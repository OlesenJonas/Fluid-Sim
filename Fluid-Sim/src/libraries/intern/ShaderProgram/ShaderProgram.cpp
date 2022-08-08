#include <array>

#include "ShaderProgram.h"

ShaderProgram::ShaderProgram(
    GLuint shaderMask, const std::initializer_list<std::string> shaderFiles,
    const std::initializer_list<DefinePair> defines)
{
    shaderName = std::data(shaderFiles)[shaderFiles.size() - 1];
    size_t pos = shaderName.find_last_of('/') + 1;
    shaderName = shaderName.substr(pos, shaderName.find_last_of('.') - pos);

    int nextName = 0;
    std::array<GLuint, 6> shaderStageIDs = {
        {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}};
    struct ShaderStage
    {
        GLenum shaderBit;
        GLenum shaderEnum;
    };
    constexpr static std::array<ShaderStage, 6> stages = {
        {{VERTEX_SHADER_BIT, GL_VERTEX_SHADER},
         {TESS_CONTROL_BIT, GL_TESS_CONTROL_SHADER},
         {TESS_EVAL_BIT, GL_TESS_EVALUATION_SHADER},
         {GEOMETRY_SHADER_BIT, GL_GEOMETRY_SHADER},
         {FRAGMENT_SHADER_BIT, GL_FRAGMENT_SHADER},
         {COMPUTE_SHADER_BIT, GL_COMPUTE_SHADER}}};
    for(int i = 0; i < stages.size(); i++)
    {
        if((shaderMask & stages[i].shaderBit) != 0u)
        {
            bool success = true;
            shaderStageIDs[i] = glCreateShader(stages[i].shaderEnum);
            std::string_view stagePath = std::data(shaderFiles)[nextName++];
            loadShaderSource(shaderStageIDs[i], stagePath.data(), defines);
            glCompileShader(shaderStageIDs[i]);
            success &= checkShader(shaderStageIDs[i]);
            if(!success)
            {
                // todo: something here
                programID = 0xFFFFFFFF;
                return;
            }
            size_t pos = stagePath.find_last_of('/') + 1;
            std::string_view stageName = stagePath.substr(pos);
            glObjectLabel(GL_SHADER, shaderStageIDs[i], -1, stageName.data());
        }
    }

    // link shader programs
    programID = glCreateProgram();
    for(int i = 0; i < stages.size(); i++)
    {
        if((shaderMask & stages[i].shaderBit) != 0u)
        {
            glAttachShader(programID, shaderStageIDs[i]);
        }
    }
    glLinkProgram(programID);

    if(!checkProgram(programID))
    {
        programID = 0xFFFFFFFF;
        return;
    }
    glObjectLabel(GL_PROGRAM, programID, -1, shaderName.c_str());
}

ShaderProgram::~ShaderProgram()
{
    if(programID != 0xFFFFFFFF)
        glDeleteProgram(programID);
}

GLuint ShaderProgram::getProgramID()
{
    return programID;
}

void ShaderProgram::useProgram()
{
    glUseProgram(programID);
}

void ShaderProgram::loadShaderSource(
    GLint shaderID, const char* path, const std::initializer_list<DefinePair> defines)
{
    std::ifstream file(path);
    if(!file.is_open())
    {
        std::cerr << "ERROR: Unable to open file " << path << std::endl;
        return;
    }
    if(file.eof())
    {
        std::cerr << "ERROR: File is empty " << path << std::endl;
        return;
    }

    // load file
    std::string fileContent;
    std::string line;

    // cant insert text before #version line, so track if it has been parsed yet
    bool versionFound = false;
    uint16_t lineNumber = 1;
    do
    {
        getline(file, line);
        lineNumber++;
        versionFound = line.substr(0, 8) == "#version";
        fileContent += line + "\n";
    }
    while(!file.eof() && !versionFound);

    for(const auto& definePair : defines)
    {
        fileContent += "#define ";
        fileContent += definePair.first;
        fileContent += " ";
        fileContent += definePair.second;
        fileContent += "\n";
    }
    fileContent += "#line " + std::to_string(lineNumber) + "\n";

    while(!file.eof())
    {
        getline(file, line);
        if(line.length() >= 8 && line.substr(0, 8) == "#include")
        {
            // todo: handle includes
            //  shaderString += insertFile(line);
        }
        else
        {
            fileContent += line + "\n";
        }
    }
    file.close();

    const char* source = fileContent.c_str();
    const auto size = static_cast<GLint>(strlen(source));
    glShaderSource(shaderID, 1, &source, &size);
}

bool ShaderProgram::checkShader(GLuint shaderID)
{
    GLint compileStatus = 0;
    glGetShaderiv(shaderID, GL_COMPILE_STATUS, &compileStatus);

    if(compileStatus == GL_FALSE)
    {
        std::cout << "!Failed to compile Shader: " << shaderID << " - " << shaderName << "!" << std::endl;
        glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &compileStatus);
        auto* infoLog = new GLchar[compileStatus + 1];
        glGetShaderInfoLog(shaderID, compileStatus, nullptr, infoLog);
        std::cout << infoLog << std::endl;
        delete[] infoLog;
        return false;
    }
    /*else {
        std::cout << "Shader compiled successfully." << std::endl;
    }*/
    return true;
}

bool ShaderProgram::checkProgram(GLuint programID)
{
    GLint linkStatus = 0;
    glGetProgramiv(programID, GL_LINK_STATUS, &linkStatus);

    if(linkStatus == GL_FALSE)
    {
        std::cout << "!Failed to link Program: " << programID << " - " << shaderName << "!" << std::endl;
        glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &linkStatus);
        auto* infoLog = new GLchar[linkStatus + 1];
        glGetProgramInfoLog(programID, linkStatus, nullptr, infoLog);
        std::cout << infoLog << std::endl;
        delete[] infoLog;
        return false;
    }
    /*else {
        std::cout << "Shader Program '" << shaderName << "' linked successfully." << std::endl;
    }*/
    return true;
}