// TODO: make this a test instead of a standard executable

#include "intern/Computepass/Computepass.h"
#include <glad/glad/glad.h>

#include <GLFW/glfw3.h>

#include <array>

#include <intern/Context/Context.h>
#include <intern/InputManager/InputManager.h>
#include <intern/Mesh/FullscreenTri.h>
#include <intern/Misc/Misc.h>
#include <intern/Misc/OpenGLErrorHandler.h>
#include <intern/ShaderProgram/ShaderProgram.h>
#include <intern/Texture/Texture.h>
#include <intern/Texture/Texture3D.h>
#include <intern/Window/Window.h>

int main()
{
    Context ctx{};

    //----------------------- INIT WINDOW

    int WIDTH = 1200;
    int HEIGHT = 800;

    GLFWwindow* window = initAndCreateGLFWWindow(WIDTH, HEIGHT, "Multigrid Shader Test");
    ctx.setWindow(window);

    //----------------------- INIT OPENGL
    // init OpenGL context
    if(gladLoadGL() == 0)
    {
        std::cout << "Failed to initialize OpenGL context" << std::endl;
        return -1;
    }
    setupOpenGLMessageCallback();
    glClearColor(0.3f, 0.7f, 1.0f, 1.0f);

    //----------------------- INIT REST
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int depth = 64;
    std::vector<float> data(depth * width * height * 3);
    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            for(int z = 0; z < depth; z++)
            {
                int index = 3 * (z * width * height + y * width + x);
                data[index] = 0;                           // R
                data[index + 1] = float(y) / (height - 1); // G
                data[index + 2] = 0;                       // B
            }
        }
    }
    Texture3D testTex3D{
        {.name = "testTex3D",
         .width = width,
         .height = height,
         .depth = depth,
         .internalFormat = GL_RGBA32F,
         .data = (uint8_t*)data.data(),
         .dataFormat = GL_RGB,
         .dataType = GL_FLOAT}};
    Texture3D divTex{
        {.name = "divergenceTex",
         .width = width + 1,
         .height = height + 1,
         .depth = depth + 1,
         .internalFormat = GL_R32F}};

    ShaderProgram divergenceShader(
        COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/divergenceMultigrid.comp"}, {{"FORMAT", "r32f"}});
    Computepass divergencePass(ComputepassDesc{
        .name = "Divergence",
        .shaderPtr = &divergenceShader,
        .numX = UintDivAndCeil(width + 1, 16),
        .numY = UintDivAndCeil(height + 1, 16),
        .numZ = static_cast<uint32_t>(depth + 1),
        .barriers = GL_ALL_BARRIER_BITS,
        .textureBindings = {{.unit = 0, .texture = &testTex3D}},
        .imageBindings = {
            {.unit = 0,
             .texture = &divTex,
             .layered = GL_TRUE,
             .access = GL_WRITE_ONLY,
             .format = GL_R32F}}});

    glfwSetTime(0.0);

    divergencePass.execute();

    std::array<float, 8> filldata{};
    std::fill(filldata.begin(), filldata.end(), 3.0f);
    glTextureSubImage3D(divTex.getTextureID(), 0, 0, 0, 0, 2, 2, 2, GL_RED, GL_FLOAT, filldata.data());

    std::vector<float> divergenceData((depth + 1) * (width + 1) * (height + 1));
    glGetTextureImage(
        divTex.getTextureID(),
        0,
        GL_RED,
        GL_FLOAT,
        divergenceData.size() * sizeof(float),
        divergenceData.data());

    while(glfwWindowShouldClose(window) == 0)
    {
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}