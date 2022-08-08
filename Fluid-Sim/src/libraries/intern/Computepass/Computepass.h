#pragma once

#include <glad/glad/glad.h>

#include <cstdint>
#include <vector>

#include <intern/ShaderProgram/ShaderProgram.h>
#include <intern/Texture/GLTexture.h>

// can be expanded as needed, currently not taking buffers as inputs/outputs, just textures/images

/*
    ComputepassDescriptor holds all parameters and information needed to execute the Computepass.
    Grouped in a single struct so constructor needs just one argument
*/
struct ImageBinding
{
    GLuint unit = 0xFFFFFFFF;
    GLTexture* texture = nullptr;
    GLint level = 0;
    GLboolean layered = GL_FALSE;
    GLint layer = 0;
    GLenum access = 0xFFFFFFFF;
    // todo: format could be retrieved from texture if not otherwise specified I think
    GLenum format = 0xFFFFFFFF;
};
struct TextureBinding
{
    GLuint unit = 0xFFFFFFFF;
    GLTexture* texture = nullptr;
};
/*
struct UniformBufferBinding {
    GLuint index = 0xFFFFFFFF;
    //until theres a buffer abstraction, just use ptr as indirection
    GLuint* buffer = nullptr;
    GLintptr offset = 0;
    GLsizeiptr size = 0;
};*/
struct ComputepassDesc
{
    const char* name = "";
    ShaderProgram* shaderPtr = nullptr;
    uint32_t numX = 1;
    uint32_t numY = 1;
    uint32_t numZ = 1;
    GLbitfield barriers = 0U;
    std::vector<TextureBinding> textureBindings;
    std::vector<ImageBinding> imageBindings;
    // std::vector<UniformBufferBinding> uniformBufferBindings;
};

class Computepass
{
  public:
    // does not pass by ref or ptr since it should be constructed inside call
    explicit Computepass(ComputepassDesc descriptor);

    /*Execute does everything: Sets a debug marker, binds the shader and runs it */
    void execute() const;

    /* Only do part of the pass and allow for other stuff to be executed inbetween
     * (like extra uniforms in a "hacky" way)
     * todo: something like this? https://compiler-explorer.com/z/nznjqKrnY taking lambda as parameter instead
     */
    void startMarker() const;
    void bind() const;
    void run() const;
    void endMarker() const;

  private:
    ComputepassDesc descriptor;
};