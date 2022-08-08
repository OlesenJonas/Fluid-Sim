#include "Computepass.h"

#include <glad/glad/glad.h>

#include <utility>

Computepass::Computepass(ComputepassDesc descriptor) : descriptor(std::move(descriptor))
{
    assert(descriptor.shaderPtr != nullptr);
}

void Computepass::execute() const
{
    startMarker();
    bind();
    run();
    endMarker();
}

void Computepass::startMarker() const
{
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, descriptor.name);
}

void Computepass::bind() const
{
    descriptor.shaderPtr->useProgram();
    /* For compute passes marked const, these for loops wont end up in the final compiled program
     * Instead it seems to be as optimised as manually writing the command list
     * (when using clang, msvc seems to keep the for loop :/ )
     * https://compiler-explorer.com/z/c41jzjr4c
     */
    for(const auto& binding : descriptor.textureBindings)
    {
        glBindTextureUnit(binding.unit, binding.texture->getTextureID());
    }
    for(const auto& binding : descriptor.imageBindings)
    {
        glBindImageTexture(
            binding.unit,
            binding.texture->getTextureID(),
            binding.level,
            binding.layered,
            binding.layer,
            binding.access,
            binding.format);
    }
    // for(const auto& binding : descriptor.uniformBufferBindings) {
    //     glBindBufferBase(GL_UNIFORM_BUFFER, binding.index, *binding.buffer);
    // }
}

void Computepass::run() const
{
    glDispatchCompute(descriptor.numX, descriptor.numY, descriptor.numZ);

    glMemoryBarrier(descriptor.barriers);
}

void Computepass::endMarker() const
{
    glPopDebugGroup();
}