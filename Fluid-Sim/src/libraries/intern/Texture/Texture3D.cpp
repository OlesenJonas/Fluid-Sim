#include <glad/glad/glad.h>

#include "Texture.h"
#include "Texture3D.h"

Texture3D::Texture3D(const TextureDesc descriptor)
    : width(descriptor.width), height(descriptor.height), depth(descriptor.depth)
{

    glCreateTextures(GL_TEXTURE_3D, 1, &textureID);
    glTextureStorage3D(
        textureID,
        descriptor.levels,
        descriptor.internalFormat,
        descriptor.width,
        descriptor.height,
        descriptor.depth);
    glTextureParameteri(textureID, GL_TEXTURE_MIN_FILTER, descriptor.minFilter);
    glTextureParameteri(textureID, GL_TEXTURE_MAG_FILTER, descriptor.magFilter);
    glTextureParameteri(textureID, GL_TEXTURE_WRAP_S, descriptor.wrapS);
    glTextureParameteri(textureID, GL_TEXTURE_WRAP_T, descriptor.wrapT);
    glTextureParameteri(textureID, GL_TEXTURE_WRAP_R, descriptor.wrapR);
    if(strlen(descriptor.name) > 0)
    {
        glObjectLabel(GL_TEXTURE, textureID, -1, descriptor.name);
    }
    if(descriptor.data != nullptr)
    {
        glTextureSubImage3D(
            textureID,
            0,
            0,
            0,
            0,
            width,
            height,
            depth,
            descriptor.dataFormat,
            descriptor.dataType,
            descriptor.data);
    }
    if(descriptor.generateMips)
    {
        glGenerateTextureMipmap(textureID);
    }
    initialized = true;
}

Texture3D::Texture3D(Texture3D&& other) noexcept
    : width(other.width), height(other.height), depth(other.depth)
{
    // if this texture was already initialized then somebody needs to delete the old content
    // so swap IDs and init staus instead of just moving and *other will delete on destruction
    std::swap(textureID, other.textureID);
    std::swap(initialized, other.initialized);
}

Texture3D& Texture3D::operator=(Texture3D&& other) noexcept
{
    width = other.width;
    height = other.height;
    depth = other.depth;
    // if this texture was already initialized then somebody needs to delete the old content
    // so swap IDs and init staus instead of just moving and *other will delete on destruction
    std::swap(textureID, other.textureID);
    std::swap(initialized, other.initialized);
    return *this;
}

Texture3D::~Texture3D()
{
    if(initialized)
    {
        glDeleteTextures(1, &textureID);
    }
}

void Texture3D::swap(Texture3D& other)
{
    std::swap(width, other.width);
    std::swap(height, other.height);
    std::swap(depth, other.depth);
    std::swap(textureID, other.textureID);
    std::swap(initialized, other.initialized);
}