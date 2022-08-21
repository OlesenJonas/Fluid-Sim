#include "Texture.h"

#include <glad/glad/glad.h>

#include <cassert>

#include <stb/stb_image.h>

Texture::Texture(const std::string& file, bool mipMap)
{
    // todo: load image data first, then use descriptor based constructor

    //  load texture data from file
    int bytesPerPixel = 0;
    stbi_set_flip_vertically_on_load(true);

    std::string texName = file;
    size_t pos = texName.find_last_of('/') + 1;
    texName = texName.substr(pos, texName.find_last_of('.') - pos);

    bool isHdr = stbi_is_hdr(file.c_str()) != 0;

    unsigned char* image = nullptr;
    if(!isHdr)
    {
        image = stbi_load(file.c_str(), &width, &height, &bytesPerPixel, 0);
    }
    else
    {
        assert(false && "todo: hdr image load");
        image = (unsigned char*)stbi_loadf(file.c_str(), &width, &height, &bytesPerPixel, 0);
    }

    if(bytesPerPixel == 0 || width == 0 || height == 0)
    {
        std::cout << "Could not load texture " << file << std::endl;
    }

    int levels = static_cast<int>(log2(std::max(width, height))) + 1;

    // create openGL texture object
    glCreateTextures(GL_TEXTURE_2D, 1, &textureID);
    glObjectLabel(GL_TEXTURE, textureID, static_cast<GLsizei>(texName.size()), texName.data());
    glTextureParameterf(textureID, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTextureParameterf(textureID, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTextureParameterf(textureID, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameterf(textureID, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    if(bytesPerPixel == 3)
    {
        glTextureStorage2D(textureID, levels, GL_RGB8, width, height);
        glTextureSubImage2D(textureID, 0, 0, 0, width, height, GL_RGB, GL_BYTE, image);
    }
    if(bytesPerPixel == 4)
    {
        glTextureStorage2D(textureID, levels, GL_RGB8, width, height);
        glTextureSubImage2D(textureID, 0, 0, 0, width, height, GL_RGBA, GL_BYTE, image);
    }
    if(mipMap)
    {
        glTextureParameterf(textureID, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glGenerateTextureMipmap(textureID);
    }

    // GLfloat maxAniso = 0;
    // glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &maxAniso);
    // glTextureParameterf(textureID, GL_TEXTURE_MAX_ANISOTROPY, maxAniso);

    initialized = true;
    stbi_image_free(image);
}

Texture::Texture(const TextureDesc descriptor) : width(descriptor.width), height(descriptor.height)
{

    glCreateTextures(GL_TEXTURE_2D, 1, &textureID);
    glTextureStorage2D(
        textureID, descriptor.levels, descriptor.internalFormat, descriptor.width, descriptor.height);
    glTextureParameteri(textureID, GL_TEXTURE_MIN_FILTER, descriptor.minFilter);
    glTextureParameteri(textureID, GL_TEXTURE_MAG_FILTER, descriptor.magFilter);
    glTextureParameteri(textureID, GL_TEXTURE_WRAP_S, descriptor.wrapS);
    glTextureParameteri(textureID, GL_TEXTURE_WRAP_T, descriptor.wrapT);
    // GLfloat maxAniso = 0;
    // glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &maxAniso);
    // glTextureParameterf(handle, GL_TEXTURE_MAX_ANISOTROPY, std::max(maxAniso, descriptor.maxAnisotropy));
    if(strlen(descriptor.name) > 0)
    {
        glObjectLabel(GL_TEXTURE, textureID, -1, descriptor.name);
    }
    if(descriptor.data != nullptr)
    {
        glTextureSubImage2D(
            textureID, 0, 0, 0, width, height, descriptor.dataFormat, descriptor.dataType, descriptor.data);
    }
    if(descriptor.generateMips)
    {
        glGenerateTextureMipmap(textureID);
    }
    initialized = true;
}

Texture::Texture(Texture&& other) noexcept : width(other.width), height(other.height)
{
    // if this texture was already initialized then somebody needs to delete the old content
    // so swap IDs and init staus instead of just moving and *other will delete on destruction
    std::swap(textureID, other.textureID);
    std::swap(initialized, other.initialized);
}

Texture& Texture::operator=(Texture&& other) noexcept
{
    width = other.width;
    height = other.height;
    // if this texture was already initialized then somebody needs to delete the old content
    // so swap IDs and init staus instead of just moving and *other will delete on destruction
    std::swap(textureID, other.textureID);
    std::swap(initialized, other.initialized);
    return *this;
}

Texture::~Texture()
{
    // todo: this shouldnt call glDelete, GLTexture's dtor should do that
    if(initialized)
    {
        glDeleteTextures(1, &textureID);
    }
}

void Texture::swap(Texture& other)
{
    std::swap(width, other.width);
    std::swap(height, other.height);
    std::swap(textureID, other.textureID);
    std::swap(initialized, other.initialized);
}