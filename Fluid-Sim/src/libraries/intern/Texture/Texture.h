#pragma once

#include <glad/glad/glad.h>

#include <stb/stb_image.h>

#include <iostream>
#include <string>

#include "GLTexture.h"

struct TextureDesc
{
    const char* name = "";
    GLsizei levels = 1;
    GLsizei width = -1;
    GLsizei height = -1;
    GLsizei depth = -1;
    GLenum internalFormat = 0xFFFFFFFF;
    GLint minFilter = GL_LINEAR;
    GLint magFilter = GL_LINEAR;
    GLint wrapS = GL_REPEAT;
    GLint wrapT = GL_REPEAT;
    GLint wrapR = GL_REPEAT;
    const uint8_t* data = nullptr;
    GLenum dataFormat = 0xFFFFFFFF;
    GLenum dataType = 0xFFFFFFFF;
    bool generateMips = false;
};

class Texture : public GLTexture
{
  public:
    /** delete any copy ctor/assign since the implicitly existing ones
     * would only copy the opengl handle and not the full texture
     * meaning if one gets destructed both objects would have an invalid handle
     * and I dont want to deal with that right now
     */
    Texture(const Texture&) = delete;            // copy constr
    Texture& operator=(const Texture&) = delete; // copy assign

    /** Sets up and loads a texture from a file.
     * @param file Name of the image file within the resource folder
     * @param mipMap True if texture is supposed to generate mipmapping
     */
    Texture(const std::string& file, bool mipMap);

    /** Creates an immutable Texture object based on a given descriptor.
     * @param descriptor Descriptor to use for configuring the texture
     */
    explicit Texture(TextureDesc descriptor);

    /*
        Move constructor

    */
    Texture(Texture&& other) noexcept;
    /*
        Move assignment
    */
    Texture& operator=(Texture&& other) noexcept;

    /** Deletes the OpenGL texture object.
     */
    ~Texture();

    void swap(Texture& other);

    inline int getWidth()
    {
        return width;
    };

    inline int getHeight()
    {
        return height;
    };

  private:
    /*
        does not actually store the descriptor (not needed atm)
        if that changes in the future, then make sure that move ctor, move assign and swap
        also move the descriptor
    */
    bool initialized = false;
    int width = -1;
    int height = -1;
};