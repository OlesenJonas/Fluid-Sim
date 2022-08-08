#pragma once

#include <glad/glad/glad.h>

/*
  Wrapper class to enable binding both Textures and Texture3Ds
  through the same interface

  https://compiler-explorer.com/z/jjKM89ex8
*/
class GLTexture
{
  public:
    [[nodiscard]] inline GLuint getTextureID() const
    {
        return textureID;
    };

  protected:
    GLuint textureID = 0xFFFFFFFF; // NOLINT
};