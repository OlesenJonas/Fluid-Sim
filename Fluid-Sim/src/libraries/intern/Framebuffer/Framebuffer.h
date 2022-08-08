#pragma once

#include <glad/glad/glad.h>

#include <initializer_list>
#include <vector>

#include <intern/Texture/Texture.h>

class Framebuffer
{
  public:
    Framebuffer(
        GLsizei width, GLsizei height, std::initializer_list<GLenum> colorTextureFormats,
        bool useDepthStencil);
    ~Framebuffer();

    void bind() const;

    [[nodiscard]] const std::vector<Texture>& getColorTextures() const;
    // todo: not sure about optional refernces so just return raw pointer instead
    [[nodiscard]] const Texture* getDepthTexture() const;

  private:
    GLsizei width = -1;
    GLsizei height = -1;
    bool hasDepthStencilAttachment = false;
    GLuint handle = 0xFFFFFFFF;
    std::vector<Texture> textures;
};