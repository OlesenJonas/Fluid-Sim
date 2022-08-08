#pragma once

#include <glad/glad/glad.h>

#include "GLTexture.h"

// forward declared, part of Texture.h
struct TextureDesc;

/** Texture class encapsulating and managing an OpenGL 3D texture object.
 */
class Texture3D : public GLTexture
{
  public:
    /** delete any copy ctor/assign since the implicitly existing ones
     * would only copy the opengl handle and not the full texture
     * meaning if one gets destructed both objects would have an invalid handle
     * and I dont want to deal with that right now
     */
    Texture3D(const Texture3D&) = delete;            // copy constr
    Texture3D& operator=(const Texture3D&) = delete; // copy assign

    /** Creates an immutable Texture object based on a given descriptor.
     * @param descriptor Descriptor to use for configuring the texture
     */
    explicit Texture3D(TextureDesc descriptor);

    /*
        Move constructor

    */
    Texture3D(Texture3D&& other) noexcept;
    /*
        Move assignment
    */
    Texture3D& operator=(Texture3D&& other) noexcept;

    /** Deletes the OpenGL texture object.
     */
    ~Texture3D();

    void swap(Texture3D& other);

    inline int getWidth() const
    {
        return width;
    };

    inline int getHeight() const
    {
        return height;
    };

    inline int getDepth() const
    {
        return depth;
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
    int depth = -1;
};