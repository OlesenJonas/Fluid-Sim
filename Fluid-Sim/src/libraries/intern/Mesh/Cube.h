#pragma once

#include "Mesh.h"

class Cube : public Mesh
{
  public:
    /** Creates a cube object centered around 0,0,0
     * @param size Length of all sides of the cube
     */
    explicit Cube(float size = 1.0f);
};