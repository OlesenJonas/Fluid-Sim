#pragma once

#include "Mesh.h"

class VolumeCube : public Mesh
{
  public:
    /* Creates a cube object to use for rendering volumes
     * Dimensions: [0,1]^3
     */
    explicit VolumeCube();
};