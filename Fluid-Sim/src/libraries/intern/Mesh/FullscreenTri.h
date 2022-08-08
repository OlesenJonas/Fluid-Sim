#pragma once

#include "Mesh.h"

#include <array>

class FullscreenTri : public Mesh
{
  public:
    FullscreenTri();

  private:
    // todo: dont need full tangent, normals just for fullscreen image effects
    //       could override init and get rid of them
    constexpr static std::array<VertexStruct, 3> vertices = {
        {{{-1.0f, -1.0f, 0.0f}, {0.f, 0.0f, 1.f}, {0.0f, 0.0f}, {1.f, 0.f, 0.f, 1.f}},
         {{3.0f, -1.0f, 0.0f}, {0.f, 0.0f, 1.f}, {2.0f, 0.0f}, {1.f, 0.f, 0.f, 1.f}},
         {{-1.0f, 3.0f, 0.0f}, {0.f, 0.0f, 1.f}, {0.0f, 2.0f}, {1.f, 0.f, 0.f, 1.f}}}};
    constexpr static std::array<GLuint, 3> indices = {{0, 1, 2}};
};