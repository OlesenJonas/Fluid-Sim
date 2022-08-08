#include "VolumeCube.h"

#include <array>

VolumeCube::VolumeCube()
{
    std::vector<VertexStruct> vertices = {};
    vertices.reserve(8);
    // dont care about tangents, uvs, normals for this mesh
    vertices.push_back({.pos = {1.0f, 0.0f, 1.0f}});
    vertices.push_back({.pos = {0.0f, 0.0f, 1.0f}});
    vertices.push_back({.pos = {1.0f, 1.0f, 1.0f}});
    vertices.push_back({.pos = {0.0f, 1.0f, 1.0f}});

    vertices.push_back({.pos = {0.0f, 0.0f, 0.0f}});
    vertices.push_back({.pos = {1.0f, 0.0f, 0.0f}});
    vertices.push_back({.pos = {0.0f, 1.0f, 0.0f}});
    vertices.push_back({.pos = {1.0f, 1.0f, 0.0f}});

    // index structure
    std::vector<GLuint> indices = {
        1, 0, 3, //
        3, 0, 2, //

        0, 5, 2, //
        2, 5, 7, //

        5, 4, 7, //
        7, 4, 6, //

        4, 1, 6, //
        6, 1, 3, //

        3, 2, 6, //
        6, 2, 7, //

        4, 5, 1, //
        1, 5, 0  //
    };

    init(vertices, indices);
}