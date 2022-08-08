#include "Mesh.h"

#include <numeric>

Mesh::~Mesh()
{
    if(initialized)
    {
        glDeleteVertexArrays(1, &vaoHandle);
        glDeleteBuffers(2, &vboHandles[0]);
    }
}

void Mesh::init(std::span<const VertexStruct> vertices, std::span<const GLuint> indices)
{
    // todo: warning if already initialized
    // todo: bool: recalcTangent using mikktspace

    indexCount = indices.empty() ? vertices.size() : indices.size();

    glCreateVertexArrays(1, &vaoHandle);
    glBindVertexArray(vaoHandle);

    glGenBuffers(2, &vboHandles[0]);

    glBindBuffer(GL_ARRAY_BUFFER, vboHandles[0]);
    glBufferStorage(GL_ARRAY_BUFFER, sizeof(VertexStruct) * vertices.size(), vertices.data(), 0);

    // position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexStruct), (void*)0);
    // normal
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexStruct), (void*)(3 * sizeof(float)));
    // uvs
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexStruct), (void*)(6 * sizeof(float)));
    // tangents
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(VertexStruct), (void*)(8 * sizeof(float)));

    // uses the second buffer as the source for indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboHandles[1]);
    if(!indices.empty())
    {
        glBufferStorage(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indexCount, indices.data(), 0);
    }
    else // if no index buffer was given, construct a trivial one to not break drawing api
    {
        std::vector<GLuint> tempIndices(vertices.size());
        std::iota(tempIndices.begin(), tempIndices.end(), 0);
        glBufferStorage(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * tempIndices.size(), tempIndices.data(), 0);
    }

    // unbind the VBO, we don't need it anymore
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    initialized = true;
}

void Mesh::draw() const
{
    glBindVertexArray(vaoHandle);
    glDrawElements(GL_TRIANGLES, indexCount, indexType, nullptr);
    glBindVertexArray(0);
}