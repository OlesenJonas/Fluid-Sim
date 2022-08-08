#pragma once

#define _USE_MATH_DEFINES
#include <array>
#include <cmath>
#include <string>

#include <glm/ext.hpp>
#include <glm/glm.hpp>

class Context;

/*
    TODO:
    look sensitivity, scroll sensitivity, pan sensitivity
*/

class Camera
{
  public:
    enum struct Mode
    {
        ORBIT,
        FLY
    };

    explicit Camera(Context& ctx, float aspect);

    void update();
    void move(glm::vec3 offset);
    void rotate(float dx, float dy);
    void setPosition(glm::vec3 newPosition);
    void setCenter(glm::vec3 newCenter);
    void changeRadius(bool increase);
    /* Sets vertical! Fov in radians. */
    void setFov(float _fov);
    void setAspect(float _aspect);
    void setMode(Mode mode);
    void setFlySpeed(float speed);

    void updateView();

    glm::vec3 getPosition();
    glm::mat4* getView();
    glm::mat4* getProj();
    glm::mat4* getMatricesPointer();
    float getNear() const;
    float getFar() const;
    Mode getMode() const;
    float getFlySpeed() const;

  private:
    void init();

    Context& ctx;

    Mode mode = Mode::ORBIT;
    float phi = 0.0f;
    float theta = glm::pi<float>() * 0.5;
    float radius = 1.0f;

    float aspect = 1.77f;
    float fov = glm::radians(60.0f);
    float cam_near = 0.1f;
    float cam_far = 100.0f;
    float flySpeed = 2.0f;

    // orbit uses center+viewVec as eye and center         as target
    // fly   uses center         as eye and center+viewVec as target
    glm::vec3 position;
    glm::vec3 viewVec, center;
    //[0] is view, [1] is projection, [2] inverse Projection
    std::array<glm::mat4, 3> matrices;
};