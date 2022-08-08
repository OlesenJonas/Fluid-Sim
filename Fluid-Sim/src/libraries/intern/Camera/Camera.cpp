#include "Camera.h"

#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/dual_quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

#include <GLFW/glfw3.h>

#include <intern/Context/Context.h>
#include <intern/Misc/Misc.h>

Camera::Camera(Context& ctx, float aspect) : ctx(ctx)
{
    this->aspect = aspect;
    init();
}

void Camera::init()
{
    viewVec = posFromPolar(theta, phi);
    center = glm::vec3(0.0f);
    matrices[0] = glm::lookAt(center + radius * viewVec, center, glm::vec3(0.f, 1.f, 0.f));
    position = center + radius * viewVec;
    matrices[1] = glm::perspective(fov, aspect, cam_near, cam_far);
    matrices[2] = glm::inverse(matrices[1]);
}

void Camera::update()
{
    auto* window = ctx.getWindow();
    auto* inputManager = ctx.getInputManager();
    glm::vec2 mouseDelta = inputManager->getMouseDelta();

    // todo: how much should be part of this, and how much should be inside InputManager?
    if(mode == Mode::ORBIT)
    {
        if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) != GLFW_RELEASE)
        {
            if(glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
            {
                move(glm::vec3(-mouseDelta.x * 0.005f, mouseDelta.y * 0.005f, 0.f));
            }
            else
            {
                rotate(mouseDelta.x, mouseDelta.y);
            }
        }
    }
    else if(mode == Mode::FLY)
    {
        rotate(mouseDelta.x * 0.5f, -mouseDelta.y * 0.5f); // viewVector is flipped, angle diff reversed

        glm::vec3 cam_move = glm::vec3(
            static_cast<float>(
                (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) -
                (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)),
            static_cast<float>(
                (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) -
                (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)),
            static_cast<float>(
                (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) -
                (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)));
        if(cam_move != glm::vec3(0.0f))
        {
            float dt = inputManager->getRealDeltaTime();
            move(2.0f * glm::normalize(cam_move) * dt);
        }
    }
    updateView();
}

void Camera::setPosition(glm::vec3 newPosition)
{
    if(mode == Mode::FLY)
    {
        center = newPosition;
        position = newPosition;
    }
    else if(mode == Mode::ORBIT)
    {
        position = newPosition;
        radius = glm::distance(newPosition, center);
        viewVec = glm::normalize(newPosition - center);
    }
}

void Camera::setCenter(glm::vec3 newCenter)
{
    glm::vec3 offset = position - center;
    center = newCenter;
    position = center + offset;
}

void Camera::updateView()
{
    viewVec = posFromPolar(theta, phi);
    glm::mat4& view = matrices[0];
    if(mode == Mode::ORBIT)
    {
        glm::vec3 eye = center + radius * viewVec;
        view = glm::lookAt(center + radius * viewVec, center, glm::vec3(0.f, 1.f, 0.f));
        position = center + radius * viewVec;
    }
    else if(mode == Mode::FLY)
    {
        view = glm::lookAt(center, center + viewVec, glm::vec3(0.f, 1.f, 0.f));
        position = center;
    }
}

// move the camera along its local axis
void Camera::move(glm::vec3 offset)
{
    glm::vec3 camx = glm::row(matrices[0], 0);
    glm::vec3 camy = glm::row(matrices[0], 1);
    glm::vec3 camz = glm::row(matrices[0], 2);
    glm::vec3 offs = offset.x * camx + offset.y * camy + offset.z * camz;
    center += (mode == Mode::FLY ? flySpeed : 1.0f) * offs;
}

void Camera::rotate(float dx, float dy)
{
    theta = std::min<float>(std::max<float>(theta - dy * 0.01f, 0.1f), M_PI - 0.1f);
    phi = phi - dx * 0.01f;
}

void Camera::changeRadius(bool increase)
{
    if(increase)
        radius /= 0.95f;
    else
        radius *= 0.95f;
}

void Camera::setFov(float _fov)
{
    fov = _fov;
    matrices[1] = glm::perspective(fov, aspect, cam_near, cam_far);
    matrices[2] = glm::inverse(matrices[1]);
}

void Camera::setAspect(float _aspect)
{
    aspect = _aspect;
    matrices[1] = glm::perspective(fov, aspect, cam_near, cam_far);
    matrices[2] = glm::inverse(matrices[1]);
}

void Camera::setMode(Mode mode)
{
    if(this->mode == mode)
        return;
    this->mode = mode;

    // flip viewvecter and sawp center and target
    theta = M_PI - theta;
    phi = phi + M_PI;
    center = center + radius * viewVec;
    viewVec = -viewVec;
}

void Camera::setFlySpeed(float speed)
{
    flySpeed = speed;
}

glm::mat4* Camera::getView()
{
    return &matrices[0];
}

glm::mat4* Camera::getProj()
{
    return &matrices[1];
}

glm::mat4* Camera::getMatricesPointer()
{
    return &matrices[0];
}

glm::vec3 Camera::getPosition()
{
    return position;
}

float Camera::getNear() const
{
    return cam_near;
}

float Camera::getFar() const
{
    return cam_far;
}

Camera::Mode Camera::getMode() const
{
    return mode;
}

float Camera::getFlySpeed() const
{
    return flySpeed;
}