#include "InputManager.h"

#include <limits>

#include <ImGui/imgui.h>

#include <intern/Context/Context.h>

InputManager::InputManager(Context& ctx) : ctx(ctx)
{
    glfwGetCursorPos(ctx.getWindow(), &mouseX, &mouseY);
    oldMouseX = mouseX;
    oldMouseY = mouseY;
}

void InputManager::resetTime(int64_t frameCount, double simulationTime)
{
    this->frameCount = frameCount;
    this->simulationTime = simulationTime;
}

void InputManager::disableFixedTimestep()
{
    useFixedTimestep = false;
}

void InputManager::enableFixedTimestep(double timestep)
{
    useFixedTimestep = true;
    fixedDeltaTime = timestep;
}

void InputManager::update()
{
    frameCount++;
    double currentRealTime = glfwGetTime();
    deltaTime = currentRealTime - realTime;
    realTime = currentRealTime;

    simulationTime += useFixedTimestep ? fixedDeltaTime : deltaTime;

    glfwGetCursorPos(ctx.getWindow(), &mouseX, &mouseY);
    mouseDelta = {mouseX - oldMouseX, mouseY - oldMouseY};
    oldMouseX = mouseX;
    oldMouseY = mouseY;
}

void InputManager::setupCallbacks()
{
    GLFWwindow* window = ctx.getWindow();
    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetFramebufferSizeCallback(window, resizeCallback);

    glfwSetWindowUserPointer(window, reinterpret_cast<void*>(&ctx));
}

// setCallbcaks
// set window pointer!!
// set callback functions!

//// STATIC FUNCTIONS ////

void InputManager::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    // IsWindowHovered enough? or ImGui::getIO().WantCapture[Mouse/Key]
    if(ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenBlockedByPopup))
    {
        return;
    }

    Context& ctx = *static_cast<Context*>(glfwGetWindowUserPointer(window));
    // if((button == GLFW_MOUSE_BUTTON_MIDDLE || button == GLFW_MOUSE_BUTTON_RIGHT) && action == GLFW_PRESS)
    // {
    //     glfwGetCursorPos(window, &cbStruct->mousePos->x, &cbStruct->mousePos->y);
    // }
    if(button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
    {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        ctx.getCamera()->setMode(Camera::Mode::FLY);
    }
    if(button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
    {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        ctx.getCamera()->setMode(Camera::Mode::ORBIT);
    }
    if(button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        // ...
    }
}

void InputManager::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    Context& ctx = *static_cast<Context*>(glfwGetWindowUserPointer(window));
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, true);
    }
}

void InputManager::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    Context& ctx = *static_cast<Context*>(glfwGetWindowUserPointer(window));
    // IsWindowHovered enough? or ImGui::getIO().WantCapture[Mouse/Key]
    if(!ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow))
    {
        Camera* cam = ctx.getCamera();
        if(cam->getMode() == Camera::Mode::ORBIT)
        {
            cam->changeRadius(yoffset < 0);
        }
        else if(cam->getMode() == Camera::Mode::FLY)
        {
            float factor = (yoffset > 0) ? 1.1f : 1 / 1.1f;
            cam->setFlySpeed(cam->getFlySpeed() * factor);
        }
    }
}

void InputManager::resizeCallback(GLFWwindow* window, int width, int height)
{
    // todo
}