#pragma once

#include <GLFW/glfw3.h>

#include <intern/Camera/Camera.h>
#include <intern/InputManager/InputManager.h>
#include <intern/SimulationCapture/SimulationCapture.h>

// very simple context struct to pass "global" objects around
class Context
{
  public:
    Context() = default;

    GLFWwindow* getWindow();
    void setWindow(GLFWwindow* window);
    InputManager* getInputManager();
    void setInputManager(InputManager* inputManager);
    Camera* getCamera();
    void setCamera(Camera* camera);
    SimulationCapture* getSimulationCapture();
    void setSimulationCapture(SimulationCapture* simCapture);

  private:
    // todo: window abstraction that stores width/height for glViewport calls in renderloop
    GLFWwindow* window;
    InputManager* inputManager;
    Camera* camera;
    SimulationCapture* simulationCapture;
};