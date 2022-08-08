#include "Context.h"

GLFWwindow* Context::getWindow()
{
    return window;
}
void Context::setWindow(GLFWwindow* window)
{
    this->window = window;
}

InputManager* Context::getInputManager()
{
    return inputManager;
}
void Context::setInputManager(InputManager* inputManager)
{
    this->inputManager = inputManager;
}

Camera* Context::getCamera()
{
    return camera;
}
void Context::setCamera(Camera* camera)
{
    this->camera = camera;
}

SimulationCapture* Context::getSimulationCapture()
{
    return simulationCapture;
}
void Context::setSimulationCapture(SimulationCapture* simCapture)
{
    this->simulationCapture = simCapture;
}