#pragma once

#include <GLFW/glfw3.h>
#include <initializer_list>
#include <utility>

/*
    This calls glfwInit and glfwCreateWindow internally.
    GLFW monitor and share arguments are not used and set to nullptr.
    Some default window hints will be set, but they can be overriden using
    the hint list.
    (For example during debug builds a debug context will be automatically requested,
    this can be overriden by passing {GLFW_OPENGL_DEBUG_CONTEXT, GLFW_FALSE} as hint)
*/
using WindowHint = std::pair<int, int>;
GLFWwindow* initAndCreateGLFWWindow(
    int width, int height, const char* title, std::initializer_list<WindowHint> hints = {});