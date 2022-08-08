#include <cassert>

#include "Window.h"

GLFWwindow* initAndCreateGLFWWindow(
    const int width, const int height, const char* title, const std::initializer_list<WindowHint> hints)
{

    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    // todo: 4.5 to get access to most DSA functions. Bump to 4.6 if needed
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifndef NDEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif

    /*
        enable sRGB capability on default framebuffer
        (actual functionality will still be disabled
        until GL_FRAMEBUFFER_SRGB is enabled)
    */
    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

    for(const WindowHint& hint : hints)
    {
        glfwWindowHint(hint.first, hint.second);
    }

    GLFWwindow* window = nullptr;
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    glfwMakeContextCurrent(window);

    assert(window != nullptr);
    return window;
}