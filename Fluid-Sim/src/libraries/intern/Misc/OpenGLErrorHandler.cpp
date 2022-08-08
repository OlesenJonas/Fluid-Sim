#include <glad/glad/glad.h>

#include <cassert>

#include "OpenGLErrorHandler.h"

void printOpenGLErrors()
{
    GLenum err = 0;
    while((err = glGetError()) != GL_NO_ERROR)
    {
        printf("opengl error occured!\n");
        printf("%#05x\n", err);
    }
}

void GLAPIENTRY OpenGLMessageCallback(
    GLenum source, GLenum type, GLuint msgId, GLenum severity, GLsizei length, const GLchar* message,
    const void* userParam)
{
    (void)userParam; // silence unused parameter warning
    (void)source;
    (void)msgId;
    (void)length;
#ifndef NDEBUG
    // some code to enable breakpoints
    if(type == GL_DEBUG_TYPE_ERROR)
    {
        {
            int x = 0;
        }
    }
#endif
    int written = fprintf(
        stderr,
        "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
        (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
        type,
        severity,
        message);
    assert(written > 0);
}

void setupOpenGLMessageCallback()
{
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS); // synch or asynch debug output
    glDebugMessageCallback(OpenGLMessageCallback, 0);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, 0, GL_FALSE); // DISABLE ALL MESSAGES
    // ENABLE ERRORS
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_HIGH, 0, 0, GL_TRUE);
    // ENABLE MAJOR WARNINGS
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_MEDIUM, 0, 0, GL_TRUE);
    // WARN ON DEPRECATED BEHAVIOUR
    glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR, GL_DONT_CARE, 0, 0, GL_TRUE);
    // WARN ON UNDEFINED BEHAVIOUR
    glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR, GL_DONT_CARE, 0, 0, GL_TRUE);
    // WARN ON PERFORMANCE
    glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_PERFORMANCE, GL_DONT_CARE, 0, 0, GL_TRUE);
}