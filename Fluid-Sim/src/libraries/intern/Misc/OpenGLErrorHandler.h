#pragma once

#include <glad/glad/glad.h>

#include <iostream>

/*
    Asynchronous function to retrieve all
    unhandled OpenGL errors
*/
void printOpenGLErrors();

/*
    Callback function to automatically handle an
    OpenGL error when it occurs
*/
void GLAPIENTRY OpenGLMessageCallback(
    GLenum source, GLenum type, GLuint msgId, GLenum severity, GLsizei length, const GLchar* message,
    const void* userParam);

void setupOpenGLMessageCallback();