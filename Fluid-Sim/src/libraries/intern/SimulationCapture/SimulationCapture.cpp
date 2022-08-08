#include "SimulationCapture.h"

#include <glad/glad/glad.h>

#include <GLFW/glfw3.h>

#include <array>
#include <string>
#include <thread>

#include <stb/stb_image_write.h>

#include <intern/Context/Context.h>
#include <intern/InputManager/InputManager.h>

SimulationCapture::SimulationCapture(Context& ctx) : ctx(ctx)
{
}

void SimulationCapture::handleFrame()
{
    InputManager* input = ctx.getInputManager();
    GLFWwindow* window = ctx.getWindow();

    if(!settings.isCapturing)
    {
        return;
    }
    int64_t currentFrame = input->getFrameCount();
    if(currentFrame >= settings.startFrame)
    {
        if(currentFrame % settings.simulationSubsteps == 0)
        {
            int width = 0;
            int height = 0;
            glfwGetFramebufferSize(window, &width, &height);
            GLsizei nrChannels = 3;
            GLsizei stride = nrChannels * width;
            stride += (stride % 4) != 0 ? (4 - stride % 4) : 0;
            GLsizei bufferSize = stride * height;
            char* buffer = new char[bufferSize];
            glPixelStorei(GL_PACK_ALIGNMENT, 4);
            glReadBuffer(GL_FRONT);
            glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer);
            // fire and forget async writing image file.
            // not really safe, doesnt even check if file actually got written 🤷
            int simSubsteps = settings.simulationSubsteps;
            std::thread saveFrameThread(
                [buffer, width, height, simSubsteps, currentFrame, nrChannels, stride]() -> void
                {
                    std::array<char, 5> paddedNumber{};
                    (void)sprintf(paddedNumber.data(), "%04llu", currentFrame / simSubsteps);
                    std::string filePath =
                        MISC_PATH "/output/temp/screenshot_" + std::string{paddedNumber.data()} + ".png";
                    stbi_flip_vertically_on_write(1);
                    stbi_write_png(filePath.c_str(), width, height, nrChannels, buffer, stride);
                    delete[] buffer;
                });
            saveFrameThread.detach();

            // update window title to act as progress bar
            settings.progress = static_cast<float>(currentFrame - settings.startFrame) /
                                static_cast<float>(settings.endFrame - settings.startFrame);
        }
        if(currentFrame >= settings.endFrame)
        {
            settings.isCapturing = false;
            input->disableFixedTimestep();
        }
    }
}

void SimulationCapture::startCapturing()
{
    InputManager* input = ctx.getInputManager();
    int internalFramerate = settings.framerate * settings.simulationSubsteps;
    settings.isCapturing = true;
    settings.startFrame = settings.startTimeSeconds * internalFramerate;                            // NOLINT
    settings.endFrame = (settings.startTimeSeconds + settings.durationSeconds) * internalFramerate; // NOLINT
    settings.endFrame -= 1;
    /* reset time
     * Next frame should be frame 0 with time = 0
     * but frame starts with an update, so set to -1 and -1dt, this way after the update
     * it should be 0 and 0 (not sure if this causes errors somewhere)
     */
    input->resetTime(-1, -1.0 / internalFramerate);
    input->enableFixedTimestep(1.0 / internalFramerate);
}