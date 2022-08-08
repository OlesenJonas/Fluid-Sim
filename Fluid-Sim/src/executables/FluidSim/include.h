#pragma once

#include <GLFW/glfw3.h>

#include <intern/Context/Context.h>
#include <intern/FluidSolver/FluidSolver.h>
#include <intern/InputManager/InputManager.h>
#include <intern/Misc/RollingAverage.h>
#include <intern/ShaderProgram/ShaderProgram.h>
#include <intern/SimulationCapture/SimulationCapture.h>
#include <intern/VolumeRenderer/VolumeRenderer.h>

// for convenience
using FrametimeAverage_t = RollingAverage<float, 128>;

// todo: add to Context? could just add a userData void* member for simplicity
struct PostProcessSettings
{
    float simpleExposure = 1.0f;
    int mode = 1;
};
extern PostProcessSettings* postProcessSettingsGlobal; // NOLINT

void drawFrameInfoUI(
    Context& ctx, const FrametimeAverage_t& frametimeAverage, FluidSolver& fluidSolver,
    VolumeRenderer& volumeRenderer, PostProcessSettings& postProcessSettings);