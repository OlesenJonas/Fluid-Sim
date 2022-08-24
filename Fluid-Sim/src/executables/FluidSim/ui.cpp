#include <glad/glad/glad.h>

#include <array>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <utility>

#include <GLFW/glfw3.h>

#include <stb/stb_image_write.h>

#include <ImGui/imgui.h>
#include <ImGui/imgui_impl_glfw.h>
#include <ImGui/imgui_impl_opengl3.h>

#include <intern/Camera/Camera.h>
#include <intern/Computepass/Computepass.h>
#include <intern/FluidSolver/FluidSolver.h>
#include <intern/InputManager/InputManager.h>
#include <intern/Misc/GPUTimer.h>
#include <intern/Misc/ImGuiExtensions.h>
#include <intern/Misc/OpenGLErrorHandler.h>
#include <intern/Misc/RollingAverage.h>
#include <intern/ShaderProgram/ShaderProgram.h>
#include <intern/SimulationCapture/SimulationCapture.h>
#include <intern/VolumeRenderer/VolumeRenderer.h>
#include <intern/Window/Window.h>

#include "include.h"

void drawFrameInfoUI(
    Context& ctx, const FrametimeAverage_t& frametimeAverage, FluidSolver& fluidSolver,
    VolumeRenderer& volumeRenderer, PostProcessSettings& postProcessSettings)
{
    InputManager* input = ctx.getInputManager();
    SimulationCapture* simulationCapture = ctx.getSimulationCapture();
    Camera* cam = ctx.getCamera();

    // evaluate timers
    fluidSolver.getTimer().evaluate();
    auto& fsTimers = fluidSolver.getComponentTimers();
    // https://compiler-explorer.com/z/cb4PvjsxW
    for(auto index :
        {FluidSolver::Timer::Advection,
         FluidSolver::Timer::Impulse,
         FluidSolver::Timer::Buoyancy,
         FluidSolver::Timer::Divergence,
         FluidSolver::Timer::PressureSolve,
         FluidSolver::Timer::PressureSubtraction})
    {
        fsTimers[index].evaluate();
    }
    volumeRenderer.getTimer().evaluate();

    ImGui::Begin("Info", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    {
        constexpr auto averagedFrametimes = FrametimeAverage_t::length();
        static std::array<float, averagedFrametimes> frametimesSorted;
        float averageFrametime = frametimeAverage.average();

        ImGui::Text(
            "Application average %.3f ms/frame (%.1f FPS)", averageFrametime * 1000, 1.0f / averageFrametime);
        { // Graphing the frametimes
            // have to "unroll" the ringbuffer
            frametimeAverage.unroll(frametimesSorted);
            ImGui::PlotLines(
                "",
                frametimesSorted.data(),
                frametimesSorted.size(),
                0,
                nullptr,
                0.0f,
                (2.0f * averageFrametime),
                ImVec2(0, 80));

            ImGui::HorizontalBar(
                0.0f, fluidSolver.getTimer().timeMilliseconds() / (averageFrametime * 1000), ImVec2(400, 0));
            ImGui::SameLine();
            ImGui::Text("Fluid Solver    : %.3f ms", fluidSolver.getTimer().timeMilliseconds());

            ImGui::HorizontalBar(
                0.0f,
                volumeRenderer.getTimer().timeMilliseconds() / (averageFrametime * 1000),
                ImVec2(400, 0));
            ImGui::SameLine();
            ImGui::Text("Volume Rendering: %.3f ms", volumeRenderer.getTimer().timeMilliseconds());
        }
    }
    ImGui::End();
    ImGui::Begin("Simulation", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    {
        if(ImGui::CollapsingHeader("Simulation Settings"))
        {
            ImGui::Text("Simulation Time: %f", input->getSimulationTime());
            if(ImGui::Button("Reset"))
            {
                // reset time
                input->resetTime();
                // reset fluidsim
                fluidSolver.clear();
            }
            if(ImGui::Button("Use Real Delta Time"))
            {
                input->disableFixedTimestep();
            }
            static double fixedDt = 0.01;
            if(ImGui::Button("Use Fixed Delta Time"))
            {
                input->enableFixedTimestep(fixedDt);
            }
            ImGui::SameLine();
            ImGui::InputDouble("##inputFixedDt", &fixedDt);
        }

        if(ImGui::CollapsingHeader("Capture Simulation"))
        {
            auto& captureSettings = simulationCapture->getSettings();
            bool wasCapturing = captureSettings.isCapturing;
            // Disable UI when currently capturing
            if(wasCapturing)
            {
                ImGui::BeginDisabled();
            }
            if(ImGui::InputInt("Start after (s)", &captureSettings.startTimeSeconds))
            {
                captureSettings.startTimeSeconds = std::max(captureSettings.startTimeSeconds, 0);
            }
            if(ImGui::InputInt("Duration (s)", &captureSettings.durationSeconds))
            {
                captureSettings.durationSeconds = std::max(captureSettings.durationSeconds, 0);
            }

            if(ImGui::InputInt("Output Framerate", &captureSettings.framerate))
            {
                captureSettings.framerate = std::max(captureSettings.framerate, 1);
            }

            if(ImGui::InputInt("Simulation Substeps", &captureSettings.simulationSubsteps))
            {
                captureSettings.simulationSubsteps = std::max(captureSettings.simulationSubsteps, 1);
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if(ImGui::IsItemHovered())
            {
                ImGui::BeginTooltip();
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
                ImGui::TextUnformatted("How many simulation steps will be performed for each rendered frame");
                ImGui::PopTextWrapPos();
                ImGui::EndTooltip();
            }
            if(ImGui::Button("Record"))
            {
                // (this also resets time and framecount)
                simulationCapture->startCapturing();
                // reset cam
                cam->setPosition({0.0f, -0.042496f, 1.699469f});
                cam->update();
                // reset fluidsim
                fluidSolver.clear();
            }
            if(wasCapturing)
            {
                ImGui::EndDisabled();
                ImGui::ProgressBar(captureSettings.progress);
            }
        }
    }
    ImGui::End();
    ImGui::Begin("Fluid Solver", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    {
        double fsCombinedTime = fluidSolver.getTimer().timeMilliseconds();
        ImGui::Text("%.3f ms", fsCombinedTime);
        ImGui::Separator();

        auto& fluidSolverSettings = fluidSolver.getSettings();
        if(ImGui::Button("Clear"))
        {
            fluidSolver.clear();
        }
        double timeCumulative = 0.0;
        /**hoping that compiler can unroll loop over constexpr array
         * because its nicer/easier to write this way
         * https://compiler-explorer.com/z/c41caT95o
         * seems like clang does it but not msvc
         */
        struct timerToDisplay
        {
            FluidSolver::Timer timer;
            std::string_view name;
        };
        constexpr std::array<const timerToDisplay, 6> timersToDisplay = {
            {{FluidSolver::Timer::Advection, "Advection"},
             {FluidSolver::Timer::Impulse, "Impulse"},
             {FluidSolver::Timer::Buoyancy, "Buoyancy"},
             {FluidSolver::Timer::Divergence, "Divergence"},
             {FluidSolver::Timer::PressureSolve, "Pressure Solve"},
             {FluidSolver::Timer::PressureSubtraction, "Pressure Sub"}}};
        // get the longest name for alignement in text
        constexpr size_t maxNameLength = std::max_element(
                                             std::begin(timersToDisplay),
                                             std::end(timersToDisplay),
                                             [](const timerToDisplay& lhs, const timerToDisplay& rhs)
                                             {
                                                 return lhs.name.size() < rhs.name.size();
                                             })
                                             ->name.size();
        //---
        for(const auto& timerInfo : timersToDisplay)
        {
            double timerRes = fsTimers[timerInfo.timer].timeMilliseconds();
            double newTime = timeCumulative + timerRes;
            ImGui::HorizontalBar(
                timeCumulative / fsCombinedTime, newTime / fsCombinedTime, ImVec2(400, 0), "");
            ImGui::SameLine();
            ImGui::Text("%-*s: %.3f ms", static_cast<int>(maxNameLength), timerInfo.name.data(), timerRes);
            timeCumulative = newTime;
        }
        //---

        // Component Settings
        if(ImGui::CollapsingHeader("Advection"))
        {
            bool settingsChanged = false;
            settingsChanged |=
                ImGui::Combo("Advection Mode", &fluidSolverSettings.advectSettings.mode, "Euler\0RK4\0");
            settingsChanged |= ImGui::Checkbox(
                "Use BFECC for Temperature Advection", &fluidSolverSettings.useBFECCTemperature);
            settingsChanged |=
                ImGui::Checkbox("Use BFECC for Density Advection", &fluidSolverSettings.useBFECCDensity);
            settingsChanged |=
                ImGui::Checkbox("Use BFECC for Velocity Advection", &fluidSolverSettings.useBFECCVelocity);
            bool BFECCused = fluidSolverSettings.useBFECCVelocity || fluidSolverSettings.useBFECCDensity ||
                             fluidSolverSettings.useBFECCTemperature;
            if(BFECCused)
            {
                settingsChanged |= ImGui::Combo(
                    "BFECC Limiter Mode",
                    &fluidSolverSettings.advectSettings.limiter,
                    "None\0Clamp\0Revert\0");
            }
            ImGui::SliderFloat(
                "Temperature Dissipation", &fluidSolverSettings.temperatureDissipation, 0.0f, 1.0f);
            ImGui::SliderFloat("Density Dissipation", &fluidSolverSettings.densityDissipation, 0.0f, 1.0f);
            ImGui::SliderFloat("Velocity Dissipation", &fluidSolverSettings.velocityDissipation, 0.0f, 1.0f);
            if(settingsChanged)
            {
                fluidSolver.updateSettingsBuffer();
            }
        }
        if(ImGui::CollapsingHeader("Pressure Solver"))
        {
            ImGui::Checkbox(
                "Use last frames pressure as initial guess", &fluidSolverSettings.useLastFrameAsInitialGuess);
            ImGui::Combo(
                "Solver Mode", (int*)&fluidSolverSettings.solverMode, "Jacobi\0RBGS\0Multigrid (Jacobi)\0");
            if(fluidSolverSettings.solverMode < FluidSolver::PressureSolver::Multigrid)
            {
                int temp = fluidSolverSettings.iterations;
                if(ImGui::SliderInt("Solver Iteration", &temp, 0, 255))
                {
                    fluidSolverSettings.iterations = temp;
                }
            }
            else
            {
                int tempIter = fluidSolverSettings.mgPrePostSmoothIterations;
                if(ImGui::SliderInt("Pre- & Postsmoothing Iterations", &tempIter, 0, 255))
                {
                    fluidSolverSettings.mgPrePostSmoothIterations = tempIter;
                }
                int tempLevels = fluidSolverSettings.mgLevels;
                if(ImGui::SliderInt("Multigrid Levels", &tempLevels, 1, fluidSolver.getLevels()))
                {
                    fluidSolverSettings.mgLevels = tempLevels;
                }
            }
        }
        if(ImGui::CollapsingHeader("Divergence Remainder"))
        {
            ImGui::Checkbox(
                "Calculate remaining divergence", &fluidSolverSettings.calculateRemainingDivergence);
            if(fluidSolverSettings.calculateRemainingDivergence)
            {
                auto remainingDivergence = fluidSolver.getRemainingDivergence();
                ImGui::SameLine();
                fsTimers[FluidSolver::Timer::DivergenceRemainder].evaluate();
                float fullRatio = remainingDivergence.totalDivAfter / remainingDivergence.totalDivBefore;
                float innerRatio =
                    remainingDivergence.totalDivAfterInner / remainingDivergence.totalDivBeforeInner;
                float pixels = fluidSolver.getVelocityTexture().getWidth() *
                               fluidSolver.getVelocityTexture().getHeight() *
                               fluidSolver.getVelocityTexture().getDepth();
                ImGui::Text("- %.3fms", fsTimers[FluidSolver::Timer::DivergenceRemainder].timeMilliseconds());
                ImGui::TextUnformatted("Total absolute remaining divergence:");
                ImGui::TextUnformatted("All:");
                ImGui::Text("Before:%6.3f", remainingDivergence.totalDivBefore);
                ImGui::Text("After:%6.3f", remainingDivergence.totalDivAfter);
                ImGui::Text("After (per pixel):%6.3f", remainingDivergence.totalDivAfter / pixels);
                ImGui::Text("Left over:%6.3f", fullRatio);
                ImGui::TextUnformatted("Without Border Texel:");
                ImGui::Text("Before:%6.3f", remainingDivergence.totalDivBeforeInner);
                ImGui::Text("After:%6.3f", remainingDivergence.totalDivAfterInner);
                ImGui::Text("After (per pixel):%6.3f", remainingDivergence.totalDivAfterInner / pixels);
                ImGui::Text("Left over:%6.3f", innerRatio);
            }
        }
    }
    ImGui::End();
    ImGui::Begin("Rendering", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    {
        VolumeRenderer::Settings& volumeSettings = volumeRenderer.getSettings();
        ImGui::TextUnformatted("Directional Light");
        glm::vec3 oldDir = fluidSolver.getTransform() * glm::vec4(volumeSettings.lightVectorLocal, 0.0);
        float theta = acos(oldDir.y);
        float phi = atan2(oldDir.z, oldDir.x);
        bool changed = false;
        changed |= ImGui::SliderFloat("Theta", &theta, 0.0f, glm::pi<float>());
        changed |= ImGui::DragFloat("Phi", &phi, 0.01f);
        if(changed)
        {
            glm::vec3 newDir = {cos(phi) * sin(theta), cos(theta), sin(phi) * sin(theta)};

            // Also update for volume renderer
            volumeSettings.lightVectorLocal = fluidSolver.getInvTransform() * glm::vec4(newDir, 0.0f);
            volumeRenderer.updateSettingsBuffer();
        }
        if(ImGui::Button("Reset Direction##lightDir"))
        {
            glm::vec3 newDir = {0.f, 1.f, 0.f};
            // Also update for volume renderer
            volumeSettings.lightVectorLocal = fluidSolver.getInvTransform() * glm::vec4(newDir, 0.0f);
            volumeRenderer.updateSettingsBuffer();
        }
        if(ImGui::ColorEdit4(
               "Color and Strength##directional",
               &volumeSettings.sunColorAndStrength.x,
               ImGuiColorEditFlags_Float))
        {
            volumeRenderer.updateSettingsBuffer();
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Ambient Light");
        if(ImGui::ColorEdit4(
               "Color and Strength##ambient",
               &volumeSettings.skyColorAndStrength.x,
               ImGuiColorEditFlags_Float))
        {
            volumeRenderer.updateSettingsBuffer();
        }

        if(ImGui::CollapsingHeader("Volume Settings"))
        {
            static std::array<char, 100> inputText;
            static bool first = true;
            if(first)
            {
                first = false;
                inputText.fill('\0');
            }
            // todo: scan for presets
            ImGui::InputText("Preset Name", inputText.data(), inputText.size() - 1);
            if(ImGui::Button("Save Preset"))
            {
                std::string fullPath = std::string(MISC_PATH "/volumePresets/") + inputText.data() + ".bin";
                volumeRenderer.saveSettings(fullPath);
            }
            ImGui::SameLine();
            if(ImGui::Button("Load Preset"))
            {
                std::string fullPath = std::string(MISC_PATH "/volumePresets/") + inputText.data() + ".bin";
                volumeRenderer.loadSettings(fullPath);
            }
            ImGui::Separator();
            VolumeRenderer::Settings& settings = volumeSettings;
            bool changed = false;
            changed |= ImGui::SliderInt("Plane Alignment", &settings.planeAlignment, 0, 1, "");
            changed |= ImGui::SliderInt("Max Steps", &settings.maxSteps, 16, 128);
            changed |= ImGui::SliderInt("Hard Step Limit", &settings.hardStepsLimit, 16, 256);
            changed |= ImGui::SliderInt("Shadow Steps", &settings.shadowSteps, 16, 64);
            ImGui::Indent(15.0f);
            changed |= ImGui::SliderFloat("Jitter", &settings.jitter, 0.0f, 1.0f);
            changed |= ImGui::Combo("Noise Type", &settings.noiseType, "White\0Blue\0");
            ImGui::Unindent(15.0f);
            ImGui::Separator();
            changed |= ImGui::ColorEdit4("Base Color", &settings.baseColor.x, ImGuiColorEditFlags_Float);
            changed |= ImGui::SliderFloat("Density Scale", &settings.densityScale, 0.0f, 100.0f);
            changed |= ImGui::ColorEdit3(
                "Shadow Density Factor",
                &settings.shadowDensityFactor.x,
                ImGuiColorEditFlags_Float | ImGuiColorEditFlags_DisplayHSV);
            changed |= ImGui::SliderFloat("Ambient Density", &settings.ambientDensity, 0.0f, 1.0f);
            ImGui::Separator();
            ImGui::Checkbox("Use Emissive", &volumeRenderer.useEmissive);
            changed |= ImGui::SliderFloat("Temperature Scale", &settings.temperatureScale, 0.0f, 2.0f);
            changed |= ImGui::SliderFloat("Emissive Strength", &settings.emissiveStrength, 0.0f, 1.0f);
            ImGui::Separator();
            changed |= ImGui::SliderInt("Use Phase Function", &settings.usePhaseFunction, 0, 1, "");
            changed |= ImGui::SliderFloat("Anisotropy", &settings.henyeyAnisotropy, -1.0f, 1.0f);

            if(changed)
            {
                volumeRenderer.updateSettingsBuffer();
            }
        }
        if(ImGui::CollapsingHeader("Post Process"))
        {
            ImGui::SliderFloat("Simple Exposure", &postProcessSettings.simpleExposure, 0.0f, 3.0f);
            ImGui::Combo("Aces fit", &postProcessSettings.mode, "KN\0SH\0");
        }
    }
    ImGui::End();
}