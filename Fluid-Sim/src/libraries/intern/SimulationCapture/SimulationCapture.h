#pragma once

#include <cstdint>

class Context;

/** Capture the simulations rendered frames. Works with a fixed timestep
 */
class SimulationCapture
{
  public:
    explicit SimulationCapture(Context& ctx);

    void handleFrame();

    void startCapturing();

    inline auto& getSettings()
    {
        return settings;
    }

  private:
    Context& ctx;

    struct RecordingSettings
    {
        bool isCapturing = false;
        int startTimeSeconds = 0;
        int64_t startFrame = 0;
        int durationSeconds = 12;
        int64_t endFrame = 0; // 1st frame that should not be captured anymore
        int framerate = 30;
        int simulationSubsteps = 4;
        float progress = 0.0f;
    } settings;
};
