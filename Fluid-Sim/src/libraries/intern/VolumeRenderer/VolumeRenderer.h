#pragma once

#include <glm/glm.hpp>

#include <intern/FluidSolver/FluidSolver.h>
#include <intern/Mesh/VolumeCube.h>
#include <intern/Misc/GPUTimer.h>
#include <intern/Texture/Texture.h>

class VolumeRenderer
{
  public:
    // todo: re-enable depthbuffer and blending with scene depth
    explicit VolumeRenderer(Context& ctx, const FluidSolver& fluidSolver /*, const Texture& depthbuffer*/);

    void render();

    [[nodiscard]] inline GPUTimer<128>& getTimer()
    {
        return timer;
    }

    struct Settings
    {
        int planeAlignment = 0; // boolean
        int maxSteps = 64;
        int hardStepsLimit = 256;
        int shadowSteps = 16;

        float jitter = 1;
        int noiseType = 1;            // 0 = white, 1 = blue
        float henyeyAnisotropy = 0.2; //[-1,1]
        float ambientDensity = 0.7;

        glm::vec4 baseColor = glm::vec4(1.0);

        glm::vec3 lightVectorLocal = normalize(glm::vec3(0.0f, 1.0f, 0.0f));
        float densityScale = 41.0;

        glm::vec4 sunColorAndStrength = glm::vec4(1.0, 1.0, 1.0, 1.0);

        glm::vec3 shadowDensityFactor = glm::vec3(8);
        int usePhaseFunction = 0; //"boolean"

        glm::vec4 skyColorAndStrength = glm::vec4(0.2, 0.2, 0.25, 1.0);

        float temperatureScale = 1.0f;
        float emissiveStrength = 1.0f;
    };

    inline Settings& getSettings()
    {
        return settings;
    }

    void updateSettingsBuffer();

    void saveSettings(const std::string& file);
    void loadSettings(const std::string& file);

    bool useEmissive = false;

  private:
    Context& ctx;

    const FluidSolver& fluidSolver;
    // const Texture& depthbuffer;

    Settings settings;
    GLuint settingsUBO;

    GPUTimer<128> timer;

    VolumeCube cube;
    ShaderProgram volumeRenderShader{
        VERTEX_SHADER_BIT | FRAGMENT_SHADER_BIT,
        {SHADERS_PATH "/FluidSim/volumeRender.vert", SHADERS_PATH "/FluidSim/volumeRender.frag"}};
    ShaderProgram volumeRenderEmissiveShader{
        VERTEX_SHADER_BIT | FRAGMENT_SHADER_BIT,
        {SHADERS_PATH "/FluidSim/volumeRender.vert", SHADERS_PATH "/FluidSim/volumeRender.frag"},
        {{"USE_EMISSIVE", "1"}}};
    Texture noiseTex{MISC_PATH "/textures/LDR_RGBA_0.png", false};
    GLuint borderSampler = 0xFFFFFFFF;
};