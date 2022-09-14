#pragma once

#include <glad/glad/glad.h>

#include <stdint.h>
#include <vector>

#include <intern/Computepass/Computepass.h>
#include <intern/Context/Context.h>
#include <intern/Misc/GPUTimer.h>
#include <intern/ShaderProgram/ShaderProgram.h>
#include <intern/Texture/Texture3D.h>

class FluidSolver
{
  public:
    enum struct Precision
    {
        Half,
        Full
    };

    // copy ctor
    FluidSolver(const FluidSolver&) = delete;
    // move ctor
    FluidSolver(const FluidSolver&&) = delete;
    // rvalue binding
    FluidSolver(FluidSolver&&) = delete;
    // copy assign
    FluidSolver& operator=(const FluidSolver&) = delete;
    // move assign
    FluidSolver& operator=(const FluidSolver&&) = delete;

    FluidSolver(Context& ctx, GLsizei width, GLsizei height, GLsizei depth, Precision precision);

    void update();

    void clear();

    inline const Texture3D& getVelocityTexture() const
    {
        return velocityTexFront;
    }
    inline const Texture3D& getDensityTexture() const
    {
        return densityTexFront;
    }
    inline const Texture3D& getTemperatureTexture() const
    {
        return temperatureTexFront;
    }

    enum Timer
    {
        Advection = 0,
        Impulse,
        Buoyancy,
        Divergence,
        PressureSolve,
        PressureSubtraction,
        DivergenceRemainder,
        count
    };
    inline auto& getTimer()
    {
        return timer;
    }
    // inline std::array<GPUTimer<128>, Timer::count>& getTimers() {
    //     return timers;
    // }
    inline auto& getComponentTimers()
    {
        return componentTimers;
    }

    struct AdvectSettings
    {
        int32_t mode = 1;
        int32_t limiter = 2;
        // renderdoc complains if ubo binding < 16 bytes
        uint32_t misc[2] = {}; // NOLINT
    };
    enum PressureSolver : int
    {
        Jacobi = 0,
        RBGaussSeidel = 1,
        Multigrid = 2,
    };
    struct Settings
    {
        //---
        AdvectSettings advectSettings;
        bool useBFECCTemperature = false;
        bool useBFECCDensity = true;
        bool useBFECCVelocity = true;
        float temperatureDissipation = 0.0f;
        float densityDissipation = 0.005f;
        float velocityDissipation = 0.005f;
        // ---
        bool useLastFrameAsInitialGuess = false;
        PressureSolver solverMode = PressureSolver::Jacobi;
        uint16_t iterations = 10;
        struct LevelSettings
        {
            uint16_t preSmoothIterations = 10;
            uint16_t postSmoothIterations = 10;
            uint16_t iterations = 10;
        };
        uint16_t mgLevels = 1;
        std::vector<LevelSettings> mgLevelSettings;
        bool calculateRemainingDivergence = false;
    };

    struct Impulse
    {
        glm::vec3 position{0.f};
        float density{0.f};
        glm::vec3 velocity{0.f};
        float temperature{0.f};
        float size{0.f};
    };

    [[nodiscard]] inline Settings& getSettings()
    {
        return settings;
    }

    void updateSettingsBuffer();

    struct RemainingDivergence
    {
        float totalDivBefore;
        float totalDivBeforeInner;
        float totalDivAfter;
        float totalDivAfterInner;
    };
    [[nodiscard]] RemainingDivergence getRemainingDivergence() const;

    void setTransform(glm::mat4 mat);

    inline glm::mat4 getTransform() const
    {
        return localToWorld;
    }

    inline glm::mat4 getInvTransform() const
    {
        return worldToLocal;
    }

    void setImpulse(struct Impulse imp);

    inline int getLevels() const
    {
        return levels;
    }

  private:
    void advectQuantities();
    void addImpulse();
    void addBuoyancy();
    void calculateDivergence();
    void solvePressure();
    void subtractPressureGradient();

    Context& ctx;

    const GLsizei width = -1;
    const GLsizei height = -1;
    const GLsizei depth = -1;
    int levels = -1;

    // if more than just these few are needed do constexpr LUT
    const GLenum scalarInternalFormat = GL_R16F;
    const GLenum vectorInternalFormat = GL_RGBA16F;
    const char* scalarShaderFormat = "r16f";
    const char* vectorShaderFormat = "rgba16f";

    ShaderProgram vectorAdvectShader;
    ShaderProgram vectorBFECCBackwardsStepShader;
    ShaderProgram vectorBFECCForwardsStepShader;
    ShaderProgram scalarAdvectShader;
    ShaderProgram scalarBFECCBackwardsStepShader;
    ShaderProgram scalarBFECCForwardsStepShader;
    ShaderProgram impulseShader;
    ShaderProgram buoyancyShader;
    ShaderProgram divergenceShader;
    ShaderProgram divergenceMultigridShader;
    ShaderProgram jacobiShader;
    ShaderProgram jacobiMultigridShader;
    ShaderProgram gaussSeidelShader;
    ShaderProgram residualShader;
    ShaderProgram restrictShader;
    ShaderProgram correctShader;
    ShaderProgram pressureSubShader;
    ShaderProgram pressureSubMultigridShader;
    ShaderProgram divergenceRemainderShader;

    // todo: optimize, re-use textures for different passes
    //       use references to keep names for readability
    Texture3D velocityTexFront;
    Texture3D velocityTexBack;
    Texture3D vectorPhiTildeTex;
    Texture3D vectorPhiTilde2Tex;
    Texture3D densityTexFront;
    Texture3D densityTexBack;
    Texture3D temperatureTexFront;
    Texture3D temperatureTexBack;
    Texture3D scalarPhiTildeTex;
    Texture3D scalarPhiTilde2Tex;
    Texture3D divergenceTex;
    Texture3D pressureTexFront;
    Texture3D pressureTexBack;

    std::vector<Texture3D> lhsTexturesFront;
    std::vector<Texture3D> lhsTexturesBack;
    std::vector<Texture3D> rhsTextures;

    const Computepass advectVelocitySimplePass;
    const Computepass advectVelocityBFECCPass1;
    const Computepass advectVelocityBFECCPass2;
    const Computepass advectVelocityBFECCPass3;
    const Computepass advectDensitySimplePass;
    const Computepass advectDensityBFECCPass1;
    const Computepass advectDensityBFECCPass2;
    const Computepass advectDensityBFECCPass3;
    const Computepass advectTempSimplePass;
    const Computepass advectTempBFECCPass1;
    const Computepass advectTempBFECCPass2;
    const Computepass advectTempBFECCPass3;
    const Computepass buoyancyPass;
    const Computepass impulsePass;
    const Computepass divergencePass;
    const Computepass divergenceMultigridPass;
    const Computepass pressureSubPass;
    const Computepass pressureSubMultigridPass;
    const Computepass divergenceRemainderPass;

    Settings settings;

    glm::mat4 localToWorld{1.0f};
    glm::mat4 worldToLocal{1.0f};

    struct TimeVars
    {
        float deltaTime = 0.f;
        float pad[3] = {}; // NOLINT
    } timeVars;

    struct Impulse impulse;

    // todo: no buffer abstraction yet
    GLuint advectionSettingsUBO = 0xFFFFFFFF;
    GLuint timeVarsUBO = 0xFFFFFFFF;
    GLuint divergenceSSBOs[2] = {0xFFFFFFFF, 0xFFFFFFFF}; // NOLINT
    RemainingDivergence remainingDivergence{};

    GPUTimer<128> timer;
    std::array<GPUTimer<128>, Timer::count> componentTimers;

    uint8_t backBufferIndex = 0;
    uint8_t frontBufferIndex = 1;
};
