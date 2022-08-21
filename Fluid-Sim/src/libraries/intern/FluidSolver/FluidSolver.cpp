#include <glad/glad/glad.h>

#include <GLFW/glfw3.h>

#include <cstddef>
#include <format>

#include <intern/Computepass/Computepass.h>
#include <intern/FluidSolver/FluidSolver.h>
#include <intern/InputManager/InputManager.h>
#include <intern/Misc/Misc.h>
#include <intern/ShaderProgram/ShaderProgram.h>
#include <intern/Texture/Texture.h>

FluidSolver::FluidSolver(Context& ctx, GLsizei width, GLsizei height, GLsizei depth, Precision precision)
    : ctx(ctx), width(width), height(height), depth(depth),
      scalarInternalFormat(precision == Precision::Half ? GL_R16F : GL_R32F),
      vectorInternalFormat(precision == Precision::Half ? GL_RGBA16F : GL_RGBA32F),
      scalarShaderFormat(precision == Precision::Half ? "r16f" : "r32f"),
      vectorShaderFormat(precision == Precision::Half ? "rgba16f" : "rgba32f"),

      vectorAdvectShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/vectorAdvect.comp"}, {{"FORMAT", vectorShaderFormat}}),
      vectorBFECCBackwardsStepShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/vectorBFECCBackwardsStep.comp"},
          {{"FORMAT", vectorShaderFormat}}),
      vectorBFECCForwardsStepShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/vectorBFECCForwardsStep.comp"},
          {{"FORMAT", vectorShaderFormat}}),

      scalarAdvectShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/scalarAdvect.comp"}, {{"FORMAT", scalarShaderFormat}}),
      scalarBFECCBackwardsStepShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/scalarBFECCBackwardsStep.comp"},
          {{"FORMAT", scalarShaderFormat}}),
      scalarBFECCForwardsStepShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/scalarBFECCForwardsStep.comp"},
          {{"FORMAT", scalarShaderFormat}}),

      impulseShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/impulse.comp"},
          {{"SCALAR_FORMAT", scalarShaderFormat}, {"VECTOR_FORMAT", vectorShaderFormat}}),
      buoyancyShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/buoyancy.comp"}, {{"FORMAT", vectorShaderFormat}}),

      divergenceShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/divergence.comp"}, {{"FORMAT", scalarShaderFormat}}),
      jacobiShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/jacobi.comp"}, {{"FORMAT", scalarShaderFormat}}),
      gaussSeidelShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/rbGaussSeidel.comp"},
          {{"FORMAT", scalarShaderFormat}}),
      pressureSubShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/subPressureGrad.comp"},
          {{"FORMAT", vectorShaderFormat}}),

      divergenceRemainderShader(COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/divergenceRemainder.comp"}),

      velocityTex0({
          .name = "velocityTex0",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = vectorInternalFormat,
          .wrapS = GL_CLAMP_TO_EDGE,
          .wrapT = GL_CLAMP_TO_EDGE,
          .wrapR = GL_CLAMP_TO_EDGE,
      }),
      velocityTex1({
          .name = "velocityTex1",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = vectorInternalFormat,
          .wrapS = GL_CLAMP_TO_EDGE,
          .wrapT = GL_CLAMP_TO_EDGE,
          .wrapR = GL_CLAMP_TO_EDGE,
      }),
      vectorPhiTildeTex({
          .name = "phiTildeVector",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = vectorInternalFormat,
          .wrapS = GL_CLAMP_TO_EDGE,
          .wrapT = GL_CLAMP_TO_EDGE,
          .wrapR = GL_CLAMP_TO_EDGE,
      }),
      vectorPhiTilde2Tex({
          .name = "phiTilde2Vector",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = vectorInternalFormat,
          .wrapS = GL_CLAMP_TO_EDGE,
          .wrapT = GL_CLAMP_TO_EDGE,
          .wrapR = GL_CLAMP_TO_EDGE,
      }),
      densityTex0({
          .name = "densityTex0",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = scalarInternalFormat,
          .wrapS = GL_CLAMP_TO_EDGE,
          .wrapT = GL_CLAMP_TO_EDGE,
          .wrapR = GL_CLAMP_TO_EDGE,
      }),
      densityTex1({
          .name = "densityTex1",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = scalarInternalFormat,
          .wrapS = GL_CLAMP_TO_EDGE,
          .wrapT = GL_CLAMP_TO_EDGE,
          .wrapR = GL_CLAMP_TO_EDGE,
      }),
      temperatureTex0({
          .name = "temperatureTex0",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = scalarInternalFormat,
          .wrapS = GL_CLAMP_TO_EDGE,
          .wrapT = GL_CLAMP_TO_EDGE,
          .wrapR = GL_CLAMP_TO_EDGE,
      }),
      temperatureTex1({
          .name = "temperatureTex1",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = scalarInternalFormat,
          .wrapS = GL_CLAMP_TO_EDGE,
          .wrapT = GL_CLAMP_TO_EDGE,
          .wrapR = GL_CLAMP_TO_EDGE,
      }),
      scalarPhiTildeTex({
          .name = "phiTildeScalar",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = scalarInternalFormat,
          .wrapS = GL_CLAMP_TO_EDGE,
          .wrapT = GL_CLAMP_TO_EDGE,
          .wrapR = GL_CLAMP_TO_EDGE,
      }),
      scalarPhiTilde2Tex({
          .name = "phiTilde2Scalar",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = scalarInternalFormat,
          .wrapS = GL_CLAMP_TO_EDGE,
          .wrapT = GL_CLAMP_TO_EDGE,
          .wrapR = GL_CLAMP_TO_EDGE,
      }),
      divergenceTex({
          .name = "divergenceTex",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = scalarInternalFormat,
          .wrapS = GL_CLAMP_TO_EDGE,
          .wrapT = GL_CLAMP_TO_EDGE,
          .wrapR = GL_CLAMP_TO_EDGE,
      }),
      pressureTex0({
          .name = "pressureTex0",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = scalarInternalFormat,
          .wrapS = GL_CLAMP_TO_EDGE,
          .wrapT = GL_CLAMP_TO_EDGE,
          .wrapR = GL_CLAMP_TO_EDGE,
      }),
      pressureTex1({
          .name = "pressureTex1",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = scalarInternalFormat,
          .wrapS = GL_CLAMP_TO_EDGE,
          .wrapT = GL_CLAMP_TO_EDGE,
          .wrapR = GL_CLAMP_TO_EDGE,
      }),

      advectVelocitySimplePass(ComputepassDesc{
          .name = "Advect Velocity Simple",
          .shaderPtr = &vectorAdvectShader,
          .numX = UintDivAndCeil(width, 16),
          .numY = UintDivAndCeil(height, 16),
          .numZ = static_cast<uint32_t>(depth),
          .barriers = GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
          .textureBindings =
              {{.unit = 0, .texture = &velocityTex0}, //
               {.unit = 1, .texture = &velocityTex0}},
          .imageBindings =
              {{.unit = 0,
                .texture = &velocityTex1,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = vectorInternalFormat}}}),

      advectVelocityBFECCPass1(ComputepassDesc{
          .name = "Advect Velocity BFECC Step 1",
          .shaderPtr = &vectorAdvectShader,
          .numX = UintDivAndCeil(width, 16),
          .numY = UintDivAndCeil(height, 16),
          .numZ = static_cast<uint32_t>(depth),
          .barriers = GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
          .textureBindings =
              {{.unit = 0, .texture = &velocityTex0}, //
               {.unit = 1, .texture = &velocityTex0}},
          .imageBindings =
              {{.unit = 0,
                .texture = &vectorPhiTildeTex,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = vectorInternalFormat}}}),

      advectVelocityBFECCPass2(ComputepassDesc{
          .name = "Advect Velocity BFECC Step 2",
          .shaderPtr = &vectorBFECCBackwardsStepShader,
          .numX = UintDivAndCeil(width, 16),
          .numY = UintDivAndCeil(height, 16),
          .numZ = static_cast<uint32_t>(depth),
          .barriers = GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
          .textureBindings =
              {{.unit = 0, .texture = &velocityTex0}, //
               {.unit = 1, .texture = &velocityTex0}, //
               {.unit = 2, .texture = &vectorPhiTildeTex}},
          .imageBindings =
              {{.unit = 0,
                .texture = &vectorPhiTilde2Tex,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = vectorInternalFormat}}}),

      advectVelocityBFECCPass3(ComputepassDesc{
          .name = "Advect Velocity BFECC Step 3",
          .shaderPtr = &vectorBFECCForwardsStepShader,
          .numX = UintDivAndCeil(width, 16),
          .numY = UintDivAndCeil(height, 16),
          .numZ = static_cast<uint32_t>(depth),
          .barriers = GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
          .textureBindings =
              {{.unit = 0, .texture = &velocityTex0}, //
               {.unit = 1, .texture = &velocityTex0}, //
               {.unit = 2, .texture = &vectorPhiTilde2Tex}},
          .imageBindings =
              {{.unit = 0,
                .texture = &velocityTex1,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = vectorInternalFormat}}}),

      advectDensitySimplePass(ComputepassDesc{
          .name = "Advect Density Simple",
          .shaderPtr = &scalarAdvectShader,
          .numX = UintDivAndCeil(width, 16),
          .numY = UintDivAndCeil(height, 16),
          .numZ = static_cast<uint32_t>(depth),
          .barriers = GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
          .textureBindings =
              {{.unit = 0, .texture = &velocityTex0}, //
               {.unit = 1, .texture = &densityTex0}},
          .imageBindings =
              {{.unit = 0,
                .texture = &densityTex1,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = scalarInternalFormat}}}),

      advectDensityBFECCPass1(ComputepassDesc{
          .name = "Advect Density BFECC Step 1",
          .shaderPtr = &scalarAdvectShader,
          .numX = UintDivAndCeil(width, 16),
          .numY = UintDivAndCeil(height, 16),
          .numZ = static_cast<uint32_t>(depth),
          .barriers = GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
          .textureBindings =
              {{.unit = 0, .texture = &velocityTex0}, //
               {.unit = 1, .texture = &densityTex0}},
          .imageBindings =
              {{.unit = 0,
                .texture = &scalarPhiTildeTex,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = scalarInternalFormat}}}),

      advectDensityBFECCPass2(ComputepassDesc{
          .name = "Advect Density BFECC Step 2",
          .shaderPtr = &scalarBFECCBackwardsStepShader,
          .numX = UintDivAndCeil(width, 16),
          .numY = UintDivAndCeil(height, 16),
          .numZ = static_cast<uint32_t>(depth),
          .barriers = GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
          .textureBindings =
              {{.unit = 0, .texture = &velocityTex0}, //
               {.unit = 1, .texture = &densityTex0},  //
               {.unit = 2, .texture = &scalarPhiTildeTex}},
          .imageBindings =
              {{.unit = 0,
                .texture = &scalarPhiTilde2Tex,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = scalarInternalFormat}}}),

      advectDensityBFECCPass3(ComputepassDesc{
          .name = "Advect Density BFECC Step 3",
          .shaderPtr = &scalarBFECCForwardsStepShader,
          .numX = UintDivAndCeil(width, 16),
          .numY = UintDivAndCeil(height, 16),
          .numZ = static_cast<uint32_t>(depth),
          .barriers = GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
          .textureBindings =
              {{.unit = 0, .texture = &velocityTex0}, //
               {.unit = 1, .texture = &densityTex0},  //
               {.unit = 2, .texture = &scalarPhiTilde2Tex}},
          .imageBindings =
              {{.unit = 0,
                .texture = &densityTex1,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = scalarInternalFormat}}}),

      advectTempSimplePass(ComputepassDesc{
          .name = "Advect Temperature Simple",
          .shaderPtr = &scalarAdvectShader,
          .numX = UintDivAndCeil(width, 16),
          .numY = UintDivAndCeil(height, 16),
          .numZ = static_cast<uint32_t>(depth),
          .barriers = GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
          .textureBindings =
              {{.unit = 0, .texture = &velocityTex0}, //
               {.unit = 1, .texture = &temperatureTex0}},
          .imageBindings =
              {{.unit = 0,
                .texture = &temperatureTex1,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = scalarInternalFormat}}}),

      advectTempBFECCPass1(ComputepassDesc{
          .name = "Advect Temperature BFECC Step 1",
          .shaderPtr = &scalarAdvectShader,
          .numX = UintDivAndCeil(width, 16),
          .numY = UintDivAndCeil(height, 16),
          .numZ = static_cast<uint32_t>(depth),
          .barriers = GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
          .textureBindings =
              {{.unit = 0, .texture = &velocityTex0}, //
               {.unit = 1, .texture = &temperatureTex0}},
          .imageBindings =
              {{.unit = 0,
                .texture = &scalarPhiTildeTex,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = scalarInternalFormat}}}),

      advectTempBFECCPass2(ComputepassDesc{
          .name = "Advect Temperature BFECC Step 2",
          .shaderPtr = &scalarBFECCBackwardsStepShader,
          .numX = UintDivAndCeil(width, 16),
          .numY = UintDivAndCeil(height, 16),
          .numZ = static_cast<uint32_t>(depth),
          .barriers = GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
          .textureBindings =
              {{.unit = 0, .texture = &velocityTex0},    //
               {.unit = 1, .texture = &temperatureTex0}, //
               {.unit = 2, .texture = &scalarPhiTildeTex}},
          .imageBindings =
              {{.unit = 0,
                .texture = &scalarPhiTilde2Tex,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = scalarInternalFormat}}}),

      advectTempBFECCPass3(ComputepassDesc{
          .name = "Advect Temperature BFECC Step 3",
          .shaderPtr = &scalarBFECCForwardsStepShader,
          .numX = UintDivAndCeil(width, 16),
          .numY = UintDivAndCeil(height, 16),
          .numZ = static_cast<uint32_t>(depth),
          .barriers = GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
          .textureBindings =
              {{.unit = 0, .texture = &velocityTex0},    //
               {.unit = 1, .texture = &temperatureTex0}, //
               {.unit = 2, .texture = &scalarPhiTilde2Tex}},
          .imageBindings =
              {{.unit = 0,
                .texture = &temperatureTex1,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = scalarInternalFormat}}}),

      buoyancyPass(ComputepassDesc{
          .name = "Buoyancy",
          .shaderPtr = &buoyancyShader,
          .numX = UintDivAndCeil(width, 16),
          .numY = UintDivAndCeil(height, 16),
          .numZ = static_cast<uint32_t>(depth),
          .barriers = GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
          .textureBindings =
              {{.unit = 0, .texture = &velocityTex0},
               {.unit = 1, .texture = &densityTex0},
               {.unit = 2, .texture = &temperatureTex0}},
          .imageBindings =
              {{.unit = 0,
                .texture = &velocityTex1,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = vectorInternalFormat}}}),

      impulsePass(ComputepassDesc{
          .name = "Impulse",
          .shaderPtr = &impulseShader,
          .numX = UintDivAndCeil(width, 16),
          .numY = UintDivAndCeil(height, 16),
          .numZ = static_cast<uint32_t>(depth),
          .barriers = GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
          .textureBindings =
              {{.unit = 0, .texture = &velocityTex0},
               {.unit = 1, .texture = &temperatureTex0},
               {.unit = 2, .texture = &densityTex0}},
          .imageBindings =
              {{.unit = 0,
                .texture = &velocityTex1,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = vectorInternalFormat},
               {.unit = 1,
                .texture = &temperatureTex1,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = scalarInternalFormat},
               {.unit = 2,
                .texture = &densityTex1,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = scalarInternalFormat}}}),

      divergencePass(ComputepassDesc{
          .name = "Divergence",
          .shaderPtr = &divergenceShader,
          .numX = UintDivAndCeil(width, 16),
          .numY = UintDivAndCeil(height, 16),
          .numZ = static_cast<uint32_t>(depth),
          .barriers = GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
          .textureBindings = {{.unit = 0, .texture = &velocityTex0}},
          .imageBindings =
              {{.unit = 0,
                .texture = &divergenceTex,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = scalarInternalFormat}}}),

      pressureSubPass(ComputepassDesc{
          .name = "Pressure Sub",
          .shaderPtr = &pressureSubShader,
          .numX = UintDivAndCeil(width, 16),
          .numY = UintDivAndCeil(height, 16),
          .numZ = static_cast<uint32_t>(depth),
          .barriers = GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
          .textureBindings =
              {{.unit = 0, .texture = &pressureTex0}, //
               {.unit = 1, .texture = &velocityTex0}},
          .imageBindings =
              {{.unit = 0,
                .texture = &velocityTex1,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = vectorInternalFormat}}}),

      divergenceRemainderPass(ComputepassDesc{
          .name = "Divergence Remainder",
          .shaderPtr = &divergenceRemainderShader,
          .numX = UintDivAndCeil(width, 16),
          .numY = UintDivAndCeil(height, 16),
          .numZ = static_cast<uint32_t>(depth),
          .barriers = GL_BUFFER_UPDATE_BARRIER_BIT,
          .textureBindings = {{.unit = 0, .texture = &velocityTex0}}})

{
    // not allocating these as 1 texture with multiple mips since then .swap() would swap all levels
    // (since that just swaps the internal OGL object) which I dont want that to happen
    // for now just assume sizes are all POT
    int levels = int(log2(std::max(width, std::max(depth, height)))) + 1;
    divergenceTextures.reserve(levels);
    pressureTextures0.reserve(levels);
    pressureTextures1.reserve(levels);

    for(int i = 0; i < levels; i++)
    {
        int levelWidth = width / (1 << i);
        int levelHeight = height / (1 << i);
        int levelDepth = depth / (1 << i);
        divergenceTextures.emplace_back(TextureDesc{
            .name = std::format("divergenceTexLevel{}", i).c_str(),
            .width = width,
            .height = height,
            .depth = depth,
            .internalFormat = scalarInternalFormat,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .wrapR = GL_CLAMP_TO_EDGE,
        });
        divergenceTextures.emplace_back(TextureDesc{
            .name = std::format("divergenceTexLevel{}", i).c_str(),
            .width = width,
            .height = height,
            .depth = depth,
            .internalFormat = scalarInternalFormat,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .wrapR = GL_CLAMP_TO_EDGE,
        });
        divergenceTextures.emplace_back(TextureDesc{
            .name = std::format("divergenceTexLevel{}", i).c_str(),
            .width = width,
            .height = height,
            .depth = depth,
            .internalFormat = scalarInternalFormat,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .wrapR = GL_CLAMP_TO_EDGE,
        });
    }

    // not part of compute bindinds since these calls need to be made only once
    //(total amount of buffers is low enough that no binding point has to be reused)

    glCreateBuffers(1, &timeVarsUBO);
    glNamedBufferStorage(timeVarsUBO, sizeof(TimeVars), &timeVars, GL_DYNAMIC_STORAGE_BIT);
    glObjectLabel(GL_BUFFER, timeVarsUBO, -1, "Time Vars UBO");
    glBindBufferBase(GL_UNIFORM_BUFFER, 10, timeVarsUBO);

    glCreateBuffers(1, &advectionSettingsUBO);
    glNamedBufferStorage(
        advectionSettingsUBO, sizeof(AdvectSettings), &settings.advectSettings, GL_DYNAMIC_STORAGE_BIT);
    glObjectLabel(GL_BUFFER, advectionSettingsUBO, -1, "Advection Settings UBO");
    glBindBufferBase(GL_UNIFORM_BUFFER, 11, advectionSettingsUBO);

    glCreateBuffers(2, &divergenceSSBOs[0]);
    glNamedBufferStorage(divergenceSSBOs[0], sizeof(GLfloat), nullptr, 0);
    glObjectLabel(GL_BUFFER, divergenceSSBOs[0], -1, "Remaining Divergence SSBO0");
    glNamedBufferStorage(divergenceSSBOs[1], sizeof(GLfloat), nullptr, 0);
    glObjectLabel(GL_BUFFER, divergenceSSBOs[1], -1, "Remaining Divergence SSBO1");
    GLfloat zero = 0;
    glClearNamedBufferData(divergenceSSBOs[0], GL_R32F, GL_RED, GL_FLOAT, &zero);
    glClearNamedBufferData(divergenceSSBOs[1], GL_R32F, GL_RED, GL_FLOAT, &zero);
    // glBindBufferBase(GL_SHADER_BIT_STORAGE_BUFFER, 10, divergenceSSBO);

    clear();
}

void FluidSolver::clear()
{
    const glm::vec4 clear = glm::vec4(0.0f);
    glClearTexImage(velocityTex0.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(velocityTex1.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(vectorPhiTilde2Tex.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(vectorPhiTildeTex.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(densityTex0.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(densityTex1.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(temperatureTex0.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(temperatureTex1.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(scalarPhiTildeTex.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(scalarPhiTilde2Tex.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(divergenceTex.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(pressureTex0.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(pressureTex1.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
}

void FluidSolver::update()
{
    InputManager* input = ctx.getInputManager();

    timer.start();

    // update time dependant variables
    timeVars.deltaTime = input->getSimulationDeltaTime();
    // glInvalidateBufferData(timeVarsUBO);
    glNamedBufferSubData(timeVarsUBO, 0, sizeof(TimeVars), &timeVars);

    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Fluid Solver");

    componentTimers[Advection].start();
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Advection");

    // dissipation factors are authored based on 60fps simulation, so calculate the appropriate
    // factor for the current framerate here
    float temperatureFactor = pow(1.0f - settings.temperatureDissipation, 60 * timeVars.deltaTime);
    float densityFactor = pow(1.0f - settings.densityDissipation, 60 * timeVars.deltaTime);
    float velocityFactor = pow(1.0f - settings.velocityDissipation, 60 * timeVars.deltaTime);

    if(settings.useBFECCTemperature)
    {
        advectTempBFECCPass1.startMarker();
        advectTempBFECCPass1.bind();
        glUniform1f(0, 1.0f); // Location specified in shader
        advectTempBFECCPass1.run();
        advectTempBFECCPass1.endMarker();

        advectTempBFECCPass2.execute();

        advectTempBFECCPass3.startMarker();
        advectTempBFECCPass3.bind();
        glUniform1f(0, temperatureFactor); // Location specified in shader
        advectTempBFECCPass3.run();
        advectTempBFECCPass3.endMarker();
    }
    else
    {
        advectTempSimplePass.startMarker();
        advectTempSimplePass.bind();
        glUniform1f(0, temperatureFactor); // Location specified in shader
        advectTempSimplePass.run();
        advectTempSimplePass.endMarker();
    }
    temperatureTex0.swap(temperatureTex1);

    if(settings.useBFECCDensity)
    {
        advectDensityBFECCPass1.startMarker();
        advectDensityBFECCPass1.bind();
        glUniform1f(0, 1.0f); // Location specified in shader
        advectDensityBFECCPass1.run();
        advectDensityBFECCPass1.endMarker();

        advectDensityBFECCPass2.execute();

        advectDensityBFECCPass3.startMarker();
        advectDensityBFECCPass3.bind();
        glUniform1f(0, densityFactor); // Location specified in shader
        advectDensityBFECCPass3.run();
        advectDensityBFECCPass3.endMarker();
    }
    else
    {
        advectDensitySimplePass.startMarker();
        advectDensitySimplePass.bind();
        glUniform1f(0, densityFactor); // Location specified in shader
        advectDensitySimplePass.run();
        advectDensitySimplePass.endMarker();
    }
    densityTex0.swap(densityTex1);

    if(settings.useBFECCVelocity)
    {
        advectVelocityBFECCPass1.startMarker();
        advectVelocityBFECCPass1.bind();
        glUniform1f(0, 1.0f); // Location specified in shader
        advectVelocityBFECCPass1.run();
        advectVelocityBFECCPass1.endMarker();

        advectVelocityBFECCPass2.execute();

        advectVelocityBFECCPass3.startMarker();
        advectVelocityBFECCPass3.bind();
        glUniform1f(0, velocityFactor); // Location specified in shader
        advectVelocityBFECCPass3.run();
        advectVelocityBFECCPass3.endMarker();
    }
    else
    {
        advectVelocitySimplePass.startMarker();
        advectVelocitySimplePass.bind();
        glUniform1f(0, velocityFactor); // Location specified in shader
        advectVelocitySimplePass.run();
        advectVelocitySimplePass.endMarker();
    }
    velocityTex0.swap(velocityTex1);
    glPopDebugGroup();

    componentTimers[Advection].end();

    componentTimers[Impulse].start();
    impulsePass.startMarker();
    impulsePass.bind();
    // todo: UBO
    glUniform3fv(0, 1, &impulse.position.x);
    glUniform1f(1, impulse.density);
    glUniform3fv(2, 1, &impulse.velocity.x);
    glUniform1f(3, impulse.temperature);
    glUniform1f(4, impulse.size);
    impulsePass.run();
    impulsePass.endMarker();
    velocityTex0.swap(velocityTex1);
    temperatureTex0.swap(temperatureTex1);
    densityTex0.swap(densityTex1);
    componentTimers[Impulse].end();

    componentTimers[Buoyancy].start();
    buoyancyPass.execute();
    velocityTex0.swap(velocityTex1);
    componentTimers[Buoyancy].end();

    componentTimers[Divergence].start();
    divergencePass.execute();
    componentTimers[Divergence].end();

    componentTimers[PressureSolve].start();
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Pressure Solve");
    // not using passes for these since they dont work well with loops yet (eg lots of
    // duplicate/unnecessary bindings)
    if(!settings.useLastFrameAsInitialGuess)
    {
        const glm::vec4 clear = glm::vec4(0.0f);
        glClearTexImage(pressureTex0.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    }
    if(settings.solverMode == 0)
    {
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Jacobi");
        jacobiShader.useProgram();
        glBindTextureUnit(1, divergenceTex.getTextureID());
        for(decltype(settings.iterations) i = 0; i < settings.iterations; i++)
        {
            glBindImageTexture(
                0, pressureTex1.getTextureID(), 0, GL_TRUE, 0, GL_WRITE_ONLY, scalarInternalFormat);
            glBindTextureUnit(0, pressureTex0.getTextureID());
            glDispatchCompute(
                UintDivAndCeil(width, 16), UintDivAndCeil(height, 16), static_cast<uint32_t>(depth));
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
            pressureTex0.swap(pressureTex1);
        }
        glPopDebugGroup();
    }
    else if(settings.solverMode == 1)
    {
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Gauss Seidel");
        gaussSeidelShader.useProgram();
        glBindImageTexture(
            0, pressureTex0.getTextureID(), 0, GL_TRUE, 0, GL_READ_WRITE, scalarInternalFormat);
        glBindTextureUnit(0, divergenceTex.getTextureID());
        for(decltype(settings.iterations) i = 0; i < settings.iterations; i += 2)
        {
            glUniform1ui(0, 0U); // Location specified in shader
            glDispatchCompute(
                UintDivAndCeil(width, 16), UintDivAndCeil(height, 16), static_cast<uint32_t>(depth / 2));

            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

            glUniform1ui(0, 1U); // Location specified in shader
            glDispatchCompute(
                UintDivAndCeil(width, 16), UintDivAndCeil(height, 16), static_cast<uint32_t>(depth / 2));

            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        }
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
        glPopDebugGroup();
    }
    glPopDebugGroup();
    componentTimers[PressureSolve].end();

    componentTimers[PressureSubtraction].start();
    pressureSubPass.execute();
    velocityTex0.swap(velocityTex1);
    componentTimers[PressureSubtraction].end();

    if(settings.calculateRemainingDivergence)
    {
        componentTimers[DivergenceRemainder].start();
        GLfloat zero = 0;
        glClearNamedBufferData(divergenceSSBOs[frontBufferIndex], GL_R32F, GL_RED, GL_FLOAT, &zero);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, divergenceSSBOs[frontBufferIndex]);
        divergenceRemainderPass.execute();
        // todo: make more efficient, but low prio since not necessary for solver
        glGetNamedBufferSubData(divergenceSSBOs[backBufferIndex], 0, sizeof(GLfloat), &remainingDivergence);
        componentTimers[DivergenceRemainder].end();
    }

    glPopDebugGroup();

    timer.end();

    frontBufferIndex = !frontBufferIndex;
    backBufferIndex = !backBufferIndex;
}

void FluidSolver::updateSettingsBuffer()
{
    // todo: just subdata vs invalidate + subdata vs map with invalidate
    //       doesnt seem to matter for such a small buffer
    glNamedBufferSubData(timeVarsUBO, 0, sizeof(AdvectSettings), &settings.advectSettings);
}

float FluidSolver::getRemainingDivergence() const
{
    // return static_cast<float>(remainingDivergence) / 65536.0f;
    return remainingDivergence;
}

void FluidSolver::setTransform(glm::mat4 mat)
{
    localToWorld = mat;
    worldToLocal = glm::inverse(mat);
}

void FluidSolver::setImpulse(struct Impulse imp)
{
    impulse = imp;
}