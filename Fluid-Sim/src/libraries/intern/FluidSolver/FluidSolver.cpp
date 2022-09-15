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
      levels(int(log2(std::max(width, std::max(depth, height)))) + 1),
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
      divergenceMultigridShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/divergenceMultigrid.comp"},
          {{"FORMAT", scalarShaderFormat}}),
      jacobiShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/jacobi.comp"}, {{"FORMAT", scalarShaderFormat}}),
      jacobiMultigridShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/jacobiMultigrid.comp"},
          {{"FORMAT", scalarShaderFormat}}),
      gaussSeidelShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/rbGaussSeidel.comp"},
          {{"FORMAT", scalarShaderFormat}}),
      residualShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/calcResidual.comp"}, {{"FORMAT", scalarShaderFormat}}),
      restrictShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/restrict.comp"}, {{"FORMAT", scalarShaderFormat}}),
      correctShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/correct.comp"}, {{"FORMAT", scalarShaderFormat}}),
      pressureSubShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/subPressureGrad.comp"},
          {{"FORMAT", vectorShaderFormat}}),
      pressureSubMultigridShader(
          COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/subPressureGradMultigrid.comp"},
          {{"FORMAT", vectorShaderFormat}}),

      divergenceRemainderShader(COMPUTE_SHADER_BIT, {SHADERS_PATH "/FluidSim/divergenceRemainder.comp"}),

      velocityTexFront({
          .name = "velocityTexFront",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = vectorInternalFormat,
      }),
      velocityTexBack({
          .name = "velocityTexBack",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = vectorInternalFormat,
      }),
      vectorPhiTildeTex({
          .name = "phiTildeVector",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = vectorInternalFormat,
      }),
      vectorPhiTilde2Tex({
          .name = "phiTilde2Vector",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = vectorInternalFormat,
      }),
      densityTexFront({
          .name = "densityTexFront",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = scalarInternalFormat,
      }),
      densityTexBack({
          .name = "densityTexBack",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = scalarInternalFormat,
      }),
      temperatureTexFront({
          .name = "temperatureTexFront",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = scalarInternalFormat,
      }),
      temperatureTexBack({
          .name = "temperatureTexBack",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = scalarInternalFormat,
      }),
      scalarPhiTildeTex({
          .name = "phiTildeScalar",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = scalarInternalFormat,

      }),
      scalarPhiTilde2Tex({
          .name = "phiTilde2Scalar",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = scalarInternalFormat,
      }),
      divergenceTex({
          .name = "divergenceTex",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = scalarInternalFormat,
      }),
      pressureTexFront({
          .name = "pressureTexFront",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = scalarInternalFormat,
      }),
      pressureTexBack({
          .name = "pressureTexBack",
          .width = width,
          .height = height,
          .depth = depth,
          .internalFormat = scalarInternalFormat,
      }),

      lhsTexturesFront(levels), lhsTexturesBack(levels), rhsTextures(levels),

      advectVelocitySimplePass(ComputepassDesc{
          .name = "Advect Velocity Simple",
          .shaderPtr = &vectorAdvectShader,
          .numX = UintDivAndCeil(width, 16),
          .numY = UintDivAndCeil(height, 16),
          .numZ = static_cast<uint32_t>(depth),
          .barriers = GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
          .textureBindings =
              {{.unit = 0, .texture = &velocityTexFront}, //
               {.unit = 1, .texture = &velocityTexFront}},
          .imageBindings =
              {{.unit = 0,
                .texture = &velocityTexBack,
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
              {{.unit = 0, .texture = &velocityTexFront}, //
               {.unit = 1, .texture = &velocityTexFront}},
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
              {{.unit = 0, .texture = &velocityTexFront}, //
               {.unit = 1, .texture = &velocityTexFront}, //
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
              {{.unit = 0, .texture = &velocityTexFront}, //
               {.unit = 1, .texture = &velocityTexFront}, //
               {.unit = 2, .texture = &vectorPhiTilde2Tex}},
          .imageBindings =
              {{.unit = 0,
                .texture = &velocityTexBack,
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
              {{.unit = 0, .texture = &velocityTexFront}, //
               {.unit = 1, .texture = &densityTexFront}},
          .imageBindings =
              {{.unit = 0,
                .texture = &densityTexBack,
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
              {{.unit = 0, .texture = &velocityTexFront}, //
               {.unit = 1, .texture = &densityTexFront}},
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
              {{.unit = 0, .texture = &velocityTexFront}, //
               {.unit = 1, .texture = &densityTexFront},  //
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
              {{.unit = 0, .texture = &velocityTexFront}, //
               {.unit = 1, .texture = &densityTexFront},  //
               {.unit = 2, .texture = &scalarPhiTilde2Tex}},
          .imageBindings =
              {{.unit = 0,
                .texture = &densityTexBack,
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
              {{.unit = 0, .texture = &velocityTexFront}, //
               {.unit = 1, .texture = &temperatureTexFront}},
          .imageBindings =
              {{.unit = 0,
                .texture = &temperatureTexBack,
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
              {{.unit = 0, .texture = &velocityTexFront}, //
               {.unit = 1, .texture = &temperatureTexFront}},
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
              {{.unit = 0, .texture = &velocityTexFront},    //
               {.unit = 1, .texture = &temperatureTexFront}, //
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
              {{.unit = 0, .texture = &velocityTexFront},    //
               {.unit = 1, .texture = &temperatureTexFront}, //
               {.unit = 2, .texture = &scalarPhiTilde2Tex}},
          .imageBindings =
              {{.unit = 0,
                .texture = &temperatureTexBack,
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
              {{.unit = 0, .texture = &velocityTexFront},
               {.unit = 1, .texture = &densityTexFront},
               {.unit = 2, .texture = &temperatureTexFront}},
          .imageBindings =
              {{.unit = 0,
                .texture = &velocityTexBack,
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
              {{.unit = 0, .texture = &velocityTexFront},
               {.unit = 1, .texture = &temperatureTexFront},
               {.unit = 2, .texture = &densityTexFront}},
          .imageBindings =
              {{.unit = 0,
                .texture = &velocityTexBack,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = vectorInternalFormat},
               {.unit = 1,
                .texture = &temperatureTexBack,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = scalarInternalFormat},
               {.unit = 2,
                .texture = &densityTexBack,
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
          .textureBindings = {{.unit = 0, .texture = &velocityTexFront}},
          .imageBindings =
              {{.unit = 0,
                .texture = &divergenceTex,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = scalarInternalFormat}}}),

      divergenceMultigridPass(ComputepassDesc{
          .name = "Divergence",
          .shaderPtr = &divergenceMultigridShader,
          .numX = UintDivAndCeil(width + 1, 16),
          .numY = UintDivAndCeil(height + 1, 16),
          .numZ = static_cast<uint32_t>(depth + 1),
          .barriers = GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
          .textureBindings = {{.unit = 0, .texture = &velocityTexFront}},
          .imageBindings =
              {{.unit = 0,
                .texture = &rhsTextures[0],
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
              {{.unit = 0, .texture = &pressureTexFront}, //
               {.unit = 1, .texture = &velocityTexFront}},
          .imageBindings =
              {{.unit = 0,
                .texture = &velocityTexBack,
                .layered = GL_TRUE,
                .access = GL_WRITE_ONLY,
                .format = vectorInternalFormat}}}),

      pressureSubMultigridPass(ComputepassDesc{
          .name = "Pressure Sub",
          .shaderPtr = &pressureSubMultigridShader,
          .numX = UintDivAndCeil(width, 16),
          .numY = UintDivAndCeil(height, 16),
          .numZ = static_cast<uint32_t>(depth),
          .barriers = GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
          .textureBindings =
              {{.unit = 0, .texture = &lhsTexturesFront[0]}, //
               {.unit = 1, .texture = &velocityTexFront}},
          .imageBindings =
              {{.unit = 0,
                .texture = &velocityTexBack,
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
          .textureBindings = {{.unit = 0, .texture = &velocityTexFront}}})

{
    // for now just assume sizes are all POT

    for(int i = 0; i < levels; i++)
    {
        int levelWidth = std::max(width / (1 << i), 1);
        int levelHeight = std::max(height / (1 << i), 1);
        int levelDepth = std::max(depth / (1 << i), 1);
        rhsTextures[i] = Texture3D{
            {.name = std::format("rhsTexture_level{}", i).c_str(),
             .width = levelWidth + 1,
             .height = levelHeight + 1,
             .depth = levelDepth + 1,
             .internalFormat = scalarInternalFormat,
             .minFilter = GL_LINEAR,
             .magFilter = GL_LINEAR,
             .wrapS = GL_MIRRORED_REPEAT,
             .wrapT = GL_MIRRORED_REPEAT,
             .wrapR = GL_MIRRORED_REPEAT}};
        lhsTexturesFront[i] = Texture3D{
            {.name = std::format("lhsTexture0_level{}", i).c_str(),
             .width = levelWidth + 1,
             .height = levelHeight + 1,
             .depth = levelDepth + 1,
             .internalFormat = scalarInternalFormat,
             .minFilter = GL_LINEAR,
             .magFilter = GL_LINEAR,
             .wrapS = GL_MIRRORED_REPEAT,
             .wrapT = GL_MIRRORED_REPEAT,
             .wrapR = GL_MIRRORED_REPEAT}};
        lhsTexturesBack[i] = Texture3D{
            {.name = std::format("lhsTexture1_level{}", i).c_str(),
             .width = levelWidth + 1,
             .height = levelHeight + 1,
             .depth = levelDepth + 1,
             .internalFormat = scalarInternalFormat,
             .minFilter = GL_LINEAR,
             .magFilter = GL_LINEAR,
             .wrapS = GL_MIRRORED_REPEAT,
             .wrapT = GL_MIRRORED_REPEAT,
             .wrapR = GL_MIRRORED_REPEAT}};
    }

    settings.mgLevelSettings.resize(levels);

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
    glNamedBufferStorage(divergenceSSBOs[0], 4 * sizeof(GLfloat), nullptr, 0);
    glObjectLabel(GL_BUFFER, divergenceSSBOs[0], -1, "Remaining Divergence SSBO0");
    glNamedBufferStorage(divergenceSSBOs[1], 4 * sizeof(GLfloat), nullptr, 0);
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
    glClearTexImage(velocityTexFront.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(velocityTexBack.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(vectorPhiTilde2Tex.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(vectorPhiTildeTex.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(densityTexFront.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(densityTexBack.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(temperatureTexFront.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(temperatureTexBack.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(scalarPhiTildeTex.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(scalarPhiTilde2Tex.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(divergenceTex.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(pressureTexFront.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    glClearTexImage(pressureTexBack.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    for(int i = 0; i < levels; i++)
    {
        glClearTexImage(rhsTextures[i].getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
        glClearTexImage(lhsTexturesFront[i].getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
        glClearTexImage(lhsTexturesBack[i].getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
    }
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

    advectQuantities();
    addImpulse();
    addBuoyancy();
    calculateDivergence();
    solvePressure();
    subtractPressureGradient();

    if(settings.calculateRemainingDivergence)
    {
        // todo: make more efficient, but low prio since not necessary for solver
        componentTimers[DivergenceRemainder].start();
        GLfloat zero = 0;
        glClearNamedBufferData(divergenceSSBOs[frontBufferIndex], GL_R32F, GL_RED, GL_FLOAT, &zero);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, divergenceSSBOs[frontBufferIndex]);
        glBindTextureUnit(0, velocityTexFront.getTextureID());
        // glBindTextureUnit(1, rhsTextures[0].getTextureID());
        divergenceRemainderPass.execute();
        glGetNamedBufferSubData(
            divergenceSSBOs[backBufferIndex], 0, 4 * sizeof(GLfloat), &remainingDivergence);
        componentTimers[DivergenceRemainder].end();
    }

    glPopDebugGroup();

    timer.end();

    frontBufferIndex = !frontBufferIndex;
    backBufferIndex = !backBufferIndex;
}

void FluidSolver::advectQuantities()
{
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
    temperatureTexFront.swap(temperatureTexBack);

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
    densityTexFront.swap(densityTexBack);

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
    velocityTexFront.swap(velocityTexBack);
    glPopDebugGroup();

    componentTimers[Advection].end();
}

void FluidSolver::addImpulse()
{
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
    velocityTexFront.swap(velocityTexBack);
    temperatureTexFront.swap(temperatureTexBack);
    densityTexFront.swap(densityTexBack);
    componentTimers[Impulse].end();
}

void FluidSolver::addBuoyancy()
{
    componentTimers[Buoyancy].start();
    buoyancyPass.execute();
    velocityTexFront.swap(velocityTexBack);
    componentTimers[Buoyancy].end();
}

void FluidSolver::calculateDivergence()
{
    componentTimers[Divergence].start();
    if(settings.solverMode == PressureSolver::Jacobi || settings.solverMode == PressureSolver::RBGaussSeidel)
    {
        divergencePass.execute();
    }
    else if(settings.solverMode == PressureSolver::Multigrid)
    {
        divergenceMultigridPass.execute();
    }
    componentTimers[Divergence].end();
}

void FluidSolver::solvePressure()
{
    componentTimers[PressureSolve].start();
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Pressure Solve");
    // not using passes for these since they dont work well with loops yet (eg lots of
    // duplicate/unnecessary bindings)
    if(settings.solverMode == PressureSolver::Jacobi)
    {
        if(!settings.useLastFrameAsInitialGuess)
        {
            const glm::vec4 clear = glm::vec4(0.0f);
            glClearTexImage(pressureTexFront.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
        }
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Jacobi");
        jacobiShader.useProgram();
        glBindTextureUnit(1, divergenceTex.getTextureID());
        for(decltype(settings.iterations) i = 0; i < settings.iterations; i++)
        {
            glBindTextureUnit(0, pressureTexFront.getTextureID());
            glBindImageTexture(
                0, pressureTexBack.getTextureID(), 0, GL_TRUE, 0, GL_WRITE_ONLY, scalarInternalFormat);
            glDispatchCompute(
                UintDivAndCeil(width, 16), UintDivAndCeil(height, 16), static_cast<uint32_t>(depth));
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
            pressureTexFront.swap(pressureTexBack);
        }
        glPopDebugGroup();
    }
    else if(settings.solverMode == PressureSolver::RBGaussSeidel)
    {
        if(!settings.useLastFrameAsInitialGuess)
        {
            const glm::vec4 clear = glm::vec4(0.0f);
            glClearTexImage(pressureTexFront.getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
        }
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Gauss Seidel");
        gaussSeidelShader.useProgram();
        glBindTextureUnit(0, divergenceTex.getTextureID());
        glBindImageTexture(
            0, pressureTexFront.getTextureID(), 0, GL_TRUE, 0, GL_READ_WRITE, scalarInternalFormat);
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
    else if(settings.solverMode == PressureSolver::Multigrid)
    {
        if(!settings.useLastFrameAsInitialGuess)
        {
            const glm::vec4 clear = glm::vec4(0.0f);
            glClearTexImage(lhsTexturesFront[0].getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
        }
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Multigrid");

        //  V Cycle
        //  Downwards
        for(int level = 0; level < settings.mgLevels - 1; level++)
        {
            jacobiMultigridShader.useProgram();
            glBindTextureUnit(1, rhsTextures[level].getTextureID());
            for(decltype(settings.mgLevelSettings[level].preSmoothIterations) i = 0;
                i < settings.mgLevelSettings[level].preSmoothIterations;
                i++)
            {
                glBindTextureUnit(0, lhsTexturesFront[level].getTextureID());
                glBindImageTexture(
                    0,
                    lhsTexturesBack[level].getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_WRITE_ONLY,
                    scalarInternalFormat);
                glDispatchCompute(
                    UintDivAndCeil(lhsTexturesFront[level].getDepth(), 16),
                    UintDivAndCeil(lhsTexturesFront[level].getHeight(), 16),
                    static_cast<uint32_t>(lhsTexturesFront[level].getDepth()));
                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
                lhsTexturesFront[level].swap(lhsTexturesBack[level]);
            }
            // pre smoothed pressure is now in lhsTexturesFront

            // restriction

            // write current level residual into lhsTexturesBack
            residualShader.useProgram();
            glBindTextureUnit(0, lhsTexturesFront[level].getTextureID());
            glBindTextureUnit(1, rhsTextures[level].getTextureID());
            glBindImageTexture(
                0, lhsTexturesBack[level].getTextureID(), 0, GL_TRUE, 0, GL_WRITE_ONLY, scalarInternalFormat);
            glDispatchCompute(
                UintDivAndCeil(lhsTexturesBack[level].getDepth(), 16),
                UintDivAndCeil(lhsTexturesBack[level].getHeight(), 16),
                static_cast<uint32_t>(lhsTexturesBack[level].getDepth()));
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

            // restrict from lhsTexturesBack into rhsTextures[level + 1]
            restrictShader.useProgram();
            glBindTextureUnit(0, lhsTexturesBack[level].getTextureID());
            glBindImageTexture(
                0, rhsTextures[level + 1].getTextureID(), 0, GL_TRUE, 0, GL_WRITE_ONLY, scalarInternalFormat);
            glDispatchCompute(
                UintDivAndCeil(rhsTextures[level + 1].getDepth(), 16),
                UintDivAndCeil(rhsTextures[level + 1].getHeight(), 16),
                static_cast<uint32_t>(rhsTextures[level + 1].getDepth()));

            // clear new levels initial lhs texture
            const glm::vec4 clear = glm::vec4(0.0f);
            glClearTexImage(lhsTexturesFront[level + 1].getTextureID(), 0, GL_RED, GL_FLOAT, &clear);
        }

        // do iterations on last level
        jacobiMultigridShader.useProgram();
        glBindTextureUnit(1, rhsTextures[settings.mgLevels - 1].getTextureID());
        for(decltype(settings.mgLevelSettings[settings.mgLevels - 1].iterations) i = 0;
            i < settings.mgLevelSettings[settings.mgLevels - 1].iterations;
            i++)
        {
            glBindTextureUnit(0, lhsTexturesFront[settings.mgLevels - 1].getTextureID());
            glBindImageTexture(
                0,
                lhsTexturesBack[settings.mgLevels - 1].getTextureID(),
                0,
                GL_TRUE,
                0,
                GL_WRITE_ONLY,
                scalarInternalFormat);
            glDispatchCompute(
                UintDivAndCeil(lhsTexturesFront[settings.mgLevels - 1].getDepth(), 16),
                UintDivAndCeil(lhsTexturesFront[settings.mgLevels - 1].getHeight(), 16),
                static_cast<uint32_t>(lhsTexturesFront[settings.mgLevels - 1].getDepth()));
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
            lhsTexturesFront[settings.mgLevels - 1].swap(lhsTexturesBack[settings.mgLevels - 1]);
        }

        //  Upwards
        for(int level = settings.mgLevels - 2; level >= 0; level--)
        {
            // interpolate and correct with solution on previous level

            // currently interpolation and correction can easily be done in one step since
            // interpolation just takes nearest neighbour
            correctShader.useProgram();
            glBindTextureUnit(0, lhsTexturesFront[level].getTextureID());
            glBindTextureUnit(1, lhsTexturesFront[level + 1].getTextureID());
            glBindImageTexture(
                0, lhsTexturesBack[level].getTextureID(), 0, GL_TRUE, 0, GL_WRITE_ONLY, scalarInternalFormat);
            glDispatchCompute(
                UintDivAndCeil(lhsTexturesBack[level].getDepth(), 16),
                UintDivAndCeil(lhsTexturesBack[level].getHeight(), 16),
                static_cast<uint32_t>(lhsTexturesFront[level].getDepth()));
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
            lhsTexturesFront[level].swap(lhsTexturesBack[level]);

            // postsmoothing iterations
            jacobiMultigridShader.useProgram();
            glBindTextureUnit(1, rhsTextures[level].getTextureID());
            for(decltype(settings.mgLevelSettings[level].postSmoothIterations) i = 0;
                i < settings.mgLevelSettings[level].postSmoothIterations;
                i++)
            {
                glBindTextureUnit(0, lhsTexturesFront[level].getTextureID());
                glBindImageTexture(
                    0,
                    lhsTexturesBack[level].getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_WRITE_ONLY,
                    scalarInternalFormat);
                glDispatchCompute(
                    UintDivAndCeil(lhsTexturesFront[level].getDepth(), 16),
                    UintDivAndCeil(lhsTexturesFront[level].getHeight(), 16),
                    static_cast<uint32_t>(lhsTexturesFront[level].getDepth()));
                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
                lhsTexturesFront[level].swap(lhsTexturesBack[level]);
            }
        }

        glPopDebugGroup();
    }
    glPopDebugGroup();
    componentTimers[PressureSolve].end();
}

void FluidSolver::subtractPressureGradient()
{
    componentTimers[PressureSubtraction].start();
    if(settings.solverMode == PressureSolver::Jacobi || settings.solverMode == PressureSolver::RBGaussSeidel)
    {
        pressureSubPass.execute();
    }
    else if(settings.solverMode == PressureSolver::Multigrid)
    {
        pressureSubMultigridPass.execute();
    }
    velocityTexFront.swap(velocityTexBack);
    componentTimers[PressureSubtraction].end();
}

void FluidSolver::updateSettingsBuffer()
{
    // todo: just subdata vs invalidate + subdata vs map with invalidate
    //       doesnt seem to matter for such a small buffer
    glNamedBufferSubData(timeVarsUBO, 0, sizeof(AdvectSettings), &settings.advectSettings);
}

FluidSolver::RemainingDivergence FluidSolver::getRemainingDivergence() const
{
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