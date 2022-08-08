#include "VolumeRenderer.h"

#include <glad/glad/glad.h>

#include <cstdio>
#include <thread>

#include <intern/Camera/Camera.h>
#include <intern/FluidSolver/FluidSolver.h>

VolumeRenderer::VolumeRenderer(Context& ctx, const FluidSolver& fluidSolver /*, const Texture& depthbuffer*/)
    : ctx(ctx), fluidSolver(fluidSolver) /*, depthbuffer(depthbuffer)*/
{
    // when rendering the volume, override the density sampler so that all values outside [0,1] return 0
    glCreateSamplers(1, &borderSampler);
    glSamplerParameteri(borderSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glSamplerParameteri(borderSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(borderSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glSamplerParameteri(borderSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glSamplerParameteri(borderSampler, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glm::vec4 borderColor{0.0f, 0.0f, 0.0f, 0.0f};
    glSamplerParameterfv(borderSampler, GL_TEXTURE_BORDER_COLOR, &borderColor[0]);

    volumeRenderShader.useProgram();
    glUniform1f(glGetUniformLocation(volumeRenderShader.getProgramID(), "zNear"), ctx.getCamera()->getNear());
    glUniform1f(glGetUniformLocation(volumeRenderShader.getProgramID(), "zFar"), ctx.getCamera()->getFar());

    glCreateBuffers(1, &settingsUBO);
    glNamedBufferStorage(settingsUBO, sizeof(Settings), &settings, GL_DYNAMIC_STORAGE_BIT);
    glObjectLabel(GL_BUFFER, settingsUBO, -1, "Volume Render Settings UBO");
    glBindBufferBase(GL_UNIFORM_BUFFER, 12, settingsUBO);
}

void VolumeRenderer::render()
{
    // todo: enable rendering when inside volume

    Camera* cam = ctx.getCamera();

    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Volume Rendering");

    timer.start();

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    // test if camera is inside volume
    glm::vec4 camPos = glm::vec4{cam->getPosition(), 1.0};
    // would need to transform into local spacer here *if* the cube/volume had any transform
    // glm::bvec3 inside = glm::lessThan(glm::abs(camPos), glm::vec3(0.5f));
    // if(glm::all(inside)) {
    //     // disabled for now, since inside rendering not fully functional
    //     glCullFace(GL_FRONT);
    // }

    // glDisable(GL_BLEND);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    glBindTextureUnit(0, fluidSolver.getDensityTexture().getTextureID());
    glBindSampler(0, borderSampler);
    glBindTextureUnit(1, noiseTex.getTextureID());
    // glBindTextureUnit(2, depthbuffer.getTextureID());

    if(useEmissive)
    {
        glBindTextureUnit(3, fluidSolver.getTemperatureTexture().getTextureID());
    }

    ShaderProgram& shaderToUse = useEmissive ? volumeRenderEmissiveShader : volumeRenderShader;

    shaderToUse.useProgram();
    glUniformMatrix4fv(0, 1, GL_FALSE, glm::value_ptr(fluidSolver.getTransform()));
    glUniformMatrix4fv(1, 1, GL_FALSE, glm::value_ptr(*cam->getView()));
    glUniformMatrix4fv(2, 1, GL_FALSE, glm::value_ptr(*cam->getProj()));

    glUniform3fv(3, 1, glm::value_ptr(fluidSolver.getInvTransform() * camPos));
    glm::mat4 invProjection = glm::inverse(*cam->getProj());
    glUniformMatrix4fv(4, 1, GL_FALSE, glm::value_ptr(invProjection));
    // worldToLocal * viewToWorld
    glm::mat4 viewToLocal = fluidSolver.getInvTransform() * glm::inverse(*cam->getView());
    glUniformMatrix4fv(5, 1, GL_FALSE, glm::value_ptr(viewToLocal));

    cube.draw();

    // reset state
    // unbind sampler override
    glBindSampler(0, 0);
    glCullFace(GL_BACK);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    timer.end();

    glPopDebugGroup();
}

void VolumeRenderer::updateSettingsBuffer()
{
    glNamedBufferSubData(settingsUBO, 0, sizeof(Settings), &settings);
}

void VolumeRenderer::saveSettings(const std::string& file)
{
    assert(file.data()[file.size()] == '\0' && "Filepath needs to be null terminated");
    Settings settingsToSave = settings;
    std::thread saveSettingsThread(
        [settingsToSave, file]() -> void
        {
            FILE* settingsFile = fopen(file.data(), "wb");
            assert(settingsFile != NULL);
            size_t ret = fwrite(&settingsToSave, sizeof(settingsToSave), 1, settingsFile);
            assert(ret == 1);
            int retclose = fclose(settingsFile);
            assert(retclose == 0);
        });
    saveSettingsThread.detach();
}

void VolumeRenderer::loadSettings(const std::string& file)
{
    // dont override lightposition and colors!!
    Settings newSettings;
    std::ifstream settingsFile(file.data(), std::ios::out | std::ios::binary);
    assert(settingsFile);
    settingsFile.read((char*)&newSettings, sizeof(newSettings) / sizeof(char));
    settingsFile.close();
    newSettings.sunColorAndStrength = settings.sunColorAndStrength;
    newSettings.skyColorAndStrength = settings.skyColorAndStrength;
    newSettings.lightVectorLocal = settings.lightVectorLocal;
    settings = newSettings;
    updateSettingsBuffer();
}