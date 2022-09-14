#include <glad/glad/glad.h>

#include <array>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <utility>

#include <GLFW/glfw3.h>

#include <stb/stb_image_write.h>

#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>

#include <ImGui/imgui.h>
#include <ImGui/imgui_impl_glfw.h>
#include <ImGui/imgui_impl_opengl3.h>

#include <intern/Camera/Camera.h>
#include <intern/Computepass/Computepass.h>
#include <intern/Context/Context.h>
#include <intern/FluidSolver/FluidSolver.h>
#include <intern/Framebuffer/Framebuffer.h>
#include <intern/InputManager/InputManager.h>
#include <intern/Mesh/Cube.h>
#include <intern/Mesh/FullscreenTri.h>
#include <intern/Misc/GPUTimer.h>
#include <intern/Misc/ImGuiExtensions.h>
#include <intern/Misc/OpenGLErrorHandler.h>
#include <intern/Misc/RollingAverage.h>
#include <intern/ShaderProgram/ShaderProgram.h>
#include <intern/SimulationCapture/SimulationCapture.h>
#include <intern/Splines/BezierSpline.h>
#include <intern/VolumeRenderer/VolumeRenderer.h>
#include <intern/Window/Window.h>

#include "include.h"

int main()
{
    Context ctx{};

    //----------------------- INIT WINDOW

    int WIDTH = 1200;
    int HEIGHT = 800;

    GLFWwindow* window = initAndCreateGLFWWindow(WIDTH, HEIGHT, "Fluid Sim", {{GLFW_MAXIMIZED, GLFW_TRUE}});
    ctx.setWindow(window);
    // disable VSYNC
    glfwSwapInterval(0);
    // In case window was set to start maximized, retrieve size for framebuffer here
    glfwGetWindowSize(window, &WIDTH, &HEIGHT);

    //----------------------- INIT OPENGL
    // init OpenGL context
    if(gladLoadGL() == 0)
    {
        std::cout << "Failed to initialize OpenGL context" << std::endl;
        return -1;
    }
    // glEnable(GL_FRAMEBUFFER_SRGB);
#ifndef NDEBUG
    setupOpenGLMessageCallback();
#endif
    glClearColor(0.3f, 0.7f, 1.0f, 1.0f);
    glDisable(GL_BLEND);

    //----------------------- INIT IMGUI

    InputManager input(ctx);
    input.enableFixedTimestep(0.01);
    ctx.setInputManager(&input);
    input.setupCallbacks();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigDockingWithShift = false;
    // io.Fonts->AddFontFromFileTTF("C:/WINDOWS/Fonts/verdana.ttf", FONT_SIZE, NULL, NULL);
    ImGui::StyleColorsDark();
    // platform/renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");

    //----------------------- INIT REST
    FullscreenTri fullScreenTri;
    ShaderProgram postProcessShader{
        VERTEX_SHADER_BIT | FRAGMENT_SHADER_BIT,
        {SHADERS_PATH "/General/screenQuad.vert", SHADERS_PATH "/General/hdrTonemapSimple.frag"}};
    PostProcessSettings postProcessSettings;
    Framebuffer internalFBO{WIDTH, HEIGHT, {GL_RGBA16F}, true};

    // todo: not sure if I want the setXXX() functions to be part of the constructors
    //       any occasions where it would be undesirable?
    Camera cam{ctx, static_cast<float>(WIDTH) / static_cast<float>(HEIGHT)};
    ctx.setCamera(&cam);

    SimulationCapture simulationCapture{ctx};
    ctx.setSimulationCapture(&simulationCapture);

    FrametimeAverage_t frametimeAverage;

    FluidSolver fluidSolver(ctx, 128, 128, 128, FluidSolver::Precision::Full);
    fluidSolver.setTransform(glm::translate(glm::vec3(-0.5f)));
    // todo: wrong to use internal FBO depthbuffer here, but not actually depth testing in the volume atm
    // anyways
    VolumeRenderer volumeRenderer(ctx, fluidSolver);

    // todo: factor out
    //  Bezier curve to control fluid impulse
    std::vector<glm::vec3> impulseControlPoints = {
        glm::vec3(0.f, 0.f, 1.f),
        glm::vec3(2.f, 1.0f, 1.0f),
        glm::vec3(0.f, 2.0f, 1.0f),
        glm::vec3(2.f, 3.0f, 1.0f),
        glm::vec3(0.f, 2.0f, 1.0f),
        glm::vec3(2.f, 1.0f, 1.0f)};
    constexpr float tangentScale = 2.0f;
    std::vector<glm::vec3> impulseTangentList = {
        glm::vec3(0.0f, 0.0f, tangentScale),
        glm::vec3(0.0f, 0.0f, -tangentScale),
        glm::vec3(0.0f, 0.0f, tangentScale),
        glm::vec3(0.0f, 0.0f, -tangentScale),
        glm::vec3(0.0f, 0.0f, tangentScale),
        glm::vec3(0.0f, 0.0f, -tangentScale)};
    for(int i = 0; i < impulseControlPoints.size(); i++)
    {
        auto& point = impulseControlPoints[i];
        auto& tangent = impulseTangentList[i];
        // point = fluidSolver.getInvTransform() * glm::vec4(point, 1.0);
        point = point / glm::vec3{2.f, 3.f, 2.f};
        float scaleFactor = 0.75f;
        point = scaleFactor * (point - 0.5f) + 0.5f;
        tangent *= scaleFactor;
    }
    BezierSpline impulseSpline{impulseControlPoints, impulseTangentList, true};

    // Cube cube{};
    // ShaderProgram simpleShader{
    //     VERTEX_SHADER_BIT | FRAGMENT_SHADER_BIT,
    //     {SHADERS_PATH "/General/simple.vert", SHADERS_PATH "/General/simple.frag"}};

    // reset time to 0 before renderloop starts
    // otherwise first simulation step will probably be larger than expected
    glfwSetTime(0.0);
    input.resetTime();

    while(glfwWindowShouldClose(window) == 0)
    {
        ImGui::FrameStart();

        input.update();
        // input needs to be updated (checks for pressed buttons etc.)
        // but dont update if UI is using user inputs
        if(!ImGui::GetIO().WantCaptureMouse && !ImGui::GetIO().WantCaptureKeyboard)
        {
            cam.update();
        }

        frametimeAverage.update(input.getRealDeltaTime());

        auto currentTime = static_cast<float>(input.getSimulationTime());

        float impSize = 0.5f + 0.7f *                                  //
                                   ((0.2f * sin(2.0f * currentTime)) + //
                                    (0.1f * cos(3.0f * currentTime)) + //
                                    (0.1f * sin(5.0f * currentTime)));
        impSize *= 0.05f;
        const auto& positionAndTangent = impulseSpline.getPositionAndTangent(glm::fract(0.05f * currentTime));
        glm::vec3 impPosition = positionAndTangent.first;
        glm::vec3 impVelocity = positionAndTangent.second;
        impVelocity = normalize(impVelocity);
        impVelocity *= -0.5f;
        impVelocity *= sin(currentTime) * 0.2f + 1.0f;
        impVelocity *= cos(25.0f * currentTime) * 0.5f + 0.5f;
        fluidSolver.setImpulse(
            {.position = impPosition,
             .density = 5.0f,
             .velocity = impVelocity,
             .temperature = 0.1f,
             .size = impSize});
        fluidSolver.update();

        internalFBO.bind();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // simpleShader.useProgram();
        // glUniformMatrix4fv(0, 1, GL_FALSE, glm::value_ptr(glm::scale(glm::vec3{0.2f})));
        // glUniformMatrix4fv(1, 1, GL_FALSE, glm::value_ptr(*cam.getView()));
        // glUniformMatrix4fv(2, 1, GL_FALSE, glm::value_ptr(*cam.getProj()));
        // cube.draw();

        volumeRenderer.render();

        // Post Processing
        {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            // overwriting full screen anyways, dont need to clear
            glDisable(GL_DEPTH_TEST);
            glBindTextureUnit(0, internalFBO.getColorTextures()[0].getTextureID());
            postProcessShader.useProgram();
            glUniform1f(0, postProcessSettings.simpleExposure);
            glUniform1i(1, postProcessSettings.mode);
            fullScreenTri.draw();
            glEnable(GL_DEPTH_TEST);
        }

        drawFrameInfoUI(ctx, frametimeAverage, fluidSolver, volumeRenderer, postProcessSettings);
        // sRGB is broken in Dear ImGui
        //  glDisable(GL_FRAMEBUFFER_SRGB);
        ImGui::FrameEnd();
        // glEnable(GL_FRAMEBUFFER_SRGB);

        glfwSwapBuffers(window);
        glfwPollEvents();

        simulationCapture.handleFrame();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}