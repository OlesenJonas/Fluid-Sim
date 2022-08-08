#include <glad/glad/glad.h>

#include <GLFW/glfw3.h>
#include <glm/ext.hpp>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <ImGui/imgui.h>
#include <ImGui/imgui_impl_glfw.h>
#include <ImGui/imgui_impl_opengl3.h>

#include <intern/Misc/GPUTimer.h>
#include <intern/Misc/ImGuiExtensions.h>
#include <intern/Misc/RollingAverage.h>
#include <intern/ShaderProgram/ShaderProgram.h>

// screen
int width = 1200, height = 800, window_off_x = 50, window_off_y = 50;
int FONT_SIZE = 14;

// todo: High dpi support!

GLFWwindow* window;

void mouseButtonCallbackDemo(GLFWwindow* window, int button, int action, int mods)
{
}

void keyCallbackDemo(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

void scrollCallbackDemo(GLFWwindow* window, double xoffset, double yoffset)
{
}

void framebufferSizeCallbackDemo(GLFWwindow* window, int w, int h)
{
    width = w, height = h;
    glViewport(0, 0, w, h);
}

int main()
{
    glfwInit();

    // initliaze window values

    window = glfwCreateWindow(width, height, "Imgui Demo", nullptr, nullptr);
    glfwSetWindowPos(window, window_off_x, window_off_y);
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallbackDemo);
    glfwSetMouseButtonCallback(window, mouseButtonCallbackDemo);
    glfwSetScrollCallback(window, scrollCallbackDemo);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallbackDemo);
    glfwSwapInterval(0);

    // init opengl
    if(!gladLoadGL())
    {
        printf("Something went wrong!\n");
        exit(-1);
    }

    // SETUP IMGUI CONTEXT
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    // not sure what happens if font cant be found, disable call for now
    // io.Fonts->AddFontFromFileTTF("C:/WINDOWS/Fonts/verdana.ttf", FONT_SIZE, NULL, NULL);
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    // style.TabRounding = ...

    // platform/renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();

    // Demo state variables
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    glClearColor(1, 1, 1, 1);
    glEnable(GL_DEPTH_TEST);

    // The different constructs used for measuring perf
    // RollingAverage class is enough to measure time taken on CPU
    RollingAverage<float, 128> frametimeAverage;
    // GPUTimer is a wrapper around RollingAverage that measures time on the GPU
    GPUTimer<128> uiDrawTimer;

    // time
    double last_frame = 0.0;

    // set time to 0 before renderloop
    glfwSetTime(0.0);
    // renderloop
    while(!glfwWindowShouldClose(window))
    {

        glfwPollEvents();

        ImGui::FrameStart();

        // get ImGui's IO for eventual checks
        ImGuiIO& io = ImGui::GetIO();

        // process time manually, or use ImGui's deltaTime
        double current_frame = glfwGetTime();
        double delta_time = current_frame - last_frame;
        last_frame = current_frame;
        (void)io.DeltaTime;
        frametimeAverage.update(delta_time);

        // clear last frame
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse
        // its code to learn more about Dear ImGui!).
        if(show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named
        // window.
        { // Dummy scopes are useful when using variables inside window setup to hide them
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Hello, world!"); // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text."); // Display some text (you can use a format strings too)
            ImGui::Checkbox(
                "Demo Window", &show_demo_window); // Edit bools storing our window open/close state
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f); // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

            if(ImGui::Button("Button")) // Buttons return true when clicked (most widgets return true when
                                        // edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text(
                "Application average %.3f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate,
                ImGui::GetIO().Framerate);
            ImGui::End();
        }

        // 3. Show another simple window.
        if(show_another_window)
        {
            ImGui::Begin(
                "Another Window",
                &show_another_window); // Pass a pointer to our bool variable (the window will have a closing
                                       // button that will clear the bool when clicked)
            ImGui::Text("Hello from another window!");
            if(ImGui::Button("Close Me"))
                show_another_window = false;
            ImGui::End();
        }

        // custom graphing
        {
            ImGui::SetNextWindowPos(ImVec2(10, 100), ImGuiCond_Once);
            ImGui::Begin("GRAPHS", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
            std::vector<float> xValues = {0, 1, 2, 3, 4, 5};
            std::vector<float> yValues = {0, 1, 4, 9, 16, 25};
            ImGui::Text("Default Graph");
            ImGui::PlotLines("", &yValues[0], xValues.size(), 0, nullptr, -1, 26, ImVec2(0, 120));
            ImGui::Text("Custom 2D");
            ImGui::PlotLines2D(
                "",
                &xValues[0],
                &yValues[0],
                xValues.size(),
                nullptr,
                ImVec2(0, 5),
                ImVec2(-1, 26),
                ImVec2(0, 120),
                ImGui::GetColorU32({1, 1, 0, 1}));
            std::vector<float> yValues2 = {25, 25, 15, 10, 8, 0};
            float* xPointers[2] = {&xValues[0], &xValues[0]};
            float* yPointers[2] = {&yValues[0], &yValues2[0]};
            int count[2] = {static_cast<int>(xValues.size()), static_cast<int>(xValues.size())};
            ImU32 colors[2] = {
                ImGui::GetColorU32({1, 1, 0, 1}),
                0xFF00FF00}; // or ImGui::GetColorU32(ImGuiCol_PlotLines) for default color
            ImGui::Text("Multiple 2D");
            ImGui::PlotLines2D(
                "",
                xPointers,
                yPointers,
                count,
                2,
                nullptr,
                ImVec2(0, 5),
                ImVec2(-1, 26),
                ImVec2(0, 120),
                colors);
            ImGui::End();
        }

        {
            ImGui::Begin("Performance Example", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
            constexpr auto averagedFrametimes = frametimeAverage.length();
            static std::array<float, averagedFrametimes> frametimesSorted;
            ImGui::TextUnformatted("ImGui Measurement:");
            ImGuiIO& io = ImGui::GetIO();
            ImGui::Text(
                "Application average %.3f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate,
                io.Framerate);
            ImGui::TextUnformatted("Custom Measurement:");
            ImGui::Text(
                "Application average %.3f ms/frame (%.1f FPS)",
                frametimeAverage.average() * 1000,
                1.0f / frametimeAverage.average());
            { // Graphing the frametimes
                // have to "unroll" the ringbuffer
                frametimeAverage.unroll(frametimesSorted);
                ImGui::PlotLines(
                    "Frame Times (s)",
                    frametimesSorted.data(),
                    frametimesSorted.size(),
                    0,
                    nullptr,
                    0.0f,
                    (2.0f / io.Framerate),
                    ImVec2(0, 80));
            }
            // double frametimeMilliseconds = frametimeAverage.average() * 1000.0;
            double frametimeMilliseconds = 1000.0f / ImGui::GetIO().Framerate;
            ImGui::Text("UI draw time: %.3f ms", uiDrawTimer.timeMilliseconds());
            ImGui::HorizontalBar(
                0.0f, uiDrawTimer.timeMilliseconds() / frametimeMilliseconds, ImVec2(0, 0), "");
            ImGui::End();
        }

        uiDrawTimer.start();
        // Create ImGui Renderdata and draw it
        ImGui::FrameEnd();
        uiDrawTimer.end();
        // evaluates last frames result, this guarantees no stalling
        uiDrawTimer.evaluate();

        // Swap buffer
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}