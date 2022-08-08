#include <glad/glad/glad.h>

#include "glm/common.hpp"

#include <ImGui/imgui.h>
#include <ImGui/imgui_impl_glfw.h>
#include <ImGui/imgui_impl_opengl3.h>

#include "ImGuiExtensions.h"

#include <vector>

void ImGui::FrameStart()
{
    // start imgui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGui::FrameEnd()
{
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "UI Rendering");
    ImGui::Render();
    // Draw the Render Data into framebuffer
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glPopDebugGroup();
}

// only copy pasted with minor adjustments from existing ImGui code, heavy WIP!

// size_arg (for each axis) < 0.0f: align to end, 0.0f: auto, > 0.0f: specified size
void ImGui::HorizontalBar(
    float fraction_start, float fraction_end, const ImVec2& size_arg, const char* overlay)
{
    ImGuiWindow* window = GetCurrentWindow();
    if(window->SkipItems)
        return;

    ImGuiContext& g = *GImGui;
    const ImGuiStyle& style = g.Style;

    ImVec2 pos = window->DC.CursorPos;
    ImVec2 size = CalcItemSize(size_arg, CalcItemWidth(), g.FontSize + style.FramePadding.y * 2.0f);
    ImRect bb(pos, pos + size);
    ItemSize(size, style.FramePadding.y);
    if(!ItemAdd(bb, 0))
        return;

    // Render
    RenderFrame(bb.Min, bb.Max, GetColorU32(ImGuiCol_FrameBg), true, style.FrameRounding);
    bb.Expand(ImVec2(-style.FrameBorderSize, -style.FrameBorderSize));
    const ImVec2 fill_br = ImVec2(ImLerp(bb.Min.x, bb.Max.x, fraction_end), bb.Max.y);
    RenderRectFilledRangeH(
        window->DrawList,
        bb,
        GetColorU32(ImGuiCol_PlotHistogram),
        fraction_start,
        fraction_end,
        style.FrameRounding);

    // Default displaying the fraction as percentage string, but user can override it
    char overlay_buf[32];
    if(!overlay)
    {
        ImFormatString(overlay_buf, IM_ARRAYSIZE(overlay_buf), "%.0f%%", fraction_end * 100 + 0.01f);
        overlay = overlay_buf;
    }

    ImVec2 overlay_size = CalcTextSize(overlay, NULL);
    if(overlay_size.x > 0.0f)
        RenderTextClipped(
            ImVec2(
                ImClamp(
                    fill_br.x + style.ItemSpacing.x,
                    bb.Min.x,
                    bb.Max.x - overlay_size.x - style.ItemInnerSpacing.x),
                bb.Min.y),
            bb.Max,
            overlay,
            NULL,
            &overlay_size,
            ImVec2(0.0f, 0.5f),
            &bb);
}

int ImGui::PlotLines2D(
    const char* label, float* xValues, float* yValues, int count, const char* overlay_text, ImVec2 xRange,
    ImVec2 yRange, ImVec2 graphSize, ImU32 color)
{
    ImGuiContext& g = *GImGui;
    ImGuiWindow* window = GetCurrentWindow();
    if(window->SkipItems)
        return -1;

    const ImGuiStyle& style = g.Style;
    const ImGuiID id = window->GetID(label);

    const ImVec2 label_size = CalcTextSize(label, NULL, true);
    if(graphSize.x == 0.0f)
        graphSize.x = CalcItemWidth();
    if(graphSize.y == 0.0f)
        graphSize.y = label_size.y + (style.FramePadding.y * 2);

    const ImRect frame_bb(window->DC.CursorPos, window->DC.CursorPos + graphSize);
    const ImRect inner_bb(frame_bb.Min + style.FramePadding, frame_bb.Max - style.FramePadding);
    const ImRect total_bb(
        frame_bb.Min,
        frame_bb.Max + ImVec2(label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f, 0));
    ItemSize(total_bb, style.FramePadding.y);
    if(!ItemAdd(total_bb, 0, &frame_bb))
        return -1;
    const bool hovered = ItemHoverable(frame_bb, id);

    // Determine scale from values if not specified
    if(xRange.x == FLT_MAX || xRange.y == FLT_MAX || yRange.x == FLT_MAX || yRange.y == FLT_MAX)
    {
        float tempMinX = FLT_MAX;
        float tempMinY = FLT_MAX;
        float tempMaxX = -FLT_MAX;
        float tempMaxY = -FLT_MAX;
        for(int i = 0; i < count; i++)
        {
            const float xCur = xValues[i];
            const float yCur = yValues[i];
            if(xCur == xCur) // Ignore NaN values
            {
                tempMinX = ImMin(tempMinX, xCur);
                tempMaxX = ImMax(tempMaxX, xCur);
            }
            if(yCur == yCur) // Ignore NaN values
            {
                tempMinY = ImMin(tempMinY, yCur);
                tempMaxY = ImMax(tempMaxY, yCur);
            }
        }
        if(xRange.x == FLT_MAX)
            xRange.x = tempMinX;
        if(xRange.y == FLT_MAX)
            xRange.y = tempMaxX;
        if(yRange.x == FLT_MAX)
            yRange.x = tempMinY;
        if(yRange.y == FLT_MAX)
            yRange.y = tempMaxY;
    }

    RenderFrame(frame_bb.Min, frame_bb.Max, GetColorU32(ImGuiCol_FrameBg), true, style.FrameRounding);

    const int values_count_min = 2;
    int idx_hovered = -1;
    if(count >= values_count_min)
    {
        int res_w = ImMin((int)graphSize.x, count) - 1;

        // const float t_step = 1.0f / (float)res_w;
        const float inv_scale_x = (xRange.x == xRange.y) ? 0.0f : (1.0f / (xRange.y - xRange.x));
        const float inv_scale_y = (yRange.x == yRange.y) ? 0.0f : (1.0f / (yRange.y - yRange.x));

        // float v0 = values_getter(data, (0 + values_offset) % values_count);
        float x0 = xValues[0];
        float y0 = yValues[0];
        ImVec2 p0 = ImVec2(
            ImSaturate((x0 - xRange.x) * inv_scale_x),
            1.0f -
                ImSaturate(
                    (y0 - yRange.x) * inv_scale_y)); // Point in the normalized space of our target rectangle

        for(int n = 0; n < res_w; n++)
        {
            const float x1 = xValues[n + 1];
            const float y1 = yValues[n + 1];
            // IM_ASSERT(v1_idx >= 0 && v1_idx < values_count); //figure out what was doink
            ImVec2 p1 = ImVec2(
                ImSaturate((x1 - xRange.x) * inv_scale_x), 1.0f - ImSaturate((y1 - yRange.x) * inv_scale_y));

            // NB: Draw calls are merged together by the DrawList system. Still, we should render our batch
            // are lower level to save a bit of CPU.
            ImVec2 pos0 = ImLerp(inner_bb.Min, inner_bb.Max, p0);
            ImVec2 pos1 = ImLerp(inner_bb.Min, inner_bb.Max, p1);

            window->DrawList->AddLine(pos0, pos1, color);

            p0 = p1;
        }
    }

    // Text overlay
    if(overlay_text != nullptr)
        RenderTextClipped(
            ImVec2(frame_bb.Min.x, frame_bb.Min.y + style.FramePadding.y),
            frame_bb.Max,
            overlay_text,
            nullptr,
            nullptr,
            ImVec2(0.5f, 0.0f));

    if(label_size.x > 0.0f)
        RenderText(ImVec2(frame_bb.Max.x + style.ItemInnerSpacing.x, inner_bb.Min.y), label);

    // Return hovered index or -1 if none are hovered.
    // This is currently not exposed in the public API because we need a larger redesign of the whole thing,
    // but in the short-term we are making it available in PlotEx().
    return -1;
}

int ImGui::PlotLines2D(
    const char* label, float** xValues, float** yValues, int* count, int datasets, const char* overlay_text,
    ImVec2 xRange, ImVec2 yRange, ImVec2 graphSize, ImU32* colors)
{
    ImGuiContext& g = *GImGui;
    ImGuiWindow* window = GetCurrentWindow();
    if(window->SkipItems)
        return -1;

    const ImGuiStyle& style = g.Style;
    const ImGuiID id = window->GetID(label);

    const ImVec2 label_size = CalcTextSize(label, nullptr, true);
    if(graphSize.x == 0.0f)
        graphSize.x = CalcItemWidth();
    if(graphSize.y == 0.0f)
        graphSize.y = label_size.y + (style.FramePadding.y * 2);

    const ImRect frame_bb(window->DC.CursorPos, window->DC.CursorPos + graphSize);
    const ImRect inner_bb(frame_bb.Min + style.FramePadding, frame_bb.Max - style.FramePadding);
    const ImRect total_bb(
        frame_bb.Min,
        frame_bb.Max + ImVec2(label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f, 0));
    ItemSize(total_bb, style.FramePadding.y);
    if(!ItemAdd(total_bb, 0, &frame_bb))
        return -1;
    const bool hovered = ItemHoverable(frame_bb, id);

    // Determine scale from values if not specified
    if(xRange.x == FLT_MAX || xRange.y == FLT_MAX || yRange.x == FLT_MAX || yRange.y == FLT_MAX)
    {
        float tempMinX = FLT_MAX;
        float tempMinY = FLT_MAX;
        float tempMaxX = -FLT_MAX;
        float tempMaxY = -FLT_MAX;
        for(int d = 0; d < datasets; d++)
        {
            for(int i = 0; i < count[d]; i++)
            {
                const float xCur = xValues[d][i];
                const float yCur = yValues[d][i];
                if(xCur == xCur) // Ignore NaN values
                {
                    tempMinX = ImMin(tempMinX, xCur);
                    tempMaxX = ImMax(tempMaxX, xCur);
                }
                if(yCur == yCur) // Ignore NaN values
                {
                    tempMinY = ImMin(tempMinY, yCur);
                    tempMaxY = ImMax(tempMaxY, yCur);
                }
            }
        }
        if(xRange.x == FLT_MAX)
            xRange.x = tempMinX;
        if(xRange.y == FLT_MAX)
            xRange.y = tempMaxX;
        if(yRange.x == FLT_MAX)
            yRange.x = tempMinY;
        if(yRange.y == FLT_MAX)
            yRange.y = tempMaxY;
    }

    RenderFrame(frame_bb.Min, frame_bb.Max, GetColorU32(ImGuiCol_FrameBg), true, style.FrameRounding);

    const int values_count_min = 2;
    int idx_hovered = -1;
    for(int d = 0; d < datasets; d++)
    {
        if(count[d] >= values_count_min)
        {
            int res_w = ImMin((int)graphSize.x, count[d]) - 1;

            // const float t_step = 1.0f / (float)res_w;
            const float inv_scale_x = (xRange.x == xRange.y) ? 0.0f : (1.0f / (xRange.y - xRange.x));
            const float inv_scale_y = (yRange.x == yRange.y) ? 0.0f : (1.0f / (yRange.y - yRange.x));

            // float v0 = values_getter(data, (0 + values_offset) % values_count);
            float x0 = xValues[d][0];
            float y0 = yValues[d][0];
            ImVec2 p0 = ImVec2(
                ImSaturate((x0 - xRange.x) * inv_scale_x),
                1.0f - ImSaturate(
                           (y0 - yRange.x) *
                           inv_scale_y)); // Point in the normalized space of our target rectangle

            for(int n = 0; n < res_w; n++)
            {
                const float x1 = xValues[d][n + 1];
                const float y1 = yValues[d][n + 1];
                // IM_ASSERT(v1_idx >= 0 && v1_idx < values_count); //figure out what was doink
                ImVec2 p1 = ImVec2(
                    ImSaturate((x1 - xRange.x) * inv_scale_x),
                    1.0f - ImSaturate((y1 - yRange.x) * inv_scale_y));

                // NB: Draw calls are merged together by the DrawList system. Still, we should render our
                // batch are lower level to save a bit of CPU.
                ImVec2 pos0 = ImLerp(inner_bb.Min, inner_bb.Max, p0);
                ImVec2 pos1 = ImLerp(inner_bb.Min, inner_bb.Max, p1);

                window->DrawList->AddLine(pos0, pos1, colors[d]);

                p0 = p1;
            }
        }
    }

    // Text overlay
    if(overlay_text != nullptr)
        RenderTextClipped(
            ImVec2(frame_bb.Min.x, frame_bb.Min.y + style.FramePadding.y),
            frame_bb.Max,
            overlay_text,
            nullptr,
            nullptr,
            ImVec2(0.5f, 0.0f));

    if(label_size.x > 0.0f)
        RenderText(ImVec2(frame_bb.Max.x + style.ItemInnerSpacing.x, inner_bb.Min.y), label);

    // Return hovered index or -1 if none are hovered.
    // This is currently not exposed in the public API because we need a larger redesign of the whole thing,
    // but in the short-term we are making it available in PlotEx().
    return -1;
}

float ImGui::GetTextScrollFactor(double time)
{
    // https://www.desmos.com/calculator/vopzdlvbit
    const double pixelWidth = ImGui::GetScrollMaxX();
    const double pixelPerSecond = 50.0f;
    const double timeForScroll = pixelWidth / pixelPerSecond;
    const double freeze = 0.5f;
    // return pixelWidth *
    //        glm::smoothstep(freeze, freeze + timeForScroll, glm::mod(time, 2 * freeze + timeForScroll));
    return pixelWidth *
           glm::clamp((glm::mod(time, 2 * freeze + timeForScroll) - freeze) / timeForScroll, 0.0, 1.0);
}