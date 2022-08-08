#pragma once

#include <glad/glad/glad.h>

#include <cstdint>

#include "RollingAverage.h"

/**Timer to measure GPU time taken between submitting start() and end() *max once* per Frame.
 * Averaged across S frames
 */
template <uint8_t S>
class GPUTimer
{
  public:
    GPUTimer()
    {
        // generate queries
        glGenQueries(2, queries[0]);
        glGenQueries(2, queries[1]);
        // dummy calls, otherwise first frame cant retrieve results
        glQueryCounter(queries[frontBufferIndex][0], GL_TIMESTAMP);
        glQueryCounter(queries[frontBufferIndex][1], GL_TIMESTAMP);
    }

    ~GPUTimer()
    {
        glDeleteQueries(2, queries[0]);
        glDeleteQueries(2, queries[1]);
    }

    void start()
    {
        glQueryCounter(queries[backBufferIndex][0], GL_TIMESTAMP);
    }

    void end()
    {
        glQueryCounter(queries[backBufferIndex][1], GL_TIMESTAMP);
    }

    void evaluate()
    {
        GLuint64 resultStart = 0;
        GLuint64 resultEnd = 0;
        glGetQueryObjectui64v(queries[frontBufferIndex][0], GL_QUERY_RESULT, &resultStart);
        glGetQueryObjectui64v(queries[frontBufferIndex][1], GL_QUERY_RESULT, &resultEnd);
        average.update(resultEnd - resultStart);
        // swap buffers
        frontBufferIndex = !frontBufferIndex;
        backBufferIndex = !backBufferIndex;
    }

    [[nodiscard]] double timeMilliseconds() const
    {
        return average.template average<GLuint64>() / 1000000.0;
    }

    [[nodiscard]] GLuint64 timeNanoseconds() const
    {
        return average.template average<GLuint64>();
    }

    [[nodiscard]] constexpr uint8_t framesAveraged() const
    {
        return S;
    }

  private:
    GLuint queries[2][2]; // NOLINT
    RollingAverage<GLuint64, S> average;
    uint8_t backBufferIndex = 0;
    uint8_t frontBufferIndex = 1;
};