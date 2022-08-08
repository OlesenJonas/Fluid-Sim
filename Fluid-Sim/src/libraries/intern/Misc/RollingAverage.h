#pragma once

#include <array>
#include <type_traits>

/**Rolling Average of 0 to 255 numeric elements of Type T.
 * Implemented as a Ringbuffer.
 * (255 limit because index is uint8_t, trivial to change)
 */
template <
    typename T, uint8_t Length, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
class RollingAverage
{
  public:
    RollingAverage()
    {
        values.fill(0);
    };
    ~RollingAverage() = default;
    void update(T value)
    {
        sum -= values[index];
        values[index] = value;
        sum += values[index];
        index = (index + 1) % Length;
    }

    template <typename F = float>
    F average() const
    {
        return static_cast<F>(sum) / Length;
    }

    void unroll(std::array<T, Length>& arrayToFill) const
    {
        const uint16_t length2 = Length - index;
        const T* start2 = values.data() + index;
        // part after index is the oldest data, insert first
        memcpy(arrayToFill.data(), start2, length2 * sizeof(T));
        uint16_t length1 = index;
        memcpy(arrayToFill.data() + length2, values.data(), (length1) * sizeof(T));
    }

    [[nodiscard]] static constexpr uint8_t length()
    {
        return Length;
    }

  private:
    std::array<T, Length> values;
    // (at least) for integral types this should probably be a wider type (ie sum of uint32_t could end up
    // uint64_t)
    T sum = 0;
    uint8_t index = 0;
};