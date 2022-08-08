#pragma once

#include <glm/glm.hpp>

#include <span>
#include <utility>
#include <vector>

class BezierSpline
{
  public:
    /* Position and outgoing! tangent of each controlpoint */
    BezierSpline(std::span<glm::vec3> positions, std::span<glm::vec3> tangents, bool closed);
    /* Get position along the full spline using the splines arc length table
     * factor should be in [0,1]
     */
    glm::vec3 getPosition(float factor);
    std::pair<glm::vec3, glm::vec3> getPositionAndTangent(float factor);

  private:
    /* List of all control points. 4 consecutive control points define a Bezier Curve */
    std::vector<glm::vec3> controlPoints;

    using CubicBezierCurve = std::span<glm::vec3, 4>;
    std::vector<CubicBezierCurve> cubicBezierCurves;
    glm::vec3 getPosition(int curveIndex, float parameter);
    static glm::vec3 getPosition(CubicBezierCurve bezierCurve, float parameter);
    static glm::vec3 getTangent(CubicBezierCurve bezierCurve, float parameter);

    std::pair<int, float> getCurveAndParameter(float factor);

    // todo: instead of arclengthtable + LUT and spline calculations
    //       just reparameterize the spline into equal-length segments?
    struct ArcLengthTableEntry
    {
        float accumLength = 0.0f;
        int curveIndex = 0;
        float parameter = 0;
    };
    std::vector<ArcLengthTableEntry> arcLengthTable;
};