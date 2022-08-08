#include "BezierSpline.h"

BezierSpline::BezierSpline(std::span<glm::vec3> positions, std::span<glm::vec3> tangents, bool closed)
{
    assert(positions.size() > 1);
    assert(positions.size() == tangents.size());

    // Transform positions + tangents list into list of control points for bezier curve

    int curves = int(positions.size()) - 1;
    curves += int(closed);
    // all curves except for the first one share one controlpoint
    int totalControlPoints = curves * 3 + 1;

    controlPoints.reserve(totalControlPoints);

    // first point + tangent
    controlPoints.push_back(positions[0]);
    controlPoints.push_back(positions[0] + tangents[0] / 3.0f);
    // all interior points
    for(int i = 1; i < positions.size() - 1; i++)
    {
        controlPoints.push_back(positions[i] - tangents[i] / 3.0f);
        controlPoints.push_back(positions[i]);
        controlPoints.push_back(positions[i] + tangents[i] / 3.0f);
    }
    // last point
    auto lastIndex = positions.size() - 1;
    controlPoints.push_back(positions[lastIndex] - tangents[lastIndex] / 3.0f);
    controlPoints.push_back(positions[lastIndex]);
    if(closed)
    {
        controlPoints.push_back(positions[lastIndex] + tangents[lastIndex] / 3.0f);
        controlPoints.push_back(positions[0] - tangents[0] / 3.0f);
        controlPoints.push_back(positions[0]);
    }

    // Try to distribute arc length table entries evenly to all segments
    // ie amount of entries should be proportional to segment's fraction of total length
    // todo: test how well this actually works

    cubicBezierCurves.reserve(curves);
    std::vector<float> segmentLengths;
    segmentLengths.reserve(curves);
    float totalLength = 0.0f;
    for(int i = 0; i < curves; i++)
    {
        CubicBezierCurve& newCurve =
            cubicBezierCurves.emplace_back(CubicBezierCurve{&controlPoints[i * 3], 4});

        float segmentLength = 0.0f;
        int res = 20;
        glm::vec3 oldPos = newCurve[0];
        for(int j = 1; j <= res; j++)
        {
            glm::vec3 newPos = getPosition(newCurve, float(j) / float(res));
            segmentLength += glm::distance(newPos, oldPos);
            oldPos = newPos;
        }
        segmentLengths.push_back(segmentLength);
        totalLength += segmentLength;
    }

    int arcLengthTableSize = 200;

    // try to balance rounding errors for all curves by keeping track of the rounding error for previous
    // segments
    float roundingError = 0.0f;
    int entriesUsed = 0;
    std::vector<int> entriesPerSegment(curves);
    for(int i = 0; i < curves - 1; i++)
    {
        float entriesToUseOptimal = (segmentLengths[i] / totalLength) * arcLengthTableSize + roundingError;
        int entriesToUse = std::round(entriesToUseOptimal);
        roundingError = entriesToUseOptimal - entriesToUse;
        entriesPerSegment[i] = entriesToUse;
        entriesUsed += entriesToUse;
    }
    // make sure total amount of entries used is !exactly! arcLengthTableSize
    int entriesToUse = arcLengthTableSize - entriesUsed;
    entriesPerSegment[curves - 1] = entriesToUse;
    entriesUsed += entriesToUse;
    assert(entriesUsed == arcLengthTableSize);

    //+1 since first entry is { length 0 -> (curveIndex 0, parameter 0) }
    arcLengthTable.reserve(arcLengthTableSize + 1);
    arcLengthTable.emplace_back(ArcLengthTableEntry{0.0f, 0, 0.0f});

    float accumLength = 0.0f;
    glm::vec3 oldPos = cubicBezierCurves[0][0];
    for(int i = 0; i < curves; i++)
    {
        CubicBezierCurve curve = cubicBezierCurves[i];
        for(int j = 1; j <= entriesPerSegment[i]; j++)
        {
            float t = float(j) / float(entriesPerSegment[i]);
            glm::vec3 newPos = getPosition(curve, t);
            float distance = glm::distance(newPos, oldPos);
            accumLength += distance;
            arcLengthTable.emplace_back(ArcLengthTableEntry{accumLength, i, t});
            oldPos = newPos;
        }
    }

    // normalize arc length table
    for(auto& entry : arcLengthTable)
    {
        entry.accumLength = entry.accumLength / accumLength;
    }
}

glm::vec3 BezierSpline::getPosition(int segment, float parameter)
{
    return getPosition(cubicBezierCurves[segment], parameter);
}

glm::vec3 BezierSpline::getPosition(CubicBezierCurve bezierSegment, float parameter)
{
    float t = parameter; // NOLINT
    float tSq = t * t;
    float tCub = t * t * t;
    float OneMinusT = 1.0f - parameter;
    float OneMinusTSq = OneMinusT * OneMinusT;
    float OneMinusTCub = OneMinusT * OneMinusT * OneMinusT;

    return OneMinusTCub * bezierSegment[0] +        //
           3 * OneMinusTSq * t * bezierSegment[1] + //
           3 * OneMinusT * tSq * bezierSegment[2] + //
           tCub * bezierSegment[3];
}

glm::vec3 BezierSpline::getTangent(CubicBezierCurve bezierSegment, float parameter)
{
    float t = parameter; // NOLINT
    float tSq = t * t;
    float OneMinusT = 1.0f - parameter;
    float OneMinusTSq = OneMinusT * OneMinusT;

    return 3 * OneMinusTSq * (bezierSegment[1] - bezierSegment[0]) +   //
           6 * OneMinusT * t * (bezierSegment[2] - bezierSegment[1]) + //
           3 * tSq * (bezierSegment[3] - bezierSegment[2]);
}

std::pair<int, float> BezierSpline::getCurveAndParameter(float factor)
{
    // binary search through arcLengthTable then interpolate closest two entries
    int left = 0;
    int right = arcLengthTable.size() - 1;
    int middle = -1;
    while(right - left > 1)
    {
        middle = left + (right - left) / 2;
        if(factor < arcLengthTable[middle].accumLength)
        {
            right = middle;
        }
        else
        {
            left = middle;
        }
    }
    // inverse interpolate factor to get the fraction between two entries
    // then interpolate to get final parameter
    // take care across boundaries
    // (ie: if left is part of a different segment) then we shouldnt use that entries parameter for
    // interpolation

    float factorLeft = arcLengthTable[left].accumLength;
    float factorRight = arcLengthTable[right].accumLength;
    float fraction = (factor - factorLeft) / (factorRight - factorLeft);

    // always use right entry's segment
    int curveIndex = arcLengthTable[right].curveIndex;
    float parameterLeft = arcLengthTable[left].parameter;
    float parameterRight = arcLengthTable[right].parameter;
    // if the left entry's curveIndex is != right entry's curveIndex
    // then instead of treating it as t=1 of the last curve treat it as t=0 of the current curve
    if(arcLengthTable[left].curveIndex != curveIndex)
    {
        parameterLeft = 0.0f;
    }
    // return interpolated parameter
    float parameter = (1.0f - fraction) * parameterLeft + fraction * parameterRight;

    return {curveIndex, parameter};
}

glm::vec3 BezierSpline::getPosition(float factor)
{
    const auto segmentAndParameter = getCurveAndParameter(factor);
    return getPosition(cubicBezierCurves[segmentAndParameter.first], segmentAndParameter.second);
}

std::pair<glm::vec3, glm::vec3> BezierSpline::getPositionAndTangent(float factor)
{
    const auto segmentAndParameter = getCurveAndParameter(factor);
    return {
        getPosition(cubicBezierCurves[segmentAndParameter.first], segmentAndParameter.second),
        getTangent(cubicBezierCurves[segmentAndParameter.first], segmentAndParameter.second)};
}