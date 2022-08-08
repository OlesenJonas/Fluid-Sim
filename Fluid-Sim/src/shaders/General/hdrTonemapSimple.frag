#version 430

uniform layout (binding = 0) sampler2D sceneColor;

uniform layout (location = 0) float simpleExposure = 1.0;
uniform layout (location = 1) int mode = 0;

in vec2 passTextureCoord;

out vec4 fragmentColor;

// ----

vec3 linearToGammaSRGB(vec3 linearRGB)
{
    bvec3 cutoff = lessThan(linearRGB.rgb, vec3(0.0031308));
    vec3 higher = vec3(1.055)*pow(linearRGB.rgb, vec3(1.0/2.4)) - vec3(0.055);
    vec3 lower = linearRGB.rgb * vec3(12.92);

    return mix(higher, lower, cutoff);
}

// ----

// by Krzysztof Narkowicz: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 ACESFilm(vec3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

// by Stephan Hill: https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
const mat3 ACESInputMat = mat3(
    0.59719, 0.07600, 0.02840,
    0.35458, 0.90834, 0.13383,
    0.04823, 0.01566, 0.83777
);
const mat3 ACESOutputMat = mat3(
    1.60475, -0.10208, -0.00327,
    -0.53108,  1.10813, -0.07276,
    -0.07367, -0.00605,  1.07602
);
vec3 RRTAndODTFit(vec3 v)
{
    vec3 a = v * (v + 0.0245786f) - 0.000090537f;
    vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}
vec3 ACESFitted(vec3 color)
{
    color = ACESInputMat*color;

    // Apply RRT and ODT
    color = RRTAndODTFit(color);

    color = ACESOutputMat*color;

    color = clamp(color, 0.0, 1.0);

    return color;
}

// ----

void main() {
    vec3 sceneCol = texelFetch(sceneColor, ivec2(gl_FragCoord), 0).rgb;

    sceneCol *= simpleExposure;

    if(mode == 0)
    {
        sceneCol = ACESFilm(sceneCol);
    }
    else
    {
        sceneCol = ACESFitted(sceneCol);
    }
    
    fragmentColor.xyz = linearToGammaSRGB(sceneCol);
    fragmentColor.a = 1.0;
}