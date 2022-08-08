#version 430

#ifdef GLSLANGVALIDATOR
    #define USE_EMISSIVE
#endif

// [0,1]
in vec3 localCord;

// ----

layout (binding = 0) uniform sampler3D tex;
layout (binding = 1) uniform sampler2D noise;
// layout (binding = 2) uniform sampler2D sceneDepth;
#ifdef USE_EMISSIVE
    layout (binding = 3) uniform sampler3D temperatureTex;
#endif

//cam position in volume local space
layout (location=3) uniform vec3 camPosLocal;
layout (location=4) uniform mat4 invProjection;
layout (location=5) uniform mat4 viewToLocal;

layout (std140, binding = 12) uniform renderSettings
{
    int planeAlignment;
    int maxSteps;
    int hardStepsLimit;
    int shadowSteps;

    float jitter;
    int noiseType;
    float henyeyAnisotropy;
    float ambientDensity;
    
    vec4 baseColor;
    
    vec3 lightVectorLocalUniform;
    float densityScale;
    
    vec4 sunColorAndStrength;
    
    vec3 shadowDensityFactor;
    int usePhaseFunction;
    
    vec4 skyColorAndStrength;

    float temperatureScale;
    float emissiveStrength;
};

// ----

out vec4 outColor;

// ----

//todo find
vec3 BlackBodyColor( float Temp )
{
    float u = ( 0.860117757f + 1.54118254e-4f * Temp + 1.28641212e-7f * Temp*Temp ) / ( 1.0f + 8.42420235e-4f * Temp + 7.08145163e-7f * Temp*Temp );
    float v = ( 0.317398726f + 4.22806245e-5f * Temp + 4.20481691e-8f * Temp*Temp ) / ( 1.0f - 2.89741816e-5f * Temp + 1.61456053e-7f * Temp*Temp );

    float x = 3*u / ( 2*u - 8*v + 4 );
    float y = 2*v / ( 2*u - 8*v + 4 );
    float z = 1 - x - y;

    float Y = 1;
    float X = Y/y * x;
    float Z = Y/y * z;

    const mat3 XYZtoRGB = mat3(
        3.2404542, -0.9692660, 0.0556434,
        -1.5371385,  1.8760108,  -0.2040259,
        -0.4985314, 0.0415560,  1.0572252
    );

    return ( XYZtoRGB * vec3( X, Y, Z ) ) * pow( 0.0004 * Temp, 4 );
}


// ----

// vec4 getViewPositionFromDepthAtPixel(ivec2 p)
// {
//     //this can definitly be more optimized
//     vec2 uv = vec2(p) / textureSize(sceneDepth, 0).xy;
//     float depth = texelFetch(sceneDepth, p, 0).x;
//     vec4 positionClipSpace = vec4(uv * 2.0 - 1.0, 2.0 * depth - 1.0, 1.0);
//     vec4 position = invProjection * positionClipSpace;
//     return (position / position.w);
// }

float henyeyGreensteinPhase(vec3 wi, vec3 wo)
{
    const float cosTheta = dot(wi, wo);
    const float g2 = henyeyAnisotropy*henyeyAnisotropy;
    const float numerator = 1-g2;
    const float denominator = pow(1 + g2 - 2*henyeyAnisotropy*cosTheta, 1.5);
    return (numerator/denominator)/(4*3.1415926538);
}

// from: https://www.jcgt.org/published/0009/03/02/paper.pdf
uvec3 pcg3d(uvec3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
    v ^= v >> 16u;
    v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
    return v;
}

float getShadowDistance(vec3 shadowSamplePos, int i, vec3 lightVectorLocal)
{
    float rndF = 0.0;
            //todo: offset noises by time?
    if(noiseType == 0)
    {
        /* pretty sure pcg3d is white noise, looks bad but still better than stepping imo */
        uvec3 rnd = pcg3d(floatBitsToUint(shadowSamplePos));
        rndF = float(rnd.x)/0xFFFFFFFFu;
    }
    else
    {
        /* texture based blue noise by Christoph Peters http://momentsingraphics.de/BlueNoise.html */
        ivec2 noiseCoord = (ivec2(gl_FragCoord.xy) + 17*i)%1024;
        float pixelNoise = texelFetch(noise, noiseCoord, 0).x;
        rndF = pixelNoise;
    }
    shadowSamplePos += lightVectorLocal*rndF*jitter;
    
    float shadowDistance = 0.0;
    for(int s=0; s<shadowSteps; s++)
    {
        shadowSamplePos += lightVectorLocal;
        // terminate early if shadowsample is outside [0,1]
        if( any( greaterThan(abs(shadowSamplePos-0.5),vec3(0.5)) ) )
        {
            //optionally also early out if some density limit is reached (see blog)
            break;
        }
        float shadowSample = texture(tex, shadowSamplePos).r;
        shadowDistance += shadowSample;
    }

    return shadowDistance;
}

float getAmbientShadowDistance(vec3 samplePos)
{
    float shadowDistance = 0.0;

    const vec3 sampleOffset1 = vec3(0,0,0.025);
    const vec3 sampleOffset2 = vec3(0,0,0.05);
    const vec3 sampleOffset3 = vec3(0,0,0.15);
    vec3 shadowSamplePos = samplePos + sampleOffset1;
    float shadowSample = texture(tex, shadowSamplePos).r;
    shadowDistance += shadowSample;
    shadowSamplePos = samplePos + sampleOffset2;
    shadowSample = texture(tex, shadowSamplePos).r;
    shadowDistance += shadowSample;
    shadowSamplePos = samplePos + sampleOffset3;
    shadowSample = texture(tex, shadowSamplePos).r;
    shadowDistance += shadowSample;

    return shadowDistance;
}

// ----

void main() {

    //https://shaderbits.com/blog/creating-volumetric-ray-marcher
    // https://github.com/sp0lsh/UEShaderBits-GDC-Pack

    // viewDir is ray direction, ie starting from camera along view direction
    vec3 viewDirLocal = normalize(localCord-camPosLocal);

    //calculating ray start & step
    //todo: test against scene depth!
    vec3 invRayDir = 1.0/viewDirLocal;
    vec3 firstIntersections = (0 - camPosLocal)*invRayDir;
    vec3 secondIntersections = (1 - camPosLocal)*invRayDir;
    vec3 closest = min(firstIntersections, secondIntersections);
    vec3 furthest = max(firstIntersections, secondIntersections);
    float t0 = max(closest.x, max(closest.y, closest.z));
    float t1 = min(furthest.x, min(furthest.y, furthest.z));
    float planeoffset = 1.0-fract((t0-length(camPosLocal-0.5))*maxSteps);
    t0 += (planeoffset / maxSteps) * planeAlignment;
    t0 = max(0, t0);
    //account for scene depth
    // const vec4 sceneDepthPosViewSpace = getViewPositionFromDepthAtPixel(ivec2(gl_FragCoord.xy));
    // const vec4 sceneDepthPosLocalSpace = viewToLocal * sceneDepthPosViewSpace;
    // float tmax = distance(camPosLocal,sceneDepthPosLocalSpace.xyz);
    // t1 = min(t1, tmax);

    float boxthickness = max(0, t1 - t0);

    vec3 entryPos = camPosLocal + (max(0,t0) * viewDirLocal);

    int steps = int(floor(boxthickness*maxSteps));
    float finalStepFactor = fract(boxthickness*maxSteps);
    //hardcoded step limits
    steps = clamp(steps,0,hardStepsLimit);


    //ray marcher - simple but expensive brute force out scattering
    vec3 samplePos = entryPos;
    float stepSize = 1.0/maxSteps;
    float shadowStepSize = 1.0/shadowSteps;
    
    vec3 lightVectorLocal = normalize(lightVectorLocalUniform);
    vec3 lightColor = sunColorAndStrength.xyz * sunColorAndStrength.w;
    if(usePhaseFunction == 1)
    {
        lightColor *= henyeyGreensteinPhase(-lightVectorLocal, -viewDirLocal);
    }
    lightVectorLocal *= shadowStepSize;
    vec3 shadowDensity = shadowDensityFactor * shadowStepSize;

    float density = densityScale * stepSize;


    //begin raymarch loop
    float currentDensity = 0;
    float transmittance = 1;

    vec3 light = vec3(0.0);
    vec3 emissive = vec3(0.0);

    for(int i=0; i<steps; i++)
    {
        float curSample = texture(tex, samplePos).r;

        if(curSample > 0.001)
        {
            float shadowDistance = getShadowDistance(samplePos, i, lightVectorLocal);

            currentDensity = 1.0 - exp(-curSample*density);
            vec3 shadowFactor = exp(-shadowDistance * shadowDensity.xyz);
            vec3 absorbedLight = shadowFactor * currentDensity;
            light += absorbedLight * transmittance * lightColor;
            #ifdef USE_EMISSIVE
                //todo: 4000 and 1.0 as parameter
                float sampleTemp = texture(temperatureTex, samplePos).r * 4000 * temperatureScale;
                emissive += BlackBodyColor(sampleTemp) * currentDensity * transmittance * emissiveStrength;
            #endif
            transmittance *= 1-currentDensity;


            shadowDistance = getAmbientShadowDistance(samplePos);
            
            shadowFactor = vec3(exp(-shadowDistance * ambientDensity));
            absorbedLight = shadowFactor * currentDensity;
            light += absorbedLight * transmittance * skyColorAndStrength.rgb*skyColorAndStrength.a;

        }
        samplePos += viewDirLocal*stepSize;
    }

    //final fractional step (last iteration in loop added a full step at end, so subtract 1.0-fraction again)
    samplePos -= viewDirLocal*stepSize*(1.0-finalStepFactor);

    {
        float curSample = texture(tex, samplePos).r;

        if(curSample > 0.001)
        {
            float shadowDistance = getShadowDistance(samplePos, 7759, lightVectorLocal);

            currentDensity = 1.0 - exp(-curSample*density*finalStepFactor);
            vec3 shadowFactor = exp(-shadowDistance * shadowDensity.xyz);
            vec3 absorbedLight = shadowFactor * currentDensity;
            light += absorbedLight * transmittance * lightColor;
            transmittance *= 1-currentDensity;

            shadowDistance = getAmbientShadowDistance(samplePos);
            
            shadowFactor = vec3(exp(-shadowDistance * ambientDensity));
            absorbedLight = shadowFactor * currentDensity;
            light += absorbedLight * transmittance * skyColorAndStrength.rgb*skyColorAndStrength.a;

        }
        samplePos += viewDirLocal*stepSize;
    }

    vec3 color = baseColor.a * baseColor.rgb * light;
    color     += baseColor.a * emissive;
    float alpha = 1.0-transmittance;

    outColor = vec4(color,alpha);
}