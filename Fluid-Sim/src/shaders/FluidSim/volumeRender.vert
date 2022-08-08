#version 430

layout (location = 0) in vec4 position;
layout (location = 2) in vec2 textureCoord;

layout (location = 0) uniform mat4 modelMatrix;
layout (location = 1) uniform mat4 viewMatrix;
layout (location = 2) uniform mat4 projectionMatrix;

out vec3 localCord;

void main(){
    localCord = position.xyz;
    vec4 worldPos =  modelMatrix * position;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
}