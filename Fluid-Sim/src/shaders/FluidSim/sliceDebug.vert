#version 430

layout (location = 0) in vec4 position;
layout (location = 2) in vec2 textureCoord;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

out vec3 passTextureCoord;

void main(){
    // passTextureCoord = textureCoord;
    vec4 worldPos =  modelMatrix * position;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
    passTextureCoord = 0.5 + 0.5*worldPos.xyz;
}