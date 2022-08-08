#version 430

layout (location = 0) in vec4 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 textureCoord;
layout (location = 3) in vec3 tangent;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

out vec3 passPositionCamera;
out vec3 passPositionWorld;
out vec3 passNormalCamera;
out vec2 passTextureCoord;
out vec3 passTangentCamera;

void main(){
    passPositionCamera = (viewMatrix * modelMatrix * position).xyz;
    passPositionWorld = (modelMatrix * position).xyz;
    passNormalCamera = normalize(vec3(transpose(inverse(viewMatrix * modelMatrix)) * vec4(normal, 0.0)));
    passTextureCoord = textureCoord;
    passTangentCamera = normalize(vec3(transpose(inverse(viewMatrix * modelMatrix)) * vec4(tangent, 0.0)));
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * position;
}