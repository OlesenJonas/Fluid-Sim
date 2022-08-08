#version 430

layout (location = 0) in vec4 position;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

out vec3 passPositionCamera;

void main(){
    passPositionCamera = (viewMatrix * modelMatrix * position).xyz;
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * position;
}