#version 430

layout (location = 0) in vec4 position;

insert /Buffers/lightBuffer.txt

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform uint lightID;

void main(){
    Light light = lights[lightID];
    gl_Position = light.shadowMatrix * modelMatrix * position;
}