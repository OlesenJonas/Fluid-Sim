#version 430

layout (location = 0) in vec4 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 textureCoord;
layout (location = 3) in vec3 tangent;

insert /Buffers/objectBuffer.txt

uniform uint objectID;
uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

insert /Functions/objectModel.txt

out vec3 passPositionCamera;
out vec3 passPositionWorld;
out vec3 passNormalCamera;
out vec2 passTextureCoord;
out vec3 passTangentCamera;

void main(){
    Object object = objects[objectID];
    mat4 objectModelMatrix = getModelMatrix(object.position, object.rotation, object.scale);
    passPositionCamera = (viewMatrix * objectModelMatrix * position).xyz;
    passPositionWorld = (objectModelMatrix * position).xyz;
    passNormalCamera = normalize(vec3(transpose(inverse(viewMatrix * objectModelMatrix)) * vec4(normal, 0.0)));
    passTextureCoord = textureCoord;
    passTangentCamera = normalize(vec3(transpose(inverse(viewMatrix * objectModelMatrix)) * vec4(tangent, 0.0)));
    gl_Position = projectionMatrix * viewMatrix * objectModelMatrix * position;
}