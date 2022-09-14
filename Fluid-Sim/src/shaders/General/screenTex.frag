#version 430

in vec2 passTextureCoord;

out vec4 fragmentColor;

layout(binding=0) uniform sampler2D tex;

void main()
{
    fragmentColor = textureLod(tex,passTextureCoord,0);
}