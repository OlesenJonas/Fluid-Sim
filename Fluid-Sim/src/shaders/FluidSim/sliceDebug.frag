#version 430

in vec3 passTextureCoord;

uniform layout (binding = 0) sampler3D tex;

out vec4 outColor;

void main() {
    // outColor = vec4(texture(tex, passTextureCoord).rgb*0.5 + 0.5, 1.0);
    outColor = vec4(texture(tex, passTextureCoord).rgb, 1.0);
}