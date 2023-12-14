#version 450

uniform sampler2D uTexture;
uniform float uColorScale;

in vec2 vUV;
out vec4 fFragColor;

void main()
{
	vec4 color = texture(uTexture, vUV) * uColorScale;
	fFragColor = abs(color);
}
