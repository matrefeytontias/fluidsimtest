#version 450

uniform sampler2D uTexture;
uniform float uColorScale;

in vec2 vUV;
out vec4 fFragColor;

void main()
{
	float color = texture(uTexture, vUV).r * uColorScale;
	fFragColor = vec4(max(0, color), 0, max(0, -color), 1);
}
