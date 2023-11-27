#version 450

uniform sampler3D uTexture;
uniform float uColorScale;

in vec3 vUV;
out vec4 fFragColor;

void main()
{
	vec4 color = texture(uTexture, vUV) * uColorScale;
	fFragColor = abs(color);
}
