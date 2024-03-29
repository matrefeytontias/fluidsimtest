#version 450

uniform sampler3D uTexture;
uniform usampler3D uIntTexture;

uniform bool uUseIntTexture;
uniform float uColorScale;

in vec3 vUV;
out vec4 fFragColor;

vec4 colors[4] = { vec4(0., 0., 0., 1.), vec4(1., 1., 1., 1.), vec4(0., 1., 0., 1.), vec4(1., 0., 0., 1.) };

void main()
{
	if (uUseIntTexture)
	{
		uint index = texture(uIntTexture, vUV).r;
		fFragColor = colors[index];
	}
	else
	{
		vec4 color = texture(uTexture, vUV) * uColorScale;
		fFragColor = abs(color);
	}
}
