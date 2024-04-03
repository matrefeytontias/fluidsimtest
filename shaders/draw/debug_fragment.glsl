#version 450

uniform sampler2DArray uTexture;
uniform usampler2DArray uIntTexture;

uniform bool uUseIntTexture;
uniform float uColorScale;

in vec3 vUV;
out vec4 fFragColor;

vec4 colors[4] = { vec4(0., 0., 0., 1.), vec4(1., 1., 1., 1.), vec4(0., 1., 0., 1.), vec4(1., 0., 0., 1.) };

float sampleTex(sampler2DArray tex, vec3 uv)
{
	vec3 size = textureSize(tex, 0);
	uv.z = uv.z * size.z - 0.5;

	float down = texture(tex, uv + vec3(0, 0, -0.5)).r;
	float up = texture(tex, uv + vec3(0, 0, 0.5)).r;

	return mix(down, up, fract(uv.z));
}

void main()
{
	if (uUseIntTexture)
	{
		uint index = texture(uIntTexture, vUV * vec3(1, 1, textureSize(uIntTexture, 0).z)).r;
		fFragColor = colors[index];
	}
	else
	{
		float color = sampleTex(uTexture, vUV) * uColorScale;
		fFragColor = vec4(max(0, color), max(0, -color), 0, 1);
	}
}
