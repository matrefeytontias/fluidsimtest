#version 450

layout(local_size_x = 32, local_size_y = 32) in;

uniform float udx;
uniform float udt;

layout(binding = 0, rg32f) uniform readonly image2D uVelocity;

layout(binding = 1) uniform sampler2D uFieldIn;
layout(binding = 2, r32f) uniform restrict writeonly image2D uFieldOut;

vec2 texelSpaceToGridSpace(ivec2 p)
{
	return (vec2(p) + 0.5) * udx;
}

vec2 gridSpaceToUV(vec2 p)
{
	return p / (udx * vec2(textureSize(uFieldIn, 0)));
}

void main()
{
	ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
	if(any(texel == 0) || any(texel == imageSize(uFieldOut) - 1))
		return;
	
	vec2 velocity = imageLoad(uVelocity, texel).rg;

	vec2 oldPosition = texelSpaceToGridSpace(texel) - velocity * udt;
	vec4 oldValue = texture(uFieldIn, gridSpaceToUV(oldPosition));
	imageStore(uFieldOut, texel, oldValue);
}