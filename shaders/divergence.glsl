#version 450

layout(local_size_x = 32, local_size_y = 32) in;

uniform float uHalfOneOverDx;

layout(binding = 0, rg32f) uniform restrict readonly image2D uVelocity;
layout(binding = 1, r32f) uniform restrict writeonly image2D uFieldOut;

void main()
{
	ivec2 texel = ivec2(gl_GlobalInvocationID.xy);

	float xleft = imageLoad(uVelocity, texel + ivec2(-1,  0)).x,
		 xright = imageLoad(uVelocity, texel + ivec2( 1,  0)).x,
		    yup = imageLoad(uVelocity, texel + ivec2( 0,  1)).y,
		  ydown = imageLoad(uVelocity, texel + ivec2( 0, -1)).y;

	float divergence = (xright - xleft + yup - ydown) * uHalfOneOverDx;
	imageStore(uFieldOut, texel, vec4(divergence));
}
