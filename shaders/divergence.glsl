#version 450

layout(local_size_x = 32, local_size_y = 32) in;

uniform float uHalfOneOverDx;

layout(binding = 0, r32f) uniform restrict readonly image2D uVelocityX;
layout(binding = 1, r32f) uniform restrict readonly image2D uVelocityY;
layout(binding = 2, r32f) uniform restrict writeonly image2D uFieldOut;

void main()
{
	ivec2 texel = ivec2(gl_GlobalInvocationID.xy);

	float xleft = imageLoad(uVelocityX, texel + ivec2(-1,  0)).r,
		 xright = imageLoad(uVelocityX, texel + ivec2( 1,  0)).r,
		    yup = imageLoad(uVelocityY, texel + ivec2( 0,  1)).r,
		  ydown = imageLoad(uVelocityY, texel + ivec2( 0, -1)).r;

	float divergence = (xright - xleft + yup - ydown) * uHalfOneOverDx;
	imageStore(uFieldOut, texel, vec4(divergence));
}
