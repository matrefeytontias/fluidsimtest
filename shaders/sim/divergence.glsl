#version 450

layout(local_size_x = 32, local_size_y = 32) in;

uniform float uOneOverDx;

layout(binding = 0, r32f) uniform restrict readonly image2D uVelocityX;
layout(binding = 1, r32f) uniform restrict readonly image2D uVelocityY;
layout(binding = 2, r32f) uniform restrict writeonly image2D uDivergence;

// Velocity textures are staggered, and the divergence texture is centered.
// This means that divergence samples are in the middle of velocity samples,
// which allows for quick and accurate finite difference derivatives.

// TEST: collocated

void main()
{
	ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(uVelocityX) - 1;
	ivec2 zero = ivec2(0);

	float xleft = imageLoad(uVelocityX, max(zero, texel + ivec2(-1,  0))).r,
		 xright = imageLoad(uVelocityX, min(size, texel + ivec2( 1,  0))).r,
		    yup = imageLoad(uVelocityY, min(size, texel + ivec2( 0,  1))).r,
		  ydown = imageLoad(uVelocityY, max(zero, texel + ivec2( 0, -1))).r;

	float divergence = (xright - xleft + yup - ydown) * uOneOverDx * 0.5;
	imageStore(uDivergence, texel, vec4(divergence));
}
