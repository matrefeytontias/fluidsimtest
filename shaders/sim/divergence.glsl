#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

uniform float uOneOverDx;

layout(binding = 0, r32f) uniform restrict readonly image2DArray uVelocityX;
layout(binding = 1, r32f) uniform restrict readonly image2DArray uVelocityY;
layout(binding = 2, r32f) uniform restrict readonly image2DArray uVelocityZ;
layout(binding = 3, r32f) uniform restrict writeonly image2DArray uDivergence;

// Velocity textures are staggered, and the divergence texture is centered.
// This means that divergence samples are in the middle of velocity samples,
// which allows for quick and accurate finite difference derivatives.

void main()
{
	ivec3 texel = ivec3(gl_GlobalInvocationID);
	ivec3 size = imageSize(uVelocityX) - 1;
	ivec2 s = ivec2(1, 0);
	ivec3 zero = ivec3(0);

	// Clamp coordinates so gradients are 0 on the boundary
	// TEST: collocated grid
	float xleft = imageLoad(uVelocityX, max(zero, texel - s.xyy)).r,
		 xright = imageLoad(uVelocityX, min(size, texel + s.xyy)).r,
		    yup = imageLoad(uVelocityY, min(size, texel + s.yxy)).r,
		  ydown = imageLoad(uVelocityY, max(zero, texel - s.yxy)).r,
		 zfront = imageLoad(uVelocityZ, min(size, texel + s.yyx)).r,
		  zback = imageLoad(uVelocityZ, max(zero, texel - s.yyx)).r;

	// TEST: collocated grid
	float divergence = (xright - xleft + yup - ydown + zfront - zback) * uOneOverDx * 0.5;
	imageStore(uDivergence, texel, vec4(divergence));
}
