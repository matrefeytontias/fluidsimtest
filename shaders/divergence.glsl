#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

uniform float uHalfOneOverDx;

layout(binding = 0, r32f) uniform restrict readonly image3D uVelocityX;
layout(binding = 1, r32f) uniform restrict readonly image3D uVelocityY;
layout(binding = 2, r32f) uniform restrict readonly image3D uVelocityZ;
layout(binding = 3, r32f) uniform restrict writeonly image3D uDivergence;

// Velocity textures are staggered, and the divergence texture is centered.
// This means that divergence samples are in the middle of velocity samples,
// which allows for quick and accurate finite difference derivatives.

void main()
{
	ivec3 texel = ivec3(gl_GlobalInvocationID);

	float xleft = imageLoad(uVelocityX, texel                 ).r,
		 xright = imageLoad(uVelocityX, texel + ivec3(1, 0, 0)).r,
		    yup = imageLoad(uVelocityY, texel + ivec3(0, 1, 0)).r,
		  ydown = imageLoad(uVelocityY, texel                 ).r,
		 zfront = imageLoad(uVelocityZ, texel + ivec3(0, 0, 1)).r,
		  zback = imageLoad(uVelocityZ, texel                 ).r;

	float divergence = (xright - xleft + yup - ydown + zfront - zback) * uHalfOneOverDx;
	imageStore(uDivergence, texel, vec4(divergence));
}
