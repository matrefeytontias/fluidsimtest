#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

uniform float uOneOverDx;

layout(binding = 0, r32f) uniform restrict image2DArray uVelocityX;
layout(binding = 1, r32f) uniform restrict image2DArray uVelocityY;
layout(binding = 2, r32f) uniform restrict image2DArray uVelocityZ;
layout(binding = 3, r32f) uniform restrict readonly image2DArray uPressure;

// Velocity textures are staggered, and the pressure texture is centered.
// This means that pressure samples are in the middle of velocity samples,
// which allows for quick and accurate finite difference derivatives.
//
// Because we handle two staggered textures at once, we don't use
// the entry point and do our own boundary detection logic.

void main()
{
	ivec3 texel = ivec3(gl_GlobalInvocationID);
	ivec3 size = imageSize(uPressure) - 1;
	ivec2 s = ivec2(1, 0);
	ivec3 zero = ivec3(0);

	// Velocity X, Y and Z texels are in different locations, so they each need a
	// coordinate from different gradients, which happen to share a texel.
	// Clamp coordinates so gradients are 0 on the boundary
	// TEST: collocated grid
	float pleft = imageLoad(uPressure, max(zero, texel - s.xyy)).r,
		 pright = imageLoad(uPressure, min(size, texel + s.xyy)).r,
		    pup = imageLoad(uPressure, min(size, texel + s.yxy)).r,
		  pdown = imageLoad(uPressure, max(zero, texel - s.yxy)).r,
		 pfront = imageLoad(uPressure, min(size, texel + s.yyx)).r,
		  pback = imageLoad(uPressure, max(zero, texel - s.yyx)).r;

	// TEST: collocated grid
	vec3 pressureGradientComponents = uOneOverDx * vec3(pright - pleft, pup - pdown, pfront - pback) * 0.5;
	
	float oldx = imageLoad(uVelocityX, texel).r;
	float oldy = imageLoad(uVelocityY, texel).r;
	float oldz = imageLoad(uVelocityZ, texel).r;
	float newx = oldx - pressureGradientComponents.x;
	float newy = oldy - pressureGradientComponents.y;
	float newz = oldz - pressureGradientComponents.z;

	// TEST: collocated grid
	imageStore(uVelocityX, texel, vec4(/*texel.x == 0 ? 0 : */newx));
	imageStore(uVelocityY, texel, vec4(/*texel.y == 0 ? 0 : */newy));
	imageStore(uVelocityZ, texel, vec4(/*texel.z == 0 ? 0 : */newz));
}
