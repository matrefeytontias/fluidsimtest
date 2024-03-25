#version 450

layout(local_size_x = 32, local_size_y = 32) in;

uniform float uOneOverDx;

layout(binding = 0, r32f) uniform restrict image2D uVelocityX;
layout(binding = 1, r32f) uniform restrict image2D uVelocityY;
layout(binding = 2, r32f) uniform restrict readonly image2D uPressure;

// Velocity textures are staggered, and the pressure texture is centered.
// This means that pressure samples are in the middle of velocity samples,
// which allows for quick and accurate finite difference derivatives.
//
// Because we handle two staggered textures at once, we give
// uStaggeredField = 0 to the entry point and do our own boundary detection
// logic.
void main()
{
	ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(uVelocityX) - 1;
	ivec2 zero = ivec2(0);

	// Velocity X and Y texels are in different locations, so they each need a
	// different half-pressure gradient. Both halves happen to share a texel.
	// TEST: collocated
	float pleft = imageLoad(uPressure, max(zero, texel + ivec2(-1,  0))).r,
		 pright = imageLoad(uPressure, min(size, texel + ivec2( 1,  0))).r,
		    pup = imageLoad(uPressure, min(size, texel + ivec2( 0,  1))).r,
		  pdown = imageLoad(uPressure, max(zero, texel + ivec2( 0, -1))).r;

	vec2 pressureGradientHalves = uOneOverDx * vec2(pright - pleft, pup - pdown) * 0.5;
	
	float oldx = imageLoad(uVelocityX, texel).r;
	float oldy = imageLoad(uVelocityY, texel).r;
	float newx = oldx - pressureGradientHalves.x;
	float newy = oldy - pressureGradientHalves.y;

	// Boundary detection for each velocity field
	bvec2 boundaryVelX = equal(texel, ivec2(0)) || equal(texel, size - 1);
	bvec2 boundaryVelY = equal(texel, ivec2(0)) || equal(texel, size - 1);

	// Velocity respects the staggered no-slip boundary condition
	imageStore(uVelocityX, texel, vec4(/*texel.x == 0
		? 0.
		: boundaryVelX.x
			? 0.
			: */newx));
	imageStore(uVelocityY, texel, vec4(/*texel.y == 0
		? 0.
		: boundaryVelY.y
			? 0.
			: */newy));
}
