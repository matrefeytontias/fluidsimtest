#version 450

uniform float uHalfOneOverDx;

layout(binding = 0, r32f) uniform restrict image2D uVelocityX;
layout(binding = 1, r32f) uniform restrict image2D uVelocityY;
layout(binding = 2, r32f) uniform restrict readonly image2D uPressure;

// Velocity textures are staggered, and the pressure texture is centered.
// This means that pressure samples are in the middle of velocity samples,
// which allows for quick and accurate finite difference derivatives.

void compute(ivec2 texel, ivec2 outputTexel, bool boundaryTexel)
{
	// Velocity X and Y texels are in different locations, so they each need a
	// different half-pressure gradient. Both halves happen to share a texel.

	float pleft = imageLoad(uPressure, texel + ivec2(-1,  0)).r,
		 pright = imageLoad(uPressure, texel                ).r,
		    pup = pright,
		  pdown = imageLoad(uPressure, texel + ivec2( 0, -1)).r;

	vec2 pressureGradientHalves = uHalfOneOverDx * vec2(pright - pleft, pup - pdown);
	
	float oldx = imageLoad(uVelocityX, texel).r;
	float oldy = imageLoad(uVelocityY, texel).r;
	float newx = oldx - pressureGradientHalves.x;
	float newy = oldy - pressureGradientHalves.y;

	// Velocity respects the staggered no-slip boundary condition
	imageStore(uVelocityX, outputTexel, vec4(boundaryTexel ? 0. : newx));
	imageStore(uVelocityY, outputTexel, vec4(boundaryTexel ? 0. : newy));
}
