#version 450

uniform float uHalfOneOverDx;

layout(binding = 0, rg32f) uniform restrict image2D uVelocity;
layout(binding = 1, r32f) uniform restrict readonly image2D uPressure;

void compute(ivec2 texel, ivec2 outputTexel, bool boundaryTexel)
{
	float pleft = imageLoad(uPressure, texel + ivec2(-1,  0)).r,
		 pright = imageLoad(uPressure, texel + ivec2( 1,  0)).r,
		    pup = imageLoad(uPressure, texel + ivec2( 0,  1)).r,
		  pdown = imageLoad(uPressure, texel + ivec2( 0, -1)).r;

	vec2 pressureGradient = uHalfOneOverDx * vec2(pright - pleft, pup - pdown);
	
	vec4 old = imageLoad(uVelocity, texel);
	vec4 newValue = old - pressureGradient.xyxx;

	// Velocity respects the no-slip boundary condition
	imageStore(uVelocity, outputTexel, boundaryTexel ? -newValue : newValue);
}
