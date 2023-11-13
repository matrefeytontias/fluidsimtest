#version 450

uniform float uHalfOneOverDx;

layout(binding = 0, r32f) uniform restrict image2D uVelocityX;
layout(binding = 1, r32f) uniform restrict image2D uVelocityY;
layout(binding = 2, r32f) uniform restrict readonly image2D uPressure;

void compute(ivec2 texel, ivec2 outputTexel, bool boundaryTexel)
{
	float pleft = imageLoad(uPressure, texel + ivec2(-1,  0)).r,
		 pright = imageLoad(uPressure, texel + ivec2( 1,  0)).r,
		    pup = imageLoad(uPressure, texel + ivec2( 0,  1)).r,
		  pdown = imageLoad(uPressure, texel + ivec2( 0, -1)).r;

	vec2 pressureGradient = uHalfOneOverDx * vec2(pright - pleft, pup - pdown);
	
	float oldx = imageLoad(uVelocityX, texel).r;
	float oldy = imageLoad(uVelocityY, texel).r;
	float newx = oldx - pressureGradient.x;
	float newy = oldy - pressureGradient.y;

	// Velocity respects the no-slip boundary condition
	imageStore(uVelocityX, outputTexel, vec4(boundaryTexel ? -newx : newx));
	imageStore(uVelocityY, outputTexel, vec4(boundaryTexel ? -newy : newy));
}
