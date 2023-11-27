#version 450

uniform float uHalfOneOverDx;

layout(binding = 0, r32f) uniform restrict image3D uVelocityX;
layout(binding = 1, r32f) uniform restrict image3D uVelocityY;
layout(binding = 2, r32f) uniform restrict image3D uVelocityZ;
layout(binding = 3, r32f) uniform restrict readonly image3D uPressure;

// Velocity textures are staggered, and the pressure texture is centered.
// This means that pressure samples are in the middle of velocity samples,
// which allows for quick and accurate finite difference derivatives.

void compute(ivec3 texel, ivec3 outputTexel, bool boundaryTexel)
{
	// Velocity X, Y and Z texels are in different locations, so they each need a
	// coordinate from a different gradient, which each happen to share a texel.

	float pleft = imageLoad(uPressure, texel + ivec3(-1,  0,  0)).r,
		 pright = imageLoad(uPressure, texel                    ).r,
		    pup = pright,
		  pdown = imageLoad(uPressure, texel + ivec3( 0, -1,  0)).r,
		 pfront = pright,
		  pback = imageLoad(uPressure, texel + ivec3( 0,  0, -1)).r;

	vec3 pressureGradientComponents = uHalfOneOverDx * vec3(pright - pleft, pup - pdown, pfront - pback);
	
	float oldx = imageLoad(uVelocityX, texel).r;
	float oldy = imageLoad(uVelocityY, texel).r;
	float oldz = imageLoad(uVelocityZ, texel).r;
	float newx = oldx - pressureGradientComponents.x;
	float newy = oldy - pressureGradientComponents.y;
	float newz = oldz - pressureGradientComponents.z;

	// Velocity respects the staggered no-slip boundary condition
	imageStore(uVelocityX, outputTexel, vec4(boundaryTexel ? 0. : newx));
	imageStore(uVelocityY, outputTexel, vec4(boundaryTexel ? 0. : newy));
	imageStore(uVelocityZ, outputTexel, vec4(boundaryTexel ? 0. : newz));
}
