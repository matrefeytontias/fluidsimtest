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
	ivec3 size = imageSize(uVelocityX) - 1;

	// Velocity X, Y and Z texels are in different locations, so they each need a
	// coordinate from different gradients, which happen to share a texel.

	float pleft = imageLoad(uPressure, texel + ivec3(-1,  0,  0)).r,
		 pright = imageLoad(uPressure, texel                    ).r,
		    pup = pright,
		  pdown = imageLoad(uPressure, texel + ivec3( 0, -1,  0)).r,
		 pfront = pright,
		  pback = imageLoad(uPressure, texel + ivec3( 0,  0, -1)).r;

	vec3 pressureGradientComponents = uOneOverDx * vec3(pright - pleft, pup - pdown, pfront - pback);
	
	float oldx = imageLoad(uVelocityX, texel).r;
	float oldy = imageLoad(uVelocityY, texel).r;
	float oldz = imageLoad(uVelocityZ, texel).r;
	float newx = oldx - pressureGradientComponents.x;
	float newy = oldy - pressureGradientComponents.y;
	float newz = oldz - pressureGradientComponents.z;

	// Boundary detection for each velocity field
	bvec3 boundaryVelX = lessThanEqual(texel, ivec3(1, 0, 0)) || equal(texel, size);
	bvec3 boundaryVelY = lessThanEqual(texel, ivec3(0, 1, 0)) || equal(texel, size);
	bvec3 boundaryVelZ = lessThanEqual(texel, ivec3(0, 0, 1)) || equal(texel, size);

	// Velocity respects the staggered no-slip boundary condition
	imageStore(uVelocityX, texel, vec4(any(boundaryVelX) ? 0. : newx));
	imageStore(uVelocityY, texel, vec4(any(boundaryVelY) ? 0. : newy));
	imageStore(uVelocityZ, texel, vec4(any(boundaryVelZ) ? 0. : newz));
}
