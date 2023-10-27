#version 450

layout(local_size_x = 32, local_size_y = 32) in;

uniform float uHalfOneOverDx;

layout(binding = 0, r32f) uniform restrict image2D uVelocityX;
layout(binding = 1, r32f) uniform restrict image2D uVelocityY;
// layout(binding = 2, r32f) uniform restrict image2D uVelocityZ;

layout(binding = 3, r32f) uniform restrict readonly image2D uPressure;

void main()
{
	ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
	if(any(texel == 0) || any(texel == imageSize(uVelocityX) - 1))
		return;

	float pleft = imageLoad(uPressure, texel + ivec2(-1,  0)).r,
		 pright = imageLoad(uPressure, texel + ivec2( 1,  0)).r,
		    pup = imageLoad(uPressure, texel + ivec2( 0,  1)).r,
		  pdown = imageLoad(uPressure, texel + ivec2( 0, -1)).r;

	vec2 pressureGradient = uHalfOneOverDx * vec2(pright - pleft, pup - pdown);
	
	float xold = imageLoad(uVelocityX, texel).r;
	float yold = imageLoad(uVelocityY, texel).r;

	imageStore(uVelocityX, texel, vec4(xold - pressureGradient.x));
	imageStore(uVelocityY, texel, vec4(yold - pressureGradient.y));
	// imageStore(uVelocityZ, texel, vec4(yold - pressureGradient.z));
}
