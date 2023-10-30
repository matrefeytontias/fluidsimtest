#version 450

layout(local_size_x = 32, local_size_y = 32) in;

uniform float uHalfOneOverDx;

layout(binding = 0, rg32f) uniform restrict image2D uVelocity;
layout(binding = 1, r32f) uniform restrict readonly image2D uPressure;

void main()
{
	ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
	if(any(texel == 0) || any(texel == imageSize(uVelocity) - 1))
		return;

	float pleft = imageLoad(uPressure, texel + ivec2(-1,  0)).r,
		 pright = imageLoad(uPressure, texel + ivec2( 1,  0)).r,
		    pup = imageLoad(uPressure, texel + ivec2( 0,  1)).r,
		  pdown = imageLoad(uPressure, texel + ivec2( 0, -1)).r;

	vec2 pressureGradient = uHalfOneOverDx * vec2(pright - pleft, pup - pdown);
	
	vec4 old = imageLoad(uVelocity, texel);

	imageStore(uVelocity, texel, old - pressureGradient.xyxx);
}
