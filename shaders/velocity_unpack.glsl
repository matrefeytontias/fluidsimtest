#version 450

layout(local_size_x = 32, local_size_y = 32) in;

layout(binding = 0, rg32f) uniform restrict readonly image2D uVelocityIn;
layout(binding = 1, r32f) uniform restrict writeonly image2D uVelocityXOut;
layout(binding = 2, r32f) uniform restrict writeonly image2D uVelocityYOut;

void main()
{
	ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
	
	vec2 velocity = imageLoad(uVelocityIn, texel).xy;
	imageStore(uVelocityXOut, texel, velocity.xxxx);
	imageStore(uVelocityYOut, texel, velocity.yyyy);
}
