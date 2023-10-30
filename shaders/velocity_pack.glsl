#version 450

layout(local_size_x = 32, local_size_y = 32) in;

layout(binding = 0, rg32f) uniform restrict writeonly image2D uVelocityOut;
layout(binding = 1, r32f) uniform restrict readonly image2D uVelocityXIn;
layout(binding = 2, r32f) uniform restrict readonly image2D uVelocityYIn;

void main()
{
	ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
	
	float velocityX = imageLoad(uVelocityXIn, texel).r,
		  velocityY = imageLoad(uVelocityYIn, texel).r;
	imageStore(uVelocityOut, texel, vec2(velocityX, velocityY).xyxx);
}
