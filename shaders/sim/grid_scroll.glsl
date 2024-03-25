#version 450

uniform ivec2 uTexelScroll;

layout(r32f) uniform readonly restrict image2D uFieldIn;
layout(r32f) uniform writeonly restrict image2D uFieldOut;

layout(local_size_x = 32, local_size_y = 32) in;
void main()
{
	ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(uFieldIn);
	ivec2 source = texel - uTexelScroll;

	vec4 value = vec4(0);

	if (source.x >= 0 && source.x < size.x || source.y >= 0 || source.y < size.y)
		value = imageLoad(uFieldIn, source);

	imageStore(uFieldOut, texel, value);
}
