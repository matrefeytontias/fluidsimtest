#version 450

uniform ivec3 uTexelScroll;

layout(r32f) uniform readonly restrict image3D uFieldIn;
layout(r32f) uniform writeonly restrict image3D uFieldOut;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main()
{
	ivec3 texel = ivec3(gl_GlobalInvocationID);
	ivec3 size = imageSize(uFieldIn);
	ivec3 zero = ivec3(0);
	ivec3 source = texel - uTexelScroll;

	vec4 value = vec4(0);

	if (all(greaterThanEqual(texel, zero)) && all(lessThan(texel, size)))
		value = imageLoad(uFieldIn, source);

	imageStore(uFieldOut, texel, value);
}
