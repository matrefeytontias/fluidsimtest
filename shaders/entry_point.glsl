#version 450

layout(local_size_x = 32, local_size_y = 32) in;

uniform float uBoundaryCondition;

layout(binding = 0, r32f) uniform restrict writeonly image2D uFieldOut;

float compute(ivec2 texel);

void main()
{
	ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
	
	uvec2 size = gl_WorkGroupSize.xy * gl_NumWorkGroups.xy;

	// On the inside texels, compute the new value normally.
	// On the boundary, compute and enforce boundary conditions.
	bvec2 bBottomLeft = equal(texel, ivec2(0)), bTopRight = equal(texel, size - 1);
	bool isBoundaryTexel = any(bBottomLeft) || any(bTopRight);
	ivec2 boundaryOffset = ivec2(bBottomLeft) - ivec2(bTopRight);

	float newValue = compute(isBoundaryTexel ? texel + boundaryOffset : texel);

	imageStore(uFieldOut, texel, vec4(isBoundaryTexel ? newValue * uBoundaryCondition : newValue));
}
