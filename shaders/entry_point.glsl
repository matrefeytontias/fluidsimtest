#version 450

layout(local_size_x = 32, local_size_y = 32) in;

uniform bvec2 uFieldStagger;

// Unify computations and boundary condition enforcement
void compute(ivec2 inputTexel, ivec2 outputTexel, bool boundaryTexel);

void main()
{
	ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
	
	uvec2 size = gl_WorkGroupSize.xy * gl_NumWorkGroups.xy;

	// On the inside texels, compute the new value normally.
	// On the boundary, compute and enforce boundary conditions.
	// Staggered fields' bottom and right boundary is at x or
	// y == 1 rather than 0 to match the physical location of
	// the boundary on centered fields.
	bvec2 bBottomLeft = lessThanEqual(texel, ivec2(any(uFieldStagger))), bTopRight = equal(texel, size - 1);
	bool isBoundaryTexel = any(bBottomLeft) || any(bTopRight);
	ivec2 boundaryOffset = ivec2(bBottomLeft) - ivec2(bTopRight);

	compute(texel + boundaryOffset, texel, isBoundaryTexel);
}
