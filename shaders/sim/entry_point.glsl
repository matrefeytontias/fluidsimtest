#version 450

layout(local_size_x = 32, local_size_y = 32) in;

uniform bvec2 uFieldStagger;

// Unify computations and boundary condition enforcement
void compute(ivec2 inputTexel, ivec2 outputTexel, bool applyBoundaryConditions, bool unused);

void main()
{
	ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
	
	uvec2 size = gl_WorkGroupSize.xy * gl_NumWorkGroups.xy;

	// On the inside texels, compute the new value normally.
	// On the boundary, compute and enforce boundary conditions.
	//
	// Staggered fields' bottom and left boundary are at x or
	// y == 1 rather than 0 to match the physical location of
	// the boundary on centered fields; the 0 row and column are
	// unused. Texels also don't need an offset since they always
	// compute their own value.
	
	/*
	bvec2 bBottomLeft = equal(texel, ivec2(uFieldStagger)), bTopRight = equal(texel, size - 1);
	bvec2 isBoundaryTexel = bBottomLeft || bTopRight;
	bool applyBoundaryConditions = false; // !any(uFieldStagger) && any(isBoundaryTexel) || any(uFieldStagger && isBoundaryTexel);
	ivec2 boundaryOffset = ivec2(bBottomLeft) - ivec2(bTopRight);
	*/
	bool unused = any(lessThan(texel, ivec2(uFieldStagger)));

	compute(texel, texel, false, unused);
}
