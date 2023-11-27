#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

uniform bvec3 uFieldStagger;

// Unify computations and boundary condition enforcement
void compute(ivec3 inputTexel, ivec3 outputTexel, bool boundaryTexel);

void main()
{
	ivec3 texel = ivec3(gl_GlobalInvocationID);
	
	uvec3 size = gl_WorkGroupSize * gl_NumWorkGroups;

	// On the inside texels, compute the new value normally.
	// On the boundary, compute and enforce boundary conditions.
	// Staggered fields' bottom, left and back boundary is at
	// x,y,z == 1 rather than 0 to match the physical location of
	// the boundary on centered fields.
	bvec3 bBottomLeftBack = lessThanEqual(texel, ivec3(any(uFieldStagger))),
		bTopRightFront = equal(texel, size - 1);
	bool isBoundaryTexel = any(bBottomLeftBack) || any(bTopRightFront);
	ivec3 boundaryOffset = ivec3(bBottomLeftBack) - ivec3(bTopRightFront);

	compute(texel + boundaryOffset, texel, isBoundaryTexel);
}
