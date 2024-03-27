#version 450

uniform float uAlpha;
uniform float uOneOverBeta;
uniform float uBoundaryCondition;

layout(binding = 0, r32f) uniform readonly image2DArray uFieldSource;
layout(binding = 1, r32f) uniform readonly image2DArray uFieldIn;
layout(binding = 2, r32f) uniform writeonly restrict image2DArray uFieldOut;

// Performs one Jacobi iteration to solve a Poisson equation
// Lx = b
// Where L is a laplacian operator defined by alpha and beta.
// b is the source field.
// For a full derivation, see supplementary material for
// Rabbani, Guertin et al., 2022. Compact Poisson Filters for Fast Fluid Simulation.
// https://dl.acm.org/action/downloadSupplement?doi=10.1145%2F3528233.3530737&file=supplementary.pdf
void compute(ivec3 texel, ivec3 outputTexel, bool boundaryTexel)
{
	// Field is 0 outside of the texture
	float left = imageLoad(uFieldIn, texel + ivec3(-1,  0,  0)).r,
	     right = imageLoad(uFieldIn, texel + ivec3( 1,  0,  0)).r,
		    up = imageLoad(uFieldIn, texel + ivec3( 0,  1,  0)).r,
		  down = imageLoad(uFieldIn, texel + ivec3( 0, -1,  0)).r,
		 front = imageLoad(uFieldIn, texel + ivec3( 0,  0,  1)).r,
		  back = imageLoad(uFieldIn, texel + ivec3( 0,  0, -1)).r,
		source = imageLoad(uFieldSource, texel).r;
	
	float value = (left + right + up + down + front + back + uAlpha * source) * uOneOverBeta;

	imageStore(uFieldOut, outputTexel, vec4(boundaryTexel ? uBoundaryCondition * value : value));
}
