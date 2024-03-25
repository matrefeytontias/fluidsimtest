#version 450

uniform float uAlpha;
uniform float uOneOverBeta;
uniform float uBoundaryCondition;
uniform bvec2 uFieldStagger;

layout(r32f) uniform readonly image2D uFieldSource;
layout(r32f) uniform readonly image2D uFieldIn;
layout(r32f) uniform writeonly restrict image2D uFieldOut;

// Performs one Jacobi iteration to solve a Poisson equation
// Lx = b
// Where L is a laplacian operator defined by alpha and beta.
// b is the source field.
// For a full derivation, see supplementary material for
// Rabbani, Guertin et al., 2022. Compact Poisson Filters for Fast Fluid Simulation.
// https://dl.acm.org/action/downloadSupplement?doi=10.1145%2F3528233.3530737&file=supplementary.pdf
void compute(ivec2 texel, ivec2 outputTexel, bool applyBoundaryConditions, bool unused)
{
	// Field is 0 outside of the texture
	float left = imageLoad(uFieldIn, texel + ivec2(-1,  0)).r,
	     right = imageLoad(uFieldIn, texel + ivec2( 1,  0)).r,
		    up = imageLoad(uFieldIn, texel + ivec2( 0,  1)).r,
		  down = imageLoad(uFieldIn, texel + ivec2( 0, -1)).r,
		source = imageLoad(uFieldSource, texel).r;
	
	float value = (left + right + up + down + uAlpha * source) * uOneOverBeta;

	imageStore(uFieldOut, outputTexel, vec4(unused
		? 0.
		: /*applyBoundaryConditions
			? uBoundaryCondition * value
			: */value));
}
