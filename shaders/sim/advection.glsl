#version 450

struct GridParams
{
	float dx;
	float oneOverDx;
	vec2 oneOverGridSize;
};
uniform GridParams uGridParams;

uniform float udt;
uniform float uBoundaryCondition;
uniform bvec2 uFieldStagger;

layout(binding = 0) uniform sampler2D uVelocityX;
layout(binding = 1) uniform sampler2D uVelocityY;
layout(binding = 2) uniform sampler2D uFieldIn;
layout(binding = 3, r32f) uniform restrict writeonly image2D uFieldOut;

// Velocity X is staggered by velocityStagger.xy, while
// velocity Y is staggered by velocityStagger.yx
const vec2 velocityStagger = vec2(0.5, 0);

vec2 texelSpaceToGridSpace(ivec2 p, vec2 stagger)
{
	return (vec2(p) - stagger + 0.5) * uGridParams.dx;
}

vec2 gridSpaceToUV(vec2 p, vec2 stagger)
{
	return (p * uGridParams.oneOverDx + stagger) * uGridParams.oneOverGridSize;
}

vec2 bilerpVelocity(vec2 position)
{
	vec2 velocity;
	velocity.x = texture(uVelocityX, gridSpaceToUV(position, velocityStagger.xy)).r;
	velocity.y = texture(uVelocityY, gridSpaceToUV(position, velocityStagger.yx)).r;
	return velocity;
}

// Semi-lagrangian advection via 3rd-order Runge-Kutta time integration
// Fluid Simulation for Computer Graphics, Second Edition, Robert Bridson
// Appendix A.2.2 Time Integration
vec2 traceBack(vec2 position)
{
	vec2 k1 = bilerpVelocity(position);
	vec2 k2 = bilerpVelocity(position - udt * 0.5 * k1);
	vec2 k3 = bilerpVelocity(position - udt * 0.75 * k2);

	return position - (k1 * 2 + k2 * 3 + k3 * 4) * udt / 9.;
}

// Visual Simulation of Smoke, Ronald Fedkiw, Jos Stam and Henrik Wann Jensen: Proceedings of SIGGRAPH'2001
// Appendix B Monotonic Cubic Interpolation
// and
// https://jbrd.github.io/2020/12/27/monotone-cubic-interpolation.html
// A More Efficient Monotone-Cubic Sampler
float monotonicCubicInterpolation(float qprev, float q0, float q1, float qnext, float t)
{
	float delta = q1 - q0;
	float d0 = (q1 - qprev) * 0.5;
	float d1 = (qnext - q0) * 0.5;

	// Enforce monotonicity + clamp gradients
	d0 = sign(delta) != sign(d0) ? 0. : d0 / delta > 3 ? delta * 3 : d0;
	d1 = sign(delta) != sign(d1) ? 0. : d1 / delta > 3 ? delta * 3 : d1;

	float a0 = q0;
	float a1 = d0;
	float a2 = delta * 3 - d0 * 2 - d1;
	// WARNING : the Stam paper forgets the 2 here
	float a3 = d0 + d1 - delta * 2;

	return ((a3 * t + a2) * t + a1) * t + a0;
}

// Monotonic bicubic interpolation
float interpolateField(vec2 uv)
{
	vec2 size = textureSize(uFieldIn, 0).xy;

	// Gather exactly in the corner of the texel so the correct texels are always fetched.
	// Not doing this introduces irregularities on texel boundaries.
	vec2 realTexelSample = uv * size - 0.5;
	vec2 cornerTexelSample = floor(realTexelSample);
	vec2 gatherUV = (cornerTexelSample + 1.) * uGridParams.oneOverGridSize;

	// Interpolate along X then Y
	vec4 topLeftBlock = textureGatherOffset(uFieldIn, gatherUV, ivec2(-1, 1));
	vec4 topRightBlock = textureGatherOffset(uFieldIn, gatherUV, ivec2(1, 1));
	vec4 bottomLeftBlock = textureGatherOffset(uFieldIn, gatherUV, ivec2(-1, -1));
	vec4 bottomRightBlock = textureGatherOffset(uFieldIn, gatherUV, ivec2(1, -1));

	vec2 t = realTexelSample - cornerTexelSample;
	
	// Y goes up
	float q0 = monotonicCubicInterpolation(bottomLeftBlock.w, bottomLeftBlock.z, bottomRightBlock.w, bottomRightBlock.z, t.x);
	float q1 = monotonicCubicInterpolation(bottomLeftBlock.x, bottomLeftBlock.y, bottomRightBlock.x, bottomRightBlock.y, t.x);
	float q2 = monotonicCubicInterpolation(topLeftBlock.w, topLeftBlock.z, topRightBlock.w, topRightBlock.z, t.x);
	float q3 = monotonicCubicInterpolation(topLeftBlock.x, topLeftBlock.y, topRightBlock.x, topRightBlock.y, t.x);

	return monotonicCubicInterpolation(q0, q1, q2, q3, t.y);
}

void compute(ivec2 texel, ivec2 outputTexel, bool boundaryTexel)
{
	vec2 fieldStagger = ivec2(uFieldStagger) * 0.5;
	vec2 samplePosition = texelSpaceToGridSpace(texel, fieldStagger);
	float newValue = interpolateField(gridSpaceToUV(traceBack(samplePosition), fieldStagger));

	imageStore(uFieldOut, outputTexel, vec4(boundaryTexel ? uBoundaryCondition * newValue : newValue));
}
