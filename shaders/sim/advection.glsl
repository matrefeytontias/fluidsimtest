#version 450

struct GridParams
{
	float dx;
	float oneOverDx;
	vec3 oneOverGridSize;
};
uniform GridParams uGridParams;

uniform float udt;
uniform float uBoundaryCondition;
uniform bvec3 uFieldStagger;

layout(binding = 0) uniform sampler2DArray uVelocityX;
layout(binding = 1) uniform sampler2DArray uVelocityY;
layout(binding = 2) uniform sampler2DArray uVelocityZ;
layout(binding = 3) uniform sampler2DArray uFieldIn;
layout(binding = 4) uniform restrict writeonly image2DArray uFieldOut;

// Velocity X is staggered by velocityStagger.xyy
// Velocity Y is staggered by velocityStagger.yxy
// Velocity Z is staggered by velocityStagger.yyx
// TEST: collocated grid
const vec2 velocityStagger = vec2(0, 0);

vec3 texelSpaceToGridSpace(ivec3 p, vec3 stagger)
{
	return (vec3(p) - stagger + 0.5) * uGridParams.dx;
}

vec3 gridSpaceToUV(vec3 p, vec3 stagger)
{
	return (p * uGridParams.oneOverDx + stagger) * uGridParams.oneOverGridSize;
}

float sampleTex(sampler2DArray tex, vec3 uv)
{
	ivec3 size = textureSize(tex, 0);
	uv.z = uv.z * size.z - 0.5;

	float down = texture(tex, uv + vec3(0, 0, -0.5)).r;
	float up = texture(tex, uv + vec3(0, 0, 0.5)).r;

	return mix(uv.z < 0. ? 0 : down, uv.z >= size.z - 1. ? 0 : up, fract(uv.z));
}

vec3 bilerpVelocity(vec3 position)
{
	vec3 velocity;
	velocity.x = sampleTex(uVelocityX, gridSpaceToUV(position, velocityStagger.xyy));
	velocity.y = sampleTex(uVelocityY, gridSpaceToUV(position, velocityStagger.yxy));
	velocity.z = sampleTex(uVelocityZ, gridSpaceToUV(position, velocityStagger.yyx));
	return velocity;
}

// Semi-lagrangian advection via 3rd-order Runge-Kutta time integration
// Fluid Simulation for Computer Graphics, Second Edition, Robert Bridson
// Appendix A.2.2 Time Integration
vec3 traceBack(vec3 position)
{
	vec3 k1 = bilerpVelocity(position);
	vec3 k2 = bilerpVelocity(position - udt * 0.5 * k1);
	vec3 k3 = bilerpVelocity(position - udt * 0.75 * k2);

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

// Monotonic tricubic interpolation
float interpolateField(vec3 uv)
{
	ivec3 size = textureSize(uFieldIn, 0);

	// Gather exactly in the corner of the texel so the correct texels are always fetched.
	// Not doing this introduces irregularities on texel boundaries.
	vec3 realTexelSample = uv * size - 0.5;
	vec3 cornerTexelSample = floor(realTexelSample);
	vec3 gatherUV = (cornerTexelSample + 1.) * vec3(uGridParams.oneOverGridSize.xy, 1);

	// Interpolation coefficients
	vec3 t = realTexelSample - cornerTexelSample;

	// Interpolate along X then Y then Z
	vec4 zValues = vec4(0);
	gatherUV.z -= 2.;
	for (int i = 0; i < 4; i++, gatherUV.z += 1.)
	{
		if (gatherUV.z < 0 || gatherUV.z >= size.z)
			continue;

		vec4 topLeftBlock = textureGatherOffset(uFieldIn, gatherUV, ivec2(-1, 1));
		vec4 topRightBlock = textureGatherOffset(uFieldIn, gatherUV, ivec2(1, 1));
		vec4 bottomLeftBlock = textureGatherOffset(uFieldIn, gatherUV, ivec2(-1, -1));
		vec4 bottomRightBlock = textureGatherOffset(uFieldIn, gatherUV, ivec2(1, -1));
	
		// Y goes up
		float q0 = monotonicCubicInterpolation(bottomLeftBlock.w, bottomLeftBlock.z, bottomRightBlock.w, bottomRightBlock.z, t.x);
		float q1 = monotonicCubicInterpolation(bottomLeftBlock.x, bottomLeftBlock.y, bottomRightBlock.x, bottomRightBlock.y, t.x);
		float q2 = monotonicCubicInterpolation(topLeftBlock.w, topLeftBlock.z, topRightBlock.w, topRightBlock.z, t.x);
		float q3 = monotonicCubicInterpolation(topLeftBlock.x, topLeftBlock.y, topRightBlock.x, topRightBlock.y, t.x);

		zValues[i] = monotonicCubicInterpolation(q0, q1, q2, q3, t.y);
	}

	return monotonicCubicInterpolation(zValues[0], zValues[1], zValues[2], zValues[3], t.z);
}

void compute(ivec3 texel, ivec3 outputTexel, bool boundaryTexel, bool unused)
{
	vec3 fieldStagger = ivec3(uFieldStagger) * 0.5;
	vec3 samplePosition = texelSpaceToGridSpace(texel, fieldStagger);
	float newValue = interpolateField(gridSpaceToUV(traceBack(samplePosition), fieldStagger));

	// TEST: collocated grid
	imageStore(uFieldOut, outputTexel, vec4(/*unused ? 0 : boundaryTexel ? uBoundaryCondition * newValue :*/ newValue));
}
