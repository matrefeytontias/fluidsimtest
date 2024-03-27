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
layout(binding = 4, r32f) uniform restrict writeonly image2DArray uFieldOut;

// Velocity X is staggered by velocityStagger.xyy
// Velocity Y is staggered by velocityStagger.yxy
// Velocity Z is staggered by velocityStagger.yyx
const vec2 velocityStagger = vec2(0.5, 0);

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

	return mix(down, up, fract(uv.z));
}

vec3 bilerpVelocity(vec3 position)
{
	vec3 velocity;
	velocity.x = sampleTex(uVelocityX, gridSpaceToUV(position, velocityStagger.xyy));
	velocity.y = sampleTex(uVelocityY, gridSpaceToUV(position, velocityStagger.yxy));
	velocity.z = sampleTex(uVelocityZ, gridSpaceToUV(position, velocityStagger.yyx));
	return velocity;
}

vec3 traceBack(vec3 position)
{
	return position - bilerpVelocity(position) * udt;
}

float interpolateField(vec3 uv)
{
	return sampleTex(uFieldIn, uv);
}

void compute(ivec3 texel, ivec3 outputTexel, bool boundaryTexel)
{
	vec3 fieldStagger = ivec3(uFieldStagger) * 0.5;
	vec3 samplePosition = texelSpaceToGridSpace(texel, fieldStagger);
	float newValue = interpolateField(gridSpaceToUV(traceBack(samplePosition), fieldStagger));

	imageStore(uFieldOut, outputTexel, vec4(boundaryTexel ? uBoundaryCondition * newValue : newValue));
}
