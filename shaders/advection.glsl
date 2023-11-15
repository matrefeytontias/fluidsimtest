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

layout(binding = 0) uniform sampler3D uVelocityX;
layout(binding = 1) uniform sampler3D uVelocityY;
layout(binding = 2) uniform sampler3D uVelocityZ;
layout(binding = 3) uniform sampler3D uFieldIn;
layout(binding = 4, r32f) uniform restrict writeonly image3D uFieldOut;

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

void compute(ivec3 texel, ivec3 outputTexel, bool boundaryTexel)
{
	vec3 fieldStagger = ivec3(uFieldStagger) * 0.5;
	vec3 samplePosition = texelSpaceToGridSpace(texel, fieldStagger);

	vec3 velocity;
	velocity.x = texture(uVelocityX, gridSpaceToUV(samplePosition, velocityStagger.xyy)).r;
	velocity.y = texture(uVelocityY, gridSpaceToUV(samplePosition, velocityStagger.yxy)).r;
	velocity.z = texture(uVelocityZ, gridSpaceToUV(samplePosition, velocityStagger.yyx)).r;

	vec3 lastPosition = samplePosition - velocity * udt;
	vec3 uv = gridSpaceToUV(lastPosition, fieldStagger);
	float newValue = texture(uFieldIn, uv).r;
	imageStore(uFieldOut, outputTexel, vec4(boundaryTexel ? uBoundaryCondition * newValue : newValue));
}
