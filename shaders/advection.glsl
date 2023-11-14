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

void compute(ivec2 texel, ivec2 outputTexel, bool boundaryTexel)
{
	vec2 fieldStagger = ivec2(uFieldStagger) * 0.5;
	vec2 samplePosition = texelSpaceToGridSpace(texel, fieldStagger);

	vec2 velocity;
	velocity.x = texture(uVelocityX, gridSpaceToUV(samplePosition, velocityStagger.xy)).r;
	velocity.y = texture(uVelocityY, gridSpaceToUV(samplePosition, velocityStagger.yx)).r;

	vec2 lastPosition = samplePosition - velocity * udt;
	vec2 uv = gridSpaceToUV(lastPosition, fieldStagger);
	float newValue = texture(uFieldIn, uv).r;
	imageStore(uFieldOut, outputTexel, vec4(boundaryTexel ? uBoundaryCondition * newValue : newValue));
}
