#version 450

uniform float udx;
uniform float udt;
uniform vec2 uOneOverGridSizeTimesDx;
uniform float uBoundaryCondition;

layout(binding = 0) uniform sampler2D uVelocityX;
layout(binding = 1) uniform sampler2D uVelocityY;
layout(binding = 2) uniform sampler2D uFieldIn;
layout(binding = 3, r32f) uniform restrict writeonly image2D uFieldOut;

vec2 texelSpaceToGridSpace(ivec2 p)
{
	return (vec2(p) + 0.5) * udx;
}

vec2 gridSpaceToUV(vec2 p)
{
	return p * uOneOverGridSizeTimesDx;
}

void compute(ivec2 texel, ivec2 outputTexel, bool boundaryTexel)
{
	vec2 velocity;
	velocity.x = texelFetch(uVelocityX, texel, 0).r;
	velocity.y = texelFetch(uVelocityY, texel, 0).r;

	vec2 lastPosition = texelSpaceToGridSpace(texel) - velocity * udt;
	vec2 uv = gridSpaceToUV(lastPosition);
	float newValue = texture(uFieldIn, uv).r;
	imageStore(uFieldOut, outputTexel, vec4(boundaryTexel ? uBoundaryCondition * newValue : newValue));
}
