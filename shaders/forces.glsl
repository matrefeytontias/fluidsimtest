#version 450

uniform vec2 uMouseClick;
uniform float uForceMagnitude;
uniform float uOneOverForceRadius;
uniform float uBoundaryCondition;

layout(binding = 0, r32f) uniform restrict image2D uField;

void compute(ivec2 texel, ivec2 outputTexel, bool boundaryTexel)
{
	vec2 vector = vec2(texel) + 0.5 - uMouseClick;
	float factor = exp2(-dot(vector, vector) * uOneOverForceRadius);

	float newValue = uForceMagnitude * factor + imageLoad(uField, texel).r;
	imageStore(uField, outputTexel, vec4(boundaryTexel ? uBoundaryCondition * newValue : newValue));
}
