#version 450

uniform vec2 uMouseClick;
uniform float uForceMagnitude;
uniform float uOneOverForceRadius;
uniform float uBoundaryCondition;
uniform bvec2 uFieldStagger;

layout(binding = 0, r32f) uniform restrict image2D uField;

void compute(ivec2 texel, ivec2 outputTexel, bool applyBoundaryConditions, bool unused)
{
	vec2 fieldStagger = ivec2(uFieldStagger) * 0.5;

	vec2 vector = vec2(texel) - fieldStagger + 0.5 - uMouseClick;
	float factor = exp2(-dot(vector, vector) * uOneOverForceRadius);

	float value = uForceMagnitude * factor + imageLoad(uField, texel).r;
	
	imageStore(uField, outputTexel, vec4(/*unused
		? 0.
		: applyBoundaryConditions
			? uBoundaryCondition * value
			: */value));
}
