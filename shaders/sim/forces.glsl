#version 450

uniform vec3 uForceCenter;
uniform float uOneOverForceRadius;

uniform float uForceMagnitude;
uniform float uBoundaryCondition;
uniform bvec3 uFieldStagger;

layout(binding = 0, r32f) uniform restrict image2DArray uField;

void compute(ivec3 texel, ivec3 outputTexel, bool boundaryTexel)
{
	vec3 fieldStagger = ivec3(uFieldStagger) * 0.5;

	vec3 vector = vec3(texel) - fieldStagger + 0.5 - uForceCenter;
	float factor = exp2(-dot(vector, vector) * uOneOverForceRadius);

	float newValue = uForceMagnitude * factor + imageLoad(uField, texel).r;
	imageStore(uField, outputTexel, vec4(boundaryTexel ? uBoundaryCondition * newValue : newValue));
}
