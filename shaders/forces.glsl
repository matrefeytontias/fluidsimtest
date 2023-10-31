#version 450

uniform float udt;
uniform vec2 uMouseClick;
uniform vec2 uForceMagnitude;
uniform float uOneOverForceRadius;
uniform float uInkAmount;

layout(binding = 0, rg32f) uniform restrict image2D uVelocity;
layout(binding = 1, r32f) uniform restrict image2D uInkDensity;

void compute(ivec2 texel, ivec2 outputTexel, bool boundaryTexel)
{
	vec2 vector = vec2(texel) - uMouseClick;
	float factor = exp2(-dot(vector, vector) * uOneOverForceRadius);

	vec2 newVelocity = uForceMagnitude * factor + imageLoad(uVelocity, texel).xy;
	// Velocity respects the no-slip boundary condition
	imageStore(uVelocity, outputTexel, boundaryTexel ? -newVelocity.xyxx: newVelocity.xyxx);
	
	float newInk = udt * uInkAmount * factor + imageLoad(uInkDensity, texel).r;
	// Ink respects the zero boundary condition
	imageStore(uInkDensity, outputTexel, vec4(boundaryTexel ? 0. : newInk));
}
