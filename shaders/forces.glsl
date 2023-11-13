#version 450

uniform float udt;
uniform vec2 uMouseClick;
uniform vec2 uForceMagnitude;
uniform float uOneOverForceRadius;
uniform float uInkAmount;

layout(binding = 0, r32f) uniform restrict image2D uVelocityX;
layout(binding = 1, r32f) uniform restrict image2D uVelocityY;
layout(binding = 2, r32f) uniform restrict image2D uInkDensity;

void compute(ivec2 texel, ivec2 outputTexel, bool boundaryTexel)
{
	vec2 vector = vec2(texel) + 0.5 - uMouseClick;
	float factor = exp2(-dot(vector, vector) * uOneOverForceRadius);

	float newx = uForceMagnitude.x * factor + imageLoad(uVelocityX, texel).r;
	float newy = uForceMagnitude.y * factor + imageLoad(uVelocityY, texel).r;
	// Velocity respects the no-slip boundary condition
	imageStore(uVelocityX, outputTexel, vec4(boundaryTexel ? -newx : newx));
	imageStore(uVelocityY, outputTexel, vec4(boundaryTexel ? -newy : newy));
	
	float newInk = udt * uInkAmount * factor + imageLoad(uInkDensity, texel).r;
	// Ink respects the zero boundary condition
	imageStore(uInkDensity, outputTexel, vec4(boundaryTexel ? 0. : newInk));
}
