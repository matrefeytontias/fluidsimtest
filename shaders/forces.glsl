#version 450

layout(local_size_x = 32, local_size_y = 32) in;

uniform float udt;
uniform vec2 uMouseClick;
uniform vec2 uForceMagnitude;
uniform float uForceRadius;
uniform float uInkAmount;

layout(binding = 0, rg32f) uniform restrict image2D uVelocity;
layout(binding = 1, r32f) uniform restrict image2D uInkDensity;

void main()
{
	ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
	if(any(texel == 0) || any(texel == imageSize(uVelocity) - 1))
		return;

	vec2 vector = vec2(texel) - uMouseClick;
	float factor = udt * exp2(-dot(vector, vector) / uForceRadius);
	vec2 force = uForceMagnitude * factor;
	
	imageStore(uVelocity, texel, imageLoad(uVelocity, texel) + force.xyxx);
	
	float newInk = imageLoad(uInkDensity, texel).r + uInkAmount * factor;

	imageStore(uInkDensity, texel, vec4(newInk));
}
