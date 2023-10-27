#version 450

layout(local_size_x = 32, local_size_y = 32) in;

uniform float udt;
uniform vec2 uMouseClick;
uniform vec2 uForceMagnitude;
uniform float uForceRadius;
uniform float uInkAmount;

layout(binding = 0, r32f) uniform restrict image2D uVelocityX;
layout(binding = 1, r32f) uniform restrict image2D uVelocityY;
// layout(binding = 2, r32f) uniform restrict image2D uVelocityZ;

layout(binding = 3, r32f) uniform restrict image2D uInkDensity;

void main()
{
	ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
	if(any(texel == 0) || any(texel == imageSize(uVelocityX) - 1))
		return;

	vec2 vector = vec2(texel) - uMouseClick;
	float factor = udt * exp2(-dot(vector, vector) / uForceRadius);
	vec2 force = uForceMagnitude * factor;
	
	imageStore(uVelocityX, texel, vec4(imageLoad(uVelocityX, texel).r + force.x));
	imageStore(uVelocityY, texel, vec4(imageLoad(uVelocityY, texel).r + force.y));
	// imageStore(uVelocityZ, texel, vec4(imageLoad(uVelocityZ, texel).r + force.z));
	
	float newInk = imageLoad(uInkDensity, texel).r + uInkAmount * factor;

	imageStore(uInkDensity, texel, vec4(newInk));
}
