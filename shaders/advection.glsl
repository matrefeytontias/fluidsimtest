#version 450

uniform float udx;
uniform vec2 uOneOverGridSizeTimesDx;
uniform float udt;

layout(binding = 0) uniform sampler2D uVelocity;
layout(binding = 1) uniform sampler2D uInkDensity;
layout(binding = 2, rg32f) uniform restrict writeonly image2D uVelocityOut;
layout(binding = 3, r32f) uniform restrict writeonly image2D uInkDensityOut;

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
	vec2 velocity = texelFetch(uVelocity, texel, 0).xy;

	// Advect velocity first, then everything else
	vec2 lastPosition = texelSpaceToGridSpace(texel) - velocity * udt;
	vec2 uv = gridSpaceToUV(lastPosition);
	vec4 newVelocity = texture(uVelocity, uv);
	// Velocity respects the no-slip boundary condition
	imageStore(uVelocityOut, outputTexel, boundaryTexel ? -newVelocity : newVelocity);

	// Ink density
	lastPosition = texelSpaceToGridSpace(texel) - newVelocity.xy * udt;
	uv = gridSpaceToUV(lastPosition);
	vec4 newInk = texture(uInkDensity, uv);
	// Ink respects the zero boundary condition
	imageStore(uInkDensityOut, outputTexel, boundaryTexel ? vec4(0.) : newInk);
}
