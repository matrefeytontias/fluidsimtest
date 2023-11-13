#version 450

uniform float udx;
uniform vec2 uOneOverGridSizeTimesDx;
uniform float udt;

layout(binding = 0) uniform sampler2D uVelocityX;
layout(binding = 1) uniform sampler2D uVelocityY;
layout(binding = 2) uniform sampler2D uInkDensity;
layout(binding = 3, r32f) uniform restrict writeonly image2D uVelocityXOut;
layout(binding = 4, r32f) uniform restrict writeonly image2D uVelocityYOut;
layout(binding = 5, r32f) uniform restrict writeonly image2D uInkDensityOut;

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

	// Advect velocity first, then everything else
	vec2 lastPosition = texelSpaceToGridSpace(texel) - velocity * udt;
	vec2 uv = gridSpaceToUV(lastPosition);
	float newx = texture(uVelocityX, uv).r;
	float newy = texture(uVelocityY, uv).r;
	// Velocity respects the no-slip boundary condition
	imageStore(uVelocityXOut, outputTexel, vec4(boundaryTexel ? -newx : newx));
	imageStore(uVelocityYOut, outputTexel, vec4(boundaryTexel ? -newy : newy));


	// Ink density
	lastPosition = texelSpaceToGridSpace(texel) - vec2(newx, newy) * udt;
	uv = gridSpaceToUV(lastPosition);
	vec4 newInk = texture(uInkDensity, uv);
	// Ink respects the zero boundary condition
	imageStore(uInkDensityOut, outputTexel, boundaryTexel ? vec4(0.) : newInk);
}
