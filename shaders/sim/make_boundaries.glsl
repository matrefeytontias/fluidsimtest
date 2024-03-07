#version 450

layout(local_size_x = 32) in;

const uint BOUNDARY_EMPTY = 0;
const uint BOUNDARY_WALL = 1;
const uint BOUNDARY_INLET = 2;
const uint BOUNDARY_OUTLET = 3;

vec2 boundaryNormals[4] = { vec2(0, -1), vec2(0, 1), vec2(1, 0), vec2(-1, 0) };

layout(binding = 0, r8ui) restrict writeonly uniform uimage2D uBoundariesTex;

uniform vec2 uExteriorVelocity;

ivec2 texelCoordinateFromThreadID(uvec2 tid, uvec2 size)
{
	// tid.x goes across a boundary, tid.y is 0-3 in order
	// top, bottom, left, right boundary.
	return ivec2(
		tid.y < 2
			? tid.x
			: tid.y == 2
				? 0
				: size.x - 1,
		tid.y > 1
			? tid.x
			: tid.y == 0
				? size.y - 1
				: 0
	);
}

void main()
{
	uvec2 size = imageSize(uBoundariesTex).xy;
	uvec2 tid = gl_GlobalInvocationID.xy;
	
	ivec2 texel = texelCoordinateFromThreadID(tid, size);

	if (texel.x >= size.x || texel.y >= size.y)
		return;
	
	vec2 boundaryNormal = boundaryNormals[tid.y];

	// Mark boundaries in the direction of the exterior velocity as inlet, the others
	// as outlet.
	// There is a race condition at corners but those never get sampled.
	imageStore(uBoundariesTex, texel, uvec4(dot(uExteriorVelocity, boundaryNormal) > 0 ? BOUNDARY_INLET : BOUNDARY_OUTLET));
}
