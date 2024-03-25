#version 450

layout(binding = 0, r32f) uniform readonly restrict image2D uVelocityXIn;
layout(binding = 1, r32f) uniform readonly restrict image2D uVelocityYIn;
layout(binding = 2, r32f) uniform writeonly restrict image2D uVelocityXOut;
layout(binding = 3, r32f) uniform writeonly restrict image2D uVelocityYOut;

// Inward boundary normals
ivec2 boundaryNormals[4] = { ivec2(0, 1), ivec2(0, -1), ivec2(1, 0), ivec2(-1, 0) };

ivec2 texelCoordinateFromThreadID(uvec2 tid, uvec2 size)
{
	// tid.x goes across a boundary, tid.y is 0-3 in order:
	// bottom, top, left, right boundary.
	return ivec2(
		tid.y < 2
			? tid.x
			: tid.y == 2
				? 0
				: size.x - 1,
		tid.y > 1
			? tid.x
			: tid.y == 0
				? 0
				: size.y - 1
	);
}

layout(local_size_x = 32) in;
void main()
{
	uvec2 size = imageSize(uVelocityXIn);
	uvec2 tid = gl_GlobalInvocationID.xy;
	
	ivec2 texel = texelCoordinateFromThreadID(tid, size);

	if (texel.x >= size.x || texel.y >= size.y)
		return;
	
	ivec2 boundaryNormal = boundaryNormals[tid.y];

	float neighbourX = imageLoad(uVelocityXIn, texel + boundaryNormal).r;
	float neighbourY = imageLoad(uVelocityYIn, texel + boundaryNormal).r;

	// No-stick boundary condition: dot(u, n) = 0
	// => neighbour + value = 0 (constraint on the normal component to the boundary)

	// Top and bottom => velocity Y constraint
	if (tid.y < 2)
		imageStore(uVelocityYOut, texel, vec4(-neighbourY));
	// Left and right => velocity X constraint
	else
		imageStore(uVelocityXOut, texel, vec4(-neighbourX));
}
