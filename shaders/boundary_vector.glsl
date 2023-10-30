#version 450

layout(local_size_x = 32) in;

uniform float uBoundaryCondition;

layout(binding = 3, rg32f) uniform restrict image2D uField;

void main()
{
	ivec2 id = ivec2(gl_GlobalInvocationID.xy);
	
	ivec2 size = imageSize(uField);

	// id.y encodes direction of the boundary
	// 0 -> bottom
	// 1 -> top
	// 2 -> left
	// 3 -> right

	ivec2 texel = id.y < 2 ? ivec2(id.x, id.y * (size.y - 1)) : ivec2((id.y - 2) * (size.x - 1), id.x);
	if (any(greaterThan(texel, size)))
		return;
	ivec2 offset = id.y < 2 ? ivec2(0, -id.y * 2 + 1) : ivec2(-id.y * 2 + 5, 0);

	vec4 boundaryValue = imageLoad(uField, texel + offset);

	imageStore(uField, texel, uBoundaryCondition * boundaryValue);
}
