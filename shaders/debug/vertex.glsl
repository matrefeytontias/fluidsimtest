#version 450

vec2 vertices[] = { vec2(-1, 1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(1, -1), vec2(-1, -1) };

uniform vec2 uTextureSizeOverScreenSize;

out vec2 vUV;

void main()
{
	vUV = (vertices[gl_VertexID] + 1.) * 0.5;
	gl_Position = vec4(vertices[gl_VertexID] * uTextureSizeOverScreenSize, 0., 1.);
}
