#version 450

vec2 vertices[] = { vec2(0, 1), vec2(1, 1), vec2(0, 0), vec2(1, 1), vec2(1, 0), vec2(0, 0) };

uniform vec4 uRect;
uniform vec2 uOneOverScreenSize;
uniform float uUVZ;

out vec3 vUV;

void main()
{
	vUV = vec3(vertices[gl_VertexID], uUVZ);
	vec2 topLeftCorner = (uRect.xy + vec2(0., uRect.w)) * uOneOverScreenSize;
	topLeftCorner.y = 1. - topLeftCorner.y;
	vec2 pos = vertices[gl_VertexID] * uRect.zw  * uOneOverScreenSize + topLeftCorner;
	pos = pos * 2. - 1.;
	gl_Position = vec4(pos, 0., 1.);
}
