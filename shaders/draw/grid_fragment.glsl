#version 450

uniform vec3 uLineColor;

out vec4 fFragColor;

void main()
{
	fFragColor = vec4(uLineColor, 1.);
}
