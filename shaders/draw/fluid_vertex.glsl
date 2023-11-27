#version 450

uniform mat4 uMV;
uniform mat4 uP;

in vec3 aPosition;
out vec4 vPosition;

void main()
{
	vPosition = uMV * vec4(aPosition, 1.);
	gl_Position = uP * vPosition;
}
