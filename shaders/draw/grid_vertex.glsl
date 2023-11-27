#version 450

uniform mat4 uMVP;

in vec3 aPosition;

void main()
{
	gl_Position = uMVP * vec4(aPosition, 1.);
}
