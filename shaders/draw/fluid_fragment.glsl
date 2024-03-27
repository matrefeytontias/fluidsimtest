#version 450

#define RAY_SAMPLES 128

uniform mat4 uCameraToFluidSim;
uniform sampler2DArray uInkDensity;
uniform vec3 uInkColor;
uniform float uInkMultiplier;

in vec4 vPosition;
out vec4 fFragColor;

float sampleFluid(vec3 p)
{
    vec3 uv = p * 0.5 + 0.5;
    
    vec3 size = textureSize(uInkDensity, 0);
	uv.z *= size.z;

	float down = texture(uInkDensity, uv + vec3(0, 0, -0.5)).r;
	float up = texture(uInkDensity, uv + vec3(0, 0, 0.5)).r;

	return mix(down, up, fract(uv.z));
}

// All calculations take place in fluid sim space, ie the
// centered unit cube [-1; 1]³. It then becomes easy to intersect
// with it.

// Adapted from https://iquilezles.org/articles/intersectors/
// Intersection with centered unit cube
float boxIntersection( in vec3 rayOrigin, in vec3 rayDirection) 
{
    vec3 m = 1.0/rayDirection;
    vec3 n = m*rayOrigin;
    vec3 k = abs(m)*2.; // boxSize = vec3(2.) for the centered unit cube
    vec3 t1 = -n - k;
    // vec3 t2 = -n + k;
    float tN = max( max( t1.x, t1.y ), t1.z );
    // float tF = min( min( t2.x, t2.y ), t2.z ); // tF comes for free in the form of vPosition
    // if( tN>tF || tF<0.0) return -1.0; // in our case, there is always an intersection
    return tN;
}

void main()
{
	vec3 rayOrigin = uCameraToFluidSim[3].xyz;
	vec3 rayEnd = (uCameraToFluidSim * vPosition).xyz;
    vec3 rayDirection = rayEnd - rayOrigin;
    float t = boxIntersection(rayOrigin, rayDirection);
    // t is negative if the entry point of the ray into the cube is behind the camera.
    // In this case, just use the camera position.
    vec3 intersectionStart = max(0., t) * rayDirection + rayOrigin;

    float density = 0.;
    vec3 rayStep = (rayEnd - intersectionStart) / RAY_SAMPLES;
    float weight = length(rayStep);
    float weightsSum = distance(rayEnd, intersectionStart);
    vec3 rayPosition = intersectionStart;

    for (int i = 0; i < RAY_SAMPLES; i++)
    {
        density += sampleFluid(rayPosition) * weight;
        rayPosition += rayStep;
    }

    fFragColor = vec4(uInkColor, uInkMultiplier * density / weightsSum);
}
