#version 450

vec2 vertices[] = { vec2(0., 0.), vec2(1., 0.), vec2(0.9, 0.1), vec2(1., 0.), vec2(0.9, -0.1), vec2(1., 0.) };

uniform sampler2D uVelocityX;
uniform sampler2D uVelocityY;

uniform vec2 uTextureSizeOverScreenSize;
uniform vec2 uBottomLeftCornerUV;
uniform vec2 uMouseUV;
uniform float uVelocityScale;

// Velocity X is staggered by velocityStagger.xy, while
// velocity Y is staggered by velocityStagger.yx
const vec2 velocityStagger = vec2(0.5, 0) / textureSize(uVelocityX, 0);

// ReferenceBase and ReferenceTarget must be unit length
vec3 TransferUnitVectorRotation(vec3 ReferenceBase, vec3 ReferenceTarget, vec3 Vector)
{
	const vec3 HalfVector = normalize(ReferenceBase + ReferenceTarget);
	// This is a quaternion rotation qvq⁻¹ with
	// q = float4(cross(ReferenceBase, HalfVector), dot(ReferenceBase, HalfVector))
	// fully expanded out and simplified.
	const float HB = dot(HalfVector, ReferenceBase);
	const float VB = dot(Vector, ReferenceBase);
	const float VH = dot(HalfVector, Vector);
	return Vector + ((HB * VB * 2 - VH) * HalfVector - VB * ReferenceBase) * 2;
}

void main()
{
	vec2 velocity = vec2(texture(uVelocityX, uMouseUV + velocityStagger.xy).r,
						 texture(uVelocityY, uMouseUV + velocityStagger.yx).r);

	float norm = sqrt(dot(velocity, velocity));
	vec2 vertex = TransferUnitVectorRotation(vec3(1., 0., 0.), vec3(velocity / norm, 0.), vec3(vertices[gl_VertexID], 0.)).xy * log(1 + norm) * uVelocityScale;
	
	gl_Position = vec4((vertex + uMouseUV * uTextureSizeOverScreenSize + uBottomLeftCornerUV) * 2. - 1., 0., 1.);
}
