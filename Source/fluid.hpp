#pragma once

#include <Empty/math/vec.h>

#include "fields.hpp"

// *********************************
// Types related to fluid simulation
// *********************************

struct FluidGridParameters
{
	// In texels
	Empty::math::uvec3 size;
	// In meters
	float cellSize;
};

struct FluidPhysicalProperties
{
	// In kg/dm²
	float density;
	// In m²/s
	float kinematicViscosity;
};

struct FluidSimMouseClickImpulse
{
	Empty::math::vec3 position = Empty::math::vec3::zero;
	Empty::math::vec3 magnitude = Empty::math::vec3::zero;
	float inkAmount = 20.f;
	float radius = 40.f;
};

struct FluidState
{
	FluidState(const FluidGridParameters& grid, const FluidPhysicalProperties& physics)
		: grid{ grid }
		, physics{ physics }
		, exteriorVelocity{ Empty::math::vec2::zero }
		, velocityX{ "Velocity X", grid.size }
		, velocityY{ "Velocity Y", grid.size }
		, velocityZ{ "Velocity Z", grid.size }
		, pressure{ "Pressure", grid.size }
		, divergenceTex("Divergence")
		, divergenceCheckTex("Divergence zero check")
		, boundariesTex("Boundaries")
		, inkDensity{ "Ink density", grid.size }
	{
		divergenceTex.setStorage(1, grid.size.x, grid.size.y, grid.size.z);
		divergenceCheckTex.setStorage(1, grid.size.x, grid.size.y, grid.size.z);
	}

	void reset()
	{
		velocityX.clear();
		velocityY.clear();
		velocityZ.clear();
		pressure.clear();
		divergenceTex.template clearLevel<Empty::gl::DataFormat::Red, Empty::gl::DataType::Float>(0);
		inkDensity.clear();
	}
	
	FluidGridParameters grid;
	FluidPhysicalProperties physics;
	Empty::math::vec2 exteriorVelocity;

	// Fields we need
	BufferedScalarField velocityX;
	BufferedScalarField velocityY;
	BufferedScalarField velocityZ;
	BufferedScalarField pressure;
	GPUScalarField divergenceTex;
	GPUScalarField divergenceCheckTex;
	Empty::gl::Texture<Empty::gl::TextureTarget::Texture2D, Empty::gl::TextureFormat::Red8ui> boundariesTex;

	// Fields we don't need but are cool
	BufferedScalarField inkDensity;
};
