#pragma once

#include <Empty/math/vec.h>

#include "fields.hpp"

// *********************************
// Types related to fluid simulation
// *********************************

struct FluidGridParameters
{
	// In texels
	Empty::math::uvec2 size;
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
	Empty::math::vec2 position;
	Empty::math::vec2 magnitude;
	float radius;
	float inkAmount;
};

struct FluidState
{
	FluidState(const FluidGridParameters& grid, const FluidPhysicalProperties& physics)
		: grid{ grid }
		, physics{ physics }
		, exteriorVelocity{ Empty::math::vec2::zero }
		, velocityX{ "Velocity X", grid.size }
		, velocityY{ "Velocity Y", grid.size }
		, pressure{ "Pressure", grid.size }
		, divergenceTex("Divergence")
		, divergenceCheckTex("Divergence zero check")
		, boundariesTex("Boundaries")
		, inkDensity{ "Ink density", grid.size }
	{
		divergenceTex.setStorage(1, grid.size.x, grid.size.y);
		divergenceCheckTex.setStorage(1, grid.size.x, grid.size.y);
		boundariesTex.setStorage(1, grid.size.x, grid.size.y);
	}

	void reset()
	{
		velocityX.clear();
		velocityY.clear();
		pressure.clear();
		divergenceTex.template clearLevel<Empty::gl::DataFormat::Red, Empty::gl::DataType::Float>(0);
		divergenceCheckTex.template clearLevel<Empty::gl::DataFormat::Red, Empty::gl::DataType::Float>(0);
		boundariesTex.template clearLevel<Empty::gl::DataFormat::RedInt, Empty::gl::DataType::Byte>(0);
		inkDensity.clear();
	}
	
	FluidGridParameters grid;
	FluidPhysicalProperties physics;
	Empty::math::vec2 exteriorVelocity;

	// Fields we need
	BufferedScalarField velocityX;
	BufferedScalarField velocityY;
	BufferedScalarField pressure;
	GPUScalarField divergenceTex;
	GPUScalarField divergenceCheckTex;
	Empty::gl::Texture<Empty::gl::TextureTarget::Texture2D, Empty::gl::TextureFormat::Red8ui> boundariesTex;

	// Fields we don't need but are cool
	BufferedScalarField inkDensity;
};
