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
		, velocityX{ "Velocity X", grid.size }
		, velocityY{ "Velocity Y", grid.size }
		, pressure{ "Pressure", grid.size }
		, divergenceTex("Divergence")
		, inkDensity{ "Ink density", grid.size }
	{
		divergenceTex.setStorage(1, grid.size.x, grid.size.y);
	}

	void reset()
	{
		velocityX.clear();
		velocityY.clear();
		pressure.clear();
		divergenceTex.template clearLevel<Empty::gl::DataFormat::Red, Empty::gl::DataType::Float>(0);
		inkDensity.clear();
	}
	
	FluidGridParameters grid;
	FluidPhysicalProperties physics;

	// Fields we need
	BufferedScalarField velocityX;
	BufferedScalarField velocityY;
	BufferedScalarField pressure;
	GPUScalarField divergenceTex;

	// Fields we don't need but are cool
	BufferedScalarField inkDensity;
};
