#pragma once

#include <Empty/math/vec.h>

#include "fields.hpp"

// *********************************
// Types related to fluid simulation
// *********************************

struct FluidSimParameters
{
	const Empty::math::uvec3 gridSize;
	float gridCellSize;
	float density;
	float viscosity;
};

struct FluidSimImpulse
{
	Empty::math::vec3 position;
	Empty::math::vec3 magnitude;
	float inkAmount;
	float radius;
};

struct FluidRenderParameters
{
	FluidRenderParameters(Empty::math::vec3 center, Empty::math::uvec3 gridSize, float gridCellSizeInUnits)
	{
		centerPosition = center;
		topLeftCorner = center - Empty::math::vec3(gridSize) * gridCellSizeInUnits / 2.f;
		this->gridCellSizeInUnits = gridCellSizeInUnits;
	}

	Empty::math::vec3 centerPosition;
	Empty::math::vec3 topLeftCorner;
	float gridCellSizeInUnits;
};

struct FluidState
{
	FluidState(Empty::math::uvec3 gridSize, float gridCellSize, float density, float viscosity) :
		parameters{ gridSize, gridCellSize, density, viscosity },
		velocityX{ "Velocity X", gridSize },
		velocityY{ "Velocity Y", gridSize },
		velocityZ{ "Velocity Z", gridSize },
		pressure{ "Pressure", gridSize },
		divergenceTex("Divergence"),
		inkDensity{ "Ink density", gridSize }
	{
		divergenceTex.setStorage(1, gridSize.x, gridSize.y, gridSize.z);
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
	
	FluidSimParameters parameters;

	// Fields we need
	BufferedScalarField velocityX;
	BufferedScalarField velocityY;
	BufferedScalarField velocityZ;
	BufferedScalarField pressure;
	GPUScalarField divergenceTex;

	// Fields we don't need but are cool
	BufferedScalarField inkDensity;
};
