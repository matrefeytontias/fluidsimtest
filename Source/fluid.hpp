#pragma once

#include <Empty/math/vec.h>

#include "fields.hpp"

// *********************************
// Types related to fluid simulation
// *********************************

struct FluidSimParameters
{
	const Empty::math::uvec2 gridSize;
	float gridCellSize;
	float density;
	float viscosity;
};

struct FluidSimMouseClickImpulse
{
	Empty::math::vec2 position;
	Empty::math::vec2 magnitude;
	float inkAmount;
	float radius;
};

struct FluidRenderParameters
{
	FluidRenderParameters(Empty::math::uvec2 frame, Empty::math::uvec2 gridSize, float gridCellSizeInPx)
	{
		topLeftCorner = (Empty::math::vec2(frame) - Empty::math::vec2(gridSize) * gridCellSizeInPx) / 2.f;
		this->gridCellSizeInPx = gridCellSizeInPx;
	}

	Empty::math::vec2 topLeftCorner;
	float gridCellSizeInPx;
};

struct FluidState
{
	FluidState(Empty::math::uvec2 gridSize, float gridCellSize, float density, float viscosity) :
		parameters{ gridSize, gridCellSize, density, viscosity },
		velocityX{ "Velocity X", gridSize },
		velocityY{ "Velocity Y", gridSize },
		pressure{ "Pressure", gridSize },
		divergenceTex("Divergence"),
		inkDensity{ "Ink density", gridSize }
	{
		divergenceTex.setStorage(1, gridSize.x, gridSize.y);
	}

	void reset()
	{
		velocityX.clear();
		velocityY.clear();
		pressure.clear();
		divergenceTex.template clearLevel<Empty::gl::DataFormat::Red, Empty::gl::DataType::Float>(0);
		inkDensity.clear();
	}
	
	FluidSimParameters parameters;

	// Fields we need
	BufferedScalarField velocityX;
	BufferedScalarField velocityY;
	BufferedScalarField pressure;
	GPUScalarField divergenceTex;

	// Fields we don't need but are cool
	BufferedScalarField inkDensity;
};
