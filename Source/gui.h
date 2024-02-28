#pragma once

#include <Empty/gl/ShaderProgram.hpp>
#include <Empty/math/vec.h>

#include "fluid.hpp"
#include "solver.hpp"

struct SimulationControls
{
	bool capFPS = false;
	bool pauseSimulation = false;
	bool runOneStep = false;

	bool displayDebugTexture = false;
	int whichDebugTexture = 0;
	int whenDebugTexture = 0;
	float colorScale = 1.f;
	float forceScale = 5.f;

	FluidSimMouseClickImpulse impulse;

	FluidSimHookId debugTextureLambdaHookId;
};

struct FluidSimRenderParameters
{
	FluidSimRenderParameters(int frameWidth, int frameHeight, int gridWidth, int gridHeight, float cellSizeInPx)
		: cellSizeInPx(cellSizeInPx)
	{
		topLeftCorner = Empty::math::vec2(frameWidth - gridWidth * cellSizeInPx, frameHeight - gridHeight * cellSizeInPx) * 0.5f;
	}
	Empty::math::vec2 topLeftCorner;
	float cellSizeInPx;
};

void doGUI(FluidSim& fluidSim, FluidState& fluidState, SimulationControls& simControls, Empty::gl::ShaderProgram& debugDrawProgram, float dt);
