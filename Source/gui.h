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
		: frame(frameWidth, frameHeight)
		, topLeftCorner((frameWidth - gridWidth * cellSizeInPx) * 0.5f, (frameHeight - gridHeight * cellSizeInPx) * 0.5f)
		, cellSizeInPx(cellSizeInPx)
	{ }

	Empty::math::vec2 frame;
	Empty::math::vec2 topLeftCorner;
	float cellSizeInPx;
	float mouseVectorScale = 0.01f;
};

void doGUI(FluidSim& fluidSim, FluidState& fluidState, SimulationControls& simControls, FluidSimRenderParameters& renderParams, Empty::gl::ShaderProgram& debugDrawProgram, float dt);
void displayTexture(Empty::gl::ShaderProgram& debugDrawProgram, FluidState& fluidState, int whichDebugTexture);
void drawVelocityUnderMouse(Empty::math::vec2 mousePos, FluidState& fluidState, Empty::gl::ShaderProgram& mouseVectorProgram, const FluidSimRenderParameters& renderParams);
