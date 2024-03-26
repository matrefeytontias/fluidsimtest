#pragma once

#include <Empty/gl/ShaderProgram.hpp>
#include <Empty/math/vec.h>

#include "fluid.hpp"
#include "render.hpp"
#include "solver.hpp"

struct SimulationControls
{
	bool capFPS = false;
	bool pauseSimulation = false;
	bool runOneStep = false;

	bool displayDebugTexture = false;
	int whichDebugTexture = 0;
	int whenDebugTexture = 0;
	int debugTextureSlice = 0;
	float colorScale = 1.f;
	float forceScale = 5.f;
	int gaussianImpulseAxis = 0;

	Empty::math::ivec3 gridScroll = Empty::math::ivec3::zero;

	Empty::math::vec4 debugRect = Empty::math::vec4(10, 10, 200, 200);

	FluidSimMouseClickImpulse impulse;

	FluidSimHookId debugTextureLambdaHookId;
};

void doGUI(FluidSim& fluidSim, FluidState& fluidState, SimulationControls& simControls, FluidSimRenderParameters& renderParams, Empty::gl::ShaderProgram& debugDrawProgram, float dt);
void displayTexture(Empty::gl::ShaderProgram& debugDrawProgram, FluidState& fluidState, int whichDebugTexture);
