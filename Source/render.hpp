#pragma once

#include <Empty/gl/Buffer.h>
#include <Empty/gl/ShaderProgram.hpp>
#include <Empty/gl/VertexArray.h>
#include <Empty/gl/VertexStructure.h>
#include <Empty/math/vec.h>

#include "Camera.h"
#include "fluid.hpp"

struct FluidSimRenderParameters
{
	FluidSimRenderParameters(Empty::math::vec3 position, Empty::math::uvec3 gridSize, float gridCellSizeInUnits);

	Empty::math::vec3 position;
	Empty::math::uvec3 gridSizeInCells;
	float gridCellSizeInUnits;

	Empty::math::vec3 inkColor;
	float inkMultiplier;

	Empty::gl::Buffer gridVerticesBuf;
	Empty::gl::Buffer gridOutlineIndicesBuf;
	Empty::gl::Buffer gridFacesIndicesBuf;
};

struct FluidSimRenderer
{
	FluidSimRenderer(int frameWidth, int frameHeight);

	void renderFluidSim(FluidState& fluidState, const FluidSimRenderParameters& params, const Camera& camera, int highlightSlice = -1);

private:
	Empty::gl::VertexArray _vao;
	Empty::gl::ShaderProgram _fluidProgram;
	Empty::gl::ShaderProgram _gridProgram;
	Empty::gl::VertexStructure _vs;
};