#include "render.hpp"

#include <Empty/math/funcs.h>

#include "Context.h"

using namespace Empty::gl;
using namespace Empty::math;

// ####################################################

FluidSimRenderParameters::FluidSimRenderParameters(vec3 position, uvec3 gridSize, float gridCellSizeInUnits)
	: position(position)
	, gridSizeInCells(gridSize)
	, gridCellSizeInUnits(gridCellSizeInUnits)
	, inkColor{0.f, 1.f, 0.f}
	, inkMultiplier(1.f)
	, gridVerticesBuf("Fluid volume geometry buffer")
	, gridOutlineIndicesBuf("Fluid grid outline indices buffer")
	, gridFacesIndicesBuf("Fluid volume faces indices buffer")
{
	const vec3 gridVertices[] = {
		{ -1, -1,  1 },
		{  1, -1,  1 },
		{  1, -1, -1 },
		{ -1, -1, -1 },
		{ -1,  1,  1 },
		{  1,  1,  1 },
		{  1,  1, -1 },
		{ -1,  1, -1 },
	};
	const uvec2 gridOutline[] = {
		{ 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
		{ 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
		{ 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 },
	};
	const uvec3 gridFaces[] = {
		{ 0, 2, 1 }, { 0, 3, 2 },
		{ 4, 5, 6 }, { 4, 6, 7 },
		{ 0, 4, 7 }, { 0, 7, 3 },
		{ 1, 2, 6 }, { 1, 6, 5 },
		{ 0, 1, 5 }, { 0, 5, 4 },
		{ 2, 3, 7 }, { 2, 7, 6 },
	};

	// Upload
	gridVerticesBuf.setStorage(sizeof(gridVertices), BufferUsage::StaticDraw, gridVertices);
	gridOutlineIndicesBuf.setStorage(sizeof(gridOutline), BufferUsage::StaticDraw, gridOutline);
	gridFacesIndicesBuf.setStorage(sizeof(gridFaces), BufferUsage::StaticDraw, gridFaces);
}

// ####################################################

FluidSimRenderer::FluidSimRenderer(int frameWidth, int frameHeight)
	: _vao("Fluid sim render VAO")
	, _fluidProgram("Fluid render program")
	, _gridProgram("Grid render program")
	, _vs()
{
	_fluidProgram.attachFile(ShaderType::Vertex, "shaders/draw/fluid_vertex.glsl", "Fluid render vertex shader");
	_fluidProgram.attachFile(ShaderType::Fragment, "shaders/draw/fluid_fragment.glsl", "Fluid render vertex shader");
	_fluidProgram.build();

	_gridProgram.attachFile(ShaderType::Vertex, "shaders/draw/grid_vertex.glsl", "Sim grid render vertex shader");
	_gridProgram.attachFile(ShaderType::Fragment, "shaders/draw/grid_fragment.glsl", "Sim grid render fragment shader");
	_gridProgram.build();

	_vs.add("aPosition", VertexAttribType::Float, 3);
	_gridProgram.locateAttributes(_vs);
}

void FluidSimRenderer::renderFluidSim(FluidState& fluidState, const FluidSimRenderParameters& params, const Camera& camera, int highlightSlice)
{
	Context& context = Context::get();

	mat4 m = scale(vec3(params.gridSizeInCells) * params.gridCellSizeInUnits / 2.f);
	m.column(3).xyz() = params.position;
	mat4 v = inverse(camera.m);
	mat4 mv = v * m;
	mat4 vp = camera.p * v;
	mat4 mvp = camera.p * mv;

	_vao.attachVertexBuffer(params.gridVerticesBuf, _vs);
	context.bind(_vao);

	// Draw grid
	{
		_gridProgram.uniform("uMVP", mvp);
		_gridProgram.uniform("uLineColor", vec3::one);
	
		_vao.attachElementBuffer(params.gridOutlineIndicesBuf);

		context.setShaderProgram(_gridProgram);
		context.drawElements(PrimitiveType::Lines, ElementType::Int, 0, 24);
	}

	// Draw fluid sim
	{
		context.enable(ContextCapability::CullFace);
		context.faceCullingMode(FaceCullingMode::Front);
	
		_fluidProgram.uniform("uMV", mv);
		_fluidProgram.uniform("uP", camera.p);
		_fluidProgram.uniform("uCameraToFluidSim", inverse(m) * camera.m);
		_fluidProgram.registerTexture("uInkDensity", fluidState.inkDensity.getInput());
		_fluidProgram.uniform("uInkColor", params.inkColor);
		_fluidProgram.uniform("uInkMultiplier", params.inkMultiplier);

		_vao.attachElementBuffer(params.gridFacesIndicesBuf);

		context.setShaderProgram(_fluidProgram);
		context.drawElements(PrimitiveType::Triangles, ElementType::Int, 0, 36);
	}

	// Draw highlighted slice
	if (highlightSlice >= 0 && highlightSlice < params.gridSizeInCells.z)
	{
		vec3 s{ params.gridSizeInCells };
		s.z = 1.f;
		mat4 m = scale(s * params.gridCellSizeInUnits / 2.f);
		m.column(3).xyz() = params.position + vec3(0, 0, (highlightSlice - params.gridSizeInCells.z / 2.f + 0.5f) * params.gridCellSizeInUnits);
		mat4 mvp = vp * m;

		_gridProgram.uniform("uMVP", mvp);
		_gridProgram.uniform("uLineColor", vec3(0.25f, 0.25f, 1.f));

		_vao.attachElementBuffer(params.gridOutlineIndicesBuf);

		context.setShaderProgram(_gridProgram);
		context.drawElements(PrimitiveType::Lines, ElementType::Int, 0, 24);
	}
}
