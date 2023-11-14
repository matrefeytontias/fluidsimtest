#include "solver.hpp"

#include <algorithm>

#include <Empty/gl/Buffer.h>
#include <Empty/gl/ShaderProgram.hpp>
#include <Empty/utils/macros.h>

#include "Context.h"
#include "fluid.hpp"

using namespace Empty::gl;

// ********************************************
// Shared constants related to fluid simulation
// ********************************************

constexpr int allVelocityXBinding = 0;
constexpr int allVelocityYBinding = 1;

constexpr int advectionFieldInBinding = 2;
constexpr int advectionFieldOutBinding = 3;

constexpr int forcesFieldBinding = 0;

constexpr int divergenceOutBinding = 2;

constexpr int projectionPressureBinding = 2;

// f(boundary) + f(neighbour) = 0 -> f(boundary) = -f(neighbour)
constexpr float noSlipBoundaryCondition = -1.f;
// f(boundary) - f(neighbour) = 0 -> f(boundary) = f(neighbour)
constexpr float neumannBoundaryCondition = 1.f;
// f(boundary) = 0
constexpr float zeroBoundaryCondition = 0.f;

constexpr int entryPointWorkGroupX = 32;
constexpr int entryPointWorkGroupY = 32;

// *******************************************
// Classes representing fluid simulation steps
// *******************************************

fluidsim::AdvectionStep::AdvectionStep(Shader& entryPointShader)
	: advectionProgram("Advection program")
{
	advectionProgram.attachShader(entryPointShader);
	advectionProgram.attachFile(ShaderType::Compute, "shaders/advection.glsl", "Advection shader");
	advectionProgram.build();
}

void fluidsim::AdvectionStep::compute(FluidState& fluidState, float dt)
{
	Context& context = Context::get();

	auto& params = fluidState.parameters;

	advectionProgram.uniform("udx", params.gridCellSize);
	{
		Empty::math::vec2 data(1.f / (params.gridSize.x * params.gridCellSize), 1.f / (params.gridSize.y * params.gridCellSize));
		advectionProgram.uniform("uOneOverGridSizeTimesDx", data);
	}
	advectionProgram.uniform("udt", dt);

	// Inputs are exposed with samplers to benefit from bilinear filtering

	auto& velocityXTex = fluidState.velocityX.getInput();
	advectionProgram.registerTexture("uVelocityX", velocityXTex, false);
	context.bind(velocityXTex.getLevel(0), allVelocityXBinding, AccessPolicy::ReadOnly, GPUScalarField::Format);

	auto& velocityYTex = fluidState.velocityY.getInput();
	advectionProgram.registerTexture("uVelocityY", velocityYTex, false);
	context.bind(velocityYTex.getLevel(0), allVelocityYBinding, AccessPolicy::ReadOnly, GPUScalarField::Format);

	context.setShaderProgram(advectionProgram);

	auto advect = [this, &context](BufferedScalarField& field, float boundaryCondition)
		{
			auto& fieldIn = field.getInput();
			advectionProgram.registerTexture("uFieldIn", fieldIn, false);
			context.bind(fieldIn, advectionFieldInBinding);

			auto& fieldOut = field.getOutput();
			advectionProgram.registerTexture("uFieldOut", fieldOut, false);
			context.bind(fieldOut.getLevel(0), advectionFieldOutBinding, AccessPolicy::WriteOnly, GPUScalarField::Format);

			advectionProgram.uniform("uBoundaryCondition", boundaryCondition);

			context.dispatchComputeIndirect();
		};

	context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
	advect(fluidState.velocityX, noSlipBoundaryCondition);
	advect(fluidState.velocityY, noSlipBoundaryCondition);
	advect(fluidState.inkDensity, zeroBoundaryCondition);

	fluidState.velocityX.swap();
	fluidState.velocityY.swap();
	fluidState.inkDensity.swap();
}

fluidsim::JacobiIterator::JacobiIterator(const std::string& label, Empty::math::uvec2 gridSize)
	: _workingField(label + " working field")
	, _fieldSource(nullptr)
	, _field(nullptr)
	, _fieldSourceBinding(-1)
	, _fieldInBinding(-1)
	, _workingFieldBinding(-1)
	, _fieldOutBinding(-1)
	, _numIterations(0)
	, _currentIteration(0)
	, _writeToWorkingField(true)
	, _iterationFieldInBinding(-1)
	, _iterationFieldOutBinding(-1)
{
	_workingField.setStorage(1, gridSize.x, gridSize.y);
}

void fluidsim::JacobiIterator::init(GPUScalarField& fieldSource, BufferedScalarField& field,
	int fieldSourceBinding, int fieldInBinding, int workingFieldBinding, int fieldOutBinding, int jacobiIterations)
{
	assert(jacobiIterations > 0);

	_fieldSource = &fieldSource;
	_field = &field;

	_fieldSourceBinding = fieldSourceBinding;
	_fieldInBinding = fieldInBinding;
	_workingFieldBinding = workingFieldBinding;
	_fieldOutBinding = fieldOutBinding;
	_numIterations = jacobiIterations;
	_currentIteration = 0;
	_writeToWorkingField = (_numIterations & 1) == 0;

	// Set state that doesn't change in between steps
	Context& context = Context::get();

	context.bind(_fieldSource->getLevel(0), _fieldSourceBinding, AccessPolicy::ReadOnly, GPUScalarField::Format);
	context.bind(_field->getInput().getLevel(0), _fieldInBinding, AccessPolicy::ReadWrite, GPUScalarField::Format);
	context.bind(_workingField.getLevel(0), _workingFieldBinding, AccessPolicy::ReadWrite, GPUScalarField::Format);
	context.bind(_field->getOutput().getLevel(0), _fieldOutBinding, AccessPolicy::ReadWrite, GPUScalarField::Format);

	// Alternate writes between the working texture and the output field so we write to the output
	// field last. The first step uses the actual input field as input, the other steps alternate between
	// working field and output field.
	_iterationFieldInBinding = _fieldInBinding;
	_iterationFieldOutBinding = _writeToWorkingField ? _workingFieldBinding : _fieldOutBinding;
}

void fluidsim::JacobiIterator::step(ShaderProgram& jacobiProgram)
{
	assert(_field != nullptr);
	assert(_currentIteration < _numIterations);

	Context& context = Context::get();

	jacobiProgram.uniform("uFieldSource", _fieldSourceBinding);
	jacobiProgram.uniform("uFieldIn", _iterationFieldInBinding);
	jacobiProgram.uniform("uFieldOut", _iterationFieldOutBinding);

	context.dispatchComputeIndirect();

	// I could simply swap _iterationFieldInBinding and _iterationFieldOutBinding but _iterationFieldInBinding
	// is _fieldInBinding for the first step only, and we can never write to that.
	_writeToWorkingField = !_writeToWorkingField;
	_iterationFieldInBinding = _iterationFieldOutBinding;
	_iterationFieldOutBinding = _writeToWorkingField ? _workingFieldBinding : _fieldOutBinding;

	++_currentIteration;
}

void fluidsim::JacobiIterator::reset()
{
	assert(_currentIteration == _numIterations);

	_fieldSource = nullptr;
	_field = nullptr;
	_fieldSourceBinding = -1;
	_fieldInBinding = -1;
	_workingFieldBinding = -1;
	_fieldOutBinding = -1;
	_numIterations = 0;
	_currentIteration = 0;
	_writeToWorkingField = true;
	_iterationFieldInBinding = -1;
	_iterationFieldOutBinding = -1;
}

fluidsim::DiffusionStep::DiffusionStep(Empty::math::uvec2 gridSize)
	: jacobiX("Diffuse Jacobi X", gridSize)
	, jacobiY("Diffuse Jacobi Y", gridSize)
{
}

void fluidsim::DiffusionStep::compute(ShaderProgram& jacobiProgram, FluidState& fluidState, float dt, int jacobiIterations)
{
	const auto& params = fluidState.parameters;

	Context& context = Context::get();

	// Perform Jacobi iterations on individual components
	jacobiX.init(fluidState.velocityX.getInput(), fluidState.velocityX, 0, 1, 2, 3, jacobiIterations);
	jacobiY.init(fluidState.velocityY.getInput(), fluidState.velocityY, 4, 5, 6, 7, jacobiIterations);

	// Upload solver parameters
	{
		float alpha = params.gridCellSize * params.gridCellSize / (params.viscosity * dt);
		float oneOverBeta = 1.f / (alpha + 4.f);
		jacobiProgram.uniform("uAlpha", alpha);
		jacobiProgram.uniform("uOneOverBeta", oneOverBeta);
		jacobiProgram.uniform("uBoundaryCondition", noSlipBoundaryCondition);
	}

	context.setShaderProgram(jacobiProgram);

	for (int i = 0; i < jacobiIterations; i++)
	{
		context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
		jacobiX.step(jacobiProgram);
		jacobiY.step(jacobiProgram);
	}

	jacobiX.reset();
	jacobiY.reset();

	fluidState.velocityX.swap();
	fluidState.velocityY.swap();
}

fluidsim::ForcesStep::ForcesStep(Shader& entryPointShader)
	: forcesProgram("Forces program")
{
	forcesProgram.attachShader(entryPointShader);
	forcesProgram.attachFile(ShaderType::Compute, "shaders/forces.glsl", "Forces shader");
	forcesProgram.build();
}

void fluidsim::ForcesStep::compute(FluidState& fluidState, const FluidSimMouseClickImpulse& impulse, float dt, bool velocityOnly)
{
	Context& context = Context::get();

	auto& velocityXTex = fluidState.velocityX.getInput();
	auto& velocityYTex = fluidState.velocityY.getInput();
	auto& inkTex = fluidState.inkDensity.getInput();

	forcesProgram.uniform("uMouseClick", impulse.position);
	forcesProgram.uniform("uOneOverForceRadius", 1.f / impulse.radius);

	context.setShaderProgram(forcesProgram);

	auto applyForce = [this, &context](GPUScalarField& field, float forceMagnitude, float boundaryCondition)
	{
		forcesProgram.registerTexture("uField", field, false);
		context.bind(field.getLevel(0), forcesFieldBinding, AccessPolicy::ReadWrite, GPUScalarField::Format);

		forcesProgram.uniform("uForceMagnitude", forceMagnitude);
		forcesProgram.uniform("uBoundaryCondition", boundaryCondition);
		context.dispatchComputeIndirect();
	};

	context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
	applyForce(fluidState.velocityX.getInput(), impulse.magnitude.x, noSlipBoundaryCondition);
	applyForce(fluidState.velocityY.getInput(), impulse.magnitude.y, noSlipBoundaryCondition);
	applyForce(fluidState.inkDensity.getInput(), impulse.inkAmount * dt, zeroBoundaryCondition);

	// Don't swap textures since we read from and write to the same textures
}

fluidsim::DivergenceStep::DivergenceStep()
	: divergenceProgram("Divergence program")
{
	divergenceProgram.attachFile(ShaderType::Compute, "shaders/divergence.glsl", "Divergence shader");
	divergenceProgram.build();
}

void fluidsim::DivergenceStep::compute(FluidState& fluidState)
{
	const auto& params = fluidState.parameters;

	Context& context = Context::get();

	auto& velocityXTex = fluidState.velocityX.getInput();
	auto& velocityYTex = fluidState.velocityY.getInput();

	divergenceProgram.uniform("uHalfOneOverDx", 1.f / (2.f * params.gridCellSize));
	divergenceProgram.registerTexture("uVelocityX", velocityXTex, false);
	divergenceProgram.registerTexture("uVelocityY", velocityYTex, false);
	divergenceProgram.registerTexture("uFieldOut", fluidState.divergenceTex, false);
	context.bind(velocityXTex.getLevel(0), allVelocityXBinding, AccessPolicy::ReadOnly, GPUScalarField::Format);
	context.bind(velocityYTex.getLevel(0), allVelocityYBinding, AccessPolicy::ReadOnly, GPUScalarField::Format);
	context.bind(fluidState.divergenceTex.getLevel(0), divergenceOutBinding, AccessPolicy::WriteOnly, GPUScalarField::Format);

	context.setShaderProgram(divergenceProgram);
	context.dispatchComputeIndirect();
}

fluidsim::PressureStep::PressureStep(Empty::math::uvec2 gridSize)
	: jacobi("Pressure jacobi", gridSize)
{
}

void fluidsim::PressureStep::compute(ShaderProgram& jacobiProgram, FluidState& fluidState, int jacobiIterations)
{
	const auto& params = fluidState.parameters;
	Context& context = Context::get();

	fluidState.pressure.clear();

	jacobi.init(fluidState.divergenceTex, fluidState.pressure, 0, 1, 2, 3, jacobiIterations);

	// Upload solver parameters
	{
		float alpha = -params.gridCellSize * params.gridCellSize * params.density;
		float oneOverBeta = 1.f / 4.f;
		jacobiProgram.uniform("uAlpha", alpha);
		jacobiProgram.uniform("uOneOverBeta", oneOverBeta);
		jacobiProgram.uniform("uBoundaryCondition", neumannBoundaryCondition);
	}

	context.setShaderProgram(jacobiProgram);

	for (int i = 0; i < jacobiIterations; i++)
	{
		context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
		jacobi.step(jacobiProgram);
	}

	jacobi.reset();

	fluidState.pressure.swap();
}

fluidsim::ProjectionStep::ProjectionStep(Shader& entryPointShader)
	: projectionProgram("Projection program")
{
	projectionProgram.attachShader(entryPointShader);
	projectionProgram.attachFile(ShaderType::Compute, "shaders/projection.glsl", "Projection shader");
	projectionProgram.build();
}

void fluidsim::ProjectionStep::compute(FluidState& fluidState)
{
	const auto& params = fluidState.parameters;

	Context& context = Context::get();

	auto& velocityXTex = fluidState.velocityX.getInput();
	auto& velocityYTex = fluidState.velocityY.getInput();
	auto& pressureTex = fluidState.pressure.getInput();

	projectionProgram.uniform("uHalfOneOverDx", 1.f / (2.f * params.gridCellSize));
	projectionProgram.registerTexture("uVelocityX", velocityXTex, false);
	projectionProgram.registerTexture("uVelocityY", velocityYTex, false);
	projectionProgram.registerTexture("uPressure", pressureTex, false);
	context.bind(velocityXTex.getLevel(0), allVelocityXBinding, AccessPolicy::ReadWrite, GPUScalarField::Format);
	context.bind(velocityYTex.getLevel(0), allVelocityYBinding, AccessPolicy::ReadWrite, GPUScalarField::Format);
	context.bind(pressureTex.getLevel(0), projectionPressureBinding, AccessPolicy::ReadOnly, GPUScalarField::Format);

	context.setShaderProgram(projectionProgram);
	context.dispatchComputeIndirect();

	// Don't swap textures since we read from and write to the same textures
}

FluidSim::FluidSim(Empty::math::uvec2 gridSize)
	: diffusionJacobiSteps(100)
	, pressureJacobiSteps(100)
	, runAdvection(true)
	, runDiffusion(true)
	, runDivergence(true)
	, runPressure(true)
	, runProjection(true)
	, _hooks()
	, _nextHookId(0)
	, _entryPointShader(ShaderType::Compute, "Entry point shader")
	, _jacobiProgram("Jacobi program")
	, _entryPointIndirectDispatchBuffer("Entry point indirect dispatch args")
{
	if (!_entryPointShader.setSourceFromFile("shaders/entry_point.glsl"))
		FATAL("Failed to compile entry point shader:\n" << _entryPointShader.getLog());

	_jacobiProgram.attachShader(_entryPointShader);
	_jacobiProgram.attachFile(ShaderType::Compute, "shaders/jacobi.glsl", "Jacobi shader");
	_jacobiProgram.build();

	Empty::math::uvec3 dispatch(gridSize.x / entryPointWorkGroupX, gridSize.y / entryPointWorkGroupY, 1);
	_entryPointIndirectDispatchBuffer.setStorage(sizeof(dispatch), BufferUsage::StaticDraw, dispatch);

	_advectionStep = std::make_unique<fluidsim::AdvectionStep>(_entryPointShader);
	_diffusionStep = std::make_unique<fluidsim::DiffusionStep>(gridSize);
	_forcesStep = std::make_unique<fluidsim::ForcesStep>(_entryPointShader);
	_divergenceStep = std::make_unique<fluidsim::DivergenceStep>();
	_pressureStep = std::make_unique<fluidsim::PressureStep>(gridSize);
	_projectionStep = std::make_unique<fluidsim::ProjectionStep>(_entryPointShader);
}

FluidSimHookId FluidSim::registerHook(FluidSimHook hook, FluidSimHookStage when)
{
	_hooks[_nextHookId] = std::make_pair(hook, when);

	return _nextHookId++;
}

bool FluidSim::modifyHookStage(FluidSimHookId id, FluidSimHookStage newWhen)
{
	if (_hooks.find(id) == _hooks.end())
		return false;

	_hooks[id].second = newWhen;

	return true;
}

void FluidSim::unregisterHook(FluidSimHookId id)
{
	_hooks.erase(id);
}

void FluidSim::applyForces(FluidState& fluidState, FluidSimMouseClickImpulse& impulse, bool velocityOnly, float dt)
{
	_forcesStep->compute(fluidState, impulse, dt, velocityOnly);
}

void FluidSim::advance(FluidState& fluidState, float dt)
{
	Context& context = Context::get();

	context.bind(_entryPointIndirectDispatchBuffer, BufferTarget::DispatchIndirect);

	for (auto& pair : _hooks)
		if (pair.second.second == FluidSimHookStage::Start)
			pair.second.first(fluidState, dt);

	if (runAdvection)
	{
		context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
		_advectionStep->compute(fluidState, dt);

		for (auto& pair : _hooks)
			if (pair.second.second == FluidSimHookStage::AfterAdvection)
				pair.second.first(fluidState, dt);
	}

	if (runDiffusion)
	{
		context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
		_diffusionStep->compute(_jacobiProgram, fluidState, dt, diffusionJacobiSteps);

		for (auto& pair : _hooks)
			if (pair.second.second == FluidSimHookStage::AfterDiffusion)
				pair.second.first(fluidState, dt);
	}

	if (runDivergence)
	{
		context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
		_divergenceStep->compute(fluidState);

		for (auto& pair : _hooks)
			if (pair.second.second == FluidSimHookStage::AfterDivergence)
				pair.second.first(fluidState, dt);
	}

	if (runPressure)
	{
		context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
		_pressureStep->compute(_jacobiProgram, fluidState, pressureJacobiSteps);

		for (auto& pair : _hooks)
			if (pair.second.second == FluidSimHookStage::AfterPressure)
				pair.second.first(fluidState, dt);
	}

	if (runProjection)
	{
		context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
		_projectionStep->compute(fluidState);

		for (auto& pair : _hooks)
			if (pair.second.second == FluidSimHookStage::AfterProjection)
				pair.second.first(fluidState, dt);
	}
}
