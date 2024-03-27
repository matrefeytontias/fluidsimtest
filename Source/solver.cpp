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
constexpr int allVelocityZBinding = 2;

constexpr int advectionFieldInBinding = 3;
constexpr int advectionFieldOutBinding = 4;

constexpr int jacobiFieldSourceBinding = 0;
constexpr int jacobiFieldInBinding = 1;
constexpr int jacobiFieldOutBinding = 2;

constexpr int forcesFieldBinding = 0;

constexpr int divergenceOutBinding = 3;

constexpr int projectionPressureBinding = 3;

const Empty::math::bvec3 anyStagger(true, true, true);
const Empty::math::bvec3 xStagger(true, false, false);
const Empty::math::bvec3 yStagger(false, true, false);
const Empty::math::bvec3 zStagger(false, false, true);
const Empty::math::bvec3 noStagger(false, false, false);

// f(boundary) + f(neighbour) = 0 -> f(boundary) = -f(neighbour)
constexpr float noSlipBoundaryCondition = -1.f;
// On a staggered grid, we store boundary values directly.
constexpr float staggeredNoSlipBoundaryCondition = 0.f;
// f(boundary) - f(neighbour) = 0 -> f(boundary) = f(neighbour)
constexpr float neumannBoundaryCondition = 1.f;
// f(boundary) = 0
constexpr float zeroBoundaryCondition = 0.f;

constexpr int entryPointWorkGroupX = 8;
constexpr int entryPointWorkGroupY = 8;
constexpr int entryPointWorkGroupZ = 8;

// *******************************************
// Classes representing fluid simulation steps
// *******************************************

struct FluidSim::GridScrollStep
{
	GridScrollStep()
		: scrollProgram("Grid scroll program")
	{
		scrollProgram.attachFile(ShaderType::Compute, "shaders/sim/grid_scroll.glsl", "Grid scroll shader");
		if (!scrollProgram.build())
		{
			FATAL("Could not build grid scroll program:\n" << scrollProgram.getLog());
		}
	}

	void compute(FluidState& fluidState, Empty::math::ivec3 scroll)
	{
		Context& context = Context::get();

		scrollProgram.uniform("uTexelScroll", scroll);
		context.setShaderProgram(scrollProgram);

		auto doScroll = [this, &context](BufferedScalarField& field)
			{
				auto& fieldIn = field.getInput();
				scrollProgram.registerTexture("uFieldIn", fieldIn, false);
				context.bind(fieldIn.getLevel(0), 0, AccessPolicy::ReadOnly, GPUScalarField::Format);

				auto& fieldOut = field.getOutput();
				scrollProgram.registerTexture("uFieldOut", fieldOut, false);
				context.bind(fieldOut.getLevel(0), 1, AccessPolicy::WriteOnly, GPUScalarField::Format);

				context.dispatchComputeIndirect();
			};

		doScroll(fluidState.velocityX);
		doScroll(fluidState.velocityY);
		doScroll(fluidState.velocityZ);
		doScroll(fluidState.pressure);
		doScroll(fluidState.inkDensity);
	}

	ShaderProgram scrollProgram;
};

struct FluidSim::AdvectionStep
{
	AdvectionStep(Shader& entryPointShader)
		: advectionProgram("Advection program")
	{
		advectionProgram.attachShader(entryPointShader);
		advectionProgram.attachFile(ShaderType::Compute, "shaders/sim/advection.glsl", "Advection shader");
		advectionProgram.build();
	}

	void compute(FluidState& fluidState, float dt)
	{
		Context& context = Context::get();

		auto& params = fluidState.grid;

		advectionProgram.uniform("uGridParams.dx", params.cellSize);
		advectionProgram.uniform("uGridParams.oneOverDx", 1.f / params.cellSize);
		{
			Empty::math::vec3 data(1.f / params.size.x, 1.f / params.size.y, 1.f / params.size.z);
			advectionProgram.uniform("uGridParams.oneOverGridSize", data);
		}
		advectionProgram.uniform("udt", dt);

		// Inputs are exposed with samplers to benefit from bilinear filtering

		auto& velocityXTex = fluidState.velocityX.getInput();
		advectionProgram.registerTexture("uVelocityX", velocityXTex, false);
		context.bind(velocityXTex, allVelocityXBinding);

		auto& velocityYTex = fluidState.velocityY.getInput();
		advectionProgram.registerTexture("uVelocityY", velocityYTex, false);
		context.bind(velocityYTex, allVelocityYBinding);

		auto& velocityZTex = fluidState.velocityZ.getInput();
		advectionProgram.registerTexture("uVelocityZ", velocityZTex, false);
		context.bind(velocityZTex, allVelocityZBinding);

		context.setShaderProgram(advectionProgram);

		auto advect = [this, &context](BufferedScalarField& field, float boundaryCondition, Empty::math::bvec3 stagger)
			{
				auto& fieldIn = field.getInput();
				advectionProgram.registerTexture("uFieldIn", fieldIn, false);
				context.bind(fieldIn, advectionFieldInBinding);

				auto& fieldOut = field.getOutput();
				advectionProgram.registerTexture("uFieldOut", fieldOut, false);
				context.bind(fieldOut.getLevel(0), advectionFieldOutBinding, AccessPolicy::WriteOnly, GPUScalarField::Format);

				advectionProgram.uniform("uBoundaryCondition", boundaryCondition);
				advectionProgram.uniform("uFieldStagger", stagger);

				context.dispatchComputeIndirect();
			};

		advect(fluidState.velocityX, staggeredNoSlipBoundaryCondition, xStagger);
		advect(fluidState.velocityY, staggeredNoSlipBoundaryCondition, yStagger);
		advect(fluidState.velocityZ, staggeredNoSlipBoundaryCondition, zStagger);
		advect(fluidState.inkDensity, zeroBoundaryCondition, noStagger);

		fluidState.velocityX.swap();
		fluidState.velocityY.swap();
		fluidState.velocityZ.swap();
		fluidState.inkDensity.swap();
	}


	Empty::gl::ShaderProgram advectionProgram;
};

struct JacobiIterator
{
	JacobiIterator(const std::string& label, Empty::math::uvec3 gridSize)
		: _workingField(label + " working field")
		, _fieldSource(nullptr)
		, _field(nullptr)
		, _numIterations(-1)
		, _currentIteration(0)
		, _writeToWorkingField(true)
		, _iterationFieldIn()
		, _iterationFieldOut()
	{
		_workingField.setStorage(1, gridSize.x, gridSize.y, gridSize.z);
	}

	void init(GPUScalarField& fieldSource, BufferedScalarField& field, int jacobiIterations)
	{
		assert(jacobiIterations > 0);

		_fieldSource = &fieldSource;
		_field = &field;

		_numIterations = jacobiIterations;
		_currentIteration = 0;
		_writeToWorkingField = (_numIterations & 1) == 0;

		// Alternate writes between the working texture and the output field so we write to the output
		// field last. The first step uses the actual input field as input, the other steps alternate between
		// working field and output field.
		_iterationFieldIn = field.getInput().getLevel(0);
		_iterationFieldOut = _writeToWorkingField ? _workingField.getLevel(0) : field.getOutput().getLevel(0);
	}

	// Expects all parameters except textures to be set in the jacobi program, and it to be active.
	void step(ShaderProgram& jacobiProgram)
	{
		assert(_field != nullptr);
		assert(_currentIteration < _numIterations);

		Context& context = Context::get();

		context.bind(_fieldSource->getLevel(0), jacobiFieldSourceBinding, AccessPolicy::ReadOnly, GPUScalarField::Format);
		context.bind(_iterationFieldIn, jacobiFieldInBinding, AccessPolicy::ReadOnly, GPUScalarField::Format);
		context.bind(_iterationFieldOut, jacobiFieldOutBinding, AccessPolicy::WriteOnly, GPUScalarField::Format);

		context.dispatchComputeIndirect();

		// I could simply swap _iterationFieldInBinding and _iterationFieldOutBinding but _iterationFieldInBinding
		// is _fieldInBinding for the first step only, and we can never write to that.
		_writeToWorkingField = !_writeToWorkingField;
		_iterationFieldIn = _iterationFieldOut;
		_iterationFieldOut = _writeToWorkingField ? _workingField.getLevel(0) : _field->getOutput().getLevel(0);

		++_currentIteration;
	}

	void reset()
	{
		assert(_currentIteration == _numIterations);

		_fieldSource = nullptr;
		_field = nullptr;
		_numIterations = -1;
		_currentIteration = 0;
		_writeToWorkingField = true;
		_iterationFieldIn = TextureLevelInfo{};
		_iterationFieldOut = TextureLevelInfo{};
	}

private:
	GPUScalarField _workingField;

	GPUScalarField* _fieldSource;
	BufferedScalarField* _field;

	int _numIterations;
	int _currentIteration;
	bool _writeToWorkingField;
	Empty::gl::TextureLevelInfo _iterationFieldIn;
	Empty::gl::TextureLevelInfo _iterationFieldOut;
};

struct FluidSim::DiffusionStep
{
	DiffusionStep(Empty::math::uvec3 gridSize)
		: jacobiX("Diffuse Jacobi X", gridSize)
		, jacobiY("Diffuse Jacobi Y", gridSize)
		, jacobiZ("Diffuse Jacobi Z", gridSize)
	{ }

	void compute(ShaderProgram& jacobiProgram, FluidState& fluidState, float dt, int jacobiIterations)
	{
		const auto& params = fluidState.grid;

		Context& context = Context::get();

		// Perform Jacobi iterations on individual components
		jacobiX.init(fluidState.velocityX.getInput(), fluidState.velocityX, jacobiIterations);
		jacobiY.init(fluidState.velocityY.getInput(), fluidState.velocityY, jacobiIterations);
		jacobiZ.init(fluidState.velocityZ.getInput(), fluidState.velocityZ, jacobiIterations);

		// Upload solver parameters
		{
			float alpha = params.cellSize * params.cellSize / (fluidState.physics.kinematicViscosity * dt);
			float oneOverBeta = 1.f / (alpha + 6.f);
			jacobiProgram.uniform("uAlpha", alpha);
			jacobiProgram.uniform("uOneOverBeta", oneOverBeta);
			jacobiProgram.uniform("uBoundaryCondition", staggeredNoSlipBoundaryCondition);
		}

		context.setShaderProgram(jacobiProgram);

		for (int i = 0; i < jacobiIterations; i++)
		{
			if (i > 0)
				context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
			jacobiProgram.uniform("uFieldStagger", xStagger);
			jacobiX.step(jacobiProgram);
			jacobiProgram.uniform("uFieldStagger", yStagger);
			jacobiY.step(jacobiProgram);
			jacobiProgram.uniform("uFieldStagger", zStagger);
			jacobiZ.step(jacobiProgram);
		}

		jacobiX.reset();
		jacobiY.reset();
		jacobiZ.reset();

		fluidState.velocityX.swap();
		fluidState.velocityY.swap();
		fluidState.velocityZ.swap();
	}

	JacobiIterator jacobiX;
	JacobiIterator jacobiY;
	JacobiIterator jacobiZ;
};

struct FluidSim::ForcesStep
{
	ForcesStep(Shader& entryPointShader)
		: forcesProgram("Forces program")
	{
		forcesProgram.attachShader(entryPointShader);
		forcesProgram.attachFile(ShaderType::Compute, "shaders/sim/forces.glsl", "Forces shader");
		forcesProgram.build();
	}

	void compute(FluidState& fluidState, const FluidSimMouseClickImpulse& impulse, float dt, bool velocityOnly)
	{
		Context& context = Context::get();

		forcesProgram.uniform("uForceCenter", impulse.position);
		forcesProgram.uniform("uOneOverForceRadius", 1.f / impulse.radius);

		context.setShaderProgram(forcesProgram);

		auto applyForce = [this, &context](GPUScalarField& field, float forceMagnitude, float boundaryCondition, Empty::math::bvec3 stagger)
			{
				forcesProgram.registerTexture("uField", field, false);
				context.bind(field.getLevel(0), forcesFieldBinding, AccessPolicy::ReadWrite, GPUScalarField::Format);

				forcesProgram.uniform("uForceMagnitude", forceMagnitude);
				forcesProgram.uniform("uBoundaryCondition", boundaryCondition);
				forcesProgram.uniform("uFieldStagger", stagger);
				context.dispatchComputeIndirect();
			};

	applyForce(fluidState.velocityX.getInput(), impulse.magnitude.x, staggeredNoSlipBoundaryCondition, xStagger);
	applyForce(fluidState.velocityY.getInput(), impulse.magnitude.y, staggeredNoSlipBoundaryCondition, yStagger);
	applyForce(fluidState.velocityZ.getInput(), impulse.magnitude.z, staggeredNoSlipBoundaryCondition, zStagger);
		if (!velocityOnly)
		applyForce(fluidState.inkDensity.getInput(), impulse.inkAmount * dt, zeroBoundaryCondition, noStagger);

		// Don't swap textures since we read from and write to the same textures
	}

	Empty::gl::ShaderProgram forcesProgram;
};

struct FluidSim::DivergenceStep
{
	DivergenceStep()
		: divergenceProgram("Divergence program")
	{
		divergenceProgram.attachFile(ShaderType::Compute, "shaders/sim/divergence.glsl", "Divergence shader");
		divergenceProgram.build();
	}

	void compute(FluidState& fluidState, GPUScalarField& tex)
	{
		const auto& params = fluidState.grid;

		Context& context = Context::get();

		auto& velocityXTex = fluidState.velocityX.getInput();
		auto& velocityYTex = fluidState.velocityY.getInput();
		auto& velocityZTex = fluidState.velocityZ.getInput();

		divergenceProgram.uniform("uOneOverDx", 1.f / params.cellSize);
		divergenceProgram.registerTexture("uVelocityX", velocityXTex, false);
		divergenceProgram.registerTexture("uVelocityY", velocityYTex, false);
		divergenceProgram.registerTexture("uVelocityZ", velocityZTex, false);
		divergenceProgram.registerTexture("uDivergence", tex, false);
		context.bind(velocityXTex.getLevel(0), allVelocityXBinding, AccessPolicy::ReadOnly, GPUScalarField::Format);
		context.bind(velocityYTex.getLevel(0), allVelocityYBinding, AccessPolicy::ReadOnly, GPUScalarField::Format);
		context.bind(velocityZTex.getLevel(0), allVelocityZBinding, AccessPolicy::ReadOnly, GPUScalarField::Format);
		context.bind(tex.getLevel(0), divergenceOutBinding, AccessPolicy::WriteOnly, GPUScalarField::Format);

		context.setShaderProgram(divergenceProgram);
		context.dispatchComputeIndirect();
	}

	Empty::gl::ShaderProgram divergenceProgram;
};

struct FluidSim::PressureStep
{
	PressureStep(Empty::math::uvec3 gridSize)
		: jacobi("Pressure jacobi", gridSize)
	{ }

	void compute(ShaderProgram& jacobiProgram, FluidState& fluidState, int jacobiIterations, bool reuseLastPressure)
	{
		const auto& params = fluidState.grid;
		Context& context = Context::get();

		if (!reuseLastPressure)
			fluidState.pressure.clear();

		jacobi.init(fluidState.divergenceTex, fluidState.pressure, jacobiIterations);

		// Upload solver parameters
		{
			float alpha = -params.cellSize * params.cellSize * fluidState.physics.density;
			float oneOverBeta = 1.f / 6.f;
			jacobiProgram.uniform("uAlpha", alpha);
			jacobiProgram.uniform("uOneOverBeta", oneOverBeta);
			jacobiProgram.uniform("uBoundaryCondition", neumannBoundaryCondition);
			jacobiProgram.uniform("uFieldStagger", noStagger);
		}

		context.setShaderProgram(jacobiProgram);

		for (int i = 0; i < jacobiIterations; i++)
		{
			if (i > 0)
				context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
			jacobi.step(jacobiProgram);
		}

		jacobi.reset();

		fluidState.pressure.swap();
	}

	JacobiIterator jacobi;
};

struct FluidSim::ProjectionStep
{
	ProjectionStep(Shader& entryPointShader)
		: projectionProgram("Projection program")
	{
		projectionProgram.attachShader(entryPointShader);
		projectionProgram.attachFile(ShaderType::Compute, "shaders/sim/projection.glsl", "Projection shader");
		projectionProgram.build();
	}

	void compute(FluidState& fluidState)
	{
		const auto& params = fluidState.grid;

		Context& context = Context::get();

		auto& velocityXTex = fluidState.velocityX.getInput();
		auto& velocityYTex = fluidState.velocityY.getInput();
		auto& velocityZTex = fluidState.velocityZ.getInput();
		auto& pressureTex = fluidState.pressure.getInput();

		projectionProgram.uniform("uOneOverDx", 1.f / params.cellSize);
		projectionProgram.registerTexture("uVelocityX", velocityXTex, false);
		projectionProgram.registerTexture("uVelocityY", velocityYTex, false);
		projectionProgram.registerTexture("uVelocityZ", velocityZTex, false);
		projectionProgram.registerTexture("uPressure", pressureTex, false);
		projectionProgram.uniform("uFieldStagger", anyStagger);
		context.bind(velocityXTex.getLevel(0), allVelocityXBinding, AccessPolicy::ReadWrite, GPUScalarField::Format);
		context.bind(velocityYTex.getLevel(0), allVelocityYBinding, AccessPolicy::ReadWrite, GPUScalarField::Format);
		context.bind(velocityZTex.getLevel(0), allVelocityZBinding, AccessPolicy::ReadWrite, GPUScalarField::Format);
		context.bind(pressureTex.getLevel(0), projectionPressureBinding, AccessPolicy::ReadOnly, GPUScalarField::Format);

		context.setShaderProgram(projectionProgram);
		context.dispatchComputeIndirect();

		// Don't swap textures since we read from and write to the same textures
	}

	Empty::gl::ShaderProgram projectionProgram;
};

// **********************
// Main fluid sim methods
// **********************

FluidSim::FluidSim(Empty::math::uvec3 gridSize)
	: diffusionJacobiSteps(100)
	, pressureJacobiSteps(100)
	, reuseLastPressure(true)
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
	if (!_entryPointShader.setSourceFromFile("shaders/sim/entry_point.glsl"))
		FATAL("Failed to compile entry point shader:\n" << _entryPointShader.getLog());

	_jacobiProgram.attachShader(_entryPointShader);
	_jacobiProgram.attachFile(ShaderType::Compute, "shaders/sim/jacobi.glsl", "Jacobi shader");
	_jacobiProgram.build();

	Empty::math::uvec3 dispatch(gridSize.x / entryPointWorkGroupX, gridSize.y / entryPointWorkGroupY, gridSize.z / entryPointWorkGroupZ);
	_entryPointIndirectDispatchBuffer.setStorage(sizeof(dispatch), BufferUsage::StaticDraw, dispatch);

	_gridScrollStep = std::make_unique<GridScrollStep>();
	_advectionStep = std::make_unique<AdvectionStep>(_entryPointShader);
	_diffusionStep = std::make_unique<DiffusionStep>(gridSize);
	_forcesStep = std::make_unique<ForcesStep>(_entryPointShader);
	_divergenceStep = std::make_unique<DivergenceStep>();
	_pressureStep = std::make_unique<PressureStep>(gridSize);
	_projectionStep = std::make_unique<ProjectionStep>(_entryPointShader);
}

FluidSim::~FluidSim() = default;

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
	Context::get().bind(_entryPointIndirectDispatchBuffer, BufferTarget::DispatchIndirect);
	_forcesStep->compute(fluidState, impulse, dt, velocityOnly);
}

void FluidSim::scrollGrid(FluidState& fluidState, Empty::math::ivec3 scroll)
{
	Context::get().bind(_entryPointIndirectDispatchBuffer, BufferTarget::DispatchIndirect);
	_gridScrollStep->compute(fluidState, scroll);
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
	}

	for (auto& pair : _hooks)
		if (pair.second.second == FluidSimHookStage::AfterAdvection)
			pair.second.first(fluidState, dt);

	if (runDiffusion)
	{
		context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
		_diffusionStep->compute(_jacobiProgram, fluidState, dt, diffusionJacobiSteps);
	}

	for (auto& pair : _hooks)
		if (pair.second.second == FluidSimHookStage::AfterDiffusion)
			pair.second.first(fluidState, dt);

	if (runDivergence)
	{
		context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
		_divergenceStep->compute(fluidState, fluidState.divergenceTex);
	}

	for (auto& pair : _hooks)
		if (pair.second.second == FluidSimHookStage::AfterDivergence)
			pair.second.first(fluidState, dt);

	if (runPressure)
	{
		context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
		_pressureStep->compute(_jacobiProgram, fluidState, pressureJacobiSteps, reuseLastPressure);
	}

	for (auto& pair : _hooks)
		if (pair.second.second == FluidSimHookStage::AfterPressure)
			pair.second.first(fluidState, dt);

	if (runProjection)
	{
		context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
		_projectionStep->compute(fluidState);
	}

	// Re-compute divergence to check that it is in fact 0
	context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
	_divergenceStep->compute(fluidState, fluidState.divergenceCheckTex);

	for (auto& pair : _hooks)
		if (pair.second.second == FluidSimHookStage::AfterProjection)
			pair.second.first(fluidState, dt);
}
