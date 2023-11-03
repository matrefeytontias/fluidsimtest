#pragma once

#include <functional>
#include <unordered_map>
#include <utility>

#include <Empty/gl/Buffer.h>
#include <Empty/gl/Shader.h>
#include <Empty/gl/ShaderProgram.hpp>

#include "fields.hpp"
#include "fluid.hpp"

// ****************************************
// Steps comprising a fluid simulation step
// ****************************************

namespace fluidsim
{

	struct AdvectionStep
	{
		AdvectionStep(Empty::gl::Shader& entryPointShader);

		void compute(FluidState& fluidState, float dt);

		Empty::gl::ShaderProgram advectionProgram;
	};
	
	struct JacobiIterator
	{
		JacobiIterator(const std::string& label, Empty::math::uvec2 gridSize);

		void init(GPUScalarField& fieldSource, BufferedScalarField& field,
			int fieldSourceBinding, int fieldInBinding, int workingFieldBinding, int fieldOutBinding, int jacobiIterations);
		// Expects all parameters except textures to be set in the jacobi program, and it to be active.
		void step(Empty::gl::ShaderProgram& jacobiProgram);
		void reset();

	private:
		GPUScalarField _workingField;

		GPUScalarField* _fieldSource;
		BufferedScalarField* _field;
		int _fieldSourceBinding;
		int _fieldInBinding;
		int _workingFieldBinding;
		int _fieldOutBinding;

		int _numIterations;
		int _currentIteration;
		bool _writeToWorkingField;
		int _iterationFieldInBinding;
		int _iterationFieldOutBinding;
	};

	struct DiffusionStep
	{
		DiffusionStep(Empty::math::uvec2 gridSize);

		void compute(Empty::gl::ShaderProgram& jacobiProgram, FluidState& fluidState, float dt, int jacobiIterations);

		Empty::gl::ShaderProgram velocityUnpackProgram;
		Empty::gl::ShaderProgram velocityPackProgram;
		BufferedScalarField velocityX;
		BufferedScalarField velocityY;
		JacobiIterator jacobiX;
		JacobiIterator jacobiY;
	};

	struct ForcesStep
	{
		ForcesStep(Empty::gl::Shader& entryPointShader);

		void compute(FluidState& fluidState, const FluidSimMouseClickImpulse& impulse, float dt, bool velocityOnly);

		Empty::gl::ShaderProgram forcesProgram;
	};

	struct DivergenceStep
	{
		DivergenceStep();

		void compute(FluidState& fluidState);

		Empty::gl::ShaderProgram divergenceProgram;
	};

	struct PressureStep
	{
		PressureStep(Empty::math::uvec2 gridSize);

		void compute(Empty::gl::ShaderProgram& jacobiProgram, FluidState& fluidState, int jacobiIterations);

		JacobiIterator jacobi;
	};

	struct ProjectionStep
	{
		ProjectionStep(Empty::gl::Shader& entryPointShader);

		void compute(FluidState& fluidState);

		Empty::gl::ShaderProgram projectionProgram;
	};

}

enum struct FluidSimHookStage : int
{
	Start,
	AfterAdvection,
	AfterDiffusion,
	AfterDivergence,
	AfterPressure,
	AfterProjection,
	Never,
};

using FluidSimHook = std::function<void(FluidState& fluidState, float dt)>;
using FluidSimHookId = uint64_t;

struct FluidSim
{
	FluidSim(Empty::math::uvec2 gridSize);

	FluidSimHookId registerHook(FluidSimHook hook, FluidSimHookStage when);
	bool modifyHookStage(FluidSimHookId, FluidSimHookStage newWhen);
	void unregisterHook(FluidSimHookId);

	void applyForces(FluidState& fluidState, FluidSimMouseClickImpulse& impulse, bool velocityOnly, float dt);
	void advance(FluidState& fluidState, float dt);

	int diffusionJacobiSteps;
	int pressureJacobiSteps;

	bool runAdvection;
	bool runDiffusion;
	bool runDivergence;
	bool runPressure;
	bool runProjection;

private:
	std::unordered_map<FluidSimHookId, std::pair<FluidSimHook, FluidSimHookStage>> _hooks;
	FluidSimHookId _nextHookId;
	Empty::gl::Shader _entryPointShader;
	Empty::gl::ShaderProgram _jacobiProgram;

	Empty::gl::Buffer _entryPointIndirectDispatchBuffer;

	std::unique_ptr<fluidsim::AdvectionStep> _advectionStep;
	std::unique_ptr<fluidsim::DiffusionStep> _diffusionStep;
	std::unique_ptr<fluidsim::ForcesStep> _forcesStep;
	std::unique_ptr<fluidsim::DivergenceStep> _divergenceStep;
	std::unique_ptr<fluidsim::PressureStep> _pressureStep;
	std::unique_ptr<fluidsim::ProjectionStep> _projectionStep;
};
