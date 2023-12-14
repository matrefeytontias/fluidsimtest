#pragma once

#include <functional>
#include <unordered_map>
#include <utility>

#include <Empty/gl/Buffer.h>
#include <Empty/gl/Shader.h>
#include <Empty/gl/ShaderProgram.hpp>

#include "fields.hpp"
#include "fluid.hpp"

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
	FluidSim(Empty::math::uvec3 gridSize);
	~FluidSim();

	FluidSimHookId registerHook(FluidSimHook hook, FluidSimHookStage when);
	bool modifyHookStage(FluidSimHookId, FluidSimHookStage newWhen);
	void unregisterHook(FluidSimHookId);

	void applyForces(FluidState& fluidState, FluidSimImpulse& impulse, bool velocityOnly, float dt);
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

	struct AdvectionStep;
	struct DiffusionStep;
	struct ForcesStep;
	struct DivergenceStep;
	struct PressureStep;
	struct ProjectionStep;

	std::unique_ptr<AdvectionStep> _advectionStep;
	std::unique_ptr<DiffusionStep> _diffusionStep;
	std::unique_ptr<ForcesStep> _forcesStep;
	std::unique_ptr<DivergenceStep> _divergenceStep;
	std::unique_ptr<PressureStep> _pressureStep;
	std::unique_ptr<ProjectionStep> _projectionStep;
};
