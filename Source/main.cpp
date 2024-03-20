#include <Empty/gl/Buffer.h>
#include <Empty/gl/VertexArray.h>
#include <Empty/gl/ShaderProgram.hpp>
#include <Empty/math/funcs.h>
#include <Empty/utils/macros.h>

#include "fields.hpp"
#include "fluid.hpp"
#include "gui.h"
#include "solver.hpp"

#define IM_VEC2_CLASS_EXTRA                                                   \
        constexpr ImVec2(const Empty::math::vec2& f) : x(f.x), y(f.y) {}      \
        operator Empty::math::vec2() const { return Empty::math::vec2(x,y); }

#include "Context.h"

using namespace Empty::gl;

Context Context::_instance;

void debugCallback(DebugMessageSource source, DebugMessageType type, DebugMessageSeverity severity, int id, const std::string& text, const void* userData)
{
	std::cout << Empty::utils::name(source) << " (" << Empty::utils::name(type) << ", " << Empty::utils::name(severity) << "): " << text << std::endl;
}

int _main(int argc, char* argv[])
{
	Context& context = Context::get();

	if (!context.init("Fluid simulation tests", 1920, 1080))
	{
		TRACE("Didn't work");
		return 1;
	}

	context.debugMessageControl(DebugMessageSource::DontCare, DebugMessageType::DontCare, DebugMessageSeverity::DontCare, false);
	context.debugMessageControl(DebugMessageSource::DontCare, DebugMessageType::Error, DebugMessageSeverity::DontCare, true);
	context.debugMessageCallback(debugCallback, nullptr);

	// Fluid setup
	FluidGridParameters grid;
	grid.size = Empty::math::uvec2(256, 256);
	grid.cellSize = 0.8f;
	FluidPhysicalProperties physics;
	physics.density = 1.f;
	physics.kinematicViscosity = 0.0025f;
	FluidState fluidState(grid, physics);
	FluidSim fluidSim(fluidState.grid.size);

	double then = glfwGetTime();
	Empty::math::vec2 mouseThen = ImGui::GetMousePos();

	SimulationControls simControls;
	simControls.impulse.radius = 40.f;
	simControls.impulse.inkAmount = 20.f;

	VertexArray debugVAO("Debug VAO");
	context.bind(debugVAO);

	// Velocity under mouse draw
	const char mouseVectorFragmentSource[] = "#version 450\n"
		"out vec4 fFragColor; void main() { fFragColor = vec4(1.); }";
	ShaderProgram mouseVectorProgram("Mouse vector program");
	mouseVectorProgram.attachFile(ShaderType::Vertex, "shaders/draw/mouse_vector_vertex.glsl", "Mouse vector vertex");
	mouseVectorProgram.attachSource(ShaderType::Fragment, mouseVectorFragmentSource, "Mouse vector fragment");
	mouseVectorProgram.build();

	// Debug texture draw
	FluidSimRenderParameters renderParams(context.frameWidth, context.frameHeight, grid.size.x, grid.size.y, 4.f);

	ShaderProgram debugDrawProgram("Debug draw program");
	debugDrawProgram.attachFile(ShaderType::Vertex, "shaders/draw/debug_vertex.glsl", "Debug draw vertex");
	debugDrawProgram.attachFile(ShaderType::Fragment, "shaders/draw/debug_fragment.glsl", "Debug draw fragment");
	debugDrawProgram.build();
	debugDrawProgram.uniform("uTextureSizeOverScreenSize", Empty::math::vec2(grid.size) * renderParams.cellSizeInPx / Empty::math::vec2(context.frameWidth, context.frameHeight));
	debugDrawProgram.uniform("uColorScale", simControls.colorScale);

	auto debugTextureLambda = [&simControls, &debugDrawProgram](FluidState& fluidState, float dt)
		{
			Context::get().memoryBarrier(MemoryBarrierType::ShaderImageAccess);
			if (simControls.displayDebugTexture)
				displayTexture(debugDrawProgram, fluidState, simControls.whichDebugTexture);
		};

	simControls.debugTextureLambdaHookId = fluidSim.registerHook(debugTextureLambda, FluidSimHookStage::Start);

	while (!glfwWindowShouldClose(context.window))
	{
		context.newFrame();

		double now = glfwGetTime();
		Empty::math::vec2 mouseNow = ImGui::GetMousePos();
		float dt = static_cast<float>(now - then);

		doGUI(fluidSim, fluidState, simControls, renderParams, debugDrawProgram, dt);

		/// Advance simulation

		if (!simControls.pauseSimulation || simControls.runOneStep)
		{
			float simdt = simControls.runOneStep ? 1 / 60.f : dt;

			// Apply an impulse and inject ink when the left mouse button is down,
			// or no ink if the right mouse button is down
			bool rightMouseDown = ImGui::IsMouseDown(ImGuiMouseButton_Right);
			if (!ImGui::GetIO().WantCaptureMouse && (ImGui::IsMouseDown(ImGuiMouseButton_Left) || rightMouseDown))
			{
				simControls.impulse.magnitude = (mouseNow - mouseThen) * simControls.forceScale;
				simControls.impulse.magnitude.y *= -1;
				simControls.impulse.position = mouseNow - renderParams.topLeftCorner;
				simControls.impulse.position /= renderParams.cellSizeInPx;
				simControls.impulse.position.y = fluidState.grid.size.y - simControls.impulse.position.y;

				context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
				fluidSim.applyForces(fluidState, simControls.impulse, rightMouseDown, simdt);
			}

			fluidSim.advance(fluidState, simdt);

			simControls.runOneStep = false;
		}
		else
		{
			// Only display the debug texture
			if (simControls.displayDebugTexture)
				displayTexture(debugDrawProgram, fluidState, simControls.whichDebugTexture);
		}

		// Display it
		{
			auto* drawList = ImGui::GetBackgroundDrawList();

			drawList->AddRect(renderParams.topLeftCorner - Empty::math::vec2(1, 1),
				renderParams.topLeftCorner + Empty::math::vec2(fluidState.grid.size) * renderParams.cellSizeInPx + Empty::math::vec2(1, 1),
				ImColor(0, 255, 0));

			if (!simControls.displayDebugTexture)
				displayTexture(debugDrawProgram, fluidState, 6); // ink texture
		}

		// Draw the velocity value at mouse cursor
		drawVelocityUnderMouse(mouseNow, fluidState, mouseVectorProgram, renderParams);

		// ImGui::ShowDemoWindow();

		context.swap();

		then = now;
		mouseThen = mouseNow;
	}

	context.terminate();

	return 0;
}

int main(int argc, char* argv[])
{
	try
	{
		return _main(argc, argv);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
		return 1;
	}
}