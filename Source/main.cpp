#include <Empty/gl/Buffer.h>
#include <Empty/gl/VertexArray.h>
#include <Empty/gl/ShaderProgram.hpp>
#include <Empty/math/funcs.h>
#include <Empty/utils/macros.h>

#include "Camera.h"
#include "fields.hpp"
#include "fluid.hpp"
#include "gui.h"
#include "render.hpp"
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

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
	{
		bool *cameraFreeze = reinterpret_cast<bool*>(glfwGetWindowUserPointer(window));
		*cameraFreeze ^= true;
		glfwSetInputMode(window, GLFW_CURSOR, *cameraFreeze ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
	}
};

int main(int argc, char* argv[])
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

	context.enable(ContextCapability::Blend);
	context.blendFunction(BlendFunction::SourceAlpha, BlendFunction::OneMinusSourceAlpha);

	// Fluid setup
	FluidGridParameters grid;
	grid.size = Empty::math::uvec3(64, 64, 64);
	grid.cellSize = 0.8f;
	FluidPhysicalProperties physics;
	physics.density = 1.f;
	physics.kinematicViscosity = 0.0025f;
	FluidState fluidState(grid, physics);
	FluidSim fluidSim(fluidState.grid.size);

	// Fluid rendering
	VertexArray debugVAO("Debug VAO");
	FluidSimRenderParameters fluidRenderParameters(Empty::math::vec3(0., 0., -3), fluidState.grid.size, 0.01f);
	FluidSimRenderer fluidRenderer(context.frameWidth, context.frameHeight);

	// Setup camera and input
	Camera camera(90.f, (float)context.frameWidth / context.frameHeight, 0.001f, 100.f);
	glfwSetInputMode(context.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetWindowUserPointer(context.window, &camera.freeze);
	glfwSetKeyCallback(context.window, keyCallback);

	double then = glfwGetTime();
	Empty::math::vec2 mouseThen(0, 0);

	SimulationControls simControls;

	// Debug texture draw
	ShaderProgram debugDrawProgram("Debug draw program");
	debugDrawProgram.attachFile(ShaderType::Vertex, "shaders/draw/debug_vertex.glsl", "Debug draw vertex");
	debugDrawProgram.attachFile(ShaderType::Fragment, "shaders/draw/debug_fragment.glsl", "Debug draw fragment");
	debugDrawProgram.build();
	debugDrawProgram.uniform("uRect", simControls.debugRect);
	debugDrawProgram.uniform("uOneOverScreenSize", Empty::math::vec2(1.f / context.frameWidth, 1.f / context.frameHeight));
	debugDrawProgram.uniform("uColorScale", simControls.colorScale);
	debugDrawProgram.uniform("uUVZ", 0.f);

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

		doGUI(fluidSim, fluidState, simControls, fluidRenderParameters, debugDrawProgram, dt);

		/// Simulation steps

		// Apply an impulse and inject ink when the left mouse button is down,
		// or no ink if the right mouse button is down
		bool rightMouseDown = ImGui::IsMouseDown(ImGuiMouseButton_Right);
		if (!ImGui::GetIO().WantCaptureMouse && (ImGui::IsMouseDown(ImGuiMouseButton_Left) || rightMouseDown))
		{
			auto& impulse = simControls.impulse;
			impulse.magnitude.xy() = (mouseNow - mouseThen) * simControls.forceScale;
			impulse.magnitude.z = 0.f;
			impulse.magnitude.y *= -1;
			impulse.position.xy() = (mouseNow - simControls.debugRect.xy()) * Empty::math::vec2(fluidState.grid.size.xy()) / simControls.debugRect.zw();
			impulse.position.z = simControls.debugTextureSlice + 0.5f;
			impulse.position.y = fluidState.grid.size.y - impulse.position.y;

			context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
			fluidSim.applyForces(fluidState, impulse, rightMouseDown, dt);
		}

		context.bind(debugVAO);

		// Advance simulation
		if (!simControls.pauseSimulation || simControls.runOneStep)
		{
			fluidSim.advance(fluidState, simControls.runOneStep ? 1 / 60.f : dt);

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
			fluidRenderer.renderFluidSim(fluidState, fluidRenderParameters, camera, simControls.debugTextureSlice);

			// Display the debug view
			auto* drawList = ImGui::GetBackgroundDrawList();

			drawList->AddRect(simControls.debugRect.xy() - Empty::math::vec2(1, 1),
				simControls.debugRect.xy() + simControls.debugRect.zw() + Empty::math::vec2(2, 2),
				ImColor(0, 255, 0));

			context.bind(debugVAO);
			if (!simControls.displayDebugTexture)
				displayTexture(debugDrawProgram, fluidState, 7); // ink density
		}

		// ImGui::ShowDemoWindow();

		context.swap();

		camera.processInput(
			glfwGetKey(context.window, GLFW_KEY_W) == GLFW_PRESS,
			glfwGetKey(context.window, GLFW_KEY_S) == GLFW_PRESS,
			glfwGetKey(context.window, GLFW_KEY_E) == GLFW_PRESS,
			glfwGetKey(context.window, GLFW_KEY_Q) == GLFW_PRESS,
			glfwGetKey(context.window, GLFW_KEY_A) == GLFW_PRESS,
			glfwGetKey(context.window, GLFW_KEY_D) == GLFW_PRESS,
			mouseNow.x - mouseThen.x,
			mouseNow.y - mouseThen.y,
			dt
			);

		then = now;
		mouseThen = mouseNow;
	}

	context.terminate();

	return 0;
}
