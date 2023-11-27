#include <Empty/gl/Buffer.h>
#include <Empty/gl/VertexArray.h>
#include <Empty/gl/ShaderProgram.hpp>
#include <Empty/math/funcs.h>
#include <Empty/utils/macros.h>

#include "Camera.h"
#include "fields.hpp"
#include "fluid.hpp"
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

TextureInfo selectDebugTexture(FluidState& fluidState, int whichDebugTexture)
{
	switch (whichDebugTexture)
	{
	case 0:
		return fluidState.velocityX.getInput();
	case 1:
		return fluidState.velocityY.getInput();
	case 2:
		return fluidState.velocityZ.getInput();
	case 3:
		return fluidState.pressure.getInput();
	case 4:
		return fluidState.divergenceTex;
	default:
		FATAL("invalid requested debug texture");
	}
}

void displayTexture(ShaderProgram& debugDrawProgram, const TextureInfo&& texture)
{
	Context& context = Context::get();

	debugDrawProgram.registerTexture("uTexture", texture);

	context.setShaderProgram(debugDrawProgram);
	context.drawArrays(PrimitiveType::Triangles, 0, 6);
}

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

	// Fluid setup
	FluidState fluidState(Empty::math::uvec3(64, 64, 64), 0.8f, 1.f, 0.0025f);
	FluidSim fluidSim(fluidState.parameters.gridSize);

	// Fluid rendering
	VertexArray debugVAO("Debug VAO");
	FluidSimRenderParameters fluidRenderParameters(Empty::math::vec3(0., 0., -3), fluidState.parameters.gridSize, 0.01f);
	FluidSimRenderer fluidRenderer(context.frameWidth, context.frameHeight);

	// Setup camera and input
	Camera camera(90.f, (float)context.frameWidth / context.frameHeight, 0.001f, 100.f);
	glfwSetInputMode(context.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetWindowUserPointer(context.window, &camera.freeze);
	glfwSetKeyCallback(context.window, keyCallback);

	double then = glfwGetTime();
	Empty::math::vec2 mouseThen(0, 0);

	FluidSimImpulse impulse;
	impulse.radius = 40.f;
	impulse.inkAmount = 20.f;
	float forceScale = 5.f;
	int gaussianImpulseAxis = 0;

	// Simulation control variables
	bool capFPS = false;
	bool pauseSimulation = false;
	bool runOneStep = false;

	bool displayDebugTexture = false;
	int whichDebugTexture = 0;
	int whenDebugTexture = 0;
	float colorScale = 1.f;
	Empty::math::vec4 debugRect(10, 10, 200, 200);
	int debugTextureSlice = 0;

	// Debug texture draw
	ShaderProgram debugDrawProgram("Debug draw program");
	debugDrawProgram.attachFile(ShaderType::Vertex, "shaders/draw/debug_vertex.glsl", "Debug draw vertex");
	debugDrawProgram.attachFile(ShaderType::Fragment, "shaders/draw/debug_fragment.glsl", "Debug draw fragment");
	debugDrawProgram.build();
	debugDrawProgram.uniform("uRect", debugRect);
	debugDrawProgram.uniform("uOneOverScreenSize", Empty::math::vec2(1.f / context.frameWidth, 1.f / context.frameHeight));
	debugDrawProgram.uniform("uColorScale", colorScale);
	debugDrawProgram.uniform("uUVZ", 0.f);

	auto debugTextureLambda = [&displayDebugTexture, &debugDrawProgram, &whichDebugTexture](FluidState& fluidState, float dt)
		{
			if (displayDebugTexture)
				displayTexture(debugDrawProgram, selectDebugTexture(fluidState, whichDebugTexture));
		};

	FluidSimHookId debugTextureLambdaHookId = fluidSim.registerHook(debugTextureLambda, FluidSimHookStage::Start);

	while (!glfwWindowShouldClose(context.window))
	{
		context.newFrame();

		double now = glfwGetTime();
		Empty::math::vec2 mouseNow = ImGui::GetMousePos();
		float dt = static_cast<float>(now - then);

		if (ImGui::Begin("Fluid simulation", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
		{
			ImGui::TextDisabled("%.1f fps", 1.f / dt);
			if (ImGui::Checkbox("Cap FPS", &capFPS))
				glfwSwapInterval(capFPS);
			ImGui::Checkbox("Pause simulation (P)", &pauseSimulation);
			if (ImGui::IsKeyPressed(ImGuiKey_P))
				pauseSimulation = !pauseSimulation;
			if (ImGui::Button("Run one step (R)"))
				runOneStep = true;
			if (ImGui::IsKeyPressed(ImGuiKey_R) && !ImGui::GetIO().WantCaptureKeyboard)
				runOneStep = true;
			if (ImGui::Button("Reset"))
				fluidState.reset();

			ImGui::Indent();
			ImGui::Checkbox("Advection", &fluidSim.runAdvection);
			ImGui::Checkbox("Diffusion", &fluidSim.runDiffusion);
			ImGui::Checkbox("Divergence", &fluidSim.runDivergence);
			ImGui::Checkbox("Pressure", &fluidSim.runPressure);
			ImGui::Checkbox("Projection", &fluidSim.runProjection);
			ImGui::Unindent();

			ImGui::Separator();
			ImGui::TextDisabled("Jacobi solver parameters");
			ImGui::DragInt("Diffusion Jacobi steps", &fluidSim.diffusionJacobiSteps, 1, 1);
			ImGui::DragInt("Pressure Jacobi steps", &fluidSim.pressureJacobiSteps, 1, 1);
			ImGui::Separator();
			ImGui::TextDisabled("Fluid physics properties");
			ImGui::SliderFloat("Grid cell size (m)", &fluidState.parameters.gridCellSize, 0.0001f, 1.f);
			ImGui::SliderFloat("Density (kg/dm^3)", &fluidState.parameters.density, 0.0001f, 1.f);
			ImGui::SliderFloat("Kinematic viscosity (m^2/s)", &fluidState.parameters.viscosity, 0.f, 0.005f, "%.5f");
			ImGui::Separator();
			ImGui::ColorEdit3("Ink color", fluidRenderParameters.inkColor);
			ImGui::DragFloat("Ink color scale", &fluidRenderParameters.inkMultiplier, 0.01f, 0.0f, 1.f);
			ImGui::Separator();
			ImGui::TextDisabled("Mouse click impulse parameters");
			ImGui::DragFloat("Force scale", &forceScale, 0.1f, 0.f, 20.f);
			ImGui::DragFloat("Force radius", &impulse.radius, 1.f, 1.f);
			ImGui::DragFloat("Ink injection", &impulse.inkAmount, 0.5f, 0.f, 50.f);
			{
				bool pressed = ImGui::Button("Apply centered gaussian");
				ImGui::SameLine();
				ImGui::Combo("Along which axis", &gaussianImpulseAxis, "X\0Y\0Z\0\0");
				if (pressed)
				{
					float scale = 20.f;
					FluidSimImpulse gImpulse;
					gImpulse.inkAmount = impulse.inkAmount * scale;
					auto axis = Empty::math::vec3::zero;
					axis[gaussianImpulseAxis] = 1.f;
					gImpulse.magnitude = axis * forceScale * scale;
					gImpulse.radius = impulse.radius;
					gImpulse.position = Empty::math::vec3(fluidState.parameters.gridSize) / 2.f;

					fluidSim.applyForces(fluidState, gImpulse, false, dt);
				}
			}
			ImGui::Separator();
			ImGui::TextDisabled("Debug texture display");
			ImGui::Checkbox("Display debug texture", &displayDebugTexture);
			if (ImGui::SliderInt("Debug texture Z slice", &debugTextureSlice, 0, fluidState.parameters.gridSize.z - 1))
				debugDrawProgram.uniform("uUVZ", (debugTextureSlice + 0.5f) / fluidState.parameters.gridSize.z);
			ImGui::Combo("Display which", &whichDebugTexture, "Velocity X\0Velocity Y\0Velocity Z\0Pressure\0Velocity divergence\0");
			if (ImGui::Combo("Display when", &whenDebugTexture, "Start of frame\0After advection\0After diffusion\0After divergence\0After pressure computation\0After projection\0\0"))
				fluidSim.modifyHookStage(debugTextureLambdaHookId, static_cast<FluidSimHookStage>(whenDebugTexture));

			if (ImGui::DragFloat("Debug color scale", &colorScale, 0.001f, 0.0f, 1.f))
				debugDrawProgram.uniform("uColorScale", colorScale);
		}
		ImGui::End();

		/// Simulation steps

		// Apply an impulse and inject ink when the left mouse button is down,
		// or no ink if the right mouse button is down
		bool rightMouseDown = ImGui::IsMouseDown(ImGuiMouseButton_Right);
		if (!ImGui::GetIO().WantCaptureMouse && (ImGui::IsMouseDown(ImGuiMouseButton_Left) || rightMouseDown))
		{
			impulse.magnitude.xy() = (mouseNow - mouseThen) * forceScale;
			impulse.magnitude.z = 0.f;
			impulse.magnitude.y *= -1;
			impulse.position.xy() = (mouseNow - debugRect.xy()) * Empty::math::vec2(fluidState.parameters.gridSize.xy()) / debugRect.zw();
			impulse.position.z = debugTextureSlice + 0.5f;
			impulse.position.y = fluidState.parameters.gridSize.y - impulse.position.y;

			context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
			fluidSim.applyForces(fluidState, impulse, rightMouseDown, dt);
		}

		context.bind(debugVAO);

		// Advance simulation
		if (!pauseSimulation || runOneStep)
		{
			fluidSim.advance(fluidState, dt);

			runOneStep = false;
		}
		else
		{
			// Only display the debug texture
			if (displayDebugTexture)
				displayTexture(debugDrawProgram, selectDebugTexture(fluidState, whichDebugTexture));
		}

		// Display it
		{
			fluidRenderer.renderFluidSim(fluidState, fluidRenderParameters, camera, debugTextureSlice);

			// Display the debug view
			auto* drawList = ImGui::GetBackgroundDrawList();

			drawList->AddRect(debugRect.xy() - Empty::math::vec2(1, 1),
				debugRect.xy() + debugRect.zw() + Empty::math::vec2(2, 2),
				ImColor(0, 255, 0));

			context.bind(debugVAO);
			if (!displayDebugTexture)
				displayTexture(debugDrawProgram, fluidState.inkDensity.getInput().getInfo());
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
			mouseNow.y - mouseThen.y
			);

		then = now;
		mouseThen = mouseNow;
	}

	context.terminate();

	return 0;
}
