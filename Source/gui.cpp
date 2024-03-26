#include "gui.h"

#include "Context.h"

void doGUI(FluidSim& fluidSim, FluidState& fluidState, SimulationControls& simControls, FluidSimRenderParameters& renderParams, Empty::gl::ShaderProgram& debugDrawProgram, float dt)
{
	if (ImGui::Begin("Fluid simulation", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
	{
		ImGui::TextDisabled("%.1f fps", 1.f / dt);
		if (ImGui::Checkbox("Cap FPS", &simControls.capFPS))
			glfwSwapInterval(simControls.capFPS);
		ImGui::Checkbox("Pause simulation (P)", &simControls.pauseSimulation);
		if (ImGui::IsKeyPressed(ImGuiKey_P))
			simControls.pauseSimulation = !simControls.pauseSimulation;
		if (ImGui::Button("Run one step (R)"))
			simControls.runOneStep = true;
		if (ImGui::IsKeyPressed(ImGuiKey_R) && !ImGui::GetIO().WantCaptureKeyboard)
			simControls.runOneStep = true;
		if (ImGui::Button("Reset"))
			fluidState.reset();

		ImGui::Checkbox("Advection", &fluidSim.runAdvection);
		ImGui::Checkbox("Diffusion", &fluidSim.runDiffusion);
		ImGui::Checkbox("Divergence", &fluidSim.runDivergence);
		ImGui::Checkbox("Pressure", &fluidSim.runPressure);
		ImGui::Checkbox("Projection", &fluidSim.runProjection);

		ImGui::Separator();
		ImGui::DragInt3("Grid scroll", simControls.gridScroll);
		ImGui::SameLine();
		if (ImGui::Button("Apply"))
			fluidSim.scrollGrid(fluidState, simControls.gridScroll);

		ImGui::Separator();
		ImGui::TextDisabled("Jacobi solver parameters");
		ImGui::DragInt("Diffusion Jacobi steps", &fluidSim.diffusionJacobiSteps, 1, 1);
		ImGui::DragInt("Pressure Jacobi steps", &fluidSim.pressureJacobiSteps, 1, 1);
		ImGui::Checkbox("Reuse pressure from last step", &fluidSim.reuseLastPressure);
		ImGui::Separator();
		ImGui::TextDisabled("Fluid physics properties");
		ImGui::SliderFloat("Grid cell size (m)", &fluidState.grid.cellSize, 0.0001f, 1.f);
		ImGui::SliderFloat("Density (kg/dm^3)", &fluidState.physics.density, 0.0001f, 1.f);
		ImGui::SliderFloat("Kinematic viscosity (m^2/s)", &fluidState.physics.kinematicViscosity, 0.f, 0.005f, "%.5f");
		ImGui::Separator();
		ImGui::TextDisabled("Fluid rendering options");
		ImGui::DragFloat("In-world sim cell size", &renderParams.gridCellSizeInUnits, 0.001f);
		ImGui::ColorEdit3("Ink color", renderParams.inkColor);
		ImGui::DragFloat("Ink color scale", &renderParams.inkMultiplier, 0.01f, 0.0f, 5.f);
		ImGui::Separator();
		ImGui::TextDisabled("Mouse click impulse parameters");
		ImGui::DragFloat("Force scale", &simControls.forceScale, 0.1f, 0.f, 20.f);
		ImGui::DragFloat("Force radius", &simControls.impulse.radius, 1.f, 1.f);
		ImGui::DragFloat("Ink injection", &simControls.impulse.inkAmount, 0.5f, 0.f, 50.f);
		{
			bool pressed = ImGui::Button("Apply centered gaussian");
			ImGui::SameLine();
			ImGui::Combo("Along which axis", &simControls.gaussianImpulseAxis, "X\0Y\0Z\0\0");
			if (pressed)
			{
				float scale = 20.f;
				FluidSimMouseClickImpulse gImpulse;
				gImpulse.inkAmount = simControls.impulse.inkAmount * scale;
				auto axis = Empty::math::vec3::zero;
				axis[simControls.gaussianImpulseAxis] = 1.f;
				gImpulse.magnitude = axis * simControls.forceScale * scale;
				gImpulse.radius = simControls.impulse.radius;
				gImpulse.position = Empty::math::vec3(fluidState.grid.size) / 2.f;

				fluidSim.applyForces(fluidState, gImpulse, false, dt);
			}
		}
		ImGui::Separator();
		ImGui::TextDisabled("Debug texture display");
		ImGui::Checkbox("Display debug texture", &simControls.displayDebugTexture);
		if (ImGui::SliderInt("Debug texture Z slice", &simControls.debugTextureSlice, 0, fluidState.grid.size.z - 1))
			debugDrawProgram.uniform("uUVZ", (simControls.debugTextureSlice + 0.5f) / fluidState.grid.size.z);
		ImGui::Combo("Display which", &simControls.whichDebugTexture, "Velocity X\0Velocity Y\0Velocity Z\0Pressure\0Velocity divergence\0Divergence zero check\0Boundaries\0");
		if (ImGui::Combo("Display when", &simControls.whenDebugTexture, "Start of frame\0After advection\0After diffusion\0After divergence\0After pressure computation\0After projection\0"))
			fluidSim.modifyHookStage(simControls.debugTextureLambdaHookId, static_cast<FluidSimHookStage>(simControls.whenDebugTexture));

		if (ImGui::DragFloat("Debug color scale", &simControls.colorScale, 0.001f, 0.0f, 1.f))
			debugDrawProgram.uniform("uColorScale", simControls.colorScale);
	}
	ImGui::End();
}

void displayTexture(Empty::gl::ShaderProgram& debugDrawProgram, FluidState& fluidState, int whichDebugTexture)
{
	Context& context = Context::get();

	Empty::gl::TextureInfo texture;
	bool intTexture = false;

	switch (whichDebugTexture)
	{
	case 0:
		texture = fluidState.velocityX.getInput();
		break;
	case 1:
		texture = fluidState.velocityY.getInput();
		break;
	case 2:
		texture = fluidState.velocityZ.getInput();
		break;
	case 3:
		texture = fluidState.pressure.getInput();
		break;
	case 4:
		texture = fluidState.divergenceTex;
		break;
	case 5:
		texture = fluidState.divergenceCheckTex;
		break;
	case 6:
		texture = fluidState.boundariesTex;
		intTexture = true;
		break;
	case 7:
		texture = fluidState.inkDensity.getInput();
		break;
	default:
		FATAL("invalid requested debug texture");
	}

	if (intTexture)
		debugDrawProgram.registerTexture("uIntTexture", texture);
	else
		debugDrawProgram.registerTexture("uTexture", texture);

	debugDrawProgram.uniform("uUseIntTexture", intTexture);

	context.setShaderProgram(debugDrawProgram);
	context.drawArrays(Empty::gl::PrimitiveType::Triangles, 0, 6);
}
