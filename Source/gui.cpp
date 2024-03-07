#include "gui.h"

#include <GLFW/glfw3.h>
#include <imgui.h>

void doGUI(FluidSim& fluidSim, FluidState& fluidState, SimulationControls& simControls, Empty::gl::ShaderProgram& debugDrawProgram, float dt)
{
	if (ImGui::Begin("Fluid simulation", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
	{
		ImGui::TextDisabled("%.1f fps", 1.f / dt);
		if (ImGui::Checkbox("VSync", &simControls.capFPS))
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
		ImGui::TextDisabled("Jacobi solver parameters");
		ImGui::DragInt("Diffusion Jacobi steps", &fluidSim.diffusionJacobiSteps, 1, 1);
		ImGui::DragInt("Pressure Jacobi steps", &fluidSim.pressureJacobiSteps, 1, 1);
		ImGui::Separator();
		ImGui::TextDisabled("Fluid physics properties");
		ImGui::SliderFloat("Grid cell size (m)", &fluidState.grid.cellSize, 0.0001f, 1.f);
		ImGui::SliderFloat("Density (kg/dm^3)", &fluidState.physics.density, 0.0001f, 1.f);
		ImGui::SliderFloat("Kinematic viscosity (m^2/s)", &fluidState.physics.kinematicViscosity, 0.f, 0.005f, "%.5f");
		ImGui::Separator();
		ImGui::TextDisabled("Mouse click impulse parameters");
		ImGui::DragFloat("Force scale", &simControls.forceScale, 0.1f, 0.f, 20.f);
		ImGui::DragFloat("Force radius", &simControls.impulse.radius, 1.f, 1.f);
		ImGui::DragFloat("Ink injection", &simControls.impulse.inkAmount, 0.5f, 0.f, 50.f);
		ImGui::Separator();
		ImGui::TextDisabled("Other parameters");
		if (ImGui::DragFloat2("Exterior velocity", fluidState.exteriorVelocity))
		{
			fluidState.velocityX.setBoundaryValue(fluidState.exteriorVelocity.xxxx());
			fluidState.velocityY.setBoundaryValue(fluidState.exteriorVelocity.yyyy());
		}
		ImGui::Separator();
		ImGui::TextDisabled("Debug texture display");
		ImGui::Checkbox("Display debug texture", &simControls.displayDebugTexture);
		ImGui::Combo("Display which", &simControls.whichDebugTexture, "Velocity X\0Velocity Y\0Pressure\0Velocity divergence\0Boundaries\0");
		if (ImGui::Combo("Display when", &simControls.whenDebugTexture, "Start of frame\0After advection\0After diffusion\0After divergence\0After pressure computation\0After projection\0"))
			fluidSim.modifyHookStage(simControls.debugTextureLambdaHookId, static_cast<FluidSimHookStage>(simControls.whenDebugTexture));

		if (ImGui::DragFloat("Debug color scale", &simControls.colorScale, 0.001f, 0.0f, 1.f))
			debugDrawProgram.uniform("uColorScale", simControls.colorScale);
	}
	ImGui::End();
}