#include <Empty/gl/Buffer.h>
#include <Empty/gl/VertexArray.h>
#include <Empty/gl/ShaderProgram.hpp>
#include <Empty/math/funcs.h>
#include <Empty/utils/macros.h>

#define IM_VEC2_CLASS_EXTRA                                                   \
        constexpr ImVec2(const Empty::math::vec2& f) : x(f.x), y(f.y) {}      \
        operator Empty::math::vec2() const { return Empty::math::vec2(x,y); }

#include "Context.h"

using namespace Empty::gl;

Context Context::_instance;

// *********************************
// Types related to fluid simulation
// *********************************

constexpr TextureFormat gpuScalarFieldFormat = TextureFormat::Red32f;
using GPUScalarField = Texture<TextureTarget::Texture2D, gpuScalarFieldFormat>;

struct BufferedScalarField
{
	BufferedScalarField(const std::string& name, Empty::math::uvec2 size) :
		fields{ { name + " 1" } , { name + " 2"} },
		writingBackBuffer(true)
	{
		for (int i : { 0, 1 })
		{
			fields[i].setStorage(1, size.x, size.y);
			fields[i].template clearLevel<DataFormat::Red, DataType::Float>(0, 0.f);
			fields[i].setParameter<TextureParam::WrapS>(TextureParamValue::ClampToEdge);
			fields[i].setParameter<TextureParam::WrapT>(TextureParamValue::ClampToEdge);
		}
	}

	void clear()
	{
		fields[0].template clearLevel<DataFormat::Red, DataType::Float>(0, 0.f);
		fields[1].template clearLevel<DataFormat::Red, DataType::Float>(0, 0.f);
		writingBackBuffer = true;
	}

	auto& getInput() { return fields[writingBackBuffer ? 0 : 1]; }
	auto& getOutput() { return fields[writingBackBuffer ? 1 : 0]; }

	void swap() { writingBackBuffer = !writingBackBuffer; }

private:
	GPUScalarField fields[2];
	bool writingBackBuffer;
};

struct FluidSimParameters
{
	const Empty::math::uvec2 gridSize;
	float gridCellSize;
	float density;
	float viscosity;
};

struct FluidSimMouseClickImpulse
{
	Empty::math::vec2 position;
	Empty::math::vec2 magnitude;
	float inkAmount;
	float radius;
};

struct FluidRenderParameters
{
	FluidRenderParameters(Empty::math::uvec2 frame, Empty::math::uvec2 gridSize, float gridCellSizeInPx)
	{
		topLeftCorner = (Empty::math::vec2(frame) - gridSize * gridCellSizeInPx) / 2.f;
		this->gridCellSizeInPx = gridCellSizeInPx;
	}
	
	Empty::math::vec2 topLeftCorner;
	float gridCellSizeInPx;
};

struct FluidState
{
	FluidState(Empty::math::uvec2 gridSize, float gridCellSize, float density, float viscosity) :
		parameters{ gridSize, gridCellSize, density, viscosity },
		velocityX{ "Velocity X", gridSize },
		velocityY{ "Velocity Y", gridSize },
		velocityZ{ "Velocity Z", gridSize },
		pressure{ "Pressure", gridSize },
		divergenceTex("Divergence")
	{
		divergenceTex.setStorage(1, gridSize.x, gridSize.y);
	}

	FluidSimParameters parameters;
	BufferedScalarField velocityX;
	BufferedScalarField velocityY;
	BufferedScalarField velocityZ;
	BufferedScalarField pressure;
	GPUScalarField divergenceTex;
};

// ********************************************
// Shared constants related to fluid simulation
// ********************************************

constexpr int allVelocityXBinding = 0;
constexpr int allVelocityYBinding = 1;
constexpr int allVelocityZBinding = 2;
constexpr int allFieldInBinding = 3;
constexpr int allFieldOutBinding = 4;

constexpr int jacobiSourceBinding = 0;

constexpr int forcesInkBinding = 3;

constexpr int projectionPressureBinding = 3;

constexpr int boundaryFieldBinding = 5;

// f(boundary) + f(neighbour) = 0 -> f(boundary) = -f(neighbour)
constexpr float noSlipBoundaryCondition = -1.f;
// f(boundary) - f(neighbour) = 0 -> f(boundary) = f(neighbour)
constexpr float neumannBoundaryCondition = 1.f;
// f(boundary) = 0
constexpr float zeroBoundaryCondition = 0.f;

// ********************************************
// Functions for advancing the fluid simulation
// ********************************************

void performBoundaryConditions(ShaderProgram& boundaryProgram, GPUScalarField& field, const Empty::math::uvec2& gridSize, float boundaryCondition)
{
	constexpr int boundaryWorkGroupX = 32;

	Context& context = Context::get();

	boundaryProgram.uniform("uBoundaryCondition", boundaryCondition);
	boundaryProgram.registerTexture("uField", field, false);
	context.bind(field.getLevel(0), boundaryFieldBinding, AccessPolicy::ReadWrite, gpuScalarFieldFormat);

	context.setShaderProgram(boundaryProgram);
	context.dispatchCompute(gridSize.x / boundaryWorkGroupX, 4, 1);
}

void setupAdvection(ShaderProgram& advectionProgram, FluidState& fluidState, float dt)
{
	Context& context = Context::get();

	auto& velocityXTex = fluidState.velocityX.getInput();
	auto& velocityYTex = fluidState.velocityY.getInput();
	// auto& velocityZTex = fluidState.velocityZ.getInput();

	advectionProgram.uniform("udt", dt);
	advectionProgram.uniform("udx", fluidState.parameters.gridCellSize);
	advectionProgram.registerTexture("uVelocityX", velocityXTex, false);
	advectionProgram.registerTexture("uVelocityY", velocityYTex, false);
	// advectionProgram.registerTexture("uVelocityZ", velocityZTex, false);
	context.bind(velocityXTex.getLevel(0), allVelocityXBinding, AccessPolicy::ReadOnly, gpuScalarFieldFormat);
	context.bind(velocityYTex.getLevel(0), allVelocityYBinding, AccessPolicy::ReadOnly, gpuScalarFieldFormat);
	// context.bind(velocityZTex.getLevel(0), allVelocityZBinding, AccessPolicy::ReadOnly, gpuScalarFieldFormat);

	context.setShaderProgram(advectionProgram);
}

void advectField(ShaderProgram& advectionProgram, BufferedScalarField& field, Empty::math::uvec2 gridSize)
{
	constexpr int advectionWorkGroupX = 32;
	constexpr int advectionWorkGroupY = 32;

	Context& context = Context::get();

	GPUScalarField& fieldIn = field.getInput();
	GPUScalarField& fieldOut = field.getOutput();

	// Input field is exposed as a texture to benefit from bilinear filtering
	advectionProgram.registerTexture("uFieldIn", fieldIn, false);
	advectionProgram.registerTexture("uFieldOut", fieldOut, false);

	context.bind(fieldIn, allFieldInBinding);
	context.bind(fieldOut.getLevel(0), allFieldOutBinding, AccessPolicy::WriteOnly, gpuScalarFieldFormat);

	using namespace Empty::utils;

	context.dispatchCompute(gridSize.x / advectionWorkGroupX, gridSize.y / advectionWorkGroupY, 1);

	field.swap();
}

void performJacobiIterations(ShaderProgram& jacobiProgram, ShaderProgram& boundaryProgram, GPUScalarField& fieldSource, GPUScalarField& fieldIn, GPUScalarField& fieldOut, Empty::math::uvec2 gridSize, int jacobiSteps, float boundaryCondition)
{
	constexpr int jacobiWorkGroupX = 32;
	constexpr int jacobiWorkGroupY = 32;

	assert(jacobiSteps > 0);

	static std::unique_ptr<GPUScalarField> workingField;
	static std::unique_ptr<Buffer> diffusionDispatchArgs;

	if (!workingField)
	{
		workingField = std::make_unique<GPUScalarField>("Jacobi working field");
		workingField->setStorage(1, gridSize.x, gridSize.y);
	}

	if (!diffusionDispatchArgs)
	{
		diffusionDispatchArgs = std::make_unique<Buffer>("Diffusion indirect dispatch args");
		diffusionDispatchArgs->setStorage(sizeof(Empty::math::uvec3), BufferUsage::StreamDraw);
	}
	{
		Empty::math::uvec3 dispatch(gridSize.x / jacobiWorkGroupX, gridSize.y / jacobiWorkGroupY, 1);
		diffusionDispatchArgs->uploadData(0, sizeof(dispatch), dispatch);
	}

	Context& context = Context::get();
	context.bind(*diffusionDispatchArgs, BufferTarget::DispatchIndirect);

	jacobiProgram.registerTexture("uFieldSource", fieldSource, false);
	context.bind(fieldSource.getLevel(0), jacobiSourceBinding, AccessPolicy::ReadOnly, gpuScalarFieldFormat);

	// Alternate writes between the working texture and the output field so we write to the output
	// field last. The first step uses the actual input field as input, the other steps alternate between
	// working field and output field.
	bool writeToWorkingField = (jacobiSteps & 1) == 0;
	GPUScalarField* iterationFieldIn = &fieldIn;
	GPUScalarField* iterationFieldOut = writeToWorkingField ? workingField.get() : &fieldOut;

	for (int iteration = 0; iteration < jacobiSteps; ++iteration)
	{
		jacobiProgram.registerTexture("uFieldIn", *iterationFieldIn, false);
		jacobiProgram.registerTexture("uFieldOut", *iterationFieldOut, false);
		context.bind(iterationFieldIn->getLevel(0), allFieldInBinding, AccessPolicy::ReadOnly, gpuScalarFieldFormat);
		context.bind(iterationFieldOut->getLevel(0), allFieldOutBinding, AccessPolicy::WriteOnly, gpuScalarFieldFormat);

		// The boundary program gets set every iteration so the jacobi one also needs to be set
		context.setShaderProgram(jacobiProgram);
		context.dispatchComputeIndirect();

		context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);

		performBoundaryConditions(boundaryProgram, *iterationFieldOut, gridSize, boundaryCondition);

		// I could simply swap iterationFieldIn and iterationFieldOut but I can't overwrite the actual fieldIn,
		// which is used as iterationFieldIn for the first iteration.
		writeToWorkingField = !writeToWorkingField;
		iterationFieldIn = iterationFieldOut;
		iterationFieldOut = writeToWorkingField ? workingField.get() : &fieldOut;
	}
}

void performDiffusion(ShaderProgram& jacobiProgram, ShaderProgram& boundaryProgram, FluidState& fluidState, float dt, int jacobiSteps)
{
	const auto& params = fluidState.parameters;

	// Upload solver parameters
	{
		float alpha = params.gridCellSize * params.gridCellSize / (params.viscosity * dt);
		float oneOverBeta = 1.f / (alpha + 4.f);
		jacobiProgram.uniform("uAlpha", alpha);
		jacobiProgram.uniform("uOneOverBeta", oneOverBeta);
	}

	for (auto* fields : { &fluidState.velocityX, &fluidState.velocityY /*, fluidState.velocityZ */ })
	{
		GPUScalarField& fieldIn = fields->getInput();
		GPUScalarField& fieldOut = fields->getOutput();

		performJacobiIterations(jacobiProgram, boundaryProgram, fieldIn, fieldIn, fieldOut, params.gridSize, jacobiSteps, noSlipBoundaryCondition);

		fields->swap();
	}
}

void performForcesApplication(ShaderProgram& forcesProgram, FluidState& fluidState, BufferedScalarField& inkDensity, FluidSimMouseClickImpulse& impulse, float dt)
{
	constexpr int forcesWorkGroupX = 32;
	constexpr int forcesWorkGroupY = 32;

	const auto& params = fluidState.parameters;

	Context& context = Context::get();

	auto& velocityXTex = fluidState.velocityX.getInput();
	auto& velocityYTex = fluidState.velocityY.getInput();
	// auto& velocityZTex = fluidState.velocityZ.getInput();
	auto& inkTex = inkDensity.getInput();

	forcesProgram.uniform("udt", dt);
	forcesProgram.uniform("uMouseClick", impulse.position);
	forcesProgram.uniform("uForceMagnitude", impulse.magnitude);
	forcesProgram.uniform("uForceRadius", impulse.radius);
	forcesProgram.uniform("uInkAmount", impulse.inkAmount);
	forcesProgram.registerTexture("uVelocityX", velocityXTex, false);
	forcesProgram.registerTexture("uVelocityY", velocityYTex, false);
	// forcesProgram.registerTexture("uVelocityZ", velocityZTex, false);
	forcesProgram.registerTexture("uInkDensity", inkTex, false);
	context.bind(velocityXTex.getLevel(0), allVelocityXBinding, AccessPolicy::ReadWrite, gpuScalarFieldFormat);
	context.bind(velocityYTex.getLevel(0), allVelocityYBinding, AccessPolicy::ReadWrite, gpuScalarFieldFormat);
	// context.bind(velocityZTex.getLevel(0), allVelocityZBinding, AccessPolicy::ReadWrite, gpuScalarFieldFormat);
	context.bind(inkTex.getLevel(0), forcesInkBinding, AccessPolicy::ReadWrite, gpuScalarFieldFormat);

	context.setShaderProgram(forcesProgram);
	context.dispatchCompute(params.gridSize.x / forcesWorkGroupX, params.gridSize.y / forcesWorkGroupY, 1);

	// Don't swap textures since we read from and write to the same textures
}

void performDivergenceComputation(ShaderProgram& divergenceProgram, FluidState& fluidState)
{
	constexpr int divergenceWorkGroupX = 32;
	constexpr int divergenceWorkGroupY = 32;

	const auto& params = fluidState.parameters;

	Context& context = Context::get();

	auto& velocityXTex = fluidState.velocityX.getInput();
	auto& velocityYTex = fluidState.velocityY.getInput();
	// auto& velocityZTex = fluidState.velocityZ.getInput();

	divergenceProgram.uniform("uHalfOneOverDx", 1.f / (2.f * params.gridCellSize));
	divergenceProgram.registerTexture("uVelocityX", velocityXTex, false);
	divergenceProgram.registerTexture("uVelocityY", velocityYTex, false);
	divergenceProgram.registerTexture("uFieldOut", fluidState.divergenceTex, false);
	// divergenceProgram.registerTexture("uVelocityZ", velocityZTex, false);
	context.bind(velocityXTex.getLevel(0), allVelocityXBinding, AccessPolicy::ReadOnly, gpuScalarFieldFormat);
	context.bind(velocityYTex.getLevel(0), allVelocityYBinding, AccessPolicy::ReadOnly, gpuScalarFieldFormat);
	// context.bind(velocityZTex.getLevel(0), allVelocityZBinding, AccessPolicy::ReadOnly, gpuScalarFieldFormat);
	context.bind(fluidState.divergenceTex.getLevel(0), allFieldOutBinding, AccessPolicy::WriteOnly, gpuScalarFieldFormat);

	context.setShaderProgram(divergenceProgram);
	context.dispatchCompute(params.gridSize.x / divergenceWorkGroupX, params.gridSize.y / divergenceWorkGroupY, 1);
}

void performPressureComputation(ShaderProgram& jacobiProgram, ShaderProgram& boundaryProgram, FluidState& fluidState, int jacobiSteps)
{
	const auto& params = fluidState.parameters;

	// Upload solver parameters
	{
		float alpha = -params.gridCellSize * params.gridCellSize * params.density;
		float oneOverBeta = 1.f / 4.f;
		jacobiProgram.uniform("uAlpha", alpha);
		jacobiProgram.uniform("uOneOverBeta", oneOverBeta);
	}

	GPUScalarField& fieldIn = fluidState.pressure.getInput();
	GPUScalarField& fieldOut = fluidState.pressure.getOutput();

	fieldIn.template clearLevel<DataFormat::Red, DataType::Float>(0, 0.f);

	performJacobiIterations(jacobiProgram, boundaryProgram, fluidState.divergenceTex, fieldIn, fieldOut, params.gridSize, jacobiSteps, neumannBoundaryCondition);

	fluidState.pressure.swap();
}

void performProjection(ShaderProgram& projectionProgram, FluidState& fluidState)
{
	constexpr int projectionWorkGroupX = 32;
	constexpr int projectionWorkGroupY = 32;

	const auto& params = fluidState.parameters;

	Context& context = Context::get();

	auto& velocityXTex = fluidState.velocityX.getInput();
	auto& velocityYTex = fluidState.velocityY.getInput();
	// auto& velocityZTex = fluidState.velocityZ.getInput();
	auto& pressureTex = fluidState.pressure.getInput();

	projectionProgram.uniform("uHalfOneOverDx", 1.f / (2.f * params.gridCellSize));
	projectionProgram.registerTexture("uVelocityX", velocityXTex, false);
	projectionProgram.registerTexture("uVelocityY", velocityYTex, false);
	// projectionProgram.registerTexture("uVelocityZ", velocityZTex, false);
	projectionProgram.registerTexture("uPressure", pressureTex, false);
	context.bind(velocityXTex.getLevel(0), allVelocityXBinding, AccessPolicy::ReadWrite, gpuScalarFieldFormat);
	context.bind(velocityYTex.getLevel(0), allVelocityYBinding, AccessPolicy::ReadWrite, gpuScalarFieldFormat);
	// context.bind(velocityZTex.getLevel(0), allVelocityZBinding, AccessPolicy::ReadWrite, gpuScalarFieldFormat);
	context.bind(pressureTex.getLevel(0), projectionPressureBinding, AccessPolicy::ReadOnly, gpuScalarFieldFormat);

	context.setShaderProgram(projectionProgram);
	context.dispatchCompute(params.gridSize.x / projectionWorkGroupX, params.gridSize.y / projectionWorkGroupY, 1);

	// Don't swap textures since we read from and write to the same textures
}

// ***************
// Other functions
// ***************

void debugCallback(DebugMessageSource source, DebugMessageType type, DebugMessageSeverity severity, int id, const std::string& text, const void* userData)
{
	std::cout << Empty::utils::name(source) << " (" << Empty::utils::name(type) << ", " << Empty::utils::name(severity) << "): " << text << std::endl;
}

GPUScalarField& selectDebugTexture(FluidState& fluidState, int whichDebugTexture)
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
		assert(0);
	}
}

void displayTexture(ShaderProgram& debugDrawProgram, GPUScalarField& texture)
{
	Context& context = Context::get();

	debugDrawProgram.registerTexture("uTexture", texture);

	auto temp = texture.getParameter<TextureParam::MinFilter>();
	texture.setParameter<TextureParam::MinFilter>(TextureParamValue::FilterNearest);

	context.setShaderProgram(debugDrawProgram);
	context.drawArrays(PrimitiveType::Triangles, 0, 6);

	texture.setParameter<TextureParam::MinFilter>(temp);
}

int main(int argc, char* argv[])
{
	Context& context = Context::get();

	if (!context.init("Fluid simulation tests", 1920, 1080))
	{
		TRACE("Didn't work");
		return 1;
	}

	Texture<TextureTarget::Texture2D, TextureFormat::RGB> test("Test texture");

	context.debugMessageControl(DebugMessageSource::DontCare, DebugMessageType::DontCare, DebugMessageSeverity::DontCare, false);
	context.debugMessageControl(DebugMessageSource::DontCare, DebugMessageType::Error, DebugMessageSeverity::DontCare, true);
	context.debugMessageCallback(debugCallback, nullptr);

	// Fluid setup
	FluidState fluidState(Empty::math::uvec2(256, 256), 0.8f, 1.f, 0.0025f);
	FluidRenderParameters fluidRenderParameters{ Empty::math::uvec2(context.frameWidth, context.frameHeight), fluidState.parameters.gridSize, 4.f };

	// Fun things to advect by the fluid
	BufferedScalarField inkDensity("Ink density", fluidState.parameters.gridSize);

	ShaderProgram advectionProgram("Advection program");
	advectionProgram.attachFile(ShaderType::Compute, "shaders/advection.glsl", "Compute advection");
	advectionProgram.build();

	ShaderProgram jacobiProgram("Jacobi program");
	jacobiProgram.attachFile(ShaderType::Compute, "shaders/jacobi.glsl", "Compute jacobi iteration");
	jacobiProgram.build();

	ShaderProgram forcesProgram("Force application program");
	forcesProgram.attachFile(ShaderType::Compute, "shaders/forces.glsl", "Compute force application");
	forcesProgram.build();

	ShaderProgram divergenceProgram("Divergence program");
	divergenceProgram.attachFile(ShaderType::Compute, "shaders/divergence.glsl", "Compute divergence");
	divergenceProgram.build();

	ShaderProgram projectionProgram("Projection program");
	projectionProgram.attachFile(ShaderType::Compute, "shaders/projection.glsl", "Compute projection");
	projectionProgram.build();

	ShaderProgram boundaryProgram("Boundary conditions program");
	boundaryProgram.attachFile(ShaderType::Compute, "shaders/boundary.glsl", "Compute boundary conditions");
	boundaryProgram.build();

	double then = glfwGetTime();
	Empty::math::vec2 mouseThen = ImGui::GetMousePos();

	int diffusionJacobiSteps = 100;
	int pressureJacobiSteps = 100;
	FluidSimMouseClickImpulse impulse;
	impulse.radius = 240.f;
	impulse.inkAmount = 7.f;
	float forceScale = 500.f;

	// Simulation control variables
	bool pauseSimulation = false;
	bool runOneStep = false;
	bool runAdvection = true;
	bool runDiffusion = true;
	bool runDivergence = true;
	bool runPressure = true;
	bool runProjection = true;

	bool displayDebugTexture = false;
	int whichDebugTexture = 0;
	int whenDebugTexture = 0;
	float colorScale = 1.f;

	// Debug texture draw
	VertexArray debugVAO("Debug VAO");
	context.bind(debugVAO);
	ShaderProgram debugDrawProgram("Debug draw program");
	debugDrawProgram.attachFile(ShaderType::Vertex, "shaders/debug/vertex.glsl", "Debug draw vertex");
	debugDrawProgram.attachFile(ShaderType::Fragment, "shaders/debug/fragment.glsl", "Debug draw fragment");
	debugDrawProgram.build();
	debugDrawProgram.uniform("uTextureSizeOverScreenSize", Empty::math::vec2(fluidState.parameters.gridSize) * fluidRenderParameters.gridCellSizeInPx / Empty::math::vec2(context.frameWidth, context.frameHeight));
	debugDrawProgram.uniform("uColorScale", colorScale);

	while (!glfwWindowShouldClose(context.window))
	{
		context.newFrame();

		double now = glfwGetTime();
		Empty::math::vec2 mouseNow = ImGui::GetMousePos();
		float dt = static_cast<float>(now - then);

		if (ImGui::Begin("Fluid simulation", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
		{
			ImGui::TextDisabled("%.1f fps", 1.f / dt);
			ImGui::Checkbox("Pause simulation (P)", &pauseSimulation);
			if (ImGui::IsKeyPressed(ImGuiKey_P))
				pauseSimulation = !pauseSimulation;
			if (ImGui::Button("Run one step (R)"))
				runOneStep = true;
			if (ImGui::IsKeyPressed(ImGuiKey_R) && !ImGui::GetIO().WantCaptureKeyboard)
				runOneStep = true;
			if (ImGui::Button("Reset"))
			{
				fluidState.velocityX.clear();
				fluidState.velocityY.clear();
				fluidState.velocityZ.clear();
				fluidState.pressure.clear();
				fluidState.divergenceTex.template clearLevel<DataFormat::Red, DataType::Float>(0, 0.f);
				inkDensity.clear();
			}

			ImGui::Indent();
			ImGui::Checkbox("Advection", &runAdvection);
			ImGui::Checkbox("Diffusion", &runDiffusion);
			ImGui::Checkbox("Divergence", &runDivergence);
			ImGui::Checkbox("Pressure", &runPressure);
			ImGui::Checkbox("Projection", &runProjection);
			ImGui::Unindent();

			ImGui::Separator();
			ImGui::TextDisabled("Jacobi solver parameters");
			ImGui::DragInt("Diffusion Jacobi steps", &diffusionJacobiSteps, 1, 1);
			ImGui::DragInt("Pressure Jacobi steps", &pressureJacobiSteps, 1, 1);
			ImGui::Separator();
			ImGui::TextDisabled("Fluid physics properties");
			ImGui::SliderFloat("Grid cell size (m)", &fluidState.parameters.gridCellSize, 0.0001f, 1.f);
			ImGui::SliderFloat("Density (kg/dm^3)", &fluidState.parameters.density, 0.0001f, 1.f);
			ImGui::SliderFloat("Kinematic viscosity (m^2/s)", &fluidState.parameters.viscosity, 0.f, 0.005f, "%.5f");
			ImGui::Separator();
			ImGui::TextDisabled("Mouse click impulse parameters");
			ImGui::DragFloat("Force scale", &forceScale);
			ImGui::DragFloat("Force radius", &impulse.radius, 1.f, 0.001f);
			ImGui::DragFloat("Ink injection", &impulse.inkAmount);
			ImGui::Separator();
			ImGui::TextDisabled("Debug texture display");
			ImGui::Checkbox("Display debug texture", &displayDebugTexture);
			ImGui::Combo("Display which", &whichDebugTexture, "Velocity X\0Velocity Y\0Velocity Z\0Pressure\0Velocity divergence\0");
			ImGui::Combo("Display when", &whenDebugTexture, "Start of frame\0After advection\0After diffusion\0After impulse\0After divergence\0After pressure computation\0After projection\0\0");
			if (ImGui::DragFloat("Debug color scale", &colorScale, 0.001f, 0.0f, 1.f))
				debugDrawProgram.uniform("uColorScale", colorScale);
		}
		ImGui::End();

		/// Advance simulation

		if (!pauseSimulation || runOneStep)
		{
			if (displayDebugTexture && whenDebugTexture == 0)
				displayTexture(debugDrawProgram, selectDebugTexture(fluidState, whichDebugTexture));

			// Advect fields along the velocity
			if (runAdvection)
			{
				setupAdvection(advectionProgram, fluidState, dt);
				advectField(advectionProgram, fluidState.velocityX, fluidState.parameters.gridSize);
				advectField(advectionProgram, fluidState.velocityY, fluidState.parameters.gridSize);
				// advectField(advectionProgram, fluidState.velocityZ, fluidState.parameters.gridSize);
				advectField(advectionProgram, inkDensity, fluidState.parameters.gridSize);

				context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);

				performBoundaryConditions(boundaryProgram, fluidState.velocityX.getInput(), fluidState.parameters.gridSize, noSlipBoundaryCondition);
				performBoundaryConditions(boundaryProgram, fluidState.velocityY.getInput(), fluidState.parameters.gridSize, noSlipBoundaryCondition);
				// performBoundaryConditions(boundaryProgram, fluidState.velocityZ.getInput(), fluidState.parameters.gridSize, noSlipBoundaryCondition);
				performBoundaryConditions(boundaryProgram, inkDensity.getInput(), fluidState.parameters.gridSize, zeroBoundaryCondition);
			}

			if (displayDebugTexture && whenDebugTexture == 1)
				displayTexture(debugDrawProgram, selectDebugTexture(fluidState, whichDebugTexture));

			// Compute viscous diffusion
			if (runDiffusion)
				performDiffusion(jacobiProgram, boundaryProgram, fluidState, dt, diffusionJacobiSteps);

			if (displayDebugTexture && whenDebugTexture == 2)
				displayTexture(debugDrawProgram, selectDebugTexture(fluidState, whichDebugTexture));

			// Apply an impulse and inject ink when the left mouse button is clicked
			if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && !ImGui::GetIO().WantCaptureMouse)
			{
				impulse.magnitude = (mouseNow - mouseThen) * forceScale;
				impulse.magnitude.y *= -1;
				impulse.position = mouseNow - fluidRenderParameters.topLeftCorner;
				impulse.position /= fluidRenderParameters.gridCellSizeInPx;
				impulse.position.y = fluidState.parameters.gridSize.y - impulse.position.y;

				performForcesApplication(forcesProgram, fluidState, inkDensity, impulse, dt);

				context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);

				performBoundaryConditions(boundaryProgram, fluidState.velocityX.getInput(), fluidState.parameters.gridSize, noSlipBoundaryCondition);
				performBoundaryConditions(boundaryProgram, fluidState.velocityY.getInput(), fluidState.parameters.gridSize, noSlipBoundaryCondition);
				performBoundaryConditions(boundaryProgram, inkDensity.getInput(), fluidState.parameters.gridSize, zeroBoundaryCondition);
			}
			// Only apply the force when the right mouse button is clicked
			else if (ImGui::IsMouseDown(ImGuiMouseButton_Right) && !ImGui::GetIO().WantCaptureMouse)
			{
				impulse.magnitude = (mouseNow - mouseThen) * forceScale * fluidState.parameters.gridCellSize;
				impulse.magnitude.y *= -1;
				impulse.position = mouseNow - fluidRenderParameters.topLeftCorner;
				impulse.position /= fluidRenderParameters.gridCellSizeInPx;
				impulse.position.y = fluidState.parameters.gridSize.y - impulse.position.y;
				
				float temp = impulse.inkAmount;
				impulse.inkAmount = 0;

				performForcesApplication(forcesProgram, fluidState, inkDensity, impulse, dt);

				context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);

				performBoundaryConditions(boundaryProgram, fluidState.velocityX.getInput(), fluidState.parameters.gridSize, noSlipBoundaryCondition);
				performBoundaryConditions(boundaryProgram, fluidState.velocityY.getInput(), fluidState.parameters.gridSize, noSlipBoundaryCondition);
				performBoundaryConditions(boundaryProgram, inkDensity.getInput(), fluidState.parameters.gridSize, zeroBoundaryCondition);

				impulse.inkAmount = temp;
			}

			if (displayDebugTexture && whenDebugTexture == 3)
				displayTexture(debugDrawProgram, selectDebugTexture(fluidState, whichDebugTexture));

			// Compute divergence for pressure computation
			if (runDivergence)
				performDivergenceComputation(divergenceProgram, fluidState);

			if (displayDebugTexture && whenDebugTexture == 4)
				displayTexture(debugDrawProgram, selectDebugTexture(fluidState, whichDebugTexture));

			// Compute pressure field
			if (runPressure)
				performPressureComputation(jacobiProgram, boundaryProgram, fluidState, pressureJacobiSteps);

			if (displayDebugTexture && whenDebugTexture == 5)
				displayTexture(debugDrawProgram, selectDebugTexture(fluidState, whichDebugTexture));

			// Project over divergence-free fields by subtracting the pressure gradient from the velocity
			if (runProjection)
			{
				performProjection(projectionProgram, fluidState);

				context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);

				performBoundaryConditions(boundaryProgram, fluidState.velocityX.getInput(), fluidState.parameters.gridSize, noSlipBoundaryCondition);
				performBoundaryConditions(boundaryProgram, fluidState.velocityY.getInput(), fluidState.parameters.gridSize, noSlipBoundaryCondition);
			}

			if (displayDebugTexture && whenDebugTexture == 6)
				displayTexture(debugDrawProgram, selectDebugTexture(fluidState, whichDebugTexture));

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
			auto* drawList = ImGui::GetBackgroundDrawList();

			drawList->AddRect(fluidRenderParameters.topLeftCorner - Empty::math::vec2(1, 1),
				fluidRenderParameters.topLeftCorner + Empty::math::vec2(fluidState.parameters.gridSize) * fluidRenderParameters.gridCellSizeInPx + Empty::math::vec2(2, 2),
				ImColor(0, 255, 0));

			if (!displayDebugTexture)
				displayTexture(debugDrawProgram, inkDensity.getInput());
		}

		// ImGui::ShowDemoWindow();

		context.swap();

		then = now;
		mouseThen = mouseNow;
	}

	context.terminate();

	return 0;
}
