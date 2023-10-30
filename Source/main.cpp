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

constexpr DataFormat gpuScalarDataFormat = DataFormat::Red;
constexpr DataFormat gpuVectorDataFormat = DataFormat::RG;
using GPUScalarField = Texture<TextureTarget::Texture2D, TextureFormat::Red32f>;
using GPUVectorField = Texture<TextureTarget::Texture2D, TextureFormat::RG32f>;

template <typename F, DataFormat Format>
struct BufferedField
{
	using Field = F;

	BufferedField(const std::string& name, Empty::math::uvec2 size) :
		fields{ { name + " 1" } , { name + " 2"} },
		writingBackBuffer(true)
	{
		for (int i : { 0, 1 })
		{
			fields[i].setStorage(1, size.x, size.y);
			fields[i].template clearLevel<Format, DataType::Float>(0);
			fields[i].setParameter<TextureParam::WrapS>(TextureParamValue::ClampToEdge);
			fields[i].setParameter<TextureParam::WrapT>(TextureParamValue::ClampToEdge);
		}
	}

	void clear()
	{
		fields[0].template clearLevel<Format, DataType::Float>(0);
		fields[1].template clearLevel<Format, DataType::Float>(0);
		writingBackBuffer = true;
	}

	auto& getInput() { return fields[writingBackBuffer ? 0 : 1]; }
	auto& getOutput() { return fields[writingBackBuffer ? 1 : 0]; }

	void swap() { writingBackBuffer = !writingBackBuffer; }

private:
	Field fields[2];
	bool writingBackBuffer;
};

using BufferedScalarField = BufferedField<GPUScalarField, gpuScalarDataFormat>;
using BufferedVectorField = BufferedField<GPUVectorField, gpuVectorDataFormat>;

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
		velocity{ "Velocity", gridSize },
		pressure{ "Pressure", gridSize },
		divergenceTex("Divergence")
	{
		divergenceTex.setStorage(1, gridSize.x, gridSize.y);
	}

	FluidSimParameters parameters;
	BufferedVectorField velocity;
	BufferedScalarField pressure;
	GPUScalarField divergenceTex;
};

// ********************************************
// Shared constants related to fluid simulation
// ********************************************

constexpr int allVelocityBinding = 0;
constexpr int allInkDensityBinding = 1;

constexpr int advectionVelocityOutBinding = 2;
constexpr int advectionInkDensityOutBinding = 3;

constexpr int jacobiSourceBinding = 0;
constexpr int jacobiFieldInBinding = 1;
constexpr int jacobiFieldOutBinding = 2;

constexpr int divergenceOutBinding = 1;

constexpr int projectionPressureBinding = 1;

constexpr int velocityPackingXBinding = 1;
constexpr int velocityPackingYBinding = 2;

// f(boundary) + f(neighbour) = 0 -> f(boundary) = -f(neighbour)
constexpr float noSlipBoundaryCondition = -1.f;
// f(boundary) - f(neighbour) = 0 -> f(boundary) = f(neighbour)
constexpr float neumannBoundaryCondition = 1.f;
// f(boundary) = 0
constexpr float zeroBoundaryCondition = 0.f;

// ********************************************
// Functions for advancing the fluid simulation
// ********************************************

void performAdvection(ShaderProgram& advectionProgram, FluidState& fluidState, BufferedScalarField& inkDensity, float dt, Empty::math::uvec2 gridSize)
{
	constexpr int advectionWorkGroupX = 32;
	constexpr int advectionWorkGroupY = 32;

	Context& context = Context::get();

	auto& params = fluidState.parameters;

	advectionProgram.uniform("udx", params.gridCellSize);
	advectionProgram.uniform("udt", dt);
	{
		Empty::math::vec2 data(1.f / (params.gridSize.x * params.gridCellSize), 1.f / (params.gridSize.y * params.gridCellSize));
		advectionProgram.uniform("uOneOverGridSizeTimesDx", data);
	}

	// Inputs are exposed with samplers to benefit from bilinear filtering

	auto& velocityTex = fluidState.velocity.getInput();
	advectionProgram.registerTexture("uVelocity", velocityTex, false);
	context.bind(velocityTex, allVelocityBinding);

	auto& inkTex = inkDensity.getInput();
	advectionProgram.registerTexture("uInkDensity", inkTex, false);
	context.bind(inkTex, allInkDensityBinding);

	auto& velocityOut = fluidState.velocity.getOutput();
	advectionProgram.registerTexture("uVelocityOut", velocityOut, false);
	context.bind(velocityOut.getLevel(0), advectionVelocityOutBinding, AccessPolicy::WriteOnly, GPUVectorField::Format);

	auto& inkOut = inkDensity.getOutput();
	advectionProgram.registerTexture("uInkDensityOut", inkOut, false);
	context.bind(inkOut.getLevel(0), advectionInkDensityOutBinding, AccessPolicy::WriteOnly, GPUScalarField::Format);

	context.setShaderProgram(advectionProgram);
	context.dispatchCompute(gridSize.x / advectionWorkGroupX, gridSize.y / advectionWorkGroupY, 1);

	fluidState.velocity.swap();
	inkDensity.swap();
}

void performJacobiIterations(ShaderProgram& jacobiProgram, GPUScalarField& fieldSource, GPUScalarField& fieldIn, GPUScalarField& fieldOut, Empty::math::uvec2 gridSize, int jacobiSteps)
{
	constexpr int jacobiWorkGroupX = 32;
	constexpr int jacobiWorkGroupY = 32;

	assert(jacobiSteps > 0);

	static std::unique_ptr<GPUScalarField> workingField;
	static std::unique_ptr<Buffer> jacobiDispatchArgs;

	if (!workingField)
	{
		workingField = std::make_unique<GPUScalarField>("Jacobi working field");
		workingField->setStorage(1, gridSize.x, gridSize.y);
	}

	if (!jacobiDispatchArgs)
	{
		jacobiDispatchArgs = std::make_unique<Buffer>("Jacobi indirect dispatch args");
		jacobiDispatchArgs->setStorage(sizeof(Empty::math::uvec3), BufferUsage::StreamDraw);
	}
	{
		Empty::math::uvec3 dispatch(gridSize.x / jacobiWorkGroupX, gridSize.y / jacobiWorkGroupY, 1);
		jacobiDispatchArgs->uploadData(0, sizeof(dispatch), dispatch);
	}

	Context& context = Context::get();
	context.bind(*jacobiDispatchArgs, BufferTarget::DispatchIndirect);

	jacobiProgram.registerTexture("uFieldSource", fieldSource, false);
	context.bind(fieldSource.getLevel(0), jacobiSourceBinding, AccessPolicy::ReadOnly, GPUScalarField::Format);

	// Alternate writes between the working texture and the output field so we write to the output
	// field last. The first step uses the actual input field as input, the other steps alternate between
	// working field and output field.
	bool writeToWorkingField = (jacobiSteps & 1) == 0;
	GPUScalarField* iterationFieldIn = &fieldIn;
	GPUScalarField* iterationFieldOut = writeToWorkingField ? workingField.get() : &fieldOut;

	context.setShaderProgram(jacobiProgram);

	for (int iteration = 0; iteration < jacobiSteps; ++iteration)
	{
		jacobiProgram.registerTexture("uFieldIn", *iterationFieldIn, false);
		jacobiProgram.registerTexture("uFieldOut", *iterationFieldOut, false);
		context.bind(iterationFieldIn->getLevel(0), jacobiFieldInBinding, AccessPolicy::ReadOnly, GPUScalarField::Format);
		context.bind(iterationFieldOut->getLevel(0), jacobiFieldOutBinding, AccessPolicy::WriteOnly, GPUScalarField::Format);

		context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
		context.dispatchComputeIndirect();

		// I could simply swap iterationFieldIn and iterationFieldOut but I can't overwrite the actual fieldIn,
		// which is used as iterationFieldIn for the first iteration.
		writeToWorkingField = !writeToWorkingField;
		iterationFieldIn = iterationFieldOut;
		iterationFieldOut = writeToWorkingField ? workingField.get() : &fieldOut;
	}
}

void performDiffusion(ShaderProgram& jacobiProgram, ShaderProgram& velocityUnpackProgram, ShaderProgram& velocityPackProgram,
	FluidState& fluidState, float dt, int jacobiSteps)
{
	constexpr int velocityPackingWorkGroupX = 32;
	constexpr int velocityPackingWorkGroupY = 32;

	static std::unique_ptr<BufferedScalarField> velocityX;
	static std::unique_ptr<BufferedScalarField> velocityY;

	const auto& params = fluidState.parameters;

	if (!velocityX)
	{
		velocityX = std::make_unique<BufferedScalarField>("Unpacked velocity X", params.gridSize);
		velocityY = std::make_unique<BufferedScalarField>("Unpacked velocity Y", params.gridSize);
	}

	Context& context = Context::get();

	// Upload solver parameters
	{
		float alpha = params.gridCellSize * params.gridCellSize / (params.viscosity * dt);
		float oneOverBeta = 1.f / (alpha + 4.f);
		jacobiProgram.uniform("uAlpha", alpha);
		jacobiProgram.uniform("uOneOverBeta", oneOverBeta);
		jacobiProgram.uniform("uBoundaryCondition", noSlipBoundaryCondition);
	}

	// Unpack velocity field
	{
		auto& velocityTex = fluidState.velocity.getInput();
		auto& velocityXTex = velocityX->getOutput();
		auto& velocityYTex = velocityY->getOutput();

		velocityUnpackProgram.registerTexture("uVelocityIn", velocityTex, false);
		velocityUnpackProgram.registerTexture("uVelocityXOut", velocityXTex, false);
		velocityUnpackProgram.registerTexture("uVelocityYOut", velocityYTex, false);
		context.bind(velocityTex.getLevel(0), allVelocityBinding, AccessPolicy::ReadOnly, GPUVectorField::Format);
		context.bind(velocityXTex.getLevel(0), velocityPackingXBinding, AccessPolicy::WriteOnly, GPUScalarField::Format);
		context.bind(velocityYTex.getLevel(0), velocityPackingYBinding, AccessPolicy::WriteOnly, GPUScalarField::Format);
		context.setShaderProgram(velocityUnpackProgram);
		context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
		context.dispatchCompute(params.gridSize.x / velocityPackingWorkGroupX, params.gridSize.y / velocityPackingWorkGroupY, 1);

		velocityX->swap();
		velocityY->swap();
	}

	// Perform Jacobi iterations on individual components
	for (auto fields : { velocityX.get(), velocityY.get() /*, velocityZ.get() */})
	{
		GPUScalarField& fieldIn = fields->getInput();
		GPUScalarField& fieldOut = fields->getOutput();

		performJacobiIterations(jacobiProgram, fieldIn, fieldIn, fieldOut, params.gridSize, jacobiSteps);

		fields->swap();
	}

	// Pack velocity field
	{
		auto& velocityTex = fluidState.velocity.getOutput();
		auto& velocityXTex = velocityX->getInput();
		auto& velocityYTex = velocityY->getInput();

		velocityPackProgram.registerTexture("uVelocityOut", velocityTex, false);
		velocityPackProgram.registerTexture("uVelocityXIn", velocityXTex, false);
		velocityPackProgram.registerTexture("uVelocityYIn", velocityYTex, false);
		context.bind(velocityTex.getLevel(0), allVelocityBinding, AccessPolicy::WriteOnly, GPUVectorField::Format);
		context.bind(velocityXTex.getLevel(0), velocityPackingXBinding, AccessPolicy::ReadOnly, GPUScalarField::Format);
		context.bind(velocityYTex.getLevel(0), velocityPackingYBinding, AccessPolicy::ReadOnly, GPUScalarField::Format);
		context.setShaderProgram(velocityPackProgram);
		context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
		context.dispatchCompute(params.gridSize.x / velocityPackingWorkGroupX, params.gridSize.y / velocityPackingWorkGroupY, 1);
	}

	fluidState.velocity.swap();
}

void performForcesApplication(ShaderProgram& forcesProgram, FluidState& fluidState, BufferedScalarField& inkDensity, FluidSimMouseClickImpulse& impulse, float dt)
{
	constexpr int forcesWorkGroupX = 32;
	constexpr int forcesWorkGroupY = 32;

	const auto& params = fluidState.parameters;

	Context& context = Context::get();

	auto& velocityTex = fluidState.velocity.getInput();
	auto& inkTex = inkDensity.getInput();

	forcesProgram.uniform("udt", dt);
	forcesProgram.uniform("uMouseClick", impulse.position);
	forcesProgram.uniform("uForceMagnitude", impulse.magnitude);
	forcesProgram.uniform("uOneOverForceRadius", 1.f / impulse.radius);
	forcesProgram.uniform("uInkAmount", impulse.inkAmount);
	forcesProgram.registerTexture("uVelocity", velocityTex, false);
	forcesProgram.registerTexture("uInkDensity", inkTex, false);
	context.bind(velocityTex.getLevel(0), allVelocityBinding, AccessPolicy::ReadWrite, GPUVectorField::Format);
	context.bind(inkTex.getLevel(0), allInkDensityBinding, AccessPolicy::ReadWrite, GPUScalarField::Format);

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

	auto& velocityTex = fluidState.velocity.getInput();

	divergenceProgram.uniform("uHalfOneOverDx", 1.f / (2.f * params.gridCellSize));
	divergenceProgram.registerTexture("uVelocity", velocityTex, false);
	divergenceProgram.registerTexture("uFieldOut", fluidState.divergenceTex, false);
	context.bind(velocityTex.getLevel(0), allVelocityBinding, AccessPolicy::ReadOnly, GPUVectorField::Format);
	context.bind(fluidState.divergenceTex.getLevel(0), divergenceOutBinding, AccessPolicy::WriteOnly, GPUScalarField::Format);

	context.setShaderProgram(divergenceProgram);
	context.dispatchCompute(params.gridSize.x / divergenceWorkGroupX, params.gridSize.y / divergenceWorkGroupY, 1);
}

void performPressureComputation(ShaderProgram& jacobiProgram, FluidState& fluidState, int jacobiSteps)
{
	const auto& params = fluidState.parameters;

	// Upload solver parameters
	{
		float alpha = -params.gridCellSize * params.gridCellSize * params.density;
		float oneOverBeta = 1.f / 4.f;
		jacobiProgram.uniform("uAlpha", alpha);
		jacobiProgram.uniform("uOneOverBeta", oneOverBeta);
		jacobiProgram.uniform("uBoundaryCondition", neumannBoundaryCondition);
	}

	GPUScalarField& fieldIn = fluidState.pressure.getInput();
	GPUScalarField& fieldOut = fluidState.pressure.getOutput();

	fieldIn.template clearLevel<DataFormat::Red, DataType::Float>(0);

	performJacobiIterations(jacobiProgram, fluidState.divergenceTex, fieldIn, fieldOut, params.gridSize, jacobiSteps);

	fluidState.pressure.swap();
}

void performProjection(ShaderProgram& projectionProgram, FluidState& fluidState)
{
	constexpr int projectionWorkGroupX = 32;
	constexpr int projectionWorkGroupY = 32;

	const auto& params = fluidState.parameters;

	Context& context = Context::get();

	auto& velocityTex = fluidState.velocity.getInput();
	auto& pressureTex = fluidState.pressure.getInput();

	projectionProgram.uniform("uHalfOneOverDx", 1.f / (2.f * params.gridCellSize));
	projectionProgram.registerTexture("uVelocity", velocityTex, false);
	projectionProgram.registerTexture("uPressure", pressureTex, false);
	context.bind(velocityTex.getLevel(0), allVelocityBinding, AccessPolicy::ReadWrite, GPUVectorField::Format);
	context.bind(pressureTex.getLevel(0), projectionPressureBinding, AccessPolicy::ReadOnly, GPUScalarField::Format);

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

TextureInfo selectDebugTexture(FluidState& fluidState, int whichDebugTexture)
{
	switch (whichDebugTexture)
	{
	case 0:
		return fluidState.velocity.getInput();
	case 1:
		return fluidState.pressure.getInput();
	case 2:
		return fluidState.divergenceTex;
	default:
		assert(0);
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

	Texture<TextureTarget::Texture2D, TextureFormat::RGB> test("Test texture");

	context.debugMessageControl(DebugMessageSource::DontCare, DebugMessageType::DontCare, DebugMessageSeverity::DontCare, false);
	context.debugMessageControl(DebugMessageSource::DontCare, DebugMessageType::Error, DebugMessageSeverity::DontCare, true);
	context.debugMessageCallback(debugCallback, nullptr);

	// Fluid setup
	FluidState fluidState(Empty::math::uvec2(256, 256), 0.8f, 1.f, 0.0025f);
	FluidRenderParameters fluidRenderParameters{ Empty::math::uvec2(context.frameWidth, context.frameHeight), fluidState.parameters.gridSize, 4.f };

	// Fun things to advect by the fluid
	BufferedScalarField inkDensity("Ink density", fluidState.parameters.gridSize);

	// Programs and shaders
	Shader entryPointShader(ShaderType::Compute, "Entry point shader");
	if (!entryPointShader.setSourceFromFile("shaders/entry_point.glsl"))
		FATAL("Failed to compile scalar entry point shader:\n" << entryPointShader.getLog());

	ShaderProgram advectionProgram("Advection program");
	advectionProgram.attachShader(entryPointShader);
	advectionProgram.attachFile(ShaderType::Compute, "shaders/advection.glsl", "Advection shader");
	advectionProgram.build();

	ShaderProgram velocityUnpackProgram("Velocity unpack program");
	velocityUnpackProgram.attachFile(ShaderType::Compute, "shaders/velocity_unpack.glsl", "Velocity unpack shader");
	velocityUnpackProgram.build();

	ShaderProgram velocityPackProgram("Velocity pack program");
	velocityPackProgram.attachFile(ShaderType::Compute, "shaders/velocity_pack.glsl", "Velocity pack shader");
	velocityPackProgram.build();

	ShaderProgram jacobiProgram("Jacobi program");
	jacobiProgram.attachShader(entryPointShader);
	jacobiProgram.attachFile(ShaderType::Compute, "shaders/jacobi.glsl", "Jacobi shader");
	jacobiProgram.build();

	ShaderProgram forcesProgram("Force application program");
	forcesProgram.attachShader(entryPointShader);
	forcesProgram.attachFile(ShaderType::Compute, "shaders/forces.glsl", "Force application shader");
	forcesProgram.build();

	ShaderProgram divergenceProgram("Divergence program");
	divergenceProgram.attachFile(ShaderType::Compute, "shaders/divergence.glsl", "Divergence shader");
	divergenceProgram.build();

	ShaderProgram projectionProgram("Projection program");
	projectionProgram.attachShader(entryPointShader);
	projectionProgram.attachFile(ShaderType::Compute, "shaders/projection.glsl", "Projection shader");
	projectionProgram.build();

	double then = glfwGetTime();
	Empty::math::vec2 mouseThen = ImGui::GetMousePos();

	int diffusionJacobiSteps = 100;
	int pressureJacobiSteps = 100;
	FluidSimMouseClickImpulse impulse;
	impulse.radius = 240.f;
	impulse.inkAmount = 7.f;
	float forceScale = 500.f;

	// Simulation control variables
	bool capFPS = false;
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
			{
				fluidState.velocity.clear();
				fluidState.pressure.clear();
				fluidState.divergenceTex.template clearLevel<DataFormat::Red, DataType::Float>(0);
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
			ImGui::Combo("Display which", &whichDebugTexture, "Velocity\0Pressure\0Velocity divergence\0");
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
				performAdvection(advectionProgram, fluidState, inkDensity, dt, fluidState.parameters.gridSize);

			if (displayDebugTexture && whenDebugTexture == 1)
				displayTexture(debugDrawProgram, selectDebugTexture(fluidState, whichDebugTexture));

			// Compute viscous diffusion
			if (runDiffusion)
				performDiffusion(jacobiProgram, velocityUnpackProgram, velocityPackProgram, fluidState, dt, diffusionJacobiSteps);

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

				context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
				performForcesApplication(forcesProgram, fluidState, inkDensity, impulse, dt);
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

				context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
				performForcesApplication(forcesProgram, fluidState, inkDensity, impulse, dt);

				impulse.inkAmount = temp;
			}

			if (displayDebugTexture && whenDebugTexture == 3)
				displayTexture(debugDrawProgram, selectDebugTexture(fluidState, whichDebugTexture));

			// Compute divergence for pressure computation
			if (runDivergence)
			{
				context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
				performDivergenceComputation(divergenceProgram, fluidState);
			}

			if (displayDebugTexture && whenDebugTexture == 4)
				displayTexture(debugDrawProgram, selectDebugTexture(fluidState, whichDebugTexture));

			// Compute pressure field
			if (runPressure)
			{
				context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
				performPressureComputation(jacobiProgram, fluidState, pressureJacobiSteps);
			}

			if (displayDebugTexture && whenDebugTexture == 5)
				displayTexture(debugDrawProgram, selectDebugTexture(fluidState, whichDebugTexture));

			// Project over divergence-free fields by subtracting the pressure gradient from the velocity
			if (runProjection)
			{
				context.memoryBarrier(MemoryBarrierType::ShaderImageAccess);
				performProjection(projectionProgram, fluidState);
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
				displayTexture(debugDrawProgram, inkDensity.getInput().getInfo());
		}

		// ImGui::ShowDemoWindow();

		context.swap();

		then = now;
		mouseThen = mouseNow;
	}

	context.terminate();

	return 0;
}
