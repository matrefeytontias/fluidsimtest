#pragma once

#include <Empty/gl/GLEnums.hpp>
#include <Empty/gl/Texture.h>

// *****************************************
// Types related to scalar and vector fields
// *****************************************

constexpr Empty::gl::DataFormat gpuScalarDataFormat = Empty::gl::DataFormat::Red;
using GPUScalarField = Empty::gl::Texture<Empty::gl::TextureTarget::Texture2D, Empty::gl::TextureFormat::Red32f>;

template <typename F, Empty::gl::DataFormat Format>
struct BufferedField
{
	using Field = F;

	BufferedField(const std::string& name, Empty::math::uvec2 size) :
		fields{ { name + " 1" } , { name + " 2"} },
		writingBackBuffer(true)
	{
		using namespace Empty::gl;

		for (int i : { 0, 1 })
		{
			fields[i].setStorage(1, size.x, size.y);
			fields[i].template clearLevel<Format, DataType::Float>(0);
			fields[i].template setParameter<TextureParam::WrapS>(TextureParamValue::ClampToBorder);
			fields[i].template setParameter<TextureParam::WrapT>(TextureParamValue::ClampToBorder);
		}
	}

	void clear()
	{
		using namespace Empty::gl;

		fields[0].template clearLevel<Format, DataType::Float>(0);
		fields[1].template clearLevel<Format, DataType::Float>(0);
		writingBackBuffer = true;
	}

	auto& getInput() { return fields[writingBackBuffer ? 0 : 1]; }
	auto& getOutput() { return fields[writingBackBuffer ? 1 : 0]; }

	void swap() { writingBackBuffer = !writingBackBuffer; }

	/*
	* Set a boundary value. This will only record the same number of components
	* that the field itself has (eg a one component field won't have a four component
	* boundary value). This is implemented using the border color.
	*/
	void setBoundaryValue(Empty::math::vec4 boundary)
	{
		fields[0].template setParameter<Empty::gl::TextureParam::BorderColor>(boundary);
		fields[1].template setParameter<Empty::gl::TextureParam::BorderColor>(boundary);
	}

private:
	Field fields[2];
	bool writingBackBuffer;
};

using BufferedScalarField = BufferedField<GPUScalarField, gpuScalarDataFormat>;
