#pragma once

#include <Empty/gl/GLEnums.hpp>
#include <Empty/gl/Texture.h>

// *****************************************
// Types related to scalar and vector fields
// *****************************************

constexpr Empty::gl::DataFormat gpuScalarDataFormat = Empty::gl::DataFormat::Red;
using GPUScalarField = Empty::gl::Texture<Empty::gl::TextureTarget::Texture2DArray, Empty::gl::TextureFormat::Red32f>;

template <typename F, Empty::gl::DataFormat Format>
struct BufferedField
{
	using Field = F;

	BufferedField(const std::string& name, Empty::math::uvec3 size) :
		fields{ { name + " 1" } , { name + " 2"} },
		writingBackBuffer(true)
	{
		using namespace Empty::gl;

		for (int i : { 0, 1 })
		{
			fields[i].setStorage(1, size.x, size.y, size.z);
			fields[i].template clearLevel<Format, DataType::Float>(0);
			fields[i].setParameter<TextureParam::WrapS>(TextureParamValue::ClampToBorder);
			fields[i].setParameter<TextureParam::WrapT>(TextureParamValue::ClampToBorder);
			fields[i].setParameter<TextureParam::WrapR>(TextureParamValue::ClampToBorder);
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

private:
	Field fields[2];
	bool writingBackBuffer;
};

using BufferedScalarField = BufferedField<GPUScalarField, gpuScalarDataFormat>;
