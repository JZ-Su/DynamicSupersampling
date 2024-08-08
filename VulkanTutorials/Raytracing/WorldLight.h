#pragma once
#include <vector>
#include "../NCLCoreClasses/Maths.h"
#include "../NCLCoreClasses/Vector.h"

namespace NCL::Rendering::Vulkan {
	struct ParallelLight
	{
		// use vec4 for data alignment
		Vector4 colour;
		Vector4 direction;
	};
	struct DotLight
	{
		Vector4 colour;
		Vector3 position;
		// shared with vec3 padding
		float radius;
	};

	class WorldLight
	{
	public:
		WorldLight();
		~WorldLight();

		WorldLight& AddParallelLight(const Vector4& inColour, const Vector3& inDirection);
		WorldLight& AddDotLight(const Vector4& inColour, const Vector3& inPosition, float inRadius);

		std::vector<ParallelLight> GetParalleLight() const { return parallelLight; }
		std::vector<DotLight> GetDotLight() const { return dotLight; }

	protected:
		std::vector<ParallelLight> parallelLight;
		std::vector<DotLight> dotLight;
	};

}