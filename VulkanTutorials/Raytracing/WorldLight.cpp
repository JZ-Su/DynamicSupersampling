#include "WorldLight.h"

using namespace NCL;
using namespace Rendering;
using namespace Vulkan;

WorldLight::WorldLight() {
}

WorldLight::~WorldLight() {
}

WorldLight& WorldLight::AddParallelLight(const Vector4& inColour, const Vector3& inDirection) {
	ParallelLight l;
	l.direction = Vector4(inDirection, 0.0);
	l.colour = inColour;
	parallelLight.push_back(l);
	return *this;
}

WorldLight& WorldLight::AddDotLight(const Vector4& inColour, const Vector3& inPosition, float inRadius) {
	DotLight l;
	l.position = Vector4(inPosition, 0.0);
	l.colour = inColour;
	l.radius = inRadius;
	dotLight.push_back(l);
	return *this;
}