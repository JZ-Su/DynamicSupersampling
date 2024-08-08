/******************************************************************************
This file is part of the Newcastle Vulkan Tutorial Series

Author:Rich Davison
Contact:richgdavison@gmail.com
License: MIT (see LICENSE file at the top of the source tree)
*//////////////////////////////////////////////////////////////////////////////
#version 460
#extension GL_GOOGLE_include_directive		: enable
#extension GL_ARB_separate_shader_objects	: enable
#extension GL_ARB_shading_language_420pack	: enable
#extension GL_EXT_ray_tracing : require

#include "RayStructs.glslh"

layout(location = 0) rayPayloadInEXT BasicPayload payload;

layout(set = 6, binding = 0) uniform samplerCube skyTex;

void main() {
	vec3 rayDir = gl_WorldRayDirectionEXT;
	payload.hitValue = texture(skyTex, rayDir).rgb;
	payload.shadow_count = 0;
	payload.objectIndex = -1;
}