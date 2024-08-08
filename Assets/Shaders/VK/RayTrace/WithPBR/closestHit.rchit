#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive		: enable
#extension GL_ARB_separate_shader_objects	: enable
#extension GL_ARB_shading_language_420pack	: enable
#extension GL_EXT_nonuniform_qualifier : require

hitAttributeEXT vec2 attribs;
#include "RayStructs.glslh"

layout(location = 0) rayPayloadInEXT BasicPayload payload;

layout(location = 1) rayPayloadEXT bool inShadow;

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

layout(set = 1, binding = 0) uniform CameraInfo {
	mat4 viewMatrix;
	mat4 projMatrix;
};

layout(set = 2, binding  = 0) uniform InverseCameraInfo {
	mat4 invViewMatrix;
	mat4 invProjMatrix;
};

layout(set = 2, binding = 1) uniform CameraPositionInfo {
	vec3 cameraPos;
};

layout(set = 2, binding = 2) uniform RenderTypeInfo {
	int renderType;
};

layout(set = 4, binding = 0) uniform sampler2D textures[];

layout(set = 5, binding = 0) buffer VertexData {
	vec3 position[192496];
	vec2 texCoord[192496];
	vec3 normal[192496];
};

layout(set = 5, binding = 1) buffer IndexData {
	int startIndex[103];
	int baseIndex[103];
	int indices[786801];
};

layout(set = 7, binding = 0) uniform LightCount {
	int parallelLightCount;
	int dotLightCount;
};

// No need to declare the size if the buffer binding only has 1 object
layout(set = 7, binding = 1) buffer ParallelLightData {
	ParallelLight parallelLight[];
};

layout(set = 7, binding = 2) buffer DotLightData {
	DotLight dotLight[];
};

void main() {

	uint index0 = baseIndex[gl_GeometryIndexEXT] + indices[startIndex[gl_GeometryIndexEXT] + gl_PrimitiveID * 3];
	uint index1 = baseIndex[gl_GeometryIndexEXT] + indices[startIndex[gl_GeometryIndexEXT] + gl_PrimitiveID * 3 + 1];
	uint index2 = baseIndex[gl_GeometryIndexEXT] + indices[startIndex[gl_GeometryIndexEXT] + gl_PrimitiveID * 3 + 2];

	vec3 p0 = position[index0];
	vec3 p1 = position[index1];
	vec3 p2 = position[index2];

	vec2 uv0 = texCoord[index0];
	vec2 uv1 = texCoord[index1];
	vec2 uv2 = texCoord[index2];

	vec3 n0 = normal[index0];
	vec3 n1 = normal[index1];
	vec3 n2 = normal[index2];
	
	uint texIndex = gl_GeometryIndexEXT * 3;

	vec3 baryCentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
	// vec3 baryCentrics = gl_HitTriangleBaryCoordEXT;

	vec2 hitTexCoord = uv0 * baryCentrics.x + uv1 * baryCentrics.y + uv2 * baryCentrics.z;
	vec3 hitPos 	 = p0 * baryCentrics.x + p1 * baryCentrics.y + p2 * baryCentrics.z;
	vec3 hitNormal 	 = n0 * baryCentrics.x + n1 * baryCentrics.y + n2 * baryCentrics.z;

	vec3 disToCamera = hitPos - cameraPos.xyz;
	float dis = sqrt(disToCamera.x * disToCamera.x + disToCamera.y * disToCamera.y + disToCamera.z * disToCamera.z);
	float lod = 0.005 * dis;
	lod = clamp(lod, 1.0, 2.0);
	lod = 1;
	
	vec3 albedo		= vec3(
		pow(textureLod(textures[texIndex], hitTexCoord, lod).r, 2.2),
    	pow(textureLod(textures[texIndex], hitTexCoord, lod).g, 2.2),
    	pow(textureLod(textures[texIndex], hitTexCoord, lod).b, 2.2)
	);
	float roughness	= textureLod(textures[texIndex + 1], hitTexCoord, lod).r;
	float metallic	= textureLod(textures[texIndex + 2], hitTexCoord, lod).r;
	float ao = 50.0;

	vec3 N  = normalize(hitNormal);
	vec3 V  = normalize(cameraPos.xyz - hitPos);
	vec3 F0 = vec3(0.04);
	F0 = mix(F0, albedo, metallic);

	vec3 Lo = vec3(0.0);
	for(int i = 0; i < dotLightCount; i++) {
		vec3 lightPosition = vec3(dotLight[i].posRadius.xyz);
		vec3 L = normalize(lightPosition - hitPos);
		vec3 H = normalize(V + L);
		float distances = length(lightPosition - hitPos);
		float attenuation = dotLight[i].posRadius.w / (distances * distances);
		vec3  radiance    = vec3(dotLight[i].color.xyz) * attenuation;

		float NDF = DistributionGGX(N, H, roughness);
		float G   = GeometrySmith(N, V, L, roughness);
		vec3  F   = fresnelSchlick(max(dot(H, V), 0.0), F0);

		vec3 kS = F;
		vec3 kD = vec3(1.0) - kS;
		kD *= 1.0 - metallic;

		vec3  nominator   = NDF * G * F;
		float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
		vec3  specular    = nominator / denominator;

		float NdotL = max(dot(N, L), 0.0);
		Lo += (kD * albedo / PI + specular) * radiance * NdotL;
	}

	vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 color = ambient + Lo;
	color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));

	// only identify if it is in the shadow once
	// when the payload is default
	if(payload.shadow_count == -1) {
		payload.shadow_count = 0;

		vec3 rayOrigin = hitPos + hitNormal * 0.0001;
		uint flag = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT;

		for(int i = 0; i < parallelLightCount; i++) {
			vec3 rayDirection = parallelLight[i].direction;

			traceRayEXT(tlas,			// acceleration structure
						flag,			// rayFlags
						0xFF,			// cullMask
						1,				// sbtRecordOffset
						0,				// sbtRecordStride
						1,				// missIndex
						rayOrigin,		// ray origin
						0.001,			// ray min range
						rayDirection,   // ray direction
						10000.0,		// ray max range
						1				// payload (location = 1)
			);
			if(inShadow) {
				payload.shadow_count++;
				// float ambientStrength = 0.5f;
				// vec3 lightColor = parallelLight[i].color;
				// vec3 shadowAmbient = ambientStrength * lightColor;
				// color = shadowAmbient * color;
			}
		}

		// vec4 projPosi0 = viewMatrix * projMatrix * vec4(p0, 1);
		// vec4 projPosi1 = viewMatrix * projMatrix * vec4(p1, 1);
		// vec4 projPosi2 = viewMatrix * projMatrix * vec4(p2, 1);

		// vec3 uvP0 = normalize(projPosi0).xyz;
		// vec3 uvP1 = normalize(projPosi1).xyz;
		// vec3 uvP2 = normalize(projPosi2).xyz;

		// float S0 = length(cross(p1 - p0, p2 - p0)) / 2;
		// float S1 = length(cross(uvP1 - uvP0, uvP2 - uvP0)) / 2;
		// payload.extraSample = S0 / S1;



		//vec3 inRayDir = gl_WorldRayDirectionEXT;
		//float cosine = dot(inRayDir, hitNormal) / (length(inRayDir) * length(hitNormal));
		//payload.extraSample = 1.0 / cosine;
	}



	payload.hitValue = color;
	payload.objectIndex = gl_GeometryIndexEXT;

	
  	// traceRayEXT(tlas,		// acceleration structure
    // 	        flag,			// rayFlags
    // 	        0xFF,			// cullMask
    //     	    1,				// sbtRecordOffset
    //         	0,				// sbtRecordStride
    //         	1,				// missIndex
	//          rayOrigin,		// ray origin
    //          0.001,			// ray min range
    // 	      	rayDirection,	// ray direction
    //        	10000.0,		// ray max range
    //        	1				// payload (location = 1)
	// );
	
	// if (inShadow) {
	// 	float ambientStrength = 0.5f;
	// 	vec3 lightColor = vec3(1.0f, 1.0f, 1.0f);
    // 	vec3 ambient = ambientStrength * lightColor;

    // 	vec3 result = ambient * color;
	// 	payload.hitValue = result;

	// } else {
	// 	payload.hitValue = color;
	// }

}