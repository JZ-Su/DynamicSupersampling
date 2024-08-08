#pragma once
#include "VulkanBVHBuilder.h"
#include "VulkanRTShader.h"
#include "../VulkanTutorials/VulkanTutorial.h"
#include "VulkanShaderBindingTableBuilder.h"
#include "../GLTFLoader/GLTFLoader.h"

#include "WorldLight.h"

namespace NCL::Rendering::Vulkan {
	enum RenderType {
		Default,
		Dynamic,
		Fullscene,
		MAX_SIZE
	};

	using UniqueVulkanRTShader = std::unique_ptr<VulkanRTShader>;
	using SharedVulkanRTShader = std::shared_ptr<VulkanRTShader>;

	class RayTracingWithPBR : public VulkanTutorial {
	public:
		RayTracingWithPBR(Window& window, VulkanInitialisation& vkInit);
		~RayTracingWithPBR();

	protected:
		void RenderFrame(float dt) override;

		void SwitchRenderType();

		void CreateTexturesBuffer(vk::Device device, vk::DescriptorPool pool);
		void CreateVerticesBuffer(vk::Device device, vk::DescriptorPool pool);
		void CreateSkyboxBuffer(vk::Device device, vk::DescriptorPool pool);
		void CreateLightBuffer(vk::Device device, vk::DescriptorPool pool);

		RenderType renderType;

		GLTFScene scene;

		VulkanPipeline		displayPipeline;
		UniqueVulkanShader	displayShader;

		VulkanPipeline		rtPipeline;
		UniqueVulkanMesh	quadMesh;

		vk::UniqueDescriptorSetLayout	rayTraceLayout;
		vk::UniqueDescriptorSet			rayTraceDescriptor;

		vk::UniqueDescriptorSetLayout	imageLayout;
		vk::UniqueDescriptorSet			imageDescriptor;

		vk::UniqueDescriptorSetLayout	configLayout;
		vk::UniqueDescriptorSet			configDescriptor;
		VulkanBuffer					inverseMatrices;
		VulkanBuffer					cameraPosition;
		VulkanBuffer					renderMethod;

		vk::UniqueDescriptorSetLayout	displayImageLayout;
		vk::UniqueDescriptorSet			displayImageDescriptor;

		vk::UniqueDescriptorSetLayout	textureLayout;
		vk::UniqueDescriptorSet			textureDescriptor;
		VulkanBuffer					sampleSizeBuffer;

		vk::UniqueDescriptorSetLayout	vertexDataLayout;
		vk::UniqueDescriptorSet			vertexDataDescriptor;
		VulkanBuffer					vertexDataBuffer;
		VulkanBuffer					indexDataBuffer;

		vk::UniqueDescriptorSetLayout	lightLayout;
		vk::UniqueDescriptorSet			lightDescriptor;
		VulkanBuffer					lightCount;
		VulkanBuffer					parallelBuffer;
		VulkanBuffer					dotBuffer;

		vk::UniqueDescriptorSetLayout	cubeMapLayout;
		vk::UniqueDescriptorSet			cubeMapDescriptor;
		UniqueVulkanTexture				cubeMap;

		UniqueVulkanTexture				rayTexture;
		vk::ImageView					imageWriteView;

		ShaderBindingTable				bindingTable;
		VulkanBVHBuilder				bvhBuilder;
		vk::UniqueAccelerationStructureKHR	tlas;

		UniqueVulkanRTShader	raygenShader;
		UniqueVulkanRTShader	hitShader;
		UniqueVulkanRTShader	missShader;
		UniqueVulkanRTShader	anyHitShader;
		UniqueVulkanRTShader	shadowAHit;
		UniqueVulkanRTShader	shadowCHit;
		UniqueVulkanRTShader	shadowMiss;

		vk::PhysicalDeviceRayTracingPipelinePropertiesKHR	rayPipelineProperties;
		vk::PhysicalDeviceAccelerationStructureFeaturesKHR	rayAccelFeatures;

		WorldLight worldLight;

		uint32_t FindMemoryType(vk::PhysicalDevice physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties);
		std::vector<Vector4> ReadImageRGB(vk::Device device, vk::Queue queue, vk::CommandPool commandPool, vk::Image image, vk::Format format, uint32_t width, uint32_t height);
	};
}

