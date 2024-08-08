#include "RayTracingWithPBR.h"
#include "VulkanRayTracingPipelineBuilder.h"
#include "../GLTFLoader/GLTFLoader.h"
#include "FourierTransform.h"
#include "opencv2/opencv.hpp"

using namespace NCL;
using namespace Rendering;
using namespace Vulkan;

TUTORIAL_ENTRY(RayTracingWithPBR)

RayTracingWithPBR::RayTracingWithPBR(Window& window, VulkanInitialisation& vkInit) : VulkanTutorial(window) {
	vkInit.majorVersion = 1;
	vkInit.minorVersion = 3;
	vkInit.autoBeginDynamicRendering = false;
	vkInit.skipDynamicState = true;

	vkInit.deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
	vkInit.deviceExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
	vkInit.deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
	vkInit.deviceExtensions.emplace_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
	vkInit.vmaFlags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

	static vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelFeatures = {
		.accelerationStructure = true
	};

	static vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rayFeatures = {
		.rayTracingPipeline = true
	};

	static vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR deviceAddressFeature = {
		.bufferDeviceAddress = true
	};

	static vk::PhysicalDeviceDescriptorIndexingFeatures runtimeDescriptorFeature = {
		.runtimeDescriptorArray = true
	};

	vkInit.features.push_back((void*)&accelFeatures);
	vkInit.features.push_back((void*)&rayFeatures);
	vkInit.features.push_back((void*)&deviceAddressFeature);
	vkInit.features.push_back((void*)&runtimeDescriptorFeature);

	renderer = new VulkanRenderer(window, vkInit);
	InitTutorialObjects();

	FrameState const& state = renderer->GetFrameState();
	vk::Device device = renderer->GetDevice();
	vk::DescriptorPool pool = renderer->GetDescriptorPool();

	renderType = Dynamic;

	GLTFLoader::Load("Sponza/Sponza.gltf", scene);

	//GLTFLoader::Load("CesiumMan/CesiumMan.gltf", scene);

	for (const auto& m : scene.meshes) {
		VulkanMesh* loadedMesh = (VulkanMesh*)m.get();
		loadedMesh->UploadToGPU(renderer, vk::BufferUsageFlagBits::eShaderDeviceAddress |
			vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR);
	}

	vk::PhysicalDeviceProperties2 props;
	props.pNext = &rayPipelineProperties;
	renderer->GetPhysicalDevice().getProperties2(&props);

	raygenShader = UniqueVulkanRTShader(new VulkanRTShader("RayTrace/WithPBR/rayGen.rgen.spv", device));
	hitShader = UniqueVulkanRTShader(new VulkanRTShader("RayTrace/WithPBR/closestHit.rchit.spv", device));
	missShader = UniqueVulkanRTShader(new VulkanRTShader("RayTrace/WithPBR/miss.rmiss.spv", device));
	anyHitShader = UniqueVulkanRTShader(new VulkanRTShader("RayTrace/WithPBR/anyHit.rahit.spv", device));
	//shadowAHit		= UniqueVulkanRTShader(new VulkanRTShader("RayTrace/WithPBR/shadowAnyHit.rahit.spv", device));
	shadowCHit = UniqueVulkanRTShader(new VulkanRTShader("RayTrace/WithPBR/shadowCHit.rchit.spv", device));
	shadowMiss = UniqueVulkanRTShader(new VulkanRTShader("RayTrace/WithPBR/shadowMiss.rmiss.spv", device));

	rayTraceLayout = DescriptorSetLayoutBuilder(device)
		.WithAccelStructures(0, 1)
		.Build("Ray Trace TLAS Layout");

	imageLayout = DescriptorSetLayoutBuilder(device)
		.WithStorageImages(0, 1)
		.Build("Ray Trace Image Layout");

	configLayout = DescriptorSetLayoutBuilder(device)
		.WithUniformBuffers(0, 1)		// inverse camera matrices
		.WithUniformBuffers(1, 1)		// camera position
		.WithUniformBuffers(2, 1)		// render type
		.Build("Camera Inverse Matrix Layout");

	for (const auto& n : scene.sceneNodes) {
		if (n.mesh == nullptr) {
			continue;
		}
		VulkanMesh* vm = (VulkanMesh*)n.mesh;
		//bvhBuilder.WithObject(vm, n.worldMatrix);
		bvhBuilder.WithObject(vm, Matrix::Translation(Vector3{ 0,0,0 }));
	}

	tlas = bvhBuilder
		.WithCommandQueue(renderer->GetQueue(CommandType::AsyncCompute))
		.WithCommandPool(renderer->GetCommandPool(CommandType::AsyncCompute))
		.WithDevice(device)
		.WithAllocator(renderer->GetMemoryAllocator())
		.Build(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace, "GLTF BLAS");

	rayTraceDescriptor = CreateDescriptorSet(device, pool, *rayTraceLayout);
	imageDescriptor = CreateDescriptorSet(device, pool, *imageLayout);
	configDescriptor = CreateDescriptorSet(device, pool, *configLayout);

	inverseMatrices = BufferBuilder(device, renderer->GetMemoryAllocator())
		.WithBufferUsage(vk::BufferUsageFlagBits::eUniformBuffer)
		.WithHostVisibility()
		.WithPersistentMapping()
		.Build(sizeof(Matrix4) * 2, "InverseMatrices");
	cameraPosition = BufferBuilder(device, renderer->GetMemoryAllocator())
		.WithBufferUsage(vk::BufferUsageFlagBits::eUniformBuffer)
		.WithHostVisibility()
		.WithPersistentMapping()
		.Build(sizeof(Vector4), "CameraPosition");
	renderMethod = BufferBuilder(device, renderer->GetMemoryAllocator())
		.WithBufferUsage(vk::BufferUsageFlagBits::eUniformBuffer)
		.WithHostVisibility()
		.WithPersistentMapping()
		.Build(sizeof(int), "RenderMethod");

	Vector2i windowSize = hostWindow.GetScreenSize();

	rayTexture = TextureBuilder(device, renderer->GetMemoryAllocator())
		.UsingPool(renderer->GetCommandPool(CommandType::Graphics))
		.UsingQueue(renderer->GetQueue(CommandType::Graphics))
		.WithDimension(windowSize.x, windowSize.y, 1)
		.WithUsages(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage)
		.WithPipeFlags(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
		.WithLayout(vk::ImageLayout::eGeneral)
		.WithFormat(vk::Format::eB8G8R8A8Unorm)
		.Build("RaytraceResult");

	WriteStorageImageDescriptor(device, *imageDescriptor, 0, *rayTexture, *defaultSampler, vk::ImageLayout::eGeneral);

	WriteBufferDescriptor(device, *configDescriptor, 0, vk::DescriptorType::eUniformBuffer, inverseMatrices);
	WriteBufferDescriptor(device, *configDescriptor, 1, vk::DescriptorType::eUniformBuffer, cameraPosition);
	WriteBufferDescriptor(device, *configDescriptor, 2, vk::DescriptorType::eUniformBuffer, renderMethod);

	WriteTLASDescriptor(device, *rayTraceDescriptor, 0, *tlas);


	// new
	CreateTexturesBuffer(device, pool);

	CreateVerticesBuffer(device, pool);

	CreateSkyboxBuffer(device, pool);

	worldLight = WorldLight()
		.AddParallelLight(Vector4(1.0, 1.0, 1.0, 1.0), Vector3(0.1, 1.0, 0.2))
		.AddDotLight(Vector4(1.0, 1.0, 1.0, 1.0), Vector3(0, 100, 100), 130);
	CreateLightBuffer(device, pool);
	camera.SetPosition(Vector3(0, 100, 100));

	auto rtPipeBuilder = VulkanRayTracingPipelineBuilder(device)
		.WithRecursionDepth(2)

		.WithShader(*raygenShader, vk::ShaderStageFlagBits::eRaygenKHR)		//0
		.WithShader(*missShader, vk::ShaderStageFlagBits::eMissKHR)			//1
		.WithShader(*shadowMiss, vk::ShaderStageFlagBits::eMissKHR)			//2
		.WithShader(*hitShader, vk::ShaderStageFlagBits::eClosestHitKHR)	//3
		.WithShader(*shadowCHit, vk::ShaderStageFlagBits::eClosestHitKHR)	//4
		//.WithShader(*anyHitShader, vk::ShaderStageFlagBits::eAnyHitKHR)
		//.WithShader(*shadowAHit,	vk::ShaderStageFlagBits::eAnyHitKHR)

		.WithRayGenGroup(0)	//Group for the raygen shader	//Uses shader 0
		.WithMissGroup(1)	//Group for the miss shader		//Uses shader 1
		.WithMissGroup(2)
		.WithTriangleHitGroup(3)	//Hit group 0			//Uses shader 3
		.WithTriangleHitGroup(4)

		.WithDescriptorSetLayout(0, *rayTraceLayout)
		.WithDescriptorSetLayout(1, *cameraLayout)
		.WithDescriptorSetLayout(2, *configLayout)
		.WithDescriptorSetLayout(3, *imageLayout)
		.WithDescriptorSetLayout(4, *textureLayout)
		.WithDescriptorSetLayout(5, *vertexDataLayout)
		.WithDescriptorSetLayout(6, *cubeMapLayout)
		.WithDescriptorSetLayout(7, *lightLayout);

	rtPipeline = rtPipeBuilder.Build("RT Pipeline");

	bindingTable = VulkanShaderBindingTableBuilder("SBT")
		.WithProperties(rayPipelineProperties)
		.WithPipeline(rtPipeline, rtPipeBuilder.GetCreateInfo())
		.Build(device, renderer->GetMemoryAllocator());

	//We also need some Vulkan things for displaying the result!
	displayImageLayout = DescriptorSetLayoutBuilder(device)
		.WithImageSamplers(0, 1, vk::ShaderStageFlagBits::eFragment)
		.Build("Raster Image Layout");

	displayImageDescriptor = CreateDescriptorSet(device, pool, *displayImageLayout);
	WriteImageDescriptor(device, *displayImageDescriptor, 0, *rayTexture, *defaultSampler, vk::ImageLayout::eShaderReadOnlyOptimal);

	quadMesh = GenerateQuad();

	displayShader = ShaderBuilder(device)
		.WithVertexBinary("Display.vert.spv")
		.WithFragmentBinary("Display.frag.spv")
		.Build("Result Display Shader");

	displayPipeline = PipelineBuilder(device)
		.WithVertexInputState(quadMesh->GetVertexInputState())
		.WithTopology(vk::PrimitiveTopology::eTriangleStrip)
		.WithShader(displayShader)
		.WithDescriptorSetLayout(0, *displayImageLayout)

		.WithColourAttachment(state.colourFormat)

		.WithDepthAttachment(renderer->GetDepthBuffer()->GetFormat())
		.Build("Result display pipeline");
}

RayTracingWithPBR::~RayTracingWithPBR() {
}

void RayTracingWithPBR::RenderFrame(float dt) {
	if (Window::GetKeyboard()->KeyPressed(KeyCodes::TAB)) {
		SwitchRenderType();
		std::cout << "Render Type: " << renderType << std::endl;
	}
	
	Matrix4* inverseMatrixData = (Matrix4*)inverseMatrices.Data();
	inverseMatrixData[0] = Matrix::Inverse(camera.BuildViewMatrix());
	inverseMatrixData[1] = Matrix::Inverse(camera.BuildProjectionMatrix(hostWindow.GetScreenAspect()));

	Vector4* cameraPositionData = (Vector4*)cameraPosition.Data();
	cameraPositionData[0] = Vector4(camera.GetPosition(), 0.0);

	int* renderMethodData = (int*)renderMethod.Data();
	renderMethodData[0] = static_cast<int>(renderType);

	FrameState const& frameState = renderer->GetFrameState();
	vk::CommandBuffer cmdBuffer = frameState.cmdBuffer;

	cmdBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, rtPipeline);

	vk::DescriptorSet sets[8] = {
		*rayTraceDescriptor,		//Set 0
		*cameraDescriptor,			//Set 1
		*configDescriptor,			//Set 2
		*imageDescriptor,			//Set 3
		*textureDescriptor,			//Set 4
		*vertexDataDescriptor,		//Set 5
		*cubeMapDescriptor,			//Set 6
		*lightDescriptor			//Set 7
	};

	cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, *rtPipeline.layout, 0, 8, sets, 0, nullptr);

	//Flip texture back for the next frame
	ImageTransitionBarrier(cmdBuffer, *rayTexture,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eGeneral,
		vk::ImageAspectFlagBits::eColor,
		vk::PipelineStageFlagBits2::eFragmentShader,
		vk::PipelineStageFlagBits2::eRayTracingShaderKHR);

	uint32_t renderWidth, renderHeight;

	if(renderType == Default) {
		renderWidth = frameState.defaultViewport.width;
		renderHeight = std::abs(frameState.defaultViewport.height);
	}
	else {
		renderWidth = frameState.defaultViewport.width * 3;
		renderHeight = std::abs(frameState.defaultViewport.height) * 3;
	}

	cmdBuffer.traceRaysKHR(
		&bindingTable.regions[BindingTableOrder::RayGen],
		&bindingTable.regions[BindingTableOrder::Miss],
		&bindingTable.regions[BindingTableOrder::Hit],
		&bindingTable.regions[BindingTableOrder::Call],
		renderWidth, renderHeight, 1	// gl_LaunchSizeEXT size
	);

	//Flip the texture we rendered into into display mode
	ImageTransitionBarrier(cmdBuffer, *rayTexture,
		vk::ImageLayout::eGeneral,
		vk::ImageLayout::eShaderReadOnlyOptimal,
		vk::ImageAspectFlagBits::eColor,
		vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
		vk::PipelineStageFlagBits2::eFragmentShader);

	// Get each pixel RGBA in rayTexture
	/*std::vector<Vector4> ImageRGBA = ReadImageRGB(renderer->GetDevice(), renderer->GetQueue(CommandType::Graphics), renderer->GetCommandPool(CommandType::Graphics),
		rayTexture.get()->GetImage(), vk::Format::eB8G8R8A8Unorm, frameState.defaultViewport.width, std::abs(frameState.defaultViewport.height));*/

	//Now display the results on screen!
	cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, displayPipeline);

	renderer->BeginDefaultRendering(cmdBuffer);
	cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *displayPipeline.layout, 0, 1, &*displayImageDescriptor, 0, nullptr);
	quadMesh->Draw(cmdBuffer);
	cmdBuffer.endRendering();
}


void RayTracingWithPBR::CreateTexturesBuffer(vk::Device device, vk::DescriptorPool pool) {
	const auto& scMat = scene.materials[0].allLayers;
	UniqueVulkanTexture emptyTexture = LoadTexture("blank.png");
	textureLayout = DescriptorSetLayoutBuilder(device)
		.WithImageSamplers(0, scMat.size() * 3)
		.WithStorageBuffers(1, 1)
		.Build("Ray Trace Texture Layout");
	textureDescriptor = CreateDescriptorSet(device, pool, *textureLayout);

	// Record results
	std::map<std::string, int> sampleSizeMap;
	std::vector<int> sampleSize;
	const std::string imagePath(NCL::Assets::GLTFDIR + "Sponza/");

	// convolution kernel
	cv::Mat kernel(1024, 1024, CV_32FC1);
	for (int y = 0; y < 1024; ++y) {
		for (int x = 0; x < 1024; ++x) {
			float distance = std::sqrt(std::pow(x - 512, 2) + std::pow(y - 512, 2));
			kernel.at<float>(y, x) = distance;
		}
	}
	cv::normalize(kernel, kernel, 0, 1, cv::NORM_MINMAX);


	for (size_t i = 0; i < scMat.size(); i++) {
		WriteImageDescriptor(device, *textureDescriptor, 0, i * 3, ((VulkanTexture*)scMat[i].albedo.get())->GetDefaultView(), *defaultSampler);

		std::string fileName = scene.loadedTexturesMap.at(scMat[i].albedo.get());

		std::map<std::string, int>::iterator it = sampleSizeMap.find(fileName);
		if (it != sampleSizeMap.end()) {
			// the image has been spectralised
			sampleSize.push_back(it->second);
		}
		else {
			cv::Mat magnitureImage = FourierTransform::Spectralisation(imagePath + fileName);
			cv::Mat convolutionResult = FourierTransform::FFTConvolve2D(magnitureImage, kernel);

			std::vector<cv::Point2f> points;
			for (int i = 1; i < convolutionResult.cols; i++) {
				float difference = convolutionResult.at<float>(i, i) - convolutionResult.at<float>(i - 1, i - 1);
				points.emplace_back(i, difference);
			}

			// umcomment the follows to show the curve image:
			/*cv::Mat m(800, 1024, CV_8UC3, cv::Scalar(0, 0, 0));
			for (int i = 1; i < points.size(); i++) {
				cv::Point prev = points[i - 1];
				cv::Point cur = points[i];
				cv::line(m, cv::Point(prev.x, 700 - prev.y / 5), cv::Point(cur.x, 700 - cur.y / 5), cv::Scalar(255, 0, 0), 1);
			}
			cv::imshow(fileName + "Curve Plot", m);*/

			std::vector<cv::Point2f> gradPoints;
			float maximum = 0.0f;
			for (int i = 1; i < points.size() - 1; i++) {
				float value = (points[i + 1].y - points[i - 1].y) / 2;
				if (value > maximum) maximum = value;
				gradPoints.emplace_back((i - 1), value);
			}

			// show the image:
			/*cv::Mat grad(800, 1024, CV_8UC3, cv::Scalar(0, 0, 0));
			for (int i = 1; i < gradPoints.size(); i++) {
				cv::Point prev = gradPoints[i - 1];
				cv::Point curr = gradPoints[i];
				cv::line(grad, cv::Point(prev.x, 700 - prev.y), cv::Point(curr.x, 700 - curr.y), cv::Scalar(255, 0, 0), 1);
			}
			cv::imshow(fileName + "Grad Plot", grad);*/

			int peakCount = 0;
			for (const auto& point : gradPoints) {
				if (point.y > 0.2 * maximum) {
					peakCount++;
					if (peakCount == 9) break;	// reach maximum
				}
			}
			//peakCount = std::min(peakCount, 9);
			sampleSizeMap.insert({ fileName, peakCount });
			sampleSize.push_back(peakCount);
		}

		if (scMat[i].bump.get()) {
			WriteImageDescriptor(device, *textureDescriptor, 0, (i * 3) + 1, ((VulkanTexture*)scMat[i].bump.get())->GetDefaultView(), *defaultSampler);
		}
		else {
			WriteImageDescriptor(device, *textureDescriptor, 0, (i * 3) + 1, emptyTexture.get()->GetDefaultView(), *defaultSampler);
		}
		if (scMat[i].metallic.get()) {
			WriteImageDescriptor(device, *textureDescriptor, 0, (i * 3) + 2, ((VulkanTexture*)scMat[i].metallic.get())->GetDefaultView(), *defaultSampler);
		}
		else {
			WriteImageDescriptor(device, *textureDescriptor, 0, (i * 3) + 2, emptyTexture.get()->GetDefaultView(), *defaultSampler);
		}
	}
	
	sampleSizeBuffer = BufferBuilder(device, renderer->GetMemoryAllocator())
		.WithBufferUsage(vk::BufferUsageFlagBits::eStorageBuffer)
		.WithPersistentMapping()
		.Build(sampleSize.size() * sizeof(int), "Sample Size Buffer");
	void* sampleSizeBufferData = (void*)sampleSizeBuffer.Data();
	memcpy(sampleSizeBufferData, sampleSize.data(), sampleSize.size() * sizeof(int));
	WriteBufferDescriptor(device, *textureDescriptor, 1, vk::DescriptorType::eStorageBuffer, sampleSizeBuffer);
}

void RayTracingWithPBR::CreateVerticesBuffer(vk::Device device, vk::DescriptorPool pool) {
	// copy vertex data
	const auto& meshData = scene.meshes[0].get();
	size_t meshCount = meshData->GetVertexCount();
	const auto& subMeshData = meshData->GetSubMeshData();
	// use vec4 for data alignment
	std::vector<Vector4> positionData;	// vec4(pos.x, pos.y, pos.z, 0)
	std::vector<Vector4> normalData;	// vec4(nor.x, nor.y, nor.z, 0)
	std::vector<Vector2> texCoordData;
	for (size_t i = 0; i < meshCount; i++) {
		positionData.push_back(Vector4(meshData->GetPositionData()[i], 0.0f));
		texCoordData.push_back(meshData->GetTextureCoordData()[i]);
		normalData.push_back(Vector4(meshData->GetNormalData()[i], 0.0f));
	}
	// create buffer
	size_t positionDataSize = meshCount * sizeof(Vector4);
	size_t texCoordDataSize = meshCount * sizeof(Vector2);
	size_t normalDataSize = meshCount * sizeof(Vector4);
	size_t vertexTotalSize = positionDataSize + texCoordDataSize + normalDataSize;
	vertexDataLayout = DescriptorSetLayoutBuilder(device)
		.WithStorageBuffers(0, 1)
		.WithStorageBuffers(1, 1)
		.Build("Mesh Data Layout");
	vertexDataDescriptor = CreateDescriptorSet(device, pool, *vertexDataLayout);
	vertexDataBuffer = BufferBuilder(device, renderer->GetMemoryAllocator())
		.WithBufferUsage(vk::BufferUsageFlagBits::eStorageBuffer)
		.WithPersistentMapping()
		.Build(vertexTotalSize, "Mesh Data Buffer");
	// write buffer
	void* mdBufferData = (void*)vertexDataBuffer.Data();
	memcpy(mdBufferData, positionData.data(), positionDataSize);
	memcpy(static_cast<char*>(mdBufferData) + positionDataSize, texCoordData.data(), texCoordDataSize);
	memcpy(static_cast<char*>(mdBufferData) + positionDataSize + texCoordDataSize, normalData.data(), normalDataSize);
	WriteBufferDescriptor(device, *vertexDataDescriptor, 0, vk::DescriptorType::eStorageBuffer, vertexDataBuffer); // binding = 0

	// write index data
	std::vector<int> startIndexData;
	std::vector<int> baseIndexData;
	for (const auto& element : scene.meshes[0].get()->GetSubMeshData()) {
		startIndexData.emplace_back(element.start);
		baseIndexData.emplace_back(element.base);
	}
	std::vector<int> indicesData;
	for (const auto& element : scene.meshes[0].get()->GetIndexData()) {
		indicesData.emplace_back(static_cast<int>(element));
	}
	size_t startIndexDataSize = startIndexData.size() * sizeof(int);
	size_t baseIndexDataSize = baseIndexData.size() * sizeof(int);
	size_t indicesDataSize = indicesData.size() * sizeof(int);
	size_t indexTotalSize = startIndexDataSize + baseIndexDataSize + indicesDataSize;
	indexDataBuffer = BufferBuilder(device, renderer->GetMemoryAllocator())
		.WithBufferUsage(vk::BufferUsageFlagBits::eStorageBuffer)
		.WithPersistentMapping()
		.Build(indexTotalSize, "Index Data Buffer");
	void* idBufferData = (void*)indexDataBuffer.Data();
	memcpy(idBufferData, startIndexData.data(), startIndexDataSize);
	memcpy(static_cast<char*>(idBufferData) + startIndexDataSize, baseIndexData.data(), baseIndexDataSize);
	memcpy(static_cast<char*>(idBufferData) + startIndexDataSize + baseIndexDataSize, indicesData.data(), indicesDataSize);
	WriteBufferDescriptor(device, *vertexDataDescriptor, 1, vk::DescriptorType::eStorageBuffer, indexDataBuffer); // binding = 1
}

void RayTracingWithPBR::CreateSkyboxBuffer(vk::Device device, vk::DescriptorPool pool) {
	// skybox cubemap
	cubeMap = LoadCubemap(
		"Cubemap/skyrender0004.png", "Cubemap/skyrender0001.png",
		"Cubemap/skyrender0003.png", "Cubemap/skyrender0006.png",
		"Cubemap/skyrender0002.png", "Cubemap/skyrender0005.png",
		"Skybox"
	);
	cubeMapLayout = DescriptorSetLayoutBuilder(device)
		.WithImageSamplers(0, 1)
		.Build("Skybox Layout");
	cubeMapDescriptor = CreateDescriptorSet(device, pool, *cubeMapLayout);
	WriteImageDescriptor(device, *cubeMapDescriptor, 0, cubeMap.get()->GetDefaultView(), *defaultSampler);
}

void RayTracingWithPBR::CreateLightBuffer(vk::Device device, vk::DescriptorPool pool) {
	int parallelCount = worldLight.GetParalleLight().size();
	int dotCount = worldLight.GetDotLight().size();
	size_t parallelSize = parallelCount * sizeof(ParallelLight);
	size_t dotSize = dotCount * sizeof(DotLight);

	lightLayout = DescriptorSetLayoutBuilder(device)
		.WithUniformBuffers(0, 1)
		.WithStorageBuffers(1, 1)
		.WithStorageBuffers(2, 1)
		.Build("Light Layout");
	lightDescriptor = CreateDescriptorSet(device, pool, *lightLayout);

	lightCount = BufferBuilder(device, renderer->GetMemoryAllocator())
		.WithBufferUsage(vk::BufferUsageFlagBits::eUniformBuffer)
		.WithPersistentMapping()
		.Build(sizeof(int) * 2, "Light Count Buffer");
	int* lightCountData = (int*)lightCount.Data();
	lightCountData[0] = parallelCount;
	lightCountData[1] = dotCount;
	WriteBufferDescriptor(device, *lightDescriptor, 0, vk::DescriptorType::eUniformBuffer, lightCount); // binding = 0

	// avoid building with 0 size
	if (parallelSize) {
		parallelBuffer = BufferBuilder(device, renderer->GetMemoryAllocator())
			.WithBufferUsage(vk::BufferUsageFlagBits::eStorageBuffer)
			.WithPersistentMapping()
			.Build(parallelSize, "Parallel Light Buffer");
		void* lightBufferData = (void*)parallelBuffer.Data();
		memcpy(lightBufferData, worldLight.GetParalleLight().data(), parallelSize);
		WriteBufferDescriptor(device, *lightDescriptor, 1, vk::DescriptorType::eStorageBuffer, parallelBuffer); // binding = 1
	}

	if (dotSize) {
		dotBuffer = BufferBuilder(device, renderer->GetMemoryAllocator())
			.WithBufferUsage(vk::BufferUsageFlagBits::eStorageBuffer)
			.WithPersistentMapping()
			.Build(dotSize, "Dot Light Buffer");
		void* dotBufferData = (void*)dotBuffer.Data();
		memcpy(dotBufferData, worldLight.GetDotLight().data(), dotSize);
		WriteBufferDescriptor(device, *lightDescriptor, 2, vk::DescriptorType::eStorageBuffer, dotBuffer); // binding = 2
	}
}

void RayTracingWithPBR::SwitchRenderType() {
	renderType = static_cast<RenderType>(static_cast<int>(renderType) + 1);
	if (renderType == MAX_SIZE) renderType = Default;
}

uint32_t RayTracingWithPBR::FindMemoryType(vk::PhysicalDevice physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
	vk::PhysicalDeviceMemoryProperties memProperties;
	physicalDevice.getMemoryProperties(&memProperties);

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type!");
}

//	This function can get the vk::image RGBA, but takes a lot of time
std::vector<Vector4> RayTracingWithPBR::ReadImageRGB(vk::Device device, vk::Queue queue, vk::CommandPool commandPool, 
	vk::Image image, vk::Format format, uint32_t width, uint32_t height) {
	vk::DeviceSize imageSize = width * height * 4; // 4 bytes per pixel (RGBA)

	// Create a buffer to store the image data
	vk::BufferCreateInfo bufferInfo = {};
	bufferInfo.size = imageSize;
	bufferInfo.usage = vk::BufferUsageFlagBits::eTransferDst;
	bufferInfo.sharingMode = vk::SharingMode::eExclusive;

	vk::Buffer buffer = device.createBuffer(bufferInfo);

	// Allocate memory for the buffer
	vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buffer);
	vk::MemoryAllocateInfo allocInfo = {};
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = FindMemoryType(renderer->GetPhysicalDevice(), memRequirements.memoryTypeBits,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

	vk::DeviceMemory bufferMemory = device.allocateMemory(allocInfo);
	device.bindBufferMemory(buffer, bufferMemory, 0);

	// Create a command buffer for the copy operation
	vk::CommandBufferAllocateInfo cmdAllocInfo = {};
	cmdAllocInfo.level = vk::CommandBufferLevel::ePrimary;
	cmdAllocInfo.commandPool = commandPool;
	cmdAllocInfo.commandBufferCount = 1;

	vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(cmdAllocInfo)[0];

	vk::CommandBufferBeginInfo beginInfo = {};
	commandBuffer.begin(beginInfo);

	// Transition the image layout to transfer source optimal
	vk::ImageMemoryBarrier barrier = {};
	barrier.oldLayout = vk::ImageLayout::eUndefined;
	barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;
	barrier.srcAccessMask = vk::AccessFlagBits::eNoneKHR;
	barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

	commandBuffer.pipelineBarrier(
		vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
		vk::DependencyFlags(), nullptr, nullptr, barrier);

	// Define the region to copy
	vk::BufferImageCopy copyRegion = {};
	copyRegion.bufferOffset = 0;
	copyRegion.bufferRowLength = 0;
	copyRegion.bufferImageHeight = 0;
	copyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
	copyRegion.imageSubresource.mipLevel = 0;
	copyRegion.imageSubresource.baseArrayLayer = 0;
	copyRegion.imageSubresource.layerCount = 1;
	copyRegion.imageOffset = vk::Offset3D{ 0, 0, 0 };
	copyRegion.imageExtent = vk::Extent3D{ width, height, 1 };

	commandBuffer.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, buffer, copyRegion);

	// Transition the image layout back to original layout
	barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
	barrier.newLayout = vk::ImageLayout::eGeneral; // or any original layout
	barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

	commandBuffer.pipelineBarrier(
		vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
		vk::DependencyFlags(), nullptr, nullptr, barrier);

	commandBuffer.end();

	// Submit the command buffer
	vk::SubmitInfo submitInfo = {};
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	queue.submit(submitInfo, vk::Fence{});
	queue.waitIdle();

	// Map the buffer memory and read the data
	void* data = device.mapMemory(bufferMemory, 0, imageSize, {});
	std::vector<uint8_t> imageData(imageSize);
	memcpy(imageData.data(), data, static_cast<size_t>(imageSize));
	device.unmapMemory(bufferMemory);

	// Clean up
	device.freeMemory(bufferMemory);
	device.destroyBuffer(buffer);

	// Process the image data (for example, print RGB values)
	std::vector<Vector4> RGBA;
	for (uint32_t i = 0; i < width * height; ++i) {
		uint8_t r = imageData[i * 4];
		uint8_t g = imageData[i * 4 + 1];
		uint8_t b = imageData[i * 4 + 2];
		uint8_t a = imageData[i * 4 + 3];
		RGBA.push_back(Vector4(r, g, b, a));
	}
	return RGBA;
}