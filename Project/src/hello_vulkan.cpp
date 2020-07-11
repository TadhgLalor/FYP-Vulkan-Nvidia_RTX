#include <sstream>
#include <vulkan/vulkan.hpp>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_inverse.hpp"


#include "descriptorsets_vkpp.hpp"
#include "hello_vulkan.h"
#include "manipulator.h"
#include "obj_loader.h"
#include "pipeline_vkpp.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "commands_vkpp.hpp"
#include "renderpass_vkpp.hpp"
#include "stb_image.h"
#include "utilities_vkpp.hpp"

// Holding the camera matrices
struct CameraMatrices
{
  glm::mat4 view;
  glm::mat4 proj;
  glm::mat4 viewInverse;
  // #VKRay
  glm::mat4 projInverse;
};

// OBJ representation of a vertex
struct Vertex
{
  glm::vec3 pos;
  glm::vec3 nrm;
  glm::vec3 color;
  glm::vec2 texCoord;
  int       matID = 0;
};

//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void HelloVulkan::init(const vk::Device&         device,
                       const vk::PhysicalDevice& physicalDevice,
                       uint32_t                  queueFamily,
                       const vk::Extent2D&       size)
{
  m_alloc.init(device, physicalDevice);
  m_device         = device;
  m_physicalDevice = physicalDevice;
  m_queueIndex     = queueFamily;
  m_size           = size;
  m_debug.setup(m_device);
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void HelloVulkan::updateUniformBuffer()
{
  const float aspectRatio = m_size.width / static_cast<float>(m_size.height);

  CameraMatrices ubo = {};
  ubo.view           = CameraManip.getMatrix();
  ubo.proj           = glm::perspective(glm::radians(65.0f), aspectRatio, 0.1f, 1000.0f);
  ubo.proj[1][1] *= -1;  // Inverting Y for Vulkan
  ubo.viewInverse = glm::inverse(ubo.view);
  // #VKRay
  ubo.projInverse = glm::inverse(ubo.proj);

  void* data = m_device.mapMemory(m_cameraMat.allocation, 0, sizeof(ubo));
  memcpy(data, &ubo, sizeof(ubo));
  m_device.unmapMemory(m_cameraMat.allocation);
}

//--------------------------------------------------------------------------------------------------
// Describing the layout pushed when rendering
//
void HelloVulkan::createDescriptorSetLayout()
{
  using vkDS     = vk::DescriptorSetLayoutBinding;
  using vkDT     = vk::DescriptorType;
  using vkSS     = vk::ShaderStageFlagBits;
  uint32_t nbTxt = static_cast<uint32_t>(m_textures.size());
  uint32_t nbObj = static_cast<uint32_t>(m_objModel.size());

  // Camera matrices (binding = 0)
  m_descSetLayoutBind.emplace_back(
      vkDS(0, vkDT::eUniformBuffer, 1, vkSS::eVertex | vkSS::eRaygenNV));
  // Materials (binding = 1)
  m_descSetLayoutBind.emplace_back(
      vkDS(1, vkDT::eStorageBuffer, nbObj, vkSS::eVertex | vkSS::eFragment | vkSS::eClosestHitNV));
  // Scene description (binding = 2)
  m_descSetLayoutBind.emplace_back(  //
      vkDS(2, vkDT::eStorageBuffer, 1, vkSS::eVertex | vkSS::eFragment | vkSS::eClosestHitNV));
  // Textures (binding = 3)
  m_descSetLayoutBind.emplace_back(
      vkDS(3, vkDT::eCombinedImageSampler, nbTxt, vkSS::eFragment | vkSS::eClosestHitNV));
  // Storing vertices (binding = 4)
  m_descSetLayoutBind.emplace_back(  //
      vkDS(4, vkDT::eStorageBuffer, nbObj, vkSS::eClosestHitNV));
  // Storing indices (binding = 5)
  m_descSetLayoutBind.emplace_back(  //
      vkDS(5, vkDT::eStorageBuffer, nbObj, vkSS::eClosestHitNV));

  m_descSetLayout = nvvkpp::util::createDescriptorSetLayout(m_device, m_descSetLayoutBind);
  m_descPool      = nvvkpp::util::createDescriptorPool(m_device, m_descSetLayoutBind, 1);
  m_descSet       = nvvkpp::util::createDescriptorSet(m_device, m_descPool, m_descSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Setting up the buffers in the descriptor set
//
void HelloVulkan::updateDescriptorSet()
{
  std::vector<vk::WriteDescriptorSet> writes;

  // Camera matrices and scene description
  vk::DescriptorBufferInfo dbiUnif{m_cameraMat.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet, m_descSetLayoutBind[0], &dbiUnif));
  vk::DescriptorBufferInfo dbiSceneDesc{m_sceneDesc.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet, m_descSetLayoutBind[2], &dbiSceneDesc));

  // All material buffers, 1 buffer per OBJ
  std::vector<vk::DescriptorBufferInfo> dbiMat;
  std::vector<vk::DescriptorBufferInfo> dbiVert;
  std::vector<vk::DescriptorBufferInfo> dbiIdx;
  for(size_t i = 0; i < m_objModel.size(); ++i)
  {
    dbiMat.push_back({m_objModel[i].matColorBuffer.buffer, 0, VK_WHOLE_SIZE});
    dbiVert.push_back({m_objModel[i].vertexBuffer.buffer, 0, VK_WHOLE_SIZE});
    dbiIdx.push_back({m_objModel[i].indexBuffer.buffer, 0, VK_WHOLE_SIZE});
  }
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet, m_descSetLayoutBind[1], dbiMat.data()));
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet, m_descSetLayoutBind[4], dbiVert.data()));
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet, m_descSetLayoutBind[5], dbiIdx.data()));

  // All texture samplers
  std::vector<vk::DescriptorImageInfo> diit;
  for(size_t i = 0; i < m_textures.size(); ++i)
  {
    diit.push_back(m_textures[i].descriptor);
  }
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet, m_descSetLayoutBind[3], diit.data()));

  // Writing the information
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Creating the pipeline layout
//
void HelloVulkan::createGraphicsPipeline(const vk::RenderPass& renderPass)
{
  using vkSS = vk::ShaderStageFlagBits;

  vk::PushConstantRange pushConstantRanges = {vkSS::eVertex | vkSS::eFragment, 0,
                                              sizeof(ObjPushConstant)};

  // Creating the Pipeline Layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  vk::DescriptorSetLayout      descSetLayout(m_descSetLayout);
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&descSetLayout);
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstantRanges);
  m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Creating the Pipeline
  nvvkpp::GraphicsPipelineGenerator gpb(m_device, m_pipelineLayout, m_offscreenRenderPass);
  gpb.depthStencilState = {true};
  gpb.addShader(nvvkpp::util::readFile("shaders/vert_shader.vert.spv"), vkSS::eVertex);
  gpb.addShader(nvvkpp::util::readFile("shaders/frag_shader.frag.spv"), vkSS::eFragment);
  gpb.vertexInputState.bindingDescriptions   = {{0, sizeof(Vertex)}};
  gpb.vertexInputState.attributeDescriptions = {
      {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)},
      {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, nrm)},
      {2, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)},
      {3, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord)},
      {4, 0, vk::Format::eR32Sint, offsetof(Vertex, matID)}};

  m_graphicsPipeline = gpb.create();
  m_debug.setObjectName(m_graphicsPipeline, "Graphics");
}

//--------------------------------------------------------------------------------------------------
// Loading the OBJ file and setting up all buffers
//
void HelloVulkan::loadModel(const std::string& filename, glm::mat4 transform)
{
  using vkBU = vk::BufferUsageFlagBits;

  ObjLoader<Vertex> loader;
  loader.loadModel(filename);

  // Converting from Srgb to linear
  for(auto& m : loader.m_materials)
  {
    m.ambient  = glm::pow(m.ambient, glm::vec3(2.2f));
    m.diffuse  = glm::pow(m.diffuse, glm::vec3(2.2f));
    m.specular = glm::pow(m.specular, glm::vec3(2.2f));
  }

  ObjInstance instance;
  instance.objIndex    = static_cast<uint32_t>(m_objModel.size());
  instance.transform   = transform;
  instance.transformIT = glm::inverseTranspose(transform);
  instance.txtOffset   = static_cast<uint32_t>(m_textures.size());

  ObjModel model;
  model.nbIndices  = static_cast<uint32_t>(loader.m_indices.size());
  model.nbVertices = static_cast<uint32_t>(loader.m_vertices.size());

  // Create the buffers on Device and copy vertices, indices and materials
  nvvkpp::SingleCommandBuffer cmdBufGet(m_device, m_queueIndex);
  vk::CommandBuffer           cmdBuf = cmdBufGet.createCommandBuffer();
  model.vertexBuffer =
      m_alloc.createBuffer(cmdBuf, loader.m_vertices, vkBU::eVertexBuffer | vkBU::eStorageBuffer);
  model.indexBuffer =
      m_alloc.createBuffer(cmdBuf, loader.m_indices, vkBU::eIndexBuffer | vkBU::eStorageBuffer);
  model.matColorBuffer = m_alloc.createBuffer(cmdBuf, loader.m_materials, vkBU::eStorageBuffer);
  // Creates all textures found
  createTextureImages(cmdBuf, loader.m_textures);
  cmdBufGet.flushCommandBuffer(cmdBuf);
  m_alloc.flushStaging();

  std::string objNb = std::to_string(instance.objIndex);
  m_debug.setObjectName(model.vertexBuffer.buffer, (std::string("vertex_" + objNb).c_str()));
  m_debug.setObjectName(model.indexBuffer.buffer, (std::string("index_" + objNb).c_str()));
  m_debug.setObjectName(model.matColorBuffer.buffer, (std::string("mat_" + objNb).c_str()));

  m_objModel.emplace_back(model);
  m_objInstance.emplace_back(instance);
}


//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the camera matrices
// - Buffer is host visible
//
void HelloVulkan::createUniformBuffer()
{
  using vkBU = vk::BufferUsageFlagBits;
  using vkMP = vk::MemoryPropertyFlagBits;

  m_cameraMat = m_alloc.createBuffer(sizeof(CameraMatrices), vkBU::eUniformBuffer,
                                     vkMP::eHostVisible | vkMP::eHostCoherent);
  m_debug.setObjectName(m_cameraMat.buffer, "cameraMat");
}

//--------------------------------------------------------------------------------------------------
// Create a storage buffer containing the description of the scene elements
// - Which geometry is used by which instance
// - Transformation
// - Offset for texture
//
void HelloVulkan::createSceneDescriptionBuffer()
{
  using vkBU = vk::BufferUsageFlagBits;
  nvvkpp::SingleCommandBuffer cmdGen(m_device, m_queueIndex);

  auto cmdBuf = cmdGen.createCommandBuffer();
  m_sceneDesc = m_alloc.createBuffer(cmdBuf, m_objInstance, vkBU::eStorageBuffer);
  cmdGen.flushCommandBuffer(cmdBuf);
  m_alloc.flushStaging();
  m_debug.setObjectName(m_sceneDesc.buffer, "sceneDesc");
}

//--------------------------------------------------------------------------------------------------
// Creating all textures and samplers
//
void HelloVulkan::createTextureImages(const vk::CommandBuffer&        cmdBuf,
                                      const std::vector<std::string>& textures)
{
  using vkIU = vk::ImageUsageFlagBits;

  vk::SamplerCreateInfo samplerCreateInfo{
      {}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
  samplerCreateInfo.setMaxLod(FLT_MAX);
  vk::Format format = vk::Format::eR8G8B8A8Srgb;

  // If no textures are present, create a dummy one to accommodate the pipeline layout
  if(textures.empty() && m_textures.empty())
  {
    nvvkTexture texture;

    glm::u8vec4*   color           = new glm::u8vec4(255, 255, 255, 255);
    vk::DeviceSize bufferSize      = sizeof(glm::u8vec4);
    auto           imgSize         = vk::Extent2D(1, 1);
    auto           imageCreateInfo = nvvkpp::image::create2DInfo(imgSize, format);

    // Creating the VKImage
    texture = m_alloc.createImage(cmdBuf, bufferSize, color, imageCreateInfo);
    // Setting up the descriptor used by the shader
    texture.descriptor =
        nvvkpp::image::create2DDescriptor(m_device, texture.image, samplerCreateInfo, format);
    // The image format must be in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    nvvkpp::image::setImageLayout(cmdBuf, texture.image, vk::ImageLayout::eUndefined,
                                  vk::ImageLayout::eShaderReadOnlyOptimal);
    m_textures.push_back(texture);
  }
  else
  {
    // Uploading all images
    for(const auto& texture : textures)
    {
      std::stringstream o;
      int               texWidth, texHeight, texChannels;
      o << "../media/textures/" << texture;

      stbi_uc* pixels =
          stbi_load(o.str().c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

      // Handle failure
      if(!pixels)
      {
        texWidth = texHeight = 1;
        texChannels          = 4;
        glm::u8vec4* color   = new glm::u8vec4(255, 0, 255, 255);
        pixels               = reinterpret_cast<stbi_uc*>(color);
      }

      vk::DeviceSize bufferSize = static_cast<uint64_t>(texWidth) * texHeight * sizeof(glm::u8vec4);
      auto           imgSize    = vk::Extent2D(texWidth, texHeight);
      auto imageCreateInfo = nvvkpp::image::create2DInfo(imgSize, format, vkIU::eSampled, true);

      nvvkTexture texture;
      texture = m_alloc.createImage(cmdBuf, bufferSize, pixels, imageCreateInfo);

      nvvkpp::image::generateMipmaps(cmdBuf, texture.image, format, imgSize,
                                     imageCreateInfo.mipLevels);
      texture.descriptor =
          nvvkpp::image::create2DDescriptor(m_device, texture.image, samplerCreateInfo, format);
      m_textures.push_back(texture);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void HelloVulkan::destroyResources()
{
  m_device.destroy(m_graphicsPipeline);
  m_device.destroy(m_pipelineLayout);
  m_device.destroy(m_descPool);
  m_device.destroy(m_descSetLayout);
  m_alloc.destroy(m_cameraMat);
  m_alloc.destroy(m_sceneDesc);

  for(auto& m : m_objModel)
  {
    m_alloc.destroy(m.vertexBuffer);
    m_alloc.destroy(m.indexBuffer);
    m_alloc.destroy(m.matColorBuffer);
  }

  for(auto& t : m_textures)
  {
    m_alloc.destroy(t);
  }

  //#Post
  m_device.destroy(m_postPipeline);
  m_device.destroy(m_postPipelineLayout);
  m_device.destroy(m_postDescPool);
  m_device.destroy(m_postDescSetLayout);
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);
  m_device.destroy(m_offscreenRenderPass);
  m_device.destroy(m_offscreenFramebuffer);

  // #VKRay
  m_rtBuilder.destroy();
  m_device.destroy(m_rtDescPool);
  m_device.destroy(m_rtDescSetLayout);
  m_device.destroy(m_rtPipeline);
  m_device.destroy(m_rtPipelineLayout);
  m_alloc.destroy(m_rtSBTBuffer);
}

//--------------------------------------------------------------------------------------------------
// Drawing the scene in raster mode
//
void HelloVulkan::rasterize(const vk::CommandBuffer& cmdBuf)
{
  using vkPBP = vk::PipelineBindPoint;
  using vkSS  = vk::ShaderStageFlagBits;
  vk::DeviceSize offset{0};

  m_debug.beginLabel(cmdBuf, "Rasterize");

  // Dynamic Viewport
  cmdBuf.setViewport(0, {vk::Viewport(0, 0, (float)m_size.width, (float)m_size.height, 0, 1)});
  cmdBuf.setScissor(0, {{{0, 0}, {m_size.width, m_size.height}}});


  // Drawing all triangles
  cmdBuf.bindPipeline(vkPBP::eGraphics, m_graphicsPipeline);
  cmdBuf.bindDescriptorSets(vkPBP::eGraphics, m_pipelineLayout, 0, {m_descSet}, {});
  for(int i = 0; i < m_objInstance.size(); ++i)
  {
    auto& inst                = m_objInstance[i];
    auto& model               = m_objModel[inst.objIndex];
    m_pushConstant.instanceId = i;  // Telling which instance is drawn
    cmdBuf.pushConstants<ObjPushConstant>(m_pipelineLayout, vkSS::eVertex | vkSS::eFragment, 0,
                                          m_pushConstant);

    cmdBuf.bindVertexBuffers(0, 1, &model.vertexBuffer.buffer, &offset);
    cmdBuf.bindIndexBuffer(model.indexBuffer.buffer, 0, vk::IndexType::eUint32);
    cmdBuf.drawIndexed(model.nbIndices, 1, 0, 0, 0);
  }
  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void HelloVulkan::resize(const vk::Extent2D& size)
{
  m_size = size;
  createOffscreenRender();
  updatePostDescriptorSet();
  updateRtDescriptorSet();
}


//////////////////////////////////////////////////////////////////////////
// Post-processing
//////////////////////////////////////////////////////////////////////////


//--------------------------------------------------------------------------------------------------
// Creating an offscreen frame buffer and the associated render pass
//
void HelloVulkan::createOffscreenRender()
{
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);

  // Creating the color image
  auto colorCreateInfo = nvvkpp::image::create2DInfo(m_size, m_offscreenColorFormat,
                                                     vk::ImageUsageFlagBits::eColorAttachment
                                                         | vk::ImageUsageFlagBits::eSampled
                                                         | vk::ImageUsageFlagBits::eStorage);
  m_offscreenColor     = m_alloc.createImage(colorCreateInfo);


  m_offscreenColor.descriptor =
      nvvkpp::image::create2DDescriptor(m_device, m_offscreenColor.image, vk::SamplerCreateInfo{},
                                        m_offscreenColorFormat, vk::ImageLayout::eGeneral);

  // Creating the depth buffer
  auto depthCreateInfo =
      nvvkpp::image::create2DInfo(m_size, m_offscreenDepthFormat,
                                  vk::ImageUsageFlagBits::eDepthStencilAttachment);
  m_offscreenDepth = m_alloc.createImage(depthCreateInfo);

  vk::ImageViewCreateInfo depthStencilView;
  depthStencilView.setViewType(vk::ImageViewType::e2D);
  depthStencilView.setFormat(m_offscreenDepthFormat);
  depthStencilView.setSubresourceRange({vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1});
  depthStencilView.setImage(m_offscreenDepth.image);
  m_offscreenDepth.descriptor.imageView = m_device.createImageView(depthStencilView);

  // Setting the image layout for both color and depth
  {
    nvvkpp::SingleCommandBuffer genCmdBuf(m_device, m_queueIndex);
    auto                        cmdBuf = genCmdBuf.createCommandBuffer();
    nvvkpp::image::setImageLayout(cmdBuf, m_offscreenColor.image, vk::ImageLayout::eUndefined,
                                  vk::ImageLayout::eGeneral);
    nvvkpp::image::setImageLayout(cmdBuf, m_offscreenDepth.image, vk::ImageAspectFlagBits::eDepth,
                                  vk::ImageLayout::eUndefined,
                                  vk::ImageLayout::eDepthStencilAttachmentOptimal);

    genCmdBuf.flushCommandBuffer(cmdBuf);
  }

  // Creating a renderpass for the offscreen
  if(!m_offscreenRenderPass)
  {
    m_offscreenRenderPass =
        nvvkpp::util::createRenderPass(m_device, {m_offscreenColorFormat}, m_offscreenDepthFormat,
                                       1, true, true, vk::ImageLayout::eGeneral,
                                       vk::ImageLayout::eGeneral);
  }


  // Creating the frame buffer for offscreen
  std::vector<vk::ImageView> attachments = {m_offscreenColor.descriptor.imageView,
                                            m_offscreenDepth.descriptor.imageView};

  m_device.destroy(m_offscreenFramebuffer);
  vk::FramebufferCreateInfo info;
  info.setRenderPass(m_offscreenRenderPass);
  info.setAttachmentCount(2);
  info.setPAttachments(attachments.data());
  info.setWidth(m_size.width);
  info.setHeight(m_size.height);
  info.setLayers(1);
  m_offscreenFramebuffer = m_device.createFramebuffer(info);
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void HelloVulkan::createPostPipeline(const vk::RenderPass& renderPass)
{
  // Push constants in the fragment shader
  vk::PushConstantRange pushConstantRanges = {vk::ShaderStageFlagBits::eFragment, 0, sizeof(float)};

  // Creating the pipeline layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&m_postDescSetLayout);
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstantRanges);
  m_postPipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Pipeline: completely generic, no vertices
  nvvkpp::GraphicsPipelineGenerator pipelineGenerator(m_device, m_postPipelineLayout, renderPass);
  pipelineGenerator.addShader(nvvkpp::util::readFile("shaders/passthrough.vert.spv"),
                              vk::ShaderStageFlagBits::eVertex);
  pipelineGenerator.addShader(nvvkpp::util::readFile("shaders/post.frag.spv"),
                              vk::ShaderStageFlagBits::eFragment);
  pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
  m_postPipeline = pipelineGenerator.create();
  m_debug.setObjectName(m_postPipeline, "post");
}

//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void HelloVulkan::createPostDescriptor()
{
  using vkDS = vk::DescriptorSetLayoutBinding;
  using vkDT = vk::DescriptorType;
  using vkSS = vk::ShaderStageFlagBits;

  m_postDescSetLayoutBind.emplace_back(vkDS(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
  m_postDescSetLayout = nvvkpp::util::createDescriptorSetLayout(m_device, m_postDescSetLayoutBind);
  m_postDescPool      = nvvkpp::util::createDescriptorPool(m_device, m_postDescSetLayoutBind);
  m_postDescSet = nvvkpp::util::createDescriptorSet(m_device, m_postDescPool, m_postDescSetLayout);
}


//--------------------------------------------------------------------------------------------------
// Update the output
//
void HelloVulkan::updatePostDescriptorSet()
{
  vk::WriteDescriptorSet writeDescriptorSets =
      nvvkpp::util::createWrite(m_postDescSet, m_postDescSetLayoutBind[0],
                                &m_offscreenColor.descriptor);
  m_device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Draw a full screen quad with the attached image
//
void HelloVulkan::drawPost(vk::CommandBuffer cmdBuf)
{
  m_debug.beginLabel(cmdBuf, "Post");

  cmdBuf.setViewport(0, {vk::Viewport(0, 0, (float)m_size.width, (float)m_size.height, 0, 1)});
  cmdBuf.setScissor(0, {{{0, 0}, {m_size.width, m_size.height}}});

  auto aspectRatio = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);
  cmdBuf.pushConstants<float>(m_postPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0,
                              aspectRatio);
  cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, m_postPipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_postPipelineLayout, 0,
                            m_postDescSet, {});
  cmdBuf.draw(3, 1, 0, 0);

  m_debug.endLabel(cmdBuf);
}

//////////////////////////////////////////////////////////////////////////
// Raytracing
//////////////////////////////////////////////////////////////////////////


//--------------------------------------------------------------------------------------------------
// Initialize Vulkan ray tracing
// #VKRay
void HelloVulkan::initRayTracing()
{
  // Requesting ray tracing properties
  auto properties = m_physicalDevice.getProperties2<vk::PhysicalDeviceProperties2,
                                                    vk::PhysicalDeviceRayTracingPropertiesNV>();
  m_rtProperties  = properties.get<vk::PhysicalDeviceRayTracingPropertiesNV>();

  m_rtBuilder.setup(m_device, m_physicalDevice, m_queueIndex);
}

//--------------------------------------------------------------------------------------------------
// Converting a OBJ primitive to the ray tracing geometry used for the BLAS
//
vk::GeometryNV HelloVulkan::objectToVkGeometryNV(const ObjModel& model)
{
  vk::GeometryTrianglesNV triangles;
  triangles.setVertexData(model.vertexBuffer.buffer);
  triangles.setVertexOffset(0);  // Start at the beginning of the buffer
  triangles.setVertexCount(model.nbVertices);
  triangles.setVertexStride(sizeof(Vertex));
  triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat);  // 3xfloat32 for vertices
  triangles.setIndexData(model.indexBuffer.buffer);
  triangles.setIndexOffset(0 * sizeof(uint32_t));
  triangles.setIndexCount(model.nbIndices);
  triangles.setIndexType(vk::IndexType::eUint32);  // 32-bit indices
  vk::GeometryDataNV geoData;
  geoData.setTriangles(triangles);
  vk::GeometryNV geometry;
  geometry.setGeometry(geoData);
  // Consider the geometry opaque for optimization
  geometry.setFlags(vk::GeometryFlagBitsNV::eOpaque);
  return geometry;
}

void HelloVulkan::createBottomLevelAS()
{
  // BLAS - Storing each primitive in a geometry
  std::vector<std::vector<vk::GeometryNV>> blas;
  blas.reserve(m_objModel.size());
  for(size_t i = 0; i < m_objModel.size(); i++)
  {
    auto geo = objectToVkGeometryNV(m_objModel[i]);
    // We could add more geometry in each BLAS, but we add only one for now
    blas.push_back({geo});
  }
  m_rtBuilder.buildBlas(blas);
}

void HelloVulkan::createTopLevelAS()
{
  std::vector<nvvkpp::RaytracingBuilder::Instance> tlas;
  tlas.reserve(m_objInstance.size());
  for(int i = 0; i < static_cast<int>(m_objInstance.size()); i++)
  {
    nvvkpp::RaytracingBuilder::Instance rayInst;
    rayInst.transform  = m_objInstance[i].transform;  // Position of the instance
    rayInst.instanceId = i;                           // gl_InstanceID
    rayInst.blasId     = m_objInstance[i].objIndex;
    rayInst.hitGroupId = 0;  // We will use the same hit group for all objects
    rayInst.flags      = vk::GeometryInstanceFlagBitsNV::eTriangleCullDisable;
    tlas.emplace_back(rayInst);
  }
  m_rtBuilder.buildTlas(tlas, vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace);
}

//--------------------------------------------------------------------------------------------------
// This descriptor set holds the Acceleration structure and the output image
//
void HelloVulkan::createRtDescriptorSet()
{
  using vkDT   = vk::DescriptorType;
  using vkSS   = vk::ShaderStageFlagBits;
  using vkDSLB = vk::DescriptorSetLayoutBinding;

  m_rtDescSetLayoutBind.emplace_back(
      vkDSLB(0, vkDT::eAccelerationStructureNV, 1, vkSS::eRaygenNV | vkSS::eClosestHitNV));  // TLAS
  m_rtDescSetLayoutBind.emplace_back(
      vkDSLB(1, vkDT::eStorageImage, 1, vkSS::eRaygenNV));  // Output image

  m_rtDescPool      = nvvkpp::util::createDescriptorPool(m_device, m_rtDescSetLayoutBind);
  m_rtDescSetLayout = nvvkpp::util::createDescriptorSetLayout(m_device, m_rtDescSetLayoutBind);
  m_rtDescSet       = m_device.allocateDescriptorSets({m_rtDescPool, 1, &m_rtDescSetLayout})[0];

  vk::WriteDescriptorSetAccelerationStructureNV descASInfo;
  descASInfo.setAccelerationStructureCount(1);
  descASInfo.setPAccelerationStructures(&m_rtBuilder.getAccelerationStructure());
  vk::DescriptorImageInfo imageInfo{
      {}, m_offscreenColor.descriptor.imageView, vk::ImageLayout::eGeneral};

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(
      nvvkpp::util::createWrite(m_rtDescSet, m_rtDescSetLayoutBind[0], &descASInfo));
  writes.emplace_back(nvvkpp::util::createWrite(m_rtDescSet, m_rtDescSetLayoutBind[1], &imageInfo));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Writes the output image to the descriptor set
// - Required when changing resolution
//
void HelloVulkan::updateRtDescriptorSet()
{
  using vkDT = vk::DescriptorType;

  // (1) Output buffer
  vk::DescriptorImageInfo imageInfo{
      {}, m_offscreenColor.descriptor.imageView, vk::ImageLayout::eGeneral};
  vk::WriteDescriptorSet wds{m_rtDescSet, 1, 0, 1, vkDT::eStorageImage, &imageInfo};
  m_device.updateDescriptorSets(wds, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void HelloVulkan::createRtPipeline()
{
  vk::ShaderModule raygenSM =
      nvvkpp::util::createShaderModule(m_device,  //
                                       nvvkpp::util::readFile("shaders/raytrace.rgen.spv"));
  vk::ShaderModule missSM =
      nvvkpp::util::createShaderModule(m_device,  //
                                       nvvkpp::util::readFile("shaders/raytrace.rmiss.spv"));

  // The second miss shader is invoked when a shadow ray misses the geometry. It
  // simply indicates that no occlusion has been found
  vk::ShaderModule shadowmissSM =
      nvvkpp::util::createShaderModule(m_device,  //
                                       nvvkpp::util::readFile("shaders/raytraceShadow.rmiss.spv"));

  std::vector<vk::PipelineShaderStageCreateInfo> stages;

  // Raygen
  vk::RayTracingShaderGroupCreateInfoNV rg{vk::RayTracingShaderGroupTypeNV::eGeneral,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
  stages.push_back({{}, vk::ShaderStageFlagBits::eRaygenNV, raygenSM, "main"});
  rg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(rg);
  // Miss
  vk::RayTracingShaderGroupCreateInfoNV mg{vk::RayTracingShaderGroupTypeNV::eGeneral,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
  stages.push_back({{}, vk::ShaderStageFlagBits::eMissNV, missSM, "main"});
  mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(mg);
  // Shadow Miss
  stages.push_back({{}, vk::ShaderStageFlagBits::eMissNV, shadowmissSM, "main"});
  mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(mg);

  // Hit Group - Closest Hit + AnyHit
  vk::ShaderModule chitSM =
      nvvkpp::util::createShaderModule(m_device,  //
                                       nvvkpp::util::readFile("shaders/raytrace.rchit.spv"));

  vk::RayTracingShaderGroupCreateInfoNV hg{vk::RayTracingShaderGroupTypeNV::eTrianglesHitGroup,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
  stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitNV, chitSM, "main"});
  hg.setClosestHitShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(hg);

  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;

  // Push constant: we want to be able to update constants used by the shaders
  vk::PushConstantRange pushConstant{vk::ShaderStageFlagBits::eRaygenNV
                                         | vk::ShaderStageFlagBits::eClosestHitNV
                                         | vk::ShaderStageFlagBits::eMissNV,
                                     0, sizeof(RtPushConstant)};
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstant);

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<vk::DescriptorSetLayout> rtDescSetLayouts = {m_rtDescSetLayout, m_descSetLayout};
  pipelineLayoutCreateInfo.setSetLayoutCount(static_cast<uint32_t>(rtDescSetLayouts.size()));
  pipelineLayoutCreateInfo.setPSetLayouts(rtDescSetLayouts.data());

  m_rtPipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  vk::RayTracingPipelineCreateInfoNV rayPipelineInfo;
  rayPipelineInfo.setStageCount(static_cast<uint32_t>(stages.size()));  // Stages are shaders
  rayPipelineInfo.setPStages(stages.data());

  rayPipelineInfo.setGroupCount(static_cast<uint32_t>(
      m_rtShaderGroups.size()));  // 1-raygen, n-miss, n-(hit[+anyhit+intersect])
  rayPipelineInfo.setPGroups(m_rtShaderGroups.data());

  // The ray tracing process can shoot rays from the camera, and a shadow ray can be shot from the
  // hit points of the camera rays, hence a recursion level of 2. This number should be kept as low
  // as possible for performance reasons. Even recursive ray tracing should be flattened into a loop
  // in the ray generation to avoid deep recursion.
  rayPipelineInfo.setMaxRecursionDepth(2);  // Ray depth
  rayPipelineInfo.setLayout(m_rtPipelineLayout);
  m_rtPipeline = m_device.createRayTracingPipelineNV({}, rayPipelineInfo);

  m_device.destroy(raygenSM);
  m_device.destroy(missSM);
  m_device.destroy(shadowmissSM);
  m_device.destroy(chitSM);
}

//--------------------------------------------------------------------------------------------------
// The Shader Binding Table (SBT)
// - getting all shader handles and writing them in a SBT buffer
// - Besides exception, this could be always done like this
//   See how the SBT buffer is used in run()
//
void HelloVulkan::createRtShaderBindingTable()
{
  auto groupCount =
      static_cast<uint32_t>(m_rtShaderGroups.size());               // 3 shaders: raygen, miss, chit
  uint32_t groupHandleSize = m_rtProperties.shaderGroupHandleSize;  // Size of a program identifier

  // Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
  uint32_t sbtSize = groupCount * groupHandleSize;

  std::vector<uint8_t> shaderHandleStorage(sbtSize);
  m_device.getRayTracingShaderGroupHandlesNV(m_rtPipeline, 0, groupCount, sbtSize,
                                             shaderHandleStorage.data());
  // Write the handles in the SBT
  nvvkpp::SingleCommandBuffer genCmdBuf(m_device, m_queueIndex);
  vk::CommandBuffer           cmdBuf = genCmdBuf.createCommandBuffer();

  m_rtSBTBuffer =
      m_alloc.createBuffer(cmdBuf, shaderHandleStorage, vk::BufferUsageFlagBits::eRayTracingNV);
  m_debug.setObjectName(m_rtSBTBuffer.buffer, "SBT");


  genCmdBuf.flushCommandBuffer(cmdBuf);

  m_alloc.flushStaging();
}

//--------------------------------------------------------------------------------------------------
// Ray Tracing the scene
//
void HelloVulkan::raytrace(const vk::CommandBuffer& cmdBuf, const glm::vec4& clearColor)
{
  m_debug.beginLabel(cmdBuf, "Ray trace");
  // Initializing push constant values
  m_rtPushConstants.clearColor     = clearColor;
  m_rtPushConstants.lightPosition  = m_pushConstant.lightPosition;
  m_rtPushConstants.lightIntensity = m_pushConstant.lightIntensity;
  m_rtPushConstants.lightType      = m_pushConstant.lightType;

  cmdBuf.bindPipeline(vk::PipelineBindPoint::eRayTracingNV, m_rtPipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingNV, m_rtPipelineLayout, 0,
                            {m_rtDescSet, m_descSet}, {});
  cmdBuf.pushConstants<RtPushConstant>(m_rtPipelineLayout,
                                       vk::ShaderStageFlagBits::eRaygenNV
                                           | vk::ShaderStageFlagBits::eClosestHitNV
                                           | vk::ShaderStageFlagBits::eMissNV,
                                       0, m_rtPushConstants);

  vk::DeviceSize progSize = m_rtProperties.shaderGroupHandleSize;  // Size of a program identifier
  vk::DeviceSize rayGenOffset   = 0u * progSize;  // Start at the beginning of m_sbtBuffer
  vk::DeviceSize missOffset     = 1u * progSize;  // Jump over raygen
  vk::DeviceSize missStride     = progSize;
  vk::DeviceSize hitGroupOffset = 3u * progSize;  // Jump over the raygen and 2 miss shaders
  vk::DeviceSize hitGroupStride = progSize;

  // m_sbtBuffer holds all the shader handles: raygen, n-miss, hit...
  cmdBuf.traceRaysNV(m_rtSBTBuffer.buffer, rayGenOffset,                    //
                     m_rtSBTBuffer.buffer, missOffset, missStride,          //
                     m_rtSBTBuffer.buffer, hitGroupOffset, hitGroupStride,  //
                     m_rtSBTBuffer.buffer, 0, 0,                            //
                     m_size.width, m_size.height,                           //
                     1);                                                    // depth

  m_debug.endLabel(cmdBuf);
}
