/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once
#include <set>
#include <vulkan/vulkan.hpp>

#include "GLFW/glfw3.h"
#include "GLFW/glfw3native.h"
#include "context_vkpp.hpp"
#include "swapchain_vkpp.hpp"

namespace nvvkpp {
//--------------------------------------------------------------------------------------------------
/**
 This is the base class for many examples. It does the basic for calling the initialization of
 Vulkan, the creation of the logical device, but also is a placeholder for the render passes and
 the swapchain
*/

class AppBase
{
public:
  AppBase()          = default;
  virtual ~AppBase() = default;


  void setupVulkan(nvvkpp::ContextCreateInfo deviceInfo, GLFWwindow* window)
  {
    // Creating the Vulkan instance and device
    m_vkctx.initInstance(deviceInfo);
    m_instance = m_vkctx.m_instance;

    // Create Window Surface
    {
      VkSurfaceKHR surface;
      auto         err = glfwCreateWindowSurface(m_vkctx.m_instance, window, nullptr, &surface);
      assert(err == VK_SUCCESS);
      m_surface = surface;
    }

    // Find all compatible devices
    auto compatibleDevices = m_vkctx.getCompatibleDevices(deviceInfo);
    assert(!compatibleDevices.empty());

    // Use a compatible device
    m_vkctx.initDevice(compatibleDevices[0], deviceInfo);

    setup(m_vkctx.m_device, m_vkctx.m_physicalDevice, m_vkctx.m_queueGCT.familyIndex);
  }

  //--------------------------------------------------------------------------------------------------
  // Setup the low level Vulkan for various operations
  //
  void setup(const vk::Device&         device,
             const vk::PhysicalDevice& physicalDevice,
             uint32_t                  graphicsQueueIndex)
  {
    m_device             = device;
    m_physicalDevice     = physicalDevice;
    m_graphicsQueueIndex = graphicsQueueIndex;
    m_queue              = m_device.getQueue(m_graphicsQueueIndex, 0);
    m_cmdPool            = m_device.createCommandPool(
        {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, graphicsQueueIndex});
    m_pipelineCache = device.createPipelineCache(vk::PipelineCacheCreateInfo());
  }


  //--------------------------------------------------------------------------------------------------
  // To call on exit
  //
  void destroy()
  {
    m_device.waitIdle();

    m_device.destroyRenderPass(m_renderPass);
    m_device.destroyImageView(m_depthView);
    m_device.destroyImage(m_depthImage);
    m_device.freeMemory(m_depthMemory);
    m_device.destroyPipelineCache(m_pipelineCache);
    m_device.destroySemaphore(m_acquireComplete);
    m_device.destroySemaphore(m_renderComplete);

    for(uint32_t i = 0; i < m_swapChain.imageCount; i++)
    {
      m_device.destroyFence(m_waitFences[i]);
      m_device.destroyFramebuffer(m_framebuffers[i]);
      m_device.freeCommandBuffers(m_cmdPool, m_commandBuffers[i]);
    }
    m_swapChain.deinit();

    m_device.destroyCommandPool(m_cmdPool);
    m_instance.destroySurfaceKHR(m_surface);
    m_vkctx.deinit();
  }


  //--------------------------------------------------------------------------------------------------
  // Creating the surface for rendering
  //
  void createSurface(uint32_t   width,
                     uint32_t   height,
                     vk::Format colorFormat = vk::Format::eB8G8R8A8Unorm,
                     vk::Format depthFormat = vk::Format::eD32SfloatS8Uint,
                     bool       vsync       = false)
  {
    m_size        = vk::Extent2D(width, height);
    m_depthFormat = depthFormat;
    m_colorFormat = colorFormat;
    m_vsync       = vsync;

    m_swapChain.init(m_physicalDevice, m_device, m_queue, m_graphicsQueueIndex, m_surface,
                     colorFormat);
    m_swapChain.update(m_size, vsync);


    // Create Synchronization Primitives
    m_waitFences.resize(m_swapChain.imageCount);
    for(auto& fence : m_waitFences)
    {
      fence = m_device.createFence({vk::FenceCreateFlagBits::eSignaled});
    }

    // Command buffers store a reference to the frame buffer inside their render pass info
    // so for static usage without having to rebuild them each frame, we use one per frame buffer
    m_commandBuffers = m_device.allocateCommandBuffers(
        {m_cmdPool, vk::CommandBufferLevel::ePrimary, m_swapChain.imageCount});

    m_acquireComplete = m_device.createSemaphore({});
    m_renderComplete  = m_device.createSemaphore({});
  }

  //--------------------------------------------------------------------------------------------------
  // Create the framebuffers in which the image will be rendered
  // - Swapchain need to be created before calling this
  //
  void createFrameBuffers()
  {
    // Recreate the frame buffers
    for(auto framebuffer : m_framebuffers)
    {
      m_device.destroyFramebuffer(framebuffer);
    }

    // Depth/Stencil attachment is the same for all frame buffers
    // First one is set by the swapChain
    vk::ImageView attachments[2];
    attachments[1] = m_depthView;

    // Create frame buffers for every swap chain image
    m_framebuffers = m_swapChain.createFramebuffers(
        {{}, m_renderPass, 2, attachments, m_size.width, m_size.height, 1});
  }

  //--------------------------------------------------------------------------------------------------
  // Creating a default render pass, very simple one.
  // Other examples will mostly override this one.
  //
  void createRenderPass()
  {
    if(m_renderPass)
    {
      m_device.destroyRenderPass(m_renderPass);
    }

    std::array<vk::AttachmentDescription, 2> attachments;
    // Color attachment
    attachments[0].setFormat(m_colorFormat);
    attachments[0].setLoadOp(vk::AttachmentLoadOp::eClear);
    attachments[0].setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

    // Depth attachment
    attachments[1].setFormat(m_depthFormat);
    attachments[1].setLoadOp(vk::AttachmentLoadOp::eClear);
    attachments[1].setStencilLoadOp(vk::AttachmentLoadOp::eClear);
    attachments[1].setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

    // One color, one depth
    const vk::AttachmentReference colorReference{0, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::AttachmentReference depthReference{1,
                                                 vk::ImageLayout::eDepthStencilAttachmentOptimal};

    std::array<vk::SubpassDependency, 1> subpassDependencies;
    // Transition from final to initial (VK_SUBPASS_EXTERNAL refers to all commands executed outside of the actual renderpass)
    subpassDependencies[0].setSrcSubpass(VK_SUBPASS_EXTERNAL);
    subpassDependencies[0].setDstSubpass(0);
    subpassDependencies[0].setSrcStageMask(vk::PipelineStageFlagBits::eBottomOfPipe);
    subpassDependencies[0].setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
    subpassDependencies[0].setSrcAccessMask(vk::AccessFlagBits::eMemoryRead);
    subpassDependencies[0].setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead
                                            | vk::AccessFlagBits::eColorAttachmentWrite);
    subpassDependencies[0].setDependencyFlags(vk::DependencyFlagBits::eByRegion);

    vk::SubpassDescription subpassDescription;
    subpassDescription.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics);
    subpassDescription.setColorAttachmentCount(1);
    subpassDescription.setPColorAttachments(&colorReference);
    subpassDescription.setPDepthStencilAttachment(&depthReference);

    vk::RenderPassCreateInfo renderPassInfo;
    renderPassInfo.setAttachmentCount(static_cast<uint32_t>(attachments.size()));
    renderPassInfo.setPAttachments(attachments.data());
    renderPassInfo.setSubpassCount(1);
    renderPassInfo.setPSubpasses(&subpassDescription);
    renderPassInfo.setDependencyCount(static_cast<uint32_t>(subpassDependencies.size()));
    renderPassInfo.setPDependencies(subpassDependencies.data());

    m_renderPass = m_device.createRenderPass(renderPassInfo);
  }

  //--------------------------------------------------------------------------------------------------
  // Creating an image to be used as depth buffer
  //
  void createDepthBuffer()
  {
    m_device.destroyImageView(m_depthView);
    m_device.destroyImage(m_depthImage);
    m_device.freeMemory(m_depthMemory);

    // Depth information
    const vk::ImageAspectFlags aspect =
        vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    vk::ImageCreateInfo depthStencilCreateInfo;
    depthStencilCreateInfo.setImageType(vk::ImageType::e2D);
    depthStencilCreateInfo.setExtent(vk::Extent3D{m_size.width, m_size.height, 1});
    depthStencilCreateInfo.setFormat(m_depthFormat);
    depthStencilCreateInfo.setMipLevels(1);
    depthStencilCreateInfo.setArrayLayers(1);
    depthStencilCreateInfo.setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment
                                    | vk::ImageUsageFlagBits::eTransferSrc);
    // Create the depth image
    m_depthImage = m_device.createImage(depthStencilCreateInfo);

    // Allocate the memory
    const vk::MemoryRequirements memReqs = m_device.getImageMemoryRequirements(m_depthImage);
    vk::MemoryAllocateInfo       memAllocInfo;
    memAllocInfo.allocationSize = memReqs.size;
    memAllocInfo.memoryTypeIndex =
        getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    m_depthMemory = m_device.allocateMemory(memAllocInfo);

    // Bind image and memory
    m_device.bindImageMemory(m_depthImage, m_depthMemory, 0);

    // Create an image barrier to change the layout from undefined to DepthStencilAttachmentOptimal
    vk::CommandBuffer             cmdBuffer;
    vk::CommandBufferAllocateInfo cmdBufAllocateInfo;
    cmdBufAllocateInfo.commandPool        = m_cmdPool;
    cmdBufAllocateInfo.level              = vk::CommandBufferLevel::ePrimary;
    cmdBufAllocateInfo.commandBufferCount = 1;
    cmdBuffer                             = m_device.allocateCommandBuffers(cmdBufAllocateInfo)[0];
    cmdBuffer.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // Put barrier on top, Put barrier inside setup command buffer
    vk::ImageSubresourceRange subresourceRange;
    subresourceRange.aspectMask = aspect;
    subresourceRange.levelCount = 1;
    subresourceRange.layerCount = 1;
    vk::ImageMemoryBarrier imageMemoryBarrier;
    imageMemoryBarrier.oldLayout               = vk::ImageLayout::eUndefined;
    imageMemoryBarrier.newLayout               = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    imageMemoryBarrier.image                   = m_depthImage;
    imageMemoryBarrier.subresourceRange        = subresourceRange;
    imageMemoryBarrier.srcAccessMask           = vk::AccessFlags();
    imageMemoryBarrier.dstAccessMask           = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    const vk::PipelineStageFlags srcStageMask  = vk::PipelineStageFlagBits::eTopOfPipe;
    const vk::PipelineStageFlags destStageMask = vk::PipelineStageFlagBits::eEarlyFragmentTests;

    cmdBuffer.pipelineBarrier(srcStageMask, destStageMask, vk::DependencyFlags(), nullptr, nullptr,
                              imageMemoryBarrier);
    cmdBuffer.end();
    m_queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &cmdBuffer}, vk::Fence());
    m_queue.waitIdle();
    m_device.freeCommandBuffers(m_cmdPool, cmdBuffer);

    // Setting up the view
    vk::ImageViewCreateInfo depthStencilView;
    depthStencilView.setViewType(vk::ImageViewType::e2D);
    depthStencilView.setFormat(m_depthFormat);
    depthStencilView.setSubresourceRange({aspect, 0, 1, 0, 1});
    depthStencilView.setImage(m_depthImage);
    m_depthView = m_device.createImageView(depthStencilView);
  }

  //--------------------------------------------------------------------------------------------------
  // Convenient function to call before rendering
  //
  void prepareFrame()
  {
    // Acquire the next image from the swap chain
    const vk::Result res = m_swapChain.acquire(m_acquireComplete, &m_curFramebuffer);

    // Recreate the swapchain if it's no longer compatible with the surface (OUT_OF_DATE) or no longer optimal for presentation (SUBOPTIMAL)
    if((res == vk::Result::eErrorOutOfDateKHR) || (res == vk::Result::eSuboptimalKHR))
    {
      // Need new window size !!
      onWindowResize(m_size.width, m_size.height);
    }

    // Use a fence to wait until the command buffer has finished execution before using it again
    const vk::Device device(m_device);
    while(device.waitForFences(m_waitFences[m_curFramebuffer], VK_TRUE, 10000)
          == vk::Result::eTimeout)
    {
    }
  }

  //--------------------------------------------------------------------------------------------------
  // Convenient function to call for submitting the rendering command
  //
  void submitFrame()
  {
    m_device.resetFences(m_waitFences[m_curFramebuffer]);

    // In case of using NVLINK
    const uint32_t deviceMask    = m_useNvlink ? 0b0000'0011 : 0b0000'0001;
    const uint32_t deviceIndex[] = {0, 1};

    vk::DeviceGroupSubmitInfo deviceGroupSubmitInfo;
    deviceGroupSubmitInfo.setWaitSemaphoreCount(1);
    deviceGroupSubmitInfo.setCommandBufferCount(1);
    deviceGroupSubmitInfo.setPCommandBufferDeviceMasks(&deviceMask);
    deviceGroupSubmitInfo.setSignalSemaphoreCount(m_useNvlink ? 2 : 1);
    deviceGroupSubmitInfo.setPSignalSemaphoreDeviceIndices(deviceIndex);
    deviceGroupSubmitInfo.setPWaitSemaphoreDeviceIndices(deviceIndex);

    // Pipeline stage at which the queue submission will wait (via pWaitSemaphores)
    const vk::PipelineStageFlags waitStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    // The submit info structure specifies a command buffer queue submission batch
    vk::SubmitInfo submitInfo;
    submitInfo.setPWaitDstStageMask(
        &waitStageMask);  // Pointer to the list of pipeline stages that the semaphore waits will occur at
    submitInfo.setPWaitSemaphores(
        &m_acquireComplete);  // Semaphore(s) to wait upon before the submitted command buffer starts executing
    submitInfo.setWaitSemaphoreCount(1);  // One wait semaphore
    submitInfo.setPSignalSemaphores(
        &m_renderComplete);  // Semaphore(s) to be signaled when command buffers have completed
    submitInfo.setSignalSemaphoreCount(1);  // One signal semaphore
    submitInfo.setPCommandBuffers(
        &m_commandBuffers
            [m_curFramebuffer]);  // Command buffers(s) to execute in this batch (submission)
    submitInfo.setCommandBufferCount(1);  // One command buffer
    submitInfo.setPNext(&deviceGroupSubmitInfo);

    // Submit to the graphics queue passing a wait fence
    m_queue.submit(submitInfo, m_waitFences[m_curFramebuffer]);

    const vk::Result res = m_swapChain.present(m_curFramebuffer, m_renderComplete);
    if(!((res == vk::Result::eSuccess) || (res == vk::Result::eSuboptimalKHR)))
    {
      if(res == vk::Result::eErrorOutOfDateKHR)
      {
        // Swap chain is no longer compatible with the surface and needs to be recreated
        // Need new window size !!
        onWindowResize(m_size.width, m_size.height);
        return;
      }
    }

    // Increasing the current frame buffer
    //m_curFramebuffer = (m_curFramebuffer + 1) % m_swapChain.imageCount;
  }


  //--------------------------------------------------------------------------------------------------
  // When the pipeline is set for using dynamic, this becomes useful
  //
  void setViewport(const vk::CommandBuffer& cmdBuf)
  {
    cmdBuf.setViewport(0, {vk::Viewport(0, 0, (float)m_size.width, (float)m_size.height, 0, 1)});
    cmdBuf.setScissor(0, {{{0, 0}, {m_size.width, m_size.height}}});
  }

  //--------------------------------------------------------------------------------------------------
  // Window callback when the it is resized
  // - Destroy allocated frames, then rebuild them with the new size
  // - Call onResize() of the derived class
  //
  void onWindowResize(int w, int h)
  {
    if(w == 0 || h == 0)
    {
      return;
    }

    m_size.width  = w;
    m_size.height = h;

    m_device.waitIdle();
    m_queue.waitIdle();

    m_swapChain.update(m_size, m_vsync);
    createDepthBuffer();
    createFrameBuffers();
  }

  vk::Instance                          getInstance() { return m_instance; }
  vk::Device                            getDevice() { return m_device; }
  vk::PhysicalDevice                    getPhysicalDevice() { return m_physicalDevice; }
  vk::Queue                             getQueue() { return m_queue; }
  uint32_t                              getQueueFamily() { return m_graphicsQueueIndex; }
  vk::CommandPool                       getCommandPool() { return m_cmdPool; }
  vk::RenderPass                        getRenderPass() { return m_renderPass; }
  vk::Extent2D                          getSize() { return m_size; }
  vk::PipelineCache                     getPipelineCache() { return m_pipelineCache; }
  vk::SurfaceKHR                        getSurface() { return m_surface; }
  const std::vector<vk::Framebuffer>&   getFramebuffers() { return m_framebuffers; }
  const std::vector<vk::CommandBuffer>& getCommandBuffers() { return m_commandBuffers; }
  uint32_t                              getCurFrame() { return m_curFramebuffer; }

protected:
  uint32_t getMemoryType(uint32_t typeBits, const vk::MemoryPropertyFlags& properties) const
  {
    auto deviceMemoryProperties = m_physicalDevice.getMemoryProperties();
    for(uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++)
    {
      if(((typeBits & (1 << i)) > 0)
         && (deviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
      {
        return i;
      }
    }
    std::string err = "Unable to find memory type " + vk::to_string(properties);
    assert(0);
    return ~0u;
  }


  //--------------------------------------------------------------------------------------------------
  nvvkpp::Context m_vkctx;

  // Vulkan elements
  vk::Instance       m_instance;
  vk::Device         m_device;
  vk::PhysicalDevice m_physicalDevice;
  vk::SurfaceKHR     m_surface;
  vk::Queue          m_queue;
  uint32_t           m_graphicsQueueIndex{VK_QUEUE_FAMILY_IGNORED};
  vk::CommandPool    m_cmdPool;

  // Drawing/Surface
  nvvkpp::SwapChain              m_swapChain;
  std::vector<vk::Framebuffer>   m_framebuffers;    // All framebuffers, correspond to the Swapchain
  std::vector<vk::CommandBuffer> m_commandBuffers;  // Command buffer per nb element in Swapchain
  std::vector<vk::Fence>         m_waitFences;      // Fences per nb element in Swapchain
  vk::Semaphore                  m_acquireComplete;  // Swap chain image presentation
  vk::Semaphore                  m_renderComplete;   // Command buffer submission and execution
  vk::Image                      m_depthImage;       // Depth/Stencil
  vk::DeviceMemory               m_depthMemory;      // Depth/Stencil
  vk::ImageView                  m_depthView;        // Depth/Stencil
  vk::RenderPass                 m_renderPass;       // Base render pass
  vk::Extent2D                   m_size{0, 0};       // Size of the window
  vk::PipelineCache              m_pipelineCache;    // Cache for pipeline/shaders
  bool                           m_vsync{false};     // Swapchain with vsync
  BOOL                           m_useNvlink{false};

  uint32_t m_curFramebuffer{0};  // Remember the current framebuffer in use

  // Surface buffer formats
  vk::Format m_colorFormat{vk::Format::eB8G8R8A8Unorm};
  vk::Format m_depthFormat{vk::Format::eUndefined};
};  // namespace nvvkpp


}  // namespace nvvkpp
