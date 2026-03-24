// This is an AI translated version of the original Nvidia source for GPU compute,
// but ported to WGPU for cross-platform support
// https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/main/testBenches/chanModels/src/sls_chan_src/sls_chan_large_scale_GPU.cu

#include "sls-chan-wgpu.h"

#include <iostream>

/**
 * To check available WGPU devices:
 *
 * pip install wgpu
 * python -c "import wgpu; [print(adapter.info) for adapter in wgpu.gpu.enumerate_adapters_sync()]"
 */
static WGPUDevice
createDevice()
{
    WGPUInstanceDescriptor idesc{};
    WGPUInstance instance = wgpuCreateInstance(&idesc);
    assert(instance);

    // ── Adapter — callback fires synchronously in wgpu-native ─────────────
    WGPUAdapter adapter = nullptr;

    WGPURequestAdapterOptions aopts{};
    aopts.powerPreference = WGPUPowerPreference_HighPerformance;

    wgpuInstanceRequestAdapter(
        instance,
        &aopts,
        WGPURequestAdapterCallbackInfo{
            .callback =
                [](WGPURequestAdapterStatus status,
                   WGPUAdapter a,
                   WGPUStringView msg,
                   void* ud1,
                   void*) {
                    if (status == WGPURequestAdapterStatus_Success)
                    {
                        *static_cast<WGPUAdapter*>(ud1) = a;
                    }
                    else
                    {
                        fprintf(stderr, "requestAdapter failed: %.*s\n", (int)msg.length, msg.data);
                    }
                },
            .userdata1 = &adapter});
    assert(adapter && "no suitable adapter");

    // ── Device — same pattern ──────────────────────────────────────────────
    WGPUDevice device = nullptr;

    WGPUDeviceDescriptor ddesc{};
    ddesc.uncapturedErrorCallbackInfo.callback =
        [](const WGPUDevice*, WGPUErrorType t, WGPUStringView msg, void*, void*) {
            fprintf(stderr, "[wgpu error %d] %.*s\n", (int)t, (int)msg.length, msg.data);
        };

    wgpu::Limits::WGPULimits requiredLimits{};
    requiredLimits.maxBindGroups = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxBufferSize = WGPU_LIMIT_U64_UNDEFINED;
    requiredLimits.maxColorAttachmentBytesPerSample = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxColorAttachments = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxComputeInvocationsPerWorkgroup = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxComputeWorkgroupSizeX = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxComputeWorkgroupSizeY = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxComputeWorkgroupSizeZ = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxComputeWorkgroupStorageSize = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxComputeWorkgroupStorageSize = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxComputeWorkgroupsPerDimension = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxDynamicStorageBuffersPerPipelineLayout = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxDynamicUniformBuffersPerPipelineLayout = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxInterStageShaderVariables = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxSampledTexturesPerShaderStage = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxSamplersPerShaderStage = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxStorageBufferBindingSize = WGPU_LIMIT_U64_UNDEFINED;
    requiredLimits.maxStorageBuffersPerShaderStage = 13;
    requiredLimits.maxStorageTexturesPerShaderStage = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxTextureDimension1D = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxTextureDimension2D = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxTextureDimension3D = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxUniformBufferBindingSize = WGPU_LIMIT_U64_UNDEFINED;
    requiredLimits.maxUniformBuffersPerShaderStage = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxVertexAttributes = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxVertexBufferArrayStride = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxVertexBuffers = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.minStorageBufferOffsetAlignment = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.minStorageBufferOffsetAlignment = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.minUniformBufferOffsetAlignment = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.minUniformBufferOffsetAlignment = WGPU_LIMIT_U32_UNDEFINED;
    ddesc.requiredLimits = &requiredLimits;

    wgpuAdapterRequestDevice(
        adapter,
        &ddesc,
        WGPURequestDeviceCallbackInfo{
            .callback =
                [](WGPURequestDeviceStatus status,
                   WGPUDevice d,
                   WGPUStringView msg,
                   void* ud1,
                   void*) {
                    if (status == WGPURequestDeviceStatus_Success)
                    {
                        *static_cast<WGPUDevice*>(ud1) = d;
                    }
                    else
                    {
                        fprintf(stderr, "requestDevice failed: %.*s\n", (int)msg.length, msg.data);
                    }
                },
            .userdata1 = &device});
    assert(device && "no device");

    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    return device;
}

int
main()
{
    wgpu::Device device = createDevice();
    SlsChanWgpu sls(device);

    // ── Build a minimal cell + UT layout for testing ──────────────────────
    const uint32_t nSite = 7;
    const uint32_t nUT = 210;
    const uint32_t nSector = 3;

    std::vector<CellParam> cells(nSite * nSector);
    for (uint32_t s = 0; s < nSite; ++s)
    {
        for (uint32_t k = 0; k < nSector; ++k)
        {
            auto& c = cells[s * nSector + k];
            c.loc = {(float)(s * 200), 0.f, 25.f, 0.f}; // 200 m ISD, 25 m height
        }
    }

    std::vector<UtParam> uts(nUT);
    for (uint32_t u = 0; u < nUT; ++u)
    {
        auto& ut = uts[u];
        ut.loc = {(float)((int)u * 14 - 700), (float)((int)u * 7), 1.5f, 0.f};
        ut.d_2d_in = 0.f;
        ut.outdoor_ind = 1u; // outdoor
        ut.o2i_penetration_loss = 0.f;
    }

    sls.uploadCellParams(cells);
    sls.uploadUtParams(uts);

    // Generate CRN grids (UMa correlation distances from 3GPP TR 38.901 Table 7.6.3.1-2)
    float corrLos[7] = {37.f, 12.f, 30.f, 18.f, 15.f, 15.f, 15.f};
    float corrNlos[6] = {50.f, 40.f, 50.f, 50.f, 50.f, 50.f};
    float corrO2i[6] = {10.f, 10.f, 11.f, 17.f, 25.f, 25.f};
    sls.generateCRN(700.f, -700.f, 700.f, -700.f, corrLos, corrNlos, corrO2i);

    // Run main link parameter kernel
    sls.calLinkParam(nSite,
                     nUT,
                     nSector,
                     700.f,
                     -700.f,
                     700.f,
                     -700.f,
                     /*updatePL=*/true,
                     /*updateAllLSPs=*/true,
                     /*updateLos=*/true,
                     /*nX=*/141,
                     /*nY=*/141);

    auto results = sls.readLinkParams(nSite, nUT);

    for (uint32_t nLink = 0; nLink < results.size(); nLink++)
    {
        std::cout << "Link[" << nLink << "]: PL=" << results[nLink].pathloss
                  << " dB, DS=" << results[nLink].DS << " ns, LOS=" << results[nLink].losInd
                  << ", d2d=" << results[nLink].d2d << ", d3d=" << results[nLink].d3d << "\n";
    }
    return 0;
}
