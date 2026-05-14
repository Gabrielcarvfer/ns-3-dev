// This is an AI translated version of the original Nvidia source for GPU compute,
// but ported to WGPU for cross-platform support
// https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/main/testBenches/chanModels/src/sls_chan_src/sls_chan_large_scale_GPU.cu

#define WEBGPU_CPP_IMPLEMENTATION
#include "sls-chan-wgpu.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>

#ifdef SLS_CHAN_HDF5
#include <H5Cpp.h>
#endif

static std::string
readFile(const char* path)
{
    std::ifstream f(path);
    return {std::istreambuf_iterator<char>(f), {}};
}

static std::pair<wgpu::Device, uint64_t>
createDevice()
{
    WGPUInstanceDescriptor idesc{};
    WGPUInstance instance = wgpuCreateInstance(&idesc);
    assert(instance);

    WGPUAdapter adapter = nullptr;
    WGPURequestAdapterOptions aopts{};

    // Try D3D12 backend first (best on Windows)
    aopts.backendType = WGPUBackendType_D3D12;
    fprintf(stderr, "[DEBUG] Trying D3D12 backend...\n");
    wgpuInstanceRequestAdapter(
            instance,
            &aopts,
            WGPURequestAdapterCallbackInfo{.mode = WGPUCallbackMode_AllowSpontaneous,
                    .callback =
                    [](WGPURequestAdapterStatus status,
                       WGPUAdapter a,
                       WGPUStringView,
                       void* ud1,
                       void*) {
                        if (status == WGPURequestAdapterStatus_Success)
                        {
                            fprintf(stderr, "[DEBUG] D3D12 adapter acquired\n");
                            *static_cast<WGPUAdapter*>(ud1) = a;
                        }
                        else
                        {
                            fprintf(stderr, "[DEBUG] D3D12 adapter failed (status=%d), trying Vulkan...\n", status);
                        }
                    },
                    .userdata1 = &adapter});
    wgpuInstanceProcessEvents(instance);  // process async callback
    if (!adapter) {
        // Try Vulkan backend as fallback
        aopts.backendType = WGPUBackendType_Vulkan;
        fprintf(stderr, "[DEBUG] Trying Vulkan backend...\n");
        wgpuInstanceRequestAdapter(
                instance,
                &aopts,
                WGPURequestAdapterCallbackInfo{.mode = WGPUCallbackMode_AllowSpontaneous,
                        .callback =
                        [](WGPURequestAdapterStatus status,
                           WGPUAdapter a,
                           WGPUStringView,
                           void* ud1,
                           void*) {
                            if (status == WGPURequestAdapterStatus_Success)
                            {
                                fprintf(stderr, "[DEBUG] Vulkan adapter acquired\n");
                                *static_cast<WGPUAdapter*>(ud1) = a;
                            }
                            else
                            {
                                fprintf(stderr, "[DEBUG] Vulkan adapter failed (status=%d), trying auto...\n", status);
                            }
                        },
                        .userdata1 = &adapter});
        wgpuInstanceProcessEvents(instance);  // process async callback
    }
    if (!adapter) {
        // Try auto-select
        aopts.backendType = WGPUBackendType_Undefined;
        fprintf(stderr, "[DEBUG] Trying auto-select backend...\n");
        wgpuInstanceRequestAdapter(
                instance,
                &aopts,
                WGPURequestAdapterCallbackInfo{.mode = WGPUCallbackMode_AllowSpontaneous,
                        .callback =
                        [](WGPURequestAdapterStatus status,
                           WGPUAdapter a,
                           WGPUStringView,
                           void* ud1,
                           void*) {
                            if (status == WGPURequestAdapterStatus_Success)
                            {
                                fprintf(stderr, "[DEBUG] Auto-selected adapter acquired\n");
                                *static_cast<WGPUAdapter*>(ud1) = a;
                            }
                            else
                            {
                                fprintf(stderr, "[DEBUG] Auto-select adapter failed (status=%d)\n", status);
                            }
                        },
                        .userdata1 = &adapter});
        wgpuInstanceProcessEvents(instance);  // process async callback
    }
    if (!adapter) {
        fprintf(stderr, "[FATAL] No GPU adapter available!\n");
        wgpuInstanceRelease(instance);
        exit(1);
    }

    // Query what the adapter actually supports first
    WGPULimits supported{};
    wgpuAdapterGetLimits(adapter, &supported);

    WGPUDevice device = nullptr;

    // Request max-buffer-size feature so we can create buffers > 256MB
    // Note: WGPUFeatureName_MaxBufferSize may not be in this version of wgpu-native
    // So we just set the limit to WGPU_LIMIT_U64_UNDEFINED (unlimited)
    WGPUDeviceDescriptor ddesc{};
    ddesc.uncapturedErrorCallbackInfo.callback =
            [](const WGPUDevice*, WGPUErrorType t, WGPUStringView msg, void*, void*) {
                fprintf(stderr, "[wgpu error %d] %.*s\n", (int)t, (int)msg.length, msg.data);
            };
    // Set unlimited buffer size limit
    WGPULimits limits{};
    limits.maxBufferSize = WGPU_LIMIT_U64_UNDEFINED;
    ddesc.requiredLimits = &limits;
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
    requiredLimits.maxComputeWorkgroupsPerDimension = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxDynamicStorageBuffersPerPipelineLayout = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxDynamicUniformBuffersPerPipelineLayout = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxInterStageShaderVariables = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxSampledTexturesPerShaderStage = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.maxSamplersPerShaderStage = WGPU_LIMIT_U32_UNDEFINED;
    // Request the largest maxStorageBufferBindingSize the adapter supports
    requiredLimits.maxStorageBufferBindingSize = supported.maxStorageBufferBindingSize;
    requiredLimits.maxStorageBuffersPerShaderStage = 30;
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
    requiredLimits.minUniformBufferOffsetAlignment = WGPU_LIMIT_U32_UNDEFINED;
    ddesc.requiredLimits = &requiredLimits;
    fprintf(stderr, "[DEBUG] Device created with maxStorageBufSize=%llu\n",
            (unsigned long long)requiredLimits.maxStorageBufferBindingSize);
    wgpuAdapterRequestDevice(
            adapter,
            &ddesc,
            WGPURequestDeviceCallbackInfo{
                    .callback =
                    [](WGPURequestDeviceStatus status, WGPUDevice d, WGPUStringView, void* ud1, void*) {
                        if (status == WGPURequestDeviceStatus_Success)
                        {
                            *static_cast<WGPUDevice*>(ud1) = d;
                        }
                        else
                        {
                            fprintf(stderr, "requestDevice failed\n");
                        }
                    },
                    .userdata1 = &device});
    wgpuInstanceProcessEvents(instance);  // process async callback
    if (!device) {
        fprintf(stderr, "[FATAL] Failed to create WGPU device!\n");
        wgpuAdapterRelease(adapter);
        wgpuInstanceRelease(instance);
        exit(1);
    }
    wgpu::Device dev(device);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    return std::make_pair(dev, supported.maxStorageBufferBindingSize);
}

// ── Constructor ───────────────────────────────────────────────────────────────
SlsChanWgpu::SlsChanWgpu()
{
    auto result = createDevice();
    device_ = std::move(result.first);
    m_maxGpuBuffer_ = result.second;
    fprintf(stderr, "[DEBUG] Constructor: GPU max buffer size = %llu bytes (%.1f GB)\n",
            (unsigned long long)m_maxGpuBuffer_, (double)m_maxGpuBuffer_ / (1024.0*1024.0*1024.0));
    queue_ = device_.getQueue();
    std::string wgsl = readFile("C:/tools/sources/ns-3-dev/src/spectrum/model/sls-chan.wgsl");
    if (wgsl.empty()) {
        fprintf(stderr, "ERROR: Failed to read WGSL shader\n");
        throw std::runtime_error("Failed to read WGSL shader");
    }
    std::cout << "Loaded WGSL shader: " << wgsl.size() << " bytes\n";

    WGPUShaderSourceWGSL wgslSource{};
    wgslSource.chain.next = nullptr;
    wgslSource.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSource.code = sv(wgsl.c_str());

    WGPUShaderModuleDescriptor smDescC{};
    smDescC.nextInChain = &wgslSource.chain;
    fprintf(stderr, "[DEBUG] Creating WGSL shader module (%zu bytes)...\n", wgsl.size());
    shader_ = wgpu::ShaderModule(wgpuDeviceCreateShaderModule(device_, &smDescC));
    if (!shader_) {
        fprintf(stderr, "[FATAL] WGSL shader module creation failed\n");
        throw std::runtime_error("WGSL failed to compile");
    }
    fprintf(stderr, "[DEBUG] WGSL shader module created successfully\n");

    auto makePipeline = [&](const char* ep) -> wgpu::ComputePipeline {
        wgpu::ComputePipelineDescriptor desc{};
        desc.compute.module = shader_;
        desc.compute.entryPoint = sv(ep);
        fprintf(stderr, "[DEBUG] Creating pipeline '%s'...\n", ep);
        fflush(stderr);
        auto pipe = device_.createComputePipeline(desc);
        fprintf(stderr, "[DEBUG] Pipeline '%s': %s\n", ep, pipe ? "OK" : "FAILED");
        fflush(stderr);
        return pipe;
    };

    linkParamPipeline_ = makePipeline("cal_link_param_kernel");
    fprintf(stderr, "[DEBUG] After cal_link_param_kernel\n");
    assert(linkParamPipeline_ && "missing cal_link_param_kernel in WGSL");
    crnFillPipeline_ = makePipeline("fill_crn_kernel");
    fprintf(stderr, "[DEBUG] After fill_crn_kernel\n");
    assert(crnFillPipeline_ && "missing fill_crn_kernel in WGSL");
    crnConvPipeline_ = makePipeline("convolve_crn_kernel");
    fprintf(stderr, "[DEBUG] After convolve_crn_kernel\n");
    assert(crnConvPipeline_ && "missing convolve_crn_kernel in WGSL");
    crnNormPipeline_ = makePipeline("normalize_crn_kernel");
    fprintf(stderr, "[DEBUG] After normalize_crn_kernel\n");
    assert(crnNormPipeline_ && "missing normalize_crn_kernel in WGSL");
    clusterRayPipeline_ = makePipeline("cal_cluster_ray_kernel");
    fprintf(stderr, "[DEBUG] After cal_cluster_ray_kernel\n");
    assert(clusterRayPipeline_ && "missing cal_cluster_ray_kernel in WGSL");
    generateCIRPipeline_ = makePipeline("generate_cir_kernel");
    fprintf(stderr, "[DEBUG] After generate_cir_kernel\n");
    assert(generateCIRPipeline_ && "missing generate_cir_kernel in WGSL");
    generateCFRPipeline_ = makePipeline("generate_cfr_kernel_mode1");
    fprintf(stderr, "[DEBUG] After generate_cfr_kernel_mode1\n");
    assert(generateCFRPipeline_ && "missing generate_cfr_kernel_mode1 in WGSL");
}

// ── Buffer helper ─────────────────────────────────────────────────────────────
wgpu::Buffer
SlsChanWgpu::makeBuffer(uint64_t size, wgpu::BufferUsage usage, const void* data)
{
    const uint64_t alignedSize = (size + 3) & ~uint64_t(3);

    wgpu::BufferDescriptor desc{};
    desc.size = alignedSize;
    desc.usage = usage | WGPUBufferUsage_CopyDst;
    desc.mappedAtCreation = false; // no longer needed

    wgpu::Buffer buf = device_.createBuffer(desc);

    if (data)
    {
        // writeBuffer also requires size % 4 == 0; pad if needed
        if (alignedSize == size)
        {
            queue_.writeBuffer(buf, 0, data, size);
        }
        else
        {
            std::vector<uint8_t> tmp(alignedSize, 0);
            std::memcpy(tmp.data(), data, size);
            queue_.writeBuffer(buf, 0, tmp.data(), alignedSize);
        }
    }
    return buf;
}

void
SlsChanWgpu::waitIdle()
{
    device_.poll(true, nullptr);
}

// ── Large-scale topology uploads ──────────────────────────────────────────────
void
SlsChanWgpu::uploadCellParams(const std::vector<CellParam>& cells)
{
    cellParamsBuf_ =
        makeBuffer(cells.size() * sizeof(CellParam), WGPUBufferUsage_Storage, cells.data());
    nSite_ = uint32_t(cells.size() / 3);
}

void
SlsChanWgpu::uploadUtParams(const std::vector<UtParam>& uts)
{
    utParamsBuf_ = makeBuffer(uts.size() * sizeof(UtParam), WGPUBufferUsage_Storage, uts.data());
}

// ── Small-scale topology uploads ─────────────────────────────────────────────
void
SlsChanWgpu::uploadCellParamsSS(const std::vector<CellParamSS>& cells)
{
    ssCellParamsBuf_ =
        makeBuffer(cells.size() * sizeof(CellParamSS), WGPUBufferUsage_Storage, cells.data());
}

void
SlsChanWgpu::uploadUtParamsSS(const std::vector<UtParamSS>& uts)
{
    ssUtParamsBuf_ =
        makeBuffer(uts.size() * sizeof(UtParamSS), WGPUBufferUsage_Storage, uts.data());
}

// ── Antenna panel configs ─────────────────────────────────────────────────────
void
SlsChanWgpu::uploadAntPanelConfigs(const std::vector<AntPanelConfigGPU>& configs,
                                   const std::vector<float>& antThetaFlat,
                                   const std::vector<float>& antPhiFlat)
{
    antPanelConfigBuf_ = makeBuffer(configs.size() * sizeof(AntPanelConfigGPU),
                                    WGPUBufferUsage_Storage,
                                    configs.data());
    antThetaBuf_ = makeBuffer(antThetaFlat.size() * sizeof(float),
                              WGPUBufferUsage_Storage,
                              antThetaFlat.data());
    antPhiBuf_ =
        makeBuffer(antPhiFlat.size() * sizeof(float), WGPUBufferUsage_Storage, antPhiFlat.data());
    // validation uses configs[0] = BS panel, configs[1] = UE panel
    ssNBsAnt_ = configs[0].nAnt;
    ssNUeAnt_ = configs[1].nAnt;
}

// ── mapAsync readback helper ──────────────────────────────────────────────────
template <typename T>
std::vector<T>
SlsChanWgpu::mapReadBuffer(wgpu::Buffer& staging, uint64_t byteSize)
{
    std::vector<T> result(byteSize / sizeof(T));

    struct MapCtx
    {
        bool done = false;
    };

    MapCtx ctx;

    wgpu::BufferMapCallbackInfo cbInfo = wgpu::Default;
    cbInfo.mode = wgpu::CallbackMode::AllowProcessEvents;
    cbInfo.callback = [](WGPUMapAsyncStatus, WGPUStringView, void* ud1, void*) {
        static_cast<MapCtx*>(ud1)->done = true;
    };
    cbInfo.userdata1 = &ctx;

    staging.mapAsync(wgpu::MapMode::Read, 0, byteSize, cbInfo);
    while (!ctx.done)
    {
        device_.poll(false, nullptr);
    }

    std::memcpy(result.data(), staging.getMappedRange(0, byteSize), byteSize);
    staging.unmap();
    return result;
}

// ── readLinkParams ────────────────────────────────────────────────────────────
std::vector<LinkParams>
SlsChanWgpu::readLinkParams(uint32_t nSite, uint32_t nUT)
{
    uint64_t sz = uint64_t(nSite) * nUT * sizeof(LinkParams);
    return mapReadBuffer<LinkParams>(stagingBuf_, sz);
}

// ── generateCRN ───────────────────────────────────────────────────────────────
void
SlsChanWgpu::generateCRN(float maxX,
                         float minX,
                         float maxY,
                         float minY,
                         const float corrLos[8],
                         const float corrNlos[7],
                         const float corrO2i[7])
{
    fprintf(stderr, "[DEBUG] generateCRN: nSite=%u, maxX=%.1f, minX=%.1f, maxY=%.1f, minY=%.1f\n", nSite_, maxX, minX, maxY, minY);
    fflush(stderr);
    // Calculate grid dimensions matching WGSL shader: round(bound + 1 + 2*D) where D=3*corrDist
    float maxCorrDist = 0.0f;
    for (int i = 0; i < 8; i++) maxCorrDist = std::max(maxCorrDist, corrLos[i]);
    for (int i = 0; i < 7; i++) maxCorrDist = std::max(maxCorrDist, corrNlos[i]);
    for (int i = 0; i < 7; i++) maxCorrDist = std::max(maxCorrDist, corrO2i[i]);
    
    fprintf(stderr, "[DEBUG] generateCRN: maxCorrDist=%.1f\n", maxCorrDist);
    float D = 3.0f * maxCorrDist;
    const int32_t nX = static_cast<int32_t>((maxX - minX) + 1.0f + 2.0f * D + 0.5f);
    const int32_t nY = static_cast<int32_t>((maxY - minY) + 1.0f + 2.0f * D + 0.5f);
    fprintf(stderr, "[DEBUG] generateCRN: nX=%d, nY=%d, gridSz=%llu\n", nX, nY, (unsigned long long)uint64_t(nX) * nY);

    nX_ = nX;
    nY_ = nY;

    const uint64_t gridSz = uint64_t(nX) * nY;
    const uint64_t losBufSz = uint64_t(nSite_) * 8 * gridSz * sizeof(float);
    const uint64_t nlosBufSz = uint64_t(nSite_) * 7 * gridSz * sizeof(float);
    const uint64_t o2iBufSz = uint64_t(nSite_) * 7 * gridSz * sizeof(float);
    fprintf(stderr, "[DEBUG] generateCRN: losBufSz=%llu, nlosBufSz=%llu, o2iBufSz=%llu\n", (unsigned long long)losBufSz, (unsigned long long)nlosBufSz, (unsigned long long)o2iBufSz);

    auto tempBufBytes = [&](float cd) -> uint64_t {
        // Match WGSL/CPU: D = 3*corrDist, grid = round(bound + 1 + 2*D)
        float D = 3.0f * cd;
        const uint64_t pnx = static_cast<uint64_t>((maxX - minX) + 1.0f + 2.0f * D + 0.5f);
        const uint64_t pny = static_cast<uint64_t>((maxY - minY) + 1.0f + 2.0f * D + 0.5f);
        return pnx * pny * sizeof(float);
    };

    float maxCorr = 0.0f;
    for (int i = 0; i < 8; i++)
    {
        maxCorr = std::max(maxCorr, corrLos[i]);
    }
    for (int i = 0; i < 7; i++)
    {
        maxCorr = std::max(maxCorr, corrNlos[i]);
    }
    for (int i = 0; i < 7; i++)
    {
        maxCorr = std::max(maxCorr, corrO2i[i]);
    }
    fprintf(stderr, "[DEBUG] generateCRN: maxCorr=%.1f, tempBufBytes=%llu\n", maxCorr, (unsigned long long)tempBufBytes(maxCorr));

    wgpu::Buffer tempBuf =
        makeBuffer(tempBufBytes(maxCorr), WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    fprintf(stderr, "[DEBUG] generateCRN: tempBuf created\n");

    // Check if GPU can handle the CRN output buffers
    // CRN buffers are nSite * channels * gridSz * sizeof(float) each
    // With nSite=19, gridSz=20259001: LOS=11.5GB, NLOS/O2I=10GB
    // Use the adapter's actual maxStorageBufferBindingSize (set in createDevice)
    fprintf(stderr, "[DEBUG] generateCRN: GPU max buffer size = %llu bytes (%.1f GB)\n",
            (unsigned long long)m_maxGpuBuffer_, (double)m_maxGpuBuffer_ / (1024.0*1024.0*1024.0));

    // NOTE: This wgpu-native version does NOT support the max-buffer-size feature
    // (WGPUFeatureName_MaxBufferSize is not in the enum). Large buffers (>256MB) will
    // always fail validation. We rely entirely on CPU-side buffers + staging for output.

    const uint32_t nCrnRng = 128u * 256u;
    std::vector<RngState> crnSeeds(nCrnRng);
    for (uint32_t i = 0; i < nCrnRng; i++)
    {
        crnSeeds[i] = {i * 1234567891u + 1,
                       i * 2345678901u + 1,
                       i * 3456789011u + 1,
                       (uint32_t)(i * 4567890121u + 1)};
    }

    wgpu::Buffer crnRngBuf =
        makeBuffer(uint64_t(nCrnRng) * sizeof(RngState), WGPUBufferUsage_Storage, crnSeeds.data());
    wgpu::Buffer gridBuf =
        makeBuffer(gridSz * sizeof(float),
                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst);

    auto dispatchGrid = [&](wgpu::Buffer& outputBuf, float corrDist, uint32_t gridIndex, uint32_t rowOffset = 0, uint32_t chunkY = 0) {
        // Calculate grid size matching WGSL shader: round(bound + 1 + 2*D) where D=3*corrDist
        float D = 3.0f * corrDist;
        const uint64_t pnx = static_cast<uint64_t>((maxX - minX) + 1.0f + 2.0f * D + 0.5f);
        const uint64_t pny = static_cast<uint64_t>((maxY - minY) + 1.0f + 2.0f * D + 0.5f);
        const uint64_t curTempBytes = pnx * pny * sizeof(float);
        // Use chunkY for the output grid size if chunking; otherwise use full gridSz
        const uint64_t gridBytes = (chunkY > 0) ? static_cast<uint64_t>(chunkY) * nX * sizeof(float) : gridSz * sizeof(float);
        // destOffset = channel offset (full grid) + Y offset from rowOffset
        const uint64_t fullGridBytes = static_cast<uint64_t>(nX) * nY * sizeof(float);
        // When chunking, write to staging buffer at rowOffset position; otherwise write to output buffer
        const uint64_t destOffset = (chunkY > 0) ? uint64_t(rowOffset) * nX * sizeof(float) : uint64_t(gridIndex) * fullGridBytes + uint64_t(rowOffset) * nX * sizeof(float);
        const uint64_t rngBytes = uint64_t(nCrnRng) * sizeof(RngState);

        // Clamp to tempBuf size to avoid OOB
        const uint64_t maxTempBytes = tempBufBytes(maxCorr);
        const uint64_t actualTempBytes = std::min(curTempBytes, maxTempBytes);

        // Bind group sizes must match actual buffer sizes, not curTempBytes
        // (curTempBytes may be smaller than nX*nY when corrDist < maxCorrDist)
        const uint64_t tempBufActualSize = maxTempBytes;
        const uint64_t gridBufActualSize = gridSz * sizeof(float);

        struct CRNGenUniforms
        {
            float maxX, minX, maxY, minY;
            float corrDist;
            uint32_t maxRngStates, outputGridOffset, _pad;
            uint32_t nX, nY;
            float step;
            uint32_t _pad2;
            float boundX, boundY;
            uint32_t rowOffset;
            uint32_t chunkY;
        };
        const float boundX_val = maxX - minX;
        const float boundY_val = maxY - minY;
        CRNGenUniforms genUni{maxX,
                              minX,
                              maxY,
                              minY,
                              corrDist,
                              nCrnRng,
                              0u,
                              0u,
                              (uint32_t)nX,
                              (uint32_t)nY,
                              10.0f,
                              0u,
                              boundX_val,
                              boundY_val,
                              rowOffset,
                              chunkY};
        wgpu::Buffer genUniBuf =
            makeBuffer(sizeof(genUni), WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage, &genUni);

        struct NormUniforms
        {
            uint32_t totalElements, gridOffset, _p0, _p1;
        };
        NormUniforms normUni{(uint32_t)gridSz, 0u, 0u, 0u};
        wgpu::Buffer normUniBuf = makeBuffer(sizeof(normUni),
                                             WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage,
                                             &normUni);

        // ── Fill bind group ──
        fprintf(stderr, "[DEBUG] dispatchGrid: corrDist=%.1f, curTempBytes=%llu, actualTempBytes=%llu, pnx=%llu, pny=%llu\n", corrDist, (unsigned long long)curTempBytes, (unsigned long long)actualTempBytes, (unsigned long long)pnx, (unsigned long long)pny);
        fprintf(stderr, "[DEBUG] dispatchGrid: genUniBuf=%p, tempBuf=%p, crnRngBuf=%p\n", (void*)genUniBuf, (void*)tempBuf, (void*)crnRngBuf);
        fprintf(stderr, "[DEBUG] dispatchGrid: genUni values: maxX=%.1f minX=%.1f maxY=%.1f minY=%.1f corrDist=%.1f maxRngStates=%u nX=%u nY=%u step=%.1f boundX=%.1f boundY=%.1f\n",
                genUni.maxX, genUni.minX, genUni.maxY, genUni.minY, genUni.corrDist, genUni.maxRngStates, genUni.nX, genUni.nY, genUni.step, genUni.boundX, genUni.boundY);
        auto fillLayout = crnFillPipeline_.getBindGroupLayout(0);
        std::vector<wgpu::BindGroupEntry> fillEntries(3, wgpu::Default);
        fillEntries[0].binding = 0;
        fillEntries[0].buffer = genUniBuf;
        fillEntries[0].size = sizeof(genUni);
        fillEntries[1].binding = 1;
        fillEntries[1].buffer = tempBuf;
        fillEntries[1].size = tempBufActualSize;
        fillEntries[2].binding = 2;
        fillEntries[2].buffer = crnRngBuf;
        fillEntries[2].size = rngBytes;
        wgpu::BindGroupDescriptor fillBgDesc = wgpu::Default;
        fillBgDesc.layout = fillLayout;
        fillBgDesc.entryCount = 3;
        fillBgDesc.entries = fillEntries.data();
        fprintf(stderr, "[DEBUG] dispatchGrid: creating fill bind group...\n");
        wgpu::BindGroup fillBg = device_.createBindGroup(fillBgDesc);
        fprintf(stderr, "[DEBUG] dispatchGrid: fill bind group created, fillBg=%p\n", (void*)fillBg);

        // ── Conv bind group ──
        auto convLayout = crnConvPipeline_.getBindGroupLayout(0);
        std::vector<wgpu::BindGroupEntry> convEntries(3, wgpu::Default);
        convEntries[0].binding = 0;
        convEntries[0].buffer = genUniBuf;
        convEntries[0].size = sizeof(genUni);
        convEntries[1].binding = 1;
        convEntries[1].buffer = tempBuf;
        convEntries[1].size = tempBufActualSize;
        convEntries[2].binding = 3;
        convEntries[2].buffer = gridBuf;
        convEntries[2].size = gridBufActualSize;
        wgpu::BindGroupDescriptor convBgDesc = wgpu::Default;
        convBgDesc.layout = convLayout;
        convBgDesc.entryCount = 3;
        convBgDesc.entries = convEntries.data();
        wgpu::BindGroup convBg = device_.createBindGroup(convBgDesc);

        // ── Norm bind group ──
        auto normLayout = crnNormPipeline_.getBindGroupLayout(0);
        std::vector<wgpu::BindGroupEntry> normEntries(2, wgpu::Default);
        normEntries[0].binding = 4;
        normEntries[0].buffer = normUniBuf;
        normEntries[0].size = sizeof(normUni);
        normEntries[1].binding = 5;
        normEntries[1].buffer = gridBuf;
        normEntries[1].size = gridBufActualSize;
        wgpu::BindGroupDescriptor normBgDesc = wgpu::Default;
        normBgDesc.layout = normLayout;
        normBgDesc.entryCount = 2;
        normBgDesc.entries = normEntries.data();
        wgpu::BindGroup normBg = device_.createBindGroup(normBgDesc);

        // Pass 1: fill
        {
            // Calculate dispatch workgroups using 1D grid to stay under WebGPU 65535 limit
            // chunk_total = padded_nx * chunk_ny (elements to process in this chunk)
            const uint64_t chunk_total = (chunkY > 0) ? (pnx * chunkY) : (pnx * pny);
            const uint32_t workgroups = static_cast<uint32_t>((chunk_total + 255u) / 256u);
            fprintf(stderr, "[DEBUG] dispatchGrid: dispatching %u workgroups for %llu elements\\n",
                    workgroups, (unsigned long long)chunk_total);
            wgpu::CommandEncoder enc1 = device_.createCommandEncoder(wgpu::Default);
            auto pass = enc1.beginComputePass(wgpu::Default);
            pass.setPipeline(crnFillPipeline_);
            pass.setBindGroup(0u, fillBg, (size_t)0, nullptr);
            pass.dispatchWorkgroups(workgroups, 1u, 1u);
            pass.end();
            queue_.submit(enc1.finish(wgpu::Default));
            waitIdle();
        }

        // Pass 2: convolve + normalize + copy
        {
            wgpu::CommandEncoder enc2 = device_.createCommandEncoder(wgpu::Default);
            {
                // Convolve dispatch: use 1D grid to stay under WebGPU 65535 limit
                const uint64_t chunk_total = (chunkY > 0) ? (pnx * chunkY) : (pnx * pny);
                const uint32_t convWorkgroups = static_cast<uint32_t>((chunk_total + 255u) / 256u);
                fprintf(stderr, "[DEBUG] dispatchGrid: convolve dispatching %u workgroups for %llu elements\\n",
                        convWorkgroups, (unsigned long long)chunk_total);
                auto pass1 = enc2.beginComputePass(wgpu::Default);
                pass1.setPipeline(crnConvPipeline_);
                pass1.setBindGroup(0u, convBg, (size_t)0, nullptr);
                pass1.dispatchWorkgroups(convWorkgroups, 1u, 1u);
                pass1.end();
            }
            {
                auto pass2 = enc2.beginComputePass(wgpu::Default);
                pass2.setPipeline(crnNormPipeline_);
                pass2.setBindGroup(0u, normBg, (size_t)0, nullptr);
                pass2.dispatchWorkgroups(1u, 1u, 1u);
                pass2.end();
            }
            enc2.copyBufferToBuffer(gridBuf, 0, outputBuf, destOffset, gridBytes);
            queue_.submit(enc2.finish(wgpu::Default));
            waitIdle();
        }
    };

    // ── Chunk the grid along Y axis to keep output buffers under 256MB ──
    // gridBuf size = chunkY * nX * sizeof(float); must be <= 256MB
    // Also limit chunk size to stay under WebGPU 65535 workgroup limit:
    //   maxChunkY <= (65535 * 256) / pnx where pnx ~ boundX + 2*3*corrDist
    // For pnx=4423: maxChunkY = 16777216/4423 = 3794
    // Use 128MB to get maxChunkY=7108, nChunksY=1 (reduces iterations from 8360 to 570)
    const uint64_t maxGridBufBytes = 128ULL * 1024ULL * 1024ULL;  // 128MB per chunk
    const uint64_t maxChunkY = maxGridBufBytes / (static_cast<uint64_t>(nX) * sizeof(float));
    // Clamp to respect WebGPU 65535 workgroup limit per dimension:
    // workgroups = (pnx * chunkY + 255) / 256 <= 65535
    // => pnx * chunkY <= 16776961  (pnx <= nX, so nX is safe upper bound)
    const uint64_t wgLimitChunkY = 16776961u / static_cast<uint64_t>(nX);
    const uint64_t clampedMaxChunkY = std::min(maxChunkY, std::max(static_cast<uint64_t>(1u), wgLimitChunkY));
    const uint64_t nChunksY = (static_cast<uint64_t>(nY) + clampedMaxChunkY - 1) / clampedMaxChunkY;
    fprintf(stderr, "[DEBUG] CRN chunking: nX=%d nY=%d maxChunkY=%llu nChunksY=%llu\n",
            nX, nY, (unsigned long long)clampedMaxChunkY, (unsigned long long)nChunksY);

    // Also chunk the output buffers: create them on CPU side to avoid GPU memory limit
    // LOS: nSite * 8 channels, NLOS: nSite * 7 channels, O2I: nSite * 7 channels
    const size_t losBufSize = static_cast<size_t>(nSite_) * 8ULL * static_cast<uint64_t>(nX) * nY;
    const size_t nlosBufSize = static_cast<size_t>(nSite_) * 7ULL * static_cast<uint64_t>(nX) * nY;
    const size_t o2iBufSize = static_cast<size_t>(nSite_) * 7ULL * static_cast<uint64_t>(nX) * nY;
    std::vector<float> losOutBuf(losBufSize, 0.0f);
    std::vector<float> nlosOutBuf(nlosBufSize, 0.0f);
    std::vector<float> o2iOutBuf(o2iBufSize, 0.0f);
    fprintf(stderr, "[DEBUG] CRN: GPU output buffers created (los=%lluMB, nlos=%lluMB, o2i=%lluMB)\n",
            (unsigned long long)(losOutBuf.size() * sizeof(float)) / (1024 * 1024),
            (unsigned long long)(nlosOutBuf.size() * sizeof(float)) / (1024 * 1024),
            (unsigned long long)(o2iOutBuf.size() * sizeof(float)) / (1024 * 1024));

    // Create staging buffers for chunked output (small enough to fit in GPU memory)
    const uint64_t stagingGridBytes = static_cast<uint64_t>(clampedMaxChunkY) * nX * sizeof(float);
    wgpu::Buffer stagingBuf = makeBuffer(stagingGridBytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst);
    fprintf(stderr, "[DEBUG] CRN: stagingBuf created (%llu bytes)\n", (unsigned long long)stagingGridBytes);

    fprintf(stderr, "[DEBUG] Starting CRN generation loop\n");
    for (uint32_t s = 0; s < nSite_; s++)
    {
        fprintf(stderr, "[DEBUG] LOS site %u\n", s);
        for (uint32_t ch = 0; ch < 8; ch++)
        {
            for (uint32_t c = 0; c < nChunksY; c++)
            {
                const uint32_t yOff = static_cast<uint32_t>(c * clampedMaxChunkY);
                const uint32_t yRows = static_cast<uint32_t>(std::min(static_cast<uint64_t>(clampedMaxChunkY),
                                                                       static_cast<uint64_t>(nY) - c * clampedMaxChunkY));
                const uint32_t gridIdx = static_cast<uint32_t>(s) * 8 + ch;
                fprintf(stderr, "[DEBUG]   dispatchGrid LOS s=%u ch=%u c=%u yOff=%u yRows=%u\n", s, ch, c, yOff, yRows);
                // Use stagingBuf as output instead of crnLosBuf_
                dispatchGrid(stagingBuf, corrLos[ch], gridIdx, yOff, yRows);
                // Copy staging buffer to CPU output buffer
                const uint64_t cpuOffset = static_cast<uint64_t>(s) * 8 * nX * nY +
                                           static_cast<uint64_t>(ch) * nX * nY +
                                           static_cast<uint64_t>(yOff) * nX;
                const uint64_t chunkBytes = static_cast<uint64_t>(yRows) * nX * sizeof(float);
                std::vector<float> stagingData(chunkBytes / sizeof(float));
                wgpu::Buffer stagingReadBuf = makeBuffer(chunkBytes, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
                // Copy staging to stagingReadBuf
                {
                    wgpu::CommandEncoder enc = device_.createCommandEncoder(wgpu::Default);
                    enc.copyBufferToBuffer(stagingBuf, 0, stagingReadBuf, 0, chunkBytes);
                    queue_.submit(enc.finish(wgpu::Default));
                    waitIdle();
                }
                // Map and read using wgpu-native C API
                static WGPUMapMode g_mapStatus = WGPUMapMode_None;
                auto g_callback = [](WGPUMapAsyncStatus status, WGPUStringView message, void* userdata, void* /*another*/) {
                    (void)message;
                    *(reinterpret_cast<WGPUMapMode*>(userdata)) = status == WGPUMapAsyncStatus_Success ? WGPUMapMode_Read : WGPUMapMode_None;
                };
                WGPUBufferMapCallbackInfo mapCbInfo = {};
                mapCbInfo.nextInChain = nullptr;
                mapCbInfo.callback = g_callback;
                mapCbInfo.userdata1 = &g_mapStatus;
                mapCbInfo.userdata2 = nullptr;
                wgpuBufferMapAsync(stagingReadBuf, WGPUMapMode_Read, 0, static_cast<uint64_t>(chunkBytes), mapCbInfo);
                waitIdle();
                if (g_mapStatus != WGPUMapMode_Read) {
                    fprintf(stderr, "[ERROR] Failed to map stagingReadBuf\n");
                    continue;
                }
                const void* mapped = wgpuBufferGetConstMappedRange(stagingReadBuf, 0, chunkBytes);
                if (!mapped) {
                    fprintf(stderr, "[ERROR] Failed to get mapped range\n");
                    continue;
                }
                memcpy(stagingData.data(), mapped, chunkBytes);
                wgpuBufferUnmap(stagingReadBuf);
                // Copy to CPU buffer
                for (uint64_t i = 0; i < chunkBytes / sizeof(float); i++)
                {
                    losOutBuf[cpuOffset + i] = stagingData[i];
                }
                fprintf(stderr, "[DEBUG]   LOS chunk %u copied to CPU buffer\n", c);
            }
        }
    }
    for (uint32_t s = 0; s < nSite_; s++)
    {
        fprintf(stderr, "[DEBUG] NLOS site %u\n", s);
        for (uint32_t ch = 0; ch < 7; ch++)
        {
            for (uint32_t c = 0; c < nChunksY; c++)
            {
                const uint32_t yOff = static_cast<uint32_t>(c * clampedMaxChunkY);
                const uint32_t yRows = static_cast<uint32_t>(std::min(static_cast<uint64_t>(clampedMaxChunkY),
                                                                       static_cast<uint64_t>(nY) - c * clampedMaxChunkY));
                const uint32_t gridIdx = static_cast<uint32_t>(s) * 7 + ch;
                fprintf(stderr, "[DEBUG]   dispatchGrid NLOS s=%u ch=%u c=%u yOff=%u yRows=%u\n", s, ch, c, yOff, yRows);
                dispatchGrid(stagingBuf, corrNlos[ch], gridIdx, yOff, yRows);
                const uint64_t cpuOffset = static_cast<uint64_t>(s) * 7 * nX * nY +
                                           static_cast<uint64_t>(ch) * nX * nY +
                                           static_cast<uint64_t>(yOff) * nX;
                const uint64_t chunkBytes = static_cast<uint64_t>(yRows) * nX * sizeof(float);
                std::vector<float> stagingData(chunkBytes / sizeof(float));
                wgpu::Buffer stagingReadBuf = makeBuffer(chunkBytes, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
                {
                    wgpu::CommandEncoder enc = device_.createCommandEncoder(wgpu::Default);
                    enc.copyBufferToBuffer(stagingBuf, 0, stagingReadBuf, 0, chunkBytes);
                    queue_.submit(enc.finish(wgpu::Default));
                    waitIdle();
                }
                // Map and read using wgpu-native C API
                static WGPUMapMode g_mapStatusNlos = WGPUMapMode_None;
                auto g_callbackNlos = [](WGPUMapAsyncStatus status, WGPUStringView message, void* userdata, void* /*another*/) {
                    (void)message;
                    *(reinterpret_cast<WGPUMapMode*>(userdata)) = status == WGPUMapAsyncStatus_Success ? WGPUMapMode_Read : WGPUMapMode_None;
                };
                WGPUBufferMapCallbackInfo mapCbInfoNlos = {};
                mapCbInfoNlos.nextInChain = nullptr;
                mapCbInfoNlos.callback = g_callbackNlos;
                mapCbInfoNlos.userdata1 = &g_mapStatusNlos;
                mapCbInfoNlos.userdata2 = nullptr;
                wgpuBufferMapAsync(stagingReadBuf, WGPUMapMode_Read, 0, static_cast<uint64_t>(chunkBytes), mapCbInfoNlos);
                waitIdle();
                if (g_mapStatusNlos != WGPUMapMode_Read) {
                    fprintf(stderr, "[ERROR] Failed to map stagingReadBuf (NLOS)\n");
                    continue;
                }
                const void* mappedNlos = wgpuBufferGetConstMappedRange(stagingReadBuf, 0, chunkBytes);
                if (!mappedNlos) {
                    fprintf(stderr, "[ERROR] Failed to get mapped range (NLOS)\n");
                    continue;
                }
                memcpy(stagingData.data(), mappedNlos, chunkBytes);
                wgpuBufferUnmap(stagingReadBuf);
                for (uint64_t i = 0; i < chunkBytes / sizeof(float); i++)
                {
                    nlosOutBuf[cpuOffset + i] = stagingData[i];
                }
                fprintf(stderr, "[DEBUG]   NLOS chunk %u copied to CPU buffer\n", c);
            }
        }
    }
    for (uint32_t s = 0; s < nSite_; s++)
    {
        fprintf(stderr, "[DEBUG] O2I site %u\n", s);
        for (uint32_t ch = 0; ch < 7; ch++)
        {
            for (uint32_t c = 0; c < nChunksY; c++)
            {
                const uint32_t yOff = static_cast<uint32_t>(c * clampedMaxChunkY);
                const uint32_t yRows = static_cast<uint32_t>(std::min(static_cast<uint64_t>(clampedMaxChunkY),
                                                                       static_cast<uint64_t>(nY) - c * clampedMaxChunkY));
                const uint32_t gridIdx = static_cast<uint32_t>(s) * 7 + ch;
                fprintf(stderr, "[DEBUG]   dispatchGrid O2I s=%u ch=%u c=%u yOff=%u yRows=%u\n", s, ch, c, yOff, yRows);
                dispatchGrid(stagingBuf, corrO2i[ch], gridIdx, yOff, yRows);
                const uint64_t cpuOffset = static_cast<uint64_t>(s) * 7 * nX * nY +
                                           static_cast<uint64_t>(ch) * nX * nY +
                                           static_cast<uint64_t>(yOff) * nX;
                const uint64_t chunkBytes = static_cast<uint64_t>(yRows) * nX * sizeof(float);
                std::vector<float> stagingData(chunkBytes / sizeof(float));
                wgpu::Buffer stagingReadBuf = makeBuffer(chunkBytes, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
                {
                    wgpu::CommandEncoder enc = device_.createCommandEncoder(wgpu::Default);
                    enc.copyBufferToBuffer(stagingBuf, 0, stagingReadBuf, 0, chunkBytes);
                    queue_.submit(enc.finish(wgpu::Default));
                    waitIdle();
                }
                // Map and read using wgpu-native C API
                static WGPUMapMode g_mapStatusO2i = WGPUMapMode_None;
                auto g_callbackO2i = [](WGPUMapAsyncStatus status, WGPUStringView message, void* userdata, void* /*another*/) {
                    (void)message;
                    *(reinterpret_cast<WGPUMapMode*>(userdata)) = status == WGPUMapAsyncStatus_Success ? WGPUMapMode_Read : WGPUMapMode_None;
                };
                WGPUBufferMapCallbackInfo mapCbInfoO2i = {};
                mapCbInfoO2i.nextInChain = nullptr;
                mapCbInfoO2i.callback = g_callbackO2i;
                mapCbInfoO2i.userdata1 = &g_mapStatusO2i;
                mapCbInfoO2i.userdata2 = nullptr;
                wgpuBufferMapAsync(stagingReadBuf, WGPUMapMode_Read, 0, static_cast<uint64_t>(chunkBytes), mapCbInfoO2i);
                waitIdle();
                if (g_mapStatusO2i != WGPUMapMode_Read) {
                    fprintf(stderr, "[ERROR] Failed to map stagingReadBuf (O2I)\n");
                    continue;
                }
                const void* mappedO2i = wgpuBufferGetConstMappedRange(stagingReadBuf, 0, chunkBytes);
                if (!mappedO2i) {
                    fprintf(stderr, "[ERROR] Failed to get mapped range (O2I)\n");
                    continue;
                }
                memcpy(stagingData.data(), mappedO2i, chunkBytes);
                wgpuBufferUnmap(stagingReadBuf);
                for (uint64_t i = 0; i < chunkBytes / sizeof(float); i++)
                {
                    o2iOutBuf[cpuOffset + i] = stagingData[i];
                }
                fprintf(stderr, "[DEBUG]   O2I chunk %u copied to CPU buffer\n", c);
            }
        }
    }
    fprintf(stderr, "[DEBUG] All CRN generation complete\n");
    
    // Create GPU-side CRN buffers and copy CPU data to them
    if (!crnLosBuf_) {
        crnLosBuf_ = makeBuffer(losOutBuf.size() * sizeof(float),
                                WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
                                losOutBuf.data());
        // Initialize offset buffer directly using mappedAtCreation
        std::vector<uint32_t> losOffsets(nSite_ * 8);
        for (uint32_t i = 0; i < nSite_ * 8; i++) {
            losOffsets[i] = i * nX_ * nY_;
        }
        crnLosOffBuf_ = makeBuffer(nSite_ * 8ULL * sizeof(uint32_t),
                                   WGPUBufferUsage_Storage,
                                   losOffsets.data());
    }
    if (!crnNlosBuf_) {
        crnNlosBuf_ = makeBuffer(nlosOutBuf.size() * sizeof(float),
                                 WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
                                 nlosOutBuf.data());
        std::vector<uint32_t> nlosOffsets(nSite_ * 7);
        for (uint32_t i = 0; i < nSite_ * 7; i++) {
            nlosOffsets[i] = i * nX_ * nY_;
        }
        crnNlosOffBuf_ = makeBuffer(nSite_ * 7ULL * sizeof(uint32_t),
                                    WGPUBufferUsage_Storage,
                                    nlosOffsets.data());
    }
    if (!crnO2iBuf_) {
        crnO2iBuf_ = makeBuffer(o2iOutBuf.size() * sizeof(float),
                                WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
                                o2iOutBuf.data());
        std::vector<uint32_t> o2iOffsets(nSite_ * 7);
        for (uint32_t i = 0; i < nSite_ * 7; i++) {
            o2iOffsets[i] = i * nX_ * nY_;
        }
        crnO2iOffBuf_ = makeBuffer(nSite_ * 7ULL * sizeof(uint32_t),
                                   WGPUBufferUsage_Storage,
                                   o2iOffsets.data());
    }
}

// ── calLinkParam ──────────────────────────────────────────────────────────────
void
SlsChanWgpu::calLinkParam(uint32_t nSite,
                          uint32_t nUT,
                          uint32_t nSectorPerSite,
                          float maxX,
                          float minX,
                          float maxY,
                          float minY,
                         bool updatePL,
                         bool updateAllLSPs,
                         bool updateLos,
                         bool updateOptionalPL,
                         int32_t nX,
                         int32_t nY)
{
    assert(!isDead());

    struct LinkParamUniforms
    {
        float maxX, minX, maxY, minY;
        uint32_t nSite, nUT, nSectorPerSite;
        uint32_t updatePL, updateAllLSPs, updateLos, updateOptionalPL;
        int32_t nX, nY;
    };

    nSite_ = nSite;

    LinkParamUniforms uniData{maxX,
                              minX,
                              maxY,
                              minY,
                              nSite,
                              nUT,
                              nSectorPerSite,
                              (uint32_t)updatePL,
                              (uint32_t)updateAllLSPs,
                              (uint32_t)updateLos,
                              (uint32_t)updateOptionalPL,
                              nX,
                              nY};
    wgpu::Buffer uniBuf = makeBuffer(sizeof(uniData), WGPUBufferUsage_Uniform, &uniData);

    const uint64_t cellParamsSz = uint64_t(nSite) * sizeof(CellParam);
    const uint64_t utParamsSz = nUT * sizeof(UtParam);
    const uint64_t linkParamsSz = uint64_t(nSite) * nUT * sizeof(LinkParams);
    const uint64_t rngStatesSz = uint64_t(nSite) * nUT * sizeof(RngState);

    if (!linkParamsBuf_)
    {
        linkParamsBuf_ =
            makeBuffer(linkParamsSz, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    }
    if (!cellParamsBuf_)
    {
        cellParamsBuf_ =
            makeBuffer(cellParamsSz, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    }
    if (!utParamsBuf_)
    {
        utParamsBuf_ = makeBuffer(utParamsSz, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    }

    if (!sysConfigBuf_)
    {
        SystemLevelConfigGPU slc{0, 1, 0, 0, -1.0f, -1.0f, {0, 0}};
        sysConfigBuf_ = makeBuffer(sizeof(SystemLevelConfigGPU),
                                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
                                   &slc);
    }
    if (!simConfigBuf_)
    {
        SimConfigGPU sc{3.5e9f, 0, 0, 0};
        simConfigBuf_ = makeBuffer(sizeof(SimConfigGPU),
                                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
                                   &sc);
    }
    if (!cmnLinkBuf_)
    {
        CmnLinkParamsGPU cl{};
        for (int i = 0; i < 7; i++)
        {
            cl.sqrtCorrMatLos[i * 7 + i] = 1.0f;
        }
        for (int i = 0; i < 6; i++)
        {
            cl.sqrtCorrMatNlos[i * 6 + i] = 1.0f;
        }
        for (int i = 0; i < 6; i++)
        {
            cl.sqrtCorrMatO2i[i * 6 + i] = 1.0f;
        }
        cl.lgfc = std::log10(3.5f);
        cl.mu_K[0] = 9.0f;
        cl.sigma_K[0] = 3.5f;
        cl.mu_lgDS[0] = -7.03f;
        cl.sigma_lgDS[0] = 0.66f;
        cl.mu_lgDS[1] = -6.44f;
        cl.sigma_lgDS[1] = 0.39f;
        cl.mu_lgDS[2] = -6.62f;
        cl.sigma_lgDS[2] = 0.32f;
        cl.mu_lgASD[0] = 1.15f;
        cl.sigma_lgASD[0] = 0.28f;
        cl.mu_lgASD[1] = 1.41f;
        cl.sigma_lgASD[1] = 0.28f;
        cl.mu_lgASD[2] = 1.25f;
        cl.sigma_lgASD[2] = 0.42f;
        cl.mu_lgASA[0] = 1.81f;
        cl.sigma_lgASA[0] = 0.20f;
        cl.mu_lgASA[1] = 1.87f;
        cl.sigma_lgASA[1] = 0.20f;
        cl.mu_lgASA[2] = 1.76f;
        cl.sigma_lgASA[2] = 0.16f;
        cl.mu_lgZSA[0] = 0.95f;
        cl.sigma_lgZSA[0] = 0.16f;
        cl.mu_lgZSA[1] = 1.26f;
        cl.sigma_lgZSA[1] = 0.35f;
        cl.mu_lgZSA[2] = 1.01f;
        cl.sigma_lgZSA[2] = 0.43f;
        cmnLinkBuf_ =
            makeBuffer(sizeof(cl), WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc, &cl);
    }
    if (!rngStatesBuf_)
    {
        std::vector<RngState> seeds(nSite * nUT);
        for (uint32_t i = 0; i < nSite * nUT; ++i)
        {
            uint32_t h = i + 1;
            seeds[i] = {h * 747796405u + 1,
                        h * 2246822519u + 1,
                        h * 2654435769u + 1,
                        h * 3266489917u + 1};
        }
        rngStatesBuf_ = makeBuffer(rngStatesSz,
                                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
                                   seeds.data());
    }

    if (!stagingBuf_)
    {
        stagingBuf_ = makeBuffer(linkParamsSz, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    }

    auto bg0Layout = linkParamPipeline_.getBindGroupLayout(0);
    // Bindings 0-19 required by WGSL shader, but we only use 6-19
    std::vector<wgpu::BindGroupEntry> entries(20, wgpu::Default);
    auto E = [&](int i, uint32_t binding, wgpu::Buffer buf, uint64_t size = WGPU_WHOLE_SIZE) {
        entries[i].binding = binding;
        entries[i].buffer = buf;
        entries[i].offset = 0;
        entries[i].size = size;
        if (!buf)
        {
            std::cerr << "null buffer at entry " << i << " (binding " << binding << ")" << std::endl;
            abort();
        }
    };
    E(6, 6, uniBuf, sizeof(uniData));
    E(7, 7, cellParamsBuf_, cellParamsSz);
    E(8, 8, utParamsBuf_, utParamsSz);
    E(9, 9, sysConfigBuf_, sizeof(SystemLevelConfigGPU));
    E(10, 10, simConfigBuf_, sizeof(SimConfigGPU));
    E(11, 11, cmnLinkBuf_, sizeof(CmnLinkParamsGPU));
    E(12, 12, linkParamsBuf_, linkParamsSz);
    E(13, 13, rngStatesBuf_, rngStatesSz);
    E(14, 14, crnLosBuf_, nSite_ * 8ULL * uint64_t(nX_) * nY_ * sizeof(float));
    E(15, 15, crnNlosBuf_, nSite_ * 7ULL * uint64_t(nX_) * nY_ * sizeof(float));
    E(16, 16, crnO2iBuf_, nSite_ * 7ULL * uint64_t(nX_) * nY_ * sizeof(float));
    E(17, 17, crnLosOffBuf_, nSite_ * 8ULL * sizeof(uint32_t));
    E(18, 18, crnNlosOffBuf_, nSite_ * 7ULL * sizeof(uint32_t));
    E(19, 19, crnO2iOffBuf_, nSite_ * 7ULL * sizeof(uint32_t));

    // Debug: verify all buffers are valid
    fprintf(stderr, "[DEBUG] calLinkParam: checking bind group buffers...\n");
    for (int i = 0; i < 20; i++) {
        if (entries[i].buffer) {
            fprintf(stderr, "[DEBUG]   entry[%d] (binding %u): buf=%p size=%llu\n", i, entries[i].binding, (void*)entries[i].buffer, (unsigned long long)entries[i].size);
        } else {
            fprintf(stderr, "[DEBUG]   entry[%d] (binding %u): NULL BUFFER!\n", i, entries[i].binding);
        }
    }

    wgpu::BindGroupDescriptor bgDesc = wgpu::Default;
    bgDesc.layout = bg0Layout;
    bgDesc.entryCount = (uint32_t)entries.size();
    bgDesc.entries = entries.data();
    wgpu::BindGroup bg0 = device_.createBindGroup(bgDesc);

    wgpu::CommandEncoder enc = device_.createCommandEncoder(wgpu::Default);
    auto pass = enc.beginComputePass(wgpu::Default);
    pass.setPipeline(linkParamPipeline_);
    pass.setBindGroup(0u, bg0, (size_t)0, nullptr);
    pass.dispatchWorkgroups(nSite, (nUT + 255u) / 256u, 1u);
    pass.end();
    enc.copyBufferToBuffer(linkParamsBuf_, 0, stagingBuf_, 0, linkParamsSz);
    wgpu::CommandBufferDescriptor cbDesc = wgpu::Default;
    assert(!isDead());
    queue_.submit(enc.finish(cbDesc));
    waitIdle();
    assert(!isDead());
}

// ── uploadSmallScaleConfig ────────────────────────────────────────────────────
// Populates BOTH binding 20 (uni_sim / SmallScaleSimConfig)
//           AND binding 21 (uni_sys / SmallScaleSysConfig).
void
SlsChanWgpu::uploadSmallScaleConfig(float scSpacingHz,
                                    uint32_t fftSize,
                                    uint32_t nPrb,
                                    uint32_t nPrbg,
                                    uint32_t nSnapshotPerSlot,
                                    uint32_t enablePropagationDelay,
                                    uint32_t disableSmallScaleFading,
                                    uint32_t disablePlShadowing,
                                    uint32_t optionalCfrDim,
                                    float lambda0)
{
    // binding 20 — uni_sim
    SmallScaleSimConfig sc{};
    sc.scSpacingHz = scSpacingHz;
    sc.fftSize = fftSize;
    sc.nPrb = nPrb;
    sc.nPrbg = nPrbg;
    sc.nSnapshotPerSlot = nSnapshotPerSlot;
    sc.enablePropagationDelay = enablePropagationDelay;
    sc.disableSmallScaleFading = disableSmallScaleFading;
    sc.disablePlShadowing = disablePlShadowing;
    sc.optionalCfrDim = optionalCfrDim;
    sc.lambda0 = lambda0;
    ssSimConfigBuf_ =
        makeBuffer(sizeof(sc), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopySrc, &sc);
    ssNPrbg_ = nPrbg;

    // binding 21 — uni_sys
    SmallScaleSysConfig sys{};
    sys.enablePropagationDelay = enablePropagationDelay;
    sys.disableSmallScaleFading = disableSmallScaleFading;
    sys.disablePlShadowing = disablePlShadowing;
    sys._pad0 = 0u;
    ssSysConfigBuf_ =
        makeBuffer(sizeof(sys), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopySrc, &sys);
}

void
SlsChanWgpu::uploadCmnLinkParamsSmallScale(const SsCmnParams& cmn)
{
    ssCmnLinkBuf_ = makeBuffer(sizeof(cmn), WGPUBufferUsage_Storage, &cmn);
}

// ── calClusterRay ─────────────────────────────────────────────────────────────
void
SlsChanWgpu::calClusterRay(uint32_t nSite, uint32_t nUT)
{
    assert(ssSimConfigBuf_ && "call uploadSmallScaleConfig before calClusterRay");
    assert(ssCellParamsBuf_ && "call uploadCellParamsSS before calClusterRay");
    assert(ssUtParamsBuf_ && "call uploadUtParamsSS before calClusterRay");
    assert(ssCmnLinkBuf_ && "call uploadCmnLinkParamsSmallScale before calClusterRay");
    assert(!isDead());

    const uint64_t clusterSz = uint64_t(nSite) * nUT * sizeof(ClusterParamsGpu);
    if (!clusterParamsBuf_)
    {
        clusterParamsBuf_ =
            makeBuffer(clusterSz, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    }

    // Build dispatch uniform first — it goes into the single bind group
    struct DispUni
    {
        uint32_t nSite, nUT, nActiveLinks;
        float refTime, cfr_norm;
        uint32_t _p0, _p1, _p2;
    };

    DispUni du{nSite, nUT, 0u, 0.0f, 1.0f, 0, 0, 0};
    ssDispatchBuf_ = makeBuffer(sizeof(du), WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage, &du);

    constexpr uint32_t MAXCLUSTERS = 20;
    constexpr uint32_t MAXRAYS = 20;
    constexpr uint32_t MAXCR = MAXCLUSTERS * MAXRAYS;                       // 400
    constexpr uint32_t MAXCR4 = MAXCLUSTERS * MAXRAYS * 4;                  // 1600
    size_t szXpr = uint64_t(nSite) * nUT * MAXCR * sizeof(float);           // 400 f32/link
    size_t szRandomPhases = uint64_t(nSite) * nUT * MAXCR4 * sizeof(float); // 1600 f32/link
    size_t szPhinmAoA = uint64_t(nSite) * nUT * MAXCR * sizeof(float);      // 400 f32/link
    size_t szPhinmAoD = uint64_t(nSite) * nUT * MAXCR * sizeof(float);
    size_t szThetanmZOA = uint64_t(nSite) * nUT * MAXCR * sizeof(float);
    size_t szThetanmZOD = uint64_t(nSite) * nUT * MAXCR * sizeof(float);
    xpr_Buf_ =
        makeBuffer(szXpr,
                   WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    randomPhases_Buf_ =
        makeBuffer(szRandomPhases,
                   WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    phi_nm_AoA_Buf_ =
        makeBuffer(szPhinmAoA,
                   WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    phi_nm_AoD_Buf_ =
        makeBuffer(szPhinmAoD,
                   WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    theta_nm_ZOA_Buf_ =
        makeBuffer(szThetanmZOA,
                   WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    theta_nm_ZOD_Buf_ =
        makeBuffer(szThetanmZOD,
                   WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);

    auto layout1 = clusterRayPipeline_.getBindGroupLayout(1);
    std::vector<wgpu::BindGroupEntry> e0(12, wgpu::Default);
    auto E = [](std::vector<wgpu::BindGroupEntry>& v,
                int i,
                uint32_t b,
                wgpu::Buffer buf,
                uint64_t sz = WGPU_WHOLE_SIZE) {
        v[i].binding = b;
        v[i].buffer = buf;
        v[i].size = sz;
    };
    E(e0, 0, 0, linkParamsBuf_);      // cray_buf_link
    E(e0, 1, 1, ssUtParamsBuf_);      // cray_buf_ut
    E(e0, 2, 2, ssCmnLinkBuf_, sizeof(SsCmnParams));       // cray_buf_cmn (non-array storage)
    E(e0, 3, 3, clusterParamsBuf_);   // cray_buf_cluster
    E(e0, 4, 4, rngStatesBuf_);       // cray_buf_rng
    E(e0, 5, 5, ssDispatchBuf_);      // cray_disp
    E(e0, 6, 6, xpr_Buf_);            // cray_buf_xpr
    E(e0, 7, 7, randomPhases_Buf_);   // cray_buf_randomPhases
    E(e0, 8, 8, phi_nm_AoA_Buf_);     // cray_buf_phi_nm_AoA
    E(e0, 9, 9, phi_nm_AoD_Buf_);     // cray_buf_phi_nm_AoD
    E(e0, 10, 10, theta_nm_ZOA_Buf_); //  cray_buf_theta_nm_ZOA
    E(e0, 11, 11, theta_nm_ZOD_Buf_); //  cray_buf_theta_nm_ZOD

    wgpu::BindGroupDescriptor bgd1 = wgpu::Default;
    bgd1.layout = layout1;
    bgd1.entryCount = e0.size();
    bgd1.entries = e0.data();
    auto bg0 = emptyBg(clusterRayPipeline_, 0);
    auto bg1 = device_.createBindGroup(bgd1);

    auto enc = device_.createCommandEncoder(wgpu::Default);
    auto pass = enc.beginComputePass(wgpu::Default);
    pass.setPipeline(clusterRayPipeline_);
    pass.setBindGroup(0u, bg0, (size_t)0, nullptr);
    pass.setBindGroup(1u, bg1, (size_t)0, nullptr);
    pass.dispatchWorkgroups(nSite, (nUT + 255u) / 256u, 1u);
    pass.end();
    assert(!isDead());
    queue_.submit(enc.finish(wgpu::Default));
    waitIdle();
    assert(!isDead());
}

// ── generateCIR ───────────────────────────────────────────────────────────────
void
SlsChanWgpu::generateCIR(const std::vector<ActiveLink>& activeLinks,
                         uint32_t nActiveLinks,
                         uint32_t nSnapshots,
                         float refTime)
{
    assert(ssSimConfigBuf_ && "call uploadSmallScaleConfig before generateCIR");
    assert(ssSysConfigBuf_ && "call uploadSmallScaleConfig before generateCIR");
    assert(ssCellParamsBuf_ && "call uploadCellParamsSS before generateCIR");
    assert(ssUtParamsBuf_ && "call uploadUtParamsSS before generateCIR");
    assert(ssCmnLinkBuf_ && "call uploadCmnLinkParamsSmallScale before generateCIR");
    assert(!isDead());

    activeLinkBuf_ = makeBuffer(activeLinks.size() * sizeof(ActiveLink),
                                WGPUBufferUsage_Storage,
                                activeLinks.data());

    if (!cirCoeBuf_)
    {
        const uint64_t cirCoeElems =
            uint64_t(nActiveLinks) * nSnapshots * ssNBsAnt_ * ssNUeAnt_ * 24u;
        cirCoeBuf_ = makeBuffer(cirCoeElems * sizeof(float) * 2,
                                WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    }
    if (!cirNormDelayBuf_)
    {
        cirNormDelayBuf_ = makeBuffer(uint64_t(nActiveLinks) * 24u * sizeof(uint32_t),
                                      WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    }
    if (!cirNtapsBuf_)
    {
        cirNtapsBuf_ = makeBuffer(uint64_t(nActiveLinks) * sizeof(uint32_t),
                                  WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    }

    struct DispUni
    {
        uint32_t nSite, nUT, nActiveLinks;
        float refTime, cfr_norm;
        uint32_t _p0, _p1, _p2;
    };

    DispUni du{0u, 0u, nActiveLinks, refTime, 1.0f, 0, 0, 0};
    ssDispatchBuf_ = makeBuffer(sizeof(du), WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage, &du);

    auto layout2 = generateCIRPipeline_.getBindGroupLayout(2);
    std::vector<wgpu::BindGroupEntry> e0(21, wgpu::Default);
    auto E = [](std::vector<wgpu::BindGroupEntry>& v,
                int i,
                uint32_t b,
                wgpu::Buffer buf,
                uint64_t sz = WGPU_WHOLE_SIZE) {
        v[i].binding = b;
        v[i].buffer = buf;
        v[i].size = sz;
    };
    E(e0, 0, 0, ssSimConfigBuf_);     // cir_uni_sim
    E(e0, 1, 1, ssSysConfigBuf_);     // cir_uni_sys
    E(e0, 2, 2, ssCmnLinkBuf_);       // cir_buf_cmn
    E(e0, 3, 3, ssCellParamsBuf_);    // cir_buf_cell
    E(e0, 4, 4, ssUtParamsBuf_);      // cir_buf_ut
    E(e0, 5, 5, linkParamsBuf_);      // cir_buf_link
    E(e0, 6, 6, clusterParamsBuf_);   // cir_buf_cluster
    E(e0, 7, 7, antPanelConfigBuf_);  // cir_buf_antCfg
    E(e0, 8, 8, antThetaBuf_);        // cir_buf_antTheta
    E(e0, 9, 9, antPhiBuf_);          // cir_buf_antPhi
    E(e0, 10, 10, activeLinkBuf_);    // cir_buf_activeLink
    E(e0, 11, 11, cirCoeBuf_);        // cir_buf_cirCoe
    E(e0, 12, 12, cirNormDelayBuf_);  // cir_buf_cirNormDelay
    E(e0, 13, 13, cirNtapsBuf_);      // cir_buf_cirNtaps
    E(e0, 14, 14, ssDispatchBuf_);    // cir_disp
    E(e0, 15, 15, xpr_Buf_);          // cir_buf_xpr
    E(e0, 16, 16, randomPhases_Buf_); // cir_buf_randomPhases
    E(e0, 17, 17, phi_nm_AoA_Buf_);   // cir_buf_phi_nm_AoA
    E(e0, 18, 18, phi_nm_AoD_Buf_);   // cir_buf_phi_nm_AoD
    E(e0, 19, 19, theta_nm_ZOA_Buf_); // cir_buf_theta_nm_ZOA
    E(e0, 20, 20, theta_nm_ZOD_Buf_); // cir_buf_theta_nm_ZOD

    wgpu::BindGroupDescriptor bgd2 = wgpu::Default;
    bgd2.layout = layout2;
    bgd2.entryCount = e0.size();
    bgd2.entries = e0.data();
    auto bg0 = emptyBg(generateCIRPipeline_, 0);
    auto bg1 = emptyBg(generateCIRPipeline_, 1);
    auto bg2 = device_.createBindGroup(bgd2);

    auto enc = device_.createCommandEncoder(wgpu::Default);
    auto pass = enc.beginComputePass(wgpu::Default);
    pass.setPipeline(generateCIRPipeline_);
    pass.setBindGroup(0u, bg0, (size_t)0, nullptr);
    pass.setBindGroup(1u, bg1, (size_t)0, nullptr);
    pass.setBindGroup(2u, bg2, (size_t)0, nullptr);
    pass.dispatchWorkgroups(nActiveLinks, nSnapshots, 1u);
    pass.end();
    assert(!isDead());
    queue_.submit(enc.finish(wgpu::Default));
    waitIdle();
    assert(!isDead());
}

// ── generateCFR ───────────────────────────────────────────────────────────────
void
SlsChanWgpu::generateCFR(const std::vector<ActiveLink>& activeLinks,
                         uint32_t nActiveLinks,
                         uint32_t nSnapshots)
{
    assert(ssSimConfigBuf_ && "call uploadSmallScaleConfig before generateCFR");
    assert(ssCellParamsBuf_ && "call uploadCellParamsSS before generateCFR");
    assert(ssUtParamsBuf_ && "call uploadUtParamsSS before generateCFR");
    assert(antPanelConfigBuf_ && "call uploadAntPanelConfigs before generateCFR");
    assert(activeLinkBuf_ && "call generateCIR before generateCFR (missing activeLinkBuf_)");
    assert(cirCoeBuf_ && "call generateCIR before generateCFR (missing cirCoeBuf_)");
    assert(cirNormDelayBuf_ && "call generateCIR before generateCFR (missing cirNormDelayBuf_)");
    assert(cirNtapsBuf_ && "call generateCIR before generateCFR (missing cirNtapsBuf_)");
    assert(!isDead());

    // Probe device health before doing anything
    wgpuDevicePoll(device_, false, nullptr);
    std::cerr << "CFR entry: device alive" << std::endl;

    if (!freqChanPrbgBuf_)
    {
        const uint64_t cfrElems =
            uint64_t(nActiveLinks) * nSnapshots * ssNBsAnt_ * ssNUeAnt_ * ssNPrbg_;
        const uint64_t bufBytes = cfrElems * sizeof(float) * 2;
        fprintf(stderr,
                "CFR: cfrElems=%llu, freqChanPrbgBuf size=%.1f MB\n",
                (unsigned long long)cfrElems,
                (double)bufBytes / (1024.0 * 1024.0));
        freqChanPrbgBuf_ = makeBuffer(cfrElems * sizeof(float) * 2,
                                      WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    }

    struct DispUni
    {
        uint32_t nSite, nUT, nActiveLinks;
        float refTime, cfr_norm;
        uint32_t _p0, _p1, _p2;
    };

    DispUni du{0u, 0u, nActiveLinks, 0.0f, 1.0f, 0, 0, 0};
    ssDispatchBuf_ = makeBuffer(sizeof(du), WGPUBufferUsage_Uniform, &du);

    auto layout3 = generateCFRPipeline_.getBindGroupLayout(3);
    std::vector<wgpu::BindGroupEntry> e0(10, wgpu::Default);
    auto E = [](std::vector<wgpu::BindGroupEntry>& v,
                int i,
                uint32_t b,
                wgpu::Buffer buf,
                uint64_t sz = WGPU_WHOLE_SIZE) {
        v[i].binding = b;
        v[i].buffer = buf;
        v[i].size = sz;
    };
    E(e0, 0, 0, ssSimConfigBuf_);            // cfr_uni_sim
    E(e0, 1, 1, ssCellParamsBuf_);           // cfr_buf_cell
    E(e0, 2, 2, ssUtParamsBuf_);             // cfr_buf_ut
    E(e0, 3, 3, antPanelConfigBuf_);         // cfr_buf_antCfg
    E(e0, 4, 4, activeLinkBuf_);             // cfr_buf_activeLink
    E(e0, 5, 5, cirCoeBuf_);                 // cfr_buf_cirCoe
    E(e0, 6, 6, cirNormDelayBuf_);           // cfr_buf_cirNormDelay
    E(e0, 7, 7, cirNtapsBuf_);               // cfr_buf_cirNtaps
    E(e0, 8, 8, freqChanPrbgBuf_);           // cfr_buf_freqChanPrbg
    E(e0, 9, 9, ssDispatchBuf_, sizeof(du)); // cfr_disp

    wgpu::BindGroupDescriptor bgd3 = wgpu::Default;
    bgd3.layout = layout3;
    bgd3.entryCount = e0.size();
    bgd3.entries = e0.data();
    auto bg0 = emptyBg(generateCFRPipeline_, 0);
    auto bg1 = emptyBg(generateCFRPipeline_, 1);
    auto bg2 = emptyBg(generateCFRPipeline_, 2);
    auto bg3 = device_.createBindGroup(bgd3);

    auto enc = device_.createCommandEncoder(wgpu::Default);
    auto pass = enc.beginComputePass(wgpu::Default);
    pass.setPipeline(generateCFRPipeline_);
    pass.setBindGroup(0u, bg0, (size_t)0, nullptr);
    pass.setBindGroup(1u, bg1, (size_t)0, nullptr);
    pass.setBindGroup(2u, bg2, (size_t)0, nullptr);
    pass.setBindGroup(3u, bg3, (size_t)0, nullptr);
    pass.dispatchWorkgroups(nActiveLinks, nSnapshots, 1u);
    pass.end();
    assert(!isDead());
    queue_.submit(enc.finish(wgpu::Default));
    waitIdle();
    assert(!isDead());
}

// ── Batched CFR (splits large active-link sets into smaller GPU batches) ────────
void
SlsChanWgpu::generateCFRBatched(const std::vector<ActiveLink>& activeLinks,
                                uint32_t nActiveLinks,
                                uint32_t nSnapshots)
{
    assert(ssSimConfigBuf_ && "call uploadSmallScaleConfig before generateCFR");
    assert(ssCellParamsBuf_ && "call uploadCellParamsSS before generateCFR");
    assert(ssUtParamsBuf_ && "call uploadUtParamsSS before generateCFR");
    assert(antPanelConfigBuf_ && "call uploadAntPanelConfigs before generateCFR");
    assert(activeLinkBuf_ && "call generateCIR before generateCFR (missing activeLinkBuf_)");
    assert(cirCoeBuf_ && "call generateCIR before generateCFR (missing cirCoeBuf_)");
    assert(cirNormDelayBuf_ && "call generateCIR before generateCFR (missing cirNormDelayBuf_)");
    assert(cirNtapsBuf_ && "call generateCIR before generateCFR (missing cirNtapsBuf_)");
    assert(!isDead());

    // Precompute per-link element counts
    const uint32_t elemsPerLink = nSnapshots * ssNUeAnt_ * ssNBsAnt_ * ssNPrbg_;

    // Host-side accumulator for CFR results (float2 per element)
    const uint64_t totalHostElems = uint64_t(nActiveLinks) * elemsPerLink;
    std::vector<std::complex<float>> cfrHostBuf(totalHostElems);

    // Batch the active links
    const uint32_t nBatches = (nActiveLinks + CFR_BATCH_SIZE - 1u) / CFR_BATCH_SIZE;
    for (uint32_t b = 0; b < nBatches; ++b)
    {
        const uint32_t batchStart = b * CFR_BATCH_SIZE;
        const uint32_t batchLen = std::min(CFR_BATCH_SIZE, nActiveLinks - batchStart);
        if (batchLen == 0u)
            break;

        // Allocate per-batch GPU buffer
        const uint64_t batchElems = uint64_t(batchLen) * elemsPerLink;
        const uint64_t batchBufBytes = batchElems * sizeof(float) * 2;

        // Check if device is healthy before allocating
        wgpuDevicePoll(device_, false, nullptr);
        if (isDead())
        {
            std::cerr << "CFR batch " << b << ": device lost before buffer alloc" << std::endl;
            return;
        }

        auto batchBuf = makeBuffer(batchBufBytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc);

        // Build per-batch active-link info: same layout as ActiveLink for WGSL
        struct BatchLinkInfo
        {
            uint32_t cid;
            uint32_t uid;
            uint32_t linkIdx;
            uint32_t lspReadIdx;
            uint32_t cirCoeOffset;
            uint32_t cirNormDelayOffset;
            uint32_t cirNtapsOffset;
            uint32_t freqChanPrbgOffset; // offset into the per-batch output buffer
        };
        std::vector<BatchLinkInfo> batchLinks(batchLen);
        for (uint32_t i = 0; i < batchLen; ++i)
        {
            batchLinks[i].cid = activeLinks[batchStart + i].cid;
            batchLinks[i].uid = activeLinks[batchStart + i].uid;
            batchLinks[i].linkIdx = activeLinks[batchStart + i].linkIdx;
            batchLinks[i].lspReadIdx = activeLinks[batchStart + i].lspReadIdx;
            batchLinks[i].cirCoeOffset = activeLinks[batchStart + i].cirCoeOffset;
            batchLinks[i].cirNormDelayOffset = activeLinks[batchStart + i].cirNormDelayOffset;
            batchLinks[i].cirNtapsOffset = activeLinks[batchStart + i].cirNtapsOffset;
            batchLinks[i].freqChanPrbgOffset = i * elemsPerLink; // relative offset within per-batch buffer
        }
        auto batchLinkBuf = makeBuffer(sizeof(BatchLinkInfo) * batchLen,
                                       WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                                       batchLinks.data());

        // Dispatch uniforms
        struct DispUni
        {
            uint32_t nSite, nUT, nActiveLinks;
            float refTime, cfr_norm;
            uint32_t _p0, _p1, _p2;
        };
        DispUni du{0u, 0u, batchLen, 0.0f, 1.0f, 0, 0, 0};
        auto ssDispatchBuf = makeBuffer(sizeof(du), WGPUBufferUsage_Uniform, &du);

        // Build bind group
        auto layout3 = generateCFRPipeline_.getBindGroupLayout(3);
        std::vector<wgpu::BindGroupEntry> entries(10, wgpu::Default);
        auto E = [](std::vector<wgpu::BindGroupEntry>& v,
                    int i,
                    uint32_t b,
                    wgpu::Buffer buf,
                    uint64_t sz = WGPU_WHOLE_SIZE) {
            v[i].binding = b;
            v[i].buffer = buf;
            v[i].size = sz;
        };
        E(entries, 0, 0, ssSimConfigBuf_);
        E(entries, 1, 1, ssCellParamsBuf_);
        E(entries, 2, 2, ssUtParamsBuf_);
        E(entries, 3, 3, antPanelConfigBuf_);
        E(entries, 4, 4, batchLinkBuf);
        E(entries, 5, 5, cirCoeBuf_);
        E(entries, 6, 6, cirNormDelayBuf_);
        E(entries, 7, 7, cirNtapsBuf_);
        E(entries, 8, 8, batchBuf);
        E(entries, 9, 9, ssDispatchBuf, sizeof(du));

        wgpu::BindGroupDescriptor bgd = wgpu::Default;
        bgd.layout = layout3;
        bgd.entryCount = entries.size();
        bgd.entries = entries.data();
        auto bg0 = emptyBg(generateCFRPipeline_, 0);
        auto bg1 = emptyBg(generateCFRPipeline_, 1);
        auto bg2 = emptyBg(generateCFRPipeline_, 2);
        auto bg3 = device_.createBindGroup(bgd);

        // Compute pass — dispatch CFR per batch
        auto enc = device_.createCommandEncoder(wgpu::Default);
        auto pass = enc.beginComputePass(wgpu::Default);
        pass.setPipeline(generateCFRPipeline_);
        pass.setBindGroup(0u, bg0, (size_t)0, nullptr);
        pass.setBindGroup(1u, bg1, (size_t)0, nullptr);
        pass.setBindGroup(2u, bg2, (size_t)0, nullptr);
        pass.setBindGroup(3u, bg3, (size_t)0, nullptr);
        pass.dispatchWorkgroups(batchLen, nSnapshots, 1u);
        pass.end();
        assert(!isDead());
        queue_.submit(enc.finish(wgpu::Default));
        waitIdle();
        assert(!isDead());

        // Read back this batch (create a NEW encoder — the previous one is already finished)
        auto staging = makeBuffer(batchBufBytes, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
        auto enc2 = device_.createCommandEncoder(wgpu::Default);
        enc2.copyBufferToBuffer(batchBuf, 0, staging, 0, batchBufBytes);
        queue_.submit(enc2.finish(wgpu::Default));
        waitIdle();

        // Copy into host accumulator
        auto mapped = mapReadBuffer<std::complex<float>>(staging, batchBufBytes);
        for (uint32_t i = 0; i < batchLen; ++i)
        {
            const uint64_t base = uint64_t(batchStart + i) * elemsPerLink;
            const uint64_t len = uint64_t(elemsPerLink);
            for (uint64_t j = 0; j < len; ++j)
            {
                cfrHostBuf[base + j] = mapped[i * elemsPerLink + j];
            }
        }
    }

    // Store result in member for later readback by readFreqChanPrbgBatched
    m_cfrBatchedResult_ = std::move(cfrHostBuf);
    m_cfrBatchedNActiveLinks_ = nActiveLinks;
    m_cfrBatchedNSnapshots_ = nSnapshots;

    std::cerr << "CFR batched done: " << nActiveLinks << " links in " << nBatches << " batches" << std::endl;
}

void
SlsChanWgpu::readFreqChanPrbgBatched(std::vector<std::complex<float>>& outBuf)
{
    assert(m_cfrBatchedNActiveLinks_ > 0 && "call generateCFRBatched first");
    outBuf = m_cfrBatchedResult_;
}
std::vector<std::complex<float>>
SlsChanWgpu::readCirCoe(uint32_t nActiveLinks,
                        uint32_t nSnapshots,
                        uint32_t nUtAnt,
                        uint32_t nBsAnt)
{
    const uint64_t sz = uint64_t(nActiveLinks) * nSnapshots * nUtAnt * nBsAnt * 24 * 8;
    if (!cirStagingBuf_)
    {
        cirStagingBuf_ = makeBuffer(sz, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    }
    auto enc = device_.createCommandEncoder(wgpu::Default);
    enc.copyBufferToBuffer(cirCoeBuf_, 0, cirStagingBuf_, 0, sz);
    queue_.submit(enc.finish(wgpu::Default));
    waitIdle();
    return mapReadBuffer<std::complex<float>>(cirStagingBuf_, sz);
}

std::vector<std::complex<float>>
SlsChanWgpu::readFreqChanPrbg(uint32_t nActiveLinks,
                              uint32_t nSnapshots,
                              uint32_t nUtAnt,
                              uint32_t nBsAnt)
{
    const uint64_t sz =
        uint64_t(nActiveLinks) * nSnapshots * nUtAnt * nBsAnt * ssNPrbg_ * sizeof(float) * 2;
    if (!cfrStagingBuf_)
    {
        cfrStagingBuf_ = makeBuffer(sz, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    }
    auto enc = device_.createCommandEncoder(wgpu::Default);
    enc.copyBufferToBuffer(freqChanPrbgBuf_, 0, cfrStagingBuf_, 0, sz);
    queue_.submit(enc.finish(wgpu::Default));
    waitIdle();
    return mapReadBuffer<std::complex<float>>(cfrStagingBuf_, sz);
}

void
SlsChanWgpu::invalidateOutputBuffers()
{
    cirCoeBuf_ = {};
    cirNormDelayBuf_ = {};
    cirNtapsBuf_ = {};
    freqChanPrbgBuf_ = {};
    cirStagingBuf_ = {};
    cfrStagingBuf_ = {};
}

wgpu::BindGroup
SlsChanWgpu::emptyBg(wgpu::ComputePipeline& pip, uint32_t slot)
{
    wgpu::BindGroupDescriptor d = wgpu::Default;
    d.layout = pip.getBindGroupLayout(slot);
    d.entryCount = 0;
    d.entries = nullptr;
    return device_.createBindGroup(d);
}

bool
SlsChanWgpu::isDead()
{
    WGPUBufferDescriptor probeDesc{};
    probeDesc.size = 4;
    probeDesc.usage = WGPUBufferUsage_CopyDst;

    WGPUBuffer probe = wgpuDeviceCreateBuffer(device_, &probeDesc);
    if (probe)
    {
        fprintf(stderr, "Device alive\n");
        return false;
    }
    else
    {
        fprintf(stderr, "Device dead\n");
        return true;
    }
}

std::vector<ClusterParamsGpu>
SlsChanWgpu::readClusterParams(uint32_t nSite, uint32_t nUT)
{
    assert(clusterParamsBuf_ && "call calClusterRay before readClusterParams");

    const uint64_t sz = uint64_t(nSite) * nUT * sizeof(ClusterParamsGpu);
    wgpu::Buffer staging = makeBuffer(sz, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);

    wgpu::CommandEncoder enc = device_.createCommandEncoder(wgpu::Default);
    enc.copyBufferToBuffer(clusterParamsBuf_, 0, staging, 0, sz);
    queue_.submit(enc.finish(wgpu::Default));
    waitIdle();

    return mapReadBuffer<ClusterParamsGpu>(staging, sz);
}

std::vector<uint32_t>
SlsChanWgpu::readCirNtaps()
{
    const uint64_t sz = cirNtapsBuf_.getSize();
    wgpu::Buffer staging = makeBuffer(sz, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    auto enc = device_.createCommandEncoder(wgpu::Default);
    enc.copyBufferToBuffer(cirNtapsBuf_, 0, staging, 0, sz);
    queue_.submit(enc.finish(wgpu::Default));
    waitIdle();
    return mapReadBuffer<uint32_t>(staging, sz);
}

std::vector<float>
SlsChanWgpu::readXpr()
{
    const uint64_t sz = xpr_Buf_.getSize();
    wgpu::Buffer staging = makeBuffer(sz, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    auto enc = device_.createCommandEncoder(wgpu::Default);
    enc.copyBufferToBuffer(xpr_Buf_, 0, staging, 0, sz);
    queue_.submit(enc.finish(wgpu::Default));
    waitIdle();
    return mapReadBuffer<float>(staging, sz);
}

std::vector<float>
SlsChanWgpu::readPhiNmAoA()
{
    const uint64_t sz = phi_nm_AoA_Buf_.getSize();
    wgpu::Buffer staging = makeBuffer(sz, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    auto enc = device_.createCommandEncoder(wgpu::Default);
    enc.copyBufferToBuffer(phi_nm_AoA_Buf_, 0, staging, 0, sz);
    queue_.submit(enc.finish(wgpu::Default));
    waitIdle();
    return mapReadBuffer<float>(staging, sz);
}

std::vector<float>
SlsChanWgpu::readPhiNmAoD()
{
    const uint64_t sz = phi_nm_AoD_Buf_.getSize();
    wgpu::Buffer staging = makeBuffer(sz, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    auto enc = device_.createCommandEncoder(wgpu::Default);
    enc.copyBufferToBuffer(phi_nm_AoD_Buf_, 0, staging, 0, sz);
    queue_.submit(enc.finish(wgpu::Default));
    waitIdle();
    return mapReadBuffer<float>(staging, sz);
}

std::vector<float>
SlsChanWgpu::readThetaNmZOA()
{
    const uint64_t sz = theta_nm_ZOA_Buf_.getSize();
    wgpu::Buffer staging = makeBuffer(sz, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    auto enc = device_.createCommandEncoder(wgpu::Default);
    enc.copyBufferToBuffer(theta_nm_ZOA_Buf_, 0, staging, 0, sz);
    queue_.submit(enc.finish(wgpu::Default));
    waitIdle();
    return mapReadBuffer<float>(staging, sz);
}

std::vector<float>
SlsChanWgpu::readThetaNmZOD()
{
    const uint64_t sz = theta_nm_ZOD_Buf_.getSize();
    wgpu::Buffer staging = makeBuffer(sz, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    auto enc = device_.createCommandEncoder(wgpu::Default);
    enc.copyBufferToBuffer(theta_nm_ZOD_Buf_, 0, staging, 0, sz);
    queue_.submit(enc.finish(wgpu::Default));
    waitIdle();
    return mapReadBuffer<float>(staging, sz);
}

// ── Save all channel metrics to HDF5 (matches NVIDIA slsChan::saveSlsChanToH5File) ──
#ifdef SLS_CHAN_HDF5
namespace {

void
createGroup(hid_t loc, const char* name)
{
    if (H5Lexists(loc, name, H5P_DEFAULT) <= 0)
    {
        hid_t grp = H5Gcreate2(loc, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Gclose(grp);
    }
}

void
writeDatasetFloat(hid_t loc, const char* name, const float* data, hsize_t count)
{
    hid_t dset = H5Tcopy(H5T_NATIVE_FLOAT);
    hid_t space = H5Screate_simple(1, &count, nullptr);
    hid_t dsetId = H5Dcreate(loc, name, dset, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dsetId, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dsetId);
    H5Sclose(space);
    H5Tclose(dset);
}

void
writeDatasetUint32(hid_t loc, const char* name, const uint32_t* data, hsize_t count)
{
    hid_t dset = H5Tcopy(H5T_NATIVE_UINT32);
    hid_t space = H5Screate_simple(1, &count, nullptr);
    hid_t dsetId = H5Dcreate(loc, name, dset, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dsetId, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dsetId);
    H5Sclose(space);
    H5Tclose(dset);
}

} // namespace

void
saveSlsChanToHdf5(
    const std::string& filename,
    const std::vector<LinkParams>& links,
    uint32_t nSite, uint32_t nUT,
    const std::vector<ClusterParamsGpu>& clusterParams,
    const std::vector<ActiveLink>& activeLinks,
    const std::vector<std::complex<float>>& cirCoe,
    const std::vector<uint32_t>& cirNormDelay,
    const std::vector<uint32_t>& cirNtaps,
    const std::vector<std::complex<float>>& cfrPrbg,
    uint32_t nPrbg,
    const std::vector<float>& xpr,
    const std::vector<float>& phiNmAoA,
    const std::vector<float>& phiNmAoD,
    const std::vector<float>& thetaNmZOA,
    const std::vector<float>& thetaNmZOD,
    float scSpacingHz, uint32_t fftSize, uint32_t nPrb,
    uint32_t nSnapshotPerSlot, float centerFreqHz, float bandwidthHz,
    uint32_t nUeAnt, uint32_t nBsAnt,
    const SsCmnParams& ssCmn,
    const std::vector<CellParam>& cells,
    const std::vector<CellParamSS>& cellsSS,
    const std::vector<UtParam>& uts,
    float isd, float bsHeight, float minBsUeDist2d,
    float maxBsUeDist2dIndoor, float indoorUtPercent,
    uint32_t nSectorPerSite)
{
    const uint32_t nLinks = nSite * nUT;
    const uint32_t nCells = nSite * nSectorPerSite;

    const size_t nCFRComplexElems = cfrPrbg.size();

    // Open HDF5 file
    // On Windows, H5F_ACC_TRUNC may fail (GetLastError == 33) if the file
    // is already open by another process.  Check first and give a clear
    // message instead of cascading HDF5 errors.
    if (std::filesystem::exists(filename))
    {
        std::cerr << "saveSlsChanToHdf5: " << filename
                  << " already exists — close it in any external program and retry"
                  << std::endl;
        return;
    }

    hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0)
    {
        std::cerr << "saveSlsChanToHdf5: failed to create " << filename << std::endl;
        return;
    }

    // ── simConfig (compound dataset for Python compatibility) ──
    {
        struct SimConfigRecord {
            uint32_t link_sim_ind;
            float center_freq_hz;
            float bandwidth_hz;
            float sc_spacing_hz;
            uint32_t fft_size;
            uint32_t n_prb;
            uint32_t n_prbg;
            uint32_t n_snapshot_per_slot;
            uint32_t run_mode;
            uint32_t internal_memory_mode;
            uint32_t freq_convert_type;
            uint32_t sc_sampling;
            uint32_t proc_sig_freq;
        };

        hid_t compoundType = H5Tcreate(H5T_COMPOUND, sizeof(SimConfigRecord));
        H5Tinsert(compoundType, "link_sim_ind",  0 * 4, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "center_freq_hz", 1 * 4, H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "bandwidth_hz",   2 * 4, H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "sc_spacing_hz",  3 * 4, H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "fft_size",       4 * 4, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "n_prb",          5 * 4, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "n_prbg",         6 * 4, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "n_snapshot_per_slot", 7 * 4, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "run_mode",       8 * 4, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "internal_memory_mode", 9 * 4, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "freq_convert_type", 10 * 4, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "sc_sampling",    11 * 4, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "proc_sig_freq",  12 * 4, H5T_NATIVE_UINT32);

        SimConfigRecord sc{};
        sc.link_sim_ind          = 0;  // not available from caller
        sc.center_freq_hz        = centerFreqHz;
        sc.bandwidth_hz          = bandwidthHz;
        sc.sc_spacing_hz         = scSpacingHz;
        sc.fft_size              = fftSize;
        sc.n_prb                 = nPrb;
        sc.n_prbg               = nPrbg;
        sc.n_snapshot_per_slot   = nSnapshotPerSlot;
        sc.run_mode              = 0;   // not available
        sc.internal_memory_mode  = 0;   // not available
        sc.freq_convert_type     = 0;   // not available
        sc.sc_sampling           = 0;   // not available
        sc.proc_sig_freq         = 0;   // not available

        hsize_t dims = 1;
        hid_t dataspace = H5Screate_simple(1, &dims, nullptr);
        hid_t dataset = H5Dcreate2(file, "simConfig", compoundType, dataspace,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t memspace = H5Screate_simple(1, &dims, nullptr);
        H5Dwrite(dataset, compoundType, memspace, dataspace, H5P_DEFAULT, &sc);
        H5Sclose(memspace);
        H5Sclose(dataspace);
        H5Tclose(compoundType);
        H5Dclose(dataset);
    }

    // ── systemLevelConfig (required by Python analysis scripts) ──
    {
        struct SystemLevelConfigRecord
        {
            uint32_t scenario;
            float isd;
            uint32_t n_site;
            uint32_t n_sector_per_site;
            uint32_t n_ut;
            int32_t optional_pl_ind;
            int32_t o2i_building_penetr_loss_ind;
            int32_t o2i_car_penetr_loss_ind;
            int32_t enable_near_field_effect;
            int32_t enable_non_stationarity;
            float force_los_prob[1];
            float force_ut_speed[1];
            float force_indoor_ratio;
            int32_t disable_pl_shadowing;
            int32_t disable_small_scale_fading;
            int32_t enable_per_tti_lsp;
            int32_t enable_propagation_delay;
            int32_t ut_drop_option;
        };

        hsize_t dim1 = 1;
        hid_t arrType1f = H5Tarray_create2(H5T_NATIVE_FLOAT, 1, &dim1);
        hid_t compoundType = H5Tcreate(H5T_COMPOUND, sizeof(SystemLevelConfigRecord));

        H5Tinsert(compoundType, "scenario",                         0,                  H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "isd",                              4,                  H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "n_site",                           8,                  H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "n_sector_per_site",               12,                 H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "n_ut",                            16,                 H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "optional_pl_ind",                 20,                 H5T_NATIVE_INT32);
        H5Tinsert(compoundType, "o2i_building_penetr_loss_ind",    24,                 H5T_NATIVE_INT32);
        H5Tinsert(compoundType, "o2i_car_penetr_loss_ind",        28,                 H5T_NATIVE_INT32);
        H5Tinsert(compoundType, "enable_near_field_effect",       32,                 H5T_NATIVE_INT32);
        H5Tinsert(compoundType, "enable_non_stationarity",        36,                 H5T_NATIVE_INT32);
        H5Tinsert(compoundType, "force_los_prob",                 40,                 arrType1f);
        H5Tinsert(compoundType, "force_ut_speed",                 44,                 arrType1f);
        H5Tinsert(compoundType, "force_indoor_ratio",             48,                 H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "disable_pl_shadowing",           52,                 H5T_NATIVE_INT32);
        H5Tinsert(compoundType, "disable_small_scale_fading",     56,                 H5T_NATIVE_INT32);
        H5Tinsert(compoundType, "enable_per_tti_lsp",             60,                 H5T_NATIVE_INT32);
        H5Tinsert(compoundType, "enable_propagation_delay",       64,                 H5T_NATIVE_INT32);
        H5Tinsert(compoundType, "ut_drop_option",                 68,                 H5T_NATIVE_INT32);

        SystemLevelConfigRecord slc{};
        slc.scenario                          = 0; // UMa
        slc.isd                               = isd;
        slc.n_site                            = nSite;
        slc.n_sector_per_site                 = nSectorPerSite;
        slc.n_ut                              = nUT;
        slc.optional_pl_ind                   = 0;
        slc.o2i_building_penetr_loss_ind      = 0;
        slc.o2i_car_penetr_loss_ind           = 0;
        slc.enable_near_field_effect          = 0;
        slc.enable_non_stationarity           = 0;
        slc.force_los_prob[0]                 = 0.0f;
        slc.force_ut_speed[0]                 = 0.0f;
        slc.force_indoor_ratio                = indoorUtPercent / 100.0f;
        slc.disable_pl_shadowing              = 0;
        slc.disable_small_scale_fading        = 0;
        slc.enable_per_tti_lsp                = 0;
        slc.enable_propagation_delay          = 0;
        slc.ut_drop_option                    = 0;

        hsize_t dims = 1;
        hid_t dataspace = H5Screate_simple(1, &dims, nullptr);
        hid_t dataset = H5Dcreate2(file, "systemLevelConfig", compoundType, dataspace,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t memspace = H5Screate_simple(1, &dims, nullptr);
        H5Dwrite(dataset, compoundType, memspace, dataspace, H5P_DEFAULT, &slc);
        H5Sclose(memspace);
        H5Sclose(dataspace);
        H5Tclose(compoundType);
        H5Tclose(arrType1f);
        H5Dclose(dataset);
    }

    // ── linkParams (compound dataset for Python compatibility) ──
    {
        hid_t compoundType = H5Tcreate(H5T_COMPOUND, sizeof(LinkParamsHdf5));
        // Offset 0: cid (4)
        H5Tinsert(compoundType, "cid", 0 * sizeof(uint32_t), H5T_NATIVE_UINT32);
        // Offset 4: d2d (4)
        H5Tinsert(compoundType, "d2d", 1 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 8: d2d_in (4)
        H5Tinsert(compoundType, "d2d_in", 2 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 12: d2d_out (4)
        H5Tinsert(compoundType, "d2d_out", 3 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 16: d3d (4)
        H5Tinsert(compoundType, "d3d", 4 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 20: d3d_in (4)
        H5Tinsert(compoundType, "d3d_in", 5 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 24: d3d_out (4)
        H5Tinsert(compoundType, "d3d_out", 6 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 28: phi_LOS_AOD (4)
        H5Tinsert(compoundType, "phi_LOS_AOD", 7 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 32: theta_LOS_ZOD (4)
        H5Tinsert(compoundType, "theta_LOS_ZOD", 8 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 36: phi_LOS_AOA (4)
        H5Tinsert(compoundType, "phi_LOS_AOA", 9 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 40: theta_LOS_ZOA (4)
        H5Tinsert(compoundType, "theta_LOS_ZOA", 10 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 44: losInd (4)
        H5Tinsert(compoundType, "losInd", 11 * sizeof(float), H5T_NATIVE_UINT32);
        // Offset 48: pathloss (4)
        H5Tinsert(compoundType, "pathloss", 12 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 52: SF (4)
        H5Tinsert(compoundType, "SF", 13 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 56: K (4)
        H5Tinsert(compoundType, "K", 14 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 60: DS (4)
        H5Tinsert(compoundType, "DS", 15 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 64: ASD (4)
        H5Tinsert(compoundType, "ASD", 16 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 68: ASA (4)
        H5Tinsert(compoundType, "ASA", 17 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 72: ZSD (4)
        H5Tinsert(compoundType, "ZSD", 18 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 76: ZSA (4)
        H5Tinsert(compoundType, "ZSA", 19 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 80: mu_lgZSD (4)
        H5Tinsert(compoundType, "mu_lgZSD", 20 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 84: sigma_lgZSD (4)
        H5Tinsert(compoundType, "sigma_lgZSD", 21 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 88: mu_offset_ZOD (4)
        H5Tinsert(compoundType, "mu_offset_ZOD", 22 * sizeof(float), H5T_NATIVE_FLOAT);
        // Offset 92: delta_tau (4)
        H5Tinsert(compoundType, "delta_tau", 23 * sizeof(float), H5T_NATIVE_FLOAT);

        // Cast from LinkParams -> LinkParamsHdf5
        std::vector<LinkParamsHdf5> hdf5Links(nLinks);
        for (uint32_t i = 0; i < nLinks; ++i)
        {
            const LinkParams& lk = links[i];
            hdf5Links[i].cid          = activeLinks[i].cid;  // Serving cell (sector) ID
            hdf5Links[i].d2d          = lk.d2d;
            hdf5Links[i].d2d_in       = lk.d2d_in;
            hdf5Links[i].d2d_out      = lk.d2d_out;
            hdf5Links[i].d3d          = lk.d3d;
            hdf5Links[i].d3d_in       = lk.d3d_in;
            hdf5Links[i].d3d_out      = lk.d3d_out;
            hdf5Links[i].phi_LOS_AOD  = lk.phi_LOS_AOD;
            hdf5Links[i].theta_LOS_ZOD= lk.theta_LOS_ZOD;
            hdf5Links[i].phi_LOS_AOA  = lk.phi_LOS_AOA;
            hdf5Links[i].theta_LOS_ZOA= lk.theta_LOS_ZOA;
            hdf5Links[i].losInd       = lk.losInd;
            hdf5Links[i].pathloss     = lk.pathloss;
            hdf5Links[i].SF           = lk.SF;
            hdf5Links[i].K            = lk.K;
            hdf5Links[i].DS           = lk.DS;
            hdf5Links[i].ASD          = lk.ASD;
            hdf5Links[i].ASA          = lk.ASA;
            hdf5Links[i].ZSD          = lk.ZSD;
            hdf5Links[i].ZSA          = lk.ZSA;
            hdf5Links[i].mu_lgZSD     = lk.mu_lgZSD;
            hdf5Links[i].sigma_lgZSD  = lk.sigma_lgZSD;
            hdf5Links[i].mu_offset_ZOD= lk.mu_offset_ZOD;
            hdf5Links[i].delta_tau    = lk._pad; // delta_tau was 0 in the original code
        }

        hsize_t dims = static_cast<hsize_t>(nLinks);
        hid_t dataspace = H5Screate_simple(1, &dims, nullptr);
        hid_t dataset = H5Dcreate2(file, "linkParams", compoundType, dataspace,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t memspace = H5Screate_simple(1, &dims, nullptr);
        H5Dwrite(dataset, compoundType, memspace, dataspace, H5P_DEFAULT, hdf5Links.data());
        H5Sclose(memspace);
        H5Sclose(dataspace);
        H5Tclose(compoundType);
        H5Dclose(dataset);
    }

    // ── clusterParams (compound dataset for Python compatibility) ──
    {
        // Define compound datatype matching ClusterParamsGpu layout (496 bytes)
        hid_t compoundType = H5Tcreate(H5T_COMPOUND, sizeof(ClusterParamsGpu));

        // Scalar fields
        H5Tinsert(compoundType, "nCluster", 0, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "nRayPerCluster", 4, H5T_NATIVE_UINT32);

        // Fixed-size array fields using H5Tarray_create2 (requires hsize_t*)
        hsize_t dim20 = static_cast<hsize_t>(MAX_CLUSTERS);
        hid_t float20 = H5Tarray_create2(H5T_NATIVE_FLOAT, 1, &dim20);
        H5Tinsert(compoundType, "delays", 8, float20);
        H5Tinsert(compoundType, "powers", 88, float20); // delays(80) + 8 = 88

        hsize_t dim2 = 2;
        hid_t uint2 = H5Tarray_create2(H5T_NATIVE_UINT32, 1, &dim2);
        H5Tinsert(compoundType, "strongest2clustersIdx", 168, uint2); // powers(80) + 8 = 168

        H5Tinsert(compoundType, "phi_n_AoA", 176, float20); // strongest2(8) + 168 = 176
        H5Tinsert(compoundType, "phi_n_AoD", 256, float20);
        H5Tinsert(compoundType, "theta_n_ZOA", 336, float20);
        H5Tinsert(compoundType, "theta_n_ZOD", 416, float20);

        H5Tclose(float20);
        H5Tclose(uint2);

        hsize_t dims = nLinks;
        hid_t dataspace = H5Screate_simple(1, &dims, nullptr);
        hid_t memspace = H5Screate_simple(1, &dims, nullptr);
        hid_t dataset = H5Dcreate2(file, "clusterParams", compoundType, dataspace,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dataset, compoundType, memspace, dataspace, H5P_DEFAULT, clusterParams.data());
        H5Sclose(memspace);
        H5Sclose(dataspace);
        H5Tclose(compoundType);
        H5Dclose(dataset);
    }

    // ── Per-ray small-scale parameters (separate datasets for Python compatibility) ──
    {
        // xpr, phiNmAoA, phiNmAoD, thetaNmZOA, thetaNmZOD are per-ray arrays
        // size = nLinks * MAX_RAYS where MAX_RAYS = MAX_CLUSTERS * MAX_CLUSTERS = 400
        const uint32_t MAX_RAYS = MAX_CLUSTERS * MAX_CLUSTERS;
        const hsize_t nRayElems = static_cast<hsize_t>(nLinks) * MAX_RAYS;

        hid_t dset = H5Tcopy(H5T_NATIVE_FLOAT);
        hid_t space = H5Screate_simple(1, &nRayElems, nullptr);

        // xpr
        {
            hid_t dsetId = H5Dcreate(file, "xpr", dset, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dsetId, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, xpr.data());
            H5Dclose(dsetId);
        }
        // phi_n_m_AoA
        {
            hid_t dsetId = H5Dcreate(file, "phi_n_m_AoA", dset, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dsetId, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, phiNmAoA.data());
            H5Dclose(dsetId);
        }
        // phi_n_m_AoD
        {
            hid_t dsetId = H5Dcreate(file, "phi_n_m_AoD", dset, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dsetId, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, phiNmAoD.data());
            H5Dclose(dsetId);
        }
        // theta_n_m_ZOA
        {
            hid_t dsetId = H5Dcreate(file, "theta_n_m_ZOA", dset, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dsetId, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, thetaNmZOA.data());
            H5Dclose(dsetId);
        }
        // theta_n_m_ZOD
        {
            hid_t dsetId = H5Dcreate(file, "theta_n_m_ZOD", dset, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dsetId, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, thetaNmZOD.data());
            H5Dclose(dsetId);
        }

        H5Sclose(space);
        H5Tclose(dset);
    }

    // ── activeLinkParams (compound dataset for Python compatibility) ──
    {
        // Define compound datatype
        hid_t compoundType = H5Tcreate(H5T_COMPOUND, sizeof(uint32_t) * 4);
        H5Tinsert(compoundType, "cid", 0 * sizeof(uint32_t), H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "uid", 1 * sizeof(uint32_t), H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "linkIdx", 2 * sizeof(uint32_t), H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "lspReadIdx", 3 * sizeof(uint32_t), H5T_NATIVE_UINT32);
        
        struct ActiveLinkRecord {
            uint32_t cid;
            uint32_t uid;
            uint32_t linkIdx;
            uint32_t lspReadIdx;
        };
        
        std::vector<ActiveLinkRecord> activeLinkRecords(nLinks);
        for (uint32_t i = 0; i < nLinks; ++i) {
            activeLinkRecords[i].cid = activeLinks[i].cid;
            activeLinkRecords[i].uid = activeLinks[i].uid;
            activeLinkRecords[i].linkIdx = activeLinks[i].linkIdx;
            activeLinkRecords[i].lspReadIdx = activeLinks[i].lspReadIdx;
        }
        
        hsize_t dims = nLinks;
        
        hid_t dataspace = H5Screate_simple(1, &dims, &dims);
        hid_t dataset = H5Dcreate2(file, "activeLinkParams", compoundType, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t memspace = H5Screate_simple(1, &dims, nullptr);
        H5Dwrite(dataset, compoundType, memspace, dataspace, H5P_DEFAULT, activeLinkRecords.data());
        H5Sclose(memspace);
        H5Sclose(dataspace);
        H5Tclose(compoundType);
        H5Dclose(dataset);
    }

    // ── cirPerCell ── (per-cell complex CIR datasets for Python compatibility)
    createGroup(file, "cirPerCell");
    hid_t grpCir = H5Gopen2(file, "cirPerCell", H5P_DEFAULT);

    // Build per-cell grouping: map cell_id -> list of active link indices
    std::map<uint32_t, std::vector<uint32_t>> cellToLinks;
    for (uint32_t i = 0; i < nLinks; ++i)
    {
        cellToLinks[activeLinks[i].cid].push_back(i);
    }

    // CIR data layout per link: [nSnapshots, nBsAnt, nUeAnt, NMAXTAPS]
    constexpr uint32_t NMAXTAPS = 24;
    const uint32_t nSnapshots = nSnapshotPerSlot;
    const uint32_t nBsAntN = nBsAnt;
    const uint32_t nUeAntN = nUeAnt;
    const uint32_t nTapsPerLink = nBsAntN * nUeAntN * NMAXTAPS;

    // Write per-cell complex datasets matching Python API:
    //   cirPerCell/cirCoe_cell{cell_id}  -> complex dtype shape (nActiveUt, nSnapshots, nUtAnt, nBsAnt, nTaps)
    //   cirPerCell/cirNtaps_cell{cell_id} -> uint32 shape (nActiveUt,)
    for (auto& [cellId, linkIdxs] : cellToLinks)
    {
        const uint32_t nActiveUtForCell = static_cast<uint32_t>(linkIdxs.size());

        // Create complex datatype (real + imag)
        hid_t complexType = H5Tcreate(H5T_COMPOUND, sizeof(float) * 2);
        H5Tinsert(complexType, "real", 0 * sizeof(float), H5T_NATIVE_FLOAT);
        H5Tinsert(complexType, "imag", 1 * sizeof(float), H5T_NATIVE_FLOAT);

        // Dataspace shape: [nActiveUt, nSnapshots, nBsAnt, nUeAnt, NMAXTAPS]
        hsize_t cirDims[5];
        cirDims[0] = nActiveUtForCell;
        cirDims[1] = nSnapshots;
        cirDims[2] = nBsAntN;
        cirDims[3] = nUeAntN;
        cirDims[4] = NMAXTAPS;
        hid_t cirDataspace = H5Screate_simple(5, cirDims, nullptr);

        // Allocate host buffer: [nActiveUt, nSnapshots, nBsAnt, nUeAnt, NMAXTAPS] as complex
        std::vector<std::complex<float>> cellCir(
            static_cast<size_t>(nActiveUtForCell) * nSnapshots * nBsAntN * nUeAntN * NMAXTAPS);

        // Scatter from flat cirCoe buffer into per-cell buffer
        // Flat layout: [link, snap, bs_ant, ue_ant, tap] -> [link * nSnapshots * nBsAnt * nUeAnt * NMAXTAPS
        //                                                     + snap * nBsAnt * nUeAnt * NMAXTAPS
        //                                                     + bs_ant * nUeAnt * NMAXTAPS
        //                                                     + ue_ant * NMAXTAPS
        //                                                     + tap]
        // Per-cell layout: [ueInCell, snap, bs_ant, ue_ant, tap]
        for (uint32_t ueInCell = 0; ueInCell < nActiveUtForCell; ++ueInCell)
        {
            const uint32_t globalLinkIdx = linkIdxs[ueInCell];
            const uint32_t cirOffset = activeLinks[globalLinkIdx].cirCoeOffset;
            for (uint32_t snap = 0; snap < nSnapshots; ++snap)
            {
                for (uint32_t bs_ant = 0; bs_ant < nBsAntN; ++bs_ant)
                {
                    for (uint32_t ue_ant = 0; ue_ant < nUeAntN; ++ue_ant)
                    {
                        for (uint32_t tap = 0; tap < NMAXTAPS; ++tap)
                        {
                            const size_t flatIdx = cirOffset
                                + static_cast<size_t>(snap) * nBsAntN * nUeAntN * NMAXTAPS
                                + static_cast<size_t>(bs_ant) * nUeAntN * NMAXTAPS
                                + static_cast<size_t>(ue_ant) * NMAXTAPS
                                + tap;
                            cellCir[(ueInCell * nSnapshots + snap) * nTapsPerLink
                                    + bs_ant * nUeAntN * NMAXTAPS
                                    + ue_ant * NMAXTAPS
                                    + tap] = cirCoe[flatIdx];
                        }
                    }
                }
            }
        }

        // Create dataset name
        std::string cirDatasetName = "cirCoe_cell" + std::to_string(cellId);

        // Write CIR data
        hid_t cirDataset = H5Dcreate2(grpCir, cirDatasetName.c_str(), complexType,
                                       cirDataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(cirDataset, complexType, H5S_ALL, H5S_ALL, H5P_DEFAULT, cellCir.data());
        H5Dclose(cirDataset);
        H5Sclose(cirDataspace);
        H5Tclose(complexType);

        // Write ntaps per active UE for this cell
        {
            std::vector<uint32_t> cellNtaps(nActiveUtForCell);
            for (uint32_t ueInCell = 0; ueInCell < nActiveUtForCell; ++ueInCell)
            {
                const uint32_t globalLinkIdx = linkIdxs[ueInCell];
                cellNtaps[ueInCell] = cirNtaps[activeLinks[globalLinkIdx].cirNtapsOffset];
            }

            std::string ntapsDatasetName = "cirNtaps_cell" + std::to_string(cellId);
            hsize_t ntapsDim = nActiveUtForCell;
            hid_t ntapsDataspace = H5Screate_simple(1, &ntapsDim, nullptr);
            hid_t ntapsDataset = H5Dcreate2(grpCir, ntapsDatasetName.c_str(),
                                            H5T_NATIVE_UINT32, ntapsDataspace,
                                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(ntapsDataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, cellNtaps.data());
            H5Dclose(ntapsDataset);
            H5Sclose(ntapsDataspace);
        }

        // Write UE-to-row mapping for this cell
        {
            std::vector<uint16_t> ueMapping(nActiveUtForCell);
            for (uint32_t ueInCell = 0; ueInCell < nActiveUtForCell; ++ueInCell)
            {
                const uint32_t globalLinkIdx = linkIdxs[ueInCell];
                ueMapping[ueInCell] = static_cast<uint16_t>(activeLinks[globalLinkIdx].uid);
            }

            std::string mappingName = "ue_mapping_cell" + std::to_string(cellId);
            hsize_t mappingDim = nActiveUtForCell;
            hid_t mappingDataspace = H5Screate_simple(1, &mappingDim, nullptr);
            hid_t mappingDataset = H5Dcreate2(grpCir, mappingName.c_str(),
                                              H5T_NATIVE_UINT16, mappingDataspace,
                                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(mappingDataset, H5T_NATIVE_UINT16, H5S_ALL, H5S_ALL, H5P_DEFAULT, ueMapping.data());
            H5Dclose(mappingDataset);
            H5Sclose(mappingDataspace);
        }
    }
    H5Gclose(grpCir);

    // ── cfrPrbgPerCell ──
    createGroup(file, "cfrPrbgPerCell");
    hid_t grpCfr = H5Gopen2(file, "cfrPrbgPerCell", H5P_DEFAULT);

    std::vector<float> cfrCoeFloat(2 * nCFRComplexElems);
    for (size_t i = 0; i < nCFRComplexElems; ++i)
    {
        cfrCoeFloat[2 * i] = cfrPrbg[i].real();
        cfrCoeFloat[2 * i + 1] = cfrPrbg[i].imag();
    }
    writeDatasetFloat(grpCfr, "CFR_Coeff", cfrCoeFloat.data(), 2 * nCFRComplexElems);
    H5Gclose(grpCfr);

    // ── cfrScPerCell (empty dataset, matches NVIDIA structure) ──
    createGroup(file, "cfrScPerCell");
    hid_t grpCfrSc = H5Gopen2(file, "cfrScPerCell", H5P_DEFAULT);
    H5Gclose(grpCfrSc);

    // ── configurationMetadata ──
    createGroup(file, "configurationMetadata");
    hid_t grpCfg = H5Gopen2(file, "configurationMetadata", H5P_DEFAULT);

    createGroup(grpCfg, "simulation_config");
    hid_t grpSimCfg = H5Gopen2(grpCfg, "simulation_config", H5P_DEFAULT);
    writeDatasetFloat(grpSimCfg, "scSpacingHz", &scSpacingHz, 1);
    writeDatasetUint32(grpSimCfg, "fftSize", &fftSize, 1);
    writeDatasetUint32(grpSimCfg, "nPrb", &nPrb, 1);
    writeDatasetUint32(grpSimCfg, "nSnapshotPerSlot", &nSnapshotPerSlot, 1);
    writeDatasetFloat(grpSimCfg, "centerFreqHz", &centerFreqHz, 1);
    writeDatasetFloat(grpSimCfg, "bandwidthHz", &bandwidthHz, 1);
    writeDatasetUint32(grpSimCfg, "nUeAnt", &nUeAnt, 1);
    writeDatasetUint32(grpSimCfg, "nBsAnt", &nBsAnt, 1);
    H5Gclose(grpSimCfg);

    createGroup(grpCfg, "system_level_config");
    hid_t grpSysCfg = H5Gopen2(grpCfg, "system_level_config", H5P_DEFAULT);
    writeDatasetFloat(grpSysCfg, "isd", &isd, 1);
    writeDatasetFloat(grpSysCfg, "bsHeight", &bsHeight, 1);
    writeDatasetFloat(grpSysCfg, "minBsUeDist2d", &minBsUeDist2d, 1);
    writeDatasetFloat(grpSysCfg, "maxBsUeDist2dIndoor", &maxBsUeDist2dIndoor, 1);
    writeDatasetFloat(grpSysCfg, "indoorUtPercent", &indoorUtPercent, 1);
    writeDatasetUint32(grpSysCfg, "nSectorPerSite", &nSectorPerSite, 1);
    writeDatasetUint32(grpSysCfg, "nSite", &nSite, 1);
    writeDatasetUint32(grpSysCfg, "nUT", &nUT, 1);
    writeDatasetUint32(grpSysCfg, "nActiveLinks", &nLinks, 1);
    H5Gclose(grpSysCfg);

    createGroup(grpCfg, "ss_common_params");
    hid_t grpSsCfg = H5Gopen2(grpCfg, "ss_common_params", H5P_DEFAULT);
    writeDatasetFloat(grpSsCfg, "lgfc", &ssCmn.lgfc, 1);
    writeDatasetFloat(grpSsCfg, "lambda_0", &ssCmn.lambda_0, 1);
    writeDatasetFloat(grpSsCfg, "r_tao_0", &ssCmn.r_tao[0], 1);
    writeDatasetFloat(grpSsCfg, "r_tao_1", &ssCmn.r_tao[1], 1);
    writeDatasetFloat(grpSsCfg, "r_tao_2", &ssCmn.r_tao[2], 1);
    writeDatasetFloat(grpSsCfg, "mu_XPR_0", &ssCmn.mu_XPR[0], 1);
    writeDatasetFloat(grpSsCfg, "mu_XPR_1", &ssCmn.mu_XPR[1], 1);
    writeDatasetFloat(grpSsCfg, "mu_XPR_2", &ssCmn.mu_XPR[2], 1);
    writeDatasetFloat(grpSsCfg, "sigma_XPR_0", &ssCmn.sigma_XPR[0], 1);
    writeDatasetFloat(grpSsCfg, "sigma_XPR_1", &ssCmn.sigma_XPR[1], 1);
    writeDatasetFloat(grpSsCfg, "sigma_XPR_2", &ssCmn.sigma_XPR[2], 1);
    writeDatasetUint32(grpSsCfg, "nCluster_0", &ssCmn.nCluster[0], 1);
    writeDatasetUint32(grpSsCfg, "nCluster_1", &ssCmn.nCluster[1], 1);
    writeDatasetUint32(grpSsCfg, "nCluster_2", &ssCmn.nCluster[2], 1);
    writeDatasetUint32(grpSsCfg, "nSubCluster", &ssCmn.nSubCluster, 1);
    writeDatasetUint32(grpSsCfg, "nUeAnt", &ssCmn.nUeAnt, 1);
    writeDatasetUint32(grpSsCfg, "nBsAnt", &ssCmn.nBsAnt, 1);
    H5Gclose(grpSsCfg);
    H5Gclose(grpCfg);

    // ── topology ──
    createGroup(file, "topology");
    hid_t grpTopo = H5Gopen2(file, "topology", H5P_DEFAULT);

    // ── topology/cellParams (compound dataset for Python compatibility) ──
    // Matches the layout expected by analysis_channel_stats.py _parse_cell_params_h5
    {
        // Struct for the compound record
        struct CellParamsRecord
        {
            uint32_t cid;
            uint32_t siteId;
            float loc[3];       // [x, y, z]
            uint32_t antPanelIdx;
            float antPanelOrientation[3]; // [theta_tilt, phi_tilt, zeta_offset]
        };

        // Build compound type
        hsize_t dim3 = 3;
        hid_t locType = H5Tarray_create2(H5T_NATIVE_FLOAT, 1, &dim3);
        hid_t antOrType = H5Tarray_create2(H5T_NATIVE_FLOAT, 1, &dim3);

        hid_t compoundType = H5Tcreate(H5T_COMPOUND, sizeof(CellParamsRecord));
        H5Tinsert(compoundType, "cid",                     0,             H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "siteId",                  4,             H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "loc",                     8,             locType);
        H5Tinsert(compoundType, "antPanelIdx",             20,            H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "antPanelOrientation",     24,            antOrType);

        std::vector<CellParamsRecord> records(nCells);
        for (uint32_t i = 0; i < nCells; ++i)
        {
            records[i].cid                      = i;
            records[i].siteId                   = i / nSectorPerSite;
            records[i].loc[0]                   = cells[i].loc.x;
            records[i].loc[1]                   = cells[i].loc.y;
            records[i].loc[2]                   = cells[i].loc.z;
            records[i].antPanelIdx              = cellsSS[i].antPanelIdx;
            records[i].antPanelOrientation[0]   = cellsSS[i].antPanelOrientation[0];
            records[i].antPanelOrientation[1]   = cellsSS[i].antPanelOrientation[1];
            records[i].antPanelOrientation[2]   = cellsSS[i].antPanelOrientation[2];
        }

        hsize_t dims = static_cast<hsize_t>(nCells);
        hid_t dataspace = H5Screate_simple(1, &dims, &dims);
        hid_t dataset = H5Dcreate2(file, "cellParams", compoundType, dataspace,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t memspace = H5Screate_simple(1, &dims, nullptr);
        H5Dwrite(dataset, compoundType, memspace, dataspace, H5P_DEFAULT, records.data());
        H5Sclose(memspace);
        H5Sclose(dataspace);
        H5Tclose(compoundType);
        H5Tclose(locType);
        H5Tclose(antOrType);
        H5Dclose(dataset);
    }

    const uint32_t cellFloatsPer = 4; // loc[x,y,z,_p] only
    std::vector<float> cellParamsFlat(nCells * cellFloatsPer, 0.0f);
    for (uint32_t i = 0; i < nCells; ++i)
    {
        uint32_t base = i * cellFloatsPer;
        cellParamsFlat[base + 0] = cells[i].loc.x;
        cellParamsFlat[base + 1] = cells[i].loc.y;
        cellParamsFlat[base + 2] = cells[i].loc.z;
        cellParamsFlat[base + 3] = cells[i].loc._p;
    }
    writeDatasetFloat(grpTopo, "cell_params", cellParamsFlat.data(), nCells * cellFloatsPer);

    std::vector<float> siteParamsFlat(nSite * 7, 0.0f);
    for (uint32_t i = 0; i < nSite; ++i)
    {
        uint32_t base = i * 7;
        siteParamsFlat[base + 0] = cells[i * nSectorPerSite].loc.x;
        siteParamsFlat[base + 1] = cells[i * nSectorPerSite].loc.y;
        siteParamsFlat[base + 2] = cells[i * nSectorPerSite].loc.z;
        siteParamsFlat[base + 3] = cells[i * nSectorPerSite].loc._p;
        siteParamsFlat[base + 4] = isd;
        siteParamsFlat[base + 5] = 0.0f;
        siteParamsFlat[base + 6] = 0.0f;
    }
    writeDatasetFloat(grpTopo, "site_params", siteParamsFlat.data(), nSite * 7);

    std::vector<float> utParamsFlat(nUT * 7, 0.0f);
    for (uint32_t i = 0; i < nUT; ++i)
    {
        uint32_t base = i * 7;
        utParamsFlat[base + 0] = uts[i].loc.x;
        utParamsFlat[base + 1] = uts[i].loc.y;
        utParamsFlat[base + 2] = uts[i].loc.z;
        utParamsFlat[base + 3] = uts[i].loc._p;
        utParamsFlat[base + 4] = uts[i].d_2d_in;
        utParamsFlat[base + 5] = static_cast<float>(uts[i].outdoor_ind);
        utParamsFlat[base + 6] = uts[i].o2i_penetration_loss;
    }
    writeDatasetFloat(grpTopo, "ut_params", utParamsFlat.data(), nUT * 7);

    // Topology scalars (matches NVIDIA slsChan::saveSlsChanToH5File)
    writeDatasetUint32(grpTopo, "nSite", &nSite, 1);
    writeDatasetUint32(grpTopo, "nSector", &nSectorPerSite, 1);
    writeDatasetUint32(grpTopo, "nUT", &nUT, 1);
    writeDatasetFloat(grpTopo, "isd", &isd, 1);
    writeDatasetFloat(grpTopo, "bsHeight", &bsHeight, 1);
    writeDatasetFloat(grpTopo, "minBsUeDist2d", &minBsUeDist2d, 1);
    writeDatasetFloat(grpTopo, "maxBsUeDist2dIndoor", &maxBsUeDist2dIndoor, 1);
    writeDatasetFloat(grpTopo, "indoorUtPercent", &indoorUtPercent, 1);
    H5Gclose(grpTopo);

    H5Fclose(file);
    std::cout << "Wrote HDF5: " << filename
              << " (" << nLinks << " links, " << nLinks << " active)\n";
}
#endif // SLS_CHAN_HDF5
