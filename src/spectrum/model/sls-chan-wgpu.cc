// This is an AI translated version of the original Nvidia source for GPU compute,
// but ported to WGPU for cross-platform support
// https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/main/testBenches/chanModels/src/sls_chan_src/sls_chan_large_scale_GPU.cu

#define WEBGPU_CPP_IMPLEMENTATION
#include "sls-chan-wgpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <vector>

#ifdef SLS_CHAN_HDF5
#include <H5Cpp.h>
#endif

static std::string
readFile(const char* path)
{
    std::ifstream f(path);
    return {std::istreambuf_iterator<char>(f), {}};
}

// Map a backend enum value to its short textual name (also used as the
// `SLS_BACKEND` env-var keyword).
static const char*
backendName(WGPUBackendType b)
{
    switch (b)
    {
    case WGPUBackendType_D3D12: return "D3D12";
    case WGPUBackendType_D3D11: return "D3D11";
    case WGPUBackendType_Vulkan: return "Vulkan";
    case WGPUBackendType_Metal: return "Metal";
    case WGPUBackendType_OpenGL: return "OpenGL";
    case WGPUBackendType_OpenGLES: return "OpenGLES";
    case WGPUBackendType_Undefined: return "Auto";
    case WGPUBackendType_WebGPU: return "WebGPU";
    case WGPUBackendType_Null: return "Null";
    default: return "Unknown";
    }
}

// Request an adapter for the given backend; returns nullptr on failure.
static WGPUAdapter
tryAdapter(WGPUInstance instance, WGPUBackendType backend)
{
    WGPUAdapter adapter = nullptr;
    WGPURequestAdapterOptions aopts{};
    aopts.backendType = backend;
    SLS_LOG("[DEBUG] Trying %s backend...\n", backendName(backend));
    wgpuInstanceRequestAdapter(
        instance,
        &aopts,
        WGPURequestAdapterCallbackInfo{
            .mode = WGPUCallbackMode_AllowSpontaneous,
            .callback =
                [](WGPURequestAdapterStatus status,
                   WGPUAdapter a,
                   WGPUStringView,
                   void* ud1,
                   void*) {
                    if (status == WGPURequestAdapterStatus_Success)
                    {
                        *static_cast<WGPUAdapter*>(ud1) = a;
                    }
                },
            .userdata1 = &adapter});
    wgpuInstanceProcessEvents(instance);
    if (adapter)
    {
        SLS_LOG("[DEBUG] %s adapter acquired\n", backendName(backend));
    }
    else
    {
        SLS_LOG("[DEBUG] %s adapter unavailable\n", backendName(backend));
    }
    return adapter;
}

// Resolve the desired backend order:
//   - If SLS_BACKEND is set to D3D12 / D3D11 / Vulkan / Metal / OpenGL /
//     OpenGLES / Auto (case-insensitive), try ONLY that backend, hard-fail
//     if it isn't available.
//   - Otherwise fall through to the default Windows-friendly chain:
//     D3D12 -> Vulkan -> Auto.
// Result is a small vector of backends to try in order; the first one
// that returns a usable adapter wins.
static std::vector<WGPUBackendType>
resolveBackendOrder()
{
    const char* env = std::getenv("SLS_BACKEND");
    if (env && *env)
    {
        std::string sel(env);
        for (auto& c : sel)
        {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        if (sel == "d3d12")
        {
            return {WGPUBackendType_D3D12};
        }
        if (sel == "d3d11")
        {
            return {WGPUBackendType_D3D11};
        }
        if (sel == "vulkan" || sel == "vk")
        {
            return {WGPUBackendType_Vulkan};
        }
        if (sel == "metal" || sel == "mtl")
        {
            return {WGPUBackendType_Metal};
        }
        if (sel == "opengl" || sel == "gl")
        {
            return {WGPUBackendType_OpenGL};
        }
        if (sel == "opengles" || sel == "gles")
        {
            return {WGPUBackendType_OpenGLES};
        }
        if (sel == "auto" || sel == "undefined")
        {
            return {WGPUBackendType_Undefined};
        }
        SLS_LOG("[WARN] SLS_BACKEND=%s not recognised; falling through to default chain\n",
                env);
    }
    // Default chain — best on Windows, falls back for headless / non-D3D12 systems.
    return {WGPUBackendType_D3D12, WGPUBackendType_Vulkan, WGPUBackendType_Undefined};
}

struct DeviceBundle
{
    wgpu::Instance instance;
    wgpu::Device device;
    uint64_t maxStorageBufferBindingSize;
};

static DeviceBundle
createDevice()
{
    WGPUInstanceDescriptor idesc{};
#ifdef WEBGPU_BACKEND_DAWN
    // Dawn rejects `instance.waitAny(..., timeoutNS != 0)` unless the
    // instance was created with the TimedWaitAny capability. wgpu-native
    // doesn't expose this struct, so it stays inside the Dawn ifdef.
    WGPUInstanceCapabilities caps{};
    caps.timedWaitAnyEnable = true;
    idesc.capabilities = caps;
#endif
    WGPUInstance instance = wgpuCreateInstance(&idesc);
    assert(instance);

    WGPUAdapter adapter = nullptr;
    WGPUBackendType chosen = WGPUBackendType_Undefined;
    for (WGPUBackendType b : resolveBackendOrder())
    {
        adapter = tryAdapter(instance, b);
        if (adapter)
        {
            chosen = b;
            break;
        }
    }
    if (!adapter)
    {
        SLS_LOG("[FATAL] No GPU adapter available for any requested backend!\n");
        wgpuInstanceRelease(instance);
        exit(1);
    }
    // Log the winning backend even when SLS_DEBUG is off — useful for
    // quick "which backend am I on?" diagnostics during testing.
    std::fprintf(stderr, "[SlsChanWgpu] using %s backend\n", backendName(chosen));

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
            SLS_LOG("[wgpu error %d] %.*s\n", (int)t, (int)msg.length, msg.data);
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
    // We use ~20 storage buffers per stage in cal_link_param / cluster_ray.
    // Ask for what the adapter actually supports (Dawn rejects
    // requests above adapter limits; wgpu-native is lenient).
    // The shader uses ~13 storage buffers per stage in
    // cal_link_param_kernel. Different runtimes handle this
    // differently: wgpu-native v24 is lenient (over-request OK), Dawn
    // enforces the WebGPU spec default of 10 even if the underlying
    // hardware supports more. For wgpu-native we keep the historical
    // 30; Dawn callers should refactor the shader to fewer SSBOs per
    // stage or enable the relevant adapter feature.
#ifdef WEBGPU_BACKEND_DAWN
    requiredLimits.maxStorageBuffersPerShaderStage =
        supported.maxStorageBuffersPerShaderStage;
#else
    requiredLimits.maxStorageBuffersPerShaderStage = 30;
#endif
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
    SLS_LOG("[DEBUG] Device created with maxStorageBufSize=%llu\n",
            (unsigned long long)requiredLimits.maxStorageBufferBindingSize);
    wgpuAdapterRequestDevice(
        adapter,
        &ddesc,
        WGPURequestDeviceCallbackInfo{
            .mode = WGPUCallbackMode_AllowSpontaneous,
            .callback =
                [](WGPURequestDeviceStatus status, WGPUDevice d, WGPUStringView, void* ud1, void*) {
                    if (status == WGPURequestDeviceStatus_Success)
                    {
                        *static_cast<WGPUDevice*>(ud1) = d;
                    }
                    else
                    {
                        SLS_LOG("requestDevice failed\n");
                    }
                },
            .userdata1 = &device});
    wgpuInstanceProcessEvents(instance); // process async callback
    if (!device)
    {
        SLS_LOG("[FATAL] Failed to create WGPU device!\n");
        wgpuAdapterRelease(adapter);
        wgpuInstanceRelease(instance);
        exit(1);
    }
    wgpu::Device dev(device);
    wgpu::Instance inst(instance);
    wgpuAdapterRelease(adapter);
    // Don't release the instance here — its lifetime is tied to the
    // device callbacks. The wrapper takes ownership; it'll release on
    // its destructor.
    return DeviceBundle{inst, dev, supported.maxStorageBufferBindingSize};
}

// ── Constructor ───────────────────────────────────────────────────────────────
SlsChanWgpu::SlsChanWgpu()
{
    auto result = createDevice();
    instance_ = std::move(result.instance);
    device_ = std::move(result.device);
    m_maxGpuBuffer_ = result.maxStorageBufferBindingSize;
    SLS_LOG("[DEBUG] Constructor: GPU max buffer size = %llu bytes (%.1f GB)\n",
            (unsigned long long)m_maxGpuBuffer_,
            (double)m_maxGpuBuffer_ / (1024.0 * 1024.0 * 1024.0));
    queue_ = device_.getQueue();
    std::string wgsl = readFile("C:/tools/sources/ns-3-dev/src/spectrum/model/sls-chan.wgsl");
    if (wgsl.empty())
    {
        SLS_LOG("ERROR: Failed to read WGSL shader\n");
        throw std::runtime_error("Failed to read WGSL shader");
    }
    SLS_COUT << "Loaded WGSL shader: " << wgsl.size() << " bytes\n";

    WGPUShaderSourceWGSL wgslSource{};
    wgslSource.chain.next = nullptr;
    wgslSource.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSource.code = sv(wgsl.c_str());

    WGPUShaderModuleDescriptor smDescC{};
    smDescC.nextInChain = &wgslSource.chain;
    SLS_LOG("[DEBUG] Creating WGSL shader module (%zu bytes)...\n", wgsl.size());
    shader_ = wgpu::ShaderModule(wgpuDeviceCreateShaderModule(device_, &smDescC));
    if (!shader_)
    {
        SLS_LOG("[FATAL] WGSL shader module creation failed\n");
        throw std::runtime_error("WGSL failed to compile");
    }
    SLS_LOG("[DEBUG] WGSL shader module created successfully\n");

    // Large-scale pipelines compile eagerly — every caller (LSP-only
    // ns-3 batch + full validation harness) needs them.
    linkParamPipeline_ = makePipeline("cal_link_param_kernel");
    assert(linkParamPipeline_ && "missing cal_link_param_kernel in WGSL");
    crnFillPipeline_ = makePipeline("fill_crn_kernel");
    assert(crnFillPipeline_ && "missing fill_crn_kernel in WGSL");
    crnConvPipeline_ = makePipeline("convolve_crn_kernel");
    assert(crnConvPipeline_ && "missing convolve_crn_kernel in WGSL");
    crnNormPipeline_ = makePipeline("normalize_crn_kernel");
    assert(crnNormPipeline_ && "missing normalize_crn_kernel in WGSL");
    // Small-scale pipelines (cluster-ray / CIR / CFR) compile lazily
    // in their owning methods. Some wgpu-native backends (notably the
    // bundled v24 Vulkan / OpenGL on certain drivers) crash inside
    // createComputePipeline when handed cal_cluster_ray_kernel; the
    // LSP-only ns-3 path never reaches that method, so deferring lets
    // those backends still produce LinkParams.
}

wgpu::ComputePipeline
SlsChanWgpu::makePipeline(const char* entryPoint)
{
    wgpu::ComputePipelineDescriptor desc{};
    desc.compute.module = shader_;
    desc.compute.entryPoint = sv(entryPoint);
    SLS_LOG("[DEBUG] Creating pipeline '%s'...\n", entryPoint);
    std::fflush(stderr);
    auto pipe = device_.createComputePipeline(desc);
    SLS_LOG("[DEBUG] Pipeline '%s': %s\n", entryPoint, pipe ? "OK" : "FAILED");
    std::fflush(stderr);
    return pipe;
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

// Shim: process queued GPU work. wgpu-native has `device.poll(wait,
// submissionIdx)`; Dawn has `device.tick()` which is always
// non-blocking. For Dawn we approximate "wait until idle" by calling
// `queue.onSubmittedWorkDone` and ticking until the future resolves --
// not strictly identical to wgpu-native's blocking poll, but adequate
// for our pattern of "submit, then read back" since we sleep between
// ticks.
#ifdef WEBGPU_BACKEND_DAWN
// Drive Dawn's event loop. `Instance::processEvents()` is what fires
// callbacks registered with `AllowProcessEvents` mode (mapAsync,
// onSubmittedWorkDone, ...). `Device::tick()` only drives
// device-internal state and does NOT fire those user callbacks.
inline void
SlsChanWgpu::dawnPumpEvents()
{
    instance_.processEvents();
    device_.tick();
}
#endif

void
SlsChanWgpu::waitIdle()
{
#ifdef WEBGPU_BACKEND_DAWN
    // Wait for any submitted work to finish. We use waitAny on a
    // queue.onSubmittedWorkDone future with WaitAnyOnly mode — this is
    // the synchronous equivalent of wgpu-native's poll(wait=true).
    auto fut = queue_.onSubmittedWorkDone(
        wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::QueueWorkDoneStatus) {});
    wgpu::FutureWaitInfo waitInfo{};
    waitInfo.future = fut;
    waitInfo.completed = false;
    instance_.waitAny(1, &waitInfo, UINT64_MAX);
    // Also drive any AllowProcessEvents-mode callbacks (mapAsync,
    // ...) that may be pending. wgpu-native's poll(wait=true) does
    // both implicitly; Dawn splits them across waitAny + processEvents.
    instance_.processEvents();
#else
    device_.poll(true, nullptr);
#endif
}

// ── Large-scale topology uploads ──────────────────────────────────────────────
void
SlsChanWgpu::uploadCellParams(const std::vector<CellParam>& cells, uint32_t nSectorPerSite)
{
    cellParamsBuf_ =
        makeBuffer(cells.size() * sizeof(CellParam), WGPUBufferUsage_Storage, cells.data());
    // nSite_ is the number of *sites*. With a sectorised deployment the
    // CellParam vector holds nSite * nSectorPerSite entries — the
    // Phase-1 calibration harness uses 3-sectors-per-site. Callers that
    // pass a single cell per site (e.g. the ThreeGppChannelModel batch
    // back-end) should pass `nSectorPerSite=1`.
    assert(nSectorPerSite > 0 && "nSectorPerSite must be > 0");
    nSite_ = static_cast<uint32_t>(cells.size() / nSectorPerSite);
    cellsCache_ = cells;
    nSectorPerSiteCache_ = nSectorPerSite;
}

void
SlsChanWgpu::uploadUtParams(const std::vector<UtParam>& uts)
{
    utParamsBuf_ = makeBuffer(uts.size() * sizeof(UtParam), WGPUBufferUsage_Storage, uts.data());
    utsCache_ = uts;
    nUTCache_ = static_cast<uint32_t>(uts.size());
}

void
SlsChanWgpu::setSystemLevelConfig(uint32_t scenario,
                                  uint32_t enablePropagationDelay,
                                  uint32_t o2iBldg,
                                  uint32_t o2iCar,
                                  float forceLosIndoor,
                                  float forceLosOutdoor)
{
    SystemLevelConfigGPU slc{scenario,
                             enablePropagationDelay,
                             o2iBldg,
                             o2iCar,
                             forceLosIndoor,
                             forceLosOutdoor,
                             {0.0f, 0.0f}};
    // `calLinkParam` lazily allocates this buffer if it doesn't already
    // exist; for an explicit caller override we (re)create it with the
    // provided values so subsequent calLinkParam runs pick them up.
    sysConfigBuf_ = makeBuffer(sizeof(SystemLevelConfigGPU),
                               WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
                               &slc);
}

void
SlsChanWgpu::setCenterFrequencyHz(float centerFreqHz)
{
    centerFreqHzCache_ = centerFreqHz;
    SimConfigGPU sc{centerFreqHz, 0.0f, 0.0f, 0.0f};
    simConfigBuf_ = makeBuffer(sizeof(SimConfigGPU),
                               WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
                               &sc);
}

// ── Small-scale topology uploads ─────────────────────────────────────────────
void
SlsChanWgpu::uploadCellParamsSS(const std::vector<CellParamSS>& cells)
{
    ssCellParamsBuf_ =
        makeBuffer(cells.size() * sizeof(CellParamSS), WGPUBufferUsage_Storage, cells.data());
    cellsSSCache_ = cells;
}

void
SlsChanWgpu::uploadUtParamsSS(const std::vector<UtParamSS>& uts)
{
    ssUtParamsBuf_ =
        makeBuffer(uts.size() * sizeof(UtParamSS), WGPUBufferUsage_Storage, uts.data());
    utsSSCache_ = uts;
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
    antCfgsCache_ = configs;
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

#ifdef WEBGPU_BACKEND_DAWN
    // Dawn: use WaitAnyOnly + waitAny on the returned future for a true
    // blocking wait. AllowProcessEvents + spin-on-tick races with the
    // submit's onSubmittedWorkDone callback (different future);
    // WaitAnyOnly is the synchronous primitive Dawn provides for this
    // exact pattern.
    auto fut = staging.mapAsync(
        wgpu::MapMode::Read,
        0,
        byteSize,
        wgpu::CallbackMode::WaitAnyOnly,
        [&ctx](wgpu::MapAsyncStatus, wgpu::StringView) { ctx.done = true; });
    wgpu::FutureWaitInfo waitInfo{};
    waitInfo.future = fut;
    waitInfo.completed = false;
    instance_.waitAny(1, &waitInfo, UINT64_MAX);
#else
    // wgpu-native v24 takes (mode, offset, size, BufferMapCallbackInfo).
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
#endif

    // Dawn validates that buffers mapped with MapMode::Read are read
    // back via `getConstMappedRange` (writable getMappedRange returns
    // null for read maps). wgpu-native accepts either, but the const
    // variant is the WebGPU-spec way for read-only buffers.
    const void* mapped = staging.getConstMappedRange(0, byteSize);
    if (mapped)
    {
        std::memcpy(result.data(), mapped, byteSize);
    }
    else
    {
        SLS_LOG("[ERROR] mapReadBuffer: getConstMappedRange returned null\n");
    }
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
    SLS_LOG("[DEBUG] generateCRN: nSite=%u, maxX=%.1f, minX=%.1f, maxY=%.1f, minY=%.1f\n",
            nSite_,
            maxX,
            minX,
            maxY,
            minY);
    fflush(stderr);
    // Calculate grid dimensions matching WGSL shader: round(bound + 1 + 2*D) where D=3*corrDist
    float maxCorrDist = 0.0f;
    for (int i = 0; i < 8; i++)
    {
        maxCorrDist = std::max(maxCorrDist, corrLos[i]);
    }
    for (int i = 0; i < 7; i++)
    {
        maxCorrDist = std::max(maxCorrDist, corrNlos[i]);
    }
    for (int i = 0; i < 7; i++)
    {
        maxCorrDist = std::max(maxCorrDist, corrO2i[i]);
    }

    SLS_LOG("[DEBUG] generateCRN: maxCorrDist=%.1f step=%.1f\n", maxCorrDist, crnStep_);
    // All grid sizing is in PIXELS. corrDist (metres) maps to corrDist/step
    // pixels of correlation; D = 3*corr_px is the per-side filter pad; the
    // total grid is round(bound/step + 1 + 2*D) pixels per axis.
    const float step_m = std::max(crnStep_, 1.0f);
    const float D_px = 3.0f * (maxCorrDist / step_m);
    const int32_t nX = static_cast<int32_t>((maxX - minX) / step_m + 1.0f + 2.0f * D_px + 0.5f);
    const int32_t nY = static_cast<int32_t>((maxY - minY) / step_m + 1.0f + 2.0f * D_px + 0.5f);
    SLS_LOG("[DEBUG] generateCRN: nX=%d, nY=%d, gridSz=%llu\n",
            nX,
            nY,
            (unsigned long long)uint64_t(nX) * nY);

    nX_ = nX;
    nY_ = nY;

    const uint64_t gridSz = uint64_t(nX) * nY;
    const uint64_t losBufSz = uint64_t(nSite_) * 8 * gridSz * sizeof(float);
    const uint64_t nlosBufSz = uint64_t(nSite_) * 7 * gridSz * sizeof(float);
    const uint64_t o2iBufSz = uint64_t(nSite_) * 7 * gridSz * sizeof(float);
    SLS_LOG("[DEBUG] generateCRN: losBufSz=%llu, nlosBufSz=%llu, o2iBufSz=%llu\n",
            (unsigned long long)losBufSz,
            (unsigned long long)nlosBufSz,
            (unsigned long long)o2iBufSz);

    // Hard safety guard. On D3D12 we have observed that requesting a CRN buffer
    // anywhere near the device's maxStorageBufferBindingSize triggers a BSOD
    // (VIDEO_SCHEDULER_INTERNAL_ERROR / 0x119) — likely a dxgkrnl TDR cascading
    // on the multi-GB allocation + chunked dispatch. Refuse to allocate any CRN
    // buffer larger than 1 GB. The buffer scales linearly with deployment area
    // squared and site count; if you blow this budget, shrink the deployment
    // (and/or coarsen the CRN grid step via the third arg to generateCRN).
    const uint64_t maxCrn = std::max({losBufSz, nlosBufSz, o2iBufSz});
    constexpr uint64_t CRN_BUFFER_SAFETY_CAP = 1ULL << 30; // 1 GB
    uint64_t effectiveCap = CRN_BUFFER_SAFETY_CAP;
    if (m_maxGpuBuffer_ != 0)
    {
        effectiveCap = std::min(effectiveCap, m_maxGpuBuffer_);
    }
    if (maxCrn > effectiveCap)
    {
        SLS_LOG("[FATAL] CRN buffer size %.2f GB exceeds safety cap %.2f GB "
                "(device limit %.2f GB).\n"
                "        nSite=%u, gridSz=%llu (nX=%d nY=%d).\n"
                "        Reduce deployment radius / site count or coarsen the "
                "CRN grid step.\n",
                (double)maxCrn / (1024.0 * 1024.0 * 1024.0),
                (double)effectiveCap / (1024.0 * 1024.0 * 1024.0),
                (double)m_maxGpuBuffer_ / (1024.0 * 1024.0 * 1024.0),
                nSite_,
                (unsigned long long)gridSz,
                nX,
                nY);
        std::fflush(stderr);
        std::abort();
    }

    auto tempBufBytes = [&, step_m](float cd) -> uint64_t {
        // Must match what fill_crn_kernel writes: padded_nx × padded_ny
        // floats, where padded = final + (L-1) and L = 2*floor(3*cd/step)+1.
        // The convolve kernel ALSO reads at padded stride. Sizing this for
        // the *final* grid (which is what the previous version did) left
        // the filter-padding tail spilling into adjacent allocations.
        const float Dpx = 3.0f * (cd / step_m);
        const uint64_t iDpx = static_cast<uint64_t>(Dpx);
        const uint64_t L = (cd > 0.0f) ? (2 * iDpx + 1) : 1ULL;
        const uint64_t pad = (L > 0) ? (L - 1) : 0;
        const uint64_t finalX =
            static_cast<uint64_t>((maxX - minX) / step_m + 1.0f + 2.0f * Dpx + 0.5f);
        const uint64_t finalY =
            static_cast<uint64_t>((maxY - minY) / step_m + 1.0f + 2.0f * Dpx + 0.5f);
        return (finalX + pad) * (finalY + pad) * sizeof(float);
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
    SLS_LOG("[DEBUG] generateCRN: maxCorr=%.1f, tempBufBytes=%llu\n",
            maxCorr,
            (unsigned long long)tempBufBytes(maxCorr));

    wgpu::Buffer tempBuf =
        makeBuffer(tempBufBytes(maxCorr), WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    SLS_LOG("[DEBUG] generateCRN: tempBuf created\n");

    // Check if GPU can handle the CRN output buffers
    // CRN buffers are nSite * channels * gridSz * sizeof(float) each
    // With nSite=19, gridSz=20259001: LOS=11.5GB, NLOS/O2I=10GB
    // Use the adapter's actual maxStorageBufferBindingSize (set in createDevice)
    SLS_LOG("[DEBUG] generateCRN: GPU max buffer size = %llu bytes (%.1f GB)\n",
            (unsigned long long)m_maxGpuBuffer_,
            (double)m_maxGpuBuffer_ / (1024.0 * 1024.0 * 1024.0));

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

    auto dispatchGrid = [&](wgpu::Buffer& outputBuf,
                            float corrDist,
                            uint32_t gridIndex,
                            uint32_t rowOffset = 0,
                            uint32_t chunkY = 0) {
        // Match WGSL: grid in pixels = round(bound/step + 1 + 2*D)
        // where D = 3 * (corrDist / step).
        const float Dpx = 3.0f * (corrDist / step_m);
        const uint64_t pnx =
            static_cast<uint64_t>((maxX - minX) / step_m + 1.0f + 2.0f * Dpx + 0.5f);
        const uint64_t pny =
            static_cast<uint64_t>((maxY - minY) / step_m + 1.0f + 2.0f * Dpx + 0.5f);
        const uint64_t curTempBytes = pnx * pny * sizeof(float);
        // Use chunkY for the output grid size if chunking; otherwise use full gridSz
        const uint64_t gridBytes = (chunkY > 0) ? static_cast<uint64_t>(chunkY) * nX * sizeof(float)
                                                : gridSz * sizeof(float);
        // destOffset = channel offset (full grid) + Y offset from rowOffset
        const uint64_t fullGridBytes = static_cast<uint64_t>(nX) * nY * sizeof(float);
        // When chunking, write to staging buffer at rowOffset position; otherwise write to output
        // buffer
        const uint64_t destOffset = (chunkY > 0) ? uint64_t(rowOffset) * nX * sizeof(float)
                                                 : uint64_t(gridIndex) * fullGridBytes +
                                                       uint64_t(rowOffset) * nX * sizeof(float);
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
                              step_m,
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
        SLS_LOG("[DEBUG] dispatchGrid: corrDist=%.1f, curTempBytes=%llu, actualTempBytes=%llu, "
                "pnx=%llu, pny=%llu\n",
                corrDist,
                (unsigned long long)curTempBytes,
                (unsigned long long)actualTempBytes,
                (unsigned long long)pnx,
                (unsigned long long)pny);
        SLS_LOG("[DEBUG] dispatchGrid: genUniBuf=%p, tempBuf=%p, crnRngBuf=%p\n",
                (void*)genUniBuf,
                (void*)tempBuf,
                (void*)crnRngBuf);
        SLS_LOG("[DEBUG] dispatchGrid: genUni values: maxX=%.1f minX=%.1f maxY=%.1f minY=%.1f "
                "corrDist=%.1f maxRngStates=%u nX=%u nY=%u step=%.1f boundX=%.1f boundY=%.1f\n",
                genUni.maxX,
                genUni.minX,
                genUni.maxY,
                genUni.minY,
                genUni.corrDist,
                genUni.maxRngStates,
                genUni.nX,
                genUni.nY,
                genUni.step,
                genUni.boundX,
                genUni.boundY);
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
        SLS_LOG("[DEBUG] dispatchGrid: creating fill bind group...\n");
        wgpu::BindGroup fillBg = device_.createBindGroup(fillBgDesc);
        SLS_LOG("[DEBUG] dispatchGrid: fill bind group created, fillBg=%p\n", (void*)fillBg);

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
            SLS_LOG("[DEBUG] dispatchGrid: dispatching %u workgroups for %llu elements\\n",
                    workgroups,
                    (unsigned long long)chunk_total);
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
                SLS_LOG(
                    "[DEBUG] dispatchGrid: convolve dispatching %u workgroups for %llu elements\\n",
                    convWorkgroups,
                    (unsigned long long)chunk_total);
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
    const uint64_t maxGridBufBytes = 128ULL * 1024ULL * 1024ULL; // 128MB per chunk
    const uint64_t maxChunkY = maxGridBufBytes / (static_cast<uint64_t>(nX) * sizeof(float));
    // Clamp to respect WebGPU 65535 workgroup limit per dimension:
    // workgroups = (pnx * chunkY + 255) / 256 <= 65535
    // => pnx * chunkY <= 16776961  (pnx <= nX, so nX is safe upper bound)
    const uint64_t wgLimitChunkY = 16776961u / static_cast<uint64_t>(nX);
    const uint64_t clampedMaxChunkY =
        std::min(maxChunkY, std::max(static_cast<uint64_t>(1u), wgLimitChunkY));
    const uint64_t nChunksY = (static_cast<uint64_t>(nY) + clampedMaxChunkY - 1) / clampedMaxChunkY;
    SLS_LOG("[DEBUG] CRN chunking: nX=%d nY=%d maxChunkY=%llu nChunksY=%llu\n",
            nX,
            nY,
            (unsigned long long)clampedMaxChunkY,
            (unsigned long long)nChunksY);

    // Also chunk the output buffers: create them on CPU side to avoid GPU memory limit
    // LOS: nSite * 8 channels, NLOS: nSite * 7 channels, O2I: nSite * 7 channels
    const size_t losBufSize = static_cast<size_t>(nSite_) * 8ULL * static_cast<uint64_t>(nX) * nY;
    const size_t nlosBufSize = static_cast<size_t>(nSite_) * 7ULL * static_cast<uint64_t>(nX) * nY;
    const size_t o2iBufSize = static_cast<size_t>(nSite_) * 7ULL * static_cast<uint64_t>(nX) * nY;
    std::vector<float> losOutBuf(losBufSize, 0.0f);
    std::vector<float> nlosOutBuf(nlosBufSize, 0.0f);
    std::vector<float> o2iOutBuf(o2iBufSize, 0.0f);
    SLS_LOG("[DEBUG] CRN: GPU output buffers created (los=%lluMB, nlos=%lluMB, o2i=%lluMB)\n",
            (unsigned long long)(losOutBuf.size() * sizeof(float)) / (1024 * 1024),
            (unsigned long long)(nlosOutBuf.size() * sizeof(float)) / (1024 * 1024),
            (unsigned long long)(o2iOutBuf.size() * sizeof(float)) / (1024 * 1024));

    // Create staging buffers for chunked output (small enough to fit in GPU memory)
    const uint64_t stagingGridBytes = static_cast<uint64_t>(clampedMaxChunkY) * nX * sizeof(float);
    wgpu::Buffer stagingBuf =
        makeBuffer(stagingGridBytes,
                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst);
    SLS_LOG("[DEBUG] CRN: stagingBuf created (%llu bytes)\n", (unsigned long long)stagingGridBytes);

    SLS_LOG("[DEBUG] Starting CRN generation loop\n");
    for (uint32_t s = 0; s < nSite_; s++)
    {
        SLS_LOG("[DEBUG] LOS site %u\n", s);
        for (uint32_t ch = 0; ch < 8; ch++)
        {
            for (uint32_t c = 0; c < nChunksY; c++)
            {
                const uint32_t yOff = static_cast<uint32_t>(c * clampedMaxChunkY);
                const uint32_t yRows = static_cast<uint32_t>(
                    std::min(static_cast<uint64_t>(clampedMaxChunkY),
                             static_cast<uint64_t>(nY) - c * clampedMaxChunkY));
                const uint32_t gridIdx = static_cast<uint32_t>(s) * 8 + ch;
                SLS_LOG("[DEBUG]   dispatchGrid LOS s=%u ch=%u c=%u yOff=%u yRows=%u\n",
                        s,
                        ch,
                        c,
                        yOff,
                        yRows);
                // Use stagingBuf as output instead of crnLosBuf_
                dispatchGrid(stagingBuf, corrLos[ch], gridIdx, yOff, yRows);
                // Copy staging buffer to CPU output buffer
                const uint64_t cpuOffset = static_cast<uint64_t>(s) * 8 * nX * nY +
                                           static_cast<uint64_t>(ch) * nX * nY +
                                           static_cast<uint64_t>(yOff) * nX;
                const uint64_t chunkBytes = static_cast<uint64_t>(yRows) * nX * sizeof(float);
                std::vector<float> stagingData(chunkBytes / sizeof(float));
                wgpu::Buffer stagingReadBuf =
                    makeBuffer(chunkBytes, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
                // Copy staging to stagingReadBuf
                {
                    wgpu::CommandEncoder enc = device_.createCommandEncoder(wgpu::Default);
                    enc.copyBufferToBuffer(stagingBuf, 0, stagingReadBuf, 0, chunkBytes);
                    queue_.submit(enc.finish(wgpu::Default));
                    waitIdle();
                }
                // Map and read using wgpu-native C API
                static WGPUMapMode g_mapStatus = WGPUMapMode_None;
                auto g_callback = [](WGPUMapAsyncStatus status,
                                     WGPUStringView message,
                                     void* userdata,
                                     void* /*another*/) {
                    (void)message;
                    *(reinterpret_cast<WGPUMapMode*>(userdata)) =
                        status == WGPUMapAsyncStatus_Success ? WGPUMapMode_Read : WGPUMapMode_None;
                };
                WGPUBufferMapCallbackInfo mapCbInfo = {};
                mapCbInfo.nextInChain = nullptr;
                mapCbInfo.mode = WGPUCallbackMode_AllowProcessEvents;
                mapCbInfo.callback = g_callback;
                mapCbInfo.userdata1 = &g_mapStatus;
                mapCbInfo.userdata2 = nullptr;
                wgpuBufferMapAsync(stagingReadBuf,
                                   WGPUMapMode_Read,
                                   0,
                                   static_cast<uint64_t>(chunkBytes),
                                   mapCbInfo);
                waitIdle();
                if (g_mapStatus != WGPUMapMode_Read)
                {
                    SLS_LOG("[ERROR] Failed to map stagingReadBuf\n");
                    continue;
                }
                const void* mapped = wgpuBufferGetConstMappedRange(stagingReadBuf, 0, chunkBytes);
                if (!mapped)
                {
                    SLS_LOG("[ERROR] Failed to get mapped range\n");
                    continue;
                }
                memcpy(stagingData.data(), mapped, chunkBytes);
                wgpuBufferUnmap(stagingReadBuf);
                // Copy to CPU buffer
                for (uint64_t i = 0; i < chunkBytes / sizeof(float); i++)
                {
                    losOutBuf[cpuOffset + i] = stagingData[i];
                }
                SLS_LOG("[DEBUG]   LOS chunk %u copied to CPU buffer\n", c);
            }
        }
    }
    for (uint32_t s = 0; s < nSite_; s++)
    {
        SLS_LOG("[DEBUG] NLOS site %u\n", s);
        for (uint32_t ch = 0; ch < 7; ch++)
        {
            for (uint32_t c = 0; c < nChunksY; c++)
            {
                const uint32_t yOff = static_cast<uint32_t>(c * clampedMaxChunkY);
                const uint32_t yRows = static_cast<uint32_t>(
                    std::min(static_cast<uint64_t>(clampedMaxChunkY),
                             static_cast<uint64_t>(nY) - c * clampedMaxChunkY));
                const uint32_t gridIdx = static_cast<uint32_t>(s) * 7 + ch;
                SLS_LOG("[DEBUG]   dispatchGrid NLOS s=%u ch=%u c=%u yOff=%u yRows=%u\n",
                        s,
                        ch,
                        c,
                        yOff,
                        yRows);
                dispatchGrid(stagingBuf, corrNlos[ch], gridIdx, yOff, yRows);
                const uint64_t cpuOffset = static_cast<uint64_t>(s) * 7 * nX * nY +
                                           static_cast<uint64_t>(ch) * nX * nY +
                                           static_cast<uint64_t>(yOff) * nX;
                const uint64_t chunkBytes = static_cast<uint64_t>(yRows) * nX * sizeof(float);
                std::vector<float> stagingData(chunkBytes / sizeof(float));
                wgpu::Buffer stagingReadBuf =
                    makeBuffer(chunkBytes, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
                {
                    wgpu::CommandEncoder enc = device_.createCommandEncoder(wgpu::Default);
                    enc.copyBufferToBuffer(stagingBuf, 0, stagingReadBuf, 0, chunkBytes);
                    queue_.submit(enc.finish(wgpu::Default));
                    waitIdle();
                }
                // Map and read using wgpu-native C API
                static WGPUMapMode g_mapStatusNlos = WGPUMapMode_None;
                auto g_callbackNlos = [](WGPUMapAsyncStatus status,
                                         WGPUStringView message,
                                         void* userdata,
                                         void* /*another*/) {
                    (void)message;
                    *(reinterpret_cast<WGPUMapMode*>(userdata)) =
                        status == WGPUMapAsyncStatus_Success ? WGPUMapMode_Read : WGPUMapMode_None;
                };
                WGPUBufferMapCallbackInfo mapCbInfoNlos = {};
                mapCbInfoNlos.nextInChain = nullptr;
                mapCbInfoNlos.mode = WGPUCallbackMode_AllowProcessEvents;
                mapCbInfoNlos.callback = g_callbackNlos;
                mapCbInfoNlos.userdata1 = &g_mapStatusNlos;
                mapCbInfoNlos.userdata2 = nullptr;
                wgpuBufferMapAsync(stagingReadBuf,
                                   WGPUMapMode_Read,
                                   0,
                                   static_cast<uint64_t>(chunkBytes),
                                   mapCbInfoNlos);
                waitIdle();
                if (g_mapStatusNlos != WGPUMapMode_Read)
                {
                    SLS_LOG("[ERROR] Failed to map stagingReadBuf (NLOS)\n");
                    continue;
                }
                const void* mappedNlos =
                    wgpuBufferGetConstMappedRange(stagingReadBuf, 0, chunkBytes);
                if (!mappedNlos)
                {
                    SLS_LOG("[ERROR] Failed to get mapped range (NLOS)\n");
                    continue;
                }
                memcpy(stagingData.data(), mappedNlos, chunkBytes);
                wgpuBufferUnmap(stagingReadBuf);
                for (uint64_t i = 0; i < chunkBytes / sizeof(float); i++)
                {
                    nlosOutBuf[cpuOffset + i] = stagingData[i];
                }
                SLS_LOG("[DEBUG]   NLOS chunk %u copied to CPU buffer\n", c);
            }
        }
    }
    for (uint32_t s = 0; s < nSite_; s++)
    {
        SLS_LOG("[DEBUG] O2I site %u\n", s);
        for (uint32_t ch = 0; ch < 7; ch++)
        {
            for (uint32_t c = 0; c < nChunksY; c++)
            {
                const uint32_t yOff = static_cast<uint32_t>(c * clampedMaxChunkY);
                const uint32_t yRows = static_cast<uint32_t>(
                    std::min(static_cast<uint64_t>(clampedMaxChunkY),
                             static_cast<uint64_t>(nY) - c * clampedMaxChunkY));
                const uint32_t gridIdx = static_cast<uint32_t>(s) * 7 + ch;
                SLS_LOG("[DEBUG]   dispatchGrid O2I s=%u ch=%u c=%u yOff=%u yRows=%u\n",
                        s,
                        ch,
                        c,
                        yOff,
                        yRows);
                dispatchGrid(stagingBuf, corrO2i[ch], gridIdx, yOff, yRows);
                const uint64_t cpuOffset = static_cast<uint64_t>(s) * 7 * nX * nY +
                                           static_cast<uint64_t>(ch) * nX * nY +
                                           static_cast<uint64_t>(yOff) * nX;
                const uint64_t chunkBytes = static_cast<uint64_t>(yRows) * nX * sizeof(float);
                std::vector<float> stagingData(chunkBytes / sizeof(float));
                wgpu::Buffer stagingReadBuf =
                    makeBuffer(chunkBytes, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
                {
                    wgpu::CommandEncoder enc = device_.createCommandEncoder(wgpu::Default);
                    enc.copyBufferToBuffer(stagingBuf, 0, stagingReadBuf, 0, chunkBytes);
                    queue_.submit(enc.finish(wgpu::Default));
                    waitIdle();
                }
                // Map and read using wgpu-native C API
                static WGPUMapMode g_mapStatusO2i = WGPUMapMode_None;
                auto g_callbackO2i = [](WGPUMapAsyncStatus status,
                                        WGPUStringView message,
                                        void* userdata,
                                        void* /*another*/) {
                    (void)message;
                    *(reinterpret_cast<WGPUMapMode*>(userdata)) =
                        status == WGPUMapAsyncStatus_Success ? WGPUMapMode_Read : WGPUMapMode_None;
                };
                WGPUBufferMapCallbackInfo mapCbInfoO2i = {};
                mapCbInfoO2i.nextInChain = nullptr;
                mapCbInfoO2i.mode = WGPUCallbackMode_AllowProcessEvents;
                mapCbInfoO2i.callback = g_callbackO2i;
                mapCbInfoO2i.userdata1 = &g_mapStatusO2i;
                mapCbInfoO2i.userdata2 = nullptr;
                wgpuBufferMapAsync(stagingReadBuf,
                                   WGPUMapMode_Read,
                                   0,
                                   static_cast<uint64_t>(chunkBytes),
                                   mapCbInfoO2i);
                waitIdle();
                if (g_mapStatusO2i != WGPUMapMode_Read)
                {
                    SLS_LOG("[ERROR] Failed to map stagingReadBuf (O2I)\n");
                    continue;
                }
                const void* mappedO2i =
                    wgpuBufferGetConstMappedRange(stagingReadBuf, 0, chunkBytes);
                if (!mappedO2i)
                {
                    SLS_LOG("[ERROR] Failed to get mapped range (O2I)\n");
                    continue;
                }
                memcpy(stagingData.data(), mappedO2i, chunkBytes);
                wgpuBufferUnmap(stagingReadBuf);
                for (uint64_t i = 0; i < chunkBytes / sizeof(float); i++)
                {
                    o2iOutBuf[cpuOffset + i] = stagingData[i];
                }
                SLS_LOG("[DEBUG]   O2I chunk %u copied to CPU buffer\n", c);
            }
        }
    }
    SLS_LOG("[DEBUG] All CRN generation complete\n");

    // Final guard before the large GPU uploads. The earlier guard at the top of
    // generateCRN already rejects oversized configs, but keep this as a tripwire
    // so any future code path that gets here with a huge buffer fails loudly.
    auto guardCrnSize = [&](const char* tag, uint64_t bytes) {
        const uint64_t cap =
            (m_maxGpuBuffer_ == 0) ? (1ULL << 30) : std::min<uint64_t>(1ULL << 30, m_maxGpuBuffer_);
        if (bytes > cap)
        {
            SLS_LOG("[FATAL] CRN %s upload %.2f GB > safety cap %.2f GB\n",
                    tag,
                    (double)bytes / (1024.0 * 1024.0 * 1024.0),
                    (double)cap / (1024.0 * 1024.0 * 1024.0));
            std::fflush(stderr);
            std::abort();
        }
    };

    // Build the combined CRN data buffer: concat(LOS | NLOS | O2I).
    // Build the combined offsets buffer with each section's offsets
    // PRE-BAKED to address into the concatenated data buffer:
    //   losOffs[i]  = i*nX*nY                            (no shift)
    //   nlosOffs[i] = losDataElems + i*nX*nY             (skip LOS region)
    //   o2iOffs[i]  = losDataElems + nlosDataElems + i*nX*nY
    // so the kernel can do a single indexed read without per-section
    // arithmetic.
    if (!crnDataBuf_)
    {
        const uint64_t losBytes = losOutBuf.size() * sizeof(float);
        const uint64_t nlosBytes = nlosOutBuf.size() * sizeof(float);
        const uint64_t o2iBytes = o2iOutBuf.size() * sizeof(float);
        const uint64_t combinedBytes = losBytes + nlosBytes + o2iBytes;
        guardCrnSize("combined", combinedBytes);

        std::vector<float> combinedData;
        combinedData.reserve(combinedBytes / sizeof(float));
        combinedData.insert(combinedData.end(), losOutBuf.begin(), losOutBuf.end());
        combinedData.insert(combinedData.end(), nlosOutBuf.begin(), nlosOutBuf.end());
        combinedData.insert(combinedData.end(), o2iOutBuf.begin(), o2iOutBuf.end());
        crnDataBuf_ = makeBuffer(combinedBytes,
                                 WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
                                 combinedData.data());

        const uint32_t losElems = static_cast<uint32_t>(losOutBuf.size());
        const uint32_t nlosElems = static_cast<uint32_t>(nlosOutBuf.size());
        const uint32_t losOffsetCount = nSite_ * 8u;
        const uint32_t nlosOffsetCount = nSite_ * 7u;
        const uint32_t o2iOffsetCount = nSite_ * 7u;
        const uint32_t totalOffsets = losOffsetCount + nlosOffsetCount + o2iOffsetCount;

        std::vector<uint32_t> combinedOffsets(totalOffsets);
        const uint32_t gridSz = uint32_t(nX_) * uint32_t(nY_);
        // LOS section
        for (uint32_t i = 0; i < losOffsetCount; ++i)
        {
            combinedOffsets[i] = i * gridSz;
        }
        // NLOS section — element offsets shifted past the LOS region.
        for (uint32_t i = 0; i < nlosOffsetCount; ++i)
        {
            combinedOffsets[losOffsetCount + i] = losElems + i * gridSz;
        }
        // O2I section — shifted past LOS + NLOS.
        for (uint32_t i = 0; i < o2iOffsetCount; ++i)
        {
            combinedOffsets[losOffsetCount + nlosOffsetCount + i] =
                losElems + nlosElems + i * gridSz;
        }
        crnOffsetsBuf_ = makeBuffer(totalOffsets * sizeof(uint32_t),
                                    WGPUBufferUsage_Storage,
                                    combinedOffsets.data());
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

    // Cell buffer is one entry per (site, sector); WGSL kernel indexes it as
    // cell_params[site_idx * nSectorPerSite + sector]. Sizing the binding by
    // nSite alone clips the buffer to the first nSite/nSectorPerSite-th of
    // the data and the kernel reads zeros for higher site indices.
    const uint64_t cellParamsSz = uint64_t(nSite) * uint64_t(nSectorPerSite) * sizeof(CellParam);
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
    // cal_link_param_kernel now uses bindings 6-15 only (10 entries:
    // 1 uniform + 9 storage). The combined crn_data + crn_offsets
    // bindings replaced the previous 3+3 split.
    std::vector<wgpu::BindGroupEntry> entries(10, wgpu::Default);
    auto E = [&](int i, uint32_t binding, wgpu::Buffer buf, uint64_t size = WGPU_WHOLE_SIZE) {
        entries[i].binding = binding;
        entries[i].buffer = buf;
        entries[i].offset = 0;
        entries[i].size = size;
        if (!buf)
        {
            SLS_CERR << "null buffer at entry " << i << " (binding " << binding << ")" << std::endl;
            abort();
        }
    };
    const uint64_t combinedCrnSz = uint64_t(nSite_) * 22ULL * uint64_t(nX_) * nY_ * sizeof(float);
    const uint64_t combinedOffSz = uint64_t(nSite_) * 22ULL * sizeof(uint32_t);
    E(0, 6, uniBuf, sizeof(uniData));
    E(1, 7, cellParamsBuf_, cellParamsSz);
    E(2, 8, utParamsBuf_, utParamsSz);
    E(3, 9, sysConfigBuf_, sizeof(SystemLevelConfigGPU));
    E(4, 10, simConfigBuf_, sizeof(SimConfigGPU));
    E(5, 11, cmnLinkBuf_, sizeof(CmnLinkParamsGPU));
    E(6, 12, linkParamsBuf_, linkParamsSz);
    E(7, 13, rngStatesBuf_, rngStatesSz);
    E(8, 14, crnDataBuf_, combinedCrnSz);
    E(9, 15, crnOffsetsBuf_, combinedOffSz);

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
    scSpacingHzCache_ = scSpacingHz;
    fftSizeCache_ = fftSize;
    nPrbCache_ = nPrb;
    nSnapshotPerSlotCache_ = nSnapshotPerSlot;

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
    ssCmnCache_ = cmn;
    ssCmnCacheValid_ = true;
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
    if (!clusterRayPipeline_)
    {
        clusterRayPipeline_ = makePipeline("cal_cluster_ray_kernel");
        assert(clusterRayPipeline_ && "missing cal_cluster_ray_kernel in WGSL "
                                      "(or backend can't compile it)");
    }

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

    // Pack 6 logically-separate per-link f32 arrays (xpr,
    // randomPhases, phi_nm_AoA/AoD, theta_nm_ZOA/ZOD) into one
    // storage buffer. Layout per link (PACKED_LINK_STRIDE = 3600 f32):
    //   offset 0    .. 399  : xpr           (MAX_CR  = 400)
    //   offset 400  .. 1999 : randomPhases  (MAX_CR4 = 1600)
    //   offset 2000 .. 2399 : phi_nm_AoA    (MAX_CR  = 400)
    //   offset 2400 .. 2799 : phi_nm_AoD    (MAX_CR  = 400)
    //   offset 2800 .. 3199 : theta_nm_ZOA  (MAX_CR  = 400)
    //   offset 3200 .. 3599 : theta_nm_ZOD  (MAX_CR  = 400)
    // The PACKED_OFF_* / PACKED_LINK_STRIDE constants in
    // sls-chan.wgsl must mirror these exactly.
    constexpr uint32_t PACKED_OFF_XPR = 0u;
    constexpr uint32_t PACKED_OFF_RNDP = 400u;
    constexpr uint32_t PACKED_OFF_AOA = 2000u;
    constexpr uint32_t PACKED_OFF_AOD = 2400u;
    constexpr uint32_t PACKED_OFF_ZOA = 2800u;
    constexpr uint32_t PACKED_OFF_ZOD = 3200u;
    constexpr uint32_t PACKED_LINK_STRIDE = 3600u;
    (void)PACKED_OFF_XPR;
    (void)PACKED_OFF_RNDP;
    (void)PACKED_OFF_AOA;
    (void)PACKED_OFF_AOD;
    (void)PACKED_OFF_ZOA;
    (void)PACKED_OFF_ZOD;
    const uint64_t clusterOutputsSz =
        uint64_t(nSite) * nUT * PACKED_LINK_STRIDE * sizeof(float);
    clusterOutputsBuf_ =
        makeBuffer(clusterOutputsSz,
                   WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);

    auto layout1 = clusterRayPipeline_.getBindGroupLayout(1);
    // Was 12 bindings (6 separate output arrays at bindings 6..11);
    // packed into one at binding 6 -> 7 entries total now.
    std::vector<wgpu::BindGroupEntry> e0(7, wgpu::Default);
    auto E = [](std::vector<wgpu::BindGroupEntry>& v,
                int i,
                uint32_t b,
                wgpu::Buffer buf,
                uint64_t sz = WGPU_WHOLE_SIZE) {
        v[i].binding = b;
        v[i].buffer = buf;
        v[i].size = sz;
    };
    E(e0, 0, 0, linkParamsBuf_);                     // cray_buf_link
    E(e0, 1, 1, ssUtParamsBuf_);                     // cray_buf_ut
    E(e0, 2, 2, ssCmnLinkBuf_, sizeof(SsCmnParams)); // cray_buf_cmn (non-array storage)
    E(e0, 3, 3, clusterParamsBuf_);                  // cray_buf_cluster
    E(e0, 4, 4, rngStatesBuf_);                      // cray_buf_rng
    E(e0, 5, 5, ssDispatchBuf_);                     // cray_disp
    E(e0, 6, 6, clusterOutputsBuf_);                 // cray_packed_out

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
    activeLinksCache_ = activeLinks;
    nSnapshotsCache_ = nSnapshots;
    if (!generateCIRPipeline_)
    {
        generateCIRPipeline_ = makePipeline("generate_cir_kernel");
        assert(generateCIRPipeline_ && "missing generate_cir_kernel in WGSL "
                                       "(or backend can't compile it)");
    }
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

    if (!cirDbgBuf_)
    {
        cirDbgBuf_ = makeBuffer(uint64_t(nActiveLinks) * 16ULL * sizeof(float),
                                WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    }

    auto layout2 = generateCIRPipeline_.getBindGroupLayout(2);
    // Was 22 (bindings 0..21 with 6 separate cluster-output arrays at
    // 15..20). Now 17 -- the 6 outputs collapsed into one packed view
    // at binding 15 and cir_dbg moved to binding 16.
    std::vector<wgpu::BindGroupEntry> e0(17, wgpu::Default);
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
    E(e0, 15, 15, clusterOutputsBuf_); // cir_packed_in
    E(e0, 16, 16, cirDbgBuf_);         // cir_dbg (was binding 21)

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
    if (!generateCFRPipeline_)
    {
        generateCFRPipeline_ = makePipeline("generate_cfr_kernel_mode1");
        assert(generateCFRPipeline_ && "missing generate_cfr_kernel_mode1 in WGSL "
                                       "(or backend can't compile it)");
    }
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
#ifdef WEBGPU_BACKEND_DAWN
    dawnPumpEvents();
#else
    wgpuDevicePoll(device_, false, nullptr);
#endif
    SLS_CERR << "CFR entry: device alive" << std::endl;

    if (!freqChanPrbgBuf_)
    {
        const uint64_t cfrElems =
            uint64_t(nActiveLinks) * nSnapshots * ssNBsAnt_ * ssNUeAnt_ * ssNPrbg_;
        const uint64_t bufBytes = cfrElems * sizeof(float) * 2;
        SLS_LOG("CFR: cfrElems=%llu, freqChanPrbgBuf size=%.1f MB\n",
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
    if (!generateCFRPipeline_)
    {
        generateCFRPipeline_ = makePipeline("generate_cfr_kernel_mode1");
        assert(generateCFRPipeline_ && "missing generate_cfr_kernel_mode1 in WGSL "
                                       "(or backend can't compile it)");
    }
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
        {
            break;
        }

        // Allocate per-batch GPU buffer
        const uint64_t batchElems = uint64_t(batchLen) * elemsPerLink;
        const uint64_t batchBufBytes = batchElems * sizeof(float) * 2;

        // Check if device is healthy before allocating
    #ifdef WEBGPU_BACKEND_DAWN
    dawnPumpEvents();
#else
    wgpuDevicePoll(device_, false, nullptr);
#endif
        if (isDead())
        {
            SLS_CERR << "CFR batch " << b << ": device lost before buffer alloc" << std::endl;
            return;
        }

        auto batchBuf =
            makeBuffer(batchBufBytes,
                       WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc);

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
            batchLinks[i].freqChanPrbgOffset =
                i * elemsPerLink; // relative offset within per-batch buffer
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

    SLS_CERR << "CFR batched done: " << nActiveLinks << " links in " << nBatches << " batches"
             << std::endl;
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
        SLS_LOG("Device alive\n");
        return false;
    }
    else
    {
        SLS_LOG("Device dead\n");
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
SlsChanWgpu::readCirDebug(uint32_t nActiveLinks)
{
    if (!cirDbgBuf_)
    {
        return {};
    }
    const uint64_t sz = uint64_t(nActiveLinks) * 16ULL * sizeof(float);
    wgpu::Buffer staging = makeBuffer(sz, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    auto enc = device_.createCommandEncoder(wgpu::Default);
    enc.copyBufferToBuffer(cirDbgBuf_, 0, staging, 0, sz);
    queue_.submit(enc.finish(wgpu::Default));
    waitIdle();
    return mapReadBuffer<float>(staging, sz);
}

namespace
{
// Mirrors PACKED_OFF_* / PACKED_LINK_STRIDE in calClusterRay() above
// and in sls-chan.wgsl. These describe how the six per-link f32
// arrays are laid out inside clusterOutputsBuf_ -- they must stay
// in sync across the three places (calClusterRay alloc, WGSL
// constants, here).
constexpr uint32_t kPackedOffXpr = 0u;
constexpr uint32_t kPackedOffRndp = 400u;
constexpr uint32_t kPackedOffAoA = 2000u;
constexpr uint32_t kPackedOffAoD = 2400u;
constexpr uint32_t kPackedOffZoA = 2800u;
constexpr uint32_t kPackedOffZoD = 3200u;
constexpr uint32_t kPackedLinkStride = 3600u;
constexpr uint32_t kMaxCr = 400u; // MAX_CLUSTERS * MAX_RAYS

} // namespace

// Slice one of the six per-link sub-views out of the packed
// cluster-output buffer into a flat std::vector<float>. Each
// sub-view is `perLinkLen` f32 per link (400 for the angle /
// xpr arrays, 1600 for randomPhases), and lives at byte
// offset `linkOff*sizeof(float)` inside each link's
// kPackedLinkStride-f32 slab.
std::vector<float>
SlsChanWgpu::sliceClusterOutput(uint32_t linkOffF32, uint32_t perLinkLenF32)
{
    if (!clusterOutputsBuf_)
    {
        return {};
    }
    const uint64_t totalBytes = clusterOutputsBuf_.getSize();
    const uint64_t totalF32 = totalBytes / sizeof(float);
    assert(totalF32 % kPackedLinkStride == 0u
           && "clusterOutputsBuf_ size is not a multiple of kPackedLinkStride");
    const uint64_t nLinks = totalF32 / kPackedLinkStride;

    // Copy the whole buffer to host (one mapAsync round-trip) and
    // gather. Simpler than a per-link copyBufferToBuffer chain and
    // dwarfed by the actual GPU work for any meaningful nLinks.
    wgpu::Buffer staging =
        makeBuffer(totalBytes, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    auto enc = device_.createCommandEncoder(wgpu::Default);
    enc.copyBufferToBuffer(clusterOutputsBuf_, 0, staging, 0, totalBytes);
    queue_.submit(enc.finish(wgpu::Default));
    waitIdle();
    auto whole = mapReadBuffer<float>(staging, totalBytes);

    std::vector<float> out(static_cast<size_t>(nLinks) * perLinkLenF32);
    for (uint64_t l = 0; l < nLinks; ++l)
    {
        const float* src = whole.data() + l * kPackedLinkStride + linkOffF32;
        float* dst = out.data() + l * perLinkLenF32;
        std::memcpy(dst, src, perLinkLenF32 * sizeof(float));
    }
    return out;
}

std::vector<float>
SlsChanWgpu::readXpr()
{
    return sliceClusterOutput(kPackedOffXpr, kMaxCr);
}

std::vector<float>
SlsChanWgpu::readPhiNmAoA()
{
    return sliceClusterOutput(kPackedOffAoA, kMaxCr);
}

std::vector<float>
SlsChanWgpu::readPhiNmAoD()
{
    return sliceClusterOutput(kPackedOffAoD, kMaxCr);
}

std::vector<float>
SlsChanWgpu::readThetaNmZOA()
{
    return sliceClusterOutput(kPackedOffZoA, kMaxCr);
}

std::vector<float>
SlsChanWgpu::readThetaNmZOD()
{
    return sliceClusterOutput(kPackedOffZoD, kMaxCr);
}

// ── genChannelMatrix ───────────────────────────────────────────────────────────
// Dispatch gen_channel_matrix_kernel. The kernel currently emits a placeholder
// (zeros) into the per-link output; the mezanine reads the matrix back and
// detects all-zero entries to decide whether to fall back to the CPU
// GetNewChannel. The actual ray-sum / antenna-pattern math will replace the
// placeholder cell once the kernel body is written.
void
SlsChanWgpu::genChannelMatrix(const std::vector<ActiveLink>& activeLinks,
                              uint32_t nActiveLinks,
                              uint32_t uSize,
                              uint32_t sSize,
                              uint32_t numOverallCluster,
                              uint32_t numReducedCluster,
                              uint32_t nRays,
                              uint8_t cluster1st,
                              uint8_t cluster2nd)
{
    assert(linkParamsBuf_ && "call calLinkParam before genChannelMatrix");
    assert(clusterParamsBuf_ && "call calClusterRay before genChannelMatrix");
    assert(clusterOutputsBuf_ && "call calClusterRay before genChannelMatrix");
    assert(activeLinkBuf_ && "call generateCIR (or set activeLinkBuf_) before genChannelMatrix");

    if (!channelMatrixPipeline_)
    {
        channelMatrixPipeline_ = makePipeline("gen_channel_matrix_kernel");
        assert(channelMatrixPipeline_ &&
               "missing gen_channel_matrix_kernel in WGSL (or backend can't compile it)");
    }

    // (Re)alloc output buffer if shape changed. Sized for the maximum page
    // count (kMatMaxPages) so the kernel doesn't have to special-case the
    // strongest-2 subcluster split.
    const uint64_t perLink =
        uint64_t(uSize) * sSize * kMatMaxPages * sizeof(float) * 2;
    const uint64_t totalBytes = perLink * nActiveLinks;
    if (!channelMatrixBuf_ || channelMatrixCfgUSize_ != uSize ||
        channelMatrixCfgSSize_ != sSize || channelMatrixCfgNLinks_ != nActiveLinks)
    {
        channelMatrixBuf_ =
            makeBuffer(totalBytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
        channelMatrixCfgUSize_ = uSize;
        channelMatrixCfgSSize_ = sSize;
        channelMatrixCfgNLinks_ = nActiveLinks;
    }

    struct MatDispUni
    {
        uint32_t nActiveLinks;
        uint32_t nOverallCluster;
        uint32_t uSize;
        uint32_t sSize;
        uint32_t nRays;
        uint32_t nReducedCluster;
        uint32_t cluster1st;
        uint32_t cluster2nd;
    };
    MatDispUni du{nActiveLinks,
                  numOverallCluster,
                  uSize,
                  sSize,
                  nRays,
                  numReducedCluster,
                  cluster1st,
                  cluster2nd};
    matrixDispatchBuf_ =
        makeBuffer(sizeof(du), WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage, &du);

    auto layout0 = channelMatrixPipeline_.getBindGroupLayout(0);
    std::vector<wgpu::BindGroupEntry> entries(11, wgpu::Default);
    auto E = [&](int i, uint32_t b, wgpu::Buffer buf, uint64_t sz = WGPU_WHOLE_SIZE) {
        entries[i].binding = b;
        entries[i].buffer = buf;
        entries[i].offset = 0;
        entries[i].size = sz;
    };
    // Bindings 30-40 chosen to avoid clashing with the LSP kernels'
    // group(0) bindings (0..15). See sls-chan.wgsl for the matching
    // declarations.
    E(0, 30, matrixDispatchBuf_, sizeof(du));
    E(1, 31, linkParamsBuf_);
    E(2, 32, clusterParamsBuf_);
    E(3, 33, clusterOutputsBuf_);
    E(4, 34, activeLinkBuf_);
    E(5, 35, ssCellParamsBuf_);
    E(6, 36, ssUtParamsBuf_);
    E(7, 37, antPanelConfigBuf_);
    E(8, 38, antThetaBuf_);
    E(9, 39, antPhiBuf_);
    E(10, 40, channelMatrixBuf_);

    wgpu::BindGroupDescriptor bgd = wgpu::Default;
    bgd.layout = layout0;
    bgd.entryCount = static_cast<uint32_t>(entries.size());
    bgd.entries = entries.data();
    wgpu::BindGroup bg0 = device_.createBindGroup(bgd);

    auto enc = device_.createCommandEncoder(wgpu::Default);
    auto pass = enc.beginComputePass(wgpu::Default);
    pass.setPipeline(channelMatrixPipeline_);
    pass.setBindGroup(0u, bg0, (size_t)0, nullptr);
    // workgroup_size = (64, 1, 1)
    // dispatch z = ceil(uSize*sSize / 64)
    const uint32_t usThreads = uSize * sSize;
    const uint32_t usGroups = (usThreads + 63u) / 64u;
    pass.dispatchWorkgroups(nActiveLinks, numOverallCluster, usGroups);
    pass.end();
    assert(!isDead());
    queue_.submit(enc.finish(wgpu::Default));
    waitIdle();
    (void)activeLinks; // bound via activeLinkBuf_ from generateCIR
}

std::vector<std::complex<float>>
SlsChanWgpu::readChannelMatrix(uint32_t nLinks, uint32_t uSize, uint32_t sSize)
{
    if (!channelMatrixBuf_)
    {
        return {};
    }
    const uint64_t totalBytes =
        uint64_t(nLinks) * uSize * sSize * kMatMaxPages * sizeof(std::complex<float>);
    assert(channelMatrixBuf_.getSize() >= totalBytes);
    wgpu::Buffer staging =
        makeBuffer(totalBytes, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    auto enc = device_.createCommandEncoder(wgpu::Default);
    enc.copyBufferToBuffer(channelMatrixBuf_, 0, staging, 0, totalBytes);
    queue_.submit(enc.finish(wgpu::Default));
    waitIdle();
    return mapReadBuffer<std::complex<float>>(staging, totalBytes);
}

// ── Save all channel metrics to HDF5 (matches NVIDIA slsChan::saveSlsChanToH5File) ──
#ifdef SLS_CHAN_HDF5
namespace
{

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

static void
writeSlsChanHdf5File(const std::string& filename,
                     const std::vector<LinkParams>& links,
                     uint32_t nSite,
                     uint32_t nUT,
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
                     float scSpacingHz,
                     uint32_t fftSize,
                     uint32_t nPrb,
                     uint32_t nSnapshotPerSlot,
                     float centerFreqHz,
                     float bandwidthHz,
                     uint32_t nUeAnt,
                     uint32_t nBsAnt,
                     const SsCmnParams& ssCmn,
                     const std::vector<CellParam>& cells,
                     const std::vector<CellParamSS>& cellsSS,
                     const std::vector<UtParam>& uts,
                     float isd,
                     float bsHeight,
                     float minBsUeDist2d,
                     float maxBsUeDist2dIndoor,
                     float indoorUtPercent,
                     uint32_t nSectorPerSite);

} // namespace

// Minimal HDF5 writer for the ns-3 LSP-only batch path. Emits just the
// datasets `minimal_analyzer.py` / `analysis_channel_stats.py` need for
// outdoor coupling-loss / SIR / SINR calibration: `linkParams`,
// `activeLinkParams`, and `topology/{nSite,nUT,nSector}`. Used when
// the small-scale GPU buffers (clusters, rays, CIR, CFR) weren't
// populated.
namespace
{

static void
writeLspOnlyHdf5(const std::string& filename,
                 const std::vector<LinkParams>& links,
                 const std::vector<CellParam>& cells,
                 const std::vector<UtParam>& uts,
                 uint32_t nSite,
                 uint32_t nUT,
                 uint32_t nSectorPerSite,
                 uint32_t gpuScenario,
                 float centerFreqHz)
{
    if (std::filesystem::exists(filename))
    {
        SLS_CERR << "writeLspOnlyHdf5: " << filename << " already exists" << std::endl;
        return;
    }
    hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0)
    {
        SLS_CERR << "writeLspOnlyHdf5: failed to create " << filename << std::endl;
        return;
    }
    const uint32_t nLinks = nSite * nUT;

    // linkParams compound matches the LinkParamsHdf5 layout the
    // NVIDIA analyzer reads. All 24 fields are emitted; the LSP-only
    // path leaves the small-scale-specific ones (mu_lgZSD etc) at
    // their GPU-computed values from cal_link_param_kernel.
    {
        hid_t compoundType = H5Tcreate(H5T_COMPOUND, sizeof(LinkParamsHdf5));
        H5Tinsert(compoundType, "cid", offsetof(LinkParamsHdf5, cid), H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "d2d", offsetof(LinkParamsHdf5, d2d), H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "d2d_in", offsetof(LinkParamsHdf5, d2d_in), H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "d2d_out", offsetof(LinkParamsHdf5, d2d_out), H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "d3d", offsetof(LinkParamsHdf5, d3d), H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "d3d_in", offsetof(LinkParamsHdf5, d3d_in), H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "d3d_out", offsetof(LinkParamsHdf5, d3d_out), H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType,
                  "phi_LOS_AOD",
                  offsetof(LinkParamsHdf5, phi_LOS_AOD),
                  H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType,
                  "theta_LOS_ZOD",
                  offsetof(LinkParamsHdf5, theta_LOS_ZOD),
                  H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType,
                  "phi_LOS_AOA",
                  offsetof(LinkParamsHdf5, phi_LOS_AOA),
                  H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType,
                  "theta_LOS_ZOA",
                  offsetof(LinkParamsHdf5, theta_LOS_ZOA),
                  H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "losInd", offsetof(LinkParamsHdf5, losInd), H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "pathloss", offsetof(LinkParamsHdf5, pathloss), H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "SF", offsetof(LinkParamsHdf5, SF), H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "K", offsetof(LinkParamsHdf5, K), H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "DS", offsetof(LinkParamsHdf5, DS), H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "ASD", offsetof(LinkParamsHdf5, ASD), H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "ASA", offsetof(LinkParamsHdf5, ASA), H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType,
                  "mu_lgZSD",
                  offsetof(LinkParamsHdf5, mu_lgZSD),
                  H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType,
                  "sigma_lgZSD",
                  offsetof(LinkParamsHdf5, sigma_lgZSD),
                  H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType,
                  "mu_offset_ZOD",
                  offsetof(LinkParamsHdf5, mu_offset_ZOD),
                  H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "ZSD", offsetof(LinkParamsHdf5, ZSD), H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "ZSA", offsetof(LinkParamsHdf5, ZSA), H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType,
                  "delta_tau",
                  offsetof(LinkParamsHdf5, delta_tau),
                  H5T_NATIVE_FLOAT);

        std::vector<LinkParamsHdf5> h5(nLinks);
        for (uint32_t i = 0; i < nLinks && i < links.size(); ++i)
        {
            const LinkParams& lk = links[i];
            h5[i].cid = (nUT > 0) ? i / nUT : 0u;
            h5[i].d2d = lk.d2d;
            h5[i].d2d_in = lk.d2d_in;
            h5[i].d2d_out = lk.d2d_out;
            h5[i].d3d = lk.d3d;
            h5[i].d3d_in = lk.d3d_in;
            h5[i].d3d_out = lk.d3d_out;
            // Note: LinkParams stores phi/theta in
            //   phi_LOS_AOD, phi_LOS_AOA, theta_LOS_ZOD, theta_LOS_ZOA
            // which matches the HDF5 names.
            h5[i].phi_LOS_AOD = lk.phi_LOS_AOD;
            h5[i].phi_LOS_AOA = lk.phi_LOS_AOA;
            h5[i].theta_LOS_ZOD = lk.theta_LOS_ZOD;
            h5[i].theta_LOS_ZOA = lk.theta_LOS_ZOA;
            h5[i].losInd = lk.losInd;
            h5[i].pathloss = lk.pathloss;
            h5[i].SF = lk.SF;
            h5[i].K = lk.K;
            h5[i].DS = lk.DS;
            h5[i].ASD = lk.ASD;
            h5[i].ASA = lk.ASA;
            h5[i].ZSD = lk.ZSD;
            h5[i].ZSA = lk.ZSA;
            h5[i].mu_lgZSD = lk.mu_lgZSD;
            h5[i].sigma_lgZSD = lk.sigma_lgZSD;
            h5[i].mu_offset_ZOD = lk.mu_offset_ZOD;
            h5[i].delta_tau = 0.0f;
        }
        hsize_t dims = nLinks;
        hid_t space = H5Screate_simple(1, &dims, nullptr);
        hid_t dset = H5Dcreate2(file,
                                "linkParams",
                                compoundType,
                                space,
                                H5P_DEFAULT,
                                H5P_DEFAULT,
                                H5P_DEFAULT);
        H5Dwrite(dset, compoundType, H5S_ALL, H5S_ALL, H5P_DEFAULT, h5.data());
        H5Dclose(dset);
        H5Sclose(space);
        H5Tclose(compoundType);
    }

    // activeLinkParams compound — synthesised from site-major flat
    // layout. minimal_analyzer.py reads this just to obtain the link
    // count.
    {
        struct ActiveRec
        {
            uint32_t cid, uid, linkIdx, lspReadIdx;
        };
        hid_t compoundType = H5Tcreate(H5T_COMPOUND, sizeof(ActiveRec));
        H5Tinsert(compoundType, "cid", offsetof(ActiveRec, cid), H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "uid", offsetof(ActiveRec, uid), H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "linkIdx", offsetof(ActiveRec, linkIdx), H5T_NATIVE_UINT32);
        H5Tinsert(compoundType,
                  "lspReadIdx",
                  offsetof(ActiveRec, lspReadIdx),
                  H5T_NATIVE_UINT32);

        std::vector<ActiveRec> rec(nLinks);
        for (uint32_t i = 0; i < nLinks; ++i)
        {
            rec[i].cid = (nUT > 0) ? i / nUT : 0u;
            rec[i].uid = (nUT > 0) ? i % nUT : 0u;
            rec[i].linkIdx = i;
            rec[i].lspReadIdx = i;
        }
        hsize_t dims = nLinks;
        hid_t space = H5Screate_simple(1, &dims, nullptr);
        hid_t dset = H5Dcreate2(file,
                                "activeLinkParams",
                                compoundType,
                                space,
                                H5P_DEFAULT,
                                H5P_DEFAULT,
                                H5P_DEFAULT);
        H5Dwrite(dset, compoundType, H5S_ALL, H5S_ALL, H5P_DEFAULT, rec.data());
        H5Dclose(dset);
        H5Sclose(space);
        H5Tclose(compoundType);
    }

    // topology group with nSite / nUT / nSector scalars + cellParams +
    // utParams (group-of-arrays layout the analyzer expects).
    {
        hid_t topo = H5Gcreate2(file, "topology", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        auto writeU32 = [&](const char* name, uint32_t val) {
            hsize_t one = 1;
            hid_t space = H5Screate_simple(1, &one, nullptr);
            hid_t dset = H5Dcreate2(topo,
                                    name,
                                    H5T_NATIVE_UINT32,
                                    space,
                                    H5P_DEFAULT,
                                    H5P_DEFAULT,
                                    H5P_DEFAULT);
            H5Dwrite(dset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);
            H5Dclose(dset);
            H5Sclose(space);
        };
        writeU32("nSite", nSite);
        writeU32("nUT", nUT);
        writeU32("nSector", nSectorPerSite);

        // topology/cellParams: compound dataset with cid, siteId,
        // loc[3], antPanelIdx, antPanelOrientation[3].
        if (!cells.empty())
        {
            struct CellRec
            {
                uint32_t cid;
                uint32_t siteId;
                float loc[3];
                uint32_t antPanelIdx;
                float antPanelOrientation[3];
            };
            hid_t arr3 = H5Tarray_create(H5T_NATIVE_FLOAT, 1, (const hsize_t[]){3});
            hid_t cellType = H5Tcreate(H5T_COMPOUND, sizeof(CellRec));
            H5Tinsert(cellType, "cid", offsetof(CellRec, cid), H5T_NATIVE_UINT32);
            H5Tinsert(cellType, "siteId", offsetof(CellRec, siteId), H5T_NATIVE_UINT32);
            H5Tinsert(cellType, "loc", offsetof(CellRec, loc), arr3);
            H5Tinsert(cellType,
                      "antPanelIdx",
                      offsetof(CellRec, antPanelIdx),
                      H5T_NATIVE_UINT32);
            H5Tinsert(cellType,
                      "antPanelOrientation",
                      offsetof(CellRec, antPanelOrientation),
                      arr3);

            std::vector<CellRec> rec(cells.size());
            for (size_t i = 0; i < cells.size(); ++i)
            {
                rec[i].cid = cells[i].cid;
                rec[i].siteId = cells[i].siteId;
                rec[i].loc[0] = cells[i].loc[0];
                rec[i].loc[1] = cells[i].loc[1];
                rec[i].loc[2] = cells[i].loc[2];
                rec[i].antPanelIdx = cells[i].antPanelIdx;
                rec[i].antPanelOrientation[0] = cells[i].antPanelOrientation[0];
                rec[i].antPanelOrientation[1] = cells[i].antPanelOrientation[1];
                rec[i].antPanelOrientation[2] = cells[i].antPanelOrientation[2];
            }
            hsize_t dims = rec.size();
            hid_t space = H5Screate_simple(1, &dims, nullptr);
            hid_t dset = H5Dcreate2(topo,
                                    "cellParams",
                                    cellType,
                                    space,
                                    H5P_DEFAULT,
                                    H5P_DEFAULT,
                                    H5P_DEFAULT);
            H5Dwrite(dset, cellType, H5S_ALL, H5S_ALL, H5P_DEFAULT, rec.data());
            H5Dclose(dset);
            H5Sclose(space);
            H5Tclose(cellType);
            H5Tclose(arr3);
        }

        // topology/utParams: group of separate-array datasets. The
        // analyzer reads uid / loc_x / loc_y / loc_z / outdoor_ind /
        // antPanelIdx / velocity_x/y/z / d_2d_in.
        if (!uts.empty())
        {
            hid_t utg = H5Gcreate2(topo, "utParams", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            const size_t n = uts.size();
            std::vector<uint32_t> uid(n), outdoor(n), antIdx(n);
            std::vector<float> lx(n), ly(n), lz(n), vx(n), vy(n), vz(n), d2din(n);
            for (size_t i = 0; i < n; ++i)
            {
                uid[i] = static_cast<uint32_t>(i);
                lx[i] = uts[i].loc.x;
                ly[i] = uts[i].loc.y;
                lz[i] = uts[i].loc.z;
                outdoor[i] = uts[i].outdoor_ind;
                antIdx[i] = 0u;
                vx[i] = 0.0f;
                vy[i] = 0.0f;
                vz[i] = 0.0f;
                d2din[i] = uts[i].d_2d_in;
            }
            auto writeArr = [&](const char* name, hid_t dtype, const void* data) {
                hsize_t dims = n;
                hid_t space = H5Screate_simple(1, &dims, nullptr);
                hid_t dset = H5Dcreate2(utg,
                                        name,
                                        dtype,
                                        space,
                                        H5P_DEFAULT,
                                        H5P_DEFAULT,
                                        H5P_DEFAULT);
                H5Dwrite(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
                H5Dclose(dset);
                H5Sclose(space);
            };
            writeArr("uid", H5T_NATIVE_UINT32, uid.data());
            writeArr("loc_x", H5T_NATIVE_FLOAT, lx.data());
            writeArr("loc_y", H5T_NATIVE_FLOAT, ly.data());
            writeArr("loc_z", H5T_NATIVE_FLOAT, lz.data());
            writeArr("outdoor_ind", H5T_NATIVE_UINT32, outdoor.data());
            writeArr("antPanelIdx", H5T_NATIVE_UINT32, antIdx.data());
            writeArr("velocity_x", H5T_NATIVE_FLOAT, vx.data());
            writeArr("velocity_y", H5T_NATIVE_FLOAT, vy.data());
            writeArr("velocity_z", H5T_NATIVE_FLOAT, vz.data());
            writeArr("d_2d_in", H5T_NATIVE_FLOAT, d2din.data());
            H5Gclose(utg);
        }
        H5Gclose(topo);
    }

    // systemLevelConfig compound: the analyzer reads scenario + isd +
    // n_site etc here. Minimal-but-sufficient set.
    {
        struct SysRec
        {
            uint32_t scenario;
            float isd;
            uint32_t n_site;
            uint32_t n_sector_per_site;
            uint32_t n_ut;
            uint32_t optional_pl_ind;
            uint32_t o2i_building_penetr_loss_ind;
            uint32_t o2i_car_penetr_loss_ind;
            uint32_t enable_near_field_effect;
            uint32_t enable_non_stationarity;
            float force_los_prob[2];
            float force_ut_speed[3];
            float force_indoor_ratio;
            uint32_t disable_pl_shadowing;
            uint32_t disable_small_scale_fading;
            uint32_t enable_per_tti_lsp;
            uint32_t enable_propagation_delay;
        };
        SysRec rec{};
        rec.scenario = gpuScenario;
        rec.isd = 0.0f;
        rec.n_site = nSite;
        rec.n_sector_per_site = nSectorPerSite;
        rec.n_ut = nUT;
        rec.force_los_prob[0] = -1.0f;
        rec.force_los_prob[1] = -1.0f;
        rec.enable_propagation_delay = 1u;

        hid_t arr2 = H5Tarray_create(H5T_NATIVE_FLOAT, 1, (const hsize_t[]){2});
        hid_t arr3 = H5Tarray_create(H5T_NATIVE_FLOAT, 1, (const hsize_t[]){3});
        hid_t sysType = H5Tcreate(H5T_COMPOUND, sizeof(SysRec));
        H5Tinsert(sysType, "scenario", offsetof(SysRec, scenario), H5T_NATIVE_UINT32);
        H5Tinsert(sysType, "isd", offsetof(SysRec, isd), H5T_NATIVE_FLOAT);
        H5Tinsert(sysType, "n_site", offsetof(SysRec, n_site), H5T_NATIVE_UINT32);
        H5Tinsert(sysType,
                  "n_sector_per_site",
                  offsetof(SysRec, n_sector_per_site),
                  H5T_NATIVE_UINT32);
        H5Tinsert(sysType, "n_ut", offsetof(SysRec, n_ut), H5T_NATIVE_UINT32);
        H5Tinsert(sysType,
                  "optional_pl_ind",
                  offsetof(SysRec, optional_pl_ind),
                  H5T_NATIVE_UINT32);
        H5Tinsert(sysType,
                  "o2i_building_penetr_loss_ind",
                  offsetof(SysRec, o2i_building_penetr_loss_ind),
                  H5T_NATIVE_UINT32);
        H5Tinsert(sysType,
                  "o2i_car_penetr_loss_ind",
                  offsetof(SysRec, o2i_car_penetr_loss_ind),
                  H5T_NATIVE_UINT32);
        H5Tinsert(sysType,
                  "enable_near_field_effect",
                  offsetof(SysRec, enable_near_field_effect),
                  H5T_NATIVE_UINT32);
        H5Tinsert(sysType,
                  "enable_non_stationarity",
                  offsetof(SysRec, enable_non_stationarity),
                  H5T_NATIVE_UINT32);
        H5Tinsert(sysType, "force_los_prob", offsetof(SysRec, force_los_prob), arr2);
        H5Tinsert(sysType, "force_ut_speed", offsetof(SysRec, force_ut_speed), arr3);
        H5Tinsert(sysType,
                  "force_indoor_ratio",
                  offsetof(SysRec, force_indoor_ratio),
                  H5T_NATIVE_FLOAT);
        H5Tinsert(sysType,
                  "disable_pl_shadowing",
                  offsetof(SysRec, disable_pl_shadowing),
                  H5T_NATIVE_UINT32);
        H5Tinsert(sysType,
                  "disable_small_scale_fading",
                  offsetof(SysRec, disable_small_scale_fading),
                  H5T_NATIVE_UINT32);
        H5Tinsert(sysType,
                  "enable_per_tti_lsp",
                  offsetof(SysRec, enable_per_tti_lsp),
                  H5T_NATIVE_UINT32);
        H5Tinsert(sysType,
                  "enable_propagation_delay",
                  offsetof(SysRec, enable_propagation_delay),
                  H5T_NATIVE_UINT32);

        hsize_t one = 1;
        hid_t space = H5Screate_simple(1, &one, nullptr);
        hid_t dset = H5Dcreate2(file,
                                "systemLevelConfig",
                                sysType,
                                space,
                                H5P_DEFAULT,
                                H5P_DEFAULT,
                                H5P_DEFAULT);
        H5Dwrite(dset, sysType, H5S_ALL, H5S_ALL, H5P_DEFAULT, &rec);
        H5Dclose(dset);
        H5Sclose(space);
        H5Tclose(sysType);
        H5Tclose(arr2);
        H5Tclose(arr3);
    }

    // simConfig compound with at least center_freq_hz, so the analyzer
    // knows the scenario frequency.
    {
        struct SimRec
        {
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
        SimRec r{};
        r.center_freq_hz = centerFreqHz;
        hid_t simType = H5Tcreate(H5T_COMPOUND, sizeof(SimRec));
        H5Tinsert(simType,
                  "link_sim_ind",
                  offsetof(SimRec, link_sim_ind),
                  H5T_NATIVE_UINT32);
        H5Tinsert(simType,
                  "center_freq_hz",
                  offsetof(SimRec, center_freq_hz),
                  H5T_NATIVE_FLOAT);
        H5Tinsert(simType,
                  "bandwidth_hz",
                  offsetof(SimRec, bandwidth_hz),
                  H5T_NATIVE_FLOAT);
        H5Tinsert(simType,
                  "sc_spacing_hz",
                  offsetof(SimRec, sc_spacing_hz),
                  H5T_NATIVE_FLOAT);
        H5Tinsert(simType, "fft_size", offsetof(SimRec, fft_size), H5T_NATIVE_UINT32);
        H5Tinsert(simType, "n_prb", offsetof(SimRec, n_prb), H5T_NATIVE_UINT32);
        H5Tinsert(simType, "n_prbg", offsetof(SimRec, n_prbg), H5T_NATIVE_UINT32);
        H5Tinsert(simType,
                  "n_snapshot_per_slot",
                  offsetof(SimRec, n_snapshot_per_slot),
                  H5T_NATIVE_UINT32);
        H5Tinsert(simType, "run_mode", offsetof(SimRec, run_mode), H5T_NATIVE_UINT32);
        H5Tinsert(simType,
                  "internal_memory_mode",
                  offsetof(SimRec, internal_memory_mode),
                  H5T_NATIVE_UINT32);
        H5Tinsert(simType,
                  "freq_convert_type",
                  offsetof(SimRec, freq_convert_type),
                  H5T_NATIVE_UINT32);
        H5Tinsert(simType,
                  "sc_sampling",
                  offsetof(SimRec, sc_sampling),
                  H5T_NATIVE_UINT32);
        H5Tinsert(simType,
                  "proc_sig_freq",
                  offsetof(SimRec, proc_sig_freq),
                  H5T_NATIVE_UINT32);
        hsize_t one = 1;
        hid_t space = H5Screate_simple(1, &one, nullptr);
        hid_t dset = H5Dcreate2(file,
                                "simConfig",
                                simType,
                                space,
                                H5P_DEFAULT,
                                H5P_DEFAULT,
                                H5P_DEFAULT);
        H5Dwrite(dset, simType, H5S_ALL, H5S_ALL, H5P_DEFAULT, &r);
        H5Dclose(dset);
        H5Sclose(space);
        H5Tclose(simType);
    }

    H5Fclose(file);
    SLS_COUT << "Wrote LSP-only HDF5: " << filename << " (" << nLinks << " links)" << std::endl;
}

} // namespace

void
SlsChanWgpu::saveSlsChanToHdf5(const std::string& filename, const SceneMeta& meta)
{
    // Pull outputs from the GPU buffers. Each readback method returns
    // an empty vector if its source buffer wasn't populated (e.g. the
    // ns-3 LSP-only path never runs calClusterRay / generateCIR).
    std::vector<LinkParams> links;
    if (linkParamsBuf_)
    {
        links = readLinkParams(nSite_, nUTCache_);
    }

    // If the small-scale path didn't run, drop down to the lightweight
    // writer: it just emits linkParams + activeLinkParams (synthesised)
    // + topology, which is everything `minimal_analyzer.py` /
    // `analysis_channel_stats.py` need to compute the outdoor
    // coupling-loss / SIR / SINR calibration.
    if (activeLinksCache_.empty() || ssNUeAnt_ == 0 || ssNBsAnt_ == 0)
    {
        // gpuScenario isn't directly cached; default to UMa(0). Callers
        // that care can call setSystemLevelConfig before saving and the
        // simConfig will reflect the right centerFreqHz at least.
        writeLspOnlyHdf5(filename,
                         links,
                         cellsCache_,
                         utsCache_,
                         nSite_,
                         nUTCache_,
                         nSectorPerSiteCache_,
                         /*gpuScenario=*/0u,
                         centerFreqHzCache_);
        (void)meta; // bandwidth/isd/etc. carried in SceneMeta are
                    // unused by the LSP-only writer for now.
        return;
    }

    std::vector<ClusterParamsGpu> clusterParams;
    if (clusterParamsBuf_)
    {
        clusterParams = readClusterParams(nSite_, nUTCache_);
    }

    std::vector<uint32_t> cirNtaps;
    if (cirNtapsBuf_)
    {
        cirNtaps = readCirNtaps();
    }

    std::vector<float> xprVec, phiNmAoAVec, phiNmAoDVec, thetaNmZOAVec, thetaNmZODVec;
    if (clusterOutputsBuf_)
    {
        xprVec = readXpr();
        phiNmAoAVec = readPhiNmAoA();
        phiNmAoDVec = readPhiNmAoD();
        thetaNmZOAVec = readThetaNmZOA();
        thetaNmZODVec = readThetaNmZOD();
    }

    // CIR coefficients and CFR are typically the largest payloads. We
    // skip the readback when the buffers aren't there (LSP-only path)
    // or when the caller hasn't built activeLinks. The caller is
    // expected to call generateCIR / generateCFR first if they want
    // these populated.
    std::vector<std::complex<float>> cirCoe;
    std::vector<std::complex<float>> cfrPrbg;
    const std::vector<ActiveLink>& activeLinks = activeLinksCache_;
    if (cirCoeBuf_ && !activeLinks.empty() && ssNUeAnt_ > 0 && ssNBsAnt_ > 0)
    {
        cirCoe = readCirCoe(static_cast<uint32_t>(activeLinks.size()),
                            nSnapshotsCache_,
                            ssNUeAnt_,
                            ssNBsAnt_);
    }
    if (!m_cfrBatchedResult_.empty())
    {
        cfrPrbg = m_cfrBatchedResult_;
    }

    // 24-tap normalised-delay scratch vector. The kernel produces a
    // monotonically-increasing 0..NMAXTAPS-1 ramp, so we recreate it
    // host-side rather than reading back.
    constexpr uint32_t kNmaxTaps = 24;
    std::vector<uint32_t> cirNormDelay(kNmaxTaps);
    for (uint32_t i = 0; i < kNmaxTaps; ++i)
    {
        cirNormDelay[i] = i;
    }

    writeSlsChanHdf5File(filename,
                         links,
                         nSite_,
                         nUTCache_,
                         clusterParams,
                         activeLinks,
                         cirCoe,
                         cirNormDelay,
                         cirNtaps,
                         cfrPrbg,
                         ssNPrbg_,
                         xprVec,
                         phiNmAoAVec,
                         phiNmAoDVec,
                         thetaNmZOAVec,
                         thetaNmZODVec,
                         scSpacingHzCache_,
                         fftSizeCache_,
                         nPrbCache_,
                         nSnapshotPerSlotCache_,
                         centerFreqHzCache_,
                         meta.bandwidthHz,
                         ssNUeAnt_,
                         ssNBsAnt_,
                         ssCmnCache_,
                         cellsCache_,
                         cellsSSCache_,
                         utsCache_,
                         meta.isd,
                         meta.bsHeight,
                         meta.minBsUeDist2d,
                         meta.maxBsUeDist2dIndoor,
                         meta.indoorUtPercent,
                         nSectorPerSiteCache_);
}

namespace
{

static void
writeSlsChanHdf5File(const std::string& filename,
                     const std::vector<LinkParams>& links,
                     uint32_t nSite,
                     uint32_t nUT,
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
                     float scSpacingHz,
                     uint32_t fftSize,
                     uint32_t nPrb,
                     uint32_t nSnapshotPerSlot,
                     float centerFreqHz,
                     float bandwidthHz,
                     uint32_t nUeAnt,
                     uint32_t nBsAnt,
                     const SsCmnParams& ssCmn,
                     const std::vector<CellParam>& cells,
                     const std::vector<CellParamSS>& cellsSS,
                     const std::vector<UtParam>& uts,
                     float isd,
                     float bsHeight,
                     float minBsUeDist2d,
                     float maxBsUeDist2dIndoor,
                     float indoorUtPercent,
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
        SLS_CERR << "saveSlsChanToHdf5: " << filename
                 << " already exists — close it in any external program and retry" << std::endl;
        return;
    }

    hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0)
    {
        SLS_CERR << "saveSlsChanToHdf5: failed to create " << filename << std::endl;
        return;
    }

    // ── simConfig (compound dataset for Python compatibility) ──
    {
        struct SimConfigRecord
        {
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
        H5Tinsert(compoundType, "link_sim_ind", 0 * 4, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "center_freq_hz", 1 * 4, H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "bandwidth_hz", 2 * 4, H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "sc_spacing_hz", 3 * 4, H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "fft_size", 4 * 4, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "n_prb", 5 * 4, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "n_prbg", 6 * 4, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "n_snapshot_per_slot", 7 * 4, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "run_mode", 8 * 4, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "internal_memory_mode", 9 * 4, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "freq_convert_type", 10 * 4, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "sc_sampling", 11 * 4, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "proc_sig_freq", 12 * 4, H5T_NATIVE_UINT32);

        SimConfigRecord sc{};
        sc.link_sim_ind = 0; // not available from caller
        sc.center_freq_hz = centerFreqHz;
        sc.bandwidth_hz = bandwidthHz;
        sc.sc_spacing_hz = scSpacingHz;
        sc.fft_size = fftSize;
        sc.n_prb = nPrb;
        sc.n_prbg = nPrbg;
        sc.n_snapshot_per_slot = nSnapshotPerSlot;
        sc.run_mode = 0;             // not available
        sc.internal_memory_mode = 0; // not available
        sc.freq_convert_type = 0;    // not available
        sc.sc_sampling = 0;          // not available
        sc.proc_sig_freq = 0;        // not available

        hsize_t dims = 1;
        hid_t dataspace = H5Screate_simple(1, &dims, nullptr);
        hid_t dataset = H5Dcreate2(file,
                                   "simConfig",
                                   compoundType,
                                   dataspace,
                                   H5P_DEFAULT,
                                   H5P_DEFAULT,
                                   H5P_DEFAULT);
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

        H5Tinsert(compoundType, "scenario", 0, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "isd", 4, H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "n_site", 8, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "n_sector_per_site", 12, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "n_ut", 16, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "optional_pl_ind", 20, H5T_NATIVE_INT32);
        H5Tinsert(compoundType, "o2i_building_penetr_loss_ind", 24, H5T_NATIVE_INT32);
        H5Tinsert(compoundType, "o2i_car_penetr_loss_ind", 28, H5T_NATIVE_INT32);
        H5Tinsert(compoundType, "enable_near_field_effect", 32, H5T_NATIVE_INT32);
        H5Tinsert(compoundType, "enable_non_stationarity", 36, H5T_NATIVE_INT32);
        H5Tinsert(compoundType, "force_los_prob", 40, arrType1f);
        H5Tinsert(compoundType, "force_ut_speed", 44, arrType1f);
        H5Tinsert(compoundType, "force_indoor_ratio", 48, H5T_NATIVE_FLOAT);
        H5Tinsert(compoundType, "disable_pl_shadowing", 52, H5T_NATIVE_INT32);
        H5Tinsert(compoundType, "disable_small_scale_fading", 56, H5T_NATIVE_INT32);
        H5Tinsert(compoundType, "enable_per_tti_lsp", 60, H5T_NATIVE_INT32);
        H5Tinsert(compoundType, "enable_propagation_delay", 64, H5T_NATIVE_INT32);
        H5Tinsert(compoundType, "ut_drop_option", 68, H5T_NATIVE_INT32);

        SystemLevelConfigRecord slc{};
        slc.scenario = 0; // UMa
        slc.isd = isd;
        slc.n_site = nSite;
        slc.n_sector_per_site = nSectorPerSite;
        slc.n_ut = nUT;
        slc.optional_pl_ind = 0;
        slc.o2i_building_penetr_loss_ind = 0;
        slc.o2i_car_penetr_loss_ind = 0;
        slc.enable_near_field_effect = 0;
        slc.enable_non_stationarity = 0;
        slc.force_los_prob[0] = 0.0f;
        slc.force_ut_speed[0] = 0.0f;
        slc.force_indoor_ratio = indoorUtPercent / 100.0f;
        slc.disable_pl_shadowing = 0;
        slc.disable_small_scale_fading = 0;
        slc.enable_per_tti_lsp = 0;
        slc.enable_propagation_delay = 0;
        slc.ut_drop_option = 0;

        hsize_t dims = 1;
        hid_t dataspace = H5Screate_simple(1, &dims, nullptr);
        hid_t dataset = H5Dcreate2(file,
                                   "systemLevelConfig",
                                   compoundType,
                                   dataspace,
                                   H5P_DEFAULT,
                                   H5P_DEFAULT,
                                   H5P_DEFAULT);
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
        const bool haveActiveLinks = activeLinks.size() >= nLinks;
        for (uint32_t i = 0; i < nLinks; ++i)
        {
            const LinkParams& lk = links[i];
            // LSP-only callers (e.g. the ns-3 batch path) don't populate
            // activeLinks. Fall back to deriving the serving cell ID
            // from the link's site-major index: cid = i / nUT.
            hdf5Links[i].cid = haveActiveLinks
                                   ? activeLinks[i].cid
                                   : (nUT > 0 ? i / nUT : 0u);
            hdf5Links[i].d2d = lk.d2d;
            hdf5Links[i].d2d_in = lk.d2d_in;
            hdf5Links[i].d2d_out = lk.d2d_out;
            hdf5Links[i].d3d = lk.d3d;
            hdf5Links[i].d3d_in = lk.d3d_in;
            hdf5Links[i].d3d_out = lk.d3d_out;
            hdf5Links[i].phi_LOS_AOD = lk.phi_LOS_AOD;
            hdf5Links[i].theta_LOS_ZOD = lk.theta_LOS_ZOD;
            hdf5Links[i].phi_LOS_AOA = lk.phi_LOS_AOA;
            hdf5Links[i].theta_LOS_ZOA = lk.theta_LOS_ZOA;
            hdf5Links[i].losInd = lk.losInd;
            hdf5Links[i].pathloss = lk.pathloss;
            hdf5Links[i].SF = lk.SF;
            hdf5Links[i].K = lk.K;
            hdf5Links[i].DS = lk.DS;
            hdf5Links[i].ASD = lk.ASD;
            hdf5Links[i].ASA = lk.ASA;
            hdf5Links[i].ZSD = lk.ZSD;
            hdf5Links[i].ZSA = lk.ZSA;
            hdf5Links[i].mu_lgZSD = lk.mu_lgZSD;
            hdf5Links[i].sigma_lgZSD = lk.sigma_lgZSD;
            hdf5Links[i].mu_offset_ZOD = lk.mu_offset_ZOD;
            hdf5Links[i].delta_tau = lk._pad; // delta_tau was 0 in the original code
        }

        hsize_t dims = static_cast<hsize_t>(nLinks);
        hid_t dataspace = H5Screate_simple(1, &dims, nullptr);
        hid_t dataset = H5Dcreate2(file,
                                   "linkParams",
                                   compoundType,
                                   dataspace,
                                   H5P_DEFAULT,
                                   H5P_DEFAULT,
                                   H5P_DEFAULT);
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
        hid_t dataset = H5Dcreate2(file,
                                   "clusterParams",
                                   compoundType,
                                   dataspace,
                                   H5P_DEFAULT,
                                   H5P_DEFAULT,
                                   H5P_DEFAULT);
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
            hid_t dsetId =
                H5Dcreate(file, "xpr", dset, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dsetId, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, xpr.data());
            H5Dclose(dsetId);
        }
        // phi_n_m_AoA
        {
            hid_t dsetId =
                H5Dcreate(file, "phi_n_m_AoA", dset, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dsetId, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, phiNmAoA.data());
            H5Dclose(dsetId);
        }
        // phi_n_m_AoD
        {
            hid_t dsetId =
                H5Dcreate(file, "phi_n_m_AoD", dset, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dsetId, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, phiNmAoD.data());
            H5Dclose(dsetId);
        }
        // theta_n_m_ZOA
        {
            hid_t dsetId = H5Dcreate(file,
                                     "theta_n_m_ZOA",
                                     dset,
                                     space,
                                     H5P_DEFAULT,
                                     H5P_DEFAULT,
                                     H5P_DEFAULT);
            H5Dwrite(dsetId, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, thetaNmZOA.data());
            H5Dclose(dsetId);
        }
        // theta_n_m_ZOD
        {
            hid_t dsetId = H5Dcreate(file,
                                     "theta_n_m_ZOD",
                                     dset,
                                     space,
                                     H5P_DEFAULT,
                                     H5P_DEFAULT,
                                     H5P_DEFAULT);
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

        struct ActiveLinkRecord
        {
            uint32_t cid;
            uint32_t uid;
            uint32_t linkIdx;
            uint32_t lspReadIdx;
        };

        std::vector<ActiveLinkRecord> activeLinkRecords(nLinks);
        const bool haveActiveLinks2 = activeLinks.size() >= nLinks;
        for (uint32_t i = 0; i < nLinks; ++i)
        {
            if (haveActiveLinks2)
            {
                activeLinkRecords[i].cid = activeLinks[i].cid;
                activeLinkRecords[i].uid = activeLinks[i].uid;
                activeLinkRecords[i].linkIdx = activeLinks[i].linkIdx;
                activeLinkRecords[i].lspReadIdx = activeLinks[i].lspReadIdx;
            }
            else
            {
                // Synthesise records from the site-major flat layout
                // when no GPU CIR pipeline was run.
                activeLinkRecords[i].cid = (nUT > 0) ? i / nUT : 0u;
                activeLinkRecords[i].uid = (nUT > 0) ? i % nUT : 0u;
                activeLinkRecords[i].linkIdx = i;
                activeLinkRecords[i].lspReadIdx = i;
            }
        }

        hsize_t dims = nLinks;

        hid_t dataspace = H5Screate_simple(1, &dims, &dims);
        hid_t dataset = H5Dcreate2(file,
                                   "activeLinkParams",
                                   compoundType,
                                   dataspace,
                                   H5P_DEFAULT,
                                   H5P_DEFAULT,
                                   H5P_DEFAULT);
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
    //   cirPerCell/cirCoe_cell{cell_id}  -> complex dtype shape (nActiveUt, nSnapshots, nUtAnt,
    //   nBsAnt, nTaps) cirPerCell/cirNtaps_cell{cell_id} -> uint32 shape (nActiveUt,)
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
        std::vector<std::complex<float>> cellCir(static_cast<size_t>(nActiveUtForCell) *
                                                 nSnapshots * nBsAntN * nUeAntN * NMAXTAPS);

        // Scatter from flat cirCoe buffer into per-cell buffer
        // Flat layout: [link, snap, bs_ant, ue_ant, tap] -> [link * nSnapshots * nBsAnt * nUeAnt *
        // NMAXTAPS
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
                            const size_t flatIdx =
                                cirOffset +
                                static_cast<size_t>(snap) * nBsAntN * nUeAntN * NMAXTAPS +
                                static_cast<size_t>(bs_ant) * nUeAntN * NMAXTAPS +
                                static_cast<size_t>(ue_ant) * NMAXTAPS + tap;
                            cellCir[(ueInCell * nSnapshots + snap) * nTapsPerLink +
                                    bs_ant * nUeAntN * NMAXTAPS + ue_ant * NMAXTAPS + tap] =
                                cirCoe[flatIdx];
                        }
                    }
                }
            }
        }

        // Create dataset name
        std::string cirDatasetName = "cirCoe_cell" + std::to_string(cellId);

        // Write CIR data
        hid_t cirDataset = H5Dcreate2(grpCir,
                                      cirDatasetName.c_str(),
                                      complexType,
                                      cirDataspace,
                                      H5P_DEFAULT,
                                      H5P_DEFAULT,
                                      H5P_DEFAULT);
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
            hid_t ntapsDataset = H5Dcreate2(grpCir,
                                            ntapsDatasetName.c_str(),
                                            H5T_NATIVE_UINT32,
                                            ntapsDataspace,
                                            H5P_DEFAULT,
                                            H5P_DEFAULT,
                                            H5P_DEFAULT);
            H5Dwrite(ntapsDataset,
                     H5T_NATIVE_UINT32,
                     H5S_ALL,
                     H5S_ALL,
                     H5P_DEFAULT,
                     cellNtaps.data());
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
            hid_t mappingDataset = H5Dcreate2(grpCir,
                                              mappingName.c_str(),
                                              H5T_NATIVE_UINT16,
                                              mappingDataspace,
                                              H5P_DEFAULT,
                                              H5P_DEFAULT,
                                              H5P_DEFAULT);
            H5Dwrite(mappingDataset,
                     H5T_NATIVE_UINT16,
                     H5S_ALL,
                     H5S_ALL,
                     H5P_DEFAULT,
                     ueMapping.data());
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
            float loc[3]; // [x, y, z]
            uint32_t antPanelIdx;
            float antPanelOrientation[3]; // [theta_tilt, phi_tilt, zeta_offset]
        };

        // Build compound type
        hsize_t dim3 = 3;
        hid_t locType = H5Tarray_create2(H5T_NATIVE_FLOAT, 1, &dim3);
        hid_t antOrType = H5Tarray_create2(H5T_NATIVE_FLOAT, 1, &dim3);

        hid_t compoundType = H5Tcreate(H5T_COMPOUND, sizeof(CellParamsRecord));
        H5Tinsert(compoundType, "cid", 0, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "siteId", 4, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "loc", 8, locType);
        H5Tinsert(compoundType, "antPanelIdx", 20, H5T_NATIVE_UINT32);
        H5Tinsert(compoundType, "antPanelOrientation", 24, antOrType);

        std::vector<CellParamsRecord> records(nCells);
        for (uint32_t i = 0; i < nCells; ++i)
        {
            records[i].cid = i;
            records[i].siteId = i / nSectorPerSite;
            records[i].loc[0] = cells[i].loc[0];
            records[i].loc[1] = cells[i].loc[1];
            records[i].loc[2] = cells[i].loc[2];
            records[i].antPanelIdx = cellsSS[i].antPanelIdx;
            records[i].antPanelOrientation[0] = cellsSS[i].antPanelOrientation[0];
            records[i].antPanelOrientation[1] = cellsSS[i].antPanelOrientation[1];
            records[i].antPanelOrientation[2] = cellsSS[i].antPanelOrientation[2];
        }

        hsize_t dims = static_cast<hsize_t>(nCells);
        hid_t dataspace = H5Screate_simple(1, &dims, &dims);
        hid_t dataset = H5Dcreate2(file,
                                   "cellParams",
                                   compoundType,
                                   dataspace,
                                   H5P_DEFAULT,
                                   H5P_DEFAULT,
                                   H5P_DEFAULT);
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
        cellParamsFlat[base + 0] = cells[i].loc[0];
        cellParamsFlat[base + 1] = cells[i].loc[1];
        cellParamsFlat[base + 2] = cells[i].loc[2];
        cellParamsFlat[base + 3] = 0.0f;
    }
    writeDatasetFloat(grpTopo, "cell_params", cellParamsFlat.data(), nCells * cellFloatsPer);

    std::vector<float> siteParamsFlat(nSite * 7, 0.0f);
    for (uint32_t i = 0; i < nSite; ++i)
    {
        uint32_t base = i * 7;
        siteParamsFlat[base + 0] = cells[i * nSectorPerSite].loc[0];
        siteParamsFlat[base + 1] = cells[i * nSectorPerSite].loc[1];
        siteParamsFlat[base + 2] = cells[i * nSectorPerSite].loc[2];
        siteParamsFlat[base + 3] = 0.0f;
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
    SLS_COUT << "Wrote HDF5: " << filename << " (" << nLinks << " links, " << nLinks
             << " active)\n";
}

} // namespace
#endif // SLS_CHAN_HDF5
