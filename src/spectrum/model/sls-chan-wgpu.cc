// This is an AI translated version of the original Nvidia source for GPU compute,
// but ported to WGPU for cross-platform support
// https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/main/testBenches/chanModels/src/sls_chan_src/sls_chan_large_scale_GPU.cu

#define WEBGPU_CPP_IMPLEMENTATION
#include "sls-chan-wgpu.h"

#include <cassert>
#include <cstring>
#include <fstream>

static std::string
readFile(const char* path)
{
    std::ifstream f(path);
    return {std::istreambuf_iterator<char>(f), {}};
}

// ── Constructor ────────────────────────────────────────────────────────────
SlsChanWgpu::SlsChanWgpu(wgpu::Device device)
    : device_(device),
      queue_(device.getQueue())
{
    // ── Fix 1: ShaderModuleWGSLDescriptor → WGPUShaderSourceWGSL (C API) ──
    std::string wgsl = readFile("/home/gabriel/ns-3-maintainer/src/spectrum/model/sls-chan.wgsl");

    WGPUShaderSourceWGSL wgslSource{};
    wgslSource.chain.next = nullptr;
    wgslSource.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSource.code = sv(wgsl.c_str()); // WGPUStringView

    WGPUShaderModuleDescriptor smDescC{};
    smDescC.nextInChain = &wgslSource.chain;
    // Create via C API then wrap — the C++ wrapper's createShaderModule
    // still accepts a WGPUShaderModuleDescriptor pointer under the hood
    shader_ = wgpu::ShaderModule(wgpuDeviceCreateShaderModule(device_, &smDescC));
    assert(shader_ && "WGSL failed to compile — check entry point names and file path");

    // ── Fix 2: entryPoint is WGPUStringView, not const char* ──────────────
    auto makePipeline = [&](const char* ep) -> wgpu::ComputePipeline {
        wgpu::ComputePipelineDescriptor desc{};
        desc.compute.module = shader_;
        desc.compute.entryPoint = sv(ep); // ← WGPUStringView, not const char*
        return device_.createComputePipeline(desc);
    };
    linkParamPipeline_ = makePipeline("cal_link_param_kernel");
    assert(linkParamPipeline_ && "missing cal_link_param_kernel in WGSL");

    crnFillPipeline_ = makePipeline("fill_crn_kernel");
    assert(crnFillPipeline_ && "missing fill_crn_kernel in WGSL");

    crnConvPipeline_ = makePipeline("convolve_crn_kernel");
    assert(crnConvPipeline_ && "missing convolve_crn_kernel in WGSL");

    crnNormPipeline_ = makePipeline("normalize_crn_kernel");
    assert(crnNormPipeline_ && "missing normalize_crn_kernel in WGSL");
}

// ── Buffer helper ──────────────────────────────────────────────────────────
wgpu::Buffer
SlsChanWgpu::makeBuffer(uint64_t size, wgpu::BufferUsage usage, const void* data)
{
    wgpu::BufferDescriptor desc{};
    desc.size = (size + 3) & ~uint64_t(3);
    desc.usage = usage | WGPUBufferUsage_CopyDst;
    desc.mappedAtCreation = (data != nullptr);
    wgpu::Buffer buf = device_.createBuffer(desc);
    if (data)
    {
        std::memcpy(buf.getMappedRange(0, size), const_cast<void*>(data), size);
        buf.unmap();
    }
    return buf;
}

// ── Fix 3: poll takes (Bool, SubmissionIndex*) in v24 ─────────────────────
void
SlsChanWgpu::waitIdle()
{
    device_.poll(true, nullptr); // Bool wait=true, no submission index filter
}

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

// ── Fix 7: mapAsync needs BufferMapCallbackInfo with function ptr + userdata
std::vector<LinkParams>
SlsChanWgpu::readLinkParams(uint32_t nSite, uint32_t nUT)
{
    uint64_t sz = (uint64_t)nSite * nUT * sizeof(LinkParams);
    std::vector<LinkParams> result(nSite * nUT);

    // In wgpu-native v24, the callback is:
    //   void cb(WGPUMapAsyncStatus, WGPUStringView msg, void* ud1, void* ud2)
    struct MapCtx
    {
        bool done = false;
    };

    MapCtx ctx;

    wgpu::BufferMapCallbackInfo cbInfo = wgpu::Default;
    cbInfo.mode = wgpu::CallbackMode::AllowProcessEvents;
    cbInfo.callback =
        [](WGPUMapAsyncStatus status, WGPUStringView /*msg*/, void* ud1, void* /*ud2*/) {
            static_cast<MapCtx*>(ud1)->done = true;
        };
    cbInfo.userdata1 = &ctx;

    stagingBuf_.mapAsync(wgpu::MapMode::Read, 0, sz, cbInfo);

    // Tick until the mapping completes
    while (!ctx.done)
    {
        device_.poll(false, nullptr); // non-blocking tick
    }

    const void* data = stagingBuf_.getMappedRange(0, sz);
    std::memcpy(result.data(), data, sz);
    stagingBuf_.unmap();
    return result;
}

void
SlsChanWgpu::generateCRN(float maxX,
                         float minX,
                         float maxY,
                         float minY,
                         const float corrLos[7],
                         const float corrNlos[6],
                         const float corrO2i[6])
{
    const float kStep = 10.0f;
    const int32_t nX = std::abs(int32_t((maxX - minX) / kStep)) + 1;
    const int32_t nY = std::abs(int32_t((maxY - minY) / kStep)) + 1;

    nX_ = nX;
    nY_ = nY; // store for calLinkParam

    const uint64_t gridSz = uint64_t(nX) * nY;

    const uint64_t losBufSz = uint64_t(nSite_) * 7 * gridSz * sizeof(float);
    const uint64_t nlosBufSz = uint64_t(nSite_) * 6 * gridSz * sizeof(float);
    const uint64_t o2iBufSz = uint64_t(nSite_) * 6 * gridSz * sizeof(float);

    // ── Helper: mirrors WGSL padded-size formula exactly ─────────────────────
    // WGSL: corr_px = corrDist/step (0 when corrDist==0)
    //        iD = u32(3*corr_px)    L = corr_px>0 ? 2*iD+1 : 1
    //        padded_NX = nX + L - 1
    auto tempBufBytes = [&](float cd) -> uint64_t {
        const float cp = (cd == 0.0f) ? 0.0f : (cd / kStep);
        const uint32_t iD = static_cast<uint32_t>(3.0f * cp);
        const uint32_t L = (cp == 0.0f) ? 1u : (2u * iD + 1u);
        const uint64_t pnx = static_cast<uint64_t>(nX) + (L - 1u);
        const uint64_t pny = static_cast<uint64_t>(nY) + (L - 1u);
        return pnx * pny * sizeof(float);
    };

    // Allocate tempBuf for worst-case corrDist across all dispatches
    float maxCorr = 0.0f;
    for (int i = 0; i < 7; i++)
    {
        maxCorr = std::max(maxCorr, corrLos[i]);
    }
    for (int i = 0; i < 6; i++)
    {
        maxCorr = std::max(maxCorr, corrNlos[i]);
    }
    for (int i = 0; i < 6; i++)
    {
        maxCorr = std::max(maxCorr, corrO2i[i]);
    }

    wgpu::Buffer tempBuf =
        makeBuffer(tempBufBytes(maxCorr), WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);

    crnLosBuf_ = makeBuffer(losBufSz, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    crnNlosBuf_ = makeBuffer(nlosBufSz, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    crnO2iBuf_ = makeBuffer(o2iBufSz, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);

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

    auto dispatchGrid = [&](wgpu::Buffer& outputBuf, float corrDist, uint32_t gridIndex) {
        const uint64_t gridBytes = gridSz * sizeof(float);
        const uint64_t destOffset = uint64_t(gridIndex) * gridBytes;

        // Per-dispatch exact temp size — guaranteed <= tempBuf.size
        const uint64_t curTempBytes = tempBufBytes(corrDist);
        const uint64_t rngBytes = uint64_t(nCrnRng) * sizeof(RngState);

        struct CRNGenUniforms
        {
            float maxX, minX, maxY, minY;
            float corrDist;
            uint32_t maxRngStates;
            uint32_t outputGridOffset;
            uint32_t _pad;
            uint32_t nX, nY;
            float step;
            uint32_t _pad2;
        };
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
                              kStep,
                              0u};
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

        // ── Fill bind group: {0=crn_gen_uni, 1=fill_temp, 2=fill_rng} ──────
        auto fillLayout = crnFillPipeline_.getBindGroupLayout(0);
        std::vector<wgpu::BindGroupEntry> fillEntries(3, wgpu::Default);
        fillEntries[0].binding = 0;
        fillEntries[0].buffer = genUniBuf;
        fillEntries[0].size = sizeof(genUni);
        fillEntries[1].binding = 1;
        fillEntries[1].buffer = tempBuf;
        fillEntries[1].size = curTempBytes;
        fillEntries[2].binding = 2;
        fillEntries[2].buffer = crnRngBuf;
        fillEntries[2].size = rngBytes;
        wgpu::BindGroupDescriptor fillBgDesc = wgpu::Default;
        fillBgDesc.layout = fillLayout;
        fillBgDesc.entryCount = 3;
        fillBgDesc.entries = fillEntries.data();
        wgpu::BindGroup fillBg = device_.createBindGroup(fillBgDesc);

        // ── Conv bind group: {0=crn_gen_uni, 1=conv_temp, 3=conv_output} ───
        auto convLayout = crnConvPipeline_.getBindGroupLayout(0);
        std::vector<wgpu::BindGroupEntry> convEntries(3, wgpu::Default);
        convEntries[0].binding = 0;
        convEntries[0].buffer = genUniBuf;
        convEntries[0].size = sizeof(genUni);
        convEntries[1].binding = 1;
        convEntries[1].buffer = tempBuf;
        convEntries[1].size = curTempBytes;
        convEntries[2].binding = 3;
        convEntries[2].buffer = gridBuf;
        convEntries[2].size = gridBytes;
        wgpu::BindGroupDescriptor convBgDesc = wgpu::Default;
        convBgDesc.layout = convLayout;
        convBgDesc.entryCount = 3;
        convBgDesc.entries = convEntries.data();
        wgpu::BindGroup convBg = device_.createBindGroup(convBgDesc);

        // ── Norm bind group: {4=norm_uni, 5=norm_buffer} ───────────────────
        auto normLayout = crnNormPipeline_.getBindGroupLayout(0);
        std::vector<wgpu::BindGroupEntry> normEntries(2, wgpu::Default);
        normEntries[0].binding = 4;
        normEntries[0].buffer = normUniBuf;
        normEntries[0].size = sizeof(normUni);
        normEntries[1].binding = 5;
        normEntries[1].buffer = gridBuf;
        normEntries[1].size = gridBytes;
        wgpu::BindGroupDescriptor normBgDesc = wgpu::Default;
        normBgDesc.layout = normLayout;
        normBgDesc.entryCount = 2;
        normBgDesc.entries = normEntries.data();
        wgpu::BindGroup normBg = device_.createBindGroup(normBgDesc);

        // ── Pass 1: fill crn_temp ─────────────────────────────────────────────
        {
            wgpu::CommandEncoder enc1 = device_.createCommandEncoder(wgpu::Default);
            auto pass = enc1.beginComputePass(wgpu::Default);
            pass.setPipeline(crnFillPipeline_);
            pass.setBindGroup(0u, fillBg, (size_t)0, nullptr);
            pass.dispatchWorkgroups(128u, 1u, 1u);
            pass.end();
            queue_.submit(enc1.finish(wgpu::Default));
            waitIdle();
        }

        // ── Pass 2: convolve + normalize + copy to flat buffer ────────────────
        {
            wgpu::CommandEncoder enc2 = device_.createCommandEncoder(wgpu::Default);
            {
                auto pass = enc2.beginComputePass(wgpu::Default);
                pass.setPipeline(crnConvPipeline_);
                pass.setBindGroup(0u, convBg, (size_t)0, nullptr);
                pass.dispatchWorkgroups(128u, 1u, 1u);
                pass.end();
            }
            {
                auto pass = enc2.beginComputePass(wgpu::Default);
                pass.setPipeline(crnNormPipeline_);
                pass.setBindGroup(0u, normBg, (size_t)0, nullptr);
                pass.dispatchWorkgroups(1u, 1u, 1u);
                pass.end();
            }
            enc2.copyBufferToBuffer(gridBuf, 0, outputBuf, destOffset, gridBytes);
            queue_.submit(enc2.finish(wgpu::Default));
            waitIdle();
        }
    };

    for (uint32_t s = 0; s < nSite_; s++)
    {
        for (uint32_t lsp = 0; lsp < 7; lsp++)
        {
            dispatchGrid(crnLosBuf_, corrLos[lsp], s * 7 + lsp);
        }
    }

    for (uint32_t s = 0; s < nSite_; s++)
    {
        for (uint32_t lsp = 0; lsp < 6; lsp++)
        {
            dispatchGrid(crnNlosBuf_, corrNlos[lsp], s * 6 + lsp);
        }
    }

    for (uint32_t s = 0; s < nSite_; s++)
    {
        for (uint32_t lsp = 0; lsp < 6; lsp++)
        {
            dispatchGrid(crnO2iBuf_, corrO2i[lsp], s * 6 + lsp);
        }
    }
}

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
                          int32_t nX,
                          int32_t nY)
{
    struct LinkParamUniforms
    {
        float maxX, minX, maxY, minY;
        uint32_t nSite, nUT, nSectorPerSite;
        uint32_t updatePL, updateAllLSPs, updateLos;
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
                              nX,
                              nY};
    wgpu::Buffer uniBuf = makeBuffer(sizeof(uniData), WGPUBufferUsage_Uniform, &uniData);

    const uint64_t cellParamsSz = uint64_t(nSite) * sizeof(CellParam);
    const uint64_t utParamsSz = nUT * sizeof(UtParam);
    const uint64_t linkParamsSz = uint64_t(nSite) * nUT * sizeof(LinkParams);
    const uint64_t rngStatesSz = uint64_t(nSite) * nUT * sizeof(RngState);
    const uint64_t crnGridSz = uint64_t(nX) * nY;
    const uint64_t crnLosSz = uint64_t(nSite) * 7 * crnGridSz * sizeof(float);
    const uint64_t crnNlosSz = uint64_t(nSite) * 6 * crnGridSz * sizeof(float);
    const uint64_t crnO2iSz = uint64_t(nSite) * 6 * crnGridSz * sizeof(float);
    const uint64_t crnLosOffSz = uint64_t(nSite) * 7 * sizeof(uint32_t);
    const uint64_t crnNlosOffSz = uint64_t(nSite) * 6 * sizeof(uint32_t);
    const uint64_t crnO2iOffSz = uint64_t(nSite) * 6 * sizeof(uint32_t);
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
        SystemLevelConfigGPU slc{0, 0, 0, 0, -1.0f, -1.0f, {0, 0}};
        sysConfigBuf_ = makeBuffer(sizeof(SystemLevelConfigGPU),
                                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
                                   &slc);
    }
    if (!simConfigBuf_)
    {
        // NOTE: shader expects fc in GHz, not Hz — field name is misleading
        SimConfigGPU sc{3.5e9f, 0, 0, 0};
        simConfigBuf_ = makeBuffer(sizeof(SimConfigGPU),
                                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
                                   &sc);
    }
    if (!cmnLinkBuf_)
    {
        CmnLinkParamsGPU cl{};
        // Identity Cholesky (uncorrelated LSPs — valid for initial testing)
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

        // lsp_idx: 0=LOS, 1=NLOS, 2=O2I
        // 3GPP TR 38.901 Table 7.5-6, UMa, fc=3.5 GHz
        cl.lgfc = std::log10(3.5f);

        // lsp_idx: 0=LOS, 1=NLOS, 2=O2I  ← match WGSL select() logic
        cl.mu_K[0] = 9.0f;
        cl.sigma_K[0] = 3.5f; // LOS
        cl.mu_K[1] = 0.0f;
        cl.sigma_K[1] = 0.0f; // NLOS (no K)
        cl.mu_K[2] = 0.0f;
        cl.sigma_K[2] = 0.0f; // O2I

        cl.mu_lgDS[0] = -7.03f;
        cl.sigma_lgDS[0] = 0.66f; // LOS
        cl.mu_lgDS[1] = -6.44f;
        cl.sigma_lgDS[1] = 0.39f; // NLOS
        cl.mu_lgDS[2] = -6.62f;
        cl.sigma_lgDS[2] = 0.32f; // O2I

        cl.mu_lgASD[0] = 1.15f;
        cl.sigma_lgASD[0] = 0.28f; // LOS
        cl.mu_lgASD[1] = 1.41f;
        cl.sigma_lgASD[1] = 0.28f; // NLOS
        cl.mu_lgASD[2] = 1.25f;
        cl.sigma_lgASD[2] = 0.42f; // O2I

        cl.mu_lgASA[0] = 1.81f;
        cl.sigma_lgASA[0] = 0.20f; // LOS
        cl.mu_lgASA[1] = 1.87f;
        cl.sigma_lgASA[1] = 0.20f; // NLOS
        cl.mu_lgASA[2] = 1.76f;
        cl.sigma_lgASA[2] = 0.16f; // O2I

        cl.mu_lgZSA[0] = 0.95f;
        cl.sigma_lgZSA[0] = 0.16f; // LOS
        cl.mu_lgZSA[1] = 1.26f;
        cl.sigma_lgZSA[1] = 0.35f; // NLOS
        cl.mu_lgZSA[2] = 1.01f;
        cl.sigma_lgZSA[2] = 0.43f; // O2I

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
    if (!crnLosBuf_)
    {
        crnLosBuf_ = makeBuffer(crnLosSz, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    }
    if (!crnNlosBuf_)
    {
        crnNlosBuf_ = makeBuffer(crnNlosSz, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    }
    if (!crnO2iBuf_)
    {
        crnO2iBuf_ = makeBuffer(crnO2iSz, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    }

    std::vector<uint32_t> losOff(nSite * 7), nlosOff(nSite * 6), o2iOff(nSite * 6);
    for (uint32_t i = 0; i < nSite * 7; i++)
    {
        losOff[i] = i * crnGridSz;
    }
    for (uint32_t i = 0; i < nSite * 6; i++)
    {
        nlosOff[i] = i * crnGridSz;
    }
    for (uint32_t i = 0; i < nSite * 6; i++)
    {
        o2iOff[i] = i * crnGridSz;
    }
    if (!crnLosOffBuf_)
    {
        crnLosOffBuf_ = makeBuffer(crnLosOffSz,
                                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
                                   losOff.data());
        ;
    }
    if (!crnNlosOffBuf_)
    {
        crnNlosOffBuf_ = makeBuffer(crnNlosOffSz,
                                    WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
                                    nlosOff.data());
    }
    if (!crnO2iOffBuf_)
    {
        crnO2iOffBuf_ = makeBuffer(crnO2iOffSz,
                                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
                                   o2iOff.data());
    }
    if (!stagingBuf_)
    {
        stagingBuf_ = makeBuffer(linkParamsSz, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    }

    // ── Fix 4: BindGroupEntry — use wgpu::Default + explicit field sets ────
    auto bg0Layout = linkParamPipeline_.getBindGroupLayout(0);

    std::vector<wgpu::BindGroupEntry> entries(14, wgpu::Default);
    auto E = [&](int i, uint32_t binding, wgpu::Buffer buf, uint64_t size = WGPU_WHOLE_SIZE) {
        entries[i].binding = binding;
        entries[i].buffer = buf;
        entries[i].offset = 0;
        entries[i].size = size;
        if (!buf)
        {
            std::cerr << "null buffer: " << i << std::endl;
            abort();
        }
    };
    E(0, 6, uniBuf, sizeof(uniData));
    E(1, 7, cellParamsBuf_, cellParamsSz);
    E(2, 8, utParamsBuf_, utParamsSz);
    E(3, 9, sysConfigBuf_, sizeof(SystemLevelConfigGPU));
    E(4, 10, simConfigBuf_, sizeof(SimConfigGPU));
    E(5, 11, cmnLinkBuf_, sizeof(CmnLinkParamsGPU));
    E(6, 12, linkParamsBuf_, linkParamsSz);
    E(7, 13, rngStatesBuf_, rngStatesSz);
    E(8, 14, crnLosBuf_, crnLosSz);
    E(9, 15, crnNlosBuf_, crnNlosSz);
    E(10, 16, crnO2iBuf_, crnO2iSz);
    E(11, 17, crnLosOffBuf_, crnLosOffSz);
    E(12, 18, crnNlosOffBuf_, crnNlosOffSz);
    E(13, 19, crnO2iOffBuf_, crnO2iOffSz);

    wgpu::BindGroupDescriptor bgDesc = wgpu::Default;
    bgDesc.layout = bg0Layout;
    bgDesc.entryCount = (uint32_t)entries.size();
    bgDesc.entries = entries.data();
    wgpu::BindGroup bg0 = device_.createBindGroup(bgDesc);

    // ── Encode ─────────────────────────────────────────────────────────────
    wgpu::CommandEncoderDescriptor encDesc = wgpu::Default;
    wgpu::CommandEncoder enc = device_.createCommandEncoder(encDesc);
    {
        wgpu::ComputePassDescriptor passDesc = wgpu::Default;
        wgpu::ComputePassEncoder pass = enc.beginComputePass(passDesc);
        pass.setPipeline(linkParamPipeline_);

        // ── Fix 5: setBindGroup needs (uint32_t, BindGroup, size_t, uint32_t*) ──
        pass.setBindGroup(0u, bg0, (size_t)0, nullptr);

        pass.dispatchWorkgroups(nSite, (nUT + 255u) / 256u, 1u);
        pass.end();
    }
    enc.copyBufferToBuffer(linkParamsBuf_, 0, stagingBuf_, 0, linkParamsSz);

    // ── Fix 6: finish() needs CommandBufferDescriptor; store before submit ─
    wgpu::CommandBufferDescriptor cbDesc = wgpu::Default;
    wgpu::CommandBuffer cmdBuf = enc.finish(cbDesc);
    queue_.submit(cmdBuf); // single-buffer overload

    waitIdle();
}
