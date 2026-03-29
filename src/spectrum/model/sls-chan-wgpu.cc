// This is an AI translated version of the original Nvidia source for GPU compute,
// but ported to WGPU for cross-platform support
// https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/main/testBenches/chanModels/src/sls_chan_src/sls_chan_large_scale_GPU.cu

#define WEBGPU_CPP_IMPLEMENTATION
#include "sls-chan-wgpu.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>

static std::string
readFile(const char* path)
{
    std::ifstream f(path);
    return {std::istreambuf_iterator<char>(f), {}};
}

// ── Constructor ───────────────────────────────────────────────────────────────
SlsChanWgpu::SlsChanWgpu(wgpu::Device device)
    : device_(std::move(device)),
      queue_(device.getQueue())
{
    std::string wgsl = readFile("C:/tools/sources/ns-3-dev/src/spectrum/model/sls-chan.wgsl");

    WGPUShaderSourceWGSL wgslSource{};
    wgslSource.chain.next = nullptr;
    wgslSource.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSource.code = sv(wgsl.c_str());

    WGPUShaderModuleDescriptor smDescC{};
    smDescC.nextInChain = &wgslSource.chain;
    shader_ = wgpu::ShaderModule(wgpuDeviceCreateShaderModule(device_, &smDescC));
    assert(shader_ && "WGSL failed to compile — check entry point names and file path");

    auto makePipeline = [&](const char* ep) -> wgpu::ComputePipeline {
        wgpu::ComputePipelineDescriptor desc{};
        desc.compute.module = shader_;
        desc.compute.entryPoint = sv(ep);
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
    clusterRayPipeline_ = makePipeline("cal_cluster_ray_kernel");
    assert(clusterRayPipeline_ && "missing cal_cluster_ray_kernel in WGSL");
    generateCIRPipeline_ = makePipeline("generate_cir_kernel");
    assert(generateCIRPipeline_ && "missing generate_cir_kernel in WGSL");
    generateCFRPipeline_ = makePipeline("generate_cfr_kernel_mode1");
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
                         const float corrLos[7],
                         const float corrNlos[6],
                         const float corrO2i[6])
{
    const float kStep = 10.0f;
    const int32_t nX = std::abs(int32_t((maxX - minX) / kStep)) + 1;
    const int32_t nY = std::abs(int32_t((maxY - minY) / kStep)) + 1;

    nX_ = nX;
    nY_ = nY;

    const uint64_t gridSz = uint64_t(nX) * nY;
    const uint64_t losBufSz = uint64_t(nSite_) * 7 * gridSz * sizeof(float);
    const uint64_t nlosBufSz = uint64_t(nSite_) * 6 * gridSz * sizeof(float);
    const uint64_t o2iBufSz = uint64_t(nSite_) * 6 * gridSz * sizeof(float);

    auto tempBufBytes = [&](float cd) -> uint64_t {
        const float cp = (cd == 0.0f) ? 0.0f : (cd / kStep);
        const uint32_t iD = static_cast<uint32_t>(3.0f * cp);
        const uint32_t L = (cp == 0.0f) ? 1u : (2u * iD + 1u);
        const uint64_t pnx = static_cast<uint64_t>(nX) + (L - 1u);
        const uint64_t pny = static_cast<uint64_t>(nY) + (L - 1u);
        return pnx * pny * sizeof(float);
    };

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
        const uint64_t curTempBytes = tempBufBytes(corrDist);
        const uint64_t rngBytes = uint64_t(nCrnRng) * sizeof(RngState);

        struct CRNGenUniforms
        {
            float maxX, minX, maxY, minY;
            float corrDist;
            uint32_t maxRngStates, outputGridOffset, _pad;
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

        // ── Fill bind group ──
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

        // ── Conv bind group ──
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

        // ── Norm bind group ──
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

        // Pass 1: fill
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

        // Pass 2: convolve + normalize + copy
        {
            wgpu::CommandEncoder enc2 = device_.createCommandEncoder(wgpu::Default);
            {
                auto pass1 = enc2.beginComputePass(wgpu::Default);
                pass1.setPipeline(crnConvPipeline_);
                pass1.setBindGroup(0u, convBg, (size_t)0, nullptr);
                pass1.dispatchWorkgroups(128u, 1u, 1u);
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
                          int32_t nX,
                          int32_t nY)
{
    assert(!isDead());

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

    auto bg0Layout = linkParamPipeline_.getBindGroupLayout(0);
    std::vector<wgpu::BindGroupEntry> entries(14, wgpu::Default);
    auto E = [&](int i, uint32_t binding, wgpu::Buffer buf, uint64_t size = WGPU_WHOLE_SIZE) {
        entries[i].binding = binding;
        entries[i].buffer = buf;
        entries[i].offset = 0;
        entries[i].size = size;
        if (!buf)
        {
            std::cerr << "null buffer at entry " << i << std::endl;
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
    E(e0, 2, 2, ssCmnLinkBuf_);       // cray_buf_cmn
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

// ── Readback helpers ──────────────────────────────────────────────────────────
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
