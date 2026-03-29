// sls-chan-validation.cc
// Sweeps UTs across distances, writes per-link CSV for 3GPP TR 38.901 UMa validation.

#include "sls-chan-wgpu.h"
#include "wgpu.h"

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

static wgpu::Device
createDevice()
{
    WGPUInstanceDescriptor idesc{};
    WGPUInstance instance = wgpuCreateInstance(&idesc);
    assert(instance);

    WGPUAdapter adapter = nullptr;
    WGPURequestAdapterOptions aopts{};
    aopts.backendType = WGPUBackendType_D3D12; // Force D3D12, not Vulkan

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
                                                   *static_cast<WGPUAdapter*>(ud1) = a;
                                               }
                                               else
                                               {
                                                   fprintf(stderr, "requestAdapter failed\n");
                                               }
                                           },
                                       .userdata1 = &adapter});
    assert(adapter);

    // Query what the adapter actually supports first
    WGPULimits supported{};
    wgpuAdapterGetLimits(adapter, &supported);

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
    requiredLimits.maxStorageBufferBindingSize =
        std::min(supported.maxStorageBufferBindingSize, uint64_t(1) << 31);
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
    requiredLimits.minStorageBufferOffsetAlignment = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.minUniformBufferOffsetAlignment = WGPU_LIMIT_U32_UNDEFINED;
    requiredLimits.minUniformBufferOffsetAlignment = WGPU_LIMIT_U32_UNDEFINED;
    ddesc.requiredLimits = &requiredLimits;
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
    assert(device);
    wgpu::Device dev(device);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    return dev;
}

static float
uma_d_bp(float h_bs, float h_ut, float fc_ghz)
{
    const float h_e = 1.0f;
    return 4.0f * (h_bs - h_e) * (h_ut - h_e) * fc_ghz * 1e9f / 3e8f;
}

static float
uma_los_pl_ref(float d2d, float d3d, float h_bs, float h_ut, float fc_ghz)
{
    const float d_bp = uma_d_bp(h_bs, h_ut, fc_ghz);
    const float pl1 = 28.0f + 22.0f * std::log10(d3d) + 20.0f * std::log10(fc_ghz);
    const float pl2 = 28.0f + 40.0f * std::log10(d3d) + 20.0f * std::log10(fc_ghz) -
                      9.0f * std::log10(d_bp * d_bp + (h_bs - h_ut) * (h_bs - h_ut));
    return (d2d <= d_bp) ? pl1 : pl2;
}

static float
uma_nlos_pl_ref(float d3d, float fc_ghz)
{
    return 32.4f + 20.0f * std::log10(fc_ghz) + 30.0f * std::log10(d3d);
}

static void
buildHexCells(uint32_t nSite,
              uint32_t nSector,
              float isd,
              float h_bs,
              std::vector<CellParam>& cells)
{
    cells.resize(nSite * nSector);
    for (uint32_t s = 0; s < nSite; ++s)
    {
        float sx = 0.0f, sy = 0.0f;
        if (s > 0)
        {
            const float angle = float(s - 1) * float(M_PI) / 3.0f;
            sx = isd * std::cos(angle);
            sy = isd * std::sin(angle);
        }
        for (uint32_t k = 0; k < nSector; ++k)
        {
            cells[s * nSector + k].loc = {sx, sy, h_bs, 0.0f};
        }
    }
}

int
main()
{
    wgpu::Device device = createDevice();
    SlsChanWgpu sls(device);

    const uint32_t nSite = 7;
    const uint32_t nSector = 3;
    const uint32_t nCell = nSite * nSector;
    const float fc_ghz = 3.5f;
    const float h_bs = 25.0f;
    const float h_ut = 1.5f;
    const float isd = 500.0f;

    const uint32_t nUT = 1000;
    const float maxDist = 2000.0f;

    std::vector<CellParam> cells;
    buildHexCells(nSite, nSector, isd, h_bs, cells);
    sls.uploadCellParams(cells);

    std::vector<UtParam> uts(nUT);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uDist(-1.0f, 1.0f);
    for (uint32_t u = 0; u < nUT; u++)
    {
        float x, y;
        double closestCellDistance = std::numeric_limits<double>::max();
        do
        {
            // Create UE position
            x = uDist(rng) * maxDist;
            y = uDist(rng) * maxDist;

            // Check UE is further away from the closest UE at least 10m
            for (auto& cell : cells)
            {
                double d = sqrt(pow(cell.loc.x - x, 2) + pow(cell.loc.y - y, 2));
                if (d < closestCellDistance)
                {
                    closestCellDistance = d;
                }
            }
        } while (closestCellDistance < 10);

        uts[u].loc = {x, y, h_ut, 0.0f};
        uts[u].d_2d_in = 0.0f;
        uts[u].outdoor_ind = 1u;
        uts[u].o2i_penetration_loss = 0.0f;
    }
    sls.uploadUtParams(uts);

    float corrLos[7] = {37.f, 12.f, 30.f, 18.f, 15.f, 15.f, 15.f};
    float corrNlos[6] = {50.f, 40.f, 50.f, 50.f, 50.f, 50.f};
    float corrO2i[6] = {10.f, 10.f, 11.f, 17.f, 25.f, 25.f};

    const float area = maxDist + 100.0f;

    // Check large system parameters
    sls.generateCRN(area, -area, area, -area, corrLos, corrNlos, corrO2i);
    std::cerr << "generateCRN ok" << std::endl;

    sls.calLinkParam(nSite,
                     nUT,
                     nSector,
                     area,
                     -area,
                     area,
                     -area,
                     true,
                     true,
                     true,
                     sls.nX(),
                     sls.nY());
    std::cerr << "calLinkParam ok" << std::endl;

    auto links = sls.readLinkParams(nSite, nUT);
    std::cerr << "readLinkParams ok" << std::endl;

    std::ofstream csv("link_params.csv");
    csv << "cell_id,ut_id,is_los,is_outdoor,d2d_m,d3d_m,pl_sim_db,pl_ref_db,sf_db,k_db,ds_ns,asd_"
           "deg,asa_deg,zsd_deg,zsa_deg\n";

    for (uint32_t site = 0; site < nSite; ++site)
    {
        for (uint32_t u = 0; u < nUT; ++u)
        {
            const LinkParams& lk = links[site * nUT + u];
            const float d3d = std::max(lk.d3d, 1e-3f);
            const float pl_ref = lk.losInd ? uma_los_pl_ref(lk.d2d, d3d, h_bs, h_ut, fc_ghz)
                                           : uma_nlos_pl_ref(d3d, fc_ghz);

            csv << site << "," << u << "," << lk.losInd << "," << uts[u].outdoor_ind << ","
                << lk.d2d << "," << d3d << "," << lk.pathloss << "," << pl_ref << "," << lk.SF
                << "," << lk.K << "," << lk.DS << "," << lk.ASD << "," << lk.ASA << "," << lk.ZSD
                << "," << lk.ZSA << "\n";
        }
    }

    std::cout << "Wrote link_params.csv (" << uint64_t(nCell) * nUT << " links)\n";

    // Check small scale parameters
    sls.uploadSmallScaleConfig(
        /*scSpacingHz=*/15000.0f,
        /*fftSize=*/4096,
        /*nPrb=*/106,
        /*nPrbg=*/53,
        /*nSnapshotPerSlot=*/14,
        /*enablePropagationDelay=*/0,
        /*disableSmallScaleFading=*/0,
        /*disablePlShadowing=*/0,
        /*optionalCfrDim=*/0,       // 0 = use nPrbg as CFR dimension
        /*lambda0=*/3e8f / (3.5e9f) // c/fc = ~0.0857 m at 3.5 GHz
    );
    std::cerr << "uploadSmallScaleConfig ok" << std::endl;

    // ── Antenna panel configs (matches default_antenna_config.yaml) ──────────
    // panel 0 = BS: nAnt=4, antSize=[1,1,1,2,2], antSpacing=[0,0,0.5,0.5], polarAngles=[45,-45]
    // panel 1 = UE: nAnt=1, antSize=[1,1,2,2,1], antSpacing=[0,0,0.5,0.5], polarAngles=[0,0]
    std::vector<AntPanelConfigGPU> antCfgs(2);

    // BS panel (index 0)
    antCfgs[0].nAnt = 4;
    antCfgs[0].antModel = 1; // directional
    antCfgs[0].antSize[0] = 1;
    antCfgs[0].antSize[1] = 1;
    antCfgs[0].antSize[2] = 1;
    antCfgs[0].antSize[3] = 2;
    antCfgs[0].antSize[4] = 2;
    antCfgs[0].antSpacing[0] = 0.0f;
    antCfgs[0].antSpacing[1] = 0.0f;
    antCfgs[0].antSpacing[2] = 0.5f;
    antCfgs[0].antSpacing[3] = 0.5f;
    antCfgs[0].antPolarAngles[0] = 45.0f;
    antCfgs[0].antPolarAngles[1] = -45.0f;
    antCfgs[0].thetaOffset = 0; // first 181 entries in flat theta table
    antCfgs[0].phiOffset = 0;   // first 360 entries in flat phi table

    // UE panel (index 1)
    antCfgs[1].nAnt = 1;
    antCfgs[1].antModel = 0; // isotropic
    antCfgs[1].antSize[0] = 1;
    antCfgs[1].antSize[1] = 1;
    antCfgs[1].antSize[2] = 2;
    antCfgs[1].antSize[3] = 2;
    antCfgs[1].antSize[4] = 1;
    antCfgs[1].antSpacing[0] = 0.0f;
    antCfgs[1].antSpacing[1] = 0.0f;
    antCfgs[1].antSpacing[2] = 0.5f;
    antCfgs[1].antSpacing[3] = 0.5f;
    antCfgs[1].antPolarAngles[0] = 0.0f;
    antCfgs[1].antPolarAngles[1] = 0.0f;
    antCfgs[1].thetaOffset = 181; // second block in flat theta table
    antCfgs[1].phiOffset = 360;   // second block in flat phi table

    // Flat antenna pattern tables: 181 theta entries + 360 phi entries per panel
    // For isotropic (antModel=0): all zeros in dB → gain=1 after 10^(0/20)
    // For directional (antModel=1): simplified flat 0 dB table for validation
    std::vector<float> antThetaFlat(2 * 181, 0.0f); // 0 dB for all angles
    std::vector<float> antPhiFlat(2 * 360, 0.0f);

    const uint32_t nBsAnt = antCfgs[0].nAnt; // 4
    const uint32_t nUeAnt = antCfgs[1].nAnt; // 1
    sls.uploadAntPanelConfigs(antCfgs, antThetaFlat, antPhiFlat);
    std::cerr << "uploadAntPanelConfigs ok" << std::endl;

    // Small-scale common parameters for WGSL binding 2 (SsCmnParams).
    // These are safe placeholder values to make the buffer valid and bounded.
    // Replace them with your calibrated 38.901 UMa numbers after the crash is fixed.
    SsCmnParams ssCmn{};

    for (int i = 0; i < 3; ++i)
    {
        ssCmn.mu_lgDS[i] = -7.0f;
        ssCmn.sigma_lgDS[i] = 0.3f;
        ssCmn.mu_lgASD[i] = 1.2f;
        ssCmn.sigma_lgASD[i] = 0.2f;
        ssCmn.mu_lgASA[i] = 1.7f;
        ssCmn.sigma_lgASA[i] = 0.2f;
        ssCmn.mu_lgZSA[i] = 0.9f;
        ssCmn.sigma_lgZSA[i] = 0.2f;
        ssCmn.mu_K[i] = (i == 1) ? 9.0f : 0.0f;
        ssCmn.sigma_K[i] = (i == 1) ? 3.5f : 0.0f;
        ssCmn.r_tao[i] = 2.2f;
        ssCmn.mu_XPR[i] = 8.0f;
        ssCmn.sigma_XPR[i] = 4.0f;
        ssCmn.nCluster[i] = 12;
        ssCmn.nRayPerCluster[i] = 20;
        ssCmn.C_DS[i] = 1.0f;
        ssCmn.C_ASD[i] = 1.0f;
        ssCmn.C_ASA[i] = 1.0f;
        ssCmn.C_ZSA[i] = 1.0f;
        ssCmn.xi[i] = 3.0f;
    }
    ssCmn.C_phi_LOS = 1.0f;
    ssCmn.C_phi_NLOS = 1.0f;
    ssCmn.C_phi_O2I = 1.0f;
    ssCmn.C_theta_LOS = 1.0f;
    ssCmn.C_theta_NLOS = 1.0f;
    ssCmn.C_theta_O2I = 1.0f;
    ssCmn.lgfc = std::log10(fc_ghz);
    ssCmn.lambda_0 = 3e8f / 3.5e9f;
    ssCmn.raysInSubClusterSizes[0] = 10;
    ssCmn.raysInSubClusterSizes[1] = 6;
    ssCmn.raysInSubClusterSizes[2] = 4;
    for (uint32_t i = 0; i < 10; ++i)
    {
        ssCmn.raysInSubCluster0[i] = i;
    }
    for (uint32_t i = 0; i < 6; ++i)
    {
        ssCmn.raysInSubCluster1[i] = 10 + i;
    }
    for (uint32_t i = 0; i < 4; ++i)
    {
        ssCmn.raysInSubCluster2[i] = 16 + i;
    }
    ssCmn.nSubCluster = 3;
    ssCmn.nUeAnt = nUeAnt;
    ssCmn.nBsAnt = nBsAnt;
    sls.uploadCmnLinkParamsSmallScale(ssCmn);
    std::cerr << "uploadCmnLinkParamsSmallScale ok" << std::endl;

    // ── Build active links: all nSite × nUT pairs ────────────────────────────
    const uint32_t nSnapshots = 14; // n_snapshot_per_slot
    const uint32_t nPrbg = 53;

    std::vector<ActiveLink> activeLinks;
    activeLinks.reserve(nSite * nUT);
    for (uint32_t site = 0; site < nSite; ++site)
    {
        for (uint32_t u = 0; u < nUT; ++u)
        {
            const uint32_t linkIdx = site * nUT + u;
            const uint32_t elemsPerLink = nSnapshots * nUeAnt * nBsAnt * 24u; // NMAXTAPS=24
            ActiveLink al;
            al.cid = site * nSector; // first sector of this site
            al.uid = u;
            al.linkIdx = linkIdx;
            al.lspReadIdx = linkIdx;
            al.cirCoeOffset = linkIdx * elemsPerLink;
            al.cirNormDelayOffset = linkIdx * 24u;
            al.cirNtapsOffset = linkIdx;
            al.freqChanPrbgOffset = linkIdx * nSnapshots * nUeAnt * nBsAnt * nPrbg;
            activeLinks.push_back(al);
        }
    }
    const uint32_t nActiveLinks = static_cast<uint32_t>(activeLinks.size());

    // Build small-scale cell params (antPanelIdx + orientation)
    std::vector<CellParamSS> cellsSS(nCell);
    for (uint32_t i = 0; i < nCell; ++i)
    {
        cellsSS[i].antPanelIdx = 0;                 // BS panel = index 0
        cellsSS[i].antPanelOrientation[0] = 102.0f; // theta_tilt (degrees downtilt)
        cellsSS[i].antPanelOrientation[1] = float(i % nSector) * 120.0f; // phi per sector
        cellsSS[i].antPanelOrientation[2] = 0.0f;                        // zeta_offset
        cellsSS[i]._pad0 = 0;
    }
    sls.uploadCellParamsSS(cellsSS);
    std::cerr << "uploadCellParamsSS ok" << std::endl;

    // Build small-scale UT params
    std::vector<UtParamSS> utsSS(nUT);
    for (uint32_t u = 0; u < nUT; ++u)
    {
        utsSS[u].antPanelIdx = 1; // UE panel = index 1
        utsSS[u].outdoor_ind = uts[u].outdoor_ind;
        utsSS[u].antPanelOrientation[0] = 0.0f;
        utsSS[u].antPanelOrientation[1] = 0.0f;
        utsSS[u].antPanelOrientation[2] = 0.0f;
        utsSS[u].velocity[0] = 0.0f; // stationary UTs
        utsSS[u].velocity[1] = 0.0f;
        utsSS[u].velocity[2] = 0.0f;
        utsSS[u]._pad0 = 0;
    }
    sls.uploadUtParamsSS(utsSS);
    std::cerr << "uploadUtParamsSS ok" << std::endl;

    // ── Small-scale pipeline ─────────────────────────────────────────────────
    sls.calClusterRay(nSite, nUT);
    std::cerr << "calClusterRay ok" << std::endl;

    sls.generateCIR(activeLinks, nActiveLinks, nSnapshots, /*refTime=*/0.0f);
    std::cerr << "generateCIR ok" << std::endl;

    sls.generateCFR(activeLinks, nActiveLinks, nSnapshots);
    std::cerr << "generateCFR ok" << std::endl;

    // ── Readback ─────────────────────────────────────────────────────────────
    auto cirCoe = sls.readCirCoe(nActiveLinks, nSnapshots, nUeAnt, nBsAnt);
    auto freqChanPrbg = sls.readFreqChanPrbg(nActiveLinks, nSnapshots, nUeAnt, nBsAnt);
    auto clusterParams = sls.readClusterParams(nSite, nUT);
    auto cirNtaps = sls.readCirNtaps();
    auto xpr = sls.readXpr();
    auto phiNmAoA = sls.readPhiNmAoA();
    auto phiNmAoD = sls.readPhiNmAoD();
    auto thetaNmZOA = sls.readThetaNmZOA();
    auto thetaNmZOD = sls.readThetaNmZOD();

    // ── Write small-scale CSV ─────────────────────────────────────────────────
    // Per link: mean CIR power across snapshots/antennas/taps, mean CFR power
    std::ofstream csvSS("small_scale_params.csv");
    std::ofstream csvDetail("small_scale_detail.csv");
    std::ofstream csvRays("ray_params.csv");

    csvSS << "site,ut,is_los,d2d_m,cir_power_db,cfr_power_db\n";
    csvDetail << "site,ut,is_los,d2d_m,ds_ns,asd_deg,asa_deg,zsa_deg,"
                 "n_cluster,n_ray_per_cluster,n_taps,strongest0,strongest1,"
                 "cir_power_db,cfr_power_db\n";
    csvRays << "site,ut,is_los,d2d_m,cluster_idx,ray_idx,strongest_cluster,"
               "cluster_delay_ns,cluster_power_lin,cluster_power_db,"
               "aoa_deg,aod_deg,zoa_deg,zod_deg,xpr_linear,xpr_db\n";

    for (uint32_t site = 0; site < nSite; ++site)
    {
        for (uint32_t u = 0; u < nUT; ++u)
        {
            const uint32_t linkIdx = site * nUT + u;
            const LinkParams& lk = links[linkIdx];
            const ClusterParamsGpu& cp = clusterParams[linkIdx];

            // CIR mean power over all snapshots * nUeAnt * nBsAnt * NMAXTAPS
            const uint32_t cirBase = linkIdx * nSnapshots * nUeAnt * nBsAnt * 24u;
            float cirPow = 0.0f;
            const uint32_t cirCount = nSnapshots * nUeAnt * nBsAnt * 24u;
            for (uint32_t i = 0; i < cirCount; ++i)
            {
                const auto& c = cirCoe[cirBase + i];
                cirPow += c.real() * c.real() + c.imag() * c.imag();
            }
            cirPow /= float(cirCount);

            // CFR mean power over all snapshots * nUeAnt * nBsAnt * nPrbg
            const uint32_t cfrBase = linkIdx * nSnapshots * nUeAnt * nBsAnt * nPrbg;
            float cfrPow = 0.0f;
            const uint32_t cfrCount = nSnapshots * nUeAnt * nBsAnt * nPrbg;
            for (uint32_t i = 0; i < cfrCount; ++i)
            {
                const auto& c = freqChanPrbg[cfrBase + i];
                cfrPow += c.real() * c.real() + c.imag() * c.imag();
            }
            cfrPow /= float(cfrCount);

            const float cirDb = (cirPow > 0.0f) ? 10.0f * std::log10(cirPow) : -999.0f;
            const float cfrDb = (cfrPow > 0.0f) ? 10.0f * std::log10(cfrPow) : -999.0f;

            csvSS << site << "," << u << "," << lk.losInd << "," << lk.d2d << "," << cirDb << ","
                  << cfrDb << "\n";

            csvDetail << site << "," << u << "," << lk.losInd << "," << lk.d2d << "," << lk.DS
                      << "," << lk.ASD << "," << lk.ASA << "," << lk.ZSA << "," << cp.nCluster
                      << "," << cp.nRayPerCluster << "," << cirNtaps[linkIdx] << ","
                      << cp.strongest2clustersIdx[0] << "," << cp.strongest2clustersIdx[1] << ","
                      << cirDb << "," << cfrDb << "\n";
            uint32_t MAXCR = cp.nCluster * cp.nRayPerCluster;
            const uint32_t rayBase = linkIdx * MAXCR;
            for (uint32_t c = 0; c < cp.nCluster; ++c)
            {
                const bool strongest =
                    (c == cp.strongest2clustersIdx[0]) || (c == cp.strongest2clustersIdx[1]);
                const float clPowLin = cp.powers[c];
                const float clPowDb = (clPowLin > 0.0f) ? 10.0f * std::log10(clPowLin) : -999.0f;

                for (uint32_t r = 0; r < cp.nRayPerCluster; ++r)
                {
                    const uint32_t idx = rayBase + c * cp.nRayPerCluster + r;
                    const float xprLin = xpr[idx];
                    const float xprDb = (xprLin > 0.0f) ? 10.0f * std::log10(xprLin) : -999.0f;

                    csvRays << site << "," << u << "," << lk.losInd << "," << lk.d2d << "," << c
                            << "," << r << "," << (strongest ? 1 : 0) << "," << cp.delays[c] << ","
                            << clPowLin << "," << clPowDb << "," << phiNmAoA[idx] << ","
                            << phiNmAoD[idx] << "," << thetaNmZOA[idx] << "," << thetaNmZOD[idx]
                            << "," << xprLin << "," << xprDb << "\n";
                }
            }
        }
    }
    std::cout << "Wrote small_scale_params.csv (" << nActiveLinks << " links)\n";
    std::cout << "Wrote small_scale_detail.csv (" << nActiveLinks << " links)\n";
    std::cout << "Wrote ray_params.csv (" << "variable rows" << ")\n";
    return 0;
}
