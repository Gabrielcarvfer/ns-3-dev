// sls-chan-validation.cc
// Sweeps UTs across distances, writes per-link CSV for 3GPP TR 38.901 UMa validation.

#include "sls-chan-wgpu.h"

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
    aopts.powerPreference = WGPUPowerPreference_HighPerformance;
    wgpuInstanceRequestAdapter(
        instance,
        &aopts,
        WGPURequestAdapterCallbackInfo{.callback =
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
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    return wgpu::Device(device);
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

    const uint32_t nUT = 2000;
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

    sls.generateCRN(area, -area, area, -area, corrLos, corrNlos, corrO2i);

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

    auto links = sls.readLinkParams(nSite, nUT);

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
    return 0;
}
