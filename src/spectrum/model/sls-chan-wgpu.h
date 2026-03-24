// This is an AI translated version of the original Nvidia source for GPU compute,
// but ported to WGPU for cross-platform support
// https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/main/testBenches/chanModels/src/sls_chan_src/sls_chan_large_scale_GPU.cu

#ifndef SLS_CHAN_WGPU_H
#define SLS_CHAN_WGPU_H
#include <cstdint>
#include <string>
#include <vector>
#include <webgpu/webgpu.hpp> // C++ wrapper around webgpu.h

// Plain-old-data structs that map exactly to the WGSL struct layouts.
// __attribute__((packed)) / alignas mirror the WGSL std430 rules.
struct alignas(16) Vec3f
{
    float x, y, z, _p;
};

struct alignas(16) CellParam
{
    Vec3f loc; /* add remaining fields */
};

struct alignas(16) UtParam
{
    Vec3f loc;
    float d_2d_in;
    uint32_t outdoor_ind;
    float o2i_penetration_loss;
    float _p;
};

struct LinkParams
{
    float d2d, d2d_in, d2d_out, d3d;
    float d3d_in, d3d_out;
    float phi_LOS_AOD, phi_LOS_AOA;
    float theta_LOS_ZOD, theta_LOS_ZOA;
    uint32_t losInd;
    float pathloss;
    float SF, K, DS, ASD, ASA, ZSD, ZSA;
    float mu_lgZSD, sigma_lgZSD, mu_offset_ZOD, _pad;
};

struct alignas(16) RngState
{
    uint32_t s0, s1, s2, s3;
};

struct LinkParamUniforms
{
    float maxX, minX, maxY, minY;
    uint32_t nSite, nUT, nSectorPerSite;
    uint32_t updatePL, updateAllLSPs, updateLosState;
    int32_t nX, nY;
};

struct SystemLevelConfigGPU
{
    uint32_t scenario, o2i_bldg, o2i_car, _p;
    float force_los_indoor;  // -1.0 = use formula
    float force_los_outdoor; // -1.0 = use formula
    float _p2[2];
};

struct SimConfigGPU
{
    float center_freq_hz;
    float _p0, _p1, _p2;
};

struct CmnLinkParamsGPU
{
    float sqrtCorrMatLos[49];  // 7×7 lower-triangular Cholesky, identity for now
    float sqrtCorrMatNlos[36]; // 6×6
    float sqrtCorrMatO2i[36];  // 6×6
    float mu_K[4], sigma_K[4]; // Actually 3 for each below, but we add an extra for padding
    float mu_lgDS[4], sigma_lgDS[4];
    float mu_lgASD[4], sigma_lgASD[4];
    float mu_lgASA[4], sigma_lgASA[4];
    float mu_lgZSA[4], sigma_lgZSA[4];
    float lgfc;
    float _pad[3];
};

// ── String helper for v24 WGPUStringView ──────────────────────────────────
// WGPU_STRLEN = SIZE_MAX means "null-terminated, compute length internally"
static inline WGPUStringView
sv(const char* s)
{
    return {s, WGPU_STRLEN};
}

class SlsChanWgpu
{
  public:
    explicit SlsChanWgpu(wgpu::Device device);
    ~SlsChanWgpu() = default;

    // Upload topology/config once (or whenever it changes)
    void uploadCellParams(const std::vector<CellParam>& cells);
    void uploadUtParams(const std::vector<UtParam>& uts);

    // Mirrors slsChan::generateCRNGPU()
    void generateCRN(float maxX,
                     float minX,
                     float maxY,
                     float minY,
                     const float corrDistsLos[7],
                     const float corrDistsNlos[6],
                     const float corrDistsO2i[6]);

    // Mirrors slsChan::calLinkParamGPU()
    void calLinkParam(uint32_t nSite,
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
                      int32_t nY);

    // Readback results to CPU
    std::vector<LinkParams> readLinkParams(uint32_t nSite, uint32_t nUT);

    constexpr int32_t nSite()
    {
        return nSite_;
    };

    constexpr int32_t nX()
    {
        return nX_;
    };

    constexpr int32_t nY()
    {
        return nY_;
    };

  private:
    wgpu::Device device_;
    wgpu::Queue queue_;
    wgpu::ShaderModule shader_;
    wgpu::ComputePipeline linkParamPipeline_;
    wgpu::ComputePipeline crnFillPipeline_;
    wgpu::ComputePipeline crnConvPipeline_;
    wgpu::ComputePipeline crnNormPipeline_;

    wgpu::Buffer cellParamsBuf_, utParamsBuf_;
    wgpu::Buffer sysConfigBuf_, simConfigBuf_, cmnLinkBuf_;
    wgpu::Buffer linkParamsBuf_, rngStatesBuf_, crnTempBuf_;
    wgpu::Buffer crnLosBuf_, crnNlosBuf_, crnO2iBuf_;
    wgpu::Buffer crnLosOffBuf_, crnNlosOffBuf_, crnO2iOffBuf_;
    wgpu::Buffer stagingBuf_;
    uint32_t nSite_ = 0;
    int32_t nX_ = 0;
    int32_t nY_ = 0;

    wgpu::Buffer makeBuffer(uint64_t size, wgpu::BufferUsage usage, const void* data = nullptr);
    void waitIdle();
};
#endif
