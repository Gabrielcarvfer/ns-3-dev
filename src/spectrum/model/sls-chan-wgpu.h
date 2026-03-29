// This is an AI translated version of the original Nvidia source for GPU compute,
// but ported to WGPU for cross-platform support
// https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/main/testBenches/chanModels/src/sls_chan_src/sls_chan_large_scale_GPU.cu

#ifndef SLS_CHAN_WGPU_H
#define SLS_CHAN_WGPU_H
#include <cassert>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>
#include <webgpu/webgpu.hpp> // C++ wrapper around webgpu.h

// ─────────────────────────────────────────────────────────────────────────────
// Plain-old-data structs that map exactly to the WGSL struct layouts.
// alignas values mirror the WGSL std430 rules.
// ─────────────────────────────────────────────────────────────────────────────

struct alignas(16) Vec3f
{
    float x, y, z, _p;
};

// ── Large-scale CellParam / UtParam ──────────────────────────────────────────
// Used by calLinkParam (bindings 7 & 8 in cal_link_param_kernel)
struct alignas(16) CellParam
{
    Vec3f loc;
    // add remaining large-scale fields from sls_chan.cuh here
};

struct alignas(16) UtParam
{
    Vec3f loc;
    float d_2d_in;
    uint32_t outdoor_ind;
    float o2i_penetration_loss;
    float _p;
    // add remaining large-scale fields from sls_chan.cuh here
};

// ── Small-scale CellParam / UtParam ──────────────────────────────────────────
// Used by calClusterRay / generateCIR / generateCFR (bindings 23 & 24)
// Must match the WGSL small-scale CellParam / UtParam structs exactly.
struct alignas(16) CellParamSS
{
    uint32_t antPanelIdx;
    float antPanelOrientation[3]; // [theta_tilt, phi_tilt, zeta_offset]
    uint32_t _pad0;
};

struct alignas(16) UtParamSS
{
    uint32_t antPanelIdx;
    uint32_t outdoor_ind;         // 1 = outdoor, 0 = indoor (O2I)
    float antPanelOrientation[3]; // [theta_tilt, phi_tilt, zeta_offset]
    float velocity[3];
    uint32_t _pad0;
};

// ── Large-scale LinkParams ────────────────────────────────────────────────────
// NOTE: the small-scale WGSL uses a slightly different LinkParams layout
//       (phi/theta field order swapped, ZSD/ZSA at the end).
//       Two layouts are intentionally kept separate – this one is written by
//       calLinkParam and read back via readLinkParams.
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

// ── Large-scale uniforms / config structs ─────────────────────────────────────

struct LinkParamUniforms
{
    float maxX, minX, maxY, minY;
    uint32_t nSite, nUT, nSectorPerSite;
    uint32_t updatePL, updateAllLSPs, updateLosState;
    int32_t nX, nY;
};

// Binding 9  — sys_config (large-scale kernel)
struct SystemLevelConfigGPU
{
    uint32_t scenario, o2i_bldg, o2i_car, _p;
    float force_los_indoor;  // -1.0 = use formula
    float force_los_outdoor; // -1.0 = use formula
    float _p2[2];
};

// Binding 10 — sim_config (large-scale kernel)
struct SimConfigGPU
{
    float center_freq_hz;
    float _p0, _p1, _p2;
};

struct CmnLinkParamsGPU
{
    float sqrtCorrMatLos[49];  // 7×7 lower-triangular Cholesky
    float sqrtCorrMatNlos[36]; // 6×6
    float sqrtCorrMatO2i[36];  // 6×6
    float mu_K[4], sigma_K[4];
    float mu_lgDS[4], sigma_lgDS[4];
    float mu_lgASD[4], sigma_lgASD[4];
    float mu_lgASA[4], sigma_lgASA[4];
    float mu_lgZSA[4], sigma_lgZSA[4];
    float lgfc;
    float _pad[3];
};

// ── Small-scale structs ───────────────────────────────────────────────────────
// MAX_CLUSTERS=20, MAX_RAYS=20 — must match WGSL constants

static constexpr uint32_t MAX_CLUSTERS = 20;
static constexpr uint32_t MAX_RAYS = 20;

// Exact host-side mirror of WGSL `ClusterParams`.
// IMPORTANT: per-ray data (phinm*, thetanm*, xpr, randomPhases)
// lives in separate GPU buffers and is NOT part of this struct.
struct ClusterParamsGpu
{
    uint32_t nCluster;
    uint32_t nRayPerCluster;
    float delays[MAX_CLUSTERS];
    float powers[MAX_CLUSTERS];
    uint32_t strongest2clustersIdx[2];
    float phinAoA[MAX_CLUSTERS];
    float phinAoD[MAX_CLUSTERS];
    float thetanZOA[MAX_CLUSTERS];
    float thetanZOD[MAX_CLUSTERS];
};

static_assert(sizeof(ClusterParamsGpu) == 496,
              "ClusterParamsGpu must match WGSL ClusterParams layout");

struct ActiveLink
{
    uint32_t cid;
    uint32_t uid;
    uint32_t linkIdx;
    uint32_t lspReadIdx;
    uint32_t cirCoeOffset;       // element offset into flat cirCoe buffer
    uint32_t cirNormDelayOffset; // element offset into flat cirNormDelay buffer
    uint32_t cirNtapsOffset;     // element offset into flat cirNtaps buffer
    uint32_t freqChanPrbgOffset; // element offset into flat freqChanPrbg buffer
};

// Binding 20 — uni_sim (small-scale kernels)
// Fields must be in exactly this order and with these types to match WGSL.
struct SmallScaleSimConfig
{
    float scSpacingHz;                // +0   sc_spacing_hz
    uint32_t fftSize;                 // +4   fft_size
    uint32_t nPrb;                    // +8   n_prb
    uint32_t nPrbg;                   // +12  n_prbg
    uint32_t nSnapshotPerSlot;        // +16  n_snapshot_per_slot
    uint32_t enablePropagationDelay;  // +20
    uint32_t disableSmallScaleFading; // +24
    uint32_t disablePlShadowing;      // +28
    uint32_t optionalCfrDim;          // +32
    float lambda0;                    // +36
    float _pad0;                      // +40
    float _pad1;                      // +44
}; // = 48 bytes

static_assert(sizeof(SmallScaleSimConfig) == 48,
              "SmallScaleSimConfig size mismatch vs WGSL uni_sim");

// Binding 21 — uni_sys (small-scale kernels)
// This is NOT the same struct as SystemLevelConfigGPU (binding 9).
struct SmallScaleSysConfig
{
    uint32_t enablePropagationDelay;  // +0
    uint32_t disableSmallScaleFading; // +4
    uint32_t disablePlShadowing;      // +8
    uint32_t _pad0;                   // +12
}; // = 16 bytes

static_assert(sizeof(SmallScaleSysConfig) == 16,
              "SmallScaleSysConfig size mismatch vs WGSL uni_sys");

// Binding 2 in calClusterRay / generateCIR.
// Must match WGSL struct SsCmnParams exactly.
struct alignas(16) SsCmnParams
{
    float mu_lgDS[3];
    float sigma_lgDS[3];
    float mu_lgASD[3];
    float sigma_lgASD[3];
    float mu_lgASA[3];
    float sigma_lgASA[3];
    float mu_lgZSA[3];
    float sigma_lgZSA[3];
    float mu_K[3];
    float sigma_K[3];
    float r_tao[3];
    float mu_XPR[3];
    float sigma_XPR[3];
    uint32_t nCluster[3];
    uint32_t nRayPerCluster[3];
    float C_DS[3];
    float C_ASD[3];
    float C_ASA[3];
    float C_ZSA[3];
    float xi[3];
    float C_phi_LOS, C_phi_NLOS, C_phi_O2I;
    float C_theta_LOS, C_theta_NLOS, C_theta_O2I;
    float lgfc;
    float lambda_0;
    uint32_t raysInSubCluster0[10];
    uint32_t raysInSubCluster1[10];
    uint32_t raysInSubCluster2[10];
    uint32_t raysInSubClusterSizes[3];
    uint32_t nSubCluster, nUeAnt, nBsAnt;
    float RayOffsetAngles[20];
};

struct AntPanelConfigGPU
{
    uint32_t nAnt;
    uint32_t antModel;       // 0=isotropic, 1=directional, 2=direct pattern
    uint32_t antSize[5];     // Mg, Ng, M, N, P
    float antSpacing[4];     // dgh, dgv, dh, dv
    float antPolarAngles[2]; // roll angle first/second polarization
    uint32_t thetaOffset;    // offset into flat antTheta buffer (181 entries per panel)
    uint32_t phiOffset;      // offset into flat antPhi buffer   (360 entries per panel)
    uint32_t _pad0;
    uint32_t _pad1;
};

// ─────────────────────────────────────────────────────────────────────────────
// String helper for v24 WGPUStringView
// WGPU_STRLEN = SIZE_MAX → null-terminated, compute length internally
// ─────────────────────────────────────────────────────────────────────────────
static inline WGPUStringView
sv(const char* s)
{
    return {s, WGPU_STRLEN};
}

// ─────────────────────────────────────────────────────────────────────────────
class SlsChanWgpu
{
  public:
    explicit SlsChanWgpu(wgpu::Device device);
    ~SlsChanWgpu() = default;

    // Upload topology/config once (or whenever it changes)
    void uploadCellParams(const std::vector<CellParam>& cells);
    void uploadUtParams(const std::vector<UtParam>& uts);

    // Upload small-scale cell/UT params (different layout from large-scale)
    void uploadCellParamsSS(const std::vector<CellParamSS>& cells);
    void uploadUtParamsSS(const std::vector<UtParamSS>& uts);

    // Upload antenna configs + flat theta/phi tables (call once after topology)
    void uploadAntPanelConfigs(const std::vector<AntPanelConfigGPU>& configs,
                               const std::vector<float>& antThetaFlat,
                               const std::vector<float>& antPhiFlat);

    // Upload CmnLinkParams extended with small-scale fields
    void uploadCmnLinkParamsSmallScale(const SsCmnParams& cmn);

    constexpr int32_t nSite()
    {
        return nSite_;
    }

    constexpr int32_t nX()
    {
        return nX_;
    }

    constexpr int32_t nY()
    {
        return nY_;
    }

    ///////////////////////////////
    //// Large-scale pipeline
    ///////////////////////////////
    void generateCRN(float maxX,
                     float minX,
                     float maxY,
                     float minY,
                     const float corrDistsLos[7],
                     const float corrDistsNlos[6],
                     const float corrDistsO2i[6]);

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

    std::vector<LinkParams> readLinkParams(uint32_t nSite, uint32_t nUT);

    ///////////////////////////////
    //// Small-scale pipeline
    ///////////////////////////////

    // Must be called before calClusterRay / generateCIR / generateCFR.
    // Populates both ssSimConfigBuf_ (binding 20) and ssSysConfigBuf_ (binding 21).
    void uploadSmallScaleConfig(float scSpacingHz,
                                uint32_t fftSize,
                                uint32_t nPrb,
                                uint32_t nPrbg,
                                uint32_t nSnapshotPerSlot,
                                uint32_t enablePropagationDelay,
                                uint32_t disableSmallScaleFading,
                                uint32_t disablePlShadowing,
                                uint32_t optionalCfrDim,
                                float lambda0);

    void calClusterRay(uint32_t nSite, uint32_t nUT);

    void generateCIR(const std::vector<ActiveLink>& activeLinks,
                     uint32_t nActiveLinks,
                     uint32_t nSnapshots,
                     float refTime);

    void generateCFR(const std::vector<ActiveLink>& activeLinks,
                     uint32_t nActiveLinks,
                     uint32_t nSnapshots);

    std::vector<std::complex<float>> readCirCoe(uint32_t nActiveLinks,
                                                uint32_t nSnapshots,
                                                uint32_t nUtAnt,
                                                uint32_t nBsAnt);
    std::vector<std::complex<float>> readFreqChanPrbg(uint32_t nActiveLinks,
                                                      uint32_t nSnapshots,
                                                      uint32_t nUtAnt,
                                                      uint32_t nBsAnt);
    void invalidateOutputBuffers();
    wgpu::BindGroup emptyBg(wgpu::ComputePipeline& pip, uint32_t slot);
    template <typename T>
    std::vector<T> mapReadBuffer(wgpu::Buffer& staging, uint64_t byteSize);
    bool isDead();
    std::vector<ClusterParamsGpu> readClusterParams(uint32_t nSite, uint32_t nUT);
    std::vector<uint32_t> readCirNtaps();
    std::vector<float> readXpr();
    std::vector<float> readPhiNmAoA();
    std::vector<float> readPhiNmAoD();
    std::vector<float> readThetaNmZOA();
    std::vector<float> readThetaNmZOD();

  private:
    wgpu::Device device_;
    wgpu::Queue queue_;
    wgpu::ShaderModule shader_;

    uint32_t nSite_ = 0;
    int32_t nX_ = 0;
    int32_t nY_ = 0;

    wgpu::Buffer makeBuffer(uint64_t size, wgpu::BufferUsage usage, const void* data = nullptr);
    void waitIdle();

    ///////////////////////////////
    //// Large-scale buffers & pipelines
    ///////////////////////////////
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

    ///////////////////////////////
    //// Small-scale buffers & pipelines
    ///////////////////////////////
    wgpu::ComputePipeline clusterRayPipeline_;
    wgpu::ComputePipeline generateCIRPipeline_;
    wgpu::ComputePipeline generateCFRPipeline_;

    wgpu::Buffer clusterParamsBuf_;
    wgpu::Buffer activeLinkBuf_;
    wgpu::Buffer cirCoeBuf_;
    wgpu::Buffer cirNormDelayBuf_;
    wgpu::Buffer cirNtapsBuf_;
    wgpu::Buffer freqChanPrbgBuf_;
    wgpu::Buffer antPanelConfigBuf_;
    wgpu::Buffer antThetaBuf_;
    wgpu::Buffer antPhiBuf_;
    wgpu::Buffer ssSimConfigBuf_; // SmallScaleSimConfig  — binding 20
    wgpu::Buffer ssCmnLinkBuf_;   // SsCmnParams — binding 2 in calClusterRay/generateCIR
    wgpu::Buffer ssSysConfigBuf_; // SmallScaleSysConfig  — binding 21
    wgpu::Buffer ssDispatchBuf_;  // DispatchUniforms     — binding 36
    wgpu::Buffer xpr_Buf_;
    wgpu::Buffer randomPhases_Buf_;
    wgpu::Buffer phi_nm_AoA_Buf_;
    wgpu::Buffer phi_nm_AoD_Buf_;
    wgpu::Buffer theta_nm_ZOA_Buf_;
    wgpu::Buffer theta_nm_ZOD_Buf_;
    // Small-scale cell/UT param buffers (separate from large-scale ones)
    wgpu::Buffer ssCellParamsBuf_; // bindings 23
    wgpu::Buffer ssUtParamsBuf_;   // binding  24

    // Staging for small-scale readback
    wgpu::Buffer cirStagingBuf_;
    wgpu::Buffer cfrStagingBuf_;

    uint32_t ssNUeAnt_ = 0;
    uint32_t ssNBsAnt_ = 0;
    uint32_t ssNPrbg_ = 0;
};
#endif // SLS_CHAN_WGPU_H
