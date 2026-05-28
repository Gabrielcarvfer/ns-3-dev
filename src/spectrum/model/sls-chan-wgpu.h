// This is an AI translated version of the original Nvidia source for GPU compute,
// but ported to WGPU for cross-platform support
// https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/main/testBenches/chanModels/src/sls_chan_src/sls_chan_large_scale_GPU.cu

#ifndef SLS_CHAN_WGPU_H
#define SLS_CHAN_WGPU_H
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// HDF5 include (optional, conditional on SLS_CHAN_HDF5)
#ifdef SLS_CHAN_HDF5
#include <H5Cpp.h>
#endif

#include <webgpu/webgpu.hpp> // C++ wrapper around webgpu.h

// ─────────────────────────────────────────────────────────────────────────────
// Debug logging — disabled by default, opt in with `SLS_DEBUG=1` in the
// environment. Use SLS_LOG(fmt, ...) for printf-style messages; SLS_CERR
// and SLS_COUT for stream-style. None of these produce any output unless
// SLS_DEBUG is set to a non-empty, non-"0" value at process start.
// ─────────────────────────────────────────────────────────────────────────────
inline bool
slsDebugEnabled()
{
    static const bool v = []() {
        const char* e = std::getenv("SLS_DEBUG");
        return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
    }();
    return v;
}

#define SLS_LOG(...)                                                                               \
    do                                                                                             \
    {                                                                                              \
        if (slsDebugEnabled())                                                                     \
        {                                                                                          \
            std::fprintf(stderr, __VA_ARGS__);                                                     \
        }                                                                                          \
    } while (0)

#define SLS_CERR                                                                                   \
    if (!slsDebugEnabled())                                                                        \
    {                                                                                              \
    }                                                                                              \
    else                                                                                           \
        std::cerr
#define SLS_COUT                                                                                   \
    if (!slsDebugEnabled())                                                                        \
    {                                                                                              \
    }                                                                                              \
    else                                                                                           \
        std::cout

// ─────────────────────────────────────────────────────────────────────────────
// Plain-old-data structs that map exactly to the WGSL struct layouts.
// alignas values mirror the WGSL std430 rules.
// ─────────────────────────────────────────────────────────────────────────────

struct alignas(16) Vec3f
{
    float x, y, z, _p;
};

// ── Large-scale CellParam / UtParam ──────────────────────────────────────────
// Used by calLinkParam (bindings 7 & 8 in cal_link_param_kernel).
// Layout MUST match the WGSL `CellParam` struct exactly. Per the WGSL spec,
// `vec3<f32>` has alignment 16 *but size 12* — i.e. the next field can start
// 12 bytes after a vec3 begins, not 16. So we model vec3 with `float[3]` and
// insert explicit padding before each one to satisfy alignment. Total struct
// size is 80 bytes.
struct alignas(16) CellParam
{
    uint32_t cid;                       // 0
    uint32_t siteId;                    // 4
    uint32_t _pad_a[2];                 // 8  — pad to 16 for `loc`
    float loc[3];                       // 16-27 (vec3<f32>, 12 bytes)
    uint32_t antPanelIdx;               // 28 — fits in the implicit-pad slot
    float antPanelOrientation[3];       // 32-43 (vec3<f32>, 12 bytes)
    uint32_t monostaticInd;             // 44
    uint32_t _pad1;                     // 48
    uint32_t _pad2;                     // 52
    uint32_t secondAntPanelIdx;         // 56
    uint32_t _pad_b;                    // 60 — pad to 64 for next vec3
    float secondAntPanelOrientation[3]; // 64-75
    // alignas(16) pads to 80
};

static_assert(sizeof(CellParam) == 80, "CellParam size must match WGSL layout (80 B)");
static_assert(offsetof(CellParam, loc) == 16, "CellParam.loc must be at offset 16");
static_assert(offsetof(CellParam, antPanelIdx) == 28, "CellParam.antPanelIdx offset 28");
static_assert(offsetof(CellParam, antPanelOrientation) == 32, "antPanelOrientation offset 32");
static_assert(offsetof(CellParam, monostaticInd) == 44, "monostaticInd offset 44");
static_assert(offsetof(CellParam, secondAntPanelIdx) == 56, "secondAntPanelIdx offset 56");
static_assert(offsetof(CellParam, secondAntPanelOrientation) == 64,
              "secondAntPanelOrientation offset 64");

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
// Used by calClusterRay / generateCIR / generateCFR (bindings 23 & 24).
// Layout MUST match the WGSL SspCellParam / SspUtParam structs. The
// element stride for arrays of these in storage buffers is 32 / 48 B in
// the running wgpu-native implementation (16-byte structure stride). The
// WGSL structs include explicit trailing padding to land at that stride;
// the host structs use `alignas(16)` to do the same.
struct alignas(16) CellParamSS
{
    uint32_t antPanelIdx;
    float antPanelOrientation[3]; // [theta_tilt, phi_tilt, zeta_offset]
    uint32_t _pad0;
};

static_assert(sizeof(CellParamSS) == 32, "CellParamSS must be 32 bytes to match WGSL stride");

struct alignas(16) UtParamSS
{
    uint32_t antPanelIdx;
    uint32_t outdoor_ind;         // 1 = outdoor, 0 = indoor (O2I)
    float antPanelOrientation[3]; // [theta_tilt, phi_tilt, zeta_offset]
    float velocity[3];
    uint32_t _pad0;
};

static_assert(sizeof(UtParamSS) == 48, "UtParamSS must be 48 bytes to match WGSL stride");

// ── Large-scale LinkParams ────────────────────────────────────────────────────
struct LinkParamsHdf5
{
    uint32_t cid; ///< Serving cell (sector) ID
    float d2d;
    float d2d_in;
    float d2d_out;
    float d3d;
    float d3d_in;
    float d3d_out;
    float phi_LOS_AOD;
    float theta_LOS_ZOD;
    float phi_LOS_AOA;
    float theta_LOS_ZOA;
    uint32_t losInd;
    float pathloss;
    float SF;
    float K;
    float DS;
    float ASD;
    float ASA;
    float mu_lgZSD;
    float sigma_lgZSD;
    float mu_offset_ZOD;
    float ZSD;
    float ZSA;
    float delta_tau;
};

static_assert(sizeof(LinkParamsHdf5) == 96, "LinkParamsHdf5 size mismatch");

// ── Large-scale LinkParams ────────
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
    uint32_t updatePL, updateAllLSPs, updateLosState, updateOptionalPL;
    int32_t nX, nY;
};

static_assert(sizeof(LinkParamUniforms) == 52, "LinkParamUniforms size mismatch with WGSL");

// Binding 9  — sys_config (large-scale kernel)
struct SystemLevelConfigGPU
{
    uint32_t scenario;
    uint32_t enable_propagation_delay;
    uint32_t o2i_bldg;
    uint32_t o2i_car;
    float force_los_indoor;  // -1.0 = use formula
    float force_los_outdoor; // -1.0 = use formula
    float _p2[2];
};

static_assert(sizeof(SystemLevelConfigGPU) == 32, "SystemLevelConfigGPU size mismatch");

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
    float mu_lgDT[4], sigma_lgDT[4];
    float lgfc;
    float _pad[2];
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
// Must match WGSL struct SsCmnParams exactly (520 bytes, no extra padding).
struct SsCmnParams
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
    float mu_lgDT[3];    // mean of log10(delta_tau) by scenario (NLOS=0, LOS=1, O2I=2)
    float sigma_lgDT[3]; // stddev of log10(delta_tau) by scenario
    uint32_t raysInSubCluster0[10];
    uint32_t raysInSubCluster1[10];
    uint32_t raysInSubCluster2[10];
    uint32_t raysInSubClusterSizes[3];
    uint32_t nSubCluster, nUeAnt, nBsAnt;
    float RayOffsetAngles[20];
};

// AntPanelConfigGPU: layout must match WGSL `AntPanelConfig` (64 bytes).
// The previous version had two trailing u32 pads (68 bytes), which made the
// UE panel (index 1) read with a 4-byte shift — the GPU saw nUtAnt = 0 and
// silently skipped the UE-antenna loop in calc_los_coeff / calc_ray_coeff,
// producing zero CIR for every link.
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
};

static_assert(sizeof(AntPanelConfigGPU) == 64, "AntPanelConfigGPU must be 64 B (WGSL stride)");

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
    SlsChanWgpu();
    ~SlsChanWgpu() = default;

    // Upload topology/config once (or whenever it changes).
    // `nSectorPerSite` is the number of sector cells per radio site; the
    // total number of sites (used to size per-site buffers in
    // `generateCRN`) is computed as `cells.size() / nSectorPerSite`.
    // Defaults to 3 to preserve the Phase-1 calibration harness's prior
    // behaviour; the ThreeGppChannelModel batch back-end passes 1.
    void uploadCellParams(const std::vector<CellParam>& cells, uint32_t nSectorPerSite = 3);
    void uploadUtParams(const std::vector<UtParam>& uts);

    // Override the system-level config that calLinkParam would otherwise
    // lazily initialise to "UMa / formula LOS / no O2I". Pass force_los_*
    // as 1.0 to force LOS for that UT class, 0.0 to force NLOS, or
    // -1.0 to use the TR 38.901 LOS-probability formula. Call this
    // BEFORE calLinkParam; subsequent calls overwrite the buffer.
    void setSystemLevelConfig(uint32_t scenario,
                              uint32_t enablePropagationDelay,
                              uint32_t o2iBldg,
                              uint32_t o2iCar,
                              float forceLosIndoor,
                              float forceLosOutdoor);

    // Override the GPU's sim-level center frequency. calLinkParam will
    // otherwise lazily initialise this to 3.5 GHz. Call BEFORE
    // calLinkParam if you want path-loss to use a different fc.
    // The value is also cached and emitted into the HDF5 file by
    // `saveSlsChanToHdf5`.
    void setCenterFrequencyHz(float centerFreqHz);

    // Check if running in GPU mode (always true now)
    bool isCpuOnlyMode() const
    {
        return false;
    }

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
                     const float corrDistsLos[8],
                     const float corrDistsNlos[7],
                     const float corrDistsO2i[7]);

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
                      bool updateOptionalPL,
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
    void generateCFRBatched(const std::vector<ActiveLink>& activeLinks,
                            uint32_t nActiveLinks,
                            uint32_t nSnapshots);
    void invalidateOutputBuffers();
    void readFreqChanPrbgBatched(std::vector<std::complex<float>>& outBuf);
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

    // Dispatch gen_channel_matrix_kernel. Inputs already on the GPU
    // (linkParamsBuf_, clusterParamsBuf_, activeLinkBuf_, the packed
    // cluster outputs at clusterOutputsBuf_); output goes into a
    // dedicated channelMatrixBuf_ sized for nActiveLinks * uSize *
    // sSize * MAT_MAX_PAGES vec2<f32>. Caller passes per-link
    // (cluster1st, cluster2nd, numReducedCluster); the kernel
    // currently emits zeros into the matrix and the host falls back
    // to the CPU build (math still in progress).
    void genChannelMatrix(const std::vector<ActiveLink>& activeLinks,
                          uint32_t nActiveLinks,
                          uint32_t uSize,
                          uint32_t sSize,
                          uint32_t numOverallCluster,
                          uint32_t numReducedCluster,
                          uint32_t nRays,
                          uint8_t cluster1st,
                          uint8_t cluster2nd);

    // Maximum per-link page count the channel-matrix kernel
    // writes -- mirrors MAT_MAX_PAGES in sls-chan.wgsl. Output
    // buffer size is nLinks * uSize * sSize * MAX_OVERALL_CLUSTER
    // vec2<f32>.
    static constexpr uint32_t kMatMaxPages = 24u;

    // Read back the per-link channel matrix block sized
    // uSize * sSize * MAT_MAX_PAGES per link. Caller slices the
    // result into per-link Complex3DVector(uSize, sSize, numPages).
    std::vector<std::complex<float>> readChannelMatrix(uint32_t nLinks,
                                                       uint32_t uSize,
                                                       uint32_t sSize);

  private:
    // Gather one of the per-link sub-views out of the packed
    // clusterOutputsBuf_. linkOffF32 is the f32 offset into each
    // link's PACKED_LINK_STRIDE slab; perLinkLenF32 is how many
    // f32 to copy per link. Public read* helpers delegate to this.
    std::vector<float> sliceClusterOutput(uint32_t linkOffF32, uint32_t perLinkLenF32);

  public:
    // 16 f32 per link, populated by generate_cir_kernel. Layout described in
    // sls-chan.wgsl near the `cir_dbg` binding.
    std::vector<float> readCirDebug(uint32_t nActiveLinks);

    /**
     * Scene-level metadata to write alongside the channel state. The
     * pipeline doesn't actually use any of these — they're recorded so
     * downstream analyzers (e.g. analysis_channel_stats.py) can see
     * how the deployment was constructed.
     */
    struct SceneMeta
    {
        float isd = 0.0f;
        float bsHeight = 0.0f;
        float minBsUeDist2d = 0.0f;
        float maxBsUeDist2dIndoor = 0.0f;
        float indoorUtPercent = 0.0f;
        float bandwidthHz = 0.0f;
    };

#ifdef SLS_CHAN_HDF5
    /**
     * Write the full GPU channel state to `filename` in the
     * NVIDIA-compatible HDF5 layout. Reads back every output buffer
     * via the existing read*() methods and pulls topology / config
     * inputs from the host-side caches populated by `upload*` and
     * `uploadSmallScaleConfig`. Small-scale buffers that haven't been
     * computed yet (e.g. when the caller only ran `calLinkParam`)
     * are written as empty datasets rather than skipped, so the
     * resulting file's schema is always the same.
     *
     * Only compiled when the spectrum library is built with HDF5
     * (controlled by the `HDF5_FOUND` branch in
     * `src/spectrum/CMakeLists.txt`).
     */
    void saveSlsChanToHdf5(const std::string& filename, const SceneMeta& meta);
#endif

  private:
    // Keep the instance alive for the lifetime of the device. Dawn's
    // `Instance::processEvents()` is the only way to fire
    // `AllowProcessEvents`-mode callbacks (mapAsync, onSubmittedWorkDone).
    // wgpu-native v24 doesn't need this (its `Device::poll` handles
    // both), so the field is empty (`wgpu::Instance{}`) there.
    wgpu::Instance instance_;
    wgpu::Device device_;
    wgpu::Queue queue_;
    wgpu::ShaderModule shader_;

    uint32_t nSite_ = 0;
    int32_t nX_ = 0;
    int32_t nY_ = 0;
    // Metres per CRN grid cell. The CRN sizing scales as O(area / step)^2, so a
    // coarser step keeps the per-buffer footprint bounded for spec-sized
    // deployments. Default matches the CUDA reference (10 m).
    float crnStep_ = 10.0f;

    // generateCRN cache: every per-tick call to generateCRN spends most of
    // its time inside the convolve_crn_kernel on a grid that, for any
    // single sim, is a function of (deployment bounds, site count, the
    // correlation-distance constants, crnStep_). For a stationary
    // deployment those don't change across consecutive UpdateChannel
    // ticks, so the kernel is producing the same field every tick. When
    // UEs move slightly the new bounds usually fit inside the
    // previously-generated grid, so cache validity is "current_bounds is
    // a subset of cached_bounds AND other inputs unchanged" rather than
    // exact equality. On a miss we re-generate with extra padding so the
    // next few ticks' motion still hits the cache.
    //
    // Per 3GPP TR 38.901, large-scale parameters are spatially-correlated
    // random fields that are constant within a channel consistency window
    // (~100 ms), so re-using the field across ticks within that window is
    // statistically equivalent to drawing fresh shadowing each tick (and
    // the previous code's per-tick draws were already wasting statistical
    // information by ignoring within-window correlation).
    float crnCacheMaxX_ = 0.f;
    float crnCacheMinX_ = 0.f;
    float crnCacheMaxY_ = 0.f;
    float crnCacheMinY_ = 0.f;
    uint32_t crnCacheNSite_ = 0;
    uint64_t crnCacheCorrKey_ = 0;
    bool crnCacheValid_ = false;

    wgpu::Buffer makeBuffer(uint64_t size, wgpu::BufferUsage usage, const void* data = nullptr);
    void waitIdle();
#ifdef WEBGPU_BACKEND_DAWN
    // Pump Dawn's event queue. Used inside busy-wait loops on
    // `AllowProcessEvents`-mode callbacks (mapAsync, etc).
    void dawnPumpEvents();
#endif

    // Compile a single compute kernel into a ComputePipeline. Used both
    // eagerly (large-scale kernels in the constructor) and lazily
    // (small-scale kernels only when their owning method is called) so
    // backends that can't compile the small-scale kernels (Vulkan /
    // OpenGL with some drivers and the current wgpu-native v24) can
    // still be used for the LSP-only ns-3 batch path.
    wgpu::ComputePipeline makePipeline(const char* entryPoint);

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
    // Combined CRN data + offsets buffer. The old 3+3 split blew past
    // the WebGPU per-stage SSBO limit of 10; concatenating reduces
    // cal_link_param_kernel's storage-buffer count from 13 to 9.
    // Layout of crnDataBuf_:    [LOS region | NLOS region | O2I region]
    // Layout of crnOffsetsBuf_: [losOffs   | nlosOffs   | o2iOffs   ]
    // with offsets pre-baked to address into the combined data buffer
    // (so losOffs[i] is in [0, lossSize), nlosOffs[i] in [losSize,
    // losSize+nlosSize), etc).
    wgpu::Buffer crnDataBuf_;
    wgpu::Buffer crnOffsetsBuf_;
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
    static constexpr uint32_t CFR_BATCH_SIZE = 1024u;
    wgpu::Buffer antPanelConfigBuf_;
    wgpu::Buffer antThetaBuf_;
    wgpu::Buffer antPhiBuf_;
    wgpu::Buffer ssSimConfigBuf_; // SmallScaleSimConfig  — binding 20
    wgpu::Buffer ssCmnLinkBuf_;   // SsCmnParams — binding 2 in calClusterRay/generateCIR
    wgpu::Buffer ssSysConfigBuf_; // SmallScaleSysConfig  — binding 21
    wgpu::Buffer ssDispatchBuf_;  // DispatchUniforms     — binding 36
    // Single buffer that packs six per-link f32 arrays (xpr,
    // randomPhases, phi_nm_AoA/AoD, theta_nm_ZOA/ZOD) into one
    // PACKED_LINK_STRIDE-sized slab per link. Lets the cluster-ray
    // kernel fit Dawn-D3D12's 10 SSBO-per-stage hard cap. The
    // readXpr / readPhiNmA*/ readThetaNmZ* helpers slice this back
    // into independent views on the host side.
    wgpu::Buffer clusterOutputsBuf_;
    // Output of gen_channel_matrix_kernel: per-link block of
    // uSize * sSize * MAT_MAX_PAGES vec2<f32>. Sized at the
    // first genChannelMatrix() call.
    wgpu::Buffer channelMatrixBuf_;
    wgpu::ComputePipeline channelMatrixPipeline_;
    wgpu::Buffer matrixDispatchBuf_;
    uint32_t channelMatrixCfgUSize_{0};
    uint32_t channelMatrixCfgSSize_{0};
    uint32_t channelMatrixCfgNLinks_{0};
    // Small-scale cell/UT param buffers (separate from large-scale ones)
    wgpu::Buffer ssCellParamsBuf_; // bindings 23
    wgpu::Buffer ssUtParamsBuf_;   // binding  24

    // Debug buffer for generate_cir_kernel — 16 f32 per link (binding 21).
    wgpu::Buffer cirDbgBuf_;

    // Staging for small-scale readback
    wgpu::Buffer cirStagingBuf_;
    wgpu::Buffer cfrStagingBuf_;

    // Batched CFR results storage
    std::vector<std::complex<float>> m_cfrBatchedResult_;
    uint32_t m_cfrBatchedNActiveLinks_ = 0;
    uint32_t m_cfrBatchedNSnapshots_ = 0;

    uint64_t m_maxGpuBuffer_ = 0;

    uint32_t ssNUeAnt_ = 0;
    uint32_t ssNBsAnt_ = 0;
    uint32_t ssNPrbg_ = 0;

    // ── Cached host-side copies of the most recent upload* inputs ──
    // Used by saveSlsChanToHdf5() to write topology metadata back to
    // the HDF5 file without forcing every caller to thread these
    // values through a 30-arg function. Populated automatically by
    // the corresponding upload* / config methods; missing entries
    // (e.g. small-scale buffers when the caller only ran the LSP
    // pipeline) are written to the HDF5 as empty datasets.
    std::vector<CellParam> cellsCache_;
    std::vector<UtParam> utsCache_;
    std::vector<CellParamSS> cellsSSCache_;
    std::vector<UtParamSS> utsSSCache_;
    std::vector<AntPanelConfigGPU> antCfgsCache_;
    SsCmnParams ssCmnCache_{};
    bool ssCmnCacheValid_ = false;
    uint32_t nSectorPerSiteCache_ = 1;
    uint32_t nUTCache_ = 0;
    float scSpacingHzCache_ = 0.0f;
    uint32_t fftSizeCache_ = 0;
    uint32_t nPrbCache_ = 0;
    uint32_t nSnapshotPerSlotCache_ = 0;
    float centerFreqHzCache_ = 0.0f;
    std::vector<ActiveLink> activeLinksCache_;
    uint32_t nSnapshotsCache_ = 0;
};
#endif // SLS_CHAN_WGPU_H
