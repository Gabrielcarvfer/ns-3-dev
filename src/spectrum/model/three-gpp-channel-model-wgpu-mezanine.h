//
// Created by gabri on 04/04/2026.
//

#ifndef THREE_GPP_CHANNEL_MODEL_WGPU_MEZANINE_H
#define THREE_GPP_CHANNEL_MODEL_WGPU_MEZANINE_H

#include "three-gpp-channel-model.h"

#include "ns3/phased-array-model.h"

#include <array>
#include <memory>
#include <optional>

class SlsChanWgpu; // forward declaration because we use an opaque pointer here

namespace ns3
{
/***
 * @ingroup spectrum
 *
 * @brief 3GPP Channel Model WGPU Mezanine
 * The only reason for this class to exist is to be able to configure
 * the 3GPP WGPU channel model to load settings from the CPU 3GPP
 * channel model, without exposing WGPU channel model internals.
 */
class ThreeGppChannelModelWgpuMezanine : public ThreeGppChannelModel
{
  public:
    ThreeGppChannelModelWgpuMezanine();
    ~ThreeGppChannelModelWgpuMezanine();
    static TypeId GetTypeId();
    void UpdateChannel();
    // Phase D: decouple channel-state advancement from PRX events.
    // The mez schedules itself on a periodic NS-3 timer and refreshes
    // the GPU caches independent of when PRX evaluates. PRX events
    // become pure cache consumers. NR reconfigurations (antenna
    // reshuffle from MaxRsrpInitialAssociation, BWP/numerology change
    // on MIB/SIB1) are caught at the next refresh tick.
    void PeriodicRefresh();
    // Override the base batch-fresh hook so the small-scale GPU
    // pipeline runs at the start of each tick instead of waiting for
    // an external 10 ms self-scheduled loop. Base does LSP draw; this
    // override additionally calls UpdateChannel() to write GPU cluster
    // delays / ray angles / XPR into m_channelParamsMap, so the
    // per-link GetNewChannel that follows skips the CPU small-scale
    // draw entirely.
    void EnsureBatchFresh() override;
    // Override GetChannel so we control both the matrix AND the
    // channelParams lookup as a unit. The base class's GetChannel
    // calls GenerateChannelParameters mid-call (with UpdatePeriod=0
    // it fires on every eval), which can leave the cached matrix
    // out of sync with the freshly-written m_alpha / m_D / m_angle
    // vectors -- PRX::CalcBeamformingGain then asserts
    // numCluster <= m_alpha.size(). We short-circuit when the
    // mezanine has populated both maps with a matching (params,
    // matrix) pair this tick.
    Ptr<const ChannelMatrix> GetChannel(
        Ptr<const MobilityModel> aMob,
        Ptr<const MobilityModel> bMob,
        Ptr<const PhasedArrayModel> aAntenna,
        Ptr<const PhasedArrayModel> bAntenna) override;
    Ptr<MatrixBasedChannelModel::ChannelMatrix> GetNewChannel(
        Ptr<const ThreeGppChannelParams> channelParams,
        Ptr<const ParamsTable> table3gpp,
        Ptr<const MobilityModel> sMob,
        Ptr<const MobilityModel> uMob,
        Ptr<const PhasedArrayModel> sAntenna,
        Ptr<const PhasedArrayModel> uAntenna) const override;

    // Surfaces longTerm computed by gen_long_term_kernel during the
    // most recent UpdateChannel. PRX::GetLongTerm calls this before
    // falling through to CalcLongTerm; on a match (channelMatrix is
    // the cached one, beam weights identical to the snapshot) we
    // bypass the CPU compute entirely.
    Ptr<const Complex3DVector> GetCachedLongTerm(
        Ptr<const ChannelMatrix> channelMatrix,
        Ptr<const PhasedArrayModel> sAnt,
        Ptr<const PhasedArrayModel> uAnt,
        const PhasedArrayModel::ComplexVector& sW,
        const PhasedArrayModel::ComplexVector& uW) const override;

    // Override the GenSpec hook to dispatch gen_spec_chan_kernel against
    // the longTerm matrix this same back-end produced in the most
    // recent UpdateChannel. Returns nullptr (and PRX falls back to CPU
    // GenSpec) when this link has no GPU-built longTerm entry.
    Ptr<Complex3DVector> TryGenSpectrumChannelMatrix(
        Ptr<const ChannelMatrix> channelMatrix,
        Ptr<const ChannelParams> channelParams,
        Ptr<const Complex3DVector> longTerm,
        Ptr<const SpectrumValue> inPsd,
        const std::vector<std::complex<double>>& delayT,
        const std::vector<double>& sqrtVit,
        uint32_t numRb,
        uint8_t numRxPorts,
        uint8_t numTxPorts,
        bool isReverse,
        uint64_t sWBeamHash,
        uint64_t uWBeamHash,
        bool scalarPsdOk) const override;

    // Pre-seed the RB frequency table from a representative PSD so that
    // the first UpdateChannel call can run GenSpecBatch immediately
    // rather than waiting for the first PRX eval to capture rb_freqs.
    // Call this (with txPsd) after pre-warming GetChannel for all links
    // but before the first Simulator::Run() tick to eliminate tick-0
    // TryGenSpecCpuMiss calls (Opt C).
    void CaptureRbFreqs(Ptr<const SpectrumValue> psd);

    // Bypass GPU batch dispatch for a phase of the simulation where GPU
    // pre-computation would create stale cache entries (e.g. maxRSRP initial
    // UE attachment uses temporary single-port antenna copies that differ from
    // the user-configured antennas used during real traffic).
    //
    // When true:
    //   - TryGenSpectrumChannelMatrix returns nullptr immediately (CPU fallback).
    //   - EnsureBatchFresh skips UpdateChannel (no GPU batch dispatch or cache writes).
    //
    // NR initial-association code should call SetBypassGpuBatch(true) before the
    // maxRSRP scan and SetBypassGpuBatch(false) after. The real simulation then
    // starts with a clean cache that only holds entries for the configured antennas.
    void SetBypassGpuBatch(bool bypass);

    // --- Parallel uncached batch path for initial attachment and REM maps ----
    //
    // Context: NR initial attachment (maxRSRP) and REM-map generation scan
    // every (gNB, UE) pair through many beam angles.  Going through the normal
    // per-link DoCalcRxPowerSpectralDensity path serialises these evaluations;
    // even with the bypass flag above the cost is pure CPU.
    //
    // This method accepts a flat list of (link, beam) descriptors, fires ONE
    // GPU genSpecBatch dispatch covering all of them in parallel, waits for
    // completion, and returns the per-link isotropic power -- without touching
    // m_gpuChanSpctMap.  Results are not cached.
    //
    // Intended callers:
    //   - NrInitialAssociation::PopulateRsrps (replace the per-link loop)
    //   - NrRemHelper (REM map generation)
    //
    // TODO: not yet implemented -- returns an empty vector and falls through to
    // the per-link CPU path.  Implement when profiling shows initial attachment
    // is a bottleneck (it is currently masked by the real-traffic GPU gain).
    struct UncachedBatchLink
    {
        Ptr<const MobilityModel> sMob;
        Ptr<const MobilityModel> uMob;
        Ptr<const PhasedArrayModel> sAnt; ///< may have any beam vector already set
        Ptr<const PhasedArrayModel> uAnt;
    };
    /// @return per-link isotropic received power in the same order as @p links,
    ///         or an empty vector if the GPU path is unavailable.
    std::vector<double> ComputeUncachedBatch(const std::vector<UncachedBatchLink>& links,
                                             Ptr<const SpectrumValue> txPsd);

  private:
    std::unique_ptr<SlsChanWgpu> m_wgpuChannel;
    mutable std::unordered_map<uint32_t, Ptr<const PhasedArrayModel>> m_antennaIdToObjectMap;
    // Dedup the per-link EnsureBatchFresh calls within the same
    // simulator tick so the small-scale GPU work runs at most once per
    // tick instead of once per link. std::optional so the very first
    // tick still triggers a refresh (Time::Zero() is a valid tick).
    std::optional<Time> m_lastMezBatchTime;

    // Phase D: periodic-refresh state.
    // Period at which PeriodicRefresh runs. Defaults to the channel
    // update period (m_updatePeriod) on the base, but is decoupled --
    // setting MezRefreshPeriod=0 disables the periodic loop and falls
    // back to the on-demand EnsureBatchFresh path.
    Time m_refreshPeriod;
    bool m_periodicRefreshScheduled = false;

    // Opt H: multi-tick UpdateChannel reuse.
    // EnsureBatchFresh calls UpdateChannel only once every m_ucPeriod
    // simulator ticks. Default=1 (every tick, same as before). Higher
    // values amortize the UC cost over more PRX evals at the cost of
    // using a stale channel for up to (m_ucPeriod-1) ticks.
    uint32_t m_ucPeriod{1};
    uint32_t m_ucTickCount{0};
    // UpdateChannel simtime gate.
    // Mirrors base-class semantics: m_updatePeriod=0 means static channel
    // (UpdateChannel fires exactly ONCE, at the first non-empty tick), and
    // m_updatePeriod>0 fires every m_updatePeriod of simulated time.
    // Negative sentinel means "never fired yet" so the first call always
    // triggers a dispatch regardless of m_updatePeriod.
    // This prevents staging-buffer accumulation when EnsureBatchFresh is
    // called thousands of times per simtime-second (NR with many UEs).
    Time m_lastUCSimTime{Seconds(-1.0)};

    // GPU-built longTerm cache. Keyed by (sAntId, uAntId) the same way
    // m_channelMatrixMap is. Each entry holds the longTerm matrix the
    // GPU kernel produced this tick + the snapshot of (sW, uW) that
    // went into it -- GetCachedLongTerm only returns a hit when both
    // the channelMatrix m_generatedTime matches AND the caller's beam
    // weights still equal the snapshot.
    struct GpuLongTermEntry
    {
        Ptr<const Complex3DVector> longTerm;
        // Hash the beam-weight snapshot rather than copying the
        // ComplexVector. The CPU PRX::GetLongTerm cache stores the
        // full vector (it needs operator!= for the equality check),
        // but in this back-end we control both ends -- we hash the
        // current sW/uW at insertion and re-hash at lookup time. A
        // 64-bit collision on two distinct beam vectors yielding a
        // false-positive cache hit is ~1 in 1.8e19 and not a concern
        // for any realistic sim. Avoids the 1900 x (sW+uW copy) per
        // tick the populate loop was paying in Debug.
        uint64_t sWHash{0};
        uint64_t uWHash{0};
        size_t sWSize{0};
        size_t uWSize{0};
        Time generatedTime;
        // Index into SlsChanWgpu's longTermOutBuf_ for this link, so
        // the GenSpec hook can address the per-link slab.
        uint32_t gpuLinkIdx{0};
        // longTerm shape -- the GenSpec kernel needs these to walk
        // the longTerm buffer's column-major layout correctly. uPorts
        // and sPorts come from the BS/UE antenna geometry at the
        // moment of dispatch, so they're stable for the lifetime of
        // the entry.
        uint32_t ltUPorts{0};
        uint32_t ltSPorts{0};
    };
    mutable std::unordered_map<uint64_t, GpuLongTermEntry> m_gpuLongTermMap;

    static uint64_t HashComplexVector(const PhasedArrayModel::ComplexVector& v);

    // Captured RB band layout for the batched GenSpec path. The first
    // PRX eval ever runs through the per-eval hook with inPsd in hand;
    // we snapshot the subband centre frequencies into m_batchRbFreqs
    // there, return nullptr (so PRX uses CPU GenSpec for that one
    // eval), and the NEXT UpdateChannel uses the snapshot to dispatch
    // gen_spec_batch_kernel once for all active links. Subsequent
    // evals look up their pre-built chanSpct_unscaled, multiply by
    // sqrt(PSD[rb]), and return. RB freqs are checked for drift on
    // every cache lookup -- mismatch -> fall back to CPU.
    mutable std::vector<float> m_batchRbFreqs;
    mutable uint64_t m_batchRbFreqsHash{0};
    // Per-link chanSpct_unscaled metadata indexed by matrixKey.
    // The actual float data lives in m_specBatchFlat[batchIdx * m_specBatchPerLink].
    // Storing only the index (not the data) avoids a full second copy of the
    // 4+ GB spectrum-channel flat buffer.
    struct GpuChanSpctEntry
    {
        /// Plain antenna-pair key (un-mixed). The map itself is keyed by
        /// MixBeamKey(matrixKey, sWHash, uWHash); this field lets the
        /// eviction sweep and per-link invalidation find a link's entries.
        uint64_t matrixKey{0};
        /// Start COMPLEX-element index into the owning flat H array for slot 0.
        uint32_t specHBaseIdx{0};
        uint32_t numRxPorts{0};
        uint32_t numTxPorts{0};
        uint32_t numRb{0};
        Time generatedTime;
        uint64_t rbFreqsHash{0};
        /// Identity-cached output state for one (representation, direction)
        /// combination of this entry. Repeat evaluations of a link within a
        /// slot overwhelmingly carry the SAME input PSD (full-buffer data
        /// with the pathloss frozen for the loss-cache period), so when the
        /// (batch, slot, psd-fingerprint) triple matches the previous build
        /// the cached output Ptr is returned untouched — a ~0.05 us pointer
        /// return instead of a 0.4-6.8 us rebuild. Rebuilds rotate through a
        /// 2-deep ring so an output Ptr still referenced by a LIVE signal in
        /// NrInterference (which holds spectrumChannelMatrix for the whole
        /// signal duration) is not overwritten by the very next rebuild.
        struct HitOutCache
        {
            std::array<Ptr<MatrixBasedChannelModel::Complex3DVector>, 2> ring;
            uint8_t ringIdx{0};
            uint32_t lastSlot{0xFFFFFFFFu}; ///< slot of the cached build
            Time lastBatch;                 ///< entry generatedTime at build
            double p0{-1.0};                ///< inPsd fingerprint: first RB
            double pm{-1.0};                ///< inPsd fingerprint: middle RB
            double pl{-1.0};                ///< inPsd fingerprint: last RB
        };
        /// Full per-port H outputs (DL orientation / UL transpose).
        mutable HitOutCache outFwd;
        mutable HitOutCache outRev;
        /// (1,1,numRb) scalar-power outputs for 1-rx-port, no-precoding
        /// evals: H[0,0,rb] = sqrt(pow[rb] * inPsd[rb] / numTxCaller).
        mutable HitOutCache outScalarFwd;
        mutable HitOutCache outScalarRev;
        /// Start FLOAT index of this link's slot-0 scalar-power row in the
        /// owning pow array (batch: m_specPowFlat/m_specPowRevFlat with
        /// powPerSlotStride; CPU section: m_specPowCpuFlat/...RevFlat).
        uint32_t powBaseIdx{0};
        /// FLOAT elements between consecutive slots in the batch pow
        /// arrays: nBucketLinks * numRb.
        uint32_t powPerSlotStride{0};
        /// M>1 batched-slot metadata. batchM=1 means only slot 0 is valid.
        uint32_t batchM{1};
        /// perSlotStride is the number of COMPLEX elements between
        /// consecutive slots in m_specHFlat: stride = nBucketLinks * numRb * rxtx.
        uint32_t perSlotStride{0};
        /// Simulation time (seconds) at which slot 0 was computed.
        double batchStartTimeSec{-1.0};
        /// Duration of one slot in seconds (= MezSlotDuration attribute).
        double slotDurationSec{0.0};
        /// True when specHBaseIdx indexes the CPU-miss (D-3b) flat
        /// arrays rather than the GPU batch arrays. CPU-miss entries get
        /// disjoint storage so they can never collide with the batch
        /// scatter regions (the old shared-array scheme allocated by map
        /// size and silently corrupted slot 1..M-1 sections when batchM > 1,
        /// and stale rebucketed entries could overwrite live batch data).
        bool cpuSection{false};
    };
    mutable std::unordered_map<uint64_t, GpuChanSpctEntry> m_gpuChanSpctMap;
    /// Per-link, per-slot complex per-port channel matrices in canonical
    /// (DL) orientation: H[rx,tx,rb] with the Complex3DVector page layout
    /// (rb pages, rx-fast columns). One matrix serves both orientations
    /// (UL is the transpose). Layout: slot s, bucket link bi ->
    /// m_specHFlat[s*perSlotStride + bi*numRb*rxtx + rb*rxtx + tx*nRx + rx].
    mutable std::vector<std::complex<float>> m_specHFlat;
    /// CPU-miss (D-3b) section, disjoint from the batch array above.
    /// Layout: cpu slot i -> [i*numRb*rxtx .. (i+1)*numRb*rxtx).
    mutable std::vector<std::complex<float>> m_specHCpuFlat;
    /// Scalar beamformed power per (link, slot, rb): fwd = sum_rx|sum_tx H|^2,
    /// rev = sum_tx|sum_rx H|^2. Batch arrays use per-bucket regions
    /// (m_powBucketBase allocator); CPU arrays use a running offset.
    mutable std::vector<float> m_specPowFlat;
    mutable std::vector<float> m_specPowRevFlat;
    mutable std::vector<float> m_specPowCpuFlat;
    mutable std::vector<float> m_specPowCpuRevFlat;
    /// Per-refresh running base offsets (reset alongside m_specHBucketBase).
    size_t m_powBucketBase{0};
    /// Running offset for CPU-miss pow rows.
    mutable size_t m_cpuMissPowOffset{0};
    /// Spatial consistency: grid slot (lspReadIdx) that produced each
    /// paramsKey's channel params at the previous refresh. A drift
    /// (procedure A) is only valid when the slot mapping is unchanged.
    std::unordered_map<uint64_t, uint32_t> m_prevLspReadIdx;
    /// Next free element offset in m_specHCpuFlat. Slots vary in size
    /// (numRb * rxtx differs per antenna bucket), so allocation is a
    /// running offset rather than fixed-size slots.
    mutable size_t m_cpuMissNextOffset{0};
    /// Per-refresh running base offset into m_specHFlat: each antenna
    /// bucket claims a disjoint region (reset at the top of UpdateChannel's
    /// bucket loop). Without this, multi-bucket scenarios clobbered each
    /// other's slabs.
    size_t m_specHBucketBase{0};
    mutable uint32_t m_reducedPowNumRb{0}; ///< numRb for the current batch

    static uint64_t HashFloatVector(const std::vector<float>& v);

    /// Number of future slots to pre-compute per PeriodicRefresh tick.
    /// Default 1 = current behaviour (only the current slot).
    /// M>1 dispatches genSpecBatch M times with Doppler extrapolated
    /// M-1 slots ahead, caching M reducedPow arrays per link.
    uint32_t m_batchM{1};
    /// Duration of one NR slot for M>1 lookahead. Default 0.5 ms (mu=1).
    Time m_slotDuration{MicroSeconds(500)};
    /// When true, skip GPU batch dispatch and all cache lookups/writes.
    /// Set during phases (initial attachment, REM maps) that use temporary
    /// antenna objects so they do not pollute the batch cache.
    bool m_bypassGpuBatch{false};

    /// True after a full LSP/cluster pipeline run with m_updatePeriod=0
    /// (static channel). Cleared on topology changes. Enables Path A/B
    /// fast paths in UpdateChannel that skip expensive re-runs.
    bool m_gpuClusterParamsFresh{false};
    /// Number of runtimeLinks at the time m_gpuClusterParamsFresh was set.
    /// Path A/B are only active when the current link count matches this.
    uint32_t m_lastRuntimeLinksCount{0};
};

} // namespace ns3

#endif // THREE_GPP_CHANNEL_MODEL_WGPU_MEZANINE_H
