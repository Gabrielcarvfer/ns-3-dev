//
// Created by gabri on 04/04/2026.
//

#ifndef THREE_GPP_CHANNEL_MODEL_WGPU_MEZANINE_H
#define THREE_GPP_CHANNEL_MODEL_WGPU_MEZANINE_H

#include "three-gpp-channel-model.h"

#include "ns3/phased-array-model.h"

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
        bool isReverse) const override;

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
    // Per-link chanSpct_unscaled buffers indexed by matrixKey. The
    // value pair is (linkSlab f32, generatedTime + rbFreqsHash + dims
    // snapshot). Same matrixKey policy as m_gpuLongTermMap so
    // invalidation rules line up.
    struct GpuChanSpctEntry
    {
        std::vector<std::complex<float>> chanSpctUnscaled;
        uint32_t numRxPorts{0};
        uint32_t numTxPorts{0};
        uint32_t numRb{0};
        Time generatedTime;
        uint64_t rbFreqsHash{0};
    };
    mutable std::unordered_map<uint64_t, GpuChanSpctEntry> m_gpuChanSpctMap;

    static uint64_t HashFloatVector(const std::vector<float>& v);
};

} // namespace ns3

#endif // THREE_GPP_CHANNEL_MODEL_WGPU_MEZANINE_H
