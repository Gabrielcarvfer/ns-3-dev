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
    // Override the base batch-fresh hook so the small-scale GPU
    // pipeline runs at the start of each tick instead of waiting for
    // an external 10 ms self-scheduled loop. Base does LSP draw; this
    // override additionally calls UpdateChannel() to write GPU cluster
    // delays / ray angles / XPR into m_channelParamsMap, so the
    // per-link GetNewChannel that follows skips the CPU small-scale
    // draw entirely.
    void EnsureBatchFresh() override;
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
};

} // namespace ns3

#endif // THREE_GPP_CHANNEL_MODEL_WGPU_MEZANINE_H
