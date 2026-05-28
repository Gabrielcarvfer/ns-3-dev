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

  private:
    std::unique_ptr<SlsChanWgpu> m_wgpuChannel;
    mutable std::unordered_map<uint32_t, Ptr<const PhasedArrayModel>> m_antennaIdToObjectMap;
    // Dedup the per-link EnsureBatchFresh calls within the same
    // simulator tick so the small-scale GPU work runs at most once per
    // tick instead of once per link. std::optional so the very first
    // tick still triggers a refresh (Time::Zero() is a valid tick).
    std::optional<Time> m_lastMezBatchTime;
};

} // namespace ns3

#endif // THREE_GPP_CHANNEL_MODEL_WGPU_MEZANINE_H
