/*
 * Copyright (c) 2020 Universita' degli Studi di Napoli Federico II
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#ifndef QOS_FRAME_EXCHANGE_MANAGER_H
#define QOS_FRAME_EXCHANGE_MANAGER_H

#include "frame-exchange-manager.h"

#include <optional>

namespace ns3
{

/**
 * @ingroup wifi
 *
 * QosFrameExchangeManager handles the frame exchange sequences
 * for QoS stations.
 * Note that Basic Block Ack is not supported.
 */
class QosFrameExchangeManager : public FrameExchangeManager
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();
    QosFrameExchangeManager();
    ~QosFrameExchangeManager() override;

    bool StartTransmission(Ptr<Txop> edca, MHz_u allowedWidth) override;

    /**
     * Recompute the protection and acknowledgment methods to use if the given MPDU
     * is added to the frame being built (as described by the given TX parameters)
     * and check whether the duration of the frame exchange sequence (including
     * protection and acknowledgment) does not exceed the given available time.
     * The protection and acknowledgment methods held by the given TX parameters are
     * only updated if the given MPDU can be added.
     *
     * @param mpdu the MPDU to add to the frame being built
     * @param txParams the TX parameters describing the frame being built
     * @param availableTime the time limit on the frame exchange sequence
     * @return true if the given MPDU can be added to the frame being built
     */
    bool TryAddMpdu(Ptr<const WifiMpdu> mpdu, WifiTxParameters& txParams, Time availableTime) const;

    /**
     * Check whether the given MPDU can be added to the frame being built (as described
     * by the given TX parameters) without violating the given constraint on the
     * PPDU transmission duration.
     *
     * @param mpdu the MPDU to add to the frame being built
     * @param txParams the TX parameters describing the frame being built
     * @param ppduDurationLimit the time limit on the PPDU transmission duration
     * @return true if the given MPDU can be added to the frame being built
     */
    virtual bool IsWithinLimitsIfAddMpdu(Ptr<const WifiMpdu> mpdu,
                                         const WifiTxParameters& txParams,
                                         Time ppduDurationLimit) const;

    /**
     * Check whether the transmission time of the frame being built (as described
     * by the given TX parameters) does not exceed the given PPDU duration limit
     * if the size of the PSDU addressed to the given receiver becomes
     * <i>ppduPayloadSize</i>. Also check that the PSDU size does not exceed the
     * max PSDU size for the modulation class being used.
     *
     * @param ppduPayloadSize the new PSDU size
     * @param receiver the MAC address of the receiver of the PSDU
     * @param txParams the TX parameters describing the frame being built
     * @param ppduDurationLimit the limit on the PPDU duration
     * @return true if the constraints on the PPDU duration limit and the maximum PSDU size are met
     */
    virtual bool IsWithinSizeAndTimeLimits(uint32_t ppduPayloadSize,
                                           Mac48Address receiver,
                                           const WifiTxParameters& txParams,
                                           Time ppduDurationLimit) const;

    /**
     * Create an alias of the given MPDU for transmission by this Frame Exchange Manager.
     * This is required by 11be MLDs to support translation of MAC addresses. For single
     * link devices, the given MPDU is simply returned.
     *
     * @param mpdu the given MPDU
     * @return the alias of the given MPDU for transmission on this link
     */
    virtual Ptr<WifiMpdu> CreateAliasIfNeeded(Ptr<WifiMpdu> mpdu) const;

    /**
     * @return the TXOP holder (if any)
     */
    std::optional<Mac48Address> GetTxopHolder() const;

  protected:
    void DoDispose() override;

    void ReceiveMpdu(Ptr<const WifiMpdu> mpdu,
                     RxSignalInfo rxSignalInfo,
                     const WifiTxVector& txVector,
                     bool inAmpdu) override;
    void PreProcessFrame(Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector) override;
    void PostProcessFrame(Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector) override;
    void NavResetTimeout() override;
    void UpdateNav(const WifiMacHeader& hdr,
                   const WifiTxVector& txVector,
                   const Time& surplus = Time{0}) override;
    Time GetFrameDurationId(const WifiMacHeader& header,
                            uint32_t size,
                            const WifiTxParameters& txParams,
                            Ptr<Packet> fragmentedPacket) const override;
    Time GetRtsDurationId(const WifiTxVector& rtsTxVector,
                          Time txDuration,
                          Time response) const override;
    Time GetCtsToSelfDurationId(const WifiTxVector& ctsTxVector,
                                Time txDuration,
                                Time response) const override;
    void TransmissionSucceeded() override;
    void TransmissionFailed(bool forceCurrentCw = false) override;
    void ForwardMpduDown(Ptr<WifiMpdu> mpdu, WifiTxVector& txVector) override;
    void ReceivedMacHdr(const WifiMacHeader& macHdr,
                        const WifiTxVector& txVector,
                        Time psduDuration) override;

    /**
     * Request the FrameExchangeManager to start a frame exchange sequence.
     *
     * @param edca the EDCA that gained channel access
     * @param txopDuration the duration of a TXOP. This value is only used when a
     *                     new TXOP is started (and hence the TXOP limit for the
     *                     given EDCAF is non-zero)
     * @return true if a frame exchange sequence was started, false otherwise
     */
    virtual bool StartTransmission(Ptr<QosTxop> edca, Time txopDuration);

    /**
     * Start a frame exchange (including protection frames and acknowledgment frames
     * as needed) that fits within the given <i>availableTime</i> (if different than
     * Time::Min()).
     *
     * @param edca the EDCAF which has been granted the opportunity to transmit
     * @param availableTime the amount of time allowed for the frame exchange. Pass
     *                      Time::Min() in case the TXOP limit is null
     * @param initialFrame true if the frame being transmitted is the initial frame
     *                     of the TXOP. This is used to determine whether the TXOP
     *                     limit can be exceeded
     * @return true if a frame exchange is started, false otherwise
     */
    virtual bool StartFrameExchange(Ptr<QosTxop> edca, Time availableTime, bool initialFrame);

    /**
     * Perform a PIFS recovery as a response to transmission failure within a TXOP.
     * If the carrier sense indicates that the medium is idle, continue the TXOP.
     * Otherwise, release the channel.
     *
     * @param forceCurrentCw whether to force the contention window to stay equal to the current
     *                       value if PIFs recovery fails (normally, contention window is updated)
     */
    void PifsRecovery(bool forceCurrentCw);

    /**
     * Send a CF-End frame to indicate the completion of the TXOP, provided that
     * the remaining duration is long enough to transmit this frame.
     *
     * @return true if a CF-End frame was sent, false otherwise
     */
    virtual bool SendCfEndIfNeeded();

    /**
     * Determine the holder of the TXOP, if possible, based on the received frame
     *
     * @param hdr the MAC header of an MPDU included in the received PSDU
     * @param txVector TX vector of the received PSDU
     * @return the holder of the TXOP, if one was found
     */
    virtual std::optional<Mac48Address> FindTxopHolder(const WifiMacHeader& hdr,
                                                       const WifiTxVector& txVector);

    /**
     * Clear the TXOP holder if the NAV counted down to zero (includes the case of NAV reset).
     */
    virtual void ClearTxopHolderIfNeeded();

    Ptr<QosTxop> m_edca;                      //!< the EDCAF that gained channel access
    std::optional<Mac48Address> m_txopHolder; //!< MAC address of the TXOP holder
    bool m_setQosQueueSize;                   /**< whether to set the Queue Size subfield of the
                                                   QoS Control field of QoS data frames */
    bool m_protectSingleExchange; /**< true if the Duration/ID field in frames establishing
                                     protection only covers the immediate frame exchange instead of
                                     rest of the TXOP limit when the latter is non-zero */
    Time m_singleExchangeProtectionSurplus; /**< additional time to protect beyond end of the
                                               immediate frame exchange in case of non-zero TXOP
                                               limit when a single frame exchange is protected */

  private:
    /**
     * Set the TXOP holder, if needed, based on the received frame
     *
     * @param hdr the MAC header of the received PSDU
     * @param txVector TX vector of the received PSDU
     */
    void SetTxopHolder(const WifiMacHeader& hdr, const WifiTxVector& txVector);

    /**
     * Cancel the PIFS recovery event and have the EDCAF attempting PIFS recovery
     * release the channel.
     */
    void CancelPifsRecovery();

    bool m_initialFrame;         //!< true if transmitting the initial frame of a TXOP
    bool m_pifsRecovery;         //!< true if performing a PIFS recovery after failure
    EventId m_pifsRecoveryEvent; //!< event associated with an attempt of PIFS recovery
    Ptr<Txop> m_edcaBackingOff;  //!< channel access function that invoked backoff during TXOP
};

} // namespace ns3

#endif /* QOS_FRAME_EXCHANGE_MANAGER_H */
