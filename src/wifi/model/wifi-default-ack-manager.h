/*
 * Copyright (c) 2020 Universita' degli Studi di Napoli Federico II
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#ifndef WIFI_DEFAULT_ACK_MANAGER_H
#define WIFI_DEFAULT_ACK_MANAGER_H

#include "wifi-ack-manager.h"

namespace ns3
{

class WifiTxParameters;
class WifiMpdu;

/**
 * @ingroup wifi
 *
 * WifiDefaultAckManager is the default ack manager.
 */
class WifiDefaultAckManager : public WifiAckManager
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    WifiDefaultAckManager();
    ~WifiDefaultAckManager() override;

    std::unique_ptr<WifiAcknowledgment> TryAddMpdu(Ptr<const WifiMpdu> mpdu,
                                                   const WifiTxParameters& txParams) override;
    std::unique_ptr<WifiAcknowledgment> TryAggregateMsdu(Ptr<const WifiMpdu> msdu,
                                                         const WifiTxParameters& txParams) override;

    /**
     * Get the maximum distance between the starting sequence number of the Block
     * Ack agreement which the given MPDU belongs to and each of the sequence numbers
     * of the given MPDU and of all the QoS data frames included in the given TX
     * parameters.
     *
     * @param mpdu the given MPDU
     * @param txParams the given TX parameters
     * @return the maximum distance between the starting sequence number of the Block
     *         Ack agreement which the given MPDU belongs to and each of the sequence
     *         numbers of the given MPDU and of all the QoS data frames included in
     *         the given TX parameters
     */
    uint16_t GetMaxDistFromStartingSeq(Ptr<const WifiMpdu> mpdu,
                                       const WifiTxParameters& txParams) const;

  protected:
    /**
     * Determine whether the (A-)MPDU containing the given MPDU and the MPDUs (if any)
     * included in the given TX parameters requires an immediate response (Normal Ack,
     * Block Ack or Block Ack Request followed by Block Ack).
     *
     * @param mpdu the given MPDU.
     * @param txParams the given TX parameters.
     * @return true if the given PSDU requires an immediate response
     */
    bool IsResponseNeeded(Ptr<const WifiMpdu> mpdu, const WifiTxParameters& txParams) const;

    /**
     * @param mpdu the given MPDU
     * @return whether there exist MPDUs with lower sequence number than the given MPDU that are
     * inflight on the same link as the given MPDU
     */
    bool ExistInflightOnSameLink(Ptr<const WifiMpdu> mpdu) const;

  private:
    /**
     * Compute the information about the acknowledgment of the current multi-user frame
     * (as described by the given TX parameters) if the given MPDU is added. If the
     * computed information is the same as the current one, a null pointer is returned.
     * Otherwise, the computed information is returned.
     * This method can only be called if the selected acknowledgment method for DL
     * multi-user frames consists of a sequence of BlockAckReq and BlockAck frames.
     *
     * @param mpdu the given MPDU
     * @param txParams the TX parameters describing the current multi-user frame
     * @return the new acknowledgment method or a null pointer if the acknowledgment method
     *         is unchanged
     */
    virtual std::unique_ptr<WifiAcknowledgment> GetAckInfoIfBarBaSequence(
        Ptr<const WifiMpdu> mpdu,
        const WifiTxParameters& txParams);
    /**
     * Compute the information about the acknowledgment of the current multi-user frame
     * (as described by the given TX parameters) if the given MPDU is added. If the
     * computed information is the same as the current one, a null pointer is returned.
     * Otherwise, the computed information is returned.
     * This method can only be called if the selected acknowledgment method for DL
     * multi-user frames consists of a MU-BAR Trigger Frame sent as single-user frame.
     *
     * @param mpdu the given MPDU
     * @param txParams the TX parameters describing the current multi-user frame
     * @return the new acknowledgment method or a null pointer if the acknowledgment method
     *         is unchanged
     */
    virtual std::unique_ptr<WifiAcknowledgment> GetAckInfoIfTfMuBar(
        Ptr<const WifiMpdu> mpdu,
        const WifiTxParameters& txParams);
    /**
     * Compute the information about the acknowledgment of the current multi-user frame
     * (as described by the given TX parameters) if the given MPDU is added. If the
     * computed information is the same as the current one, a null pointer is returned.
     * Otherwise, the computed information is returned.
     * This method can only be called if the selected acknowledgment method for DL
     * multi-user frames consists of MU-BAR Trigger Frames aggregated to the PSDUs of
     * the MU PPDU.
     *
     * @param mpdu the given MPDU
     * @param txParams the TX parameters describing the current multi-user frame
     * @return the new acknowledgment method or a null pointer if the acknowledgment method
     *         is unchanged
     */
    virtual std::unique_ptr<WifiAcknowledgment> GetAckInfoIfAggregatedMuBar(
        Ptr<const WifiMpdu> mpdu,
        const WifiTxParameters& txParams);

    /**
     * Calculate the acknowledgment method for the TB PPDUs solicited by the given
     * Trigger Frame.
     *
     * @param mpdu the given Trigger Frame
     * @param txParams the current TX parameters (just the TXVECTOR needs to be set)
     * @return the acknowledgment method for the TB PPDUs solicited by the given Trigger Frame
     */
    virtual std::unique_ptr<WifiAcknowledgment> TryUlMuTransmission(
        Ptr<const WifiMpdu> mpdu,
        const WifiTxParameters& txParams);

    bool m_useExplicitBar; //!< true for sending BARs, false for using Implicit BAR policy
    double m_baThreshold;  //!< Threshold to determine when a BlockAck must be requested
    WifiAcknowledgment::Method m_dlMuAckType; //!< Type of the ack sequence for DL MU PPDUs
    uint8_t m_maxMcsForBlockAckInTbPpdu;      //!< Max MCS used to send a BlockAck in a TB PPDU
};

} // namespace ns3

#endif /* WIFI_DEFAULT_ACK_MANAGER_H */
