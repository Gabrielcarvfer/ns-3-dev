/*
 * Copyright (c) 2021 DERONNE SOFTWARE ENGINEERING
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#ifndef EHT_PHY_H
#define EHT_PHY_H

#include "ns3/he-phy.h"

/**
 * @file
 * @ingroup wifi
 * Declaration of ns3::EhtPhy class.
 */

namespace ns3
{

/**
 * This defines the BSS membership value for EHT PHY.
 */
#define EHT_PHY 121 // FIXME: not defined yet as of 802.11be D1.5

/**
 * @brief PHY entity for EHT (11be)
 * @ingroup wifi
 *
 * EHT PHY is based on HE PHY.
 *
 * Refer to P802.11be/D1.5.
 */
class EhtPhy : public HePhy
{
  public:
    /**
     * Constructor for EHT PHY
     *
     * @param buildModeList flag used to add EHT modes to list (disabled
     *                      by child classes to only add child classes' modes)
     */
    EhtPhy(bool buildModeList = true);
    /**
     * Destructor for EHT PHY
     */
    ~EhtPhy() override;

    const PpduFormats& GetPpduFormats() const override;
    Time GetDuration(WifiPpduField field, const WifiTxVector& txVector) const override;
    Ptr<WifiPpdu> BuildPpdu(const WifiConstPsduMap& psdus,
                            const WifiTxVector& txVector,
                            Time ppduDuration) override;
    WifiMode GetSigBMode(const WifiTxVector& txVector) const override;
    Time CalculateNonHeDurationForHeTb(const WifiTxVector& txVector) const override;
    Time CalculateNonHeDurationForHeMu(const WifiTxVector& txVector) const override;

    /**
     * Initialize all EHT modes.
     */
    static void InitializeModes();

    /**
     * Return the EHT MCS corresponding to
     * the provided index.
     *
     * @param index the index of the MCS
     * @return an EHT MCS
     */
    static WifiMode GetEhtMcs(uint8_t index);

    /**
     * Return MCS 0 from EHT MCS values.
     *
     * @return MCS 0 from EHT MCS values
     */
    static WifiMode GetEhtMcs0();
    /**
     * Return MCS 1 from EHT MCS values.
     *
     * @return MCS 1 from EHT MCS values
     */
    static WifiMode GetEhtMcs1();
    /**
     * Return MCS 2 from EHT MCS values.
     *
     * @return MCS 2 from EHT MCS values
     */
    static WifiMode GetEhtMcs2();
    /**
     * Return MCS 3 from EHT MCS values.
     *
     * @return MCS 3 from EHT MCS values
     */
    static WifiMode GetEhtMcs3();
    /**
     * Return MCS 4 from EHT MCS values.
     *
     * @return MCS 4 from EHT MCS values
     */
    static WifiMode GetEhtMcs4();
    /**
     * Return MCS 5 from EHT MCS values.
     *
     * @return MCS 5 from EHT MCS values
     */
    static WifiMode GetEhtMcs5();
    /**
     * Return MCS 6 from EHT MCS values.
     *
     * @return MCS 6 from EHT MCS values
     */
    static WifiMode GetEhtMcs6();
    /**
     * Return MCS 7 from EHT MCS values.
     *
     * @return MCS 7 from EHT MCS values
     */
    static WifiMode GetEhtMcs7();
    /**
     * Return MCS 8 from EHT MCS values.
     *
     * @return MCS 8 from EHT MCS values
     */
    static WifiMode GetEhtMcs8();
    /**
     * Return MCS 9 from EHT MCS values.
     *
     * @return MCS 9 from EHT MCS values
     */
    static WifiMode GetEhtMcs9();
    /**
     * Return MCS 10 from EHT MCS values.
     *
     * @return MCS 10 from EHT MCS values
     */
    static WifiMode GetEhtMcs10();
    /**
     * Return MCS 11 from EHT MCS values.
     *
     * @return MCS 11 from EHT MCS values
     */
    static WifiMode GetEhtMcs11();
    /**
     * Return MCS 12 from EHT MCS values.
     *
     * @return MCS 12 from EHT MCS values
     */
    static WifiMode GetEhtMcs12();
    /**
     * Return MCS 13 from EHT MCS values.
     *
     * @return MCS 13 from EHT MCS values
     */
    static WifiMode GetEhtMcs13();

    /**
     * Return the coding rate corresponding to
     * the supplied EHT MCS index. This function is used
     * as a callback for WifiMode operation.
     *
     * @param mcsValue the MCS index
     * @return the coding rate.
     */
    static WifiCodeRate GetCodeRate(uint8_t mcsValue);

    /**
     * Return the constellation size corresponding
     * to the supplied EHT MCS index. This function is used
     * as a callback for WifiMode operation.
     *
     * @param mcsValue the MCS index
     * @return the size of modulation constellation.
     */
    static uint16_t GetConstellationSize(uint8_t mcsValue);

    /**
     * Return the PHY rate corresponding to the supplied EHT MCS
     * index, channel width, guard interval, and number of
     * spatial stream. This function calls HtPhy::CalculatePhyRate
     * and is mainly used as a callback for WifiMode operation.
     *
     * @param mcsValue the EHT MCS index
     * @param channelWidth the considered channel width
     * @param guardInterval the considered guard interval duration
     * @param nss the considered number of stream
     *
     * @return the physical bit rate of this signal in bps.
     */
    static uint64_t GetPhyRate(uint8_t mcsValue,
                               MHz_u channelWidth,
                               Time guardInterval,
                               uint8_t nss);

    /**
     * Return the PHY rate corresponding to
     * the supplied TXVECTOR for the STA-ID.
     *
     * @param txVector the TXVECTOR used for the transmission
     * @param staId the station ID for MU (unused if SU)
     * @return the physical bit rate of this signal in bps.
     */
    static uint64_t GetPhyRateFromTxVector(const WifiTxVector& txVector,
                                           uint16_t staId = SU_STA_ID);

    /**
     * Return the data rate corresponding to
     * the supplied TXVECTOR for the STA-ID.
     *
     * @param txVector the TXVECTOR used for the transmission
     * @param staId the station ID for MU (unused if SU)
     * @return the data bit rate in bps.
     */
    static uint64_t GetDataRateFromTxVector(const WifiTxVector& txVector,
                                            uint16_t staId = SU_STA_ID);

    /**
     * Return the data rate corresponding to
     * the supplied EHT MCS index, channel width,
     * guard interval, and number of spatial
     * streams.
     *
     * @param mcsValue the EHT MCS index
     * @param channelWidth the channel width
     * @param guardInterval the guard interval duration
     * @param nss the number of spatial streams
     * @return the data bit rate in bps.
     */
    static uint64_t GetDataRate(uint8_t mcsValue,
                                MHz_u channelWidth,
                                Time guardInterval,
                                uint8_t nss);

    /**
     * Calculate the rate in bps of the non-HT Reference Rate corresponding
     * to the supplied HE MCS index. This function calls CalculateNonHtReferenceRate
     * and is used as a callback for WifiMode operation.
     *
     * @param mcsValue the HE MCS index
     * @return the rate in bps of the non-HT Reference Rate.
     */
    static uint64_t GetNonHtReferenceRate(uint8_t mcsValue);

  protected:
    void BuildModeList() override;
    WifiMode GetSigMode(WifiPpduField field, const WifiTxVector& txVector) const override;
    PhyFieldRxStatus DoEndReceiveField(WifiPpduField field, Ptr<Event> event) override;
    PhyFieldRxStatus ProcessSig(Ptr<Event> event,
                                PhyFieldRxStatus status,
                                WifiPpduField field) override;
    WifiPhyRxfailureReason GetFailureReason(WifiPpduField field) const override;
    uint32_t GetSigBSize(const WifiTxVector& txVector) const override;
    dBm_u GetCcaThreshold(const Ptr<const WifiPpdu> ppdu,
                          WifiChannelListType channelType) const override;
    const std::map<MHz_u, WifiChannelListType>& GetCcaSecondaryChannels() const override;
    PhyEntity::CcaIndication GetCcaIndicationOnSecondary(const Ptr<const WifiPpdu> ppdu) override;
    std::vector<Time> GetPer20MHzDurations(const Ptr<const WifiPpdu> ppdu) override;

    /**
     * Create and return the EHT MCS corresponding to
     * the provided index.
     * This method binds all the callbacks used by WifiMode.
     *
     * @param index the index of the MCS
     * @return an EHT MCS
     */
    static WifiMode CreateEhtMcs(uint8_t index);

    /**
     * Return the rate (in bps) of the non-HT Reference Rate
     * which corresponds to the supplied code rate and
     * constellation size.
     *
     * @param codeRate the convolutional coding rate
     * @param constellationSize the size of modulation constellation
     * @returns the rate in bps.
     *
     * To convert an HE MCS to its corresponding non-HT Reference Rate
     * use the modulation and coding rate of the HT MCS
     * and lookup in Table 10-10 of IEEE P802.11ax/D6.0.
     */
    static uint64_t CalculateNonHtReferenceRate(WifiCodeRate codeRate, uint16_t constellationSize);

    /**
     * @param channelWidth the channel width
     * @return he number of usable subcarriers for data
     */
    static uint16_t GetUsableSubcarriers(MHz_u channelWidth);

    /**
     * Compute the CCA threshold for Per 20MHz check.
     * This threshold is used for CCA check when signal is not occupying P20 or for Per 20MHz CCA
     * check.
     *
     * @param ppdu the incoming PPDU or nullptr for any signal
     * @return the CCA threshold for Per 20MHz check
     */
    dBm_u Per20MHzCcaThreshold(const Ptr<const WifiPpdu> ppdu) const;

    static const PpduFormats m_ehtPpduFormats; //!< EHT PPDU formats
};

} // namespace ns3

#endif /* EHT_PHY_H */
