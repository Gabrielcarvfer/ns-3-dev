/*
 * Copyright (c) 2015
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Ghada Badawy <gbadawy@gmail.com>
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#ifndef VHT_CAPABILITIES_H
#define VHT_CAPABILITIES_H

#include "ns3/wifi-information-element.h"

namespace ns3
{

/**
 * @ingroup wifi
 *
 * The IEEE 802.11ac VHT Capabilities
 */
class VhtCapabilities : public WifiInformationElement
{
  public:
    VhtCapabilities();

    // Implementations of pure virtual methods of WifiInformationElement
    WifiInformationElementId ElementId() const override;
    void Print(std::ostream& os) const override;

    /**
     * Set the VHT Capabilities Info field in the VHT Capabilities information element.
     *
     * @param ctrl the VHT Capabilities Info field in the VHT Capabilities information element
     */
    void SetVhtCapabilitiesInfo(uint32_t ctrl);
    /**
     * Set the MCS and NSS field in the VHT Capabilities information element.
     *
     * @param ctrl the MCS and NSS field in the VHT Capabilities information element
     */
    void SetSupportedMcsAndNssSet(uint64_t ctrl);

    /**
     * Return the VHT Capabilities Info field in the VHT Capabilities information element.
     *
     * @return the VHT Capabilities Info field in the VHT Capabilities information element
     */
    uint32_t GetVhtCapabilitiesInfo() const;
    /**
     * Return the MCS and NSS field in the VHT Capabilities information element.
     *
     * @return the MCS and NSS field in the VHT Capabilities information element
     */
    uint64_t GetSupportedMcsAndNssSet() const;

    // Capabilities Info fields
    /**
     * Set the maximum MPDU length.
     *
     * @param length the maximum MPDU length (3895, 7991 or 11454)
     */
    void SetMaxMpduLength(uint16_t length);
    /**
     * Set the supported channel width set.
     *
     * @param channelWidthSet the supported channel width set
     */
    void SetSupportedChannelWidthSet(uint8_t channelWidthSet);
    /**
     * Set the receive LDPC.
     *
     * @param rxLdpc the receive LDPC
     */
    void SetRxLdpc(uint8_t rxLdpc);
    /**
     * Set the short guard interval 80 MHz.
     *
     * @param shortGuardInterval the short guard interval 80 MHz
     */
    void SetShortGuardIntervalFor80Mhz(uint8_t shortGuardInterval);
    /**
     * Set the short guard interval 160 MHz.
     *
     * @param shortGuardInterval the short guard interval 160 MHz
     */
    void SetShortGuardIntervalFor160Mhz(uint8_t shortGuardInterval);
    /**
     * Set the receive STBC.
     *
     * @param rxStbc the receive STBC
     */
    void SetRxStbc(uint8_t rxStbc);
    /**
     * Set the transmit STBC.
     *
     * @param txStbc the receive STBC
     */
    void SetTxStbc(uint8_t txStbc);
    /**
     * Set the maximum AMPDU length.
     *
     * @param maxAmpduLength 2^(13 + x) - 1, x in the range 0 to 7
     */
    void SetMaxAmpduLength(uint32_t maxAmpduLength);

    /**
     * Get the maximum MPDU length.
     *
     * @return the maximum MPDU length in bytes
     */
    uint16_t GetMaxMpduLength() const;
    /**
     * Get the supported channel width set.
     *
     * @returns the supported channel width set
     */
    uint8_t GetSupportedChannelWidthSet() const;
    /**
     * Get the receive LDPC.
     *
     * @returns the receive LDPC
     */
    uint8_t GetRxLdpc() const;
    /**
     * Get the receive STBC.
     *
     * @returns the receive STBC
     */
    uint8_t GetRxStbc() const;
    /**
     * Get the transmit STBC.
     *
     * @returns the transmit STBC
     */
    uint8_t GetTxStbc() const;

    /**
     * @param mcs Max MCS value (between 7 and 9)
     * @param nss Spatial stream for which the Max MCS value is being set
     */
    void SetRxMcsMap(uint8_t mcs, uint8_t nss);
    /**
     * @param mcs Max MCS value (between 7 and 9)
     * @param nss Spatial stream for which the Max MCS value is being set
     */
    void SetTxMcsMap(uint8_t mcs, uint8_t nss);
    /**
     * Set the receive highest supported LGI data rate.
     *
     * @param supportedDatarate receive highest supported LGI data rate
     */
    void SetRxHighestSupportedLgiDataRate(uint16_t supportedDatarate);
    /**
     * Set the transmit highest supported LGI data rate.
     *
     * @param supportedDatarate transmit highest supported LGI data rate
     */
    void SetTxHighestSupportedLgiDataRate(uint16_t supportedDatarate);
    /**
     * Get the is MCS supported.
     *
     * @param mcs the MCS
     * @param nss the NSS
     * @returns the is MCS supported
     */
    bool IsSupportedMcs(uint8_t mcs, uint8_t nss) const;

    /**
     * Get the receive highest supported LGI data rate.
     *
     * @returns the receive highest supported LGI data rate.
     */
    uint16_t GetRxHighestSupportedLgiDataRate() const;

    /**
     * Returns true if transmit MCS is supported.
     *
     * @param mcs the MCS
     * @returns whether transmit MCS is supported
     */
    bool IsSupportedTxMcs(uint8_t mcs) const;
    /**
     * Returns true if receive MCS is supported.
     *
     * @param mcs the MCS
     * @returns whether receive MCS is supported
     */
    bool IsSupportedRxMcs(uint8_t mcs) const;

    /**
     * Return the maximum A-MPDU length.
     *
     * @return the maximum A-MPDU length in bytes
     */
    uint32_t GetMaxAmpduLength() const;

  private:
    uint16_t GetInformationFieldSize() const override;
    void SerializeInformationField(Buffer::Iterator start) const override;
    uint16_t DeserializeInformationField(Buffer::Iterator start, uint16_t length) override;

    // Capabilities Info fields
    uint8_t m_maxMpduLength;               ///< maximum MPDU length
    uint8_t m_supportedChannelWidthSet;    ///< supported channel width set
    uint8_t m_rxLdpc;                      ///< receive LDPC
    uint8_t m_shortGuardIntervalFor80Mhz;  ///< short guard interval for 80 MHz
    uint8_t m_shortGuardIntervalFor160Mhz; ///< short guard interval for 160 MHz
    uint8_t m_txStbc;                      ///< transmit STBC
    uint8_t m_rxStbc;                      ///< receive STBC
    uint8_t m_suBeamformerCapable;         ///< SU beamformer capable
    uint8_t m_suBeamformeeCapable;         ///< SU beamformee capable
    uint8_t m_beamformeeStsCapable;        ///< beamformee STS capable
    uint8_t m_numberOfSoundingDimensions;  ///< number of sounding dimensions
    uint8_t m_muBeamformerCapable;         ///< MU beamformer capable
    uint8_t m_muBeamformeeCapable;         ///< MU beamformee capable
    uint8_t m_vhtTxopPs;                   ///< VHT TXOP PS
    uint8_t m_htcVhtCapable;               ///< HTC VHT capable
    uint8_t m_maxAmpduLengthExponent;      ///< maximum A-MPDU length exponent
    uint8_t m_vhtLinkAdaptationCapable;    ///< VHT link adaptation capable
    uint8_t m_rxAntennaPatternConsistency; ///< receive antenna pattern consistency
    uint8_t m_txAntennaPatternConsistency; ///< transmit antenna pattern consistency

    // MCS and NSS field information
    std::vector<uint8_t> m_rxMcsMap;                        ///< receive MCS map
    uint16_t m_rxHighestSupportedLongGuardIntervalDataRate; ///< receive highest supported long
                                                            ///< guard interval data rate
    std::vector<uint8_t> m_txMcsMap;                        ///< transmit MCS map
    uint16_t m_txHighestSupportedLongGuardIntervalDataRate; ///< transmit highest supported long
                                                            ///< guard interval data rate
};

} // namespace ns3

#endif /* VHT_CAPABILITY_H */
