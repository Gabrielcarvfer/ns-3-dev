/*
 * Copyright (c) 2006, 2009 INRIA
 * Copyright (c) 2009 MIRKO BANCHI
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *          Mirko Banchi <mk.banchi@gmail.com>
 */

#ifndef AP_WIFI_MAC_H
#define AP_WIFI_MAC_H

#include "wifi-mac-header.h"
#include "wifi-mac.h"

#include "ns3/attribute-container.h"
#include "ns3/enum.h"
#include "ns3/pair.h"

#include <unordered_map>
#include <variant>

namespace ns3
{

struct AllSupportedRates;
class CapabilityInformation;
class DsssParameterSet;
class ErpInformation;
class EdcaParameterSet;
class MuEdcaParameterSet;
class ReducedNeighborReport;
class HtOperation;
class VhtOperation;
class HeOperation;
class EhtOperation;
class CfParameterSet;
class UniformRandomVariable;
class MgtEmlOmn;
class ApEmlsrManager;
class GcrManager;

/// variant holding a  reference to a (Re)Association Request
using AssocReqRefVariant = std::variant<std::reference_wrapper<MgtAssocRequestHeader>,
                                        std::reference_wrapper<MgtReassocRequestHeader>>;

/**
 * @brief Wi-Fi AP state machine
 * @ingroup wifi
 *
 * Handle association, dis-association and authentication,
 * of STAs within an infrastructure BSS.  By default, beacons are
 * sent with PIFS access, zero backoff, and are generated roughly
 * every 102.4 ms by default (configurable by an attribute) and
 * with some jitter to de-synchronize beacon transmissions in
 * multi-BSS scenarios.
 */
class ApWifiMac : public WifiMac
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    ApWifiMac();
    ~ApWifiMac() override;

    void SetLinkUpCallback(Callback<void> linkUp) override;
    bool CanForwardPacketsTo(Mac48Address to) const override;
    bool SupportsSendFrom() const override;
    Ptr<WifiMacQueue> GetTxopQueue(AcIndex ac) const override;
    int64_t AssignStreams(int64_t stream) override;

    /**
     * Set the AP EMLSR Manager.
     *
     * @param apEmlsrManager the AP EMLSR Manager
     */
    void SetApEmlsrManager(Ptr<ApEmlsrManager> apEmlsrManager);

    /**
     * @return the AP EMLSR Manager
     */
    Ptr<ApEmlsrManager> GetApEmlsrManager() const;

    /**
     * Set the GCR Manager.
     *
     * @param gcrManager the GCR Manager
     */
    void SetGcrManager(Ptr<GcrManager> gcrManager);

    /**
     * @return the GCR Manager
     */
    Ptr<GcrManager> GetGcrManager() const;

    /**
     * @param interval the interval between two beacon transmissions.
     */
    void SetBeaconInterval(Time interval);
    /**
     * @return the interval between two beacon transmissions.
     */
    Time GetBeaconInterval() const;

    /**
     * Get a const reference to the map of associated stations on the given link.
     * Each station is specified by an (association ID, MAC address) pair. Make sure
     * not to use the returned reference after that this object has been deallocated.
     *
     * @param linkId the ID of the given link
     * @return a const reference to the map of associated stations
     */
    const std::map<uint16_t, Mac48Address>& GetStaList(uint8_t linkId) const;
    /**
     * @param addr the address of the associated station
     * @param linkId the ID of the link on which the station is associated
     * @return the Association ID allocated by the AP to the station, SU_STA_ID if unallocated
     */
    uint16_t GetAssociationId(Mac48Address addr, uint8_t linkId) const;

    /// @return the next Association ID to be allocated by the AP
    uint16_t GetNextAssociationId() const;

    /**
     * Get the ID of a link (if any) that has been setup with the station having the given MAC
     * address. The address can be either a link address or an MLD address. In the former case,
     * the returned ID is the ID of the link connecting the AP to the STA with the given address.
     *
     * @param address the given MAC address
     * @return the ID of a link (if any) that has been setup with the given station
     */
    std::optional<uint8_t> IsAssociated(const Mac48Address& address) const;

    /**
     * @param aid the given AID
     * @return the MLD address (in case of MLD) or link address (in case of single link device)
     *         of the STA having the given AID, if any
     */
    std::optional<Mac48Address> GetMldOrLinkAddressByAid(uint16_t aid) const;

    /**
     * Return the value of the Queue Size subfield of the last QoS Data or QoS Null
     * frame received from the station with the given MAC address and belonging to
     * the given TID.
     *
     * The Queue Size value is the total size, rounded up to the nearest multiple
     * of 256 octets and expressed in units of 256 octets, of all  MSDUs and A-MSDUs
     * buffered at the STA (excluding the MSDU or A-MSDU of the present QoS Data frame).
     * A queue size value of 254 is used for all sizes greater than 64 768 octets.
     * A queue size value of 255 is used to indicate an unspecified or unknown size.
     * See Section 9.2.4.5.6 of 802.11-2016
     *
     * @param tid the given TID
     * @param address the given MAC address
     * @return the value of the Queue Size subfield
     */
    uint8_t GetBufferStatus(uint8_t tid, Mac48Address address) const;
    /**
     * Store the value of the Queue Size subfield of the last QoS Data or QoS Null
     * frame received from the station with the given MAC address and belonging to
     * the given TID.
     *
     * @param tid the given TID
     * @param address the given MAC address
     * @param size the value of the Queue Size subfield
     */
    void SetBufferStatus(uint8_t tid, Mac48Address address, uint8_t size);
    /**
     * Return the maximum among the values of the Queue Size subfield of the last
     * QoS Data or QoS Null frames received from the station with the given MAC address
     * and belonging to any TID.
     *
     * @param address the given MAC address
     * @return the maximum among the values of the Queue Size subfields
     */
    uint8_t GetMaxBufferStatus(Mac48Address address) const;

    /**
     * Return whether GCR is used to transmit a packet.
     *
     * @param hdr the header of the packet to transmit
     * @return true if a GCR manager is set and the packet is a groupcast QoS data, false otherwise
     */
    bool UseGcr(const WifiMacHeader& hdr) const;

    /**
     * Check if a GCR Block Ack agreement has been successfully established with all members of
     * its group.
     *
     * @param groupAddress the GCR group address.
     * @param tid the traffic ID.
     * @return true if a GCR Block Ack agreement has been successfully established with all members
     * of its group, false otherwise.
     */
    bool IsGcrBaAgreementEstablishedWithAllMembers(const Mac48Address& groupAddress,
                                                   uint8_t tid) const;

    /// ACI-indexed map of access parameters of type unsigned integer (CWmin, CWmax and AIFSN)
    using UintAccessParamsMap = std::map<AcIndex, std::vector<uint64_t>>;

    /// ACI-indexed map of access parameters of type Time (TxopLimit)
    using TimeAccessParamsMap = std::map<AcIndex, std::vector<Time>>;

    /// AttributeValue type of a pair (ACI, access parameters of type unsigned integer)
    using UintAccessParamsPairValue =
        PairValue<EnumValue<AcIndex>, AttributeContainerValue<UintegerValue, ',', std::vector>>;

    /// AttributeValue type of a pair (ACI, access parameters of type Time)
    using TimeAccessParamsPairValue =
        PairValue<EnumValue<AcIndex>, AttributeContainerValue<TimeValue, ',', std::vector>>;

    /// AttributeValue type of an ACI-indexed map of access parameters of type unsigned integer
    using UintAccessParamsMapValue = AttributeContainerValue<UintAccessParamsPairValue, ';'>;

    /// AttributeValue type of ACI-indexed map of access parameters of type Time
    using TimeAccessParamsMapValue = AttributeContainerValue<TimeAccessParamsPairValue, ';'>;

    /**
     * Get a checker for the CwMinsForSta, CwMaxsForSta and AifsnsForSta attributes, which can
     * be used to deserialize an ACI-indexed map of access parameters of type unsigned integer
     * (CWmin, CWmax and AIFSN) from a string:
     *
     * @code
     *   ApWifiMac::UintAccessParamsMapValue value;
     *   value.DeserializeFromString("BE 31,31; VO 15,15",
     *                               ApWifiMac::GetUintAccessParamsChecker<uint32_t>());
     *   auto map = value.Get();
     * @endcode
     *
     * The type of \p map is ApWifiMac::UintAccessParamsMapValue::result_type, which is
     * std::list<std::pair<AcIndex, std::vector<uint64_t>>>.
     *
     * @tparam T \explicit the type of the unsigned integer access parameter
     * @return a checker for the CwMinsForSta, CwMaxsForSta and AifsnsForSta attributes
     */
    template <class T>
    static Ptr<const AttributeChecker> GetUintAccessParamsChecker();

    /**
     * Get a checker for the TxopLimitsForSta attribute, which can be used to deserialize an
     * ACI-indexed map of access parameters of type Time (TxopLimit) from a string:
     *
     * @code
     *   ApWifiMac::TimeAccessParamsMapValue value;
     *   value.DeserializeFromString("BE 3200us; VO 3232us",
     *                               ApWifiMac::GetTimeAccessParamsChecker());
     *   auto map = value.Get();
     * @endcode
     *
     * The type of \p map is ApWifiMac::TimeAccessParamsMapValue::result_type, which is
     * std::list<std::pair<AcIndex, std::vector<Time>>>.
     *
     * @return a checker for the TxopLimitsForSta attribute
     */
    static Ptr<const AttributeChecker> GetTimeAccessParamsChecker();

  protected:
    /**
     * Structure holding information specific to a single link. Here, the meaning of
     * "link" is that of the 11be amendment which introduced multi-link devices. For
     * previous amendments, only one link can be created.
     */
    struct ApLinkEntity : public WifiMac::LinkEntity
    {
        /// Destructor (a virtual method is needed to make this struct polymorphic)
        ~ApLinkEntity() override;

        EventId beaconEvent;                      //!< Event to generate one beacon
        std::map<uint16_t, Mac48Address> staList; //!< Map of all stations currently associated
                                                  //!< to the AP with their association ID
        uint16_t numNonHtStations{0}; //!< Number of non-HT stations currently associated to the AP
        uint16_t numNonErpStations{
            0}; //!< Number of non-ERP stations currently associated to the AP
        bool shortSlotTimeEnabled{
            false}; //!< Flag whether short slot time is enabled within the BSS
        bool shortPreambleEnabled{false}; //!< Flag whether short preamble is enabled in the BSS
    };

    /**
     * Get a reference to the link associated with the given ID.
     *
     * @param linkId the given link ID
     * @return a reference to the link associated with the given ID
     */
    ApLinkEntity& GetLink(uint8_t linkId) const;

    std::map<uint16_t, Mac48Address>
        m_aidToMldOrLinkAddress; //!< Maps AIDs to MLD addresses (for MLDs) or link addresses (in
                                 //!< case of single link devices)

  private:
    std::unique_ptr<LinkEntity> CreateLinkEntity() const override;
    Mac48Address DoGetLocalAddress(const Mac48Address& remoteAddr) const override;
    void Receive(Ptr<const WifiMpdu> mpdu, uint8_t linkId) override;
    void DoCompleteConfig() override;
    void Enqueue(Ptr<WifiMpdu> mpdu, Mac48Address to, Mac48Address from) override;

    /**
     * Check whether the supported rate set included in the received (Re)Association
     * Request frame is compatible with our Basic Rate Set. If so, record all the station's
     * supported modes in its associated WifiRemoteStation and return true.
     * Otherwise, return false.
     *
     * @param assoc the frame body of the received (Re)Association Request
     * @param from the Transmitter Address field of the frame
     * @param linkId the ID of the link on which the frame was received
     * @return true if the (Re)Association request can be accepted, false otherwise
     */
    bool ReceiveAssocRequest(const AssocReqRefVariant& assoc,
                             const Mac48Address& from,
                             uint8_t linkId);

    /**
     * Given a (Re)Association Request frame body containing a Multi-Link Element,
     * check if a link can be setup with each of the reported stations (STA MAC address
     * and a (Re)Association Request frame body must be present, the Link ID identifies
     * a valid link other than the one the frame was received on and the supported
     * rates are compatible with our basic rate set).
     *
     * @param assoc the frame body of the received (Re)Association Request
     * @param from the Transmitter Address field of the frame
     * @param linkId the ID of the link on which the frame was received
     */
    void ParseReportedStaInfo(const AssocReqRefVariant& assoc, Mac48Address from, uint8_t linkId);

    /**
     * Take necessary actions upon receiving the given EML Operating Mode Notification frame
     * from the given station on the given link.
     *
     * @param frame the received EML Operating Mode Notification frame
     * @param sender the MAC address of the sender of the frame
     * @param linkId the ID of the link over which the frame was received
     */
    void ReceiveEmlOmn(MgtEmlOmn& frame, const Mac48Address& sender, uint8_t linkId);

    /**
     * The packet we sent was successfully received by the receiver
     * (i.e. we received an Ack from the receiver).  If the packet
     * was an association response to the receiver, we record that
     * the receiver is now associated with us.
     *
     * @param mpdu the MPDU that we successfully sent
     */
    void TxOk(Ptr<const WifiMpdu> mpdu);
    /**
     * The packet we sent was successfully received by the receiver
     * (i.e. we did not receive an Ack from the receiver).  If the packet
     * was an association response to the receiver, we record that
     * the receiver is not associated with us yet.
     *
     * @param timeoutReason the reason why the TX timer was started (\see WifiTxTimer::Reason)
     * @param mpdu the MPDU that we failed to sent
     */
    void TxFailed(WifiMacDropReason timeoutReason, Ptr<const WifiMpdu> mpdu);

    /**
     * This method is called to de-aggregate an A-MSDU and forward the
     * constituent packets up the stack. We override the WifiMac version
     * here because, as an AP, we also need to think about redistributing
     * to other associated STAs.
     *
     * @param mpdu the MPDU containing the A-MSDU.
     */
    void DeaggregateAmsduAndForward(Ptr<const WifiMpdu> mpdu) override;

    /**
     * Get Probe Response Per-STA Profile for the given link.
     *
     * @param linkId the ID of the given link
     * @return the Probe Response header
     */
    MgtProbeResponseHeader GetProbeRespProfile(uint8_t linkId) const;

    /**
     * Get Probe Response based on the given Probe Request Multi-link Element (if any)
     *
     * @param linkId the ID of link the Probe Response is to be sent
     * @param reqMle Probe Request Multi-link Element
     * @return Probe Response header
     */
    MgtProbeResponseHeader GetProbeResp(uint8_t linkId,
                                        const std::optional<MultiLinkElement>& reqMle);

    /**
     * Send a packet prepared using the given Probe Response to the given receiver on the given
     * link.
     *
     * @param probeResp the Probe Response header
     * @param to the address of the STA we are sending a probe response to
     * @param linkId the ID of the given link
     */
    void EnqueueProbeResp(const MgtProbeResponseHeader& probeResp, Mac48Address to, uint8_t linkId);

    /**
     * Get the Association Response frame to send on a given link. The returned frame
     * never includes a Multi-Link Element.
     *
     * @param to the address of the STA we are sending an association response to
     * @param linkId the ID of the given link
     * @return the Association Response frame
     */
    MgtAssocResponseHeader GetAssocResp(Mac48Address to, uint8_t linkId);
    /// Map of (link ID, remote STA address) of the links to setup
    using LinkIdStaAddrMap = std::map<uint8_t, Mac48Address>;
    /**
     * Set the AID field of the given Association Response frame. In case of
     * multi-link setup, the selected AID value must be assigned to all the STAs
     * corresponding to the setup links. The AID value is selected among the AID
     * values that are possibly already assigned to the STAs affiliated with the
     * non-AP MLD we are associating with. If no STA has an assigned AID value,
     * a new AID value is selected.
     *
     * @param assoc the given Association Response frame
     * @param linkIdStaAddrMap a map of (link ID, remote STA address) of the links to setup
     */
    void SetAid(MgtAssocResponseHeader& assoc, const LinkIdStaAddrMap& linkIdStaAddrMap);
    /**
     * Get a map of (link ID, remote STA address) of the links to setup. Information
     * is taken from the given Association Response that is sent over the given link
     * to the given station.
     *
     * @param assoc the given Association Response frame
     * @param to the Receiver Address (RA) of the Association Response frame
     * @param linkId the ID of the link on which the Association Response frame is sent
     * @return a map of (link ID, remote STA address) of the links to setup
     */
    LinkIdStaAddrMap GetLinkIdStaAddrMap(MgtAssocResponseHeader& assoc,
                                         const Mac48Address& to,
                                         uint8_t linkId);
    /**
     * Forward an association or a reassociation response packet to the DCF/EDCA.
     *
     * @param to the address of the STA we are sending an association response to
     * @param isReassoc indicates whether it is a reassociation response
     * @param linkId the ID of the link on which the association response must be sent
     */
    void SendAssocResp(Mac48Address to, bool isReassoc, uint8_t linkId);
    /**
     * Forward a beacon packet to the beacon special DCF for transmission
     * on the given link.
     *
     * @param linkId the ID of the given link
     */
    void SendOneBeacon(uint8_t linkId);

    /**
     * Get the FILS Discovery frame to send on the given link.
     *
     * @param linkId the ID of the given link
     * @return the FILS Discovery frame to send on the given link
     */
    Ptr<WifiMpdu> GetFilsDiscovery(uint8_t linkId) const;

    /**
     * Schedule the transmission of FILS Discovery frames or unsolicited Probe Response frames
     * on the given link
     *
     * @param linkId the ID of the given link
     */
    void ScheduleFilsDiscOrUnsolProbeRespFrames(uint8_t linkId);

    /**
     * Process the Power Management bit in the Frame Control field of an MPDU
     * successfully received on the given link.
     *
     * @param mpdu the successfully received MPDU
     * @param linkId the ID of the given link
     */
    void ProcessPowerManagementFlag(Ptr<const WifiMpdu> mpdu, uint8_t linkId);
    /**
     * Perform the necessary actions when a given station switches from active mode
     * to powersave mode.
     *
     * @param staAddr the MAC address of the given station
     * @param linkId the ID of the link on which the given station is operating
     */
    void StaSwitchingToPsMode(const Mac48Address& staAddr, uint8_t linkId);
    /**
     * Perform the necessary actions when a given station deassociates or switches
     * from powersave mode to active mode.
     *
     * @param staAddr the MAC address of the given station
     * @param linkId the ID of the link on which the given station is operating
     */
    void StaSwitchingToActiveModeOrDeassociated(const Mac48Address& staAddr, uint8_t linkId);

    /**
     * Return the Capability information of the current AP for the given link.
     *
     * @param linkId the ID of the given link
     * @return the Capability information that we support
     */
    CapabilityInformation GetCapabilities(uint8_t linkId) const;
    /**
     * Return the ERP information of the current AP for the given link.
     *
     * @param linkId the ID of the given link
     * @return the ERP information that we support for the given link
     */
    ErpInformation GetErpInformation(uint8_t linkId) const;
    /**
     * Return the EDCA Parameter Set of the current AP for the given link.
     *
     * @param linkId the ID of the given link
     * @return the EDCA Parameter Set that we support for the given link
     */
    EdcaParameterSet GetEdcaParameterSet(uint8_t linkId) const;
    /**
     * Return the MU EDCA Parameter Set of the current AP, if one needs to be advertised
     *
     * @return the MU EDCA Parameter Set that needs to be advertised (if any)
     */
    std::optional<MuEdcaParameterSet> GetMuEdcaParameterSet() const;
    /**
     * Return the Reduced Neighbor Report (RNR) element that the current AP sends
     * on the given link, if one needs to be advertised.
     *
     * @param linkId the ID of the link to send the RNR element onto
     * @return the Reduced Neighbor Report element
     */
    std::optional<ReducedNeighborReport> GetReducedNeighborReport(uint8_t linkId) const;

    /**
     * Return the Multi-Link Element that the current AP includes in the management
     * frames of the given type it transmits on the given link.
     *
     * @param linkId the ID of the link to send the Multi-Link Element onto
     * @param frameType the type of the frame containing the Multi-Link Element
     * @param to the Receiver Address of the frame containing the Multi-Link Element
     * @param mlProbeReqMle the Multi-Link Element included in the multi-link probe request, in
     *                      case the Multi-Link Element is requested for the response to such a
     *                      frame
     * @return the Multi-Link Element
     */
    MultiLinkElement GetMultiLinkElement(
        uint8_t linkId,
        WifiMacType frameType,
        const Mac48Address& to = Mac48Address::GetBroadcast(),
        const std::optional<MultiLinkElement>& mlProbeReqMle = std::nullopt);

    /**
     * Return the HT operation of the current AP for the given link.
     *
     * @param linkId the ID of the given link
     * @return the HT operation that we support
     */
    HtOperation GetHtOperation(uint8_t linkId) const;
    /**
     * Return the VHT operation of the current AP for the given link.
     *
     * @param linkId the ID of the given link
     * @return the VHT operation that we support
     */
    VhtOperation GetVhtOperation(uint8_t linkId) const;
    /**
     * Return the HE operation of the current AP for the given link.
     *
     * @param linkId the ID of the given link
     * @return the HE operation that we support
     */
    HeOperation GetHeOperation(uint8_t linkId) const;
    /**
     * Return the EHT operation of the current AP for the given link.
     *
     * @param linkId the ID of the given link
     * @return the EHT operation that we support
     */
    EhtOperation GetEhtOperation(uint8_t linkId) const;
    /**
     * Return an instance of SupportedRates that contains all rates that we support
     * for the given link (including HT rates).
     *
     * @param linkId the ID of the given link
     * @return all rates that we support
     */
    AllSupportedRates GetSupportedRates(uint8_t linkId) const;
    /**
     * Return the DSSS Parameter Set that we support on the given link
     *
     * @param linkId the ID of the given link
     * @return the DSSS Parameter Set that we support on the given link
     */
    DsssParameterSet GetDsssParameterSet(uint8_t linkId) const;
    /**
     * Enable or disable beacon generation of the AP.
     *
     * @param enable enable or disable beacon generation
     */
    void SetBeaconGeneration(bool enable);

    /**
     * Update whether short slot time should be enabled or not in the BSS
     * corresponding to the given link.
     * Typically, short slot time is enabled only when there is no non-ERP station
     * associated  to the AP, and that short slot time is supported by the AP and by all
     * other ERP stations that are associated to the AP. Otherwise, it is disabled.
     *
     * @param linkId the ID of the given link
     */
    void UpdateShortSlotTimeEnabled(uint8_t linkId);
    /**
     * Update whether short preamble should be enabled or not in the BSS
     * corresponding to the given link.
     * Typically, short preamble is enabled only when the AP and all associated
     * stations support short PHY preamble. Otherwise, it is disabled.
     *
     * @param linkId the ID of the given link
     */
    void UpdateShortPreambleEnabled(uint8_t linkId);

    /**
     * Return whether protection for non-ERP stations is used in the BSS
     * corresponding to the given link.
     *
     * @param linkId the ID of the given link
     * @return true if protection for non-ERP stations is used in the BSS,
     *         false otherwise
     */
    bool GetUseNonErpProtection(uint8_t linkId) const;

    void DoDispose() override;
    void DoInitialize() override;

    Ptr<Txop> m_beaconTxop;        //!< Dedicated Txop for beacons
    bool m_enableBeaconGeneration; //!< Flag whether beacons are being generated
    Time m_beaconInterval;         //!< Beacon interval
    Ptr<UniformRandomVariable>
        m_beaconJitter; //!< UniformRandomVariable used to randomize the time of the first beacon
    bool m_enableBeaconJitter; //!< Flag whether the first beacon should be generated at random time
    bool m_enableNonErpProtection; //!< Flag whether protection mechanism is used or not when
                                   //!< non-ERP STAs are present within the BSS
    Time m_bsrLifetime;            //!< Lifetime of Buffer Status Reports
    /// transition timeout events running for EMLSR clients
    std::map<Mac48Address, EventId> m_transitionTimeoutEvents;
    uint8_t m_grpAddrBuIndicExp; //!< Group Addressed BU Indication Exponent of EHT Operation IE
    Ptr<ApEmlsrManager> m_apEmlsrManager; ///< AP EMLSR Manager
    Ptr<GcrManager> m_gcrManager;         //!< GCR Manager

    UintAccessParamsMap m_cwMinsForSta;     //!< Per-AC CW min values to advertise to stations
    UintAccessParamsMap m_cwMaxsForSta;     //!< Per-AC CW max values to advertise to stations
    UintAccessParamsMap m_aifsnsForSta;     //!< Per-AC AIFS values to advertise to stations
    TimeAccessParamsMap m_txopLimitsForSta; //!< Per-AC TXOP limits values to advertise to stations

    Time m_fdBeaconInterval6GHz;    //!< Time elapsing between a beacon and FILS Discovery (FD)
                                    //!< frame or between two FD frames on 6GHz links
    Time m_fdBeaconIntervalNon6GHz; //!< Time elapsing between a beacon and FILS Discovery (FD)
                                    //!< frame or between two FD frames on 2.4GHz and 5GHz links
    bool m_sendUnsolProbeResp;      //!< send unsolicited Probe Response instead of FILS Discovery

    /// store value and timestamp for each Buffer Status Report
    struct BsrType
    {
        uint8_t value;  //!< value of BSR
        Time timestamp; //!< timestamp of BSR
    };

    /// Per (MAC address, TID) buffer status reports
    std::unordered_map<WifiAddressTidPair, BsrType, WifiAddressTidHash> m_bufferStatus;

    /**
     * TracedCallback signature for association/deassociation events.
     *
     * @param aid the AID of the station
     * @param address the MAC address of the station
     */
    typedef void (*AssociationCallback)(uint16_t aid, Mac48Address address);

    TracedCallback<uint16_t /* AID */, Mac48Address> m_assocLogger;   ///< association logger
    TracedCallback<uint16_t /* AID */, Mac48Address> m_deAssocLogger; ///< deassociation logger
};

} // namespace ns3

#endif /* AP_WIFI_MAC_H */
