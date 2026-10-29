/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#ifndef WIFI_MONITOR_NET_DEVICE_H
#define WIFI_MONITOR_NET_DEVICE_H

#include "wifi-phy-common.h"
#include "wifi-tx-vector.h"
#include "wifi-types.h"

#include "ns3/net-device.h"

#include <optional>
namespace ns3
{

class WifiNetDevice;

/**
 * @ingroup wifi
 *
 * A monitor interface of the radio of a WifiNetDevice, like the monitor
 * interfaces of Linux (iw dev wlan0 interface add mon0 type monitor): it
 * delivers every MPDU that the PHYs of the WifiNetDevice receive or transmit,
 * whoever it is addressed to and whatever its type, prefixed by a Radiotap
 * header and without the FCS (WifiPhyHelper::DLT_IEEE802_11_RADIO pcap traces
 * have the same frames). The WifiNetDevice keeps working as usual.
 *
 * Frames are delivered with protocol 0x0004 (ETH_P_802_2), as Linux does: to
 * the protocol handlers of the node, and to its promiscuous handlers as
 * PACKET_OTHERHOST. A monitor interface does not transmit: Send fails.
 */
class WifiMonitorNetDevice : public NetDevice
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    WifiMonitorNetDevice();
    ~WifiMonitorNetDevice() override;

    /// The protocol of the delivered frames (ETH_P_802_2)
    static constexpr uint16_t PROTOCOL = 0x0004;

    /**
     * Monitor the radio of a WifiNetDevice: all its PHYs.
     *
     * @param device the WifiNetDevice, with its PHYs
     */
    void SetWifiNetDevice(Ptr<WifiNetDevice> device);

    /// @return the monitored WifiNetDevice
    Ptr<WifiNetDevice> GetWifiNetDevice() const;

    // Inherited from NetDevice base class.
    void SetIfIndex(const uint32_t index) override;
    uint32_t GetIfIndex() const override;
    Ptr<Channel> GetChannel() const override;
    void SetAddress(Address address) override;
    Address GetAddress() const override;
    bool SetMtu(const uint16_t mtu) override;
    uint16_t GetMtu() const override;
    bool IsLinkUp() const override;
    void AddLinkChangeCallback(Callback<void> callback) override;
    bool IsBroadcast() const override;
    Address GetBroadcast() const override;
    bool IsMulticast() const override;
    Address GetMulticast(Ipv4Address multicastGroup) const override;
    Address GetMulticast(Ipv6Address addr) const override;
    bool IsPointToPoint() const override;
    bool IsBridge() const override;
    bool Send(Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber) override;
    bool SendFrom(Ptr<Packet> packet,
                  const Address& source,
                  const Address& dest,
                  uint16_t protocolNumber) override;
    Ptr<Node> GetNode() const override;
    void SetNode(const Ptr<Node> node) override;
    bool NeedsArp() const override;
    void SetReceiveCallback(NetDevice::ReceiveCallback cb) override;
    void SetPromiscReceiveCallback(PromiscReceiveCallback cb) override;
    bool SupportsSendFrom() const override;

  protected:
    void DoDispose() override;

  private:
    /**
     * Called for each MPDU transmitted by a PHY of the monitored device.
     *
     * @param phyId the ID of the PHY
     * @param packet the MPDU
     * @param channelFreqMhz the channel frequency
     * @param txVector the TXVECTOR
     * @param aMpdu the A-MPDU information
     * @param staId the STA-ID
     */
    void SniffTx(uint8_t phyId,
                 Ptr<const Packet> packet,
                 uint16_t channelFreqMhz,
                 WifiTxVector txVector,
                 MpduInfo aMpdu,
                 uint16_t staId);

    /**
     * Called for each MPDU received by a PHY of the monitored device.
     *
     * @param phyId the ID of the PHY
     * @param packet the MPDU
     * @param channelFreqMhz the channel frequency
     * @param txVector the TXVECTOR
     * @param aMpdu the A-MPDU information
     * @param signalNoise the signal and noise
     * @param staId the STA-ID
     */
    void SniffRx(uint8_t phyId,
                 Ptr<const Packet> packet,
                 uint16_t channelFreqMhz,
                 WifiTxVector txVector,
                 MpduInfo aMpdu,
                 SignalNoiseDbm signalNoise,
                 uint16_t staId);

    /**
     * Deliver a frame to the node.
     *
     * @param phyId the ID of the PHY
     * @param packet the MPDU, as traced by the PHY
     * @param channelFreqMhz the channel frequency
     * @param txVector the TXVECTOR
     * @param aMpdu the A-MPDU information
     * @param staId the STA-ID
     * @param signalNoise the signal and noise (received frames only)
     */
    void Deliver(uint8_t phyId,
                 Ptr<const Packet> packet,
                 uint16_t channelFreqMhz,
                 const WifiTxVector& txVector,
                 MpduInfo aMpdu,
                 uint16_t staId,
                 std::optional<SignalNoiseDbm> signalNoise);

    Ptr<Node> m_node;                         //!< the node
    Ptr<WifiNetDevice> m_device;              //!< the monitored device
    uint32_t m_ifIndex;                       //!< the interface index
    uint16_t m_mtu;                           //!< the MTU
    NetDevice::ReceiveCallback m_rxCallback;  //!< the receive callback
    PromiscReceiveCallback m_promiscCallback; //!< the promiscuous receive callback
};

} // namespace ns3

#endif /* WIFI_MONITOR_NET_DEVICE_H */
