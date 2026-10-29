/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "wifi-monitor-net-device.h"

#include "wifi-mac-header.h"
#include "wifi-net-device.h"
#include "wifi-phy.h"
#include "wifi-radiotap.h"

#include "ns3/channel.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/uinteger.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("WifiMonitorNetDevice");

NS_OBJECT_ENSURE_REGISTERED(WifiMonitorNetDevice);

TypeId
WifiMonitorNetDevice::GetTypeId()
{
    static TypeId tid = TypeId("ns3::WifiMonitorNetDevice")
                            .SetParent<NetDevice>()
                            .AddConstructor<WifiMonitorNetDevice>()
                            .SetGroupName("Wifi")
                            .AddAttribute("Mtu",
                                          "The MAC-level Maximum Transmission Unit",
                                          UintegerValue(2304),
                                          MakeUintegerAccessor(&WifiMonitorNetDevice::SetMtu,
                                                               &WifiMonitorNetDevice::GetMtu),
                                          MakeUintegerChecker<uint16_t>(1, 2304));
    return tid;
}

WifiMonitorNetDevice::WifiMonitorNetDevice()
    : m_ifIndex(0),
      m_mtu(2304)
{
    NS_LOG_FUNCTION_NOARGS();
}

WifiMonitorNetDevice::~WifiMonitorNetDevice()
{
    NS_LOG_FUNCTION_NOARGS();
}

void
WifiMonitorNetDevice::DoDispose()
{
    NS_LOG_FUNCTION_NOARGS();
    m_node = nullptr;
    m_device = nullptr;
    m_rxCallback = MakeNullCallback<bool,
                                    Ptr<NetDevice>,
                                    Ptr<const Packet>,
                                    uint16_t,
                                    const Address&>();
    m_promiscCallback = MakeNullCallback<bool,
                                         Ptr<NetDevice>,
                                         Ptr<const Packet>,
                                         uint16_t,
                                         const Address&,
                                         const Address&,
                                         PacketType>();
    NetDevice::DoDispose();
}

void
WifiMonitorNetDevice::SetWifiNetDevice(Ptr<WifiNetDevice> device)
{
    NS_LOG_FUNCTION(this << device);
    NS_ASSERT_MSG(!m_device, "The monitor interface already has a device");
    NS_ASSERT_MSG(device->GetNPhys() > 0, "The device has no PHY yet");
    m_device = device;
    for (uint8_t phyId = 0; phyId < device->GetNPhys(); ++phyId)
    {
        auto phy = device->GetPhy(phyId);
        phy->TraceConnectWithoutContext(
            "MonitorSnifferTx",
            MakeCallback(&WifiMonitorNetDevice::SniffTx, this).Bind(phyId));
        phy->TraceConnectWithoutContext(
            "MonitorSnifferRx",
            MakeCallback(&WifiMonitorNetDevice::SniffRx, this).Bind(phyId));
    }
}

Ptr<WifiNetDevice>
WifiMonitorNetDevice::GetWifiNetDevice() const
{
    return m_device;
}

void
WifiMonitorNetDevice::SniffTx(uint8_t phyId,
                              Ptr<const Packet> packet,
                              uint16_t channelFreqMhz,
                              WifiTxVector txVector,
                              MpduInfo aMpdu,
                              uint16_t staId)
{
    Deliver(phyId, packet, channelFreqMhz, txVector, aMpdu, staId, std::nullopt);
}

void
WifiMonitorNetDevice::SniffRx(uint8_t phyId,
                              Ptr<const Packet> packet,
                              uint16_t channelFreqMhz,
                              WifiTxVector txVector,
                              MpduInfo aMpdu,
                              SignalNoiseDbm signalNoise,
                              uint16_t staId)
{
    Deliver(phyId, packet, channelFreqMhz, txVector, aMpdu, staId, signalNoise);
}

void
WifiMonitorNetDevice::Deliver(uint8_t phyId,
                              Ptr<const Packet> packet,
                              uint16_t channelFreqMhz,
                              const WifiTxVector& txVector,
                              MpduInfo aMpdu,
                              uint16_t staId,
                              std::optional<SignalNoiseDbm> signalNoise)
{
    NS_LOG_FUNCTION(this << +phyId << packet << channelFreqMhz << txVector << staId);
    bool eof;
    Ptr<Packet> frame = GetMonitorMpdu(packet, txVector, eof);
    WifiMacHeader hdr;
    frame->PeekHeader(hdr);
    // the transmitter, if the frame has one (ACK and CTS frames do not)
    Mac48Address from = hdr.IsCts() || hdr.IsAck() ? Mac48Address() : hdr.GetAddr2();
    Mac48Address to = hdr.GetAddr1();
    frame->AddHeader(GetRadiotapHeader(channelFreqMhz,
                                       m_device->GetPhy(phyId)->GetPrimary20Index(),
                                       txVector,
                                       aMpdu,
                                       eof,
                                       staId,
                                       signalNoise));
    if (!m_promiscCallback.IsNull())
    {
        m_promiscCallback(this, frame, PROTOCOL, from, to, NetDevice::PACKET_OTHERHOST);
    }
    if (!m_rxCallback.IsNull())
    {
        m_rxCallback(this, frame, PROTOCOL, from);
    }
}

void
WifiMonitorNetDevice::SetIfIndex(const uint32_t index)
{
    m_ifIndex = index;
}

uint32_t
WifiMonitorNetDevice::GetIfIndex() const
{
    return m_ifIndex;
}

Ptr<Channel>
WifiMonitorNetDevice::GetChannel() const
{
    return m_device ? m_device->GetChannel() : nullptr;
}

void
WifiMonitorNetDevice::SetAddress(Address address)
{
    NS_LOG_FUNCTION(this << address);
    NS_LOG_WARN("The address of a monitor interface is that of its radio");
}

Address
WifiMonitorNetDevice::GetAddress() const
{
    return m_device ? m_device->GetAddress() : Address(Mac48Address());
}

bool
WifiMonitorNetDevice::SetMtu(const uint16_t mtu)
{
    m_mtu = mtu;
    return true;
}

uint16_t
WifiMonitorNetDevice::GetMtu() const
{
    return m_mtu;
}

bool
WifiMonitorNetDevice::IsLinkUp() const
{
    return m_device != nullptr;
}

void
WifiMonitorNetDevice::AddLinkChangeCallback(Callback<void> callback)
{
}

bool
WifiMonitorNetDevice::IsBroadcast() const
{
    return false;
}

Address
WifiMonitorNetDevice::GetBroadcast() const
{
    return Mac48Address::GetBroadcast();
}

bool
WifiMonitorNetDevice::IsMulticast() const
{
    return false;
}

Address
WifiMonitorNetDevice::GetMulticast(Ipv4Address multicastGroup) const
{
    return Mac48Address::GetMulticast(multicastGroup);
}

Address
WifiMonitorNetDevice::GetMulticast(Ipv6Address addr) const
{
    return Mac48Address::GetMulticast(addr);
}

bool
WifiMonitorNetDevice::IsPointToPoint() const
{
    return false;
}

bool
WifiMonitorNetDevice::IsBridge() const
{
    return false;
}

bool
WifiMonitorNetDevice::Send(Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber)
{
    NS_LOG_FUNCTION(this << packet << dest << protocolNumber);
    NS_LOG_WARN("A monitor interface does not transmit");
    return false;
}

bool
WifiMonitorNetDevice::SendFrom(Ptr<Packet> packet,
                               const Address& source,
                               const Address& dest,
                               uint16_t protocolNumber)
{
    return Send(packet, dest, protocolNumber);
}

Ptr<Node>
WifiMonitorNetDevice::GetNode() const
{
    return m_node;
}

void
WifiMonitorNetDevice::SetNode(const Ptr<Node> node)
{
    m_node = node;
}

bool
WifiMonitorNetDevice::NeedsArp() const
{
    return false;
}

void
WifiMonitorNetDevice::SetReceiveCallback(NetDevice::ReceiveCallback cb)
{
    m_rxCallback = cb;
}

void
WifiMonitorNetDevice::SetPromiscReceiveCallback(PromiscReceiveCallback cb)
{
    m_promiscCallback = cb;
}

bool
WifiMonitorNetDevice::SupportsSendFrom() const
{
    return false;
}

} // namespace ns3
