/*
 * Copyright (c) 2008 INRIA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#include "loopback-net-device.h"

#include "ns3/channel.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("LoopbackNetDevice");

NS_OBJECT_ENSURE_REGISTERED(LoopbackNetDevice);

TypeId
LoopbackNetDevice::GetTypeId()
{
    static TypeId tid = TypeId("ns3::LoopbackNetDevice")
                            .SetParent<NetDevice>()
                            .SetGroupName("Internet")
                            .AddConstructor<LoopbackNetDevice>();
    return tid;
}

LoopbackNetDevice::LoopbackNetDevice()
    : m_node(nullptr),
      m_mtu(0xffff),
      m_ifIndex(0),
      m_address(Mac48Address("00:00:00:00:00:00"))
{
    NS_LOG_FUNCTION(this);
}

void
LoopbackNetDevice::Receive(Ptr<Packet> packet,
                           uint16_t protocol,
                           Mac48Address to,
                           Mac48Address from)
{
    NS_LOG_FUNCTION(packet << " " << protocol << " " << to << " " << from);
    NetDevice::PacketType packetType;
    if (to == m_address || to.IsBroadcast())
    {
        packetType = NetDevice::PACKET_HOST;
    }
    else if (to.IsGroup())
    {
        packetType = NetDevice::PACKET_MULTICAST;
    }
    else
    {
        packetType = NetDevice::PACKET_OTHERHOST;
    }
    m_rxCallback(this, packet, protocol, from);
    if (!m_promiscCallback.IsNull())
    {
        m_promiscCallback(this, packet, protocol, from, to, packetType);
    }
}

void
LoopbackNetDevice::SetIfIndex(const uint32_t index)
{
    m_ifIndex = index;
}

uint32_t
LoopbackNetDevice::GetIfIndex() const
{
    return m_ifIndex;
}

Ptr<Channel>
LoopbackNetDevice::GetChannel() const
{
    return nullptr;
}

void
LoopbackNetDevice::SetAddress(Address address)
{
    m_address = Mac48Address::ConvertFrom(address);
}

Address
LoopbackNetDevice::GetAddress() const
{
    return m_address;
}

bool
LoopbackNetDevice::SetMtu(const uint16_t mtu)
{
    m_mtu = mtu;
    return true;
}

uint16_t
LoopbackNetDevice::GetMtu() const
{
    return m_mtu;
}

bool
LoopbackNetDevice::IsLinkUp() const
{
    return true;
}

void
LoopbackNetDevice::AddLinkChangeCallback(Callback<void> callback)
{
}

bool
LoopbackNetDevice::IsBroadcast() const
{
    return true;
}

Address
LoopbackNetDevice::GetBroadcast() const
{
    // This is typically set to all zeros rather than all ones in real systems
    return Mac48Address("00:00:00:00:00:00");
}

bool
LoopbackNetDevice::IsMulticast() const
{
    // Multicast loopback will need to be supported for outgoing
    // datagrams but this will probably be handled in multicast sockets
    return false;
}

Address
LoopbackNetDevice::GetMulticast(Ipv4Address multicastGroup) const
{
    return Mac48Address::GetMulticast(multicastGroup);
}

Address
LoopbackNetDevice::GetMulticast(Ipv6Address addr) const
{
    return Mac48Address::GetMulticast(addr);
}

bool
LoopbackNetDevice::IsPointToPoint() const
{
    return false;
}

bool
LoopbackNetDevice::IsBridge() const
{
    return false;
}

bool
LoopbackNetDevice::Send(Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber)
{
    NS_LOG_FUNCTION(packet << " " << dest << " " << protocolNumber);
    Mac48Address to = Mac48Address::ConvertFrom(dest);
    NS_ASSERT_MSG(to == GetBroadcast() || to == m_address, "Invalid destination address");
    Simulator::ScheduleWithContext(m_node->GetId(),
                                   Seconds(0),
                                   &LoopbackNetDevice::Receive,
                                   this,
                                   packet,
                                   protocolNumber,
                                   to,
                                   m_address);
    return true;
}

bool
LoopbackNetDevice::SendFrom(Ptr<Packet> packet,
                            const Address& source,
                            const Address& dest,
                            uint16_t protocolNumber)
{
    NS_LOG_FUNCTION(packet << " " << source << " " << dest << " " << protocolNumber);
    Mac48Address to = Mac48Address::ConvertFrom(dest);
    Mac48Address from = Mac48Address::ConvertFrom(source);
    NS_ASSERT_MSG(to.IsBroadcast() || to == m_address, "Invalid destination address");
    Simulator::ScheduleWithContext(m_node->GetId(),
                                   Seconds(0),
                                   &LoopbackNetDevice::Receive,
                                   this,
                                   packet,
                                   protocolNumber,
                                   to,
                                   from);
    return true;
}

Ptr<Node>
LoopbackNetDevice::GetNode() const
{
    return m_node;
}

void
LoopbackNetDevice::SetNode(Ptr<Node> node)
{
    m_node = node;
}

bool
LoopbackNetDevice::NeedsArp() const
{
    return false;
}

void
LoopbackNetDevice::SetReceiveCallback(NetDevice::ReceiveCallback cb)
{
    m_rxCallback = cb;
}

void
LoopbackNetDevice::DoDispose()
{
    m_node = nullptr;
    NetDevice::DoDispose();
}

void
LoopbackNetDevice::SetPromiscReceiveCallback(PromiscReceiveCallback cb)
{
    m_promiscCallback = cb;
}

bool
LoopbackNetDevice::SupportsSendFrom() const
{
    return true;
}

} // namespace ns3
