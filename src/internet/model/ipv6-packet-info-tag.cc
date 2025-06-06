/*
 * Copyright (c) 2010 Hajime Tazaki
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Hajime Tazaki <tazaki@sfc.wide.ad.jp>
 */

#include "ipv6-packet-info-tag.h"

#include "ns3/ipv6-address.h"

#include <cstdint>

namespace ns3
{

Ipv6PacketInfoTag::Ipv6PacketInfoTag()
    : m_addr(Ipv6Address()),
      m_ifindex(0),
      m_hoplimit(0),
      m_tclass(0)
{
}

void
Ipv6PacketInfoTag::SetAddress(Ipv6Address addr)
{
    m_addr = addr;
}

Ipv6Address
Ipv6PacketInfoTag::GetAddress() const
{
    return m_addr;
}

void
Ipv6PacketInfoTag::SetRecvIf(uint32_t ifindex)
{
    m_ifindex = ifindex;
}

uint32_t
Ipv6PacketInfoTag::GetRecvIf() const
{
    return m_ifindex;
}

void
Ipv6PacketInfoTag::SetHoplimit(uint8_t ttl)
{
    m_hoplimit = ttl;
}

uint8_t
Ipv6PacketInfoTag::GetHoplimit() const
{
    return m_hoplimit;
}

void
Ipv6PacketInfoTag::SetTrafficClass(uint8_t tclass)
{
    m_tclass = tclass;
}

uint8_t
Ipv6PacketInfoTag::GetTrafficClass() const
{
    return m_tclass;
}

TypeId
Ipv6PacketInfoTag::GetTypeId()
{
    static TypeId tid = TypeId("ns3::Ipv6PacketInfoTag")
                            .SetParent<Tag>()
                            .SetGroupName("Internet")
                            .AddConstructor<Ipv6PacketInfoTag>();
    return tid;
}

TypeId
Ipv6PacketInfoTag::GetInstanceTypeId() const
{
    return GetTypeId();
}

uint32_t
Ipv6PacketInfoTag::GetSerializedSize() const
{
    return 16 + sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint8_t);
}

void
Ipv6PacketInfoTag::Serialize(TagBuffer i) const
{
    uint8_t buf[16];
    m_addr.Serialize(buf);
    i.Write(buf, 16);
    i.WriteU8(m_ifindex);
    i.WriteU8(m_hoplimit);
    i.WriteU8(m_tclass);
}

void
Ipv6PacketInfoTag::Deserialize(TagBuffer i)
{
    uint8_t buf[16];
    i.Read(buf, 16);
    m_addr = Ipv6Address::Deserialize(buf);
    m_ifindex = i.ReadU8();
    m_hoplimit = i.ReadU8();
    m_tclass = i.ReadU8();
}

void
Ipv6PacketInfoTag::Print(std::ostream& os) const
{
    os << "Ipv6 PKTINFO [DestAddr: " << m_addr;
    os << ", RecvIf:" << (uint32_t)m_ifindex;
    os << ", TTL:" << (uint32_t)m_hoplimit;
    os << ", TClass:" << (uint32_t)m_tclass;
    os << "] ";
}
} // namespace ns3
