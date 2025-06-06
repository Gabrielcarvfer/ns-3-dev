/*
 * Copyright (c) 2005 INRIA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Emmanuelle Laprise <emmanuelle.laprise@bluekazoo.ca>
 */

#include "ethernet-header.h"

#include "address-utils.h"

#include "ns3/assert.h"
#include "ns3/header.h"
#include "ns3/log.h"

#include <iomanip>
#include <iostream>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("EthernetHeader");

NS_OBJECT_ENSURE_REGISTERED(EthernetHeader);

EthernetHeader::EthernetHeader(bool hasPreamble)
    : m_enPreambleSfd(hasPreamble),
      m_lengthType(0)
{
    NS_LOG_FUNCTION(this << hasPreamble);
}

EthernetHeader::EthernetHeader()
    : m_enPreambleSfd(false),
      m_lengthType(0)
{
    NS_LOG_FUNCTION(this);
}

void
EthernetHeader::SetLengthType(uint16_t lengthType)
{
    NS_LOG_FUNCTION(this << lengthType);
    m_lengthType = lengthType;
}

uint16_t
EthernetHeader::GetLengthType() const
{
    NS_LOG_FUNCTION(this);
    return m_lengthType;
}

void
EthernetHeader::SetPreambleSfd(uint64_t preambleSfd)
{
    NS_LOG_FUNCTION(this << preambleSfd);
    m_preambleSfd = preambleSfd;
}

uint64_t
EthernetHeader::GetPreambleSfd() const
{
    NS_LOG_FUNCTION(this);
    return m_preambleSfd;
}

void
EthernetHeader::SetSource(Mac48Address source)
{
    NS_LOG_FUNCTION(this << source);
    m_source = source;
}

Mac48Address
EthernetHeader::GetSource() const
{
    NS_LOG_FUNCTION(this);
    return m_source;
}

void
EthernetHeader::SetDestination(Mac48Address dst)
{
    NS_LOG_FUNCTION(this << dst);
    m_destination = dst;
}

Mac48Address
EthernetHeader::GetDestination() const
{
    NS_LOG_FUNCTION(this);
    return m_destination;
}

ethernet_header_t
EthernetHeader::GetPacketType() const
{
    NS_LOG_FUNCTION(this);
    return LENGTH;
}

uint32_t
EthernetHeader::GetHeaderSize() const
{
    NS_LOG_FUNCTION(this);
    return GetSerializedSize();
}

TypeId
EthernetHeader::GetTypeId()
{
    static TypeId tid = TypeId("ns3::EthernetHeader")
                            .SetParent<Header>()
                            .SetGroupName("Network")
                            .AddConstructor<EthernetHeader>();
    return tid;
}

TypeId
EthernetHeader::GetInstanceTypeId() const
{
    return GetTypeId();
}

void
EthernetHeader::Print(std::ostream& os) const
{
    NS_LOG_FUNCTION(this << &os);
    // ethernet, right ?
    if (m_enPreambleSfd)
    {
        os << "preamble/sfd=" << m_preambleSfd << ",";
    }

    os << " length/type=0x" << std::hex << m_lengthType << std::dec << ", source=" << m_source
       << ", destination=" << m_destination;
}

uint32_t
EthernetHeader::GetSerializedSize() const
{
    NS_LOG_FUNCTION(this);
    if (m_enPreambleSfd)
    {
        return PREAMBLE_SIZE + LENGTH_SIZE + 2 * MAC_ADDR_SIZE;
    }
    else
    {
        return LENGTH_SIZE + 2 * MAC_ADDR_SIZE;
    }
}

void
EthernetHeader::Serialize(Buffer::Iterator start) const
{
    NS_LOG_FUNCTION(this << &start);
    Buffer::Iterator i = start;

    if (m_enPreambleSfd)
    {
        i.WriteU64(m_preambleSfd);
    }
    WriteTo(i, m_destination);
    WriteTo(i, m_source);
    i.WriteHtonU16(m_lengthType);
}

uint32_t
EthernetHeader::Deserialize(Buffer::Iterator start)
{
    NS_LOG_FUNCTION(this << &start);
    Buffer::Iterator i = start;

    if (m_enPreambleSfd)
    {
        m_enPreambleSfd = i.ReadU64();
    }

    ReadFrom(i, m_destination);
    ReadFrom(i, m_source);
    m_lengthType = i.ReadNtohU16();

    return GetSerializedSize();
}

} // namespace ns3
