/*
 * Copyright (c) 2008,2009 IITP RAS
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Kirill Andreev <andreev@iitp.ru>
 */

#include "ie-dot11s-prep.h"

#include "ns3/address-utils.h"
#include "ns3/assert.h"
#include "ns3/packet.h"

namespace ns3
{
namespace dot11s
{
/********************************
 * IePrep
 *******************************/
IePrep::~IePrep()
{
}

IePrep::IePrep()
    : m_flags(0),
      m_hopcount(0),
      m_ttl(0),
      m_destinationAddress(Mac48Address::GetBroadcast()),
      m_destSeqNumber(0),
      m_lifetime(0),
      m_metric(0),
      m_originatorAddress(Mac48Address::GetBroadcast()),
      m_originatorSeqNumber(0)
{
}

WifiInformationElementId
IePrep::ElementId() const
{
    return IE_PREP;
}

void
IePrep::SetFlags(uint8_t flags)
{
    m_flags = flags;
}

void
IePrep::SetHopcount(uint8_t hopcount)
{
    m_hopcount = hopcount;
}

void
IePrep::SetTtl(uint8_t ttl)
{
    m_ttl = ttl;
}

void
IePrep::SetDestinationSeqNumber(uint32_t destSeqNumber)
{
    m_destSeqNumber = destSeqNumber;
}

void
IePrep::SetDestinationAddress(Mac48Address destAddress)
{
    m_destinationAddress = destAddress;
}

void
IePrep::SetMetric(uint32_t metric)
{
    m_metric = metric;
}

void
IePrep::SetOriginatorAddress(Mac48Address originatorAddress)
{
    m_originatorAddress = originatorAddress;
}

void
IePrep::SetOriginatorSeqNumber(uint32_t originatorSeqNumber)
{
    m_originatorSeqNumber = originatorSeqNumber;
}

void
IePrep::SetLifetime(uint32_t lifetime)
{
    m_lifetime = lifetime;
}

uint8_t
IePrep::GetFlags() const
{
    return m_flags;
}

uint8_t
IePrep::GetHopcount() const
{
    return m_hopcount;
}

uint32_t
IePrep::GetTtl() const
{
    return m_ttl;
}

uint32_t
IePrep::GetDestinationSeqNumber() const
{
    return m_destSeqNumber;
}

Mac48Address
IePrep::GetDestinationAddress() const
{
    return m_destinationAddress;
}

uint32_t
IePrep::GetMetric() const
{
    return m_metric;
}

Mac48Address
IePrep::GetOriginatorAddress() const
{
    return m_originatorAddress;
}

uint32_t
IePrep::GetOriginatorSeqNumber() const
{
    return m_originatorSeqNumber;
}

uint32_t
IePrep::GetLifetime() const
{
    return m_lifetime;
}

void
IePrep::DecrementTtl()
{
    m_ttl--;
    m_hopcount++;
}

void
IePrep::IncrementMetric(uint32_t metric)
{
    m_metric += metric;
}

void
IePrep::SerializeInformationField(Buffer::Iterator i) const
{
    i.WriteU8(m_flags);
    i.WriteU8(m_hopcount);
    i.WriteU8(m_ttl);
    WriteTo(i, m_destinationAddress);
    i.WriteHtolsbU32(m_destSeqNumber);
    i.WriteHtolsbU32(m_lifetime);
    i.WriteHtolsbU32(m_metric);
    WriteTo(i, m_originatorAddress);
    i.WriteHtolsbU32(m_originatorSeqNumber);
}

uint16_t
IePrep::DeserializeInformationField(Buffer::Iterator start, uint16_t length)
{
    Buffer::Iterator i = start;
    m_flags = i.ReadU8();
    m_hopcount = i.ReadU8();
    m_ttl = i.ReadU8();
    ReadFrom(i, m_destinationAddress);
    m_destSeqNumber = i.ReadLsbtohU32();
    m_lifetime = i.ReadLsbtohU32();
    m_metric = i.ReadLsbtohU32();
    ReadFrom(i, m_originatorAddress);
    m_originatorSeqNumber = i.ReadLsbtohU32();
    return i.GetDistanceFrom(start);
}

uint16_t
IePrep::GetInformationFieldSize() const
{
    uint32_t retval = 1    // Flags
                      + 1  // Hopcount
                      + 1  // Ttl
                      + 6  // Dest address
                      + 4  // Dest seqno
                      + 4  // Lifetime
                      + 4  // metric
                      + 6  // Originator address
                      + 4; // Originator seqno
    return retval;
}

void
IePrep::Print(std::ostream& os) const
{
    os << "PREP=(Flags=" << +m_flags << ", Hopcount=" << +m_hopcount << ", TTL=" << m_ttl
       << ",Destination=" << m_destinationAddress << ", Dest. seqnum=" << m_destSeqNumber
       << ", Lifetime=" << m_lifetime << ", Metric=" << m_metric
       << ", Originator=" << m_originatorAddress << ", Orig. seqnum=" << m_originatorSeqNumber
       << ")";
}

bool
operator==(const IePrep& a, const IePrep& b)
{
    return ((a.m_flags == b.m_flags) && (a.m_hopcount == b.m_hopcount) && (a.m_ttl == b.m_ttl) &&
            (a.m_destinationAddress == b.m_destinationAddress) &&
            (a.m_destSeqNumber == b.m_destSeqNumber) && (a.m_lifetime == b.m_lifetime) &&
            (a.m_metric == b.m_metric) && (a.m_originatorAddress == b.m_originatorAddress) &&
            (a.m_originatorSeqNumber == b.m_originatorSeqNumber));
}

std::ostream&
operator<<(std::ostream& os, const IePrep& a)
{
    a.Print(os);
    return os;
}
} // namespace dot11s
} // namespace ns3
