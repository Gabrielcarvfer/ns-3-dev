/*
 * Copyright (c) 2005 INRIA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "ipv4-header.h"

#include "ns3/abort.h"
#include "ns3/assert.h"
#include "ns3/header.h"
#include "ns3/log.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("Ipv4Header");

NS_OBJECT_ENSURE_REGISTERED(Ipv4Header);

Ipv4Header::Ipv4Header()
    : m_calcChecksum(false),
      m_payloadSize(0),
      m_identification(0),
      m_tos(0),
      m_ttl(0),
      m_protocol(0),
      m_flags(0),
      m_fragmentOffset(0),
      m_checksum(0),
      m_goodChecksum(true),
      m_headerSize(5 * 4)
{
}

void
Ipv4Header::EnableChecksum()
{
    NS_LOG_FUNCTION(this);
    m_calcChecksum = true;
}

void
Ipv4Header::SetPayloadSize(uint16_t size)
{
    NS_LOG_FUNCTION(this << size);
    m_payloadSize = size;
}

uint16_t
Ipv4Header::GetPayloadSize() const
{
    NS_LOG_FUNCTION(this);
    return m_payloadSize;
}

uint16_t
Ipv4Header::GetIdentification() const
{
    NS_LOG_FUNCTION(this);
    return m_identification;
}

void
Ipv4Header::SetIdentification(uint16_t identification)
{
    NS_LOG_FUNCTION(this << identification);
    m_identification = identification;
}

void
Ipv4Header::SetTos(uint8_t tos)
{
    NS_LOG_FUNCTION(this << static_cast<uint32_t>(tos));
    m_tos = tos;
}

void
Ipv4Header::SetDscp(DscpType dscp)
{
    NS_LOG_FUNCTION(this << dscp);
    m_tos &= 0x3; // Clear out the DSCP part, retain 2 bits of ECN
    m_tos |= (dscp << 2);
}

void
Ipv4Header::SetEcn(EcnType ecn)
{
    NS_LOG_FUNCTION(this << ecn);
    m_tos &= 0xFC; // Clear out the ECN part, retain 6 bits of DSCP
    m_tos |= ecn;
}

Ipv4Header::DscpType
Ipv4Header::GetDscp() const
{
    NS_LOG_FUNCTION(this);
    // Extract only first 6 bits of TOS byte, i.e 0xFC
    return DscpType((m_tos & 0xFC) >> 2);
}

std::string
Ipv4Header::DscpTypeToString(DscpType dscp) const
{
    NS_LOG_FUNCTION(this << dscp);
    switch (dscp)
    {
    case DscpDefault:
        return "Default";
    case DSCP_CS1:
        return "CS1";
    case DSCP_AF11:
        return "AF11";
    case DSCP_AF12:
        return "AF12";
    case DSCP_AF13:
        return "AF13";
    case DSCP_CS2:
        return "CS2";
    case DSCP_AF21:
        return "AF21";
    case DSCP_AF22:
        return "AF22";
    case DSCP_AF23:
        return "AF23";
    case DSCP_CS3:
        return "CS3";
    case DSCP_AF31:
        return "AF31";
    case DSCP_AF32:
        return "AF32";
    case DSCP_AF33:
        return "AF33";
    case DSCP_CS4:
        return "CS4";
    case DSCP_AF41:
        return "AF41";
    case DSCP_AF42:
        return "AF42";
    case DSCP_AF43:
        return "AF43";
    case DSCP_CS5:
        return "CS5";
    case DSCP_EF:
        return "EF";
    case DSCP_CS6:
        return "CS6";
    case DSCP_CS7:
        return "CS7";
    default:
        return "Unrecognized DSCP";
    };
}

Ipv4Header::EcnType
Ipv4Header::GetEcn() const
{
    NS_LOG_FUNCTION(this);
    // Extract only last 2 bits of TOS byte, i.e 0x3
    return EcnType(m_tos & 0x3);
}

std::string
Ipv4Header::EcnTypeToString(EcnType ecn) const
{
    NS_LOG_FUNCTION(this << ecn);
    switch (ecn)
    {
    case ECN_NotECT:
        return "Not-ECT";
    case ECN_ECT1:
        return "ECT (1)";
    case ECN_ECT0:
        return "ECT (0)";
    case ECN_CE:
        return "CE";
    default:
        return "Unknown ECN";
    };
}

uint8_t
Ipv4Header::GetTos() const
{
    NS_LOG_FUNCTION(this);
    return m_tos;
}

void
Ipv4Header::SetMoreFragments()
{
    NS_LOG_FUNCTION(this);
    m_flags |= MORE_FRAGMENTS;
}

void
Ipv4Header::SetLastFragment()
{
    NS_LOG_FUNCTION(this);
    m_flags &= ~MORE_FRAGMENTS;
}

bool
Ipv4Header::IsLastFragment() const
{
    NS_LOG_FUNCTION(this);
    return !(m_flags & MORE_FRAGMENTS);
}

void
Ipv4Header::SetDontFragment()
{
    NS_LOG_FUNCTION(this);
    m_flags |= DONT_FRAGMENT;
}

void
Ipv4Header::SetMayFragment()
{
    NS_LOG_FUNCTION(this);
    m_flags &= ~DONT_FRAGMENT;
}

bool
Ipv4Header::IsDontFragment() const
{
    NS_LOG_FUNCTION(this);
    return (m_flags & DONT_FRAGMENT);
}

void
Ipv4Header::SetFragmentOffset(uint16_t offsetBytes)
{
    NS_LOG_FUNCTION(this << offsetBytes);
    // check if the user is trying to set an invalid offset
    NS_ABORT_MSG_IF((offsetBytes & 0x7), "offsetBytes must be multiple of 8 bytes");
    m_fragmentOffset = offsetBytes;
}

uint16_t
Ipv4Header::GetFragmentOffset() const
{
    NS_LOG_FUNCTION(this);
    // -fstrict-overflow sensitive, see bug 1868
    if (m_fragmentOffset + m_payloadSize > 65535 - 5 * 4)
    {
        NS_LOG_WARN("Fragment will exceed the maximum packet size once reassembled");
    }

    return m_fragmentOffset;
}

void
Ipv4Header::SetTtl(uint8_t ttl)
{
    NS_LOG_FUNCTION(this << static_cast<uint32_t>(ttl));
    m_ttl = ttl;
}

uint8_t
Ipv4Header::GetTtl() const
{
    NS_LOG_FUNCTION(this);
    return m_ttl;
}

uint8_t
Ipv4Header::GetProtocol() const
{
    NS_LOG_FUNCTION(this);
    return m_protocol;
}

void
Ipv4Header::SetProtocol(uint8_t protocol)
{
    NS_LOG_FUNCTION(this << static_cast<uint32_t>(protocol));
    m_protocol = protocol;
}

namespace
{
/// Type of the loose source route option, from the IANA IP option numbers
constexpr uint8_t IPV4_OPTION_LSRR = 131;
/// Type of the end of option list
constexpr uint8_t IPV4_OPTION_END = 0;
/// Type of the no operation option, used as padding
constexpr uint8_t IPV4_OPTION_NOP = 1;
} // namespace

void
Ipv4Header::SetLooseSourceRoute(const std::vector<Ipv4Address>& route)
{
    NS_LOG_FUNCTION(this << route.size());
    m_sourceRoute = route;
    m_sourceRoutePointer = 4;
    m_headerSize = 20 + OptionsLength();
}

bool
Ipv4Header::HasLooseSourceRoute() const
{
    return !m_sourceRoute.empty();
}

std::vector<Ipv4Address>
Ipv4Header::GetLooseSourceRoute() const
{
    return m_sourceRoute;
}

uint8_t
Ipv4Header::GetSourceRoutePointer() const
{
    return m_sourceRoutePointer;
}

void
Ipv4Header::SetSourceRoutePointer(uint8_t pointer)
{
    m_sourceRoutePointer = pointer;
}

uint16_t
Ipv4Header::OptionsLength() const
{
    if (m_sourceRoute.empty())
    {
        return 0;
    }

    // Type, length and pointer, then the addresses, padded to a word
    uint16_t length = 3 + 4 * m_sourceRoute.size();
    return (length + 3) & ~0x3;
}

void
Ipv4Header::SetSource(Ipv4Address source)
{
    NS_LOG_FUNCTION(this << source);
    m_source = source;
}

Ipv4Address
Ipv4Header::GetSource() const
{
    NS_LOG_FUNCTION(this);
    return m_source;
}

void
Ipv4Header::SetDestination(Ipv4Address dst)
{
    NS_LOG_FUNCTION(this << dst);
    m_destination = dst;
}

Ipv4Address
Ipv4Header::GetDestination() const
{
    NS_LOG_FUNCTION(this);
    return m_destination;
}

bool
Ipv4Header::IsChecksumOk() const
{
    NS_LOG_FUNCTION(this);
    return m_goodChecksum;
}

TypeId
Ipv4Header::GetTypeId()
{
    static TypeId tid = TypeId("ns3::Ipv4Header")
                            .SetParent<Header>()
                            .SetGroupName("Internet")
                            .AddConstructor<Ipv4Header>();
    return tid;
}

TypeId
Ipv4Header::GetInstanceTypeId() const
{
    NS_LOG_FUNCTION(this);
    return GetTypeId();
}

void
Ipv4Header::Print(std::ostream& os) const
{
    NS_LOG_FUNCTION(this << &os);
    // ipv4, right ?
    std::string flags;
    if (m_flags == 0)
    {
        flags = "none";
    }
    else if ((m_flags & MORE_FRAGMENTS) && (m_flags & DONT_FRAGMENT))
    {
        flags = "MF|DF";
    }
    else if (m_flags & DONT_FRAGMENT)
    {
        flags = "DF";
    }
    else if (m_flags & MORE_FRAGMENTS)
    {
        flags = "MF";
    }
    else
    {
        flags = "XX";
    }
    os << "tos 0x" << std::hex << m_tos << std::dec << " "
       << "DSCP " << DscpTypeToString(GetDscp()) << " "
       << "ECN " << EcnTypeToString(GetEcn()) << " "
       << "ttl " << m_ttl << " "
       << "id " << m_identification << " "
       << "protocol " << m_protocol << " "
       << "offset (bytes) " << m_fragmentOffset << " "
       << "flags [" << flags << "] "
       << "length: " << (m_payloadSize + 5 * 4) << " " << m_source << " > " << m_destination;
}

uint32_t
Ipv4Header::GetSerializedSize() const
{
    NS_LOG_FUNCTION(this);
    // return 5 * 4;
    return m_headerSize;
}

void
Ipv4Header::Serialize(Buffer::Iterator start) const
{
    NS_LOG_FUNCTION(this << &start);
    Buffer::Iterator i = start;

    uint16_t optionsLength = OptionsLength();
    uint16_t headerLength = 20 + optionsLength;
    uint8_t verIhl = (4 << 4) | (headerLength / 4);
    i.WriteU8(verIhl);
    i.WriteU8(m_tos);
    i.WriteHtonU16(m_payloadSize + headerLength);
    i.WriteHtonU16(m_identification);
    uint32_t fragmentOffset = m_fragmentOffset / 8;
    uint8_t flagsFrag = (fragmentOffset >> 8) & 0x1f;
    if (m_flags & DONT_FRAGMENT)
    {
        flagsFrag |= (1 << 6);
    }
    if (m_flags & MORE_FRAGMENTS)
    {
        flagsFrag |= (1 << 5);
    }
    i.WriteU8(flagsFrag);
    uint8_t frag = fragmentOffset & 0xff;
    i.WriteU8(frag);
    i.WriteU8(m_ttl);
    i.WriteU8(m_protocol);
    i.WriteHtonU16(0);
    i.WriteHtonU32(m_source.Get());
    i.WriteHtonU32(m_destination.Get());

    if (optionsLength > 0)
    {
        // The loose source route option, as RFC 791 lays it out
        i.WriteU8(IPV4_OPTION_LSRR);
        i.WriteU8(3 + 4 * m_sourceRoute.size());
        i.WriteU8(m_sourceRoutePointer);
        for (const auto& address : m_sourceRoute)
        {
            i.WriteHtonU32(address.Get());
        }
        // The padding bringing the options to a word boundary
        for (uint16_t pad = 3 + 4 * m_sourceRoute.size(); pad < optionsLength; ++pad)
        {
            i.WriteU8(pad + 1 == optionsLength ? IPV4_OPTION_END : IPV4_OPTION_NOP);
        }
    }

    if (m_calcChecksum)
    {
        i = start;
        uint16_t checksum = i.CalculateIpChecksum(20);
        NS_LOG_LOGIC("checksum=" << checksum);
        i = start;
        i.Next(10);
        i.WriteU16(checksum);
    }
}

uint32_t
Ipv4Header::Deserialize(Buffer::Iterator start)
{
    NS_LOG_FUNCTION(this << &start);
    Buffer::Iterator i = start;

    uint8_t verIhl = i.ReadU8();
    uint8_t ihl = verIhl & 0x0f;
    uint16_t headerSize = ihl * 4;

    if ((verIhl >> 4) != 4)
    {
        NS_LOG_WARN("Trying to decode a non-IPv4 header, refusing to do it.");
        return 0;
    }

    m_tos = i.ReadU8();
    uint16_t size = i.ReadNtohU16();
    m_payloadSize = size - headerSize;
    m_identification = i.ReadNtohU16();
    uint8_t flags = i.ReadU8();
    m_flags = 0;
    if (flags & (1 << 6))
    {
        m_flags |= DONT_FRAGMENT;
    }
    if (flags & (1 << 5))
    {
        m_flags |= MORE_FRAGMENTS;
    }
    i.Prev();
    m_fragmentOffset = i.ReadU8() & 0x1f;
    m_fragmentOffset <<= 8;
    m_fragmentOffset |= i.ReadU8();
    m_fragmentOffset <<= 3;
    m_ttl = i.ReadU8();
    m_protocol = i.ReadU8();
    m_checksum = i.ReadU16();
    /* i.Next (2); // checksum */
    m_source.Set(i.ReadNtohU32());
    m_destination.Set(i.ReadNtohU32());
    m_headerSize = headerSize;

    m_sourceRoute.clear();
    m_sourceRoutePointer = 4;
    uint16_t optionsLength = headerSize > 20 ? headerSize - 20 : 0;
    while (optionsLength > 0)
    {
        uint8_t type = i.ReadU8();
        optionsLength--;

        if (type == IPV4_OPTION_END)
        {
            break;
        }
        if (type == IPV4_OPTION_NOP)
        {
            continue;
        }
        if (optionsLength == 0)
        {
            NS_LOG_WARN("IPv4 option without a length field");
            break;
        }

        uint8_t length = i.ReadU8();
        optionsLength--;
        if (length < 2 || length - 2 > optionsLength)
        {
            NS_LOG_WARN("Illegal IPv4 option length " << static_cast<uint32_t>(length));
            break;
        }

        if (type == IPV4_OPTION_LSRR && length >= 3 && (length - 3) % 4 == 0)
        {
            m_sourceRoutePointer = i.ReadU8();
            for (uint8_t read = 3; read < length; read += 4)
            {
                m_sourceRoute.emplace_back(i.ReadNtohU32());
            }
        }
        else
        {
            i.Next(length - 2);
        }
        optionsLength -= length - 2;
    }

    if (m_calcChecksum)
    {
        i = start;
        uint16_t checksum = i.CalculateIpChecksum(headerSize);
        NS_LOG_LOGIC("checksum=" << checksum);

        m_goodChecksum = (checksum == 0);
    }
    return GetSerializedSize();
}

} // namespace ns3
