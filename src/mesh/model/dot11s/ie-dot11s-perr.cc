/*
 * Copyright (c) 2008,2009 IITP RAS
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Kirill Andreev <andreev@iitp.ru>
 */

#include "ie-dot11s-perr.h"

#include "ns3/address-utils.h"
#include "ns3/packet.h"

namespace ns3
{
namespace dot11s
{

namespace
{
/// Initial element TTL of the PERRs sent (dot11MeshElementTTL default, IEEE 802.11-2012, C.3)
constexpr uint8_t PERR_TTL = 31;
/// AE subfield of the per-destination Flags field (IEEE 802.11-2012, 8.4.2.115)
constexpr uint8_t PERR_FLAG_AE = 1 << 6;
/// MESH-PATH-ERROR-DESTINATION-UNREACHABLE (IEEE 802.11-2012, Table 8-36)
constexpr uint16_t PERR_REASON_DESTINATION_UNREACHABLE = 63;
} // namespace

IePerr::IePerr()
{
}

IePerr::~IePerr()
{
}

WifiInformationElementId
IePerr::ElementId() const
{
    return IE_PERR;
}

void
IePerr::Print(std::ostream& os) const
{
    os << "PERR=(Number of failed destinations=" << m_addressUnits.size();
    for (unsigned int j = 0; j < m_addressUnits.size(); j++)
    {
        os << "(Failed destination address=" << m_addressUnits[j].destination
           << ", sequence number = " << m_addressUnits[j].seqnum << ")";
    }
    os << ")";
}

uint8_t
IePerr::GetNumOfDest() const
{
    return m_addressUnits.size();
}

void
IePerr::SerializeInformationField(Buffer::Iterator i) const
{
    // Each mesh STA sends its own PERR rather than forwarding the one it received: the element
    // TTL is always the initial one. Receivers drop PERRs with a TTL of 1 or less.
    i.WriteU8(PERR_TTL);
    i.WriteU8(m_addressUnits.size()); // number of Destinations
    for (unsigned int j = 0; j < m_addressUnits.size(); j++)
    {
        i.WriteU8(0); // Flags: no destination external address (AE subfield, bit 6)
        WriteTo(i, m_addressUnits[j].destination);
        i.WriteU32(m_addressUnits[j].seqnum);
        i.WriteHtolsbU16(PERR_REASON_DESTINATION_UNREACHABLE);
    }
}

uint16_t
IePerr::DeserializeInformationField(Buffer::Iterator start, uint16_t length)
{
    Buffer::Iterator i = start;
    i.Next(1); // TTL: not used
    uint8_t numOfDest = i.ReadU8();
    for (unsigned int j = 0; j < numOfDest && i.GetDistanceFrom(start) + 13 <= length; j++)
    {
        uint8_t flags = i.ReadU8();
        HwmpProtocol::FailedDestination unit;
        ReadFrom(i, unit.destination);
        unit.seqnum = i.ReadU32();
        if (flags & PERR_FLAG_AE)
        {
            i.Next(6); // Destination External Address: not used
        }
        m_addressUnits.push_back(unit);
        i.Next(2); // Reason Code: not used
    }
    i.Next(length - i.GetDistanceFrom(start));
    return i.GetDistanceFrom(start);
}

uint16_t
IePerr::GetInformationFieldSize() const
{
    uint16_t retval = 1                           // TTL   //ModeFlags
                      + 1                         // NumOfDests
                      + 1 * m_addressUnits.size() // ModeFlags
                      + (6 + 4) * m_addressUnits.size() + 2 * m_addressUnits.size(); // Reason Code
    return retval;
}

void
IePerr::AddAddressUnit(HwmpProtocol::FailedDestination unit)
{
    for (unsigned int i = 0; i < m_addressUnits.size(); i++)
    {
        if (m_addressUnits[i].destination == unit.destination)
        {
            return;
        }
    }
    if ((m_addressUnits.size() + 1) * 13 + 2 > 255)
    {
        return;
    }
    m_addressUnits.push_back(unit);
}

bool
IePerr::IsFull() const
{
    // -fstrict-overflow sensitive, see bug 1868
    return (GetInformationFieldSize() > 255 - 2 /* ID + LENGTH*/
                                            -
                                            13 // 10 /* Size of Mac48Address + uint32_t (one unit)*/
    );
}

std::vector<HwmpProtocol::FailedDestination>
IePerr::GetAddressUnitVector() const
{
    return m_addressUnits;
}

void
IePerr::DeleteAddressUnit(Mac48Address address)
{
    for (auto i = m_addressUnits.begin(); i != m_addressUnits.end(); i++)
    {
        if (i->destination == address)
        {
            m_addressUnits.erase(i);
            break;
        }
    }
}

void
IePerr::ResetPerr()
{
    m_addressUnits.clear();
}

bool
operator==(const IePerr& a, const IePerr& b)
{
    if (a.m_addressUnits.size() != b.m_addressUnits.size())
    {
        return false;
    }
    for (unsigned int i = 0; i < a.m_addressUnits.size(); i++)
    {
        if (a.m_addressUnits[i].destination != b.m_addressUnits[i].destination)
        {
            return false;
        }
        if (a.m_addressUnits[i].seqnum != b.m_addressUnits[i].seqnum)
        {
            return false;
        }
    }
    return true;
}

std::ostream&
operator<<(std::ostream& os, const IePerr& a)
{
    a.Print(os);
    return os;
}
} // namespace dot11s
} // namespace ns3
