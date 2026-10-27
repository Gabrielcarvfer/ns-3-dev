/*
 * Copyright (c) 2008,2009 IITP RAS
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Kirill Andreev <andreev@iitp.ru>
 *          Aleksey Kovalenko <kovalenko@iitp.ru>
 */

#include "ie-dot11s-peer-management.h"

#include "ns3/assert.h"
#include "ns3/packet.h"

namespace ns3
{
namespace dot11s
{

namespace
{
/// Mesh Peering Protocol Identifier of the Mesh Peering Management protocol (Table 8-178)
constexpr uint16_t MESH_PEERING_MANAGEMENT_PROTOCOL = 0;
/// Size of the Chosen PMK field (AMPE only)
constexpr uint16_t CHOSEN_PMK_SIZE = 16;
} // namespace

IePeerManagement::IePeerManagement()
    : m_length(4),
      m_subtype(PEER_OPEN),
      m_protocolId(MESH_PEERING_MANAGEMENT_PROTOCOL),
      m_localLinkId(0),
      m_peerLinkId(0),
      m_reasonCode(REASON11S_RESERVED)
{
}

WifiInformationElementId
IePeerManagement::ElementId() const
{
    return IE_MESH_PEERING_MANAGEMENT;
}

void
IePeerManagement::SetPeerOpen(uint16_t localLinkId)
{
    m_length = 4;
    m_subtype = PEER_OPEN;
    m_localLinkId = localLinkId;
}

void
IePeerManagement::SetPeerClose(uint16_t localLinkId, uint16_t peerLinkId, PmpReasonCode reasonCode)
{
    m_length = 8;
    m_subtype = PEER_CLOSE;
    m_localLinkId = localLinkId;
    m_peerLinkId = peerLinkId;
    m_reasonCode = reasonCode;
}

void
IePeerManagement::SetPeerConfirm(uint16_t localLinkId, uint16_t peerLinkId)
{
    m_length = 6;
    m_subtype = PEER_CONFIRM;
    m_localLinkId = localLinkId;
    m_peerLinkId = peerLinkId;
}

PmpReasonCode
IePeerManagement::GetReasonCode() const
{
    return m_reasonCode;
}

uint16_t
IePeerManagement::GetLocalLinkId() const
{
    return m_localLinkId;
}

uint16_t
IePeerManagement::GetPeerLinkId() const
{
    return m_peerLinkId;
}

uint16_t
IePeerManagement::GetInformationFieldSize() const
{
    return m_length;
}

uint8_t
IePeerManagement::GetSubtype() const
{
    return m_subtype;
}

bool
IePeerManagement::SubtypeIsOpen() const
{
    return (m_subtype == PEER_OPEN);
}

bool
IePeerManagement::SubtypeIsClose() const
{
    return (m_subtype == PEER_CLOSE);
}

bool
IePeerManagement::SubtypeIsConfirm() const
{
    return (m_subtype == PEER_CONFIRM);
}

void
IePeerManagement::SetSubtype(Subtype subtype)
{
    m_subtype = subtype;
    if (m_subtype == PEER_CLOSE && m_length == 6)
    {
        // A Close without peer link ID: the field after the local link ID is the reason code
        m_reasonCode = static_cast<PmpReasonCode>(m_peerLinkId);
        m_peerLinkId = 0;
    }
}

void
IePeerManagement::SerializeInformationField(Buffer::Iterator i) const
{
    i.WriteHtolsbU16(m_protocolId);
    i.WriteHtolsbU16(m_localLinkId);
    if (m_length == 6 && m_subtype == PEER_CLOSE)
    {
        i.WriteHtolsbU16(m_reasonCode);
        return;
    }
    if (m_length >= 6)
    {
        i.WriteHtolsbU16(m_peerLinkId);
    }
    if (m_length >= 8)
    {
        i.WriteHtolsbU16(m_reasonCode);
    }
}

uint16_t
IePeerManagement::DeserializeInformationField(Buffer::Iterator start, uint16_t length)
{
    Buffer::Iterator i = start;
    m_length = length;
    m_protocolId = i.ReadLsbtohU16();
    m_localLinkId = i.ReadLsbtohU16();
    m_peerLinkId = 0;
    m_reasonCode = REASON11S_RESERVED;
    // A trailing Chosen PMK (AMPE) is skipped: only open peering is supported
    uint16_t fields = length >= 4 + CHOSEN_PMK_SIZE ? length - CHOSEN_PMK_SIZE : length;
    if (fields >= 6)
    {
        m_peerLinkId = i.ReadLsbtohU16();
    }
    if (fields >= 8)
    {
        m_reasonCode = static_cast<PmpReasonCode>(i.ReadLsbtohU16());
    }
    i.Next(length - i.GetDistanceFrom(start));
    return i.GetDistanceFrom(start);
}

void
IePeerManagement::Print(std::ostream& os) const
{
    os << "PeerMgmt=(Subtype=" << (uint16_t)m_subtype << ", Length=" << (uint16_t)m_length
       << ", ProtocolId=" << m_protocolId << ", LocalLinkId=" << m_localLinkId
       << ", PeerLinkId=" << m_peerLinkId << ", ReasonCode=" << m_reasonCode << ")";
}

bool
operator==(const IePeerManagement& a, const IePeerManagement& b)
{
    // The subtype is not part of the element
    return ((a.m_length == b.m_length) && (a.m_protocolId == b.m_protocolId) &&
            (a.m_localLinkId == b.m_localLinkId) && (a.m_peerLinkId == b.m_peerLinkId) &&
            (a.m_reasonCode == b.m_reasonCode));
}

std::ostream&
operator<<(std::ostream& os, const IePeerManagement& a)
{
    a.Print(os);
    return os;
}
} // namespace dot11s
} // namespace ns3
