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
#include "ns3/log.h"
#include "ns3/packet.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("IePeerManagement");

namespace dot11s
{

IePeerManagement::IePeerManagement()
    : m_length(4),
      m_subtype(PEER_OPEN),
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
IePeerManagement::SerializeInformationField(Buffer::Iterator i) const
{
    // IEEE 802.11-2020 9.4.2.102 Mesh Peering Management element: Mesh Peering
    // Protocol Identifier, Local Link ID, Peer Link ID (Confirm, and Close if known)
    // and Reason Code (Close). The frame type is that of the Self-protected Action
    // frame carrying the element.
    i.WriteHtolsbU16(MESH_PEERING_PROTOCOL);
    i.WriteHtolsbU16(m_localLinkId);
    if (m_subtype == PEER_CONFIRM || (m_subtype == PEER_CLOSE && m_length == 8))
    {
        i.WriteHtolsbU16(m_peerLinkId);
    }
    if (m_subtype == PEER_CLOSE)
    {
        i.WriteHtolsbU16(m_reasonCode);
    }
}

uint16_t
IePeerManagement::DeserializeInformationField(Buffer::Iterator start, uint16_t length)
{
    Buffer::Iterator i = start;
    m_length = length;
    i.ReadLsbtohU16(); // Mesh Peering Protocol Identifier
    m_localLinkId = i.ReadLsbtohU16();
    m_peerLinkId = 0;
    m_reasonCode = REASON11S_RESERVED;
    // The frame type is that of the frame carrying the element (see SetSubtype):
    // 4 octets for Open, 6 for Confirm (or Close without Peer Link ID), 8 for Close
    switch (length)
    {
    case 4:
        m_subtype = PEER_OPEN;
        break;
    case 6:
        m_subtype = PEER_CONFIRM;
        m_peerLinkId = i.ReadLsbtohU16();
        break;
    case 8:
        m_subtype = PEER_CLOSE;
        m_peerLinkId = i.ReadLsbtohU16();
        m_reasonCode = (PmpReasonCode)i.ReadLsbtohU16();
        break;
    default:
        NS_LOG_WARN("Unexpected length of the Mesh Peering Management element: " << length);
    }
    return length;
}

void
IePeerManagement::SetSubtype(Subtype subtype)
{
    if (subtype == PEER_CLOSE && m_length == 6)
    {
        // a Close without Peer Link ID: the last field is the Reason Code
        m_reasonCode = static_cast<PmpReasonCode>(m_peerLinkId);
        m_peerLinkId = 0;
    }
    m_subtype = subtype;
}

void
IePeerManagement::Print(std::ostream& os) const
{
    os << "PeerMgmt=(Subtype=" << (uint16_t)m_subtype << ", Length=" << (uint16_t)m_length
       << ", LocalLinkId=" << m_localLinkId << ", PeerLinkId=" << m_peerLinkId
       << ", ReasonCode=" << m_reasonCode << ")";
}

bool
operator==(const IePeerManagement& a, const IePeerManagement& b)
{
    return ((a.m_length == b.m_length) && (a.m_subtype == b.m_subtype) &&
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
