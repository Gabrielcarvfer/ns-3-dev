/*
 * Copyright (c) 2009 MIRKO BANCHI
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Mirko Banchi <mk.banchi@gmail.com>
 */

#include "ctrl-headers.h"

#include "wifi-tx-vector.h"
#include "wifi-utils.h"

#include "ns3/address-utils.h"
#include "ns3/he-phy.h"

#include <algorithm>

namespace ns3
{

/***********************************
 *       Block ack request
 ***********************************/

NS_OBJECT_ENSURE_REGISTERED(CtrlBAckRequestHeader);

CtrlBAckRequestHeader::CtrlBAckRequestHeader()
    : m_barAckPolicy(false),
      m_barType(BlockAckReqType::BASIC)
{
}

CtrlBAckRequestHeader::~CtrlBAckRequestHeader()
{
}

TypeId
CtrlBAckRequestHeader::GetTypeId()
{
    static TypeId tid = TypeId("ns3::CtrlBAckRequestHeader")
                            .SetParent<Header>()
                            .SetGroupName("Wifi")
                            .AddConstructor<CtrlBAckRequestHeader>();
    return tid;
}

TypeId
CtrlBAckRequestHeader::GetInstanceTypeId() const
{
    return GetTypeId();
}

void
CtrlBAckRequestHeader::Print(std::ostream& os) const
{
    os << "TID_INFO=" << m_tidInfo << ", StartingSeq=" << std::hex << m_startingSeq << std::dec;
}

uint32_t
CtrlBAckRequestHeader::GetSerializedSize() const
{
    uint32_t size = 0;
    size += 2; // Bar control
    switch (m_barType.m_variant)
    {
    case BlockAckReqType::BASIC:
    case BlockAckReqType::COMPRESSED:
    case BlockAckReqType::EXTENDED_COMPRESSED:
        size += 2;
        break;
    case BlockAckReqType::MULTI_TID:
        size += (2 + 2) * (m_tidInfo + 1);
        break;
    case BlockAckReqType::GCR:
        size += (2 + 6); // SSC plus GCR Group Address
        break;
    default:
        NS_FATAL_ERROR("Invalid BA type");
        break;
    }
    return size;
}

void
CtrlBAckRequestHeader::Serialize(Buffer::Iterator start) const
{
    Buffer::Iterator i = start;
    i.WriteHtolsbU16(GetBarControl());
    switch (m_barType.m_variant)
    {
    case BlockAckReqType::BASIC:
    case BlockAckReqType::COMPRESSED:
    case BlockAckReqType::EXTENDED_COMPRESSED:
        i.WriteHtolsbU16(GetStartingSequenceControl());
        break;
    case BlockAckReqType::MULTI_TID:
        NS_FATAL_ERROR("Multi-tid block ack is not supported.");
        break;
    case BlockAckReqType::GCR:
        i.WriteHtolsbU16(GetStartingSequenceControl());
        WriteTo(i, m_gcrAddress);
        break;
    default:
        NS_FATAL_ERROR("Invalid BA type");
        break;
    }
}

uint32_t
CtrlBAckRequestHeader::Deserialize(Buffer::Iterator start)
{
    Buffer::Iterator i = start;
    SetBarControl(i.ReadLsbtohU16());
    switch (m_barType.m_variant)
    {
    case BlockAckReqType::BASIC:
    case BlockAckReqType::COMPRESSED:
    case BlockAckReqType::EXTENDED_COMPRESSED:
        SetStartingSequenceControl(i.ReadLsbtohU16());
        break;
    case BlockAckReqType::MULTI_TID:
        NS_FATAL_ERROR("Multi-tid block ack is not supported.");
        break;
    case BlockAckReqType::GCR:
        SetStartingSequenceControl(i.ReadLsbtohU16());
        ReadFrom(i, m_gcrAddress);
        break;
    default:
        NS_FATAL_ERROR("Invalid BA type");
        break;
    }
    return i.GetDistanceFrom(start);
}

uint16_t
CtrlBAckRequestHeader::GetBarControl() const
{
    uint16_t res = 0;
    switch (m_barType.m_variant)
    {
    case BlockAckReqType::BASIC:
        break;
    case BlockAckReqType::COMPRESSED:
        res |= (0x02 << 1);
        break;
    case BlockAckReqType::EXTENDED_COMPRESSED:
        res |= (0x01 << 1);
        break;
    case BlockAckReqType::MULTI_TID:
        res |= (0x03 << 1);
        break;
    case BlockAckReqType::GCR:
        res |= (0x06 << 1);
        break;
    default:
        NS_FATAL_ERROR("Invalid BA type");
        break;
    }
    res |= (m_tidInfo << 12) & (0xf << 12);
    return res;
}

void
CtrlBAckRequestHeader::SetBarControl(uint16_t bar)
{
    m_barAckPolicy = ((bar & 0x01) == 1);
    if (((bar >> 1) & 0x0f) == 0x06)
    {
        m_barType.m_variant = BlockAckReqType::GCR;
    }
    else if (((bar >> 1) & 0x0f) == 0x03)
    {
        m_barType.m_variant = BlockAckReqType::MULTI_TID;
    }
    else if (((bar >> 1) & 0x0f) == 0x01)
    {
        m_barType.m_variant = BlockAckReqType::EXTENDED_COMPRESSED;
    }
    else if (((bar >> 1) & 0x0f) == 0x02)
    {
        m_barType.m_variant = BlockAckReqType::COMPRESSED;
    }
    else
    {
        m_barType.m_variant = BlockAckReqType::BASIC;
    }
    m_tidInfo = (bar >> 12) & 0x0f;
}

uint16_t
CtrlBAckRequestHeader::GetStartingSequenceControl() const
{
    return (m_startingSeq << 4) & 0xfff0;
}

void
CtrlBAckRequestHeader::SetStartingSequenceControl(uint16_t seqControl)
{
    m_startingSeq = (seqControl >> 4) & 0x0fff;
}

void
CtrlBAckRequestHeader::SetHtImmediateAck(bool immediateAck)
{
    m_barAckPolicy = immediateAck;
}

void
CtrlBAckRequestHeader::SetType(BlockAckReqType type)
{
    m_barType = type;
}

BlockAckReqType
CtrlBAckRequestHeader::GetType() const
{
    return m_barType;
}

void
CtrlBAckRequestHeader::SetTidInfo(uint8_t tid)
{
    m_tidInfo = static_cast<uint16_t>(tid);
}

void
CtrlBAckRequestHeader::SetStartingSequence(uint16_t seq)
{
    m_startingSeq = seq;
}

bool
CtrlBAckRequestHeader::MustSendHtImmediateAck() const
{
    return m_barAckPolicy;
}

uint8_t
CtrlBAckRequestHeader::GetTidInfo() const
{
    auto tid = static_cast<uint8_t>(m_tidInfo);
    return tid;
}

uint16_t
CtrlBAckRequestHeader::GetStartingSequence() const
{
    return m_startingSeq;
}

void
CtrlBAckRequestHeader::SetGcrGroupAddress(const Mac48Address& address)
{
    NS_ASSERT(IsGcr());
    m_gcrAddress = address;
}

Mac48Address
CtrlBAckRequestHeader::GetGcrGroupAddress() const
{
    NS_ASSERT(IsGcr());
    return m_gcrAddress;
}

bool
CtrlBAckRequestHeader::IsBasic() const
{
    return m_barType.m_variant == BlockAckReqType::BASIC;
}

bool
CtrlBAckRequestHeader::IsCompressed() const
{
    return m_barType.m_variant == BlockAckReqType::COMPRESSED;
}

bool
CtrlBAckRequestHeader::IsExtendedCompressed() const
{
    return m_barType.m_variant == BlockAckReqType::EXTENDED_COMPRESSED;
}

bool
CtrlBAckRequestHeader::IsMultiTid() const
{
    return m_barType.m_variant == BlockAckReqType::MULTI_TID;
}

bool
CtrlBAckRequestHeader::IsGcr() const
{
    return m_barType.m_variant == BlockAckReqType::GCR;
}

/***********************************
 *       Block ack response
 ***********************************/

NS_OBJECT_ENSURE_REGISTERED(CtrlBAckResponseHeader);

CtrlBAckResponseHeader::CtrlBAckResponseHeader()
    : m_baAckPolicy(false),
      m_tidInfo(0)
{
    SetType(BlockAckType::BASIC);
}

CtrlBAckResponseHeader::~CtrlBAckResponseHeader()
{
}

TypeId
CtrlBAckResponseHeader::GetTypeId()
{
    static TypeId tid = TypeId("ns3::CtrlBAckResponseHeader")
                            .SetParent<Header>()
                            .SetGroupName("Wifi")
                            .AddConstructor<CtrlBAckResponseHeader>();
    return tid;
}

TypeId
CtrlBAckResponseHeader::GetInstanceTypeId() const
{
    return GetTypeId();
}

void
CtrlBAckResponseHeader::Print(std::ostream& os) const
{
    if (m_baType.m_variant != BlockAckType::MULTI_STA)
    {
        os << "TID_INFO=" << m_tidInfo << ", StartingSeq=0x" << std::hex
           << m_baInfo[0].m_startingSeq << std::dec;
    }
    else
    {
        for (std::size_t i = 0; i < m_baInfo.size(); i++)
        {
            os << "{AID=" << GetAid11(i) << ", TID=" << GetTidInfo(i) << ", StartingSeq=0x"
               << std::hex << m_baInfo[i].m_startingSeq << std::dec << "}";
        }
    }
}

uint32_t
CtrlBAckResponseHeader::GetSerializedSize() const
{
    // This method only makes use of the configured BA type, so that functions like
    // GetBlockAckSize () can easily return the size of a Block Ack of a given type
    uint32_t size = 0;
    size += 2; // BA control
    switch (m_baType.m_variant)
    {
    case BlockAckType::BASIC:
    case BlockAckType::COMPRESSED:
    case BlockAckType::EXTENDED_COMPRESSED:
        size += (2 + m_baType.m_bitmapLen[0]);
        break;
    case BlockAckType::MULTI_TID:
        size += (2 + 2 + 8) * (m_tidInfo + 1); // Multi-TID block ack
        break;
    case BlockAckType::GCR:
        size += (2 + 6 + m_baType.m_bitmapLen[0]);
        break;
    case BlockAckType::MULTI_STA:
        for (auto& bitmapLen : m_baType.m_bitmapLen)
        {
            size += 2 /* AID TID Info */ + (bitmapLen > 0 ? 2 : 0) /* BA SSC */ + bitmapLen;
        }
        break;
    default:
        NS_FATAL_ERROR("Invalid BA type");
        break;
    }
    return size;
}

void
CtrlBAckResponseHeader::Serialize(Buffer::Iterator start) const
{
    Buffer::Iterator i = start;
    i.WriteHtolsbU16(GetBaControl());
    switch (m_baType.m_variant)
    {
    case BlockAckType::BASIC:
    case BlockAckType::COMPRESSED:
    case BlockAckType::EXTENDED_COMPRESSED:
        i.WriteHtolsbU16(GetStartingSequenceControl());
        i = SerializeBitmap(i);
        break;
    case BlockAckType::GCR:
        i.WriteHtolsbU16(GetStartingSequenceControl());
        WriteTo(i, m_baInfo[0].m_address);
        i = SerializeBitmap(i);
        break;
    case BlockAckType::MULTI_STA:
        for (std::size_t index = 0; index < m_baInfo.size(); index++)
        {
            i.WriteHtolsbU16(m_baInfo[index].m_aidTidInfo);
            if (GetAid11(index) != 2045)
            {
                if (!m_baInfo[index].m_bitmap.empty())
                {
                    i.WriteHtolsbU16(GetStartingSequenceControl(index));
                    i = SerializeBitmap(i, index);
                }
            }
            else
            {
                uint32_t reserved = 0;
                i.WriteHtolsbU32(reserved);
                WriteTo(i, m_baInfo[index].m_address);
            }
        }
        break;
    case BlockAckType::MULTI_TID:
        NS_FATAL_ERROR("Multi-tid block ack is not supported.");
        break;
    default:
        NS_FATAL_ERROR("Invalid BA type");
        break;
    }
}

uint32_t
CtrlBAckResponseHeader::Deserialize(Buffer::Iterator start)
{
    Buffer::Iterator i = start;
    SetBaControl(i.ReadLsbtohU16());
    switch (m_baType.m_variant)
    {
    case BlockAckType::BASIC:
    case BlockAckType::COMPRESSED:
    case BlockAckType::EXTENDED_COMPRESSED:
        SetStartingSequenceControl(i.ReadLsbtohU16());
        i = DeserializeBitmap(i);
        break;
    case BlockAckType::GCR:
        SetStartingSequenceControl(i.ReadLsbtohU16());
        ReadFrom(i, m_baInfo[0].m_address);
        i = DeserializeBitmap(i);
        break;
    case BlockAckType::MULTI_STA: {
        std::size_t index = 0;
        while (i.GetRemainingSize() > 0)
        {
            m_baInfo.emplace_back();
            m_baType.m_bitmapLen.push_back(0); // updated by next call to SetStartingSequenceControl

            m_baInfo.back().m_aidTidInfo = i.ReadLsbtohU16();

            if (GetAid11(index) != 2045)
            {
                // the Block Ack Starting Sequence Control and Block Ack Bitmap subfields
                // are only present in Block acknowledgement context, i.e., if the Ack Type
                // subfield is set to 0 and the TID subfield is set to a value from 0 to 7.
                if (!GetAckType(index) && GetTidInfo(index) < 8)
                {
                    SetStartingSequenceControl(i.ReadLsbtohU16(), index);
                    i = DeserializeBitmap(i, index);
                }
            }
            else
            {
                i.ReadLsbtohU32(); // next 4 bytes are reserved
                ReadFrom(i, m_baInfo.back().m_address);
                // the length of this Per AID TID Info subfield is 12, so set
                // the bitmap length to 8 to simulate the correct size
                m_baType.m_bitmapLen.back() = 8;
            }
            index++;
        }
    }
    break;
    case BlockAckType::MULTI_TID:
        NS_FATAL_ERROR("Multi-tid block ack is not supported.");
        break;
    default:
        NS_FATAL_ERROR("Invalid BA type");
        break;
    }
    return i.GetDistanceFrom(start);
}

void
CtrlBAckResponseHeader::SetHtImmediateAck(bool immediateAck)
{
    m_baAckPolicy = immediateAck;
}

void
CtrlBAckResponseHeader::SetType(BlockAckType type)
{
    m_baType = type;
    m_baInfo.clear();

    for (auto& bitmapLen : m_baType.m_bitmapLen)
    {
        BaInfoInstance baInfoInstance{.m_aidTidInfo = 0,
                                      .m_startingSeq = 0,
                                      .m_bitmap = std::vector<uint8_t>(bitmapLen, 0),
                                      .m_address = Mac48Address()};

        m_baInfo.emplace_back(baInfoInstance);
    }
}

BlockAckType
CtrlBAckResponseHeader::GetType() const
{
    return m_baType;
}

void
CtrlBAckResponseHeader::SetTidInfo(uint8_t tid, std::size_t index)
{
    NS_ASSERT_MSG(m_baType.m_variant == BlockAckType::MULTI_STA || index == 0,
                  "index can only be non null for Multi-STA Block Ack");
    NS_ASSERT(index < m_baInfo.size());

    if (m_baType.m_variant != BlockAckType::MULTI_STA)
    {
        m_tidInfo = static_cast<uint16_t>(tid);
    }
    else
    {
        m_baInfo[index].m_aidTidInfo |= ((static_cast<uint16_t>(tid) & 0x000f) << 12);
    }
}

void
CtrlBAckResponseHeader::SetStartingSequence(uint16_t seq, std::size_t index)
{
    NS_ASSERT_MSG(m_baType.m_variant == BlockAckType::MULTI_STA || index == 0,
                  "index can only be non null for Multi-STA Block Ack");
    NS_ASSERT(index < m_baInfo.size());

    m_baInfo[index].m_startingSeq = seq;
}

bool
CtrlBAckResponseHeader::MustSendHtImmediateAck() const
{
    return m_baAckPolicy;
}

uint8_t
CtrlBAckResponseHeader::GetTidInfo(std::size_t index) const
{
    NS_ASSERT_MSG(m_baType.m_variant == BlockAckType::MULTI_STA || index == 0,
                  "index can only be non null for Multi-STA Block Ack");
    NS_ASSERT(index < m_baInfo.size());

    uint8_t tid = 0;

    if (m_baType.m_variant != BlockAckType::MULTI_STA)
    {
        tid = static_cast<uint8_t>(m_tidInfo);
    }
    else
    {
        tid = static_cast<uint8_t>((m_baInfo[index].m_aidTidInfo >> 12) & 0x000f);
    }
    return tid;
}

uint16_t
CtrlBAckResponseHeader::GetStartingSequence(std::size_t index) const
{
    NS_ASSERT_MSG(m_baType.m_variant == BlockAckType::MULTI_STA || index == 0,
                  "index can only be non null for Multi-STA Block Ack");
    NS_ASSERT(index < m_baInfo.size());

    return m_baInfo[index].m_startingSeq;
}

bool
CtrlBAckResponseHeader::IsBasic() const
{
    return m_baType.m_variant == BlockAckType::BASIC;
}

bool
CtrlBAckResponseHeader::IsCompressed() const
{
    return m_baType.m_variant == BlockAckType::COMPRESSED;
}

bool
CtrlBAckResponseHeader::IsExtendedCompressed() const
{
    return m_baType.m_variant == BlockAckType::EXTENDED_COMPRESSED;
}

bool
CtrlBAckResponseHeader::IsMultiTid() const
{
    return m_baType.m_variant == BlockAckType::MULTI_TID;
}

bool
CtrlBAckResponseHeader::IsMultiSta() const
{
    return m_baType.m_variant == BlockAckType::MULTI_STA;
}

bool
CtrlBAckResponseHeader::IsGcr() const
{
    return m_baType.m_variant == BlockAckType::GCR;
}

void
CtrlBAckResponseHeader::SetAid11(uint16_t aid, std::size_t index)
{
    NS_ASSERT(m_baType.m_variant == BlockAckType::MULTI_STA && index < m_baInfo.size());

    m_baInfo[index].m_aidTidInfo |= (aid & 0x07ff);
}

uint16_t
CtrlBAckResponseHeader::GetAid11(std::size_t index) const
{
    NS_ASSERT(m_baType.m_variant == BlockAckType::MULTI_STA && index < m_baInfo.size());

    return m_baInfo[index].m_aidTidInfo & 0x07ff;
}

void
CtrlBAckResponseHeader::SetAckType(bool type, std::size_t index)
{
    NS_ASSERT(m_baType.m_variant == BlockAckType::MULTI_STA && index < m_baInfo.size());

    if (type)
    {
        m_baInfo[index].m_aidTidInfo |= (1 << 11);
    }
}

bool
CtrlBAckResponseHeader::GetAckType(std::size_t index) const
{
    NS_ASSERT(m_baType.m_variant == BlockAckType::MULTI_STA && index < m_baInfo.size());

    return ((m_baInfo[index].m_aidTidInfo >> 11) & 0x0001) != 0;
}

void
CtrlBAckResponseHeader::SetUnassociatedStaAddress(const Mac48Address& ra, std::size_t index)
{
    NS_ASSERT(GetAid11(index) == 2045);

    m_baInfo[index].m_address = ra;
}

Mac48Address
CtrlBAckResponseHeader::GetUnassociatedStaAddress(std::size_t index) const
{
    NS_ASSERT(GetAid11(index) == 2045);

    return m_baInfo[index].m_address;
}

std::size_t
CtrlBAckResponseHeader::GetNPerAidTidInfoSubfields() const
{
    NS_ASSERT(m_baType.m_variant == BlockAckType::MULTI_STA);
    return m_baInfo.size();
}

std::vector<uint32_t>
CtrlBAckResponseHeader::FindPerAidTidInfoWithAid(uint16_t aid) const
{
    NS_ASSERT(m_baType.m_variant == BlockAckType::MULTI_STA);

    std::vector<uint32_t> ret;
    ret.reserve(m_baInfo.size());
    for (uint32_t i = 0; i < m_baInfo.size(); i++)
    {
        if (GetAid11(i) == aid)
        {
            ret.push_back(i);
        }
    }
    return ret;
}

void
CtrlBAckResponseHeader::SetGcrGroupAddress(const Mac48Address& address)
{
    NS_ASSERT(IsGcr());
    m_baInfo[0].m_address = address;
}

Mac48Address
CtrlBAckResponseHeader::GetGcrGroupAddress() const
{
    NS_ASSERT(IsGcr());
    return m_baInfo[0].m_address;
}

uint16_t
CtrlBAckResponseHeader::GetBaControl() const
{
    uint16_t res = 0;
    if (m_baAckPolicy)
    {
        res |= 0x1;
    }
    switch (m_baType.m_variant)
    {
    case BlockAckType::BASIC:
        break;
    case BlockAckType::COMPRESSED:
        res |= (0x02 << 1);
        break;
    case BlockAckType::EXTENDED_COMPRESSED:
        res |= (0x01 << 1);
        break;
    case BlockAckType::MULTI_TID:
        res |= (0x03 << 1);
        break;
    case BlockAckType::GCR:
        res |= (0x06 << 1);
        break;
    case BlockAckType::MULTI_STA:
        res |= (0x0b << 1);
        break;
    default:
        NS_FATAL_ERROR("Invalid BA type");
        break;
    }
    if (m_baType.m_variant != BlockAckType::MULTI_STA)
    {
        res |= (m_tidInfo << 12) & (0xf << 12);
    }
    return res;
}

void
CtrlBAckResponseHeader::SetBaControl(uint16_t ba)
{
    m_baAckPolicy = ((ba & 0x01) == 1);
    if (((ba >> 1) & 0x0f) == 0x06)
    {
        SetType(BlockAckType::GCR);
    }
    else if (((ba >> 1) & 0x0f) == 0x03)
    {
        SetType(BlockAckType::MULTI_TID);
    }
    else if (((ba >> 1) & 0x0f) == 0x01)
    {
        SetType(BlockAckType::EXTENDED_COMPRESSED);
    }
    else if (((ba >> 1) & 0x0f) == 0x02)
    {
        SetType(BlockAckType::COMPRESSED);
    }
    else if (((ba >> 1) & 0x0f) == 0)
    {
        SetType(BlockAckType::BASIC);
    }
    else if (((ba >> 1) & 0x0f) == 0x0b)
    {
        SetType(BlockAckType::MULTI_STA);
    }
    else
    {
        NS_FATAL_ERROR("Invalid BA type");
    }
    if (m_baType.m_variant != BlockAckType::MULTI_STA)
    {
        m_tidInfo = (ba >> 12) & 0x0f;
    }
}

uint16_t
CtrlBAckResponseHeader::GetStartingSequenceControl(std::size_t index) const
{
    NS_ASSERT_MSG(m_baType.m_variant == BlockAckType::MULTI_STA || index == 0,
                  "index can only be non null for Multi-STA Block Ack");
    NS_ASSERT(index < m_baInfo.size());

    uint16_t ret = (m_baInfo[index].m_startingSeq << 4) & 0xfff0;

    // The Fragment Number subfield encodes the length of the bitmap for Compressed and Multi-STA
    // variants (see sections 9.3.1.8.2 and 9.3.1.8.7 of 802.11ax-2021 and 802.11be Draft 4.0).
    // Note that Fragmentation Level 3 is not supported.
    if (m_baType.m_variant == BlockAckType::COMPRESSED || m_baType.m_variant == BlockAckType::GCR)
    {
        switch (m_baType.m_bitmapLen[0])
        {
        case 8:
            // do nothing
            break;
        case 32:
            ret |= 0x0004;
            break;
        case 64:
            ret |= 0x0008;
            break;
        case 128:
            ret |= 0x000a;
            break;
        default:
            NS_ABORT_MSG("Unsupported bitmap length: " << +m_baType.m_bitmapLen[0] << " bytes");
        }
    }
    else if (m_baType.m_variant == BlockAckType::MULTI_STA)
    {
        NS_ASSERT(m_baInfo.size() == m_baType.m_bitmapLen.size());
        NS_ASSERT_MSG(!m_baInfo[index].m_bitmap.empty(),
                      "This Per AID TID Info subfield has no Starting Sequence Control subfield");

        switch (m_baType.m_bitmapLen[index])
        {
        case 8:
            // do nothing
            break;
        case 16:
            ret |= 0x0002;
            break;
        case 32:
            ret |= 0x0004;
            break;
        case 4:
            ret |= 0x0006;
            break;
        case 64:
            ret |= 0x0008;
            break;
        case 128:
            ret |= 0x000a;
            break;
        default:
            NS_ABORT_MSG("Unsupported bitmap length: " << +m_baType.m_bitmapLen[index] << " bytes");
        }
    }
    return ret;
}

void
CtrlBAckResponseHeader::SetStartingSequenceControl(uint16_t seqControl, std::size_t index)
{
    NS_ASSERT_MSG(m_baType.m_variant == BlockAckType::MULTI_STA || index == 0,
                  "index can only be non null for Multi-STA Block Ack");
    NS_ASSERT(index < m_baInfo.size());

    // The Fragment Number subfield encodes the length of the bitmap for Compressed and Multi-STA
    // variants (see sections 9.3.1.8.2 and 9.3.1.8.7 of 802.11ax-2021 and 802.11be Draft 4.0).
    // Note that Fragmentation Level 3 is not supported.
    if (m_baType.m_variant == BlockAckType::COMPRESSED || m_baType.m_variant == BlockAckType::GCR)
    {
        uint16_t fragNumber = seqControl & 0x000f;

        if ((fragNumber & 0x0001) == 1)
        {
            NS_FATAL_ERROR("Fragmentation Level 3 unsupported");
        }
        switch (fragNumber)
        {
        case 0:
            SetType({m_baType.m_variant, {8}});
            break;
        case 4:
            SetType({m_baType.m_variant, {32}});
            break;
        case 8:
            SetType({m_baType.m_variant, {64}});
            break;
        case 10:
            SetType({m_baType.m_variant, {128}});
            break;
        default:
            NS_ABORT_MSG("Unsupported fragment number: " << fragNumber);
        }
    }
    else if (m_baType.m_variant == BlockAckType::MULTI_STA)
    {
        uint16_t fragNumber = seqControl & 0x000f;

        if ((fragNumber & 0x0001) == 1)
        {
            NS_FATAL_ERROR("Fragmentation Level 3 unsupported");
        }
        uint8_t bitmapLen = 0;
        switch (fragNumber)
        {
        case 0:
            bitmapLen = 8;
            break;
        case 2:
            bitmapLen = 16;
            break;
        case 4:
            bitmapLen = 32;
            break;
        case 6:
            bitmapLen = 4;
            break;
        case 8:
            bitmapLen = 64;
            break;
        case 10:
            bitmapLen = 128;
            break;
        default:
            NS_ABORT_MSG("Unsupported fragment number: " << fragNumber);
        }
        m_baType.m_bitmapLen[index] = bitmapLen;
        m_baInfo[index].m_bitmap.assign(bitmapLen, 0);
    }

    m_baInfo[index].m_startingSeq = (seqControl >> 4) & 0x0fff;
}

Buffer::Iterator
CtrlBAckResponseHeader::SerializeBitmap(Buffer::Iterator start, std::size_t index) const
{
    NS_ASSERT_MSG(m_baType.m_variant == BlockAckType::MULTI_STA || index == 0,
                  "index can only be non null for Multi-STA Block Ack");
    NS_ASSERT(index < m_baInfo.size());

    Buffer::Iterator i = start;
    switch (m_baType.m_variant)
    {
    case BlockAckType::BASIC:
    case BlockAckType::COMPRESSED:
    case BlockAckType::EXTENDED_COMPRESSED:
    case BlockAckType::GCR:
    case BlockAckType::MULTI_STA:
        for (const auto& byte : m_baInfo[index].m_bitmap)
        {
            i.WriteU8(byte);
        }
        break;
    case BlockAckType::MULTI_TID:
        NS_FATAL_ERROR("Multi-tid block ack is not supported.");
        break;
    default:
        NS_FATAL_ERROR("Invalid BA type");
        break;
    }
    return i;
}

Buffer::Iterator
CtrlBAckResponseHeader::DeserializeBitmap(Buffer::Iterator start, std::size_t index)
{
    NS_ASSERT_MSG(m_baType.m_variant == BlockAckType::MULTI_STA || index == 0,
                  "index can only be non null for Multi-STA Block Ack");
    NS_ASSERT(index < m_baInfo.size());

    Buffer::Iterator i = start;
    switch (m_baType.m_variant)
    {
    case BlockAckType::BASIC:
    case BlockAckType::COMPRESSED:
    case BlockAckType::EXTENDED_COMPRESSED:
    case BlockAckType::GCR:
    case BlockAckType::MULTI_STA:
        for (uint8_t j = 0; j < m_baType.m_bitmapLen[index]; j++)
        {
            m_baInfo[index].m_bitmap[j] = i.ReadU8();
        }
        break;
    case BlockAckType::MULTI_TID:
        NS_FATAL_ERROR("Multi-tid block ack is not supported.");
        break;
    default:
        NS_FATAL_ERROR("Invalid BA type");
        break;
    }
    return i;
}

void
CtrlBAckResponseHeader::SetReceivedPacket(uint16_t seq, std::size_t index)
{
    NS_ASSERT_MSG(m_baType.m_variant == BlockAckType::MULTI_STA || index == 0,
                  "index can only be non null for Multi-STA Block Ack");
    NS_ASSERT(index < m_baInfo.size());

    if (!IsInBitmap(seq, index))
    {
        return;
    }
    switch (m_baType.m_variant)
    {
    case BlockAckType::BASIC:
        /* To set correctly basic block ack bitmap we need fragment number too.
            So if it's not specified, we consider packet not fragmented. */
        m_baInfo[index].m_bitmap[IndexInBitmap(seq) * 2] |= 0x01;
        break;
    case BlockAckType::COMPRESSED:
    case BlockAckType::EXTENDED_COMPRESSED:
    case BlockAckType::GCR:
    case BlockAckType::MULTI_STA: {
        uint16_t i = IndexInBitmap(seq, index);
        m_baInfo[index].m_bitmap[i / 8] |= (uint8_t(0x01) << (i % 8));
        break;
    }
    case BlockAckType::MULTI_TID:
        NS_FATAL_ERROR("Multi-tid block ack is not supported.");
        break;
    default:
        NS_FATAL_ERROR("Invalid BA type");
        break;
    }
}

void
CtrlBAckResponseHeader::SetReceivedFragment(uint16_t seq, uint8_t frag)
{
    NS_ASSERT(frag < 16);
    if (!IsInBitmap(seq))
    {
        return;
    }
    switch (m_baType.m_variant)
    {
    case BlockAckType::BASIC:
        m_baInfo[0].m_bitmap[IndexInBitmap(seq) * 2 + frag / 8] |= (0x01 << (frag % 8));
        break;
    case BlockAckType::COMPRESSED:
    case BlockAckType::EXTENDED_COMPRESSED:
    case BlockAckType::GCR:
    case BlockAckType::MULTI_STA:
        /* We can ignore this...compressed block ack doesn't support
           acknowledgment of single fragments */
        break;
    case BlockAckType::MULTI_TID:
        NS_FATAL_ERROR("Multi-tid block ack is not supported.");
        break;
    default:
        NS_FATAL_ERROR("Invalid BA type");
        break;
    }
}

bool
CtrlBAckResponseHeader::IsPacketReceived(uint16_t seq, std::size_t index) const
{
    NS_ASSERT_MSG(m_baType.m_variant == BlockAckType::MULTI_STA || index == 0,
                  "index can only be non null for Multi-STA Block Ack");
    NS_ASSERT(index < m_baInfo.size());

    if (m_baType.m_variant == BlockAckType::MULTI_STA && GetAckType(index) &&
        GetTidInfo(index) == 14)
    {
        // All-ack context
        return true;
    }
    if (!IsInBitmap(seq, index))
    {
        return false;
    }
    switch (m_baType.m_variant)
    {
    case BlockAckType::BASIC:
        /*It's impossible to say if an entire packet was correctly received. */
        return false;
    case BlockAckType::COMPRESSED:
    case BlockAckType::EXTENDED_COMPRESSED:
    case BlockAckType::GCR:
    case BlockAckType::MULTI_STA: {
        uint16_t i = IndexInBitmap(seq, index);
        uint8_t mask = uint8_t(0x01) << (i % 8);
        return (m_baInfo[index].m_bitmap[i / 8] & mask) != 0;
    }
    case BlockAckType::MULTI_TID:
        NS_FATAL_ERROR("Multi-tid block ack is not supported.");
        break;
    default:
        NS_FATAL_ERROR("Invalid BA type");
        break;
    }
    return false;
}

bool
CtrlBAckResponseHeader::IsFragmentReceived(uint16_t seq, uint8_t frag) const
{
    NS_ASSERT(frag < 16);
    if (!IsInBitmap(seq))
    {
        return false;
    }
    switch (m_baType.m_variant)
    {
    case BlockAckType::BASIC:
        return (m_baInfo[0].m_bitmap[IndexInBitmap(seq) * 2 + frag / 8] & (0x01 << (frag % 8))) !=
               0;
    case BlockAckType::COMPRESSED:
    case BlockAckType::EXTENDED_COMPRESSED:
    case BlockAckType::GCR:
    case BlockAckType::MULTI_STA:
        /* We can ignore this...compressed block ack doesn't support
           acknowledgement of single fragments */
        return false;
    case BlockAckType::MULTI_TID: {
        NS_FATAL_ERROR("Multi-tid block ack is not supported.");
        break;
    }
    default: {
        NS_FATAL_ERROR("Invalid BA type");
        break;
    }
    }
    return false;
}

uint16_t
CtrlBAckResponseHeader::IndexInBitmap(uint16_t seq, std::size_t index) const
{
    uint16_t i;
    if (seq >= m_baInfo[index].m_startingSeq)
    {
        i = seq - m_baInfo[index].m_startingSeq;
    }
    else
    {
        i = SEQNO_SPACE_SIZE - m_baInfo[index].m_startingSeq + seq;
    }

    uint16_t nAckedMpdus = m_baType.m_bitmapLen[index] * 8;

    if (m_baType.m_variant == BlockAckType::BASIC)
    {
        nAckedMpdus = nAckedMpdus / 16;
    }

    NS_ASSERT(i < nAckedMpdus);
    return i;
}

bool
CtrlBAckResponseHeader::IsInBitmap(uint16_t seq, std::size_t index) const
{
    NS_ASSERT_MSG(m_baType.m_variant == BlockAckType::MULTI_STA || index == 0,
                  "index can only be non null for Multi-STA Block Ack");
    NS_ASSERT(index < m_baType.m_bitmapLen.size());

    uint16_t nAckedMpdus = m_baType.m_bitmapLen[index] * 8;

    if (m_baType.m_variant == BlockAckType::BASIC)
    {
        nAckedMpdus = nAckedMpdus / 16;
    }

    return (seq - m_baInfo[index].m_startingSeq + SEQNO_SPACE_SIZE) % SEQNO_SPACE_SIZE <
           nAckedMpdus;
}

const std::vector<uint8_t>&
CtrlBAckResponseHeader::GetBitmap(std::size_t index) const
{
    NS_ASSERT_MSG(m_baType.m_variant == BlockAckType::MULTI_STA || index == 0,
                  "index can only be non null for Multi-STA Block Ack");
    NS_ASSERT(index < m_baInfo.size());

    return m_baInfo[index].m_bitmap;
}

void
CtrlBAckResponseHeader::ResetBitmap(std::size_t index)
{
    NS_ASSERT_MSG(m_baType.m_variant == BlockAckType::MULTI_STA || index == 0,
                  "index can only be non null for Multi-STA Block Ack");
    NS_ASSERT(index < m_baInfo.size());

    m_baInfo[index].m_bitmap.assign(m_baType.m_bitmapLen[index], 0);
}

/***********************************
 * Trigger frame - User Info field
 ***********************************/

CtrlTriggerUserInfoField::CtrlTriggerUserInfoField(TriggerFrameType triggerType,
                                                   TriggerFrameVariant variant)
    : m_variant(variant),
      m_aid12(0),
      m_ruAllocation(0),
      m_ulFecCodingType(false),
      m_ulMcs(0),
      m_ulDcm(false),
      m_ps160(false),
      m_ulTargetRssi(0),
      m_triggerType(triggerType),
      m_basicTriggerDependentUserInfo(0)
{
    memset(&m_bits26To31, 0, sizeof(m_bits26To31));
}

CtrlTriggerUserInfoField::~CtrlTriggerUserInfoField()
{
}

CtrlTriggerUserInfoField&
CtrlTriggerUserInfoField::operator=(const CtrlTriggerUserInfoField& userInfo)
{
    NS_ABORT_MSG_IF(m_triggerType != userInfo.m_triggerType, "Trigger Frame type mismatch");

    // check for self-assignment
    if (&userInfo == this)
    {
        return *this;
    }

    m_variant = userInfo.m_variant;
    m_aid12 = userInfo.m_aid12;
    m_ruAllocation = userInfo.m_ruAllocation;
    m_ulFecCodingType = userInfo.m_ulFecCodingType;
    m_ulMcs = userInfo.m_ulMcs;
    m_ulDcm = userInfo.m_ulDcm;
    m_ps160 = userInfo.m_ps160;
    m_bits26To31 = userInfo.m_bits26To31;
    m_ulTargetRssi = userInfo.m_ulTargetRssi;
    m_basicTriggerDependentUserInfo = userInfo.m_basicTriggerDependentUserInfo;
    m_muBarTriggerDependentUserInfo = userInfo.m_muBarTriggerDependentUserInfo;
    return *this;
}

void
CtrlTriggerUserInfoField::Print(std::ostream& os) const
{
    os << ", USER_INFO " << (m_variant == TriggerFrameVariant::HE ? "HE" : "EHT")
       << " variant AID=" << m_aid12 << ", RU_Allocation=" << +m_ruAllocation
       << ", MCS=" << +m_ulMcs;
}

uint32_t
CtrlTriggerUserInfoField::GetSerializedSize() const
{
    uint32_t size = 0;
    size += 5; // User Info (excluding Trigger Dependent User Info)

    switch (m_triggerType)
    {
    case TriggerFrameType::BASIC_TRIGGER:
    case TriggerFrameType::BFRP_TRIGGER:
        size += 1;
        break;
    case TriggerFrameType::MU_BAR_TRIGGER:
        size +=
            m_muBarTriggerDependentUserInfo.GetSerializedSize(); // BAR Control and BAR Information
        break;
    default:;
        // The Trigger Dependent User Info subfield is not present in the other variants
    }

    return size;
}

Buffer::Iterator
CtrlTriggerUserInfoField::Serialize(Buffer::Iterator start) const
{
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::BFRP_TRIGGER,
                    "BFRP Trigger frame is not supported");
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::GCR_MU_BAR_TRIGGER,
                    "GCR-MU-BAR Trigger frame is not supported");
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::NFRP_TRIGGER,
                    "NFRP Trigger frame is not supported");

    Buffer::Iterator i = start;

    uint32_t userInfo = 0; // User Info except the MSB
    userInfo |= (m_aid12 & 0x0fff);
    userInfo |= (m_ruAllocation << 12);
    userInfo |= (m_ulFecCodingType ? 1 << 20 : 0);
    userInfo |= (m_ulMcs & 0x0f) << 21;
    if (m_variant == TriggerFrameVariant::HE)
    {
        userInfo |= (m_ulDcm ? 1 << 25 : 0);
    }

    if (m_aid12 != 0 && m_aid12 != 2045)
    {
        userInfo |= (m_bits26To31.ssAllocation.startingSs & 0x07) << 26;
        userInfo |= (m_bits26To31.ssAllocation.nSs & 0x07) << 29;
    }
    else
    {
        userInfo |= (m_bits26To31.raRuInformation.nRaRu & 0x1f) << 26;
        userInfo |= (m_bits26To31.raRuInformation.moreRaRu ? 1 << 31 : 0);
    }

    i.WriteHtolsbU32(userInfo);
    // Here we need to write 8 bits covering the UL Target RSSI (7 bits) and B39, which is
    // reserved in the HE variant and the PS160 subfield in the EHT variant.
    uint8_t bit32To39 = m_ulTargetRssi;
    if (m_variant == TriggerFrameVariant::EHT)
    {
        bit32To39 |= (m_ps160 ? 1 << 7 : 0);
    }

    i.WriteU8(bit32To39);

    if (m_triggerType == TriggerFrameType::BASIC_TRIGGER)
    {
        i.WriteU8(m_basicTriggerDependentUserInfo);
    }
    else if (m_triggerType == TriggerFrameType::MU_BAR_TRIGGER)
    {
        m_muBarTriggerDependentUserInfo.Serialize(i);
        i.Next(m_muBarTriggerDependentUserInfo.GetSerializedSize());
    }

    return i;
}

Buffer::Iterator
CtrlTriggerUserInfoField::Deserialize(Buffer::Iterator start)
{
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::BFRP_TRIGGER,
                    "BFRP Trigger frame is not supported");
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::GCR_MU_BAR_TRIGGER,
                    "GCR-MU-BAR Trigger frame is not supported");
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::NFRP_TRIGGER,
                    "NFRP Trigger frame is not supported");

    Buffer::Iterator i = start;

    uint32_t userInfo = i.ReadLsbtohU32();

    m_aid12 = userInfo & 0x0fff;
    NS_ABORT_MSG_IF(m_aid12 == 4095, "Cannot deserialize a Padding field");
    m_ruAllocation = (userInfo >> 12) & 0xff;
    m_ulFecCodingType = (userInfo >> 20) & 0x01;
    m_ulMcs = (userInfo >> 21) & 0x0f;
    if (m_variant == TriggerFrameVariant::HE)
    {
        m_ulDcm = (userInfo >> 25) & 0x01;
    }

    if (m_aid12 != 0 && m_aid12 != 2045)
    {
        m_bits26To31.ssAllocation.startingSs = (userInfo >> 26) & 0x07;
        m_bits26To31.ssAllocation.nSs = (userInfo >> 29) & 0x07;
    }
    else
    {
        m_bits26To31.raRuInformation.nRaRu = (userInfo >> 26) & 0x1f;
        m_bits26To31.raRuInformation.moreRaRu = (userInfo >> 31) & 0x01;
    }

    uint8_t bit32To39 = i.ReadU8();
    m_ulTargetRssi = bit32To39 & 0x7f; // B39 is reserved in HE variant
    if (m_variant == TriggerFrameVariant::EHT)
    {
        m_ps160 = (bit32To39 >> 7) == 1;
    }

    if (m_triggerType == TriggerFrameType::BASIC_TRIGGER)
    {
        m_basicTriggerDependentUserInfo = i.ReadU8();
    }
    else if (m_triggerType == TriggerFrameType::MU_BAR_TRIGGER)
    {
        uint32_t len = m_muBarTriggerDependentUserInfo.Deserialize(i);
        i.Next(len);
    }

    return i;
}

TriggerFrameType
CtrlTriggerUserInfoField::GetType() const
{
    return m_triggerType;
}

WifiPreamble
CtrlTriggerUserInfoField::GetPreambleType() const
{
    switch (m_variant)
    {
    case TriggerFrameVariant::HE:
        return WIFI_PREAMBLE_HE_TB;
    case TriggerFrameVariant::EHT:
        return WIFI_PREAMBLE_EHT_TB;
    default:
        NS_ABORT_MSG("Unexpected variant: " << +static_cast<uint8_t>(m_variant));
    }
    return WIFI_PREAMBLE_LONG; // to silence warning
}

void
CtrlTriggerUserInfoField::SetAid12(uint16_t aid)
{
    NS_ASSERT_MSG((m_variant == TriggerFrameVariant::HE) || (aid != AID_SPECIAL_USER),
                  std::to_string(AID_SPECIAL_USER)
                      << " is reserved for Special User Info Field in EHT variant");
    m_aid12 = aid & 0x0fff;
}

uint16_t
CtrlTriggerUserInfoField::GetAid12() const
{
    return m_aid12;
}

bool
CtrlTriggerUserInfoField::HasRaRuForAssociatedSta() const
{
    return (m_aid12 == 0);
}

bool
CtrlTriggerUserInfoField::HasRaRuForUnassociatedSta() const
{
    return (m_aid12 == 2045);
}

void
CtrlTriggerUserInfoField::SetRuAllocation(WifiRu::RuSpec ru)
{
    const auto ruIndex = WifiRu::GetIndex(ru);
    const auto ruType = WifiRu::GetRuType(ru);
    NS_ABORT_MSG_IF(ruIndex == 0, "Valid indices start at 1");
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::MU_RTS_TRIGGER,
                    "SetMuRtsRuAllocation() must be used for MU-RTS");

    switch (ruType)
    {
    case RuType::RU_26_TONE:
        m_ruAllocation = ruIndex - 1;
        NS_ABORT_MSG_IF(!WifiRu::IsHe(ru) && (m_ruAllocation == 18), "Reserved value.");
        NS_ASSERT(m_ruAllocation <= 36);
        break;
    case RuType::RU_52_TONE:
        m_ruAllocation = ruIndex + 36;
        NS_ASSERT(m_ruAllocation <= 52);
        break;
    case RuType::RU_106_TONE:
        m_ruAllocation = ruIndex + 52;
        NS_ASSERT(m_ruAllocation <= 60);
        break;
    case RuType::RU_242_TONE:
        m_ruAllocation = ruIndex + 60;
        NS_ASSERT(m_ruAllocation <= 64);
        break;
    case RuType::RU_484_TONE:
        m_ruAllocation = ruIndex + 64;
        NS_ASSERT(m_ruAllocation <= 67);
        break;
    case RuType::RU_996_TONE:
        m_ruAllocation = 67;
        break;
    case RuType::RU_2x996_TONE:
        m_ruAllocation = 68;
        break;
    case RuType::RU_4x996_TONE:
        m_ruAllocation = 69;
        break;
    default:
        NS_FATAL_ERROR("RU type unknown.");
        break;
    }

    NS_ABORT_MSG_IF(m_ruAllocation > 69, "Reserved value.");

    auto b0 = (WifiRu::IsHe(ru) && !std::get<HeRu::RuSpec>(ru).GetPrimary80MHz()) ||
              (WifiRu::IsEht(ru) && !std::get<EhtRu::RuSpec>(ru).GetPrimary80MHzOrLower80MHz());
    m_ps160 = (WifiRu::IsEht(ru) && !std::get<EhtRu::RuSpec>(ru).GetPrimary160MHz());

    m_ruAllocation <<= 1;
    if (b0)
    {
        m_ruAllocation++;
    }
}

WifiRu::RuSpec
CtrlTriggerUserInfoField::GetRuAllocation() const
{
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::MU_RTS_TRIGGER,
                    "GetMuRtsRuAllocation() must be used for MU-RTS");

    RuType ruType;
    std::size_t index;

    const auto primaryOrLower80MHz = ((m_ruAllocation & 0x01) == 0);

    uint8_t val = m_ruAllocation >> 1;

    if (val < 37)
    {
        ruType = RuType::RU_26_TONE;
        index = val + 1;
    }
    else if (val < 53)
    {
        ruType = RuType::RU_52_TONE;
        index = val - 36;
    }
    else if (val < 61)
    {
        ruType = RuType::RU_106_TONE;
        index = val - 52;
    }
    else if (val < 65)
    {
        ruType = RuType::RU_242_TONE;
        index = val - 60;
    }
    else if (val < 67)
    {
        ruType = RuType::RU_484_TONE;
        index = val - 64;
    }
    else if (val == 67)
    {
        ruType = RuType::RU_996_TONE;
        index = 1;
    }
    else if (val == 68)
    {
        ruType = RuType::RU_2x996_TONE;
        index = 1;
    }
    else if (val == 69)
    {
        NS_ASSERT(m_variant == TriggerFrameVariant::EHT);
        ruType = RuType::RU_4x996_TONE;
        index = 1;
    }
    else
    {
        NS_FATAL_ERROR("Reserved value.");
    }

    if (m_variant == TriggerFrameVariant::EHT)
    {
        return EhtRu::RuSpec{ruType, index, !m_ps160, primaryOrLower80MHz};
    }

    return HeRu::RuSpec{ruType, index, primaryOrLower80MHz};
}

void
CtrlTriggerUserInfoField::SetMuRtsRuAllocation(uint8_t value)
{
    NS_ABORT_MSG_IF(m_triggerType != TriggerFrameType::MU_RTS_TRIGGER,
                    "SetMuRtsRuAllocation() can only be used for MU-RTS");
    NS_ABORT_MSG_IF(
        value < 61 || value > 69,
        "Value "
            << +value
            << " is not admitted for B7-B1 of the RU Allocation subfield of MU-RTS Trigger Frames");

    m_ruAllocation = (value << 1);
    if (value >= 68)
    {
        // set B0 for 160 MHz, 80+80 MHz and 320 MHz indication
        m_ruAllocation++;
    }
    if (value == 69)
    {
        // set 160 for 320 MHz indication
        m_ps160 = true;
    }
}

uint8_t
CtrlTriggerUserInfoField::GetMuRtsRuAllocation() const
{
    NS_ABORT_MSG_IF(m_triggerType != TriggerFrameType::MU_RTS_TRIGGER,
                    "GetMuRtsRuAllocation() can only be used for MU-RTS");
    uint8_t value = (m_ruAllocation >> 1);
    NS_ABORT_MSG_IF(
        value < 61 || value > 69,
        "Value "
            << +value
            << " is not admitted for B7-B1 of the RU Allocation subfield of MU-RTS Trigger Frames");
    return value;
}

void
CtrlTriggerUserInfoField::SetUlFecCodingType(bool ldpc)
{
    m_ulFecCodingType = ldpc;
}

bool
CtrlTriggerUserInfoField::GetUlFecCodingType() const
{
    return m_ulFecCodingType;
}

void
CtrlTriggerUserInfoField::SetUlMcs(uint8_t mcs)
{
    uint8_t maxMcs = m_variant == TriggerFrameVariant::EHT ? 13 : 11;
    NS_ABORT_MSG_IF(mcs > maxMcs, "Invalid MCS index");
    m_ulMcs = mcs;
}

uint8_t
CtrlTriggerUserInfoField::GetUlMcs() const
{
    return m_ulMcs;
}

void
CtrlTriggerUserInfoField::SetUlDcm(bool dcm)
{
    NS_ASSERT_MSG(m_variant == TriggerFrameVariant::HE, "UL DCM flag only present in HE variant");
    m_ulDcm = dcm;
}

bool
CtrlTriggerUserInfoField::GetUlDcm() const
{
    NS_ASSERT_MSG(m_variant == TriggerFrameVariant::HE, "UL DCM flag only present in HE variant");
    return m_ulDcm;
}

void
CtrlTriggerUserInfoField::SetSsAllocation(uint8_t startingSs, uint8_t nSs)
{
    NS_ABORT_MSG_IF(m_aid12 == 0 || m_aid12 == 2045, "SS Allocation subfield not present");
    NS_ABORT_MSG_IF(!startingSs || startingSs > 8, "Starting SS must be from 1 to 8");
    NS_ABORT_MSG_IF(!nSs || nSs > 8, "Number of SS must be from 1 to 8");

    m_bits26To31.ssAllocation.startingSs = startingSs - 1;
    m_bits26To31.ssAllocation.nSs = nSs - 1;
}

uint8_t
CtrlTriggerUserInfoField::GetStartingSs() const
{
    if (m_aid12 == 0 || m_aid12 == 2045)
    {
        return 1;
    }
    return m_bits26To31.ssAllocation.startingSs + 1;
}

uint8_t
CtrlTriggerUserInfoField::GetNss() const
{
    if (m_aid12 == 0 || m_aid12 == 2045)
    {
        return 1;
    }
    return m_bits26To31.ssAllocation.nSs + 1;
}

void
CtrlTriggerUserInfoField::SetRaRuInformation(uint8_t nRaRu, bool moreRaRu)
{
    NS_ABORT_MSG_IF(m_aid12 != 0 && m_aid12 != 2045, "RA-RU Information subfield not present");
    NS_ABORT_MSG_IF(!nRaRu || nRaRu > 32, "Number of contiguous RA-RUs must be from 1 to 32");

    m_bits26To31.raRuInformation.nRaRu = nRaRu - 1;
    m_bits26To31.raRuInformation.moreRaRu = moreRaRu;
}

uint8_t
CtrlTriggerUserInfoField::GetNRaRus() const
{
    NS_ABORT_MSG_IF(m_aid12 != 0 && m_aid12 != 2045, "RA-RU Information subfield not present");

    return m_bits26To31.raRuInformation.nRaRu + 1;
}

bool
CtrlTriggerUserInfoField::GetMoreRaRu() const
{
    NS_ABORT_MSG_IF(m_aid12 != 0 && m_aid12 != 2045, "RA-RU Information subfield not present");

    return m_bits26To31.raRuInformation.moreRaRu;
}

void
CtrlTriggerUserInfoField::SetUlTargetRssiMaxTxPower()
{
    m_ulTargetRssi = 127; // see Table 9-25i of 802.11ax amendment D3.0
}

void
CtrlTriggerUserInfoField::SetUlTargetRssi(int8_t dBm)
{
    NS_ABORT_MSG_IF(dBm < -110 || dBm > -20, "Invalid values for signal power");

    m_ulTargetRssi = static_cast<uint8_t>(110 + dBm);
}

bool
CtrlTriggerUserInfoField::IsUlTargetRssiMaxTxPower() const
{
    return (m_ulTargetRssi == 127);
}

int8_t
CtrlTriggerUserInfoField::GetUlTargetRssi() const
{
    NS_ABORT_MSG_IF(m_ulTargetRssi == 127, "STA must use its max TX power");

    return static_cast<int8_t>(m_ulTargetRssi) - 110;
}

void
CtrlTriggerUserInfoField::SetBasicTriggerDepUserInfo(uint8_t spacingFactor,
                                                     uint8_t tidLimit,
                                                     AcIndex prefAc)
{
    NS_ABORT_MSG_IF(m_triggerType != TriggerFrameType::BASIC_TRIGGER, "Not a Basic Trigger Frame");

    m_basicTriggerDependentUserInfo = (spacingFactor & 0x03) |
                                      (tidLimit & 0x07) << 2
                                      // B5 is reserved
                                      | (prefAc & 0x03) << 6;
}

uint8_t
CtrlTriggerUserInfoField::GetMpduMuSpacingFactor() const
{
    NS_ABORT_MSG_IF(m_triggerType != TriggerFrameType::BASIC_TRIGGER, "Not a Basic Trigger Frame");

    return m_basicTriggerDependentUserInfo & 0x03;
}

uint8_t
CtrlTriggerUserInfoField::GetTidAggregationLimit() const
{
    NS_ABORT_MSG_IF(m_triggerType != TriggerFrameType::BASIC_TRIGGER, "Not a Basic Trigger Frame");

    return (m_basicTriggerDependentUserInfo & 0x1c) >> 2;
}

AcIndex
CtrlTriggerUserInfoField::GetPreferredAc() const
{
    NS_ABORT_MSG_IF(m_triggerType != TriggerFrameType::BASIC_TRIGGER, "Not a Basic Trigger Frame");

    return AcIndex((m_basicTriggerDependentUserInfo & 0xc0) >> 6);
}

void
CtrlTriggerUserInfoField::SetMuBarTriggerDepUserInfo(const CtrlBAckRequestHeader& bar)
{
    NS_ABORT_MSG_IF(m_triggerType != TriggerFrameType::MU_BAR_TRIGGER,
                    "Not a MU-BAR Trigger frame");
    NS_ABORT_MSG_IF(bar.GetType().m_variant != BlockAckReqType::COMPRESSED &&
                        bar.GetType().m_variant != BlockAckReqType::MULTI_TID,
                    "BAR Control indicates it is neither the Compressed nor the Multi-TID variant");
    m_muBarTriggerDependentUserInfo = bar;
}

const CtrlBAckRequestHeader&
CtrlTriggerUserInfoField::GetMuBarTriggerDepUserInfo() const
{
    NS_ABORT_MSG_IF(m_triggerType != TriggerFrameType::MU_BAR_TRIGGER,
                    "Not a MU-BAR Trigger frame");

    return m_muBarTriggerDependentUserInfo;
}

/*****************************************
 * Trigger frame - Special User Info field
 *****************************************/

CtrlTriggerSpecialUserInfoField::CtrlTriggerSpecialUserInfoField(TriggerFrameType triggerType)
    : m_triggerType(triggerType)
{
}

CtrlTriggerSpecialUserInfoField&
CtrlTriggerSpecialUserInfoField::operator=(const CtrlTriggerSpecialUserInfoField& other)
{
    // check for self-assignment
    if (&other == this)
    {
        return *this;
    }

    m_triggerType = other.m_triggerType;
    m_ulBwExt = other.m_ulBwExt;
    m_muBarTriggerDependentUserInfo = other.m_muBarTriggerDependentUserInfo;

    return *this;
}

uint32_t
CtrlTriggerSpecialUserInfoField::GetSerializedSize() const
{
    uint32_t size = 0;
    size += 5; // User Info (excluding Trigger Dependent User Info)

    switch (m_triggerType)
    {
    case TriggerFrameType::BASIC_TRIGGER:
    case TriggerFrameType::BFRP_TRIGGER:
        size += 1;
        break;
    case TriggerFrameType::MU_BAR_TRIGGER:
        size +=
            m_muBarTriggerDependentUserInfo.GetSerializedSize(); // BAR Control and BAR Information
        break;
    default:;
        // The Trigger Dependent User Info subfield is not present in the other variants
    }

    return size;
}

Buffer::Iterator
CtrlTriggerSpecialUserInfoField::Serialize(Buffer::Iterator start) const
{
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::BFRP_TRIGGER,
                    "BFRP Trigger frame is not supported");
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::GCR_MU_BAR_TRIGGER,
                    "GCR-MU-BAR Trigger frame is not supported");
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::NFRP_TRIGGER,
                    "NFRP Trigger frame is not supported");

    Buffer::Iterator i = start;

    uint32_t userInfo = 0;
    userInfo |= (AID_SPECIAL_USER & 0x0fff);
    userInfo |= (static_cast<uint32_t>(m_ulBwExt) << 15);
    i.WriteHtolsbU32(userInfo);
    i.WriteU8(0);
    // TODO: EHT Spatial Reuse and U-SIG Disregard And Validate

    if (m_triggerType == TriggerFrameType::BASIC_TRIGGER)
    {
        // The length is one octet and all the subfields are reserved in a Basic Trigger frame and
        // in a BFRP Trigger frame
        i.WriteU8(0);
    }
    else if (m_triggerType == TriggerFrameType::MU_BAR_TRIGGER)
    {
        m_muBarTriggerDependentUserInfo.Serialize(i);
        i.Next(m_muBarTriggerDependentUserInfo.GetSerializedSize());
    }

    return i;
}

Buffer::Iterator
CtrlTriggerSpecialUserInfoField::Deserialize(Buffer::Iterator start)
{
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::BFRP_TRIGGER,
                    "BFRP Trigger frame is not supported");
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::GCR_MU_BAR_TRIGGER,
                    "GCR-MU-BAR Trigger frame is not supported");
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::NFRP_TRIGGER,
                    "NFRP Trigger frame is not supported");

    Buffer::Iterator i = start;

    const auto userInfo = i.ReadLsbtohU32();
    i.ReadU8();
    // TODO: EHT Spatial Reuse and U-SIG Disregard And Validate

    const uint16_t aid12 = userInfo & 0x0fff;
    NS_ABORT_MSG_IF(aid12 != AID_SPECIAL_USER, "Failed to deserialize Special User Info field");
    m_ulBwExt = (userInfo >> 15) & 0x03;

    if (m_triggerType == TriggerFrameType::BASIC_TRIGGER)
    {
        i.ReadU8();
    }
    else if (m_triggerType == TriggerFrameType::MU_BAR_TRIGGER)
    {
        const auto len = m_muBarTriggerDependentUserInfo.Deserialize(i);
        i.Next(len);
    }

    return i;
}

TriggerFrameType
CtrlTriggerSpecialUserInfoField::GetType() const
{
    return m_triggerType;
}

void
CtrlTriggerSpecialUserInfoField::SetUlBwExt(MHz_u bw)
{
    switch (static_cast<uint16_t>(bw))
    {
    case 20:
    case 40:
    case 80:
        m_ulBwExt = 0;
        break;
    case 160:
        m_ulBwExt = 1;
        break;
    case 320:
        m_ulBwExt = 2;
        // TODO: differentiate channelization 1 from channelization 2
        break;
    default:
        NS_FATAL_ERROR("Bandwidth value not allowed.");
        break;
    }
}

uint8_t
CtrlTriggerSpecialUserInfoField::GetUlBwExt() const
{
    return m_ulBwExt;
}

void
CtrlTriggerSpecialUserInfoField::SetMuBarTriggerDepUserInfo(const CtrlBAckRequestHeader& bar)
{
    NS_ABORT_MSG_IF(m_triggerType != TriggerFrameType::MU_BAR_TRIGGER,
                    "Not a MU-BAR Trigger frame");
    NS_ABORT_MSG_IF(bar.GetType().m_variant != BlockAckReqType::COMPRESSED &&
                        bar.GetType().m_variant != BlockAckReqType::MULTI_TID,
                    "BAR Control indicates it is neither the Compressed nor the Multi-TID variant");
    m_muBarTriggerDependentUserInfo = bar;
}

const CtrlBAckRequestHeader&
CtrlTriggerSpecialUserInfoField::GetMuBarTriggerDepUserInfo() const
{
    NS_ABORT_MSG_IF(m_triggerType != TriggerFrameType::MU_BAR_TRIGGER,
                    "Not a MU-BAR Trigger frame");

    return m_muBarTriggerDependentUserInfo;
}

/***********************************
 *       Trigger frame
 ***********************************/

NS_OBJECT_ENSURE_REGISTERED(CtrlTriggerHeader);

CtrlTriggerHeader::CtrlTriggerHeader()
    : m_variant(TriggerFrameVariant::HE),
      m_triggerType(TriggerFrameType::BASIC_TRIGGER),
      m_ulLength(0),
      m_moreTF(false),
      m_csRequired(false),
      m_ulBandwidth(0),
      m_giAndLtfType(0),
      m_apTxPower(0),
      m_ulSpatialReuse(0),
      m_padding(0)
{
}

CtrlTriggerHeader::CtrlTriggerHeader(TriggerFrameType type, const WifiTxVector& txVector)
    : CtrlTriggerHeader()
{
    m_triggerType = type;
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::MU_RTS_TRIGGER,
                    "This constructor cannot be used for MU-RTS");

    switch (txVector.GetPreambleType())
    {
    case WIFI_PREAMBLE_HE_TB:
        m_variant = TriggerFrameVariant::HE;
        break;
    case WIFI_PREAMBLE_EHT_TB:
        m_variant = TriggerFrameVariant::EHT;
        break;
    default:
        NS_ABORT_MSG("Cannot create a TF out of a TXVECTOR with preamble type: "
                     << txVector.GetPreambleType());
    }

    // special user is always present if solicited TB PPDU format is EHT or later
    if (txVector.GetModulationClass() >= WifiModulationClass::WIFI_MOD_CLASS_EHT)
    {
        NS_ASSERT(m_variant == TriggerFrameVariant::EHT);
        m_specialUserInfoField.emplace(m_triggerType);
    }

    SetUlBandwidth(txVector.GetChannelWidth());
    SetUlLength(txVector.GetLength());
    const auto gi = txVector.GetGuardInterval().GetNanoSeconds();
    if ((gi == 800) || (gi == 1600))
    {
        m_giAndLtfType = 1;
    }
    else
    {
        m_giAndLtfType = 2;
    }

    for (auto& userInfo : txVector.GetHeMuUserInfoMap())
    {
        CtrlTriggerUserInfoField& ui = AddUserInfoField();
        ui.SetAid12(userInfo.first);
        ui.SetRuAllocation(userInfo.second.ru);
        ui.SetUlMcs(userInfo.second.mcs);
        ui.SetSsAllocation(1, userInfo.second.nss); // MU-MIMO is not supported
    }
}

CtrlTriggerHeader::~CtrlTriggerHeader()
{
}

CtrlTriggerHeader&
CtrlTriggerHeader::operator=(const CtrlTriggerHeader& trigger)
{
    // check for self-assignment
    if (&trigger == this)
    {
        return *this;
    }

    m_variant = trigger.m_variant;
    m_triggerType = trigger.m_triggerType;
    m_ulLength = trigger.m_ulLength;
    m_moreTF = trigger.m_moreTF;
    m_csRequired = trigger.m_csRequired;
    m_ulBandwidth = trigger.m_ulBandwidth;
    m_giAndLtfType = trigger.m_giAndLtfType;
    m_apTxPower = trigger.m_apTxPower;
    m_ulSpatialReuse = trigger.m_ulSpatialReuse;
    m_padding = trigger.m_padding;
    m_specialUserInfoField.reset();
    m_specialUserInfoField = trigger.m_specialUserInfoField;
    m_userInfoFields.clear();
    m_userInfoFields = trigger.m_userInfoFields;
    return *this;
}

TypeId
CtrlTriggerHeader::GetTypeId()
{
    static TypeId tid = TypeId("ns3::CtrlTriggerHeader")
                            .SetParent<Header>()
                            .SetGroupName("Wifi")
                            .AddConstructor<CtrlTriggerHeader>();
    return tid;
}

TypeId
CtrlTriggerHeader::GetInstanceTypeId() const
{
    return GetTypeId();
}

void
CtrlTriggerHeader::Print(std::ostream& os) const
{
    os << "TriggerType=" << GetTypeString() << ", Bandwidth=" << +GetUlBandwidth()
       << ", UL Length=" << m_ulLength;

    for (auto& ui : m_userInfoFields)
    {
        ui.Print(os);
    }
}

void
CtrlTriggerHeader::SetVariant(TriggerFrameVariant variant)
{
    NS_ABORT_MSG_IF(!m_userInfoFields.empty(),
                    "Cannot change Common Info field variant if User Info fields are present");
    m_variant = variant;
    // special user is always present if User Info field variant is EHT or later
    if (!m_specialUserInfoField && (m_variant >= TriggerFrameVariant::EHT))
    {
        m_specialUserInfoField.emplace(m_triggerType);
    }
}

TriggerFrameVariant
CtrlTriggerHeader::GetVariant() const
{
    return m_variant;
}

uint32_t
CtrlTriggerHeader::GetSerializedSize() const
{
    uint32_t size = 0;
    size += 8; // Common Info (excluding Trigger Dependent Common Info)

    // Add the size of the Trigger Dependent Common Info subfield
    if (m_triggerType == TriggerFrameType::GCR_MU_BAR_TRIGGER)
    {
        size += 4;
    }

    if (m_specialUserInfoField)
    {
        NS_ASSERT(m_variant == TriggerFrameVariant::EHT);
        size += m_specialUserInfoField->GetSerializedSize();
    }

    for (auto& ui : m_userInfoFields)
    {
        size += ui.GetSerializedSize();
    }

    size += m_padding;

    return size;
}

void
CtrlTriggerHeader::Serialize(Buffer::Iterator start) const
{
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::BFRP_TRIGGER,
                    "BFRP Trigger frame is not supported");
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::GCR_MU_BAR_TRIGGER,
                    "GCR-MU-BAR Trigger frame is not supported");
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::NFRP_TRIGGER,
                    "NFRP Trigger frame is not supported");

    Buffer::Iterator i = start;

    uint64_t commonInfo = 0;
    commonInfo |= (static_cast<uint8_t>(m_triggerType) & 0x0f);
    commonInfo |= (m_ulLength & 0x0fff) << 4;
    commonInfo |= (m_moreTF ? 1 << 16 : 0);
    commonInfo |= (m_csRequired ? 1 << 17 : 0);
    commonInfo |= (m_ulBandwidth & 0x03) << 18;
    commonInfo |= (m_giAndLtfType & 0x03) << 20;
    commonInfo |= static_cast<uint64_t>(m_apTxPower & 0x3f) << 28;
    commonInfo |= static_cast<uint64_t>(m_ulSpatialReuse) << 37;
    if (m_variant == TriggerFrameVariant::HE)
    {
        uint64_t ulHeSigA2 = 0x01ff; // nine bits equal to 1
        commonInfo |= ulHeSigA2 << 54;
    }

    i.WriteHtolsbU64(commonInfo);

    if (m_specialUserInfoField)
    {
        NS_ASSERT(m_variant == TriggerFrameVariant::EHT);
        i = m_specialUserInfoField->Serialize(i);
    }

    for (auto& ui : m_userInfoFields)
    {
        i = ui.Serialize(i);
    }

    for (std::size_t count = 0; count < m_padding; count++)
    {
        i.WriteU8(0xff); // Padding field
    }
}

uint32_t
CtrlTriggerHeader::Deserialize(Buffer::Iterator start)
{
    Buffer::Iterator i = start;

    uint64_t commonInfo = i.ReadLsbtohU64();

    m_triggerType = static_cast<TriggerFrameType>(commonInfo & 0x0f);
    m_ulLength = (commonInfo >> 4) & 0x0fff;
    m_moreTF = (commonInfo >> 16) & 0x01;
    m_csRequired = (commonInfo >> 17) & 0x01;
    m_ulBandwidth = (commonInfo >> 18) & 0x03;
    m_giAndLtfType = (commonInfo >> 20) & 0x03;
    m_apTxPower = (commonInfo >> 28) & 0x3f;
    m_ulSpatialReuse = (commonInfo >> 37) & 0xffff;
    uint8_t bit54and55 = (commonInfo >> 54) & 0x03;
    m_variant = bit54and55 == 3 ? TriggerFrameVariant::HE : TriggerFrameVariant::EHT;
    m_userInfoFields.clear();
    m_padding = 0;

    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::BFRP_TRIGGER,
                    "BFRP Trigger frame is not supported");
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::GCR_MU_BAR_TRIGGER,
                    "GCR-MU-BAR Trigger frame is not supported");
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::NFRP_TRIGGER,
                    "NFRP Trigger frame is not supported");

    if (m_variant == TriggerFrameVariant::EHT)
    {
        m_specialUserInfoField.reset();
        const auto userInfo = i.ReadLsbtohU16();
        i.Prev(2);
        if (const auto aid12 = userInfo & 0x0fff; aid12 == AID_SPECIAL_USER)
        {
            m_specialUserInfoField.emplace(m_triggerType);
            i = m_specialUserInfoField->Deserialize(i);
        }
    }

    while (i.GetRemainingSize() >= 2)
    {
        // read the first 2 bytes to check if we encountered the Padding field
        if (i.ReadU16() == 0xffff)
        {
            m_padding = i.GetRemainingSize() + 2;
        }
        else
        {
            // go back 2 bytes to deserialize the User Info field from the beginning
            i.Prev(2);
            CtrlTriggerUserInfoField& ui = AddUserInfoField();
            i = ui.Deserialize(i);
        }
    }

    return i.GetDistanceFrom(start);
}

void
CtrlTriggerHeader::SetType(TriggerFrameType type)
{
    m_triggerType = type;
}

TriggerFrameType
CtrlTriggerHeader::GetType() const
{
    return m_triggerType;
}

const char*
CtrlTriggerHeader::GetTypeString() const
{
    return GetTypeString(GetType());
}

const char*
CtrlTriggerHeader::GetTypeString(TriggerFrameType type)
{
#define FOO(x)                                                                                     \
    case TriggerFrameType::x:                                                                      \
        return #x;

    switch (type)
    {
        FOO(BASIC_TRIGGER);
        FOO(BFRP_TRIGGER);
        FOO(MU_BAR_TRIGGER);
        FOO(MU_RTS_TRIGGER);
        FOO(BSRP_TRIGGER);
        FOO(GCR_MU_BAR_TRIGGER);
        FOO(BQRP_TRIGGER);
        FOO(NFRP_TRIGGER);
    default:
        return "ERROR";
    }
#undef FOO
}

bool
CtrlTriggerHeader::IsBasic() const
{
    return (m_triggerType == TriggerFrameType::BASIC_TRIGGER);
}

bool
CtrlTriggerHeader::IsBfrp() const
{
    return (m_triggerType == TriggerFrameType::BFRP_TRIGGER);
}

bool
CtrlTriggerHeader::IsMuBar() const
{
    return (m_triggerType == TriggerFrameType::MU_BAR_TRIGGER);
}

bool
CtrlTriggerHeader::IsMuRts() const
{
    return (m_triggerType == TriggerFrameType::MU_RTS_TRIGGER);
}

bool
CtrlTriggerHeader::IsBsrp() const
{
    return (m_triggerType == TriggerFrameType::BSRP_TRIGGER);
}

bool
CtrlTriggerHeader::IsGcrMuBar() const
{
    return (m_triggerType == TriggerFrameType::GCR_MU_BAR_TRIGGER);
}

bool
CtrlTriggerHeader::IsBqrp() const
{
    return (m_triggerType == TriggerFrameType::BQRP_TRIGGER);
}

bool
CtrlTriggerHeader::IsNfrp() const
{
    return (m_triggerType == TriggerFrameType::NFRP_TRIGGER);
}

void
CtrlTriggerHeader::SetUlLength(uint16_t len)
{
    m_ulLength = (len & 0x0fff);
}

uint16_t
CtrlTriggerHeader::GetUlLength() const
{
    return m_ulLength;
}

WifiTxVector
CtrlTriggerHeader::GetHeTbTxVector(uint16_t staId) const
{
    NS_ABORT_MSG_IF(m_triggerType == TriggerFrameType::MU_RTS_TRIGGER,
                    "GetHeTbTxVector() cannot be used for MU-RTS");
    auto userInfoIt = FindUserInfoWithAid(staId);
    NS_ASSERT(userInfoIt != end());

    WifiTxVector v;
    v.SetPreambleType(userInfoIt->GetPreambleType());
    v.SetChannelWidth(GetUlBandwidth());
    v.SetGuardInterval(GetGuardInterval());
    v.SetLength(GetUlLength());
    v.SetHeMuUserInfo(
        staId,
        {userInfoIt->GetRuAllocation(), userInfoIt->GetUlMcs(), userInfoIt->GetNss()});
    return v;
}

void
CtrlTriggerHeader::SetMoreTF(bool more)
{
    m_moreTF = more;
}

bool
CtrlTriggerHeader::GetMoreTF() const
{
    return m_moreTF;
}

void
CtrlTriggerHeader::SetCsRequired(bool cs)
{
    m_csRequired = cs;
}

bool
CtrlTriggerHeader::GetCsRequired() const
{
    return m_csRequired;
}

void
CtrlTriggerHeader::SetUlBandwidth(MHz_u bw)
{
    switch (static_cast<uint16_t>(bw))
    {
    case 20:
        m_ulBandwidth = 0;
        break;
    case 40:
        m_ulBandwidth = 1;
        break;
    case 80:
        m_ulBandwidth = 2;
        break;
    case 160:
    case 320:
        m_ulBandwidth = 3;
        break;
    default:
        NS_FATAL_ERROR("Bandwidth value not allowed.");
        break;
    }
    if (bw > MHz_u{160})
    {
        NS_ASSERT(m_specialUserInfoField);
    }
    if (m_specialUserInfoField)
    {
        NS_ASSERT(m_variant == TriggerFrameVariant::EHT);
        m_specialUserInfoField->SetUlBwExt(bw);
    }
}

MHz_u
CtrlTriggerHeader::GetUlBandwidth() const
{
    if (m_specialUserInfoField)
    {
        NS_ASSERT(m_variant == TriggerFrameVariant::EHT);
        if (m_specialUserInfoField->GetUlBwExt() > 1)
        {
            return MHz_u{320};
        }
    }
    return (1 << m_ulBandwidth) * MHz_u{20};
}

void
CtrlTriggerHeader::SetGiAndLtfType(Time guardInterval, uint8_t ltfType)
{
    const auto gi = guardInterval.GetNanoSeconds();
    if ((ltfType == 1) && (gi == 1600))
    {
        m_giAndLtfType = 0;
    }
    else if ((ltfType == 2) && (gi == 1600))
    {
        m_giAndLtfType = 1;
    }
    else if ((ltfType == 4) && (gi == 3200))
    {
        m_giAndLtfType = 2;
    }
    else
    {
        NS_FATAL_ERROR("Invalid combination of GI and LTF type");
    }
}

Time
CtrlTriggerHeader::GetGuardInterval() const
{
    if (m_giAndLtfType == 0 || m_giAndLtfType == 1)
    {
        return NanoSeconds(1600);
    }
    else if (m_giAndLtfType == 2)
    {
        return NanoSeconds(3200);
    }
    else
    {
        NS_FATAL_ERROR("Invalid value for GI And LTF Type subfield");
    }
}

uint8_t
CtrlTriggerHeader::GetLtfType() const
{
    if (m_giAndLtfType == 0)
    {
        return 1;
    }
    else if (m_giAndLtfType == 1)
    {
        return 2;
    }
    else if (m_giAndLtfType == 2)
    {
        return 4;
    }
    else
    {
        NS_FATAL_ERROR("Invalid value for GI And LTF Type subfield");
    }
}

void
CtrlTriggerHeader::SetApTxPower(int8_t power)
{
    // see Table 9-25f "AP Tx Power subfield encoding" of 802.11ax amendment D3.0
    NS_ABORT_MSG_IF(power < -20 || power > 40, "Out of range power values");

    m_apTxPower = static_cast<uint8_t>(power + 20);
}

int8_t
CtrlTriggerHeader::GetApTxPower() const
{
    // see Table 9-25f "AP Tx Power subfield encoding" of 802.11ax amendment D3.0
    return static_cast<int8_t>(m_apTxPower) - 20;
}

void
CtrlTriggerHeader::SetUlSpatialReuse(uint16_t sr)
{
    m_ulSpatialReuse = sr;
}

uint16_t
CtrlTriggerHeader::GetUlSpatialReuse() const
{
    return m_ulSpatialReuse;
}

void
CtrlTriggerHeader::SetPaddingSize(std::size_t size)
{
    NS_ABORT_MSG_IF(size == 1, "The Padding field, if present, shall be at least two octets");
    m_padding = size;
}

std::size_t
CtrlTriggerHeader::GetPaddingSize() const
{
    return m_padding;
}

CtrlTriggerHeader
CtrlTriggerHeader::GetCommonInfoField() const
{
    // make a copy of this Trigger Frame and remove the User Info fields (including the Special User
    // Info field) from the copy
    CtrlTriggerHeader trigger(*this);
    trigger.m_specialUserInfoField.reset();
    trigger.m_userInfoFields.clear();
    return trigger;
}

CtrlTriggerUserInfoField&
CtrlTriggerHeader::AddUserInfoField()
{
    m_userInfoFields.emplace_back(m_triggerType, m_variant);
    return m_userInfoFields.back();
}

CtrlTriggerUserInfoField&
CtrlTriggerHeader::AddUserInfoField(const CtrlTriggerUserInfoField& userInfo)
{
    NS_ABORT_MSG_IF(
        userInfo.GetType() != m_triggerType,
        "Trying to add a User Info field of a type other than the type of the Trigger Frame");
    m_userInfoFields.push_back(userInfo);
    return m_userInfoFields.back();
}

CtrlTriggerHeader::Iterator
CtrlTriggerHeader::RemoveUserInfoField(ConstIterator userInfoIt)
{
    return m_userInfoFields.erase(userInfoIt);
}

CtrlTriggerHeader::ConstIterator
CtrlTriggerHeader::begin() const
{
    return m_userInfoFields.begin();
}

CtrlTriggerHeader::ConstIterator
CtrlTriggerHeader::end() const
{
    return m_userInfoFields.end();
}

CtrlTriggerHeader::Iterator
CtrlTriggerHeader::begin()
{
    return m_userInfoFields.begin();
}

CtrlTriggerHeader::Iterator
CtrlTriggerHeader::end()
{
    return m_userInfoFields.end();
}

std::size_t
CtrlTriggerHeader::GetNUserInfoFields() const
{
    return m_userInfoFields.size();
}

CtrlTriggerHeader::ConstIterator
CtrlTriggerHeader::FindUserInfoWithAid(ConstIterator start, uint16_t aid12) const
{
    // the lambda function returns true if a User Info field has the AID12 subfield
    // equal to the given aid12 value
    return std::find_if(start, end(), [aid12](const CtrlTriggerUserInfoField& ui) -> bool {
        return (ui.GetAid12() == aid12);
    });
}

CtrlTriggerHeader::ConstIterator
CtrlTriggerHeader::FindUserInfoWithAid(uint16_t aid12) const
{
    return FindUserInfoWithAid(m_userInfoFields.begin(), aid12);
}

CtrlTriggerHeader::ConstIterator
CtrlTriggerHeader::FindUserInfoWithRaRuAssociated(ConstIterator start) const
{
    return FindUserInfoWithAid(start, 0);
}

CtrlTriggerHeader::ConstIterator
CtrlTriggerHeader::FindUserInfoWithRaRuAssociated() const
{
    return FindUserInfoWithAid(0);
}

CtrlTriggerHeader::ConstIterator
CtrlTriggerHeader::FindUserInfoWithRaRuUnassociated(ConstIterator start) const
{
    return FindUserInfoWithAid(start, 2045);
}

CtrlTriggerHeader::ConstIterator
CtrlTriggerHeader::FindUserInfoWithRaRuUnassociated() const
{
    return FindUserInfoWithAid(2045);
}

bool
CtrlTriggerHeader::IsValid() const
{
    if (m_triggerType == TriggerFrameType::MU_RTS_TRIGGER)
    {
        return true;
    }

    // check that allocated RUs do not overlap
    // TODO This is not a problem in case of UL MU-MIMO
    std::vector<WifiRu::RuSpec> prevRus;
    for (auto& ui : m_userInfoFields)
    {
        if (WifiRu::DoesOverlap(GetUlBandwidth(), ui.GetRuAllocation(), prevRus))
        {
            return false;
        }
        prevRus.push_back(ui.GetRuAllocation());
    }
    return true;
}

} // namespace ns3
