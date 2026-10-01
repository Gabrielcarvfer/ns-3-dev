/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "dns-header.h"

#include "ns3/assert.h"
#include "ns3/ipv4-address.h"
#include "ns3/ipv6-address.h"
#include "ns3/log.h"

#include <algorithm>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("DnsHeader");

NS_OBJECT_ENSURE_REGISTERED(DnsHeader);

namespace
{

constexpr uint32_t MAX_NAME_SIZE{255};       ///< maximum size of an encoded name
constexpr uint32_t MAX_LABEL_SIZE{63};       ///< maximum size of a label
constexpr uint32_t MAX_POINTERS{64};         ///< maximum compression pointers in a name
constexpr uint32_t RECORD_FIXED_SIZE{10};    ///< type, class, TTL and data length of a record
constexpr uint32_t SOA_FIXED_SIZE{20};       ///< the five 32-bit fields of an SOA record
constexpr uint16_t OPCODE_MASK{0x7800};      ///< the opcode in the flags
constexpr uint16_t RCODE_MASK{0x000f};       ///< the RCODE in the flags
constexpr uint8_t LABEL_TYPE_MASK{0xc0};     ///< the type of a label in its first octet
constexpr uint8_t COMPRESSION_POINTER{0xc0}; ///< the label is a compression pointer

/**
 * @param labels the labels of a name
 * @return the size of the encoded name
 */
uint32_t
NameSize(const std::vector<std::string>& labels)
{
    uint32_t size = 1;
    for (const auto& label : labels)
    {
        size += label.size() + 1;
    }
    return size;
}

/**
 * Split a name, aborting if it is not valid.
 * @param name the name
 * @return the labels
 */
std::vector<std::string>
Labels(const std::string& name)
{
    auto labels = DnsHeader::SplitName(name);
    NS_ASSERT_MSG(labels, "Invalid DNS name " << name);
    return *labels;
}

/**
 * Write a name, without compression.
 * @param i the iterator, advanced past the name
 * @param labels the labels of the name
 */
void
WriteName(Buffer::Iterator& i, const std::vector<std::string>& labels)
{
    for (const auto& label : labels)
    {
        i.WriteU8(label.size());
        i.Write(reinterpret_cast<const uint8_t*>(label.data()), label.size());
    }
    i.WriteU8(0);
}

/**
 * Read a name, which may be compressed (RFC 1035, section 4.1.4).
 *
 * Each compression pointer must point strictly before the previous one, which prevents loops.
 *
 * @param i the iterator, advanced past the name
 * @param start the start of the message, where the compression pointers are relative to
 * @param name the name, in presentation format
 * @return true if the name is well-formed
 */
bool
ReadName(Buffer::Iterator& i, Buffer::Iterator start, std::string& name)
{
    name.clear();
    Buffer::Iterator current = i;
    uint32_t limit = start.GetRemainingSize();
    uint32_t pointers = 0;
    uint32_t size = 0;
    bool jumped = false;
    while (!current.IsEnd())
    {
        const uint8_t length = current.ReadU8();
        if ((length & LABEL_TYPE_MASK) == COMPRESSION_POINTER)
        {
            if (current.IsEnd() || ++pointers > MAX_POINTERS)
            {
                return false;
            }
            const uint32_t target = (length & ~LABEL_TYPE_MASK) << 8 | current.ReadU8();
            const uint32_t pointerOffset = current.GetDistanceFrom(start) - 2;
            if (target >= std::min(limit, pointerOffset))
            {
                return false;
            }
            // the name ends after the first pointer
            if (!jumped)
            {
                i = current;
                jumped = true;
            }
            limit = target;
            current = start;
            current.Next(target);
            continue;
        }
        if (length & LABEL_TYPE_MASK)
        {
            return false;
        }
        size += length + 1;
        if (size > MAX_NAME_SIZE || current.GetRemainingSize() < length)
        {
            return false;
        }
        if (length == 0)
        {
            if (!jumped)
            {
                i = current;
            }
            return true;
        }
        if (!name.empty())
        {
            name += '.';
        }
        for (uint8_t n = 0; n < length; n++)
        {
            const char c = current.ReadU8();
            if (c == '.' || c == '\\')
            {
                name += '\\';
            }
            name += c;
        }
    }
    return false;
}

/**
 * @param record a record
 * @return the size of the data of the record
 */
uint32_t
RdataSize(const DnsResourceRecord& record)
{
    switch (record.type)
    {
    case DnsHeader::TYPE_A:
        return Ipv4Address::ADDRESS_LENGTH;
    case DnsHeader::TYPE_AAAA:
        return Ipv6Address::ADDRESS_LENGTH;
    case DnsHeader::TYPE_CNAME:
    case DnsHeader::TYPE_NS:
        return NameSize(Labels(record.target));
    case DnsHeader::TYPE_SOA:
        return NameSize(Labels(record.soa.mname)) + NameSize(Labels(record.soa.rname)) +
               SOA_FIXED_SIZE;
    default:
        return record.rdata.size();
    }
}

} // namespace

TypeId
DnsHeader::GetTypeId()
{
    static TypeId tid = TypeId("ns3::DnsHeader")
                            .SetParent<Header>()
                            .SetGroupName("InternetApps")
                            .AddConstructor<DnsHeader>();
    return tid;
}

TypeId
DnsHeader::GetInstanceTypeId() const
{
    return GetTypeId();
}

void
DnsHeader::Print(std::ostream& os) const
{
    os << "id " << m_id << (IsResponse() ? " response" : " query") << " opcode " << +GetOpcode()
       << " rcode " << GetRcode() << " flags" << (IsAuthoritativeAnswer() ? " AA" : "")
       << (IsTruncated() ? " TC" : "") << (IsRecursionDesired() ? " RD" : "")
       << (IsRecursionAvailable() ? " RA" : "");
    if (m_hasQuestion)
    {
        os << " question " << m_questionName << " type " << m_questionType << " class "
           << m_questionClass;
    }
    os << " answers " << m_answers.size() << " authorities " << m_authorities.size()
       << " additional " << m_additionalRecords.size();
}

uint32_t
DnsHeader::GetSerializedSize() const
{
    uint32_t size = HEADER_SIZE;
    if (m_hasQuestion)
    {
        size += NameSize(Labels(m_questionName)) + 4;
    }
    for (const auto* section : {&m_answers, &m_authorities, &m_additionalRecords})
    {
        for (const auto& record : *section)
        {
            size += NameSize(Labels(record.name)) + RECORD_FIXED_SIZE + RdataSize(record);
        }
    }
    return size;
}

void
DnsHeader::Serialize(Buffer::Iterator start) const
{
    NS_LOG_FUNCTION(this << &start);
    Buffer::Iterator i = start;
    i.WriteHtonU16(m_id);
    i.WriteHtonU16(m_flags);
    i.WriteHtonU16(m_hasQuestion ? 1 : 0);
    i.WriteHtonU16(m_answers.size());
    i.WriteHtonU16(m_authorities.size());
    i.WriteHtonU16(m_additionalRecords.size());
    if (m_hasQuestion)
    {
        WriteName(i, Labels(m_questionName));
        i.WriteHtonU16(m_questionType);
        i.WriteHtonU16(m_questionClass);
    }
    for (const auto* section : {&m_answers, &m_authorities, &m_additionalRecords})
    {
        for (const auto& record : *section)
        {
            SerializeRecord(i, record);
        }
    }
}

void
DnsHeader::SerializeRecord(Buffer::Iterator& i, const DnsResourceRecord& record)
{
    WriteName(i, Labels(record.name));
    i.WriteHtonU16(record.type);
    i.WriteHtonU16(record.rclass);
    i.WriteHtonU32(record.ttl);
    i.WriteHtonU16(RdataSize(record));
    switch (record.type)
    {
    case TYPE_A: {
        uint8_t buffer[Ipv4Address::ADDRESS_LENGTH];
        Ipv4Address::ConvertFrom(record.address).Serialize(buffer);
        i.Write(buffer, sizeof(buffer));
        break;
    }
    case TYPE_AAAA: {
        uint8_t buffer[Ipv6Address::ADDRESS_LENGTH];
        Ipv6Address::ConvertFrom(record.address).Serialize(buffer);
        i.Write(buffer, sizeof(buffer));
        break;
    }
    case TYPE_CNAME:
    case TYPE_NS:
        WriteName(i, Labels(record.target));
        break;
    case TYPE_SOA:
        WriteName(i, Labels(record.soa.mname));
        WriteName(i, Labels(record.soa.rname));
        i.WriteHtonU32(record.soa.serial);
        i.WriteHtonU32(record.soa.refresh);
        i.WriteHtonU32(record.soa.retry);
        i.WriteHtonU32(record.soa.expire);
        i.WriteHtonU32(record.soa.minimum);
        break;
    default:
        i.Write(record.rdata.data(), record.rdata.size());
        break;
    }
}

uint32_t
DnsHeader::Deserialize(Buffer::Iterator start)
{
    NS_LOG_FUNCTION(this << &start);
    Buffer::Iterator i = start;
    if (i.GetRemainingSize() < HEADER_SIZE)
    {
        return 0;
    }
    m_id = i.ReadNtohU16();
    m_flags = i.ReadNtohU16();
    const uint16_t questions = i.ReadNtohU16();
    const uint16_t answerCount = i.ReadNtohU16();
    const uint16_t authorityCount = i.ReadNtohU16();
    const uint16_t additionalCount = i.ReadNtohU16();

    m_hasQuestion = questions == 1;
    if (questions > 1)
    {
        return 0;
    }
    if (m_hasQuestion)
    {
        if (!ReadName(i, start, m_questionName) || i.GetRemainingSize() < 4)
        {
            return 0;
        }
        m_questionType = i.ReadNtohU16();
        m_questionClass = i.ReadNtohU16();
    }

    m_answers.clear();
    m_authorities.clear();
    m_additionalRecords.clear();
    for (auto [section, count] : {std::pair{&m_answers, answerCount},
                                  std::pair{&m_authorities, authorityCount},
                                  std::pair{&m_additionalRecords, additionalCount}})
    {
        for (uint16_t n = 0; n < count; n++)
        {
            DnsResourceRecord record;
            if (!DeserializeRecord(i, start, record))
            {
                return 0;
            }
            // a single OPT record, in the additional section, with the root as owner (RFC 6891,
            // section 6.1.1)
            if (record.type == TYPE_OPT &&
                (section != &m_additionalRecords || !record.name.empty() || GetOptRecord()))
            {
                return 0;
            }
            section->push_back(std::move(record));
        }
    }
    return i.GetDistanceFrom(start);
}

bool
DnsHeader::DeserializeRecord(Buffer::Iterator& i, Buffer::Iterator start, DnsResourceRecord& record)
{
    if (!ReadName(i, start, record.name) || i.GetRemainingSize() < RECORD_FIXED_SIZE)
    {
        return false;
    }
    record.type = i.ReadNtohU16();
    record.rclass = i.ReadNtohU16();
    record.ttl = i.ReadNtohU32();
    const uint16_t length = i.ReadNtohU16();
    if (i.GetRemainingSize() < length)
    {
        return false;
    }
    Buffer::Iterator end = i;
    end.Next(length);
    switch (record.type)
    {
    case TYPE_A: {
        uint8_t buffer[Ipv4Address::ADDRESS_LENGTH];
        if (length != sizeof(buffer))
        {
            return false;
        }
        i.Read(buffer, sizeof(buffer));
        record.address = Ipv4Address::Deserialize(buffer);
        break;
    }
    case TYPE_AAAA: {
        uint8_t buffer[Ipv6Address::ADDRESS_LENGTH];
        if (length != sizeof(buffer))
        {
            return false;
        }
        i.Read(buffer, sizeof(buffer));
        record.address = Ipv6Address::Deserialize(buffer);
        break;
    }
    case TYPE_CNAME:
    case TYPE_NS:
        if (!ReadName(i, start, record.target) || i.GetDistanceFrom(end) != 0)
        {
            return false;
        }
        break;
    case TYPE_SOA:
        if (!ReadName(i, start, record.soa.mname) || !ReadName(i, start, record.soa.rname) ||
            i.GetDistanceFrom(end) != SOA_FIXED_SIZE)
        {
            return false;
        }
        record.soa.serial = i.ReadNtohU32();
        record.soa.refresh = i.ReadNtohU32();
        record.soa.retry = i.ReadNtohU32();
        record.soa.expire = i.ReadNtohU32();
        record.soa.minimum = i.ReadNtohU32();
        break;
    default:
        record.rdata.resize(length);
        i.Read(record.rdata.data(), length);
        break;
    }
    return true;
}

void
DnsHeader::SetId(uint16_t id)
{
    m_id = id;
}

uint16_t
DnsHeader::GetId() const
{
    return m_id;
}

void
DnsHeader::SetFlag(Flags flag, bool value)
{
    m_flags = value ? (m_flags | flag) : (m_flags & ~flag);
}

void
DnsHeader::SetResponse(bool response)
{
    SetFlag(FLAG_QR, response);
}

bool
DnsHeader::IsResponse() const
{
    return m_flags & FLAG_QR;
}

void
DnsHeader::SetOpcode(uint8_t opcode)
{
    m_flags = (m_flags & ~OPCODE_MASK) | ((opcode << 11) & OPCODE_MASK);
}

uint8_t
DnsHeader::GetOpcode() const
{
    return (m_flags & OPCODE_MASK) >> 11;
}

void
DnsHeader::SetAuthoritativeAnswer(bool authoritative)
{
    SetFlag(FLAG_AA, authoritative);
}

bool
DnsHeader::IsAuthoritativeAnswer() const
{
    return m_flags & FLAG_AA;
}

void
DnsHeader::SetTruncated(bool truncated)
{
    SetFlag(FLAG_TC, truncated);
}

bool
DnsHeader::IsTruncated() const
{
    return m_flags & FLAG_TC;
}

void
DnsHeader::SetRecursionDesired(bool desired)
{
    SetFlag(FLAG_RD, desired);
}

bool
DnsHeader::IsRecursionDesired() const
{
    return m_flags & FLAG_RD;
}

void
DnsHeader::SetRecursionAvailable(bool available)
{
    SetFlag(FLAG_RA, available);
}

bool
DnsHeader::IsRecursionAvailable() const
{
    return m_flags & FLAG_RA;
}

void
DnsHeader::SetRcode(uint16_t rcode)
{
    m_flags = (m_flags & ~RCODE_MASK) | (rcode & RCODE_MASK);
    auto opt = std::find_if(m_additionalRecords.begin(),
                            m_additionalRecords.end(),
                            [](const auto& record) { return record.type == TYPE_OPT; });
    if (opt != m_additionalRecords.end())
    {
        opt->ttl = (opt->ttl & 0x00ffffff) | static_cast<uint32_t>(rcode >> 4) << 24;
    }
    else
    {
        NS_ASSERT_MSG(rcode >> 4 == 0, "An extended RCODE requires an OPT record");
    }
}

uint16_t
DnsHeader::GetRcode() const
{
    uint16_t rcode = m_flags & RCODE_MASK;
    if (auto opt = GetOptRecord())
    {
        rcode |= static_cast<uint16_t>((opt->ttl >> 24) << 4);
    }
    return rcode;
}

void
DnsHeader::SetQuestion(const std::string& name, uint16_t type, uint16_t rclass)
{
    m_hasQuestion = true;
    m_questionName = name;
    m_questionType = type;
    m_questionClass = rclass;
}

bool
DnsHeader::HasQuestion() const
{
    return m_hasQuestion;
}

const std::string&
DnsHeader::GetQuestionName() const
{
    return m_questionName;
}

uint16_t
DnsHeader::GetQuestionType() const
{
    return m_questionType;
}

uint16_t
DnsHeader::GetQuestionClass() const
{
    return m_questionClass;
}

void
DnsHeader::AddAnswer(const DnsResourceRecord& record)
{
    m_answers.push_back(record);
}

void
DnsHeader::AddAuthority(const DnsResourceRecord& record)
{
    m_authorities.push_back(record);
}

void
DnsHeader::AddAdditional(const DnsResourceRecord& record)
{
    m_additionalRecords.push_back(record);
}

const std::vector<DnsResourceRecord>&
DnsHeader::GetAnswers() const
{
    return m_answers;
}

const std::vector<DnsResourceRecord>&
DnsHeader::GetAuthorities() const
{
    return m_authorities;
}

const std::vector<DnsResourceRecord>&
DnsHeader::GetAdditionalRecords() const
{
    return m_additionalRecords;
}

void
DnsHeader::AddOptRecord(uint16_t udpPayloadSize)
{
    DnsResourceRecord opt;
    opt.type = TYPE_OPT;
    opt.rclass = udpPayloadSize;
    m_additionalRecords.push_back(opt);
}

std::optional<DnsResourceRecord>
DnsHeader::GetOptRecord() const
{
    auto opt = std::find_if(m_additionalRecords.begin(),
                            m_additionalRecords.end(),
                            [](const auto& record) { return record.type == TYPE_OPT; });
    if (opt == m_additionalRecords.end())
    {
        return std::nullopt;
    }
    return *opt;
}

std::optional<std::vector<std::string>>
DnsHeader::SplitName(const std::string& name)
{
    std::vector<std::string> labels;
    if (name.empty())
    {
        return labels;
    }
    std::string label;
    for (size_t pos = 0; pos <= name.size(); pos++)
    {
        if (pos == name.size() || name[pos] == '.')
        {
            if (label.empty() || label.size() > MAX_LABEL_SIZE)
            {
                return std::nullopt;
            }
            labels.push_back(std::move(label));
            label.clear();
            continue;
        }
        if (name[pos] == '\\' && ++pos == name.size())
        {
            return std::nullopt;
        }
        label += name[pos];
    }
    if (NameSize(labels) > MAX_NAME_SIZE)
    {
        return std::nullopt;
    }
    return labels;
}

} // namespace ns3
