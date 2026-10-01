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
#include <regex>
#include <string_view>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("DnsHeader");

NS_OBJECT_ENSURE_REGISTERED(DnsHeader);

namespace
{

constexpr uint32_t MAX_NAME_SIZE{255};         ///< Maximum encoded name size, in bytes
constexpr uint32_t MAX_LABEL_SIZE{63};         ///< Maximum label size, in bytes
constexpr uint32_t MAX_POINTERS{64};           ///< Maximum compression pointers per name
constexpr uint32_t RECORD_FIXED_SIZE{10};      ///< Size of TYPE, CLASS, TTL and RDLENGTH
constexpr uint32_t SOA_FIXED_SIZE{20};         ///< Size of the five 32-bit SOA fields
constexpr uint16_t OPCODE_MASK{0x7800};        ///< Opcode bits in the flags word
constexpr uint16_t RCODE_MASK{0x000f};         ///< RCODE bits in the flags word
constexpr uint8_t LABEL_TYPE_MASK{0xc0};       ///< Label type bits in the length byte
constexpr uint8_t COMPRESSION_POINTER{0xc0};   ///< Label type of a compression pointer
constexpr uint32_t QUESTION_FIXED_SIZE{4};     ///< Size of QTYPE and QCLASS
constexpr uint32_t TTL_SIGN_BIT{0x80000000};   ///< TTLs with this bit set are treated as zero
constexpr uint8_t RCODE_BITS{4};               ///< RCODE bits in the flags word
constexpr uint8_t OPT_RCODE_SHIFT{24};         ///< Extended RCODE position in the OPT TTL
constexpr uint8_t OPT_VERSION_SHIFT{16};       ///< EDNS version position in the OPT TTL
constexpr uint32_t OPT_RCODE_MASK{0xff000000}; ///< Extended RCODE bits in the OPT TTL

/**
 * Compute the encoded size of a name.
 * @param labels Name labels.
 * @return The encoded size, in bytes.
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
 * Split a name into labels, asserting that it is valid.
 * @param name Name to split.
 * @return The labels.
 */
std::vector<std::string>
Labels(const std::string& name)
{
    auto labels = DnsHeader::SplitName(name);
    NS_ASSERT_MSG(labels, "Invalid DNS name " << name);
    return *labels;
}

/**
 * Write an uncompressed name.
 * @param i Buffer iterator, advanced past the name.
 * @param labels Name labels.
 */
void
WriteName(Buffer::Iterator& i, const std::vector<std::string>& labels)
{
    for (const auto& label : labels)
    {
        i.WriteU8(label.size());
        for (char c : label)
        {
            i.WriteU8(c);
        }
    }
    i.WriteU8(0);
}

/**
 * Read a possibly compressed name (RFC 1035, section 4.1.4).
 *
 * Each compression pointer must point strictly before the previous one, so that loops are
 * impossible.
 *
 * @param i Buffer iterator, advanced past the name.
 * @param start Start of the message, used to resolve compression pointers.
 * @param name Decoded name, in presentation format.
 * @return True if the name is well-formed.
 */
bool
ReadName(Buffer::Iterator& i, Buffer::Iterator start, std::string& name)
{
    // Dots and backslashes inside labels are escaped in presentation format
    static const std::regex special(R"([.\\])");

    name.clear();
    std::optional<Buffer::Iterator> nameEnd; // Set when the first pointer is followed
    Buffer::Iterator current = i;
    uint32_t limit = start.GetRemainingSize(); // Pointers must point before this offset
    uint32_t pointers = 0;
    uint32_t size = 1; // Terminating zero
    while (!current.IsEnd())
    {
        const uint8_t length = current.ReadU8();
        if (length == 0)
        {
            i = nameEnd.value_or(current);
            return true;
        }
        if ((length & LABEL_TYPE_MASK) == COMPRESSION_POINTER)
        {
            if (current.IsEnd() || ++pointers > MAX_POINTERS)
            {
                return false;
            }
            const uint32_t pointerOffset = current.GetDistanceFrom(start) - 1;
            const uint32_t target = (length & ~LABEL_TYPE_MASK) << 8 | current.ReadU8();
            if (target >= std::min(limit, pointerOffset))
            {
                return false;
            }
            if (!nameEnd)
            {
                nameEnd = current;
            }
            limit = target;
            current = start;
            current.Next(target);
            continue;
        }
        size += length + 1;
        if ((length & LABEL_TYPE_MASK) != 0 || size > MAX_NAME_SIZE ||
            current.GetRemainingSize() < length)
        {
            return false;
        }
        std::string label;
        for (uint8_t n = 0; n < length; n++)
        {
            label += static_cast<char>(current.ReadU8());
        }
        name += name.empty() ? "" : ".";
        name += std::regex_replace(label, special, R"(\$&)");
    }
    return false;
}

/**
 * Convert an ASCII character to lower case. Unlike std::tolower, this does not depend on the
 * locale, since DNS names are only case-insensitive for ASCII (RFC 4343).
 *
 * @param c The character.
 * @return The lower case character.
 */
char
ToLowerAscii(char c)
{
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

/**
 * Check whether a name ends with an unescaped dot.
 *
 * @param name Name in presentation format.
 * @return True if the name has a trailing dot.
 */
bool
HasTrailingDot(const std::string& name)
{
    if (!name.ends_with('.'))
    {
        return false;
    }
    // The dot is escaped if it follows an odd number of backslashes. When the name is only
    // backslashes, find_last_not_of() returns npos, and npos + 1 wraps around to 0.
    const std::string_view beforeDot = std::string_view(name).substr(0, name.size() - 1);
    const size_t backslashes = beforeDot.size() - (beforeDot.find_last_not_of('\\') + 1);
    return backslashes % 2 == 0;
}

/**
 * Convert a raw TTL field to a Time. Values with the most significant bit set are treated as
 * zero (RFC 2181, section 8).
 *
 * @param ttl Raw TTL field.
 * @return The TTL.
 */
Time
TtlToTime(uint32_t ttl)
{
    return (ttl & TTL_SIGN_BIT) ? Time(0) : Seconds(ttl);
}

} // namespace

void
DnsResourceRecord::SetName(const std::string& name)
{
    m_name = name;
}

const std::string&
DnsResourceRecord::GetName() const
{
    return m_name;
}

void
DnsResourceRecord::SetType(uint16_t type)
{
    m_type = type;
}

uint16_t
DnsResourceRecord::GetType() const
{
    return m_type;
}

void
DnsResourceRecord::SetClass(uint16_t rclass)
{
    m_class = rclass;
}

uint16_t
DnsResourceRecord::GetClass() const
{
    return m_class;
}

void
DnsResourceRecord::SetAddress(const Address& address)
{
    m_address = address;
}

const Address&
DnsResourceRecord::GetAddress() const
{
    return m_address;
}

void
DnsResourceRecord::SetTarget(const std::string& target)
{
    m_target = target;
}

const std::string&
DnsResourceRecord::GetTarget() const
{
    return m_target;
}

void
DnsResourceRecord::SetSoa(const DnsResourceRecord::Soa& soa)
{
    m_soa = soa;
}

const DnsResourceRecord::Soa&
DnsResourceRecord::GetSoa() const
{
    return m_soa;
}

void
DnsResourceRecord::SetRdata(const std::vector<uint8_t>& rdata)
{
    m_rdata = rdata;
}

const std::vector<uint8_t>&
DnsResourceRecord::GetRdata() const
{
    return m_rdata;
}

void
DnsResourceRecord::SetTtl(uint32_t ttl)
{
    m_ttl = ttl;
}

uint32_t
DnsResourceRecord::GetRawTtl() const
{
    return m_ttl;
}

void
DnsResourceRecord::SetExtendedRcode(uint8_t extendedRcode)
{
    m_ttl = (m_ttl & ~OPT_RCODE_MASK) | static_cast<uint32_t>(extendedRcode) << OPT_RCODE_SHIFT;
}

uint8_t
DnsResourceRecord::GetExtendedRcode() const
{
    return m_ttl >> OPT_RCODE_SHIFT;
}

uint8_t
DnsResourceRecord::GetEdnsVersion() const
{
    return (m_ttl >> OPT_VERSION_SHIFT) & 0xff;
}

uint32_t
DnsResourceRecord::GetSerializedSize() const
{
    return NameSize(Labels(m_name)) + RECORD_FIXED_SIZE + GetRdataSize();
}

Time
DnsResourceRecord::GetTtl() const
{
    return TtlToTime(m_ttl);
}

bool
DnsResourceRecord::Matches(uint16_t type, const std::string& owner) const
{
    return m_type == type && m_class == DnsHeader::CLASS_IN &&
           DnsHeader::NormalizeName(m_name) == owner;
}

uint32_t
DnsResourceRecord::GetRdataSize() const
{
    switch (m_type)
    {
    case DnsHeader::TYPE_A:
        return Ipv4Address::ADDRESS_LENGTH;
    case DnsHeader::TYPE_AAAA:
        return Ipv6Address::ADDRESS_LENGTH;
    case DnsHeader::TYPE_CNAME:
    case DnsHeader::TYPE_NS:
        return NameSize(Labels(m_target));
    case DnsHeader::TYPE_SOA:
        return NameSize(Labels(m_soa.mname)) + NameSize(Labels(m_soa.rname)) + SOA_FIXED_SIZE;
    default:
        return m_rdata.size();
    }
}

void
DnsResourceRecord::Serialize(Buffer::Iterator& i) const
{
    WriteName(i, Labels(m_name));
    i.WriteHtonU16(m_type);
    i.WriteHtonU16(m_class);
    i.WriteHtonU32(m_ttl);
    i.WriteHtonU16(GetRdataSize());
    switch (m_type)
    {
    case DnsHeader::TYPE_A: {
        uint8_t buffer[Ipv4Address::ADDRESS_LENGTH];
        Ipv4Address::ConvertFrom(m_address).Serialize(buffer);
        i.Write(buffer, sizeof(buffer));
        break;
    }
    case DnsHeader::TYPE_AAAA: {
        uint8_t buffer[Ipv6Address::ADDRESS_LENGTH];
        Ipv6Address::ConvertFrom(m_address).Serialize(buffer);
        i.Write(buffer, sizeof(buffer));
        break;
    }
    case DnsHeader::TYPE_CNAME:
    case DnsHeader::TYPE_NS:
        WriteName(i, Labels(m_target));
        break;
    case DnsHeader::TYPE_SOA:
        WriteName(i, Labels(m_soa.mname));
        WriteName(i, Labels(m_soa.rname));
        i.WriteHtonU32(m_soa.serial);
        i.WriteHtonU32(m_soa.refresh);
        i.WriteHtonU32(m_soa.retry);
        i.WriteHtonU32(m_soa.expire);
        i.WriteHtonU32(m_soa.minimum);
        break;
    default:
        i.Write(m_rdata.data(), m_rdata.size());
        break;
    }
}

bool
DnsResourceRecord::Deserialize(Buffer::Iterator& i, Buffer::Iterator start)
{
    if (!ReadName(i, start, m_name) || i.GetRemainingSize() < RECORD_FIXED_SIZE)
    {
        return false;
    }
    m_type = i.ReadNtohU16();
    m_class = i.ReadNtohU16();
    m_ttl = i.ReadNtohU32();
    const uint16_t length = i.ReadNtohU16();
    if (i.GetRemainingSize() < length)
    {
        return false;
    }
    Buffer::Iterator end = i;
    end.Next(length);
    switch (m_type)
    {
    case DnsHeader::TYPE_A: {
        uint8_t buffer[Ipv4Address::ADDRESS_LENGTH];
        if (length != sizeof(buffer))
        {
            return false;
        }
        i.Read(buffer, sizeof(buffer));
        m_address = Ipv4Address::Deserialize(buffer);
        break;
    }
    case DnsHeader::TYPE_AAAA: {
        uint8_t buffer[Ipv6Address::ADDRESS_LENGTH];
        if (length != sizeof(buffer))
        {
            return false;
        }
        i.Read(buffer, sizeof(buffer));
        m_address = Ipv6Address::Deserialize(buffer);
        break;
    }
    case DnsHeader::TYPE_CNAME:
    case DnsHeader::TYPE_NS:
        if (!ReadName(i, start, m_target) || i.GetDistanceFrom(end) != 0)
        {
            return false;
        }
        break;
    case DnsHeader::TYPE_SOA:
        if (!ReadName(i, start, m_soa.mname) || !ReadName(i, start, m_soa.rname) ||
            i.GetDistanceFrom(end) != SOA_FIXED_SIZE)
        {
            return false;
        }
        m_soa.serial = i.ReadNtohU32();
        m_soa.refresh = i.ReadNtohU32();
        m_soa.retry = i.ReadNtohU32();
        m_soa.expire = i.ReadNtohU32();
        m_soa.minimum = i.ReadNtohU32();
        break;
    default:
        m_rdata.resize(length);
        i.Read(m_rdata.data(), length);
        break;
    }
    return true;
}

Time
DnsResourceRecord::Soa::GetMinimumTtl() const
{
    return TtlToTime(minimum);
}

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
        size += NameSize(Labels(m_questionName)) + QUESTION_FIXED_SIZE;
    }
    for (const auto* section : {&m_answers, &m_authorities, &m_additionalRecords})
    {
        for (const auto& record : *section)
        {
            size += record.GetSerializedSize();
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
            record.Serialize(i);
        }
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
        if (!ReadName(i, start, m_questionName) || i.GetRemainingSize() < QUESTION_FIXED_SIZE)
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
            if (!record.Deserialize(i, start))
            {
                return 0;
            }
            // Only one OPT record is allowed, in the additional section, owned by the root
            if (record.GetType() == TYPE_OPT &&
                (section != &m_additionalRecords || !record.GetName().empty() || GetOptRecord()))
            {
                return 0;
            }
            section->push_back(std::move(record));
        }
    }
    return i.GetDistanceFrom(start);
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
    for (auto& record : m_additionalRecords)
    {
        if (record.GetType() == TYPE_OPT)
        {
            record.SetExtendedRcode(rcode >> RCODE_BITS);
            return;
        }
    }
    NS_ASSERT_MSG(rcode >> RCODE_BITS == 0, "An extended RCODE requires an OPT record");
}

uint16_t
DnsHeader::GetRcode() const
{
    uint16_t rcode = m_flags & RCODE_MASK;
    if (auto opt = GetOptRecord())
    {
        rcode |= static_cast<uint16_t>(opt->GetExtendedRcode() << RCODE_BITS);
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
    opt.SetType(TYPE_OPT);
    opt.SetClass(udpPayloadSize);
    m_additionalRecords.push_back(opt);
}

std::optional<DnsResourceRecord>
DnsHeader::GetOptRecord() const
{
    for (const auto& record : m_additionalRecords)
    {
        if (record.GetType() == TYPE_OPT)
        {
            return record;
        }
    }
    return std::nullopt;
}

std::optional<std::vector<std::string>>
DnsHeader::SplitName(const std::string& name)
{
    // A label is a run of unescaped characters other than dots, or escaped characters
    static const std::string LABEL{R"((?:[^.\\]|\\[\s\S])+)"};
    static const std::regex label(LABEL);
    static const std::regex validName(LABEL + R"((?:\.)" + LABEL + ")*");
    static const std::regex escape(R"(\\([\s\S]))");

    std::vector<std::string> labels;
    if (name.empty())
    {
        return labels;
    }
    if (!std::regex_match(name, validName))
    {
        return std::nullopt;
    }
    for (std::sregex_iterator it(name.begin(), name.end(), label), end; it != end; ++it)
    {
        labels.push_back(std::regex_replace(it->str(), escape, "$1"));
        if (labels.back().size() > MAX_LABEL_SIZE)
        {
            return std::nullopt;
        }
    }
    if (NameSize(labels) > MAX_NAME_SIZE)
    {
        return std::nullopt;
    }
    return labels;
}

std::string
DnsHeader::NormalizeName(const std::string& name)
{
    std::string normalized = name;
    if (HasTrailingDot(normalized))
    {
        normalized.pop_back();
    }
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ToLowerAscii);
    return normalized;
}

bool
DnsHeader::IsInZone(const std::string& name, const std::string& zone)
{
    const auto nameLabels = SplitName(name);
    const auto zoneLabels = SplitName(zone);
    if (!nameLabels || !zoneLabels || zoneLabels->size() > nameLabels->size())
    {
        return false;
    }
    return std::equal(zoneLabels->rbegin(), zoneLabels->rend(), nameLabels->rbegin());
}

} // namespace ns3
