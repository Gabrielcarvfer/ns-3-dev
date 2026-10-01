/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "dns-resolver.h"

#include "ns3/boolean.h"
#include "ns3/inet-socket-address.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/ipv4-address.h"
#include "ns3/ipv6-address.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/packet.h"
#include "ns3/random-variable-stream.h"
#include "ns3/simulator.h"
#include "ns3/socket.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/uinteger.h"

#include <algorithm>
#include <limits>
#include <span>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("DnsResolver");

NS_OBJECT_ENSURE_REGISTERED(DnsResolver);

namespace
{

constexpr uint16_t DNS_CLASS_IN{1};                ///< Internet class
constexpr uint16_t DNS_FLAG_QR{0x8000};            ///< the message is a response
constexpr uint16_t DNS_FLAG_TC{0x0200};            ///< the message is truncated
constexpr uint16_t DNS_FLAG_RD{0x0100};            ///< recursion desired
constexpr uint16_t DNS_FLAG_RA{0x0080};            ///< recursion available
constexpr size_t DNS_HEADER_SIZE{12};              ///< size of the DNS header
constexpr size_t DNS_MAX_NAME_SIZE{255};           ///< maximum size of an encoded name
constexpr size_t DNS_MAX_LABEL_SIZE{63};           ///< maximum size of a label
constexpr size_t DNS_MAX_QUERIES{0x10000};         ///< number of query identifiers
constexpr uint32_t DNS_MAX_POINTERS{64};           ///< maximum compression pointers in a name
constexpr uint32_t DNS_MAX_CNAME_CHAIN{16};        ///< maximum length of a CNAME chain
constexpr uint32_t DNS_MAX_ATTEMPTS_PER_SERVER{3}; ///< maximum attempts to a server (RFC 9520)
constexpr uint16_t EDNS_MIN_UDP_PAYLOAD_SIZE{512}; ///< minimum EDNS(0) UDP payload size
constexpr uint16_t DYNAMIC_PORT_MIN{49152};        ///< first port of the dynamic range
constexpr uint32_t SOURCE_PORT_TRIES{16};          ///< random source ports tried for a query
const Time MAX_FAILURE_TTL{Minutes(5)};            ///< maximum caching of a failure (RFC 9520)

/**
 * Read a 16-bit big-endian value.
 * @param message the message
 * @param offset the offset of the value
 * @return the value
 */
uint16_t
ReadU16(const std::vector<uint8_t>& message, size_t offset)
{
    return static_cast<uint16_t>(message[offset] << 8 | message[offset + 1]);
}

/**
 * Read a 32-bit big-endian value.
 * @param message the message
 * @param offset the offset of the value
 * @return the value
 */
uint32_t
ReadU32(const std::vector<uint8_t>& message, size_t offset)
{
    return static_cast<uint32_t>(ReadU16(message, offset)) << 16 | ReadU16(message, offset + 2);
}

/**
 * Interpret a TTL, which is 0 if its most significant bit is set (RFC 2181, section 8).
 * @param ttl the TTL, as transmitted
 * @return the TTL, in seconds
 */
int64_t
TtlValue(uint32_t ttl)
{
    return (ttl & 0x80000000) ? 0 : ttl;
}

/**
 * @param a a TTL, or -1 if unknown
 * @param b another TTL, or -1 if unknown
 * @return the minimum of the known TTLs, or -1 if none is known
 */
int64_t
MinTtl(int64_t a, int64_t b)
{
    return a < 0 ? b : (b < 0 ? a : std::min(a, b));
}

/**
 * Convert an ASCII character to lower case; DNS names are case-insensitive for ASCII only
 * (RFC 4343).
 * @param c the character
 * @return the character in lower case
 */
char
ToLowerAscii(char c)
{
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

/**
 * @param name a name, with escaped dots and backslashes
 * @param position a position in the name
 * @return whether the character at the position is escaped
 */
bool
IsEscaped(const std::string& name, size_t position)
{
    size_t backslashes = 0;
    while (position > backslashes && name[position - backslashes - 1] == '\\')
    {
        backslashes++;
    }
    return backslashes % 2 == 1;
}

/**
 * Normalize a name: lower case, without the final dot.
 * @param name the name, with escaped dots and backslashes
 * @return the normalized name
 */
std::string
Normalize(const std::string& name)
{
    std::string normalized = name;
    if (!normalized.empty() && normalized.back() == '.' &&
        !IsEscaped(normalized, normalized.size() - 1))
    {
        normalized.pop_back();
    }
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ToLowerAscii);
    return normalized;
}

/**
 * Split a name into its labels.
 * @param name the name, with escaped dots and backslashes, without final dot (empty for the
 * root)
 * @param labels the labels
 * @return true if the labels are valid (not empty, at most 63 octets, at most 255 octets in
 * total once encoded)
 */
bool
SplitLabels(const std::string& name, std::vector<std::string>& labels)
{
    labels.clear();
    if (name.empty())
    {
        return true;
    }
    std::string label;
    size_t size = 1;
    for (size_t i = 0; i <= name.size(); i++)
    {
        if (i == name.size() || name[i] == '.')
        {
            if (label.empty() || label.size() > DNS_MAX_LABEL_SIZE)
            {
                return false;
            }
            size += label.size() + 1;
            labels.push_back(label);
            label.clear();
            continue;
        }
        if (name[i] == '\\')
        {
            if (++i == name.size())
            {
                return false;
            }
        }
        label += name[i];
    }
    return size <= DNS_MAX_NAME_SIZE;
}

/**
 * Encode a query.
 * @param id the identifier of the query
 * @param labels the labels of the name
 * @param type the type of the records
 * @param ednsUdpPayloadSize the EDNS(0) UDP payload size to advertise, or 0 for no EDNS(0)
 * @return the query
 */
std::vector<uint8_t>
BuildQuery(uint16_t id,
           const std::vector<std::string>& labels,
           uint16_t type,
           uint16_t ednsUdpPayloadSize)
{
    std::vector<uint8_t> message{static_cast<uint8_t>(id >> 8),
                                 static_cast<uint8_t>(id),
                                 static_cast<uint8_t>(DNS_FLAG_RD >> 8),
                                 static_cast<uint8_t>(DNS_FLAG_RD),
                                 0,
                                 1, // one question
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 static_cast<uint8_t>(ednsUdpPayloadSize > 0 ? 1 : 0)};
    for (const auto& label : labels)
    {
        message.push_back(static_cast<uint8_t>(label.size()));
        message.insert(message.end(), label.begin(), label.end());
    }
    message.push_back(0);
    message.insert(message.end(),
                   {static_cast<uint8_t>(type >> 8), static_cast<uint8_t>(type), 0, DNS_CLASS_IN});
    if (ednsUdpPayloadSize > 0)
    {
        // values below 512 are treated as 512 (RFC 6891, section 6.2.5)
        ednsUdpPayloadSize = std::max(ednsUdpPayloadSize, EDNS_MIN_UDP_PAYLOAD_SIZE);
        // OPT pseudo-record: root name, type, UDP payload size, extended RCODE, version 0 and
        // flags, no options
        message.insert(message.end(),
                       {0,
                        0,
                        DnsResolver::TYPE_OPT,
                        static_cast<uint8_t>(ednsUdpPayloadSize >> 8),
                        static_cast<uint8_t>(ednsUdpPayloadSize),
                        0,
                        0,
                        0,
                        0,
                        0,
                        0});
    }
    return message;
}

/**
 * Parse an IPv4 address in dotted-decimal notation, possibly with a final dot.
 * @param name the text
 * @param address the address
 * @return true if the text is an IPv4 address
 */
bool
ParseIpv4Literal(const std::string& name, Ipv4Address& address)
{
    const std::string text =
        (!name.empty() && name.back() == '.') ? name.substr(0, name.size() - 1) : name;
    uint32_t value = 0;
    size_t start = 0;
    for (int part = 0; part < 4; part++)
    {
        size_t end = text.find('.', start);
        end = (part == 3) ? text.size() : end;
        // without leading zeros, which some parsers read as octal
        if (end == std::string::npos || end == start || end - start > 3 ||
            (end - start > 1 && text[start] == '0'))
        {
            return false;
        }
        uint32_t byte = 0;
        for (size_t i = start; i < end; i++)
        {
            if (text[i] < '0' || text[i] > '9')
            {
                return false;
            }
            byte = byte * 10 + (text[i] - '0');
        }
        if (byte > 255)
        {
            return false;
        }
        value = value << 8 | byte;
        start = end + 1;
    }
    address = Ipv4Address(value);
    return true;
}

/**
 * Read a domain name, which may be compressed (RFC 1035, section 4.1.4).
 *
 * Each compression pointer must point strictly before the previous one, which prevents loops.
 * The dots and backslashes inside the labels are escaped with a backslash.
 *
 * @param message the message
 * @param offset the offset of the name
 * @param name the name, in lower case, with dot-separated labels (empty for the root)
 * @return the offset following the name, or 0 if the name is malformed
 */
size_t
ReadName(const std::vector<uint8_t>& message, size_t offset, std::string& name)
{
    name.clear();
    size_t next = 0;
    size_t size = 0;
    size_t limit = std::numeric_limits<size_t>::max();
    uint32_t pointers = 0;
    while (offset < message.size())
    {
        const uint8_t length = message[offset];
        if ((length & 0xc0) == 0xc0)
        {
            if (offset + 1 >= message.size() || ++pointers > DNS_MAX_POINTERS)
            {
                return 0;
            }
            const size_t target = static_cast<size_t>(length & 0x3f) << 8 | message[offset + 1];
            if (target >= std::min(limit, offset))
            {
                return 0;
            }
            if (next == 0)
            {
                next = offset + 2;
            }
            limit = target;
            offset = target;
            continue;
        }
        if ((length & 0xc0) != 0)
        {
            return 0;
        }
        size += length + 1;
        if (size > DNS_MAX_NAME_SIZE)
        {
            return 0;
        }
        if (length == 0)
        {
            return next != 0 ? next : offset + 1;
        }
        if (offset + 1 + length > message.size())
        {
            return 0;
        }
        if (!name.empty())
        {
            name += '.';
        }
        for (size_t i = 0; i < length; i++)
        {
            const char c = ToLowerAscii(static_cast<char>(message[offset + 1 + i]));
            if (c == '.' || c == '\\')
            {
                name += '\\';
            }
            name += c;
        }
        offset += length + 1;
    }
    return 0;
}

/**
 * @param name a normalized name
 * @param zone a normalized zone name
 * @return whether the name is the zone or one of its subdomains, comparing their labels
 */
bool
IsInZone(const std::string& name, const std::string& zone)
{
    std::vector<std::string> nameLabels;
    std::vector<std::string> zoneLabels;
    if (!SplitLabels(name, nameLabels) || !SplitLabels(zone, zoneLabels) ||
        zoneLabels.size() > nameLabels.size())
    {
        return false;
    }
    return std::equal(zoneLabels.rbegin(), zoneLabels.rend(), nameLabels.rbegin());
}

/// A resource record of a message
struct Record
{
    std::string owner; ///< the owner name, normalized
    uint16_t type;     ///< the type
    uint16_t rclass;   ///< the class
    uint32_t rawTtl;   ///< the TTL field, as transmitted
    size_t rdata;      ///< the offset of the data
    uint16_t length;   ///< the length of the data
};

} // namespace

TypeId
DnsResolver::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::DnsResolver")
            .SetParent<Object>()
            .SetGroupName("InternetApps")
            .AddConstructor<DnsResolver>()
            .AddAttribute("Port",
                          "The port of the DNS servers",
                          UintegerValue(53),
                          MakeUintegerAccessor(&DnsResolver::m_port),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("Timeout",
                          "The time to wait for a response before retransmitting a query, during "
                          "the first round of the servers; it doubles after each round",
                          TimeValue(Seconds(2)),
                          MakeTimeAccessor(&DnsResolver::m_timeout),
                          MakeTimeChecker(NanoSeconds(1)))
            .AddAttribute("Retransmissions",
                          "The maximum number of retransmissions of a query, each one to the "
                          "next server, with at most three attempts to each server",
                          UintegerValue(2),
                          MakeUintegerAccessor(&DnsResolver::m_retransmissions),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("CacheEnabled",
                          "Whether to cache the answers for their TTL",
                          BooleanValue(true),
                          MakeBooleanAccessor(&DnsResolver::m_cacheEnabled),
                          MakeBooleanChecker())
            .AddAttribute("MaxTtl",
                          "The maximum time an answer with addresses is cached",
                          TimeValue(Days(1)),
                          MakeTimeAccessor(&DnsResolver::m_maxTtl),
                          MakeTimeChecker(Time(0)))
            .AddAttribute("MaxNegativeTtl",
                          "The maximum time an answer without address is cached (RFC 2308, "
                          "section 5)",
                          TimeValue(Hours(3)),
                          MakeTimeAccessor(&DnsResolver::m_maxNegativeTtl),
                          MakeTimeChecker(Time(0)))
            .AddAttribute("FailureTtl",
                          "The time a resolution failure is cached (RFC 9520, section 3.2)",
                          TimeValue(Seconds(5)),
                          MakeTimeAccessor(&DnsResolver::m_failureTtl),
                          MakeTimeChecker(Seconds(1), Minutes(5)))
            .AddAttribute("MaxCacheEntries",
                          "The maximum number of cached answers and failures; the answers "
                          "closest to their expiry are removed first",
                          UintegerValue(10000),
                          MakeUintegerAccessor(&DnsResolver::m_maxCacheEntries),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("EdnsUdpPayloadSize",
                          "The EDNS(0) UDP payload size advertised in the queries (RFC 6891), at "
                          "least 512, or 0 not to use EDNS(0); larger responses are truncated by "
                          "the servers, and requested again over TCP",
                          UintegerValue(1232),
                          MakeUintegerAccessor(&DnsResolver::m_ednsUdpPayloadSize),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("NoEdnsTime",
                          "The time during which a server that rejected EDNS(0) is sent the new "
                          "queries without it (RFC 6891, section 6.2.2)",
                          TimeValue(Minutes(10)),
                          MakeTimeAccessor(&DnsResolver::m_noEdnsTime),
                          MakeTimeChecker(Time(0)))
            .AddTraceSource("Resolved",
                            "The result of a resolution: the host name and its addresses, "
                            "empty if the resolution failed",
                            MakeTraceSourceAccessor(&DnsResolver::m_resolvedTrace),
                            "ns3::DnsResolver::ResolvedTracedCallback");
    return tid;
}

DnsResolver::DnsResolver()
    : m_id(CreateObject<UniformRandomVariable>()),
      m_sourcePort(CreateObject<UniformRandomVariable>())
{
    NS_LOG_FUNCTION(this);
}

DnsResolver::~DnsResolver()
{
    NS_LOG_FUNCTION(this);
}

void
DnsResolver::DoDispose()
{
    NS_LOG_FUNCTION(this);
    // the pending resolutions are dropped without reporting, since the resolver is destroyed
    for (auto& [id, query] : m_queries)
    {
        query.timeout.Cancel();
        CloseTcp(query);
        CloseUdp(query);
    }
    m_queries.clear();
    for (auto& [id, lookup] : m_lookups)
    {
        lookup.completion.Cancel();
    }
    m_lookups.clear();
    for (auto socket : {m_udpSocket, m_udp6Socket})
    {
        if (socket)
        {
            socket->SetRecvCallback(MakeNullCallback<void, Ptr<Socket>>());
            socket->Close();
        }
    }
    m_udpSocket = nullptr;
    m_udp6Socket = nullptr;
    Object::DoDispose();
}

void
DnsResolver::SetServers(const std::vector<Address>& servers)
{
    NS_LOG_FUNCTION(this);
    for (const auto& server : servers)
    {
        NS_ABORT_MSG_UNLESS(Ipv4Address::IsMatchingType(server) ||
                                Ipv6Address::IsMatchingType(server),
                            "The DNS servers must be IPv4 or IPv6 addresses");
    }
    m_servers = servers;
    // the failures may not happen with the new servers
    std::erase_if(m_cache, [](const auto& entry) { return entry.second.failure; });
    m_failures.clear();
    std::vector<uint16_t> ids;
    for (auto& [id, query] : m_queries)
    {
        ids.push_back(id);
        if (servers.empty())
        {
            continue;
        }
        // the next server is the one following the current server or, if it is not a server
        // any more, the one that followed it
        auto it = std::find(servers.begin(), servers.end(), query.current);
        query.server = it != servers.end()
                           ? static_cast<size_t>(it - servers.begin())
                           : (query.server % servers.size() + servers.size() - 1) % servers.size();
    }
    if (servers.empty())
    {
        // the queries fail when their current attempt is abandoned, unless servers are added
        for (auto id : ids)
        {
            AbandonAttempt(id);
        }
    }
}

void
DnsResolver::AddServer(const Address& server)
{
    NS_LOG_FUNCTION(this << server);
    auto servers = m_servers;
    servers.push_back(server);
    SetServers(servers);
}

std::vector<Address>
DnsResolver::GetServers() const
{
    return m_servers;
}

void
DnsResolver::FlushCache()
{
    NS_LOG_FUNCTION(this);
    m_cache.clear();
    m_failures.clear();
}

int64_t
DnsResolver::AssignStreams(int64_t stream)
{
    NS_LOG_FUNCTION(this << stream);
    m_id->SetStream(stream);
    m_sourcePort->SetStream(stream + 1);
    return 2;
}

bool
DnsResolver::IsValidHostName(const std::string& name)
{
    std::string host = name;
    if (!host.empty() && host.back() == '.')
    {
        host.pop_back();
    }
    if (host.empty() || host.size() > 253)
    {
        return false;
    }
    size_t start = 0;
    bool numeric = false;
    while (start <= host.size())
    {
        size_t end = host.find('.', start);
        end = end == std::string::npos ? host.size() : end;
        const size_t length = end - start;
        if (length == 0 || length > DNS_MAX_LABEL_SIZE || host[start] == '-' ||
            host[end - 1] == '-')
        {
            return false;
        }
        numeric = true;
        for (size_t i = start; i < end; i++)
        {
            const char c = host[i];
            const bool digit = c >= '0' && c <= '9';
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || digit || c == '-'))
            {
                return false;
            }
            numeric = numeric && digit;
        }
        start = end + 1;
    }
    // the last label is not all-numeric (RFC 1123, section 2.1)
    return !numeric;
}

std::vector<uint8_t>
DnsResolver::EncodeQuery(uint16_t id,
                         const std::string& name,
                         RecordType type,
                         uint16_t ednsUdpPayloadSize)
{
    std::vector<std::string> labels;
    if (!IsValidHostName(name) || !SplitLabels(Normalize(name), labels))
    {
        return {};
    }
    // the case of the name is preserved in the question
    const std::string host = (name.back() == '.') ? name.substr(0, name.size() - 1) : name;
    SplitLabels(host, labels);
    return BuildQuery(id, labels, type, ednsUdpPayloadSize);
}

DnsResolver::Response
DnsResolver::DecodeResponse(const std::vector<uint8_t>& message,
                            const std::string& name,
                            RecordType type)
{
    Response response;
    if (message.size() < DNS_HEADER_SIZE)
    {
        return response;
    }
    response.id = ReadU16(message, 0);
    const uint16_t flags = ReadU16(message, 2);
    if (!(flags & DNS_FLAG_QR) || ((flags >> 11) & 0xf) != 0)
    {
        // not a response to a standard query
        return response;
    }
    response.truncated = flags & DNS_FLAG_TC;
    response.recursionAvailable = flags & DNS_FLAG_RA;
    response.rcode = flags & 0x000f;
    const uint16_t questions = ReadU16(message, 4);
    const uint16_t answers = ReadU16(message, 6);
    const uint16_t authorities = ReadU16(message, 8);
    const uint16_t additionals = ReadU16(message, 10);
    response.hasAnswers = answers > 0;

    // The question must be the one of the query (RFC 5452, section 9.1), except for the
    // FORMERR and NOTIMP responses of servers not supporting EDNS(0) (RFC 6891, section 7)
    size_t offset = DNS_HEADER_SIZE;
    const std::string qname = Normalize(name);
    response.canonicalName = qname;
    if (questions == 1)
    {
        std::string questionName;
        offset = ReadName(message, offset, questionName);
        if (offset == 0 || offset + 4 > message.size() || questionName != qname ||
            ReadU16(message, offset) != type || ReadU16(message, offset + 2) != DNS_CLASS_IN)
        {
            return response;
        }
        offset += 4;
        response.hasQuestion = true;
    }
    else if (questions != 0 || answers != 0 ||
             (response.rcode != RCODE_FORMERR && response.rcode != RCODE_NOTIMP))
    {
        return response;
    }
    response.valid = true;

    // the records; a malformed record section is a failure of the server, unless the response
    // is truncated
    std::vector<Record> records;
    for (uint32_t i = 0; i < static_cast<uint32_t>(answers) + authorities + additionals; i++)
    {
        Record record;
        offset = ReadName(message, offset, record.owner);
        if (offset == 0 || offset + 10 > message.size())
        {
            response.rcode = response.truncated ? response.rcode : RCODE_SERVFAIL;
            return response;
        }
        record.type = ReadU16(message, offset);
        record.rclass = ReadU16(message, offset + 2);
        record.rawTtl = ReadU32(message, offset + 4);
        record.length = ReadU16(message, offset + 8);
        record.rdata = offset + 10;
        offset = record.rdata + record.length;
        if (offset > message.size())
        {
            response.rcode = response.truncated ? response.rcode : RCODE_SERVFAIL;
            return response;
        }
        // EDNS(0): a single OPT record, in the additional section, with the root as owner,
        // extending the RCODE (RFC 6891, section 6.1.1)
        if (record.type == TYPE_OPT)
        {
            if (i < static_cast<uint32_t>(answers) + authorities || response.hasOpt ||
                !record.owner.empty())
            {
                response.rcode = RCODE_SERVFAIL;
                return response;
            }
            response.hasOpt = true;
            response.rcode |= static_cast<uint16_t>((record.rawTtl >> 24) << 4);
            // a response of another EDNS version is an error, unless BADVERS (RFC 6891,
            // section 6.1.3)
            if (((record.rawTtl >> 16) & 0xff) != 0 && response.rcode != RCODE_BADVERS)
            {
                response.rcode = RCODE_SERVFAIL;
                return response;
            }
        }
        records.push_back(record);
    }
    if (response.truncated)
    {
        return response;
    }
    const auto answerSection = std::span(records).subspan(0, answers);
    const auto authoritySection = std::span(records).subspan(answers, authorities);

    // the CNAME chain from the queried name (RFC 1034, section 3.6.2)
    std::string target = qname;
    for (uint32_t links = 0;; links++)
    {
        auto cname = std::find_if(answerSection.begin(), answerSection.end(), [&](const auto& r) {
            return r.type == TYPE_CNAME && r.rclass == DNS_CLASS_IN && r.owner == target;
        });
        if (cname == answerSection.end())
        {
            break;
        }
        // a name has a single alias (RFC 2181, section 10.1), possibly repeated
        std::string alias;
        bool consistent = links < DNS_MAX_CNAME_CHAIN;
        for (const auto& record : answerSection)
        {
            std::string other;
            if (consistent && record.type == TYPE_CNAME && record.rclass == DNS_CLASS_IN &&
                record.owner == target)
            {
                consistent =
                    ReadName(message, record.rdata, other) == record.rdata + record.length &&
                    (alias.empty() || other == alias);
                alias = other;
                response.chainTtl = MinTtl(response.chainTtl, TtlValue(record.rawTtl));
            }
        }
        if (!consistent)
        {
            // a loop, or a malformed or ambiguous alias: the server failed to resolve the name
            response.rcode = RCODE_SERVFAIL;
            return response;
        }
        target = alias;
    }
    response.canonicalName = target;

    // the addresses of the name at the end of the chain
    int64_t answerTtl = response.chainTtl;
    for (const auto& record : answerSection)
    {
        if (record.owner != target || record.type != type || record.rclass != DNS_CLASS_IN)
        {
            continue;
        }
        if (record.length != (type == TYPE_A ? 4 : 16))
        {
            // RFC 1035, section 3.4.1, and RFC 3596, section 2.2
            response.rcode = RCODE_SERVFAIL;
            response.addresses.clear();
            return response;
        }
        Address address;
        if (type == TYPE_A)
        {
            address = Ipv4Address(ReadU32(message, record.rdata));
        }
        else
        {
            uint8_t bytes[16];
            std::copy_n(message.begin() + record.rdata, 16, bytes);
            address = Ipv6Address(bytes);
        }
        // an RRset has no duplicate records (RFC 2181, section 5)
        if (std::find(response.addresses.begin(), response.addresses.end(), address) ==
            response.addresses.end())
        {
            response.addresses.push_back(address);
        }
        answerTtl = MinTtl(answerTtl, TtlValue(record.rawTtl));
    }
    if (!response.addresses.empty())
    {
        response.ttl = answerTtl;
        return response;
    }

    // Negative answer: the negative TTL is given by the SOA record of the zone of the name at the
    // end of the chain, as the minimum of its TTL and of its MINIMUM field (RFC 2308, section 5),
    // and is bounded by the TTL of the chain
    for (const auto& record : authoritySection)
    {
        response.hasNs = response.hasNs || record.type == TYPE_NS;
        if (response.hasSoa || record.type != TYPE_SOA || record.rclass != DNS_CLASS_IN ||
            !IsInZone(target, record.owner))
        {
            continue;
        }
        std::string mname;
        std::string rname;
        size_t field = ReadName(message, record.rdata, mname);
        field = field != 0 ? ReadName(message, field, rname) : 0;
        if (field == 0 || field + 20 != record.rdata + record.length)
        {
            continue;
        }
        response.hasSoa = true;
        response.ttl =
            MinTtl(std::min(TtlValue(record.rawTtl), TtlValue(ReadU32(message, field + 16))),
                   response.chainTtl);
    }
    return response;
}

void
DnsResolver::Resolve(const std::string& name, ResolveCallback callback, AddressFamily family)
{
    NS_LOG_FUNCTION(this << name << +family);
    const uint64_t lookup = m_nextLookup++;
    m_lookups[lookup] = {name, callback};

    std::vector<RecordType> types;
    Ipv4Address literal;
    if (ParseIpv4Literal(name, literal))
    {
        // an address is not resolved
        if (family != IPV6)
        {
            m_lookups[lookup].ipv4 = {literal};
        }
    }
    else if (!IsValidHostName(name))
    {
        NS_LOG_WARN("Invalid host name " << name);
    }
    else
    {
        if (family != IPV4)
        {
            types.push_back(TYPE_AAAA);
        }
        if (family != IPV6)
        {
            types.push_back(TYPE_A);
        }
    }
    const std::string normalized = Normalize(name);
    for (auto type : types)
    {
        auto it = m_cache.find({normalized, type});
        if (m_cacheEnabled && it != m_cache.end() && it->second.expiry > Simulator::Now())
        {
            NS_LOG_INFO("Cached " << (it->second.failure ? "failure" : "answer") << " for "
                                  << normalized << " (type " << type << ")");
            auto& addresses = type == TYPE_A ? m_lookups[lookup].ipv4 : m_lookups[lookup].ipv6;
            addresses = it->second.addresses;
            continue;
        }
        if (it != m_cache.end() && it->second.expiry <= Simulator::Now())
        {
            m_cache.erase(it);
        }
        if (m_servers.empty())
        {
            NS_LOG_WARN("No DNS server to resolve " << name);
            continue;
        }
        // a single query in progress for a question (RFC 5452, section 5)
        auto shared = std::find_if(m_queries.begin(), m_queries.end(), [&](const auto& entry) {
            return entry.second.name == normalized && entry.second.type == type;
        });
        if (shared != m_queries.end())
        {
            NS_LOG_INFO("Query for " << normalized << " (type " << type << ") in progress");
            shared->second.lookups.push_back(lookup);
            m_lookups[lookup].pendingQueries++;
            continue;
        }
        if (m_queries.size() >= DNS_MAX_QUERIES)
        {
            NS_LOG_WARN("No query identifier available to resolve " << name);
            continue;
        }
        m_lookups[lookup].pendingQueries++;
        StartQuery(lookup, normalized, type);
    }
    if (m_lookups[lookup].pendingQueries == 0)
    {
        // the result is reported after Resolve() returns, as when the servers are queried
        m_lookups[lookup].completion =
            Simulator::ScheduleNow(&DnsResolver::CompleteLookup, this, lookup);
    }
}

uint16_t
DnsResolver::NewQueryId()
{
    uint16_t id;
    do
    {
        id = static_cast<uint16_t>(m_id->GetInteger(0, 0xffff));
    } while (m_queries.contains(id));
    return id;
}

void
DnsResolver::StartQuery(uint64_t lookup, const std::string& name, RecordType type)
{
    NS_LOG_FUNCTION(this << lookup << name << type);
    const uint16_t id = NewQueryId();
    Query& query = m_queries[id];
    query.lookups = {lookup};
    query.name = name;
    query.question = name;
    query.type = type;
    SendUdp(id);
}

bool
DnsResolver::IsServer(const Address& server) const
{
    return std::find(m_servers.begin(), m_servers.end(), server) != m_servers.end();
}

std::vector<uint8_t>
DnsResolver::EncodeQuery(uint16_t id)
{
    Query& query = m_queries.at(id);
    query.currentEdns = m_ednsUdpPayloadSize > 0 && !query.noEdns;
    auto it = m_noEdnsServers.find(query.current);
    if (it != m_noEdnsServers.end())
    {
        if (it->second > Simulator::Now())
        {
            query.currentEdns = false;
        }
        else
        {
            m_noEdnsServers.erase(it);
        }
    }
    std::vector<std::string> labels;
    SplitLabels(query.question, labels);
    return BuildQuery(id, labels, query.type, query.currentEdns ? m_ednsUdpPayloadSize : 0);
}

Address
DnsResolver::GetSocketAddress(const Address& server) const
{
    if (Ipv4Address::IsMatchingType(server))
    {
        return InetSocketAddress(Ipv4Address::ConvertFrom(server), m_port);
    }
    return Inet6SocketAddress(Ipv6Address::ConvertFrom(server), m_port);
}

Ptr<Socket>
DnsResolver::GetUdpSocket(Query& query, const Address& server)
{
    const bool ipv4 = Ipv4Address::IsMatchingType(server);
    Ptr<Socket>& socket = ipv4 ? query.udpSocket : query.udp6Socket;
    if (socket)
    {
        return socket;
    }
    Ptr<Node> node = GetObject<Node>();
    NS_ABORT_MSG_IF(!node, "The DNS resolver is not aggregated to a node");
    // an unpredictable source port for each query (RFC 5452, section 9.2)
    socket = Socket::CreateSocket(node, UdpSocketFactory::GetTypeId());
    for (uint32_t i = 0; i < SOURCE_PORT_TRIES; i++)
    {
        const auto port = static_cast<uint16_t>(m_sourcePort->GetInteger(DYNAMIC_PORT_MIN, 65535));
        const int bound = ipv4 ? socket->Bind(InetSocketAddress(Ipv4Address::GetAny(), port))
                               : socket->Bind(Inet6SocketAddress(Ipv6Address::GetAny(), port));
        if (bound == 0)
        {
            socket->SetRecvCallback(MakeCallback(&DnsResolver::ReceiveUdp, this));
            return socket;
        }
    }
    // the dynamic range is busy: the socket shared by the queries
    NS_LOG_WARN("No free random source port");
    socket->Close();
    Ptr<Socket>& shared = ipv4 ? m_udpSocket : m_udp6Socket;
    if (!shared)
    {
        shared = Socket::CreateSocket(node, UdpSocketFactory::GetTypeId());
        NS_ABORT_MSG_IF((ipv4 ? shared->Bind() : shared->Bind6()) != 0,
                        "Cannot bind the UDP socket of the DNS resolver");
        shared->SetRecvCallback(MakeCallback(&DnsResolver::ReceiveUdp, this));
    }
    socket = shared;
    return socket;
}

void
DnsResolver::CloseUdp(Query& query)
{
    for (auto socket : {query.udpSocket, query.udp6Socket})
    {
        if (socket && socket != m_udpSocket && socket != m_udp6Socket)
        {
            socket->SetRecvCallback(MakeNullCallback<void, Ptr<Socket>>());
            socket->Close();
        }
    }
    query.udpSocket = nullptr;
    query.udp6Socket = nullptr;
}

Time
DnsResolver::GetAttemptTimeout(const Query& query) const
{
    // exponential backoff after each round of the servers (RFC 1123, section 6.1.3.3)
    const uint32_t round = query.retransmissions / std::max<size_t>(m_servers.size(), 1);
    return m_timeout * (static_cast<int64_t>(1) << std::min<uint32_t>(round, 10));
}

void
DnsResolver::SendUdp(uint16_t id, Address server)
{
    NS_LOG_FUNCTION(this << id << server);
    Query& query = m_queries.at(id);
    if (m_servers.empty())
    {
        // the servers were removed: the query fails when the attempt is abandoned
        AbandonAttempt(id);
        return;
    }
    if (server.IsInvalid())
    {
        server = m_servers[query.server];
    }
    query.current = server;
    query.attempts[server]++;
    NS_LOG_INFO("Query for " << query.question << " (type " << query.type << ") to " << server);
    if (std::find(query.queried.begin(), query.queried.end(), server) == query.queried.end())
    {
        query.queried.push_back(server);
    }
    auto message = EncodeQuery(id);
    if (GetUdpSocket(query, server)
            ->SendTo(Create<Packet>(message.data(), message.size()), 0, GetSocketAddress(server)) <
        0)
    {
        // e.g., no route to the server
        AbandonAttempt(id);
        return;
    }
    query.timeout =
        Simulator::Schedule(GetAttemptTimeout(query), &DnsResolver::HandleTimeout, this, id);
}

void
DnsResolver::AbandonAttempt(uint16_t id)
{
    NS_LOG_FUNCTION(this << id);
    Query& query = m_queries.at(id);
    query.timeout.Cancel();
    query.timeout = Simulator::ScheduleNow(&DnsResolver::HandleTimeout, this, id);
}

void
DnsResolver::HandleTimeout(uint16_t id)
{
    NS_LOG_FUNCTION(this << id);
    NS_LOG_INFO("No answer from " << m_queries.at(id).current);
    Retransmit(id);
}

void
DnsResolver::Retransmit(uint16_t id)
{
    NS_LOG_FUNCTION(this << id);
    Query& query = m_queries.at(id);
    query.timeout.Cancel();
    CloseTcp(query);
    if (m_servers.empty())
    {
        // not a failure of the servers: not cached
        CompleteQuery(id, {});
        return;
    }
    if (query.retransmissions < m_retransmissions)
    {
        // the next server that was not tried too many times (RFC 9520, section 3.1)
        for (size_t i = 1; i <= m_servers.size(); i++)
        {
            const size_t server = (query.server + i) % m_servers.size();
            if (query.attempts[m_servers[server]] < DNS_MAX_ATTEMPTS_PER_SERVER)
            {
                query.retransmissions++;
                query.server = server;
                SendUdp(id);
                return;
            }
        }
    }
    NS_LOG_INFO("No answer for " << query.question << " (type " << query.type << ")");
    FailQuery(id);
}

void
DnsResolver::QueryTarget(uint16_t id, const std::string& target, int64_t chainTtl)
{
    NS_LOG_FUNCTION(this << id << target);
    std::vector<std::string> labels;
    if (m_queries.at(id).restarts >= DNS_MAX_CNAME_CHAIN || !SplitLabels(target, labels) ||
        labels.empty())
    {
        FailQuery(id);
        return;
    }
    // a new query, with its own identifier, socket, servers and retransmissions; the identifier
    // of the previous query is released first, so that one is available
    Query previous = std::move(m_queries.at(id));
    m_queries.erase(id);
    previous.timeout.Cancel();
    CloseTcp(previous);
    CloseUdp(previous);
    NS_LOG_INFO("Querying " << target << ", alias of " << previous.name);
    const uint16_t newId = NewQueryId();
    Query& next = m_queries[newId];
    next.lookups = previous.lookups;
    next.name = previous.name;
    next.question = target;
    next.type = previous.type;
    next.restarts = previous.restarts + 1;
    next.chainTtl = MinTtl(previous.chainTtl, chainTtl);
    next.noEdns = previous.noEdns;
    if (!m_servers.empty() && !IsServer(previous.current))
    {
        // the server was removed: the next one
        next.server = (previous.server + 1) % m_servers.size();
        SendUdp(newId);
        return;
    }
    next.server = previous.server;
    SendUdp(newId, previous.current);
}

void
DnsResolver::ReceiveUdp(Ptr<Socket> socket)
{
    NS_LOG_FUNCTION(this << socket);
    Address from;
    while (Ptr<Packet> packet = socket->RecvFrom(from))
    {
        std::vector<uint8_t> message(packet->GetSize());
        packet->CopyData(message.data(), message.size());
        if (message.size() < 2)
        {
            continue;
        }
        const uint16_t id = ReadU16(message, 0);
        auto it = m_queries.find(id);
        if (it == m_queries.end() || it->second.tcpSocket ||
            (socket != it->second.udpSocket && socket != it->second.udp6Socket))
        {
            continue;
        }
        // a response may come from any of the servers the query was sent to
        const auto& queried = it->second.queried;
        auto server = std::find_if(queried.begin(), queried.end(), [&](const Address& s) {
            return GetSocketAddress(s) == from;
        });
        if (server == queried.end())
        {
            NS_LOG_WARN("Ignoring a DNS message from " << from);
            continue;
        }
        HandleResponse(id, message, false, *server);
    }
}

void
DnsResolver::HandleResponse(uint16_t id,
                            const std::vector<uint8_t>& message,
                            bool overTcp,
                            Address server)
{
    NS_LOG_FUNCTION(this << id << overTcp << server);
    auto it = m_queries.find(id);
    if (it == m_queries.end())
    {
        return;
    }
    Query& query = it->second;
    auto response = DecodeResponse(message, query.question, query.type);
    if (!response.valid || response.id != id)
    {
        NS_LOG_WARN("Ignoring an invalid DNS response from " << server);
        if (overTcp)
        {
            // the connection is closed: no other response can come
            Retransmit(id);
        }
        return;
    }
    // Only the answers are accepted from the servers the query is not sent to any more; the
    // errors of these servers do not affect the current attempt. A response without question
    // is only trusted as a rejection of EDNS(0).
    const bool current = server == query.current;
    const bool ednsRejected = current && query.currentEdns && !response.hasOpt &&
                              (response.rcode == RCODE_FORMERR || response.rcode == RCODE_NOTIMP);
    if (ednsRejected)
    {
        // the server does not support EDNS(0): retry without it (RFC 6891, section 7)
        NS_LOG_INFO("No EDNS(0) support in " << server);
        m_noEdnsServers[server] = Simulator::Now() + m_noEdnsTime;
        query.noEdns = true;
        if (!IsServer(server))
        {
            // the server was removed: the next one
            Retransmit(id);
            return;
        }
        query.timeout.Cancel();
        CloseTcp(query);
        SendUdp(id, server);
        return;
    }
    if (!response.hasQuestion)
    {
        return;
    }
    if (response.truncated)
    {
        if (current)
        {
            // a response over TCP is never truncated (RFC 7766, section 8)
            overTcp ? Retransmit(id) : SendTcp(id);
        }
        return;
    }
    switch (response.rcode)
    {
    case RCODE_NOERROR: {
        if (!response.addresses.empty() || response.hasSoa)
        {
            // an answer, or the absence of address of the requested type (NODATA)
            const auto addresses = response.addresses;
            const int64_t ttl = MinTtl(response.ttl, query.chainTtl);
            Cache(query.name, query.type, addresses, ttl < 0 ? Time(0) : Seconds(ttl));
            CompleteQuery(id, addresses);
        }
        else if (response.canonicalName != query.question)
        {
            // the CNAME chain ends without the data of its target: query the target
            // (RFC 1034, section 3.6.2)
            QueryTarget(id, response.canonicalName, response.chainTtl);
        }
        else if (!current)
        {
            // the other responses of a previous server do not affect the current attempt
        }
        else if (!response.hasAnswers && (response.hasNs || !response.recursionAvailable))
        {
            // a referral (RFC 2308, section 2.2): the server does not offer recursion
            NS_LOG_INFO("Referral from " << server);
            Retransmit(id);
        }
        else
        {
            CompleteQuery(id, {});
        }
        break;
    }
    case RCODE_NXDOMAIN: {
        // the name has no record of any type (RFC 8020), and the answer of the other type is
        // kept if it has addresses
        const auto name = query.name;
        const auto type = query.type;
        const int64_t ttl = response.hasSoa ? MinTtl(response.ttl, query.chainTtl) : -1;
        const Time negativeTtl = ttl < 0 ? Time(0) : Seconds(ttl);
        Cache(name, type, {}, negativeTtl);
        Cache(name, type == TYPE_A ? TYPE_AAAA : TYPE_A, {}, negativeTtl, false);
        std::vector<uint16_t> ids{id};
        for (const auto& [otherId, other] : m_queries)
        {
            if (otherId != id && other.name == name)
            {
                ids.push_back(otherId);
            }
        }
        for (auto queryId : ids)
        {
            CompleteQuery(queryId, {});
        }
        break;
    }
    default:
        // e.g., SERVFAIL or REFUSED: another server may answer
        if (current)
        {
            NS_LOG_INFO("Error " << response.rcode << " from " << server);
            Retransmit(id);
        }
        break;
    }
}

void
DnsResolver::SendTcp(uint16_t id)
{
    NS_LOG_FUNCTION(this << id);
    Query& query = m_queries.at(id);
    query.timeout.Cancel();
    CloseTcp(query);
    const Address server = query.current;
    query.tcpSocket = Socket::CreateSocket(GetObject<Node>(), TcpSocketFactory::GetTypeId());
    if ((Ipv4Address::IsMatchingType(server) ? query.tcpSocket->Bind()
                                             : query.tcpSocket->Bind6()) != 0)
    {
        AbandonAttempt(id);
        return;
    }
    using SocketCallback = Callback<void, Ptr<Socket>>;
    query.tcpSocket->SetConnectCallback(
        SocketCallback([this, id](Ptr<Socket> socket) {
            // over TCP, the message is prefixed with its length (RFC 1035, section 4.2.2)
            const auto message = EncodeQuery(id);
            std::vector<uint8_t> data{static_cast<uint8_t>(message.size() >> 8),
                                      static_cast<uint8_t>(message.size())};
            data.insert(data.end(), message.begin(), message.end());
            socket->Send(Create<Packet>(data.data(), data.size()));
        }),
        SocketCallback([this, id](Ptr<Socket> socket) { HandleTcpClose(id, socket); }));
    query.tcpSocket->SetRecvCallback(
        SocketCallback([this, id](Ptr<Socket> socket) { ReceiveTcp(id, socket); }));
    query.tcpSocket->SetCloseCallbacks(
        SocketCallback([this, id](Ptr<Socket> socket) { HandleTcpClose(id, socket); }),
        SocketCallback([this, id](Ptr<Socket> socket) { HandleTcpClose(id, socket); }));
    if (query.tcpSocket->Connect(GetSocketAddress(server)) != 0)
    {
        // e.g., no route to the server
        AbandonAttempt(id);
        return;
    }
    query.timeout =
        Simulator::Schedule(GetAttemptTimeout(query), &DnsResolver::HandleTimeout, this, id);
}

void
DnsResolver::ReceiveTcp(uint16_t id, Ptr<Socket> socket)
{
    NS_LOG_FUNCTION(this << id << socket);
    auto it = m_queries.find(id);
    if (it == m_queries.end() || it->second.tcpSocket != socket)
    {
        return;
    }
    Query& query = it->second;
    while (Ptr<Packet> packet = socket->Recv())
    {
        const size_t size = query.tcpData.size();
        query.tcpData.resize(size + packet->GetSize());
        packet->CopyData(query.tcpData.data() + size, packet->GetSize());
    }
    if (query.tcpData.size() >= 2 && query.tcpData.size() >= 2u + ReadU16(query.tcpData, 0))
    {
        std::vector<uint8_t> message(query.tcpData.begin() + 2,
                                     query.tcpData.begin() + 2 + ReadU16(query.tcpData, 0));
        const Address server = query.current;
        CloseTcp(query);
        HandleResponse(id, message, true, server);
    }
}

void
DnsResolver::HandleTcpClose(uint16_t id, Ptr<Socket> socket)
{
    NS_LOG_FUNCTION(this << id << socket);
    auto it = m_queries.find(id);
    if (it == m_queries.end() || it->second.tcpSocket != socket)
    {
        return;
    }
    // the connection was closed or failed before a complete response (RFC 7766, section 6.2)
    NS_LOG_INFO("TCP connection to " << it->second.current << " closed");
    Retransmit(id);
}

void
DnsResolver::CloseTcp(Query& query)
{
    if (query.tcpSocket)
    {
        auto null = MakeNullCallback<void, Ptr<Socket>>();
        query.tcpSocket->SetConnectCallback(null, null);
        query.tcpSocket->SetCloseCallbacks(null, null);
        query.tcpSocket->SetRecvCallback(null);
        query.tcpSocket->Close();
        query.tcpSocket = nullptr;
    }
    query.tcpData.clear();
}

void
DnsResolver::Cache(const std::string& name,
                   RecordType type,
                   const std::vector<Address>& addresses,
                   Time ttl,
                   bool overwrite,
                   bool failure)
{
    if (!failure)
    {
        m_failures.erase({name, type});
    }
    if (!m_cacheEnabled || ttl <= Time(0) || m_maxCacheEntries == 0)
    {
        return;
    }
    if (Simulator::Now() - m_lastPurge >= Minutes(1))
    {
        PurgeCache();
    }
    auto it = m_cache.find({name, type});
    if (!overwrite && it != m_cache.end() && it->second.expiry > Simulator::Now() &&
        !it->second.addresses.empty())
    {
        return;
    }
    if (it == m_cache.end() && m_cache.size() >= m_maxCacheEntries)
    {
        PurgeCache();
        if (m_cache.size() >= m_maxCacheEntries)
        {
            // the answer closest to its expiry is removed
            auto oldest = std::min_element(m_cache.begin(), m_cache.end(), [](auto& a, auto& b) {
                return a.second.expiry < b.second.expiry;
            });
            m_cache.erase(oldest);
        }
    }
    if (!failure)
    {
        ttl = std::min(ttl, addresses.empty() ? m_maxNegativeTtl : m_maxTtl);
    }
    m_cache[{name, type}] = {addresses, Simulator::Now() + ttl, failure};
}

void
DnsResolver::PurgeCache()
{
    NS_LOG_FUNCTION(this);
    m_lastPurge = Simulator::Now();
    std::erase_if(m_cache,
                  [](const auto& entry) { return entry.second.expiry <= Simulator::Now(); });
    std::erase_if(m_failures, [](const auto& entry) {
        return entry.second.expiry + MAX_FAILURE_TTL <= Simulator::Now();
    });
}

void
DnsResolver::FailQuery(uint16_t id)
{
    NS_LOG_FUNCTION(this << id);
    const Query& query = m_queries.at(id);
    // The failures are cached, to avoid repeating the queries, for a time doubled after each
    // consecutive failure (RFC 9520, section 3.2). The failures are consecutive if the previous
    // one expired less than the maximum time a failure is cached ago.
    if (m_cacheEnabled)
    {
        auto [it, inserted] = m_failures.try_emplace({query.name, query.type});
        FailureHistory& history = it->second;
        history.count = (inserted || history.expiry + MAX_FAILURE_TTL <= Simulator::Now())
                            ? 0
                            : history.count + 1;
        const Time ttl = std::min(
            m_failureTtl * (static_cast<int64_t>(1) << std::min<uint32_t>(history.count, 16)),
            MAX_FAILURE_TTL);
        history.expiry = Simulator::Now() + ttl;
        Cache(query.name, query.type, {}, ttl, true, true);
    }
    CompleteQuery(id, {});
}

void
DnsResolver::CompleteQuery(uint16_t id, const std::vector<Address>& addresses)
{
    NS_LOG_FUNCTION(this << id);
    auto it = m_queries.find(id);
    if (it == m_queries.end())
    {
        return;
    }
    Query query = std::move(it->second);
    m_queries.erase(it);
    query.timeout.Cancel();
    CloseTcp(query);
    CloseUdp(query);
    for (auto lookup : query.lookups)
    {
        auto it = m_lookups.find(lookup);
        if (it == m_lookups.end())
        {
            continue;
        }
        (query.type == TYPE_A ? it->second.ipv4 : it->second.ipv6) = addresses;
        if (--it->second.pendingQueries == 0)
        {
            CompleteLookup(lookup);
        }
    }
}

void
DnsResolver::CompleteLookup(uint64_t id)
{
    NS_LOG_FUNCTION(this << id);
    auto it = m_lookups.find(id);
    if (it == m_lookups.end())
    {
        return;
    }
    Lookup lookup = std::move(it->second);
    m_lookups.erase(it);
    std::vector<Address> addresses = lookup.ipv6;
    addresses.insert(addresses.end(), lookup.ipv4.begin(), lookup.ipv4.end());
    NS_LOG_INFO(lookup.name << " resolved to " << addresses.size() << " addresses");
    m_resolvedTrace(lookup.name, addresses);
    if (!lookup.callback.IsNull())
    {
        lookup.callback(lookup.name, addresses);
    }
}

} // namespace ns3
