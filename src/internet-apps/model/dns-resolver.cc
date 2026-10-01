/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
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

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("DnsResolver");

NS_OBJECT_ENSURE_REGISTERED(DnsResolver);

namespace
{

constexpr size_t MAX_QUERIES{0x10000};             ///< number of query identifiers
constexpr uint32_t MAX_CNAME_CHAIN{16};            ///< maximum length of a CNAME chain
constexpr uint32_t MAX_ATTEMPTS_PER_SERVER{3};     ///< maximum attempts to a server (RFC 9520)
constexpr uint16_t EDNS_MIN_UDP_PAYLOAD_SIZE{512}; ///< minimum EDNS(0) UDP payload size
constexpr uint16_t DYNAMIC_PORT_MIN{49152};        ///< first port of the dynamic range
constexpr uint32_t SOURCE_PORT_TRIES{16};          ///< random source ports tried for a query
constexpr uint32_t TCP_LENGTH_PREFIX_SIZE{2};      ///< size of the length prefix over TCP
const Time MAX_FAILURE_TTL{Minutes(5)};            ///< maximum caching of a failure (RFC 9520)

/**
 * Interpret a TTL, which is 0 if its most significant bit is set (RFC 2181, section 8).
 * @param ttl the TTL, as transmitted
 * @return the TTL
 */
Time
TtlOf(uint32_t ttl)
{
    return (ttl & 0x80000000) ? Time(0) : Seconds(ttl);
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
 * @param name a normalized name
 * @param zone a normalized zone name
 * @return whether the name is the zone or one of its subdomains, comparing their labels
 */
bool
IsInZone(const std::string& name, const std::string& zone)
{
    const auto nameLabels = DnsHeader::SplitName(name);
    const auto zoneLabels = DnsHeader::SplitName(zone);
    if (!nameLabels || !zoneLabels || zoneLabels->size() > nameLabels->size())
    {
        return false;
    }
    return std::equal(zoneLabels->rbegin(), zoneLabels->rend(), nameLabels->rbegin());
}

/**
 * @param name a host name, or an address
 * @return the address, if the name is an IPv4 address (possibly with a final dot, like a fully
 * qualified name) or an IPv6 address; an invalid Address otherwise
 */
Address
ParseAddressLiteral(const std::string& name)
{
    std::string ipv4 = name;
    if (!ipv4.empty() && ipv4.back() == '.')
    {
        ipv4.pop_back();
    }
    if (Ipv4Address::CheckCompatible(ipv4))
    {
        return Ipv4Address(ipv4.c_str());
    }
    if (Ipv6Address::CheckCompatible(name))
    {
        return Ipv6Address(name.c_str());
    }
    return Address();
}

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
    for (auto& socket : {m_udpSocket, m_udp6Socket})
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
    for (auto& [id, query] : m_queries)
    {
        if (servers.empty())
        {
            // the query fails when its current attempt is abandoned, unless servers are added
            AbandonAttempt(id);
            continue;
        }
        // the next server is the one following the current server or, if it was removed, the
        // one that followed it
        auto it = std::find(servers.begin(), servers.end(), query.current);
        if (it != servers.end())
        {
            query.server = std::distance(servers.begin(), it);
        }
        else
        {
            query.server = (query.server % servers.size() + servers.size() - 1) % servers.size();
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
    auto isLetter = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
    auto isDigit = [](char c) { return c >= '0' && c <= '9'; };
    bool numericLabel = false;
    for (size_t start = 0; start <= host.size();)
    {
        size_t end = host.find('.', start);
        end = (end == std::string::npos) ? host.size() : end;
        const std::string label = host.substr(start, end - start);
        if (label.empty() || label.size() > 63 || label.front() == '-' || label.back() == '-')
        {
            return false;
        }
        numericLabel = true;
        for (char c : label)
        {
            if (!isLetter(c) && !isDigit(c) && c != '-')
            {
                return false;
            }
            numericLabel = numericLabel && isDigit(c);
        }
        start = end + 1;
    }
    // the last label is not all-numeric (RFC 1123, section 2.1)
    return !numericLabel;
}

bool
DnsResolver::IsResponseTo(const DnsHeader& response,
                          uint16_t id,
                          const std::string& name,
                          DnsHeader::RecordType type)
{
    if (!response.IsResponse() || response.GetOpcode() != DnsHeader::OPCODE_QUERY ||
        response.GetId() != id)
    {
        return false;
    }
    if (response.HasQuestion())
    {
        return Normalize(response.GetQuestionName()) == Normalize(name) &&
               response.GetQuestionType() == type &&
               response.GetQuestionClass() == DnsHeader::CLASS_IN;
    }
    const uint16_t rcode = response.GetRcode();
    return response.GetAnswers().empty() &&
           (rcode == DnsHeader::RCODE_FORMERR || rcode == DnsHeader::RCODE_NOTIMP);
}

DnsResolver::Answer
DnsResolver::ReadAnswer(const DnsHeader& response,
                        const std::string& name,
                        DnsHeader::RecordType type)
{
    Answer answer;
    answer.rcode = response.GetRcode();
    answer.canonicalName = Normalize(name);

    // a response of another EDNS version is an error, unless BADVERS (RFC 6891, section 6.1.3)
    const auto opt = response.GetOptRecord();
    if (opt && ((opt->ttl >> 16) & 0xff) != 0 && answer.rcode != DnsHeader::RCODE_BADVERS)
    {
        answer.rcode = DnsHeader::RCODE_SERVFAIL;
        return answer;
    }

    auto isRecord = [](const DnsResourceRecord& record, uint16_t type, const std::string& owner) {
        return record.type == type && record.rclass == DnsHeader::CLASS_IN &&
               Normalize(record.name) == owner;
    };

    // the CNAME chain from the queried name (RFC 1034, section 3.6.2)
    const auto& answers = response.GetAnswers();
    for (uint32_t links = 0;; links++)
    {
        std::string alias;
        for (const auto& record : answers)
        {
            if (!isRecord(record, DnsHeader::TYPE_CNAME, answer.canonicalName))
            {
                continue;
            }
            // a name has a single alias (RFC 2181, section 10.1), possibly repeated
            const std::string target = Normalize(record.target);
            if (links >= MAX_CNAME_CHAIN || (!alias.empty() && target != alias))
            {
                // a loop, or an ambiguous alias: the server failed to resolve the name
                answer.rcode = DnsHeader::RCODE_SERVFAIL;
                return answer;
            }
            alias = target;
            answer.chainTtl = std::min(answer.chainTtl, TtlOf(record.ttl));
        }
        if (alias.empty())
        {
            break;
        }
        answer.canonicalName = alias;
    }

    // the addresses of the name at the end of the chain
    Time ttl = answer.chainTtl;
    for (const auto& record : answers)
    {
        if (!isRecord(record, type, answer.canonicalName))
        {
            continue;
        }
        // an RRset has no duplicate records (RFC 2181, section 5)
        if (std::find(answer.addresses.begin(), answer.addresses.end(), record.address) ==
            answer.addresses.end())
        {
            answer.addresses.push_back(record.address);
        }
        ttl = std::min(ttl, TtlOf(record.ttl));
    }
    if (!answer.addresses.empty())
    {
        answer.ttl = ttl;
        return answer;
    }

    // Negative answer: the negative TTL is given by the SOA record of the zone of the name at the
    // end of the chain, as the minimum of its TTL and of its MINIMUM field (RFC 2308, section 5),
    // and is bounded by the TTL of the chain
    for (const auto& record : response.GetAuthorities())
    {
        answer.hasNs = answer.hasNs || record.type == DnsHeader::TYPE_NS;
        if (!answer.hasSoa && record.type == DnsHeader::TYPE_SOA &&
            record.rclass == DnsHeader::CLASS_IN &&
            IsInZone(answer.canonicalName, Normalize(record.name)))
        {
            answer.hasSoa = true;
            answer.ttl = std::min({TtlOf(record.ttl), TtlOf(record.soa.minimum), answer.chainTtl});
        }
    }
    return answer;
}

void
DnsResolver::Resolve(const std::string& name, ResolveCallback callback, AddressFamily family)
{
    NS_LOG_FUNCTION(this << name << +family);
    const uint64_t id = m_nextLookup++;
    Lookup& lookup = m_lookups[id];
    lookup.name = name;
    lookup.callback = callback;

    std::vector<DnsHeader::RecordType> types;
    if (const Address literal = ParseAddressLiteral(name); !literal.IsInvalid())
    {
        // an address is not resolved
        if (Ipv4Address::IsMatchingType(literal) && family != IPV6)
        {
            lookup.ipv4 = {literal};
        }
        else if (Ipv6Address::IsMatchingType(literal) && family != IPV4)
        {
            lookup.ipv6 = {literal};
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
            types.push_back(DnsHeader::TYPE_AAAA);
        }
        if (family != IPV6)
        {
            types.push_back(DnsHeader::TYPE_A);
        }
    }

    const std::string normalized = Normalize(name);
    for (auto type : types)
    {
        auto cached = m_cache.find({normalized, type});
        if (cached != m_cache.end() && cached->second.expiry <= Simulator::Now())
        {
            m_cache.erase(cached);
            cached = m_cache.end();
        }
        if (m_cacheEnabled && cached != m_cache.end())
        {
            NS_LOG_INFO("Cached " << (cached->second.failure ? "failure" : "answer") << " for "
                                  << normalized << " (type " << type << ")");
            (type == DnsHeader::TYPE_A ? lookup.ipv4 : lookup.ipv6) = cached->second.addresses;
            continue;
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
            shared->second.lookups.push_back(id);
            lookup.pendingQueries++;
            continue;
        }
        if (m_queries.size() >= MAX_QUERIES)
        {
            NS_LOG_WARN("No query identifier available to resolve " << name);
            continue;
        }
        lookup.pendingQueries++;
        StartQuery(id, normalized, type);
    }
    if (lookup.pendingQueries == 0)
    {
        // the result is reported after Resolve() returns, as when the servers are queried
        lookup.completion = Simulator::ScheduleNow(&DnsResolver::CompleteLookup, this, id);
    }
}

uint16_t
DnsResolver::NewQueryId()
{
    uint16_t id;
    do
    {
        id = m_id->GetInteger(0, 0xffff);
    } while (m_queries.contains(id));
    return id;
}

void
DnsResolver::StartQuery(uint64_t lookup, const std::string& name, DnsHeader::RecordType type)
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

DnsHeader
DnsResolver::BuildQuery(uint16_t id)
{
    Query& query = m_queries.at(id);
    query.currentEdns = m_ednsUdpPayloadSize > 0 && !query.noEdns;
    if (auto it = m_noEdnsServers.find(query.current); it != m_noEdnsServers.end())
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
    DnsHeader message;
    message.SetId(id);
    message.SetRecursionDesired(true);
    message.SetQuestion(query.question, query.type);
    if (query.currentEdns)
    {
        // values below 512 are treated as 512 (RFC 6891, section 6.2.5)
        message.AddOptRecord(std::max(m_ednsUdpPayloadSize, EDNS_MIN_UDP_PAYLOAD_SIZE));
    }
    return message;
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
    for (uint32_t n = 0; n < SOURCE_PORT_TRIES; n++)
    {
        const uint16_t port = m_sourcePort->GetInteger(DYNAMIC_PORT_MIN, 65535);
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
    for (auto& socket : {query.udpSocket, query.udp6Socket})
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
    return m_timeout * (1 << std::min<uint32_t>(round, 10));
}

void
DnsResolver::SendUdp(uint16_t id)
{
    NS_LOG_FUNCTION(this << id);
    Query& query = m_queries.at(id);
    if (m_servers.empty())
    {
        // the servers were removed: the query fails when the attempt is abandoned
        AbandonAttempt(id);
        return;
    }
    const Address server = m_servers[query.server];
    query.current = server;
    query.attempts[server]++;
    NS_LOG_INFO("Query for " << query.question << " (type " << query.type << ") to " << server);
    Ptr<Packet> packet = Create<Packet>();
    packet->AddHeader(BuildQuery(id));
    if (GetUdpSocket(query, server)->SendTo(packet, 0, GetSocketAddress(server)) < 0)
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
    auto it = m_queries.find(id);
    if (it == m_queries.end())
    {
        return;
    }
    it->second.timeout.Cancel();
    it->second.timeout = Simulator::ScheduleNow(&DnsResolver::HandleTimeout, this, id);
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
        for (size_t n = 1; n <= m_servers.size(); n++)
        {
            const size_t server = (query.server + n) % m_servers.size();
            auto attempts = query.attempts.find(m_servers[server]);
            if (attempts == query.attempts.end() || attempts->second < MAX_ATTEMPTS_PER_SERVER)
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
DnsResolver::QueryTarget(uint16_t id, const std::string& target, Time chainTtl)
{
    NS_LOG_FUNCTION(this << id << target << chainTtl);
    Query& query = m_queries.at(id);
    if (query.restarts >= MAX_CNAME_CHAIN || target.empty() || !DnsHeader::SplitName(target))
    {
        FailQuery(id);
        return;
    }
    // a new query, with its own identifier, socket, servers and retransmissions; the identifier
    // of the previous query is released first, so that one is available
    Query previous = std::move(query);
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
    next.chainTtl = std::min(previous.chainTtl, chainTtl);
    next.noEdns = previous.noEdns;
    // the target is queried from the current server or, if it was removed, from the next one
    next.server = previous.server;
    if (!m_servers.empty() && !IsServer(previous.current))
    {
        next.server = (previous.server + 1) % m_servers.size();
    }
    SendUdp(newId);
}

void
DnsResolver::ReceiveUdp(Ptr<Socket> socket)
{
    NS_LOG_FUNCTION(this << socket);
    Address from;
    while (Ptr<Packet> packet = socket->RecvFrom(from))
    {
        DnsHeader response;
        if (packet->PeekHeader(response) == 0)
        {
            NS_LOG_WARN("Ignoring a malformed DNS message from " << from);
            continue;
        }
        const uint16_t id = response.GetId();
        auto it = m_queries.find(id);
        if (it == m_queries.end() || it->second.tcpSocket ||
            (socket != it->second.udpSocket && socket != it->second.udp6Socket))
        {
            continue;
        }
        // a response may come from any of the servers the query was sent to
        const auto& attempts = it->second.attempts;
        auto server = std::find_if(attempts.begin(), attempts.end(), [&](const auto& attempt) {
            return GetSocketAddress(attempt.first) == from;
        });
        if (server == attempts.end())
        {
            NS_LOG_WARN("Ignoring a DNS message from " << from);
            continue;
        }
        HandleResponse(id, response, false, server->first);
    }
}

void
DnsResolver::HandleResponse(uint16_t id,
                            const DnsHeader& response,
                            bool overTcp,
                            const Address& server)
{
    NS_LOG_FUNCTION(this << id << response << overTcp << server);
    Query& query = m_queries.at(id);
    if (!IsResponseTo(response, id, query.question, query.type))
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
    // errors of these servers do not affect the current attempt
    const bool current = server == query.current;
    const uint16_t rcode = response.GetRcode();
    if (current && query.currentEdns && !response.GetOptRecord() &&
        (rcode == DnsHeader::RCODE_FORMERR || rcode == DnsHeader::RCODE_NOTIMP))
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
        SendUdp(id);
        return;
    }
    if (!response.HasQuestion())
    {
        // a response without question is only trusted as a rejection of EDNS(0)
        return;
    }
    if (response.IsTruncated())
    {
        if (!current)
        {
            return;
        }
        if (overTcp)
        {
            // a response over TCP is never truncated (RFC 7766, section 8)
            Retransmit(id);
        }
        else
        {
            SendTcp(id);
        }
        return;
    }
    HandleAnswer(id, response, ReadAnswer(response, query.question, query.type), server);
}

void
DnsResolver::HandleAnswer(uint16_t id,
                          const DnsHeader& response,
                          const Answer& answer,
                          const Address& server)
{
    NS_LOG_FUNCTION(this << id << answer.rcode << server);
    Query& query = m_queries.at(id);
    const bool current = server == query.current;
    switch (answer.rcode)
    {
    case DnsHeader::RCODE_NOERROR:
        if (!answer.addresses.empty() || answer.hasSoa)
        {
            // an answer, or the absence of address of the requested type (NODATA)
            Cache({query.name, query.type}, answer.addresses, std::min(answer.ttl, query.chainTtl));
            CompleteQuery(id, answer.addresses);
        }
        else if (answer.canonicalName != query.question)
        {
            // the CNAME chain ends without the data of its target: query the target (RFC 1034,
            // section 3.6.2)
            QueryTarget(id, answer.canonicalName, answer.chainTtl);
        }
        else if (!current)
        {
            // the other responses of a previous server do not affect the current attempt
        }
        else if (response.GetAnswers().empty() &&
                 (answer.hasNs || !response.IsRecursionAvailable()))
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
    case DnsHeader::RCODE_NXDOMAIN: {
        // the name has no record of any type (RFC 8020), and the answer of the other type is
        // kept if it has addresses
        const std::string name = query.name;
        const auto type = query.type;
        const auto otherType = type == DnsHeader::TYPE_A ? DnsHeader::TYPE_AAAA : DnsHeader::TYPE_A;
        const Time ttl = answer.hasSoa ? std::min(answer.ttl, query.chainTtl) : Time(0);
        Cache({name, type}, {}, ttl);
        Cache({name, otherType}, {}, ttl, false);
        std::vector<uint16_t> ids;
        for (const auto& [otherId, other] : m_queries)
        {
            if (other.name == name)
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
            NS_LOG_INFO("Error " << answer.rcode << " from " << server);
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
    query.tcpData = Create<Packet>();
    const int bound =
        Ipv4Address::IsMatchingType(server) ? query.tcpSocket->Bind() : query.tcpSocket->Bind6();
    if (bound != 0)
    {
        AbandonAttempt(id);
        return;
    }
    query.tcpSocket->SetConnectCallback(MakeCallback(&DnsResolver::HandleTcpConnected, this),
                                        MakeCallback(&DnsResolver::HandleTcpClose, this));
    query.tcpSocket->SetRecvCallback(MakeCallback(&DnsResolver::ReceiveTcp, this));
    query.tcpSocket->SetCloseCallbacks(MakeCallback(&DnsResolver::HandleTcpClose, this),
                                       MakeCallback(&DnsResolver::HandleTcpClose, this));
    if (query.tcpSocket->Connect(GetSocketAddress(server)) != 0)
    {
        // e.g., no route to the server
        AbandonAttempt(id);
        return;
    }
    query.timeout =
        Simulator::Schedule(GetAttemptTimeout(query), &DnsResolver::HandleTimeout, this, id);
}

std::map<uint16_t, DnsResolver::Query>::iterator
DnsResolver::FindTcpQuery(Ptr<Socket> socket)
{
    return std::find_if(m_queries.begin(), m_queries.end(), [&](const auto& entry) {
        return entry.second.tcpSocket == socket;
    });
}

void
DnsResolver::HandleTcpConnected(Ptr<Socket> socket)
{
    NS_LOG_FUNCTION(this << socket);
    auto it = FindTcpQuery(socket);
    if (it == m_queries.end())
    {
        return;
    }
    // over TCP, the message is prefixed with its length (RFC 1035, section 4.2.2)
    Ptr<Packet> message = Create<Packet>();
    message->AddHeader(BuildQuery(it->first));
    const uint8_t prefix[TCP_LENGTH_PREFIX_SIZE] = {static_cast<uint8_t>(message->GetSize() >> 8),
                                                    static_cast<uint8_t>(message->GetSize())};
    Ptr<Packet> packet = Create<Packet>(prefix, sizeof(prefix));
    packet->AddAtEnd(message);
    socket->Send(packet);
}

void
DnsResolver::ReceiveTcp(Ptr<Socket> socket)
{
    NS_LOG_FUNCTION(this << socket);
    auto it = FindTcpQuery(socket);
    if (it == m_queries.end())
    {
        return;
    }
    const uint16_t id = it->first;
    Query& query = it->second;
    while (Ptr<Packet> packet = socket->Recv())
    {
        query.tcpData->AddAtEnd(packet);
    }
    if (query.tcpData->GetSize() < TCP_LENGTH_PREFIX_SIZE)
    {
        return;
    }
    uint8_t prefix[TCP_LENGTH_PREFIX_SIZE];
    query.tcpData->CopyData(prefix, sizeof(prefix));
    const uint32_t length = prefix[0] << 8 | prefix[1];
    if (query.tcpData->GetSize() < TCP_LENGTH_PREFIX_SIZE + length)
    {
        return;
    }
    Ptr<Packet> message = query.tcpData->CreateFragment(TCP_LENGTH_PREFIX_SIZE, length);
    const Address server = query.current;
    CloseTcp(query);
    DnsHeader response;
    if (message->PeekHeader(response) == 0)
    {
        // the connection is closed: no other response can come
        NS_LOG_WARN("Ignoring a malformed DNS response from " << server);
        Retransmit(id);
        return;
    }
    HandleResponse(id, response, true, server);
}

void
DnsResolver::HandleTcpClose(Ptr<Socket> socket)
{
    NS_LOG_FUNCTION(this << socket);
    auto it = FindTcpQuery(socket);
    if (it == m_queries.end())
    {
        return;
    }
    // the connection was closed or failed before a complete response (RFC 7766, section 6.2)
    NS_LOG_INFO("TCP connection to " << it->second.current << " closed");
    Retransmit(it->first);
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
    query.tcpData = nullptr;
}

void
DnsResolver::Cache(const CacheKey& key,
                   const std::vector<Address>& addresses,
                   Time ttl,
                   bool overwrite,
                   bool failure)
{
    NS_LOG_FUNCTION(this << key.first << key.second << addresses.size() << ttl << overwrite
                         << failure);
    if (!failure)
    {
        m_failures.erase(key);
    }
    if (!m_cacheEnabled || ttl <= Time(0) || m_maxCacheEntries == 0)
    {
        return;
    }
    if (Simulator::Now() - m_lastPurge >= Minutes(1))
    {
        PurgeCache();
    }
    auto it = m_cache.find(key);
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
    m_cache[key] = {addresses, Simulator::Now() + ttl, failure};
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
        const CacheKey key{query.name, query.type};
        auto [it, inserted] = m_failures.try_emplace(key);
        FailureHistory& history = it->second;
        if (inserted || history.expiry + MAX_FAILURE_TTL <= Simulator::Now())
        {
            history.count = 0;
        }
        else
        {
            history.count++;
        }
        const Time ttl =
            std::min(m_failureTtl * (1 << std::min<uint32_t>(history.count, 16)), MAX_FAILURE_TTL);
        history.expiry = Simulator::Now() + ttl;
        Cache(key, {}, ttl, true, true);
    }
    CompleteQuery(id, {});
}

void
DnsResolver::CompleteQuery(uint16_t id, const std::vector<Address>& addresses)
{
    NS_LOG_FUNCTION(this << id << addresses.size());
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
    for (auto lookupId : query.lookups)
    {
        auto lookup = m_lookups.find(lookupId);
        if (lookup == m_lookups.end())
        {
            continue;
        }
        (query.type == DnsHeader::TYPE_A ? lookup->second.ipv4 : lookup->second.ipv6) = addresses;
        if (--lookup->second.pendingQueries == 0)
        {
            CompleteLookup(lookupId);
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
