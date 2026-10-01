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
#include <cctype>
#include <limits>
#include <regex>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("DnsResolver");

NS_OBJECT_ENSURE_REGISTERED(DnsResolver);

namespace
{

constexpr size_t MAX_QUERIES{0x10000};             ///< Number of query IDs
constexpr uint32_t MAX_CNAME_CHAIN{16};            ///< Maximum CNAME chain length
constexpr uint32_t MAX_ATTEMPTS_PER_SERVER{3};     ///< Maximum attempts per server
constexpr uint16_t EDNS_MIN_UDP_PAYLOAD_SIZE{512}; ///< Minimum EDNS(0) UDP payload size
constexpr uint16_t DYNAMIC_PORT_MIN{49152};        ///< First port of the dynamic range
constexpr uint32_t SOURCE_PORT_TRIES{16};          ///< Random source ports to try
constexpr uint32_t TCP_LENGTH_PREFIX_SIZE{2};      ///< Length prefix size over TCP
const Time MAX_FAILURE_TTL{Minutes(5)};            ///< Maximum failure cache lifetime
const Time CACHE_PURGE_INTERVAL{Minutes(1)};       ///< Minimum time between cache purges
constexpr size_t MAX_HOST_NAME_SIZE{253};          ///< Maximum host name length
constexpr uint32_t MAX_TIMEOUT_DOUBLINGS{10};      ///< Cap on the timeout backoff exponent
constexpr uint32_t MAX_FAILURE_DOUBLINGS{16};      ///< Cap on the failure backoff exponent

/**
 * Parse an IPv4 or IPv6 address literal. A trailing dot is accepted after an IPv4 address.
 *
 * @param name Host name or address literal.
 * @return The address, or an invalid Address if the name is not an address literal.
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
                          "DNS server port",
                          UintegerValue(53),
                          MakeUintegerAccessor(&DnsResolver::m_port),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("Timeout",
                          "Initial timeout before retransmitting a query. It doubles after each "
                          "round over the servers",
                          TimeValue(Seconds(2)),
                          MakeTimeAccessor(&DnsResolver::m_timeout),
                          MakeTimeChecker(NanoSeconds(1)))
            .AddAttribute("Retransmissions",
                          "Maximum number of retransmissions of a query, each to the next server",
                          UintegerValue(2),
                          MakeUintegerAccessor(&DnsResolver::m_retransmissions),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("CacheEnabled",
                          "Enable caching of DNS answers",
                          BooleanValue(true),
                          MakeBooleanAccessor(&DnsResolver::m_cacheEnabled),
                          MakeBooleanChecker())
            .AddAttribute("MaxTtl",
                          "Maximum cache lifetime of positive answers",
                          TimeValue(Days(1)),
                          MakeTimeAccessor(&DnsResolver::m_maxTtl),
                          MakeTimeChecker(Time(0)))
            .AddAttribute("MaxNegativeTtl",
                          "Maximum cache lifetime of negative answers (NXDOMAIN and NODATA)",
                          TimeValue(Hours(3)),
                          MakeTimeAccessor(&DnsResolver::m_maxNegativeTtl),
                          MakeTimeChecker(Time(0)))
            .AddAttribute("FailureTtl",
                          "Initial cache lifetime of resolution failures",
                          TimeValue(Seconds(5)),
                          MakeTimeAccessor(&DnsResolver::m_failureTtl),
                          MakeTimeChecker(Seconds(1), Minutes(5)))
            .AddAttribute("MaxCacheEntries",
                          "Maximum number of cache entries",
                          UintegerValue(10000),
                          MakeUintegerAccessor(&DnsResolver::m_maxCacheEntries),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("EdnsUdpPayloadSize",
                          "EDNS(0) UDP payload size advertised in queries (0 disables EDNS(0)). "
                          "Larger responses are truncated by the server and retried over TCP",
                          UintegerValue(1232),
                          MakeUintegerAccessor(&DnsResolver::m_ednsUdpPayloadSize),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("NoEdnsTime",
                          "How long to stop using EDNS(0) with a server that rejected it",
                          TimeValue(Minutes(10)),
                          MakeTimeAccessor(&DnsResolver::m_noEdnsTime),
                          MakeTimeChecker(Time(0)))
            .AddTraceSource(
                "Resolved",
                "Result of a resolution: the host name and its addresses (empty on failure)",
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
    // Pending lookups are dropped without invoking their callbacks
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
    // Failures may not happen with the new servers
    for (auto it = m_cache.begin(); it != m_cache.end();)
    {
        it = it->second.failure ? m_cache.erase(it) : std::next(it);
    }
    m_failures.clear();
    for (auto& [id, query] : m_queries)
    {
        if (servers.empty())
        {
            // Fails on timeout unless servers are added in the meantime
            AbandonAttempt(id);
            continue;
        }
        // Keep pointing at the current server. If it was removed, point at the server before
        // it, so that the next retransmission goes to the one that followed it.
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
    // Labels of up to 63 letters, digits and hyphens, not starting or ending with a hyphen, the
    // last one not all-numeric, optionally followed by a trailing dot
    static const std::regex hostName(R"((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*)"
                                     R"((?=[0-9]*[a-z-])[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.?)",
                                     std::regex::icase);
    const size_t length = name.ends_with('.') ? name.size() - 1 : name.size();
    return length <= MAX_HOST_NAME_SIZE && std::regex_match(name, hostName);
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
        return DnsHeader::NormalizeName(response.GetQuestionName()) ==
                   DnsHeader::NormalizeName(name) &&
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
    answer.canonicalName = DnsHeader::NormalizeName(name);

    // Only EDNS version 0 is supported
    const auto opt = response.GetOptRecord();
    if (opt && opt->GetEdnsVersion() != 0 && answer.rcode != DnsHeader::RCODE_BADVERS)
    {
        answer.rcode = DnsHeader::RCODE_SERVFAIL;
        return answer;
    }

    // Follow the CNAME chain starting at the queried name
    const auto& answers = response.GetAnswers();
    for (uint32_t links = 0;; links++)
    {
        std::string alias;
        for (const auto& record : answers)
        {
            if (!record.Matches(DnsHeader::TYPE_CNAME, answer.canonicalName))
            {
                continue;
            }
            // A name can have only one CNAME (RFC 2181, section 10.1), although it may be
            // repeated. Anything else, or a chain that is too long, is a server failure.
            const std::string target = DnsHeader::NormalizeName(record.GetTarget());
            if (links >= MAX_CNAME_CHAIN || (!alias.empty() && target != alias))
            {
                answer.rcode = DnsHeader::RCODE_SERVFAIL;
                return answer;
            }
            alias = target;
            answer.chainTtl = std::min(answer.chainTtl, record.GetTtl());
        }
        if (alias.empty())
        {
            break;
        }
        answer.canonicalName = alias;
    }

    Time ttl = answer.chainTtl;
    for (const auto& record : answers)
    {
        if (!record.Matches(type, answer.canonicalName))
        {
            continue;
        }
        if (std::find(answer.addresses.begin(), answer.addresses.end(), record.GetAddress()) ==
            answer.addresses.end())
        {
            answer.addresses.push_back(record.GetAddress());
        }
        ttl = std::min(ttl, record.GetTtl());
    }
    if (!answer.addresses.empty())
    {
        answer.ttl = ttl;
        return answer;
    }

    // No addresses: the negative TTL is the smaller of the SOA TTL and its MINIMUM field
    // (RFC 2308, section 5), capped by the TTL of the CNAME chain
    for (const auto& record : response.GetAuthorities())
    {
        answer.hasNs = answer.hasNs || record.GetType() == DnsHeader::TYPE_NS;
        if (!answer.hasSoa && record.GetType() == DnsHeader::TYPE_SOA &&
            record.GetClass() == DnsHeader::CLASS_IN &&
            DnsHeader::IsInZone(answer.canonicalName, DnsHeader::NormalizeName(record.GetName())))
        {
            answer.hasSoa = true;
            answer.ttl =
                std::min({record.GetTtl(), record.GetSoa().GetMinimumTtl(), answer.chainTtl});
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

    const std::string normalized = DnsHeader::NormalizeName(name);
    for (auto type : types)
    {
        auto cached = m_cache.find({normalized, type});
        if (cached != m_cache.end() && cached->second.IsExpired())
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
        // Reuse a query in progress for the same question (RFC 5452, section 5)
        if (Query* shared = FindQuery(normalized, type))
        {
            NS_LOG_INFO("Query for " << normalized << " (type " << type << ") in progress");
            shared->lookups.push_back(id);
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
        // Always report the result asynchronously
        lookup.completion = Simulator::ScheduleNow(&DnsResolver::CompleteLookup, this, id);
    }
}

bool
DnsResolver::CacheEntry::IsExpired() const
{
    return expiry <= Simulator::Now();
}

uint16_t
DnsResolver::NewQueryId()
{
    uint16_t id;
    do
    {
        id = m_id->GetInteger(0, std::numeric_limits<uint16_t>::max());
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
        // Values below 512 must be treated as 512 (RFC 6891, section 6.2.5)
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
    // Use a random source port for each query (RFC 5452, section 9.2)
    socket = Socket::CreateSocket(node, UdpSocketFactory::GetTypeId());
    for (uint32_t n = 0; n < SOURCE_PORT_TRIES; n++)
    {
        const uint16_t port =
            m_sourcePort->GetInteger(DYNAMIC_PORT_MIN, std::numeric_limits<uint16_t>::max());
        const int bound = ipv4 ? socket->Bind(InetSocketAddress(Ipv4Address::GetAny(), port))
                               : socket->Bind(Inet6SocketAddress(Ipv6Address::GetAny(), port));
        if (bound == 0)
        {
            socket->SetRecvCallback(MakeCallback(&DnsResolver::ReceiveUdp, this));
            return socket;
        }
    }
    // No free port was found, fall back to the shared socket
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
    // Exponential backoff after each round over the servers
    const uint32_t round = query.retransmissions / std::max<size_t>(m_servers.size(), 1);
    return m_timeout * (1 << std::min(round, MAX_TIMEOUT_DOUBLINGS));
}

void
DnsResolver::SendUdp(uint16_t id)
{
    NS_LOG_FUNCTION(this << id);
    Query& query = m_queries.at(id);
    if (m_servers.empty())
    {
        // All servers were removed
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
        // E.g., no route to the server
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
        // Not a server failure, so it is not cached
        CompleteQuery(id, {});
        return;
    }
    if (query.retransmissions < m_retransmissions)
    {
        // Pick the next server that has not reached the per-server attempt limit
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
    // The target gets a brand new query (ID, socket and retransmissions). The old ID is
    // released first so that a new one is always available.
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
    // Ask the same server, or the next one if it was removed
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
        // Accept a late response from any server this query was sent to
        Address server;
        for (const auto& [attempted, count] : it->second.attempts)
        {
            if (GetSocketAddress(attempted) == from)
            {
                server = attempted;
                break;
            }
        }
        if (server.IsInvalid())
        {
            NS_LOG_WARN("Ignoring a DNS message from " << from);
            continue;
        }
        HandleResponse(id, response, false, server);
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
            // The connection is closed, so no other response can arrive
            Retransmit(id);
        }
        return;
    }

    // Late responses from previous servers are only used if they carry an answer; their
    // errors are ignored
    const bool current = server == query.current;
    const uint16_t rcode = response.GetRcode();
    if (current && query.currentEdns && !response.GetOptRecord() &&
        (rcode == DnsHeader::RCODE_FORMERR || rcode == DnsHeader::RCODE_NOTIMP))
    {
        // The server does not support EDNS(0), retry without it (RFC 6891, section 7)
        NS_LOG_INFO("No EDNS(0) support in " << server);
        m_noEdnsServers[server] = Simulator::Now() + m_noEdnsTime;
        query.noEdns = true;
        if (!IsServer(server))
        {
            // The server was removed, move on
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
        // A response without a question is only meaningful as an EDNS(0) rejection
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
            // Truncation is invalid over TCP
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
            // Addresses, or NODATA
            Cache({query.name, query.type}, answer.addresses, std::min(answer.ttl, query.chainTtl));
            CompleteQuery(id, answer.addresses);
        }
        else if (answer.canonicalName != query.question)
        {
            // The CNAME chain ends without data for its target, so query the target
            QueryTarget(id, answer.canonicalName, answer.chainTtl);
        }
        else if (!current)
        {
            // Ignore late responses from previous servers
        }
        else if (response.GetAnswers().empty() &&
                 (answer.hasNs || !response.IsRecursionAvailable()))
        {
            // Referral: the server does not offer recursion
            NS_LOG_INFO("Referral from " << server);
            Retransmit(id);
        }
        else
        {
            CompleteQuery(id, {});
        }
        break;
    case DnsHeader::RCODE_NXDOMAIN: {
        // NXDOMAIN applies to all record types (RFC 8020), so also complete the query for the
        // other type, unless it already has cached addresses
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
        // E.g., SERVFAIL or REFUSED: try another server
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
        // E.g., no route to the server
        AbandonAttempt(id);
        return;
    }
    query.timeout =
        Simulator::Schedule(GetAttemptTimeout(query), &DnsResolver::HandleTimeout, this, id);
}

std::map<uint16_t, DnsResolver::Query>::iterator
DnsResolver::FindTcpQuery(Ptr<Socket> socket)
{
    auto it = m_queries.begin();
    while (it != m_queries.end() && it->second.tcpSocket != socket)
    {
        ++it;
    }
    return it;
}

DnsResolver::Query*
DnsResolver::FindQuery(const std::string& name, DnsHeader::RecordType type)
{
    for (auto& [id, query] : m_queries)
    {
        if (query.name == name && query.type == type)
        {
            return &query;
        }
    }
    return nullptr;
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
    // Over TCP, messages are prefixed with a 2-byte length (RFC 1035, section 4.2.2)
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
        // The connection is closed, so no other response can arrive
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
    // Closed or failed before a complete response arrived
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
    if (Simulator::Now() - m_lastPurge >= CACHE_PURGE_INTERVAL)
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
            // Evict the entry closest to expiry
            auto oldest = m_cache.begin();
            for (auto it = m_cache.begin(); it != m_cache.end(); ++it)
            {
                if (it->second.expiry < oldest->second.expiry)
                {
                    oldest = it;
                }
            }
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
    for (auto it = m_cache.begin(); it != m_cache.end();)
    {
        it = it->second.IsExpired() ? m_cache.erase(it) : std::next(it);
    }
    for (auto it = m_failures.begin(); it != m_failures.end();)
    {
        const bool forgotten = it->second.expiry + MAX_FAILURE_TTL <= Simulator::Now();
        it = forgotten ? m_failures.erase(it) : std::next(it);
    }
}

void
DnsResolver::FailQuery(uint16_t id)
{
    NS_LOG_FUNCTION(this << id);
    const Query& query = m_queries.at(id);
    // Cache failures for a time that doubles after each consecutive failure (RFC 9520,
    // section 3.2). A failure counts as consecutive if the previous one expired less than
    // MAX_FAILURE_TTL ago.
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
            std::min(m_failureTtl * (1 << std::min(history.count, MAX_FAILURE_DOUBLINGS)),
                     MAX_FAILURE_TTL);
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
