/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

#ifndef DNS_RESOLVER_H
#define DNS_RESOLVER_H

#include "ns3/address.h"
#include "ns3/callback.h"
#include "ns3/event-id.h"
#include "ns3/nstime.h"
#include "ns3/object.h"
#include "ns3/traced-callback.h"

#include <map>
#include <string>
#include <vector>

namespace ns3
{

class Packet;
class Socket;
class UniformRandomVariable;

/**
 * @ingroup internet-apps
 * @defgroup dns-resolver DNS stub resolver
 */

/**
 * @ingroup dns-resolver
 * @brief DNS stub resolver (RFC 1035) of the IPv4 and IPv6 addresses of host names.
 *
 * The resolver is aggregated to a node, and sends recursive queries for the A (IPv4) and AAAA
 * (IPv6) records of host names to DNS servers, which are recursive resolvers (e.g., the servers
 * provided by the network). The servers are reached over UDP, on IPv4 or IPv6, and over TCP
 * when a response is truncated (RFC 7766). The queries advertise an EDNS(0) UDP payload size
 * (RFC 6891), unless a server is known not to support EDNS(0). Each query is sent from a UDP
 * port drawn at random in the dynamic range (RFC 5452), and the concurrent resolutions of the
 * same name and record type share their query.
 *
 * A response is accepted only from a server the query was sent to, with the identifier, the
 * opcode and the question of the query (RFC 5452). The addresses are taken from the records of
 * the queried name or, if it is an alias, of the name at the end of its CNAME chain. When a
 * chain ends without the data of its target, a new query is sent for the target.
 *
 * A query that is not answered within the Timeout attribute is retransmitted to the next
 * server, as is a query that the current server fails to answer (e.g., with SERVFAIL or
 * REFUSED, or with a referral since it does not offer recursion), up to the Retransmissions
 * attribute times, and at most three times to each server (RFC 9520). The timeout doubles
 * after each round of the servers. An answer from a server the query was sent to before is
 * still accepted.
 *
 * The answers are cached for the minimum TTL of their records (RFC 2181), up to the MaxTtl
 * attribute, the answers without addresses (the name does not exist, or has no address of the
 * requested family) for the negative TTL of the SOA record of their zone (RFC 2308), up to the
 * MaxNegativeTtl attribute, and the resolution failures for the FailureTtl attribute, doubled
 * after each consecutive failure up to five minutes (RFC 9520).
 *
 * The result of each resolution is reported to a callback, with the addresses of the host
 * name (Ipv4Address and Ipv6Address instances, the IPv6 addresses first), which are empty if
 * the resolution failed.
 */
class DnsResolver : public Object
{
  public:
    /// Address families to resolve
    enum AddressFamily : uint8_t
    {
        IPV4, ///< IPv4 addresses (A records)
        IPV6, ///< IPv6 addresses (AAAA records)
        ANY,  ///< IPv6 and IPv4 addresses (AAAA and A records)
    };

    /// Types of the DNS records handled by the resolver
    enum RecordType : uint16_t
    {
        TYPE_A = 1,     ///< IPv4 address
        TYPE_NS = 2,    ///< authoritative name server
        TYPE_CNAME = 5, ///< canonical name of an alias
        TYPE_SOA = 6,   ///< start of a zone of authority
        TYPE_AAAA = 28, ///< IPv6 address
        TYPE_OPT = 41,  ///< EDNS(0) options
    };

    /// Response codes handled by the resolver (RFC 1035, RFC 6891)
    enum ResponseCode : uint16_t
    {
        RCODE_NOERROR = 0,  ///< no error
        RCODE_FORMERR = 1,  ///< format error
        RCODE_SERVFAIL = 2, ///< server failure
        RCODE_NXDOMAIN = 3, ///< the name does not exist
        RCODE_NOTIMP = 4,   ///< not implemented
        RCODE_REFUSED = 5,  ///< refused
        RCODE_BADVERS = 16, ///< unsupported EDNS version
    };

    /**
     * Callback reporting the result of a resolution.
     *
     * The first argument is the host name, and the second one the addresses of the host name
     * (Ipv4Address and Ipv6Address instances), which are empty if the resolution failed.
     */
    using ResolveCallback = Callback<void, const std::string&, const std::vector<Address>&>;

    /**
     * TracedCallback signature for the result of a resolution.
     *
     * @param [in] name the host name
     * @param [in] addresses the addresses of the host name, empty if the resolution failed
     */
    typedef void (*ResolvedTracedCallback)(const std::string& name,
                                           const std::vector<Address>& addresses);

    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    DnsResolver();
    ~DnsResolver() override;

    /**
     * Set the DNS servers, in the order in which they are queried, and forget the cached
     * resolution failures.
     *
     * The queries in progress continue with the new servers. If there is none, they fail when
     * their current attempt is abandoned, after this method returns.
     *
     * @param servers the addresses of the servers (Ipv4Address or Ipv6Address instances)
     */
    void SetServers(const std::vector<Address>& servers);

    /**
     * Add a DNS server, queried after the ones already set.
     * @param server the address of the server (an Ipv4Address or an Ipv6Address)
     */
    void AddServer(const Address& server);

    /**
     * @return the addresses of the DNS servers
     */
    std::vector<Address> GetServers() const;

    /**
     * Resolve the addresses of a host name.
     *
     * The resolver must be aggregated to a node with an internet stack. The callback (if not
     * null) is always invoked after this method returns, even when the result is cached, the
     * name is invalid or there is no server (the result is then empty), unless the resolver is
     * disposed of first. An IPv4 address in dotted-decimal notation is returned as is, without
     * query.
     *
     * @param name the host name, e.g., "www.nsnam.org", possibly with a final dot
     * @param callback the callback reporting the result
     * @param family the address family to resolve
     */
    void Resolve(const std::string& name, ResolveCallback callback, AddressFamily family = IPV4);

    /// Remove all the cached answers and failures
    void FlushCache();

    /**
     * Assign a fixed random variable stream number to the random variables used by this
     * object: the query identifiers and the UDP source ports.
     *
     * @param stream first stream index to use
     * @return the number of stream indices assigned by this object
     */
    int64_t AssignStreams(int64_t stream);

    /**
     * Check whether a name is a valid host name (RFC 1123, section 2.1): dot-separated labels
     * of 1 to 63 letters, digits and hyphens, not starting or ending with a hyphen, the last
     * one not all-numeric (so that the name is not an address), of at most 253 characters,
     * possibly with a final dot.
     *
     * @param name the name
     * @return true if the name is a valid host name
     */
    static bool IsValidHostName(const std::string& name);

    /**
     * Encode a DNS query for the records of a host name.
     *
     * @param id the identifier of the query
     * @param name the host name
     * @param type the type of the records
     * @param ednsUdpPayloadSize the EDNS(0) UDP payload size to advertise (at least 512), or 0
     * for no EDNS(0)
     * @return the DNS message, or an empty vector if the name is not a valid host name
     */
    static std::vector<uint8_t> EncodeQuery(uint16_t id,
                                            const std::string& name,
                                            RecordType type,
                                            uint16_t ednsUdpPayloadSize = 0);

    /// Result of decoding a DNS response
    struct Response
    {
        /// whether the message is a response to the query (header and question)
        bool valid{false};
        uint16_t id{0};                 ///< identifier of the query it answers
        uint16_t rcode{0};              ///< response code, including the EDNS(0) extension
        bool hasQuestion{false};        ///< whether the response has the question
        bool truncated{false};          ///< whether the response is truncated
        bool recursionAvailable{false}; ///< whether the server offers recursion
        bool hasOpt{false};             ///< whether the response has an EDNS(0) OPT record
        bool hasSoa{false};             ///< whether the authority section has the zone SOA
        bool hasNs{false};              ///< whether the authority section has NS records
        bool hasAnswers{false};         ///< whether the answer section has records
        /// name at the end of the CNAME chain of the queried name (the queried name if none)
        std::string canonicalName;
        int64_t chainTtl{-1};           ///< minimum TTL of the CNAME chain, -1 if none
        std::vector<Address> addresses; ///< addresses of the queried name, or of its alias
        /// TTL of the answer, in seconds: the minimum TTL of the alias and address records, or
        /// the negative TTL of the SOA record when there is no address; -1 if unknown
        int64_t ttl{-1};
    };

    /**
     * Decode a DNS response.
     *
     * A response without question is accepted only for the FORMERR and NOTIMP response codes,
     * which servers not supporting EDNS(0) may return without question (RFC 6891). A response
     * with the question of the query, but with a malformed or inconsistent record section (e.g.,
     * an address record of the wrong length, or several aliases for a name), has the SERVFAIL
     * response code. The duplicate addresses are removed. The names are in lower case, with
     * dot-separated labels in which the dots and backslashes are escaped with a backslash, and
     * without final dot.
     *
     * @param message the DNS message
     * @param name the name of the query
     * @param type the type of the records of the query (TYPE_A or TYPE_AAAA)
     * @return the decoded response
     */
    static Response DecodeResponse(const std::vector<uint8_t>& message,
                                   const std::string& name,
                                   RecordType type);

  protected:
    void DoDispose() override;

  private:
    /// A resolution, made of one query per record type
    struct Lookup
    {
        std::string name;           ///< the host name, as requested
        ResolveCallback callback;   ///< the callback reporting the result
        uint32_t pendingQueries{0}; ///< number of queries without result
        std::vector<Address> ipv6;  ///< IPv6 addresses
        std::vector<Address> ipv4;  ///< IPv4 addresses
        EventId completion;         ///< the report of a result without query
    };

    /// A query waiting for its response
    struct Query
    {
        std::vector<uint64_t> lookups;        ///< the resolutions sharing the query
        std::string name;                     ///< the normalized host name, as requested
        std::string question;                 ///< the name in the question (the target of an alias)
        RecordType type{TYPE_A};              ///< the type of the records
        uint32_t retransmissions{0};          ///< number of retransmissions so far
        uint32_t restarts{0};                 ///< number of queries for the targets of aliases
        int64_t chainTtl{-1};                 ///< minimum TTL of the aliases followed, -1 if none
        size_t server{0};                     ///< index of the current server in the servers
        Address current;                      ///< the current server
        bool currentEdns{false};              ///< whether the current attempt uses EDNS(0)
        bool noEdns{false};                   ///< whether EDNS(0) was rejected for this query
        std::vector<Address> queried;         ///< servers the query was sent to
        Ptr<Socket> udpSocket;                ///< the UDP socket for the IPv4 servers
        Ptr<Socket> udp6Socket;               ///< the UDP socket for the IPv6 servers
        std::map<Address, uint32_t> attempts; ///< number of attempts, by server
        EventId timeout;                      ///< the timeout of the current attempt
        Ptr<Socket> tcpSocket;                ///< the TCP connection, for truncated responses
        std::vector<uint8_t> tcpData;         ///< the data received on the TCP connection
    };

    /// A cached answer, or resolution failure
    struct CacheEntry
    {
        std::vector<Address> addresses; ///< the addresses
        Time expiry;                    ///< the expiry time
        bool failure{false};            ///< whether the resolution failed
    };

    /// The consecutive resolution failures of a name and record type
    struct FailureHistory
    {
        uint32_t count{0}; ///< number of consecutive failures
        Time expiry;       ///< the expiry of the cached failure
    };

    /**
     * Start a query for the records of a name.
     * @param lookup the resolution of the query
     * @param name the normalized host name
     * @param type the type of the records
     */
    void StartQuery(uint64_t lookup, const std::string& name, RecordType type);

    /**
     * @return an identifier not used by any query
     */
    uint16_t NewQueryId();

    /**
     * @param server a server
     * @return whether it is one of the servers
     */
    bool IsServer(const Address& server) const;

    /**
     * Encode a query for its current server, with EDNS(0) unless the server or the query
     * rejected it.
     * @param id the identifier of a query
     * @return the encoded query
     */
    std::vector<uint8_t> EncodeQuery(uint16_t id);

    /**
     * Send a query over UDP, to its current server in the servers, or to another server.
     * @param id the identifier of the query
     * @param server the server, or a null address for the current server in the servers
     */
    void SendUdp(uint16_t id, Address server = Address());

    /**
     * @param query a query
     * @return the timeout of the current attempt of the query
     */
    Time GetAttemptTimeout(const Query& query) const;

    /**
     * Abandon the current attempt of a query at once, as on its timeout, from a new event.
     * @param id the identifier of the query
     */
    void AbandonAttempt(uint16_t id);

    /**
     * Retransmit a query to the next server, or give up.
     * @param id the identifier of the query
     */
    void Retransmit(uint16_t id);

    /**
     * Query the target of an alias, in a new query replacing a query.
     * @param id the identifier of the query
     * @param target the target of the alias
     * @param chainTtl the minimum TTL of the alias chain
     */
    void QueryTarget(uint16_t id, const std::string& target, int64_t chainTtl);

    /**
     * Handle the timeout of a query.
     * @param id the identifier of the query
     */
    void HandleTimeout(uint16_t id);

    /**
     * Receive the responses over UDP.
     * @param socket the socket
     */
    void ReceiveUdp(Ptr<Socket> socket);

    /**
     * Handle a response.
     * @param id the identifier of the query
     * @param message the response
     * @param overTcp whether the response was received over TCP
     * @param server the server that sent the response
     */
    void HandleResponse(uint16_t id,
                        const std::vector<uint8_t>& message,
                        bool overTcp,
                        Address server);

    /**
     * Send a query to its current server, over TCP.
     * @param id the identifier of the query
     */
    void SendTcp(uint16_t id);

    /**
     * Receive a response over TCP.
     * @param id the identifier of the query
     * @param socket the socket of the connection
     */
    void ReceiveTcp(uint16_t id, Ptr<Socket> socket);

    /**
     * Handle the closing of the TCP connection of a query by the server, or its failure.
     * @param id the identifier of the query
     * @param socket the socket of the connection
     */
    void HandleTcpClose(uint16_t id, Ptr<Socket> socket);

    /**
     * Close the TCP connection of a query, if any.
     * @param query the query
     */
    static void CloseTcp(Query& query);

    /**
     * Close the UDP sockets of a query, unless they are shared by the queries.
     * @param query the query
     */
    void CloseUdp(Query& query);

    /**
     * Cache an answer, or a resolution failure.
     * @param name the normalized host name
     * @param type the type of the records
     * @param addresses the addresses
     * @param ttl the TTL of the answer
     * @param overwrite whether to replace an unexpired answer with addresses
     * @param failure whether the resolution failed
     */
    void Cache(const std::string& name,
               RecordType type,
               const std::vector<Address>& addresses,
               Time ttl,
               bool overwrite = true,
               bool failure = false);

    /// Remove the expired answers from the cache
    void PurgeCache();

    /**
     * Complete a query, and its resolution when all its queries are complete.
     * @param id the identifier of the query
     * @param addresses the addresses of the records of the query
     */
    void CompleteQuery(uint16_t id, const std::vector<Address>& addresses);

    /**
     * Complete a query whose resolution failed, caching the failure.
     * @param id the identifier of the query
     */
    void FailQuery(uint16_t id);

    /**
     * Report the result of a resolution, and forget it.
     * @param lookup the resolution
     */
    void CompleteLookup(uint64_t lookup);

    /**
     * @param server a server
     * @return the socket address of the server
     */
    Address GetSocketAddress(const Address& server) const;

    /**
     * Get the UDP socket of a query for the address family of a server, created if needed,
     * bound to a random port in the dynamic range or, if none is free, the socket shared by the
     * queries.
     * @param query the query
     * @param server a server
     * @return the socket
     */
    Ptr<Socket> GetUdpSocket(Query& query, const Address& server);

    std::vector<Address> m_servers; ///< addresses of the DNS servers
    uint16_t m_port;                ///< port of the DNS servers
    Time m_timeout;                 ///< timeout of the attempts of the first round of servers
    uint32_t m_retransmissions;     ///< maximum number of retransmissions of a query
    bool m_cacheEnabled;            ///< whether the answers are cached
    Time m_maxTtl;                  ///< maximum time an answer with addresses is cached
    Time m_maxNegativeTtl;          ///< maximum time an answer without address is cached
    Time m_failureTtl;              ///< time a resolution failure is cached
    uint32_t m_maxCacheEntries;     ///< maximum number of cached answers
    uint16_t m_ednsUdpPayloadSize;  ///< EDNS(0) UDP payload size advertised, 0 for no EDNS(0)
    Time m_noEdnsTime;              ///< time a server is known not to support EDNS(0)

    Ptr<Socket> m_udpSocket;                 ///< shared UDP socket for the IPv4 servers
    Ptr<Socket> m_udp6Socket;                ///< shared UDP socket for the IPv6 servers
    Ptr<UniformRandomVariable> m_id;         ///< random query identifiers
    Ptr<UniformRandomVariable> m_sourcePort; ///< random UDP source ports
    std::map<uint16_t, Query> m_queries;     ///< queries in progress, by id
    std::map<uint64_t, Lookup> m_lookups;    ///< resolutions in progress, by id
    uint64_t m_nextLookup{0};                ///< identifier of the next resolution
    std::map<Address, Time> m_noEdnsServers; ///< servers not supporting EDNS(0), until when
    std::map<std::pair<std::string, uint16_t>, CacheEntry> m_cache; ///< cache, by name and type
    /// consecutive resolution failures, by name and type
    std::map<std::pair<std::string, uint16_t>, FailureHistory> m_failures;
    Time m_lastPurge; ///< last purge of the expired answers

    /// Trace fired with the result of each resolution
    TracedCallback<const std::string&, const std::vector<Address>&> m_resolvedTrace;
};

} // namespace ns3

#endif /* DNS_RESOLVER_H */
