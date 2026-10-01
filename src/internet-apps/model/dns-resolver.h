/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#ifndef DNS_RESOLVER_H
#define DNS_RESOLVER_H

#include "dns-header.h"

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
     * disposed of first. An IPv4 or IPv6 address literal is returned as is, without query, if
     * it belongs to the requested family.
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

    /// The content of a response to a query for the addresses of a name
    struct Answer
    {
        /// response code, SERVFAIL if the records are inconsistent
        uint16_t rcode{DnsHeader::RCODE_NOERROR};
        /// name at the end of the CNAME chain of the queried name (the queried name if none)
        std::string canonicalName;
        Time chainTtl{Time::Max()};     ///< minimum TTL of the CNAME chain, Time::Max() if none
        std::vector<Address> addresses; ///< addresses of the queried name, or of its alias
        /// TTL of the answer: the minimum TTL of the alias and address records, or the negative
        /// TTL of the SOA record when there is no address; zero if unknown
        Time ttl;
        bool hasSoa{false}; ///< whether the authority section has the SOA record of the zone
        bool hasNs{false};  ///< whether the authority section has NS records
    };

    /**
     * Check whether a message is the response to a query (RFC 5452, section 9.1).
     *
     * A response without question is accepted only with the FORMERR and NOTIMP response codes
     * and no answer, as sent by servers not supporting EDNS(0) (RFC 6891, section 7).
     *
     * @param response the message
     * @param id the identifier of the query
     * @param name the name of the query
     * @param type the type of the records of the query
     * @return true if the message is the response to the query
     */
    static bool IsResponseTo(const DnsHeader& response,
                             uint16_t id,
                             const std::string& name,
                             DnsHeader::RecordType type);

    /**
     * Read the answer of a response to a query, following the CNAME chain of the queried name
     * (RFC 1034, section 3.6.2) and finding the negative TTL (RFC 2308, section 5).
     *
     * A response whose records are inconsistent (e.g., several aliases for a name, or an alias
     * loop) has the SERVFAIL response code. The duplicate addresses are removed. The names are
     * normalized: in lower case, without final dot.
     *
     * @param response the response, with the question of the query
     * @param name the name of the query
     * @param type the type of the records of the query (TYPE_A or TYPE_AAAA)
     * @return the answer
     */
    static Answer ReadAnswer(const DnsHeader& response,
                             const std::string& name,
                             DnsHeader::RecordType type);

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
        std::vector<uint64_t> lookups; ///< the resolutions sharing the query
        std::string name;              ///< the normalized host name, as requested
        std::string question;          ///< the name queried (the target of an alias)
        DnsHeader::RecordType type{DnsHeader::TYPE_A}; ///< the type of the records
        uint32_t retransmissions{0};                   ///< number of retransmissions so far
        uint32_t restarts{0};                          ///< number of queries for alias targets
        Time chainTtl{Time::Max()};                    ///< minimum TTL of the aliases followed
        size_t server{0};                              ///< index of the current server
        Address current;                               ///< the current server
        bool currentEdns{false};                       ///< whether the current attempt uses EDNS(0)
        bool noEdns{false};                   ///< whether EDNS(0) was rejected for this query
        std::map<Address, uint32_t> attempts; ///< number of attempts, by server queried
        Ptr<Socket> udpSocket;                ///< the UDP socket for the IPv4 servers
        Ptr<Socket> udp6Socket;               ///< the UDP socket for the IPv6 servers
        EventId timeout;                      ///< the timeout of the current attempt
        Ptr<Socket> tcpSocket;                ///< the TCP connection, for truncated responses
        Ptr<Packet> tcpData;                  ///< the data received on the TCP connection
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

    /// Key of the cache: a normalized name and a record type
    using CacheKey = std::pair<std::string, uint16_t>;

    /**
     * Start a query for the records of a name.
     * @param lookup the resolution of the query
     * @param name the normalized host name
     * @param type the type of the records
     */
    void StartQuery(uint64_t lookup, const std::string& name, DnsHeader::RecordType type);

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
     * Build the message of a query for its current server, with EDNS(0) unless the server or
     * the query rejected it.
     * @param id the identifier of the query
     * @return the message
     */
    DnsHeader BuildQuery(uint16_t id);

    /**
     * Send a query over UDP to its current server.
     * @param id the identifier of the query
     */
    void SendUdp(uint16_t id);

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
    void QueryTarget(uint16_t id, const std::string& target, Time chainTtl);

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
     * @param response the response
     * @param overTcp whether the response was received over TCP
     * @param server the server that sent the response
     */
    void HandleResponse(uint16_t id,
                        const DnsHeader& response,
                        bool overTcp,
                        const Address& server);

    /**
     * Handle the answer of a response with the question of a query.
     * @param id the identifier of the query
     * @param response the response
     * @param answer the answer
     * @param server the server that sent the response
     */
    void HandleAnswer(uint16_t id,
                      const DnsHeader& response,
                      const Answer& answer,
                      const Address& server);

    /**
     * Send a query to its current server, over TCP.
     * @param id the identifier of the query
     */
    void SendTcp(uint16_t id);

    /**
     * @param socket the socket of a TCP connection
     * @return the query of the connection, or the end of the queries
     */
    std::map<uint16_t, Query>::iterator FindTcpQuery(Ptr<Socket> socket);

    /**
     * Send a query over its TCP connection, once established.
     * @param socket the socket of the connection
     */
    void HandleTcpConnected(Ptr<Socket> socket);

    /**
     * Receive a response over TCP.
     * @param socket the socket of the connection
     */
    void ReceiveTcp(Ptr<Socket> socket);

    /**
     * Handle the closing of the TCP connection of a query by the server, or its failure.
     * @param socket the socket of the connection
     */
    void HandleTcpClose(Ptr<Socket> socket);

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
     * @param key the name and type of the records
     * @param addresses the addresses
     * @param ttl the TTL of the answer
     * @param overwrite whether to replace an unexpired answer with addresses
     * @param failure whether the resolution failed
     */
    void Cache(const CacheKey& key,
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

    Ptr<Socket> m_udpSocket;                       ///< shared UDP socket for the IPv4 servers
    Ptr<Socket> m_udp6Socket;                      ///< shared UDP socket for the IPv6 servers
    Ptr<UniformRandomVariable> m_id;               ///< random query identifiers
    Ptr<UniformRandomVariable> m_sourcePort;       ///< random UDP source ports
    std::map<uint16_t, Query> m_queries;           ///< queries in progress, by id
    std::map<uint64_t, Lookup> m_lookups;          ///< resolutions in progress, by id
    uint64_t m_nextLookup{0};                      ///< identifier of the next resolution
    std::map<Address, Time> m_noEdnsServers;       ///< servers not supporting EDNS(0), until when
    std::map<CacheKey, CacheEntry> m_cache;        ///< cached answers and failures
    std::map<CacheKey, FailureHistory> m_failures; ///< consecutive resolution failures
    Time m_lastPurge;                              ///< last purge of the expired answers

    /// Trace fired with the result of each resolution
    TracedCallback<const std::string&, const std::vector<Address>&> m_resolvedTrace;
};

} // namespace ns3

#endif /* DNS_RESOLVER_H */
