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
 * @brief DNS stub resolver for IPv4 and IPv6 host addresses.
 *
 * The resolver is aggregated to a node and sends recursive A and AAAA queries to one or more
 * DNS servers (RFC 1035). Queries are sent over UDP, and repeated over TCP when the response
 * is truncated (RFC 7766). Queries carry an EDNS(0) OPT record (RFC 6891), unless the server
 * is known not to support it.
 *
 * To make spoofing harder, each query uses a random ID and a random UDP source port, and a
 * response is only accepted if it comes from a server the query was sent to and matches the
 * query ID and question (RFC 5452). Concurrent requests for the same name and record type
 * share a single query.
 *
 * CNAME chains are followed. If a chain ends without the records of its target, a new query
 * is sent for the target.
 *
 * If no response arrives within the Timeout attribute, or the server returns an error (e.g.,
 * SERVFAIL, REFUSED or a referral), the query is retransmitted to the next server, up to the
 * Retransmissions attribute, and at most three times per server (RFC 9520). The timeout
 * doubles after each round over the server list.
 *
 * Positive answers are cached for the smallest TTL of their records, capped by MaxTtl.
 * Negative answers (NXDOMAIN or NODATA) are cached for the SOA negative TTL (RFC 2308),
 * capped by MaxNegativeTtl. Failures are cached for FailureTtl, doubled after each
 * consecutive failure up to five minutes (RFC 9520).
 *
 * The result is reported to a callback with the list of addresses (Ipv6Address entries
 * first, then Ipv4Address entries). The list is empty if the resolution failed.
 */
class DnsResolver : public Object
{
  public:
    /// Address families to resolve
    enum AddressFamily : uint8_t
    {
        IPV4, ///< IPv4 only (A records)
        IPV6, ///< IPv6 only (AAAA records)
        ANY,  ///< IPv6 and IPv4 (AAAA and A records)
    };

    /**
     * Callback invoked with the result of a resolution.
     *
     * The arguments are the host name and the resolved addresses, which are empty if the
     * resolution failed.
     */
    using ResolveCallback = Callback<void, const std::string&, const std::vector<Address>&>;

    /**
     * TracedCallback signature for resolution results.
     *
     * @param [in] name Host name.
     * @param [in] addresses Resolved addresses, empty on failure.
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
     * Set the list of DNS servers, in order of preference.
     *
     * Cached failures are discarded. Queries in progress continue with the new list; if it is
     * empty, they fail once their current attempt is abandoned.
     *
     * @param servers Server addresses (Ipv4Address or Ipv6Address).
     */
    void SetServers(const std::vector<Address>& servers);

    /**
     * Append a DNS server to the list.
     *
     * @param server Server address (Ipv4Address or Ipv6Address).
     */
    void AddServer(const Address& server);

    /**
     * Get the list of DNS servers.
     *
     * @return The server addresses.
     */
    std::vector<Address> GetServers() const;

    /**
     * Resolve a host name.
     *
     * The resolver must be aggregated to a node with an internet stack. The callback, if not
     * null, is always invoked after this method returns, including when the result is cached,
     * the name is invalid or there are no servers (in which case the result is empty). It is
     * not invoked if the resolver is disposed first.
     *
     * An IPv4 or IPv6 address literal of the requested family is returned without sending a
     * query.
     *
     * @param name Host name, e.g., "www.nsnam.org", optionally with a trailing dot.
     * @param callback Callback invoked with the result.
     * @param family Address family to resolve.
     */
    void Resolve(const std::string& name, ResolveCallback callback, AddressFamily family = IPV4);

    /**
     * Remove all cached answers and failures.
     */
    void FlushCache();

    /**
     * Assign a fixed random variable stream number to the random variables used by this
     * object (query IDs and UDP source ports).
     *
     * @param stream First stream index to use.
     * @return The number of stream indices assigned by this object.
     */
    int64_t AssignStreams(int64_t stream);

    /**
     * Check whether a string is a valid host name (RFC 1123, section 2.1).
     *
     * A valid host name has at most 253 characters and is made of dot-separated labels of 1 to
     * 63 letters, digits and hyphens, which do not start or end with a hyphen. The last label
     * must not be all-numeric, so that the name cannot be confused with an IPv4 address. A
     * trailing dot is allowed.
     *
     * @param name The name to check.
     * @return True if the name is a valid host name.
     */
    static bool IsValidHostName(const std::string& name);

    /**
     * Information extracted from the response to an A or AAAA query.
     */
    struct Answer
    {
        uint16_t rcode{DnsHeader::RCODE_NOERROR}; ///< Response code (SERVFAIL if inconsistent)
        std::string canonicalName;                ///< End of the CNAME chain, or the queried name
        Time chainTtl{Time::Max()};     ///< Smallest TTL in the CNAME chain (Max if no chain)
        std::vector<Address> addresses; ///< Addresses of the canonical name
        Time ttl;           ///< Cache lifetime: positive or negative TTL, or zero if unknown
        bool hasSoa{false}; ///< True if the authority section has the SOA of the zone
        bool hasNs{false};  ///< True if the authority section has NS records
    };

    /**
     * Check whether a message is a valid response to a query (RFC 5452, section 9.1).
     *
     * A response without a question section is only accepted if it has no answers and its
     * response code is FORMERR or NOTIMP, which is how servers without EDNS(0) support may
     * reply (RFC 6891, section 7).
     *
     * @param response Received message.
     * @param id Query ID.
     * @param name Queried name.
     * @param type Queried record type.
     * @return True if the message is a response to the query.
     */
    static bool IsResponseTo(const DnsHeader& response,
                             uint16_t id,
                             const std::string& name,
                             DnsHeader::RecordType type);

    /**
     * Extract the answer from a response, following the CNAME chain of the queried name and
     * computing the negative TTL when there are no addresses.
     *
     * Inconsistent responses (e.g., a name with several CNAMEs, or a CNAME loop) are reported
     * as SERVFAIL. Duplicate addresses are removed, and names are normalized to lower case
     * without a trailing dot.
     *
     * @param response Response matching the query.
     * @param name Queried name.
     * @param type Queried record type (TYPE_A or TYPE_AAAA).
     * @return The extracted answer.
     */
    static Answer ReadAnswer(const DnsHeader& response,
                             const std::string& name,
                             DnsHeader::RecordType type);

  protected:
    void DoDispose() override;

  private:
    /// A resolution request, made of one query per record type
    struct Lookup
    {
        std::string name;           ///< Host name, as passed to Resolve()
        ResolveCallback callback;   ///< Callback to invoke with the result
        uint32_t pendingQueries{0}; ///< Number of queries still in progress
        std::vector<Address> ipv6;  ///< Resolved IPv6 addresses
        std::vector<Address> ipv4;  ///< Resolved IPv4 addresses
        EventId completion;         ///< Event reporting a result that needed no query
    };

    /// A query in progress
    struct Query
    {
        std::vector<uint64_t> lookups; ///< Lookups waiting on this query
        std::string name;              ///< Normalized requested name
        std::string question;          ///< Name being queried (may be a CNAME target)
        DnsHeader::RecordType type{DnsHeader::TYPE_A}; ///< Queried record type
        uint32_t retransmissions{0};                   ///< Retransmissions so far
        uint32_t restarts{0};                          ///< CNAME targets queried so far
        Time chainTtl{Time::Max()};                    ///< Smallest TTL of the CNAMEs followed
        size_t server{0};                              ///< Index of the current server
        Address current;                               ///< Current server
        bool currentEdns{false};                       ///< True if the current attempt uses EDNS(0)
        bool noEdns{false};                   ///< True if EDNS(0) was rejected for this query
        std::map<Address, uint32_t> attempts; ///< Attempts per server
        Ptr<Socket> udpSocket;                ///< UDP socket for IPv4 servers
        Ptr<Socket> udp6Socket;               ///< UDP socket for IPv6 servers
        EventId timeout;                      ///< Timeout of the current attempt
        Ptr<Socket> tcpSocket;                ///< TCP socket, used after a truncated response
        Ptr<Packet> tcpData;                  ///< Data received over TCP so far
    };

    /// A cached answer or failure
    struct CacheEntry
    {
        std::vector<Address> addresses; ///< Cached addresses
        Time expiry;                    ///< Expiration time
        bool failure{false};            ///< True if this is a cached failure

        /**
         * Check whether the entry has expired.
         *
         * @return True if the entry has expired.
         */
        bool IsExpired() const;
    };

    /// Consecutive failures for a name and record type
    struct FailureHistory
    {
        uint32_t count{0}; ///< Number of consecutive failures
        Time expiry;       ///< Expiration time of the cached failure
    };

    /// Cache key: normalized name and record type
    using CacheKey = std::pair<std::string, uint16_t>;

    /**
     * Start a new query.
     *
     * @param lookup Lookup ID.
     * @param name Normalized host name.
     * @param type Record type.
     */
    void StartQuery(uint64_t lookup, const std::string& name, DnsHeader::RecordType type);

    /**
     * Pick a random query ID that is not in use.
     *
     * @return The query ID.
     */
    uint16_t NewQueryId();

    /**
     * Check whether an address is in the server list.
     *
     * @param server Server address.
     * @return True if the address is a configured server.
     */
    bool IsServer(const Address& server) const;

    /**
     * Build the query message for the current server, with an OPT record unless EDNS(0) is
     * disabled or was rejected.
     *
     * @param id Query ID.
     * @return The query message.
     */
    DnsHeader BuildQuery(uint16_t id);

    /**
     * Send a query over UDP to its current server.
     *
     * @param id Query ID.
     */
    void SendUdp(uint16_t id);

    /**
     * Get the timeout of the current attempt, doubled after each round over the servers.
     *
     * @param query The query.
     * @return The timeout.
     */
    Time GetAttemptTimeout(const Query& query) const;

    /**
     * Abandon the current attempt by scheduling an immediate timeout.
     *
     * @param id Query ID.
     */
    void AbandonAttempt(uint16_t id);

    /**
     * Retransmit a query to the next server, or fail it if no attempts are left.
     *
     * @param id Query ID.
     */
    void Retransmit(uint16_t id);

    /**
     * Replace a query with a new one for a CNAME target.
     *
     * @param id Query ID.
     * @param target CNAME target.
     * @param chainTtl Smallest TTL of the CNAME chain.
     */
    void QueryTarget(uint16_t id, const std::string& target, Time chainTtl);

    /**
     * Handle the timeout of an attempt.
     *
     * @param id Query ID.
     */
    void HandleTimeout(uint16_t id);

    /**
     * Receive callback for the UDP sockets.
     *
     * @param socket The receiving socket.
     */
    void ReceiveUdp(Ptr<Socket> socket);

    /**
     * Handle a response received over UDP or TCP.
     *
     * @param id Query ID.
     * @param response The response.
     * @param overTcp True if the response was received over TCP.
     * @param server Server that sent the response.
     */
    void HandleResponse(uint16_t id,
                        const DnsHeader& response,
                        bool overTcp,
                        const Address& server);

    /**
     * Handle the answer extracted from a valid response.
     *
     * @param id Query ID.
     * @param response The response.
     * @param answer The extracted answer.
     * @param server Server that sent the response.
     */
    void HandleAnswer(uint16_t id,
                      const DnsHeader& response,
                      const Answer& answer,
                      const Address& server);

    /**
     * Resend a query to its current server over TCP.
     *
     * @param id Query ID.
     */
    void SendTcp(uint16_t id);

    /**
     * Find the query that owns a TCP socket.
     *
     * @param socket TCP socket.
     * @return Iterator to the query, or m_queries.end() if not found.
     */
    std::map<uint16_t, Query>::iterator FindTcpQuery(Ptr<Socket> socket);

    /**
     * Find the query in progress for a name and record type.
     *
     * @param name Normalized host name.
     * @param type Record type.
     * @return The query, or nullptr if there is none.
     */
    Query* FindQuery(const std::string& name, DnsHeader::RecordType type);

    /**
     * Connection succeeded callback: send the query over TCP.
     *
     * @param socket TCP socket.
     */
    void HandleTcpConnected(Ptr<Socket> socket);

    /**
     * Receive callback for the TCP sockets.
     *
     * @param socket TCP socket.
     */
    void ReceiveTcp(Ptr<Socket> socket);

    /**
     * Handle a TCP connection that failed or was closed by the server.
     *
     * @param socket TCP socket.
     */
    void HandleTcpClose(Ptr<Socket> socket);

    /**
     * Close the TCP connection of a query, if any.
     *
     * @param query The query.
     */
    static void CloseTcp(Query& query);

    /**
     * Close the UDP sockets of a query, unless they are the shared sockets.
     *
     * @param query The query.
     */
    void CloseUdp(Query& query);

    /**
     * Add an answer or a failure to the cache.
     *
     * @param key Name and record type.
     * @param addresses Addresses to cache.
     * @param ttl Cache lifetime.
     * @param overwrite If false, an unexpired entry with addresses is kept.
     * @param failure True to cache a failure.
     */
    void Cache(const CacheKey& key,
               const std::vector<Address>& addresses,
               Time ttl,
               bool overwrite = true,
               bool failure = false);

    /**
     * Remove expired entries from the cache.
     */
    void PurgeCache();

    /**
     * Complete a query, and its lookups once all their queries are complete.
     *
     * @param id Query ID.
     * @param addresses Resolved addresses.
     */
    void CompleteQuery(uint16_t id, const std::vector<Address>& addresses);

    /**
     * Complete a failed query and cache the failure.
     *
     * @param id Query ID.
     */
    void FailQuery(uint16_t id);

    /**
     * Report the result of a lookup and remove it.
     *
     * @param lookup Lookup ID.
     */
    void CompleteLookup(uint64_t lookup);

    /**
     * Get the socket address (address and DNS port) of a server.
     *
     * @param server Server address.
     * @return The socket address.
     */
    Address GetSocketAddress(const Address& server) const;

    /**
     * Get the UDP socket of a query for the address family of a server, creating it if needed.
     *
     * New sockets are bound to a random port in the dynamic range. If no free port is found,
     * the socket shared by all queries is returned instead.
     *
     * @param query The query.
     * @param server Server address.
     * @return The socket.
     */
    Ptr<Socket> GetUdpSocket(Query& query, const Address& server);

    std::vector<Address> m_servers; ///< DNS servers
    uint16_t m_port;                ///< DNS server port
    Time m_timeout;                 ///< Initial attempt timeout
    uint32_t m_retransmissions;     ///< Maximum number of retransmissions per query
    bool m_cacheEnabled;            ///< True if answers are cached
    Time m_maxTtl;                  ///< Maximum cache lifetime of positive answers
    Time m_maxNegativeTtl;          ///< Maximum cache lifetime of negative answers
    Time m_failureTtl;              ///< Initial cache lifetime of failures
    uint32_t m_maxCacheEntries;     ///< Maximum number of cache entries
    uint16_t m_ednsUdpPayloadSize;  ///< Advertised EDNS(0) UDP payload size (0 disables EDNS)
    Time m_noEdnsTime;              ///< How long a server rejecting EDNS(0) is remembered

    Ptr<Socket> m_udpSocket;                       ///< Shared UDP socket for IPv4 servers
    Ptr<Socket> m_udp6Socket;                      ///< Shared UDP socket for IPv6 servers
    Ptr<UniformRandomVariable> m_id;               ///< Random query IDs
    Ptr<UniformRandomVariable> m_sourcePort;       ///< Random UDP source ports
    std::map<uint16_t, Query> m_queries;           ///< Queries in progress, by query ID
    std::map<uint64_t, Lookup> m_lookups;          ///< Lookups in progress, by lookup ID
    uint64_t m_nextLookup{0};                      ///< Next lookup ID
    std::map<Address, Time> m_noEdnsServers;       ///< Servers without EDNS(0), and until when
    std::map<CacheKey, CacheEntry> m_cache;        ///< Cached answers and failures
    std::map<CacheKey, FailureHistory> m_failures; ///< Consecutive failures
    Time m_lastPurge;                              ///< Time of the last cache purge

    /// Trace source fired with the result of each resolution
    TracedCallback<const std::string&, const std::vector<Address>&> m_resolvedTrace;
};

} // namespace ns3

#endif /* DNS_RESOLVER_H */
