/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

#ifndef DNS_RESOLVER_HELPER_H
#define DNS_RESOLVER_HELPER_H

#include "ns3/address.h"
#include "ns3/attribute.h"
#include "ns3/node-container.h"
#include "ns3/object-factory.h"

#include <string>
#include <vector>

namespace ns3
{

/**
 * @ingroup dns-resolver
 * @brief Aggregates a DnsResolver to nodes.
 */
class DnsResolverHelper
{
  public:
    /**
     * Constructor.
     * @param server the address of the DNS server (an Ipv4Address or an Ipv6Address)
     */
    DnsResolverHelper(const Address& server);

    /**
     * Constructor.
     * @param servers the addresses of the DNS servers, in the order in which they are queried
     */
    DnsResolverHelper(const std::vector<Address>& servers);

    /**
     * Add a DNS server, queried after the ones already set.
     * @param server the address of the server (an Ipv4Address or an Ipv6Address)
     */
    void AddServer(const Address& server);

    /**
     * Set an attribute of the resolvers to be installed.
     * @param name the name of the attribute
     * @param value the value of the attribute
     */
    void SetAttribute(const std::string& name, const AttributeValue& value);

    /**
     * Aggregate a DnsResolver to each node, which must not have one already.
     * @param nodes the nodes
     */
    void Install(NodeContainer nodes) const;

    /**
     * Assign a fixed random variable stream number to the random variables used by the
     * resolvers of the nodes.
     * @param nodes the nodes
     * @param stream first stream index to use
     * @return the number of stream indices assigned
     */
    static int64_t AssignStreams(NodeContainer nodes, int64_t stream);

  private:
    ObjectFactory m_factory;        ///< factory of the resolvers
    std::vector<Address> m_servers; ///< addresses of the DNS servers
};

} // namespace ns3

#endif /* DNS_RESOLVER_HELPER_H */
