/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
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
 * @brief Helper to install a DnsResolver on a set of nodes.
 */
class DnsResolverHelper
{
  public:
    /**
     * Constructor.
     *
     * @param server DNS server address (Ipv4Address or Ipv6Address).
     */
    DnsResolverHelper(const Address& server);

    /**
     * Constructor.
     *
     * @param servers DNS server addresses, in order of preference.
     */
    DnsResolverHelper(const std::vector<Address>& servers);

    /**
     * Append a DNS server to the list.
     *
     * @param server Server address (Ipv4Address or Ipv6Address).
     */
    void AddServer(const Address& server);

    /**
     * Set an attribute of the DnsResolver objects to be created.
     *
     * @param name Attribute name.
     * @param value Attribute value.
     */
    void SetAttribute(const std::string& name, const AttributeValue& value);

    /**
     * Create a DnsResolver and aggregate it to each node. It is an error to install a
     * resolver on a node that already has one.
     *
     * @param nodes The nodes.
     */
    void Install(NodeContainer nodes) const;

    /**
     * Assign a fixed random variable stream number to the random variables used by the
     * resolvers installed on the nodes.
     *
     * @param nodes The nodes.
     * @param stream First stream index to use.
     * @return The number of stream indices assigned.
     */
    static int64_t AssignStreams(NodeContainer nodes, int64_t stream);

  private:
    ObjectFactory m_factory;        ///< DnsResolver factory
    std::vector<Address> m_servers; ///< DNS servers
};

} // namespace ns3

#endif /* DNS_RESOLVER_HELPER_H */
