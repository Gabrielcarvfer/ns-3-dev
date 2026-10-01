/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "dns-resolver-helper.h"

#include "ns3/abort.h"
#include "ns3/dns-resolver.h"
#include "ns3/node.h"

namespace ns3
{

DnsResolverHelper::DnsResolverHelper(const Address& server)
    : DnsResolverHelper(std::vector<Address>{server})
{
}

DnsResolverHelper::DnsResolverHelper(const std::vector<Address>& servers)
    : m_servers(servers)
{
    m_factory.SetTypeId(DnsResolver::GetTypeId());
}

void
DnsResolverHelper::AddServer(const Address& server)
{
    m_servers.push_back(server);
}

void
DnsResolverHelper::SetAttribute(const std::string& name, const AttributeValue& value)
{
    m_factory.Set(name, value);
}

void
DnsResolverHelper::Install(NodeContainer nodes) const
{
    for (auto it = nodes.Begin(); it != nodes.End(); ++it)
    {
        NS_ABORT_MSG_IF((*it)->GetObject<DnsResolver>(),
                        "Node " << (*it)->GetId() << " already has a DNS resolver");
        auto resolver = m_factory.Create<DnsResolver>();
        resolver->SetServers(m_servers);
        (*it)->AggregateObject(resolver);
    }
}

int64_t
DnsResolverHelper::AssignStreams(NodeContainer nodes, int64_t stream)
{
    int64_t currentStream = stream;
    for (auto it = nodes.Begin(); it != nodes.End(); ++it)
    {
        if (auto resolver = (*it)->GetObject<DnsResolver>())
        {
            currentStream += resolver->AssignStreams(currentStream);
        }
    }
    return currentStream - stream;
}

} // namespace ns3
