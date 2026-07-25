/*
 * Copyright (c) 2005 INRIA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "ipv4-end-point-demux.h"

#include "ipv4-end-point.h"
#include "ipv4-interface-address.h"

#include "ns3/log.h"

#include <algorithm>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("Ipv4EndPointDemux");

Ipv4EndPointDemux::Ipv4EndPointDemux()
    : m_ephemeral(49152),
      m_portLast(65535),
      m_portFirst(49152)
{
    NS_LOG_FUNCTION(this);
}

Ipv4EndPointDemux::~Ipv4EndPointDemux()
{
    NS_LOG_FUNCTION(this);
    for (auto i = m_endPoints.begin(); i != m_endPoints.end(); i++)
    {
        Ipv4EndPoint* endPoint = *i;
        delete endPoint;
    }
    m_endPoints.clear();
}

bool
Ipv4EndPointDemux::LookupPortLocal(uint16_t port)
{
    NS_LOG_FUNCTION(this << port);
    return m_endPointsByPort.find(port) != m_endPointsByPort.end();
}

bool
Ipv4EndPointDemux::LookupLocal(Ptr<NetDevice> boundNetDevice, Ipv4Address addr, uint16_t port)
{
    NS_LOG_FUNCTION(this << addr << port);
    auto bucket = m_endPointsByPort.find(port);
    if (bucket == m_endPointsByPort.end())
    {
        return false;
    }
    for (Ipv4EndPoint* endP : bucket->second)
    {
        if (endP->GetLocalAddress() == addr && endP->GetBoundNetDevice() == boundNetDevice)
        {
            return true;
        }
    }
    return false;
}

Ipv4EndPoint*
Ipv4EndPointDemux::Allocate()
{
    NS_LOG_FUNCTION(this);
    uint16_t port = AllocateEphemeralPort();
    if (port == 0)
    {
        NS_LOG_WARN("Ephemeral port allocation failed.");
        return nullptr;
    }
    auto endPoint = new Ipv4EndPoint(Ipv4Address::GetAny(), port);
    m_endPoints.push_back(endPoint);
    m_endPointsByPort[port].push_back(endPoint);
    NS_LOG_DEBUG("Now have >>" << m_endPoints.size() << "<< endpoints.");
    return endPoint;
}

Ipv4EndPoint*
Ipv4EndPointDemux::Allocate(Ipv4Address address)
{
    NS_LOG_FUNCTION(this << address);
    uint16_t port = AllocateEphemeralPort();
    if (port == 0)
    {
        NS_LOG_WARN("Ephemeral port allocation failed.");
        return nullptr;
    }
    auto endPoint = new Ipv4EndPoint(address, port);
    m_endPoints.push_back(endPoint);
    m_endPointsByPort[port].push_back(endPoint);
    NS_LOG_DEBUG("Now have >>" << m_endPoints.size() << "<< endpoints.");
    return endPoint;
}

Ipv4EndPoint*
Ipv4EndPointDemux::Allocate(Ptr<NetDevice> boundNetDevice, uint16_t port)
{
    NS_LOG_FUNCTION(this << port << boundNetDevice);

    return Allocate(boundNetDevice, Ipv4Address::GetAny(), port);
}

Ipv4EndPoint*
Ipv4EndPointDemux::Allocate(Ptr<NetDevice> boundNetDevice, Ipv4Address address, uint16_t port)
{
    NS_LOG_FUNCTION(this << address << port << boundNetDevice);
    if (LookupLocal(boundNetDevice, address, port) || LookupLocal(nullptr, address, port))
    {
        NS_LOG_WARN("Duplicated endpoint.");
        return nullptr;
    }
    auto endPoint = new Ipv4EndPoint(address, port);
    m_endPoints.push_back(endPoint);
    m_endPointsByPort[port].push_back(endPoint);
    NS_LOG_DEBUG("Now have >>" << m_endPoints.size() << "<< endpoints.");
    return endPoint;
}

Ipv4EndPoint*
Ipv4EndPointDemux::Allocate(Ptr<NetDevice> boundNetDevice,
                            Ipv4Address localAddress,
                            uint16_t localPort,
                            Ipv4Address peerAddress,
                            uint16_t peerPort)
{
    NS_LOG_FUNCTION(this << localAddress << localPort << peerAddress << peerPort << boundNetDevice);
    auto portBucket = m_endPointsByPort.find(localPort);
    if (portBucket != m_endPointsByPort.end())
    {
        for (Ipv4EndPoint* endP : portBucket->second)
        {
            if (endP->GetLocalAddress() == localAddress && endP->GetPeerPort() == peerPort &&
                endP->GetPeerAddress() == peerAddress &&
                (endP->GetBoundNetDevice() == boundNetDevice || !endP->GetBoundNetDevice()))
            {
                NS_LOG_WARN("Duplicated endpoint.");
                return nullptr;
            }
        }
    }
    auto endPoint = new Ipv4EndPoint(localAddress, localPort);
    endPoint->SetPeer(peerAddress, peerPort);
    m_endPoints.push_back(endPoint);
    m_endPointsByPort[localPort].push_back(endPoint);

    NS_LOG_DEBUG("Now have >>" << m_endPoints.size() << "<< endpoints.");

    return endPoint;
}

void
Ipv4EndPointDemux::DeAllocate(Ipv4EndPoint* endPoint)
{
    NS_LOG_FUNCTION(this << endPoint);
    for (auto i = m_endPoints.begin(); i != m_endPoints.end(); i++)
    {
        if (*i == endPoint)
        {
            auto bucket = m_endPointsByPort.find(endPoint->GetLocalPort());
            if (bucket != m_endPointsByPort.end())
            {
                auto& portEndPoints = bucket->second;
                portEndPoints.erase(
                    std::remove(portEndPoints.begin(), portEndPoints.end(), endPoint),
                    portEndPoints.end());
                if (portEndPoints.empty())
                {
                    m_endPointsByPort.erase(bucket);
                }
            }
            delete endPoint;
            m_endPoints.erase(i);
            break;
        }
    }
}

/*
 * return list of all available Endpoints
 */
Ipv4EndPointDemux::EndPoints
Ipv4EndPointDemux::GetAllEndPoints()
{
    NS_LOG_FUNCTION(this);
    EndPoints ret;

    for (auto i = m_endPoints.begin(); i != m_endPoints.end(); i++)
    {
        Ipv4EndPoint* endP = *i;
        ret.push_back(endP);
    }
    return ret;
}

/*
 * If we have an exact match, we return it.
 * Otherwise, if we find a generic match, we return it.
 * Otherwise, we return 0.
 */
Ipv4EndPointDemux::EndPoints
Ipv4EndPointDemux::Lookup(Ipv4Address daddr,
                          uint16_t dport,
                          Ipv4Address saddr,
                          uint16_t sport,
                          Ptr<Ipv4Interface> incomingInterface)
{
    NS_LOG_FUNCTION(this << daddr << dport << saddr << sport << incomingInterface);

    // Best match per specificity class, most specific wins. Classes are:
    // 1: exact on local port, wildcards on others
    // 2: exact on local port/addr, wildcards on others
    // 3: all but local address
    // 4: exact match on all 4
    Ipv4EndPoint* match[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    unsigned int matchCount[5] = {0, 0, 0, 0, 0};

    NS_LOG_DEBUG("Looking up endpoint for destination address " << daddr << ":" << dport);
    auto portBucket = m_endPointsByPort.find(dport);
    if (portBucket == m_endPointsByPort.end())
    {
        return {};
    }
    for (Ipv4EndPoint* endP : portBucket->second)
    {
        NS_LOG_DEBUG("Looking at endpoint dport="
                     << endP->GetLocalPort() << " daddr=" << endP->GetLocalAddress()
                     << " sport=" << endP->GetPeerPort() << " saddr=" << endP->GetPeerAddress());

        if (!endP->IsRxEnabled())
        {
            NS_LOG_LOGIC("Skipping endpoint " << &endP
                                              << " because endpoint can not receive packets");
            continue;
        }
        if (endP->GetBoundNetDevice())
        {
            if (endP->GetBoundNetDevice() != incomingInterface->GetDevice())
            {
                NS_LOG_LOGIC("Skipping endpoint "
                             << &endP << " because endpoint is bound to specific device and"
                             << endP->GetBoundNetDevice() << " does not match packet device "
                             << incomingInterface->GetDevice());
                continue;
            }
        }

        bool localAddressMatchesExact = false;
        bool localAddressIsAny = false;
        bool localAddressIsSubnetAny = false;

        // We have 3 cases:
        // 1) Exact local / destination address match
        // 2) Local endpoint bound to Any -> matches anything
        // 3) Local endpoint bound to x.y.z.0 -> matches Subnet-directed broadcast packet (e.g.,
        // x.y.z.255 in a /24 net) and direct destination match.

        if (endP->GetLocalAddress() == daddr)
        {
            // Case 1:
            localAddressMatchesExact = true;
        }
        else if (endP->GetLocalAddress() == Ipv4Address::GetAny())
        {
            // Case 2:
            localAddressIsAny = true;
        }
        else
        {
            // Case 3:
            for (uint32_t i = 0; i < incomingInterface->GetNAddresses(); i++)
            {
                Ipv4InterfaceAddress addr = incomingInterface->GetAddress(i);

                Ipv4Address addrNetpart = addr.GetLocal().CombineMask(addr.GetMask());
                if (endP->GetLocalAddress() == addrNetpart)
                {
                    NS_LOG_LOGIC("Endpoint is SubnetDirectedAny "
                                 << endP->GetLocalAddress() << "/"
                                 << addr.GetMask().GetPrefixLength());

                    Ipv4Address daddrNetPart = daddr.CombineMask(addr.GetMask());
                    if (addrNetpart == daddrNetPart)
                    {
                        localAddressIsSubnetAny = true;
                    }
                }
            }

            // if no match here, keep looking
            if (!localAddressIsSubnetAny)
            {
                continue;
            }
        }

        bool remotePortMatchesExact = endP->GetPeerPort() == sport;
        bool remotePortMatchesWildCard = endP->GetPeerPort() == 0;
        bool remoteAddressMatchesExact = endP->GetPeerAddress() == saddr;
        bool remoteAddressMatchesWildCard = endP->GetPeerAddress() == Ipv4Address::GetAny();

        // If remote does not match either with exact or wildcard,
        // skip this one
        if (!(remotePortMatchesExact || remotePortMatchesWildCard))
        {
            continue;
        }
        if (!(remoteAddressMatchesExact || remoteAddressMatchesWildCard))
        {
            continue;
        }

        bool localAddressMatchesWildCard = localAddressIsAny || localAddressIsSubnetAny;

        if (localAddressMatchesExact && remoteAddressMatchesExact && remotePortMatchesExact)
        { // All 4 match - this is the case of an open TCP connection, for example.
            NS_LOG_LOGIC("Found an endpoint for case 4, adding " << endP->GetLocalAddress() << ":"
                                                                 << endP->GetLocalPort());
            match[4] = endP;
            matchCount[4]++;
        }
        if (localAddressMatchesWildCard && remoteAddressMatchesExact && remotePortMatchesExact)
        { // All but local address - no idea what this case could be.
            NS_LOG_LOGIC("Found an endpoint for case 3, adding " << endP->GetLocalAddress() << ":"
                                                                 << endP->GetLocalPort());
            match[3] = endP;
            matchCount[3]++;
        }
        if (localAddressMatchesExact && remoteAddressMatchesWildCard && remotePortMatchesWildCard)
        { // Only local port and local address matches exactly - Not yet opened connection
            NS_LOG_LOGIC("Found an endpoint for case 2, adding " << endP->GetLocalAddress() << ":"
                                                                 << endP->GetLocalPort());
            match[2] = endP;
            matchCount[2]++;
        }
        if (localAddressMatchesWildCard && remoteAddressMatchesWildCard &&
            remotePortMatchesWildCard)
        { // Only local port matches exactly - Endpoint open to "any" connection
            NS_LOG_LOGIC("Found an endpoint for case 1, adding " << endP->GetLocalAddress() << ":"
                                                                 << endP->GetLocalPort());
            match[1] = endP;
            matchCount[1]++;
        }
    }

    // Here we find the most exact match
    EndPoints retval;
    for (int specificity = 4; specificity >= 1; specificity--)
    {
        if (matchCount[specificity] > 0)
        {
            NS_ABORT_MSG_IF(matchCount[specificity] > 1,
                            "Too many endpoints - perhaps you created too many sockets without "
                            "binding them to different NetDevices.");
            retval.push_back(match[specificity]);
            break;
        }
    }
    return retval; // might be empty if no matches
}

Ipv4EndPoint*
Ipv4EndPointDemux::SimpleLookup(Ipv4Address daddr,
                                uint16_t dport,
                                Ipv4Address saddr,
                                uint16_t sport)
{
    NS_LOG_FUNCTION(this << daddr << dport << saddr << sport);

    // this code is a copy/paste version of an old BSD ip stack lookup
    // function.
    uint32_t genericity = 3;
    Ipv4EndPoint* generic = nullptr;
    auto portBucket = m_endPointsByPort.find(dport);
    if (portBucket == m_endPointsByPort.end())
    {
        return nullptr;
    }
    for (Ipv4EndPoint* endP : portBucket->second)
    {
        if (endP->GetLocalAddress() == daddr && endP->GetPeerPort() == sport &&
            endP->GetPeerAddress() == saddr)
        {
            /* this is an exact match. */
            return endP;
        }
        uint32_t tmp = 0;
        if (endP->GetLocalAddress() == Ipv4Address::GetAny())
        {
            tmp++;
        }
        if (endP->GetPeerAddress() == Ipv4Address::GetAny())
        {
            tmp++;
        }
        if (tmp < genericity)
        {
            generic = endP;
            genericity = tmp;
        }
    }
    return generic;
}

uint16_t
Ipv4EndPointDemux::AllocateEphemeralPort()
{
    // Similar to counting up logic in netinet/in_pcb.c
    NS_LOG_FUNCTION(this);
    uint16_t port = m_ephemeral;
    int count = m_portLast - m_portFirst;
    do
    {
        if (count-- < 0)
        {
            return 0;
        }
        ++port;
        if (port < m_portFirst || port > m_portLast)
        {
            port = m_portFirst;
        }
    } while (LookupPortLocal(port));
    m_ephemeral = port;
    return port;
}

} // namespace ns3
