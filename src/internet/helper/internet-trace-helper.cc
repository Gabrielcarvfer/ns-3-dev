/*
 * Copyright (c) 2010 University of Washington
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "internet-trace-helper.h"

#include "ns3/abort.h"
#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/names.h"
#include "ns3/net-device.h"
#include "ns3/node.h"
#include "ns3/pcap-file-wrapper.h"
#include "ns3/ptr.h"

#include <cstdint>
#include <fstream>
#include <string>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("InternetTraceHelper");

void
PcapHelperForIpv4::EnablePcapIpv4(std::string prefix,
                                  Ptr<Ipv4> ipv4,
                                  uint32_t interface,
                                  bool explicitFilename)
{
    EnablePcapIpv4Internal(prefix, ipv4, interface, explicitFilename);
}

void
PcapHelperForIpv4::EnablePcapIpv4(std::string prefix,
                                  std::string ipv4Name,
                                  uint32_t interface,
                                  bool explicitFilename)
{
    Ptr<Ipv4> ipv4 = Names::Find<Ipv4>(ipv4Name);
    EnablePcapIpv4(prefix, ipv4, interface, explicitFilename);
}

void
PcapHelperForIpv4::EnablePcapIpv4(std::string prefix, Ipv4InterfaceContainer c)
{
    for (auto i = c.Begin(); i != c.End(); ++i)
    {
        EnablePcapIpv4(prefix, (*i).first, (*i).second, false);
    }
}

void
PcapHelperForIpv4::EnablePcapIpv4(std::string prefix, NodeContainer n)
{
    for (auto i = n.Begin(); i != n.End(); ++i)
    {
        Ptr<Node> node = *i;
        Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
        if (ipv4)
        {
            for (uint32_t j = 0; j < ipv4->GetNInterfaces(); ++j)
            {
                EnablePcapIpv4(prefix, ipv4, j, false);
            }
        }
    }
}

void
PcapHelperForIpv4::EnablePcapIpv4All(std::string prefix)
{
    EnablePcapIpv4(prefix, NodeContainer::GetGlobal());
}

void
PcapHelperForIpv4::EnablePcapIpv4(std::string prefix,
                                  uint32_t nodeid,
                                  uint32_t interface,
                                  bool explicitFilename)
{
    NodeContainer n = NodeContainer::GetGlobal();

    for (auto i = n.Begin(); i != n.End(); ++i)
    {
        Ptr<Node> node = *i;
        if (node->GetId() != nodeid)
        {
            continue;
        }

        Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
        if (ipv4)
        {
            EnablePcapIpv4(prefix, ipv4, interface, explicitFilename);
        }
        return;
    }
}

//
// Public API
//
void
AsciiTraceHelperForIpv4::EnableAsciiIpv4(std::string prefix,
                                         Ptr<Ipv4> ipv4,
                                         uint32_t interface,
                                         bool explicitFilename)
{
    EnableAsciiIpv4Internal(Ptr<OutputStreamWrapper>(), prefix, ipv4, interface, explicitFilename);
}

//
// Public API
//
void
AsciiTraceHelperForIpv4::EnableAsciiIpv4(Ptr<OutputStreamWrapper> stream,
                                         Ptr<Ipv4> ipv4,
                                         uint32_t interface)
{
    EnableAsciiIpv4Internal(stream, std::string(), ipv4, interface, false);
}

//
// Public API
//
void
AsciiTraceHelperForIpv4::EnableAsciiIpv4(std::string prefix,
                                         std::string ipv4Name,
                                         uint32_t interface,
                                         bool explicitFilename)
{
    EnableAsciiIpv4Impl(Ptr<OutputStreamWrapper>(), prefix, ipv4Name, interface, explicitFilename);
}

//
// Public API
//
void
AsciiTraceHelperForIpv4::EnableAsciiIpv4(Ptr<OutputStreamWrapper> stream,
                                         std::string ipv4Name,
                                         uint32_t interface)
{
    EnableAsciiIpv4Impl(stream, std::string(), ipv4Name, interface, false);
}

//
// Private API
//
void
AsciiTraceHelperForIpv4::EnableAsciiIpv4Impl(Ptr<OutputStreamWrapper> stream,
                                             std::string prefix,
                                             std::string ipv4Name,
                                             uint32_t interface,
                                             bool explicitFilename)
{
    Ptr<Ipv4> ipv4 = Names::Find<Ipv4>(ipv4Name);
    EnableAsciiIpv4Internal(stream, prefix, ipv4, interface, explicitFilename);
}

//
// Public API
//
void
AsciiTraceHelperForIpv4::EnableAsciiIpv4(std::string prefix, Ipv4InterfaceContainer c)
{
    EnableAsciiIpv4Impl(Ptr<OutputStreamWrapper>(), prefix, c);
}

//
// Public API
//
void
AsciiTraceHelperForIpv4::EnableAsciiIpv4(Ptr<OutputStreamWrapper> stream, Ipv4InterfaceContainer c)
{
    EnableAsciiIpv4Impl(stream, std::string(), c);
}

//
// Private API
//
void
AsciiTraceHelperForIpv4::EnableAsciiIpv4Impl(Ptr<OutputStreamWrapper> stream,
                                             std::string prefix,
                                             Ipv4InterfaceContainer c)
{
    for (auto i = c.Begin(); i != c.End(); ++i)
    {
        EnableAsciiIpv4Internal(stream, prefix, (*i).first, (*i).second, false);
    }
}

//
// Public API
//
void
AsciiTraceHelperForIpv4::EnableAsciiIpv4(std::string prefix, NodeContainer n)
{
    EnableAsciiIpv4Impl(Ptr<OutputStreamWrapper>(), prefix, n);
}

//
// Public API
//
void
AsciiTraceHelperForIpv4::EnableAsciiIpv4(Ptr<OutputStreamWrapper> stream, NodeContainer n)
{
    EnableAsciiIpv4Impl(stream, std::string(), n);
}

//
// Private API
//
void
AsciiTraceHelperForIpv4::EnableAsciiIpv4Impl(Ptr<OutputStreamWrapper> stream,
                                             std::string prefix,
                                             NodeContainer n)
{
    for (auto i = n.Begin(); i != n.End(); ++i)
    {
        Ptr<Node> node = *i;
        Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
        if (ipv4)
        {
            for (uint32_t j = 0; j < ipv4->GetNInterfaces(); ++j)
            {
                EnableAsciiIpv4Internal(stream, prefix, ipv4, j, false);
            }
        }
    }
}

//
// Public API
//
void
AsciiTraceHelperForIpv4::EnableAsciiIpv4All(std::string prefix)
{
    EnableAsciiIpv4Impl(Ptr<OutputStreamWrapper>(), prefix, NodeContainer::GetGlobal());
}

//
// Public API
//
void
AsciiTraceHelperForIpv4::EnableAsciiIpv4All(Ptr<OutputStreamWrapper> stream)
{
    EnableAsciiIpv4Impl(stream, std::string(), NodeContainer::GetGlobal());
}

//
// Public API
//
void
AsciiTraceHelperForIpv4::EnableAsciiIpv4(Ptr<OutputStreamWrapper> stream,
                                         uint32_t nodeid,
                                         uint32_t interface,
                                         bool explicitFilename)
{
    EnableAsciiIpv4Impl(stream, std::string(), nodeid, interface, explicitFilename);
}

//
// Public API
//
void
AsciiTraceHelperForIpv4::EnableAsciiIpv4(std::string prefix,
                                         uint32_t nodeid,
                                         uint32_t interface,
                                         bool explicitFilename)
{
    EnableAsciiIpv4Impl(Ptr<OutputStreamWrapper>(), prefix, nodeid, interface, explicitFilename);
}

//
// Private API
//
void
AsciiTraceHelperForIpv4::EnableAsciiIpv4Impl(Ptr<OutputStreamWrapper> stream,
                                             std::string prefix,
                                             uint32_t nodeid,
                                             uint32_t interface,
                                             bool explicitFilename)
{
    NodeContainer n = NodeContainer::GetGlobal();

    for (auto i = n.Begin(); i != n.End(); ++i)
    {
        Ptr<Node> node = *i;
        if (node->GetId() != nodeid)
        {
            continue;
        }

        Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
        if (ipv4)
        {
            EnableAsciiIpv4Internal(stream, prefix, ipv4, interface, explicitFilename);
        }

        return;
    }
}

void
PcapHelperForIpv6::EnablePcapIpv6(std::string prefix,
                                  Ptr<Ipv6> ipv6,
                                  uint32_t interface,
                                  bool explicitFilename)
{
    EnablePcapIpv6Internal(prefix, ipv6, interface, explicitFilename);
}

void
PcapHelperForIpv6::EnablePcapIpv6(std::string prefix,
                                  std::string ipv6Name,
                                  uint32_t interface,
                                  bool explicitFilename)
{
    Ptr<Ipv6> ipv6 = Names::Find<Ipv6>(ipv6Name);
    EnablePcapIpv6(prefix, ipv6, interface, explicitFilename);
}

void
PcapHelperForIpv6::EnablePcapIpv6(std::string prefix, Ipv6InterfaceContainer c)
{
    for (auto i = c.Begin(); i != c.End(); ++i)
    {
        EnablePcapIpv6(prefix, (*i).first, (*i).second, false);
    }
}

void
PcapHelperForIpv6::EnablePcapIpv6(std::string prefix, NodeContainer n)
{
    for (auto i = n.Begin(); i != n.End(); ++i)
    {
        Ptr<Node> node = *i;
        Ptr<Ipv6> ipv6 = node->GetObject<Ipv6>();
        if (ipv6)
        {
            for (uint32_t j = 0; j < ipv6->GetNInterfaces(); ++j)
            {
                EnablePcapIpv6(prefix, ipv6, j, false);
            }
        }
    }
}

void
PcapHelperForIpv6::EnablePcapIpv6All(std::string prefix)
{
    EnablePcapIpv6(prefix, NodeContainer::GetGlobal());
}

void
PcapHelperForIpv6::EnablePcapIpv6(std::string prefix,
                                  uint32_t nodeid,
                                  uint32_t interface,
                                  bool explicitFilename)
{
    NodeContainer n = NodeContainer::GetGlobal();

    for (auto i = n.Begin(); i != n.End(); ++i)
    {
        Ptr<Node> node = *i;
        if (node->GetId() != nodeid)
        {
            continue;
        }

        Ptr<Ipv6> ipv6 = node->GetObject<Ipv6>();
        if (ipv6)
        {
            EnablePcapIpv6(prefix, ipv6, interface, explicitFilename);
        }
        return;
    }
}

//
// Public API
//
void
AsciiTraceHelperForIpv6::EnableAsciiIpv6(std::string prefix,
                                         Ptr<Ipv6> ipv6,
                                         uint32_t interface,
                                         bool explicitFilename)
{
    EnableAsciiIpv6Internal(Ptr<OutputStreamWrapper>(), prefix, ipv6, interface, explicitFilename);
}

//
// Public API
//
void
AsciiTraceHelperForIpv6::EnableAsciiIpv6(Ptr<OutputStreamWrapper> stream,
                                         Ptr<Ipv6> ipv6,
                                         uint32_t interface)
{
    EnableAsciiIpv6Internal(stream, std::string(), ipv6, interface, false);
}

//
// Public API
//
void
AsciiTraceHelperForIpv6::EnableAsciiIpv6(std::string prefix,
                                         std::string ipv6Name,
                                         uint32_t interface,
                                         bool explicitFilename)
{
    EnableAsciiIpv6Impl(Ptr<OutputStreamWrapper>(), prefix, ipv6Name, interface, explicitFilename);
}

//
// Public API
//
void
AsciiTraceHelperForIpv6::EnableAsciiIpv6(Ptr<OutputStreamWrapper> stream,
                                         std::string ipv6Name,
                                         uint32_t interface)
{
    EnableAsciiIpv6Impl(stream, std::string(), ipv6Name, interface, false);
}

//
// Private API
//
void
AsciiTraceHelperForIpv6::EnableAsciiIpv6Impl(Ptr<OutputStreamWrapper> stream,
                                             std::string prefix,
                                             std::string ipv6Name,
                                             uint32_t interface,
                                             bool explicitFilename)
{
    Ptr<Ipv6> ipv6 = Names::Find<Ipv6>(ipv6Name);
    EnableAsciiIpv6Internal(stream, prefix, ipv6, interface, explicitFilename);
}

//
// Public API
//
void
AsciiTraceHelperForIpv6::EnableAsciiIpv6(std::string prefix, Ipv6InterfaceContainer c)
{
    EnableAsciiIpv6Impl(Ptr<OutputStreamWrapper>(), prefix, c);
}

//
// Public API
//
void
AsciiTraceHelperForIpv6::EnableAsciiIpv6(Ptr<OutputStreamWrapper> stream, Ipv6InterfaceContainer c)
{
    EnableAsciiIpv6Impl(stream, std::string(), c);
}

//
// Private API
//
void
AsciiTraceHelperForIpv6::EnableAsciiIpv6Impl(Ptr<OutputStreamWrapper> stream,
                                             std::string prefix,
                                             Ipv6InterfaceContainer c)
{
    for (auto i = c.Begin(); i != c.End(); ++i)
    {
        EnableAsciiIpv6Internal(stream, prefix, (*i).first, (*i).second, false);
    }
}

//
// Public API
//
void
AsciiTraceHelperForIpv6::EnableAsciiIpv6(std::string prefix, NodeContainer n)
{
    EnableAsciiIpv6Impl(Ptr<OutputStreamWrapper>(), prefix, n);
}

//
// Public API
//
void
AsciiTraceHelperForIpv6::EnableAsciiIpv6(Ptr<OutputStreamWrapper> stream, NodeContainer n)
{
    EnableAsciiIpv6Impl(stream, std::string(), n);
}

//
// Private API
//
void
AsciiTraceHelperForIpv6::EnableAsciiIpv6Impl(Ptr<OutputStreamWrapper> stream,
                                             std::string prefix,
                                             NodeContainer n)
{
    for (auto i = n.Begin(); i != n.End(); ++i)
    {
        Ptr<Node> node = *i;
        Ptr<Ipv6> ipv6 = node->GetObject<Ipv6>();
        if (ipv6)
        {
            for (uint32_t j = 0; j < ipv6->GetNInterfaces(); ++j)
            {
                EnableAsciiIpv6Internal(stream, prefix, ipv6, j, false);
            }
        }
    }
}

//
// Public API
//
void
AsciiTraceHelperForIpv6::EnableAsciiIpv6All(std::string prefix)
{
    EnableAsciiIpv6Impl(Ptr<OutputStreamWrapper>(), prefix, NodeContainer::GetGlobal());
}

//
// Public API
//
void
AsciiTraceHelperForIpv6::EnableAsciiIpv6All(Ptr<OutputStreamWrapper> stream)
{
    EnableAsciiIpv6Impl(stream, std::string(), NodeContainer::GetGlobal());
}

//
// Public API
//
void
AsciiTraceHelperForIpv6::EnableAsciiIpv6(Ptr<OutputStreamWrapper> stream,
                                         uint32_t nodeid,
                                         uint32_t interface)
{
    EnableAsciiIpv6Impl(stream, std::string(), nodeid, interface, false);
}

//
// Public API
//
void
AsciiTraceHelperForIpv6::EnableAsciiIpv6(std::string prefix,
                                         uint32_t nodeid,
                                         uint32_t interface,
                                         bool explicitFilename)
{
    EnableAsciiIpv6Impl(Ptr<OutputStreamWrapper>(), prefix, nodeid, interface, explicitFilename);
}

//
// Private API
//
void
AsciiTraceHelperForIpv6::EnableAsciiIpv6Impl(Ptr<OutputStreamWrapper> stream,
                                             std::string prefix,
                                             uint32_t nodeid,
                                             uint32_t interface,
                                             bool explicitFilename)
{
    NodeContainer n = NodeContainer::GetGlobal();

    for (auto i = n.Begin(); i != n.End(); ++i)
    {
        Ptr<Node> node = *i;
        if (node->GetId() != nodeid)
        {
            continue;
        }

        Ptr<Ipv6> ipv6 = node->GetObject<Ipv6>();
        if (ipv6)
        {
            EnableAsciiIpv6Internal(stream, prefix, ipv6, interface, explicitFilename);
        }

        return;
    }
}

} // namespace ns3
