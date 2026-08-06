/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ipv4-source-route-tag.h"

#include "ns3/log.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("Ipv4SourceRouteTag");

NS_OBJECT_ENSURE_REGISTERED(Ipv4SourceRouteTag);

TypeId
Ipv4SourceRouteTag::GetTypeId()
{
    static TypeId tid = TypeId("ns3::Ipv4SourceRouteTag")
                            .SetParent<Tag>()
                            .SetGroupName("Internet")
                            .AddConstructor<Ipv4SourceRouteTag>();
    return tid;
}

TypeId
Ipv4SourceRouteTag::GetInstanceTypeId() const
{
    return GetTypeId();
}

uint32_t
Ipv4SourceRouteTag::GetSerializedSize() const
{
    return 1 + 4 * MAX_HOPS;
}

void
Ipv4SourceRouteTag::Serialize(TagBuffer i) const
{
    i.WriteU8(m_route.size());
    for (const auto& address : m_route)
    {
        i.WriteU32(address.Get());
    }
    // The tag has a fixed size, so the unused addresses are written as zeroes
    for (uint8_t hop = m_route.size(); hop < MAX_HOPS; ++hop)
    {
        i.WriteU32(0);
    }
}

void
Ipv4SourceRouteTag::Deserialize(TagBuffer i)
{
    m_route.clear();
    uint8_t hops = i.ReadU8();
    for (uint8_t hop = 0; hop < MAX_HOPS; ++hop)
    {
        uint32_t address = i.ReadU32();
        if (hop < hops)
        {
            m_route.emplace_back(address);
        }
    }
}

void
Ipv4SourceRouteTag::Print(std::ostream& os) const
{
    os << "route=[";
    for (const auto& address : m_route)
    {
        os << " " << address;
    }
    os << " ]";
}

void
Ipv4SourceRouteTag::SetRoute(const std::vector<Ipv4Address>& route)
{
    NS_LOG_FUNCTION(this << route.size());
    NS_ABORT_MSG_IF(route.size() > MAX_HOPS, "A source route holds at most " << +MAX_HOPS);
    m_route = route;
}

std::vector<Ipv4Address>
Ipv4SourceRouteTag::GetRoute() const
{
    return m_route;
}

} // namespace ns3
