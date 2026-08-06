/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#ifndef IPV4_SOURCE_ROUTE_TAG_H
#define IPV4_SOURCE_ROUTE_TAG_H

#include "ns3/ipv4-address.h"
#include "ns3/tag.h"

#include <vector>

namespace ns3
{

/**
 * @ingroup ipv4
 *
 * @brief Tag carrying the addresses of a loose source route
 *
 * Carries a route between a socket and the IPv4 layer: on the way down, the
 * route the application asked the datagram to follow, which the layer turns
 * into the option of the header; on the way up, the route recorded in the
 * option of a received datagram, which a connection can send its answers
 * along (@RFC{9293}, Section 3.9.2.1, MUST-51 to MUST-53).
 */
class Ipv4SourceRouteTag : public Tag
{
  public:
    /**
     * @brief Get the type ID.
     * @return The object TypeId.
     */
    static TypeId GetTypeId();
    TypeId GetInstanceTypeId() const override;

    uint32_t GetSerializedSize() const override;
    void Serialize(TagBuffer i) const override;
    void Deserialize(TagBuffer i) override;
    void Print(std::ostream& os) const override;

    /**
     * @brief Set the addresses of the route.
     * @param route The addresses, the last of which is the final destination.
     */
    void SetRoute(const std::vector<Ipv4Address>& route);

    /**
     * @brief Get the addresses of the route.
     * @return The addresses, the last of which is the final destination.
     */
    std::vector<Ipv4Address> GetRoute() const;

    /// The largest route the option of a header holds
    static constexpr uint8_t MAX_HOPS = 9;

  private:
    std::vector<Ipv4Address> m_route; //!< The addresses of the route
};

} // namespace ns3

#endif /* IPV4_SOURCE_ROUTE_TAG_H */
