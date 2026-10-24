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
 * @brief Tag carrying the addresses of a loose source route to follow
 *
 * Carries the route the application, or the transport protocol, asks a
 * datagram to follow down to the IPv4 layer, which turns it into the option
 * of the header. The way back of a received datagram travels up in an
 * Ipv4ReturnRouteTag instead, so that a datagram sent on as received does not
 * follow it.
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

    /// The largest route a datagram can follow: the first hop travels in the
    /// destination field of the header, and the option holds the 9 others
    /// which fit in the option space (RFC 791). A route of this many hops
    /// must hence end at the destination, which is otherwise appended to it:
    /// the IP layer drops a datagram whose route does not fit.
    static constexpr uint8_t MAX_HOPS = 10;

  private:
    std::vector<Ipv4Address> m_route; //!< The addresses of the route
};

/**
 * @ingroup ipv4
 *
 * @brief Tag carrying the way back of a received source routed datagram
 *
 * The IPv4 layer attaches it to a datagram which reached its final
 * destination along a loose source route: the addresses the option recorded,
 * reversed and followed by the sender of the datagram. TCP sends the segments
 * of a connection along it (@RFC{9293}, Section 3.9.2.1, MUST-51 to MUST-53),
 * and ICMP its Echo Reply (@RFC{1122}, Section 3.2.2.6). The UDP and raw
 * sockets leave it on the datagrams they deliver, so that the application
 * sees the route the datagram followed (@RFC{1122}, Section 4.1.3.2): an
 * application which answers along the way back attaches it to the answer as
 * an Ipv4SourceRouteTag. The IPv4 layer never sends a datagram along this
 * tag, so a datagram an application sends on as received does not follow it.
 */
class Ipv4ReturnRouteTag : public Ipv4SourceRouteTag
{
  public:
    /**
     * @brief Get the type ID.
     * @return The object TypeId.
     */
    static TypeId GetTypeId();
    TypeId GetInstanceTypeId() const override;
};

} // namespace ns3

#endif /* IPV4_SOURCE_ROUTE_TAG_H */
