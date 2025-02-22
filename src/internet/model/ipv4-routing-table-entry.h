/*
 * Copyright (c) 2005 INRIA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#ifndef IPV4_ROUTING_TABLE_ENTRY_H
#define IPV4_ROUTING_TABLE_ENTRY_H

#include "ns3/ipv4-address.h"

#include <list>
#include <ostream>
#include <vector>

namespace ns3
{

/**
 * @ingroup ipv4Routing
 *
 * A record of an IPv4 routing table entry for Ipv4GlobalRouting and
 * Ipv4StaticRouting.  This is not a reference counted object.
 */
class Ipv4RoutingTableEntry
{
  public:
    /**
     * @brief This constructor does nothing
     */
    Ipv4RoutingTableEntry();
    /**
     * @brief Copy Constructor
     * @param route The route to copy
     */
    Ipv4RoutingTableEntry(const Ipv4RoutingTableEntry& route);
    /**
     * @brief Copy Constructor
     * @param route The route to copy
     */
    Ipv4RoutingTableEntry(const Ipv4RoutingTableEntry* route);
    /**
     * @return True if this route is a host route (mask of all ones); false otherwise
     */
    bool IsHost() const;
    /**
     * @return True if this route is not a host route (mask is not all ones); false otherwise
     *
     * This method is implemented as !IsHost ().
     */
    bool IsNetwork() const;
    /**
     * @return True if this route is a default route; false otherwise
     */
    bool IsDefault() const;
    /**
     * @return True if this route is a gateway route; false otherwise
     */
    bool IsGateway() const;
    /**
     * @return address of the gateway stored in this entry
     */
    Ipv4Address GetGateway() const;
    /**
     * @return The IPv4 address of the destination of this route
     */
    Ipv4Address GetDest() const;
    /**
     * @return The IPv4 network number of the destination of this route
     */
    Ipv4Address GetDestNetwork() const;
    /**
     * @return The IPv4 network mask of the destination of this route
     */
    Ipv4Mask GetDestNetworkMask() const;
    /**
     * @return The Ipv4 interface number used for sending outgoing packets
     */
    uint32_t GetInterface() const;
    /**
     * @return An Ipv4RoutingTableEntry object corresponding to the input parameters.
     * @param dest Ipv4Address of the destination
     * @param nextHop Ipv4Address of the next hop
     * @param interface Outgoing interface
     */
    static Ipv4RoutingTableEntry CreateHostRouteTo(Ipv4Address dest,
                                                   Ipv4Address nextHop,
                                                   uint32_t interface);
    /**
     * @return An Ipv4RoutingTableEntry object corresponding to the input parameters.
     * @param dest Ipv4Address of the destination
     * @param interface Outgoing interface
     */
    static Ipv4RoutingTableEntry CreateHostRouteTo(Ipv4Address dest, uint32_t interface);
    /**
     * @return An Ipv4RoutingTableEntry object corresponding to the input parameters.
     * @param network Ipv4Address of the destination network
     * @param networkMask Ipv4Mask of the destination network mask
     * @param nextHop Ipv4Address of the next hop
     * @param interface Outgoing interface
     */
    static Ipv4RoutingTableEntry CreateNetworkRouteTo(Ipv4Address network,
                                                      Ipv4Mask networkMask,
                                                      Ipv4Address nextHop,
                                                      uint32_t interface);
    /**
     * @return An Ipv4RoutingTableEntry object corresponding to the input parameters.
     * @param network Ipv4Address of the destination network
     * @param networkMask Ipv4Mask of the destination network mask
     * @param interface Outgoing interface
     */
    static Ipv4RoutingTableEntry CreateNetworkRouteTo(Ipv4Address network,
                                                      Ipv4Mask networkMask,
                                                      uint32_t interface);
    /**
     * @return An Ipv4RoutingTableEntry object corresponding to the input
     * parameters.  This route is distinguished; it will match any
     * destination for which a more specific route does not exist.
     * @param nextHop Ipv4Address of the next hop
     * @param interface Outgoing interface
     */
    static Ipv4RoutingTableEntry CreateDefaultRoute(Ipv4Address nextHop, uint32_t interface);

  private:
    /**
     * @brief Constructor.
     * @param network network address
     * @param mask network mask
     * @param gateway the gateway
     * @param interface the interface index
     */
    Ipv4RoutingTableEntry(Ipv4Address network,
                          Ipv4Mask mask,
                          Ipv4Address gateway,
                          uint32_t interface);
    /**
     * @brief Constructor.
     * @param dest destination address
     * @param mask network mask
     * @param interface the interface index
     */
    Ipv4RoutingTableEntry(Ipv4Address dest, Ipv4Mask mask, uint32_t interface);
    /**
     * @brief Constructor.
     * @param dest destination address
     * @param gateway the gateway
     * @param interface the interface index
     */
    Ipv4RoutingTableEntry(Ipv4Address dest, Ipv4Address gateway, uint32_t interface);
    /**
     * @brief Constructor.
     * @param dest destination address
     * @param interface the interface index
     */
    Ipv4RoutingTableEntry(Ipv4Address dest, uint32_t interface);

    Ipv4Address m_dest;         //!< destination address
    Ipv4Mask m_destNetworkMask; //!< destination network mask
    Ipv4Address m_gateway;      //!< gateway
    uint32_t m_interface;       //!< output interface
};

/**
 * @brief Stream insertion operator.
 *
 * @param os the reference to the output stream
 * @param route the Ipv4 routing table entry
 * @returns the reference to the output stream
 */
std::ostream& operator<<(std::ostream& os, const Ipv4RoutingTableEntry& route);

/**
 * @brief Equality operator.
 *
 * @param a lhs
 * @param b rhs
 * @returns true if operands are equal, false otherwise
 */
bool operator==(const Ipv4RoutingTableEntry a, const Ipv4RoutingTableEntry b);

/**
 * @ingroup ipv4Routing
 *
 * @brief A record of an IPv4 multicast route for Ipv4GlobalRouting and Ipv4StaticRouting
 */
class Ipv4MulticastRoutingTableEntry
{
  public:
    /**
     * @brief This constructor does nothing
     */
    Ipv4MulticastRoutingTableEntry();

    /**
     * @brief Copy Constructor
     * @param route The route to copy
     */
    Ipv4MulticastRoutingTableEntry(const Ipv4MulticastRoutingTableEntry& route);
    /**
     * @brief Copy Constructor
     * @param route The route to copy
     */
    Ipv4MulticastRoutingTableEntry(const Ipv4MulticastRoutingTableEntry* route);
    /**
     * @return The IPv4 address of the source of this route
     */
    Ipv4Address GetOrigin() const;
    /**
     * @return The IPv4 address of the multicast group of this route
     */
    Ipv4Address GetGroup() const;
    /**
     * @return The IPv4 address of the input interface of this route
     */
    uint32_t GetInputInterface() const;
    /**
     * @return The number of output interfaces of this route
     */
    uint32_t GetNOutputInterfaces() const;
    /**
     * @param n interface index
     * @return A specified output interface.
     */
    uint32_t GetOutputInterface(uint32_t n) const;
    /**
     * @return A vector of all of the output interfaces of this route.
     */
    std::vector<uint32_t> GetOutputInterfaces() const;
    /**
     * @return Ipv4MulticastRoutingTableEntry corresponding to the input parameters.
     * @param origin Source address for the multicast route
     * @param group Group destination address for the multicast route
     * @param inputInterface Input interface that multicast datagram must be received on
     * @param outputInterfaces vector of output interfaces to copy and forward the datagram to
     */
    static Ipv4MulticastRoutingTableEntry CreateMulticastRoute(
        Ipv4Address origin,
        Ipv4Address group,
        uint32_t inputInterface,
        std::vector<uint32_t> outputInterfaces);

  private:
    /**
     * @brief Constructor.
     * @param origin source address
     * @param group destination address
     * @param inputInterface input interface
     * @param outputInterfaces output interfaces
     */
    Ipv4MulticastRoutingTableEntry(Ipv4Address origin,
                                   Ipv4Address group,
                                   uint32_t inputInterface,
                                   std::vector<uint32_t> outputInterfaces);

    Ipv4Address m_origin;                     //!< source address
    Ipv4Address m_group;                      //!< destination address
    uint32_t m_inputInterface;                //!< input interface
    std::vector<uint32_t> m_outputInterfaces; //!< output interfaces
};

/**
 * @brief Stream insertion operator.
 *
 * @param os the reference to the output stream
 * @param route the Ipv4 multicast routing table entry
 * @returns the reference to the output stream
 */
std::ostream& operator<<(std::ostream& os, const Ipv4MulticastRoutingTableEntry& route);

/**
 * @brief Equality operator.
 *
 * @param a lhs
 * @param b rhs
 * @returns true if operands are equal, false otherwise
 */
bool operator==(const Ipv4MulticastRoutingTableEntry a, const Ipv4MulticastRoutingTableEntry b);

} // namespace ns3

#endif /* IPV4_ROUTING_TABLE_ENTRY_H */
