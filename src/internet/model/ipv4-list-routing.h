/*
 * Copyright (c) 2009 University of Washington
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

#ifndef IPV4_LIST_ROUTING_H
#define IPV4_LIST_ROUTING_H

#include "ipv4-routing-protocol.h"

#include "ns3/nstime.h"
#include "ns3/simulator.h"

#include <list>

namespace ns3
{

/**
 * @ingroup ipv4Routing
 *
 * @brief IPv4 list routing.
 *
 * This class is a specialization of Ipv4RoutingProtocol that allows
 * other instances of Ipv4RoutingProtocol to be inserted in a
 * prioritized list.  Routing protocols in the list are consulted one
 * by one, from highest to lowest priority, until a routing protocol
 * is found that will take the packet (this corresponds to a non-zero
 * return value to RouteOutput, or a return value of true to RouteInput).
 * The order by which routing protocols with the same priority value
 * are consulted is undefined.
 *
 */
class Ipv4ListRouting : public Ipv4RoutingProtocol
{
  public:
    /**
     * @brief Get the type ID of this class.
     * @return type ID
     */
    static TypeId GetTypeId();

    Ipv4ListRouting();
    ~Ipv4ListRouting() override;

    /**
     * @brief Register a new routing protocol to be used in this IPv4 stack
     *
     * @param routingProtocol new routing protocol implementation object
     * @param priority priority to give to this routing protocol.
     * Values may range between -32768 and +32767.
     */
    virtual void AddRoutingProtocol(Ptr<Ipv4RoutingProtocol> routingProtocol, int16_t priority);
    /**
     * @return number of routing protocols in the list
     */
    virtual uint32_t GetNRoutingProtocols() const;
    /**
     * Return pointer to routing protocol stored at index, with the
     * first protocol (index 0) the highest priority, the next one (index 1)
     * the second highest priority, and so on.  The priority parameter is an
     * output parameter and it returns the integer priority of the protocol.
     *
     * @return pointer to routing protocol indexed by
     * @param index index of protocol to return
     * @param priority output parameter, set to the priority of the protocol
              being returned
     */
    virtual Ptr<Ipv4RoutingProtocol> GetRoutingProtocol(uint32_t index, int16_t& priority) const;

    // Below are from Ipv4RoutingProtocol
    Ptr<Ipv4Route> RouteOutput(Ptr<Packet> p,
                               const Ipv4Header& header,
                               Ptr<NetDevice> oif,
                               Socket::SocketErrno& sockerr) override;

    bool RouteInput(Ptr<const Packet> p,
                    const Ipv4Header& header,
                    Ptr<const NetDevice> idev,
                    const UnicastForwardCallback& ucb,
                    const MulticastForwardCallback& mcb,
                    const LocalDeliverCallback& lcb,
                    const ErrorCallback& ecb) override;
    void NotifyInterfaceUp(uint32_t interface) override;
    void NotifyInterfaceDown(uint32_t interface) override;
    void NotifyAddAddress(uint32_t interface, Ipv4InterfaceAddress address) override;
    void NotifyRemoveAddress(uint32_t interface, Ipv4InterfaceAddress address) override;
    void SetIpv4(Ptr<Ipv4> ipv4) override;
    void PrintRoutingTable(Ptr<OutputStreamWrapper> stream,
                           Time::Unit unit = Time::S) const override;

  protected:
    void DoDispose() override;
    void DoInitialize() override;

  private:
    /**
     * @brief Container identifying an IPv4 Routing Protocol entry in the list.
     */
    typedef std::pair<int16_t, Ptr<Ipv4RoutingProtocol>> Ipv4RoutingProtocolEntry;
    /**
     * @brief Container of the IPv4 Routing Protocols.
     */
    typedef std::list<Ipv4RoutingProtocolEntry> Ipv4RoutingProtocolList;
    Ipv4RoutingProtocolList m_routingProtocols; //!<  List of routing protocols.

    /**
     * @brief Compare two routing protocols.
     * @param a first object to compare
     * @param b second object to compare
     * @return true if they are the same, false otherwise
     */
    static bool Compare(const Ipv4RoutingProtocolEntry& a, const Ipv4RoutingProtocolEntry& b);
    Ptr<Ipv4> m_ipv4; //!< Ipv4 this protocol is associated with.
};

} // namespace ns3

#endif /* IPV4_LIST_ROUTING_H */
