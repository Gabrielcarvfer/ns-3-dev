//
// Copyright (c) 2006 Georgia Tech Research Corporation
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: George F. Riley <riley@ece.gatech.edu>
// Author: Lalith Suresh <suresh.lalith@gmail.com>
//

#ifndef IPV4_L3_CLICK_PROTOCOL_H
#define IPV4_L3_CLICK_PROTOCOL_H

#include "ns3/deprecated.h"
#include "ns3/ipv4-interface.h"
#include "ns3/ipv4-routing-protocol.h"
#include "ns3/ipv4.h"
#include "ns3/log.h"
#include "ns3/net-device.h"
#include "ns3/packet.h"
#include "ns3/traced-callback.h"

namespace ns3
{

class Packet;
class NetDevice;
class Ipv4Interface;
class Ipv4Address;
class Ipv4Header;
class Ipv4RoutingTableEntry;
class Ipv4Route;
class Node;
class Socket;
class Ipv4RawSocketImpl;
class IpL4Protocol;
class Icmpv4L4Protocol;

/**
 * @brief Implement the Ipv4 layer specifically for Click nodes
 * to allow a clean integration of Click.
 * @ingroup click
 *
 * This is code is mostly repeated from the Ipv4L3Protocol implementation.
 * Changes include:
 *   - A stripped down version of Send().
 *   - A stripped down version of Receive().
 *   - A public version of LocalDeliver().
 *   - Modifications to AddInterface().
 */

class Ipv4L3ClickProtocol : public Ipv4
{
  public:
    /**
     * Get Type ID.
     *
     * @return The type ID.
     */
    static TypeId GetTypeId();

    /**
     * Protocol number for Ipv4 L3 (0x0800).
     */
    static const uint16_t PROT_NUMBER;

    Ipv4L3ClickProtocol();
    ~Ipv4L3ClickProtocol() override;

    void Insert(Ptr<IpL4Protocol> protocol) override;
    void Insert(Ptr<IpL4Protocol> protocol, uint32_t interfaceIndex) override;

    void Remove(Ptr<IpL4Protocol> protocol) override;
    void Remove(Ptr<IpL4Protocol> protocol, uint32_t interfaceIndex) override;

    Ptr<IpL4Protocol> GetProtocol(int protocolNumber) const override;
    Ptr<IpL4Protocol> GetProtocol(int protocolNumber, int32_t interfaceIndex) const override;

    Ipv4Address SourceAddressSelection(uint32_t interface, Ipv4Address dest) override;

    /**
     * @param ttl default ttl to use
     *
     * When we need to send an ipv4 packet, we use this default
     * ttl value.
     */
    void SetDefaultTtl(uint8_t ttl);

    /**
     * @param packet packet to send
     * @param source source address of packet
     * @param destination address of packet
     * @param protocol number of packet
     * @param route route entry
     *
     * Higher-level layers call this method to send a packet
     * to Click
     */
    void Send(Ptr<Packet> packet,
              Ipv4Address source,
              Ipv4Address destination,
              uint8_t protocol,
              Ptr<Ipv4Route> route) override;

    /**
     * @param packet packet to send
     * @param ipHeader IP Header
     * @param route route entry
     *
     * Higher-level layers call this method to send a packet with IPv4 Header
     * (Intend to be used with IpHeaderInclude attribute.)
     */
    void SendWithHeader(Ptr<Packet> packet, Ipv4Header ipHeader, Ptr<Ipv4Route> route) override;

    /**
     * @param packet packet to send down the stack
     * @param ifid interface to be used for sending down packet
     *
     * Ipv4ClickRouting calls this method to send a packet further
     * down the stack
     */
    void SendDown(Ptr<Packet> packet, int ifid);

    /**
     * Lower layer calls this method to send a packet to Click
     * @param device network device
     * @param p the packet
     * @param protocol protocol value
     * @param from address of the correspondent
     * @param to address of the destination
     * @param packetType type of the packet
     */
    void Receive(Ptr<NetDevice> device,
                 Ptr<const Packet> p,
                 uint16_t protocol,
                 const Address& from,
                 const Address& to,
                 NetDevice::PacketType packetType);

    /**
     * Ipv4ClickRouting calls this to locally deliver a packet
     * @param p the packet
     * @param ip The Ipv4Header of the packet
     * @param iif The interface on which the packet was received
     */
    void LocalDeliver(Ptr<const Packet> p, const Ipv4Header& ip, uint32_t iif);

    /**
     * Get a pointer to the i'th Ipv4Interface
     * @param i index of interface, pointer to which is to be returned
     * @returns Pointer to the i'th Ipv4Interface if any.
     */
    Ptr<Ipv4Interface> GetInterface(uint32_t i) const;

    /**
     * Adds an Ipv4Interface to the interfaces list
     * @param interface Pointer to the Ipv4Interface to be added
     * @returns Index of the device which was added
     */
    uint32_t AddIpv4Interface(Ptr<Ipv4Interface> interface);

    /**
     * Calls m_node = node and sets up Loopback if needed
     * @param node Pointer to the node
     */
    void SetNode(Ptr<Node> node);

    /**
     * Returns the Icmpv4L4Protocol for the node
     * @returns Icmpv4L4Protocol instance of the node
     */
    Ptr<Icmpv4L4Protocol> GetIcmp() const;

    /**
     * Sets up a Loopback device
     */
    void SetupLoopback();

    /**
     * Creates a raw-socket
     * @returns Pointer to the created socket
     */
    Ptr<Socket> CreateRawSocket() override;

    /**
     * Deletes a particular raw socket
     * @param socket Pointer of socket to be deleted
     */
    void DeleteRawSocket(Ptr<Socket> socket) override;

    // functions defined in base class Ipv4
    void SetRoutingProtocol(Ptr<Ipv4RoutingProtocol> routingProtocol) override;
    Ptr<Ipv4RoutingProtocol> GetRoutingProtocol() const override;

    Ptr<NetDevice> GetNetDevice(uint32_t i) override;

    uint32_t AddInterface(Ptr<NetDevice> device) override;
    uint32_t GetNInterfaces() const override;

    int32_t GetInterfaceForAddress(Ipv4Address addr) const override;
    int32_t GetInterfaceForPrefix(Ipv4Address addr, Ipv4Mask mask) const override;
    int32_t GetInterfaceForDevice(Ptr<const NetDevice> device) const override;
    bool IsDestinationAddress(Ipv4Address address, uint32_t iif) const override;

    bool AddAddress(uint32_t i, Ipv4InterfaceAddress address) override;
    Ipv4InterfaceAddress GetAddress(uint32_t interfaceIndex, uint32_t addressIndex) const override;
    uint32_t GetNAddresses(uint32_t interface) const override;
    bool RemoveAddress(uint32_t interfaceIndex, uint32_t addressIndex) override;
    bool RemoveAddress(uint32_t interfaceIndex, Ipv4Address address) override;
    Ipv4Address SelectSourceAddress(Ptr<const NetDevice> device,
                                    Ipv4Address dst,
                                    Ipv4InterfaceAddress::InterfaceAddressScope_e scope) override;

    void SetMetric(uint32_t i, uint16_t metric) override;
    uint16_t GetMetric(uint32_t i) const override;
    uint16_t GetMtu(uint32_t i) const override;
    bool IsUp(uint32_t i) const override;
    void SetUp(uint32_t i) override;
    void SetDown(uint32_t i) override;
    bool IsForwarding(uint32_t i) const override;
    void SetForwarding(uint32_t i, bool val) override;

    /**
     * Sets an interface to run on promiscuous mode.
     *
     * @param i Interface ID.
     */
    void SetPromisc(uint32_t i);

  protected:
    void DoDispose() override;
    /**
     * This function will notify other components connected to the node that a new stack member is
     * now connected This will be used to notify Layer 3 protocol of layer 4 protocol stack to
     * connect them together.
     */
    void NotifyNewAggregate() override;

  private:
    /**
     * Build IPv4 header.
     *
     * @param source IPv4 source address.
     * @param destination IPv4 destination address.
     * @param protocol Protocol.
     * @param payloadSize Payload size.
     * @param ttl Time To Live (TTL).
     * @param mayFragment Whether the packet can be fragmented or not.
     * @return The IPv4 header.
     */
    Ipv4Header BuildHeader(Ipv4Address source,
                           Ipv4Address destination,
                           uint8_t protocol,
                           uint16_t payloadSize,
                           uint8_t ttl,
                           bool mayFragment);

    void SetIpForward(bool forward) override;
    bool GetIpForward() const override;

    void SetStrongEndSystemModel(bool model) override;
    bool GetStrongEndSystemModel() const override;

    /**
     * @brief List of IPv4 interfaces.
     */
    typedef std::vector<Ptr<Ipv4Interface>> Ipv4InterfaceList;

    /**
     * @brief Container of NetDevices registered to IPv4 and their interface indexes.
     */
    typedef std::map<Ptr<const NetDevice>, uint32_t> Ipv4InterfaceReverseContainer;

    /**
     * @brief List of sockets.
     */
    typedef std::list<Ptr<Ipv4RawSocketImpl>> SocketList;

    /**
     * @brief Container of the IPv4 L4 keys: protocol number, interface index
     */
    typedef std::pair<int, int32_t> L4ListKey_t;

    /**
     * @brief Container of the IPv4 L4 instances.
     */
    typedef std::map<L4ListKey_t, Ptr<IpL4Protocol>> L4List_t;

    Ptr<Ipv4RoutingProtocol> m_routingProtocol; //!< IPv4 routing protocol
    bool m_ipForward;                           //!< Whether IP forwarding is enabled
    bool m_strongEndSystemModel;                //!< Whether to use Strong End System Model
    L4List_t m_protocols;                       //!< List of IPv4 L4 protocols
    Ipv4InterfaceList m_interfaces;             //!< List of interfaces
    Ipv4InterfaceReverseContainer
        m_reverseInterfacesContainer; //!< Container of NetDevice / Interface index associations
    uint8_t m_defaultTtl;             //!< Default TTL
    uint16_t m_identification;        //!< Identification

    Ptr<Node> m_node; //!< Node

    /** @todo Remove; this TracedCallback is never invoked. */
    TracedCallback<const Ipv4Header&, Ptr<const Packet>, uint32_t> m_sendOutgoingTrace;
    /** @todo Remove: this TracedCallback is never invoked. */
    TracedCallback<const Ipv4Header&, Ptr<const Packet>, uint32_t> m_unicastForwardTrace;
    /** @todo This TracedCallback is invoked but not accessible. */
    TracedCallback<const Ipv4Header&, Ptr<const Packet>, uint32_t> m_localDeliverTrace;

    SocketList m_sockets; //!< List of sockets

    std::vector<bool> m_promiscDeviceList; //!< List of promiscuous devices
};

} // namespace ns3

#endif /* IPV4_L3_CLICK_ROUTING_H */
