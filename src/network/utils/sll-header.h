/*
 * Copyright (c) 2015 Université Pierre et Marie Curie
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Matthieu Coudron <matthieu.coudron@lip6.fr>
 */
#ifndef SLL_HEADER_H
#define SLL_HEADER_H

#include "ns3/buffer.h"
#include "ns3/header.h"

#include <stdint.h>

namespace ns3
{

/**
  * @ingroup packet
  *
  * @brief Protocol header serialization and deserialization.
  *
  * Libpcap sometimes add an additional header to provide information that would be
  * lost otherwise due to the link-layer/capture mechanism, for instance when capturing from
  * "nlmon" device on linux
  *
  * @see http://www.tcpdump.org/linktypes/LINKTYPE_LINUX_SLL.html
  * @see https://wiki.wireshark.org/SLL
  *
  \verbatim
    +---------------------------+
    |         Packet type       |
    |         (2 Octets)        |
    +---------------------------+
    |        ARPHRD_ type       |
    |         (2 Octets)        |
    +---------------------------+
    | Link-layer address length |
    |         (2 Octets)        |
    +---------------------------+
    |    Link-layer address     |
    |         (8 Octets)        |
    +---------------------------+
    |        Protocol type      |
    |         (2 Octets)        |
    +---------------------------+
    |           Payload         |
    .                           .
    .                           .
    .                           .
   \endverbatim
  */
class SllHeader : public Header
{
  public:
    /**
     * Type of the packet.
     */
    enum PacketType
    {
        UNICAST_FROM_PEER_TO_ME = 0, /**< the packet was specifically sent to us by somebody else */
        BROADCAST_BY_PEER = 1,       /**< packet was broadcast by somebody else */
        MULTICAST_BY_PEER = 2,  /**< packet was multicast, but not broadcast, by somebody else */
        INTERCEPTED_PACKET = 3, /**< packet was sent to somebody else by somebody else **/
        SENT_BY_US              /**< the packet was sent by us */
    };

    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    SllHeader();
    ~SllHeader() override;

    /**
     * @return ARP header type field in network byte order
     *  The ARPHRD_ type field is in network byte order; it contains a Linux ARPHRD_ value for the
     * link-layer device type.
     */
    uint16_t GetArpType() const;

    /**
     * @param arphdType ARP protocol hardware identifier
     */
    void SetArpType(uint16_t arphdType);

    /**
     * @return Packet type
     */
    PacketType GetPacketType() const;

    /**
     * @param type Depends on source and address of the packet
     */
    void SetPacketType(PacketType type);

    // Inherited from ObjectBase
    TypeId GetInstanceTypeId() const override;
    // Inherited from Header
    uint32_t GetSerializedSize() const override;
    void Serialize(Buffer::Iterator start) const override;
    uint32_t Deserialize(Buffer::Iterator start) override;
    void Print(std::ostream& os) const override;

  protected:
    // declared in packet order
    PacketType m_packetType;  /**< Packet type */
    uint16_t m_arphdType;     /**< ARP protocol hardware identifier */
    uint16_t m_addressLength; /**< Address length */
    uint64_t m_address;       /**< Address */
    uint16_t m_protocolType;  /**< protocol type */
};

} // namespace ns3

#endif /* SLL_HEADER_H */
