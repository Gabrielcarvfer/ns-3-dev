/*
 * Copyright (c) 2007 Emmanuelle Laprise
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Emmanuelle Laprise <emmanuelle.laprise@bluekazoo.ca>
 */

#ifndef ETHERNET_HEADER_H
#define ETHERNET_HEADER_H

#include "mac48-address.h"

#include "ns3/header.h"

#include <string>

namespace ns3
{

/**
 * @ingroup network
 *
 * Types of ethernet packets. Indicates the type of the current
 * header.
 */
enum ethernet_header_t
{
    LENGTH, /**< Basic ethernet packet, no tags, type/length field
               indicates packet length or IP/ARP packet */
    VLAN,   /**< Single tagged packet. Header includes VLAN tag */
    QINQ    /**< Double tagged packet. Header includes two VLAN tags */
};

/**
 * @ingroup network
 *
 * @brief Packet header for Ethernet
 *
 * This class can be used to add a header to an ethernet packet that
 * will specify the source and destination addresses and the length of
 * the packet. Eventually the class will be improved to also support
 * VLAN tags in packet headers.
 */
class EthernetHeader : public Header
{
  public:
    /**
     * @brief Construct a null ethernet header
     * @param hasPreamble if true, insert and remove an ethernet preamble from the
     *       packet, if false, does not insert and remove it.
     */
    EthernetHeader(bool hasPreamble);
    /**
     * @brief Construct a null ethernet header
     * By default, does not add or remove an ethernet preamble
     */
    EthernetHeader();
    /**
     * @param size The size of the payload in bytes
     */
    void SetLengthType(uint16_t size);
    /**
     * @param source The source address of this packet
     */
    void SetSource(Mac48Address source);
    /**
     * @param destination The destination address of this packet.
     */
    void SetDestination(Mac48Address destination);
    /**
     * @param preambleSfd The value that the preambleSfd field should take
     */
    void SetPreambleSfd(uint64_t preambleSfd);
    /**
     * @return The size of the payload in bytes
     */
    uint16_t GetLengthType() const;
    /**
     * @return The type of packet (only basic Ethernet is currently supported)
     */
    ethernet_header_t GetPacketType() const;
    /**
     * @return The source address of this packet
     */
    Mac48Address GetSource() const;
    /**
     * @return The destination address of this packet
     */
    Mac48Address GetDestination() const;
    /**
     * @return The value of the PreambleSfd field
     */
    uint64_t GetPreambleSfd() const;
    /**
     * @return The size of the header
     */
    uint32_t GetHeaderSize() const;

    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();
    TypeId GetInstanceTypeId() const override;
    void Print(std::ostream& os) const override;
    uint32_t GetSerializedSize() const override;
    void Serialize(Buffer::Iterator start) const override;
    uint32_t Deserialize(Buffer::Iterator start) override;

  private:
    static const int PREAMBLE_SIZE = 8; //!< size of the preamble_sfd header field
    static const int LENGTH_SIZE = 2;   //!< size of the length_type header field
    static const int MAC_ADDR_SIZE = 6; //!< size of src/dest addr header fields

    /**
     * If false, the preamble/sfd are not serialised/deserialised.
     */
    bool m_enPreambleSfd;
    uint64_t m_preambleSfd;     //!< Value of the Preamble/SFD fields
    uint16_t m_lengthType;      //!< Length or type of the packet
    Mac48Address m_source;      //!< Source address
    Mac48Address m_destination; //!< Destination address
};

} // namespace ns3

#endif /* ETHERNET_HEADER_H */
