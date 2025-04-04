/*
 * Copyright (c) 2008-2009 Strasbourg University
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: David Gross <gdavid.devel@gmail.com>
 */

#ifndef IPV6_EXTENSION_HEADER_H
#define IPV6_EXTENSION_HEADER_H

#include "ipv6-option-header.h"

#include "ns3/header.h"
#include "ns3/ipv6-address.h"

#include <list>
#include <ostream>
#include <vector>

namespace ns3
{

/**
 * @ingroup ipv6HeaderExt
 *
 * @brief Header for IPv6 Extension.
 */
class Ipv6ExtensionHeader : public Header
{
  public:
    /**
     * @brief Get the type identificator.
     * @return type identificator
     */
    static TypeId GetTypeId();

    /**
     * @brief Get the instance type ID.
     * @return instance type ID
     */
    TypeId GetInstanceTypeId() const override;

    /**
     * @brief Constructor.
     */
    Ipv6ExtensionHeader();

    /**
     * @brief Destructor.
     */
    ~Ipv6ExtensionHeader() override;

    /**
     * @brief Set the "Next header" field.
     * @param nextHeader the next header number
     */
    void SetNextHeader(uint8_t nextHeader);

    /**
     * @brief Get the next header.
     * @return the next header number
     */
    uint8_t GetNextHeader() const;

    /**
     * brief Set the length of the extension.
     * @param length the length of the extension in bytes
     */
    void SetLength(uint16_t length);

    /**
     * @brief Get the length of the extension.
     * @return the length of the extension
     */
    uint16_t GetLength() const;

    /**
     * @brief Print some information about the packet.
     * @param os output stream
     */
    void Print(std::ostream& os) const override;

    /**
     * @brief Get the serialized size of the packet.
     * @return size
     */
    uint32_t GetSerializedSize() const override;

    /**
     * @brief Serialize the packet.
     * @param start Buffer iterator
     */
    void Serialize(Buffer::Iterator start) const override;

    /**
     * @brief Deserialize the packet.
     * @param start Buffer iterator
     * @return size of the packet
     */
    uint32_t Deserialize(Buffer::Iterator start) override;

  protected:
    /**
     * @brief The "length" field.
     */
    uint8_t m_length;

  private:
    /**
     * @brief The "next header" field.
     */
    uint8_t m_nextHeader;

    /**
     * @brief The data of the extension.
     */
    Buffer m_data;
};

/**
 * @ingroup ipv6HeaderExt
 *
 * @brief Option field for an IPv6ExtensionHeader.
 *
 * Enables adding options to an IPv6ExtensionHeader.
 *
 * Implementor's note: Make sure to add the result of
 * OptionField::GetSerializedSize () to your IPv6ExtensionHeader::GetSerializedSize ()
 * return value. Call OptionField::Serialize and OptionField::Deserialize at the
 * end of your corresponding IPv6ExtensionHeader methods.
 */
class OptionField
{
  public:
    /**
     * @brief Constructor.
     * @param optionsOffset option offset
     */
    OptionField(uint32_t optionsOffset);

    /**
     * @brief Destructor.
     */
    ~OptionField();

    /**
     * @brief Get the serialized size of the packet.
     * @return size
     */
    uint32_t GetSerializedSize() const;

    /**
     * @brief Serialize all added options.
     * @param start Buffer iterator
     */
    void Serialize(Buffer::Iterator start) const;

    /**
     * @brief Deserialize the packet.
     * @param start Buffer iterator
     * @param length length
     * @return size of the packet
     */
    uint32_t Deserialize(Buffer::Iterator start, uint32_t length);

    /**
     * @brief Serialize the option, prepending pad1 or padn option as necessary
     * @param option the option header to serialize
     */
    void AddOption(const Ipv6OptionHeader& option);

    /**
     * @brief Get the offset where the options begin, measured from the start of
     * the extension header.
     * @return the offset from the start of the extension header
     */
    uint32_t GetOptionsOffset() const;

    /**
     * @brief Get the buffer.
     * @return buffer
     */
    Buffer GetOptionBuffer();

  private:
    /**
     * @brief Calculate padding.
     * @param alignment alignment
     * @return the number of pad bytes
     */
    uint32_t CalculatePad(Ipv6OptionHeader::Alignment alignment) const;

    /**
     * @brief Data payload.
     */
    Buffer m_optionData;

    /**
     * @brief Offset.
     */
    uint32_t m_optionsOffset;
};

/**
 * @ingroup ipv6HeaderExt
 *
 * @brief Header of IPv6 Extension "Hop by Hop"
 */
class Ipv6ExtensionHopByHopHeader : public Ipv6ExtensionHeader, public OptionField
{
  public:
    /**
     * @brief Get the type identificator.
     * @return type identificator
     */
    static TypeId GetTypeId();

    /**
     * @brief Get the instance type ID.
     * @return instance type ID
     */
    TypeId GetInstanceTypeId() const override;

    /**
     * @brief Constructor.
     */
    Ipv6ExtensionHopByHopHeader();

    /**
     * @brief Destructor.
     */
    ~Ipv6ExtensionHopByHopHeader() override;

    /**
     * @brief Print some information about the packet.
     * @param os output stream
     */
    void Print(std::ostream& os) const override;

    /**
     * @brief Get the serialized size of the packet.
     * @return size
     */
    uint32_t GetSerializedSize() const override;

    /**
     * @brief Serialize the packet.
     * @param start Buffer iterator
     */
    void Serialize(Buffer::Iterator start) const override;

    /**
     * @brief Deserialize the packet.
     * @param start Buffer iterator
     * @return size of the packet
     */
    uint32_t Deserialize(Buffer::Iterator start) override;
};

/**
 * @ingroup ipv6HeaderExt
 *
 * @brief Header of IPv6 Extension Destination
 */
class Ipv6ExtensionDestinationHeader : public Ipv6ExtensionHeader, public OptionField
{
  public:
    /**
     * @brief Get the type identificator.
     * @return type identificator
     */
    static TypeId GetTypeId();

    /**
     * @brief Get the instance type ID.
     * @return instance type ID
     */
    TypeId GetInstanceTypeId() const override;

    /**
     * @brief Constructor.
     */
    Ipv6ExtensionDestinationHeader();

    /**
     * @brief Destructor.
     */
    ~Ipv6ExtensionDestinationHeader() override;

    /**
     * @brief Print some information about the packet.
     * @param os output stream
     */
    void Print(std::ostream& os) const override;

    /**
     * @brief Get the serialized size of the packet.
     * @return size
     */
    uint32_t GetSerializedSize() const override;

    /**
     * @brief Serialize the packet.
     * @param start Buffer iterator
     */
    void Serialize(Buffer::Iterator start) const override;

    /**
     * @brief Deserialize the packet.
     * @param start Buffer iterator
     * @return size of the packet
     */
    uint32_t Deserialize(Buffer::Iterator start) override;
};

/**
 * @ingroup ipv6HeaderExt
 *
 * @brief Header of IPv6 Extension Fragment
 */
class Ipv6ExtensionFragmentHeader : public Ipv6ExtensionHeader
{
  public:
    /**
     * @brief Get the type identificator.
     * @return type identificator
     */
    static TypeId GetTypeId();

    /**
     * @brief Get the instance type ID.
     * @return instance type ID
     */
    TypeId GetInstanceTypeId() const override;

    /**
     * @brief Constructor.
     */
    Ipv6ExtensionFragmentHeader();

    /**
     * @brief Destructor.
     */
    ~Ipv6ExtensionFragmentHeader() override;

    /**
     * @brief Set the "Offset" field.
     * @param offset the offset of the fragment
     */
    void SetOffset(uint16_t offset);

    /**
     * @brief Get the field "Offset".
     * @return the offset of the fragment
     */
    uint16_t GetOffset() const;

    /**
     * @brief Set the status of "More Fragment" bit.
     * @param moreFragment the bit "More Fragment"
     */
    void SetMoreFragment(bool moreFragment);

    /**
     * @brief Get the status of "More Fragment" bit.
     * @return the status of "More Fragment" bit.
     */
    bool GetMoreFragment() const;

    /**
     * @brief Set the "Identification" field.
     * @param identification the identifier of the fragment
     */
    void SetIdentification(uint32_t identification);

    /**
     * @brief Get the field "Identification".
     * @return the identifier of the fragment
     */
    uint32_t GetIdentification() const;

    /**
     * @brief Print some information about the packet.
     * @param os output stream
     */
    void Print(std::ostream& os) const override;

    /**
     * @brief Get the serialized size of the packet.
     * @return size
     */
    uint32_t GetSerializedSize() const override;

    /**
     * @brief Serialize the packet.
     * @param start Buffer iterator
     */
    void Serialize(Buffer::Iterator start) const override;

    /**
     * @brief Deserialize the packet.
     * @param start Buffer iterator
     * @return size of the packet
     */
    uint32_t Deserialize(Buffer::Iterator start) override;

  private:
    /**
     * @brief Offset of the fragment and More Fragment bit.
     */
    uint16_t m_offset;

    /**
     * @brief Identifier of the packet.
     */
    uint32_t m_identification;
};

/**
 * @ingroup ipv6HeaderExt
 *
 * @brief Header of IPv6 Extension Routing
 */
class Ipv6ExtensionRoutingHeader : public Ipv6ExtensionHeader
{
  public:
    /**
     * @brief Get the type identificator.
     * @return type identificator
     */
    static TypeId GetTypeId();

    /**
     * @brief Get the instance type ID.
     * @return instance type ID
     */
    TypeId GetInstanceTypeId() const override;

    /**
     * @brief Constructor.
     */
    Ipv6ExtensionRoutingHeader();

    /**
     * @brief Destructor.
     */
    ~Ipv6ExtensionRoutingHeader() override;

    /**
     * @brief Set the "Type of Routing" field.
     * @param typeRouting the type of routing
     */
    void SetTypeRouting(uint8_t typeRouting);

    /**
     * @brief Get the field "Type of Routing".
     * @return the type of routing
     */
    uint8_t GetTypeRouting() const;

    /**
     * @brief Set the "Segments left" field.
     * @param segmentsLeft the number of segments left
     */
    void SetSegmentsLeft(uint8_t segmentsLeft);

    /**
     * @brief Get the field "Segments left".
     * @return the number of segments left
     */
    uint8_t GetSegmentsLeft() const;

    /**
     * @brief Print some information about the packet.
     * @param os output stream
     */
    void Print(std::ostream& os) const override;

    /**
     * @brief Get the serialized size of the packet.
     * @return size
     */
    uint32_t GetSerializedSize() const override;

    /**
     * @brief Serialize the packet.
     * @param start Buffer iterator
     */
    void Serialize(Buffer::Iterator start) const override;

    /**
     * @brief Deserialize the packet.
     * @param start Buffer iterator
     * @return size of the packet
     */
    uint32_t Deserialize(Buffer::Iterator start) override;

  private:
    /**
     * @brief Type of routing.
     */
    uint8_t m_typeRouting;

    /**
     * @brief Number of left segments.
     */
    uint8_t m_segmentsLeft;
};

/**
 * @ingroup ipv6HeaderExt
 *
 * @brief Header of IPv6 Extension Routing : Type 0 (Loose Routing)
 */
class Ipv6ExtensionLooseRoutingHeader : public Ipv6ExtensionRoutingHeader
{
  public:
    /**
     * @brief Get the type identificator.
     * @return type identificator
     */
    static TypeId GetTypeId();

    /**
     * @brief Get the instance type ID.
     * @return instance type ID
     */
    TypeId GetInstanceTypeId() const override;

    /**
     * @brief Constructor.
     */
    Ipv6ExtensionLooseRoutingHeader();

    /**
     * @brief Destructor.
     */
    ~Ipv6ExtensionLooseRoutingHeader() override;

    /**
     * @brief Set the number of routers' address.
     * @param n the number of routers' address
     */
    void SetNumberAddress(uint8_t n);

    /**
     * @brief Set the vector of routers' address
     * @param routersAddress the vector of routers's address
     */
    void SetRoutersAddress(std::vector<Ipv6Address> routersAddress);

    /**
     * @brief Get the vector of routers' address
     * @return the vector of routers' address
     */
    std::vector<Ipv6Address> GetRoutersAddress() const;

    /**
     * @brief Set a Router IPv6 Address.
     * @param index the index of the IPv6 Address
     * @param addr the new IPv6 Address
     */
    void SetRouterAddress(uint8_t index, Ipv6Address addr);

    /**
     * @brief Get a Router IPv6 Address.
     * @param index the index of the IPv6 Address
     * @return the router IPv6 Address
     */
    Ipv6Address GetRouterAddress(uint8_t index) const;

    /**
     * @brief Print some information about the packet.
     * @param os output stream
     */
    void Print(std::ostream& os) const override;

    /**
     * @brief Get the serialized size of the packet.
     * @return size
     */
    uint32_t GetSerializedSize() const override;

    /**
     * @brief Serialize the packet.
     * @param start Buffer iterator
     */
    void Serialize(Buffer::Iterator start) const override;

    /**
     * @brief Deserialize the packet.
     * @param start Buffer iterator
     * @return size of the packet
     */
    uint32_t Deserialize(Buffer::Iterator start) override;

  private:
    /**
     * @brief A vector of IPv6 Address.
     */
    typedef std::vector<Ipv6Address> VectorIpv6Address_t;

    /**
     * @brief The vector of Routers' IPv6 Address.
     */
    VectorIpv6Address_t m_routersAddress;
};

/**
 * @ingroup ipv6HeaderExt
 *
 * @brief Header of IPv6 Extension ESP
 */
class Ipv6ExtensionESPHeader : public Ipv6ExtensionHeader
{
  public:
    /**
     * @brief Get the type identificator.
     * @return type identificator
     */
    static TypeId GetTypeId();

    /**
     * @brief Get the instance type ID.
     * @return instance type ID
     */
    TypeId GetInstanceTypeId() const override;

    /**
     * @brief Constructor.
     */
    Ipv6ExtensionESPHeader();

    /**
     * @brief Destructor.
     */
    ~Ipv6ExtensionESPHeader() override;

    /**
     * @brief Print some information about the packet.
     * @param os output stream
     */
    void Print(std::ostream& os) const override;

    /**
     * @brief Get the serialized size of the packet.
     * @return size
     */
    uint32_t GetSerializedSize() const override;

    /**
     * @brief Serialize the packet.
     * @param start Buffer iterator
     */
    void Serialize(Buffer::Iterator start) const override;

    /**
     * @brief Deserialize the packet.
     * @param start Buffer iterator
     * @return size of the packet
     */
    uint32_t Deserialize(Buffer::Iterator start) override;
};

/**
 * @ingroup ipv6HeaderExt
 *
 * @brief Header of IPv6 Extension AH
 */
class Ipv6ExtensionAHHeader : public Ipv6ExtensionHeader
{
  public:
    /**
     * @brief Get the type identificator.
     * @return type identificator
     */
    static TypeId GetTypeId();

    /**
     * @brief Get the instance type ID.
     * @return instance type ID
     */
    TypeId GetInstanceTypeId() const override;

    /**
     * @brief Constructor.
     */
    Ipv6ExtensionAHHeader();

    /**
     * @brief Destructor.
     */
    ~Ipv6ExtensionAHHeader() override;

    /**
     * @brief Print some information about the packet.
     * @param os output stream
     */
    void Print(std::ostream& os) const override;

    /**
     * @brief Get the serialized size of the packet.
     * @return size
     */
    uint32_t GetSerializedSize() const override;

    /**
     * @brief Serialize the packet.
     * @param start Buffer iterator
     */
    void Serialize(Buffer::Iterator start) const override;

    /**
     * @brief Deserialize the packet.
     * @param start Buffer iterator
     * @return size of the packet
     */
    uint32_t Deserialize(Buffer::Iterator start) override;
};

} // namespace ns3

#endif /* IPV6_EXTENSION_HEADER_H */
