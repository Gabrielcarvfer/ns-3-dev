/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#ifndef DNS_HEADER_H
#define DNS_HEADER_H

#include "ns3/address.h"
#include "ns3/header.h"
#include "ns3/nstime.h"

#include <optional>
#include <string>
#include <vector>

namespace ns3
{

/**
 * @ingroup dns-resolver
 * @brief DNS resource record (RFC 1035, section 3.2.1).
 *
 * The RDATA of A, AAAA, CNAME, NS and SOA records is decoded into the matching fields. For any
 * other type (e.g., OPT), the raw RDATA is kept as is (see GetRdata()).
 */
class DnsResourceRecord
{
  public:
    /// SOA record data (RFC 1035, section 3.3.13)
    struct Soa
    {
        std::string mname;   ///< Primary name server
        std::string rname;   ///< Mailbox of the zone administrator
        uint32_t serial{0};  ///< Zone serial number
        uint32_t refresh{0}; ///< Refresh interval, in seconds
        uint32_t retry{0};   ///< Retry interval, in seconds
        uint32_t expire{0};  ///< Expire time, in seconds
        uint32_t minimum{0}; ///< Negative caching TTL, in seconds

        /**
         * Get the negative caching TTL, with the same rules as DnsResourceRecord::GetTtl().
         *
         * @return The negative caching TTL.
         */
        Time GetMinimumTtl() const;
    };

    /**
     * @param name Owner name.
     */
    void SetName(const std::string& name);
    /**
     * @return The owner name.
     */
    const std::string& GetName() const;

    /**
     * @param type Record type.
     */
    void SetType(uint16_t type);
    /**
     * @return The record type.
     */
    uint16_t GetType() const;

    /**
     * @param rclass Record class (the UDP payload size for OPT records).
     */
    void SetClass(uint16_t rclass);
    /**
     * @return The record class (the UDP payload size for OPT records).
     */
    uint16_t GetClass() const;

    /**
     * @param ttl Raw TTL field, in seconds (the extended RCODE and flags for OPT records).
     */
    void SetTtl(uint32_t ttl);
    /**
     * @return The raw TTL field, as transmitted.
     */
    uint32_t GetRawTtl() const;
    /**
     * Get the TTL. Values with the most significant bit set are treated as zero (RFC 2181,
     * section 8).
     *
     * @return The TTL.
     */
    Time GetTtl() const;

    /**
     * Set the upper 8 bits of the extended RCODE carried in the TTL field of an OPT record
     * (RFC 6891, section 6.1.3).
     *
     * @param extendedRcode Upper 8 bits of the RCODE.
     */
    void SetExtendedRcode(uint8_t extendedRcode);
    /**
     * @return The upper 8 bits of the RCODE carried by an OPT record.
     */
    uint8_t GetExtendedRcode() const;
    /**
     * @return The EDNS version carried by an OPT record.
     */
    uint8_t GetEdnsVersion() const;

    /**
     * @param address Address of an A (Ipv4Address) or AAAA (Ipv6Address) record.
     */
    void SetAddress(const Address& address);
    /**
     * @return The address of an A or AAAA record.
     */
    const Address& GetAddress() const;

    /**
     * @param target Target name of a CNAME or NS record.
     */
    void SetTarget(const std::string& target);
    /**
     * @return The target name of a CNAME or NS record.
     */
    const std::string& GetTarget() const;

    /**
     * @param soa Data of an SOA record.
     */
    void SetSoa(const Soa& soa);
    /**
     * @return The data of an SOA record.
     */
    const Soa& GetSoa() const;

    /**
     * @param rdata Raw RDATA of a record type that is not decoded.
     */
    void SetRdata(const std::vector<uint8_t>& rdata);
    /**
     * @return The raw RDATA of a record type that is not decoded.
     */
    const std::vector<uint8_t>& GetRdata() const;

    /**
     * Check whether the record is of class IN and has the given type and owner name.
     *
     * @param type Record type.
     * @param owner Owner name, normalized with DnsHeader::NormalizeName().
     * @return True if the record matches.
     */
    bool Matches(uint16_t type, const std::string& owner) const;

    /**
     * @return The size of the serialized record, in bytes.
     */
    uint32_t GetSerializedSize() const;

    /**
     * Serialize the record.
     *
     * @param i Buffer iterator, advanced past the record.
     */
    void Serialize(Buffer::Iterator& i) const;

    /**
     * Deserialize the record.
     *
     * @param i Buffer iterator, advanced past the record.
     * @param start Start of the message, used to resolve compression pointers.
     * @return True if the record is well-formed.
     */
    bool Deserialize(Buffer::Iterator& i, Buffer::Iterator start);

  private:
    /**
     * @return The size of the RDATA, in bytes.
     */
    uint32_t GetRdataSize() const;

    std::string m_name;           ///< Owner name
    uint16_t m_type{0};           ///< Record type
    uint16_t m_class{1};          ///< Record class (IN by default)
    uint32_t m_ttl{0};            ///< Raw TTL field
    Address m_address;            ///< A or AAAA address
    std::string m_target;         ///< CNAME or NS target
    Soa m_soa;                    ///< SOA data
    std::vector<uint8_t> m_rdata; ///< Raw RDATA of other record types
};

/**
 * @ingroup dns-resolver
 * @brief DNS message header and body (RFC 1035, section 4), with EDNS(0) support (RFC 6891).
 *
 * A message carries at most one question. Names use presentation format: labels separated by
 * dots, with literal dots and backslashes escaped by a backslash, and no trailing dot (the root
 * is the empty string). Case is preserved.
 *
 * Deserialize() returns 0 for malformed messages, e.g., truncated messages, more than one
 * question, invalid names or compression pointers, records overrunning the message, A/AAAA/SOA
 * records with a wrong length, or a misplaced or duplicated OPT record.
 */
class DnsHeader : public Header
{
  public:
    /// Record types decoded by DnsResourceRecord
    enum RecordType : uint16_t
    {
        TYPE_A = 1,     ///< IPv4 address
        TYPE_NS = 2,    ///< Authoritative name server
        TYPE_CNAME = 5, ///< Canonical name
        TYPE_SOA = 6,   ///< Start of authority
        TYPE_AAAA = 28, ///< IPv6 address
        TYPE_OPT = 41,  ///< EDNS(0) pseudo-record
    };

    /// Response codes
    enum Rcode : uint16_t
    {
        RCODE_NOERROR = 0,  ///< No error
        RCODE_FORMERR = 1,  ///< Format error
        RCODE_SERVFAIL = 2, ///< Server failure
        RCODE_NXDOMAIN = 3, ///< Non-existent domain
        RCODE_NOTIMP = 4,   ///< Not implemented
        RCODE_REFUSED = 5,  ///< Query refused
        RCODE_BADVERS = 16, ///< Bad EDNS version
    };

    static constexpr uint16_t CLASS_IN{1};     ///< Internet class
    static constexpr uint8_t OPCODE_QUERY{0};  ///< Standard query opcode
    static constexpr uint32_t HEADER_SIZE{12}; ///< Size of the fixed header, in bytes

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

    /**
     * Set the message ID.
     * @param id Message ID.
     */
    void SetId(uint16_t id);
    /**
     * Get the message ID.
     * @return The message ID.
     */
    uint16_t GetId() const;

    /**
     * Set the QR flag.
     * @param response True for a response, false for a query.
     */
    void SetResponse(bool response);
    /**
     * Check the QR flag.
     * @return True if the message is a response.
     */
    bool IsResponse() const;

    /**
     * Set the opcode.
     * @param opcode Opcode (4 bits).
     */
    void SetOpcode(uint8_t opcode);
    /**
     * Get the opcode.
     * @return The opcode.
     */
    uint8_t GetOpcode() const;

    /**
     * Set the AA flag.
     * @param authoritative True if the answer is authoritative.
     */
    void SetAuthoritativeAnswer(bool authoritative);
    /**
     * Check the AA flag.
     * @return True if the answer is authoritative.
     */
    bool IsAuthoritativeAnswer() const;

    /**
     * Set the TC flag.
     * @param truncated True if the message is truncated.
     */
    void SetTruncated(bool truncated);
    /**
     * Check the TC flag.
     * @return True if the message is truncated.
     */
    bool IsTruncated() const;

    /**
     * Set the RD flag.
     * @param desired True if recursion is desired.
     */
    void SetRecursionDesired(bool desired);
    /**
     * Check the RD flag.
     * @return True if recursion is desired.
     */
    bool IsRecursionDesired() const;

    /**
     * Set the RA flag.
     * @param available True if the server supports recursion.
     */
    void SetRecursionAvailable(bool available);
    /**
     * Check the RA flag.
     * @return True if the server supports recursion.
     */
    bool IsRecursionAvailable() const;

    /**
     * Set the response code.
     *
     * The upper 8 bits of an extended RCODE are stored in the OPT record, which must be added
     * beforehand.
     *
     * @param rcode Response code.
     */
    void SetRcode(uint16_t rcode);
    /**
     * Get the response code, including the extended bits from the OPT record, if present.
     * @return The response code.
     */
    uint16_t GetRcode() const;

    /**
     * Set the question.
     * @param name Queried name.
     * @param type Queried record type.
     * @param rclass Queried record class.
     */
    void SetQuestion(const std::string& name, uint16_t type, uint16_t rclass = CLASS_IN);
    /**
     * Check if the message has a question.
     * @return True if a question is present.
     */
    bool HasQuestion() const;
    /**
     * Get the queried name.
     * @return The queried name.
     */
    const std::string& GetQuestionName() const;
    /**
     * Get the queried record type.
     * @return The queried record type.
     */
    uint16_t GetQuestionType() const;
    /**
     * Get the queried record class.
     * @return The queried record class.
     */
    uint16_t GetQuestionClass() const;

    /**
     * Add a record to the answer section.
     * @param record Record to add.
     */
    void AddAnswer(const DnsResourceRecord& record);
    /**
     * Add a record to the authority section.
     * @param record Record to add.
     */
    void AddAuthority(const DnsResourceRecord& record);
    /**
     * Add a record to the additional section.
     * @param record Record to add.
     */
    void AddAdditional(const DnsResourceRecord& record);
    /**
     * Get the answer section.
     * @return The answer records.
     */
    const std::vector<DnsResourceRecord>& GetAnswers() const;
    /**
     * Get the authority section.
     * @return The authority records.
     */
    const std::vector<DnsResourceRecord>& GetAuthorities() const;
    /**
     * Get the additional section.
     * @return The additional records.
     */
    const std::vector<DnsResourceRecord>& GetAdditionalRecords() const;

    /**
     * Add an EDNS(0) OPT record, without options, to the additional section.
     * @param udpPayloadSize Advertised UDP payload size.
     */
    void AddOptRecord(uint16_t udpPayloadSize);
    /**
     * Get the OPT record.
     * @return The OPT record, or std::nullopt if there is none.
     */
    std::optional<DnsResourceRecord> GetOptRecord() const;

    /**
     * Split a name in presentation format into labels.
     *
     * A name is invalid if it has an empty label, a label longer than 63 bytes, an encoded size
     * above 255 bytes, or a trailing backslash.
     *
     * @param name Name to split.
     * @return The labels (empty for the root), or std::nullopt if the name is invalid.
     */
    static std::optional<std::vector<std::string>> SplitName(const std::string& name);

    /**
     * Normalize a name for comparison: ASCII lower case, without a trailing dot.
     *
     * @param name Name in presentation format.
     * @return The normalized name.
     */
    static std::string NormalizeName(const std::string& name);

    /**
     * Check whether a name is equal to, or a subdomain of, a zone, comparing whole labels.
     *
     * @param name Normalized name.
     * @param zone Normalized zone name.
     * @return True if the name is in the zone.
     */
    static bool IsInZone(const std::string& name, const std::string& zone);

  private:
    /// Header flag bits
    enum Flags : uint16_t
    {
        FLAG_QR = 0x8000, ///< Response
        FLAG_AA = 0x0400, ///< Authoritative answer
        FLAG_TC = 0x0200, ///< Truncated
        FLAG_RD = 0x0100, ///< Recursion desired
        FLAG_RA = 0x0080, ///< Recursion available
    };

    /**
     * Set or clear a header flag.
     * @param flag Flag to change.
     * @param value True to set the flag, false to clear it.
     */
    void SetFlag(Flags flag, bool value);

    /**
     * Deserialize a resource record.
     * @param i Buffer iterator, advanced past the record.
     * @param start Start of the message, used to resolve compression pointers.
     * @param record Deserialized record.
     * @return True if the record is well-formed.
     */
    static bool DeserializeRecord(Buffer::Iterator& i,
                                  Buffer::Iterator start,
                                  DnsResourceRecord& record);

    uint16_t m_id{0};                                   ///< Message ID
    uint16_t m_flags{0};                                ///< Flags, opcode and RCODE
    bool m_hasQuestion{false};                          ///< True if a question is present
    std::string m_questionName;                         ///< Queried name
    uint16_t m_questionType{0};                         ///< Queried record type
    uint16_t m_questionClass{CLASS_IN};                 ///< Queried record class
    std::vector<DnsResourceRecord> m_answers;           ///< Answer section
    std::vector<DnsResourceRecord> m_authorities;       ///< Authority section
    std::vector<DnsResourceRecord> m_additionalRecords; ///< Additional section
};

} // namespace ns3

#endif /* DNS_HEADER_H */
