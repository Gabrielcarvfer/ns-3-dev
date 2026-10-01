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

#include <optional>
#include <string>
#include <vector>

namespace ns3
{

/**
 * @ingroup dns-resolver
 * @brief A DNS resource record (RFC 1035, section 3.2.1).
 *
 * The data of the A, AAAA, CNAME, NS and SOA records is decoded in the corresponding fields;
 * the data of the other records (e.g., the options of an OPT record) is kept as transmitted.
 */
struct DnsResourceRecord
{
    /// The data of an SOA record (RFC 1035, section 3.3.13)
    struct Soa
    {
        std::string mname;   ///< primary name server of the zone
        std::string rname;   ///< mailbox of the person responsible for the zone
        uint32_t serial{0};  ///< version number of the zone
        uint32_t refresh{0}; ///< refresh interval, in seconds
        uint32_t retry{0};   ///< retry interval, in seconds
        uint32_t expire{0};  ///< expiry time, in seconds
        uint32_t minimum{0}; ///< minimum TTL of the records of the zone, in seconds
    };

    std::string name;   ///< owner name
    uint16_t type{0};   ///< record type
    uint16_t rclass{1}; ///< record class (the UDP payload size of an OPT record)
    uint32_t ttl{0};    ///< TTL, as transmitted (the extended RCODE and flags of an OPT record)
    Address address;    ///< data of an A or AAAA record
    std::string target; ///< data of a CNAME or NS record
    Soa soa;            ///< data of an SOA record
    std::vector<uint8_t> rdata; ///< data of the other records
};

/**
 * @ingroup dns-resolver
 * @brief DNS message (RFC 1035, section 4), with the EDNS(0) OPT pseudo-record (RFC 6891).
 *
 * The message carries at most one question. The names are in presentation format: dot-separated
 * labels, in which the dots and backslashes are escaped with a backslash, without final dot
 * (the root is the empty string). Their case is preserved.
 *
 * Deserialize() returns 0 if the message is malformed: too short, with more than one question,
 * with a malformed name (e.g., a compression pointer that does not point backwards, or a name
 * longer than 255 octets), with a record extending beyond the message, with an address or SOA
 * record of the wrong length, or with an OPT record that is not the single one of the
 * additional section with the root as owner.
 */
class DnsHeader : public Header
{
  public:
    /// Types of the records decoded by DnsResourceRecord
    enum RecordType : uint16_t
    {
        TYPE_A = 1,     ///< IPv4 address
        TYPE_NS = 2,    ///< authoritative name server
        TYPE_CNAME = 5, ///< canonical name of an alias
        TYPE_SOA = 6,   ///< start of a zone of authority
        TYPE_AAAA = 28, ///< IPv6 address
        TYPE_OPT = 41,  ///< EDNS(0) pseudo-record
    };

    /// Response codes (RFC 1035, section 4.1.1, and RFC 6891, section 6.1.3)
    enum Rcode : uint16_t
    {
        RCODE_NOERROR = 0,  ///< no error
        RCODE_FORMERR = 1,  ///< format error
        RCODE_SERVFAIL = 2, ///< server failure
        RCODE_NXDOMAIN = 3, ///< the name does not exist
        RCODE_NOTIMP = 4,   ///< not implemented
        RCODE_REFUSED = 5,  ///< refused
        RCODE_BADVERS = 16, ///< unsupported EDNS version
    };

    static constexpr uint16_t CLASS_IN{1};     ///< Internet class
    static constexpr uint8_t OPCODE_QUERY{0};  ///< standard query
    static constexpr uint32_t HEADER_SIZE{12}; ///< size of the fixed header

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
     * @param id the identifier of the message
     */
    void SetId(uint16_t id);
    /**
     * @return the identifier of the message
     */
    uint16_t GetId() const;

    /**
     * @param response whether the message is a response (QR flag)
     */
    void SetResponse(bool response);
    /**
     * @return whether the message is a response (QR flag)
     */
    bool IsResponse() const;

    /**
     * @param opcode the kind of query (4 bits)
     */
    void SetOpcode(uint8_t opcode);
    /**
     * @return the kind of query
     */
    uint8_t GetOpcode() const;

    /**
     * @param authoritative whether the response is authoritative (AA flag)
     */
    void SetAuthoritativeAnswer(bool authoritative);
    /**
     * @return whether the response is authoritative (AA flag)
     */
    bool IsAuthoritativeAnswer() const;

    /**
     * @param truncated whether the message is truncated (TC flag)
     */
    void SetTruncated(bool truncated);
    /**
     * @return whether the message is truncated (TC flag)
     */
    bool IsTruncated() const;

    /**
     * @param desired whether the server should pursue the query recursively (RD flag)
     */
    void SetRecursionDesired(bool desired);
    /**
     * @return whether the server should pursue the query recursively (RD flag)
     */
    bool IsRecursionDesired() const;

    /**
     * @param available whether the server offers recursion (RA flag)
     */
    void SetRecursionAvailable(bool available);
    /**
     * @return whether the server offers recursion (RA flag)
     */
    bool IsRecursionAvailable() const;

    /**
     * Set the response code; its upper 8 bits are stored in the OPT record, which must have
     * been added first if they are not 0 (RFC 6891, section 6.1.3).
     * @param rcode the response code
     */
    void SetRcode(uint16_t rcode);
    /**
     * @return the response code, extended by the OPT record if any
     */
    uint16_t GetRcode() const;

    /**
     * Set the question of the message.
     * @param name the name
     * @param type the type of the records
     * @param rclass the class of the records
     */
    void SetQuestion(const std::string& name, uint16_t type, uint16_t rclass = CLASS_IN);
    /**
     * @return whether the message has a question
     */
    bool HasQuestion() const;
    /**
     * @return the name of the question
     */
    const std::string& GetQuestionName() const;
    /**
     * @return the type of the records of the question
     */
    uint16_t GetQuestionType() const;
    /**
     * @return the class of the records of the question
     */
    uint16_t GetQuestionClass() const;

    /**
     * Append a record to the answer section.
     * @param record the record
     */
    void AddAnswer(const DnsResourceRecord& record);
    /**
     * Append a record to the authority section.
     * @param record the record
     */
    void AddAuthority(const DnsResourceRecord& record);
    /**
     * Append a record to the additional section.
     * @param record the record
     */
    void AddAdditional(const DnsResourceRecord& record);
    /**
     * @return the records of the answer section
     */
    const std::vector<DnsResourceRecord>& GetAnswers() const;
    /**
     * @return the records of the authority section
     */
    const std::vector<DnsResourceRecord>& GetAuthorities() const;
    /**
     * @return the records of the additional section
     */
    const std::vector<DnsResourceRecord>& GetAdditionalRecords() const;

    /**
     * Add an EDNS(0) OPT record to the additional section, without options (RFC 6891).
     * @param udpPayloadSize the UDP payload size to advertise
     */
    void AddOptRecord(uint16_t udpPayloadSize);
    /**
     * @return the OPT record, if any
     */
    std::optional<DnsResourceRecord> GetOptRecord() const;

    /**
     * Split a name in presentation format into its labels.
     * @param name the name
     * @return the labels (none for the root), or nothing if the name is not valid: an empty
     * label, a label longer than 63 octets, more than 255 octets once encoded, or a final
     * backslash
     */
    static std::optional<std::vector<std::string>> SplitName(const std::string& name);

  private:
    /// Positions of the flags in the second 16-bit word of the header
    enum Flags : uint16_t
    {
        FLAG_QR = 0x8000, ///< response
        FLAG_AA = 0x0400, ///< authoritative answer
        FLAG_TC = 0x0200, ///< truncated
        FLAG_RD = 0x0100, ///< recursion desired
        FLAG_RA = 0x0080, ///< recursion available
    };

    /**
     * Set or clear a flag.
     * @param flag the flag
     * @param value the value
     */
    void SetFlag(Flags flag, bool value);

    /**
     * Serialize a record.
     * @param i the iterator, advanced past the record
     * @param record the record
     */
    static void SerializeRecord(Buffer::Iterator& i, const DnsResourceRecord& record);

    /**
     * Deserialize a record.
     * @param i the iterator, advanced past the record
     * @param start the start of the message, for the compression pointers
     * @param record the record
     * @return true if the record is well-formed
     */
    static bool DeserializeRecord(Buffer::Iterator& i,
                                  Buffer::Iterator start,
                                  DnsResourceRecord& record);

    uint16_t m_id{0};                                   ///< identifier
    uint16_t m_flags{0};                                ///< flags, opcode and RCODE
    bool m_hasQuestion{false};                          ///< whether there is a question
    std::string m_questionName;                         ///< name of the question
    uint16_t m_questionType{0};                         ///< type of the question
    uint16_t m_questionClass{CLASS_IN};                 ///< class of the question
    std::vector<DnsResourceRecord> m_answers;           ///< answer section
    std::vector<DnsResourceRecord> m_authorities;       ///< authority section
    std::vector<DnsResourceRecord> m_additionalRecords; ///< additional section
};

} // namespace ns3

#endif /* DNS_HEADER_H */
