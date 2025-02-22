/*
 * Copyright (c) 2007,2008,2009 INRIA, UDcast
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 *          Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *                               <amine.ismail@UDcast.com>
 */

/*
 *This file does not contain all MAC messages, the rest of MAC messages have
 *This been categorized as DL and UL messages and are placed in
 *This dl-mac-messages.h and ul-mac-messages.h files.
 */

#ifndef MANAGEMENT_MESSAGE_TYPE_H
#define MANAGEMENT_MESSAGE_TYPE_H

#include "ns3/header.h"

#include <stdint.h>

namespace ns3
{

/**
 * @ingroup wimax
 * Mac Management messages
 * Section 6.3.2.3 MAC Management messages page 42, Table 14 page 43
 */
class ManagementMessageType : public Header
{
  public:
    /// Message type enumeration
    enum MessageType
    {
        MESSAGE_TYPE_UCD = 0,
        MESSAGE_TYPE_DCD = 1,
        MESSAGE_TYPE_DL_MAP = 2,
        MESSAGE_TYPE_UL_MAP = 3,
        MESSAGE_TYPE_RNG_REQ = 4,
        MESSAGE_TYPE_RNG_RSP = 5,
        MESSAGE_TYPE_REG_REQ = 6,
        MESSAGE_TYPE_REG_RSP = 7,
        MESSAGE_TYPE_DSA_REQ = 11,
        MESSAGE_TYPE_DSA_RSP = 12,
        MESSAGE_TYPE_DSA_ACK = 13
    };

    ManagementMessageType();
    /**
     * Constructor
     *
     * @param type message type
     */
    ManagementMessageType(uint8_t type);
    ~ManagementMessageType() override;
    /**
     * Set type field
     * @param type the type
     */
    void SetType(uint8_t type);
    /**
     * Get type field
     * @returns the type value
     */
    uint8_t GetType() const;

    /** @returns the name field */
    std::string GetName() const;
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
    uint8_t m_type; ///< type
};

} // namespace ns3

#endif /* MANAGEMENT_MESSAGE_TYPE_H */

// ---------------------------------------------------------------------

#ifndef RNG_RSP_H
#define RNG_RSP_H

#include "cid.h"
#include "service-flow.h"

#include "ns3/header.h"
#include "ns3/mac48-address.h"

#include <stdint.h>

namespace ns3
{

/**
 * @ingroup wimax
 * This class implements the ranging response message described by "IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access
 * Systems" 6.3.2.3.6 Ranging response (RNG-RSP) message, page 50
 */
class RngRsp : public Header
{
  public:
    RngRsp();
    ~RngRsp() override;
    /**
     * @brief set the  Tx timing offset adjustment (signed 32-bit).
     * @param timingAdjust The time required to advance SS transmission so frames
     * arrive at the expected time instance at the BS.
     */
    void SetTimingAdjust(uint32_t timingAdjust);
    /**
     * @brief set the relative change in transmission power level that the SS should make in order
     * that transmissions arrive at the BS at the desired power. When subchannelization is employed,
     * the subscriber shall interpret the power offset adjustment as a required change to the
     * transmitted power density.
     * @param powerLevelAdjust the relative change in transmission power level
     */
    void SetPowerLevelAdjust(uint8_t powerLevelAdjust);
    /**
     * @brief set the relative change in transmission frequency that the SS should take in order to
     * better match the BS. This is fine-frequency adjustment within a channel, not reassignment to
     * a different channel
     * @param offsetFreqAdjust Offset frequency adjustment
     */
    void SetOffsetFreqAdjust(uint32_t offsetFreqAdjust);
    /**
     * @brief set the range status.
     * @param rangStatus Range status
     */
    void SetRangStatus(uint8_t rangStatus);
    /**
     * @brief set the Center frequency, in kHz, of new downlink channel where the SS should redo
     * initial ranging.
     * @param dlFreqOverride the Center frequency in kHz
     */
    void SetDlFreqOverride(uint32_t dlFreqOverride);
    /**
     * @brief set the identifier of the uplink channel with which the SS is to redo initial ranging
     * @param ulChnlIdOverride the uplink channel index
     */
    void SetUlChnlIdOverride(uint8_t ulChnlIdOverride);
    /**
     * @brief set the DL oper burst profile
     * @param dlOperBurstProfile the oper burt profile
     */
    void SetDlOperBurstProfile(uint16_t dlOperBurstProfile);
    /**
     * @brief set the MAC address
     * @param macAddress the MAC address
     */
    void SetMacAddress(Mac48Address macAddress);

    /**
     * @brief set basic CID.
     * @param basicCid Basic CID
     */
    void SetBasicCid(Cid basicCid);
    /**
     * @brief set primary CID.
     * @param primaryCid Primary CID
     */
    void SetPrimaryCid(Cid primaryCid);

    /**
     * @brief set AAS broadcast permission.
     * @param aasBdcastPermission AAS broadcast permission
     */
    void SetAasBdcastPermission(uint8_t aasBdcastPermission);
    /**
     * @brief set frame number.
     * @param frameNumber Frame number
     */
    void SetFrameNumber(uint32_t frameNumber);
    /**
     * @brief set initial range opp number.
     * @param initRangOppNumber Initial range opp number
     */
    void SetInitRangOppNumber(uint8_t initRangOppNumber);
    /**
     * @brief set range sub channel.
     * @param rangSubchnl Range subchannel
     */
    void SetRangSubchnl(uint8_t rangSubchnl);
    /**
     * @return Tx timing offset adjustment (signed 32-bit). The time required to advance SS
     * transmission so frames arrive at the expected time instance at the BS.
     */
    uint32_t GetTimingAdjust() const;
    /**
     * @return the relative change in transmission power level that the SS should take in order
     * that transmissions arrive at the BS at the desired power. When subchannelization is employed,
     * the subscriber shall interpret the power offset adjustment as a required change to the
     * transmitted power density.
     */
    uint8_t GetPowerLevelAdjust() const;
    /**
     * @return the relative change in transmission frequency that the SS should take in order to
     * better match the BS. This is fine-frequency adjustment within a channel, not reassignment to
     * a different channel.
     */
    uint32_t GetOffsetFreqAdjust() const;
    /**
     * @return the range status.
     */
    uint8_t GetRangStatus() const;
    /**
     * @return Center frequency, in kHz, of new downlink channel where the SS should redo initial
     * ranging.
     */
    uint32_t GetDlFreqOverride() const;
    /**
     * @return The identifier of the uplink channel with which the SS is to redo initial ranging
     */
    uint8_t GetUlChnlIdOverride() const;
    /**
     * @return DlOperBurstProfile: This parameter is sent in response to the RNG-REQ Requested
     * Downlink Burst Profile parameter
     */
    uint16_t GetDlOperBurstProfile() const;
    /**
     * @return MAC address
     */
    Mac48Address GetMacAddress() const;
    /**
     * @return basic CID
     */
    Cid GetBasicCid() const;
    /**
     * @return primary CID
     */
    Cid GetPrimaryCid() const;
    /**
     * @return AAS broadcast permission
     */
    uint8_t GetAasBdcastPermission() const;
    /**
     * @return frame number
     */
    uint32_t GetFrameNumber() const;
    /**
     * @return initial range opp number
     */
    uint8_t GetInitRangOppNumber() const;
    /**
     * @return range sub channel
     */
    uint8_t GetRangSubchnl() const;

    /**
     * @return name string
     */
    std::string GetName() const;
    /**
     * Register this type.
     * @return The TypeId.
     */
    static TypeId GetTypeId();
    TypeId GetInstanceTypeId() const override;
    void Print(std::ostream& os) const override;
    uint32_t GetSerializedSize() const override;
    void Serialize(Buffer::Iterator start) const override;
    uint32_t Deserialize(Buffer::Iterator start) override;

  private:
    uint8_t m_reserved; ///< changed as per the amendment 802.16e-2005

    // TLV Encoded Information

    /**
     * Tx timing offset adjustment (signed 32-bit). The time required to advance SS transmission so
     * frames arrive at the expected time instance at the BS.
     */
    uint32_t m_timingAdjust;

    /**
     * Specifies the relative change in transmission power level that the SS is to make in order
     * that transmissions arrive at the BS at the desired power. When subchannelization is employed,
     * the subscriber shall interpret the power offset adjustment as a required change to the
     * transmitted power density.
     */
    uint8_t m_powerLevelAdjust;

    /**
     * Specifies the relative change in transmission frequency that the SS is to make in order to
     * better match the BS. This is fine-frequency adjustment within a channel, not reassignment to
     * a different channel.
     */
    uint32_t m_offsetFreqAdjust;

    /**
     * range status.
     */
    uint8_t m_rangStatus;

    /// Center frequency, in kHz, of new downlink channel where the SS should redo initial ranging.
    uint32_t m_dlFreqOverride;

    /**
     * Licensed bands: The identifier of the uplink channel with which the SS is to redo initial
     * ranging (not used with PHYs without channelized uplinks).
     */
    uint8_t m_ulChnlIdOverride;

    /**
     * This parameter is sent in response to the RNG-REQ Requested Downlink Burst Profile parameter.
     * Byte 0: Specifies the least robust DIUC that may be used by the BS for transmissions to the
     * SS. Byte 1: Configuration Change Count value of DCD defining the burst profile associated
     * with DIUC.
     */
    uint16_t m_dlOperBurstProfile;

    Mac48Address m_macAddress;     ///< MAC address
    Cid m_basicCid;                ///< basic CID
    Cid m_primaryCid;              ///< primary CID
    uint8_t m_aasBdcastPermission; ///< AAS broadcast permission

    /**
     * Frame number where the associated RNG_REQ message was detected by the BS. Usage is mutually
     * exclusive with SS MAC Address
     */
    uint32_t m_frameNumber;

    /**
     * Initial Ranging opportunity (1–255) in which the associated RNG_REQ message was detected by
     * the BS. Usage is mutually exclusive with SS MAC Address
     */
    uint8_t m_initRangOppNumber;

    /**
     * Used to indicate the OFDM subchannel reference that was used to transmit the initial ranging
     * message (OFDM with subchannelization).
     */
    uint8_t m_rangSubchnl;
};

} // namespace ns3

#endif /* RNG_RSP_H */

// ---------------------------------------------------------------------

#ifndef DSA_REQ_H
#define DSA_REQ_H

#include "cid.h"
#include "service-flow.h"

#include "ns3/buffer.h"
#include "ns3/header.h"

#include <stdint.h>

namespace ns3
{
/**
 * @ingroup wimax
 * This class implements the DSA-REQ message described by "IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access
 * Systems" 6.3.2.3.10 DSA-REQ message, page 62
 */
class DsaReq : public Header
{
  public:
    DsaReq();
    ~DsaReq() override;
    /**
     * Constructor
     *
     * @param sf service flow
     */
    DsaReq(ServiceFlow sf);
    /**
     * @brief set the transaction ID
     * @param transactionId
     */
    void SetTransactionId(uint16_t transactionId);
    /**
     * @brief set the service flow identifier
     * @param sfid the service flow identifier
     */
    void SetSfid(uint32_t sfid);
    /**
     * @brief set the connection identifier
     * @param cid the connection identifier
     */
    void SetCid(Cid cid);
    /**
     * @brief specify a service flow to be requested by this message
     * @param sf the service flow
     */
    void SetServiceFlow(ServiceFlow sf);
    /**
     * @return the service flow requested by this message
     */
    ServiceFlow GetServiceFlow() const;
    /**
     * @return the transaction ID
     */
    uint16_t GetTransactionId() const;
    /**
     * @return the service flow identifier
     */
    uint32_t GetSfid() const;
    /**
     * @return the connection identifier
     */
    Cid GetCid() const;
    /**
     * @return the service name
     */
    std::string GetName() const;
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
    uint16_t m_transactionId; ///< transaction ID
    // TLV Encoded Information
    uint32_t m_sfid;           ///< SFID
    Cid m_cid;                 ///< CID
    ServiceFlow m_serviceFlow; ///< service flow
};

} // namespace ns3

#endif /* DSA_REQ_H */

// ---------------------------------------------------------------------

#ifndef DSA_RSP_H
#define DSA_RSP_H

#include "cid.h"

#include "ns3/buffer.h"
#include "ns3/header.h"

#include <stdint.h>

namespace ns3
{

/**
 * @ingroup wimax
 * This class implements the DSA-RSP message described by "IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access
 * Systems" 6.3.2.3.11 DSA-RSP message, page 63
 * @verbatim
   0             7             15            23
   +-------------+-------------+-------------+
   |Mngt msg type|       Transaction ID      |
   +-------------+-------------+-------------+
   | Conf Code   | Service Flow TLV          |
   +~~~~~~~~~~~~~+~~~~~~~~~~~~~+~~~~~~~~~~~~~+
   \endverbatim
 */
class DsaRsp : public Header
{
  public:
    DsaRsp();
    ~DsaRsp() override;

    /**
     * @brief set the transaction ID
     * @param transactionId
     */
    void SetTransactionId(uint16_t transactionId);
    /**
     * @return the transaction ID
     */
    uint16_t GetTransactionId() const;

    /**
     * @brief set the confirmation code
     * @param confirmationCode
     */
    void SetConfirmationCode(uint16_t confirmationCode);
    /**
     * @return the confirmation code
     */
    uint16_t GetConfirmationCode() const;
    /**
     * @brief set the service flow identifier
     * @param sfid the service flow identifier
     */
    void SetSfid(uint32_t sfid);
    /**
     * @return the service flow identifier
     */
    uint32_t GetSfid() const;
    /**
     * @brief set the connection identifier
     * @param cid the connection identifier
     */
    void SetCid(Cid cid);
    /**
     * @return the connection identifier
     */
    Cid GetCid() const;
    /**
     * @brief specify a service flow to be requested by this message
     * @param sf the service flow
     */
    void SetServiceFlow(ServiceFlow sf);
    /**
     * @return the service flow requested by this message
     */
    ServiceFlow GetServiceFlow() const;

    /**
     * @return the service name
     */
    std::string GetName() const;
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
    uint16_t m_transactionId;   ///< transaction ID
    uint8_t m_confirmationCode; ///< confirmation code
    // TLV Encoded Information
    ServiceFlow m_serviceFlow; ///< service flow
    uint32_t m_sfid;           ///< SFID
    Cid m_cid;                 ///< CID
};

} // namespace ns3

#endif /* DSA_RSP_H */

// ---------------------------------------------------------------------

#ifndef DSA_ACK_H
#define DSA_ACK_H

#include "ns3/buffer.h"
#include "ns3/header.h"

#include <stdint.h>

namespace ns3
{

/**
 * @ingroup wimax
 * This class implements the DSA-ACK message described by "IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access
 * Systems" 6.3.2.3.12 DSA-ACK message, page 64
 */
class DsaAck : public Header
{
  public:
    DsaAck();
    ~DsaAck() override;

    /**
     * Set transaction ID field
     * @param transactionId the transaction ID
     */
    void SetTransactionId(uint16_t transactionId);
    /**
     * Get transaction ID field
     * @returns the transaction ID
     */
    uint16_t GetTransactionId() const;

    /**
     * Set confirmation code field
     * @param confirmationCode the confirmation code
     */
    void SetConfirmationCode(uint16_t confirmationCode);
    /**
     * Get confirmation code field
     * @returns the confirmation code
     */
    uint16_t GetConfirmationCode() const;

    /**
     * Get name field
     * @return the name string
     */
    std::string GetName() const;
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
    uint16_t m_transactionId;   ///< transaction ID
    uint8_t m_confirmationCode; ///< confirmation code
};

} // namespace ns3

#endif /* DSA_ACK_H */

// ---------------------------------------------------------------------

#ifndef RNG_REQ_H
#define RNG_REQ_H

#include "service-flow.h"

#include "ns3/header.h"
#include "ns3/mac48-address.h"

#include <stdint.h>

namespace ns3
{

/**
 * @ingroup wimax
 * This class implements the ranging request message described by "IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access
 * Systems"
 */
class RngReq : public Header
{
  public:
    RngReq();
    ~RngReq() override;

    /**
     * Set request DL burst profile field
     * @param reqDlBurstProfile the request DL burst profile
     */
    void SetReqDlBurstProfile(uint8_t reqDlBurstProfile);
    /**
     * Set MAC address field
     * @param macAddress the MAC address
     */
    void SetMacAddress(Mac48Address macAddress);
    /**
     * Set ranging anomalies field
     * @param rangingAnomalies the rnaging anomalies
     */
    void SetRangingAnomalies(uint8_t rangingAnomalies);

    /**
     * Get request DL burst profile field
     * @returns the request DL burst profile
     */
    uint8_t GetReqDlBurstProfile() const;
    /**
     * Get MAC address field
     * @returns the MAC address
     */
    Mac48Address GetMacAddress() const;
    /**
     * Get ranging anomalies field
     * @returns the ranging anomalies
     */
    uint8_t GetRangingAnomalies() const;

    /**
     * @brief Get name field
     * @returns the name string
     */
    std::string GetName() const;
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();
    TypeId GetInstanceTypeId() const override;
    void Print(std::ostream& os) const override;
    /// Print debug function
    void PrintDebug() const;
    uint32_t GetSerializedSize() const override;
    void Serialize(Buffer::Iterator start) const override;
    uint32_t Deserialize(Buffer::Iterator start) override;

  private:
    uint8_t m_reserved; ///< changed as per the amendment 802.16e-2005

    // TLV Encoded Information
    uint8_t m_reqDlBurstProfile; ///< request DL burst profile
    Mac48Address m_macAddress;   ///< MAC address
    uint8_t m_rangingAnomalies;  ///< ranging anomalies
};

} // namespace ns3

#endif /* RNG_REQ_H */
