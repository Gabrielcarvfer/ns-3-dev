/*
 * Copyright (c) 2009 INRIA, UDcast
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 *         Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 */

#ifndef BS_SERVICE_FLOW_MANAGER_H
#define BS_SERVICE_FLOW_MANAGER_H

#include "bs-net-device.h"
#include "mac-messages.h"
#include "service-flow-manager.h"

#include "ns3/buffer.h"
#include "ns3/event-id.h"

#include <stdint.h>

namespace ns3
{

class Packet;
class ServiceFlow;
class WimaxNetDevice;
class SSRecord;
class WimaxConnection;
class BaseStationNetDevice;

/**
 * @ingroup wimax
 * @brief BsServiceFlowManager
 */
class BsServiceFlowManager : public ServiceFlowManager
{
  public:
    /// Confirmation code enumeration
    enum ConfirmationCode // as per Table 384 (not all codes implemented)
    {
        CONFIRMATION_CODE_SUCCESS,
        CONFIRMATION_CODE_REJECT
    };

    /**
     * Constructor
     *
     * @param device base station device
     */
    BsServiceFlowManager(Ptr<BaseStationNetDevice> device);
    ~BsServiceFlowManager() override;
    void DoDispose() override;
    /**
     * Register this type.
     * @return The TypeId.
     */
    static TypeId GetTypeId();

    /**
     * @brief Add a new service flow
     * @param serviceFlow the service flow to add
     */
    void AddServiceFlow(ServiceFlow* serviceFlow);
    /**
     * @param sfid the service flow identifier
     * @return the service flow which has as identifier sfid
     */
    ServiceFlow* GetServiceFlow(uint32_t sfid) const;
    /**
     * @param cid the connection identifier
     * @return the service flow which has as connection identifier cid
     */
    ServiceFlow* GetServiceFlow(Cid cid) const;
    /**
     * @param schedulingType the scheduling type
     * @return the list of service flows configured with schedulingType as a QoS class
     */
    std::vector<ServiceFlow*> GetServiceFlows(ServiceFlow::SchedulingType schedulingType) const;
    /**
     * @brief set the maximum Dynamic ServiceFlow Add (DSA) retries
     * @param maxDsaRspRetries the maximum DSA response retries
     */
    void SetMaxDsaRspRetries(uint8_t maxDsaRspRetries);

    /**
     * @return the DSA ack timeout event
     */
    EventId GetDsaAckTimeoutEvent() const;
    /**
     * @brief allocate service flows
     * @param dsaReq the DSA request
     * @param cid the connection identifier
     */
    void AllocateServiceFlows(const DsaReq& dsaReq, Cid cid);
    /**
     * @brief add a multicast service flow
     * @param sf the service flow
     * @param modulation the wimax phy modulation type
     */
    void AddMulticastServiceFlow(ServiceFlow sf, WimaxPhy::ModulationType modulation);
    /**
     * @brief process a DSA-ACK message
     * @param dsaAck the message to process
     * @param cid the identifier of the connection on which the message was received
     */
    void ProcessDsaAck(const DsaAck& dsaAck, Cid cid);

    /**
     * @brief process a DSA-Req message
     * @param dsaReq the message to process
     * @param cid the identifier of the connection on which the message was received
     * @return a pointer to the service flow
     */
    ServiceFlow* ProcessDsaReq(const DsaReq& dsaReq, Cid cid);

  private:
    /**
     * Create DSA response function
     * @param serviceFlow service flow
     * @param transactionId transaction ID
     * @return the DSA response
     */
    DsaRsp CreateDsaRsp(const ServiceFlow* serviceFlow, uint16_t transactionId);
    /**
     * @return the maximum DSA response retries
     */
    uint8_t GetMaxDsaRspRetries() const;
    /**
     * Create DSA response function
     * @param serviceFlow service flow
     * @param cid the identifier of the connection on which the message was received
     */
    void ScheduleDsaRsp(ServiceFlow* serviceFlow, Cid cid);
    Ptr<WimaxNetDevice> m_device; ///< the device
    uint32_t m_sfidIndex;         ///< SFID index
    uint8_t m_maxDsaRspRetries;   ///< maximum number of DSA response retries
    EventId m_dsaAckTimeoutEvent; ///< DSA ack timeout event
    Cid m_inuseScheduleDsaRspCid; ///< in use schedule DSA response CID
};

} // namespace ns3

#endif /* BS_SERVICE_FLOW_MANAGER_H */
