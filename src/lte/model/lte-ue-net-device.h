/*
 * Copyright (c) 2010 TELEMATICS LAB, DEE - Politecnico di Bari
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Giuseppe Piro  <g.piro@poliba.it>
 *         Nicola Baldo <nbaldo@cttc.es>
 * Modified by:
 *          Danilo Abrignani <danilo.abrignani@unibo.it> (Carrier Aggregation - GSoC 2015)
 *          Biljana Bojovic <biljana.bojovic@cttc.es> (Carrier Aggregation)
 */

#ifndef LTE_UE_NET_DEVICE_H
#define LTE_UE_NET_DEVICE_H

#include "component-carrier-ue.h"
#include "lte-net-device.h"

#include "ns3/event-id.h"
#include "ns3/nstime.h"
#include "ns3/traced-callback.h"

#include <map>
#include <vector>

namespace ns3
{

class Packet;
class PacketBurst;
class Node;
class LtePhy;
class LteUePhy;
class LteEnbNetDevice;
class LteUeMac;
class LteUeRrc;
class EpcUeNas;
class EpcTft;
class LteUeComponentCarrierManager;

/**
 * @ingroup lte
 * The LteUeNetDevice class implements the UE net device
 */
class LteUeNetDevice : public LteNetDevice
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    LteUeNetDevice();
    ~LteUeNetDevice() override;
    void DoDispose() override;

    // inherited from NetDevice
    bool Send(Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber) override;

    /**
     * @brief Get the MAC.
     * @return the LTE UE MAC
     */
    Ptr<LteUeMac> GetMac() const;

    /**
     * @brief Get the RRC.
     * @return the LTE UE RRC
     */
    Ptr<LteUeRrc> GetRrc() const;

    /**
     * @brief Get the Phy.
     * @return the LTE UE Phy
     */
    Ptr<LteUePhy> GetPhy() const;

    /**
     * @brief Get the NAS.
     * @return the LTE UE NAS
     */
    Ptr<EpcUeNas> GetNas() const;

    /**
     * @brief Get the componentn carrier manager.
     * @return the LTE UE component carrier manager
     */
    Ptr<LteUeComponentCarrierManager> GetComponentCarrierManager() const;

    /**
     * @brief Get the IMSI.
     * @return the IMSI
     */
    uint64_t GetImsi() const;

    /**
     * @return the downlink carrier frequency (EARFCN)
     *
     * Note that real-life handset typically supports more than one EARFCN, but
     * the sake of simplicity we assume only one EARFCN is supported.
     */
    uint32_t GetDlEarfcn() const;

    /**
     * @param earfcn the downlink carrier frequency (EARFCN)
     *
     * Note that real-life handset typically supports more than one EARFCN, but
     * the sake of simplicity we assume only one EARFCN is supported.
     */
    void SetDlEarfcn(uint32_t earfcn);

    /**
     * @brief Returns the CSG ID the UE is currently a member of.
     * @return the Closed Subscriber Group identity
     */
    uint32_t GetCsgId() const;

    /**
     * @brief Enlist the UE device as a member of a particular CSG.
     * @param csgId the intended Closed Subscriber Group identity
     *
     * UE is associated with a single CSG identity, and thus becoming a member of
     * this particular CSG. As a result, the UE may gain access to cells which
     * belong to this CSG. This does not revoke the UE's access to non-CSG cells.
     *
     * @note This restriction only applies to initial cell selection and
     *       EPC-enabled simulation.
     */
    void SetCsgId(uint32_t csgId);

    /**
     * @brief Set the target eNB where the UE is registered
     * @param enb
     */
    void SetTargetEnb(Ptr<LteEnbNetDevice> enb);

    /**
     * @brief Get the target eNB where the UE is registered
     * @return the pointer to the enb
     */
    Ptr<LteEnbNetDevice> GetTargetEnb();

    /**
     * @brief Set the ComponentCarrier Map for the UE
     * @param ccm the map of ComponentCarrierUe
     */
    void SetCcMap(std::map<uint8_t, Ptr<ComponentCarrierUe>> ccm);

    /**
     * @brief Get the ComponentCarrier Map for the UE
     * @returns the map of ComponentCarrierUe
     */
    std::map<uint8_t, Ptr<ComponentCarrierUe>> GetCcMap();

  protected:
    // inherited from Object
    void DoInitialize() override;

  private:
    bool m_isConstructed; ///< is constructed?

    /**
     * @brief Propagate attributes and configuration to sub-modules.
     *
     * Several attributes (e.g., the IMSI) are exported as the attributes of the
     * LteUeNetDevice from a user perspective, but are actually used also in other
     * sub-modules (the RRC, the PHY, etc.). This method takes care of updating
     * the configuration of all these sub-modules so that their copy of attribute
     * values are in sync with the one in the LteUeNetDevice.
     */
    void UpdateConfig();

    Ptr<LteEnbNetDevice> m_targetEnb; ///< target ENB

    Ptr<LteUeRrc> m_rrc;                                         ///< the RRC
    Ptr<EpcUeNas> m_nas;                                         ///< the NAS
    Ptr<LteUeComponentCarrierManager> m_componentCarrierManager; ///< the component carrier manager

    uint64_t m_imsi; ///< the IMSI

    uint32_t m_dlEarfcn; /**< downlink carrier frequency */

    uint32_t m_csgId; ///< the CSG ID

    std::map<uint8_t, Ptr<ComponentCarrierUe>> m_ccMap; ///< CC map
};

} // namespace ns3

#endif /* LTE_UE_NET_DEVICE_H */
