/*
 * Copyright (c) 2020 Universita' degli Studi di Napoli Federico II
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#include "wifi-default-protection-manager.h"

#include "ap-wifi-mac.h"
#include "sta-wifi-mac.h"
#include "wifi-mpdu.h"
#include "wifi-tx-parameters.h"

#include "ns3/boolean.h"
#include "ns3/eht-frame-exchange-manager.h"
#include "ns3/emlsr-manager.h"
#include "ns3/erp-ofdm-phy.h"
#include "ns3/log.h"

#include <type_traits>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("WifiDefaultProtectionManager");

NS_OBJECT_ENSURE_REGISTERED(WifiDefaultProtectionManager);

TypeId
WifiDefaultProtectionManager::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::WifiDefaultProtectionManager")
            .SetParent<WifiProtectionManager>()
            .SetGroupName("Wifi")
            .AddConstructor<WifiDefaultProtectionManager>()
            .AddAttribute("EnableMuRts",
                          "If enabled, always protect a DL/UL MU frame exchange with MU-RTS/CTS.",
                          BooleanValue(false),
                          MakeBooleanAccessor(&WifiDefaultProtectionManager::m_sendMuRts),
                          MakeBooleanChecker())
            .AddAttribute("SingleRtsPerTxop",
                          "If enabled, a protection mechanism (RTS or MU-RTS) is normally used no "
                          "more than once in a TXOP, regardless of the destination of the data "
                          "frame (unless required for specific purposes, such as transmitting an "
                          "Initial Control Frame to an EMLSR client).",
                          BooleanValue(false),
                          MakeBooleanAccessor(&WifiDefaultProtectionManager::m_singleRtsPerTxop),
                          MakeBooleanChecker())
            .AddAttribute("SkipMuRtsBeforeBsrp",
                          "If enabled, MU-RTS is not used to protect the transmission of a BSRP "
                          "Trigger Frame.",
                          BooleanValue(true),
                          MakeBooleanAccessor(&WifiDefaultProtectionManager::m_skipMuRtsBeforeBsrp),
                          MakeBooleanChecker());
    return tid;
}

WifiDefaultProtectionManager::WifiDefaultProtectionManager()
{
    NS_LOG_FUNCTION(this);
}

WifiDefaultProtectionManager::~WifiDefaultProtectionManager()
{
    NS_LOG_FUNCTION_NOARGS();
}

std::unique_ptr<WifiProtection>
WifiDefaultProtectionManager::TryAddMpdu(Ptr<const WifiMpdu> mpdu, const WifiTxParameters& txParams)
{
    NS_LOG_FUNCTION(this << *mpdu << &txParams);

    // Call a separate method that handles MU-RTS/CTS protection in case of DL MU PPDU containing
    // more than one PSDU or in case the MPDU being added is addressed to an EMLSR client or in
    // case the protection method is already MU-RTS/CTS.
    const auto& psduInfoMap = txParams.GetPsduInfoMap();
    auto dlMuPpdu = txParams.m_txVector.IsDlMu() && psduInfoMap.size() > 1;
    const auto& hdr = mpdu->GetHeader();
    auto isEmlsrDestination = GetWifiRemoteStationManager()->GetEmlsrEnabled(hdr.GetAddr1());

    if (dlMuPpdu || isEmlsrDestination ||
        (txParams.m_protection && txParams.m_protection->method == WifiProtection::MU_RTS_CTS))
    {
        return TryAddMpduToMuPpdu(mpdu, txParams);
    }

    // No protection for TB PPDUs (the soliciting Trigger Frame can be protected by an MU-RTS)
    if (txParams.m_txVector.IsUlMu())
    {
        if (txParams.m_protection)
        {
            NS_ASSERT(txParams.m_protection->method == WifiProtection::NONE);
            return nullptr;
        }
        return std::make_unique<WifiNoProtection>();
    }

    // if this is a Trigger Frame, call a separate method
    if (hdr.IsTrigger())
    {
        return TryUlMuTransmission(mpdu, txParams);
    }

    // if the current protection method (if any) is already RTS/CTS or CTS-to-Self,
    // it will not change by adding an MPDU
    if (txParams.m_protection && (txParams.m_protection->method == WifiProtection::RTS_CTS ||
                                  txParams.m_protection->method == WifiProtection::CTS_TO_SELF))
    {
        return nullptr;
    }

    // if a protection method is set, it must be NONE
    NS_ASSERT(!txParams.m_protection || txParams.m_protection->method == WifiProtection::NONE);

    std::unique_ptr<WifiProtection> protection;
    protection = GetPsduProtection(hdr, txParams);

    // return the newly computed method if none was set or it is not NONE
    if (!txParams.m_protection || protection->method != WifiProtection::NONE)
    {
        return protection;
    }
    // the protection method has not changed
    return nullptr;
}

std::unique_ptr<WifiProtection>
WifiDefaultProtectionManager::TryAggregateMsdu(Ptr<const WifiMpdu> msdu,
                                               const WifiTxParameters& txParams)
{
    NS_LOG_FUNCTION(this << *msdu << &txParams);

    // if the current protection method is already RTS/CTS, CTS-to-Self or MU-RTS/CTS,
    // it will not change by aggregating an MSDU
    NS_ASSERT(txParams.m_protection);
    if (txParams.m_protection->method == WifiProtection::RTS_CTS ||
        txParams.m_protection->method == WifiProtection::CTS_TO_SELF ||
        txParams.m_protection->method == WifiProtection::MU_RTS_CTS)
    {
        return nullptr;
    }

    NS_ASSERT(txParams.m_protection->method == WifiProtection::NONE);

    // No protection for TB PPDUs and DL MU PPDUs containing more than one PSDU
    if (txParams.m_txVector.IsUlMu() ||
        (txParams.m_txVector.IsDlMu() && txParams.GetPsduInfoMap().size() > 1))
    {
        return nullptr;
    }

    std::unique_ptr<WifiProtection> protection;
    protection = GetPsduProtection(msdu->GetHeader(), txParams);

    // the protection method may still be none
    if (protection->method == WifiProtection::NONE)
    {
        return nullptr;
    }

    // the protection method has changed
    return protection;
}

std::unique_ptr<WifiProtection>
WifiDefaultProtectionManager::GetPsduProtection(const WifiMacHeader& hdr,
                                                const WifiTxParameters& txParams) const
{
    NS_LOG_FUNCTION(this << hdr << &txParams);

    // a non-initial fragment does not need to be protected, unless it is being retransmitted
    if (hdr.GetFragmentNumber() > 0 && !hdr.IsRetry())
    {
        return std::make_unique<WifiNoProtection>();
    }

    // no need to use protection if destination already received an RTS in this TXOP or
    // SingleRtsPerTxop is true and a protection mechanism has been already used in this TXOP
    if (const auto& protectedStas = m_mac->GetFrameExchangeManager(m_linkId)->GetProtectedStas();
        protectedStas.contains(hdr.GetAddr1()) || (m_singleRtsPerTxop && !protectedStas.empty()))
    {
        return std::make_unique<WifiNoProtection>();
    }

    // when an EMLSR client starts an UL TXOP on a link while the MediumSyncDelay timer is running
    // or on a link on which the main PHY is not operating, it needs to send an RTS frame
    bool emlsrNeedRts = false;

    if (auto staMac = DynamicCast<StaWifiMac>(m_mac))
    {
        auto emlsrManager = staMac->GetEmlsrManager();

        emlsrNeedRts = emlsrManager && staMac->IsEmlsrLink(m_linkId) &&
                       (emlsrManager->GetElapsedMediumSyncDelayTimer(m_linkId) ||
                        m_mac->GetLinkForPhy(emlsrManager->GetMainPhyId()) != m_linkId);
    }

    // check if RTS/CTS is needed
    if (emlsrNeedRts || GetWifiRemoteStationManager()->NeedRts(hdr, txParams))
    {
        auto protection = std::make_unique<WifiRtsCtsProtection>();
        protection->rtsTxVector =
            GetWifiRemoteStationManager()->GetRtsTxVector(hdr.GetAddr1(),
                                                          txParams.m_txVector.GetChannelWidth());
        protection->ctsTxVector =
            GetWifiRemoteStationManager()->GetCtsTxVector(hdr.GetAddr1(),
                                                          protection->rtsTxVector.GetMode());
        return protection;
    }

    // check if CTS-to-Self is needed
    if (GetWifiRemoteStationManager()->NeedCtsToSelf(txParams.m_txVector, hdr))
    {
        auto protection = std::make_unique<WifiCtsToSelfProtection>();
        protection->ctsTxVector = GetWifiRemoteStationManager()->GetCtsToSelfTxVector();
        return protection;
    }

    return std::make_unique<WifiNoProtection>();
}

std::unique_ptr<WifiProtection>
WifiDefaultProtectionManager::TryAddMpduToMuPpdu(Ptr<const WifiMpdu> mpdu,
                                                 const WifiTxParameters& txParams)
{
    NS_LOG_FUNCTION(this << *mpdu << &txParams);

    auto receiver = mpdu->GetHeader().GetAddr1();
    const auto& psduInfoMap = txParams.GetPsduInfoMap();
    auto dlMuPpdu = txParams.m_txVector.IsDlMu() && psduInfoMap.size() > 1;
    auto isEmlsrDestination = GetWifiRemoteStationManager()->GetEmlsrEnabled(receiver);
    NS_ASSERT(
        dlMuPpdu || isEmlsrDestination ||
        (txParams.m_protection && txParams.m_protection->method == WifiProtection::MU_RTS_CTS));

    const auto& protectedStas = m_mac->GetFrameExchangeManager(m_linkId)->GetProtectedStas();
    const auto isProtected = protectedStas.contains(receiver);
    bool needMuRts =
        (txParams.m_protection && txParams.m_protection->method == WifiProtection::MU_RTS_CTS) ||
        (dlMuPpdu && m_sendMuRts && !isProtected &&
         (!m_singleRtsPerTxop || protectedStas.empty())) ||
        (isEmlsrDestination && !isProtected);

    if (!needMuRts)
    {
        // No protection needed
        if (txParams.m_protection && txParams.m_protection->method == WifiProtection::NONE)
        {
            return nullptr;
        }
        return std::make_unique<WifiNoProtection>();
    }

    WifiMuRtsCtsProtection* protection = nullptr;
    if (txParams.m_protection && txParams.m_protection->method == WifiProtection::MU_RTS_CTS)
    {
        protection = static_cast<WifiMuRtsCtsProtection*>(txParams.m_protection.get());
    }

    if (txParams.LastAddedIsFirstMpdu(receiver))
    {
        // we get here if this is the first MPDU for this receiver.
        NS_ABORT_MSG_IF(m_mac->GetTypeOfStation() != AP, "HE APs only can send DL MU PPDUs");
        auto modClass = txParams.m_txVector.GetModulationClass();
        auto txWidth = modClass == WIFI_MOD_CLASS_DSSS || modClass == WIFI_MOD_CLASS_HR_DSSS
                           ? MHz_u{20}
                           : txParams.m_txVector.GetChannelWidth();

        if (protection != nullptr)
        {
            // txParams.m_protection points to an existing WifiMuRtsCtsProtection object.
            // We have to return a copy of this object including the needed changes
            protection = new WifiMuRtsCtsProtection(*protection);

            // Add a User Info field for the new receiver
            // The UL HE-MCS, UL FEC Coding Type, UL DCM, SS Allocation and UL Target RSSI fields
            // in the User Info field are reserved (Sec. 9.3.1.22.5 of 802.11ax)
            AddUserInfoToMuRts(protection->muRts, txWidth, receiver);
        }
        else
        {
            // we have to create a new WifiMuRtsCtsProtection object
            protection = new WifiMuRtsCtsProtection;

            // initialize the MU-RTS Trigger Frame
            // The UL Length, GI And HE-LTF Type, MU-MIMO HE-LTF Mode, Number Of HE-LTF Symbols,
            // UL STBC, LDPC Extra Symbol Segment, AP TX Power, Pre-FEC Padding Factor,
            // PE Disambiguity, UL Spatial Reuse, Doppler and UL HE-SIG-A2 Reserved subfields in
            // the Common Info field are reserved. (Sec. 9.3.1.22.5 of 802.11ax)
            protection->muRts.SetType(TriggerFrameType::MU_RTS_TRIGGER);
            /* 35.2.2.1 MU-RTS Trigger frame transmission (IEEE P802.11be/D7.0):
             * If a non-AP EHT STA is addressed in an MU-RTS Trigger frame from an EHT AP and any of
             * the following conditions is met, the User Info field addressed to an EHT STA in the
             * MU-RTS Trigger frame shall be an EHT variant User Info field:
             * - The bandwidth of the EHT MU PPDU or non-HT duplicate PPDU carrying the MU-RTS
             * Trigger frame is 320 MHz.
             * - The EHT MU PPDU or non-HT duplicate PPDU carrying the MU-RTS Trigger frame is
             * punctured. Otherwise, the EHT AP may decide whether the User Info field in the MU-RTS
             * Trigger frame is an HE variant User Info field or an EHT variant User Info field.
             */
            const auto& inactiveSubchannels = txParams.m_txVector.GetInactiveSubchannels();
            const auto isPunctured =
                std::find(inactiveSubchannels.cbegin(), inactiveSubchannels.cend(), true) !=
                inactiveSubchannels.cend();
            const auto muRtsVariant = ((txWidth == MHz_u{320}) || isPunctured)
                                          ? TriggerFrameVariant::EHT
                                          : TriggerFrameVariant::HE;
            protection->muRts.SetVariant(muRtsVariant);
            protection->muRts.SetUlBandwidth(txWidth);

            // Add a User Info field for each of the receivers already in the TX params
            for (const auto& [address, info] : txParams.GetPsduInfoMap())
            {
                AddUserInfoToMuRts(protection->muRts, txWidth, address);
            }

            // compute the TXVECTOR to use to send the MU-RTS Trigger Frame
            protection->muRtsTxVector =
                GetWifiRemoteStationManager()->GetRtsTxVector(receiver, txWidth);
            // The transmitter of an MU-RTS Trigger frame shall not request a non-AP STA to send
            // a CTS frame response in a 20 MHz channel that is not occupied by the PPDU that
            // contains the MU-RTS Trigger frame. (Sec. 26.2.6.2 of 802.11ax)
            protection->muRtsTxVector.SetChannelWidth(txWidth);
            // OFDM is needed to transmit the PPDU over a bandwidth that is a multiple of 20 MHz
            const auto modulation = protection->muRtsTxVector.GetModulationClass();
            if (modulation == WIFI_MOD_CLASS_DSSS || modulation == WIFI_MOD_CLASS_HR_DSSS)
            {
                protection->muRtsTxVector.SetMode(ErpOfdmPhy::GetErpOfdmRate6Mbps());
            }
        }

        if (isEmlsrDestination && !isProtected)
        {
            // This MU-RTS is an ICF for some EMLSR client
            auto ehtFem =
                StaticCast<EhtFrameExchangeManager>(m_mac->GetFrameExchangeManager(m_linkId));
            ehtFem->SetIcfPaddingAndTxVector(protection->muRts, protection->muRtsTxVector);
        }

        return std::unique_ptr<WifiMuRtsCtsProtection>(protection);
    }

    // an MPDU addressed to the same receiver has been already added
    NS_ASSERT(protection != nullptr);

    // no change is needed
    return nullptr;
}

std::unique_ptr<WifiProtection>
WifiDefaultProtectionManager::TryUlMuTransmission(Ptr<const WifiMpdu> mpdu,
                                                  const WifiTxParameters& txParams)
{
    NS_LOG_FUNCTION(this << *mpdu << &txParams);
    NS_ASSERT(mpdu->GetHeader().IsTrigger());

    CtrlTriggerHeader trigger;
    mpdu->GetPacket()->PeekHeader(trigger);
    NS_ASSERT(trigger.GetNUserInfoFields() > 0);
    auto txWidth = trigger.GetUlBandwidth();

    auto protection = std::make_unique<WifiMuRtsCtsProtection>();
    // initialize the MU-RTS Trigger Frame
    // The UL Length, GI And HE-LTF Type, MU-MIMO HE-LTF Mode, Number Of HE-LTF Symbols,
    // UL STBC, LDPC Extra Symbol Segment, AP TX Power, Pre-FEC Padding Factor,
    // PE Disambiguity, UL Spatial Reuse, Doppler and UL HE-SIG-A2 Reserved subfields in
    // the Common Info field are reserved. (Sec. 9.3.1.22.5 of 802.11ax)
    protection->muRts.SetType(TriggerFrameType::MU_RTS_TRIGGER);
    protection->muRts.SetVariant(trigger.GetVariant());
    protection->muRts.SetUlBandwidth(txWidth);

    NS_ABORT_MSG_IF(m_mac->GetTypeOfStation() != AP, "HE APs only can send DL MU PPDUs");
    const auto& staList = StaticCast<ApWifiMac>(m_mac)->GetStaList(m_linkId);

    const auto& protectedStas = m_mac->GetFrameExchangeManager(m_linkId)->GetProtectedStas();
    bool allProtected = true;
    bool isUnprotectedEmlsrDst = false;

    for (const auto& userInfo : trigger)
    {
        // Add a User Info field to the MU-RTS for this solicited station
        // The UL HE-MCS, UL FEC Coding Type, UL DCM, SS Allocation and UL Target RSSI fields
        // in the User Info field are reserved (Sec. 9.3.1.22.5 of 802.11ax)
        auto staIt = staList.find(userInfo.GetAid12());
        NS_ASSERT(staIt != staList.cend());
        AddUserInfoToMuRts(protection->muRts, txWidth, staIt->second);
        const auto isProtected = protectedStas.contains(staIt->second);
        allProtected = allProtected && isProtected;

        isUnprotectedEmlsrDst =
            isUnprotectedEmlsrDst ||
            (!isProtected && GetWifiRemoteStationManager()->GetEmlsrEnabled(staIt->second));
    }

    bool needMuRts =
        (m_sendMuRts && !allProtected && (!m_singleRtsPerTxop || protectedStas.empty())) ||
        isUnprotectedEmlsrDst;

    // if we are sending a BSRP TF and SkipMuRtsBeforeBsrpTf is true, do not use MU-RTS (even in
    // case of unprotected EMLSR, because the BSRP TF is an ICF)
    needMuRts = needMuRts && (!m_skipMuRtsBeforeBsrp || !trigger.IsBsrp());

    if (!needMuRts)
    {
        // No protection needed
        return std::make_unique<WifiNoProtection>();
    }

    // compute the TXVECTOR to use to send the MU-RTS Trigger Frame
    protection->muRtsTxVector =
        GetWifiRemoteStationManager()->GetRtsTxVector(mpdu->GetHeader().GetAddr1(), txWidth);
    // The transmitter of an MU-RTS Trigger frame shall not request a non-AP STA to send
    // a CTS frame response in a 20 MHz channel that is not occupied by the PPDU that
    // contains the MU-RTS Trigger frame. (Sec. 26.2.6.2 of 802.11ax)
    protection->muRtsTxVector.SetChannelWidth(txWidth);
    // OFDM is needed to transmit the PPDU over a bandwidth that is a multiple of 20 MHz
    const auto modulation = protection->muRtsTxVector.GetModulationClass();
    if (modulation == WIFI_MOD_CLASS_DSSS || modulation == WIFI_MOD_CLASS_HR_DSSS)
    {
        protection->muRtsTxVector.SetMode(ErpOfdmPhy::GetErpOfdmRate6Mbps());
    }
    if (isUnprotectedEmlsrDst)
    {
        // This MU-RTS is an ICF for some EMLSR client
        auto ehtFem = StaticCast<EhtFrameExchangeManager>(m_mac->GetFrameExchangeManager(m_linkId));
        ehtFem->SetIcfPaddingAndTxVector(protection->muRts, protection->muRtsTxVector);
    }

    return protection;
}

} // namespace ns3
