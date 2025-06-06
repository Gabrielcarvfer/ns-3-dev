/*
 * Copyright (c) 2020 Orange Labs
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Rediet <getachew.redieteab@orange.com>
 *          Sébastien Deronne <sebastien.deronne@gmail.com> (for logic ported from wifi-phy and
 *                                                           spectrum-wifi-phy)
 *          Mathieu Lacage <mathieu.lacage@sophia.inria.fr> (for logic ported from wifi-phy)
 */

#include "phy-entity.h"

#include "frame-capture-model.h"
#include "interference-helper.h"
#include "preamble-detection-model.h"
#include "spectrum-wifi-phy.h"
#include "wifi-psdu.h"
#include "wifi-spectrum-signal-parameters.h"
#include "wifi-utils.h"

#include "ns3/assert.h"
#include "ns3/data-rate.h"
#include "ns3/log.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"

#include <algorithm>

#undef NS_LOG_APPEND_CONTEXT
#define NS_LOG_APPEND_CONTEXT WIFI_PHY_NS_LOG_APPEND_CONTEXT(m_wifiPhy)

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("PhyEntity");

std::ostream&
operator<<(std::ostream& os, const PhyEntity::PhyRxFailureAction& action)
{
    switch (action)
    {
    case PhyEntity::DROP:
        return (os << "DROP");
    case PhyEntity::ABORT:
        return (os << "ABORT");
    case PhyEntity::IGNORE:
        return (os << "IGNORE");
    default:
        NS_FATAL_ERROR("Unknown action");
        return (os << "unknown");
    }
}

std::ostream&
operator<<(std::ostream& os, const PhyEntity::PhyFieldRxStatus& status)
{
    if (status.isSuccess)
    {
        return os << "success";
    }
    else
    {
        return os << "failure (" << status.reason << "/" << status.actionIfFailure << ")";
    }
}

/*******************************************************
 *       Abstract base class for PHY entities
 *******************************************************/

uint64_t PhyEntity::m_globalPpduUid = 0;

PhyEntity::~PhyEntity()
{
    NS_LOG_FUNCTION(this);
    m_modeList.clear();
    CancelAllEvents();
}

void
PhyEntity::SetOwner(Ptr<WifiPhy> wifiPhy)
{
    NS_LOG_FUNCTION(this << wifiPhy);
    m_wifiPhy = wifiPhy;
    m_state = m_wifiPhy->m_state;
}

bool
PhyEntity::IsModeSupported(WifiMode mode) const
{
    for (const auto& m : m_modeList)
    {
        if (m == mode)
        {
            return true;
        }
    }
    return false;
}

uint8_t
PhyEntity::GetNumModes() const
{
    return m_modeList.size();
}

WifiMode
PhyEntity::GetMcs(uint8_t /* index */) const
{
    NS_ABORT_MSG(
        "This method should be used only for HtPhy and child classes. Use GetMode instead.");
    return WifiMode();
}

bool
PhyEntity::IsMcsSupported(uint8_t /* index */) const
{
    NS_ABORT_MSG("This method should be used only for HtPhy and child classes. Use IsModeSupported "
                 "instead.");
    return false;
}

bool
PhyEntity::HandlesMcsModes() const
{
    return false;
}

std::list<WifiMode>::const_iterator
PhyEntity::begin() const
{
    return m_modeList.begin();
}

std::list<WifiMode>::const_iterator
PhyEntity::end() const
{
    return m_modeList.end();
}

WifiMode
PhyEntity::GetSigMode(WifiPpduField field, const WifiTxVector& txVector) const
{
    NS_FATAL_ERROR("PPDU field is not a SIG field (no sense in retrieving the signaled mode) or is "
                   "unsupported: "
                   << field);
    return WifiMode(); // should be overloaded
}

WifiPpduField
PhyEntity::GetNextField(WifiPpduField currentField, WifiPreamble preamble) const
{
    const auto& ppduFormats = GetPpduFormats();
    const auto itPpdu = ppduFormats.find(preamble);
    if (itPpdu != ppduFormats.end())
    {
        const auto itField = std::find(itPpdu->second.begin(), itPpdu->second.end(), currentField);
        if (itField != itPpdu->second.end())
        {
            const auto itNextField = std::next(itField, 1);
            if (itNextField != itPpdu->second.end())
            {
                return *(itNextField);
            }
            NS_FATAL_ERROR("No field after " << currentField << " for " << preamble
                                             << " for the provided PPDU formats");
        }
        else
        {
            NS_FATAL_ERROR("Unsupported PPDU field " << currentField << " for " << preamble
                                                     << " for the provided PPDU formats");
        }
    }
    else
    {
        NS_FATAL_ERROR("Unsupported preamble " << preamble << " for the provided PPDU formats");
    }
    return WifiPpduField::WIFI_PPDU_FIELD_PREAMBLE; // Silence compiler warning
}

Time
PhyEntity::GetDuration(WifiPpduField field, const WifiTxVector& txVector) const
{
    if (field > WIFI_PPDU_FIELD_EHT_SIG)
    {
        NS_FATAL_ERROR("Unsupported PPDU field");
    }
    return Time(); // should be overloaded
}

Time
PhyEntity::CalculatePhyPreambleAndHeaderDuration(const WifiTxVector& txVector) const
{
    Time duration;
    for (uint8_t field = WIFI_PPDU_FIELD_PREAMBLE; field < WIFI_PPDU_FIELD_DATA; ++field)
    {
        duration += GetDuration(static_cast<WifiPpduField>(field), txVector);
    }
    return duration;
}

WifiConstPsduMap
PhyEntity::GetWifiConstPsduMap(Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector) const
{
    return WifiConstPsduMap({{SU_STA_ID, psdu}});
}

Ptr<const WifiPsdu>
PhyEntity::GetAddressedPsduInPpdu(Ptr<const WifiPpdu> ppdu) const
{
    return ppdu->GetPsdu();
}

PhyEntity::PhyHeaderSections
PhyEntity::GetPhyHeaderSections(const WifiTxVector& txVector, Time ppduStart) const
{
    PhyHeaderSections map;
    WifiPpduField field = WIFI_PPDU_FIELD_PREAMBLE; // preamble always present
    Time start = ppduStart;

    while (field != WIFI_PPDU_FIELD_DATA)
    {
        Time duration = GetDuration(field, txVector);
        map[field] = {{start, start + duration}, GetSigMode(field, txVector)};
        // Move to next field
        start += duration;
        field = GetNextField(field, txVector.GetPreambleType());
    }
    return map;
}

Ptr<WifiPpdu>
PhyEntity::BuildPpdu(const WifiConstPsduMap& psdus, const WifiTxVector& txVector, Time ppduDuration)
{
    NS_LOG_FUNCTION(this << psdus << txVector << ppduDuration);
    NS_FATAL_ERROR("This method is unsupported for the base PhyEntity class. Use the overloaded "
                   "version in the amendment-specific subclasses instead!");
    return Create<WifiPpdu>(psdus.begin()->second,
                            txVector,
                            m_wifiPhy->GetOperatingChannel()); // should be overloaded
}

Time
PhyEntity::GetDurationUpToField(WifiPpduField field, const WifiTxVector& txVector) const
{
    if (field ==
        WIFI_PPDU_FIELD_DATA) // this field is not in the map returned by GetPhyHeaderSections
    {
        return CalculatePhyPreambleAndHeaderDuration(txVector);
    }
    const auto& sections = GetPhyHeaderSections(txVector, Time());
    auto it = sections.find(field);
    NS_ASSERT(it != sections.end());
    const auto& startStopTimes = it->second.first;
    return startStopTimes
        .first; // return the start time of field relatively to the beginning of the PPDU
}

PhyEntity::SnrPer
PhyEntity::GetPhyHeaderSnrPer(WifiPpduField field, Ptr<Event> event) const
{
    const auto measurementChannelWidth = GetMeasurementChannelWidth(event->GetPpdu());
    return m_wifiPhy->m_interference->CalculatePhyHeaderSnrPer(
        event,
        measurementChannelWidth,
        GetPrimaryBand(measurementChannelWidth),
        field);
}

void
PhyEntity::StartReceiveField(WifiPpduField field, Ptr<Event> event)
{
    NS_LOG_FUNCTION(this << field << *event);
    NS_ASSERT(m_wifiPhy); // no sense if no owner WifiPhy instance
    NS_ASSERT(m_wifiPhy->m_endPhyRxEvent.IsExpired());
    NS_ABORT_MSG_IF(field == WIFI_PPDU_FIELD_PREAMBLE,
                    "Use the StartReceivePreamble method for preamble reception");
    // Handle special cases of data reception
    if (field == WIFI_PPDU_FIELD_DATA)
    {
        StartReceivePayload(event);
        return;
    }

    bool supported = DoStartReceiveField(field, event);
    NS_ABORT_MSG_IF(!supported,
                    "Unknown field "
                        << field << " for this PHY entity"); // TODO see what to do if not supported
    Time duration = GetDuration(field, event->GetPpdu()->GetTxVector());
    m_wifiPhy->m_endPhyRxEvent =
        Simulator::Schedule(duration, &PhyEntity::EndReceiveField, this, field, event);
    m_wifiPhy->NotifyCcaBusy(
        event->GetPpdu(),
        duration); // keep in CCA busy state up to reception of Data (will then switch to RX)
}

void
PhyEntity::EndReceiveField(WifiPpduField field, Ptr<Event> event)
{
    NS_LOG_FUNCTION(this << field << *event);
    NS_ASSERT(m_wifiPhy); // no sense if no owner WifiPhy instance
    NS_ASSERT(m_wifiPhy->m_endPhyRxEvent.IsExpired());
    PhyFieldRxStatus status = DoEndReceiveField(field, event);
    const auto& txVector = event->GetPpdu()->GetTxVector();
    if (status.isSuccess) // move to next field if reception succeeded
    {
        StartReceiveField(GetNextField(field, txVector.GetPreambleType()), event);
    }
    else
    {
        Ptr<const WifiPpdu> ppdu = event->GetPpdu();
        switch (status.actionIfFailure)
        {
        case ABORT:
            // Abort reception, but consider medium as busy
            AbortCurrentReception(status.reason);
            if (event->GetEndTime() > (Simulator::Now() + m_state->GetDelayUntilIdle()))
            {
                m_wifiPhy->SwitchMaybeToCcaBusy(ppdu);
            }
            break;
        case DROP:
            // Notify drop, keep in CCA busy, and perform same processing as IGNORE case
            if (status.reason == FILTERED)
            {
                // PHY-RXSTART is immediately followed by PHY-RXEND (Filtered)
                m_wifiPhy->m_phyRxPayloadBeginTrace(
                    txVector,
                    NanoSeconds(0)); // this callback (equivalent to PHY-RXSTART primitive) is also
                                     // triggered for filtered PPDUs
            }
            m_wifiPhy->NotifyRxPpduDrop(ppdu, status.reason);
            m_wifiPhy->NotifyCcaBusy(ppdu, GetRemainingDurationAfterField(ppdu, field));
        // no break
        case IGNORE:
            // Keep in Rx state and reset at end
            m_endRxPayloadEvents.push_back(
                Simulator::Schedule(GetRemainingDurationAfterField(ppdu, field),
                                    &PhyEntity::ResetReceive,
                                    this,
                                    event));
            break;
        default:
            NS_FATAL_ERROR("Unknown action in case of failure");
        }
    }
}

Time
PhyEntity::GetRemainingDurationAfterField(Ptr<const WifiPpdu> ppdu, WifiPpduField field) const
{
    const auto& txVector = ppdu->GetTxVector();
    return ppdu->GetTxDuration() -
           (GetDurationUpToField(field, txVector) + GetDuration(field, txVector));
}

bool
PhyEntity::DoStartReceiveField(WifiPpduField field, Ptr<Event> event)
{
    NS_LOG_FUNCTION(this << field << *event);
    NS_ASSERT(field != WIFI_PPDU_FIELD_PREAMBLE &&
              field != WIFI_PPDU_FIELD_DATA); // handled apart for the time being
    const auto& ppduFormats = GetPpduFormats();
    auto itFormat = ppduFormats.find(event->GetPpdu()->GetPreamble());
    if (itFormat != ppduFormats.end())
    {
        auto itField = std::find(itFormat->second.begin(), itFormat->second.end(), field);
        if (itField != itFormat->second.end())
        {
            return true; // supported field so we can start receiving
        }
    }
    return false; // unsupported otherwise
}

PhyEntity::PhyFieldRxStatus
PhyEntity::DoEndReceiveField(WifiPpduField field, Ptr<Event> event)
{
    NS_LOG_FUNCTION(this << field << *event);
    NS_ASSERT(field != WIFI_PPDU_FIELD_DATA); // handled apart for the time being
    if (field == WIFI_PPDU_FIELD_PREAMBLE)
    {
        return DoEndReceivePreamble(event);
    }
    return PhyFieldRxStatus(false); // failed reception by default
}

void
PhyEntity::StartReceivePreamble(Ptr<const WifiPpdu> ppdu,
                                RxPowerWattPerChannelBand& rxPowersW,
                                Time rxDuration)
{
    // The total RX power corresponds to the maximum over all the bands
    auto it =
        std::max_element(rxPowersW.begin(), rxPowersW.end(), [](const auto& p1, const auto& p2) {
            return p1.second < p2.second;
        });
    NS_LOG_FUNCTION(this << ppdu << it->second);

    auto event = DoGetEvent(ppdu, rxPowersW);
    if (!event)
    {
        // PPDU should be simply considered as interference (once it has been accounted for in
        // InterferenceHelper)
        return;
    }

    Time endRx = Simulator::Now() + rxDuration;
    if (ppdu->IsTruncatedTx())
    {
        NS_LOG_DEBUG("Packet reception stopped because transmitter has been switched off");
        if (endRx > (Simulator::Now() + m_state->GetDelayUntilIdle()))
        {
            m_wifiPhy->SwitchMaybeToCcaBusy(ppdu);
        }
        DropPreambleEvent(ppdu, WifiPhyRxfailureReason::TRUNCATED_TX, endRx);
        return;
    }

    switch (m_state->GetState())
    {
    case WifiPhyState::SWITCHING:
        NS_LOG_DEBUG("Drop packet because of channel switching");
        /*
         * Packets received on the upcoming channel are added to the event list
         * during the switching state. This way the medium can be correctly sensed
         * when the device listens to the channel for the first time after the
         * switching e.g. after channel switching, the channel may be sensed as
         * busy due to other devices' transmissions started before the end of
         * the switching.
         */
        DropPreambleEvent(ppdu, CHANNEL_SWITCHING, endRx);
        break;
    case WifiPhyState::RX:
        if (m_wifiPhy->m_frameCaptureModel &&
            m_wifiPhy->m_frameCaptureModel->IsInCaptureWindow(
                m_wifiPhy->m_timeLastPreambleDetected) &&
            m_wifiPhy->m_frameCaptureModel->CaptureNewFrame(m_wifiPhy->m_currentEvent, event))
        {
            AbortCurrentReception(FRAME_CAPTURE_PACKET_SWITCH);
            NS_LOG_DEBUG("Switch to new packet");
            StartPreambleDetectionPeriod(event);
        }
        else
        {
            NS_LOG_DEBUG("Drop packet because already in Rx");
            DropPreambleEvent(ppdu, RXING, endRx);
            if (!m_wifiPhy->m_currentEvent)
            {
                /*
                 * We are here because the non-legacy PHY header has not been successfully received.
                 * The PHY is kept in RX state for the duration of the PPDU, but EndReceive function
                 * is not called when the reception of the PPDU is finished, which is responsible to
                 * clear m_currentPreambleEvents. As a result, m_currentPreambleEvents should be
                 * cleared here.
                 */
                m_wifiPhy->m_currentPreambleEvents.clear();
            }
        }
        break;
    case WifiPhyState::TX:
        NS_LOG_DEBUG("Drop packet because already in Tx");
        DropPreambleEvent(ppdu, TXING, endRx);
        break;
    case WifiPhyState::CCA_BUSY:
        if (m_wifiPhy->m_currentEvent)
        {
            if (m_wifiPhy->m_frameCaptureModel &&
                m_wifiPhy->m_frameCaptureModel->IsInCaptureWindow(
                    m_wifiPhy->m_timeLastPreambleDetected) &&
                m_wifiPhy->m_frameCaptureModel->CaptureNewFrame(m_wifiPhy->m_currentEvent, event))
            {
                AbortCurrentReception(FRAME_CAPTURE_PACKET_SWITCH);
                NS_LOG_DEBUG("Switch to new packet");
                StartPreambleDetectionPeriod(event);
            }
            else
            {
                NS_LOG_DEBUG("Drop packet because already decoding preamble");
                DropPreambleEvent(ppdu, BUSY_DECODING_PREAMBLE, endRx);
            }
        }
        else
        {
            StartPreambleDetectionPeriod(event);
        }
        break;
    case WifiPhyState::IDLE:
        NS_ASSERT(!m_wifiPhy->m_currentEvent);
        StartPreambleDetectionPeriod(event);
        break;
    case WifiPhyState::SLEEP:
        NS_LOG_DEBUG("Drop packet because in sleep mode");
        DropPreambleEvent(ppdu, SLEEPING, endRx);
        break;
    case WifiPhyState::OFF:
        NS_LOG_DEBUG("Drop packet because in switched off");
        DropPreambleEvent(ppdu, WifiPhyRxfailureReason::POWERED_OFF, endRx);
        break;
    default:
        NS_FATAL_ERROR("Invalid WifiPhy state.");
        break;
    }
}

void
PhyEntity::DropPreambleEvent(Ptr<const WifiPpdu> ppdu, WifiPhyRxfailureReason reason, Time endRx)
{
    NS_LOG_FUNCTION(this << ppdu << reason << endRx);
    m_wifiPhy->NotifyRxPpduDrop(ppdu, reason);
    auto it = m_wifiPhy->m_currentPreambleEvents.find({ppdu->GetUid(), ppdu->GetPreamble()});
    if (it != m_wifiPhy->m_currentPreambleEvents.end())
    {
        m_wifiPhy->m_currentPreambleEvents.erase(it);
    }
    if (!m_wifiPhy->IsStateSleep() && !m_wifiPhy->IsStateOff() &&
        (endRx > (Simulator::Now() + m_state->GetDelayUntilIdle())))
    {
        // that PPDU will be noise _after_ the end of the current event.
        m_wifiPhy->SwitchMaybeToCcaBusy(ppdu);
    }
}

void
PhyEntity::ErasePreambleEvent(Ptr<const WifiPpdu> ppdu, Time rxDuration)
{
    NS_LOG_FUNCTION(this << ppdu << rxDuration);
    auto it = m_wifiPhy->m_currentPreambleEvents.find({ppdu->GetUid(), ppdu->GetPreamble()});
    if (it != m_wifiPhy->m_currentPreambleEvents.end())
    {
        m_wifiPhy->m_currentPreambleEvents.erase(it);
    }
    if (m_wifiPhy->m_currentPreambleEvents.empty())
    {
        m_wifiPhy->Reset();
    }

    if (rxDuration > m_state->GetDelayUntilIdle())
    {
        // this PPDU will be noise _after_ the completion of the current event
        m_wifiPhy->SwitchMaybeToCcaBusy(ppdu);
    }
}

uint16_t
PhyEntity::GetStaId(const Ptr<const WifiPpdu> /* ppdu */) const
{
    return SU_STA_ID;
}

void
PhyEntity::StartReceivePayload(Ptr<Event> event)
{
    NS_LOG_FUNCTION(this << *event);
    NS_ASSERT(m_wifiPhy->m_endPhyRxEvent.IsExpired());

    Time payloadDuration = DoStartReceivePayload(event);
    m_state->SwitchToRx(payloadDuration);
}

Time
PhyEntity::DoStartReceivePayload(Ptr<Event> event)
{
    NS_LOG_FUNCTION(this << *event);
    Ptr<const WifiPpdu> ppdu = event->GetPpdu();
    NS_LOG_DEBUG("Receiving PSDU");
    uint16_t staId = GetStaId(ppdu);
    m_signalNoiseMap.insert({{ppdu->GetUid(), staId}, SignalNoiseDbm()});
    m_statusPerMpduMap.insert({{ppdu->GetUid(), staId}, std::vector<bool>()});
    ScheduleEndOfMpdus(event);
    const auto& txVector = event->GetPpdu()->GetTxVector();
    Time payloadDuration = ppdu->GetTxDuration() - CalculatePhyPreambleAndHeaderDuration(txVector);
    m_wifiPhy->m_phyRxPayloadBeginTrace(
        txVector,
        payloadDuration); // this callback (equivalent to PHY-RXSTART primitive) is triggered only
                          // if headers have been correctly decoded and that the mode within is
                          // supported
    m_endRxPayloadEvents.push_back(
        Simulator::Schedule(payloadDuration, &PhyEntity::EndReceivePayload, this, event));
    return payloadDuration;
}

void
PhyEntity::ScheduleEndOfMpdus(Ptr<Event> event)
{
    NS_LOG_FUNCTION(this << *event);
    Ptr<const WifiPpdu> ppdu = event->GetPpdu();
    Ptr<const WifiPsdu> psdu = GetAddressedPsduInPpdu(ppdu);
    const auto& txVector = event->GetPpdu()->GetTxVector();
    uint16_t staId = GetStaId(ppdu);
    Time endOfMpduDuration;
    Time relativeStart;
    Time psduDuration = ppdu->GetTxDuration() - CalculatePhyPreambleAndHeaderDuration(txVector);
    Time remainingAmpduDuration = psduDuration;
    size_t nMpdus = psdu->GetNMpdus();
    MpduType mpduType =
        (nMpdus > 1) ? FIRST_MPDU_IN_AGGREGATE : (psdu->IsSingle() ? SINGLE_MPDU : NORMAL_MPDU);
    uint32_t totalAmpduSize = 0;
    double totalAmpduNumSymbols = 0.0;
    auto mpdu = psdu->begin();
    for (size_t i = 0; i < nMpdus && mpdu != psdu->end(); ++mpdu)
    {
        if (m_wifiPhy->m_notifyRxMacHeaderEnd)
        {
            // calculate MAC header size (including A-MPDU subframe header, if present)
            auto macHdrSize =
                (*mpdu)->GetHeader().GetSerializedSize() + (mpduType == NORMAL_MPDU ? 0 : 4);
            // calculate the (approximate) duration of the MAC header TX
            auto macHdrDuration = DataRate(txVector.GetMode(staId).GetDataRate(txVector, staId))
                                      .CalculateBytesTxTime(macHdrSize);
            const auto widthBand = GetChannelWidthAndBand(txVector, staId);
            const auto snrPer = m_wifiPhy->m_interference->CalculatePayloadSnrPer(
                event,
                widthBand.first,
                widthBand.second,
                staId,
                {relativeStart, relativeStart + macHdrDuration});
            if (GetRandomValue() > snrPer.per)
            {
                // interference level should permit to correctly decode the MAC header
                m_endOfMacHdrEvents[staId].push_back(
                    Simulator::Schedule(endOfMpduDuration + macHdrDuration, [=, this]() {
                        m_wifiPhy->m_phyRxMacHeaderEndTrace((*mpdu)->GetHeader(),
                                                            txVector,
                                                            remainingAmpduDuration -
                                                                macHdrDuration);
                    }));
            }
        }

        uint32_t size = (mpduType == NORMAL_MPDU) ? psdu->GetSize() : psdu->GetAmpduSubframeSize(i);
        Time mpduDuration = WifiPhy::GetPayloadDuration(size,
                                                        txVector,
                                                        m_wifiPhy->GetPhyBand(),
                                                        mpduType,
                                                        true,
                                                        totalAmpduSize,
                                                        totalAmpduNumSymbols,
                                                        staId);

        remainingAmpduDuration -= mpduDuration;
        if (i == (nMpdus - 1) && !remainingAmpduDuration.IsZero()) // no more MPDUs coming
        {
            if (remainingAmpduDuration < txVector.GetGuardInterval()) // enables to ignore padding
            {
                mpduDuration += remainingAmpduDuration; // apply a correction just in case rounding
                                                        // had induced slight shift
            }
        }

        endOfMpduDuration += mpduDuration;
        NS_LOG_INFO("Schedule end of MPDU #"
                    << i << " in " << endOfMpduDuration.As(Time::NS) << " (relativeStart="
                    << relativeStart.As(Time::NS) << ", mpduDuration=" << mpduDuration.As(Time::NS)
                    << ", remainingAmdpuDuration=" << remainingAmpduDuration.As(Time::NS) << ")");
        m_endOfMpduEvents.push_back(Simulator::Schedule(endOfMpduDuration,
                                                        &PhyEntity::EndOfMpdu,
                                                        this,
                                                        event,
                                                        *mpdu,
                                                        i,
                                                        relativeStart,
                                                        mpduDuration));

        // Prepare next iteration
        ++i;
        relativeStart += mpduDuration;
        mpduType = (i == (nMpdus - 1)) ? LAST_MPDU_IN_AGGREGATE : MIDDLE_MPDU_IN_AGGREGATE;
    }
}

void
PhyEntity::EndOfMpdu(Ptr<Event> event,
                     Ptr<WifiMpdu> mpdu,
                     size_t mpduIndex,
                     Time relativeStart,
                     Time mpduDuration)
{
    NS_LOG_FUNCTION(this << *event << mpduIndex << relativeStart << mpduDuration);
    const auto ppdu = event->GetPpdu();
    const auto& txVector = ppdu->GetTxVector();
    uint16_t staId = GetStaId(ppdu);

    std::pair<bool, SignalNoiseDbm> rxInfo =
        GetReceptionStatus(mpdu, event, staId, relativeStart, mpduDuration);
    NS_LOG_DEBUG("Extracted MPDU #" << mpduIndex << ": duration: " << mpduDuration.As(Time::NS)
                                    << ", correct reception: " << rxInfo.first << ", Signal/Noise: "
                                    << rxInfo.second.signal << "/" << rxInfo.second.noise << "dBm");

    auto signalNoiseIt = m_signalNoiseMap.find({ppdu->GetUid(), staId});
    NS_ASSERT(signalNoiseIt != m_signalNoiseMap.end());
    signalNoiseIt->second = rxInfo.second;

    RxSignalInfo rxSignalInfo;
    rxSignalInfo.snr = DbToRatio(dB_u{rxInfo.second.signal - rxInfo.second.noise});
    rxSignalInfo.rssi = rxInfo.second.signal;

    auto statusPerMpduIt = m_statusPerMpduMap.find({ppdu->GetUid(), staId});
    NS_ASSERT(statusPerMpduIt != m_statusPerMpduMap.end());
    statusPerMpduIt->second.push_back(rxInfo.first);

    if (rxInfo.first && GetAddressedPsduInPpdu(ppdu)->GetNMpdus() > 1)
    {
        // only done for correct MPDU that is part of an A-MPDU
        m_state->NotifyRxMpdu(Create<const WifiPsdu>(mpdu, false), rxSignalInfo, txVector);
    }
}

void
PhyEntity::EndReceivePayload(Ptr<Event> event)
{
    const auto ppdu = event->GetPpdu();
    const auto& txVector = ppdu->GetTxVector();
    NS_LOG_FUNCTION(
        this << *event << ppdu->GetTxDuration() - CalculatePhyPreambleAndHeaderDuration(txVector));
    NS_ASSERT(event->GetEndTime() == Simulator::Now());
    const auto staId = GetStaId(ppdu);
    const auto channelWidthAndBand = GetChannelWidthAndBand(txVector, staId);
    const auto snr = m_wifiPhy->m_interference->CalculateSnr(event,
                                                             channelWidthAndBand.first,
                                                             txVector.GetNss(staId),
                                                             channelWidthAndBand.second);

    Ptr<const WifiPsdu> psdu = GetAddressedPsduInPpdu(ppdu);
    m_wifiPhy->NotifyRxEnd(psdu);

    auto signalNoiseIt = m_signalNoiseMap.find({ppdu->GetUid(), staId});
    NS_ASSERT(signalNoiseIt != m_signalNoiseMap.end());
    auto statusPerMpduIt = m_statusPerMpduMap.find({ppdu->GetUid(), staId});
    NS_ASSERT(statusPerMpduIt != m_statusPerMpduMap.end());
    // store per-MPDU status, which is cleared by the call to DoEndReceivePayload below
    auto statusPerMpdu = statusPerMpduIt->second;

    RxSignalInfo rxSignalInfo;
    bool success;

    if (std::count(statusPerMpdu.cbegin(), statusPerMpdu.cend(), true))
    {
        // At least one MPDU has been successfully received
        m_wifiPhy->NotifyMonitorSniffRx(psdu,
                                        m_wifiPhy->GetFrequency(),
                                        txVector,
                                        signalNoiseIt->second,
                                        statusPerMpdu,
                                        staId);
        rxSignalInfo.snr = snr;
        rxSignalInfo.rssi = signalNoiseIt->second.signal; // same information for all MPDUs
        RxPayloadSucceeded(psdu, rxSignalInfo, txVector, staId, statusPerMpdu);
        m_wifiPhy->m_previouslyRxPpduUid =
            ppdu->GetUid(); // store UID only if reception is successful (because otherwise trigger
                            // won't be read by MAC layer)
        success = true;
    }
    else
    {
        RxPayloadFailed(psdu, snr, txVector);
        success = false;
    }

    m_state->NotifyRxPpduOutcome(ppdu, rxSignalInfo, txVector, staId, statusPerMpduIt->second);
    DoEndReceivePayload(ppdu);
    m_wifiPhy->SwitchMaybeToCcaBusy(ppdu);

    // notify the MAC through the PHY state helper as the last action. Indeed, the notification
    // of the RX end may lead the MAC to request a PHY state change (e.g., channel switch, sleep).
    // Hence, all actions the PHY has to perform when RX ends should be completed before
    // notifying the MAC.
    success ? m_state->NotifyRxPsduSucceeded(psdu, rxSignalInfo, txVector, staId, statusPerMpdu)
            : m_state->NotifyRxPsduFailed(psdu, snr);
}

void
PhyEntity::RxPayloadSucceeded(Ptr<const WifiPsdu> psdu,
                              RxSignalInfo rxSignalInfo,
                              const WifiTxVector& txVector,
                              uint16_t staId,
                              const std::vector<bool>& statusPerMpdu)
{
    NS_LOG_FUNCTION(this << *psdu << txVector);
    m_state->SwitchFromRxEndOk();
}

void
PhyEntity::RxPayloadFailed(Ptr<const WifiPsdu> psdu, double snr, const WifiTxVector& txVector)
{
    NS_LOG_FUNCTION(this << *psdu << txVector << snr);
    m_state->SwitchFromRxEndError(txVector);
}

void
PhyEntity::DoEndReceivePayload(Ptr<const WifiPpdu> ppdu)
{
    NS_LOG_FUNCTION(this << ppdu);
    NS_ASSERT(m_wifiPhy->GetLastRxEndTime() == Simulator::Now());
    NotifyInterferenceRxEndAndClear(false); // don't reset WifiPhy

    m_wifiPhy->m_currentEvent = nullptr;
    m_wifiPhy->m_currentPreambleEvents.clear();
    m_endRxPayloadEvents.clear();
}

std::pair<bool, SignalNoiseDbm>
PhyEntity::GetReceptionStatus(Ptr<WifiMpdu> mpdu,
                              Ptr<Event> event,
                              uint16_t staId,
                              Time relativeMpduStart,
                              Time mpduDuration)
{
    NS_LOG_FUNCTION(this << *mpdu << *event << staId << relativeMpduStart << mpduDuration);
    const auto channelWidthAndBand = GetChannelWidthAndBand(event->GetPpdu()->GetTxVector(), staId);
    SnrPer snrPer = m_wifiPhy->m_interference->CalculatePayloadSnrPer(
        event,
        channelWidthAndBand.first,
        channelWidthAndBand.second,
        staId,
        {relativeMpduStart, relativeMpduStart + mpduDuration});

    WifiMode mode = event->GetPpdu()->GetTxVector().GetMode(staId);
    NS_LOG_DEBUG("rate=" << (mode.GetDataRate(event->GetPpdu()->GetTxVector(), staId))
                         << ", SNR(dB)=" << RatioToDb(snrPer.snr) << ", PER=" << snrPer.per
                         << ", size=" << mpdu->GetSize()
                         << ", relativeStart = " << relativeMpduStart.As(Time::NS)
                         << ", duration = " << mpduDuration.As(Time::NS));

    // There are two error checks: PER and receive error model check.
    // PER check models is typical for Wi-Fi and is based on signal modulation;
    // Receive error model is optional, if we have an error model and
    // it indicates that the packet is corrupt, drop the packet.
    SignalNoiseDbm signalNoise;
    signalNoise.signal = WToDbm(event->GetRxPower(channelWidthAndBand.second));
    signalNoise.noise = WToDbm(event->GetRxPower(channelWidthAndBand.second) / snrPer.snr);
    if (GetRandomValue() > snrPer.per &&
        !(m_wifiPhy->m_postReceptionErrorModel &&
          m_wifiPhy->m_postReceptionErrorModel->IsCorrupt(mpdu->GetPacket()->Copy())))
    {
        NS_LOG_DEBUG("Reception succeeded: " << *mpdu);
        return {true, signalNoise};
    }
    else
    {
        NS_LOG_DEBUG("Reception failed: " << *mpdu);
        return {false, signalNoise};
    }
}

std::optional<Time>
PhyEntity::GetTimeToPreambleDetectionEnd() const
{
    if (m_endPreambleDetectionEvents.empty())
    {
        return {};
    }

    std::optional<Time> delayUntilPreambleDetectionEnd;
    for (const auto& endPreambleDetectionEvent : m_endPreambleDetectionEvents)
    {
        if (endPreambleDetectionEvent.IsPending())
        {
            delayUntilPreambleDetectionEnd =
                std::max(delayUntilPreambleDetectionEnd.value_or(Time{0}),
                         Simulator::GetDelayLeft(endPreambleDetectionEvent));
        }
    }
    return delayUntilPreambleDetectionEnd;
}

std::optional<Time>
PhyEntity::GetTimeToMacHdrEnd(uint16_t staId) const
{
    const auto it = m_endOfMacHdrEvents.find(staId);

    if (it == m_endOfMacHdrEvents.cend())
    {
        return std::nullopt;
    }

    for (const auto& endOfMacHdrEvent : it->second)
    {
        if (endOfMacHdrEvent.IsPending())
        {
            return Simulator::GetDelayLeft(endOfMacHdrEvent);
        }
    }

    return std::nullopt;
}

std::pair<MHz_u, WifiSpectrumBandInfo>
PhyEntity::GetChannelWidthAndBand(const WifiTxVector& txVector, uint16_t /* staId */) const
{
    const auto channelWidth = GetRxChannelWidth(txVector);
    return {channelWidth, GetPrimaryBand(channelWidth)};
}

const std::map<std::pair<uint64_t, WifiPreamble>, Ptr<Event>>&
PhyEntity::GetCurrentPreambleEvents() const
{
    return m_wifiPhy->m_currentPreambleEvents;
}

void
PhyEntity::AddPreambleEvent(Ptr<Event> event)
{
    NS_LOG_FUNCTION(this << *event);
    Ptr<const WifiPpdu> ppdu = event->GetPpdu();
    m_wifiPhy->m_currentPreambleEvents.insert({{ppdu->GetUid(), ppdu->GetPreamble()}, event});
}

Ptr<Event>
PhyEntity::DoGetEvent(Ptr<const WifiPpdu> ppdu, RxPowerWattPerChannelBand& rxPowersW)
{
    // We store all incoming preamble events, and a decision is made at the end of the preamble
    // detection window.
    const auto& currentPreambleEvents = GetCurrentPreambleEvents();
    const auto it = currentPreambleEvents.find({ppdu->GetUid(), ppdu->GetPreamble()});
    if (it != currentPreambleEvents.cend())
    {
        // received another signal with the same content
        NS_LOG_DEBUG("Received another PPDU for UID " << ppdu->GetUid());
        const auto foundEvent = it->second;
        HandleRxPpduWithSameContent(foundEvent, ppdu, rxPowersW);
        return nullptr;
    }

    auto event = CreateInterferenceEvent(ppdu, ppdu->GetTxDuration(), rxPowersW);
    AddPreambleEvent(event);
    return event;
}

Ptr<Event>
PhyEntity::CreateInterferenceEvent(Ptr<const WifiPpdu> ppdu,
                                   Time duration,
                                   RxPowerWattPerChannelBand& rxPower,
                                   bool isStartHePortionRxing /* = false */)
{
    return m_wifiPhy->m_interference->Add(ppdu,
                                          duration,
                                          rxPower,
                                          m_wifiPhy->GetCurrentFrequencyRange(),
                                          isStartHePortionRxing);
}

void
PhyEntity::HandleRxPpduWithSameContent(Ptr<Event> event,
                                       Ptr<const WifiPpdu> ppdu,
                                       RxPowerWattPerChannelBand& rxPower)
{
    if (const auto maxDelay =
            m_wifiPhy->GetPhyEntityForPpdu(ppdu)->GetMaxDelayPpduSameUid(ppdu->GetTxVector());
        Simulator::Now() - event->GetStartTime() > maxDelay)
    {
        // This PPDU arrived too late to be decoded properly. The PPDU is dropped and added as
        // interference
        event = CreateInterferenceEvent(ppdu, ppdu->GetTxDuration(), rxPower);
        NS_LOG_DEBUG("Drop PPDU that arrived too late");
        m_wifiPhy->NotifyRxPpduDrop(ppdu, PPDU_TOO_LATE);
        return;
    }

    // Update received power and TXVECTOR of the event associated to that transmission upon
    // reception of a signal adding up constructively (in case of a UL MU PPDU or non-HT duplicate
    // PPDU)
    m_wifiPhy->m_interference->UpdateEvent(event, rxPower);
    const auto& txVector = ppdu->GetTxVector();
    const auto& eventTxVector = event->GetPpdu()->GetTxVector();
    auto updatedTxVector{eventTxVector};
    updatedTxVector.SetChannelWidth(
        std::max(eventTxVector.GetChannelWidth(), txVector.GetChannelWidth()));
    if (updatedTxVector.GetChannelWidth() != eventTxVector.GetChannelWidth())
    {
        event->UpdatePpdu(ppdu);
    }
}

void
PhyEntity::NotifyInterferenceRxEndAndClear(bool reset)
{
    m_wifiPhy->m_interference->NotifyRxEnd(Simulator::Now(), m_wifiPhy->GetCurrentFrequencyRange());
    m_signalNoiseMap.clear();
    m_statusPerMpduMap.clear();
    for (const auto& endOfMpduEvent : m_endOfMpduEvents)
    {
        NS_ASSERT(endOfMpduEvent.IsExpired());
    }
    m_endOfMpduEvents.clear();
    for (const auto& [staId, endOfMacHdrEvents] : m_endOfMacHdrEvents)
    {
        for (const auto& endOfMacHdrEvent : endOfMacHdrEvents)
        {
            NS_ASSERT(endOfMacHdrEvent.IsExpired());
        }
    }
    m_endOfMacHdrEvents.clear();
    if (reset)
    {
        m_wifiPhy->Reset();
    }
}

PhyEntity::PhyFieldRxStatus
PhyEntity::DoEndReceivePreamble(Ptr<Event> event)
{
    NS_LOG_FUNCTION(this << *event);
    NS_ASSERT(m_wifiPhy->m_currentPreambleEvents.size() ==
              1);                  // Synched on one after detection period
    return PhyFieldRxStatus(true); // always consider that preamble has been correctly received if
                                   // preamble detection was OK
}

void
PhyEntity::StartPreambleDetectionPeriod(Ptr<Event> event)
{
    NS_LOG_FUNCTION(this << *event);
    const auto rxPower = GetRxPowerForPpdu(event);
    NS_LOG_DEBUG("Sync to signal (power=" << (rxPower > Watt_u{0.0}
                                                  ? std::to_string(WToDbm(rxPower)) + "dBm)"
                                                  : std::to_string(rxPower) + "W)"));
    m_wifiPhy->m_interference->NotifyRxStart(
        m_wifiPhy->GetCurrentFrequencyRange()); // We need to notify it now so that it starts
                                                // recording events
    m_endPreambleDetectionEvents.push_back(
        Simulator::Schedule(WifiPhy::GetPreambleDetectionDuration(),
                            &PhyEntity::EndPreambleDetectionPeriod,
                            this,
                            event));
}

void
PhyEntity::EndPreambleDetectionPeriod(Ptr<Event> event)
{
    NS_LOG_FUNCTION(this << *event);
    NS_ASSERT(!m_wifiPhy->IsStateRx());
    NS_ASSERT(m_wifiPhy->m_endPhyRxEvent.IsExpired()); // since end of preamble reception is
                                                       // scheduled by this method upon success

    // calculate PER on the measurement channel for PHY headers
    const auto measurementChannelWidth = GetMeasurementChannelWidth(event->GetPpdu());
    auto measurementBand = GetPrimaryBand(measurementChannelWidth);
    std::optional<Watt_u>
        maxRxPower; // in case current event may not be sent on measurement channel
    Ptr<Event> maxEvent;
    NS_ASSERT(!m_wifiPhy->m_currentPreambleEvents.empty());
    for (auto preambleEvent : m_wifiPhy->m_currentPreambleEvents)
    {
        const auto rxPower = preambleEvent.second->GetRxPower(measurementBand);
        if (!maxRxPower || (rxPower > *maxRxPower))
        {
            maxRxPower = rxPower;
            maxEvent = preambleEvent.second;
        }
    }

    NS_ASSERT(maxEvent);
    if (maxEvent != event)
    {
        NS_LOG_DEBUG("Receiver got a stronger packet with UID "
                     << maxEvent->GetPpdu()->GetUid()
                     << " during preamble detection: drop packet with UID "
                     << event->GetPpdu()->GetUid());
        m_wifiPhy->NotifyRxPpduDrop(event->GetPpdu(), BUSY_DECODING_PREAMBLE);

        auto it = m_wifiPhy->m_currentPreambleEvents.find(
            {event->GetPpdu()->GetUid(), event->GetPpdu()->GetPreamble()});
        m_wifiPhy->m_currentPreambleEvents.erase(it);
        // This is needed to cleanup the m_firstPowerPerBand so that the first power corresponds to
        // the power at the start of the PPDU
        m_wifiPhy->m_interference->NotifyRxEnd(maxEvent->GetStartTime(),
                                               m_wifiPhy->GetCurrentFrequencyRange());
        // Make sure InterferenceHelper keeps recording events
        m_wifiPhy->m_interference->NotifyRxStart(m_wifiPhy->GetCurrentFrequencyRange());
        return;
    }

    m_wifiPhy->m_currentEvent = event;

    const auto snr = m_wifiPhy->m_interference->CalculateSnr(m_wifiPhy->m_currentEvent,
                                                             measurementChannelWidth,
                                                             1,
                                                             measurementBand);
    NS_LOG_DEBUG("SNR(dB)=" << RatioToDb(snr) << " at end of preamble detection period");

    if (const auto power = m_wifiPhy->m_currentEvent->GetRxPower(measurementBand);
        (!m_wifiPhy->m_preambleDetectionModel && maxRxPower && (*maxRxPower > Watt_u{0.0})) ||
        (m_wifiPhy->m_preambleDetectionModel && power > Watt_u{0.0} &&
         m_wifiPhy->m_preambleDetectionModel->IsPreambleDetected(WToDbm(power),
                                                                 snr,
                                                                 measurementChannelWidth)))
    {
        // A bit convoluted but it enables to sync all PHYs
        for (auto& it : m_wifiPhy->m_phyEntities)
        {
            it.second->CancelRunningEndPreambleDetectionEvents();
        }

        for (auto it = m_wifiPhy->m_currentPreambleEvents.begin();
             it != m_wifiPhy->m_currentPreambleEvents.end();)
        {
            if (it->second != m_wifiPhy->m_currentEvent)
            {
                NS_LOG_DEBUG("Drop packet with UID " << it->first.first << " and preamble "
                                                     << it->first.second << " arrived at time "
                                                     << it->second->GetStartTime());
                WifiPhyRxfailureReason reason;
                if (m_wifiPhy->m_currentEvent->GetPpdu()->GetUid() > it->first.first)
                {
                    reason = PREAMBLE_DETECTION_PACKET_SWITCH;
                    // This is needed to cleanup the m_firstPowerPerBand so that the first power
                    // corresponds to the power at the start of the PPDU
                    m_wifiPhy->m_interference->NotifyRxEnd(
                        m_wifiPhy->m_currentEvent->GetStartTime(),
                        m_wifiPhy->GetCurrentFrequencyRange());
                }
                else
                {
                    reason = BUSY_DECODING_PREAMBLE;
                }
                m_wifiPhy->NotifyRxPpduDrop(it->second->GetPpdu(), reason);

                it = m_wifiPhy->m_currentPreambleEvents.erase(it);
            }
            else
            {
                ++it;
            }
        }

        // Make sure InterferenceHelper keeps recording events
        m_wifiPhy->m_interference->NotifyRxStart(m_wifiPhy->GetCurrentFrequencyRange());

        m_wifiPhy->NotifyRxBegin(GetAddressedPsduInPpdu(m_wifiPhy->m_currentEvent->GetPpdu()),
                                 m_wifiPhy->m_currentEvent->GetRxPowerPerBand());
        m_wifiPhy->m_timeLastPreambleDetected = Simulator::Now();

        // Continue receiving preamble
        const auto durationTillEnd =
            GetDuration(WIFI_PPDU_FIELD_PREAMBLE, event->GetPpdu()->GetTxVector()) -
            WifiPhy::GetPreambleDetectionDuration();
        m_wifiPhy->NotifyCcaBusy(event->GetPpdu(),
                                 durationTillEnd); // will be prolonged by next field
        m_wifiPhy->m_endPhyRxEvent = Simulator::Schedule(durationTillEnd,
                                                         &PhyEntity::EndReceiveField,
                                                         this,
                                                         WIFI_PPDU_FIELD_PREAMBLE,
                                                         event);
    }
    else
    {
        NS_LOG_DEBUG("Drop packet because PHY preamble detection failed");
        // Like CCA-SD, CCA-ED is governed by the 4 us CCA window to flag CCA-BUSY
        // for any received signal greater than the CCA-ED threshold.
        DropPreambleEvent(m_wifiPhy->m_currentEvent->GetPpdu(),
                          PREAMBLE_DETECT_FAILURE,
                          m_wifiPhy->m_currentEvent->GetEndTime());
        if (m_wifiPhy->m_currentPreambleEvents.empty())
        {
            // Do not erase events if there are still pending preamble events to be processed
            m_wifiPhy->m_interference->NotifyRxEnd(Simulator::Now(),
                                                   m_wifiPhy->GetCurrentFrequencyRange());
        }
        m_wifiPhy->m_currentEvent = nullptr;
        // Cancel preamble reception
        m_wifiPhy->m_endPhyRxEvent.Cancel();
    }
}

bool
PhyEntity::IsConfigSupported(Ptr<const WifiPpdu> ppdu) const
{
    WifiMode txMode = ppdu->GetTxVector().GetMode();
    if (!IsModeSupported(txMode))
    {
        NS_LOG_DEBUG("Drop packet because it was sent using an unsupported mode (" << txMode
                                                                                   << ")");
        return false;
    }
    return true;
}

void
PhyEntity::CancelAllEvents()
{
    NS_LOG_FUNCTION(this);
    CancelRunningEndPreambleDetectionEvents();
    for (auto& endRxPayloadEvent : m_endRxPayloadEvents)
    {
        endRxPayloadEvent.Cancel();
    }
    m_endRxPayloadEvents.clear();
    for (auto& endMpduEvent : m_endOfMpduEvents)
    {
        endMpduEvent.Cancel();
    }
    m_endOfMpduEvents.clear();
    for (auto& [staId, endOfMacHdrEvents] : m_endOfMacHdrEvents)
    {
        for (auto& endMacHdrEvent : endOfMacHdrEvents)
        {
            endMacHdrEvent.Cancel();
        }
    }
    m_endOfMacHdrEvents.clear();
}

void
PhyEntity::CancelRunningEndPreambleDetectionEvents()
{
    NS_LOG_FUNCTION(this);
    for (auto& endPreambleDetectionEvent : m_endPreambleDetectionEvents)
    {
        endPreambleDetectionEvent.Cancel();
    }
    m_endPreambleDetectionEvents.clear();
}

void
PhyEntity::AbortCurrentReception(WifiPhyRxfailureReason reason)
{
    NS_LOG_FUNCTION(this << reason);
    DoAbortCurrentReception(reason);
    m_wifiPhy->AbortCurrentReception(reason);
}

void
PhyEntity::DoAbortCurrentReception(WifiPhyRxfailureReason reason)
{
    NS_LOG_FUNCTION(this << reason);
    if (m_wifiPhy->m_currentEvent) // Otherwise abort has already been called just before
    {
        for (auto& endMpduEvent : m_endOfMpduEvents)
        {
            endMpduEvent.Cancel();
        }
        m_endOfMpduEvents.clear();
        for (auto& [staId, endOfMacHdrEvents] : m_endOfMacHdrEvents)
        {
            for (auto& endMacHdrEvent : endOfMacHdrEvents)
            {
                endMacHdrEvent.Cancel();
            }
        }
        m_endOfMacHdrEvents.clear();
    }
}

void
PhyEntity::ResetReceive(Ptr<Event> event)
{
    NS_LOG_FUNCTION(this << *event);
    DoResetReceive(event);
    NS_ASSERT(!m_wifiPhy->IsStateRx());
    m_wifiPhy->m_interference->NotifyRxEnd(Simulator::Now(), m_wifiPhy->GetCurrentFrequencyRange());
    NS_ASSERT(m_endRxPayloadEvents.size() == 1 && m_endRxPayloadEvents.front().IsExpired());
    m_endRxPayloadEvents.clear();
    m_wifiPhy->m_currentEvent = nullptr;
    m_wifiPhy->m_currentPreambleEvents.clear();
    m_wifiPhy->SwitchMaybeToCcaBusy(event->GetPpdu());
}

void
PhyEntity::DoResetReceive(Ptr<Event> event)
{
    NS_LOG_FUNCTION(this << *event);
    NS_ASSERT(event->GetEndTime() == Simulator::Now());
}

double
PhyEntity::GetRandomValue() const
{
    return m_wifiPhy->m_random->GetValue();
}

Watt_u
PhyEntity::GetRxPowerForPpdu(Ptr<Event> event) const
{
    return event->GetRxPower(GetPrimaryBand(GetMeasurementChannelWidth(event->GetPpdu())));
}

Ptr<const Event>
PhyEntity::GetCurrentEvent() const
{
    return m_wifiPhy->m_currentEvent;
}

WifiSpectrumBandInfo
PhyEntity::GetPrimaryBand(MHz_u bandWidth) const
{
    if (static_cast<uint16_t>(m_wifiPhy->GetChannelWidth()) % 20 != 0)
    {
        return m_wifiPhy->GetBand(bandWidth);
    }
    return m_wifiPhy->GetBand(bandWidth,
                              m_wifiPhy->GetOperatingChannel().GetPrimaryChannelIndex(bandWidth));
}

WifiSpectrumBandInfo
PhyEntity::GetSecondaryBand(MHz_u bandWidth) const
{
    NS_ASSERT(m_wifiPhy->GetChannelWidth() >= MHz_u{40});
    return m_wifiPhy->GetBand(bandWidth,
                              m_wifiPhy->GetOperatingChannel().GetSecondaryChannelIndex(bandWidth));
}

MHz_u
PhyEntity::GetRxChannelWidth(const WifiTxVector& txVector) const
{
    return std::min(m_wifiPhy->GetChannelWidth(), txVector.GetChannelWidth());
}

dBm_u
PhyEntity::GetCcaThreshold(const Ptr<const WifiPpdu> ppdu,
                           WifiChannelListType /*channelType*/) const
{
    return (!ppdu) ? m_wifiPhy->GetCcaEdThreshold() : m_wifiPhy->GetCcaSensitivityThreshold();
}

Time
PhyEntity::GetDelayUntilCcaEnd(dBm_u threshold, const WifiSpectrumBandInfo& band)
{
    return m_wifiPhy->m_interference->GetEnergyDuration(DbmToW(threshold), band);
}

void
PhyEntity::SwitchMaybeToCcaBusy(const Ptr<const WifiPpdu> ppdu)
{
    // We are here because we have received the first bit of a packet and we are
    // not going to be able to synchronize on it
    // In this model, CCA becomes busy when the aggregation of all signals as
    // tracked by the InterferenceHelper class is higher than the CcaBusyThreshold
    const auto ccaIndication = GetCcaIndication(ppdu);
    if (ccaIndication.has_value())
    {
        NS_LOG_DEBUG("CCA busy for " << ccaIndication.value().second << " during "
                                     << ccaIndication.value().first.As(Time::S));
        m_state->SwitchMaybeToCcaBusy(ccaIndication.value().first,
                                      ccaIndication.value().second,
                                      {});
        return;
    }
    if (ppdu)
    {
        SwitchMaybeToCcaBusy(nullptr);
    }
}

PhyEntity::CcaIndication
PhyEntity::GetCcaIndication(const Ptr<const WifiPpdu> ppdu)
{
    const auto channelWidth = GetMeasurementChannelWidth(ppdu);
    NS_LOG_FUNCTION(this << channelWidth);
    const auto ccaThreshold = GetCcaThreshold(ppdu, WIFI_CHANLIST_PRIMARY);
    const Time delayUntilCcaEnd = GetDelayUntilCcaEnd(ccaThreshold, GetPrimaryBand(channelWidth));
    if (delayUntilCcaEnd.IsStrictlyPositive())
    {
        return std::make_pair(delayUntilCcaEnd, WIFI_CHANLIST_PRIMARY);
    }
    return std::nullopt;
}

void
PhyEntity::NotifyCcaBusy(const Ptr<const WifiPpdu> /*ppdu*/,
                         Time duration,
                         WifiChannelListType channelType)
{
    NS_LOG_FUNCTION(this << duration << channelType);
    NS_LOG_DEBUG("CCA busy for " << channelType << " during " << duration.As(Time::S));
    m_state->SwitchMaybeToCcaBusy(duration, channelType, {});
}

uint64_t
PhyEntity::ObtainNextUid(const WifiTxVector& /* txVector */)
{
    NS_LOG_FUNCTION(this);
    return m_globalPpduUid++;
}

Time
PhyEntity::GetMaxDelayPpduSameUid(const WifiTxVector& /*txVector*/)
{
    return Seconds(0);
}

void
PhyEntity::NotifyPayloadBegin(const WifiTxVector& txVector, const Time& payloadDuration)
{
    m_wifiPhy->m_phyRxPayloadBeginTrace(txVector, payloadDuration);
}

void
PhyEntity::StartTx(Ptr<const WifiPpdu> ppdu)
{
    NS_LOG_FUNCTION(this << ppdu);
    auto txPower = m_wifiPhy->GetTxPowerForTransmission(ppdu) + m_wifiPhy->GetTxGain();
    auto txVector = ppdu->GetTxVector();
    auto txPowerSpectrum = GetTxPowerSpectralDensity(DbmToW(txPower), ppdu);
    Transmit(ppdu->GetTxDuration(), ppdu, txPower, txPowerSpectrum, "transmission");
}

void
PhyEntity::Transmit(Time txDuration,
                    Ptr<const WifiPpdu> ppdu,
                    dBm_u txPower,
                    Ptr<SpectrumValue> txPowerSpectrum,
                    const std::string& type)
{
    NS_LOG_FUNCTION(this << txDuration << ppdu << txPower << type);
    NS_LOG_DEBUG("Start " << type << ": signal power before antenna gain=" << txPower << "dBm");
    auto txParams = Create<WifiSpectrumSignalParameters>();
    txParams->duration = txDuration;
    txParams->psd = txPowerSpectrum;
    txParams->ppdu = ppdu;
    NS_LOG_DEBUG("Starting " << type << " with power " << txPower << " dBm on channel "
                             << +m_wifiPhy->GetChannelNumber() << " for "
                             << txParams->duration.As(Time::MS));
    NS_LOG_DEBUG("Starting " << type << " with integrated spectrum power "
                             << WToDbm(Integral(*txPowerSpectrum)) << " dBm; spectrum model Uid: "
                             << txPowerSpectrum->GetSpectrumModel()->GetUid());
    auto spectrumWifiPhy = DynamicCast<SpectrumWifiPhy>(m_wifiPhy);
    NS_ASSERT(spectrumWifiPhy);
    spectrumWifiPhy->Transmit(txParams);
}

MHz_u
PhyEntity::GetGuardBandwidth(MHz_u currentChannelWidth) const
{
    return m_wifiPhy->GetGuardBandwidth(currentChannelWidth);
}

std::tuple<dBr_u, dBr_u, dBr_u>
PhyEntity::GetTxMaskRejectionParams() const
{
    return m_wifiPhy->GetTxMaskRejectionParams();
}

Time
PhyEntity::CalculateTxDuration(const WifiConstPsduMap& psduMap,
                               const WifiTxVector& txVector,
                               WifiPhyBand band) const
{
    NS_ASSERT(psduMap.size() == 1);
    const auto& it = psduMap.begin();
    return WifiPhy::CalculateTxDuration(it->second->GetSize(), txVector, band, it->first);
}

bool
PhyEntity::CanStartRx(Ptr<const WifiPpdu> ppdu) const
{
    // The PHY shall not issue a PHY-RXSTART.indication primitive in response to a PPDU that does
    // not overlap the primary channel
    const auto channelWidth = m_wifiPhy->GetChannelWidth();
    const auto primaryWidth = ((static_cast<uint16_t>(channelWidth) % 20 == 0)
                                   ? MHz_u{20}
                                   : channelWidth); // if the channel width is a multiple of 20 MHz,
                                                    // then we consider the primary20 channel
    const auto p20CenterFreq =
        m_wifiPhy->GetOperatingChannel().GetPrimaryChannelCenterFrequency(primaryWidth);
    const auto p20MinFreq = p20CenterFreq - (primaryWidth / 2);
    const auto p20MaxFreq = p20CenterFreq + (primaryWidth / 2);
    const auto txChannelWidth = (ppdu->GetTxChannelWidth() / ppdu->GetTxCenterFreqs().size());
    for (auto txCenterFreq : ppdu->GetTxCenterFreqs())
    {
        const auto minTxFreq = txCenterFreq - txChannelWidth / 2;
        const auto maxTxFreq = txCenterFreq + txChannelWidth / 2;
        if ((p20MinFreq >= minTxFreq) && (p20MaxFreq <= maxTxFreq))
        {
            return true;
        }
    }
    return false;
}

Ptr<const WifiPpdu>
PhyEntity::GetRxPpduFromTxPpdu(Ptr<const WifiPpdu> ppdu)
{
    return ppdu;
}

} // namespace ns3
