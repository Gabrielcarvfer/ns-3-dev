/*
 * Copyright (c) 2004,2005,2006 INRIA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Federico Maguolo <maguolof@dei.unipd.it>
 */

#include "cara-wifi-manager.h"

#include "ns3/log.h"
#include "ns3/wifi-tx-vector.h"

#define Min(a, b) ((a < b) ? a : b)

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("CaraWifiManager");

/**
 * @brief hold per-remote-station state for CARA Wifi manager.
 *
 * This struct extends from WifiRemoteStation struct to hold additional
 * information required by the CARA Wifi manager
 */
struct CaraWifiRemoteStation : public WifiRemoteStation
{
    uint32_t m_timer;   ///< timer count
    uint32_t m_success; ///< success count
    uint32_t m_failed;  ///< failed count
    uint8_t m_rate;     ///< rate in bps
};

NS_OBJECT_ENSURE_REGISTERED(CaraWifiManager);

TypeId
CaraWifiManager::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::CaraWifiManager")
            .SetParent<WifiRemoteStationManager>()
            .SetGroupName("Wifi")
            .AddConstructor<CaraWifiManager>()
            .AddAttribute(
                "ProbeThreshold",
                "The number of consecutive transmissions failure to activate the RTS probe.",
                UintegerValue(1),
                MakeUintegerAccessor(&CaraWifiManager::m_probeThreshold),
                MakeUintegerChecker<uint32_t>())
            .AddAttribute("FailureThreshold",
                          "The number of consecutive transmissions failure to decrease the rate.",
                          UintegerValue(2),
                          MakeUintegerAccessor(&CaraWifiManager::m_failureThreshold),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("SuccessThreshold",
                          "The minimum number of successful transmissions to try a new rate.",
                          UintegerValue(10),
                          MakeUintegerAccessor(&CaraWifiManager::m_successThreshold),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("Timeout",
                          "The 'timer' in the CARA algorithm",
                          UintegerValue(15),
                          MakeUintegerAccessor(&CaraWifiManager::m_timerTimeout),
                          MakeUintegerChecker<uint32_t>())
            .AddTraceSource("Rate",
                            "Traced value for rate changes (b/s)",
                            MakeTraceSourceAccessor(&CaraWifiManager::m_currentRate),
                            "ns3::TracedValueCallback::Uint64");
    return tid;
}

CaraWifiManager::CaraWifiManager()
    : WifiRemoteStationManager(),
      m_currentRate(0)
{
    NS_LOG_FUNCTION(this);
}

CaraWifiManager::~CaraWifiManager()
{
    NS_LOG_FUNCTION(this);
}

void
CaraWifiManager::DoInitialize()
{
    NS_LOG_FUNCTION(this);
    if (GetHtSupported())
    {
        NS_FATAL_ERROR("WifiRemoteStationManager selected does not support HT rates");
    }
    if (GetVhtSupported())
    {
        NS_FATAL_ERROR("WifiRemoteStationManager selected does not support VHT rates");
    }
    if (GetHeSupported())
    {
        NS_FATAL_ERROR("WifiRemoteStationManager selected does not support HE rates");
    }
}

WifiRemoteStation*
CaraWifiManager::DoCreateStation() const
{
    NS_LOG_FUNCTION(this);
    auto station = new CaraWifiRemoteStation();
    station->m_rate = 0;
    station->m_success = 0;
    station->m_failed = 0;
    station->m_timer = 0;
    return station;
}

void
CaraWifiManager::DoReportRtsFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
}

void
CaraWifiManager::DoReportDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    auto station = static_cast<CaraWifiRemoteStation*>(st);
    station->m_timer++;
    station->m_failed++;
    station->m_success = 0;
    if (station->m_failed >= m_failureThreshold)
    {
        NS_LOG_DEBUG("self=" << station << " dec rate");
        if (station->m_rate != 0)
        {
            station->m_rate--;
        }
        station->m_failed = 0;
        station->m_timer = 0;
    }
}

void
CaraWifiManager::DoReportRxOk(WifiRemoteStation* st, double rxSnr, WifiMode txMode)
{
    NS_LOG_FUNCTION(this << st << rxSnr << txMode);
}

void
CaraWifiManager::DoReportRtsOk(WifiRemoteStation* st,
                               double ctsSnr,
                               WifiMode ctsMode,
                               double rtsSnr)
{
    NS_LOG_FUNCTION(this << st << ctsSnr << ctsMode << rtsSnr);
}

void
CaraWifiManager::DoReportDataOk(WifiRemoteStation* st,
                                double ackSnr,
                                WifiMode ackMode,
                                double dataSnr,
                                MHz_u dataChannelWidth,
                                uint8_t dataNss)
{
    NS_LOG_FUNCTION(this << st << ackSnr << ackMode << dataSnr << dataChannelWidth << +dataNss);
    auto station = static_cast<CaraWifiRemoteStation*>(st);
    station->m_timer++;
    station->m_success++;
    station->m_failed = 0;
    NS_LOG_DEBUG("self=" << station << " data ok success=" << station->m_success
                         << ", timer=" << station->m_timer);
    if (station->m_success == m_successThreshold || station->m_timer >= m_timerTimeout)
    {
        if (station->m_rate < GetNSupported(station) - 1)
        {
            station->m_rate++;
        }
        NS_LOG_DEBUG("self=" << station << " inc rate=" << +station->m_rate);
        station->m_timer = 0;
        station->m_success = 0;
    }
}

void
CaraWifiManager::DoReportFinalRtsFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
}

void
CaraWifiManager::DoReportFinalDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
}

WifiTxVector
CaraWifiManager::DoGetDataTxVector(WifiRemoteStation* st, MHz_u allowedWidth)
{
    NS_LOG_FUNCTION(this << st << allowedWidth);
    auto station = static_cast<CaraWifiRemoteStation*>(st);
    auto channelWidth = GetChannelWidth(station);
    if (channelWidth > MHz_u{20} && channelWidth != MHz_u{22})
    {
        channelWidth = MHz_u{20};
    }
    WifiMode mode = GetSupported(station, station->m_rate);
    uint64_t rate = mode.GetDataRate(channelWidth);
    if (m_currentRate != rate)
    {
        NS_LOG_DEBUG("New datarate: " << rate);
        m_currentRate = rate;
    }
    return WifiTxVector(
        mode,
        GetDefaultTxPowerLevel(),
        GetPreambleForTransmission(mode.GetModulationClass(), GetShortPreambleEnabled()),
        NanoSeconds(800),
        1,
        1,
        0,
        channelWidth,
        GetAggregation(station));
}

WifiTxVector
CaraWifiManager::DoGetRtsTxVector(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    auto station = static_cast<CaraWifiRemoteStation*>(st);
    /// @todo we could/should implement the Arf algorithm for
    /// RTS only by picking a single rate within the BasicRateSet.
    auto channelWidth = GetChannelWidth(station);
    if (channelWidth > MHz_u{20} && channelWidth != MHz_u{22})
    {
        channelWidth = MHz_u{20};
    }
    WifiMode mode;
    if (!GetUseNonErpProtection())
    {
        mode = GetSupported(station, 0);
    }
    else
    {
        mode = GetNonErpSupported(station, 0);
    }
    return WifiTxVector(
        mode,
        GetDefaultTxPowerLevel(),
        GetPreambleForTransmission(mode.GetModulationClass(), GetShortPreambleEnabled()),
        NanoSeconds(800),
        1,
        1,
        0,
        channelWidth,
        GetAggregation(station));
}

bool
CaraWifiManager::DoNeedRts(WifiRemoteStation* st, uint32_t size, bool normally)
{
    NS_LOG_FUNCTION(this << st << size << normally);
    auto station = static_cast<CaraWifiRemoteStation*>(st);
    return normally || station->m_failed >= m_probeThreshold;
}

} // namespace ns3
