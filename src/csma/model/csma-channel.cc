/*
 * Copyright (c) 2007 Emmanuelle Laprise
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Emmanuelle Laprise <emmanuelle.laprise@bluekazoo.ca>
 */

#include "csma-channel.h"

#include "csma-net-device.h"

#include "ns3/log.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("CsmaChannel");

NS_OBJECT_ENSURE_REGISTERED(CsmaChannel);

TypeId
CsmaChannel::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::CsmaChannel")
            .SetParent<Channel>()
            .SetGroupName("Csma")
            .AddConstructor<CsmaChannel>()
            .AddAttribute(
                "DataRate",
                "The transmission data rate to be provided to devices connected to the channel",
                DataRateValue(DataRate(0xffffffff)),
                MakeDataRateAccessor(&CsmaChannel::m_bps),
                MakeDataRateChecker())
            .AddAttribute("Delay",
                          "Transmission delay through the channel",
                          TimeValue(Seconds(0)),
                          MakeTimeAccessor(&CsmaChannel::m_delay),
                          MakeTimeChecker());
    return tid;
}

CsmaChannel::CsmaChannel()
    : Channel()
{
    NS_LOG_FUNCTION_NOARGS();
    m_state = IDLE;
    m_deviceList.clear();
}

CsmaChannel::~CsmaChannel()
{
    NS_LOG_FUNCTION(this);
    m_deviceList.clear();
}

int32_t
CsmaChannel::Attach(Ptr<CsmaNetDevice> device)
{
    NS_LOG_FUNCTION(this << device);
    NS_ASSERT(device);

    CsmaDeviceRec rec(device);

    m_deviceList.push_back(rec);
    return (m_deviceList.size() - 1);
}

bool
CsmaChannel::Reattach(Ptr<CsmaNetDevice> device)
{
    NS_LOG_FUNCTION(this << device);
    NS_ASSERT(device);

    for (auto it = m_deviceList.begin(); it < m_deviceList.end(); it++)
    {
        if (it->devicePtr == device)
        {
            if (!it->active)
            {
                it->active = true;
                return true;
            }
            else
            {
                return false;
            }
        }
    }
    return false;
}

bool
CsmaChannel::Reattach(uint32_t deviceId)
{
    NS_LOG_FUNCTION(this << deviceId);

    if (deviceId < m_deviceList.size())
    {
        return false;
    }

    if (m_deviceList[deviceId].active)
    {
        return false;
    }
    else
    {
        m_deviceList[deviceId].active = true;
        return true;
    }
}

bool
CsmaChannel::Detach(uint32_t deviceId)
{
    NS_LOG_FUNCTION(this << deviceId);

    if (deviceId >= m_deviceList.size())
    {
        return false;
    }

    if (!m_deviceList[deviceId].active)
    {
        NS_LOG_WARN("CsmaChannel::Detach(): Device is already detached (" << deviceId << ")");
        return false;
    }

    m_deviceList[deviceId].active = false;

    if ((m_state == TRANSMITTING) && (m_currentSrc == deviceId))
    {
        NS_LOG_WARN("CsmaChannel::Detach(): Device is currently transmitting (" << deviceId << ")");
    }

    return true;
}

bool
CsmaChannel::Detach(Ptr<CsmaNetDevice> device)
{
    NS_LOG_FUNCTION(this << device);
    NS_ASSERT(device);

    for (auto it = m_deviceList.begin(); it < m_deviceList.end(); it++)
    {
        if ((it->devicePtr == device) && (it->active))
        {
            it->active = false;
            return true;
        }
    }
    return false;
}

bool
CsmaChannel::TransmitStart(Ptr<const Packet> p, uint32_t srcId)
{
    NS_LOG_FUNCTION(this << p << srcId);
    NS_LOG_INFO("UID is " << p->GetUid() << ")");

    if (m_state != IDLE)
    {
        NS_LOG_WARN("CsmaChannel::TransmitStart(): State is not IDLE");
        return false;
    }

    if (!IsActive(srcId))
    {
        NS_LOG_ERROR(
            "CsmaChannel::TransmitStart(): Seclected source is not currently attached to network");
        return false;
    }

    NS_LOG_LOGIC("switch to TRANSMITTING");
    m_currentPkt = p;
    m_currentSrc = srcId;
    m_state = TRANSMITTING;
    return true;
}

bool
CsmaChannel::IsActive(uint32_t deviceId)
{
    return m_deviceList[deviceId].active;
}

bool
CsmaChannel::TransmitEnd()
{
    NS_LOG_FUNCTION(this << m_currentPkt << m_currentSrc);
    NS_LOG_INFO("UID is " << m_currentPkt->GetUid() << ")");

    NS_ASSERT(m_state == TRANSMITTING);
    m_state = PROPAGATING;

    bool retVal = true;

    if (!IsActive(m_currentSrc))
    {
        NS_LOG_ERROR("CsmaChannel::TransmitEnd(): Seclected source was detached before the end of "
                     "the transmission");
        retVal = false;
    }

    NS_LOG_LOGIC("Schedule event in " << m_delay.As(Time::S));

    NS_LOG_LOGIC("Receive");

    for (auto it = m_deviceList.begin(); it < m_deviceList.end(); it++)
    {
        if (it->IsActive() && it->devicePtr != m_deviceList[m_currentSrc].devicePtr)
        {
            // schedule reception events
            Simulator::ScheduleWithContext(it->devicePtr->GetNode()->GetId(),
                                           m_delay,
                                           &CsmaNetDevice::Receive,
                                           it->devicePtr,
                                           m_currentPkt,
                                           m_deviceList[m_currentSrc].devicePtr);
        }
    }

    // also schedule for the tx side to go back to IDLE
    Simulator::Schedule(m_delay, &CsmaChannel::PropagationCompleteEvent, this);
    return retVal;
}

void
CsmaChannel::PropagationCompleteEvent()
{
    NS_LOG_FUNCTION(this << m_currentPkt);
    NS_LOG_INFO("UID is " << m_currentPkt->GetUid() << ")");

    NS_ASSERT(m_state == PROPAGATING);
    m_state = IDLE;
}

uint32_t
CsmaChannel::GetNumActDevices()
{
    int numActDevices = 0;
    for (auto it = m_deviceList.begin(); it < m_deviceList.end(); it++)
    {
        if (it->active)
        {
            numActDevices++;
        }
    }
    return numActDevices;
}

std::size_t
CsmaChannel::GetNDevices() const
{
    return m_deviceList.size();
}

Ptr<CsmaNetDevice>
CsmaChannel::GetCsmaDevice(std::size_t i) const
{
    return m_deviceList[i].devicePtr;
}

int32_t
CsmaChannel::GetDeviceNum(Ptr<CsmaNetDevice> device)
{
    int i = 0;
    for (auto it = m_deviceList.begin(); it < m_deviceList.end(); it++)
    {
        if (it->devicePtr == device)
        {
            if (it->active)
            {
                return i;
            }
            else
            {
                return -2;
            }
        }
        i++;
    }
    return -1;
}

bool
CsmaChannel::IsBusy()
{
    return m_state != IDLE;
}

DataRate
CsmaChannel::GetDataRate()
{
    return m_bps;
}

Time
CsmaChannel::GetDelay()
{
    return m_delay;
}

WireState
CsmaChannel::GetState()
{
    return m_state;
}

Ptr<NetDevice>
CsmaChannel::GetDevice(std::size_t i) const
{
    return GetCsmaDevice(i);
}

CsmaDeviceRec::CsmaDeviceRec()
{
    active = false;
}

CsmaDeviceRec::CsmaDeviceRec(Ptr<CsmaNetDevice> device)
{
    devicePtr = device;
    active = true;
}

CsmaDeviceRec::CsmaDeviceRec(const CsmaDeviceRec& deviceRec)
{
    devicePtr = deviceRec.devicePtr;
    active = deviceRec.active;
}

bool
CsmaDeviceRec::IsActive() const
{
    return active;
}

} // namespace ns3
