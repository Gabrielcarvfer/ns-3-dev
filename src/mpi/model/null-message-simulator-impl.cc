/*
 *  Copyright 2013. Lawrence Livermore National Security, LLC.
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Steven Smith <smith84@llnl.gov>
 */

/**
 * @file
 * @ingroup mpi
 * Implementation of class ns3::NullMessageSimulatorImpl.
 */

#include "null-message-simulator-impl.h"

#include "mpi-interface.h"
#include "null-message-mpi-interface.h"
#include "remote-channel-bundle-manager.h"
#include "remote-channel-bundle.h"

#include "ns3/assert.h"
#include "ns3/channel.h"
#include "ns3/double.h"
#include "ns3/event-impl.h"
#include "ns3/log.h"
#include "ns3/node-container.h"
#include "ns3/pointer.h"
#include "ns3/ptr.h"
#include "ns3/scheduler.h"
#include "ns3/simulator.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("NullMessageSimulatorImpl");

NS_OBJECT_ENSURE_REGISTERED(NullMessageSimulatorImpl);

NullMessageSimulatorImpl* NullMessageSimulatorImpl::g_instance = nullptr;

TypeId
NullMessageSimulatorImpl::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::NullMessageSimulatorImpl")
            .SetParent<SimulatorImpl>()
            .SetGroupName("Mpi")
            .AddConstructor<NullMessageSimulatorImpl>()
            .AddAttribute("SchedulerTune",
                          "Null Message scheduler tuning parameter",
                          DoubleValue(1.0),
                          MakeDoubleAccessor(&NullMessageSimulatorImpl::m_schedulerTune),
                          MakeDoubleChecker<double>(0.01, 1.0));
    return tid;
}

NullMessageSimulatorImpl::NullMessageSimulatorImpl()
{
    NS_LOG_FUNCTION(this);

    m_myId = MpiInterface::GetSystemId();
    m_systemCount = MpiInterface::GetSize();

    m_stop = false;
    m_uid = EventId::UID::VALID;
    m_currentUid = EventId::UID::INVALID;
    m_currentTs = 0;
    m_currentContext = Simulator::NO_CONTEXT;
    m_unscheduledEvents = 0;
    m_eventCount = 0;
    m_events = nullptr;

    m_safeTime = Seconds(0);

    NS_ASSERT(g_instance == nullptr);
    g_instance = this;
}

NullMessageSimulatorImpl::~NullMessageSimulatorImpl()
{
    NS_LOG_FUNCTION(this);
}

void
NullMessageSimulatorImpl::DoDispose()
{
    NS_LOG_FUNCTION(this);

    while (!m_events->IsEmpty())
    {
        Scheduler::Event next = m_events->RemoveNext();
        next.impl->Unref();
    }
    m_events = nullptr;
    SimulatorImpl::DoDispose();
}

void
NullMessageSimulatorImpl::Destroy()
{
    NS_LOG_FUNCTION(this);

    while (!m_destroyEvents.empty())
    {
        Ptr<EventImpl> ev = m_destroyEvents.front().PeekEventImpl();
        m_destroyEvents.pop_front();
        NS_LOG_LOGIC("handle destroy " << ev);
        if (!ev->IsCancelled())
        {
            ev->Invoke();
        }
    }

    RemoteChannelBundleManager::Destroy();
    MpiInterface::Destroy();
}

void
NullMessageSimulatorImpl::CalculateLookAhead()
{
    NS_LOG_FUNCTION(this);

    if (MpiInterface::GetSize() > 1)
    {
        NodeContainer c = NodeContainer::GetGlobal();
        for (auto iter = c.Begin(); iter != c.End(); ++iter)
        {
            if ((*iter)->GetSystemId() != MpiInterface::GetSystemId())
            {
                continue;
            }

            for (uint32_t i = 0; i < (*iter)->GetNDevices(); ++i)
            {
                Ptr<NetDevice> localNetDevice = (*iter)->GetDevice(i);
                // only works for p2p links currently
                if (!localNetDevice->IsPointToPoint())
                {
                    continue;
                }
                Ptr<Channel> channel = localNetDevice->GetChannel();
                if (!channel)
                {
                    continue;
                }

                // grab the adjacent node
                Ptr<Node> remoteNode;
                if (channel->GetDevice(0) == localNetDevice)
                {
                    remoteNode = (channel->GetDevice(1))->GetNode();
                }
                else
                {
                    remoteNode = (channel->GetDevice(0))->GetNode();
                }

                // if it's not remote, don't consider it
                if (remoteNode->GetSystemId() == MpiInterface::GetSystemId())
                {
                    continue;
                }

                /**
                 * Add this channel to the remote channel bundle from this task to MPI task on other
                 * side of the channel.
                 */
                Ptr<RemoteChannelBundle> remoteChannelBundle =
                    RemoteChannelBundleManager::Find(remoteNode->GetSystemId());
                if (!remoteChannelBundle)
                {
                    remoteChannelBundle =
                        RemoteChannelBundleManager::Add(remoteNode->GetSystemId());
                }

                TimeValue delay;
                channel->GetAttribute("Delay", delay);
                remoteChannelBundle->AddChannel(channel, delay.Get());
            }
        }
    }

    // Completed setup of remote channel bundles.  Setup send and receive buffers.
    NullMessageMpiInterface::InitializeSendReceiveBuffers();

    // Initialized to 0 as we don't have a simulation start time.
    m_safeTime = Time(0);
}

void
NullMessageSimulatorImpl::SetScheduler(ObjectFactory schedulerFactory)
{
    NS_LOG_FUNCTION(this << schedulerFactory);

    Ptr<Scheduler> scheduler = schedulerFactory.Create<Scheduler>();

    if (m_events)
    {
        while (!m_events->IsEmpty())
        {
            Scheduler::Event next = m_events->RemoveNext();
            scheduler->Insert(next);
        }
    }
    m_events = scheduler;
}

void
NullMessageSimulatorImpl::ProcessOneEvent()
{
    NS_LOG_FUNCTION(this);

    Scheduler::Event next = m_events->RemoveNext();

    PreEventHook(EventId(next.impl, next.key.m_ts, next.key.m_context, next.key.m_uid));

    NS_ASSERT(next.key.m_ts >= m_currentTs);
    m_unscheduledEvents--;
    m_eventCount++;

    NS_LOG_LOGIC("handle " << next.key.m_ts);
    m_currentTs = next.key.m_ts;
    m_currentContext = next.key.m_context;
    m_currentUid = next.key.m_uid;
    next.impl->Invoke();
    next.impl->Unref();
}

bool
NullMessageSimulatorImpl::IsFinished() const
{
    return m_events->IsEmpty() || m_stop;
}

Time
NullMessageSimulatorImpl::Next() const
{
    NS_LOG_FUNCTION(this);

    NS_ASSERT(!m_events->IsEmpty());

    Scheduler::Event ev = m_events->PeekNext();
    return TimeStep(ev.key.m_ts);
}

void
NullMessageSimulatorImpl::ScheduleNullMessageEvent(Ptr<RemoteChannelBundle> bundle)
{
    NS_LOG_FUNCTION(this << bundle);

    Time delay(m_schedulerTune * bundle->GetDelay().GetTimeStep());

    bundle->SetEventId(Simulator::Schedule(delay,
                                           &NullMessageSimulatorImpl::NullMessageEventHandler,
                                           this,
                                           PeekPointer(bundle)));
}

void
NullMessageSimulatorImpl::RescheduleNullMessageEvent(Ptr<RemoteChannelBundle> bundle)
{
    NS_LOG_FUNCTION(this << bundle);

    Simulator::Cancel(bundle->GetEventId());

    Time delay(m_schedulerTune * bundle->GetDelay().GetTimeStep());

    bundle->SetEventId(Simulator::Schedule(delay,
                                           &NullMessageSimulatorImpl::NullMessageEventHandler,
                                           this,
                                           PeekPointer(bundle)));
}

void
NullMessageSimulatorImpl::RescheduleNullMessageEvent(uint32_t nodeSysId)
{
    NS_LOG_FUNCTION(this << nodeSysId);

    Ptr<RemoteChannelBundle> bundle = RemoteChannelBundleManager::Find(nodeSysId);
    NS_ASSERT(bundle);

    RescheduleNullMessageEvent(bundle);
}

void
NullMessageSimulatorImpl::Run()
{
    NS_LOG_FUNCTION(this);

    CalculateLookAhead();

    RemoteChannelBundleManager::InitializeNullMessageEvents();

    // Stop will be set if stop is called by simulation.
    m_stop = false;
    while (!IsFinished())
    {
        Time nextTime = Next();

        if (nextTime <= GetSafeTime())
        {
            ProcessOneEvent();
            HandleArrivingMessagesNonBlocking();
        }
        else
        {
            // Block until packet or Null Message has been received.
            HandleArrivingMessagesBlocking();
        }
    }
}

void
NullMessageSimulatorImpl::HandleArrivingMessagesNonBlocking()
{
    NS_LOG_FUNCTION(this);

    NullMessageMpiInterface::ReceiveMessagesNonBlocking();

    CalculateSafeTime();

    // Check for send completes
    NullMessageMpiInterface::TestSendComplete();
}

void
NullMessageSimulatorImpl::HandleArrivingMessagesBlocking()
{
    NS_LOG_FUNCTION(this);

    NullMessageMpiInterface::ReceiveMessagesBlocking();

    CalculateSafeTime();

    // Check for send completes
    NullMessageMpiInterface::TestSendComplete();
}

void
NullMessageSimulatorImpl::CalculateSafeTime()
{
    NS_LOG_FUNCTION(this);

    m_safeTime = RemoteChannelBundleManager::GetSafeTime();
    NS_ASSERT(m_safeTime >= Time(m_currentTs));
}

Time
NullMessageSimulatorImpl::GetSafeTime()
{
    return m_safeTime;
}

uint32_t
NullMessageSimulatorImpl::GetSystemId() const
{
    return m_myId;
}

void
NullMessageSimulatorImpl::Stop()
{
    NS_LOG_FUNCTION(this);

    m_stop = true;
}

EventId
NullMessageSimulatorImpl::Stop(const Time& delay)
{
    NS_LOG_FUNCTION(this << delay.GetTimeStep());

    return Simulator::Schedule(delay, &Simulator::Stop);
}

//
// Schedule an event for a _relative_ time in the future.
//
EventId
NullMessageSimulatorImpl::Schedule(const Time& delay, EventImpl* event)
{
    NS_LOG_FUNCTION(this << delay.GetTimeStep() << event);

    Time tAbsolute = delay + TimeStep(m_currentTs);

    NS_ASSERT(tAbsolute.IsPositive());
    NS_ASSERT(tAbsolute >= TimeStep(m_currentTs));
    Scheduler::Event ev;
    ev.impl = event;
    ev.key.m_ts = static_cast<uint64_t>(tAbsolute.GetTimeStep());
    ev.key.m_context = GetContext();
    ev.key.m_uid = m_uid;
    m_uid++;
    m_unscheduledEvents++;
    m_events->Insert(ev);
    return EventId(event, ev.key.m_ts, ev.key.m_context, ev.key.m_uid);
}

void
NullMessageSimulatorImpl::ScheduleWithContext(uint32_t context, const Time& delay, EventImpl* event)
{
    NS_LOG_FUNCTION(this << context << delay.GetTimeStep() << m_currentTs << event);

    Time tAbsolute(m_currentTs + delay.GetTimeStep());

    NS_ASSERT(tAbsolute.IsPositive());
    NS_ASSERT(tAbsolute >= TimeStep(m_currentTs));

    Scheduler::Event ev;
    ev.impl = event;
    ev.key.m_ts = tAbsolute.GetTimeStep();
    ev.key.m_context = context;
    ev.key.m_uid = m_uid;
    m_uid++;
    m_unscheduledEvents++;
    m_events->Insert(ev);
}

EventId
NullMessageSimulatorImpl::ScheduleNow(EventImpl* event)
{
    NS_LOG_FUNCTION(this << event);
    return Schedule(Time(0), event);
}

EventId
NullMessageSimulatorImpl::ScheduleDestroy(EventImpl* event)
{
    NS_LOG_FUNCTION(this << event);

    EventId id(Ptr<EventImpl>(event, false), m_currentTs, 0xffffffff, 2);
    m_destroyEvents.push_back(id);
    m_uid++;
    return id;
}

Time
NullMessageSimulatorImpl::Now() const
{
    return TimeStep(m_currentTs);
}

Time
NullMessageSimulatorImpl::GetDelayLeft(const EventId& id) const
{
    if (IsExpired(id))
    {
        return TimeStep(0);
    }
    else
    {
        return TimeStep(id.GetTs() - m_currentTs);
    }
}

void
NullMessageSimulatorImpl::Remove(const EventId& id)
{
    if (id.GetUid() == EventId::UID::DESTROY)
    {
        // destroy events.
        for (auto i = m_destroyEvents.begin(); i != m_destroyEvents.end(); i++)
        {
            if (*i == id)
            {
                m_destroyEvents.erase(i);
                break;
            }
        }
        return;
    }
    if (IsExpired(id))
    {
        return;
    }
    Scheduler::Event event;
    event.impl = id.PeekEventImpl();
    event.key.m_ts = id.GetTs();
    event.key.m_context = id.GetContext();
    event.key.m_uid = id.GetUid();
    m_events->Remove(event);
    event.impl->Cancel();
    // whenever we remove an event from the event list, we have to unref it.
    event.impl->Unref();

    m_unscheduledEvents--;
}

void
NullMessageSimulatorImpl::Cancel(const EventId& id)
{
    if (!IsExpired(id))
    {
        id.PeekEventImpl()->Cancel();
    }
}

bool
NullMessageSimulatorImpl::IsExpired(const EventId& id) const
{
    if (id.GetUid() == EventId::UID::DESTROY)
    {
        if (id.PeekEventImpl() == nullptr || id.PeekEventImpl()->IsCancelled())
        {
            return true;
        }
        // destroy events.
        for (auto i = m_destroyEvents.begin(); i != m_destroyEvents.end(); i++)
        {
            if (*i == id)
            {
                return false;
            }
        }
        return true;
    }
    return id.PeekEventImpl() == nullptr || id.GetTs() < m_currentTs ||
           (id.GetTs() == m_currentTs && id.GetUid() <= m_currentUid) ||
           id.PeekEventImpl()->IsCancelled();
}

Time
NullMessageSimulatorImpl::GetMaximumSimulationTime() const
{
    // XXX: I am fairly certain other compilers use other non-standard
    // post-fixes to indicate 64 bit constants.
    return TimeStep(0x7fffffffffffffffLL);
}

uint32_t
NullMessageSimulatorImpl::GetContext() const
{
    return m_currentContext;
}

uint64_t
NullMessageSimulatorImpl::GetEventCount() const
{
    return m_eventCount;
}

Time
NullMessageSimulatorImpl::CalculateGuaranteeTime(uint32_t nodeSysId)
{
    Ptr<RemoteChannelBundle> bundle = RemoteChannelBundleManager::Find(nodeSysId);
    NS_ASSERT(bundle);

    return Min(NullMessageSimulatorImpl::GetInstance()->Next(), GetSafeTime()) + bundle->GetDelay();
}

void
NullMessageSimulatorImpl::NullMessageEventHandler(RemoteChannelBundle* bundle)
{
    NS_LOG_FUNCTION(this << bundle);

    Time time = Min(Next(), GetSafeTime()) + bundle->GetDelay();
    NullMessageMpiInterface::SendNullMessage(time, bundle);

    ScheduleNullMessageEvent(bundle);
}

NullMessageSimulatorImpl*
NullMessageSimulatorImpl::GetInstance()
{
    NS_ASSERT(g_instance != nullptr);
    return g_instance;
}
} // namespace ns3
