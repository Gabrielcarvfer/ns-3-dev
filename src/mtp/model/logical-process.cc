/*
 * Copyright (c) 2023 State Key Laboratory for Novel Software Technology
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 *
 *
 * Author: Songyuan Bai <i@f5soft.site>
 */

/**
 * @file
 * @ingroup mtp
 *  Implementation of classes ns3::LogicalProcess
 */

#include "logical-process.h"

#include "mtp-interface.h"

#include "ns3/channel.h"
#include "ns3/node-container.h"
#include "ns3/simulator.h"

#include <algorithm>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("LogicalProcess");

LogicalProcess::LogicalProcess()
    : m_systemId(0),
      m_systemCount(0),
      m_stop(false),
      m_uid(EventId::UID::VALID),
      m_currentContext(Simulator::NO_CONTEXT),
      m_currentTs(0),
      m_eventCount(0),
      m_pendingEventCount(0),
      m_lookAhead(TimeStep(0)),
      m_grantedTs(0),
      m_execSeq(0),
      m_sent(false),
      m_executionTime(0)
{
}

LogicalProcess::~LogicalProcess()
{
    NS_LOG_INFO("system " << m_systemId << " finished with event count " << m_eventCount);
}

void
LogicalProcess::Enable(const uint32_t systemId, const uint32_t systemCount)
{
    m_systemId = systemId;
    m_systemCount = systemCount;
}

void
LogicalProcess::CalculateLookAhead()
{
    NS_LOG_FUNCTION(this);

    if (m_systemId == 0)
    {
        m_lookAhead = TimeStep(0); // No lookahead for the public LP
        for (uint32_t i = 1; i < m_systemCount; i++)
        {
            m_mailbox[i];
        }
    }
    else
    {
        m_lookAhead = Time::Max() / 2 - TimeStep(1);
        NodeContainer c = NodeContainer::GetGlobal();
        for (auto iter = c.Begin(); iter != c.End(); ++iter)
        {
            if (GetLocalSystemId(*iter) != m_systemId)
            {
                continue;
            }
            for (uint32_t i = 0; i < (*iter)->GetNDevices(); ++i)
            {
                Ptr<NetDevice> localNetDevice = (*iter)->GetDevice(i);
                Ptr<Channel> channel = localNetDevice->GetChannel();
                if (!channel)
                {
                    continue;
                }
                // The lookahead over a link is the channel delay; channels without a
                // fixed delay (e.g. wireless) offer no lookahead when they are split
                // between LPs, which is only possible with a manual partition.
                TimeValue delay(TimeStep(0));
                if (!channel->GetAttributeFailSafe("Delay", delay))
                {
                    delay = TimeValue(TimeStep(0));
                }
                for (std::size_t j = 0; j < channel->GetNDevices(); ++j)
                {
                    Ptr<Node> remoteNode = channel->GetDevice(j)->GetNode();
                    if (GetLocalSystemId(remoteNode) == m_systemId)
                    {
                        continue;
                    }
                    if (!localNetDevice->IsPointToPoint())
                    {
                        NS_LOG_WARN("Channel " << channel->GetId() << " of node "
                                               << (*iter)->GetId()
                                               << " is shared between LPs; results may be "
                                                  "nondeterministic");
                    }
                    if (delay.Get() < m_lookAhead)
                    {
                        m_lookAhead = delay.Get();
                    }
                    // add the neighbour to the mailbox
                    m_mailbox[remoteNode->GetSystemId()];
                }
            }
        }
    }

    NS_LOG_INFO("lookahead of system " << m_systemId << " is set to " << m_lookAhead.GetTimeStep());
}

uint32_t
LogicalProcess::GetLocalSystemId(Ptr<Node> node)
{
#ifdef NS3_MPI
    // for hybrid simulation, the left 16-bit indicates local system ID,
    // and the right 16-bit indicates global system ID (MPI rank)
    return node->GetSystemId() >> 16;
#else
    return node->GetSystemId();
#endif
}

void
LogicalProcess::ReceiveMessages()
{
    NS_LOG_FUNCTION(this);

    m_pendingEventCount = 0;
    // Mailboxes are visited in sender order and each one is already in the
    // sender's scheduling order, so the uid assignment is deterministic.
    for (auto& item : m_mailbox)
    {
        auto& queue = item.second;
        for (auto& msg : queue)
        {
            MtpEvent ev;
            ev.event = msg.event;
            ev.event.key.m_uid = m_uid++;
            ev.originUid = msg.originUid;
            ev.lineage = msg.lineage;
            m_events.Insert(ev);
            m_pendingEventCount++;
        }
        queue.clear();
    }
}

void
LogicalProcess::ProcessOneRound()
{
    NS_LOG_FUNCTION(this);

    // set thread context
    MtpInterface::SetSystem(m_systemId);

    // calculate time window
    Time grantedTime = MtpInterface::GetSmallestTime() + m_lookAhead;
    // Events received from other LPs in the next rounds may have a timestamp
    // equal to the granted time, so it is excluded from this round unless
    // there is no lookahead at all (in which case no progress would be made).
    bool inclusive = m_lookAhead.IsZero();
    // Events of the public LP are processed once all the other LPs have reached
    // them: this LP stops at the first event that the next public event precedes.
    const MtpEvent* nextPublic = MtpInterface::GetNextPublicEvent();
    if (nextPublic && TimeStep(nextPublic->event.key.m_ts) <= grantedTime)
    {
        if (TimeStep(nextPublic->event.key.m_ts) == grantedTime && !inclusive)
        {
            nextPublic = nullptr;
        }
        else
        {
            grantedTime = TimeStep(nextPublic->event.key.m_ts);
            inclusive = false;
        }
    }
    m_grantedTs = grantedTime.GetTimeStep();

    auto start = std::chrono::system_clock::now();

    while (const MtpEvent* next = PeekNextEvent())
    {
        const Time ts = TimeStep(next->event.key.m_ts);
        if (ts > grantedTime)
        {
            break;
        }
        if (ts == grantedTime && !inclusive && !(nextPublic && MtpEventLess(*next, *nextPublic)))
        {
            break;
        }
        Invoke(m_events.RemoveNext());
    }

    auto end = std::chrono::system_clock::now();
    m_executionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

void
LogicalProcess::ProcessPublicEvents()
{
    NS_LOG_FUNCTION(this);
    NS_ASSERT(m_systemId == 0);

    MtpInterface::SetSystem(m_systemId);

    while (const MtpEvent* next = PeekNextEvent())
    {
        const MtpEvent* bound = MtpInterface::GetNextPrivateEvent();
        if (bound && !MtpEventLess(*next, *bound))
        {
            break;
        }
        m_grantedTs = next->event.key.m_ts;
        Invoke(m_events.RemoveNext());
        // Events sent to other LPs must be delivered before going on, since
        // they may precede the next events of the public LP.
        if (m_sent)
        {
            break;
        }
    }
}

bool
LogicalProcess::TakeSentFlag()
{
    const bool sent = m_sent;
    m_sent = false;
    return sent;
}

const MtpEvent*
LogicalProcess::PeekNextEvent() const
{
    if (m_stop || m_events.IsEmpty())
    {
        return nullptr;
    }
    return &m_events.PeekNext();
}

void
LogicalProcess::Invoke(const MtpEvent& ev)
{
    m_eventCount++;
    NS_LOG_LOGIC("handle " << ev.event.key.m_ts);

    m_currentTs = ev.event.key.m_ts;
    m_currentContext = ev.event.key.m_context;

    // The current event becomes the parent of the events it schedules
    for (std::size_t i = MTP_LINEAGE_DEPTH - 1; i > 0; i--)
    {
        m_currentLineage[i] = ev.lineage[i - 1];
    }
    m_currentLineage[0] = {m_currentTs, m_systemId, ++m_execSeq, ev.originUid};

    ev.event.impl->Invoke();
    ev.event.impl->Unref();
}

EventId
LogicalProcess::Schedule(const Time& delay, EventImpl* event)
{
    MtpEvent ev;

    ev.event.impl = event;
    ev.event.key.m_ts = m_currentTs + delay.GetTimeStep();
    ev.event.key.m_context = GetContext();
    ev.event.key.m_uid = m_uid++;
    ev.originUid = ev.event.key.m_uid;
    ev.lineage = m_currentLineage;
    m_events.Insert(ev);

    return EventId(event, ev.event.key.m_ts, ev.event.key.m_context, ev.event.key.m_uid);
}

void
LogicalProcess::ScheduleAt(const uint32_t context, const Time& time, EventImpl* event)
{
    MtpEvent ev;

    ev.event.impl = event;
    ev.event.key.m_ts = time.GetTimeStep();
    ev.event.key.m_context = context;
    ev.event.key.m_uid = m_uid++;
    ev.originUid = ev.event.key.m_uid;
    ev.lineage = m_currentLineage;
    m_events.Insert(ev);
}

void
LogicalProcess::ScheduleInitial(const Scheduler::Event& event)
{
    MtpEvent ev;

    ev.event = event;
    ev.originUid = event.key.m_uid;
    ev.lineage = EventLineage{};
    m_events.Insert(ev);
}

void
LogicalProcess::ReserveUidsUpTo(uint32_t uid)
{
    m_uid = std::max(m_uid, uid);
}

void
LogicalProcess::TakePendingEvents(std::vector<Scheduler::Event>& events)
{
    while (!m_events.IsEmpty())
    {
        events.push_back(m_events.RemoveNext().event);
    }
}

void
LogicalProcess::ScheduleWithContext(LogicalProcess* remote,
                                    const uint32_t context,
                                    const Time& delay,
                                    EventImpl* event)
{
    if (remote == this)
    {
        MtpEvent ev;
        ev.event.impl = event;
        ev.event.key.m_ts = delay.GetTimeStep() + m_currentTs;
        ev.event.key.m_context = context;
        ev.event.key.m_uid = m_uid++;
        ev.originUid = ev.event.key.m_uid;
        ev.lineage = m_currentLineage;
        m_events.Insert(ev);
        return;
    }

    Message msg;
    msg.event.impl = event;
    msg.event.key.m_ts = delay.GetTimeStep() + m_currentTs;
    msg.event.key.m_context = context;
    msg.event.key.m_uid = EventId::UID::INVALID;
    msg.originUid = m_uid++;
    msg.lineage = m_currentLineage;

    NS_ASSERT_MSG(msg.event.key.m_ts >= remote->m_grantedTs,
                  "Event scheduled at " << msg.event.key.m_ts << " from system " << m_systemId
                                        << " is in the past of system " << remote->m_systemId
                                        << " (already at " << remote->m_grantedTs
                                        << "); the lookahead is too large");
    // Each sender owns its queue in the mailbox of the remote LP, and queues are
    // registered up front, so no locking is needed.
    remote->m_mailbox[m_systemId].push_back(msg);
    m_sent = true;
}

void
LogicalProcess::InvokeNow(const Scheduler::Event& event)
{
    uint32_t oldSystemId = MtpInterface::GetSystem()->GetSystemId();
    MtpInterface::SetSystem(m_systemId);

    MtpEvent ev;
    ev.event = event;
    ev.originUid = event.key.m_uid;
    ev.lineage = EventLineage{};
    Invoke(ev);

    // restore previous thread context
    MtpInterface::SetSystem(oldSystemId);
}

void
LogicalProcess::Remove(const EventId& id)
{
    MtpEvent removed;
    if (!m_events.Remove(id.GetUid(), removed))
    {
        return;
    }
    removed.event.impl->Cancel();
    // whenever we remove an event from the event list, we have to unref it.
    removed.event.impl->Unref();
}

bool
LogicalProcess::IsExpired(const EventId& id) const
{
    if (id.PeekEventImpl() == nullptr || id.PeekEventImpl()->IsCancelled())
    {
        return true;
    }
    if (id.GetTs() < m_currentTs)
    {
        return true;
    }
    if (id.GetTs() > m_currentTs)
    {
        return false;
    }
    // Events with the current timestamp are not executed in uid order, so
    // check whether the event is still pending.
    return !m_events.Contains(id.GetUid());
}

void
LogicalProcess::SetScheduler(ObjectFactory schedulerFactory)
{
    NS_LOG_FUNCTION(this << schedulerFactory);
    // Logical processes use their own event list, which reproduces the event
    // ordering of the sequential simulator; the requested scheduler is ignored.
}

Time
LogicalProcess::Next() const
{
    if (m_stop || m_events.IsEmpty())
    {
        return Time::Max();
    }
    else
    {
        return TimeStep(m_events.PeekNext().event.key.m_ts);
    }
}

} // namespace ns3
