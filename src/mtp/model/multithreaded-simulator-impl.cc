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
 *  Implementation of classes ns3::MultithreadedSimulatorImpl
 */

#include "multithreaded-simulator-impl.h"

#include "mtp-interface.h"

#include "ns3/channel.h"
#include "ns3/node-container.h"
#include "ns3/node-list.h"
#include "ns3/node.h"
#include "ns3/simulator.h"
#include "ns3/type-id.h"
#include "ns3/uinteger.h"

#include <algorithm>
#include <queue>
#include <thread>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("MultithreadedSimulatorImpl");

NS_OBJECT_ENSURE_REGISTERED(MultithreadedSimulatorImpl);

MultithreadedSimulatorImpl::MultithreadedSimulatorImpl()
{
    NS_LOG_FUNCTION(this);
    if (!MtpInterface::isPartitioned())
    {
        MtpInterface::Enable(1, 0);
        m_partition = true;
    }
    else
    {
        m_partition = false;
    }
}

MultithreadedSimulatorImpl::~MultithreadedSimulatorImpl()
{
    NS_LOG_FUNCTION(this);
}

TypeId
MultithreadedSimulatorImpl::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::MultithreadedSimulatorImpl")
            .SetParent<SimulatorImpl>()
            .SetGroupName("Mtp")
            .AddConstructor<MultithreadedSimulatorImpl>()
            .AddAttribute("MaxThreads",
                          "The maximum threads used in simulation",
                          UintegerValue(std::thread::hardware_concurrency()),
                          MakeUintegerAccessor(&MultithreadedSimulatorImpl::m_maxThreads),
                          MakeUintegerChecker<uint32_t>(1))
            .AddAttribute("MinLookahead",
                          "The minimum lookahead in a partition",
                          TimeValue(TimeStep(0)),
                          MakeTimeAccessor(&MultithreadedSimulatorImpl::m_minLookahead),
                          MakeTimeChecker(TimeStep(0)));
    return tid;
}

void
MultithreadedSimulatorImpl::Destroy()
{
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
    MtpInterface::Disable();
}

bool
MultithreadedSimulatorImpl::IsFinished() const
{
    return MtpInterface::isFinished();
}

void
MultithreadedSimulatorImpl::Stop()
{
    NS_LOG_FUNCTION(this);
    for (uint32_t i = 0; i < MtpInterface::GetSize(); i++)
    {
        MtpInterface::GetSystem(i)->Stop();
    }
}

EventId
MultithreadedSimulatorImpl::Stop(const Time& delay)
{
    NS_LOG_FUNCTION(this << delay.GetTimeStep());
    return Simulator::Schedule(delay, &Simulator::Stop);
}

EventId
MultithreadedSimulatorImpl::Schedule(const Time& delay, EventImpl* event)
{
    NS_LOG_FUNCTION(this << delay.GetTimeStep() << event);
    return MtpInterface::GetSystem()->Schedule(delay, event);
}

void
MultithreadedSimulatorImpl::ScheduleWithContext(uint32_t context,
                                                const Time& delay,
                                                EventImpl* event)
{
    NS_LOG_FUNCTION(this << context << delay.GetTimeStep() << event);
    LogicalProcess* remote =
        context == Simulator::NO_CONTEXT
            ? MtpInterface::GetSystem(0)
            : MtpInterface::GetSystem(NodeList::GetNode(context)->GetSystemId());
    MtpInterface::GetSystem()->ScheduleWithContext(remote, context, delay, event);
}

EventId
MultithreadedSimulatorImpl::ScheduleNow(EventImpl* event)
{
    return Schedule(TimeStep(0), event);
}

EventId
MultithreadedSimulatorImpl::ScheduleDestroy(EventImpl* event)
{
    EventId id(Ptr<EventImpl>(event, false),
               GetMaximumSimulationTime().GetTimeStep(),
               0xffffffff,
               EventId::DESTROY);
    MtpInterface::CriticalSection cs;
    m_destroyEvents.push_back(id);
    return id;
}

void
MultithreadedSimulatorImpl::Remove(const EventId& id)
{
    if (id.GetUid() == EventId::DESTROY)
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
    }
    else
    {
        MtpInterface::GetSystem()->Remove(id);
    }
}

void
MultithreadedSimulatorImpl::Cancel(const EventId& id)
{
    if (!IsExpired(id))
    {
        id.PeekEventImpl()->Cancel();
    }
}

bool
MultithreadedSimulatorImpl::IsExpired(const EventId& id) const
{
    if (id.GetUid() == EventId::DESTROY)
    {
        // destroy events.
        if (id.PeekEventImpl() == nullptr || id.PeekEventImpl()->IsCancelled())
        {
            return true;
        }
        for (auto i = m_destroyEvents.begin(); i != m_destroyEvents.end(); i++)
        {
            if (*i == id)
            {
                return false;
            }
        }
        return true;
    }
    else
    {
        return MtpInterface::GetSystem()->IsExpired(id);
    }
}

void
MultithreadedSimulatorImpl::Run()
{
    NS_LOG_FUNCTION(this);
    // auto partition
    if (m_partition)
    {
        Partition();
    }
    MtpInterface::Run();
}

Time
MultithreadedSimulatorImpl::Now() const
{
    // Do not add function logging here, to avoid stack overflow
    return MtpInterface::GetSystem()->Now();
}

Time
MultithreadedSimulatorImpl::GetDelayLeft(const EventId& id) const
{
    if (IsExpired(id))
    {
        return TimeStep(0);
    }
    else
    {
        return MtpInterface::GetSystem()->GetDelayLeft(id);
    }
}

Time
MultithreadedSimulatorImpl::GetMaximumSimulationTime() const
{
    return Time::Max() / 2;
}

void
MultithreadedSimulatorImpl::SetScheduler(ObjectFactory schedulerFactory)
{
    NS_LOG_FUNCTION(this << schedulerFactory);
    // Logical processes use their own event list, which reproduces the event
    // ordering of the sequential simulator; the requested scheduler is ignored.
}

uint32_t
MultithreadedSimulatorImpl::GetSystemId() const
{
    return MtpInterface::GetSystem()->GetSystemId();
}

uint32_t
MultithreadedSimulatorImpl::GetContext() const
{
    return MtpInterface::GetSystem()->GetContext();
}

uint64_t
MultithreadedSimulatorImpl::GetEventCount() const
{
    uint64_t eventCount = 0;
    for (uint32_t i = 0; i < MtpInterface::GetSize(); i++)
    {
        eventCount += MtpInterface::GetSystem(i)->GetEventCount();
    }
    return eventCount;
}

void
MultithreadedSimulatorImpl::DoDispose()
{
    SimulatorImpl::DoDispose();
}

void
MultithreadedSimulatorImpl::Partition()
{
    NS_LOG_FUNCTION(this);
    uint32_t systemId = 0;
    const NodeContainer nodes = NodeContainer::GetGlobal();
    std::vector<bool> visited(nodes.GetN(), false);
    std::queue<Ptr<Node>> q;

    // Only point-to-point links are cut when partitioning. Shared-medium channels
    // (CSMA, Wi-Fi, spectrum, ...) keep all attached nodes in the same LP, since
    // their models access the state of every device on the channel directly.
    // If m_minLookahead is not set, use the median of the delay of every p2p link.
    if (m_minLookahead == TimeStep(0))
    {
        std::vector<Time> delays;
        for (auto it = nodes.Begin(); it != nodes.End(); it++)
        {
            Ptr<Node> node = *it;
            for (uint32_t i = 0; i < node->GetNDevices(); i++)
            {
                Ptr<NetDevice> localNetDevice = node->GetDevice(i);
                Ptr<Channel> channel = localNetDevice->GetChannel();
                if (!channel || !localNetDevice->IsPointToPoint())
                {
                    continue;
                }
                TimeValue delay;
                channel->GetAttribute("Delay", delay);
                delays.push_back(delay.Get());
            }
        }
        std::sort(delays.begin(), delays.end());
        if (delays.empty())
        {
            m_minLookahead = TimeStep(0);
        }
        else if (delays.size() % 2 == 1)
        {
            m_minLookahead = delays[delays.size() / 2];
        }
        else
        {
            m_minLookahead = (delays[delays.size() / 2 - 1] + delays[delays.size() / 2]) / 2;
        }
        NS_LOG_INFO("Min lookahead is set to " << m_minLookahead);
    }

    // perform a BFS on the whole network topo to assign each node a systemId
    for (auto it = nodes.Begin(); it != nodes.End(); it++)
    {
        Ptr<Node> node = *it;
        if (visited[node->GetId()])
        {
            continue;
        }
        q.push(node);
        visited[node->GetId()] = true;
        systemId++;
        while (!q.empty())
        {
            node = q.front();
            q.pop();
            node->SetSystemId(systemId);
            NS_LOG_INFO("node " << node->GetId() << " is set to system " << systemId);

            for (uint32_t i = 0; i < node->GetNDevices(); i++)
            {
                Ptr<NetDevice> localNetDevice = node->GetDevice(i);
                Ptr<Channel> channel = localNetDevice->GetChannel();
                if (!channel)
                {
                    continue;
                }
                // cut-off p2p links whose delay is not below the threshold
                if (localNetDevice->IsPointToPoint())
                {
                    TimeValue delay;
                    channel->GetAttribute("Delay", delay);
                    if (delay.Get() >= m_minLookahead && delay.Get() > TimeStep(0))
                    {
                        continue;
                    }
                }
                // grab the adjacent nodes
                for (std::size_t j = 0; j < channel->GetNDevices(); j++)
                {
                    Ptr<Node> remote = channel->GetDevice(j)->GetNode();
                    if (!visited[remote->GetId()])
                    {
                        visited[remote->GetId()] = true;
                        q.push(remote);
                    }
                }
            }
        }
    }

    // after the partition, we finally know the system count (# of LPs)
    const uint32_t systemCount = systemId;
    const uint32_t threadCount = std::min(m_maxThreads, systemCount);
    NS_LOG_INFO("Partition done! " << systemCount << " systems share " << threadCount
                                   << " threads");

    // create new LPs
    MtpInterface::EnableNew(threadCount, systemCount);

    MtpInterface::TransferInitialEvents();
}

} // namespace ns3
