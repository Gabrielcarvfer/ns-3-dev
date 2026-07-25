/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "fq-codel-flat-queue-disc.h"

#include "ns3/log.h"
#include "ns3/net-device-queue-interface.h"
#include "ns3/queue-size.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/uinteger.h"

#include <algorithm>
#include <cmath>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("FqCoDelFlatQueueDisc");

NS_OBJECT_ENSURE_REGISTERED(FqCoDelFlatQueueDisc);

TypeId
FqCoDelFlatQueueDisc::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::FqCoDelFlatQueueDisc")
            .SetParent<QueueDisc>()
            .SetGroupName("TrafficControl")
            .AddConstructor<FqCoDelFlatQueueDisc>()
            .AddAttribute("MaxSize",
                          "The maximum number of packets accepted by this queue disc",
                          QueueSizeValue(QueueSize("10240p")),
                          MakeQueueSizeAccessor(&QueueDisc::SetMaxSize, &QueueDisc::GetMaxSize),
                          MakeQueueSizeChecker())
            .AddAttribute("Flows",
                          "The number of flow queues",
                          UintegerValue(1024),
                          MakeUintegerAccessor(&FqCoDelFlatQueueDisc::m_flows),
                          MakeUintegerChecker<uint32_t>(1))
            .AddAttribute("Quantum",
                          "The DRR quantum in bytes; 0 selects the device MTU",
                          UintegerValue(0),
                          MakeUintegerAccessor(&FqCoDelFlatQueueDisc::m_quantum),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("DropBatchSize",
                          "The maximum number of packets dropped from the fat flow",
                          UintegerValue(64),
                          MakeUintegerAccessor(&FqCoDelFlatQueueDisc::m_dropBatchSize),
                          MakeUintegerChecker<uint32_t>(1))
            .AddAttribute("Perturbation",
                          "The salt used as an additional input to the hash function",
                          UintegerValue(0),
                          MakeUintegerAccessor(&FqCoDelFlatQueueDisc::m_perturbation),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("Target",
                          "The CoDel target queue delay",
                          TimeValue(MilliSeconds(5)),
                          MakeTimeAccessor(&FqCoDelFlatQueueDisc::m_target),
                          MakeTimeChecker())
            .AddAttribute("Interval",
                          "The CoDel sliding-minimum window",
                          TimeValue(MilliSeconds(100)),
                          MakeTimeAccessor(&FqCoDelFlatQueueDisc::m_interval),
                          MakeTimeChecker());
    return tid;
}

FqCoDelFlatQueueDisc::FqCoDelFlatQueueDisc()
    : QueueDisc(QueueDiscSizePolicy::MULTIPLE_QUEUES)
{
    NS_LOG_FUNCTION(this);
}

FqCoDelFlatQueueDisc::~FqCoDelFlatQueueDisc()
{
    NS_LOG_FUNCTION(this);
}

bool
FqCoDelFlatQueueDisc::CheckConfig()
{
    NS_LOG_FUNCTION(this);
    if (GetNQueueDiscClasses() > 0)
    {
        NS_LOG_ERROR("FqCoDelFlatQueueDisc cannot have classes");
        return false;
    }
    if (GetNInternalQueues() > 0)
    {
        NS_LOG_ERROR("FqCoDelFlatQueueDisc cannot have internal queues");
        return false;
    }
    if (GetNPacketFilters() > 0)
    {
        NS_LOG_ERROR("FqCoDelFlatQueueDisc uses the packet hash, not packet filters");
        return false;
    }
    return true;
}

void
FqCoDelFlatQueueDisc::InitializeParams()
{
    NS_LOG_FUNCTION(this);
    m_flowsTable.resize(m_flows);
    if (!m_quantum)
    {
        Ptr<NetDeviceQueueInterface> ndqi = GetNetDeviceQueueInterface();
        Ptr<NetDevice> dev;
        if (ndqi && (dev = ndqi->GetObject<NetDevice>()))
        {
            m_quantum = dev->GetMtu();
        }
        if (!m_quantum)
        {
            m_quantum = 1514;
        }
        NS_LOG_DEBUG("Setting the quantum to " << m_quantum);
    }
}

bool
FqCoDelFlatQueueDisc::DoEnqueue(Ptr<QueueDiscItem> item)
{
    NS_LOG_FUNCTION(this << item);

    const uint32_t h = item->Hash(m_perturbation) % m_flows;
    Flow& flow = m_flowsTable[h];

    item->SetTimeStamp(Simulator::Now());
    flow.queue.push_back(item);
    flow.backlogBytes += item->GetSize();
    PacketEnqueued(item);

    if (flow.status == INACTIVE)
    {
        flow.status = NEW_FLOW;
        flow.deficit = m_quantum;
        m_newFlows.push_back(h);
    }

    if (GetCurrentSize() > GetMaxSize())
    {
        NS_LOG_DEBUG("Overload; dropping from the fat flow");
        DropOverlimit();
    }

    return true;
}

void
FqCoDelFlatQueueDisc::DropOverlimit()
{
    NS_LOG_FUNCTION(this);

    uint32_t maxBacklog = 0;
    uint32_t fat = 0;
    for (uint32_t i = 0; i < m_flows; i++)
    {
        if (m_flowsTable[i].backlogBytes > maxBacklog)
        {
            maxBacklog = m_flowsTable[i].backlogBytes;
            fat = i;
        }
    }

    Flow& flow = m_flowsTable[fat];
    const uint32_t threshold = maxBacklog >> 1;
    uint32_t len = 0;
    uint32_t count = 0;
    do
    {
        Ptr<QueueDiscItem> item = PopHead(flow);
        if (!item)
        {
            break;
        }
        len += item->GetSize();
        DropAfterDequeue(item, OVERLIMIT_DROP);
    } while (++count < m_dropBatchSize && len < threshold);
}

Ptr<QueueDiscItem>
FqCoDelFlatQueueDisc::PopHead(Flow& flow)
{
    if (flow.queue.empty())
    {
        return nullptr;
    }
    Ptr<QueueDiscItem> item = flow.queue.front();
    flow.queue.pop_front();
    flow.backlogBytes -= item->GetSize();
    PacketDequeued(item);
    return item;
}

Time
FqCoDelFlatQueueDisc::ControlLaw(Time t, uint32_t count) const
{
    return t + Seconds(m_interval.GetSeconds() / std::sqrt(count));
}

bool
FqCoDelFlatQueueDisc::OkToDrop(Flow& flow, Ptr<const QueueDiscItem> item, Time now)
{
    const Time sojourn = now - item->GetTimeStamp();
    if (sojourn < m_target || flow.backlogBytes <= m_quantum)
    {
        flow.firstAboveTime = Seconds(0);
        return false;
    }
    if (flow.firstAboveTime.IsZero())
    {
        flow.firstAboveTime = now + m_interval;
        return false;
    }
    return now >= flow.firstAboveTime;
}

Ptr<QueueDiscItem>
FqCoDelFlatQueueDisc::CoDelDequeue(Flow& flow)
{
    const Time now = Simulator::Now();
    Ptr<QueueDiscItem> item = PopHead(flow);
    if (!item)
    {
        flow.dropping = false;
        return nullptr;
    }

    bool okToDrop = OkToDrop(flow, item, now);
    if (flow.dropping)
    {
        if (!okToDrop)
        {
            flow.dropping = false;
        }
        else
        {
            while (flow.dropping && now >= flow.dropNext)
            {
                DropAfterDequeue(item, TARGET_EXCEEDED_DROP);
                flow.count++;
                item = PopHead(flow);
                if (!item)
                {
                    flow.dropping = false;
                    return nullptr;
                }
                if (!OkToDrop(flow, item, now))
                {
                    flow.dropping = false;
                }
                else
                {
                    flow.dropNext = ControlLaw(flow.dropNext, flow.count);
                }
            }
        }
    }
    else if (okToDrop)
    {
        DropAfterDequeue(item, TARGET_EXCEEDED_DROP);
        item = PopHead(flow);
        if (!item)
        {
            flow.dropping = false;
            return nullptr;
        }
        OkToDrop(flow, item, now);
        flow.dropping = true;
        // Restart the drop cycle, resuming the drop rate if we were
        // recently dropping.
        if (flow.count > 2 && now - flow.dropNext < 8 * m_interval)
        {
            flow.count = flow.count > flow.lastCount ? flow.count - flow.lastCount : 1;
        }
        else
        {
            flow.count = 1;
        }
        flow.lastCount = flow.count;
        flow.dropNext = ControlLaw(now, flow.count);
    }

    return item;
}

Ptr<QueueDiscItem>
FqCoDelFlatQueueDisc::DoDequeue()
{
    NS_LOG_FUNCTION(this);

    while (true)
    {
        uint32_t index;
        bool fromNew;
        if (!m_newFlows.empty())
        {
            index = m_newFlows.front();
            fromNew = true;
        }
        else if (!m_oldFlows.empty())
        {
            index = m_oldFlows.front();
            fromNew = false;
        }
        else
        {
            return nullptr;
        }

        Flow& flow = m_flowsTable[index];
        if (flow.deficit <= 0)
        {
            flow.deficit += m_quantum;
            if (fromNew)
            {
                m_newFlows.pop_front();
                flow.status = OLD_FLOW;
                m_oldFlows.push_back(index);
            }
            else
            {
                m_oldFlows.pop_front();
                m_oldFlows.push_back(index);
            }
            continue;
        }

        Ptr<QueueDiscItem> item = CoDelDequeue(flow);
        if (!item)
        {
            // The flow emptied out: a new flow becomes an old flow if other
            // new flows remain, otherwise it goes inactive.
            if (fromNew)
            {
                m_newFlows.pop_front();
                if (!m_newFlows.empty())
                {
                    flow.status = OLD_FLOW;
                    m_oldFlows.push_back(index);
                }
                else
                {
                    flow.status = INACTIVE;
                }
            }
            else
            {
                m_oldFlows.pop_front();
                flow.status = INACTIVE;
            }
            continue;
        }

        flow.deficit -= static_cast<int32_t>(item->GetSize());
        return item;
    }
}

} // namespace ns3
