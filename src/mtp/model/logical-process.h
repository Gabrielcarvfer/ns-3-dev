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
 *  Declaration of classes ns3::LogicalProcess
 */

#ifndef LOGICAL_PROCESS_H
#define LOGICAL_PROCESS_H

#include "mtp-scheduler.h"

#include "ns3/event-id.h"
#include "ns3/event-impl.h"
#include "ns3/node.h"
#include "ns3/nstime.h"
#include "ns3/object-factory.h"
#include "ns3/ptr.h"
#include "ns3/scheduler.h"

#include <atomic>
#include <chrono>
#include <map>
#include <vector>

namespace ns3
{

/**
 * @brief
 * Implementation of the logical process (LP) used by the multhreaded simulator.
 */
class LogicalProcess
{
  public:
    /** Default constructor */
    LogicalProcess();

    /** Destructor */
    ~LogicalProcess();

    /**
     * Enable this logical process object by giving it a unique systemId,
     * and let it know the total number of systems.
     *
     * @param systemId
     * @param systemCount
     */
    void Enable(const uint32_t systemId, const uint32_t systemCount);

    /**
     * @brief Calculate the lookahead value.
     */
    void CalculateLookAhead();

    /**
     * @brief Receive events sent by other logical processes in the previous round.
     */
    void ReceiveMessages();

    /**
     * @brief Process all events in the current round.
     */
    void ProcessOneRound();

    /**
     * @brief Process the events of the public LP that precede the next event
     * of every other LP.
     */
    void ProcessPublicEvents();

    /**
     * @return The next event, or nullptr if there is none or the LP is stopped
     */
    const MtpEvent* PeekNextEvent() const;

    /**
     * @brief Check and clear the flag recording that this LP sent events to other LPs.
     * @return true if events were sent since the last call
     */
    bool TakeSentFlag();

    /**
     * @brief Get the execution time of the last round.
     *
     * This method is called by MtpInterfaceused to determine the priority of each LP.
     *
     * @return The execution time of the last round
     */
    inline uint64_t GetExecutionTime() const
    {
        return m_executionTime;
    }

    /**
     * @brief Get the pending event count of the next round.
     *
     * This method is called by MtpInterfaceused to determine the priority of each LP.
     *
     * @return Number of pending events of the next round
     */
    inline uint64_t GetPendingEventCount() const
    {
        return m_pendingEventCount;
    }

    /**
     * @brief Move all the pending events of this LP to the given container.
     *
     * Used by the automatic partition to transfer the events scheduled before
     * Simulator::Run() from the public LP to the newly created LPs.
     *
     * @param events The container receiving the events, in scheduling order
     */
    void TakePendingEvents(std::vector<Scheduler::Event>& events);

    /**
     * @brief Insert an event scheduled before the partition of the simulation.
     *
     * The uid of the event is preserved, so that the EventId returned to the user
     * remains valid, and its ordering with respect to the other initial events is
     * preserved as well.
     *
     * @param ev The event
     */
    void ScheduleInitial(const Scheduler::Event& ev);

    /**
     * @brief Make sure the uids of newly scheduled events do not collide with
     * the given uid or any smaller one.
     *
     * @param uid The uid
     */
    void ReserveUidsUpTo(uint32_t uid);

    /**
     * @return The uid that the next scheduled event will receive
     */
    inline uint32_t GetNextUid() const
    {
        return m_uid;
    }

    /**
     * @brief Invoke an event immediately at the current time.
     *
     * This method is called when another thread wants to process an event of an LP
     * that does not belongs to it. It is used at the very beginning of the simulation
     * when the main thread will invoke events of newly allocated LP, whose timestamps
     * are zero.
     *
     * @param ev The event to be invoked now
     */
    void InvokeNow(const Scheduler::Event& ev);

    // The following methods are mapped from MultithreadedSimulatorImpl
    EventId Schedule(const Time& delay, EventImpl* event);
    void ScheduleAt(const uint32_t context, const Time& time, EventImpl* event);
    void ScheduleWithContext(LogicalProcess* remote,
                             const uint32_t context,
                             const Time& delay,
                             EventImpl* event);
    void Remove(const EventId& id);
    void Cancel(const EventId& id);
    bool IsExpired(const EventId& id) const;
    void SetScheduler(ObjectFactory schedulerFactory);
    Time Next() const;

    inline bool isLocalFinished() const
    {
        return m_stop || m_events.IsEmpty();
    }

    inline void Stop()
    {
        m_stop = true;
    }

    inline Time Now() const
    {
        return TimeStep(m_currentTs);
    }

    inline Time GetDelayLeft(const EventId& id) const
    {
        return TimeStep(id.GetTs() - m_currentTs);
    }

    inline uint32_t GetSystemId(void) const
    {
        return m_systemId;
    }

    inline uint32_t GetContext() const
    {
        return m_currentContext;
    }

    inline uint64_t GetEventCount() const
    {
        return m_eventCount;
    }

    /**
     * @brief Get the local (per-process) system ID of a node.
     * @param node the node
     * @return the local system ID
     */
    static uint32_t GetLocalSystemId(Ptr<Node> node);

  private:
    /// An event sent by another LP, waiting to be received
    struct Message
    {
        Scheduler::Event event; ///< the event
        uint32_t originUid;     ///< scheduling order of the event in the sender
        EventLineage lineage;   ///< the ancestors of the event
    };

    /**
     * @brief Execute an event, updating the current time, context and lineage.
     * @param ev the event
     */
    void Invoke(const MtpEvent& ev);

    uint32_t m_systemId;
    uint32_t m_systemCount;
    bool m_stop;
    uint32_t m_uid;
    uint32_t m_currentContext;
    uint64_t m_currentTs;
    uint64_t m_eventCount;
    uint64_t m_pendingEventCount;
    MtpScheduler m_events;
    Time m_lookAhead;
    uint64_t m_grantedTs;          //!< upper bound of the time window of the current round
    uint32_t m_execSeq;            //!< execution sequence number of the current event
    bool m_sent;                   //!< whether events were sent to other LPs
    EventLineage m_currentLineage; //!< the current event followed by its ancestors

    std::map<uint32_t, std::vector<Message>> m_mailbox; //!< event message mail box, per sender
    std::chrono::nanoseconds::rep m_executionTime;
};

} // namespace ns3

#endif /* LOGICAL_PROCESS_H */
