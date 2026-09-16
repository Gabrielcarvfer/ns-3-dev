/*
 * Copyright (c) 2026 Universidade de Brasilia
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

/**
 * @file
 * @ingroup mtp
 * Declaration of class ns3::MtpScheduler
 */

#ifndef MTP_SCHEDULER_H
#define MTP_SCHEDULER_H

#include "ns3/scheduler.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <unordered_map>

namespace ns3
{

/**
 * @ingroup mtp
 * @brief One ancestor of an event, i.e., an event during whose execution the
 * event (or one of its ancestors) was scheduled.
 *
 * The sequential simulator orders events with the same timestamp by the order in
 * which they were scheduled, which is fully determined by the execution order of
 * their scheduling events. Recording a few generations of ancestors lets every
 * logical process reproduce that order for events coming from other LPs.
 */
struct EventAncestor
{
    uint64_t ts{0};        ///< time at which the ancestor was executed
    uint32_t systemId{0};  ///< LP that executed the ancestor
    uint32_t seq{0};       ///< execution sequence number of the ancestor in that LP
    uint32_t originUid{0}; ///< scheduling order of the ancestor among its siblings
};

/// Number of generations of ancestors recorded for each event
constexpr std::size_t MTP_LINEAGE_DEPTH = 8;

/// The ancestors of an event, from parent to older generations
using EventLineage = std::array<EventAncestor, MTP_LINEAGE_DEPTH>;

/**
 * @ingroup mtp
 * @brief An event of a logical process, with the information needed to
 * reproduce the ordering of the sequential simulator.
 */
struct MtpEvent
{
    Scheduler::Event event; ///< the event and its key
    uint32_t originUid{0};  ///< scheduling order of the event among its siblings
    EventLineage lineage;   ///< the ancestors of the event
};

/**
 * @ingroup mtp
 * @brief Compare two events of a logical process.
 *
 * Events are ordered by timestamp first. Events with the same timestamp are
 * ordered as the sequential simulator would order them: by the execution order
 * of their scheduling events, which is compared generation by generation until
 * two ancestors executed by the same LP (whose order is known) are found.
 *
 * @param a the first event
 * @param b the second event
 * @return true if a must be executed before b
 */
bool MtpEventLess(const MtpEvent& a, const MtpEvent& b);

/**
 * @ingroup mtp
 * @brief The future event list of a logical process.
 *
 * Unlike the schedulers of the core module, events are ordered by MtpEventLess
 * rather than by (timestamp, uid), and they can be looked up by uid.
 */
class MtpScheduler
{
  public:
    /// Comparator of the event set
    struct Less
    {
        /**
         * @param a the first event
         * @param b the second event
         * @return true if a must be executed before b
         */
        bool operator()(const MtpEvent& a, const MtpEvent& b) const
        {
            return MtpEventLess(a, b);
        }
    };

    MtpScheduler() = default;
    ~MtpScheduler();
    /**
     * Copy constructor.
     * @param o the scheduler to copy
     */
    MtpScheduler(const MtpScheduler& o);
    /**
     * Copy assignment.
     * @param o the scheduler to copy
     * @return this scheduler
     */
    MtpScheduler& operator=(const MtpScheduler& o);

    /**
     * @brief Insert an event.
     * @param ev the event
     */
    void Insert(const MtpEvent& ev);
    /**
     * @return true if there are no pending events
     */
    bool IsEmpty() const;
    /**
     * @return the next event, which must exist
     */
    const MtpEvent& PeekNext() const;
    /**
     * @brief Remove and return the next event, which must exist.
     * @return the next event
     */
    MtpEvent RemoveNext();
    /**
     * @brief Remove the event with the given uid, if pending.
     * @param uid the event uid
     * @param removed set to the removed event, if any
     * @return true if the event was pending and has been removed
     */
    bool Remove(uint32_t uid, MtpEvent& removed);
    /**
     * @param uid the event uid
     * @return true if the event with the given uid is pending
     */
    bool Contains(uint32_t uid) const;

  private:
    std::set<MtpEvent, Less> m_events;                                        ///< pending events
    std::unordered_map<uint32_t, std::set<MtpEvent, Less>::iterator> m_byUid; ///< uid index
};

} // namespace ns3

#endif /* MTP_SCHEDULER_H */
