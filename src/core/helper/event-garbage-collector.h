/*
 * Copyright (c) 2007 INESC Porto
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gustavo J. A. M. Carneiro  <gjc@inescporto.pt>
 */
#ifndef EVENT_GARBAGE_COLLECTOR_H
#define EVENT_GARBAGE_COLLECTOR_H

#include "ns3/event-id.h"
#include "ns3/simulator.h"

#include <set>

/**
 * @file
 * @ingroup events
 * @ingroup core-helpers
 * ns3::EventGarbageCollector declaration.
 */

namespace ns3
{

/**
 * @ingroup events
 * @ingroup core-helpers
 *
 * @brief An object that tracks scheduled events and automatically
 * cancels them when it is destroyed.  It is useful in situations
 * where multiple instances of the same type of event can
 * simultaneously be scheduled, and when the events should be limited
 * to the lifetime of a container object.
 */
class EventGarbageCollector
{
  public:
    EventGarbageCollector();

    /**
     * @brief Tracks a new event
     * @param [in] event the Event to track
     */
    void Track(EventId event);

    ~EventGarbageCollector();

  private:
    /** Event list container */
    typedef std::multiset<EventId> EventList;

    /** Initial threshold for cleaning the event list. */
    const typename EventList::size_type CHUNK_INIT_SIZE = 8;
    /**
     * Threshold to switch from exponential to linear growth
     * in the cleanup frequency.
     */
    const typename EventList::size_type CHUNK_MAX_SIZE = 128;

    EventList::size_type m_nextCleanupSize; //!< Batch size for cleanup
    EventList m_events;                     //!< The tracked event list

    /**
     * @brief Called when a new event was added and the cleanup limit was
     * exceeded in consequence.
     */
    void Cleanup();
    /**
     * @brief Grow the cleanup limit.
     * Increase the cleanup size by the smaller of
     * the current cleanup size (exponential growth),
     * or the CHUNK_MAX_SIZE (linear growth).
     */
    void Grow();
    /**
     * @brief Shrink the cleanup limit
     * Reduce the cleanup size by factors of two until less than the
     * current event list, then Grow one step.
     */
    void Shrink();
};

} // namespace ns3

#endif /* EVENT_GARBAGE_COLLECTOR_H */
