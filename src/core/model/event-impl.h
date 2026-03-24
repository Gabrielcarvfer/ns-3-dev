/*
 * Copyright (c) 2005,2006 INRIA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#ifndef EVENT_IMPL_H
#define EVENT_IMPL_H

#include "simple-ref-count.h"

#include <cstddef>
#include <stdint.h>

/**
 * @file
 * @ingroup events
 * ns3::EventImpl declarations.
 */

namespace ns3
{

/**
 * @ingroup events
 * @brief A simulation event.
 *
 * Each subclass of this base class represents a simulation event. The
 * Invoke() method will be called by the simulation engine
 * when it reaches the time associated to this event. Most subclasses
 * are usually created by one of the many Simulator::Schedule
 * methods.
 */
class EventImpl : public SimpleRefCount<EventImpl>
{
  public:
    /**
     * Pooled allocation for event objects.
     *
     * Every Simulator::Schedule call heap-allocates one EventImpl subclass
     * object (created by MakeEvent with the bound callback arguments) and
     * frees it after Invoke(). Large simulations schedule millions of events
     * per wall-second, so the malloc/free pair per event is a measurable
     * share of the event-processing cost. These class-level operators recycle
     * freed event blocks through small size-bucketed free lists: a freed
     * block is pushed onto the bucket for its size class and reused by the
     * next allocation that fits the same class, falling back to the global
     * operator new/delete for sizes above the largest bucket.
     *
     * The pool is intentionally simple: ns-3 simulations are single-threaded
     * per Simulator, the blocks never shrink (peak in-flight events bound the
     * pool size), and the size classes cover the MakeEvent template family
     * (callback pointer + a handful of bound arguments).
     *
     * @param size the size of the event subclass object in bytes
     * @return pointer to a recycled or freshly allocated block
     */
    static void* operator new(std::size_t size);
    /**
     * Return an event block to the size-bucketed pool (or to the global
     * operator delete when it was allocated above the largest bucket).
     * @param ptr the block to free
     * @param size the size of the event subclass object in bytes
     */
    static void operator delete(void* ptr, std::size_t size);
    /**
     * Unsized fallback (used e.g. when a subclass constructor throws).
     * Bypasses the pool: without the size the bucket is unknown, and
     * releasing a pool-eligible block straight to the global delete is
     * always valid because bucket blocks originate from ::operator new.
     * @param ptr the block to free
     */
    static void operator delete(void* ptr);

    /** Default constructor. */
    EventImpl();
    /** Destructor. */
    virtual ~EventImpl() = 0;
    /**
     * Called by the simulation engine to notify the event that it is time
     * to execute.
     */
    void Invoke();
    /**
     * Marks the event as 'canceled'. The event is not removed from
     * the event list but the simulation engine will check its canceled status
     * before calling Invoke().
     */
    void Cancel();
    /**
     * @returns true if the event has been canceled.
     *
     * Checked by the simulation engine before calling Invoke().
     */
    bool IsCancelled();

  protected:
    /**
     * Implementation for Invoke().
     *
     * This typically calls a method or function pointer with the
     * arguments bound by a call to one of the MakeEvent() functions.
     */
    virtual void Notify() = 0;

  private:
    bool m_cancel; /**< Has this event been cancelled. */
};

} // namespace ns3

#endif /* EVENT_IMPL_H */
