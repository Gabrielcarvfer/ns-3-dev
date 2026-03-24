/*
 * Copyright (c) 2005 INRIA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "event-impl.h"

#include "log.h"

#include <cstdlib>
#include <new>

/**
 * @file
 * @ingroup events
 * ns3::EventImpl definitions.
 */

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("EventImpl");

namespace
{

// Size-bucketed free lists for EventImpl-derived objects (see the
// operator new/delete documentation in event-impl.h).
//
// Layout: freed blocks are reused as singly-linked list nodes (the first
// pointer-sized bytes of a dead block store the "next" pointer), so the
// pool needs no side storage. Buckets are multiples of 16 bytes; the
// MakeEvent template family (callback target + a few bound arguments)
// falls almost entirely inside the largest bucket. Blocks never return
// to the system before process exit: the pool's high-water mark equals
// the peak number of in-flight events of each size class, which for
// ns-3 simulations is bounded by the event queue length.
//
// thread_local rather than static: DistributedSimulator/multi-threaded
// helpers may run independent Simulators on different threads; per-thread
// pools keep this lock-free without atomics on the hot path.
constexpr std::size_t kBucketStep = 16;  // bucket granularity (bytes)
constexpr std::size_t kNumBuckets = 8;   // largest pooled size: 128 bytes
thread_local void* g_eventFreeList[kNumBuckets] = {};

inline std::size_t
BucketIndex(std::size_t size)
{
    // Round up to the bucket that holds `size` bytes. Bucket i serves
    // sizes ((i) * 16, (i+1) * 16]; index 0 serves 1..16.
    return (size + kBucketStep - 1) / kBucketStep - 1;
}

} // namespace

void*
EventImpl::operator new(std::size_t size)
{
    const std::size_t bucket = BucketIndex(size);
    if (bucket < kNumBuckets)
    {
        if (void* block = g_eventFreeList[bucket])
        {
            // Pop the head of this size class's free list: the dead block's
            // first bytes hold the next-pointer (written in operator delete).
            g_eventFreeList[bucket] = *static_cast<void**>(block);
            return block;
        }
        // Empty free list: allocate the FULL bucket size (not `size`) so the
        // block is reusable by any type in the same size class later.
        return ::operator new((bucket + 1) * kBucketStep);
    }
    // Oversized event object (deeply bound arguments): plain allocation.
    return ::operator new(size);
}

void
EventImpl::operator delete(void* ptr, std::size_t size)
{
    const std::size_t bucket = BucketIndex(size);
    if (bucket < kNumBuckets)
    {
        // Push onto the free list: reuse the dead block's storage for the
        // next-pointer. The object was already destroyed; the memory is ours.
        *static_cast<void**>(ptr) = g_eventFreeList[bucket];
        g_eventFreeList[bucket] = ptr;
        return;
    }
    ::operator delete(ptr);
}

void
EventImpl::operator delete(void* ptr)
{
    // Unsized fallback (see header): all pool blocks come from
    // ::operator new, so releasing without re-pooling is always valid.
    ::operator delete(ptr);
}

EventImpl::~EventImpl()
{
    NS_LOG_FUNCTION(this);
}

EventImpl::EventImpl()
    : m_cancel(false)
{
    NS_LOG_FUNCTION(this);
}

void
EventImpl::Invoke()
{
    NS_LOG_FUNCTION(this);
    if (!m_cancel)
    {
        Notify();
    }
}

void
EventImpl::Cancel()
{
    NS_LOG_FUNCTION(this);
    m_cancel = true;
}

bool
EventImpl::IsCancelled()
{
    NS_LOG_FUNCTION(this);
    return m_cancel;
}

} // namespace ns3
