/*
 * Copyright (c) 2006 INRIA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 * The idea to use a std c++ map came from GTNetS
 */

#include "map-scheduler.h"

#include "assert.h"
#include "log.h"

#include <string>

/**
 * @file
 * @ingroup scheduler
 * ns3::MapScheduler implementation.
 */

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("MapScheduler");

NS_OBJECT_ENSURE_REGISTERED(MapScheduler);

TypeId
MapScheduler::GetTypeId()
{
    static TypeId tid = TypeId("ns3::MapScheduler")
                            .SetParent<Scheduler>()
                            .SetGroupName("Core")
                            .AddConstructor<MapScheduler>();
    return tid;
}

MapScheduler::MapScheduler()
{
    NS_LOG_FUNCTION(this);
}

MapScheduler::~MapScheduler()
{
    NS_LOG_FUNCTION(this);
}

void
MapScheduler::Insert(const Event& ev)
{
    NS_LOG_FUNCTION(this << ev.impl << ev.key.m_ts << ev.key.m_uid);
    [[maybe_unused]] const auto sizeBefore = m_list.size();
    // How far ahead events are scheduled is model-dependent, but the key
    // orders by timestamp and then by a monotonically increasing uid, so a
    // newly scheduled event never sorts before one already at the same
    // timestamp. Hinting the insertion at the end therefore costs one
    // comparison when the hint is wrong, and makes the append case, which
    // covers every event scheduled at or after the latest one pending, O(1)
    // amortized instead of O(log n).
    //
    // emplace_hint does not report whether it inserted, so the duplicate-key
    // check the returned bool of insert() used to provide is expressed as a
    // size check instead.
    m_list.emplace_hint(m_list.end(), ev.key, ev.impl);
    NS_ASSERT(m_list.size() == sizeBefore + 1);
}

bool
MapScheduler::IsEmpty() const
{
    NS_LOG_FUNCTION(this);
    return m_list.empty();
}

Scheduler::Event
MapScheduler::PeekNext() const
{
    NS_LOG_FUNCTION(this);
    auto i = m_list.begin();
    NS_ASSERT(i != m_list.end());

    Event ev;
    ev.impl = i->second;
    ev.key = i->first;
    NS_LOG_DEBUG(this << ": " << ev.impl << ", " << ev.key.m_ts << ", " << ev.key.m_uid);
    return ev;
}

Scheduler::Event
MapScheduler::RemoveNext()
{
    NS_LOG_FUNCTION(this);
    auto i = m_list.begin();
    NS_ASSERT(i != m_list.end());
    Event ev;
    ev.impl = i->second;
    ev.key = i->first;
    m_list.erase(i);
    NS_LOG_DEBUG("@" << this << ": " << ev.impl << ", " << ev.key.m_ts << ", " << ev.key.m_uid);
    return ev;
}

void
MapScheduler::Remove(const Event& ev)
{
    NS_LOG_FUNCTION(this << ev.impl << ev.key.m_ts << ev.key.m_uid);
    auto i = m_list.find(ev.key);
    NS_ASSERT(i->second == ev.impl);
    m_list.erase(i);
}

} // namespace ns3
