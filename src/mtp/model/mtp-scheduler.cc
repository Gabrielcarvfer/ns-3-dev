/*
 * Copyright (c) 2026 Universidade de Brasilia
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

/**
 * @file
 * @ingroup mtp
 * Implementation of class ns3::MtpScheduler
 */

#include "mtp-scheduler.h"

#include "ns3/assert.h"

namespace ns3
{

bool
MtpEventLess(const MtpEvent& a, const MtpEvent& b)
{
    if (a.event.key.m_ts != b.event.key.m_ts)
    {
        return a.event.key.m_ts < b.event.key.m_ts;
    }
    for (std::size_t i = 0; i < MTP_LINEAGE_DEPTH; i++)
    {
        const EventAncestor& x = a.lineage[i];
        const EventAncestor& y = b.lineage[i];
        if (x.ts != y.ts)
        {
            return x.ts < y.ts;
        }
        if (x.systemId != y.systemId)
        {
            // Ancestors executed at the same time by different LPs: their
            // relative order is given by the previous generation.
            continue;
        }
        if (x.seq != y.seq)
        {
            return x.seq < y.seq;
        }
        // Common ancestor: the children were scheduled by the same event, in
        // the order given by their origin uids.
        const uint32_t ua = i == 0 ? a.originUid : a.lineage[i - 1].originUid;
        const uint32_t ub = i == 0 ? b.originUid : b.lineage[i - 1].originUid;
        if (ua != ub)
        {
            return ua < ub;
        }
        break;
    }
    return a.event.key.m_uid < b.event.key.m_uid;
}

MtpScheduler::~MtpScheduler()
{
    for (const auto& ev : m_events)
    {
        ev.event.impl->Unref();
    }
}

MtpScheduler::MtpScheduler(const MtpScheduler& o)
{
    *this = o;
}

MtpScheduler&
MtpScheduler::operator=(const MtpScheduler& o)
{
    if (this == &o)
    {
        return *this;
    }
    for (const auto& ev : m_events)
    {
        ev.event.impl->Unref();
    }
    m_events = o.m_events;
    for (const auto& ev : m_events)
    {
        ev.event.impl->Ref();
    }
    m_byUid.clear();
    for (auto it = m_events.begin(); it != m_events.end(); ++it)
    {
        m_byUid[it->event.key.m_uid] = it;
    }
    return *this;
}

void
MtpScheduler::Insert(const MtpEvent& ev)
{
    auto [it, inserted] = m_events.insert(ev);
    NS_ASSERT_MSG(inserted, "Duplicate event " << ev.event.key.m_uid);
    m_byUid[ev.event.key.m_uid] = it;
}

bool
MtpScheduler::IsEmpty() const
{
    return m_events.empty();
}

const MtpEvent&
MtpScheduler::PeekNext() const
{
    NS_ASSERT(!m_events.empty());
    return *m_events.begin();
}

MtpEvent
MtpScheduler::RemoveNext()
{
    NS_ASSERT(!m_events.empty());
    auto it = m_events.begin();
    MtpEvent ev = *it;
    m_byUid.erase(ev.event.key.m_uid);
    m_events.erase(it);
    return ev;
}

bool
MtpScheduler::Remove(uint32_t uid, MtpEvent& removed)
{
    auto it = m_byUid.find(uid);
    if (it == m_byUid.end())
    {
        return false;
    }
    removed = *it->second;
    m_events.erase(it->second);
    m_byUid.erase(it);
    return true;
}

bool
MtpScheduler::Contains(uint32_t uid) const
{
    return m_byUid.contains(uid);
}

} // namespace ns3
