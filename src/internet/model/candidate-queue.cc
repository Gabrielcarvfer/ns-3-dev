/*
 * Copyright 2007 University of Washington
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "candidate-queue.h"

#include "global-route-manager-impl.h"

#include "ns3/assert.h"
#include "ns3/log.h"

#include <algorithm>
#include <iostream>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("CandidateQueue");

/**
 * @brief Stream insertion operator.
 *
 * @param os the reference to the output stream
 * @param t the SPFVertex type
 * @returns the reference to the output stream
 */
std::ostream&
operator<<(std::ostream& os, const SPFVertex::VertexType& t)
{
    switch (t)
    {
    case SPFVertex::VertexRouter:
        os << "router";
        break;
    case SPFVertex::VertexNetwork:
        os << "network";
        break;
    default:
        os << "unknown";
        break;
    };
    return os;
}

std::ostream&
operator<<(std::ostream& os, const CandidateQueue& q)
{
    const CandidateQueue::CandidateList_t& list = q.m_candidates;

    os << "*** CandidateQueue Begin (<id, distance, LSA-type>) ***" << std::endl;
    for (auto iter = list.begin(); iter != list.end(); iter++)
    {
        os << "<" << (*iter)->GetVertexId() << ", " << (*iter)->GetDistanceFromRoot() << ", "
           << (*iter)->GetVertexType() << ">" << std::endl;
    }
    os << "*** CandidateQueue End ***";
    return os;
}

CandidateQueue::CandidateQueue()
    : m_candidates()
{
    NS_LOG_FUNCTION(this);
}

CandidateQueue::~CandidateQueue()
{
    NS_LOG_FUNCTION(this);
    Clear();
}

void
CandidateQueue::Clear()
{
    NS_LOG_FUNCTION(this);
    while (!m_candidates.empty())
    {
        SPFVertex* p = Pop();
        delete p;
        p = nullptr;
    }
}

void
CandidateQueue::Push(SPFVertex* vNew)
{
    NS_LOG_FUNCTION(this << vNew);

    auto i = std::upper_bound(m_candidates.begin(),
                              m_candidates.end(),
                              vNew,
                              &CandidateQueue::CompareSPFVertex);
    m_candidates.insert(i, vNew);
}

SPFVertex*
CandidateQueue::Pop()
{
    NS_LOG_FUNCTION(this);
    if (m_candidates.empty())
    {
        return nullptr;
    }

    SPFVertex* v = m_candidates.front();
    m_candidates.pop_front();
    return v;
}

SPFVertex*
CandidateQueue::Top() const
{
    NS_LOG_FUNCTION(this);
    if (m_candidates.empty())
    {
        return nullptr;
    }

    return m_candidates.front();
}

bool
CandidateQueue::Empty() const
{
    NS_LOG_FUNCTION(this);
    return m_candidates.empty();
}

uint32_t
CandidateQueue::Size() const
{
    NS_LOG_FUNCTION(this);
    return m_candidates.size();
}

SPFVertex*
CandidateQueue::Find(const Ipv4Address addr) const
{
    NS_LOG_FUNCTION(this);
    auto i = m_candidates.begin();

    for (; i != m_candidates.end(); i++)
    {
        SPFVertex* v = *i;
        if (v->GetVertexId() == addr)
        {
            return v;
        }
    }

    return nullptr;
}

void
CandidateQueue::Reorder()
{
    NS_LOG_FUNCTION(this);

    m_candidates.sort(&CandidateQueue::CompareSPFVertex);
    NS_LOG_LOGIC("After reordering the CandidateQueue");
    NS_LOG_LOGIC(*this);
}

/*
 * In this implementation, SPFVertex follows the ordering where
 * a vertex is ranked first if its GetDistanceFromRoot () is smaller;
 * In case of a tie, NetworkLSA is always ranked before RouterLSA.
 *
 * This ordering is necessary for implementing ECMP
 */
bool
CandidateQueue::CompareSPFVertex(const SPFVertex* v1, const SPFVertex* v2)
{
    NS_LOG_FUNCTION(&v1 << &v2);

    bool result = false;
    if (v1->GetDistanceFromRoot() < v2->GetDistanceFromRoot())
    {
        result = true;
    }
    else if (v1->GetDistanceFromRoot() == v2->GetDistanceFromRoot())
    {
        if (v1->GetVertexType() == SPFVertex::VertexNetwork &&
            v2->GetVertexType() == SPFVertex::VertexRouter)
        {
            result = true;
        }
    }
    return result;
}

} // namespace ns3
