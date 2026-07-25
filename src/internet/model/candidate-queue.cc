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
#include <vector>

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
template <typename T>
std::ostream&
operator<<(std::ostream& os, const typename SPFVertex<T>::VertexType& t)
{
    switch (t)
    {
    case SPFVertex<T>::VertexRouter:
        os << "router";
        break;
    case SPFVertex<T>::VertexNetwork:
        os << "network";
        break;
    default:
        os << "unknown";
        break;
    };
    return os;
}

template <typename T>
std::ostream&
operator<<(std::ostream& os, const CandidateQueue<T>& q)
{
    os << "*** CandidateQueue Begin (<id, distance, LSA-type>) ***" << std::endl;
    for (const auto& [priority, vertex] : q.m_queue)
    {
        os << "<" << vertex->GetVertexId() << ", " << vertex->GetDistanceFromRoot() << ", "
           << vertex->GetVertexType() << ">" << std::endl;
    }
    os << "*** CandidateQueue End ***";
    return os;
}

template <typename T>
CandidateQueue<T>::CandidateQueue()
{
    NS_LOG_FUNCTION(this);
}

template <typename T>
CandidateQueue<T>::~CandidateQueue()
{
    NS_LOG_FUNCTION(this);
    Clear();
}

template <typename T>
void
CandidateQueue<T>::Clear()
{
    NS_LOG_FUNCTION(this);
    while (!m_queue.empty())
    {
        SPFVertex<T>* p = Pop();
        delete p;
        p = nullptr;
    }
}

template <typename T>
void
CandidateQueue<T>::Push(SPFVertex<T>* vNew)
{
    NS_LOG_FUNCTION(this << vNew);
    NS_ASSERT_MSG(m_index.find(vNew->GetVertexId()) == m_index.end(),
                  "Vertex " << vNew->GetVertexId() << " is already on the candidate queue");

    const Priority priority = MakePriority(vNew);
    m_queue.emplace(priority, vNew);
    m_index[vNew->GetVertexId()] = {priority, vNew};
}

template <typename T>
SPFVertex<T>*
CandidateQueue<T>::Pop()
{
    NS_LOG_FUNCTION(this);
    if (m_queue.empty())
    {
        return nullptr;
    }

    auto top = m_queue.begin();
    SPFVertex<T>* v = top->second;
    m_queue.erase(top);
    m_index.erase(v->GetVertexId());
    return v;
}

template <typename T>
SPFVertex<T>*
CandidateQueue<T>::Top() const
{
    NS_LOG_FUNCTION(this);
    if (m_queue.empty())
    {
        return nullptr;
    }

    return m_queue.begin()->second;
}

template <typename T>
bool
CandidateQueue<T>::Empty() const
{
    NS_LOG_FUNCTION(this);
    return m_queue.empty();
}

template <typename T>
uint32_t
CandidateQueue<T>::Size() const
{
    NS_LOG_FUNCTION(this);
    return m_queue.size();
}

template <typename T>
SPFVertex<T>*
CandidateQueue<T>::Find(const IpAddress addr) const
{
    NS_LOG_FUNCTION(this);
    auto i = m_index.find(addr);
    return i != m_index.end() ? i->second.second : nullptr;
}

template <typename T>
void
CandidateQueue<T>::Reorder()
{
    NS_LOG_FUNCTION(this);

    // Stable rebuild: extract the vertices in their current (stale) order
    // and re-insert them sorted by their current priorities, so vertices
    // with equal priority keep their relative order, exactly as the former
    // stable list sort did.
    std::vector<SPFVertex<T>*> vertices;
    vertices.reserve(m_queue.size());
    for (const auto& [priority, vertex] : m_queue)
    {
        vertices.push_back(vertex);
    }
    std::stable_sort(vertices.begin(), vertices.end(), &CandidateQueue::CompareSPFVertex);
    m_queue.clear();
    m_index.clear();
    for (SPFVertex<T>* vertex : vertices)
    {
        const Priority priority = MakePriority(vertex);
        m_queue.emplace(priority, vertex);
        m_index[vertex->GetVertexId()] = {priority, vertex};
    }
    NS_LOG_LOGIC("After reordering the CandidateQueue");
    NS_LOG_LOGIC(*this);
}

template <typename T>
void
CandidateQueue<T>::Reorder(SPFVertex<T>* vertex)
{
    NS_LOG_FUNCTION(this << vertex);

    auto indexEntry = m_index.find(vertex->GetVertexId());
    NS_ASSERT_MSG(indexEntry != m_index.end(),
                  "Vertex " << vertex->GetVertexId() << " is not on the candidate queue");
    m_queue.erase(indexEntry->second.first);
    const Priority priority = MakePriority(vertex);
    m_queue.emplace(priority, vertex);
    indexEntry->second.first = priority;
}

/*
 * In this implementation, SPFVertex follows the ordering where
 * a vertex is ranked first if its GetDistanceFromRoot () is smaller;
 * In case of a tie, NetworkLSA is always ranked before RouterLSA.
 *
 * This ordering is necessary for implementing ECMP
 */
template <typename T>
bool
CandidateQueue<T>::CompareSPFVertex(const SPFVertex<T>* v1, const SPFVertex<T>* v2)
{
    NS_LOG_FUNCTION(&v1 << &v2);

    bool result = false;
    if (v1->GetDistanceFromRoot() < v2->GetDistanceFromRoot())
    {
        result = true;
    }
    else if (v1->GetDistanceFromRoot() == v2->GetDistanceFromRoot())
    {
        if (v1->GetVertexType() == SPFVertex<T>::VertexNetwork &&
            v2->GetVertexType() == SPFVertex<T>::VertexRouter)
        {
            result = true;
        }
    }
    return result;
}
template class CandidateQueue<Ipv4Manager>;
template class CandidateQueue<Ipv6Manager>;

} // namespace ns3
