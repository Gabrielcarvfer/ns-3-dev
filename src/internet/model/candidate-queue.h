/*
 * Copyright 2007 University of Washington
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author:  Craig Dowell (craigdo@ee.washington.edu)
 */

#ifndef CANDIDATE_QUEUE_H
#define CANDIDATE_QUEUE_H

#include "global-route-manager.h"

#include "ns3/ipv4-address.h"
#include "ns3/ipv6-address.h"

#include <map>
#include <stdint.h>
#include <tuple>
#include <unordered_map>

namespace ns3
{

template <typename T>
class SPFVertex;

template <typename>
class CandidateQueue;

// Forward declaration of operator<<
template <typename T>
std::ostream& operator<<(std::ostream& os, const CandidateQueue<T>& q);

/**
 * @ingroup globalrouting
 *
 * @brief A Candidate Queue used in routing calculations.
 *
 * The CandidateQueue is used in the OSPF shortest path computations.  It
 * is a priority queue used to store candidates for the shortest path to a
 * given network.
 *
 * The queue holds Shortest Path First Vertex pointers and orders them
 * according to the lowest value of the field m_distanceFromRoot.  Remaining
 * vertices are ordered according to increasing distance.  This implements a
 * priority queue.
 *
 * Although a STL priority_queue almost does what we want, the requirement
 * for a Find () operation, the dynamic nature of the data and the derived
 * requirement for a Reorder () operation led us to implement this simple
 * enhanced priority queue.
 */
template <typename T>
class CandidateQueue
{
    static_assert(std::is_same_v<T, Ipv4Manager> || std::is_same_v<T, Ipv6Manager>,
                  "T must be either Ipv4Manager or Ipv6Manager when calling CandidateQueue");

    /// Alias for determining whether the parent is Ipv4RoutingProtocol or Ipv6RoutingProtocol
    static constexpr bool IsIpv4 = std::is_same_v<Ipv4Manager, T>;

    /// Alias for Ipv4Manager and Ipv6Manager classes
    using IpManager = typename std::conditional_t<IsIpv4, Ipv4Manager, Ipv6Manager>;

    /// Alias for Ipv4 and Ipv6 classes
    using Ip = typename std::conditional_t<IsIpv4, Ipv4, Ipv6>;

    /// Alias for Ipv4Address and Ipv6Address classes
    using IpAddress = typename std::conditional_t<IsIpv4, Ipv4Address, Ipv6Address>;

  public:
    /**
     * @brief Create an empty SPF Candidate Queue.
     *
     * @see SPFVertex
     */
    CandidateQueue();

    /**
     * @brief Destroy an SPF Candidate Queue and release any resources held
     * by the contents.
     *
     * @see SPFVertex
     */
    virtual ~CandidateQueue();

    // Delete copy constructor and assignment operator to avoid misuse
    CandidateQueue(const CandidateQueue&) = delete;
    CandidateQueue& operator=(const CandidateQueue&) = delete;

    /**
     * @brief Empty the Candidate Queue and release all of the resources
     * associated with the Shortest Path First Vertex pointers in the queue.
     *
     * @see SPFVertex
     */
    void Clear();

    /**
     * @brief Push a Shortest Path First Vertex pointer onto the queue according
     * to the priority scheme.
     *
     * On completion, the top of the queue will hold the Shortest Path First
     * Vertex pointer that points to a vertex having lowest value of the field
     * m_distanceFromRoot.  Remaining vertices are ordered according to
     * increasing distance.
     *
     * @see SPFVertex
     * @param vNew The Shortest Path First Vertex to add to the queue.
     */
    void Push(SPFVertex<T>* vNew);

    /**
     * @brief Pop the Shortest Path First Vertex pointer at the top of the queue.
     *
     * The caller is given the responsibility for releasing the resources
     * associated with the vertex.
     *
     * @see SPFVertex
     * @see Top ()
     * @returns The Shortest Path First Vertex pointer at the top of the queue.
     */
    SPFVertex<T>* Pop();

    /**
     * @brief Return the Shortest Path First Vertex pointer at the top of the
     * queue.
     *
     * This method does not pop the SPFVertex* off of the queue, it simply
     * returns the pointer.
     *
     * @see SPFVertex
     * @see Pop ()
     * @returns The Shortest Path First Vertex pointer at the top of the queue.
     */
    SPFVertex<T>* Top() const;

    /**
     * @brief Test the Candidate Queue to determine if it is empty.
     *
     * @returns True if the queue is empty, false otherwise.
     */
    bool Empty() const;

    /**
     * @brief Return the number of Shortest Path First Vertex pointers presently
     * stored in the Candidate Queue.
     *
     * @see SPFVertex
     * @returns The number of SPFVertex* pointers in the Candidate Queue.
     */
    uint32_t Size() const;

    /**
     * @brief Searches the Candidate Queue for a Shortest Path First Vertex
     * pointer that points to a vertex having the given IP address.
     *
     * @see SPFVertex
     * @param addr The IP address to search for.
     * @returns The SPFVertex* pointer corresponding to the given IP address.
     */
    SPFVertex<T>* Find(const IpAddress addr) const;

    /**
     * @brief Reorders the Candidate Queue according to the priority scheme.
     *
     * On completion, the top of the queue will hold the Shortest Path First
     * Vertex pointer that points to a vertex having lowest value of the field
     * m_distanceFromRoot.  Remaining vertices are ordered according to
     * increasing distance.
     *
     * This method is provided in case the values of m_distanceFromRoot change
     * during the routing calculations. When the changed vertex is known,
     * prefer the overload taking that vertex, which reorders in logarithmic
     * instead of linearithmic time.
     *
     * @see SPFVertex
     */
    void Reorder();

    /**
     * @brief Restore the queue ordering after the distance of the given
     * vertex has been decreased.
     *
     * Equivalent to Reorder() when only @p vertex changed, but runs in
     * logarithmic time. The vertex keeps the same position relative to
     * equal-priority vertices as a stable full reordering would give it.
     *
     * @see SPFVertex
     * @param vertex The vertex whose distance from the root has changed.
     */
    void Reorder(SPFVertex<T>* vertex);

  private:
    /**
     * @brief return true if v1 < v2
     *
     * SPFVertex items are added into the queue according to the ordering
     * defined by this method. If v1 should be popped before v2, this
     * method return true; false otherwise
     *
     * @param v1 first operand
     * @param v2 second operand
     * @return True if v1 should be popped before v2; false otherwise
     */
    static bool CompareSPFVertex(const SPFVertex<T>* v1, const SPFVertex<T>* v2);

    /**
     * @brief Priority of a queued vertex.
     *
     * Vertices are popped by increasing distance from the root; on equal
     * distance a network vertex is ranked before a router vertex, and
     * remaining ties preserve insertion order through a sequence number
     * (a vertex whose distance is decreased is re-inserted with a fresh
     * sequence number, matching the ordering the former stable full sort
     * produced).
     */
    struct Priority
    {
        uint32_t distance; //!< Distance from the root vertex.
        uint8_t typeRank;  //!< 0 for network vertices, 1 otherwise.
        uint64_t sequence; //!< Insertion order tie breaker.

        /**
         * Comparison operator.
         * @param o The other priority.
         * @return True if this priority is popped before @p o.
         */
        bool operator<(const Priority& o) const
        {
            return std::tie(distance, typeRank, sequence) <
                   std::tie(o.distance, o.typeRank, o.sequence);
        }
    };

    /**
     * Compute the priority of a vertex.
     * @param vertex The vertex.
     * @return Its priority, using the next sequence number.
     */
    Priority MakePriority(const SPFVertex<T>* vertex)
    {
        const uint8_t typeRank = vertex->GetVertexType() == SPFVertex<T>::VertexNetwork ? 0 : 1;
        return Priority{vertex->GetDistanceFromRoot(), typeRank, m_sequence++};
    }

    std::map<Priority, SPFVertex<T>*> m_queue; //!< The queued vertices by priority.
    /// The queued vertices indexed by vertex ID, with their current priority.
    std::unordered_map<IpAddress, std::pair<Priority, SPFVertex<T>*>> m_index;
    uint64_t m_sequence{0}; //!< Next insertion sequence number.

    /**
     * @brief Stream insertion operator.
     *
     * @param os the reference to the output stream
     * @param q the CandidateQueue
     * @returns the reference to the output stream
     */
    friend std::ostream& operator<< <T>(std::ostream& os, const CandidateQueue& q);
};

} // namespace ns3

#endif /* CANDIDATE_QUEUE_H */
