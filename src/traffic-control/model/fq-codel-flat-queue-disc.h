/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#ifndef FQ_CODEL_FLAT_QUEUE_DISC_H
#define FQ_CODEL_FLAT_QUEUE_DISC_H

#include "queue-disc.h"

#include "ns3/nstime.h"

#include <deque>
#include <vector>

namespace ns3
{

/**
 * @ingroup traffic-control
 *
 * @brief A flat, Linux-style FqCoDel queue disc.
 *
 * Functionally an FQ-CoDel scheduler: packets are classified into one of
 * Flows queues by a flow hash, queues are served in deficit round robin
 * with new flows given priority, and each queue applies the CoDel AQM.
 *
 * Unlike FqCoDelQueueDisc, which materializes one child CoDelQueueDisc
 * object per flow (mirroring the ns-3 classful queue disc architecture),
 * this queue disc keeps all per-flow state in a preallocated flat array,
 * the way the Linux fq_codel implementation does: enqueue and dequeue
 * touch no child queue disc objects, fire no per-layer traces and take no
 * per-flow object allocations. This makes it considerably cheaper per
 * packet in large simulations, at the cost of not exposing per-flow
 * queues through the QueueDiscClass introspection API.
 *
 * The CoDel algorithm follows RFC 8289 (target, interval, and the
 * inverse-sqrt control law); packet timestamps are taken at enqueue time.
 */
class FqCoDelFlatQueueDisc : public QueueDisc
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    FqCoDelFlatQueueDisc();
    ~FqCoDelFlatQueueDisc() override;

    // Reasons for dropping packets
    static constexpr const char* OVERLIMIT_DROP = "Overlimit drop"; //!< Overlimit dropped packets
    static constexpr const char* TARGET_EXCEEDED_DROP =
        "Target exceeded drop"; //!< Sojourn time above target

  private:
    /// Scheduling state of a flow queue.
    enum FlowStatus : uint8_t
    {
        INACTIVE, //!< Not on any scheduling list
        NEW_FLOW, //!< On the new-flows list
        OLD_FLOW  //!< On the old-flows list
    };

    /// Per-flow queue and CoDel state.
    struct Flow
    {
        std::deque<Ptr<QueueDiscItem>> queue; //!< The queued packets.
        uint32_t backlogBytes{0};             //!< Bytes queued in this flow.
        int32_t deficit{0};                   //!< DRR deficit.
        FlowStatus status{INACTIVE};          //!< Scheduling status.

        // CoDel state (RFC 8289)
        uint32_t count{0};      //!< Packets dropped since entering drop state.
        uint32_t lastCount{0};  //!< count when leaving the drop state.
        bool dropping{false};   //!< In the drop state.
        Time firstAboveTime{0}; //!< When the sojourn time went above target.
        Time dropNext{0};       //!< Time of the next scheduled drop.
    };

    bool DoEnqueue(Ptr<QueueDiscItem> item) override;
    Ptr<QueueDiscItem> DoDequeue() override;
    bool CheckConfig() override;
    void InitializeParams() override;

    /**
     * @brief Drop packets from the flow with the largest backlog.
     *
     * Invoked when the queue disc is over limit; drops up to DropBatchSize
     * packets, or half the fat flow's backlog, whichever comes first.
     */
    void DropOverlimit();

    /**
     * @brief Dequeue a packet from a flow, applying the CoDel algorithm.
     * @param flow The flow.
     * @return The dequeued packet, or nullptr if the flow is empty.
     */
    Ptr<QueueDiscItem> CoDelDequeue(Flow& flow);

    /**
     * @brief Pop the head packet of a flow, updating statistics.
     * @param flow The flow.
     * @return The head packet, or nullptr if the flow is empty.
     */
    Ptr<QueueDiscItem> PopHead(Flow& flow);

    /**
     * @brief Check whether the head packet of a flow should be dropped.
     * @param flow The flow.
     * @param item The head packet.
     * @param now The current time.
     * @return True if the packet is to be dropped.
     */
    bool OkToDrop(Flow& flow, Ptr<const QueueDiscItem> item, Time now);

    /**
     * @brief The CoDel control law: interval / sqrt(count) after t.
     * @param t The base time.
     * @param count The drop count.
     * @return The time of the next drop.
     */
    Time ControlLaw(Time t, uint32_t count) const;

    uint32_t m_flows;         //!< Number of flow queues
    uint32_t m_quantum;       //!< DRR quantum (bytes); 0 selects the device MTU
    uint32_t m_dropBatchSize; //!< Max packets dropped from the fat flow per overlimit
    uint32_t m_perturbation;  //!< Hash perturbation
    Time m_target;            //!< CoDel target sojourn time
    Time m_interval;          //!< CoDel interval

    std::vector<Flow> m_flowsTable;  //!< The flow queues.
    std::deque<uint32_t> m_newFlows; //!< Indices of flows on the new list.
    std::deque<uint32_t> m_oldFlows; //!< Indices of flows on the old list.
};

} // namespace ns3

#endif /* FQ_CODEL_FLAT_QUEUE_DISC_H */
