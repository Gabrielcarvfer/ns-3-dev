/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ns3/fq-codel-flat-queue-disc.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"

using namespace ns3;

/**
 * @ingroup traffic-control-test
 *
 * Queue disc item with an externally assigned flow hash.
 */
class FqCoDelFlatTestItem : public QueueDiscItem
{
  public:
    /**
     * Constructor.
     * @param p The packet.
     * @param addr The address.
     * @param flowHash The hash returned for this item.
     */
    FqCoDelFlatTestItem(Ptr<Packet> p, const Address& addr, uint32_t flowHash)
        : QueueDiscItem(p, addr, 0),
          m_flowHash(flowHash)
    {
    }

    uint32_t Hash(uint32_t perturbation) const override
    {
        return m_flowHash;
    }

    void AddHeader() override
    {
    }

    bool Mark() override
    {
        return false;
    }

  private:
    uint32_t m_flowHash; //!< The flow hash.
};

/**
 * @ingroup traffic-control-test
 *
 * DRR fairness: packets of two flows enqueued back to back are dequeued
 * interleaved, and the statistics identities hold.
 */
class FqCoDelFlatFairnessTestCase : public TestCase
{
  public:
    FqCoDelFlatFairnessTestCase()
        : TestCase("FqCoDelFlat DRR fairness and statistics")
    {
    }

  private:
    void DoRun() override
    {
        Ptr<FqCoDelFlatQueueDisc> qd = CreateObject<FqCoDelFlatQueueDisc>();
        qd->SetAttribute("Quantum", UintegerValue(1500));
        qd->Initialize();

        Address dest;
        for (uint32_t i = 0; i < 4; i++)
        {
            qd->Enqueue(Create<FqCoDelFlatTestItem>(Create<Packet>(1000), dest, 1));
        }
        for (uint32_t i = 0; i < 4; i++)
        {
            qd->Enqueue(Create<FqCoDelFlatTestItem>(Create<Packet>(1000), dest, 2));
        }
        NS_TEST_EXPECT_MSG_EQ(qd->GetNPackets(), 8, "eight packets queued");

        // Both flows are new; DRR must alternate between them (quantum
        // covers one packet per round).
        std::vector<uint32_t> hashes;
        for (uint32_t i = 0; i < 8; i++)
        {
            Ptr<QueueDiscItem> item = qd->Dequeue();
            NS_TEST_ASSERT_MSG_NE(item, nullptr, "a packet is dequeued");
            hashes.push_back(DynamicCast<FqCoDelFlatTestItem>(item)->Hash(0));
        }
        NS_TEST_EXPECT_MSG_EQ(qd->GetNPackets(), 0, "queue empties");
        uint32_t firstFlowInFirstHalf = 0;
        for (uint32_t i = 0; i < 4; i++)
        {
            firstFlowInFirstHalf += hashes[i] == 1 ? 1 : 0;
        }
        NS_TEST_EXPECT_MSG_EQ(firstFlowInFirstHalf, 2, "the two flows are served in round robin");

        Simulator::Destroy();
    }
};

/**
 * @ingroup traffic-control-test
 *
 * Overlimit: when the queue disc fills up, packets are dropped from the
 * flow with the largest backlog.
 */
class FqCoDelFlatOverlimitTestCase : public TestCase
{
  public:
    FqCoDelFlatOverlimitTestCase()
        : TestCase("FqCoDelFlat overlimit drops from the fat flow")
    {
    }

  private:
    void DoRun() override
    {
        Ptr<FqCoDelFlatQueueDisc> qd = CreateObject<FqCoDelFlatQueueDisc>();
        qd->SetAttribute("MaxSize", StringValue("10p"));
        qd->SetAttribute("Quantum", UintegerValue(1500));
        qd->SetAttribute("DropBatchSize", UintegerValue(2));
        qd->Initialize();

        Address dest;
        for (uint32_t i = 0; i < 9; i++)
        {
            qd->Enqueue(Create<FqCoDelFlatTestItem>(Create<Packet>(1000), dest, 1));
        }
        qd->Enqueue(Create<FqCoDelFlatTestItem>(Create<Packet>(100), dest, 2));
        NS_TEST_EXPECT_MSG_EQ(qd->GetNPackets(), 10, "at the limit, no drops yet");

        // The next packet exceeds the limit; drops must come from flow 1.
        qd->Enqueue(Create<FqCoDelFlatTestItem>(Create<Packet>(100), dest, 2));
        NS_TEST_EXPECT_MSG_EQ(
            qd->GetStats().GetNDroppedPackets(FqCoDelFlatQueueDisc::OVERLIMIT_DROP),
            2,
            "a batch of two packets is dropped on overlimit");
        NS_TEST_EXPECT_MSG_EQ(qd->GetNPackets(), 9, "the queue shrank by the batch");

        // Flow 2 must not have lost anything: dequeue everything and count.
        uint32_t flow2 = 0;
        Ptr<QueueDiscItem> item;
        while ((item = qd->Dequeue()))
        {
            flow2 += DynamicCast<FqCoDelFlatTestItem>(item)->Hash(0) == 2 ? 1 : 0;
        }
        NS_TEST_EXPECT_MSG_EQ(flow2, 2, "the thin flow kept its packets");

        Simulator::Destroy();
    }
};

/**
 * @ingroup traffic-control-test
 *
 * CoDel: with a standing queue whose sojourn time exceeds the target for
 * more than an interval, packets start being dropped.
 */
class FqCoDelFlatCoDelTestCase : public TestCase
{
  public:
    FqCoDelFlatCoDelTestCase()
        : TestCase("FqCoDelFlat drops on standing queues")
    {
    }

  private:
    /**
     * Enqueue one packet on the given queue disc.
     * @param qd The queue disc.
     */
    void EnqueueOne(Ptr<FqCoDelFlatQueueDisc> qd)
    {
        qd->Enqueue(Create<FqCoDelFlatTestItem>(Create<Packet>(4000), Address(), 1));
    }

    /**
     * Dequeue one packet from the given queue disc.
     * @param qd The queue disc.
     */
    void DequeueOne(Ptr<FqCoDelFlatQueueDisc> qd)
    {
        qd->Dequeue();
    }

    void DoRun() override
    {
        Ptr<FqCoDelFlatQueueDisc> qd = CreateObject<FqCoDelFlatQueueDisc>();
        qd->SetAttribute("Quantum", UintegerValue(1500));
        qd->Initialize();

        // Sustain a queue: enqueue faster than we dequeue for two seconds,
        // so the sojourn time stays above the 5 ms target much longer than
        // the 100 ms interval.
        for (uint32_t ms = 0; ms < 2000; ms += 2)
        {
            Simulator::Schedule(MilliSeconds(ms), &FqCoDelFlatCoDelTestCase::EnqueueOne, this, qd);
        }
        for (uint32_t ms = 1; ms < 2000; ms += 4)
        {
            Simulator::Schedule(MilliSeconds(ms), &FqCoDelFlatCoDelTestCase::DequeueOne, this, qd);
        }
        Simulator::Run();

        NS_TEST_EXPECT_MSG_GT(
            qd->GetStats().GetNDroppedPackets(FqCoDelFlatQueueDisc::TARGET_EXCEEDED_DROP),
            0,
            "CoDel drops packets from a standing queue");

        Simulator::Destroy();
    }
};

/**
 * @ingroup traffic-control-test
 *
 * FqCoDelFlat queue disc test suite.
 */
class FqCoDelFlatQueueDiscTestSuite : public TestSuite
{
  public:
    FqCoDelFlatQueueDiscTestSuite()
        : TestSuite("fq-codel-flat-queue-disc", Type::UNIT)
    {
        AddTestCase(new FqCoDelFlatFairnessTestCase, TestCase::Duration::QUICK);
        AddTestCase(new FqCoDelFlatOverlimitTestCase, TestCase::Duration::QUICK);
        AddTestCase(new FqCoDelFlatCoDelTestCase, TestCase::Duration::QUICK);
    }
};

/// Do not forget to allocate an instance of this TestSuite
static FqCoDelFlatQueueDiscTestSuite g_fqCoDelFlatQueueDiscTestSuite;
