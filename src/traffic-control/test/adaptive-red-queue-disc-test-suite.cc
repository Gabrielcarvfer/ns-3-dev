/*
 * Copyright (c) 2015 NITK Surathkal
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 *
 */

#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/packet.h"
#include "ns3/red-queue-disc.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"

using namespace ns3;

/**
 * @ingroup traffic-control
 * @ingroup tests
 * @defgroup traffic-control-test traffic-control module tests
 */

/**
 * @ingroup traffic-control-test
 *
 * @brief Ared Queue Disc Test Item
 */
class AredQueueDiscTestItem : public QueueDiscItem
{
  public:
    /**
     * Constructor
     *
     * @param p packet
     * @param addr address
     */
    AredQueueDiscTestItem(Ptr<Packet> p, const Address& addr);
    ~AredQueueDiscTestItem() override;

    // Delete default constructor, copy constructor and assignment operator to avoid misuse
    AredQueueDiscTestItem() = delete;
    AredQueueDiscTestItem(const AredQueueDiscTestItem&) = delete;
    AredQueueDiscTestItem& operator=(const AredQueueDiscTestItem&) = delete;

    void AddHeader() override;
    bool Mark() override;
};

AredQueueDiscTestItem::AredQueueDiscTestItem(Ptr<Packet> p, const Address& addr)
    : QueueDiscItem(p, addr, 0)
{
}

AredQueueDiscTestItem::~AredQueueDiscTestItem()
{
}

void
AredQueueDiscTestItem::AddHeader()
{
}

bool
AredQueueDiscTestItem::Mark()
{
    return false;
}

/**
 * @ingroup traffic-control-test
 *
 * @brief Ared Queue Disc Test Case
 */
class AredQueueDiscTestCase : public TestCase
{
  public:
    AredQueueDiscTestCase();
    void DoRun() override;

  private:
    /**
     * Enqueue function
     * @param queue the queue disc
     * @param size the size
     * @param nPkt the number of packets
     */
    void Enqueue(Ptr<RedQueueDisc> queue, uint32_t size, uint32_t nPkt);
    /**
     * Enqueue with delay function
     * @param queue the queue disc
     * @param size the size
     * @param nPkt the number of packets
     */
    void EnqueueWithDelay(Ptr<RedQueueDisc> queue, uint32_t size, uint32_t nPkt);
    /**
     * Run ARED queue disc test function
     * @param mode the test mode
     */
    void RunAredDiscTest(QueueSizeUnit mode);
};

AredQueueDiscTestCase::AredQueueDiscTestCase()
    : TestCase("Sanity check on the functionality of Adaptive RED")
{
}

void
AredQueueDiscTestCase::RunAredDiscTest(QueueSizeUnit mode)
{
    uint32_t pktSize = 0;
    uint32_t modeSize = 1; // 1 for packets; pktSize for bytes
    double minTh = 70;
    double maxTh = 150;
    uint32_t qSize = 300;
    Address dest;

    // test 1: Verify automatic setting of QW. [QW = 0.0 with default LinkBandwidth]
    Ptr<RedQueueDisc> queue = CreateObject<RedQueueDisc>();

    if (mode == QueueSizeUnit::BYTES)
    {
        pktSize = 500;
        modeSize = pktSize;
        minTh = minTh * modeSize;
        maxTh = maxTh * modeSize;
        qSize = qSize * modeSize;
    }

    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("MinTh", DoubleValue(minTh)),
                          true,
                          "Verify that we can actually set the attribute MinTh");
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("MaxTh", DoubleValue(maxTh)),
                          true,
                          "Verify that we can actually set the attribute MaxTh");
    NS_TEST_ASSERT_MSG_EQ(
        queue->SetAttributeFailSafe("MaxSize", QueueSizeValue(QueueSize(mode, qSize))),
        true,
        "Verify that we can actually set the attribute MaxSize");
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("QW", DoubleValue(0.0)),
                          true,
                          "Verify that we can actually set the attribute QW");
    queue->Initialize();
    Enqueue(queue, pktSize, 300);
    QueueDisc::Stats st = queue->GetStats();
    NS_TEST_ASSERT_MSG_EQ(st.GetNDroppedPackets(RedQueueDisc::UNFORCED_DROP),
                          0,
                          "There should be zero unforced drops");

    // test 2: Verify automatic setting of QW. [QW = 0.0 with lesser LinkBandwidth]
    queue = CreateObject<RedQueueDisc>();
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("MinTh", DoubleValue(minTh)),
                          true,
                          "Verify that we can actually set the attribute MinTh");
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("MaxTh", DoubleValue(maxTh)),
                          true,
                          "Verify that we can actually set the attribute MaxTh");
    NS_TEST_ASSERT_MSG_EQ(
        queue->SetAttributeFailSafe("MaxSize", QueueSizeValue(QueueSize(mode, qSize))),
        true,
        "Verify that we can actually set the attribute MaxSize");
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("QW", DoubleValue(0.0)),
                          true,
                          "Verify that we can actually set the attribute QW");
    NS_TEST_ASSERT_MSG_EQ(
        queue->SetAttributeFailSafe("LinkBandwidth", DataRateValue(DataRate("0.015Mbps"))),
        true,
        "Verify that we can actually set the attribute LinkBandwidth");
    queue->Initialize();
    Enqueue(queue, pktSize, 300);
    st = queue->GetStats();
    NS_TEST_ASSERT_MSG_NE(st.GetNDroppedPackets(RedQueueDisc::UNFORCED_DROP),
                          0,
                          "There should be some unforced drops");

    // test 3: Verify automatic setting of QW. [QW = -1.0 with default LinkBandwidth]
    queue = CreateObject<RedQueueDisc>();
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("MinTh", DoubleValue(minTh)),
                          true,
                          "Verify that we can actually set the attribute MinTh");
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("MaxTh", DoubleValue(maxTh)),
                          true,
                          "Verify that we can actually set the attribute MaxTh");
    NS_TEST_ASSERT_MSG_EQ(
        queue->SetAttributeFailSafe("MaxSize", QueueSizeValue(QueueSize(mode, qSize))),
        true,
        "Verify that we can actually set the attribute MaxSize");
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("QW", DoubleValue(-1.0)),
                          true,
                          "Verify that we can actually set the attribute QW");
    queue->Initialize();
    Enqueue(queue, pktSize, 300);
    st = queue->GetStats();
    NS_TEST_ASSERT_MSG_EQ(st.GetNDroppedPackets(RedQueueDisc::UNFORCED_DROP),
                          0,
                          "There should be zero unforced drops");

    // test 4: Verify automatic setting of QW. [QW = -1.0 with lesser LinkBandwidth]
    queue = CreateObject<RedQueueDisc>();
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("MinTh", DoubleValue(minTh)),
                          true,
                          "Verify that we can actually set the attribute MinTh");
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("MaxTh", DoubleValue(maxTh)),
                          true,
                          "Verify that we can actually set the attribute MaxTh");
    NS_TEST_ASSERT_MSG_EQ(
        queue->SetAttributeFailSafe("MaxSize", QueueSizeValue(QueueSize(mode, qSize))),
        true,
        "Verify that we can actually set the attribute MaxSize");
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("QW", DoubleValue(-1.0)),
                          true,
                          "Verify that we can actually set the attribute QW");
    NS_TEST_ASSERT_MSG_EQ(
        queue->SetAttributeFailSafe("LinkBandwidth", DataRateValue(DataRate("0.015Mbps"))),
        true,
        "Verify that we can actually set the attribute LinkBandwidth");
    queue->Initialize();
    Enqueue(queue, pktSize, 300);
    st = queue->GetStats();
    NS_TEST_ASSERT_MSG_NE(st.GetNDroppedPackets(RedQueueDisc::UNFORCED_DROP),
                          0,
                          "There should be some unforced drops");

    // test 5: Verify automatic setting of QW. [QW = -2.0 with default LinkBandwidth]
    queue = CreateObject<RedQueueDisc>();
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("MinTh", DoubleValue(minTh)),
                          true,
                          "Verify that we can actually set the attribute MinTh");
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("MaxTh", DoubleValue(maxTh)),
                          true,
                          "Verify that we can actually set the attribute MaxTh");
    NS_TEST_ASSERT_MSG_EQ(
        queue->SetAttributeFailSafe("MaxSize", QueueSizeValue(QueueSize(mode, qSize))),
        true,
        "Verify that we can actually set the attribute MaxSize");
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("QW", DoubleValue(-2.0)),
                          true,
                          "Verify that we can actually set the attribute QW");
    queue->Initialize();
    Enqueue(queue, pktSize, 300);
    st = queue->GetStats();
    uint32_t test5 = st.GetNDroppedPackets(RedQueueDisc::UNFORCED_DROP);
    NS_TEST_ASSERT_MSG_NE(test5, 0, "There should be some unforced drops");

    // test 6: Verify automatic setting of QW. [QW = -2.0 with lesser LinkBandwidth]
    queue = CreateObject<RedQueueDisc>();
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("MinTh", DoubleValue(minTh)),
                          true,
                          "Verify that we can actually set the attribute MinTh");
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("MaxTh", DoubleValue(maxTh)),
                          true,
                          "Verify that we can actually set the attribute MaxTh");
    NS_TEST_ASSERT_MSG_EQ(
        queue->SetAttributeFailSafe("MaxSize", QueueSizeValue(QueueSize(mode, qSize))),
        true,
        "Verify that we can actually set the attribute MaxSize");
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("QW", DoubleValue(-2.0)),
                          true,
                          "Verify that we can actually set the attribute QW");
    NS_TEST_ASSERT_MSG_EQ(
        queue->SetAttributeFailSafe("LinkBandwidth", DataRateValue(DataRate("0.015Mbps"))),
        true,
        "Verify that we can actually set the attribute LinkBandwidth");
    queue->Initialize();
    Enqueue(queue, pktSize, 300);
    st = queue->GetStats();
    uint32_t test6 = st.GetNDroppedPackets(RedQueueDisc::UNFORCED_DROP);
    NS_TEST_ASSERT_MSG_NE(test6, test5, "Test 6 should have more unforced drops than Test 5");

    // test 7: Verify automatic setting of minTh and maxTh. [minTh = maxTh = 0.0, with default
    // LinkBandwidth]
    queue = CreateObject<RedQueueDisc>();
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("MinTh", DoubleValue(0.0)),
                          true,
                          "Verify that we can actually set the attribute MinTh");
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("MaxTh", DoubleValue(0.0)),
                          true,
                          "Verify that we can actually set the attribute MaxTh");
    NS_TEST_ASSERT_MSG_EQ(
        queue->SetAttributeFailSafe("MaxSize", QueueSizeValue(QueueSize(mode, qSize))),
        true,
        "Verify that we can actually set the attribute MaxSize");
    queue->Initialize();
    Enqueue(queue, pktSize, 300);
    st = queue->GetStats();
    NS_TEST_ASSERT_MSG_NE(st.GetNDroppedPackets(RedQueueDisc::UNFORCED_DROP),
                          0,
                          "There should be some unforced drops");

    // test 8: Verify automatic setting of minTh and maxTh. [minTh = maxTh = 0.0, with higher
    // LinkBandwidth]
    queue = CreateObject<RedQueueDisc>();
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("MinTh", DoubleValue(0.0)),
                          true,
                          "Verify that we can actually set the attribute MinTh");
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("MaxTh", DoubleValue(0.0)),
                          true,
                          "Verify that we can actually set the attribute MaxTh");
    NS_TEST_ASSERT_MSG_EQ(
        queue->SetAttributeFailSafe("MaxSize", QueueSizeValue(QueueSize(mode, qSize))),
        true,
        "Verify that we can actually set the attribute MaxSize");
    NS_TEST_ASSERT_MSG_EQ(
        queue->SetAttributeFailSafe("LinkBandwidth", DataRateValue(DataRate("150Mbps"))),
        true,
        "Verify that we can actually set the attribute LinkBandwidth");
    queue->Initialize();
    Enqueue(queue, pktSize, 300);
    st = queue->GetStats();
    NS_TEST_ASSERT_MSG_EQ(st.GetNDroppedPackets(RedQueueDisc::UNFORCED_DROP),
                          0,
                          "There should be zero unforced drops");

    // test 9: Default RED (automatic and adaptive settings disabled)
    queue = CreateObject<RedQueueDisc>();
    minTh = 5 * modeSize;
    maxTh = 15 * modeSize;
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("MinTh", DoubleValue(minTh)),
                          true,
                          "Verify that we can actually set the attribute MinTh");
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("MaxTh", DoubleValue(maxTh)),
                          true,
                          "Verify that we can actually set the attribute MaxTh");
    NS_TEST_ASSERT_MSG_EQ(
        queue->SetAttributeFailSafe("MaxSize", QueueSizeValue(QueueSize(mode, qSize))),
        true,
        "Verify that we can actually set the attribute MaxSize");
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("QW", DoubleValue(0.002)),
                          true,
                          "Verify that we can actually set the attribute QW");
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("LInterm", DoubleValue(2)),
                          true,
                          "Verify that we can actually set the attribute LInterm");
    queue->Initialize();
    EnqueueWithDelay(queue, pktSize, 300);
    Simulator::Stop(Seconds(5));
    Simulator::Run();
    st = queue->GetStats();
    uint32_t test9 = st.GetNDroppedPackets(RedQueueDisc::UNFORCED_DROP);
    NS_TEST_ASSERT_MSG_NE(st.GetNDroppedPackets(RedQueueDisc::UNFORCED_DROP),
                          0,
                          "There should be some unforced drops");

    // test 10: Adaptive RED (automatic and adaptive settings enabled)
    queue = CreateObject<RedQueueDisc>();
    NS_TEST_ASSERT_MSG_EQ(
        queue->SetAttributeFailSafe("MaxSize", QueueSizeValue(QueueSize(mode, qSize))),
        true,
        "Verify that we can actually set the attribute MaxSize");
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("LInterm", DoubleValue(2)),
                          true,
                          "Verify that we can actually set the attribute LInterm");
    NS_TEST_ASSERT_MSG_EQ(queue->SetAttributeFailSafe("ARED", BooleanValue(true)),
                          true,
                          "Verify that we can actually set the attribute ARED");
    queue->Initialize();
    EnqueueWithDelay(queue, pktSize, 300);
    Simulator::Stop(Seconds(5));
    Simulator::Run();
    st = queue->GetStats();
    uint32_t test10 = st.GetNDroppedPackets(RedQueueDisc::UNFORCED_DROP);
    NS_TEST_ASSERT_MSG_LT(test10, test9, "Test 10 should have less unforced drops than test 9");
}

void
AredQueueDiscTestCase::Enqueue(Ptr<RedQueueDisc> queue, uint32_t size, uint32_t nPkt)
{
    Address dest;
    for (uint32_t i = 0; i < nPkt; i++)
    {
        queue->Enqueue(Create<AredQueueDiscTestItem>(Create<Packet>(size), dest));
    }
}

void
AredQueueDiscTestCase::EnqueueWithDelay(Ptr<RedQueueDisc> queue, uint32_t size, uint32_t nPkt)
{
    Address dest;
    double delay = 0.01; // enqueue packets with delay to allow m_curMaxP to adapt
    for (uint32_t i = 0; i < nPkt; i++)
    {
        Simulator::Schedule(Seconds((i + 1) * delay),
                            &AredQueueDiscTestCase::Enqueue,
                            this,
                            queue,
                            size,
                            1);
    }
}

void
AredQueueDiscTestCase::DoRun()
{
    RunAredDiscTest(QueueSizeUnit::PACKETS);
    RunAredDiscTest(QueueSizeUnit::BYTES);
    Simulator::Destroy();
}

/**
 * @ingroup traffic-control-test
 *
 * @brief Ared Queue Disc Test Suite
 */
static class AredQueueDiscTestSuite : public TestSuite
{
  public:
    AredQueueDiscTestSuite()
        : TestSuite("adaptive-red-queue-disc", Type::UNIT)
    {
        AddTestCase(new AredQueueDiscTestCase(), TestCase::Duration::QUICK);
    }
} g_aredQueueDiscTestSuite; ///< the test suite
