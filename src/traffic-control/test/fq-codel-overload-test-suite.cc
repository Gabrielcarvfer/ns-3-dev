/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/codel-queue-disc.h"
#include "ns3/fq-codel-queue-disc.h"
#include "ns3/ipv4-address.h"
#include "ns3/ipv4-header.h"
#include "ns3/ipv4-queue-disc-item.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/test.h"

using namespace ns3;

/**
 * @ingroup traffic-control-test
 * @ingroup tests
 *
 * @brief Regression test for @issueid{332}
 *        (overload behavior of FqCoDelQueueDisc).
 *
 * The FqCoDel root is the single authority for overflow: when traffic exceeds
 * the overall (root) limit, the root must enforce that limit by invoking
 * FqCoDelDrop(), which records a drop under the parent reason code
 * FqCoDelQueueDisc::OVERLIMIT_DROP ("Overlimit drop"), rather than let a
 * capped child flow queue reject the overflowing packet first with no
 * parent-level accounting (@issueid{332}).
 *
 * The test overloads a single flow past the overall limit and checks that at
 * least one drop is recorded under FqCoDelQueueDisc::OVERLIMIT_DROP.
 *
 * The test runs entirely at simulation time 0 (no time advances), so CoDel's
 * time-based target/interval dropping cannot interfere; the result is fully
 * deterministic.
 */
class FqCoDelSingleFlowOverloadTest : public TestCase
{
  public:
    FqCoDelSingleFlowOverloadTest();
    ~FqCoDelSingleFlowOverloadTest() override;

  private:
    void DoRun() override;

    /**
     * Enqueue a single 100-byte packet described by the given IPv4 header.
     *
     * @param queue The FqCoDel queue disc under test.
     * @param hdr The IPv4 header used to classify the packet into a flow.
     */
    void AddPacket(Ptr<FqCoDelQueueDisc> queue, Ipv4Header hdr);
};

FqCoDelSingleFlowOverloadTest::FqCoDelSingleFlowOverloadTest()
    : TestCase("Single-flow overload must trigger a root FqCoDel overlimit drop (issue #332)")
{
}

FqCoDelSingleFlowOverloadTest::~FqCoDelSingleFlowOverloadTest()
{
}

void
FqCoDelSingleFlowOverloadTest::AddPacket(Ptr<FqCoDelQueueDisc> queue, Ipv4Header hdr)
{
    Ptr<Packet> p = Create<Packet>(100);
    Address dest;
    Ptr<Ipv4QueueDiscItem> item = Create<Ipv4QueueDiscItem>(p, dest, 0, hdr);
    queue->Enqueue(item);
}

void
FqCoDelSingleFlowOverloadTest::DoRun()
{
    // Overall root limit of four packets.
    const uint32_t maxPackets = 4;

    Ptr<FqCoDelQueueDisc> queueDisc =
        CreateObjectWithAttributes<FqCoDelQueueDisc>("MaxSize", StringValue("4p"));

    queueDisc->SetQuantum(1500);
    queueDisc->Initialize();

    // A single 5-tuple: every packet hashes to the SAME flow, so the single
    // child CoDel queue is the one that overloads.
    Ipv4Header hdr;
    hdr.SetPayloadSize(100);
    hdr.SetSource(Ipv4Address("10.10.1.1"));
    hdr.SetDestination(Ipv4Address("10.10.1.2"));
    hdr.SetProtocol(7);

    // Enqueue one more packet than the overall limit allows (5 > 4) into the
    // single flow.
    for (uint32_t i = 0; i < maxPackets + 1; i++)
    {
        AddPacket(queueDisc, hdr);
    }

    // Sanity: the disc must never hold more than its configured overall limit.
    // This holds on master as well (the child silently caps the count) and is
    // here only to confirm the overload was actually exercised.
    NS_TEST_EXPECT_MSG_LT_OR_EQ(queueDisc->QueueDisc::GetNPackets(),
                                maxPackets,
                                "FqCoDel must never exceed its configured overall packet limit");

    // Load-bearing assertion for @issueid{332}: the FqCoDel ROOT enforces the
    // overall limit by calling FqCoDelDrop(), which records the drop under
    // FqCoDelQueueDisc::OVERLIMIT_DROP.
    NS_TEST_EXPECT_MSG_GT_OR_EQ(
        queueDisc->GetStats().GetNDroppedPackets(FqCoDelQueueDisc::OVERLIMIT_DROP),
        1,
        "Single-flow overload must produce a root FqCoDel OVERLIMIT_DROP "
        "(issue #332): the root, not the child queue, must enforce the overall "
        "limit");

    Simulator::Destroy();
}

/**
 * @ingroup traffic-control-test
 * @ingroup tests
 *
 * @brief Test suite for the FqCoDel overload regression test (@issueid{332}).
 */
class FqCoDelOverloadTestSuite : public TestSuite
{
  public:
    FqCoDelOverloadTestSuite();
};

FqCoDelOverloadTestSuite::FqCoDelOverloadTestSuite()
    : TestSuite("fq-codel-overload", Type::UNIT)
{
    AddTestCase(new FqCoDelSingleFlowOverloadTest, TestCase::Duration::QUICK);
}

/// Do not forget to allocate an instance of this TestSuite.
static FqCoDelOverloadTestSuite g_fqCoDelOverloadTestSuite;
