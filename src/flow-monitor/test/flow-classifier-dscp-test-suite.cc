/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ns3/ipv4-flow-classifier.h"
#include "ns3/ipv4-header.h"
#include "ns3/ipv6-flow-classifier.h"
#include "ns3/ipv6-header.h"
#include "ns3/packet.h"
#include "ns3/test.h"

using namespace ns3;

/**
 * @ingroup flow-monitor
 * @defgroup flow-monitor-test Flow Monitor module tests
 */

/**
 * @ingroup flow-monitor-test
 *
 * @brief Regression test for the deterministic ordering of GetDscpCounts().
 *
 * Ipv4FlowClassifier::GetDscpCounts()/Ipv6FlowClassifier::GetDscpCounts() sort
 * the per-flow (DSCP, count) pairs by decreasing packet count using std::sort,
 * which is not stable. Before the fix for issue #1318 the comparator ordered by
 * count only, so the relative order of entries sharing the same count was
 * unspecified and could vary across standard library implementations. The
 * comparator now breaks ties by the DSCP value; this test pins that order.
 */
class FlowClassifierDscpOrderTestCase : public TestCase
{
  public:
    FlowClassifierDscpOrderTestCase()
        : TestCase("GetDscpCounts returns a deterministic, tie-broken order")
    {
    }

  private:
    void DoRun() override;

    /// Build a 4-byte payload carrying the (unused for this test) L4 ports.
    /// @return a packet large enough for the classifier to read the ports.
    static Ptr<Packet> MakePayload()
    {
        uint8_t data[4] = {0x03, 0xE8, 0x07, 0xD0}; // src port 1000, dst port 2000
        return Create<Packet>(data, 4);
    }
};

void
FlowClassifierDscpOrderTestCase::DoRun()
{
    // --- IPv4 ---
    {
        auto classifier = Create<Ipv4FlowClassifier>();
        Ipv4Header header;
        header.SetSource(Ipv4Address("10.0.0.1"));
        header.SetDestination(Ipv4Address("10.0.0.2"));
        header.SetProtocol(17); // UDP

        uint32_t flowId;
        uint32_t packetId;
        // Same 5-tuple, varying DSCP: all packets belong to the same flow, and
        // their DSCP counts accumulate. EF gets the highest count; CS1 and AF11
        // are tied, and CS1 (0x08) must precede AF11 (0x0A) after the fix.
        auto classify = [&](Ipv4Header::DscpType dscp, uint32_t times) {
            header.SetDscp(dscp);
            for (uint32_t i = 0; i < times; ++i)
            {
                classifier->Classify(header, MakePayload(), &flowId, &packetId);
            }
        };
        classify(Ipv4Header::DSCP_EF, 3);
        classify(Ipv4Header::DSCP_AF11, 2);
        classify(Ipv4Header::DSCP_CS1, 2);

        auto counts = classifier->GetDscpCounts(flowId);
        NS_TEST_ASSERT_MSG_EQ(counts.size(), 3, "Expected three distinct DSCP values");
        NS_TEST_EXPECT_MSG_EQ(counts[0].first, Ipv4Header::DSCP_EF, "Highest count must be first");
        NS_TEST_EXPECT_MSG_EQ(counts[0].second, 3, "EF count wrong");
        // Tie between CS1 and AF11 (both count 2) broken by ascending DSCP value.
        NS_TEST_EXPECT_MSG_EQ(counts[1].first, Ipv4Header::DSCP_CS1, "Tie not broken by DSCP value");
        NS_TEST_EXPECT_MSG_EQ(counts[2].first, Ipv4Header::DSCP_AF11, "Tie not broken by DSCP value");
        NS_TEST_EXPECT_MSG_EQ(counts[1].second, 2, "CS1 count wrong");
        NS_TEST_EXPECT_MSG_EQ(counts[2].second, 2, "AF11 count wrong");
    }

    // --- IPv6 ---
    {
        auto classifier = Create<Ipv6FlowClassifier>();
        Ipv6Header header;
        header.SetSource(Ipv6Address("2001:db8::1"));
        header.SetDestination(Ipv6Address("2001:db8::2"));
        header.SetNextHeader(17); // UDP

        uint32_t flowId;
        uint32_t packetId;
        auto classify = [&](Ipv6Header::DscpType dscp, uint32_t times) {
            header.SetDscp(dscp);
            for (uint32_t i = 0; i < times; ++i)
            {
                classifier->Classify(header, MakePayload(), &flowId, &packetId);
            }
        };
        classify(Ipv6Header::DSCP_EF, 3);
        classify(Ipv6Header::DSCP_AF11, 2);
        classify(Ipv6Header::DSCP_CS1, 2);

        auto counts = classifier->GetDscpCounts(flowId);
        NS_TEST_ASSERT_MSG_EQ(counts.size(), 3, "Expected three distinct DSCP values");
        NS_TEST_EXPECT_MSG_EQ(counts[0].first, Ipv6Header::DSCP_EF, "Highest count must be first");
        NS_TEST_EXPECT_MSG_EQ(counts[0].second, 3, "EF count wrong");
        NS_TEST_EXPECT_MSG_EQ(counts[1].first, Ipv6Header::DSCP_CS1, "Tie not broken by DSCP value");
        NS_TEST_EXPECT_MSG_EQ(counts[2].first, Ipv6Header::DSCP_AF11, "Tie not broken by DSCP value");
        NS_TEST_EXPECT_MSG_EQ(counts[1].second, 2, "CS1 count wrong");
        NS_TEST_EXPECT_MSG_EQ(counts[2].second, 2, "AF11 count wrong");
    }
}

/**
 * @ingroup flow-monitor-test
 *
 * @brief Flow classifier DSCP test suite.
 */
class FlowClassifierDscpTestSuite : public TestSuite
{
  public:
    FlowClassifierDscpTestSuite()
        : TestSuite("flow-classifier-dscp", Type::UNIT)
    {
        AddTestCase(new FlowClassifierDscpOrderTestCase, TestCase::Duration::QUICK);
    }
};

static FlowClassifierDscpTestSuite g_flowClassifierDscpTestSuite; //!< Static variable for test
                                                                  //!< initialization
