/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/object-factory.h"
#include "ns3/simulator.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-socket-state.h"
#include "ns3/test.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpAckCountingTest");

namespace
{
constexpr uint32_t SEGMENT_SIZE = 1000; //!< Segment size of the tests
constexpr uint32_t INITIAL_CWND = 20;   //!< Initial cwnd, in segments
constexpr uint32_t SEGS_PER_ACK = 2;    //!< Segments acknowledged per (delayed) ACK
} // namespace

/**
 * @ingroup internet-test
 *
 * @brief Test the CountAckedSegments attribute of TcpNewReno and of the models
 * derived from it
 *
 * The window is grown over one round trip of delayed ACKs, each of them
 * acknowledging two segments. Per ACK (the default), slow start grows the
 * window by half of the segments acknowledged, and congestion avoidance by
 * half of what an ACK per segment would give; per acknowledged segment, slow
 * start doubles the window, and congestion avoidance grows it twice as fast.
 */
class TcpAckCountingTest : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param congControl Type of congestion control.
     * @param slowStart True to test slow start, false for congestion avoidance.
     * @param pktsAcked True to call PktsAcked before the first increase, so
     *        that the model sets the parameters of its increase.
     */
    TcpAckCountingTest(const TypeId& congControl, bool slowStart, bool pktsAcked);

  private:
    void DoRun() override;

    /**
     * @brief Grow the window over one round trip of delayed ACKs.
     * @param countAckedSegments Value of the CountAckedSegments attribute.
     * @return the growth of the window, in bytes.
     */
    uint32_t Grow(bool countAckedSegments);

    /**
     * @brief Run the checks; scheduled after the start of the simulation, as
     * some models compute rates from the time elapsed.
     */
    void Check();

    TypeId m_congControl; //!< Type of congestion control
    bool m_slowStart;     //!< True to test slow start
    bool m_pktsAcked;     //!< True to call PktsAcked before the first increase
};

TcpAckCountingTest::TcpAckCountingTest(const TypeId& congControl, bool slowStart, bool pktsAcked)
    : TestCase(congControl.GetName() + (slowStart ? " slow start" : " congestion avoidance")),
      m_congControl(congControl),
      m_slowStart(slowStart),
      m_pktsAcked(pktsAcked)
{
}

uint32_t
TcpAckCountingTest::Grow(bool countAckedSegments)
{
    ObjectFactory factory(m_congControl.GetName());
    factory.Set("CountAckedSegments", BooleanValue(countAckedSegments));
    auto cong = factory.Create<TcpCongestionOps>();

    auto state = CreateObject<TcpSocketState>();
    state->m_segmentSize = SEGMENT_SIZE;
    state->m_cWnd = INITIAL_CWND * SEGMENT_SIZE;
    state->m_ssThresh = m_slowStart ? 1000 * SEGMENT_SIZE : INITIAL_CWND * SEGMENT_SIZE / 2;
    state->m_minRtt = MilliSeconds(100);
    state->m_lastRtt = MilliSeconds(100);

    if (m_pktsAcked)
    {
        cong->PktsAcked(state, SEGS_PER_ACK, MilliSeconds(100));
    }
    for (uint32_t i = 0; i < INITIAL_CWND / SEGS_PER_ACK; ++i)
    {
        cong->IncreaseWindow(state, SEGS_PER_ACK);
    }
    return state->m_cWnd - INITIAL_CWND * SEGMENT_SIZE;
}

void
TcpAckCountingTest::Check()
{
    uint32_t perAck = Grow(false);
    uint32_t perSegment = Grow(true);
    NS_LOG_INFO(m_congControl.GetName()
                << " growth per ACK " << perAck << " per segment " << perSegment);

    if (m_slowStart)
    {
        NS_TEST_ASSERT_MSG_EQ(perAck,
                              INITIAL_CWND / SEGS_PER_ACK * SEGMENT_SIZE,
                              "Slow start per ACK must grow by a segment per ACK");
        NS_TEST_ASSERT_MSG_EQ(perSegment,
                              INITIAL_CWND * SEGMENT_SIZE,
                              "Slow start per segment must grow by a segment per acked segment");
    }
    else
    {
        NS_TEST_ASSERT_MSG_GT(perAck, 0, "Congestion avoidance must grow the window");
        NS_TEST_ASSERT_MSG_EQ_TOL(static_cast<double>(perSegment) / perAck,
                                  2.0,
                                  0.1,
                                  "Counting segments must double the growth with delayed ACKs");
    }
}

void
TcpAckCountingTest::DoRun()
{
    Simulator::Schedule(Seconds(1), &TcpAckCountingTest::Check, this);
    Simulator::Run();
    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 *
 * @brief Test the growth of congestion avoidance of TcpNewReno over a round
 * trip of delayed ACKs, per ACK and per acknowledged segment
 */
class TcpNewRenoAckCountingCaTest : public TestCase
{
  public:
    TcpNewRenoAckCountingCaTest()
        : TestCase("TcpNewReno congestion avoidance growth per round trip")
    {
    }

  private:
    void DoRun() override
    {
        for (bool countAckedSegments : {false, true})
        {
            auto cong = CreateObject<TcpNewReno>();
            cong->SetAttribute("CountAckedSegments", BooleanValue(countAckedSegments));
            auto state = CreateObject<TcpSocketState>();
            state->m_segmentSize = SEGMENT_SIZE;
            state->m_cWnd = INITIAL_CWND * SEGMENT_SIZE;
            state->m_ssThresh = INITIAL_CWND * SEGMENT_SIZE / 2;

            for (uint32_t i = 0; i < INITIAL_CWND / SEGS_PER_ACK; ++i)
            {
                cong->IncreaseWindow(state, SEGS_PER_ACK);
            }
            // Increments of about SMSS * SMSS / cwnd: one per ACK, or one per
            // acknowledged segment
            uint32_t growth = state->m_cWnd - INITIAL_CWND * SEGMENT_SIZE;
            uint32_t expected = countAckedSegments ? SEGMENT_SIZE : SEGMENT_SIZE / 2;
            NS_TEST_ASSERT_MSG_EQ_TOL(growth,
                                      expected,
                                      expected / 20,
                                      "Unexpected growth with CountAckedSegments "
                                          << countAckedSegments);
        }
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Test that the congestion avoidance of TcpNewReno counts the
 * acknowledged segments as RFC 3465 and Linux tcp_cong_avoid_ai() do with
 * the CountAckedSegments attribute set
 *
 * A window of acknowledged segments grows the window by exactly one segment,
 * whatever the segment size, and the count is cleared when the window is
 * reduced, as Linux tcp_init_cwnd_reduction() clears snd_cwnd_cnt.
 */
class TcpNewRenoAckCountingExactTest : public TestCase
{
  public:
    TcpNewRenoAckCountingExactTest()
        : TestCase("TcpNewReno congestion avoidance counts acknowledged segments")
    {
    }

  private:
    void DoRun() override
    {
        const uint32_t mss = 1448;
        auto cong = CreateObject<TcpNewReno>();
        cong->SetAttribute("CountAckedSegments", BooleanValue(true));
        auto state = CreateObject<TcpSocketState>();
        state->m_segmentSize = mss;
        state->m_cWnd = 100 * mss;
        state->m_ssThresh = 50 * mss;

        for (uint32_t i = 0; i < 50; ++i)
        {
            cong->IncreaseWindow(state, SEGS_PER_ACK);
        }
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                              101 * mss,
                              "A window of acknowledged segments did not grow cwnd by one segment");

        for (uint32_t i = 0; i < 100; ++i)
        {
            cong->IncreaseWindow(state, 1);
        }
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(), 101 * mss, "Early window growth");

        cong->CongestionStateSet(state, TcpSocketState::CA_RECOVERY);
        state->m_ssThresh = 50 * mss;
        state->m_cWnd = 50 * mss;
        cong->CongestionStateSet(state, TcpSocketState::CA_OPEN);
        for (uint32_t i = 0; i < 49; ++i)
        {
            cong->IncreaseWindow(state, 1);
        }
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                              50 * mss,
                              "The count of acknowledged segments survived the recovery");
        cong->IncreaseWindow(state, 1);
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(), 51 * mss, "No growth after a window");
    }
};

/**
 * @ingroup internet-test
 *
 * @brief TestSuite for the CountAckedSegments attribute of TcpNewReno
 */
class TcpAckCountingTestSuite : public TestSuite
{
  public:
    TcpAckCountingTestSuite()
        : TestSuite("tcp-ack-counting", Type::UNIT)
    {
        AddTestCase(new TcpNewRenoAckCountingCaTest(), TestCase::Duration::QUICK);
        AddTestCase(new TcpNewRenoAckCountingExactTest(), TestCase::Duration::QUICK);

        // Models whose slow start is the one of TcpNewReno (TcpHybla applies
        // its increases once per ACK whatever the attribute, see tcp-hybla-test)
        for (const auto& name : {"ns3::TcpNewReno",
                                 "ns3::TcpWestwoodPlus",
                                 "ns3::TcpScalable",
                                 "ns3::TcpHighSpeed",
                                 "ns3::TcpHtcp",
                                 "ns3::TcpIllinois",
                                 "ns3::TcpYeah",
                                 "ns3::TcpVegas",
                                 "ns3::TcpVeno",
                                 "ns3::TcpLp"})
        {
            AddTestCase(new TcpAckCountingTest(TypeId::LookupByName(name), true, false),
                        TestCase::Duration::QUICK);
        }

        // Models whose congestion avoidance is the one of TcpNewReno at the
        // start of the connection, or follows its per-ACK rule (TcpHtcp)
        for (const auto& name : {"ns3::TcpNewReno",
                                 "ns3::TcpWestwoodPlus",
                                 "ns3::TcpVegas",
                                 "ns3::TcpVeno",
                                 "ns3::TcpLp",
                                 "ns3::TcpLedbat"})
        {
            AddTestCase(new TcpAckCountingTest(TypeId::LookupByName(name), false, false),
                        TestCase::Duration::QUICK);
        }
        AddTestCase(new TcpAckCountingTest(TypeId::LookupByName("ns3::TcpHtcp"), false, true),
                    TestCase::Duration::QUICK);
    }
};

static TcpAckCountingTestSuite
    g_tcpAckCountingTestSuite; //!< Static variable for test initialization
