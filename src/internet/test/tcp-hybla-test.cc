/*
 * Copyright (c) 2015 Natale Patriciello, <natale.patriciello@gmail.com>
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-hybla.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/test.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpHyblaTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief Testing the congestion avoidance increment on TcpHybla
 */
class TcpHyblaIncrementTest : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param cWnd Congestion window.
     * @param ssThresh Slow Start Threshold.
     * @param segmentSize Segment size.
     * @param rtt Round trip time.
     * @param name Test description.
     */
    TcpHyblaIncrementTest(uint32_t cWnd,
                          uint32_t ssThresh,
                          uint32_t segmentSize,
                          const Time& rtt,
                          const std::string& name);

  private:
    void DoRun() override;

    /**
     * @brief Tracks TCP Hybla rho parameter changes.
     * @param oldVal Previous value.
     * @param newVal Actual value.
     */
    void RhoUpdated(double oldVal, double newVal);

    uint32_t m_cWnd;             //!< Congestion window.
    uint32_t m_ssThresh;         //!< Slow Start Threshold.
    uint32_t m_segmentSize;      //!< Segment size.
    Time m_rtt;                  //!< Round trip time.
    double m_rho;                //!< TCP Hybla rho parameter.
    Ptr<TcpSocketState> m_state; //!< TCP socket state.
};

TcpHyblaIncrementTest::TcpHyblaIncrementTest(uint32_t cWnd,
                                             uint32_t ssThresh,
                                             uint32_t segmentSize,
                                             const Time& rtt,
                                             const std::string& name)
    : TestCase(name),
      m_cWnd(cWnd),
      m_ssThresh(ssThresh),
      m_segmentSize(segmentSize),
      m_rtt(rtt),
      m_rho(0)
{
}

void
TcpHyblaIncrementTest::RhoUpdated(double /* oldVal */, double newVal)
{
    m_rho = newVal;
}

void
TcpHyblaIncrementTest::DoRun()
{
    m_state = CreateObject<TcpSocketState>();

    m_state->m_cWnd = m_cWnd;
    m_state->m_ssThresh = m_ssThresh;
    m_state->m_segmentSize = m_segmentSize;
    m_state->m_minRtt = m_rtt;

    Ptr<TcpHybla> cong = CreateObject<TcpHybla>();
    cong->SetAttribute("RRTT", TimeValue(MilliSeconds(50)));

    cong->TraceConnectWithoutContext("Rho", MakeCallback(&TcpHyblaIncrementTest::RhoUpdated, this));

    // Each received ACK weight is "coeffA". To see an increase of 1 MSS, we need
    // to ACK at least segCwnd/coeffA ACK.

    TimeValue rRtt;
    cong->GetAttribute("RRTT", rRtt);
    cong->PktsAcked(m_state, 1, m_rtt);

    double calcRho = std::max(m_rtt.GetSeconds() / rRtt.Get().GetSeconds(), 1.0);

    NS_TEST_ASSERT_MSG_NE(m_rho, 0.0, "Rho never updated by implementation");
    NS_TEST_ASSERT_MSG_EQ_TOL(calcRho,
                              m_rho,
                              0.01,
                              "Different rho values between implementation and test");

    cong->IncreaseWindow(m_state, 1);

    if (m_cWnd <= m_ssThresh)
    {
        // We expect an increment of 2^rho - 1, which does not go beyond ssThresh
        double inc = std::pow(2, calcRho) - 1.0;
        uint32_t cWndExpected = m_cWnd + (inc * m_segmentSize);
        NS_TEST_ASSERT_MSG_LT_OR_EQ(m_state->m_cWnd.Get(),
                                    m_state->m_ssThresh.Get(),
                                    "Congestion window has gone too far");
        NS_TEST_ASSERT_MSG_EQ(m_state->m_cWnd.Get(),
                              cWndExpected,
                              "Congestion window different than expected");
    }
    else
    {
        // We expect an increment of rho^2 / cWnd
        uint32_t segCwnd = m_cWnd / m_segmentSize;
        double inc = std::pow(m_rho, 2) / ((double)segCwnd);
        uint32_t cWndExpected = m_cWnd + (inc * m_segmentSize);

        if (inc >= 1.0)
        {
            // LT because implementation does not add value less than MSS.
            NS_TEST_ASSERT_MSG_LT_OR_EQ(m_state->m_cWnd.Get(),
                                        cWndExpected,
                                        "Congestion window different than expected");
        }
    }
}

/**
 * @ingroup internet-test
 *
 * @brief Testing the default reference RTT and the slow start bound of TcpHybla
 *
 * The reference RTT is 25 ms, as in the Hybla paper and in Linux (rtt0), and
 * the slow start increment 2^rho - 1 has the integer part of rho bounded to
 * 16, as in Linux, so that it does not overflow for long RTTs.
 */
class TcpHyblaDefaultTest : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param rtt Minimum RTT.
     * @param expectedRho Expected rho.
     * @param expectedIncrement Expected slow start increment (segments).
     * @param name Test description.
     */
    TcpHyblaDefaultTest(Time rtt,
                        double expectedRho,
                        uint32_t expectedIncrement,
                        const std::string& name)
        : TestCase(name),
          m_rtt(rtt),
          m_expectedRho(expectedRho),
          m_expectedIncrement(expectedIncrement)
    {
    }

  private:
    void DoRun() override;

    /**
     * @brief Tracks TCP Hybla rho parameter changes.
     * @param oldVal Previous value.
     * @param newVal Actual value.
     */
    void RhoUpdated(double oldVal, double newVal);

    Time m_rtt;                   //!< Minimum RTT.
    double m_expectedRho;         //!< Expected rho.
    uint32_t m_expectedIncrement; //!< Expected slow start increment (segments).
    double m_rho{0};              //!< TCP Hybla rho parameter.
};

void
TcpHyblaDefaultTest::RhoUpdated(double oldVal, double newVal)
{
    m_rho = newVal;
}

void
TcpHyblaDefaultTest::DoRun()
{
    const uint32_t segmentSize = 500;
    Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
    state->m_cWnd = segmentSize;
    state->m_ssThresh = UINT32_MAX;
    state->m_segmentSize = segmentSize;
    state->m_minRtt = m_rtt;

    Ptr<TcpHybla> cong = CreateObject<TcpHybla>();
    cong->TraceConnectWithoutContext("Rho", MakeCallback(&TcpHyblaDefaultTest::RhoUpdated, this));
    cong->PktsAcked(state, 1, m_rtt);
    NS_TEST_ASSERT_MSG_EQ_TOL(m_rho, m_expectedRho, 0.01, "Unexpected rho");

    cong->IncreaseWindow(state, 1);
    NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                          segmentSize + m_expectedIncrement * segmentSize,
                          "Unexpected slow start increment");
}

/**
 * @ingroup internet-test
 *
 * @brief Testing the per-ACK increases of TcpHybla and its Linux options
 *
 * The increases of slow start (2^rho - 1) and congestion avoidance
 * (rho^2 / cwnd) apply once per ACK, as in the Hybla paper and in Linux,
 * even with the CountAckedSegments attribute of TcpNewReno set. A forked
 * instance keeps the reference RTT. The RenoOutsideOpen, RhoInitialWindow
 * and FixedPoint attributes follow Linux hybla_cong_avoid(), hybla_init()
 * and the fixed-point arithmetic of Linux, respectively; with FixedPoint, the
 * slow start threshold is half the window in whole segments, as Linux
 * tcp_reno_ssthresh().
 */
class TcpHyblaLinuxTest : public TestCase
{
  public:
    /// The behavior to check
    enum Check
    {
        SLOW_START_PER_ACK,   //!< One slow start increase for an ACK of two segments
        CONG_AVOID_PER_ACK,   //!< One congestion avoidance increase for an ACK of two segments
        FORK,                 //!< A forked instance keeps the reference RTT
        RENO_OUTSIDE_OPEN,    //!< NewReno slow start in the Loss state
        RHO_INITIAL_WINDOW,   //!< Initial window of rho segments
        FIXED_POINT,          //!< Increments truncated to 1/128 of a segment
        FIXED_POINT_SSTHRESH, //!< Slow start threshold of half the window, in segments
    };

    /**
     * @brief Constructor.
     * @param check The behavior to check.
     * @param name Test description.
     */
    TcpHyblaLinuxTest(Check check, const std::string& name)
        : TestCase(name),
          m_check(check)
    {
    }

  private:
    void DoRun() override;

    Check m_check; //!< The behavior to check
};

void
TcpHyblaLinuxTest::DoRun()
{
    const uint32_t segmentSize = 1000;
    Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
    state->m_segmentSize = segmentSize;
    state->m_ssThresh = UINT32_MAX;
    state->m_cWnd = 10 * segmentSize;
    // rho = 81.25 / 25 = 3.25: 2^rho - 1 = 8.51 segments in slow start, and
    // rho^2 = 10.56, i.e., 0.1056 segment per ACK in congestion avoidance
    // with a window of 100 segments
    state->m_minRtt = MicroSeconds(81250);

    Ptr<TcpHybla> cong = CreateObject<TcpHybla>();
    cong->SetAttribute("CountAckedSegments", BooleanValue(true));

    switch (m_check)
    {
    case SLOW_START_PER_ACK: {
        cong->PktsAcked(state, 2, state->m_minRtt);
        cong->IncreaseWindow(state, 2);
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                              10 * segmentSize + 8513,
                              "The slow start increase must apply once per ACK");
        break;
    }
    case CONG_AVOID_PER_ACK: {
        state->m_ssThresh = 100 * segmentSize;
        state->m_cWnd = 100 * segmentSize;
        cong->PktsAcked(state, 2, state->m_minRtt);
        // Nine ACKs of two segments each add 0.95 segment, which would be a
        // full segment if each acknowledged segment counted
        for (uint32_t i = 0; i < 9; ++i)
        {
            cong->IncreaseWindow(state, 2);
        }
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                              100 * segmentSize,
                              "The congestion avoidance increase must apply once per ACK");
        cong->IncreaseWindow(state, 2);
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                              101 * segmentSize,
                              "Ten ACKs must add a segment");
        break;
    }
    case FORK: {
        cong->SetAttribute("RRTT", TimeValue(MilliSeconds(50)));
        Ptr<TcpCongestionOps> forked = cong->Fork();
        double rho = 0;
        forked->TraceConnectWithoutContext(
            "Rho",
            MakeBoundCallback(+[](double* rho, double, double newVal) { *rho = newVal; }, &rho));
        forked->PktsAcked(state, 1, state->m_minRtt);
        NS_TEST_ASSERT_MSG_EQ_TOL(rho, 1.625, 1e-9, "The forked instance lost the reference RTT");
        break;
    }
    case RENO_OUTSIDE_OPEN: {
        cong->SetAttribute("RenoOutsideOpen", BooleanValue(true));
        cong->PktsAcked(state, 1, state->m_minRtt);
        cong->CongestionStateSet(state, TcpSocketState::CA_LOSS);
        cong->IncreaseWindow(state, 2);
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                              12 * segmentSize,
                              "NewReno slow start expected outside the Open state");
        cong->CongestionStateSet(state, TcpSocketState::CA_OPEN);
        cong->IncreaseWindow(state, 2);
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                              12 * segmentSize + 8513,
                              "Hybla slow start expected in the Open state");
        break;
    }
    case RHO_INITIAL_WINDOW: {
        cong->CwndEvent(state, TcpSocketState::CA_EVENT_TX_START);
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                              10 * segmentSize,
                              "The initial window must not change by default");
        cong->SetAttribute("RhoInitialWindow", BooleanValue(true));
        cong->CwndEvent(state, TcpSocketState::CA_EVENT_TX_START);
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                              3 * segmentSize,
                              "The initial window must be the integer part of rho");
        state->m_cWnd = 20 * segmentSize;
        cong->CwndEvent(state, TcpSocketState::CA_EVENT_TX_START);
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                              20 * segmentSize,
                              "Only the first transmission sets the initial window");
        break;
    }
    case FIXED_POINT: {
        cong->SetAttribute("FixedPoint", BooleanValue(true));
        cong->PktsAcked(state, 1, state->m_minRtt);
        // rho = 26/8: 2^3 * 152/128 - 1 = 8.5 segments per ACK, whose half
        // segments accumulate
        cong->IncreaseWindow(state, 1);
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(), 18 * segmentSize, "Unexpected slow start");
        cong->IncreaseWindow(state, 1);
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(), 27 * segmentSize, "Unexpected slow start");
        // (26 * 26 * 2) / 200 = 6 (instead of 6.76) in 1/128 of a segment
        // per ACK: 22 ACKs make a segment, instead of 19
        state->m_ssThresh = 200 * segmentSize;
        state->m_cWnd = 200 * segmentSize;
        for (uint32_t i = 0; i < 21; ++i)
        {
            cong->IncreaseWindow(state, 1);
        }
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                              200 * segmentSize,
                              "The congestion avoidance increment must be truncated");
        cong->IncreaseWindow(state, 1);
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                              201 * segmentSize,
                              "Unexpected congestion avoidance");
        break;
    }
    case FIXED_POINT_SSTHRESH: {
        state->m_cWnd = 205 * segmentSize;
        NS_TEST_ASSERT_MSG_EQ(cong->GetSsThresh(state, 205 * segmentSize),
                              102 * segmentSize + segmentSize / 2,
                              "The default halves the flight in bytes");
        cong->SetAttribute("FixedPoint", BooleanValue(true));
        NS_TEST_ASSERT_MSG_EQ(cong->GetSsThresh(state, 205 * segmentSize),
                              102 * segmentSize,
                              "The window must be halved in whole segments");
        NS_TEST_ASSERT_MSG_EQ(cong->GetSsThresh(state, 100 * segmentSize),
                              102 * segmentSize,
                              "The threshold must follow the window, not the flight");
        break;
    }
    }
}

/**
 * @ingroup internet-test
 *
 * @brief TCP Hybla TestSuite
 */
class TcpHyblaTestSuite : public TestSuite
{
  public:
    TcpHyblaTestSuite()
        : TestSuite("tcp-hybla-test", Type::UNIT)
    {
        AddTestCase(new TcpHyblaIncrementTest(1000,
                                              0xFFFFFFFF,
                                              500,
                                              MilliSeconds(55),
                                              "Rho=1.1, slow start"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpHyblaIncrementTest(1000,
                                              0xFFFFFFFF,
                                              500,
                                              MilliSeconds(100),
                                              "Rho=2, slow start"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpHyblaIncrementTest(1000,
                                              0xFFFFFFFF,
                                              500,
                                              MilliSeconds(750),
                                              "Rho=30, slow start"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpHyblaIncrementTest(1000, 500, 500, Seconds(0.55), "Rho=1.1, cong avoid"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpHyblaIncrementTest(1000, 500, 500, Seconds(0.1), "Rho=2, cong avoid"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpHyblaIncrementTest(1000, 500, 500, Seconds(0.75), "Rho=30, cong avoid"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpHyblaDefaultTest(MilliSeconds(100),
                                            4,
                                            15,
                                            "Default reference RTT of 25 ms, rho=4"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpHyblaDefaultTest(MilliSeconds(750),
                                            30,
                                            65535,
                                            "Slow start increment bounded for rho=30"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpHyblaLinuxTest(TcpHyblaLinuxTest::SLOW_START_PER_ACK,
                                          "Slow start increase once per ACK"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpHyblaLinuxTest(TcpHyblaLinuxTest::CONG_AVOID_PER_ACK,
                                          "Congestion avoidance increase once per ACK"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpHyblaLinuxTest(TcpHyblaLinuxTest::FORK,
                                          "Forked instance keeps the reference RTT"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpHyblaLinuxTest(TcpHyblaLinuxTest::RENO_OUTSIDE_OPEN,
                                          "NewReno growth outside the Open state"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpHyblaLinuxTest(TcpHyblaLinuxTest::RHO_INITIAL_WINDOW,
                                          "Initial window of rho segments"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpHyblaLinuxTest(TcpHyblaLinuxTest::FIXED_POINT,
                                          "Fixed-point increments of Linux"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpHyblaLinuxTest(TcpHyblaLinuxTest::FIXED_POINT_SSTHRESH,
                                          "Slow start threshold of Linux"),
                    TestCase::Duration::QUICK);
    }
};

static TcpHyblaTestSuite g_tcpHyblaTest; //!< Static variable for test initialization
