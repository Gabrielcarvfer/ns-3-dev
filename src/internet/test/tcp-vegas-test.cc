/*
 * Copyright (c) 2016 ResiliNets, ITTC, University of Kansas
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Truc Anh N. Nguyen <annguyen@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  https://resilinets.org/
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 */

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-vegas.h"
#include "ns3/test.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpVegasTestSuite");

/**
 * @brief TcpVegas congestion control algorithm test
 */
class TcpVegasTest : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param cWnd Congestion window.
     * @param segmentSize Segment size.
     * @param ssThresh Slow Start Threshold.
     * @param rtt The RTT.
     * @param segmentsAcked Number of segments ACKed.
     * @param nextTxSeq Next Tx sequence number.
     * @param lastAckedSeq Last ACKed sequence number.
     * @param name Test description.
     */
    TcpVegasTest(uint32_t cWnd,
                 uint32_t segmentSize,
                 uint32_t ssThresh,
                 Time rtt,
                 uint32_t segmentsAcked,
                 SequenceNumber32 nextTxSeq,
                 SequenceNumber32 lastAckedSeq,
                 const std::string& name);

  private:
    void DoRun() override;
    /**
     * @brief Increases the TCP window.
     * @param cong The congestion control.
     */
    void IncreaseWindow(Ptr<TcpVegas> cong);
    /**
     * brief Get and check the SSH threshold.
     * @param cong The congestion control.
     */
    void GetSsThresh(Ptr<TcpVegas> cong);

    uint32_t m_cWnd;                 //!< Congestion window.
    uint32_t m_segmentSize;          //!< Segment size.
    uint32_t m_ssThresh;             //!< Slow Start Threshold.
    Time m_rtt;                      //!< RTT.
    uint32_t m_segmentsAcked;        //!< Number of segments ACKed.
    SequenceNumber32 m_nextTxSeq;    //!< Next Tx sequence number.
    SequenceNumber32 m_lastAckedSeq; //!< Last ACKed sequence number.

    Ptr<TcpSocketState> m_state; //!< TCP socket state.
};

TcpVegasTest::TcpVegasTest(uint32_t cWnd,
                           uint32_t segmentSize,
                           uint32_t ssThresh,
                           Time rtt,
                           uint32_t segmentsAcked,
                           SequenceNumber32 nextTxSeq,
                           SequenceNumber32 lastAckedSeq,
                           const std::string& name)
    : TestCase(name),
      m_cWnd(cWnd),
      m_segmentSize(segmentSize),
      m_ssThresh(ssThresh),
      m_rtt(rtt),
      m_segmentsAcked(segmentsAcked),
      m_nextTxSeq(nextTxSeq),
      m_lastAckedSeq(lastAckedSeq)
{
}

void
TcpVegasTest::DoRun()
{
    m_state = CreateObject<TcpSocketState>();

    m_state->m_cWnd = m_cWnd;
    m_state->m_segmentSize = m_segmentSize;
    m_state->m_ssThresh = m_ssThresh;
    m_state->m_nextTxSequence = m_nextTxSeq;
    m_state->m_lastAckedSeq = m_lastAckedSeq;
    m_state->m_minRtt = m_rtt;

    Ptr<TcpVegas> cong = CreateObject<TcpVegas>();

    // Set baseRtt to 100 ms
    cong->PktsAcked(m_state, m_segmentsAcked, MilliSeconds(100));

    // Re-set Vegas to assign a new value of minRtt
    cong->CongestionStateSet(m_state, TcpSocketState::CA_OPEN);
    cong->PktsAcked(m_state, m_segmentsAcked, m_rtt);

    // 2 more calls to PktsAcked to increment cntRtt beyond 2
    cong->PktsAcked(m_state, m_segmentsAcked, m_rtt);
    cong->PktsAcked(m_state, m_segmentsAcked, m_rtt);

    // Update cwnd using Vegas algorithm
    cong->IncreaseWindow(m_state, m_segmentsAcked);

    // Our calculation of cwnd
    IncreaseWindow(cong);

    NS_TEST_ASSERT_MSG_EQ(m_state->m_cWnd.Get(), m_cWnd, "CWnd has not updated correctly");
    NS_TEST_ASSERT_MSG_EQ(m_state->m_ssThresh.Get(),
                          m_ssThresh,
                          "SsThresh has not updated correctly");
}

void
TcpVegasTest::IncreaseWindow(Ptr<TcpVegas> cong)
{
    Time baseRtt = MilliSeconds(100);
    uint32_t segCwnd = m_cWnd / m_segmentSize;

    // Calculate expected throughput
    uint64_t expectedCwnd;
    expectedCwnd =
        (uint64_t)segCwnd * (double)baseRtt.GetMilliSeconds() / (double)m_rtt.GetMilliSeconds();

    // Calculate the difference between actual and expected throughput
    uint32_t diff;
    diff = segCwnd - expectedCwnd;

    // Get the alpha,beta, and gamma attributes
    UintegerValue alpha;
    UintegerValue beta;
    UintegerValue gamma;
    cong->GetAttribute("Alpha", alpha);
    cong->GetAttribute("Beta", beta);
    cong->GetAttribute("Gamma", gamma);

    if (diff > gamma.Get() && (m_cWnd < m_ssThresh))
    { // Change from slow-start to linear increase/decrease mode
        segCwnd = std::min(segCwnd, (uint32_t)expectedCwnd + 1);
        m_cWnd = segCwnd * m_segmentSize;
        GetSsThresh(cong);
    }
    else if (m_cWnd < m_ssThresh)
    { // Execute Reno slow start
        if (m_segmentsAcked >= 1)
        {
            m_cWnd += m_segmentSize;
            m_segmentsAcked--;
        }
    }
    else
    { // Linear increase/decrease mode
        if (diff > beta.Get())
        {
            m_cWnd = (segCwnd - 1) * m_segmentSize;
            GetSsThresh(cong);
        }
        else if (diff < alpha.Get())
        {
            m_cWnd = (segCwnd + 1) * m_segmentSize;
        }
        else
        {
        }
    }
    m_ssThresh = std::max(m_ssThresh, 3 * m_cWnd / 4);
}

void
TcpVegasTest::GetSsThresh(Ptr<TcpVegas> cong)
{
    m_ssThresh = std::max(std::min(m_ssThresh, m_cWnd - m_segmentSize), 2 * m_segmentSize);
}

/**
 * @ingroup internet-test
 *
 * @brief Check that Vegas halves the window on a loss.
 *
 * Linux Vegas reacts to a loss as Reno does (tcp_reno_ssthresh()): the slow
 * start threshold becomes half the congestion window, whatever the slow start
 * threshold Vegas keeps when it adjusts the window by itself.
 */
class TcpVegasLossTest : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param cWnd Congestion window.
     * @param segmentSize Segment size.
     * @param ssThresh Slow Start Threshold before the loss.
     * @param name Test description.
     */
    TcpVegasLossTest(uint32_t cWnd,
                     uint32_t segmentSize,
                     uint32_t ssThresh,
                     const std::string& name)
        : TestCase(name),
          m_cWnd(cWnd),
          m_segmentSize(segmentSize),
          m_ssThresh(ssThresh)
    {
    }

  private:
    void DoRun() override;

    uint32_t m_cWnd;        //!< Congestion window.
    uint32_t m_segmentSize; //!< Segment size.
    uint32_t m_ssThresh;    //!< Slow Start Threshold before the loss.
};

void
TcpVegasLossTest::DoRun()
{
    Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
    state->m_cWnd = m_cWnd;
    state->m_segmentSize = m_segmentSize;
    state->m_ssThresh = m_ssThresh;

    Ptr<TcpVegas> cong = CreateObject<TcpVegas>();
    uint32_t ssThresh = cong->GetSsThresh(state, m_cWnd);

    NS_TEST_ASSERT_MSG_EQ(ssThresh,
                          std::max(m_cWnd / 2, 2 * m_segmentSize),
                          "A loss did not halve the window");
}

/**
 * @ingroup internet-test
 *
 * @brief Test the LinuxCongAvoid attribute of TcpVegas
 *
 * With a base RTT of 100 ms, an RTT of 103 ms in the cycle and a window of 50
 * segments, Diff is 2 by default (50 minus the target window of 48), so the
 * window does not change, and 1 (50 * 3 / 100, rounded down) with the attribute
 * set, so the window grows by a segment. The ACK of the data sent up to the
 * start of the cycle ends it by default; with the attribute set, the cycle
 * ends only with the ACK of data sent after its start.
 */
class TcpVegasLinuxCongAvoidTest : public TestCase
{
  public:
    /**
     * Constructor
     * @param linuxMode whether the LinuxCongAvoid attribute is set
     * @param name the test name
     */
    TcpVegasLinuxCongAvoidTest(bool linuxMode, const std::string& name)
        : TestCase(name),
          m_linux(linuxMode)
    {
    }

  private:
    void DoRun() override;

    bool m_linux; //!< LinuxCongAvoid attribute value
};

void
TcpVegasLinuxCongAvoidTest::DoRun()
{
    Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
    state->m_segmentSize = 1000;
    state->m_cWnd = 50000;
    state->m_ssThresh = 10000;
    state->m_nextTxSequence = SequenceNumber32(100000);

    Ptr<TcpVegas> cong = CreateObject<TcpVegas>();
    cong->SetAttribute("LinuxCongAvoid", BooleanValue(m_linux));

    cong->PktsAcked(state, 1, MilliSeconds(100));
    cong->CongestionStateSet(state, TcpSocketState::CA_OPEN);
    for (uint32_t i = 0; i < 3; ++i)
    {
        cong->PktsAcked(state, 1, MilliSeconds(103));
    }

    // ACK of the data sent up to the start of the cycle
    state->m_lastAckedSeq = SequenceNumber32(100000);
    cong->IncreaseWindow(state, 1);
    NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(), 50000, "Window changed at the cycle end");
    NS_TEST_ASSERT_MSG_EQ(state->m_ssThresh.Get(),
                          m_linux ? 10000 : 37500,
                          "Cycle end not detected as expected");

    // ACK of data sent after the start of the cycle
    state->m_lastAckedSeq = SequenceNumber32(101000);
    cong->IncreaseWindow(state, 1);
    // By default, this is the first ACK of a cycle without enough RTT
    // samples: NewReno adds 1000 * 1000 / 50000 bytes
    NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                          m_linux ? 51000 : 50020,
                          "Window not updated as expected");
}

/**
 * @ingroup internet-test
 *
 * @brief Test the LinuxIdleRestart attribute of TcpVegas
 *
 * The BaseRTT is 100 ms before the connection goes idle, and the RTT is 120 ms
 * once it sends again. With the attribute set, sending with no data in flight
 * restarts Vegas, BaseRTT included: the new BaseRTT is 120 ms, Diff is 0, and
 * the window grows by a segment. Without it, Diff is 50 minus 41 (the expected
 * window), above beta, and the window shrinks by a segment.
 */
class TcpVegasIdleRestartTest : public TestCase
{
  public:
    /**
     * Constructor
     * @param restart whether the LinuxIdleRestart attribute is set
     * @param name the test name
     */
    TcpVegasIdleRestartTest(bool restart, const std::string& name)
        : TestCase(name),
          m_restart(restart)
    {
    }

  private:
    void DoRun() override;

    bool m_restart; //!< LinuxIdleRestart attribute value
};

void
TcpVegasIdleRestartTest::DoRun()
{
    Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
    state->m_segmentSize = 1000;
    state->m_cWnd = 50000;
    state->m_ssThresh = 10000;
    state->m_nextTxSequence = SequenceNumber32(100000);

    Ptr<TcpVegas> cong = CreateObject<TcpVegas>();
    cong->SetAttribute("LinuxIdleRestart", BooleanValue(m_restart));

    cong->PktsAcked(state, 1, MilliSeconds(100));
    cong->CongestionStateSet(state, TcpSocketState::CA_OPEN);
    cong->CwndEvent(state, TcpSocketState::CA_EVENT_TX_START);
    for (uint32_t i = 0; i < 3; ++i)
    {
        cong->PktsAcked(state, 1, MilliSeconds(120));
    }
    state->m_lastAckedSeq = SequenceNumber32(100000);
    cong->IncreaseWindow(state, 1);
    NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                          m_restart ? 51000 : 49000,
                          "BaseRTT not handled as expected on the restart");
}

/**
 * @ingroup internet-test
 *
 * @brief TCP Vegas TestSuite
 */
class TcpVegasTestSuite : public TestSuite
{
  public:
    TcpVegasTestSuite()
        : TestSuite("tcp-vegas-test", Type::UNIT)
    {
        AddTestCase(
            new TcpVegasTest(38 * 1446,
                             1446,
                             40 * 1446,
                             MilliSeconds(106),
                             1,
                             SequenceNumber32(2893),
                             SequenceNumber32(5785),
                             "Vegas test on cWnd and ssThresh when in slow start and diff > gamma"),
            TestCase::Duration::QUICK);
        AddTestCase(
            new TcpVegasTest(5 * 536,
                             536,
                             10 * 536,
                             MilliSeconds(118),
                             1,
                             SequenceNumber32(3216),
                             SequenceNumber32(3753),
                             "Vegas test on cWnd and ssThresh when in slow start and diff < gamma"),
            TestCase::Duration::QUICK);
        AddTestCase(new TcpVegasTest(60 * 346,
                                     346,
                                     40 * 346,
                                     MilliSeconds(206),
                                     1,
                                     SequenceNumber32(20761),
                                     SequenceNumber32(21107),
                                     "Vegas test on cWnd and ssThresh when diff > beta"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpVegasTest(15 * 1446,
                                     1446,
                                     10 * 1446,
                                     MilliSeconds(106),
                                     1,
                                     SequenceNumber32(21691),
                                     SequenceNumber32(24583),
                                     "Vegas test on cWnd and ssThresh when diff < alpha"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpVegasTest(20 * 746,
                                     746,
                                     10 * 746,
                                     MilliSeconds(109),
                                     1,
                                     SequenceNumber32(14921),
                                     SequenceNumber32(15667),
                                     "Vegas test on cWnd and ssThresh when alpha <= diff <= beta"),
                    TestCase::Duration::QUICK);
        AddTestCase(
            new TcpVegasLossTest(70 * 1448, 1448, 68 * 1448, "Vegas halves the window on a loss"),
            TestCase::Duration::QUICK);
        AddTestCase(new TcpVegasLossTest(3 * 1448,
                                         1448,
                                         2 * 1448,
                                         "Vegas halves the window on a loss, small window"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpVegasLinuxCongAvoidTest(false, "Vegas cycle and Diff, default"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpVegasLinuxCongAvoidTest(true, "Vegas cycle and Diff, LinuxCongAvoid"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpVegasIdleRestartTest(false, "Vegas BaseRTT on a restart, default"),
                    TestCase::Duration::QUICK);
        AddTestCase(
            new TcpVegasIdleRestartTest(true, "Vegas BaseRTT on a restart, LinuxIdleRestart"),
            TestCase::Duration::QUICK);
    }
};

static TcpVegasTestSuite g_tcpVegasTest; //!< Static variable for test initialization
