/*
 * Copyright (c) 2015 ResiliNets, ITTC, University of Kansas
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Amir Modarresi <amodarresi@ittc.ku.edu>

 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  https://resilinets.org/
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 *
 */

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-htcp.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpHtcpTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief Testing the congestion avoidance increment on TcpHtcp
 */
class TcpHtcpIncrementTest : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param cWnd Congestion window.
     * @param segmentSize Segment size.
     * @param segmentsAcked Segments already ACKed.
     * @param lastCongestion Last congestion time.
     * @param firstAck First ACK time.
     * @param secondAck Second ACK time.
     * @param expectedCwnd Expected cWnd.
     * @param name Test description.
     */
    TcpHtcpIncrementTest(uint32_t cWnd,
                         uint32_t segmentSize,
                         uint32_t segmentsAcked,
                         Time lastCongestion,
                         Time firstAck,
                         Time secondAck,
                         uint32_t expectedCwnd,
                         const std::string& name);

  private:
    void DoRun() override;

    uint32_t m_cWnd;             //!< Congestion window.
    uint32_t m_segmentSize;      //!< Segment size.
    uint32_t m_segmentsAcked;    //!< Segments already ACKed.
    Time m_lastCongestion;       //!< Last congestion time.
    Time m_firstAck;             //!< First ACK time.
    Time m_secondAck;            //!< Second ACK time.
    uint32_t m_expectedCwnd;     //!< Expected cWnd.
    Ptr<TcpSocketState> m_state; //!< TCP socket state.
};

TcpHtcpIncrementTest::TcpHtcpIncrementTest(uint32_t cWnd,
                                           uint32_t segmentSize,
                                           uint32_t segmentsAcked,
                                           Time lastCongestion,
                                           Time firstAck,
                                           Time secondAck,
                                           uint32_t expectedCwnd,
                                           const std::string& name)
    : TestCase(name),
      m_cWnd(cWnd),
      m_segmentSize(segmentSize),
      m_segmentsAcked(segmentsAcked),
      m_lastCongestion(lastCongestion),
      m_firstAck(firstAck),
      m_secondAck(secondAck),
      m_expectedCwnd(expectedCwnd)
{
}

/**
 * @brief Since the calculation depends on the throughput and its associated
 * timing, we schedule a few exact events. We get the value from HTCP methods
 * during the simulation and compare them with their associated expected
 * values calculated from the algorithm by hand.
 */
void
TcpHtcpIncrementTest::DoRun()
{
    NS_LOG_FUNCTION(this);
    m_state = CreateObject<TcpSocketState>();

    m_state->m_cWnd = m_cWnd;
    m_state->m_segmentSize = m_segmentSize;

    Ptr<TcpHtcp> cong = CreateObject<TcpHtcp>();
    Time lastCongestion;

    NS_LOG_DEBUG("m_cWnd: " << m_cWnd << " m_segmentSize: " << m_segmentSize << " m_segmentsAcked: "
                            << m_segmentsAcked << " m_lastCongestion" << m_lastCongestion);
    Simulator::Schedule(Time(m_lastCongestion),
                        &TcpHtcp::GetSsThresh,
                        cong,
                        m_state,
                        m_state->m_cWnd);
    lastCongestion = m_lastCongestion;
    Simulator::Schedule(Time(m_firstAck),
                        &TcpHtcp::PktsAcked,
                        cong,
                        m_state,
                        m_segmentsAcked,
                        MilliSeconds(80));
    Simulator::Schedule(Time(m_secondAck),
                        &TcpHtcp::PktsAcked,
                        cong,
                        m_state,
                        m_segmentsAcked,
                        MilliSeconds(100));

    Simulator::Run();
    NS_LOG_DEBUG("Simulation ran for the scheduled events");

    cong->IncreaseWindow(m_state, m_segmentsAcked);
    NS_LOG_DEBUG("m_cwnd from function: " << m_state->m_cWnd
                                          << " expected cWnd calculated: " << m_expectedCwnd);

    NS_TEST_ASSERT_MSG_EQ(m_state->m_cWnd.Get(), m_expectedCwnd, "CWnd has not updated correctly");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 *
 * @brief Testing the adaptive backoff of TcpHtcp.
 *
 * Two congestion epochs with the same throughput, and RTT samples between
 * 95 and 100 ms, make the backoff factor the ratio of the minimum to the
 * maximum RTT, bounded to [0.5, 0.8] as in the H-TCP paper and in Linux:
 * the second loss sets the slow start threshold to 0.8 times the window.
 */
class TcpHtcpBackoffTest : public TestCase
{
  public:
    TcpHtcpBackoffTest()
        : TestCase("TcpHtcp adaptive backoff is bounded")
    {
    }

  private:
    void DoRun() override;

    /**
     * @brief Record the slow start threshold computed after a loss.
     * @param cong The congestion control
     * @param state The TCP socket state
     */
    void Loss(Ptr<TcpHtcp> cong, Ptr<TcpSocketState> state);

    uint32_t m_ssThresh{0}; //!< Last slow start threshold
};

void
TcpHtcpBackoffTest::Loss(Ptr<TcpHtcp> cong, Ptr<TcpSocketState> state)
{
    m_ssThresh = cong->GetSsThresh(state, state->m_cWnd);
}

void
TcpHtcpBackoffTest::DoRun()
{
    Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
    state->m_segmentSize = 1000;
    state->m_cWnd = 100 * 1000;
    Ptr<TcpHtcp> cong = CreateObject<TcpHtcp>();

    for (uint32_t epoch = 0; epoch < 3; epoch++)
    {
        Time start = Seconds(epoch);
        Simulator::Schedule(start + MilliSeconds(1), &TcpHtcpBackoffTest::Loss, this, cong, state);
        Simulator::Schedule(start + MilliSeconds(500),
                            &TcpHtcp::PktsAcked,
                            cong,
                            state,
                            100,
                            MilliSeconds(95));
        Simulator::Schedule(start + MilliSeconds(999),
                            &TcpHtcp::PktsAcked,
                            cong,
                            state,
                            100,
                            MilliSeconds(100));
    }
    Simulator::Stop(Seconds(2.5));
    Simulator::Run();
    Simulator::Destroy();

    NS_TEST_ASSERT_MSG_EQ(m_ssThresh, 80 * 1000, "The backoff factor is not bounded to 0.8");
}

/**
 * @ingroup internet-test
 *
 * @brief Testing the RTT scaling of the increase function of TcpHtcp.
 *
 * Two seconds after a congestion event, i.e., one second after Delta_L, the
 * increase function gives alpha = 2 * (1 - beta) * (1 + 10 + 0.25) = 11.25
 * segments per RTT with beta = 0.5. With the RttScaling attribute, as in
 * Linux, alpha is scaled by the minimum RTT over 100 ms, bounded to
 * [0.5, 10].
 */
class TcpHtcpRttScalingTest : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param rttScaling Whether the RttScaling attribute is set.
     * @param rtt The RTT of the ACKs.
     * @param expectedIncrease The expected window increase of an ACK, in bytes.
     * @param name Test description.
     */
    TcpHtcpRttScalingTest(bool rttScaling,
                          Time rtt,
                          uint32_t expectedIncrease,
                          const std::string& name)
        : TestCase(name),
          m_rttScaling(rttScaling),
          m_rtt(rtt),
          m_expectedIncrease(expectedIncrease)
    {
    }

  private:
    void DoRun() override;

    bool m_rttScaling;           //!< Whether the RttScaling attribute is set
    Time m_rtt;                  //!< RTT of the ACKs
    uint32_t m_expectedIncrease; //!< Expected window increase of an ACK
};

void
TcpHtcpRttScalingTest::DoRun()
{
    const uint32_t segmentSize = 1000;
    Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
    state->m_segmentSize = segmentSize;
    state->m_cWnd = 100 * segmentSize;
    state->m_ssThresh = 50 * segmentSize;

    Ptr<TcpHtcp> cong = CreateObject<TcpHtcp>();
    cong->SetAttribute("RttScaling", BooleanValue(m_rttScaling));

    Simulator::Schedule(MilliSeconds(1), &TcpHtcp::GetSsThresh, cong, state, state->m_cWnd);
    Simulator::Schedule(Seconds(2) + MilliSeconds(1), &TcpHtcp::PktsAcked, cong, state, 1, m_rtt);
    Simulator::Run();

    cong->IncreaseWindow(state, 1);
    NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                          100 * segmentSize + m_expectedIncrease,
                          "Unexpected window increase");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 *
 * @brief Testing the Linux arithmetic of TcpHtcp (LinuxHz attribute)
 *
 * As Linux htcp_cong_avoid(), each ACK adds the segments it acknowledged in
 * CA_OPEN to a count, and the window grows by a segment on the ACK that
 * finds the count, times alpha (1 at first, in 1/128), at least the window:
 * with ACKs of two segments, the 6th ACK grows a window of 10 segments. The
 * backoff, 0.5 at first, applies to cWnd (htcp_recalc_ssthresh()), not to
 * the bytes in flight.
 */
class TcpHtcpLinuxHzTest : public TestCase
{
  public:
    TcpHtcpLinuxHzTest()
        : TestCase("Linux arithmetic of H-TCP")
    {
    }

  private:
    void DoRun() override
    {
        const uint32_t segmentSize = 1000;
        Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
        state->m_segmentSize = segmentSize;
        state->m_cWnd = 10 * segmentSize;
        state->m_ssThresh = 5 * segmentSize;

        Ptr<TcpHtcp> cong = CreateObject<TcpHtcp>();
        cong->SetAttribute("LinuxHz", UintegerValue(100));

        Simulator::Schedule(Seconds(1), [this, cong, state]() {
            cong->Init(state);
            for (uint32_t i = 0; i < 5; ++i)
            {
                cong->PktsAcked(state, 2, MilliSeconds(100));
                cong->IncreaseWindow(state, 2);
            }
            NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(), 10 * segmentSize, "Early increase");
            cong->PktsAcked(state, 2, MilliSeconds(100));
            cong->IncreaseWindow(state, 2);
            NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(), 11 * segmentSize, "No increase");
            NS_TEST_ASSERT_MSG_EQ(cong->GetSsThresh(state, 4 * segmentSize),
                                  5 * segmentSize,
                                  "The backoff does not apply to cWnd");
        });
        Simulator::Run();
        Simulator::Destroy();
    }
};

/**
 * @ingroup internet-test
 *
 * @brief TCP Htcp TestSuite.
 *
 * The following tests simulate conditions after a congestion event and
 * return to 1/2 ssthresh. After that, two acks are scheduled and the
 * value of the cWnd is compared at the end of the event.
 * The values in each test have been chosen randomly. The first test
 * simulates receiving acks for 38 packets with segmentSize=536,
 * the second one receives ack for 100 packets with segmentSize=1 and
 * the third one receives ack for 50 segment with segmentSize=1446.
 * The cWnd values of 20382, 39 and 76803 have been
 * calculated manually from the algorithm.
 */

class TcpHtcpTestSuite : public TestSuite
{
  public:
    TcpHtcpTestSuite()
        : TestSuite("tcp-htcp-test", Type::UNIT)
    {
        AddTestCase(new TcpHtcpIncrementTest(38 * 536,
                                             536,
                                             38,
                                             ns3::MilliSeconds(1),
                                             ns3::MilliSeconds(900),
                                             ns3::MilliSeconds(1000),
                                             20382,
                                             "TcpHtcp increment test on cWnd "),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpHtcpIncrementTest(38,
                                             1,
                                             100,
                                             ns3::MilliSeconds(1),
                                             ns3::MilliSeconds(900),
                                             ns3::MilliSeconds(1100),
                                             39,
                                             "TcpHtcp increment test on cWnd "),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpHtcpIncrementTest(53 * 1446,
                                             1446,
                                             50,
                                             ns3::MilliSeconds(1),
                                             ns3::MilliSeconds(900),
                                             ns3::MilliSeconds(1500),
                                             76803,
                                             "TcpHtcp increment test on cWnd "),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpHtcpBackoffTest(), TestCase::Duration::QUICK);
        AddTestCase(new TcpHtcpLinuxHzTest(), TestCase::Duration::QUICK);
        AddTestCase(
            new TcpHtcpRttScalingTest(false, MilliSeconds(40), 112, "No RTT scaling by default"),
            TestCase::Duration::QUICK);
        AddTestCase(
            new TcpHtcpRttScalingTest(true, MilliSeconds(40), 56, "RTT scaling bounded to 0.5"),
            TestCase::Duration::QUICK);
        AddTestCase(
            new TcpHtcpRttScalingTest(true, MilliSeconds(200), 225, "RTT scaling by 2 at 200 ms"),
            TestCase::Duration::QUICK);
    }
};

static TcpHtcpTestSuite g_TcpHtcpTest; //!< Static variable for test initialization
