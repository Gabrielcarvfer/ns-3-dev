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

#include "ns3/log.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-veno.h"
#include "ns3/test.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpVenoTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief Testing the additive increase and multiplicative decrease of TcpVeno
 */
class TcpVenoTest : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param cWnd Congestion window.
     * @param segmentSize Segment size.
     * @param ssThresh Slow Start Threshold.
     * @param rtt The RTT.
     * @param segmentsAcked Number of segments acked.
     * @param numRtt Number of RTTs (i.e., rounds) in the test.
     * @param name Test description.
     */
    TcpVenoTest(uint32_t cWnd,
                uint32_t segmentSize,
                uint32_t ssThresh,
                Time rtt,
                uint32_t segmentsAcked,
                uint32_t numRtt,
                const std::string& name);

  private:
    void DoRun() override;

    /**
     * @brief TCP Veno additive increase formula.
     * @param state The congestion control state.
     * @param diff The difference between actual and expected throughput.
     * @param beta TCP Veno beta param.
     */
    void AdditiveIncrease(Ptr<TcpSocketState> state, uint32_t diff, UintegerValue beta);

    /**
     * @brief TCP Veno multiplicative decrease formula.
     * @param diff The difference between actual and expected throughput.
     * @param beta TCP Veno beta param.
     * @param bytesInFlight Bytes in flight.
     * @returns The calculated SsThresh.
     */
    uint32_t MultiplicativeDecrease(uint32_t diff,
                                    const UintegerValue& beta,
                                    uint32_t bytesInFlight) const;

    /**
     * @brief Mimics the NewReno IncreaseWindow algorithm.
     * @param state TCP socket state.
     * @param segmentsAcked Number of segments acked.
     */
    void NewReno_IncreaseWindow(Ptr<TcpSocketState> state, uint32_t segmentsAcked);

    /**
     * @brief Mimics the NewReno SlowStart algorithm.
     * @param state TCP socket state.
     * @param segmentsAcked Number of segments acked.
     * @returns The number of segments that could be sent.
     */
    uint32_t NewReno_SlowStart(Ptr<TcpSocketState> state, uint32_t segmentsAcked);

    /**
     * @brief Mimics the NewReno Congestion Avoidance algorithm.
     * @param state TCP socket state.
     * @param segmentsAcked Number of segments acked.
     */
    void NewReno_CongestionAvoidance(Ptr<TcpSocketState> state, uint32_t segmentsAcked);

    uint32_t m_cWnd;             //!< Congestion window.
    uint32_t m_segmentSize;      //!< Segment size.
    uint32_t m_ssThresh;         //!< Slow Start Threshold.
    Time m_rtt;                  //!< RTT.
    uint32_t m_segmentsAcked;    //!< Number of segments ACKed.
    uint32_t m_numRtt;           //!< Number of RTT (i.e., rounds) of the test.
    bool m_inc;                  //!< Internal flag to increase every other round.
    Ptr<TcpSocketState> m_state; //!< TCP socket state.
};

TcpVenoTest::TcpVenoTest(uint32_t cWnd,
                         uint32_t segmentSize,
                         uint32_t ssThresh,
                         Time rtt,
                         uint32_t segmentsAcked,
                         uint32_t numRtt,
                         const std::string& name)
    : TestCase(name),
      m_cWnd(cWnd),
      m_segmentSize(segmentSize),
      m_ssThresh(ssThresh),
      m_rtt(rtt),
      m_segmentsAcked(segmentsAcked),
      m_numRtt(numRtt),
      m_inc(true)
{
}

void
TcpVenoTest::DoRun()
{
    m_state = CreateObject<TcpSocketState>();

    m_state->m_cWnd = m_cWnd;
    m_state->m_segmentSize = m_segmentSize;
    m_state->m_ssThresh = m_ssThresh;
    m_state->m_minRtt = m_rtt;

    Ptr<TcpVeno> cong = CreateObject<TcpVeno>();

    // Set baseRtt to 100 ms
    Time baseRtt = MilliSeconds(100);
    cong->PktsAcked(m_state, m_segmentsAcked, baseRtt);

    // Re-set Veno to assign a new value of minRtt
    cong->CongestionStateSet(m_state, TcpSocketState::CA_OPEN);

    uint32_t segCwnd = m_cWnd / m_segmentSize;

    // Calculate expected throughput
    uint32_t expectedCwnd;
    double tmp = baseRtt.GetSeconds() / m_rtt.GetSeconds();
    expectedCwnd = segCwnd * tmp;

    // Calculate the difference between actual and expected throughput
    uint32_t diff;
    diff = segCwnd - expectedCwnd;

    // Get the beta attribute
    UintegerValue beta;
    cong->GetAttribute("Beta", beta);

    uint32_t cntRtt = 0;

    TcpSocketState state;
    state.m_cWnd = m_cWnd;
    state.m_ssThresh = m_ssThresh;
    state.m_segmentSize = m_segmentSize;

    while (m_numRtt != 0)
    {
        // Update cwnd using Veno's additive increase algorithm
        cong->PktsAcked(m_state, m_segmentsAcked, m_rtt);
        cong->IncreaseWindow(m_state, m_segmentsAcked);

        // The first round the internal m_diff of cong will be 4, just like us
        if (cntRtt == 0)
        {
            // Update ssthresh using Veno's multiplicative decrease algorithm
            uint32_t ssThresh = cong->GetSsThresh(m_state, m_state->m_cWnd);

            // Our calculation of ssthresh
            uint32_t calculatedSsThresh = MultiplicativeDecrease(diff, beta, m_state->m_cWnd.Get());

            NS_TEST_ASSERT_MSG_EQ(ssThresh,
                                  calculatedSsThresh,
                                  "Veno has not decremented cWnd correctly based on its"
                                  "multiplicative decrease algo.");
        }

        // Our calculation of cwnd. Veno counts the RTT samples since the start
        // of the connection: the one setting baseRtt, and one per ACK
        if (cntRtt + 2 <= 2)
        {
            NewReno_IncreaseWindow(&state, 1);
        }
        else
        {
            AdditiveIncrease(&state, diff, beta);
        }

        NS_TEST_ASSERT_MSG_EQ(
            m_state->m_cWnd.Get(),
            state.m_cWnd.Get(),
            "CWnd has not updated correctly based on Veno linear increase algorithm");
        m_numRtt--;
        cntRtt++;
    }
}

void
TcpVenoTest::AdditiveIncrease(Ptr<TcpSocketState> state, uint32_t diff, UintegerValue beta)
{
    if (m_cWnd < m_ssThresh)
    { // Slow start
        NewReno_SlowStart(state, 1);
    }
    else
    { // Congestion avoidance
        if (diff < beta.Get())
        { // Increase cwnd by 1 every RTT when bandwidth is not fully utilized
            NewReno_CongestionAvoidance(state, 1);
        }
        else
        { // Increase cwnd by 1 every other RTT when bandwidth is fully utilized
            if (m_inc)
            {
                NewReno_CongestionAvoidance(state, 1);
                m_inc = false;
            }
            else
            {
                m_inc = true;
            }
        }
    }
}

uint32_t
TcpVenoTest::MultiplicativeDecrease(uint32_t diff,
                                    const UintegerValue& beta,
                                    uint32_t bytesInFlight) const
{
    uint32_t calculatedSsThresh;
    if (diff < beta.Get())
    {
        // The window, in segments, is reduced to 4/5
        calculatedSsThresh = std::max(2U, bytesInFlight / m_segmentSize * 4 / 5) * m_segmentSize;
    }
    else
    {
        calculatedSsThresh = std::max(2 * m_segmentSize, bytesInFlight / 2);
    }
    return calculatedSsThresh;
}

void
TcpVenoTest::NewReno_IncreaseWindow(Ptr<TcpSocketState> state, uint32_t segmentsAcked)
{
    if (state->m_cWnd < state->m_ssThresh)
    {
        segmentsAcked = NewReno_SlowStart(state, segmentsAcked);
    }

    if (state->m_cWnd >= state->m_ssThresh)
    {
        NewReno_CongestionAvoidance(state, segmentsAcked);
    }
}

uint32_t
TcpVenoTest::NewReno_SlowStart(Ptr<TcpSocketState> state, uint32_t segmentsAcked)
{
    if (segmentsAcked >= 1)
    {
        state->m_cWnd += state->m_segmentSize;
        return segmentsAcked - 1;
    }

    return 0;
}

void
TcpVenoTest::NewReno_CongestionAvoidance(Ptr<TcpSocketState> state, uint32_t segmentsAcked)
{
    if (segmentsAcked > 0)
    {
        double adder =
            static_cast<double>(state->m_segmentSize * state->m_segmentSize) / state->m_cWnd.Get();
        adder = std::max(1.0, adder);
        state->m_cWnd += static_cast<uint32_t>(adder);
    }
}

/**
 * @ingroup internet-test
 *
 * @brief Testing the reduction of TcpVeno on a random loss
 *
 * On a loss in the non-congestive state, the window of 100 segments is
 * reduced to 4/5 of it, as in the Veno paper and in Linux
 * (tcp_veno_ssthresh()), whatever the data in flight, which the lost and
 * SACKed segments reduce.
 */
class TcpVenoRandomLossTest : public TestCase
{
  public:
    TcpVenoRandomLossTest()
        : TestCase("Random loss reduces the window by a fifth")
    {
    }

  private:
    void DoRun() override
    {
        const uint32_t segmentSize = 1000;
        Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
        state->m_segmentSize = segmentSize;
        state->m_cWnd = 100 * segmentSize + 500;
        Ptr<TcpVeno> cong = CreateObject<TcpVeno>();
        NS_TEST_ASSERT_MSG_EQ(cong->GetSsThresh(state, 90 * segmentSize),
                              80 * segmentSize,
                              "The window, in segments, must be reduced to 4/5");
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Testing the Linux arithmetic of TcpVeno
 *
 * With the LinuxArithmetic attribute, as Linux tcp_veno_cong_avoid(), a
 * window of 10 segments whose RTT doubled (a difference of 11 half segments)
 * is in the congestive state: the window grows by a segment on the ACK which
 * follows a window of acknowledged segments, and only every other such
 * window, i.e., after 11 and 35 ACKs of a segment. The window is then
 * halved in whole segments on a loss (tcp_veno_ssthresh()).
 */
class TcpVenoLinuxTest : public TestCase
{
  public:
    TcpVenoLinuxTest()
        : TestCase("Linux arithmetic in the congestive state")
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
        Ptr<TcpVeno> cong = CreateObject<TcpVeno>();
        cong->SetAttribute("CountAckedSegments", BooleanValue(true));
        cong->SetAttribute("LinuxArithmetic", BooleanValue(true));
        for (uint32_t i = 0; i < 3; ++i)
        {
            cong->PktsAcked(state, 1, MilliSeconds(100));
        }
        auto ack = [&]() {
            cong->PktsAcked(state, 1, MilliSeconds(200));
            cong->IncreaseWindow(state, 1);
        };
        for (uint32_t i = 0; i < 10; ++i)
        {
            ack();
        }
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(), 10 * segmentSize, "Unexpected increase");
        ack();
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                              11 * segmentSize,
                              "The ACK after a window must add a segment");
        for (uint32_t i = 0; i < 23; ++i)
        {
            ack();
        }
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                              11 * segmentSize,
                              "Only every other window must add a segment");
        ack();
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(), 12 * segmentSize, "Unexpected window");
        NS_TEST_ASSERT_MSG_EQ(cong->GetSsThresh(state, 11 * segmentSize),
                              6 * segmentSize,
                              "The window must be halved in whole segments");
    }
};

/**
 * @ingroup internet-test
 *
 * @brief TCP Veno TestSuite
 */
class TcpVenoTestSuite : public TestSuite
{
  public:
    TcpVenoTestSuite()
        : TestSuite("tcp-veno-test", Type::UNIT)
    {
        AddTestCase(new TcpVenoRandomLossTest(), TestCase::Duration::QUICK);
        AddTestCase(new TcpVenoLinuxTest(), TestCase::Duration::QUICK);
        AddTestCase(new TcpVenoTest(38 * 1446,
                                    1446,
                                    40 * 1446,
                                    MilliSeconds(100),
                                    1,
                                    1,
                                    "Veno test on cWnd in slow start and non-congestive loss"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpVenoTest(30 * 536,
                                    536,
                                    20 * 536,
                                    MilliSeconds(106),
                                    1,
                                    1,
                                    "Veno test on cWnd with diff < beta"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpVenoTest(60 * 536,
                                    536,
                                    40 * 536,
                                    MilliSeconds(106),
                                    1,
                                    3,
                                    "Veno increment test on cWnd with diff > beta"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpVenoTest(60 * 536,
                                    536,
                                    40 * 536,
                                    MilliSeconds(106),
                                    1,
                                    8,
                                    "Veno increment test on cWnd with diff > beta, over "
                                    "several ACKs"),
                    TestCase::Duration::QUICK);
    }
};

static TcpVenoTestSuite g_tcpVenoTest; //!< Static variable for test initialization
