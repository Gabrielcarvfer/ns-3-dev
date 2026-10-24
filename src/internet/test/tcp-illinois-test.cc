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
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-illinois.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/test.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpIllinoisTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief TcpIllinois C-AIMD algorithm tests.
 */
class TcpIllinoisTest : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param cWnd Congestion window.
     * @param ssThresh Slow Start Threshold.
     * @param segmentSize Segment size.
     * @param cntRtt RTT counter.
     * @param maxRtt Max RTT.
     * @param segmentsAcked Number of segments ACKed.
     * @param nextTxSeq Next Tx sequence number.
     * @param lastAckedSeq Last ACKed sequence number.
     * @param name Test description.
     */
    TcpIllinoisTest(uint32_t cWnd,
                    uint32_t ssThresh,
                    uint32_t segmentSize,
                    uint32_t cntRtt,
                    Time maxRtt,
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
    void IncreaseWindow(Ptr<TcpIllinois> cong);
    /**
     * @brief Recalculate the internal TCP Illinois params.
     * @param cong The congestion control.
     */
    void RecalcParam(Ptr<TcpIllinois> cong);
    /**
     * @brief Calculate the maximum delay.
     * @returns The maximum delay.
     */
    Time CalculateMaxDelay();
    /**
     * @brief Calculate the average delay.
     * @returns The average delay.
     */
    Time CalculateAvgDelay();
    /**
     * @brief Calculate the TCP Illinois alpha param.
     * @param cong The congestion control.
     * @param da Average delay (in milliseconds).
     * @param dm Maximum delay (in milliseconds).
     */
    void CalculateAlpha(Ptr<TcpIllinois> cong, double da, double dm);
    /**
     * @brief Calculate the TCP Illinois beta param.
     * @param cong The congestion control.
     * @param da Average delay (in milliseconds).
     * @param dm Maximum delay (in milliseconds).
     */
    void CalculateBeta(Ptr<TcpIllinois> cong, double da, double dm);
    /**
     * brief Get and check the SSH threshold.
     */
    void GetSsThresh();

    uint32_t m_cWnd;                 //!< Congestion window.
    uint32_t m_ssThresh;             //!< Slow Start Threshold.
    uint32_t m_segmentSize;          //!< Segment size.
    Time m_baseRtt;                  //!< Base RTT.
    Time m_maxRtt;                   //!< Max RTT.
    uint32_t m_segmentsAcked;        //!< Number of segments ACKed.
    SequenceNumber32 m_nextTxSeq;    //!< Next Tx sequence number.
    SequenceNumber32 m_lastAckedSeq; //!< Last ACKed sequence number.
    double m_alpha;                  //!< TCP Illinois alpha parameter.
    double m_beta;                   //!< TCP Illinois beta parameter.
    uint32_t m_cntRtt;               //!< RTT counter.
    Time m_sumRtt;                   //!< Sum of all the RTTs.
    bool m_rttAbove;                 //!< RTT above threshold.
    uint8_t m_rttLow;                //!< RTT low counter.
    uint32_t m_ackCnt;               //!< ACK counter.
};

TcpIllinoisTest::TcpIllinoisTest(uint32_t cWnd,
                                 uint32_t ssThresh,
                                 uint32_t segmentSize,
                                 uint32_t cntRtt,
                                 Time maxRtt,
                                 uint32_t segmentsAcked,
                                 SequenceNumber32 nextTxSeq,
                                 SequenceNumber32 lastAckedSeq,
                                 const std::string& name)
    : TestCase(name),
      m_cWnd(cWnd),
      m_ssThresh(ssThresh),
      m_segmentSize(segmentSize),
      m_baseRtt(MilliSeconds(100)),
      m_maxRtt(maxRtt),
      m_segmentsAcked(segmentsAcked),
      m_nextTxSeq(nextTxSeq),
      m_lastAckedSeq(lastAckedSeq),
      m_alpha(0.0),
      m_beta(0.0),
      m_cntRtt(cntRtt),
      m_sumRtt(0),
      m_rttAbove(false),
      m_rttLow(0),
      m_ackCnt(0)
{
}

void
TcpIllinoisTest::DoRun()
{
    Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
    state->m_cWnd = m_cWnd;
    state->m_ssThresh = m_ssThresh;
    state->m_segmentSize = m_segmentSize;
    state->m_nextTxSequence = m_nextTxSeq;
    state->m_lastAckedSeq = m_lastAckedSeq;

    Ptr<TcpIllinois> cong = CreateObject<TcpIllinois>();

    // Set baseRtt to 100 ms
    cong->PktsAcked(state, m_segmentsAcked, m_baseRtt);

    m_sumRtt += m_baseRtt;

    // Set maxRtt and update sumRtt based on cntRtt value
    for (uint32_t count = 1; count < m_cntRtt; ++count)
    {
        cong->PktsAcked(state, m_segmentsAcked, m_maxRtt);
        m_sumRtt += m_maxRtt;
    }

    /*
     * Test cWnd modification during additive increase
     */
    cong->IncreaseWindow(state, m_segmentsAcked);
    IncreaseWindow(cong);
    NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(), m_cWnd, "CWnd has not updated correctly");

    /*
     * Test ssThresh modification during multiplicative decrease
     */
    uint32_t ssThresh = cong->GetSsThresh(state, m_cWnd);
    GetSsThresh();
    NS_TEST_ASSERT_MSG_EQ(ssThresh, m_ssThresh, "SsThresh has not updated correctly");
}

void
TcpIllinoisTest::IncreaseWindow(Ptr<TcpIllinois> cong)
{
    uint32_t segCwnd = m_cWnd / m_segmentSize;

    if (m_lastAckedSeq >= m_nextTxSeq)
    {
        RecalcParam(cong);
    }
    if (m_cWnd < m_ssThresh)
    { // NewReno slow start
        if (m_segmentsAcked >= 1)
        {
            m_cWnd += m_segmentSize;
            m_segmentsAcked -= 1;
        }
        NS_LOG_INFO("In SlowStart, updated to cwnd " << m_cWnd << " ssthresh " << m_ssThresh);
    }
    else
    {
        uint32_t oldCwnd = segCwnd;

        if (m_segmentsAcked > 0)
        {
            m_ackCnt += m_segmentsAcked * m_alpha;
        }

        while (m_ackCnt >= segCwnd)
        {
            m_ackCnt -= segCwnd;
            segCwnd += 1;
        }

        if (segCwnd != oldCwnd)
        {
            m_cWnd = segCwnd * m_segmentSize;
            NS_LOG_INFO("In CongAvoid, updated to cwnd " << m_cWnd << " ssthresh " << m_ssThresh);
        }
    }
}

void
TcpIllinoisTest::RecalcParam(Ptr<TcpIllinois> cong)
{
    DoubleValue alphaBase;
    cong->GetAttribute("AlphaBase", alphaBase);
    UintegerValue winThresh;
    cong->GetAttribute("WinThresh", winThresh);

    DoubleValue betaBase;
    cong->GetAttribute("BetaBase", betaBase);

    if (m_cWnd / m_segmentSize < winThresh.Get())
    {
        NS_LOG_INFO("cWnd < winThresh, set alpha & beta to base values");
        m_alpha = alphaBase.Get();
        m_beta = betaBase.Get();
    }
    else if (m_cntRtt > 0)
    {
        double dm = CalculateMaxDelay().GetSeconds() * 1000;
        double da = CalculateAvgDelay().GetSeconds() * 1000;

        NS_LOG_INFO("Updated to dm = " << dm << " da = " << da);

        CalculateAlpha(cong, da, dm);
        CalculateBeta(cong, da, dm);
    }
}

Time
TcpIllinoisTest::CalculateMaxDelay()
{
    return (m_maxRtt - m_baseRtt);
}

Time
TcpIllinoisTest::CalculateAvgDelay()
{
    return (m_sumRtt / m_cntRtt - m_baseRtt);
}

void
TcpIllinoisTest::CalculateAlpha(Ptr<TcpIllinois> cong, double da, double dm)
{
    DoubleValue alphaMax;
    cong->GetAttribute("AlphaMax", alphaMax);
    UintegerValue theta;
    cong->GetAttribute("Theta", theta);
    DoubleValue alphaMin;
    cong->GetAttribute("AlphaMin", alphaMin);

    double d1 = dm / 100;

    if (da <= d1)
    {
        if (!m_rttAbove)
        {
            m_alpha = alphaMax.Get();
        }
        if (++m_rttLow >= theta.Get())
        {
            m_rttLow = 0;
            m_rttAbove = false;
            m_alpha = alphaMax.Get();
        }
    }
    else
    {
        m_rttAbove = true;
        dm -= d1;
        da -= d1;
        m_alpha = (dm * alphaMax.Get()) /
                  (dm + (da * (alphaMax.Get() - alphaMin.Get())) / alphaMin.Get());
    }
    NS_LOG_INFO("Updated to alpha = " << m_alpha);
}

void
TcpIllinoisTest::CalculateBeta(Ptr<TcpIllinois> cong, double da, double dm)
{
    DoubleValue betaMin;
    cong->GetAttribute("BetaMin", betaMin);
    DoubleValue betaMax;
    cong->GetAttribute("BetaMax", betaMax);

    double d2;
    double d3;
    d2 = dm / 10;
    d3 = (8 * dm) / 10;

    if (da <= d2)
    {
        m_beta = betaMin.Get();
    }
    else if (da > d2 && da < d3)
    {
        m_beta = (betaMin.Get() * d3 - betaMax.Get() * d2 + (betaMax.Get() - betaMin.Get()) * da) /
                 (d3 - d2);
    }

    else if (da >= d3 || d3 <= d2)
    {
        m_beta = betaMax.Get();
    }
    NS_LOG_INFO("Updated to beta = " << m_beta);
}

void
TcpIllinoisTest::GetSsThresh()
{
    uint32_t segCwnd = m_cWnd / m_segmentSize;
    uint32_t ssThresh = std::max(2.0, (1.0 - m_beta) * segCwnd);

    NS_LOG_DEBUG("Calculated ssThresh (in segments) = " << ssThresh);

    m_ssThresh = ssThresh * m_segmentSize;
}

/**
 * @ingroup internet-test
 *
 * @brief Test that TcpIllinois recalculates its parameters with any ISN.
 *
 * The same scenario runs twice, once with sequence numbers near zero and
 * once with sequence numbers in the upper half of the space, as a clock
 * driven initial sequence number may give. The congestion window and the
 * slow start threshold must come out the same.
 */
class TcpIllinoisUpperHalfIsnTest : public TestCase
{
  public:
    TcpIllinoisUpperHalfIsnTest()
        : TestCase("Illinois recalculates its parameters with an upper half ISN")
    {
    }

  private:
    void DoRun() override;

    /**
     * Run the scenario with the sequence numbers shifted by an offset.
     * @param offset Offset added to the sequence numbers.
     * @return The congestion window and the slow start threshold.
     */
    std::pair<uint32_t, uint32_t> Run(uint32_t offset);
};

std::pair<uint32_t, uint32_t>
TcpIllinoisUpperHalfIsnTest::Run(uint32_t offset)
{
    Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
    state->m_cWnd = 40 * 1446;
    state->m_ssThresh = 38 * 1446;
    state->m_segmentSize = 1446;
    state->m_nextTxSequence = SequenceNumber32(2893 + offset);
    state->m_lastAckedSeq = SequenceNumber32(5785 + offset);

    Ptr<TcpIllinois> cong = CreateObject<TcpIllinois>();
    cong->Init(state);
    // A queueing delay near its maximum: the recalculation lowers alpha and
    // raises beta well away from their initial values
    cong->PktsAcked(state, 10, MilliSeconds(100));
    for (uint32_t count = 1; count < 5; ++count)
    {
        cong->PktsAcked(state, 10, MilliSeconds(110));
    }
    cong->IncreaseWindow(state, 10);
    return {state->m_cWnd.Get(), cong->GetSsThresh(state, state->m_cWnd)};
}

void
TcpIllinoisUpperHalfIsnTest::DoRun()
{
    auto low = Run(0);
    auto high = Run(0x90000000);
    NS_TEST_ASSERT_MSG_EQ(high.first, low.first, "cWnd differs with an upper half ISN");
    NS_TEST_ASSERT_MSG_EQ(high.second, low.second, "ssThresh differs with an upper half ISN");
}

/**
 * @ingroup internet-test
 *
 * @brief Testing that TcpIllinois reduces the window, not the data in flight
 *
 * The window of 100 segments is reduced by beta (0.5 before any adaptation),
 * as in the Illinois paper and in Linux (tcp_illinois_ssthresh()), whatever
 * the data in flight, which the lost and SACKed segments reduce.
 */
class TcpIllinoisSsThreshWindowTest : public TestCase
{
  public:
    TcpIllinoisSsThreshWindowTest()
        : TestCase("The window is reduced by beta")
    {
    }

  private:
    void DoRun() override
    {
        const uint32_t segmentSize = 1000;
        Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
        state->m_segmentSize = segmentSize;
        state->m_cWnd = 100 * segmentSize;
        Ptr<TcpIllinois> cong = CreateObject<TcpIllinois>();
        NS_TEST_ASSERT_MSG_EQ(cong->GetSsThresh(state, 90 * segmentSize),
                              50 * segmentSize,
                              "The window must be reduced by beta");
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Testing the congestion avoidance count of TcpIllinois
 *
 * By default, each acknowledged segment adds alpha / cwnd segments, the
 * fractions carried to the next ACKs: with alpha = 1.5 (the AlphaBase after a
 * timeout) and a window of 100 segments, the 67th ACK of a segment completes
 * a segment. With CountAckedSegments, as Linux tcp_illinois_cong_avoid(),
 * the count is the segments the ACK removed from the retransmission queue
 * (PktsAcked), including those SACKed before: the ACK which ends a recovery
 * and acknowledges 100 segments grows a window of 100 segments by alpha (10)
 * segments at once.
 */
class TcpIllinoisCountTest : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param linuxCount Whether the CountAckedSegments attribute is set.
     * @param name Test description.
     */
    TcpIllinoisCountTest(bool linuxCount, const std::string& name)
        : TestCase(name),
          m_linuxCount(linuxCount)
    {
    }

  private:
    void DoRun() override
    {
        const uint32_t segmentSize = 1000;
        Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
        state->m_segmentSize = segmentSize;
        state->m_cWnd = 100 * segmentSize;
        state->m_ssThresh = 50 * segmentSize;
        state->m_nextTxSequence = SequenceNumber32(1000000);
        state->m_lastAckedSeq = SequenceNumber32(1);

        Ptr<TcpIllinois> cong = CreateObject<TcpIllinois>();
        cong->SetAttribute("CountAckedSegments", BooleanValue(m_linuxCount));
        cong->Init(state);

        if (m_linuxCount)
        {
            cong->PktsAcked(state, 100, MilliSeconds(100));
            cong->IncreaseWindow(state, 1);
            NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                                  110 * segmentSize,
                                  "The segments removed by the ACK must count");
        }
        else
        {
            cong->SetAttribute("AlphaBase", DoubleValue(1.5));
            cong->CongestionStateSet(state, TcpSocketState::CA_LOSS);
            for (uint32_t i = 0; i < 66; ++i)
            {
                cong->IncreaseWindow(state, 1);
            }
            NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                                  100 * segmentSize,
                                  "Unexpected window increase");
            cong->IncreaseWindow(state, 1);
            NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                                  101 * segmentSize,
                                  "The fractions of segment must be carried");
        }
    }

    bool m_linuxCount; //!< Whether the CountAckedSegments attribute is set
};

/**
 * @ingroup internet-test
 *
 * @brief Testing the Linux arithmetic of TcpIllinois
 *
 * RTT samples of 100, 200 and 130 ms give a maximum queueing delay of 100 ms
 * and an average one of 43.333 ms. With the LinuxArithmetic attribute, as
 * Linux tcp_illinois.c, beta is 19/64 (0.297, instead of 0.304), and the
 * decrease of a window of 100 segments is rounded down: the slow start
 * threshold is 71 segments, instead of 69. The parameters are updated only
 * once the ACK is after the right edge of the RTT.
 */
class TcpIllinoisLinuxTest : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param linuxArithmetic Whether the LinuxArithmetic attribute is set.
     * @param name Test description.
     */
    TcpIllinoisLinuxTest(bool linuxArithmetic, const std::string& name)
        : TestCase(name),
          m_linux(linuxArithmetic)
    {
    }

  private:
    void DoRun() override
    {
        const uint32_t segmentSize = 1000;
        Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
        state->m_segmentSize = segmentSize;
        state->m_cWnd = 100 * segmentSize;
        state->m_ssThresh = 50 * segmentSize;
        state->m_nextTxSequence = SequenceNumber32(1000000);
        state->m_highTxMark = SequenceNumber32(1000000);
        state->m_lastAckedSeq = SequenceNumber32(1);

        Ptr<TcpIllinois> cong = CreateObject<TcpIllinois>();
        cong->SetAttribute("LinuxArithmetic", BooleanValue(m_linux));
        cong->Init(state);
        for (auto rtt : {100, 200, 130})
        {
            cong->PktsAcked(state, 1, MilliSeconds(rtt));
        }
        state->m_lastAckedSeq = SequenceNumber32(1000000);
        cong->IncreaseWindow(state, 1);
        if (m_linux)
        {
            NS_TEST_ASSERT_MSG_EQ(cong->GetSsThresh(state, state->m_cWnd),
                                  50 * segmentSize,
                                  "The parameters must not be updated at the edge of the RTT");
        }
        state->m_lastAckedSeq = SequenceNumber32(1000001);
        state->m_cWnd = 100 * segmentSize;
        cong->IncreaseWindow(state, 1);
        state->m_cWnd = 100 * segmentSize;
        NS_TEST_ASSERT_MSG_EQ(cong->GetSsThresh(state, state->m_cWnd),
                              (m_linux ? 71 : 69) * segmentSize,
                              "Unexpected slow start threshold");
    }

    bool m_linux; //!< Whether the LinuxArithmetic attribute is set
};

/**
 * @ingroup internet-test
 *
 * @brief TCP Illinois TestSuite
 */
class TcpIllinoisTestSuite : public TestSuite
{
  public:
    TcpIllinoisTestSuite()
        : TestSuite("tcp-illinois-test", Type::UNIT)
    {
        AddTestCase(new TcpIllinoisTest(38 * 1446,
                                        40 * 1446,
                                        1446,
                                        2,
                                        MilliSeconds(105),
                                        2,
                                        SequenceNumber32(2893),
                                        SequenceNumber32(5785),
                                        "Illinois test on cWnd and ssThresh when in slow start"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpIllinoisTest(
                        60 * 346,
                        40 * 346,
                        346,
                        2,
                        MilliSeconds(100),
                        2,
                        SequenceNumber32(2893),
                        SequenceNumber32(5785),
                        "Illinois test on cWnd and ssThresh when avg queueing delay is at minimum"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpIllinoisTest(
                        38 * 1446,
                        40 * 1446,
                        1446,
                        5,
                        MilliSeconds(110),
                        2,
                        SequenceNumber32(2893),
                        SequenceNumber32(5785),
                        "Illinois test on cWnd and ssThresh when avg queueing delay is at maximum"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpIllinoisTest(40 * 1446,
                                        38 * 1446,
                                        1446,
                                        2,
                                        MilliSeconds(105),
                                        55,
                                        SequenceNumber32(2893),
                                        SequenceNumber32(5785),
                                        "Illinois test on cWnd and ssThresh when avg queueing "
                                        "delay is in between its min & max"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpIllinoisTest(10 * 1446,
                                        5 * 1446,
                                        1446,
                                        2,
                                        MilliSeconds(110),
                                        2,
                                        SequenceNumber32(2893),
                                        SequenceNumber32(5785),
                                        "Illinois test on cWnd and ssThresh when cWnd (in "
                                        "segments) is below winThresh"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpIllinoisTest(40 * 1446,
                                        38 * 1446,
                                        1446,
                                        2,
                                        MicroSeconds(100900),
                                        3,
                                        SequenceNumber32(2893),
                                        SequenceNumber32(5785),
                                        "Illinois test on cWnd and ssThresh with sub-millisecond "
                                        "queueing delays"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpIllinoisUpperHalfIsnTest(), TestCase::Duration::QUICK);
        AddTestCase(new TcpIllinoisSsThreshWindowTest(), TestCase::Duration::QUICK);
        AddTestCase(new TcpIllinoisCountTest(false, "Fractions of the increase carried"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpIllinoisCountTest(true, "Linux count of the removed segments"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpIllinoisLinuxTest(false, "Default arithmetic"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpIllinoisLinuxTest(true, "Linux arithmetic"), TestCase::Duration::QUICK);
    }
};

static TcpIllinoisTestSuite g_tcpIllinoisTest; //!< Static variable for test initialization
