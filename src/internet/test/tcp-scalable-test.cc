/*
 * Copyright (c) 2016 ResiliNets, ITTC, University of Kansas
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Truc Anh N. Nguyen <annguyen@ittc.ku.edu>

 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  https://resilinets.org/
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 *
 */

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-scalable.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/test.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpScalableTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief Testing the congestion avoidance increment on TcpScalable
 */
class TcpScalableIncrementTest : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param cWnd Congestion window.
     * @param segmentSize Segment size.
     * @param segmentsAcked Segments ACKed.
     * @param name Test description.
     */
    TcpScalableIncrementTest(uint32_t cWnd,
                             uint32_t segmentSize,
                             uint32_t segmentsAcked,
                             const std::string& name);

  private:
    void DoRun() override;

    uint32_t m_cWnd;             //!< Congestion window.
    uint32_t m_segmentSize;      //!< Segment size.
    uint32_t m_segmentsAcked;    //!< Segments ACKed.
    Ptr<TcpSocketState> m_state; //!< TCP socket state.
};

TcpScalableIncrementTest::TcpScalableIncrementTest(uint32_t cWnd,
                                                   uint32_t segmentSize,
                                                   uint32_t segmentsAcked,
                                                   const std::string& name)
    : TestCase(name),
      m_cWnd(cWnd),
      m_segmentSize(segmentSize),
      m_segmentsAcked(segmentsAcked)
{
}

void
TcpScalableIncrementTest::DoRun()
{
    m_state = CreateObject<TcpSocketState>();

    m_state->m_cWnd = m_cWnd;
    m_state->m_segmentSize = m_segmentSize;

    Ptr<TcpScalable> cong = CreateObject<TcpScalable>();

    uint32_t segCwnd = m_cWnd / m_segmentSize;

    // Get default value of additive increase factor
    UintegerValue aiFactor;
    cong->GetAttribute("AIFactor", aiFactor);

    // To see an increase of 1 MSS, the number of segments ACKed has to be at least
    // min (segCwnd, aiFactor).

    uint32_t w = std::min(segCwnd, (uint32_t)aiFactor.Get());
    uint32_t delta = m_segmentsAcked / w;

    cong->IncreaseWindow(m_state, m_segmentsAcked);

    NS_TEST_ASSERT_MSG_EQ(m_state->m_cWnd.Get(),
                          m_cWnd + delta * m_segmentSize,
                          "CWnd has not increased");
}

/**
 * @ingroup internet-test
 *
 * @brief Testing the multiplicative decrease on TcpScalable
 */
class TcpScalableDecrementTest : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param cWnd Congestion window.
     * @param segmentSize Segment size.
     * @param name Test description.
     */
    TcpScalableDecrementTest(uint32_t cWnd, uint32_t segmentSize, const std::string& name);

  private:
    void DoRun() override;

    uint32_t m_cWnd;             //!< Congestion window.
    uint32_t m_segmentSize;      //!< Segment size.
    Ptr<TcpSocketState> m_state; //!< TCP socket state.
};

TcpScalableDecrementTest::TcpScalableDecrementTest(uint32_t cWnd,
                                                   uint32_t segmentSize,
                                                   const std::string& name)
    : TestCase(name),
      m_cWnd(cWnd),
      m_segmentSize(segmentSize)
{
}

void
TcpScalableDecrementTest::DoRun()
{
    m_state = CreateObject<TcpSocketState>();

    m_state->m_cWnd = m_cWnd;
    m_state->m_segmentSize = m_segmentSize;

    Ptr<TcpScalable> cong = CreateObject<TcpScalable>();

    uint32_t segCwnd = m_cWnd / m_segmentSize;

    // Get default value of multiplicative decrease factor
    DoubleValue mdFactor;
    cong->GetAttribute("MDFactor", mdFactor);

    double b = 1.0 - mdFactor.Get();

    uint32_t ssThresh = std::max(2.0, segCwnd * b);

    uint32_t ssThreshInSegments = cong->GetSsThresh(m_state, m_state->m_cWnd) / m_segmentSize;

    NS_TEST_ASSERT_MSG_EQ(ssThreshInSegments, ssThresh, "Scalable decrement fn not used");
}

/**
 * @ingroup internet-test
 *
 * @brief Testing the additive increase of TcpScalable over several ACKs
 *
 * The window grows by one segment every 100 segments ACKed (0.01 segments
 * per segment ACKed, Linux TCP_SCALABLE_AI_CNT), or every cwnd segments
 * ACKed for a smaller window, and the ACKed segments in excess of a step
 * are kept for the next one, as in Linux tcp_cong_avoid_ai().
 */
class TcpScalableAiTest : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param cWnd Congestion window (segments).
     * @param acks Segments ACKed by each ACK.
     * @param expectedCwnd Expected final congestion window (segments).
     * @param name Test description.
     */
    TcpScalableAiTest(uint32_t cWnd,
                      std::vector<uint32_t> acks,
                      uint32_t expectedCwnd,
                      const std::string& name)
        : TestCase(name),
          m_cWnd(cWnd),
          m_acks(acks),
          m_expectedCwnd(expectedCwnd)
    {
    }

  private:
    void DoRun() override;

    uint32_t m_cWnd;              //!< Congestion window (segments).
    std::vector<uint32_t> m_acks; //!< Segments ACKed by each ACK.
    uint32_t m_expectedCwnd;      //!< Expected final congestion window (segments).
};

void
TcpScalableAiTest::DoRun()
{
    Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
    state->m_segmentSize = 1000;
    state->m_cWnd = m_cWnd * 1000;
    state->m_ssThresh = 0;

    Ptr<TcpScalable> cong = CreateObject<TcpScalable>();
    for (auto acked : m_acks)
    {
        cong->IncreaseWindow(state, acked);
    }

    NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(), m_expectedCwnd * 1000, "Unexpected window growth");
}

/**
 * @ingroup internet-test
 *
 * @brief Testing the Linux decrease of TcpScalable and the count cleared by
 * a reduction
 *
 * With LinuxDecrease, as Linux tcp_scalable_ssthresh(), the threshold is
 * cwnd - (cwnd >> 3) of cWnd: 359 segments of 410, where 0.875 of the bytes
 * in flight gives 358. A window reduction clears the acknowledged segments
 * counted towards the next increase, as Linux tp->snd_cwnd_cnt.
 */
class TcpScalableLinuxTest : public TestCase
{
  public:
    TcpScalableLinuxTest()
        : TestCase("Scalable Linux decrease and count cleared by a reduction")
    {
    }

  private:
    void DoRun() override
    {
        const uint32_t segmentSize = 1000;
        Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
        state->m_segmentSize = segmentSize;
        state->m_cWnd = 410 * segmentSize;

        Ptr<TcpScalable> rfc = CreateObject<TcpScalable>();
        NS_TEST_ASSERT_MSG_EQ(rfc->GetSsThresh(state, 410 * segmentSize),
                              358 * segmentSize,
                              "Unexpected decrease of the bytes in flight");
        Ptr<TcpScalable> cong = CreateObject<TcpScalable>();
        cong->SetAttribute("LinuxDecrease", BooleanValue(true));
        NS_TEST_ASSERT_MSG_EQ(cong->GetSsThresh(state, 300 * segmentSize),
                              359 * segmentSize,
                              "The decrease is not the one of Linux");

        // 60 of the 100 segments needed for an increase, then a reduction
        state->m_cWnd = 200 * segmentSize;
        state->m_ssThresh = 200 * segmentSize;
        cong->IncreaseWindow(state, 60);
        cong->CongestionStateSet(state, TcpSocketState::CA_RECOVERY);
        cong->IncreaseWindow(state, 60);
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                              200 * segmentSize,
                              "The count before the reduction was kept");
        cong->IncreaseWindow(state, 40);
        NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(), 201 * segmentSize, "No increase");
    }
};

/**
 * @ingroup internet-test
 *
 * @brief TcpScalable TestSuite.
 */
class TcpScalableTestSuite : public TestSuite
{
  public:
    TcpScalableTestSuite()
        : TestSuite("tcp-scalable-test", Type::UNIT)
    {
        AddTestCase(
            new TcpScalableIncrementTest(
                38 * 536,
                536,
                38,
                "Scalable increment test on cWnd = 38 segments and segmentSize = 536 bytes"),
            TestCase::Duration::QUICK);
        AddTestCase(new TcpScalableIncrementTest(
                        38,
                        1,
                        100,
                        "Scalable increment test on cWnd = 38 segments and segmentSize = 1 byte"),
                    TestCase::Duration::QUICK);
        AddTestCase(
            new TcpScalableIncrementTest(
                53 * 1446,
                1446,
                50,
                "Scalable increment test on cWnd = 53 segments and segmentSize = 1446 bytes"),
            TestCase::Duration::QUICK);

        AddTestCase(new TcpScalableAiTest(200,
                                          {100},
                                          201,
                                          "Scalable increment test: one segment per 100 ACKed"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpScalableAiTest(10,
                                          {15, 6},
                                          12,
                                          "Scalable increment test: ACKed segments carried over"),
                    TestCase::Duration::QUICK);

        AddTestCase(new TcpScalableDecrementTest(
                        38,
                        1,
                        "Scalable decrement test on cWnd = 38 segments and segmentSize = 1 byte"),
                    TestCase::Duration::QUICK);
        AddTestCase(
            new TcpScalableDecrementTest(
                100 * 536,
                536,
                "Scalable decrement test on cWnd = 100 segments and segmentSize = 536 bytes"),
            TestCase::Duration::QUICK);
        AddTestCase(
            new TcpScalableDecrementTest(
                40 * 1446,
                1446,
                "Scalable decrement test on cWnd = 40 segments and segmentSize = 1446 bytes"),
            TestCase::Duration::QUICK);
        AddTestCase(new TcpScalableLinuxTest(), TestCase::Duration::QUICK);
    }
};

static TcpScalableTestSuite g_tcpScalableTest; //!< Static variable for test initialization
