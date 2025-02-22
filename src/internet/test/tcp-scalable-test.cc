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
    }
};

static TcpScalableTestSuite g_tcpScalableTest; //!< Static variable for test initialization
