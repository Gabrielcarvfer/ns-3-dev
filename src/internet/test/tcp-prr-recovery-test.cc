/*
 * Copyright (c) 2018 NITK Surathkal
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Viyom Mittal <viyommittal@gmail.com>
 *         Vivek Jain <jain.vivek.anand@gmail.com>
 *         Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 *
 */

#include "ns3/log.h"
#include "ns3/string.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-prr-recovery.h"
#include "ns3/tcp-recovery-ops.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/test.h"

#include <algorithm>
#include <cmath>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpPrrRecoveryTestSuite");

/**
 * @brief PRR Recovery algorithm test
 */
class PrrRecoveryTest : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param cWnd Congestion window.
     * @param segmentSize Segment size.
     * @param ssThresh Slow Start Threshold.
     * @param unAckDataCount Unacknowledged data at the start of recovery.
     * @param bytesInFlight Current bytes in flight.
     * @param m_deliveredBytes Bytes SACKed on last acknowledgment.
     * @param bytesSent Bytes sent while in recovery phase.
     * @param name Test description.
     */
    PrrRecoveryTest(uint32_t cWnd,
                    uint32_t segmentSize,
                    uint32_t ssThresh,
                    uint32_t unAckDataCount,
                    uint32_t bytesInFlight,
                    uint32_t m_deliveredBytes,
                    uint32_t bytesSent,
                    const std::string& name);

  private:
    void DoRun() override;

    uint32_t m_cWnd;           //!< Congestion window.
    uint32_t m_segmentSize;    //!< Segment size.
    uint32_t m_ssThresh;       //!< Slow Start Threshold.
    uint32_t m_unAckDataCount; //!< Unacknowledged data at the start of recovery.
    uint32_t m_bytesInFlight;  //!< Current bytes in flight.
    uint32_t m_deliveredBytes; //!< Bytes SACKed on last acknowledgment.
    uint32_t m_bytesSent;      //!< Bytes sent while in recovery phase.

    Ptr<TcpSocketState> m_state; //!< TCP socket state.
};

PrrRecoveryTest::PrrRecoveryTest(uint32_t cWnd,
                                 uint32_t segmentSize,
                                 uint32_t ssThresh,
                                 uint32_t unAckDataCount,
                                 uint32_t bytesInFlight,
                                 uint32_t deliveredBytes,
                                 uint32_t bytesSent,
                                 const std::string& name)
    : TestCase(name),
      m_cWnd(cWnd),
      m_segmentSize(segmentSize),
      m_ssThresh(ssThresh),
      m_unAckDataCount(unAckDataCount),
      m_bytesInFlight(bytesInFlight),
      m_deliveredBytes(deliveredBytes),
      m_bytesSent(bytesSent)
{
}

void
PrrRecoveryTest::DoRun()
{
    m_state = CreateObject<TcpSocketState>();

    m_state->m_cWnd = m_cWnd;
    m_state->m_cWndInfl = m_cWnd;
    m_state->m_segmentSize = m_segmentSize;
    m_state->m_ssThresh = m_ssThresh;
    m_state->m_bytesInFlight = m_bytesInFlight;

    Ptr<TcpPrrRecovery> recovery = CreateObject<TcpPrrRecovery>();

    recovery->EnterRecovery(m_state, 3, m_unAckDataCount, 0);

    NS_TEST_ASSERT_MSG_GT_OR_EQ(m_state->m_cWnd.Get(),
                                m_cWnd + m_segmentSize,
                                "There should be at least one transmission on entering recovery");

    for (uint32_t iterator = 0; iterator < m_bytesSent;)
    {
        recovery->UpdateBytesSent(m_segmentSize);
        iterator += m_segmentSize;
    }

    m_bytesInFlight += m_state->m_cWnd.Get() - m_cWnd;
    m_state->m_bytesInFlight = m_bytesInFlight;
    m_cWnd = m_state->m_cWnd.Get();
    recovery->DoRecovery(m_state, m_deliveredBytes, false);

    if (m_bytesInFlight > m_state->m_ssThresh)
    {
        NS_TEST_ASSERT_MSG_LT_OR_EQ(
            m_state->m_cWnd.Get(),
            m_cWnd,
            "Updated cwnd should be less than or equal to the existing cwnd");
    }
    else
    {
        NS_TEST_ASSERT_MSG_GT_OR_EQ(
            m_state->m_cWnd.Get(),
            m_cWnd,
            "Updated cwnd should be greater than or equal to the existing cwnd");
    }
}

/**
 * @ingroup internet-test
 *
 * @brief PRR Recovery integer-overflow regression test (issue #1282)
 *
 * Drives the Proportional Rate Reduction computation with values large enough
 * that the product m_prrDelivered * m_ssThresh exceeds the range of a uint32_t.
 * With the (buggy) 32-bit multiplication the product wraps around, producing a
 * congestion window that is far too small. This test asserts the resulting
 * cwnd matches the value computed in 64-bit arithmetic.
 */
class PrrRecoveryOverflowTest : public TestCase
{
  public:
    PrrRecoveryOverflowTest()
        : TestCase("Prr test against uint32_t overflow of m_prrDelivered * m_ssThresh")
    {
    }

  private:
    void DoRun() override;
};

void
PrrRecoveryOverflowTest::DoRun()
{
    const uint32_t segmentSize = 1448;
    const uint32_t ssThresh = 1000000;
    // bytesInFlight must exceed ssThresh so the PRR (not the PRR-CRB) branch is taken.
    // It is also captured as the recovery flight size on entering recovery.
    const uint32_t bytesInFlight = ssThresh + 1;

    auto state = CreateObject<TcpSocketState>();
    state->m_segmentSize = segmentSize;
    state->m_ssThresh = ssThresh;
    state->m_bytesInFlight = bytesInFlight;
    state->m_cWnd = bytesInFlight;
    state->m_cWndInfl = bytesInFlight;

    auto recovery = CreateObject<TcpPrrRecovery>();
    // On entering recovery, m_prrDelivered becomes one segment and m_prrOut is 0.
    recovery->EnterRecovery(state, 3, bytesInFlight, 0);

    // Deliver enough additional bytes so that m_prrDelivered * m_ssThresh > 2^32.
    // m_prrDelivered after entering recovery is exactly one segment.
    const uint32_t prrDelivered = 5000;
    recovery->DoRecovery(state, prrDelivered - segmentSize, false);

    // Reference computation in 64-bit arithmetic (m_prrOut is still 0 here).
    const uint32_t prrOut = 0;
    int sendCount = static_cast<int>(std::ceil(static_cast<uint64_t>(prrDelivered) * ssThresh *
                                               1.0 / bytesInFlight)) -
                    prrOut;
    sendCount = std::max(sendCount, prrOut > 0 ? 0 : static_cast<int>(segmentSize));
    const uint32_t expectedCwnd = bytesInFlight + sendCount;

    NS_TEST_ASSERT_MSG_EQ(static_cast<uint64_t>(prrDelivered) * ssThresh > 0xFFFFFFFFULL,
                          true,
                          "Test setup error: product does not exceed uint32_t range");
    NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                          expectedCwnd,
                          "cwnd is wrong: m_prrDelivered * m_ssThresh overflowed uint32_t");
}

/**
 * @ingroup internet-test
 *
 * @brief PRR Recovery TestSuite
 */
class PrrRecoveryTestSuite : public TestSuite
{
  public:
    PrrRecoveryTestSuite()
        : TestSuite("tcp-prr-recovery-test", Type::UNIT)
    {
        AddTestCase(new PrrRecoveryTest(
                        3000,
                        500,
                        2500,
                        3000,
                        3000,
                        500,
                        1000,
                        "Prr test on cWnd when bytesInFlight is greater than ssThresh with SSRB"),
                    TestCase::Duration::QUICK);
        AddTestCase(new PrrRecoveryTest(
                        1000,
                        500,
                        2500,
                        3000,
                        1000,
                        500,
                        1000,
                        "Prr test on cWnd when bytesInFlight is lower than ssThresh with SSRB"),
                    TestCase::Duration::QUICK);
        AddTestCase(new PrrRecoveryTest(
                        3000,
                        500,
                        2500,
                        3000,
                        3000,
                        500,
                        1000,
                        "Prr test on cWnd when bytesInFlight is greater than ssThresh with CRB"),
                    TestCase::Duration::QUICK);
        AddTestCase(new PrrRecoveryTest(
                        1000,
                        500,
                        2500,
                        3000,
                        1000,
                        500,
                        1000,
                        "Prr test on cWnd when bytesInFlight is lower than ssThresh with CRB"),
                    TestCase::Duration::QUICK);
        AddTestCase(new PrrRecoveryOverflowTest(), TestCase::Duration::QUICK);
    }
};

static PrrRecoveryTestSuite g_TcpPrrRecoveryTest; //!< Static variable for test initialization
