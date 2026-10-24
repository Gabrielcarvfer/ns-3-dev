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

#include "tcp-error-model.h"
#include "tcp-general-test.h"

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/string.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-prr-recovery.h"
#include "ns3/tcp-recovery-ops.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/test.h"

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
    // This test drives DoRecovery with deliveredBytes = 0 and relies on PRR's
    // "+1 SMSS per duplicate ACK" DeliveredData estimate, which (per RFC 9937
    // Section 6.2) applies only to connections without SACK.
    m_state->m_sackEnabled = false;

    Ptr<TcpPrrRecovery> recovery = CreateObject<TcpPrrRecovery>();

    recovery->EnterRecovery(m_state, 3, m_unAckDataCount, 0, 0);

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
 * @brief PRR Recovery regression test for integer overflow and unsigned wrap
 *
 * Regression tests for issue #1282.  Calls TcpPrrRecovery::DoRecovery directly and
 * checks the resulting cwnd * against a value computed with correct 64-bit / signed
 * arithmetic. The two scenarios exercise the arithmetic defects fixed in
 * tcp-prr-recovery.cc (MR !2807):
 *  - the proportional branch (inflight > ssThresh), where the product
 *    m_prrDelivered * m_ssThresh overflowed a 32-bit integer; and
 *  - the PRR-CRB branch (inflight <= ssThresh), where m_prrDelivered - m_prrOut
 *    wrapped when m_prrOut exceeded m_prrDelivered.
 */
class PrrRecoveryArithmeticTest : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param segmentSize Segment size.
     * @param ssThresh Slow Start Threshold.
     * @param bytesInFlight Bytes in flight at recovery entry (also the RecoverFS).
     * @param bytesSentInRecovery Bytes reported sent (via UpdateBytesSent) before the tested ACK.
     * @param deliveredBytes Bytes delivered on the tested (non-dupack) acknowledgment.
     * @param expectedCWnd Expected cwnd after the tested acknowledgment.
     * @param name Test description.
     */
    PrrRecoveryArithmeticTest(uint32_t segmentSize,
                              uint32_t ssThresh,
                              uint32_t bytesInFlight,
                              uint32_t bytesSentInRecovery,
                              uint32_t deliveredBytes,
                              uint32_t expectedCWnd,
                              const std::string& name);

  private:
    void DoRun() override;

    uint32_t m_segmentSize;         //!< Segment size.
    uint32_t m_ssThresh;            //!< Slow Start Threshold.
    uint32_t m_bytesInFlight;       //!< Bytes in flight at recovery entry.
    uint32_t m_bytesSentInRecovery; //!< Bytes reported sent before the tested ACK.
    uint32_t m_deliveredBytes;      //!< Bytes delivered on the tested ACK.
    uint32_t m_expectedCWnd;        //!< Expected cwnd after the tested ACK.
};

PrrRecoveryArithmeticTest::PrrRecoveryArithmeticTest(uint32_t segmentSize,
                                                     uint32_t ssThresh,
                                                     uint32_t bytesInFlight,
                                                     uint32_t bytesSentInRecovery,
                                                     uint32_t deliveredBytes,
                                                     uint32_t expectedCWnd,
                                                     const std::string& name)
    : TestCase(name),
      m_segmentSize(segmentSize),
      m_ssThresh(ssThresh),
      m_bytesInFlight(bytesInFlight),
      m_bytesSentInRecovery(bytesSentInRecovery),
      m_deliveredBytes(deliveredBytes),
      m_expectedCWnd(expectedCWnd)
{
}

void
PrrRecoveryArithmeticTest::DoRun()
{
    Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
    state->m_segmentSize = m_segmentSize;
    state->m_ssThresh = m_ssThresh;
    state->m_bytesInFlight = m_bytesInFlight;
    // These regression scenarios rely on the "+1 SMSS per duplicate ACK"
    // DeliveredData estimate (the "dupack bump"), which PRR applies only to
    // connections without SACK (RFC 9937 Section 6.2).
    state->m_sackEnabled = false;

    Ptr<TcpPrrRecovery> recovery = CreateObject<TcpPrrRecovery>();

    // EnterRecovery sets RecoverFS = bytesInFlight and performs the first DoRecovery.
    recovery->EnterRecovery(state, 3, m_bytesInFlight, 0, 0);

    // Report data sent during recovery, advancing m_prrOut one segment at a time.
    for (uint32_t sent = 0; sent < m_bytesSentInRecovery; sent += m_segmentSize)
    {
        recovery->UpdateBytesSent(m_segmentSize);
    }

    // The acknowledgment under test (not a dupack, so deliveredBytes is used as-is).
    recovery->DoRecovery(state, m_deliveredBytes, false);

    NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                          m_expectedCWnd,
                          "cwnd does not match value computed with correct arithmetic");
}

/**
 * @ingroup internet-test
 *
 * @brief Check the SegmentQuota attribute of TcpPrrRecovery
 *
 * With 1000-byte segments, ssThresh 12000 and RecoverFS 26000, the third
 * duplicate ACK of the recovery (the first one counted by EnterRecovery)
 * brings prrDelivered to 3000 bytes, and the proportional quota to
 * CEIL(3000 * 12000 / 26000) = 1385 bytes, of which 1000 were sent (the fast
 * retransmission). Counted in bytes (RFC 9937), the 385 bytes left are less
 * than a segment; rounded up to whole segments, as Linux counts PRR in
 * packets, the quota is 2000 bytes and allows one more segment.
 */
class PrrSegmentQuotaTest : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param segmentQuota Value of the SegmentQuota attribute.
     * @param expectedSendCount Expected cwnd minus bytes in flight after the ACK.
     * @param name Test description.
     */
    PrrSegmentQuotaTest(bool segmentQuota, uint32_t expectedSendCount, const std::string& name)
        : TestCase(name),
          m_segmentQuota(segmentQuota),
          m_expectedSendCount(expectedSendCount)
    {
    }

  private:
    void DoRun() override;

    bool m_segmentQuota;          //!< Value of the SegmentQuota attribute.
    uint32_t m_expectedSendCount; //!< Expected cwnd minus bytes in flight after the ACK.
};

void
PrrSegmentQuotaTest::DoRun()
{
    const uint32_t segmentSize = 1000;
    Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
    state->m_segmentSize = segmentSize;
    state->m_ssThresh = 12000;
    state->m_bytesInFlight = 22000;
    state->m_sackEnabled = false;

    Ptr<TcpPrrRecovery> recovery = CreateObject<TcpPrrRecovery>();
    recovery->SetAttribute("SegmentQuota", BooleanValue(m_segmentQuota));

    recovery->EnterRecovery(state, 3, 26000, 0, 0);
    recovery->UpdateBytesSent(segmentSize);
    state->m_bytesInFlight = 22000;
    recovery->DoRecovery(state, 0, true);
    recovery->DoRecovery(state, 0, true);

    NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get() - state->m_bytesInFlight.Get(),
                          m_expectedSendCount,
                          "Unexpected send quota");
}

/**
 * @ingroup internet-test
 *
 * @brief Test the PriorCwndRecoverFs attribute of TcpPrrRecovery
 *
 * A congestion control has cut cWnd to one segment, below the 149 segments in
 * flight, when the recovery starts, with a slow start threshold of two
 * segments. On the first duplicate ACK (without SACK, one segment delivered),
 * RecoverFS is the flight size by default (RFC 9937): the proportional quota,
 * CEIL(1000 * 2000 / 149000) = 14 bytes, gives way to the fast retransmission
 * of one segment. With the attribute set, RecoverFS is the window before the
 * recovery (one segment, as prior_cwnd of Linux): the quota is 2000 bytes.
 */
class PrrPriorCwndTest : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param priorCwnd Value of the PriorCwndRecoverFs attribute.
     * @param expectedSendCount Expected cwnd minus bytes in flight after the ACK.
     * @param name Test description.
     */
    PrrPriorCwndTest(bool priorCwnd, uint32_t expectedSendCount, const std::string& name)
        : TestCase(name),
          m_priorCwnd(priorCwnd),
          m_expectedSendCount(expectedSendCount)
    {
    }

  private:
    void DoRun() override;

    bool m_priorCwnd;             //!< Value of the PriorCwndRecoverFs attribute.
    uint32_t m_expectedSendCount; //!< Expected cwnd minus bytes in flight after the ACK.
};

void
PrrPriorCwndTest::DoRun()
{
    const uint32_t segmentSize = 1000;
    Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
    state->m_segmentSize = segmentSize;
    state->m_cWnd = segmentSize;
    state->m_ssThresh = 2 * segmentSize;
    state->m_bytesInFlight = 149 * segmentSize;
    state->m_sackEnabled = false;

    Ptr<TcpPrrRecovery> recovery = CreateObject<TcpPrrRecovery>();
    recovery->SetAttribute("PriorCwndRecoverFs", BooleanValue(m_priorCwnd));
    recovery->EnterRecovery(state, 3, 149 * segmentSize, 0, 0);

    NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get() - state->m_bytesInFlight.Get(),
                          m_expectedSendCount,
                          "Unexpected send quota");
}

/**
 * @ingroup internet-test
 *
 * @brief A congestion control which cuts the window to one segment on every
 * ACK of a fast recovery, as TCP-LP may do
 */
class TcpRecoveryCutter : public TcpNewReno
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::TcpRecoveryCutter")
                                .SetParent<TcpNewReno>()
                                .AddConstructor<TcpRecoveryCutter>()
                                .SetGroupName("Internet");
        return tid;
    }

    void PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt) override
    {
        if (tcb->m_congState == TcpSocketState::CA_RECOVERY)
        {
            tcb->m_cWnd = tcb->m_segmentSize;
        }
    }

    Ptr<TcpCongestionOps> Fork() override
    {
        return CopyObject<TcpRecoveryCutter>(this);
    }
};

NS_OBJECT_ENSURE_REGISTERED(TcpRecoveryCutter);

/**
 * @ingroup internet-test
 *
 * @brief The window that PRR sets on an ACK of a fast recovery is the final one
 *
 * The congestion control is told of an ACK before the recovery sets the
 * window (as Linux calls pkts_acked() before tcp_cwnd_reduction()): a window
 * that the congestion control cuts on the ACK does not override the one PRR
 * computes, cWnd = pipe + sndcnt (RFC 9937), which stays above the segments in
 * flight during the recovery.
 */
class PrrCongestionControlCutTest : public TcpGeneralTest
{
  public:
    PrrCongestionControlCutTest()
        : TcpGeneralTest("Prr sets the window after the congestion control")
    {
    }

  protected:
    void ConfigureEnvironment() override
    {
        TcpGeneralTest::ConfigureEnvironment();
        SetAppPktCount(60);
        SetCongestionControl(TcpRecoveryCutter::GetTypeId());
        SetRecoveryAlgorithm(TcpPrrRecovery::GetTypeId());
    }

    void ConfigureProperties() override
    {
        TcpGeneralTest::ConfigureProperties();
        SetInitialCwnd(SENDER, 20);
    }

    Ptr<ErrorModel> CreateReceiverErrorModel() override
    {
        Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
        // The first segment of the sixth write of 500 bytes
        errorModel->AddSeqToKill(SequenceNumber32(1 + 5 * 500));
        return errorModel;
    }

    void ProcessedAck(const Ptr<const TcpSocketState> tcb,
                      const TcpHeader& h,
                      SocketWho who) override
    {
        if (who == SENDER && tcb->m_congState == TcpSocketState::CA_RECOVERY)
        {
            ++m_recoveryAcks;
            NS_TEST_ASSERT_MSG_GT(tcb->m_cWnd.Get(),
                                  tcb->m_segmentSize,
                                  "The congestion control overrode the window of PRR");
        }
    }

    void FinalChecks() override
    {
        NS_TEST_ASSERT_MSG_GT(m_recoveryAcks, 1, "No fast recovery");
    }

  private:
    uint32_t m_recoveryAcks{0}; //!< ACKs processed in fast recovery
};

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

        // Regression: proportional branch (inflight > ssThresh) must not overflow a
        // 32-bit intermediate value. EnterRecovery sets RecoverFS = 200000 and (via the
        // dupack bump) m_prrDelivered = segmentSize = 1000; the tested ACK delivers
        // 50000 bytes, so m_prrDelivered becomes 51000. Correct arithmetic gives
        // ceil(51000 * 100000 / 200000) = 25500 and cwnd = 200000 + 25500 = 225500.
        // Before the fix, 51000 * 100000 = 5.1e9 overflowed the uint32_t product.
        AddTestCase(new PrrRecoveryArithmeticTest(
                        1000,
                        100000,
                        200000,
                        0,
                        50000,
                        225500,
                        "Prr proportional branch must not overflow 32-bit product"),
                    TestCase::Duration::QUICK);

        // Regression: PRR-CRB branch (inflight <= ssThresh) must not wrap the unsigned
        // subtraction m_prrDelivered - m_prrOut. After EnterRecovery m_prrDelivered is
        // 1000; 50000 bytes are reported sent so m_prrOut is 50000; the tested ACK
        // delivers 1000 bytes (m_prrDelivered -> 2000). Signed arithmetic gives
        // max(2000 - 50000, 1000) = 1000 and cwnd = 50000 + 1000 = 51000. Before the
        // fix, the unsigned subtraction wrapped.
        AddTestCase(
            new PrrRecoveryArithmeticTest(1000,
                                          100000,
                                          50000,
                                          50000,
                                          1000,
                                          51000,
                                          "Prr CRB branch must not wrap unsigned subtraction"),
            TestCase::Duration::QUICK);

        AddTestCase(new PrrSegmentQuotaTest(false, 385, "Prr send quota counted in bytes"),
                    TestCase::Duration::QUICK);
        AddTestCase(
            new PrrSegmentQuotaTest(true, 1000, "Prr send quota rounded up to whole segments"),
            TestCase::Duration::QUICK);
        AddTestCase(new PrrPriorCwndTest(false, 1000, "Prr RecoverFS is the flight size"),
                    TestCase::Duration::QUICK);
        AddTestCase(new PrrPriorCwndTest(true, 2000, "Prr RecoverFS is the prior window"),
                    TestCase::Duration::QUICK);
        AddTestCase(new PrrCongestionControlCutTest(), TestCase::Duration::QUICK);
    }
};

static PrrRecoveryTestSuite g_TcpPrrRecoveryTest; //!< Static variable for test initialization
