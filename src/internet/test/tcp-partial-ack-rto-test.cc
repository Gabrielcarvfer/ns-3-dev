/*
 * Copyright (c) 2026 Centre Tecnològic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "tcp-error-model.h"
#include "tcp-general-test.h"

#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-tx-buffer.h"
#include "ns3/test.h"

#include <map>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpPartialAckRtoTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief Test the RTO reset behavior on partial ACKs (see @issueid{78})
 *
 * @RFC{6298} (Section 5.3) calls for restarting the retransmission timer on
 * every ACK acknowledging new data, including the partial ACKs received
 * during fast recovery: this is the Slow-but-Steady variant of NewReno
 * (@RFC{3782}, Section 4; @RFC{6582} dropped the discussion of the variants,
 * see its Appendix B), which departs from the Impatient variant of
 * @RFC{6582}, Section 3.2, step 3, resetting the timer only upon the first
 * partial acknowledgment of a fast recovery. @RFC{6298}, Section 5.3, and
 * @RFC{6675}, Section 6, permit it.
 *
 * The test drops several segments of the same window, so that (with SACK
 * disabled) the fast recovery phase spans multiple partial ACKs and lasts
 * longer than the RTO. Since every partial ACK must reset the retransmission
 * timer, the timer must not expire during the recovery.
 */
class TcpPartialAckRtoTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     */
    TcpPartialAckRtoTest(const std::string& name)
        : TcpGeneralTest(name)
    {
    }

  protected:
    void ConfigureEnvironment() override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                        const TcpSocketState::TcpCongState_t newValue) override;
    void AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who) override;
    void FinalChecks() override;

  private:
    bool m_recoveryEntered{false}; //!< True if fast recovery was entered.
    uint32_t m_rtoCount{0};        //!< Number of RTO expirations at the sender.
};

void
TcpPartialAckRtoTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(60);
    SetPropagationDelay(MilliSeconds(100));
}

Ptr<TcpSocketMsgBase>
TcpPartialAckRtoTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("InitialCwnd", UintegerValue(10));
    // The dropped sequence numbers assume that the segment payload equals the
    // configured segment size: disable the timestamp option, which would
    // decrease the payload by its size
    socket->SetAttribute("Timestamp", BooleanValue(false));
    // NewReno-style recovery with multiple partial ACKs
    socket->SetAttribute("Sack", BooleanValue(false));
    // Make the RTO shorter than the duration of the fast recovery phase
    // (5 lost segments, recovered at a rate of one per RTT of ~200 ms)
    socket->SetAttribute("MinRto", TimeValue(MilliSeconds(500)));
    return socket;
}

Ptr<ErrorModel>
TcpPartialAckRtoTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    // Drop five alternating segments of a window in the middle of the
    // transfer, when the congestion window is large enough for the drops to
    // be detected via duplicate ACKs (segment size is 500 bytes and the first
    // byte has sequence number 1)
    for (uint32_t seq : {10001, 11001, 12001, 13001, 14001})
    {
        errorModel->AddSeqToKill(SequenceNumber32(seq));
    }
    return errorModel;
}

void
TcpPartialAckRtoTest::CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                                     const TcpSocketState::TcpCongState_t newValue)
{
    NS_LOG_DEBUG("State " << TcpSocketState::TcpCongStateName[oldValue] << " -> "
                          << TcpSocketState::TcpCongStateName[newValue]);
    if (newValue == TcpSocketState::CA_RECOVERY)
    {
        m_recoveryEntered = true;
    }
}

void
TcpPartialAckRtoTest::AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who)
{
    if (who == SENDER)
    {
        NS_LOG_DEBUG("RTO expired at " << Simulator::Now().GetSeconds());
        m_rtoCount++;
    }
}

void
TcpPartialAckRtoTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_recoveryEntered, true, "Fast recovery was not entered");
    NS_TEST_ASSERT_MSG_EQ(m_rtoCount,
                          0,
                          "RTO expired during fast recovery: partial ACKs did not reset the "
                          "retransmission timer");
}

/**
 * @ingroup internet-test
 *
 * @brief Test the RTO reset behavior on partial ACKs after a timeout
 *
 * @RFC{6298} (Section 5.3) restarts the retransmission timer on every ACK
 * acknowledging new data, after a retransmission timeout as much as during
 * a fast recovery, and after every timeout, not only the first one of the
 * connection.
 *
 * The test drops several segments of a window, and their retransmissions,
 * in the order they travel, so that the recovery goes through the
 * retransmission timer several times and a partial ACK is received in the
 * CA_LOSS state after the later timeouts too. Were the partial ACKs after a
 * timeout not to reset the timer, the one following the last timeout would
 * leave the exponentially backed off timer running from the retransmission
 * instead of restarting it, and the last loss would take the doubled
 * timeout, rather than a fresh one, to recover.
 */
class TcpLossPartialAckRtoTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     */
    TcpLossPartialAckRtoTest(const std::string& name)
        : TcpGeneralTest(name)
    {
    }

  protected:
    void ConfigureEnvironment() override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who) override;
    void FinalChecks() override;

  private:
    std::vector<Time> m_rtoTimes; //!< Instants of the RTO expirations at the sender.
};

void
TcpLossPartialAckRtoTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(30);
    SetPropagationDelay(MilliSeconds(100));
}

Ptr<TcpSocketMsgBase>
TcpLossPartialAckRtoTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("InitialCwnd", UintegerValue(10));
    // The dropped sequence numbers assume that the segment payload equals the
    // configured segment size: disable the timestamp option, which would
    // decrease the payload by its size
    socket->SetAttribute("Timestamp", BooleanValue(false));
    socket->SetAttribute("Sack", BooleanValue(false));
    socket->SetAttribute("MinRto", TimeValue(MilliSeconds(500)));
    return socket;
}

Ptr<ErrorModel>
TcpLossPartialAckRtoTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    // Drop three segments of a window and their first retransmissions, so
    // that the recovery of each one has to go through the retransmission
    // timer, and the acknowledgment which follows it is a partial ACK
    // received in the CA_LOSS state
    for (uint32_t seq : {10001, 11001, 12001, 10001, 11001, 11001, 12001})
    {
        errorModel->AddSeqToKill(SequenceNumber32(seq));
    }
    return errorModel;
}

void
TcpLossPartialAckRtoTest::AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who)
{
    if (who == SENDER)
    {
        NS_LOG_DEBUG("RTO expired at " << Simulator::Now().GetSeconds());
        m_rtoTimes.push_back(Simulator::Now());
    }
}

void
TcpLossPartialAckRtoTest::FinalChecks()
{
    // One timeout per dropped transmission which nothing else can recover
    NS_TEST_ASSERT_MSG_EQ(m_rtoTimes.size(), 4, "Unexpected number of retransmission timeouts");

    // The partial ACK following the third timeout restarts the timer with a
    // fresh RTO: the last timeout comes sooner than the doubled backed off
    // timeout which was running when that partial ACK arrived
    Time backedOff = m_rtoTimes[2] - m_rtoTimes[1];
    NS_TEST_ASSERT_MSG_LT(m_rtoTimes[3] - m_rtoTimes[2],
                          backedOff,
                          "The partial ACK after a later timeout did not reset the "
                          "retransmission timer");
}

/**
 * @ingroup internet-test
 *
 * @brief Check that the ACK which ends the CA_LOSS state grows the window.
 *
 * A segment and its fast retransmission are dropped, so that the segment is
 * recovered by the retransmission timer. The receiver holds every segment
 * sent after it, and the acknowledgment of the retransmission covers all of
 * them at once, landing on the recovery point. That acknowledgment is the
 * first one of the slow start which follows the timeout, and it must grow
 * the congestion window from the one segment of the loss window
 * (RFC 5681, Section 3.1): otherwise the sender keeps sending one segment
 * per round trip time until the next acknowledgment.
 */
class TcpLossExitCwndTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     */
    TcpLossExitCwndTest(const std::string& name)
        : TcpGeneralTest(name)
    {
    }

  protected:
    void ConfigureEnvironment() override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who) override;
    void CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                        const TcpSocketState::TcpCongState_t newValue) override;
    void FinalChecks() override;

  private:
    /**
     * @brief Check the congestion window once the ACK ending CA_LOSS is processed.
     */
    void CheckCwnd();

    uint32_t m_rtos{0};      //!< Retransmission timeouts at the sender
    uint32_t m_lossExits{0}; //!< Transitions out of CA_LOSS
};

void
TcpLossExitCwndTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(30);
    SetPropagationDelay(MilliSeconds(100));
}

Ptr<TcpSocketMsgBase>
TcpLossExitCwndTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("InitialCwnd", UintegerValue(10));
    socket->SetAttribute("Timestamp", BooleanValue(false));
    socket->SetAttribute("Sack", BooleanValue(false));
    socket->SetAttribute("MinRto", TimeValue(MilliSeconds(500)));
    return socket;
}

Ptr<ErrorModel>
TcpLossExitCwndTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    // The segment and its fast retransmission
    errorModel->AddSeqToKill(SequenceNumber32(10001));
    errorModel->AddSeqToKill(SequenceNumber32(10001));
    return errorModel;
}

void
TcpLossExitCwndTest::AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who)
{
    if (who == SENDER)
    {
        m_rtos++;
    }
}

void
TcpLossExitCwndTest::CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                                    const TcpSocketState::TcpCongState_t newValue)
{
    if (oldValue == TcpSocketState::CA_LOSS && newValue == TcpSocketState::CA_OPEN)
    {
        m_lossExits++;
        // The congestion control runs after the state change, within the
        // processing of the same ACK
        Simulator::ScheduleNow(&TcpLossExitCwndTest::CheckCwnd, this);
    }
}

void
TcpLossExitCwndTest::CheckCwnd()
{
    Ptr<const TcpSocketState> tcb = GetTcb(SENDER);
    NS_TEST_ASSERT_MSG_GT_OR_EQ(tcb->m_cWnd.Get(),
                                2 * tcb->m_segmentSize,
                                "The ACK ending the CA_LOSS state did not grow the window");
}

void
TcpLossExitCwndTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_rtos, 1, "The lost retransmission was not recovered by one timeout");
    NS_TEST_ASSERT_MSG_EQ(m_lossExits, 1, "The CA_LOSS state was not left once");
}

/**
 * @ingroup internet-test
 *
 * @brief Check the transition from fast recovery to CA_LOSS on a timeout.
 *
 * Without SACK, a segment and its fast retransmission are dropped: nothing
 * reveals the loss of the retransmission, and the retransmission timer
 * expires during the fast recovery. The timeout must terminate the recovery
 * (@RFC{6675}, Section 5.1; @RFC{6582}, Section 3.2, step 4): the sender
 * moves from CA_RECOVERY to CA_LOSS, the congestion window is set to the
 * loss window of one segment (@RFC{5681}, Section 3.1), the slow start
 * threshold is kept at two segments at least, the duplicate ACK guesses are
 * dropped and only the first unacknowledged segment is retransmitted. The
 * sender then goes back to CA_OPEN, without entering fast recovery again.
 */
class TcpRecoveryRtoTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     */
    TcpRecoveryRtoTest(const std::string& name)
        : TcpGeneralTest(name)
    {
    }

  protected:
    void ConfigureEnvironment() override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void BeforeRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who) override;
    void AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who) override;
    void CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                        const TcpSocketState::TcpCongState_t newValue) override;
    void FinalChecks() override;

  private:
    static constexpr uint32_t LOST_SEQ = 10001; //!< Sequence number of the dropped segment

    std::vector<std::string> m_transitions; //!< Observed congestion state transitions
    std::map<uint32_t, uint32_t> m_txCount; //!< Transmissions per data sequence number
    std::vector<uint32_t> m_retxAfterRto;   //!< Retransmissions sent in CA_LOSS
    TcpSocketState::TcpCongState_t m_stateBeforeRto{TcpSocketState::CA_OPEN}; //!< State
    uint32_t m_rtos{0}; //!< Retransmission timeouts at the sender
};

void
TcpRecoveryRtoTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(30);
    SetPropagationDelay(MilliSeconds(100));
}

Ptr<TcpSocketMsgBase>
TcpRecoveryRtoTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("InitialCwnd", UintegerValue(10));
    socket->SetAttribute("Timestamp", BooleanValue(false));
    socket->SetAttribute("Sack", BooleanValue(false));
    socket->SetAttribute("MinRto", TimeValue(MilliSeconds(500)));
    return socket;
}

Ptr<ErrorModel>
TcpRecoveryRtoTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    // The segment and its fast retransmission
    errorModel->AddSeqToKill(SequenceNumber32(LOST_SEQ));
    errorModel->AddSeqToKill(SequenceNumber32(LOST_SEQ));
    return errorModel;
}

void
TcpRecoveryRtoTest::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who != SENDER || p->GetSize() == 0)
    {
        return;
    }
    uint32_t seq = ToRelativeSeq(SENDER, h.GetSequenceNumber()).GetValue();
    if (m_txCount[seq]++ > 0 && GetTcb(SENDER)->m_congState == TcpSocketState::CA_LOSS)
    {
        m_retxAfterRto.push_back(seq);
    }
}

void
TcpRecoveryRtoTest::BeforeRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who)
{
    if (who == SENDER)
    {
        m_stateBeforeRto = tcb->m_congState;
    }
}

void
TcpRecoveryRtoTest::AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who)
{
    if (who != SENDER)
    {
        return;
    }
    m_rtos++;
    NS_TEST_ASSERT_MSG_EQ(m_stateBeforeRto,
                          TcpSocketState::CA_RECOVERY,
                          "The timeout did not expire during the fast recovery");
    NS_TEST_ASSERT_MSG_EQ(tcb->m_congState, TcpSocketState::CA_LOSS, "CA_LOSS not entered");
    NS_TEST_ASSERT_MSG_EQ(tcb->m_cWnd.Get(), tcb->m_segmentSize, "Loss window is not one segment");
    NS_TEST_ASSERT_MSG_GT_OR_EQ(tcb->m_ssThresh.Get(),
                                2 * tcb->m_segmentSize,
                                "Slow start threshold below two segments");
    NS_TEST_ASSERT_MSG_EQ(GetTxBuffer(SENDER)->GetSacked(), 0, "Guessed SACKs left after the RTO");
    NS_TEST_ASSERT_MSG_EQ(tcb->m_bytesInFlight.Get(),
                          tcb->m_segmentSize,
                          "More than the retransmitted segment in flight after the RTO");
}

void
TcpRecoveryRtoTest::CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                                   const TcpSocketState::TcpCongState_t newValue)
{
    m_transitions.push_back(std::string(TcpSocketState::TcpCongStateName[oldValue]) + "->" +
                            TcpSocketState::TcpCongStateName[newValue]);
}

void
TcpRecoveryRtoTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_rtos, 1, "Not exactly one retransmission timeout");
    std::vector<std::string> expected{"CA_OPEN->CA_DISORDER",
                                      "CA_DISORDER->CA_RECOVERY",
                                      "CA_RECOVERY->CA_LOSS",
                                      "CA_LOSS->CA_OPEN"};
    NS_TEST_ASSERT_MSG_EQ(m_transitions.size(), expected.size(), "Unexpected transitions");
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        NS_TEST_ASSERT_MSG_EQ(m_transitions[i], expected[i], "Unexpected transition");
    }
    NS_TEST_ASSERT_MSG_EQ(m_txCount[LOST_SEQ], 3, "The lost segment was not sent three times");
    NS_TEST_ASSERT_MSG_EQ(m_retxAfterRto.size(), 1, "Not exactly one retransmission in CA_LOSS");
    NS_TEST_ASSERT_MSG_EQ(m_retxAfterRto.front(), LOST_SEQ, "Wrong segment retransmitted");
}

/**
 * @ingroup internet-test
 *
 * @brief Partial ACK RTO reset TestSuite
 */
class TcpPartialAckRtoTestSuite : public TestSuite
{
  public:
    TcpPartialAckRtoTestSuite()
        : TestSuite("tcp-partial-ack-rto", Type::UNIT)
    {
        AddTestCase(new TcpPartialAckRtoTest("Partial ACKs reset the retransmission timer"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpLossPartialAckRtoTest(
                        "Partial ACKs after a timeout reset the retransmission timer"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpLossExitCwndTest("The ACK ending CA_LOSS grows the window"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpRecoveryRtoTest("A timeout during fast recovery enters CA_LOSS"),
                    TestCase::Duration::QUICK);
    }
};

static TcpPartialAckRtoTestSuite g_tcpPartialAckRtoTestSuite; //!< static var for test init
