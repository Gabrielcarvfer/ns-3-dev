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
#include "ns3/tcp-cubic.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-socket-base.h"
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
 * @brief Check that every partial ACK in fast recovery restarts the RTO timer (see @issueid{78}).
 *
 * @RFC{6298}, Section 5.3, restarts the timer on every ACK of new data: the
 * Slow-but-Steady NewReno variant (@RFC{3782}, Section 4; see @RFC{6582},
 * Appendix B), not the Impatient one of @RFC{6582}, Section 3.2, step 3;
 * @RFC{6675}, Section 6, permits it. Without SACK, five drops in one window
 * make the recovery span several partial ACKs and last longer than the RTO.
 * Expected: no RTO expiration.
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
    void ConfigureProperties() override;
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

void
TcpPartialAckRtoTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    // Room for the timestamp option on top of the expected payload
    SetSegmentSize(SENDER, GetSegSize(SENDER) + TcpSocketBase::GetSegmentOptionsSize());
}

Ptr<TcpSocketMsgBase>
TcpPartialAckRtoTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("InitialCwnd", UintegerValue(10));
    socket->SetAttribute("Sack", BooleanValue(false));
    // Shorter than the recovery: 5 losses, one recovered per ~200 ms RTT
    socket->SetAttribute("MinRto", TimeValue(MilliSeconds(500)));
    return socket;
}

Ptr<ErrorModel>
TcpPartialAckRtoTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    // Every other 500 byte segment of one window, late enough for cwnd to
    // produce duplicate ACKs (first byte at sequence number 1)
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
 * @brief Check that partial ACKs in CA_LOSS restart the RTO timer after every timeout.
 *
 * @RFC{6298}, Section 5.3, restarts the timer on every ACK of new data,
 * including in CA_LOSS and after any timeout. Segments and their
 * retransmissions are dropped so that recovery takes several timeouts, each
 * followed by a partial ACK. Expected: the last timeout fires after a fresh
 * RTO, sooner than the backed off timeout running before that partial ACK.
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
    void ConfigureProperties() override;
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

void
TcpLossPartialAckRtoTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    // Room for the timestamp option on top of the expected payload
    SetSegmentSize(SENDER, GetSegSize(SENDER) + TcpSocketBase::GetSegmentOptionsSize());
}

Ptr<TcpSocketMsgBase>
TcpLossPartialAckRtoTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("InitialCwnd", UintegerValue(10));
    socket->SetAttribute("Sack", BooleanValue(false));
    socket->SetAttribute("MinRto", TimeValue(MilliSeconds(500)));
    return socket;
}

Ptr<ErrorModel>
TcpLossPartialAckRtoTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    // Three segments and their retransmissions: each is recovered by a
    // timeout, followed by a partial ACK in CA_LOSS
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
    // One timeout per drop that only the timer can recover
    NS_TEST_ASSERT_MSG_EQ(m_rtoTimes.size(), 4, "Unexpected number of retransmission timeouts");

    // The partial ACK after the third timeout restarts the timer with a fresh
    // RTO, shorter than the backed off one running when it arrived
    Time backedOff = m_rtoTimes[2] - m_rtoTimes[1];
    NS_TEST_ASSERT_MSG_LT(m_rtoTimes[3] - m_rtoTimes[2],
                          backedOff,
                          "The partial ACK after a later timeout did not reset the "
                          "retransmission timer");
}

/**
 * @ingroup internet-test
 *
 * @brief Check that the ACK which ends the CA_LOSS state grows the window, within bounds.
 *
 * A segment and its fast retransmission are dropped, so a timeout recovers
 * it and its ACK covers everything up to the recovery point. That first
 * slow start ACK must grow cwnd beyond the one segment loss window
 * (RFC 5681, Section 3.1). As in Linux, growth is bounded by the newly
 * delivered segments (not those SACKed before the timeout) and slow start
 * stops at ssthresh, the rest going to congestion avoidance, so TcpCubic
 * does not burst a whole window.
 */
class TcpLossExitCwndTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     * @param congControl Congestion control of the sender.
     * @param sack Whether SACK is enabled.
     */
    TcpLossExitCwndTest(const std::string& name, TypeId congControl, bool sack)
        : TcpGeneralTest(name),
          m_congControl(congControl),
          m_sack(sack)
    {
    }

  protected:
    void ConfigureProperties() override;
    void ConfigureEnvironment() override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override;
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

    TypeId m_congControl;         //!< Congestion control of the sender
    bool m_sack;                  //!< Whether SACK is enabled
    uint32_t m_rtos{0};           //!< Retransmission timeouts at the sender
    uint32_t m_lossExits{0};      //!< Transitions out of CA_LOSS
    uint32_t m_cWndBefore{0};     //!< Congestion window before the ACK ending CA_LOSS
    uint32_t m_ssThreshBefore{0}; //!< Slow start threshold before the ACK ending CA_LOSS
};

void
TcpLossExitCwndTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetCongestionControl(m_congControl);
    SetAppPktSize(500);
    SetAppPktCount(60);
    SetPropagationDelay(MilliSeconds(100));
}

void
TcpLossExitCwndTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    // Room for the timestamp option on top of the expected payload
    SetSegmentSize(SENDER, GetSegSize(SENDER) + TcpSocketBase::GetSegmentOptionsSize());
}

Ptr<TcpSocketMsgBase>
TcpLossExitCwndTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("InitialCwnd", UintegerValue(40));
    socket->SetAttribute("Sack", BooleanValue(m_sack));
    socket->SetAttribute("MinRto", TimeValue(MilliSeconds(500)));
    return socket;
}

Ptr<TcpSocketMsgBase>
TcpLossExitCwndTest::CreateReceiverSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateReceiverSocket(node);
    socket->SetAttribute("Sack", BooleanValue(m_sack));
    return socket;
}

Ptr<ErrorModel>
TcpLossExitCwndTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    // The segment and its fast retransmission
    errorModel->AddSeqToKill(SequenceNumber32(1001));
    errorModel->AddSeqToKill(SequenceNumber32(1001));
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
        // The congestion control runs after this trace, for the same ACK
        Ptr<const TcpSocketState> tcb = GetTcb(SENDER);
        m_cWndBefore = tcb->m_cWnd.Get();
        m_ssThreshBefore = tcb->m_ssThresh.Get();
        Simulator::ScheduleNow(&TcpLossExitCwndTest::CheckCwnd, this);
    }
}

void
TcpLossExitCwndTest::CheckCwnd()
{
    Ptr<const TcpSocketState> tcb = GetTcb(SENDER);
    uint32_t segmentSize = tcb->m_segmentSize;
    uint32_t cWnd = tcb->m_cWnd.Get();
    NS_TEST_ASSERT_MSG_GT_OR_EQ(cWnd,
                                2 * segmentSize,
                                "The ACK ending the CA_LOSS state did not grow the window");
    // The segments SACKed before the timeout are not credited again
    uint32_t delivered = tcb->m_lastAckedSackedBytes;
    uint32_t deliveredSegments = (delivered + segmentSize - 1) / segmentSize;
    NS_TEST_ASSERT_MSG_LT_OR_EQ(cWnd,
                                m_cWndBefore + deliveredSegments * segmentSize,
                                "The window grew by more than the newly delivered segments");
    // Slow start stops at ssthresh (rounded up to a segment), congestion
    // avoidance adds at most one segment
    NS_TEST_ASSERT_MSG_LT_OR_EQ(cWnd,
                                std::max(m_ssThreshBefore, m_cWndBefore) + 2 * segmentSize,
                                "The window grew past the slow start threshold");
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
 * Without SACK, a segment and its fast retransmission are dropped, so the
 * timer expires in fast recovery. Expected (@RFC{6675}, Section 5.1;
 * @RFC{6582}, Section 3.2, step 4; @RFC{5681}, Section 3.1): CA_RECOVERY ->
 * CA_LOSS, cwnd of one segment, ssthresh >= 2 segments, duplicate ACK
 * guesses cleared, only the first unacknowledged segment retransmitted, then
 * CA_OPEN without a new fast recovery.
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
    void ConfigureProperties() override;
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

void
TcpRecoveryRtoTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    // Room for the timestamp option on top of the expected payload
    SetSegmentSize(SENDER, GetSegSize(SENDER) + TcpSocketBase::GetSegmentOptionsSize());
}

Ptr<TcpSocketMsgBase>
TcpRecoveryRtoTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("InitialCwnd", UintegerValue(10));
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
        for (bool sack : {false, true})
        {
            std::string sackName = sack ? ", SACK" : ", no SACK";
            AddTestCase(new TcpLossExitCwndTest("The ACK ending CA_LOSS grows the window, NewReno" +
                                                    sackName,
                                                TcpNewReno::GetTypeId(),
                                                sack),
                        TestCase::Duration::QUICK);
            AddTestCase(
                new TcpLossExitCwndTest("The ACK ending CA_LOSS grows the window, Cubic" + sackName,
                                        TcpCubic::GetTypeId(),
                                        sack),
                TestCase::Duration::QUICK);
        }
        AddTestCase(new TcpRecoveryRtoTest("A timeout during fast recovery enters CA_LOSS"),
                    TestCase::Duration::QUICK);
    }
};

static TcpPartialAckRtoTestSuite g_tcpPartialAckRtoTestSuite; //!< static var for test init
