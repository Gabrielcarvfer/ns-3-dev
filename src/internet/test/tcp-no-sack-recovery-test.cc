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
#include "ns3/tcp-prr-recovery.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/test.h"

#include <limits>
#include <string>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpNoSackRecoveryTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief Check the rate reduction of a fast recovery without SACK.
 *
 * A segment is dropped in a large window, with SACK disabled, so that a long
 * train of duplicate ACKs follows. With Proportional Rate Reduction
 * (@RFC{9937}), each duplicate ACK delivers one segment (Section 6.2), and the
 * sender transmits about ssThresh/RecoverFS, here one half, new segment per
 * duplicate ACK while the flight is above ssThresh: the bytes in flight on
 * which the window is sized must account for the segment that the duplicate
 * ACK reports as delivered, otherwise one segment per duplicate ACK is sent.
 *
 * The ACK which ends the recovery covers the retransmission and the segments
 * already reported by the duplicate ACKs: it delivers the retransmitted
 * segments not reported by a duplicate ACK only (@RFC{9937}, Section 6.2: the
 * change in SND.UNA minus one SMSS per preceding duplicate ACK).
 */
class TcpNoSackPrrTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     */
    TcpNoSackPrrTest(const std::string& name)
        : TcpGeneralTest(name)
    {
    }

  protected:
    void ConfigureProperties() override;
    void ConfigureEnvironment() override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                        const TcpSocketState::TcpCongState_t newValue) override;
    void FinalChecks() override;

  private:
    SequenceNumber32 m_highTx{0};    //!< Highest data sequence number sent
    SequenceNumber32 m_lastAck{0};   //!< Last acknowledgment number received
    uint32_t m_dupAcksInRecovery{0}; //!< Duplicate ACKs received in CA_RECOVERY
    uint32_t m_newSegsInRecovery{0}; //!< New data segments sent in CA_RECOVERY
    uint32_t m_recoveryExits{0};     //!< Transitions from CA_RECOVERY to CA_OPEN
    uint32_t m_deliveredAtExit{0};   //!< Bytes delivered by the ACK ending the recovery
};

void
TcpNoSackPrrTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetRecoveryAlgorithm(TcpPrrRecovery::GetTypeId());
    SetAppPktSize(500);
    SetAppPktCount(120);
    SetAppPktInterval(MicroSeconds(10));
    SetPropagationDelay(MilliSeconds(50));
}

void
TcpNoSackPrrTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetInitialCwnd(SENDER, 40);
    // The segments carry the timestamp option on top of the data size the
    // test expects
    SetSegmentSize(SENDER, GetSegSize(SENDER) + TcpSocketBase::GetSegmentOptionsSize());
}

Ptr<TcpSocketMsgBase>
TcpNoSackPrrTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("Sack", BooleanValue(false));
    socket->SetAttribute("MinRto", TimeValue(Seconds(1)));
    return socket;
}

Ptr<TcpSocketMsgBase>
TcpNoSackPrrTest::CreateReceiverSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateReceiverSocket(node);
    socket->SetAttribute("Sack", BooleanValue(false));
    return socket;
}

Ptr<ErrorModel>
TcpNoSackPrrTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    errorModel->AddSeqToKill(SequenceNumber32(2001));
    return errorModel;
}

void
TcpNoSackPrrTest::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who != SENDER || p->GetSize() == 0)
    {
        return;
    }
    SequenceNumber32 seq = ToRelativeSeq(SENDER, h.GetSequenceNumber());
    if (seq >= m_highTx)
    {
        m_highTx = seq + p->GetSize();
        if (GetTcb(SENDER)->m_congState == TcpSocketState::CA_RECOVERY)
        {
            m_newSegsInRecovery++;
        }
    }
}

void
TcpNoSackPrrTest::Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who != SENDER || !(h.GetFlags() & TcpHeader::ACK))
    {
        return;
    }
    SequenceNumber32 ack = ToRelativeSeq(SENDER, h.GetAckNumber());
    if (ack == m_lastAck && p->GetSize() == 0 &&
        GetTcb(SENDER)->m_congState == TcpSocketState::CA_RECOVERY)
    {
        m_dupAcksInRecovery++;
    }
    m_lastAck = ack;
}

void
TcpNoSackPrrTest::CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                                 const TcpSocketState::TcpCongState_t newValue)
{
    if (oldValue == TcpSocketState::CA_RECOVERY && newValue == TcpSocketState::CA_OPEN)
    {
        m_recoveryExits++;
        m_deliveredAtExit = GetTcb(SENDER)->m_lastAckedSackedBytes;
    }
}

void
TcpNoSackPrrTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_recoveryExits, 1, "The fast recovery was not left once");
    NS_TEST_ASSERT_MSG_GT(m_dupAcksInRecovery, 20, "Too few duplicate ACKs for the test");
    NS_LOG_DEBUG("New segments " << m_newSegsInRecovery << " for " << m_dupAcksInRecovery
                                 << " duplicate ACKs in fast recovery");
    // About one new segment per two duplicate ACKs, rather than one per
    // duplicate ACK
    NS_TEST_ASSERT_MSG_LT_OR_EQ(4 * m_newSegsInRecovery,
                                3 * m_dupAcksInRecovery,
                                "The sender did not reduce its rate during the fast recovery");
    NS_TEST_ASSERT_MSG_GT_OR_EQ(4 * m_newSegsInRecovery,
                                m_dupAcksInRecovery,
                                "The sender stalled during the fast recovery");
    uint32_t segmentSize = GetTcb(SENDER)->m_segmentSize;
    NS_TEST_ASSERT_MSG_GT(m_deliveredAtExit, 0, "The ACK ending the recovery delivered nothing");
    // The retransmission, plus the segment whose ACK, delayed at the
    // receiver, was not a duplicate one
    NS_TEST_ASSERT_MSG_LT_OR_EQ(m_deliveredAtExit,
                                2 * segmentSize,
                                "The ACK ending the recovery credited again the segments "
                                "reported by the duplicate ACKs");
}

/**
 * @ingroup internet-test
 *
 * @brief Check the NonSackUndo attribute of TcpSocketBase.
 *
 * Without SACK, a segment is dropped (once, so that a fast recovery repairs
 * it, or twice, so that the retransmission timer does). The receiver holds
 * every segment sent after the lost one, and the ACK of the retransmission
 * reaches the recovery point exactly. With NonSackUndo, as Linux, the
 * recovery is held at that ACK, and the next ACK, above the recovery point,
 * ends it and restores the window and the slow start threshold of before the
 * loss. Without NonSackUndo, the recovery ends at the ACK of the recovery
 * point with the reduced slow start threshold (@RFC{6582}, @RFC{5681}).
 */
class TcpNonSackUndoTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     * @param undo Value of the NonSackUndo attribute.
     * @param rto Whether the retransmission is lost too, so that a timeout
     * repairs the loss.
     */
    TcpNonSackUndoTest(const std::string& name, bool undo, bool rto)
        : TcpGeneralTest(name),
          m_undo(undo),
          m_rto(rto)
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
     * @brief Record the window once the ACK ending the recovery is processed.
     */
    void RecordExit();

    bool m_undo;                //!< Value of the NonSackUndo attribute
    bool m_rto;                 //!< Whether the retransmission is lost too
    uint32_t m_rtos{0};         //!< Retransmission timeouts at the sender
    uint32_t m_exits{0};        //!< Transitions to CA_OPEN from CA_RECOVERY or CA_LOSS
    uint32_t m_cWndBefore{0};   //!< cWnd when the fast recovery started
    uint32_t m_ssThBefore{0};   //!< ssThresh when the fast recovery started
    uint32_t m_cWndAfter{0};    //!< cWnd after the ACK ending the recovery
    uint32_t m_ssThAfter{0};    //!< ssThresh after the ACK ending the recovery
    SequenceNumber32 m_exitAck; //!< Acknowledgment number ending the recovery
    SequenceNumber32 m_recover; //!< Highest data sent when the last recovery started
};

void
TcpNonSackUndoTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetRecoveryAlgorithm(TcpPrrRecovery::GetTypeId());
    SetAppPktSize(500);
    SetAppPktCount(120);
    SetAppPktInterval(MicroSeconds(10));
    SetPropagationDelay(MilliSeconds(50));
}

void
TcpNonSackUndoTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetInitialCwnd(SENDER, 20);
    // The segments carry the timestamp option on top of the data size the
    // test expects
    SetSegmentSize(SENDER, GetSegSize(SENDER) + TcpSocketBase::GetSegmentOptionsSize());
}

Ptr<TcpSocketMsgBase>
TcpNonSackUndoTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("Sack", BooleanValue(false));
    socket->SetAttribute("NonSackUndo", BooleanValue(m_undo));
    socket->SetAttribute("MinRto", TimeValue(MilliSeconds(500)));
    return socket;
}

Ptr<TcpSocketMsgBase>
TcpNonSackUndoTest::CreateReceiverSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateReceiverSocket(node);
    socket->SetAttribute("Sack", BooleanValue(false));
    return socket;
}

Ptr<ErrorModel>
TcpNonSackUndoTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    errorModel->AddSeqToKill(SequenceNumber32(2001));
    if (m_rto)
    {
        errorModel->AddSeqToKill(SequenceNumber32(2001));
    }
    return errorModel;
}

void
TcpNonSackUndoTest::AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who)
{
    if (who == SENDER)
    {
        m_rtos++;
        m_recover = ToRelativeSeq(SENDER, tcb->m_highTxMark.Get());
    }
}

void
TcpNonSackUndoTest::CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                                   const TcpSocketState::TcpCongState_t newValue)
{
    Ptr<const TcpSocketState> tcb = GetTcb(SENDER);
    if (newValue == TcpSocketState::CA_RECOVERY)
    {
        // The window is reduced right after the state change
        m_cWndBefore = tcb->m_cWnd.Get();
        m_ssThBefore = tcb->m_ssThresh.Get();
        m_recover = ToRelativeSeq(SENDER, tcb->m_highTxMark.Get());
    }
    else if ((oldValue == TcpSocketState::CA_RECOVERY || oldValue == TcpSocketState::CA_LOSS) &&
             newValue == TcpSocketState::CA_OPEN)
    {
        m_exits++;
        m_exitAck = ToRelativeSeq(SENDER, tcb->m_lastAckedSeq);
        Simulator::ScheduleNow(&TcpNonSackUndoTest::RecordExit, this);
    }
}

void
TcpNonSackUndoTest::RecordExit()
{
    Ptr<const TcpSocketState> tcb = GetTcb(SENDER);
    m_cWndAfter = tcb->m_cWnd.Get();
    m_ssThAfter = tcb->m_ssThresh.Get();
}

void
TcpNonSackUndoTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_rtos, (m_rto ? 1 : 0), "Unexpected number of timeouts");
    NS_TEST_ASSERT_MSG_EQ(m_exits, 1, "The recovery was not left once");
    NS_TEST_ASSERT_MSG_EQ(m_ssThBefore,
                          std::numeric_limits<uint32_t>::max(),
                          "The loss did not happen in slow start");
    if (m_undo)
    {
        NS_TEST_ASSERT_MSG_GT(m_exitAck,
                              m_recover,
                              "The recovery was not held at the recovery point");
        NS_TEST_ASSERT_MSG_EQ(m_ssThAfter,
                              m_ssThBefore,
                              "The slow start threshold was not restored");
        NS_TEST_ASSERT_MSG_GT_OR_EQ(m_cWndAfter, m_cWndBefore, "The window was not restored");
    }
    else
    {
        NS_TEST_ASSERT_MSG_EQ(m_exitAck,
                              m_recover,
                              "The recovery did not end at the recovery point");
        NS_TEST_ASSERT_MSG_LT(m_ssThAfter, m_cWndBefore, "The slow start threshold was restored");
        NS_TEST_ASSERT_MSG_LT(m_cWndAfter, m_cWndBefore, "The window was restored");
    }
}

/**
 * @ingroup internet-test
 *
 * @brief TestSuite for the loss recovery without SACK
 */
class TcpNoSackRecoveryTestSuite : public TestSuite
{
  public:
    TcpNoSackRecoveryTestSuite()
        : TestSuite("tcp-no-sack-recovery", Type::UNIT)
    {
        AddTestCase(new TcpNoSackPrrTest("PRR reduces the rate of a fast recovery without SACK"),
                    TestCase::Duration::QUICK);
        for (bool rto : {false, true})
        {
            std::string what = rto ? "timeout recovery" : "fast recovery";
            AddTestCase(new TcpNonSackUndoTest("NonSackUndo undoes a " + what, true, rto),
                        TestCase::Duration::QUICK);
            AddTestCase(
                new TcpNonSackUndoTest("Without NonSackUndo, a " + what + " is kept", false, rto),
                TestCase::Duration::QUICK);
        }
    }
};

static TcpNoSackRecoveryTestSuite g_tcpNoSackRecoveryTestSuite; //!< static var for test init
