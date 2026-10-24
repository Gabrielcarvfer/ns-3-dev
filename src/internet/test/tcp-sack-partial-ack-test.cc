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
#include "ns3/test.h"

#include <set>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpSackPartialAckTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief Test that no spurious retransmission occurs on SACK partial ACKs
 * (see @issueid{988})
 *
 * With SACK enabled, the retransmissions during loss recovery are governed by
 * the scoreboard and the pipe (@RFC{6675}, Section 5): upon receipt of a
 * partial ACK, only the segments reported lost by the scoreboard are to be
 * retransmitted (via NextSeg()). The NewReno-style forced retransmission of
 * the first unacknowledged segment on every partial ACK (@RFC{6582}) must not
 * be performed, otherwise segments are retransmitted more times than needed.
 *
 * The test drops segments with SACK enabled and checks that the number of
 * retransmitted data segments is equal to the number of dropped segments,
 * without retransmission timeouts. Two cases are covered:
 *
 * - two consecutive segments of the same window;
 * - a segment in the middle of the transfer and the last segment of an
 *   application limited transfer. Once the first loss is repaired, the
 *   partial ACK leaves only the last segment outstanding, with no SACKed
 *   data above it: it is not deemed lost (rule 1 of NextSeg()), and there
 *   is no new data to send (rule 2). NextSeg() returns it under rule 3,
 *   which ns-3 relaxes when no SACK block is held in the scoreboard,
 *   standing in for the "rescue" retransmission of rule 4 of @RFC{6675},
 *   Section 4, which is not implemented. Without it, the recovery would
 *   stall until the retransmission timer expires.
 */
class TcpSackPartialAckTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     * @param drops Sequence numbers of the segments to drop.
     * @param pktCount Number of 500 bytes packets sent by the application.
     */
    TcpSackPartialAckTest(const std::string& name,
                          const std::vector<uint32_t>& drops,
                          uint32_t pktCount)
        : TcpGeneralTest(name),
          m_drops(drops),
          m_pktCount(pktCount)
    {
    }

  protected:
    void ConfigureEnvironment() override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                        const TcpSocketState::TcpCongState_t newValue) override;
    void AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who) override;
    void FinalChecks() override;

  private:
    std::vector<uint32_t> m_drops;       //!< Sequence numbers of the dropped segments.
    uint32_t m_pktCount;                 //!< Number of packets sent by the application.
    uint32_t m_rtoCount{0};              //!< Number of RTO expirations at the sender.
    bool m_recoveryEntered{false};       //!< True if fast recovery was entered.
    uint32_t m_retransmissions{0};       //!< Number of retransmitted data segments.
    std::set<SequenceNumber32> m_seqTxs; //!< Sequence numbers of transmitted data segments.
};

void
TcpSackPartialAckTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(m_pktCount);
    SetPropagationDelay(MilliSeconds(50));
}

Ptr<TcpSocketMsgBase>
TcpSackPartialAckTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("InitialCwnd", UintegerValue(10));
    // The dropped sequence numbers assume that the segment payload equals the
    // configured segment size: disable the timestamp option, which would
    // decrease the payload by its size
    socket->SetAttribute("Timestamp", BooleanValue(false));
    socket->SetAttribute("Sack", BooleanValue(true));
    // Avoid retransmission timeouts during the test
    socket->SetAttribute("MinRto", TimeValue(Seconds(10)));
    return socket;
}

Ptr<ErrorModel>
TcpSackPartialAckTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    // The segment size is 500 bytes and the first byte has sequence number 1
    for (uint32_t seq : m_drops)
    {
        errorModel->AddSeqToKill(SequenceNumber32(seq));
    }
    return errorModel;
}

void
TcpSackPartialAckTest::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who == SENDER && p->GetSize() > 0)
    {
        if (!m_seqTxs.insert(h.GetSequenceNumber()).second)
        {
            NS_LOG_DEBUG("Retransmission of seq " << h.GetSequenceNumber());
            m_retransmissions++;
        }
    }
}

void
TcpSackPartialAckTest::CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                                      const TcpSocketState::TcpCongState_t newValue)
{
    if (newValue == TcpSocketState::CA_RECOVERY)
    {
        m_recoveryEntered = true;
    }
}

void
TcpSackPartialAckTest::AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who)
{
    if (who == SENDER)
    {
        m_rtoCount++;
    }
}

void
TcpSackPartialAckTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_recoveryEntered, true, "Fast recovery was not entered");
    NS_TEST_ASSERT_MSG_EQ(m_rtoCount, 0, "The recovery stalled until a retransmission timeout");
    NS_TEST_ASSERT_MSG_EQ(m_retransmissions,
                          m_drops.size(),
                          "Number of retransmissions differs from the number of losses");
    NS_TEST_ASSERT_MSG_EQ(m_seqTxs.size(), m_pktCount, "Not all the segments were sent");
}

/**
 * @ingroup internet-test
 *
 * @brief Test a SACK loss recovery limited by the receive window.
 *
 * The receive buffer is smaller than the congestion window reached in slow
 * start, so that the whole receive window is outstanding when a segment is
 * lost: during the recovery, no new data can be sent (rule 2 of NextSeg()
 * fails, @RFC{6675}, Section 4) even when the pipe falls below the
 * congestion window, until the cumulative ACK moves the window. The
 * recovery must complete without retransmission timeouts and with a single
 * retransmission of the lost segments.
 */
class TcpRwndLimitedRecoveryTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     * @param drops Sequence numbers of the segments to drop.
     */
    TcpRwndLimitedRecoveryTest(const std::string& name, const std::vector<uint32_t>& drops)
        : TcpGeneralTest(name),
          m_drops(drops)
    {
    }

  protected:
    void ConfigureEnvironment() override;
    void ConfigureProperties() override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                        const TcpSocketState::TcpCongState_t newValue) override;
    void AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who) override;
    void FinalChecks() override;

  private:
    static constexpr uint32_t RCV_BUFFER = 4000; //!< Receive buffer (8 segments)

    std::vector<uint32_t> m_drops;       //!< Sequence numbers of the dropped segments
    uint32_t m_recoveries{0};            //!< Number of fast recovery entries
    uint32_t m_rtoCount{0};              //!< Number of RTO expirations at the sender
    uint32_t m_retransmissions{0};       //!< Number of retransmitted data segments
    bool m_rwndLimited{false};           //!< True if rwnd < cwnd when recovery started
    std::set<SequenceNumber32> m_seqTxs; //!< Sequence numbers of transmitted data segments
};

void
TcpRwndLimitedRecoveryTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(60);
    SetPropagationDelay(MilliSeconds(50));
}

void
TcpRwndLimitedRecoveryTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetRcvBufSize(RECEIVER, RCV_BUFFER);
}

Ptr<TcpSocketMsgBase>
TcpRwndLimitedRecoveryTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("InitialCwnd", UintegerValue(10));
    socket->SetAttribute("Timestamp", BooleanValue(false));
    socket->SetAttribute("Sack", BooleanValue(true));
    // Any stall of the recovery shows up as a retransmission timeout
    socket->SetAttribute("MinRto", TimeValue(Seconds(10)));
    return socket;
}

Ptr<ErrorModel>
TcpRwndLimitedRecoveryTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    for (uint32_t seq : m_drops)
    {
        errorModel->AddSeqToKill(SequenceNumber32(seq));
    }
    return errorModel;
}

void
TcpRwndLimitedRecoveryTest::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who == SENDER && p->GetSize() > 0 && !m_seqTxs.insert(h.GetSequenceNumber()).second)
    {
        NS_LOG_DEBUG("Retransmission of seq " << h.GetSequenceNumber());
        m_retransmissions++;
    }
}

void
TcpRwndLimitedRecoveryTest::CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                                           const TcpSocketState::TcpCongState_t newValue)
{
    if (newValue == TcpSocketState::CA_RECOVERY)
    {
        m_recoveries++;
        // The window is reduced after the state change
        m_rwndLimited = GetRWnd(SENDER) < GetTcb(SENDER)->m_cWnd.Get();
    }
}

void
TcpRwndLimitedRecoveryTest::AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who)
{
    if (who == SENDER)
    {
        m_rtoCount++;
    }
}

void
TcpRwndLimitedRecoveryTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_recoveries, 1, "Fast recovery not entered exactly once");
    NS_TEST_ASSERT_MSG_EQ(m_rwndLimited, true, "The receive window was not the limit");
    NS_TEST_ASSERT_MSG_EQ(m_rtoCount, 0, "The recovery stalled until a retransmission timeout");
    NS_TEST_ASSERT_MSG_EQ(m_retransmissions,
                          m_drops.size(),
                          "Number of retransmissions differs from the number of losses");
    NS_TEST_ASSERT_MSG_EQ(m_seqTxs.size(), 60, "Not all the segments were sent");
}

/**
 * @ingroup internet-test
 *
 * @brief SACK partial ACK TestSuite
 */
class TcpSackPartialAckTestSuite : public TestSuite
{
  public:
    TcpSackPartialAckTestSuite()
        : TestSuite("tcp-sack-partial-ack", Type::UNIT)
    {
        AddTestCase(new TcpSackPartialAckTest("No spurious retransmission on SACK partial ACKs",
                                              {10001, 10501},
                                              60),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpSackPartialAckTest("Last segment recovered after a SACK partial ACK",
                                              {10001, 14501},
                                              30),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpRwndLimitedRecoveryTest("Receive window limited recovery", {10001}),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpRwndLimitedRecoveryTest("Receive window limited recovery, two losses",
                                                   {10001, 11001}),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpRwndLimitedRecoveryTest("Receive window limited recovery, loss at the "
                                                   "edge of the window",
                                                   {10001, 13001}),
                    TestCase::Duration::QUICK);
    }
};

static TcpSackPartialAckTestSuite g_tcpSackPartialAckTestSuite; //!< static var for test init
