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
#include "ns3/tcp-socket-base.h"
#include "ns3/test.h"

#include <set>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpSackPartialAckTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief Check that SACK partial ACKs cause no spurious retransmission (see @issueid{988}).
 *
 * With SACK, a partial ACK retransmits only what the scoreboard marks lost
 * (NextSeg(), @RFC{6675}, Section 5), not the first unacknowledged segment
 * as in NewReno (@RFC{6582}). Expected: retransmissions == drops, no RTO.
 * Cases: two consecutive segments; a middle segment plus the last segment of
 * an application limited transfer. In the second case the last segment has
 * no SACKed data above it, so NextSeg() returns it under rule 3, which ns-3
 * relaxes when the scoreboard holds no SACK block, in place of the
 * unimplemented rescue retransmission (rule 4, @RFC{6675}, Section 4).
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
    void ConfigureProperties() override;
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

void
TcpSackPartialAckTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    // Room for the timestamp option on top of the expected payload
    SetSegmentSize(SENDER, GetSegSize(SENDER) + TcpSocketBase::GetSegmentOptionsSize());
}

Ptr<TcpSocketMsgBase>
TcpSackPartialAckTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("InitialCwnd", UintegerValue(10));
    socket->SetAttribute("Sack", BooleanValue(true));
    // Avoid retransmission timeouts during the test
    socket->SetAttribute("MinRto", TimeValue(Seconds(10)));
    return socket;
}

Ptr<ErrorModel>
TcpSackPartialAckTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    // 500 byte segments, first byte at sequence number 1
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
 * @brief Check a SACK recovery limited by the receive window.
 *
 * The receive buffer is smaller than the slow start cwnd, so no new data can
 * be sent during recovery (NextSeg() rule 2 fails, @RFC{6675}, Section 4)
 * until the cumulative ACK moves the window. Expected: no RTO and one
 * retransmission per lost segment.
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
    // Room for the timestamp option on top of the expected payload
    SetSegmentSize(SENDER, GetSegSize(SENDER) + TcpSocketBase::GetSegmentOptionsSize());
}

Ptr<TcpSocketMsgBase>
TcpRwndLimitedRecoveryTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("InitialCwnd", UintegerValue(10));
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

        // With the initial sequence numbers driven by the clock
        auto partialAck = new TcpSackPartialAckTest(
            "No spurious retransmission on SACK partial ACKs, clock driven ISN",
            {10001, 10501},
            60);
        partialAck->SetClockDrivenIsn(true, TcpGeneralTest::UPPER_HALF_ISN_STREAM);
        AddTestCase(partialAck, TestCase::Duration::QUICK);
        auto rwndLimited =
            new TcpRwndLimitedRecoveryTest("Receive window limited recovery, clock driven ISN",
                                           {10001});
        rwndLimited->SetClockDrivenIsn(true, TcpGeneralTest::UPPER_HALF_ISN_STREAM);
        AddTestCase(rwndLimited, TestCase::Duration::QUICK);
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
