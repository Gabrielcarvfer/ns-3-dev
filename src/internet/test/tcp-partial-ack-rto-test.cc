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
 * (@RFC{3782}, Section 4). The Impatient variant, which resets the timer only
 * upon the first partial acknowledgment (as @RFC{6582}, Section 3.2, step 3
 * prescribes for fast recovery), is applied to the recovery following a
 * retransmission timeout instead.
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
    }
};

static TcpPartialAckRtoTestSuite g_tcpPartialAckRtoTestSuite; //!< static var for test init
