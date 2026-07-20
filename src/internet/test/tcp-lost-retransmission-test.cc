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

NS_LOG_COMPONENT_DEFINE("TcpLostRetransmissionTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief Test the detection of lost retransmissions (see @issueid{979})
 *
 * When a retransmitted segment is lost again, @RFC{6675} based recovery would
 * stall until the retransmission timer expires, because retransmitted
 * segments are not selected again by NextSeg(). Like Linux, ns-3 detects a
 * lost retransmission when a segment transmitted after the retransmission is
 * SACKed, and marks it for retransmission again.
 *
 * The test drops a segment and its retransmission, and checks that the
 * transfer recovers without a retransmission timeout.
 */
class TcpLostRetransmissionTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     */
    TcpLostRetransmissionTest(const std::string& name)
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
TcpLostRetransmissionTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(60);
    SetPropagationDelay(MilliSeconds(50));
}

Ptr<TcpSocketMsgBase>
TcpLostRetransmissionTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("InitialCwnd", UintegerValue(10));
    // The dropped sequence numbers assume that the segment payload equals the
    // configured segment size: disable the timestamp option, which would
    // decrease the payload by its size
    socket->SetAttribute("Timestamp", BooleanValue(false));
    socket->SetAttribute("Sack", BooleanValue(true));
    return socket;
}

Ptr<ErrorModel>
TcpLostRetransmissionTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    // Drop a segment in the middle of the transfer and then its
    // retransmission (segment size is 500 bytes, first byte has sequence
    // number 1)
    errorModel->AddSeqToKill(SequenceNumber32(10001));
    errorModel->AddSeqToKill(SequenceNumber32(10001));
    return errorModel;
}

void
TcpLostRetransmissionTest::CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                                          const TcpSocketState::TcpCongState_t newValue)
{
    if (newValue == TcpSocketState::CA_RECOVERY)
    {
        m_recoveryEntered = true;
    }
}

void
TcpLostRetransmissionTest::AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who)
{
    if (who == SENDER)
    {
        m_rtoCount++;
    }
}

void
TcpLostRetransmissionTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_recoveryEntered, true, "Fast recovery was not entered");
    NS_TEST_ASSERT_MSG_EQ(m_rtoCount,
                          0,
                          "RTO expired: the lost retransmission was not detected and "
                          "retransmitted again during fast recovery");
}

/**
 * @ingroup internet-test
 *
 * @brief Lost retransmission detection TestSuite
 */
class TcpLostRetransmissionTestSuite : public TestSuite
{
  public:
    TcpLostRetransmissionTestSuite()
        : TestSuite("tcp-lost-retransmission", Type::UNIT)
    {
        AddTestCase(new TcpLostRetransmissionTest("Lost retransmission detected and recovered"),
                    TestCase::Duration::QUICK);
    }
};

static TcpLostRetransmissionTestSuite g_tcpLostRetransmissionTestSuite; //!< static var for init
