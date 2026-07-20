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
 * The test drops two separated segments of the same window with SACK enabled
 * and checks that the number of retransmitted data segments is equal to the
 * number of dropped segments.
 */
class TcpSackPartialAckTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     */
    TcpSackPartialAckTest(const std::string& name)
        : TcpGeneralTest(name)
    {
    }

  protected:
    void ConfigureEnvironment() override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                        const TcpSocketState::TcpCongState_t newValue) override;
    void FinalChecks() override;

  private:
    static constexpr uint32_t DROPPED_SEGMENTS = 2; //!< Number of dropped segments.

    bool m_recoveryEntered{false};       //!< True if fast recovery was entered.
    uint32_t m_retransmissions{0};       //!< Number of retransmitted data segments.
    std::set<SequenceNumber32> m_seqTxs; //!< Sequence numbers of transmitted data segments.
};

void
TcpSackPartialAckTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(60);
    SetPropagationDelay(MilliSeconds(50));
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
    // Drop two consecutive segments of a window in the middle of the transfer
    // (segment size is 500 bytes and the first byte has sequence number 1)
    for (uint32_t seq : {10001, 10501})
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
TcpSackPartialAckTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_recoveryEntered, true, "Fast recovery was not entered");
    NS_TEST_ASSERT_MSG_EQ(m_retransmissions,
                          DROPPED_SEGMENTS,
                          "Number of retransmissions differs from the number of losses");
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
        AddTestCase(new TcpSackPartialAckTest("No spurious retransmission on SACK partial ACKs"),
                    TestCase::Duration::QUICK);
    }
};

static TcpSackPartialAckTestSuite g_tcpSackPartialAckTestSuite; //!< static var for test init
