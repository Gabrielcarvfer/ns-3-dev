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

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpFlightSizeTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief Check that ssthresh on entering fast recovery is half the RFC 5681
 * FlightSize, bounded by cwnd, and that PRR starts from the matching pipe
 *
 * RFC 6675, Section 5, step (4.2): ssthresh = FlightSize / 2, where FlightSize
 * is the data sent and not cumulatively acknowledged (RFC 5681, Section 2),
 * SACKed and lost segments included; the RFC 6675 pipe excludes them. The
 * FlightSize is capped at cwnd, which it exceeds once segments were sent on
 * the duplicate ACKs.
 *
 * The pipe PRR uses on the first recovery ACK must be the FlightSize minus the
 * three SACKed (or dupack-inferred) segments and the lost head.
 */
class TcpFlightSizeTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param sack Whether SACK is enabled.
     * @param name Test description.
     */
    TcpFlightSizeTest(bool sack, const std::string& name)
        : TcpGeneralTest(name),
          m_sack(sack)
    {
    }

  protected:
    void ConfigureEnvironment() override;
    void ConfigureProperties() override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void HighestTxSeqTrace(SequenceNumber32 oldValue, SequenceNumber32 newValue) override;
    void CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                        const TcpSocketState::TcpCongState_t newValue) override;
    void SsThreshTrace(uint32_t oldValue, uint32_t newValue) override;
    void CWndTrace(uint32_t oldValue, uint32_t newValue) override;
    void BytesInFlightTrace(uint32_t oldValue, uint32_t newValue) override;
    void FinalChecks() override;

  private:
    bool m_sack;                    //!< Whether SACK is enabled
    SequenceNumber32 m_highTx{0};   //!< Highest sequence number sent
    SequenceNumber32 m_lastAck{0};  //!< Last ACK number received by the sender
    uint32_t m_expectedSsThresh{0}; //!< Half the FlightSize on entering recovery
    bool m_inRecovery{false};       //!< True once fast recovery was entered
    bool m_checked{false};          //!< True once the threshold was checked
    uint32_t m_cWnd{0};             //!< Congestion window of the sender
    uint32_t m_bytesInFlight{0};    //!< Last traced data in flight (pipe) of the sender
    uint32_t m_flightSize{0};       //!< FlightSize when fast recovery was entered
    bool m_pipeChecked{false};      //!< True once the pipe was checked
};

void
TcpFlightSizeTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(60);
    SetPropagationDelay(MilliSeconds(50));
}

void
TcpFlightSizeTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetSegmentSize(SENDER, GetSegSize(SENDER) + TcpSocketBase::GetSegmentOptionsSize());
    SetInitialCwnd(SENDER, 20);
}

Ptr<TcpSocketMsgBase>
TcpFlightSizeTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("Sack", BooleanValue(m_sack));
    return socket;
}

Ptr<ErrorModel>
TcpFlightSizeTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    // The fourth segment of the initial window
    errorModel->AddSeqToKill(SequenceNumber32(1501));
    return errorModel;
}

void
TcpFlightSizeTest::Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who == SENDER && (h.GetFlags() & TcpHeader::ACK))
    {
        m_lastAck = std::max(m_lastAck, h.GetAckNumber());
    }
}

void
TcpFlightSizeTest::HighestTxSeqTrace(SequenceNumber32 oldValue, SequenceNumber32 newValue)
{
    m_highTx = newValue;
}

void
TcpFlightSizeTest::CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                                  const TcpSocketState::TcpCongState_t newValue)
{
    if (newValue == TcpSocketState::CA_RECOVERY && !m_inRecovery)
    {
        m_inRecovery = true;
        uint32_t flightSize = m_highTx - m_lastAck;
        m_flightSize = flightSize;
        NS_TEST_ASSERT_MSG_GT(flightSize,
                              m_cWnd - 2 * GetSegSize(SENDER),
                              "The test needs a FlightSize close to cwnd");
        m_expectedSsThresh =
            std::max<uint32_t>(std::min(flightSize, m_cWnd) / 2, 2 * GetSegSize(SENDER));
    }
}

void
TcpFlightSizeTest::SsThreshTrace(uint32_t oldValue, uint32_t newValue)
{
    if (m_inRecovery && !m_checked)
    {
        m_checked = true;
        NS_TEST_ASSERT_MSG_EQ(newValue,
                              m_expectedSsThresh,
                              "ssthresh must be half the FlightSize of RFC 5681");
    }
}

void
TcpFlightSizeTest::CWndTrace(uint32_t oldValue, uint32_t newValue)
{
    m_cWnd = newValue;
    if (m_checked && !m_pipeChecked)
    {
        // The window set by the recovery on its first ACK
        m_pipeChecked = true;
        NS_TEST_ASSERT_MSG_EQ(m_bytesInFlight,
                              m_flightSize - 4 * GetSegSize(SENDER),
                              "The recovery must start from the pipe of the current ACK");
    }
}

void
TcpFlightSizeTest::BytesInFlightTrace(uint32_t oldValue, uint32_t newValue)
{
    m_bytesInFlight = newValue;
}

void
TcpFlightSizeTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_checked, true, "Fast recovery was not entered");
    NS_TEST_ASSERT_MSG_EQ(m_pipeChecked, true, "The recovery did not set the window");
}

/**
 * @ingroup internet-test
 *
 * @brief TestSuite for the FlightSize used on entering fast recovery
 */
class TcpFlightSizeTestSuite : public TestSuite
{
  public:
    TcpFlightSizeTestSuite()
        : TestSuite("tcp-flight-size", Type::UNIT)
    {
        AddTestCase(new TcpFlightSizeTest(true, "ssthresh from the FlightSize, with SACK"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpFlightSizeTest(false, "ssthresh from the FlightSize, without SACK"),
                    TestCase::Duration::QUICK);
    }
};

static TcpFlightSizeTestSuite g_tcpFlightSizeTestSuite; //!< Static variable for test initialization
