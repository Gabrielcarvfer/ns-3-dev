/*
 * Copyright (c) 2015 Natale Patriciello <natale.patriciello@gmail.com>
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#include "tcp-error-model.h"
#include "tcp-general-test.h"

#include "ns3/log.h"
#include "ns3/node.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpZeroWindowTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief Testing the congestion avoidance increment on TCP ZeroWindow
 */
class TcpZeroWindowTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param desc Test description.
     */
    TcpZeroWindowTest(const std::string& desc);

  protected:
    // virtual void ReceivePacket (Ptr<Socket> socket);
    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override;

    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void ProcessedAck(const Ptr<const TcpSocketState> tcb,
                      const TcpHeader& h,
                      SocketWho who) override;
    void NormalClose(SocketWho who) override;
    void FinalChecks() override;

    void ConfigureEnvironment() override;
    void ConfigureProperties() override;

    /**
     * @brief Increase the receiver buffer size.
     */
    void IncreaseBufSize();

  protected:
    EventId m_receivePktEvent; //!< Receive packet event.
    bool m_zeroWindowProbe;    //!< ZeroWindow probe.
    bool m_windowUpdated;      //!< Window updated.
    bool m_senderFinished;     //!< Send finished.
    bool m_receiverFinished;   //!< Receiver finished.
};

TcpZeroWindowTest::TcpZeroWindowTest(const std::string& desc)
    : TcpGeneralTest(desc),
      m_zeroWindowProbe(false),
      m_windowUpdated(false),
      m_senderFinished(false),
      m_receiverFinished(false)
{
}

void
TcpZeroWindowTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktCount(20);
    SetMTU(500);
    SetTransmitStart(Seconds(2));
    SetPropagationDelay(MilliSeconds(50));
}

void
TcpZeroWindowTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetInitialCwnd(SENDER, 10);
}

void
TcpZeroWindowTest::IncreaseBufSize()
{
    SetRcvBufSize(RECEIVER, 2500);
}

Ptr<TcpSocketMsgBase>
TcpZeroWindowTest::CreateReceiverSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateReceiverSocket(node);

    socket->SetAttribute("RcvBufSize", UintegerValue(0));
    Simulator::Schedule(Seconds(10), &TcpZeroWindowTest::IncreaseBufSize, this);

    return socket;
}

void
TcpZeroWindowTest::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who == SENDER)
    {
        NS_LOG_INFO("\tSENDER TX " << h << " size " << p->GetSize());

        if (Simulator::Now().GetSeconds() <= 6.0)
        {
            NS_TEST_ASSERT_MSG_EQ(p->GetSize(), 0, "Data packet sent anyway");
        }
        else if (Simulator::Now().GetSeconds() > 6.0 && Simulator::Now().GetSeconds() <= 7.0)
        {
            NS_TEST_ASSERT_MSG_EQ(m_zeroWindowProbe, false, "Sent another probe");

            if (!m_zeroWindowProbe)
            {
                NS_TEST_ASSERT_MSG_EQ(p->GetSize(), 1, "Data packet sent instead of window probe");
                NS_TEST_ASSERT_MSG_EQ(h.GetSequenceNumber(),
                                      SequenceNumber32(1),
                                      "Data packet sent instead of window probe");
                m_zeroWindowProbe = true;
            }
        }
        else if (Simulator::Now().GetSeconds() > 7.0 && Simulator::Now().GetSeconds() < 10.0)
        {
            NS_FATAL_ERROR("No packets should be sent before the window update");
        }
    }
    else if (who == RECEIVER)
    {
        NS_LOG_INFO("\tRECEIVER TX " << h << " size " << p->GetSize());

        if (h.GetFlags() & TcpHeader::SYN)
        {
            NS_TEST_ASSERT_MSG_EQ(h.GetWindowSize(),
                                  0,
                                  "RECEIVER window size is not 0 in the SYN-ACK");
        }

        if (Simulator::Now().GetSeconds() > 6.0 && Simulator::Now().GetSeconds() <= 7.0)
        {
            NS_TEST_ASSERT_MSG_EQ(h.GetSequenceNumber(),
                                  SequenceNumber32(1),
                                  "Data packet sent instead of window probe");
            NS_TEST_ASSERT_MSG_EQ(h.GetWindowSize(), 0, "No zero window advertised by RECEIVER");
        }
        else if (Simulator::Now().GetSeconds() > 7.0 && Simulator::Now().GetSeconds() < 10.0)
        {
            NS_FATAL_ERROR("No packets should be sent before the window update");
        }
        else if (Simulator::Now().GetSeconds() >= 10.0)
        {
            NS_TEST_ASSERT_MSG_EQ(h.GetWindowSize(), 2500, "Receiver window not updated");
        }
    }

    NS_TEST_ASSERT_MSG_EQ(GetTcb(SENDER)->m_congState.Get(),
                          TcpSocketState::CA_OPEN,
                          "Sender State is not OPEN");
    NS_TEST_ASSERT_MSG_EQ(GetTcb(RECEIVER)->m_congState.Get(),
                          TcpSocketState::CA_OPEN,
                          "Receiver State is not OPEN");
}

void
TcpZeroWindowTest::Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who == SENDER)
    {
        NS_LOG_INFO("\tSENDER RX " << h << " size " << p->GetSize());

        if (h.GetFlags() & TcpHeader::SYN)
        {
            NS_TEST_ASSERT_MSG_EQ(h.GetWindowSize(),
                                  0,
                                  "RECEIVER window size is not 0 in the SYN-ACK");
        }

        if (Simulator::Now().GetSeconds() >= 10.0)
        {
            NS_TEST_ASSERT_MSG_EQ(h.GetWindowSize(), 2500, "Receiver window not updated");
            m_windowUpdated = true;
        }
    }
    else if (who == RECEIVER)
    {
        NS_LOG_INFO("\tRECEIVER RX " << h << " size " << p->GetSize());
    }
}

void
TcpZeroWindowTest::NormalClose(SocketWho who)
{
    if (who == SENDER)
    {
        m_senderFinished = true;
    }
    else if (who == RECEIVER)
    {
        m_receiverFinished = true;
    }
}

void
TcpZeroWindowTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_zeroWindowProbe, true, "Zero window probe not sent");
    NS_TEST_ASSERT_MSG_EQ(m_windowUpdated, true, "Window has not updated during the connection");
    NS_TEST_ASSERT_MSG_EQ(m_senderFinished, true, "Connection not closed successfully (SENDER)");
    NS_TEST_ASSERT_MSG_EQ(m_receiverFinished,
                          true,
                          "Connection not closed successfully (RECEIVER)");
}

void
TcpZeroWindowTest::ProcessedAck(const Ptr<const TcpSocketState> tcb,
                                const TcpHeader& h,
                                SocketWho who)
{
    if (who == SENDER)
    {
        if (h.GetFlags() & TcpHeader::SYN)
        {
            EventId persistentEvent = GetPersistentEvent(SENDER);
            NS_TEST_ASSERT_MSG_EQ(persistentEvent.IsPending(),
                                  true,
                                  "Persistent event not started");
        }
    }
    else if (who == RECEIVER)
    {
    }
}

/**
 * @ingroup internet-test
 *
 * @brief TCP ZeroWindow TestSuite
 */
class TcpZeroWindowTestSuite : public TestSuite
{
  public:
    TcpZeroWindowTestSuite()
        : TestSuite("tcp-zero-window-test", Type::UNIT)
    {
        AddTestCase(new TcpZeroWindowTest("zero window test"), TestCase::Duration::QUICK);
    }
};

static TcpZeroWindowTestSuite g_tcpZeroWindowTestSuite; //!< Static variable for test initialization
