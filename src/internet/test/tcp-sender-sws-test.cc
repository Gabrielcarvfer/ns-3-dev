/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "tcp-test-pair.h"

#include "ns3/boolean.h"
#include "ns3/config.h"
#include "ns3/inet-socket-address.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/log.h"
#include "ns3/node-container.h"
#include "ns3/packet.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/simulator.h"
#include "ns3/socket.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpSenderSwsTest");

/**
 * @ingroup internet-test
 *
 * @brief Bulk transfer into a small receive buffer, recording the data
 * segments of the sender and the right edge of the window they were sent into
 */
class TcpSenderSwsTestBase : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param desc The test description.
     * @param rcvBuf Receive buffer size of the receiver.
     * @param fillRcvWindow Value of the FillReceiveWindow attribute of the sender.
     */
    TcpSenderSwsTestBase(const std::string& desc, uint32_t rcvBuf, bool fillRcvWindow);

  protected:
    void DoRun() override;

    /**
     * @brief Check the outcome of the transfer.
     */
    virtual void Check() = 0;

    static constexpr uint32_t SEGMENT_SIZE = 500; //!< Segment size of the connection
    static constexpr uint32_t BYTES = 20000;      //!< Bytes to transfer

    uint32_t m_rcvBuf;           //!< Receive buffer size of the receiver
    bool m_fillRcvWindow;        //!< FillReceiveWindow attribute of the sender
    uint32_t m_received{0};      //!< Bytes the receiving application read
    uint32_t m_shortSegments{0}; //!< Data segments shorter than the MSS, but the last one
    uint32_t m_shortAtEdge{0};   //!< Short segments ending at the right edge of the window
    uint32_t m_rightEdge{0};     //!< Right edge of the window last offered to the sender
    Time m_lastRx;               //!< Time the receiving application read the last data

  private:
    /**
     * Trace sink of the segments the sender sends.
     * @param packet The payload.
     * @param header The TCP header.
     * @param socket The sending socket.
     */
    void SegmentSent(Ptr<const Packet> packet,
                     const TcpHeader& header,
                     Ptr<const TcpSocketBase> socket);

    /**
     * Trace sink of the segments the sender receives.
     * @param packet The payload.
     * @param header The TCP header.
     * @param socket The receiving socket.
     */
    void SegmentReceived(Ptr<const Packet> packet,
                         const TcpHeader& header,
                         Ptr<const TcpSocketBase> socket);

    /**
     * Accept callback: read everything as it arrives.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    /**
     * Receive callback of the receiver.
     * @param socket The receiving socket.
     */
    void Read(Ptr<Socket> socket);

    /**
     * Send the data.
     * @param socket The sending socket.
     */
    void SendData(Ptr<Socket> socket);

    uint32_t m_isn{0}; //!< Sequence number of the first data byte of the sender
};

TcpSenderSwsTestBase::TcpSenderSwsTestBase(const std::string& desc,
                                           uint32_t rcvBuf,
                                           bool fillRcvWindow)
    : TestCase(desc),
      m_rcvBuf(rcvBuf),
      m_fillRcvWindow(fillRcvWindow)
{
}

void
TcpSenderSwsTestBase::SegmentSent(Ptr<const Packet> packet,
                                  const TcpHeader& header,
                                  Ptr<const TcpSocketBase> socket)
{
    if (header.GetFlags() & TcpHeader::SYN)
    {
        // The first data byte follows the SYN
        m_isn = header.GetSequenceNumber().GetValue() + 1;
        return;
    }
    uint32_t size = packet->GetSize();
    uint32_t start = header.GetSequenceNumber().GetValue() - m_isn;
    if (size == 0 || size >= SEGMENT_SIZE || start + size > BYTES)
    {
        return;
    }
    if (start + size == BYTES)
    {
        // The tail of the data
        return;
    }
    m_shortSegments++;
    if (start + size == m_rightEdge)
    {
        m_shortAtEdge++;
    }
}

void
TcpSenderSwsTestBase::SegmentReceived(Ptr<const Packet> packet,
                                      const TcpHeader& header,
                                      Ptr<const TcpSocketBase> socket)
{
    if (header.GetFlags() & TcpHeader::ACK)
    {
        // The ACK of the SYN acknowledges the sequence number before the
        // first data byte
        m_rightEdge = header.GetAckNumber().GetValue() - m_isn + header.GetWindowSize();
    }
}

void
TcpSenderSwsTestBase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->SetRecvCallback(MakeCallback(&TcpSenderSwsTestBase::Read, this));
}

void
TcpSenderSwsTestBase::Read(Ptr<Socket> socket)
{
    while (Ptr<Packet> packet = socket->Recv())
    {
        m_received += packet->GetSize();
        m_lastRx = Simulator::Now();
    }
}

void
TcpSenderSwsTestBase::SendData(Ptr<Socket> socket)
{
    socket->Send(Create<Packet>(BYTES), 0);
}

void
TcpSenderSwsTestBase::DoRun()
{
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(SEGMENT_SIZE));
    // Without options the payload of a segment is a whole MSS
    Config::SetDefault("ns3::TcpSocketBase::Timestamp", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocketBase::WindowScaling", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(m_rcvBuf));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(BYTES));
    Config::SetDefault("ns3::TcpSocketBase::FillReceiveWindow", BooleanValue(m_fillRcvWindow));

    NodeContainer nodes;
    Ipv4InterfaceContainer interfaces = InstallTcpTestPair(nodes, "", MilliSeconds(10));

    const uint16_t port = 9906;
    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpSenderSwsTestBase::Accepted, this));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->TraceConnectWithoutContext("Tx",
                                       MakeCallback(&TcpSenderSwsTestBase::SegmentSent, this));
    client->TraceConnectWithoutContext("Rx",
                                       MakeCallback(&TcpSenderSwsTestBase::SegmentReceived, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));
    Simulator::Schedule(Seconds(1), &TcpSenderSwsTestBase::SendData, this, client);

    Simulator::Stop(Seconds(60));
    Simulator::Run();
    Check();
    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 *
 * @brief A receive window which never reaches one MSS still gets the data
 *
 * @RFC{9293}, Section 3.8.6.2.1: the sender sends a segment shorter than the
 * MSS when it carries at least half of the largest window the receiver has
 * offered, so the transfer completes although the receive buffer is smaller
 * than one segment.
 */
class TcpSenderSwsTinyWindowTestCase : public TcpSenderSwsTestBase
{
  public:
    TcpSenderSwsTinyWindowTestCase()
        : TcpSenderSwsTestBase("A window below one MSS still receives data", 300, false)
    {
    }

  private:
    void Check() override
    {
        NS_TEST_ASSERT_MSG_EQ(m_received, BYTES, "The transfer into a tiny window stalled");
    }
};

/**
 * @ingroup internet-test
 *
 * @brief The sender fills the receive window with short segments only with
 * FillReceiveWindow
 *
 * The receive window, 4.8 MSS, leaves a remainder that only a segment
 * shorter than the MSS can fill. With FillReceiveWindow disabled the sender
 * waits for a full segment (RFC 9293, Section 3.8.6.2.1); enabled, as Linux
 * (tcp_mss_split_point()), the bursts end with a short segment reaching the
 * right edge of the window, and the transfer is faster.
 */
class TcpFillReceiveWindowTestCase : public TestCase
{
  public:
    TcpFillReceiveWindowTestCase()
        : TestCase("FillReceiveWindow sends short segments up to the right edge of the window")
    {
    }

  private:
    /// The transfer with a given value of FillReceiveWindow
    class Run : public TcpSenderSwsTestBase
    {
      public:
        /**
         * @brief Constructor.
         * @param fill Value of the FillReceiveWindow attribute.
         */
        Run(bool fill)
            : TcpSenderSwsTestBase("", 2400, fill)
        {
        }

        void Check() override
        {
        }

        /**
         * Run the transfer.
         */
        void Go()
        {
            DoRun();
        }

        using TcpSenderSwsTestBase::m_lastRx;
        using TcpSenderSwsTestBase::m_received;
        using TcpSenderSwsTestBase::m_shortAtEdge;
        using TcpSenderSwsTestBase::m_shortSegments;
    };

    void DoRun() override
    {
        Run off(false);
        off.Go();
        Run on(true);
        on.Go();
        NS_TEST_ASSERT_MSG_EQ(off.m_received, 20000, "The transfer did not complete");
        NS_TEST_ASSERT_MSG_EQ(on.m_received, 20000, "The transfer did not complete");
        NS_TEST_ASSERT_MSG_EQ(off.m_shortSegments,
                              0,
                              "A short segment was sent with FillReceiveWindow disabled");
        NS_TEST_ASSERT_MSG_GT(on.m_shortSegments,
                              0,
                              "No short segment filled the window with FillReceiveWindow");
        NS_TEST_ASSERT_MSG_EQ(on.m_shortAtEdge,
                              on.m_shortSegments,
                              "A short segment did not end at the right edge of the window");
        NS_TEST_ASSERT_MSG_LT(on.m_lastRx,
                              off.m_lastRx,
                              "Filling the window did not speed up the transfer");
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Test suite of the sender silly window syndrome avoidance
 */
class TcpSenderSwsTestSuite : public TestSuite
{
  public:
    TcpSenderSwsTestSuite()
        : TestSuite("tcp-sender-sws", Type::UNIT)
    {
        AddTestCase(new TcpSenderSwsTinyWindowTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpFillReceiveWindowTestCase(), TestCase::Duration::QUICK);
    }
};

static TcpSenderSwsTestSuite g_tcpSenderSwsTestSuite; //!< Static variable for test initialization
