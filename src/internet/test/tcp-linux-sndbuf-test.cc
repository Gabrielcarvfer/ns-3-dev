/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "tcp-test-pair.h"

#include "ns3/boolean.h"
#include "ns3/config.h"
#include "ns3/data-rate.h"
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
#include "ns3/type-id.h"
#include "ns3/uinteger.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpLinuxSndBufTest");

/**
 * @ingroup internet-test
 *
 * @brief Test the send buffer accounted as Linux (SndSkbOverhead)
 *
 * The application writes 30000 bytes at once into a send buffer of 20000
 * bytes, with segments of 500 bytes and a receive window of 2500 bytes, so
 * that Linux fills skbs of 1000 bytes (half the window). As Linux
 * (tcp_sendmsg_locked()), the window lets 2500 bytes go during the write:
 * the third skb, which sends 500 bytes, is filled again up to 1000 bytes not
 * sent, and each skb costs 768 bytes beyond its data. A new skb is taken
 * while the cost is below the buffer size: the write queues 12500 bytes. The
 * application may write again only once the free buffer is at least half the
 * cost, after the transmissions of the ACK that frees the space
 * (tcp_check_space()), and the transfer completes.
 */
class TcpLinuxSndBufTestCase : public TestCase
{
  public:
    TcpLinuxSndBufTestCase()
        : TestCase("Writes queued and woken as Linux")
    {
    }

  private:
    void DoRun() override;

    /**
     * Write the data not yet queued, as far as the socket takes it.
     * @param socket The sending socket.
     * @param available The space the socket reports.
     */
    void Write(Ptr<Socket> socket, uint32_t available);

    static constexpr uint32_t BYTES = 30000; //!< Bytes to transfer

    uint32_t m_received{0};            //!< Bytes the receiving application read
    uint32_t m_written{0};             //!< Bytes the sending application wrote
    std::vector<uint32_t> m_writes;    //!< Bytes queued by each successful write
    std::vector<uint32_t> m_refusedAt; //!< Bytes written when a write was refused
    Time m_lastTx{Seconds(-1)};        //!< Time of the last data segment sent
    uint32_t m_wakes{0};               //!< Writes after a refused one
    uint32_t m_wakesAfterTx{0};        //!< Those once the ACK sent its data

    /**
     * Trace sink of the segments the sender sends.
     * @param packet The payload.
     * @param header The TCP header.
     * @param socket The sending socket.
     */
    void SegmentSent(Ptr<const Packet> packet,
                     const TcpHeader& header,
                     Ptr<const TcpSocketBase> socket);
};

void
TcpLinuxSndBufTestCase::SegmentSent(Ptr<const Packet> packet,
                                    const TcpHeader& header,
                                    Ptr<const TcpSocketBase> socket)
{
    if (packet->GetSize() > 0)
    {
        m_lastTx = Simulator::Now();
    }
}

void
TcpLinuxSndBufTestCase::Write(Ptr<Socket> socket, uint32_t available)
{
    if (!m_refusedAt.empty() && m_written < BYTES)
    {
        ++m_wakes;
        m_wakesAfterTx += m_lastTx == Simulator::Now() ? 1 : 0;
    }
    while (m_written < BYTES)
    {
        int sent = socket->Send(Create<Packet>(BYTES - m_written), 0);
        if (sent <= 0)
        {
            m_refusedAt.push_back(m_written);
            return;
        }
        m_writes.push_back(sent);
        m_written += sent;
    }
}

void
TcpLinuxSndBufTestCase::DoRun()
{
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(500));
    Config::SetDefault("ns3::TcpSocketBase::Timestamp", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocketBase::WindowScaling", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(100));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(2500));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(20000));
    Config::SetDefault("ns3::TcpSocketBase::ForcedPush", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocketBase::SndSkbOverhead", UintegerValue(768));

    NodeContainer nodes;
    Ipv4InterfaceContainer interfaces = InstallTcpTestPair(nodes, "", MilliSeconds(10));

    const uint16_t port = 9910;
    Ptr<Socket> server = InstallTcpByteSink(nodes.Get(1), port, m_received);

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));
    Simulator::Schedule(Seconds(1), &TcpLinuxSndBufTestCase::Write, this, client, 0);
    client->SetSendCallback(MakeCallback(&TcpLinuxSndBufTestCase::Write, this));
    client->TraceConnectWithoutContext("Tx",
                                       MakeCallback(&TcpLinuxSndBufTestCase::SegmentSent, this));

    Simulator::Stop(Seconds(60));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_received, BYTES, "The transfer did not complete");
    NS_TEST_ASSERT_MSG_GT(m_writes.size(), 1, "The first write took all the data");
    NS_TEST_ASSERT_MSG_EQ(m_writes[0], 12500, "Unexpected data queued by the first write");
    NS_TEST_ASSERT_MSG_GT(m_refusedAt.size(), 0, "No write was refused");
    NS_TEST_ASSERT_MSG_EQ(m_refusedAt[0], 12500, "The write after the first one was not refused");
    NS_TEST_ASSERT_MSG_GT(m_wakes, 0, "The application was never woken");
    NS_TEST_ASSERT_MSG_EQ(m_wakesAfterTx,
                          m_wakes,
                          "The application was woken before the ACK sent its data");
    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 *
 * @brief Test the growth of the window limited as Linux (LinuxCwndLimited)
 *
 * A bulk transfer with NewReno in slow start into a receive window of 8
 * segments: the receive window, not the congestion window, limits the
 * transmissions. Without LinuxCwndLimited the congestion window grows with
 * every ACK; with it, as Linux (tcp_is_cwnd_limited()), it stops at twice the
 * largest flight.
 */
class TcpLinuxCwndLimitedTestCase : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param limited Value of the LinuxCwndLimited attribute of the sender.
     * @param desc The test description.
     */
    TcpLinuxCwndLimitedTestCase(bool limited, const std::string& desc)
        : TestCase(desc),
          m_limited(limited)
    {
    }

  private:
    void DoRun() override;

    /**
     * Trace sink of the congestion window.
     * @param oldValue The previous window.
     * @param newValue The new window.
     */
    void CwndChange(uint32_t oldValue, uint32_t newValue);

    /**
     * Write the data.
     * @param socket The sending socket.
     */
    void Write(Ptr<Socket> socket);

    static constexpr uint32_t BYTES = 200000;      //!< Bytes to transfer
    static constexpr uint32_t SEGMENT_SIZE = 1000; //!< Segment size

    bool m_limited;         //!< LinuxCwndLimited attribute of the sender
    uint32_t m_received{0}; //!< Bytes the receiving application read
    uint32_t m_maxCwnd{0};  //!< Largest congestion window
};

void
TcpLinuxCwndLimitedTestCase::CwndChange(uint32_t oldValue, uint32_t newValue)
{
    m_maxCwnd = std::max(m_maxCwnd, newValue);
}

void
TcpLinuxCwndLimitedTestCase::Write(Ptr<Socket> socket)
{
    socket->Send(Create<Packet>(BYTES), 0);
}

void
TcpLinuxCwndLimitedTestCase::DoRun()
{
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(SEGMENT_SIZE));
    Config::SetDefault("ns3::TcpSocketBase::Timestamp", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocketBase::WindowScaling", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(2));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(8 * SEGMENT_SIZE));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(BYTES));
    Config::SetDefault("ns3::TcpSocketBase::LinuxCwndLimited", BooleanValue(m_limited));
    Config::SetDefault("ns3::TcpL4Protocol::SocketType",
                       TypeIdValue(TypeId::LookupByName("ns3::TcpNewReno")));

    NodeContainer nodes;
    Ipv4InterfaceContainer interfaces = InstallTcpTestPair(nodes, "100Mbps", MilliSeconds(10));

    const uint16_t port = 9911;
    Ptr<Socket> server = InstallTcpByteSink(nodes.Get(1), port, m_received);

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->TraceConnectWithoutContext(
        "CongestionWindow",
        MakeCallback(&TcpLinuxCwndLimitedTestCase::CwndChange, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));
    Simulator::Schedule(Seconds(1), &TcpLinuxCwndLimitedTestCase::Write, this, client);

    Simulator::Stop(Seconds(60));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_received, BYTES, "The transfer did not complete");
    if (m_limited)
    {
        NS_TEST_ASSERT_MSG_LT_OR_EQ(m_maxCwnd,
                                    16 * SEGMENT_SIZE,
                                    "The window grew past twice the largest flight");
    }
    else
    {
        NS_TEST_ASSERT_MSG_GT(m_maxCwnd, 50 * SEGMENT_SIZE, "The window did not keep growing");
    }
    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 *
 * @brief Test suite of the send buffer and window growth limited as Linux
 */
class TcpLinuxSndBufTestSuite : public TestSuite
{
  public:
    TcpLinuxSndBufTestSuite()
        : TestSuite("tcp-linux-sndbuf", Type::UNIT)
    {
        AddTestCase(new TcpLinuxSndBufTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpLinuxCwndLimitedTestCase(false, "The window grows when not limiting"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpLinuxCwndLimitedTestCase(true, "The window grows only when limiting"),
                    TestCase::Duration::QUICK);
    }
};

static TcpLinuxSndBufTestSuite
    g_tcpLinuxSndBufTestSuite; //!< Static variable for test initialization
