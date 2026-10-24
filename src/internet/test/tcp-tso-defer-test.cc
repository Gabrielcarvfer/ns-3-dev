/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ns3/boolean.h"
#include "ns3/config.h"
#include "ns3/data-rate.h"
#include "ns3/error-model.h"
#include "ns3/inet-socket-address.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-header.h"
#include "ns3/log.h"
#include "ns3/node-container.h"
#include "ns3/packet.h"
#include "ns3/pointer.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/simulator.h"
#include "ns3/socket.h"
#include "ns3/string.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"

#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpTsoDeferTest");

/**
 * @ingroup internet-test
 *
 * @brief Test the deferral of partial bursts, as Linux TSO (TsoDefer)
 *
 * A bulk transfer in congestion avoidance, with a receiver acknowledging
 * every segment over a 10 Mbit/s link, so that each ACK lets the sender send
 * a single segment, less than the two segments of a Linux GSO packet at this
 * rate. Without TsoDefer the sender sends each one at once; with it, as Linux
 * (tcp_tso_should_defer()), it waits for the next ACK and sends two segments
 * together, unless its last transmission is more than 1 ms old: more than
 * 90% of the transmissions are then single segments without it, less than
 * half with it. The transfer completes in both cases.
 */
class TcpTsoDeferTestCase : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param defer Value of the TsoDefer attribute of the sender.
     * @param desc The test description.
     */
    TcpTsoDeferTestCase(bool defer, const std::string& desc)
        : TestCase(desc),
          m_defer(defer)
    {
    }

  private:
    void DoRun() override;

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
     * Write the data.
     * @param socket The sending socket.
     */
    void Write(Ptr<Socket> socket);

    static constexpr uint32_t BYTES = 200000; //!< Bytes to transfer

    bool m_defer;           //!< TsoDefer attribute of the sender
    uint32_t m_received{0}; //!< Bytes the receiving application read
    Time m_lastTx;          //!< Time of the last data segment sent
    uint32_t m_burst{0};    //!< Segments of the current burst
    uint32_t m_singles{0};  //!< Bursts of a single segment, after the start
    uint32_t m_bursts{0};   //!< Bursts, after the start
    uint32_t m_seen{0};     //!< Bursts seen, from the start
};

void
TcpTsoDeferTestCase::SegmentSent(Ptr<const Packet> packet,
                                 const TcpHeader& header,
                                 Ptr<const TcpSocketBase> socket)
{
    if (packet->GetSize() == 0)
    {
        return;
    }
    // The segments of a burst are handed to the device at the same time
    if (Simulator::Now() != m_lastTx)
    {
        if (m_burst > 0 && ++m_seen > 50)
        {
            ++m_bursts;
            m_singles += m_burst == 1 ? 1 : 0;
        }
        m_burst = 0;
    }
    ++m_burst;
    m_lastTx = Simulator::Now();
}

void
TcpTsoDeferTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->SetRecvCallback(MakeCallback(&TcpTsoDeferTestCase::Read, this));
}

void
TcpTsoDeferTestCase::Read(Ptr<Socket> socket)
{
    while (Ptr<Packet> packet = socket->Recv())
    {
        m_received += packet->GetSize();
    }
}

void
TcpTsoDeferTestCase::Write(Ptr<Socket> socket)
{
    socket->Send(Create<Packet>(BYTES), 0);
}

void
TcpTsoDeferTestCase::DoRun()
{
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(500));
    Config::SetDefault("ns3::TcpSocketBase::Timestamp", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(10));
    Config::SetDefault("ns3::TcpSocket::InitialSlowStartThreshold", UintegerValue(5000));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(1));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(BYTES));
    Config::SetDefault("ns3::TcpSocketBase::TsoDefer", BooleanValue(m_defer));

    NodeContainer nodes;
    nodes.Create(2);
    SimpleNetDeviceHelper devHelper;
    devHelper.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Mbps")));
    devHelper.SetChannelAttribute("Delay", TimeValue(MilliSeconds(10)));
    NetDeviceContainer devices = devHelper.Install(nodes);
    InternetStackHelper stack;
    stack.Install(nodes);
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t port = 9908;
    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpTsoDeferTestCase::Accepted, this));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->TraceConnectWithoutContext("Tx", MakeCallback(&TcpTsoDeferTestCase::SegmentSent, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));
    Simulator::Schedule(Seconds(1), &TcpTsoDeferTestCase::Write, this, client);

    Simulator::Stop(Seconds(60));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_received, BYTES, "The transfer did not complete");
    NS_TEST_ASSERT_MSG_GT(m_bursts, 20, "Too few bursts");
    if (m_defer)
    {
        // Single segments remain when the last transmission is over 1 ms old
        NS_TEST_ASSERT_MSG_LT(m_singles * 2, m_bursts, "Single segments not deferred");
    }
    else
    {
        NS_TEST_ASSERT_MSG_GT(m_singles * 10, m_bursts * 9, "Single segments not sent at once");
    }
    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 *
 * @brief Test that TsoDefer does not hold the segment filling the window
 *
 * A bulk transfer into the receive window of 4.9 segments that a Linux
 * receiver advertises for a buffer of 8 KiB, over a 10 Mbit/s link with an
 * RTT of 100 ms, with FillReceiveWindow. As Linux (tcp_mss_split_point()),
 * the short segment that fills the window up to its right edge belongs to the
 * same GSO packet as the full segment before it, so TsoDefer does not defer it
 * on its own: with a receiver acknowledging every segment, each short segment
 * is sent together with a full one. A GSO packet does not start with it,
 * though, as its first segment would not fit the window (tcp_snd_wnd_test()):
 * with a receiver acknowledging every second segment, a short segment is
 * sent only along full ones, never alone.
 */
class TcpTsoDeferFillTestCase : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param delAckCount Segments the receiver acknowledges at once.
     * @param desc The test description.
     */
    TcpTsoDeferFillTestCase(uint32_t delAckCount, const std::string& desc)
        : TestCase(desc),
          m_delAckCount(delAckCount)
    {
    }

  private:
    void DoRun() override;

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
     * Write the data.
     * @param socket The sending socket.
     */
    void Write(Ptr<Socket> socket);

    static constexpr uint32_t BYTES = 50000;       //!< Bytes to transfer
    static constexpr uint32_t SEGMENT_SIZE = 1448; //!< Segment size

    uint32_t m_delAckCount;     //!< Segments the receiver acknowledges at once
    uint32_t m_received{0};     //!< Bytes the receiving application read
    uint32_t m_sent{0};         //!< Data bytes sent for the first time
    Time m_lastTx{Seconds(-1)}; //!< Time of the last data segment sent
    uint32_t m_burst{0};        //!< Segments of the current burst
    uint32_t m_short{0};        //!< Short segments sent, but the last one
    uint32_t m_shortAlone{0};   //!< Short segments sent alone
};

void
TcpTsoDeferFillTestCase::SegmentSent(Ptr<const Packet> packet,
                                     const TcpHeader& header,
                                     Ptr<const TcpSocketBase> socket)
{
    if (packet->GetSize() == 0)
    {
        return;
    }
    if (Simulator::Now() != m_lastTx)
    {
        m_burst = 0;
    }
    ++m_burst;
    m_lastTx = Simulator::Now();
    m_sent += packet->GetSize();
    if (packet->GetSize() < SEGMENT_SIZE && m_sent < BYTES)
    {
        ++m_short;
        m_shortAlone += m_burst == 1 ? 1 : 0;
    }
}

void
TcpTsoDeferFillTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->SetRecvCallback(MakeCallback(&TcpTsoDeferFillTestCase::Read, this));
}

void
TcpTsoDeferFillTestCase::Read(Ptr<Socket> socket)
{
    while (Ptr<Packet> packet = socket->Recv())
    {
        m_received += packet->GetSize();
    }
}

void
TcpTsoDeferFillTestCase::Write(Ptr<Socket> socket)
{
    socket->Send(Create<Packet>(BYTES), 0);
}

void
TcpTsoDeferFillTestCase::DoRun()
{
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(SEGMENT_SIZE));
    Config::SetDefault("ns3::TcpSocketBase::Timestamp", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(8192));
    // The receiver advertises the window of Linux, 7068 bytes, not a
    // multiple of the segment size
    Config::SetDefault("ns3::TcpSocketBase::LinuxRcvWindow", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(BYTES));
    Config::SetDefault("ns3::TcpSocketBase::FillReceiveWindow", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocketBase::TsoDefer", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(m_delAckCount));

    NodeContainer nodes;
    nodes.Create(2);
    SimpleNetDeviceHelper devHelper;
    devHelper.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Mbps")));
    devHelper.SetChannelAttribute("Delay", TimeValue(MilliSeconds(50)));
    NetDeviceContainer devices = devHelper.Install(nodes);
    InternetStackHelper stack;
    stack.Install(nodes);
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t port = 9909;
    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpTsoDeferFillTestCase::Accepted, this));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->TraceConnectWithoutContext("Tx",
                                       MakeCallback(&TcpTsoDeferFillTestCase::SegmentSent, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));
    Simulator::Schedule(Seconds(1), &TcpTsoDeferFillTestCase::Write, this, client);

    Simulator::Stop(Seconds(60));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_received, BYTES, "The transfer did not complete");
    if (m_delAckCount == 1)
    {
        NS_TEST_ASSERT_MSG_GT(m_short, 3, "Too few short segments filling the window");
    }
    NS_TEST_ASSERT_MSG_EQ(m_shortAlone, 0, "A short segment was deferred to a burst of its own");
    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 *
 * @brief Test the GSO packets that the right edge of the receive window cuts
 *
 * A transfer of 50000 bytes into the window of 7068 bytes that a Linux
 * receiver advertises for a buffer of 8 KiB, over a 10 Mbit/s link with an
 * RTT of 100 ms, as the R22 scenario compared with Linux 6.12. As
 * tcp_mss_split_point(), a GSO packet takes the segments the congestion window
 * allows, cut by the end of its skb and by the right edge of the receive
 * window, the short segment filling the window included: the short segment
 * goes with the full one before it whenever the window leaves room for less
 * than the GSO packet, and Linux sends its first short segments at 8688 and
 * 15756 bytes from the first byte.
 */
class TcpTsoDeferSplitTestCase : public TestCase
{
  public:
    TcpTsoDeferSplitTestCase()
        : TestCase("The receive window cuts the GSO packets as Linux")
    {
    }

  private:
    void DoRun() override;

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
     * Write the data.
     * @param socket The sending socket.
     */
    void Write(Ptr<Socket> socket);

    static constexpr uint32_t BYTES = 50000; //!< Bytes to transfer

    uint32_t m_received{0};         //!< Bytes the receiving application read
    uint32_t m_first{0};            //!< Sequence number of the first data byte
    uint32_t m_highest{0};          //!< Highest end of the data sent
    std::vector<uint32_t> m_shorts; //!< Starts of the short segments sent
};

void
TcpTsoDeferSplitTestCase::SegmentSent(Ptr<const Packet> packet,
                                      const TcpHeader& header,
                                      Ptr<const TcpSocketBase> socket)
{
    if (header.GetFlags() & TcpHeader::SYN)
    {
        m_first = header.GetSequenceNumber().GetValue() + 1;
        return;
    }
    uint32_t start = header.GetSequenceNumber().GetValue() - m_first;
    if (packet->GetSize() == 0 || start + packet->GetSize() <= m_highest)
    {
        return;
    }
    m_highest = start + packet->GetSize();
    if (packet->GetSize() < 1448 && m_highest < BYTES)
    {
        m_shorts.push_back(start);
    }
}

void
TcpTsoDeferSplitTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->SetRecvCallback(MakeCallback(&TcpTsoDeferSplitTestCase::Read, this));
}

void
TcpTsoDeferSplitTestCase::Read(Ptr<Socket> socket)
{
    while (Ptr<Packet> packet = socket->Recv())
    {
        m_received += packet->GetSize();
    }
}

void
TcpTsoDeferSplitTestCase::Write(Ptr<Socket> socket)
{
    socket->Send(Create<Packet>(BYTES), 0);
}

void
TcpTsoDeferSplitTestCase::DoRun()
{
    // An MSS of 1460 bytes, of which the timestamps take 12
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1460));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(8192));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(BYTES));
    Config::SetDefault("ns3::TcpSocketBase::LinuxRcvWindow", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocketBase::QuickAck", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocketBase::SackCompression", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocketBase::AdaptiveDelAck", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocketBase::FillReceiveWindow", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocketBase::ForcedPush", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocketBase::TsoDefer", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocket::TcpNoDelay", BooleanValue(false));

    NodeContainer nodes;
    nodes.Create(2);
    SimpleNetDeviceHelper devHelper;
    devHelper.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Mbps")));
    devHelper.SetChannelAttribute("Delay", TimeValue(MilliSeconds(50)));
    NetDeviceContainer devices = devHelper.Install(nodes);
    InternetStackHelper stack;
    stack.Install(nodes);
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t port = 9912;
    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpTsoDeferSplitTestCase::Accepted, this));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->TraceConnectWithoutContext("Tx",
                                       MakeCallback(&TcpTsoDeferSplitTestCase::SegmentSent, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));
    Simulator::Schedule(Seconds(1), &TcpTsoDeferSplitTestCase::Write, this, client);

    Simulator::Stop(Seconds(60));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_received, BYTES, "The transfer did not complete");
    NS_TEST_ASSERT_MSG_GT_OR_EQ(m_shorts.size(), 2, "Too few short segments");
    const std::vector<uint32_t> expected{8688, 15756};
    for (size_t i = 0; i < std::min<size_t>(m_shorts.size(), expected.size()); ++i)
    {
        NS_TEST_ASSERT_MSG_EQ(m_shorts[i], expected[i], "Short segment " << i << " elsewhere");
    }
    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 *
 * @brief Test Nagle's algorithm as Linux (TsoDefer with TcpNoDelay false)
 *
 * The application writes 2500 bytes, with segments of 1000 bytes and Nagle's
 * algorithm, then 300 bytes 1 ms later. Without TsoDefer, Nagle holds the
 * short segment of 500 bytes while data is not acknowledged; with it, as
 * Linux (tcp_nagle_check(), Minshall's variant), it holds a short segment
 * only while a short segment sent is not acknowledged: the 500 bytes go with
 * the full segments, and the 300 bytes wait for their ACK.
 */
class TcpTsoDeferNagleTestCase : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param defer Value of the TsoDefer attribute of the sender.
     * @param desc The test description.
     */
    TcpTsoDeferNagleTestCase(bool defer, const std::string& desc)
        : TestCase(desc),
          m_defer(defer)
    {
    }

  private:
    void DoRun() override;

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
     * Write data.
     * @param socket The sending socket.
     * @param size Bytes to write.
     */
    void Write(Ptr<Socket> socket, uint32_t size);

    bool m_defer;        //!< TsoDefer attribute of the sender
    Time m_halfSent;     //!< Time the last 500 bytes of the first write were sent
    Time m_lastSent;     //!< Time the data of the second write was sent
    uint32_t m_first{0}; //!< Sequence number of the first data byte
};

void
TcpTsoDeferNagleTestCase::SegmentSent(Ptr<const Packet> packet,
                                      const TcpHeader& header,
                                      Ptr<const TcpSocketBase> socket)
{
    if (header.GetFlags() & TcpHeader::SYN)
    {
        m_first = header.GetSequenceNumber().GetValue() + 1;
        return;
    }
    uint32_t start = header.GetSequenceNumber().GetValue() - m_first;
    if (start == 2000 && m_halfSent.IsZero())
    {
        m_halfSent = Simulator::Now();
    }
    if (start + packet->GetSize() > 2500 && m_lastSent.IsZero())
    {
        m_lastSent = Simulator::Now();
    }
}

void
TcpTsoDeferNagleTestCase::Write(Ptr<Socket> socket, uint32_t size)
{
    socket->Send(Create<Packet>(size), 0);
}

void
TcpTsoDeferNagleTestCase::DoRun()
{
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1000));
    Config::SetDefault("ns3::TcpSocketBase::Timestamp", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocket::TcpNoDelay", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocketBase::TsoDefer", BooleanValue(m_defer));

    NodeContainer nodes;
    nodes.Create(2);
    SimpleNetDeviceHelper devHelper;
    devHelper.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Mbps")));
    devHelper.SetChannelAttribute("Delay", TimeValue(MilliSeconds(10)));
    NetDeviceContainer devices = devHelper.Install(nodes);
    InternetStackHelper stack;
    stack.Install(nodes);
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t port = 9913;
    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->TraceConnectWithoutContext("Tx",
                                       MakeCallback(&TcpTsoDeferNagleTestCase::SegmentSent, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));
    Simulator::Schedule(Seconds(1), &TcpTsoDeferNagleTestCase::Write, this, client, 2500);
    Simulator::Schedule(Seconds(1.001), &TcpTsoDeferNagleTestCase::Write, this, client, 300);

    Simulator::Stop(Seconds(5));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_lastSent.IsZero(), false, "The last segment was not sent");
    NS_TEST_ASSERT_MSG_GT(m_lastSent, Seconds(1.02), "The last segment did not wait for an ACK");
    if (m_defer)
    {
        NS_TEST_ASSERT_MSG_LT(m_halfSent, Seconds(1.001), "The short segment was held");
    }
    else
    {
        NS_TEST_ASSERT_MSG_GT(m_halfSent, Seconds(1.02), "The short segment was not held");
    }
    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 *
 * @brief A socket exposing the GSO size of Linux for the given state
 */
class TcpTsoSegmentsProbe : public TcpSocketBase
{
  public:
    /**
     * @brief Get the GSO size for a state.
     * @param cwnd Congestion window, in segments.
     * @param ssThresh Slow start threshold, in segments.
     * @param srtt Smoothed RTT.
     * @return the number of segments of a GSO packet.
     */
    uint32_t Probe(uint32_t cwnd, uint32_t ssThresh, Time srtt)
    {
        const uint32_t mss = 1448;
        m_tcb->m_segmentSize = mss;
        m_tcb->m_cWnd = cwnd * mss;
        m_tcb->m_ssThresh = ssThresh * mss;
        m_tcb->m_srtt = srtt;
        m_tcb->m_bytesInFlight = 0;
        SetCongestionControlAlgorithm(CreateObject<TcpNewReno>());
        return GetLinuxTsoSegments();
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Test the pacing rate behind the GSO size of Linux
 *
 * As tcp_update_pacing_rate(), the rate is 200% of cwnd * mss / srtt while
 * the window is below half the slow start threshold, both counted in
 * segments, and 120% above: with 120 segments and a threshold of 241, the
 * window is not below 241 / 2 = 120 segments, so that the rate of 120%, with
 * an SRTT of 69.7 ms, is 2.99 MB/s, and a GSO packet of 1 ms holds 2
 * segments; compared in bytes, the window would be below half the threshold,
 * and the packet would hold 3.
 */
class TcpTsoSegmentsTestCase : public TestCase
{
  public:
    TcpTsoSegmentsTestCase()
        : TestCase("The GSO size follows the pacing ratio of Linux")
    {
    }

  private:
    void DoRun() override
    {
        Ptr<TcpTsoSegmentsProbe> probe = CreateObject<TcpTsoSegmentsProbe>();
        NS_TEST_ASSERT_MSG_EQ(probe->Probe(120, 241, MicroSeconds(69700)),
                              2,
                              "Pacing ratio not taken in segments");
        NS_TEST_ASSERT_MSG_EQ(probe->Probe(119, 241, MicroSeconds(69700)),
                              3,
                              "Pacing ratio of slow start not taken");
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Drop the first TCP data segment starting at a given offset from the
 * first data byte, leaving the other packets untouched
 */
class TcpOffsetErrorModel : public ErrorModel
{
  public:
    /**
     * @brief Set the offset of the segment to drop.
     * @param offset Bytes from the first data byte.
     */
    void SetOffset(uint32_t offset)
    {
        m_offset = offset;
    }

  private:
    bool DoCorrupt(Ptr<Packet> p) override
    {
        Ptr<Packet> copy = p->Copy();
        Ipv4Header ip;
        if (m_done || copy->PeekHeader(ip) == 0 || ip.GetProtocol() != 6)
        {
            return false;
        }
        copy->RemoveHeader(ip);
        TcpHeader tcp;
        copy->RemoveHeader(tcp);
        if (tcp.GetFlags() & TcpHeader::SYN)
        {
            m_first = tcp.GetSequenceNumber().GetValue() + 1;
            return false;
        }
        if (copy->GetSize() > 0 && tcp.GetSequenceNumber().GetValue() - m_first == m_offset)
        {
            m_done = true;
            return true;
        }
        return false;
    }

    void DoReset() override
    {
    }

    uint32_t m_offset{0}; //!< Offset of the segment to drop
    uint32_t m_first{0};  //!< Sequence number of the first data byte
    bool m_done{false};   //!< The segment was dropped
};

/**
 * @ingroup internet-test
 *
 * @brief Test Nagle's algorithm of Linux on the segment filling the window
 *
 * Into the window of 7240 bytes that a Linux receiver advertises in its SYN
 * for a buffer of 8 KiB, the application writes 1948 bytes, a full segment
 * and a short one of 500 bytes, then the rest of the data. The GSO packets of
 * two segments leave room for a full segment and a short one of 948 bytes up
 * to the right edge of the window. As Linux (tcp_mss_split_point() and
 * tcp_nagle_check()), with Nagle's algorithm, the GSO packet that the window
 * cuts does not take the short segment while the short segment sent before
 * is not acknowledged; without Nagle's algorithm it does.
 */
class TcpTsoDeferFillNagleTestCase : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param nagle True to enable Nagle's algorithm on the sender.
     * @param desc The test description.
     */
    TcpTsoDeferFillNagleTestCase(bool nagle, const std::string& desc)
        : TestCase(desc),
          m_nagle(nagle)
    {
    }

  private:
    void DoRun() override;

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
     * Trace sink of the ACKs the sender receives.
     * @param packet The payload.
     * @param header The TCP header.
     * @param socket The sending socket.
     */
    void AckReceived(Ptr<const Packet> packet,
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
     * Write data.
     * @param socket The sending socket.
     * @param size Bytes to write.
     */
    void Write(Ptr<Socket> socket, uint32_t size);

    static constexpr uint32_t BYTES = 50500; //!< Bytes to transfer

    bool m_nagle;             //!< Nagle's algorithm on the sender
    uint32_t m_received{0};   //!< Bytes the receiving application read
    uint32_t m_first{0};      //!< Sequence number of the first data byte
    uint32_t m_highest{0};    //!< Highest end of the data sent
    bool m_shortAcked{false}; //!< The first short segment was acknowledged
    uint32_t m_fills{0};      //!< Short segments sent before that
};

void
TcpTsoDeferFillNagleTestCase::SegmentSent(Ptr<const Packet> packet,
                                          const TcpHeader& header,
                                          Ptr<const TcpSocketBase> socket)
{
    if (header.GetFlags() & TcpHeader::SYN)
    {
        m_first = header.GetSequenceNumber().GetValue() + 1;
        return;
    }
    uint32_t start = header.GetSequenceNumber().GetValue() - m_first;
    if (packet->GetSize() == 0 || start + packet->GetSize() <= m_highest)
    {
        return;
    }
    m_highest = start + packet->GetSize();
    if (packet->GetSize() < 1448 && start > 1448 && !m_shortAcked)
    {
        ++m_fills;
    }
}

void
TcpTsoDeferFillNagleTestCase::AckReceived(Ptr<const Packet> packet,
                                          const TcpHeader& header,
                                          Ptr<const TcpSocketBase> socket)
{
    if (m_first > 0 && header.GetAckNumber().GetValue() - m_first >= 1948)
    {
        m_shortAcked = true;
    }
}

void
TcpTsoDeferFillNagleTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->SetRecvCallback(MakeCallback(&TcpTsoDeferFillNagleTestCase::Read, this));
}

void
TcpTsoDeferFillNagleTestCase::Read(Ptr<Socket> socket)
{
    while (Ptr<Packet> packet = socket->Recv())
    {
        m_received += packet->GetSize();
    }
}

void
TcpTsoDeferFillNagleTestCase::Write(Ptr<Socket> socket, uint32_t size)
{
    socket->Send(Create<Packet>(size), 0);
}

void
TcpTsoDeferFillNagleTestCase::DoRun()
{
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1460));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(8192));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(BYTES));
    Config::SetDefault("ns3::TcpSocketBase::LinuxRcvWindow", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocketBase::FillReceiveWindow", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocketBase::ForcedPush", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocketBase::TsoDefer", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocket::TcpNoDelay", BooleanValue(!m_nagle));

    NodeContainer nodes;
    nodes.Create(2);
    SimpleNetDeviceHelper devHelper;
    devHelper.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Mbps")));
    devHelper.SetChannelAttribute("Delay", TimeValue(MilliSeconds(50)));
    NetDeviceContainer devices = devHelper.Install(nodes);
    InternetStackHelper stack;
    stack.Install(nodes);
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t port = 9914;
    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpTsoDeferFillNagleTestCase::Accepted, this));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->TraceConnectWithoutContext(
        "Tx",
        MakeCallback(&TcpTsoDeferFillNagleTestCase::SegmentSent, this));
    client->TraceConnectWithoutContext(
        "Rx",
        MakeCallback(&TcpTsoDeferFillNagleTestCase::AckReceived, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));
    Simulator::Schedule(Seconds(1), &TcpTsoDeferFillNagleTestCase::Write, this, client, 1948);
    Simulator::Schedule(Seconds(1.0005),
                        &TcpTsoDeferFillNagleTestCase::Write,
                        this,
                        client,
                        BYTES - 1948);

    Simulator::Stop(Seconds(60));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_received, BYTES, "The transfer did not complete");
    if (m_nagle)
    {
        NS_TEST_ASSERT_MSG_EQ(m_fills, 0, "A short segment filled the window");
    }
    else
    {
        NS_TEST_ASSERT_MSG_GT(m_fills, 0, "No short segment filled the window");
    }
    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 *
 * @brief Test suite of the TsoDefer attribute
 */
class TcpTsoDeferTestSuite : public TestSuite
{
  public:
    TcpTsoDeferTestSuite()
        : TestSuite("tcp-tso-defer", Type::UNIT)
    {
        AddTestCase(new TcpTsoDeferTestCase(false, "Single segments sent at once"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpTsoDeferTestCase(true, "Single segments deferred as Linux TSO"),
                    TestCase::Duration::QUICK);
        AddTestCase(
            new TcpTsoDeferFillTestCase(1, "The segment filling the window is not deferred"),
            TestCase::Duration::QUICK);
        AddTestCase(new TcpTsoDeferFillTestCase(2, "No short segment sent alone"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpTsoDeferSplitTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpTsoDeferNagleTestCase(false, "Nagle holds a short segment"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpTsoDeferNagleTestCase(true, "Nagle as Linux, Minshall's variant"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpTsoSegmentsTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpTsoDeferFillNagleTestCase(true, "Nagle holds the window fill as Linux"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpTsoDeferFillNagleTestCase(false, "The window is filled without Nagle"),
                    TestCase::Duration::QUICK);
    }
};

static TcpTsoDeferTestSuite g_tcpTsoDeferTestSuite; //!< Static variable for test initialization
