/*
 * Copyright (c) 2026 Centre Tecnològic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "tcp-general-test.h"

#include "ns3/boolean.h"
#include "ns3/inet-socket-address.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/log.h"
#include "ns3/node-container.h"
#include "ns3/node.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/simple-net-device.h"
#include "ns3/simulator.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-l4-protocol.h"
#include "ns3/tcp-option-rfc793.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"

#include <map>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpMssTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief TCP MSS option test (see @issueid{946})
 *
 * Checks that the MSS option is sent in every SYN segment (@RFC{9293},
 * Section 3.7.1, SHLD-5 and MAY-3), that it advertises the largest segment
 * the interface receives, whatever the configured segment size (MUST-67,
 * and @RFC{6691}, Section 2), and that the segment size used by the sender
 * is the smaller of its configured one and the value advertised by the peer
 * (MUST-16): a receiver configured with a small segment size does not shrink
 * the segments of its peer.
 */
class TcpMssOptionTestCase : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param senderSegSize Segment size configured on the sender.
     * @param receiverSegSize Segment size configured on the receiver.
     * @param name Test description.
     * @param mtu MTU of the link.
     */
    TcpMssOptionTestCase(uint32_t senderSegSize,
                         uint32_t receiverSegSize,
                         std::string name,
                         uint32_t mtu = 1500)
        : TcpGeneralTest(name),
          m_senderSegSize(senderSegSize),
          m_receiverSegSize(receiverSegSize),
          m_mtu(mtu)
    {
    }

  protected:
    void ConfigureEnvironment() override;
    void ConfigureProperties() override;
    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;

  private:
    static constexpr uint32_t IPV4_TCP_HEADERS_SIZE = 40; //!< Base IPv4 and TCP headers

    uint32_t m_senderSegSize;   //!< Segment size configured on the sender.
    uint32_t m_receiverSegSize; //!< Segment size configured on the receiver.
    uint32_t m_mtu;             //!< MTU of the link.
};

void
TcpMssOptionTestCase::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetMTU(m_mtu);
}

void
TcpMssOptionTestCase::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetSegmentSize(SENDER, m_senderSegSize);
    SetSegmentSize(RECEIVER, m_receiverSegSize);
}

void
TcpMssOptionTestCase::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    NS_LOG_INFO(h);

    if (h.GetFlags() & TcpHeader::SYN)
    {
        NS_TEST_ASSERT_MSG_EQ(h.HasOption(TcpOption::MSS), true, "MSS option missing in SYN");
        Ptr<const TcpOptionMSS> mss = DynamicCast<const TcpOptionMSS>(h.GetOption(TcpOption::MSS));
        NS_TEST_ASSERT_MSG_EQ(mss->GetMSS(),
                              m_mtu - IPV4_TCP_HEADERS_SIZE,
                              (who == SENDER ? "Sender" : "Receiver")
                                  << " did not advertise the MSS of its interface");
    }
    else
    {
        NS_TEST_ASSERT_MSG_EQ(h.HasOption(TcpOption::MSS),
                              false,
                              "MSS option present in non-SYN segment");
        if (who == SENDER && p->GetSize() > 0)
        {
            // The segment size in use is the minimum of the configured and the
            // advertised MSS, decreased by the size of the timestamp option
            // carried by every segment (RFC 9293, Section 3.7.1), which is
            // enabled by default
            constexpr uint32_t TS_OPTION_SIZE = 12;
            NS_TEST_ASSERT_MSG_EQ(GetSegSize(SENDER),
                                  std::min(m_senderSegSize, m_mtu - IPV4_TCP_HEADERS_SIZE) -
                                      TS_OPTION_SIZE,
                                  "Sender segment size not clamped to the advertised MSS");
        }
    }
}

/**
 * @ingroup internet-test
 *
 * @brief Test that a listener negotiates each connection independently
 *
 * The options of a SYN are negotiated by the socket forked for its
 * connection, not by the listening socket, so that neither the listener nor
 * the connections accepted later are affected by the MSS a previous client
 * advertised (@RFC{9293}, Section 3.9.1.1, MUST-41): two clients whose
 * interfaces have different MTUs connect in turn, and the segments of each
 * accepted connection are sized by the MSS its own peer advertised.
 */
class TcpListenerMssTestCase : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param clockDrivenIsn Whether the initial sequence numbers are clock
     *                       driven, so that the accepted sockets, which send
     *                       the data, pick them anywhere in the space.
     */
    TcpListenerMssTestCase(bool clockDrivenIsn = false)
        : TestCase(std::string("A listener negotiates the MSS of each connection on its own") +
                   (clockDrivenIsn ? " (clock driven ISN)" : "")),
          m_clockDrivenIsn(clockDrivenIsn)
    {
    }

  private:
    void DoRun() override;

    /**
     * Accept callback, which sends data on the accepted socket and records
     * its segment size.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    /**
     * Tx trace of an accepted socket, recording the largest payload sent.
     * @param port The peer port, identifying the connection.
     * @param packet The segment sent.
     * @param header The TCP header.
     * @param socket The sending socket.
     */
    void SegmentSent(uint16_t port,
                     Ptr<const Packet> packet,
                     const TcpHeader& header,
                     Ptr<const TcpSocketBase> socket);

    bool m_clockDrivenIsn;                         //!< Whether the ISNs are clock driven
    std::map<uint16_t, uint32_t> m_largestPayload; //!< Largest payload sent, per peer port
    std::map<uint16_t, uint32_t> m_segmentSize;    //!< Segment size of the accepted socket
};

void
TcpListenerMssTestCase::SegmentSent(uint16_t port,
                                    Ptr<const Packet> packet,
                                    const TcpHeader& header,
                                    Ptr<const TcpSocketBase> socket)
{
    m_largestPayload[port] = std::max(m_largestPayload[port], packet->GetSize());
}

void
TcpListenerMssTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    uint16_t port = InetSocketAddress::ConvertFrom(from).GetPort();
    UintegerValue segmentSize;
    socket->GetAttribute("SegmentSize", segmentSize);
    m_segmentSize[port] = segmentSize.Get();
    socket->TraceConnectWithoutContext(
        "Tx",
        MakeCallback(&TcpListenerMssTestCase::SegmentSent, this).Bind(port));
    socket->Send(Create<Packet>(10000), 0);
}

void
TcpListenerMssTestCase::DoRun()
{
    const uint16_t port = 9500;
    const uint16_t smallPort = 9501;
    const uint16_t largePort = 9502;
    const uint32_t smallMss = 500;
    const uint32_t largeMss = 1400;

    // The clients advertise the MSS of their interfaces: node 0 has the small
    // one, node 2 the large one, and node 1 is the server
    NodeContainer nodes;
    nodes.Create(3);

    SimpleNetDeviceHelper devHelper;
    NetDeviceContainer devices = devHelper.Install(nodes);
    constexpr uint32_t IPV4_TCP_HEADERS_SIZE = 40;
    DynamicCast<SimpleNetDevice>(devices.Get(0))->SetMtu(smallMss + IPV4_TCP_HEADERS_SIZE);

    InternetStackHelper stack;
    stack.Install(nodes);
    for (uint32_t i = 0; i < nodes.GetN(); ++i)
    {
        nodes.Get(i)->GetObject<TcpL4Protocol>()->SetAttribute("ClockDrivenIsn",
                                                               BooleanValue(m_clockDrivenIsn));
    }

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->SetAttribute("SegmentSize", UintegerValue(largeMss));
    // The timestamps would shrink the payload below the segment size
    server->SetAttribute("Timestamp", BooleanValue(false));
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpListenerMssTestCase::Accepted, this));

    // The client with the small MSS connects first, then the one with the
    // large MSS: the second must not be clamped by the first
    Ptr<Socket> smallClient = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    smallClient->SetAttribute("SegmentSize", UintegerValue(smallMss));
    smallClient->SetAttribute("Timestamp", BooleanValue(false));
    smallClient->Bind(InetSocketAddress(interfaces.GetAddress(0), smallPort));
    Simulator::Schedule(Seconds(1),
                        &Socket::Connect,
                        smallClient,
                        InetSocketAddress(interfaces.GetAddress(1), port));

    Ptr<Socket> largeClient = Socket::CreateSocket(nodes.Get(2), TcpSocketFactory::GetTypeId());
    largeClient->SetAttribute("SegmentSize", UintegerValue(largeMss));
    largeClient->SetAttribute("Timestamp", BooleanValue(false));
    largeClient->Bind(InetSocketAddress(interfaces.GetAddress(2), largePort));
    Simulator::Schedule(Seconds(3),
                        &Socket::Connect,
                        largeClient,
                        InetSocketAddress(interfaces.GetAddress(1), port));

    Simulator::Stop(Seconds(10));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_segmentSize[smallPort],
                          smallMss,
                          "The first connection was not clamped to the MSS of its peer");
    NS_TEST_ASSERT_MSG_EQ(m_segmentSize[largePort],
                          largeMss,
                          "The second connection was clamped by the MSS of the first peer");
    NS_TEST_ASSERT_MSG_EQ(m_largestPayload[smallPort],
                          smallMss,
                          "The first connection sent segments of the wrong size");
    NS_TEST_ASSERT_MSG_EQ(m_largestPayload[largePort],
                          largeMss,
                          "The second connection sent segments of the wrong size");

    // and the listener itself keeps its configured segment size
    UintegerValue listenerSegmentSize;
    server->GetAttribute("SegmentSize", listenerSegmentSize);
    NS_TEST_ASSERT_MSG_EQ(listenerSegmentSize.Get(),
                          largeMss,
                          "The listening socket was altered by the SYN it received");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 *
 * @brief The MSS option test with clock driven initial sequence numbers
 *
 * The connection is opened with initial sequence numbers which may lie
 * anywhere in the space (@RFC{9293}, Section 3.4.1), and has to negotiate the
 * MSS and transfer its data as with the ones starting from zero.
 */
class TcpMssOptionClockDrivenIsnTestCase : public TcpMssOptionTestCase
{
  public:
    /**
     * @brief Constructor.
     * @param senderSegSize Segment size configured on the sender.
     * @param receiverSegSize Segment size configured on the receiver.
     * @param name Test description.
     */
    TcpMssOptionClockDrivenIsnTestCase(uint32_t senderSegSize,
                                       uint32_t receiverSegSize,
                                       std::string name)
        : TcpMssOptionTestCase(senderSegSize, receiverSegSize, name + " (clock driven ISN)")
    {
    }

  protected:
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override
    {
        node->GetObject<TcpL4Protocol>()->SetAttribute("ClockDrivenIsn", BooleanValue(true));
        return TcpMssOptionTestCase::CreateSenderSocket(node);
    }

    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override
    {
        node->GetObject<TcpL4Protocol>()->SetAttribute("ClockDrivenIsn", BooleanValue(true));
        return TcpMssOptionTestCase::CreateReceiverSocket(node);
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Test that writes sized from the segment size fill one segment each
 *
 * The SegmentSize attribute counts the data of a segment only, and the
 * timestamp option every segment carries comes out of it (@RFC{6691},
 * Section 2). An application whose writes should each fill one segment
 * configures SegmentSize as the size of its writes plus
 * TcpSocketBase::GetSegmentOptionsSize(); once connected, the SegmentSize
 * attribute of the socket gives the data of a full segment. Each write of an
 * application limited connection then leaves as a single segment, without a
 * runt segment after it.
 */
class TcpSegmentDataSizeTestCase : public TestCase
{
  public:
    TcpSegmentDataSizeTestCase()
        : TestCase("Writes sized from the segment size fill one segment each")
    {
    }

  private:
    void DoRun() override;

    /**
     * Tx trace of the client, recording the data segment sizes.
     * @param packet The segment sent.
     * @param header The TCP header.
     * @param socket The sending socket.
     */
    void SegmentSent(Ptr<const Packet> packet,
                     const TcpHeader& header,
                     Ptr<const TcpSocketBase> socket)
    {
        if (packet->GetSize() > 0)
        {
            m_payloads.push_back(packet->GetSize());
        }
    }

    std::vector<uint32_t> m_payloads; //!< Data sizes of the segments sent
};

void
TcpSegmentDataSizeTestCase::DoRun()
{
    const uint32_t writeSize = 1000;
    const uint32_t writes = 5;
    const uint16_t port = 9600;

    NodeContainer nodes;
    nodes.Create(2);
    SimpleNetDeviceHelper devHelper;
    NetDeviceContainer devices = devHelper.Install(nodes);
    InternetStackHelper stack;
    stack.Install(nodes);
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->SetAttribute("SegmentSize",
                         UintegerValue(writeSize + TcpSocketBase::GetSegmentOptionsSize()));
    client->TraceConnectWithoutContext(
        "Tx",
        MakeCallback(&TcpSegmentDataSizeTestCase::SegmentSent, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));
    // Application limited: every write leaves before the next one
    for (uint32_t i = 0; i < writes; ++i)
    {
        Simulator::Schedule(Seconds(1 + i),
                            [client, writeSize]() { client->Send(Create<Packet>(writeSize), 0); });
    }

    Simulator::Stop(Seconds(10));
    Simulator::Run();

    UintegerValue segmentSize;
    client->GetAttribute("SegmentSize", segmentSize);
    NS_TEST_ASSERT_MSG_EQ(segmentSize.Get(),
                          writeSize,
                          "The segment size of the connection is not its data size");
    NS_TEST_ASSERT_MSG_EQ(m_payloads.size(), writes, "The writes did not fill one segment each");
    for (uint32_t payload : m_payloads)
    {
        NS_TEST_ASSERT_MSG_EQ(payload, writeSize, "A write was split across segments");
    }

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 *
 * @brief TCP MSS option TestSuite
 */
class TcpMssTestSuite : public TestSuite
{
  public:
    TcpMssTestSuite()
        : TestSuite("tcp-mss", Type::UNIT)
    {
        AddTestCase(new TcpMssOptionTestCase(1400, 800, "MSS option, sender larger than receiver"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpMssOptionTestCase(800, 1400, "MSS option, receiver larger than sender"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpMssOptionTestCase(536, 536, "MSS option, default segment size"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpMssOptionTestCase(1400,
                                             800,
                                             "MSS option, segment size larger than the interface",
                                             1000),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpListenerMssTestCase(), TestCase::Duration::QUICK);
        AddTestCase(
            new TcpMssOptionClockDrivenIsnTestCase(1400,
                                                   800,
                                                   "MSS option, sender larger than receiver"),
            TestCase::Duration::QUICK);
        AddTestCase(
            new TcpMssOptionClockDrivenIsnTestCase(800,
                                                   1400,
                                                   "MSS option, receiver larger than sender"),
            TestCase::Duration::QUICK);
        AddTestCase(new TcpListenerMssTestCase(true), TestCase::Duration::QUICK);
        AddTestCase(new TcpSegmentDataSizeTestCase(), TestCase::Duration::QUICK);
    }
};

static TcpMssTestSuite g_tcpMssTestSuite; //!< static var for test initialization
