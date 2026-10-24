/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

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

#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpForcedPushTest");

/**
 * @ingroup internet-test
 *
 * @brief Test the PSH flags set by the ForcedPush attribute
 *
 * The application of the sender writes the given chunks at once, with a
 * segment size of 500 bytes and a receive buffer of the given size. The
 * test records the end, relative to the first data byte, of every data
 * segment carrying PSH, and compares them with the expected ones.
 */
class TcpForcedPushTestCase : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param desc The test description.
     * @param forcedPush Value of the ForcedPush attribute of the sender.
     * @param rcvBuf Receive buffer size of the receiver.
     * @param writes Sizes of the application writes.
     * @param expected Ends of the segments expected to carry PSH.
     */
    TcpForcedPushTestCase(const std::string& desc,
                          bool forcedPush,
                          uint32_t rcvBuf,
                          std::vector<uint32_t> writes,
                          std::vector<uint32_t> expected)
        : TestCase(desc),
          m_forcedPush(forcedPush),
          m_rcvBuf(rcvBuf),
          m_writes(std::move(writes)),
          m_expected(std::move(expected))
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
     * Write the chunks.
     * @param socket The sending socket.
     */
    void Write(Ptr<Socket> socket);

    bool m_forcedPush;                //!< ForcedPush attribute of the sender
    uint32_t m_rcvBuf;                //!< Receive buffer size of the receiver
    std::vector<uint32_t> m_writes;   //!< Sizes of the application writes
    std::vector<uint32_t> m_expected; //!< Ends of the segments expected to carry PSH
    std::vector<uint32_t> m_pushed;   //!< Ends of the segments sent with PSH
    uint32_t m_first{0};              //!< Sequence number of the first data byte
    uint32_t m_highest{0};            //!< Highest end of the data segments sent
};

void
TcpForcedPushTestCase::SegmentSent(Ptr<const Packet> packet,
                                   const TcpHeader& header,
                                   Ptr<const TcpSocketBase> socket)
{
    if (header.GetFlags() & TcpHeader::SYN)
    {
        m_first = header.GetSequenceNumber().GetValue() + 1;
        return;
    }
    uint32_t end = header.GetSequenceNumber().GetValue() + packet->GetSize() - m_first;
    if (packet->GetSize() == 0 || end <= m_highest)
    {
        return;
    }
    m_highest = end;
    if (header.GetFlags() & TcpHeader::PSH)
    {
        m_pushed.push_back(end);
    }
}

void
TcpForcedPushTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->SetRecvCallback(MakeCallback(&TcpForcedPushTestCase::Read, this));
}

void
TcpForcedPushTestCase::Read(Ptr<Socket> socket)
{
    while (socket->Recv())
    {
    }
}

void
TcpForcedPushTestCase::Write(Ptr<Socket> socket)
{
    for (uint32_t size : m_writes)
    {
        socket->Send(Create<Packet>(size), 0);
    }
}

void
TcpForcedPushTestCase::DoRun()
{
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(500));
    Config::SetDefault("ns3::TcpSocketBase::Timestamp", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocketBase::WindowScaling", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(100));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(m_rcvBuf));
    Config::SetDefault("ns3::TcpSocketBase::ForcedPush", BooleanValue(m_forcedPush));

    NodeContainer nodes;
    nodes.Create(2);
    SimpleNetDeviceHelper devHelper;
    devHelper.SetChannelAttribute("Delay", TimeValue(MilliSeconds(10)));
    NetDeviceContainer devices = devHelper.Install(nodes);
    InternetStackHelper stack;
    stack.Install(nodes);
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t port = 9907;
    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpForcedPushTestCase::Accepted, this));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->TraceConnectWithoutContext("Tx",
                                       MakeCallback(&TcpForcedPushTestCase::SegmentSent, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));
    Simulator::Schedule(Seconds(1), &TcpForcedPushTestCase::Write, this, client);

    Simulator::Stop(Seconds(30));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_pushed.size(), m_expected.size(), "Unexpected number of PSH");
    for (size_t i = 0; i < std::min(m_pushed.size(), m_expected.size()); ++i)
    {
        NS_TEST_ASSERT_MSG_EQ(m_pushed[i], m_expected[i], "PSH " << i << " on another segment");
    }
    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 *
 * @brief Test suite of the ForcedPush attribute
 */
class TcpForcedPushTestSuite : public TestSuite
{
  public:
    TcpForcedPushTestSuite()
        : TestSuite("tcp-forced-push", Type::UNIT)
    {
        // Three writes of 1200 bytes: PSH on the last segment only, or, as
        // Linux, on the segments ending each write
        AddTestCase(new TcpForcedPushTestCase("PSH on the last segment of the buffer",
                                              false,
                                              65535,
                                              {1200, 1200, 1200},
                                              {3600}),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpForcedPushTestCase("PSH on the segment ending each write",
                                              true,
                                              65535,
                                              {1200, 1200, 1200},
                                              {1500, 2500, 3600}),
                    TestCase::Duration::QUICK);
        // One write of 5000 bytes into a window of 2000 bytes: as Linux,
        // PSH once more than half the window was sent since the last one
        AddTestCase(new TcpForcedPushTestCase("No PSH within a write", false, 2000, {5000}, {5000}),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpForcedPushTestCase("PSH every half window",
                                              true,
                                              2000,
                                              {5000},
                                              {1500, 3000, 4500, 5000}),
                    TestCase::Duration::QUICK);
    }
};

static TcpForcedPushTestSuite g_tcpForcedPushTestSuite; //!< Static variable for test initialization
