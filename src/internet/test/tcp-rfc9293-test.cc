/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ns3/boolean.h"
#include "ns3/callback.h"
#include "ns3/config.h"
#include "ns3/icmpv4.h"
#include "ns3/inet-socket-address.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-raw-socket-factory.h"
#include "ns3/log.h"
#include "ns3/node-container.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/simulator.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-l4-protocol.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"

#include <algorithm>
#include <set>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpRfc9293TestSuite");

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that an OPEN towards an invalid remote address is rejected
 *
 * @RFC{9293}, Section 3.9.1.5 (MUST-46) requires a local OPEN call for an
 * invalid remote IP address, such as a broadcast or a multicast address, to
 * be rejected as an error.
 */
class TcpInvalidRemoteAddressTestCase : public TestCase
{
  public:
    TcpInvalidRemoteAddressTestCase();

  private:
    void DoRun() override;

    /**
     * Create a TCP socket on a node with an Internet stack.
     * @return The socket.
     */
    Ptr<Socket> CreateSocket();

    NodeContainer m_nodes; //!< The nodes hosting the sockets
};

TcpInvalidRemoteAddressTestCase::TcpInvalidRemoteAddressTestCase()
    : TestCase("Connect to a broadcast or multicast address is rejected (MUST-46)")
{
}

Ptr<Socket>
TcpInvalidRemoteAddressTestCase::CreateSocket()
{
    return Socket::CreateSocket(m_nodes.Get(0), TcpSocketFactory::GetTypeId());
}

void
TcpInvalidRemoteAddressTestCase::DoRun()
{
    m_nodes.Create(2);

    SimpleNetDeviceHelper devHelper;
    NetDeviceContainer devices = devHelper.Install(m_nodes);

    InternetStackHelper stack;
    stack.Install(m_nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t port = 4477;

    // Limited broadcast
    Ptr<Socket> socket = CreateSocket();
    NS_TEST_ASSERT_MSG_EQ(socket->Connect(InetSocketAddress(Ipv4Address::GetBroadcast(), port)),
                          -1,
                          "Connect to the limited broadcast address was accepted");
    NS_TEST_ASSERT_MSG_EQ(socket->GetErrno(),
                          Socket::ERROR_INVAL,
                          "Wrong error after connecting to the limited broadcast address");

    // Multicast
    socket = CreateSocket();
    NS_TEST_ASSERT_MSG_EQ(socket->Connect(InetSocketAddress(Ipv4Address("224.0.0.1"), port)),
                          -1,
                          "Connect to an IPv4 multicast address was accepted");
    NS_TEST_ASSERT_MSG_EQ(socket->GetErrno(),
                          Socket::ERROR_INVAL,
                          "Wrong error after connecting to an IPv4 multicast address");

    // IPv6 multicast
    socket = CreateSocket();
    NS_TEST_ASSERT_MSG_EQ(socket->Connect(Inet6SocketAddress(Ipv6Address("ff02::1"), port)),
                          -1,
                          "Connect to an IPv6 multicast address was accepted");
    NS_TEST_ASSERT_MSG_EQ(socket->GetErrno(),
                          Socket::ERROR_INVAL,
                          "Wrong error after connecting to an IPv6 multicast address");

    // A unicast address is still accepted
    socket = CreateSocket();
    NS_TEST_ASSERT_MSG_EQ(socket->Connect(InetSocketAddress(interfaces.GetAddress(1), port)),
                          0,
                          "Connect to a unicast address was rejected");

    Simulator::Run();
    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that ICMP Source Quench messages are silently discarded
 *
 * @RFC{9293}, Section 3.9.2.2 (MUST-55) requires TCP implementations to
 * silently discard any received ICMP Source Quench message, while the other
 * ICMP error messages must be acted on (MUST-54) without aborting the
 * connection (MUST-56).
 */
class TcpIcmpSourceQuenchTestCase : public TestCase
{
  public:
    TcpIcmpSourceQuenchTestCase();

  private:
    void DoRun() override;

    /**
     * ICMP callback of the socket under test.
     *
     * @param icmpSource The ICMP source address.
     * @param icmpTtl The ICMP TTL.
     * @param icmpType The ICMP type.
     * @param icmpCode The ICMP code.
     * @param icmpInfo The ICMP info.
     */
    void IcmpReceived(Ipv4Address icmpSource,
                      uint8_t icmpTtl,
                      uint8_t icmpType,
                      uint8_t icmpCode,
                      uint32_t icmpInfo);

    /**
     * State trace sink of the socket under test.
     *
     * @param oldState The previous state.
     * @param newState The new state.
     */
    void StateChanged(TcpSocket::TcpStates_t oldState, TcpSocket::TcpStates_t newState);

    TcpSocket::TcpStates_t m_state{TcpSocket::CLOSED}; //!< State of the socket under test
    uint32_t m_icmpCount{0};  //!< Number of ICMP messages delivered to the socket
    uint8_t m_lastType{0};    //!< Type of the last ICMP message delivered to the socket
    NodeContainer m_nodes;    //!< The nodes hosting the sockets
    Ptr<Socket> m_socket;     //!< The socket under test
    Ptr<Node> m_senderNode;   //!< The node hosting the socket under test
    uint16_t m_localPort{0};  //!< Local port of the socket under test
    uint16_t m_remotePort{0}; //!< Remote port of the socket under test
};

TcpIcmpSourceQuenchTestCase::TcpIcmpSourceQuenchTestCase()
    : TestCase("ICMP Source Quench messages are silently discarded (MUST-55)")
{
}

void
TcpIcmpSourceQuenchTestCase::IcmpReceived(Ipv4Address icmpSource,
                                          uint8_t icmpTtl,
                                          uint8_t icmpType,
                                          uint8_t icmpCode,
                                          uint32_t icmpInfo)
{
    m_icmpCount++;
    m_lastType = icmpType;
}

void
TcpIcmpSourceQuenchTestCase::StateChanged(TcpSocket::TcpStates_t oldState,
                                          TcpSocket::TcpStates_t newState)
{
    m_state = newState;
}

void
TcpIcmpSourceQuenchTestCase::DoRun()
{
    m_nodes.Create(2);

    SimpleNetDeviceHelper devHelper;
    NetDeviceContainer devices = devHelper.Install(m_nodes);

    InternetStackHelper stack;
    stack.Install(m_nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t serverPort = 4478;

    Ptr<Socket> server = Socket::CreateSocket(m_nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), serverPort));
    server->Listen();

    m_socket = Socket::CreateSocket(m_nodes.Get(0), TcpSocketFactory::GetTypeId());
    m_socket->Bind();
    m_socket->SetAttribute(
        "IcmpCallback",
        CallbackValue(MakeCallback(&TcpIcmpSourceQuenchTestCase::IcmpReceived, this)));
    m_socket->TraceConnectWithoutContext(
        "State",
        MakeCallback(&TcpIcmpSourceQuenchTestCase::StateChanged, this));
    m_socket->Connect(InetSocketAddress(interfaces.GetAddress(1), serverPort));

    Simulator::Stop(Seconds(1));
    Simulator::Run();

    Address local;
    m_socket->GetSockName(local);
    m_localPort = InetSocketAddress::ConvertFrom(local).GetPort();

    // The ICMP payload holds the first 8 bytes of the offending TCP segment,
    // i.e. the source and destination ports and the sequence number
    uint8_t payload[8] = {static_cast<uint8_t>(m_localPort >> 8),
                          static_cast<uint8_t>(m_localPort & 0xff),
                          static_cast<uint8_t>(serverPort >> 8),
                          static_cast<uint8_t>(serverPort & 0xff),
                          0,
                          0,
                          0,
                          0};

    Ptr<TcpL4Protocol> tcp = m_nodes.Get(0)->GetObject<TcpL4Protocol>();

    // An ICMP Source Quench (type 4) must never reach the socket
    tcp->ReceiveIcmp(interfaces.GetAddress(1),
                     64,
                     4,
                     0,
                     0,
                     interfaces.GetAddress(0),
                     interfaces.GetAddress(1),
                     payload);
    NS_TEST_ASSERT_MSG_EQ(m_icmpCount, 0, "An ICMP Source Quench was delivered to the socket");

    // Any other ICMP error message must be acted on
    tcp->ReceiveIcmp(interfaces.GetAddress(1),
                     64,
                     Icmpv4Header::ICMPV4_DEST_UNREACH,
                     Icmpv4DestinationUnreachable::ICMPV4_HOST_UNREACHABLE,
                     0,
                     interfaces.GetAddress(0),
                     interfaces.GetAddress(1),
                     payload);
    NS_TEST_ASSERT_MSG_EQ(m_icmpCount, 1, "An ICMP Destination Unreachable was not delivered");
    NS_TEST_ASSERT_MSG_EQ(m_lastType,
                          Icmpv4Header::ICMPV4_DEST_UNREACH,
                          "Wrong ICMP type delivered to the socket");

    // The soft error must not have aborted the connection (MUST-56)
    NS_TEST_ASSERT_MSG_EQ(m_state,
                          TcpSocket::ESTABLISHED,
                          "The connection was aborted by a soft ICMP error");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that a segment with malformed options resets the connection
 *
 * @RFC{9293}, Section 3.1 (MUST-7) prescribes handling an illegal option
 * length by resetting the connection and logging the error cause. The
 * segment is injected through a raw socket, so that an option length which
 * the ns-3 TCP implementation would never generate can be exercised.
 */
class TcpMalformedOptionsResetTestCase : public TestCase
{
  public:
    TcpMalformedOptionsResetTestCase();

  private:
    void DoRun() override;

    /**
     * Receive a segment on the raw socket of the prober.
     * @param socket The raw socket.
     */
    void ReceiveRaw(Ptr<Socket> socket);

    /**
     * Inject a SYN segment carrying an option with an illegal length.
     * @param socket The raw socket.
     * @param dst The destination address.
     */
    void SendMalformedSyn(Ptr<Socket> socket, Ipv4Address dst);

    uint32_t m_rstCount{0};      //!< Number of RST segments received by the prober
    uint32_t m_synAckCount{0};   //!< Number of SYN+ACK segments received by the prober
    uint16_t m_proberPort{9000}; //!< Source port used by the prober
    uint16_t m_targetPort{9001}; //!< Port the target listens on
};

TcpMalformedOptionsResetTestCase::TcpMalformedOptionsResetTestCase()
    : TestCase("A segment with malformed options resets the connection (MUST-7)")
{
}

void
TcpMalformedOptionsResetTestCase::ReceiveRaw(Ptr<Socket> socket)
{
    Address from;
    Ptr<Packet> packet = socket->RecvFrom(from);

    Ipv4Header ipv4;
    packet->RemoveHeader(ipv4);
    if (ipv4.GetProtocol() != 6)
    {
        return;
    }

    TcpHeader tcp;
    packet->RemoveHeader(tcp);
    if (tcp.GetDestinationPort() != m_proberPort)
    {
        return;
    }

    if (tcp.GetFlags() & TcpHeader::RST)
    {
        m_rstCount++;
    }
    else if ((tcp.GetFlags() & TcpHeader::SYN) && (tcp.GetFlags() & TcpHeader::ACK))
    {
        m_synAckCount++;
    }
}

void
TcpMalformedOptionsResetTestCase::SendMalformedSyn(Ptr<Socket> socket, Ipv4Address dst)
{
    // The segment is built byte by byte: TcpHeader derives its data offset
    // from the options it carries, so an option with an illegal length cannot
    // be expressed through it. The header claims one word of options holding
    // a timestamp option (kind 8) which announces a length of 10 bytes, more
    // than the option space it lies in
    const uint8_t segment[24] = {
        static_cast<uint8_t>(m_proberPort >> 8),
        static_cast<uint8_t>(m_proberPort & 0xff), // source port
        static_cast<uint8_t>(m_targetPort >> 8),
        static_cast<uint8_t>(m_targetPort & 0xff), // destination port
        0x00,
        0x00,
        0x00,
        0x01, // sequence number
        0x00,
        0x00,
        0x00,
        0x00,           // acknowledgment number
        0x60,           // data offset: 6 words
        TcpHeader::SYN, // flags
        0x10,
        0x00, // window size
        0x00,
        0x00, // checksum
        0x00,
        0x00, // urgent pointer
        0x08,
        0x0a,
        0x00,
        0x00 // malformed timestamp option
    };

    Ptr<Packet> packet = Create<Packet>(segment, sizeof(segment));
    socket->SendTo(packet, 0, InetSocketAddress(dst, 0));
}

void
TcpMalformedOptionsResetTestCase::DoRun()
{
    NodeContainer nodes;
    nodes.Create(2);

    SimpleNetDeviceHelper devHelper;
    NetDeviceContainer devices = devHelper.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    // The target listens, so that a well-formed SYN would be answered with a
    // SYN+ACK rather than with a RST
    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), m_targetPort));
    server->Listen();

    Ptr<Socket> prober = Socket::CreateSocket(nodes.Get(0), Ipv4RawSocketFactory::GetTypeId());
    prober->SetAttribute("Protocol", UintegerValue(6));
    prober->Bind(InetSocketAddress(interfaces.GetAddress(0), 0));
    prober->SetRecvCallback(MakeCallback(&TcpMalformedOptionsResetTestCase::ReceiveRaw, this));

    Simulator::Schedule(Seconds(1),
                        &TcpMalformedOptionsResetTestCase::SendMalformedSyn,
                        this,
                        prober,
                        interfaces.GetAddress(1));

    Simulator::Stop(Seconds(5));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_synAckCount,
                          0,
                          "The connection was established despite the malformed options");
    NS_TEST_ASSERT_MSG_EQ(m_rstCount, 1, "The malformed options did not reset the connection");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the initial sequence numbers are clock driven
 *
 * @RFC{9293}, Section 3.4.1 (MUST-8) requires the initial sequence number of
 * a connection to be selected from a clock, so that the sequence numbers of
 * distinct connections between the same pair of sockets do not overlap. The
 * SYN segments are injected through a raw socket, so that several connections
 * towards the same listening socket can be opened from the test.
 *
 * Clock driven initial sequence numbers are optional, so the test enables
 * them.
 */
class TcpInitialSequenceNumberTestCase : public TestCase
{
  public:
    TcpInitialSequenceNumberTestCase();

  private:
    void DoRun() override;

    /**
     * Receive a segment on the raw socket of the prober.
     * @param socket The raw socket.
     */
    void ReceiveRaw(Ptr<Socket> socket);

    /**
     * Inject a SYN segment from the given source port.
     * @param socket The raw socket.
     * @param dst The destination address.
     * @param sport The source port.
     */
    void SendSyn(Ptr<Socket> socket, Ipv4Address dst, uint16_t sport);

    std::vector<uint32_t> m_isns; //!< Initial sequence numbers of the target
    uint16_t m_targetPort{9101};  //!< Port the target listens on
};

TcpInitialSequenceNumberTestCase::TcpInitialSequenceNumberTestCase()
    : TestCase("The initial sequence numbers are clock driven (MUST-8)")
{
}

void
TcpInitialSequenceNumberTestCase::ReceiveRaw(Ptr<Socket> socket)
{
    Address from;
    Ptr<Packet> packet = socket->RecvFrom(from);

    Ipv4Header ipv4;
    packet->RemoveHeader(ipv4);
    if (ipv4.GetProtocol() != 6)
    {
        return;
    }

    TcpHeader tcp;
    packet->RemoveHeader(tcp);
    if ((tcp.GetFlags() & (TcpHeader::SYN | TcpHeader::ACK)) == (TcpHeader::SYN | TcpHeader::ACK))
    {
        m_isns.push_back(tcp.GetSequenceNumber().GetValue());
    }
}

void
TcpInitialSequenceNumberTestCase::SendSyn(Ptr<Socket> socket, Ipv4Address dst, uint16_t sport)
{
    Ptr<Packet> packet = Create<Packet>();

    TcpHeader tcp;
    tcp.SetSourcePort(sport);
    tcp.SetDestinationPort(m_targetPort);
    tcp.SetSequenceNumber(SequenceNumber32(1));
    tcp.SetAckNumber(SequenceNumber32(0));
    tcp.SetFlags(TcpHeader::SYN);
    tcp.SetWindowSize(4096);
    packet->AddHeader(tcp);

    socket->SendTo(packet, 0, InetSocketAddress(dst, 0));
}

void
TcpInitialSequenceNumberTestCase::DoRun()
{
    Config::SetDefault("ns3::TcpL4Protocol::ClockDrivenIsn", BooleanValue(true));

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
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), m_targetPort));
    server->Listen();

    Ptr<Socket> prober = Socket::CreateSocket(nodes.Get(0), Ipv4RawSocketFactory::GetTypeId());
    prober->SetAttribute("Protocol", UintegerValue(6));
    prober->Bind(InetSocketAddress(interfaces.GetAddress(0), 0));
    prober->SetRecvCallback(MakeCallback(&TcpInitialSequenceNumberTestCase::ReceiveRaw, this));

    // Open three connections from different source ports, spaced in time so
    // that the clock component of the initial sequence number advances
    const uint16_t ports[3] = {9201, 9202, 9203};
    for (uint32_t i = 0; i < 3; ++i)
    {
        Simulator::Schedule(Seconds(1 + i),
                            &TcpInitialSequenceNumberTestCase::SendSyn,
                            this,
                            prober,
                            interfaces.GetAddress(1),
                            ports[i]);
    }

    Simulator::Stop(Seconds(10));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_GT_OR_EQ(m_isns.size(), 3, "Not every SYN was answered with a SYN+ACK");

    // The initial sequence numbers must not be the constant zero of a
    // sequence-number-agnostic implementation
    uint32_t zeros = std::count(m_isns.begin(), m_isns.end(), 0);
    NS_TEST_ASSERT_MSG_EQ(zeros, 0, "An initial sequence number was zero");

    // Distinct connections must not reuse the same initial sequence number
    std::set<uint32_t> distinct(m_isns.begin(), m_isns.end());
    NS_TEST_ASSERT_MSG_EQ(distinct.size(),
                          m_isns.size(),
                          "Two connections shared an initial sequence number");

    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the urgent mechanism
 *
 * @RFC{9293}, Section 3.8.5 requires TCP implementations to support the
 * urgent mechanism (MUST-30) for a sequence of urgent data of any length
 * (MUST-31), with the urgent pointer pointing to the sequence number of the
 * octet following the urgent data (MUST-62). The receiver must inform the
 * application asynchronously when urgent data becomes pending (MUST-32) and
 * must provide a way to learn how much urgent data is pending (MUST-33).
 */
class TcpUrgentDataTestCase : public TestCase
{
  public:
    TcpUrgentDataTestCase();

  private:
    void DoRun() override;

    /**
     * Urgent data callback of the receiving socket.
     * @param socket The receiving socket.
     */
    void UrgentDataPending(Ptr<Socket> socket);

    /**
     * Trace sink of the segments sent by the sender.
     * @param packet The payload.
     * @param header The TCP header.
     * @param socket The sending socket.
     */
    void SegmentSent(Ptr<const Packet> packet,
                     const TcpHeader& header,
                     Ptr<const TcpSocketBase> socket);

    /**
     * Send the urgent data.
     * @param socket The sending socket.
     */
    void SendUrgent(Ptr<Socket> socket);

    /**
     * Send ordinary data.
     * @param socket The sending socket.
     */
    void SendNormal(Ptr<Socket> socket);

    /**
     * Set the urgent data callback on the socket forked on accept.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void AcceptedSocket(Ptr<Socket> socket, const Address& from);

    uint32_t m_notifications{0};   //!< Number of urgent data notifications
    uint32_t m_notifiedSize{0};    //!< Pending urgent bytes reported at the first notification
    uint32_t m_urgentSegments{0};  //!< Number of segments carrying the URG flag
    uint16_t m_urgentPointer{0};   //!< Urgent pointer of the first urgent segment
    uint32_t m_urgentPayload{500}; //!< Size of the urgent data
};

TcpUrgentDataTestCase::TcpUrgentDataTestCase()
    : TestCase("Urgent data is flagged, pointed at and notified (MUST-30 to MUST-33, MUST-62)")
{
}

void
TcpUrgentDataTestCase::AcceptedSocket(Ptr<Socket> socket, const Address& from)
{
    socket->SetUrgentDataCallback(MakeCallback(&TcpUrgentDataTestCase::UrgentDataPending, this));
}

void
TcpUrgentDataTestCase::UrgentDataPending(Ptr<Socket> socket)
{
    if (m_notifications == 0)
    {
        m_notifiedSize = socket->GetUrgentDataSize();
    }
    m_notifications++;
}

void
TcpUrgentDataTestCase::SegmentSent(Ptr<const Packet> packet,
                                   const TcpHeader& header,
                                   Ptr<const TcpSocketBase> socket)
{
    if (header.GetFlags() & TcpHeader::URG)
    {
        if (m_urgentSegments == 0)
        {
            m_urgentPointer = header.GetUrgentPointer();
        }
        m_urgentSegments++;
    }
}

void
TcpUrgentDataTestCase::SendNormal(Ptr<Socket> socket)
{
    socket->Send(Create<Packet>(100), 0);
}

void
TcpUrgentDataTestCase::SendUrgent(Ptr<Socket> socket)
{
    socket->Send(Create<Packet>(m_urgentPayload), Socket::MSG_FLAG_OOB);
}

void
TcpUrgentDataTestCase::DoRun()
{
    NodeContainer nodes;
    nodes.Create(2);

    SimpleNetDeviceHelper devHelper;
    NetDeviceContainer devices = devHelper.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t port = 9301;

    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    // The urgent data callback is inherited by the socket forked on accept
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpUrgentDataTestCase::AcceptedSocket, this));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->TraceConnectWithoutContext("Tx",
                                       MakeCallback(&TcpUrgentDataTestCase::SegmentSent, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));

    Simulator::Schedule(Seconds(1), &TcpUrgentDataTestCase::SendNormal, this, client);
    Simulator::Schedule(Seconds(2), &TcpUrgentDataTestCase::SendUrgent, this, client);

    Simulator::Stop(Seconds(10));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_GT(m_urgentSegments, 0, "No segment carried the URG flag");
    NS_TEST_ASSERT_MSG_EQ(m_urgentPointer,
                          m_urgentPayload,
                          "The urgent pointer does not point past the urgent data");
    NS_TEST_ASSERT_MSG_GT(m_notifications, 0, "The application was not notified of urgent data");
    NS_TEST_ASSERT_MSG_EQ(m_notifiedSize,
                          m_urgentPayload,
                          "Wrong amount of pending urgent data reported");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief TCP RFC 9293 conformance TestSuite
 */
class TcpRfc9293TestSuite : public TestSuite
{
  public:
    TcpRfc9293TestSuite()
        : TestSuite("tcp-rfc9293", Type::UNIT)
    {
        AddTestCase(new TcpInvalidRemoteAddressTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpIcmpSourceQuenchTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpMalformedOptionsResetTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpInitialSequenceNumberTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpUrgentDataTestCase(), TestCase::Duration::QUICK);
    }
};

static TcpRfc9293TestSuite g_tcpRfc9293TestSuite; //!< Static variable for test initialization
