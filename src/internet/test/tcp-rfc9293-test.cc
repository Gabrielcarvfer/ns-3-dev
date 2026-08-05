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
#include "ns3/tcp-option.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"

#include <algorithm>
#include <set>
#include <vector>

/**
 * @file
 *
 * @brief Conformance tests for the requirements of @RFC{9293}
 *
 * The requirements of @RFC{9293} are labelled MUST-1 to MUST-69 in its
 * Section 3.9.3. The status of each of them in ns-3, and whether this suite
 * covers it, is the following.
 *
 * | Clause             | Requirement                             | ns-3     | Test |
 * | :----------------- | :-------------------------------------- | :------- | :--- |
 * | MUST-1             | Window treated as unsigned              | yes      | no   |
 * | MUST-2, MUST-3     | Checksum generated and checked          | optional | no   |
 * | MUST-4 to MUST-6   | Options received, unknown ones ignored  | yes      | yes  |
 * | MUST-7             | Illegal option length handled           | yes      | yes  |
 * | MUST-8, MUST-9     | Clock driven initial sequence numbers   | optional | yes  |
 * | MUST-13            | TIME-WAIT lasts 2xMSL                   | yes      | no   |
 * | MUST-14 to MUST-16 | MSS option, defaults and effective MSS  | yes      | yes  |
 * | MUST-17            | Nagle can be disabled                   | yes      | no   |
 * | MUST-30 to MUST-33 | Urgent mechanism and its notification   | yes      | yes  |
 * | MUST-34 to MUST-37 | Window shrinking and zero window probes | yes      | no   |
 * | MUST-38, MUST-39   | SWS avoidance, sender and receiver      | yes      | no   |
 * | MUST-40            | Delayed ACK below 0.5 s                 | yes      | no   |
 * | MUST-46            | Invalid remote address rejected         | yes      | yes  |
 * | MUST-47            | Soft errors reported to the application | yes      | no   |
 * | MUST-48, MUST-49   | Diffserv field and TTL configurable     | yes      | no   |
 * | MUST-51 to MUST-53 | IP source routes                        | no       | no   |
 * | MUST-54            | ICMP errors acted upon                  | yes      | yes  |
 * | MUST-55            | ICMP Source Quench discarded            | yes      | yes  |
 * | MUST-56            | Soft ICMP errors do not abort           | yes      | yes  |
 * | MUST-57, MUST-63   | SYN to or from an invalid address       | IP layer | no   |
 * | MUST-58, MUST-59   | ACK segments aggregated                 | yes      | no   |
 * | MUST-60, MUST-61   | Data not buffered forever, PSH on last  | yes      | no   |
 * | MUST-62            | Urgent pointer past the urgent data     | yes      | yes  |
 * | MUST-64            | Options off a word boundary handled     | yes      | yes  |
 * | MUST-65            | No MSS option on non-SYN segments       | yes      | yes  |
 * | MUST-66            | RST and URG at a zero receive window    | yes      | yes  |
 * | MUST-67            | Advertised MSS bounded by the buffer    | yes      | no   |
 * | MUST-68            | Options carry a length field            | yes      | yes  |
 * | MUST-69            | Padding after EOL is zeroed             | yes      | yes  |
 *
 * The clauses left out of the table (MUST-10 to MUST-12, MUST-18 to MUST-29,
 * MUST-41 to MUST-45 and MUST-50) govern the connection state machine and the
 * semantics of the OPEN, SEND, RECEIVE, CLOSE, ABORT and STATUS calls, which
 * this suite does not exercise.
 *
 * The segment crafting test cases follow the tcpreq conformance testing
 * framework (https://github.com/TheJokr/tcpreq), whose probes are injected
 * through a raw socket here instead of being sent to a remote host.
 */

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpRfc9293TestSuite");

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Base class of the test cases injecting crafted TCP segments
 *
 * Builds a two node topology in which the probing node sends hand built
 * segments through an IPv4 raw socket, so that header contents which the ns-3
 * TCP implementation would never generate can be exercised, and collects the
 * segments the target replies with.
 */
class TcpCraftedSegmentTestCase : public TestCase
{
  public:
    /**
     * Constructor.
     * @param name The name of the test case.
     */
    TcpCraftedSegmentTestCase(std::string name);

  protected:
    /**
     * Build the topology, and make the target listen unless told otherwise.
     * @param listen Whether the target listens on the target port.
     */
    void SetupTopology(bool listen = true);

    /**
     * Inject a segment towards the target.
     *
     * @param flags The TCP flags.
     * @param options The raw option bytes, whose size must be a multiple of 4.
     * @param reserved The reserved bits, in the four least significant bits.
     * @param sport The source port.
     */
    void SendSegment(uint8_t flags,
                     const std::vector<uint8_t>& options = {},
                     uint8_t reserved = 0,
                     uint16_t sport = 0);

    /**
     * Get the segments received from the target.
     * @return The headers of the received segments.
     */
    const std::vector<TcpHeader>& GetReplies() const;

    /**
     * Count the received segments carrying all the given flags.
     * @param flags The flags to look for.
     * @return The number of matching segments.
     */
    uint32_t CountReplies(uint8_t flags) const;

    Ipv4InterfaceContainer m_interfaces; //!< The interfaces of the two nodes
    NodeContainer m_nodes;               //!< The probing and the target node
    Ptr<Socket> m_prober;                //!< The raw socket of the probing node
    Ptr<Socket> m_target;                //!< The TCP socket of the target node
    uint16_t m_targetPort{9401};         //!< The port the target listens on
    uint16_t m_proberPort{9402};         //!< The port the probes come from

  private:
    /**
     * Receive a segment on the raw socket.
     * @param socket The raw socket.
     */
    void ReceiveRaw(Ptr<Socket> socket);

    std::vector<TcpHeader> m_replies; //!< The headers of the received segments
};

TcpCraftedSegmentTestCase::TcpCraftedSegmentTestCase(std::string name)
    : TestCase(name)
{
}

void
TcpCraftedSegmentTestCase::SetupTopology(bool listen)
{
    m_nodes.Create(2);

    SimpleNetDeviceHelper devHelper;
    NetDeviceContainer devices = devHelper.Install(m_nodes);

    InternetStackHelper stack;
    stack.Install(m_nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    m_interfaces = address.Assign(devices);

    if (listen)
    {
        m_target = Socket::CreateSocket(m_nodes.Get(1), TcpSocketFactory::GetTypeId());
        m_target->Bind(InetSocketAddress(Ipv4Address::GetAny(), m_targetPort));
        m_target->Listen();
    }

    m_prober = Socket::CreateSocket(m_nodes.Get(0), Ipv4RawSocketFactory::GetTypeId());
    m_prober->SetAttribute("Protocol", UintegerValue(6));
    m_prober->Bind(InetSocketAddress(m_interfaces.GetAddress(0), 0));
    m_prober->SetRecvCallback(MakeCallback(&TcpCraftedSegmentTestCase::ReceiveRaw, this));
}

void
TcpCraftedSegmentTestCase::SendSegment(uint8_t flags,
                                       const std::vector<uint8_t>& options,
                                       uint8_t reserved,
                                       uint16_t sport)
{
    NS_ASSERT(options.size() % 4 == 0);
    if (sport == 0)
    {
        sport = m_proberPort;
    }
    const uint8_t dataOffset = 5 + options.size() / 4;

    // The segment is built byte by byte: TcpHeader derives its data offset
    // from the options it holds, and drops the reserved bits, so neither an
    // illegal option nor a reserved bit can be expressed through it
    std::vector<uint8_t> segment = {
        static_cast<uint8_t>(sport >> 8),
        static_cast<uint8_t>(sport & 0xff),
        static_cast<uint8_t>(m_targetPort >> 8),
        static_cast<uint8_t>(m_targetPort & 0xff),
        0x00,
        0x00,
        0x00,
        0x01, // sequence number
        0x00,
        0x00,
        0x00,
        0x00,                                                       // acknowledgment number
        static_cast<uint8_t>((dataOffset << 4) | (reserved & 0xf)), // offset and reserved bits
        flags,
        0x10,
        0x00, // window size
        0x00,
        0x00, // checksum
        0x00,
        0x00 // urgent pointer
    };
    segment.insert(segment.end(), options.begin(), options.end());

    Ptr<Packet> packet = Create<Packet>(segment.data(), segment.size());
    m_prober->SendTo(packet, 0, InetSocketAddress(m_interfaces.GetAddress(1), 0));
}

void
TcpCraftedSegmentTestCase::ReceiveRaw(Ptr<Socket> socket)
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
    m_replies.push_back(tcp);
}

const std::vector<TcpHeader>&
TcpCraftedSegmentTestCase::GetReplies() const
{
    return m_replies;
}

uint32_t
TcpCraftedSegmentTestCase::CountReplies(uint8_t flags) const
{
    return std::count_if(m_replies.begin(), m_replies.end(), [flags](const TcpHeader& h) {
        return (h.GetFlags() & flags) == flags;
    });
}

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
 * @brief Test the support for the End of Option List and NOP options
 *
 * Ported from the OptionSupportTest of tcpreq. @RFC{9293}, Section 3.1
 * requires the two legacy options to be supported (MUST-4), receivers to be
 * prepared to process options which do not begin on a word boundary
 * (MUST-64), and the padding after the End of Option List option to be zeroed
 * (MUST-69): a SYN carrying them must still open a connection.
 */
class TcpLegacyOptionsTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpLegacyOptionsTestCase()
        : TcpCraftedSegmentTestCase("Legacy options are supported (MUST-4, MUST-64)")
    {
    }

  private:
    void DoRun() override
    {
        SetupTopology();

        // NOP, NOP, End of Option List, and one byte of zeroed padding: the
        // option list does not begin on a word boundary
        Simulator::Schedule(Seconds(1),
                            &TcpLegacyOptionsTestCase::SendSegment,
                            this,
                            TcpHeader::SYN,
                            std::vector<uint8_t>{0x01, 0x01, 0x00, 0x00},
                            0,
                            0);

        Simulator::Stop(Seconds(5));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_EQ(CountReplies(TcpHeader::SYN | TcpHeader::ACK),
                              1,
                              "A SYN with legacy options was not answered with a SYN+ACK");
        NS_TEST_ASSERT_MSG_EQ(CountReplies(TcpHeader::RST),
                              0,
                              "A SYN with legacy options was reset");

        Simulator::Destroy();
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that an unknown option is ignored without error
 *
 * Ported from the UnknownOptionTest of tcpreq. @RFC{9293}, Section 3.1
 * (MUST-6) requires any option which is not implemented to be ignored without
 * error, as long as it has a length field (MUST-68), so a SYN carrying one
 * must still open a connection.
 */
class TcpUnknownOptionTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpUnknownOptionTestCase()
        : TcpCraftedSegmentTestCase("Unknown options are ignored without error (MUST-6)")
    {
    }

  private:
    void DoRun() override
    {
        SetupTopology();

        // Option kind 158 is reserved by IANA, and is the one tcpreq probes
        // with: kind, length, three bytes of data, and one byte of padding
        Simulator::Schedule(Seconds(1),
                            &TcpUnknownOptionTestCase::SendSegment,
                            this,
                            TcpHeader::SYN,
                            std::vector<uint8_t>{158, 0x05, 0x58, 0xfa, 0x89, 0x01, 0x01, 0x00},
                            0,
                            0);

        Simulator::Stop(Seconds(5));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_EQ(CountReplies(TcpHeader::SYN | TcpHeader::ACK),
                              1,
                              "A SYN with an unknown option was not answered with a SYN+ACK");
        NS_TEST_ASSERT_MSG_EQ(CountReplies(TcpHeader::RST),
                              0,
                              "A SYN with an unknown option was reset");

        Simulator::Destroy();
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the reserved bits of the header are ignored
 *
 * Ported from the ReservedFlagsTest of tcpreq. The bits reserved for future
 * use in the header of @RFC{9293}, Section 3.1 must be ignored by a receiver
 * which does not implement whatever uses them, so a SYN setting one must
 * still open a connection.
 */
class TcpReservedFlagsTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpReservedFlagsTestCase()
        : TcpCraftedSegmentTestCase("Reserved header bits are ignored")
    {
    }

  private:
    void DoRun() override
    {
        SetupTopology();

        // The third reserved bit, as tcpreq probes with
        Simulator::Schedule(Seconds(1),
                            &TcpReservedFlagsTestCase::SendSegment,
                            this,
                            TcpHeader::SYN,
                            std::vector<uint8_t>{},
                            0x4,
                            0);

        Simulator::Stop(Seconds(5));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_EQ(CountReplies(TcpHeader::SYN | TcpHeader::ACK),
                              1,
                              "A SYN with a reserved bit set was not answered with a SYN+ACK");

        Simulator::Destroy();
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the reset answering a SYN towards a closed port
 *
 * Ported from the RSTACKTest of tcpreq. @RFC{9293}, Section 3.10.7.1 requires
 * a segment towards a port with no listener to be answered with a reset,
 * which acknowledges the incoming sequence number when the incoming segment
 * carries no ACK of its own.
 */
class TcpResetAckTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpResetAckTestCase()
        : TcpCraftedSegmentTestCase("A SYN towards a closed port is reset")
    {
    }

  private:
    void DoRun() override
    {
        // Nobody listens on the target port
        SetupTopology(false);

        Simulator::Schedule(Seconds(1),
                            &TcpResetAckTestCase::SendSegment,
                            this,
                            TcpHeader::SYN,
                            std::vector<uint8_t>{},
                            0,
                            0);

        Simulator::Stop(Seconds(5));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_EQ(GetReplies().size(), 1, "The SYN was not answered exactly once");

        const TcpHeader& rst = GetReplies().front();
        NS_TEST_ASSERT_MSG_NE(rst.GetFlags() & TcpHeader::RST, 0, "The answer is not a reset");
        NS_TEST_ASSERT_MSG_NE(rst.GetFlags() & TcpHeader::ACK,
                              0,
                              "The reset answering a segment without ACK does not acknowledge");
        // The probes carry sequence number 1 and no payload
        NS_TEST_ASSERT_MSG_EQ(rst.GetAckNumber(),
                              SequenceNumber32(2),
                              "The reset does not acknowledge the sequence number of the SYN");

        Simulator::Destroy();
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the peer honors the advertised MSS
 *
 * Ported from the MSSSupportTest of tcpreq, which opens a connection
 * advertising a small MSS and checks the size of the segments it receives.
 * @RFC{9293}, Section 3.7.1 requires the effective send MSS to be the smaller
 * of the send MSS and the largest transmissible payload (MUST-16), and the
 * MSS option not to be sent on non-SYN segments (MUST-65).
 */
class TcpAdvertisedMssTestCase : public TestCase
{
  public:
    TcpAdvertisedMssTestCase();

  private:
    void DoRun() override;

    /**
     * Trace sink of the segments received by the receiver.
     * @param packet The payload.
     * @param header The TCP header.
     * @param socket The receiving socket.
     */
    void SegmentReceived(Ptr<const Packet> packet,
                         const TcpHeader& header,
                         Ptr<const TcpSocketBase> socket);

    /**
     * Accept callback of the sender, which starts sending upon accept.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    uint32_t m_advertisedMss{515}; //!< MSS advertised by the receiver
    uint32_t m_largestPayload{0};  //!< Largest payload received
    uint32_t m_dataSegments{0};    //!< Number of data segments received
    uint32_t m_lateMssOptions{0};  //!< MSS options seen on non-SYN segments
};

TcpAdvertisedMssTestCase::TcpAdvertisedMssTestCase()
    : TestCase("The peer honors the advertised MSS (MUST-16, MUST-65)")
{
}

void
TcpAdvertisedMssTestCase::SegmentReceived(Ptr<const Packet> packet,
                                          const TcpHeader& header,
                                          Ptr<const TcpSocketBase> socket)
{
    if (packet->GetSize() > 0)
    {
        m_dataSegments++;
        m_largestPayload = std::max(m_largestPayload, packet->GetSize());
    }

    if (!(header.GetFlags() & TcpHeader::SYN) && header.HasOption(TcpOption::MSS))
    {
        m_lateMssOptions++;
    }
}

void
TcpAdvertisedMssTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->Send(Create<Packet>(20000), 0);
}

void
TcpAdvertisedMssTestCase::DoRun()
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

    const uint16_t port = 9501;

    // The sender is free to use a large segment size: the MSS advertised by
    // the receiver is what must bound the segments it sends
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1400));

    Ptr<Socket> sender = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    sender->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    sender->Listen();
    sender->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpAdvertisedMssTestCase::Accepted, this));

    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(m_advertisedMss));

    Ptr<Socket> receiver = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    receiver->Bind();
    receiver->TraceConnectWithoutContext(
        "Rx",
        MakeCallback(&TcpAdvertisedMssTestCase::SegmentReceived, this));
    receiver->Connect(InetSocketAddress(interfaces.GetAddress(1), port));

    Simulator::Stop(Seconds(20));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_GT(m_dataSegments, 0, "No data segment was received");
    NS_TEST_ASSERT_MSG_LT_OR_EQ(m_largestPayload,
                                m_advertisedMss,
                                "A segment larger than the advertised MSS was received");
    NS_TEST_ASSERT_MSG_EQ(m_lateMssOptions, 0, "An MSS option was carried by a non-SYN segment");

    Simulator::Destroy();
    Config::Reset();
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
        AddTestCase(new TcpLegacyOptionsTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpUnknownOptionTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpReservedFlagsTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpResetAckTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpAdvertisedMssTestCase(), TestCase::Duration::QUICK);
    }
};

static TcpRfc9293TestSuite g_tcpRfc9293TestSuite; //!< Static variable for test initialization
