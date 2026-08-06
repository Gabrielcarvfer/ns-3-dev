/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ns3/boolean.h"
#include "ns3/callback.h"
#include "ns3/config.h"
#include "ns3/double.h"
#include "ns3/error-model.h"
#include "ns3/global-value.h"
#include "ns3/iana-internet-protocol-numbers.h"
#include "ns3/icmpv4.h"
#include "ns3/inet-socket-address.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/ipv4-raw-socket-factory.h"
#include "ns3/log.h"
#include "ns3/node-container.h"
#include "ns3/node.h"
#include "ns3/nstime.h"
#include "ns3/pointer.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-l4-protocol.h"
#include "ns3/tcp-option-rfc793.h"
#include "ns3/tcp-option.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"

#include <algorithm>
#include <map>
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
 * | Clause     | Requirement                                               | ns-3 | Test  |
 * | :--------- | :-------------------------------------------------------- | :--- | :---- |
 * | MUST 1     | Window treated as unsigned                                | yes  | yes   |
 * | MUST 2,3   | Checksum generated and checked                            | opt  | yes   |
 * | MUST 4-6   | Options received in any segment, unknown ones ignored     | yes  | yes   |
 * | MUST 7     | Illegal option length handled                             | yes  | yes   |
 * | MUST 8,9   | Initial sequence numbers driven by a clock and a secret   | opt  | yes   |
 * | MUST 10,11 | Simultaneous open, and how SYN-RECEIVED was reached       | yes  | yes   |
 * | MUST 12    | Normal close told apart from an abort                     | yes  | yes   |
 * | MUST 13    | TIME-WAIT lasts 2xMSL                                     | yes  | yes   |
 * | MUST 14-16 | MSS option, its defaults and the effective send MSS       | yes  | yes   |
 * | MUST 17    | Nagle algorithm can be disabled                           | yes  | yes   |
 * | MUST 18,19 | RTO estimation and congestion control                     | yes  | other |
 * | MUST 20-23 | Retransmission limits, R2 configurable and large for SYNs | yes  | other |
 * | MUST 24-29 | Keep-alives, off by default, two hours apart              | opt  | yes   |
 * | MUST 30-33 | Urgent mechanism, its notification and its pending size   | yes  | yes   |
 * | MUST 34    | Sender robust against a shrinking window                  | yes  | yes   |
 * | MUST 35-37 | Zero window probed, connection kept open                  | yes  | yes   |
 * | MUST 38,39 | SWS avoidance in the sender and in the receiver           | yes  | yes   |
 * | MUST 40    | Delayed ACK below 0.5 s                                   | yes  | yes   |
 * | MUST 41,42 | Passive open independent of the other connections         | yes  | yes   |
 * | MUST 43-45 | Local address specified, or asked to the IP layer         | yes  | yes   |
 * | MUST 46    | Invalid remote address rejected                           | yes  | yes   |
 * | MUST 47    | Soft errors reported to the application                   | yes  | yes   |
 * | MUST 48,49 | Differentiated services field and TTL configurable        | yes  | yes   |
 * | MUST 50    | IP options ignored by TCP                                 | n/a  | n/a   |
 * | MUST 51-53 | IP source routes specified, saved and preferred           | yes  | yes   |
 * | MUST 54    | ICMP errors acted upon                                    | yes  | yes   |
 * | MUST 55    | ICMP Source Quench discarded                              | yes  | yes   |
 * | MUST 56    | Soft ICMP errors do not abort the connection              | yes  | yes   |
 * | MUST 57,63 | SYN towards or coming from an invalid address ignored     | yes  | yes   |
 * | MUST 58,59 | ACK segments aggregated                                   | yes  | yes   |
 * | MUST 60,61 | Data not buffered forever, PSH set on the last segment    | yes  | yes   |
 * | MUST 62    | Urgent pointer past the last octet of urgent data         | yes  | yes   |
 * | MUST 64    | Options not starting on a word boundary handled           | yes  | yes   |
 * | MUST 65    | MSS option absent from the non-SYN segments               | yes  | yes   |
 * | MUST 66    | RST and URG processed with a zero receive window          | yes  | yes   |
 * | MUST 67    | Advertised MSS bounded by the reassembly buffer           | yes  | yes   |
 * | MUST 68    | Options other than EOL and NOP carry a length             | yes  | yes   |
 * | MUST 69    | Padding after the End of Option List is zeroed            | yes  | yes   |
 *
 * Every clause of @RFC{9293} appears above. "opt" marks what an attribute
 * enables. The rows marked "other" are covered by other suites of this module: MUST 18 by
 * tcp-rtt-estimation and tcp-rto-test, MUST 19 by tcp-slow-start-test,
 * tcp-cong-avoid-test and tcp-rto-test, and MUST 20 to MUST 23 by
 * tcp-syn-connection-failed-test, which exercises giving up on a connection
 * after the retransmissions of its SYN. The row marked "n/a" binds an
 * implementation which is handed IP options, which ns-3 never generates.
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
     * Inject a segment with the given sequence numbers and window.
     *
     * @param flags The TCP flags.
     * @param seq The sequence number.
     * @param ack The acknowledgment number.
     * @param window The window to advertise.
     * @param corruptChecksum Whether to send a checksum which does not match.
     */
    void SendSegmentFull(uint8_t flags,
                         uint32_t seq,
                         uint32_t ack,
                         uint16_t window,
                         bool corruptChecksum = false);

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
TcpCraftedSegmentTestCase::SendSegmentFull(uint8_t flags,
                                           uint32_t seq,
                                           uint32_t ack,
                                           uint16_t window,
                                           bool corruptChecksum)
{
    TcpHeader tcp;
    tcp.SetSourcePort(m_proberPort);
    tcp.SetDestinationPort(m_targetPort);
    tcp.SetSequenceNumber(SequenceNumber32(seq));
    tcp.SetAckNumber(SequenceNumber32(ack));
    tcp.SetFlags(flags);
    tcp.SetWindowSize(window);

    Ptr<Packet> packet = Create<Packet>();
    if (Node::ChecksumEnabled())
    {
        // The header computes the checksum over the pseudo header of the
        // connection, which a corrupted one is built from by moving the
        // destination address one host away
        tcp.EnableChecksums();
        Ipv4Address source = m_interfaces.GetAddress(0);
        Ipv4Address destination = m_interfaces.GetAddress(1);
        tcp.InitializeChecksum(source,
                               corruptChecksum ? Ipv4Address("10.1.1.99") : destination,
                               iana::internetprotocolnumbers::TCP);
    }
    packet->AddHeader(tcp);

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
 * connection (MUST-56). The callback they reach is also the mechanism
 * reporting the soft error conditions to the application (MUST-47).
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
 * @brief Test the simultaneous open of a connection
 *
 * @RFC{9293}, Section 3.5 (MUST-10) requires the simultaneous open sequence
 * to be supported, in which both endpoints actively open the connection and
 * no listening socket is involved. The endpoint also has to keep track of
 * whether SYN-RECEIVED was reached from a passive or an active open
 * (MUST-11), which the connection succeeding on both ends shows here.
 */
class TcpSimultaneousOpenTestCase : public TestCase
{
  public:
    TcpSimultaneousOpenTestCase()
        : TestCase("A simultaneous open establishes the connection (MUST-10, MUST-11)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Connection succeeded callback.
     * @param socket The connected socket.
     */
    void Connected(Ptr<Socket> socket);

    /**
     * Connection failed callback.
     * @param socket The socket which failed to connect.
     */
    void Failed(Ptr<Socket> socket);

    uint32_t m_connected{0}; //!< Number of endpoints which reached ESTABLISHED
    uint32_t m_failed{0};    //!< Number of endpoints which gave up
};

void
TcpSimultaneousOpenTestCase::Connected(Ptr<Socket> socket)
{
    m_connected++;
}

void
TcpSimultaneousOpenTestCase::Failed(Ptr<Socket> socket)
{
    m_failed++;
}

void
TcpSimultaneousOpenTestCase::DoRun()
{
    NodeContainer nodes;
    nodes.Create(2);

    // The link needs a delay, so that the two SYN segments cross each other
    // instead of the first one arriving before the second one is sent
    SimpleNetDeviceHelper devHelper;
    devHelper.SetChannelAttribute("Delay", TimeValue(MilliSeconds(10)));
    NetDeviceContainer devices = devHelper.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t portA = 9601;
    const uint16_t portB = 9602;

    // Neither endpoint listens: both open the connection actively, towards
    // the port the other one is bound to
    Ptr<Socket> a = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    a->Bind(InetSocketAddress(Ipv4Address::GetAny(), portA));
    a->SetConnectCallback(MakeCallback(&TcpSimultaneousOpenTestCase::Connected, this),
                          MakeCallback(&TcpSimultaneousOpenTestCase::Failed, this));

    Ptr<Socket> b = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    b->Bind(InetSocketAddress(Ipv4Address::GetAny(), portB));
    b->SetConnectCallback(MakeCallback(&TcpSimultaneousOpenTestCase::Connected, this),
                          MakeCallback(&TcpSimultaneousOpenTestCase::Failed, this));

    Simulator::Schedule(Seconds(1),
                        &Socket::Connect,
                        a,
                        InetSocketAddress(interfaces.GetAddress(1), portB));
    Simulator::Schedule(Seconds(1),
                        &Socket::Connect,
                        b,
                        InetSocketAddress(interfaces.GetAddress(0), portA));

    Simulator::Stop(Seconds(20));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_failed, 0, "An endpoint of the simultaneous open gave up");
    NS_TEST_ASSERT_MSG_EQ(m_connected, 2, "The simultaneous open did not establish both ends");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that a normal close is told apart from an abort
 *
 * @RFC{9293}, Section 3.6 (MUST-12) requires the application to be informed
 * whether the connection was closed normally or was aborted, when the remote
 * side closes it with a FIN or with a RST respectively.
 */
class TcpCloseNotificationTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpCloseNotificationTestCase()
        : TcpCraftedSegmentTestCase("A normal close is told apart from an abort (MUST-12)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Normal close callback.
     * @param socket The closed socket.
     */
    void NormalClose(Ptr<Socket> socket);

    /**
     * Error close callback.
     * @param socket The aborted socket.
     */
    void ErrorClose(Ptr<Socket> socket);

    /**
     * Accept callback, which installs the close callbacks on the forked socket.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    /**
     * Close the peer socket.
     * @param socket The socket to close.
     */
    void CloseSocket(Ptr<Socket> socket);

    uint32_t m_normalCloses{0}; //!< Number of normal close notifications
    uint32_t m_errorCloses{0};  //!< Number of error close notifications
};

void
TcpCloseNotificationTestCase::NormalClose(Ptr<Socket> socket)
{
    m_normalCloses++;
}

void
TcpCloseNotificationTestCase::ErrorClose(Ptr<Socket> socket)
{
    m_errorCloses++;
}

void
TcpCloseNotificationTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->SetCloseCallbacks(MakeCallback(&TcpCloseNotificationTestCase::NormalClose, this),
                              MakeCallback(&TcpCloseNotificationTestCase::ErrorClose, this));
}

void
TcpCloseNotificationTestCase::CloseSocket(Ptr<Socket> socket)
{
    socket->Close();
}

void
TcpCloseNotificationTestCase::DoRun()
{
    // A peer closing with a FIN notifies a normal close
    {
        SetupTopology();

        m_target->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                    MakeCallback(&TcpCloseNotificationTestCase::Accepted, this));

        Ptr<Socket> peer = Socket::CreateSocket(m_nodes.Get(0), TcpSocketFactory::GetTypeId());
        peer->Bind();
        peer->Connect(InetSocketAddress(m_interfaces.GetAddress(1), m_targetPort));
        Simulator::Schedule(Seconds(2), &TcpCloseNotificationTestCase::CloseSocket, this, peer);

        Simulator::Stop(Seconds(10));
        Simulator::Run();
        Simulator::Destroy();
    }

    NS_TEST_ASSERT_MSG_EQ(m_normalCloses, 1, "A close with a FIN was not notified as normal");
    NS_TEST_ASSERT_MSG_EQ(m_errorCloses, 0, "A close with a FIN was notified as an abort");
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that a passive open is independent of the other connections
 *
 * @RFC{9293}, Section 3.9.1.1 requires a passive OPEN to create a new
 * connection record without affecting the previously created ones (MUST-41),
 * and an application to be allowed to listen on a port while a connection
 * block with the same local port is in SYN-SENT or SYN-RECEIVED (MUST-42).
 */
class TcpPassiveOpenTestCase : public TestCase
{
  public:
    TcpPassiveOpenTestCase()
        : TestCase("A passive open leaves the other connections alone (MUST-41, MUST-42)")
    {
    }

  private:
    void DoRun() override;

    /**
     * State trace sink of the connecting socket.
     * @param oldState The previous state.
     * @param newState The new state.
     */
    void PendingState(TcpSocket::TcpStates_t oldState, TcpSocket::TcpStates_t newState);

    /**
     * Accept callback of the listening socket.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    TcpSocket::TcpStates_t m_pendingState{TcpSocket::CLOSED}; //!< State of the pending connection
    uint32_t m_accepted{0};                                   //!< Number of accepted connections
};

void
TcpPassiveOpenTestCase::PendingState(TcpSocket::TcpStates_t oldState,
                                     TcpSocket::TcpStates_t newState)
{
    m_pendingState = newState;
}

void
TcpPassiveOpenTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    m_accepted++;
}

void
TcpPassiveOpenTestCase::DoRun()
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

    const uint16_t port = 9701;

    // A connection towards a host which does not answer stays in SYN-SENT
    Ptr<Socket> pending = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    pending->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    pending->TraceConnectWithoutContext("State",
                                        MakeCallback(&TcpPassiveOpenTestCase::PendingState, this));
    pending->Connect(InetSocketAddress(Ipv4Address("10.1.1.99"), port));

    NS_TEST_ASSERT_MSG_EQ(m_pendingState,
                          TcpSocket::SYN_SENT,
                          "The pending connection is not in SYN-SENT");

    // Listening on the same local port must be allowed while it is pending
    Ptr<Socket> listening = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    NS_TEST_ASSERT_MSG_EQ(listening->Bind(InetSocketAddress(Ipv4Address::GetAny(), port)),
                          0,
                          "Binding a listener to the port of a pending connection failed");
    NS_TEST_ASSERT_MSG_EQ(listening->Listen(), 0, "Listening on that port failed");
    listening->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                 MakeCallback(&TcpPassiveOpenTestCase::Accepted, this));

    // The listener serves a connection of its own
    Ptr<Socket> peer = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    peer->Bind();
    Simulator::Schedule(Seconds(1),
                        &Socket::Connect,
                        peer,
                        InetSocketAddress(interfaces.GetAddress(0), port));

    Simulator::Stop(Seconds(10));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_accepted, 1, "The listener did not accept a connection");
    NS_TEST_ASSERT_MSG_EQ(m_pendingState,
                          TcpSocket::SYN_SENT,
                          "The passive open disturbed the pending connection");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the selection of the local address of a connection
 *
 * @RFC{9293}, Section 3.9.1.1 requires the local IP address to be
 * specifiable (MUST-43), the IP layer to be asked for one when the
 * application did not specify it (MUST-44), and the specified one to be used
 * otherwise (MUST-45). The addresses are observed on the multihomed peer,
 * which the two interfaces of the local node both reach.
 */
class TcpLocalAddressTestCase : public TestCase
{
  public:
    TcpLocalAddressTestCase()
        : TestCase("The local address is specifiable or asked to IP (MUST-43 to MUST-45)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Record the source address of the segments reaching the peer.
     * @param socket The raw socket of the peer.
     */
    void ReceiveRaw(Ptr<Socket> socket);

    std::map<uint16_t, Ipv4Address> m_sources; //!< Source address seen per source port
};

void
TcpLocalAddressTestCase::ReceiveRaw(Ptr<Socket> socket)
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
    m_sources.emplace(tcp.GetSourcePort(), ipv4.GetSource());
}

void
TcpLocalAddressTestCase::DoRun()
{
    NodeContainer nodes;
    nodes.Create(2);

    // Two links, so that the local node has an address the routing protocol
    // would not select to reach the peer over the first one
    SimpleNetDeviceHelper devHelper;
    NetDeviceContainer first = devHelper.Install(nodes);
    NetDeviceContainer second = devHelper.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer firstIfs = address.Assign(first);
    address.SetBase("10.1.2.0", "255.255.255.0");
    Ipv4InterfaceContainer secondIfs = address.Assign(second);

    const uint16_t boundPort = 9801;
    const uint16_t unboundPort = 9802;

    Ptr<Socket> sniffer = Socket::CreateSocket(nodes.Get(1), Ipv4RawSocketFactory::GetTypeId());
    sniffer->SetAttribute("Protocol", UintegerValue(6));
    sniffer->Bind();
    sniffer->SetRecvCallback(MakeCallback(&TcpLocalAddressTestCase::ReceiveRaw, this));

    // Bound to the address of the second link, but connecting over the first
    Ptr<Socket> bound = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    bound->Bind(InetSocketAddress(secondIfs.GetAddress(0), boundPort));
    bound->Connect(InetSocketAddress(firstIfs.GetAddress(1), 9803));

    // Not bound to any address, so the IP layer selects one
    Ptr<Socket> unbound = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    unbound->Bind(InetSocketAddress(Ipv4Address::GetAny(), unboundPort));
    unbound->Connect(InetSocketAddress(firstIfs.GetAddress(1), 9804));

    Simulator::Stop(Seconds(5));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_sources.count(boundPort), 1, "The bound socket sent no segment");
    NS_TEST_ASSERT_MSG_EQ(m_sources[boundPort],
                          secondIfs.GetAddress(0),
                          "The specified local address was not the source of the segments");

    NS_TEST_ASSERT_MSG_EQ(m_sources.count(unboundPort), 1, "The unbound socket sent no segment");
    NS_TEST_ASSERT_MSG_EQ(m_sources[unboundPort],
                          firstIfs.GetAddress(0),
                          "The IP layer did not select the address of the outgoing interface");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the duration of the TIME-WAIT state
 *
 * @RFC{9293}, Section 3.6 (MUST-13) requires a connection closed actively to
 * linger in TIME-WAIT for twice the maximum segment lifetime. The lifetime is
 * shortened here through the MaxSegLifetime attribute, so that the test does
 * not have to run for the four minutes of its default.
 */
class TcpTimeWaitTestCase : public TestCase
{
  public:
    TcpTimeWaitTestCase()
        : TestCase("TIME-WAIT lasts twice the maximum segment lifetime (MUST-13)")
    {
    }

  private:
    void DoRun() override;

    /**
     * State trace sink of the closing socket.
     * @param oldState The previous state.
     * @param newState The new state.
     */
    void StateChanged(TcpSocket::TcpStates_t oldState, TcpSocket::TcpStates_t newState);

    /**
     * Accept callback, which closes the forked socket when the peer closes.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    /**
     * Close the socket the peer closed, so that its FIN reaches the peer.
     * @param socket The socket to close.
     */
    void PeerClosed(Ptr<Socket> socket);

    Time m_enteredTimeWait{Seconds(0)}; //!< Instant TIME-WAIT was entered
    Time m_closed{Seconds(0)};          //!< Instant CLOSED was reached
};

void
TcpTimeWaitTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->SetCloseCallbacks(MakeCallback(&TcpTimeWaitTestCase::PeerClosed, this),
                              MakeNullCallback<void, Ptr<Socket>>());
}

void
TcpTimeWaitTestCase::PeerClosed(Ptr<Socket> socket)
{
    socket->Close();
}

void
TcpTimeWaitTestCase::StateChanged(TcpSocket::TcpStates_t oldState, TcpSocket::TcpStates_t newState)
{
    if (newState == TcpSocket::TIME_WAIT)
    {
        m_enteredTimeWait = Simulator::Now();
    }
    else if (newState == TcpSocket::CLOSED && m_enteredTimeWait > Seconds(0))
    {
        m_closed = Simulator::Now();
    }
}

void
TcpTimeWaitTestCase::DoRun()
{
    const Time msl = Seconds(1);
    Config::SetDefault("ns3::TcpSocketBase::MaxSegLifetime", DoubleValue(msl.GetSeconds()));

    NodeContainer nodes;
    nodes.Create(2);

    SimpleNetDeviceHelper devHelper;
    NetDeviceContainer devices = devHelper.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t port = 9901;

    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpTimeWaitTestCase::Accepted, this));

    // The active close is the one which lingers in TIME-WAIT
    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->TraceConnectWithoutContext("State",
                                       MakeCallback(&TcpTimeWaitTestCase::StateChanged, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));
    Simulator::Schedule(Seconds(1), &Socket::Close, client);

    Simulator::Stop(Seconds(20));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_GT(m_enteredTimeWait, Seconds(0), "The socket never entered TIME-WAIT");
    NS_TEST_ASSERT_MSG_GT(m_closed, Seconds(0), "The socket never left TIME-WAIT");
    NS_TEST_ASSERT_MSG_EQ(m_closed - m_enteredTimeWait,
                          Seconds(2) * msl.GetSeconds(),
                          "TIME-WAIT did not last twice the maximum segment lifetime");

    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the Nagle algorithm can be disabled
 *
 * @RFC{9293}, Section 3.7.4 (MUST-17) requires a way for an application to
 * disable the Nagle algorithm on a connection, which the TcpNoDelay attribute
 * of TcpSocket provides: small writes issued while data is unacknowledged are
 * coalesced with it enabled, and sent as they come with it disabled.
 */
class TcpNagleTestCase : public TestCase
{
  public:
    TcpNagleTestCase()
        : TestCase("The Nagle algorithm can be disabled (MUST-17)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Run the scenario.
     * @param noDelay Whether the Nagle algorithm is disabled.
     * @return The number of data segments the receiver got.
     */
    uint32_t CountSegments(bool noDelay);

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
     * Write a small amount of data.
     * @param socket The sending socket.
     */
    void SendSmall(Ptr<Socket> socket);

    /**
     * Accept callback, tracing the accepted socket.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    uint32_t m_dataSegments{0}; //!< Data segments received
};

void
TcpNagleTestCase::SegmentReceived(Ptr<const Packet> packet,
                                  const TcpHeader& header,
                                  Ptr<const TcpSocketBase> socket)
{
    if (packet->GetSize() > 0)
    {
        m_dataSegments++;
    }
}

void
TcpNagleTestCase::SendSmall(Ptr<Socket> socket)
{
    socket->Send(Create<Packet>(10), 0);
}

void
TcpNagleTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->TraceConnectWithoutContext("Rx",
                                       MakeCallback(&TcpNagleTestCase::SegmentReceived, this));
}

uint32_t
TcpNagleTestCase::CountSegments(bool noDelay)
{
    m_dataSegments = 0;
    Config::SetDefault("ns3::TcpSocket::TcpNoDelay", BooleanValue(noDelay));

    NodeContainer nodes;
    nodes.Create(2);

    // A link delay long enough for several writes to be issued while the
    // previous data is still unacknowledged
    SimpleNetDeviceHelper devHelper;
    devHelper.SetChannelAttribute("Delay", TimeValue(MilliSeconds(50)));
    NetDeviceContainer devices = devHelper.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t port = 9902;

    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpNagleTestCase::Accepted, this));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));

    for (uint32_t i = 0; i < 5; ++i)
    {
        Simulator::Schedule(Seconds(1) + MilliSeconds(10 * i),
                            &TcpNagleTestCase::SendSmall,
                            this,
                            client);
    }

    Simulator::Stop(Seconds(10));
    Simulator::Run();
    Simulator::Destroy();
    Config::Reset();

    return m_dataSegments;
}

void
TcpNagleTestCase::DoRun()
{
    uint32_t coalesced = CountSegments(false);
    uint32_t immediate = CountSegments(true);

    NS_TEST_ASSERT_MSG_GT(coalesced, 0, "No data reached the receiver with Nagle enabled");
    NS_TEST_ASSERT_MSG_EQ(immediate, 5, "The writes were coalesced with Nagle disabled");
    NS_TEST_ASSERT_MSG_LT(coalesced,
                          immediate,
                          "Nagle did not coalesce the writes issued while data was unacknowledged");
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the delayed ACK and the aggregation of the acknowledgments
 *
 * @RFC{9293}, Section 3.8.6.3 requires an acknowledgment not to be delayed by
 * more than 0.5 s (MUST-40), and Section 3.10.7 requires the received
 * segments to be processed before the acknowledgments are sent (MUST-58),
 * with all of them processed before any acknowledgment is sent (MUST-59), so
 * that a burst of segments is not acknowledged one by one.
 */
class TcpDelayedAckTestCase : public TestCase
{
  public:
    TcpDelayedAckTestCase()
        : TestCase("Delayed and aggregated acknowledgments (MUST-40, MUST-58, MUST-59)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Trace sink of the segments the sender receives.
     * @param packet The payload.
     * @param header The TCP header.
     * @param socket The receiving socket.
     */
    void AckReceived(Ptr<const Packet> packet,
                     const TcpHeader& header,
                     Ptr<const TcpSocketBase> socket);

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
     * Send the data.
     * @param socket The sending socket.
     * @param bytes The amount of data.
     */
    void SendData(Ptr<Socket> socket, uint32_t bytes);

    Time m_lastDataSent{Seconds(0)}; //!< Instant the last data segment was sent
    Time m_ackDelay{Seconds(0)};     //!< Delay of the acknowledgment of a lone segment
    uint32_t m_dataSegments{0};      //!< Data segments sent
    uint32_t m_acks{0};              //!< Acknowledgments received
};

void
TcpDelayedAckTestCase::SegmentSent(Ptr<const Packet> packet,
                                   const TcpHeader& header,
                                   Ptr<const TcpSocketBase> socket)
{
    if (packet->GetSize() > 0)
    {
        m_dataSegments++;
        m_lastDataSent = Simulator::Now();
    }
}

void
TcpDelayedAckTestCase::AckReceived(Ptr<const Packet> packet,
                                   const TcpHeader& header,
                                   Ptr<const TcpSocketBase> socket)
{
    if (packet->GetSize() == 0 && (header.GetFlags() & TcpHeader::ACK) &&
        !(header.GetFlags() & TcpHeader::SYN) && !(header.GetFlags() & TcpHeader::FIN))
    {
        m_acks++;
        if (m_ackDelay == Seconds(0) && m_lastDataSent > Seconds(0))
        {
            m_ackDelay = Simulator::Now() - m_lastDataSent;
        }
    }
}

void
TcpDelayedAckTestCase::SendData(Ptr<Socket> socket, uint32_t bytes)
{
    socket->Send(Create<Packet>(bytes), 0);
}

void
TcpDelayedAckTestCase::DoRun()
{
    const uint32_t segmentSize = 500;
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(segmentSize));

    NodeContainer nodes;
    nodes.Create(2);

    SimpleNetDeviceHelper devHelper;
    NetDeviceContainer devices = devHelper.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t port = 9903;

    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->TraceConnectWithoutContext("Tx",
                                       MakeCallback(&TcpDelayedAckTestCase::SegmentSent, this));
    client->TraceConnectWithoutContext("Rx",
                                       MakeCallback(&TcpDelayedAckTestCase::AckReceived, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));

    // A lone segment, whose acknowledgment is the delayed one
    Simulator::Schedule(Seconds(1), &TcpDelayedAckTestCase::SendData, this, client, segmentSize);
    // A burst, which must not be acknowledged segment by segment
    Simulator::Schedule(Seconds(3),
                        &TcpDelayedAckTestCase::SendData,
                        this,
                        client,
                        segmentSize * 10);

    Simulator::Stop(Seconds(10));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_GT(m_ackDelay, Seconds(0), "The lone segment was not acknowledged");
    NS_TEST_ASSERT_MSG_LT(m_ackDelay,
                          Seconds(0.5),
                          "The acknowledgment of the lone segment was delayed by 0.5 s or more");

    NS_TEST_ASSERT_MSG_GT(m_dataSegments, 10, "The burst was not sent");
    NS_TEST_ASSERT_MSG_LT(m_acks, m_dataSegments, "The segments were acknowledged one by one");

    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the differentiated services field and the TTL are settable
 *
 * @RFC{9293}, Section 3.9.1.9 requires the application to be able to specify
 * the differentiated services field of the segments of a connection
 * (MUST-48), and their TTL to be configurable (MUST-49). The values are read
 * from the IP header of the segments reaching the peer.
 */
class TcpTosAndTtlTestCase : public TestCase
{
  public:
    TcpTosAndTtlTestCase()
        : TestCase("The differentiated services field and the TTL are settable (MUST-48, MUST-49)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Record the header fields of the segments reaching the peer.
     * @param socket The raw socket of the peer.
     */
    void ReceiveRaw(Ptr<Socket> socket);

    uint8_t m_tos{0}; //!< Differentiated services field seen on the segments
    uint8_t m_ttl{0}; //!< TTL seen on the segments
};

void
TcpTosAndTtlTestCase::ReceiveRaw(Ptr<Socket> socket)
{
    Address from;
    Ptr<Packet> packet = socket->RecvFrom(from);

    Ipv4Header ipv4;
    packet->RemoveHeader(ipv4);
    if (ipv4.GetProtocol() != 6 || m_ttl != 0)
    {
        return;
    }

    m_tos = ipv4.GetTos();
    m_ttl = ipv4.GetTtl();
}

void
TcpTosAndTtlTestCase::DoRun()
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

    const uint16_t port = 9904;
    const uint8_t tos = 0xb8; // Expedited forwarding
    const uint8_t ttl = 33;

    Ptr<Socket> sniffer = Socket::CreateSocket(nodes.Get(1), Ipv4RawSocketFactory::GetTypeId());
    sniffer->SetAttribute("Protocol", UintegerValue(6));
    sniffer->Bind();
    sniffer->SetRecvCallback(MakeCallback(&TcpTosAndTtlTestCase::ReceiveRaw, this));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->SetIpTos(tos);
    client->SetIpTtl(ttl);
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));

    Simulator::Stop(Seconds(5));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_ttl, ttl, "The TTL of the segments is not the one which was set");
    NS_TEST_ASSERT_MSG_EQ(m_tos,
                          tos,
                          "The differentiated services field is not the one which was set");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the zero window probing and the silly window avoidance
 *
 * @RFC{9293}, Section 3.8.6 requires a window which shrank to zero to be
 * probed in the standard way (MUST-35), the probing of zero windows to be
 * supported (MUST-36), the connection to stay open as long as the peer
 * answers the probes (MUST-37), and a silly window syndrome avoidance
 * algorithm in the sender (MUST-38) and in the receiver (MUST-39), which
 * keeps the receiver from advertising the tiny windows its application frees.
 */
class TcpZeroWindowProbeTestCase : public TestCase
{
  public:
    TcpZeroWindowProbeTestCase()
        : TestCase("Zero window probing and silly window avoidance (MUST-35 to MUST-39)")
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
     * Trace sink of the segments the sender receives.
     * @param packet The payload.
     * @param header The TCP header.
     * @param socket The receiving socket.
     */
    void SegmentReceived(Ptr<const Packet> packet,
                         const TcpHeader& header,
                         Ptr<const TcpSocketBase> socket);

    /**
     * Send the data.
     * @param socket The sending socket.
     * @param bytes The amount of data.
     */
    void SendData(Ptr<Socket> socket, uint32_t bytes);

    /**
     * Read a small amount of data, freeing a small part of the receive buffer.
     */
    void ReadALittle();

    /**
     * Accept callback, holding the accepted socket without reading from it.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    Ptr<Socket> m_accepted;      //!< The socket of the receiver
    uint32_t m_probes{0};        //!< Segments sent while the peer window was zero
    uint32_t m_zeroWindows{0};   //!< Zero windows advertised by the receiver
    uint32_t m_tinyWindows{0};   //!< Non-zero windows below one segment
    uint32_t m_runtSegments{0};  //!< Data segments below one segment while more data waits
    bool m_windowIsZero{false};  //!< True while the last advertised window was zero
    uint32_t m_segmentSize{500}; //!< Segment size of the connection
    uint32_t m_lastWindow{0};    //!< Last window advertised by the receiver
    TcpSocket::TcpStates_t m_state{TcpSocket::CLOSED}; //!< State of the sender

    /**
     * State trace sink of the sender.
     * @param oldState The previous state.
     * @param newState The new state.
     */
    void StateChanged(TcpSocket::TcpStates_t oldState, TcpSocket::TcpStates_t newState);
};

void
TcpZeroWindowProbeTestCase::StateChanged(TcpSocket::TcpStates_t oldState,
                                         TcpSocket::TcpStates_t newState)
{
    m_state = newState;
}

void
TcpZeroWindowProbeTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    // The application does not read, so the receive buffer fills up
    m_accepted = socket;
}

void
TcpZeroWindowProbeTestCase::SegmentSent(Ptr<const Packet> packet,
                                        const TcpHeader& header,
                                        Ptr<const TcpSocketBase> socket)
{
    if (packet->GetSize() == 0)
    {
        return;
    }

    if (m_windowIsZero)
    {
        m_probes++;
    }
    else if (m_lastWindow > 0 && m_lastWindow < m_segmentSize && packet->GetSize() < m_segmentSize)
    {
        // A segment smaller than one full one, sent into a window which
        // cannot hold a full one, is what the sender must avoid (MUST-38)
        m_runtSegments++;
    }
}

void
TcpZeroWindowProbeTestCase::SegmentReceived(Ptr<const Packet> packet,
                                            const TcpHeader& header,
                                            Ptr<const TcpSocketBase> socket)
{
    uint32_t window = header.GetWindowSize();
    m_windowIsZero = (window == 0);
    m_lastWindow = window;

    if (window == 0)
    {
        m_zeroWindows++;
    }
    else if (window < m_segmentSize)
    {
        m_tinyWindows++;
    }
}

void
TcpZeroWindowProbeTestCase::SendData(Ptr<Socket> socket, uint32_t bytes)
{
    socket->Send(Create<Packet>(bytes), 0);
}

void
TcpZeroWindowProbeTestCase::ReadALittle()
{
    if (m_accepted)
    {
        m_accepted->Recv(10, 0);
    }
}

void
TcpZeroWindowProbeTestCase::DoRun()
{
    const uint32_t segmentSize = 500;
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(segmentSize));
    // Without the timestamp option the payload is a whole segment, so that
    // the receive buffer fills up exactly and the window reaches zero
    Config::SetDefault("ns3::TcpSocketBase::Timestamp", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(segmentSize * 4));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(50000));
    Config::SetDefault("ns3::TcpSocketBase::PersistTimeout", TimeValue(Seconds(1)));

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

    const uint16_t port = 9905;

    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpZeroWindowProbeTestCase::Accepted, this));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->TraceConnectWithoutContext(
        "Tx",
        MakeCallback(&TcpZeroWindowProbeTestCase::SegmentSent, this));
    client->TraceConnectWithoutContext(
        "Rx",
        MakeCallback(&TcpZeroWindowProbeTestCase::SegmentReceived, this));
    client->TraceConnectWithoutContext(
        "State",
        MakeCallback(&TcpZeroWindowProbeTestCase::StateChanged, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));

    // More data than the receive buffer holds, so the window closes
    Simulator::Schedule(Seconds(1), &TcpZeroWindowProbeTestCase::SendData, this, client, 20000);

    // The receiving application frees a few bytes at a time, which the
    // receiver must not advertise as a usable window
    for (uint32_t i = 0; i < 20; ++i)
    {
        Simulator::Schedule(Seconds(5) + MilliSeconds(100 * i),
                            &TcpZeroWindowProbeTestCase::ReadALittle,
                            this);
    }

    Simulator::Stop(Seconds(30));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_GT(m_zeroWindows, 0, "The receiver never advertised a zero window");
    NS_TEST_ASSERT_MSG_GT(m_probes, 0, "The sender never probed the zero window");
    NS_TEST_ASSERT_MSG_EQ(m_state,
                          TcpSocket::ESTABLISHED,
                          "The connection did not stay open while the window was zero");
    NS_TEST_ASSERT_MSG_EQ(m_tinyWindows,
                          0,
                          "The receiver advertised a window smaller than one segment");
    NS_TEST_ASSERT_MSG_EQ(m_runtSegments,
                          0,
                          "The sender sent a runt segment into a window below one segment");

    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that buffered data is sent and flagged with PSH
 *
 * @RFC{9293}, Section 3.9.1.3 requires an implementation which does not offer
 * the PUSH flag on the SEND call not to buffer data indefinitely (MUST-60),
 * and to set the PSH bit on the last buffered segment, the one after which
 * there is no more queued data (MUST-61).
 */
class TcpPushFlagTestCase : public TestCase
{
  public:
    TcpPushFlagTestCase()
        : TestCase("The last buffered segment is sent and carries PSH (MUST-60, MUST-61)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Trace sink of the segments the receiver gets.
     * @param packet The payload.
     * @param header The TCP header.
     * @param socket The receiving socket.
     */
    void SegmentReceived(Ptr<const Packet> packet,
                         const TcpHeader& header,
                         Ptr<const TcpSocketBase> socket);

    /**
     * Accept callback, tracing the accepted socket.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    /**
     * Send the data.
     * @param socket The sending socket.
     * @param bytes The amount of data.
     */
    void SendData(Ptr<Socket> socket, uint32_t bytes);

    uint32_t m_dataSegments{0};    //!< Data segments received
    uint32_t m_pushed{0};          //!< Data segments carrying PSH
    uint32_t m_bytes{0};           //!< Payload bytes received
    bool m_lastCarriedPush{false}; //!< True if the last data segment carried PSH
};

void
TcpPushFlagTestCase::SegmentReceived(Ptr<const Packet> packet,
                                     const TcpHeader& header,
                                     Ptr<const TcpSocketBase> socket)
{
    if (packet->GetSize() == 0)
    {
        return;
    }

    m_dataSegments++;
    m_bytes += packet->GetSize();
    m_lastCarriedPush = (header.GetFlags() & TcpHeader::PSH) != 0;
    if (m_lastCarriedPush)
    {
        m_pushed++;
    }
}

void
TcpPushFlagTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->TraceConnectWithoutContext("Rx",
                                       MakeCallback(&TcpPushFlagTestCase::SegmentReceived, this));
}

void
TcpPushFlagTestCase::SendData(Ptr<Socket> socket, uint32_t bytes)
{
    socket->Send(Create<Packet>(bytes), 0);
}

void
TcpPushFlagTestCase::DoRun()
{
    const uint32_t segmentSize = 500;
    const uint32_t bytes = segmentSize * 3;
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(segmentSize));

    NodeContainer nodes;
    nodes.Create(2);

    SimpleNetDeviceHelper devHelper;
    NetDeviceContainer devices = devHelper.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t port = 9906;

    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpPushFlagTestCase::Accepted, this));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));
    Simulator::Schedule(Seconds(1), &TcpPushFlagTestCase::SendData, this, client, bytes);

    Simulator::Stop(Seconds(10));
    Simulator::Run();

    // The data is not held in the buffer waiting for more (MUST-60)
    NS_TEST_ASSERT_MSG_EQ(m_bytes, bytes, "The data written was not sent");
    // The segment after which the buffer is empty announces it (MUST-61)
    NS_TEST_ASSERT_MSG_EQ(m_lastCarriedPush, true, "The last segment did not carry PSH");
    NS_TEST_ASSERT_MSG_LT(m_pushed, m_dataSegments, "Every segment carried PSH");

    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the advertised MSS fits what can be received
 *
 * @RFC{9293}, Section 3.7.1 (MUST-67) bounds the MSS value sent in an MSS
 * option by MMS_R - 20, the largest transport message which can be received
 * and reassembled, which the MTU of the interface gives here.
 */
class TcpAdvertisedMssBoundTestCase : public TestCase
{
  public:
    TcpAdvertisedMssBoundTestCase()
        : TestCase("The advertised MSS fits the interface MTU (MUST-67)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Record the MSS option of the segments reaching the peer.
     * @param socket The raw socket of the peer.
     */
    void ReceiveRaw(Ptr<Socket> socket);

    uint16_t m_advertisedMss{0}; //!< MSS advertised by the local node
};

void
TcpAdvertisedMssBoundTestCase::ReceiveRaw(Ptr<Socket> socket)
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
    if (tcp.HasOption(TcpOption::MSS))
    {
        Ptr<const TcpOptionMSS> mss =
            DynamicCast<const TcpOptionMSS>(tcp.GetOption(TcpOption::MSS));
        m_advertisedMss = mss->GetMSS();
    }
}

void
TcpAdvertisedMssBoundTestCase::DoRun()
{
    const uint16_t mtu = 600;
    // A segment size larger than the interface can carry
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1400));

    NodeContainer nodes;
    nodes.Create(2);

    SimpleNetDeviceHelper devHelper;
    NetDeviceContainer devices = devHelper.Install(nodes);
    for (uint32_t i = 0; i < devices.GetN(); ++i)
    {
        devices.Get(i)->SetMtu(mtu);
    }

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    Ptr<Socket> sniffer = Socket::CreateSocket(nodes.Get(1), Ipv4RawSocketFactory::GetTypeId());
    sniffer->SetAttribute("Protocol", UintegerValue(6));
    sniffer->Bind();
    sniffer->SetRecvCallback(MakeCallback(&TcpAdvertisedMssBoundTestCase::ReceiveRaw, this));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), 9907));

    Simulator::Stop(Seconds(5));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_GT(m_advertisedMss, 0, "No MSS option was advertised");
    NS_TEST_ASSERT_MSG_LT_OR_EQ(m_advertisedMss,
                                mtu - 40,
                                "The advertised MSS does not fit the interface MTU");

    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that a SYN towards a broadcast or multicast address is dropped
 *
 * @RFC{9293}, Section 3.10.7.2 requires an incoming SYN addressed to a
 * broadcast or a multicast address to be discarded (MUST-57), and a SYN with
 * an invalid source address to be ignored either by TCP or by the IP layer
 * (MUST-63).
 */
class TcpBroadcastSynTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpBroadcastSynTestCase()
        : TcpCraftedSegmentTestCase("A SYN towards a broadcast address is dropped (MUST-57)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Inject a SYN towards the given destination.
     * @param destination The destination address.
     */
    void SendSynTo(Ipv4Address destination);
};

void
TcpBroadcastSynTestCase::SendSynTo(Ipv4Address destination)
{
    const uint8_t segment[20] = {
        static_cast<uint8_t>(m_proberPort >> 8),
        static_cast<uint8_t>(m_proberPort & 0xff),
        static_cast<uint8_t>(m_targetPort >> 8),
        static_cast<uint8_t>(m_targetPort & 0xff),
        0x00,
        0x00,
        0x00,
        0x01, // sequence number
        0x00,
        0x00,
        0x00,
        0x00, // acknowledgment number
        0x50, // data offset
        TcpHeader::SYN,
        0x10,
        0x00, // window size
        0x00,
        0x00, // checksum
        0x00,
        0x00 // urgent pointer
    };

    Ptr<Packet> packet = Create<Packet>(segment, sizeof(segment));
    m_prober->SendTo(packet, 0, InetSocketAddress(destination, 0));
}

void
TcpBroadcastSynTestCase::DoRun()
{
    SetupTopology();

    // The subnet broadcast address reaches the listening socket of the target
    Simulator::Schedule(Seconds(1),
                        &TcpBroadcastSynTestCase::SendSynTo,
                        this,
                        Ipv4Address("10.1.1.255"));

    Simulator::Stop(Seconds(5));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(GetReplies().size(), 0, "A SYN towards a broadcast address was answered");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the window is treated as an unsigned number
 *
 * @RFC{9293}, Section 3.1 (MUST-1) requires the window size to be treated as
 * an unsigned number, since a window above 32767 would otherwise look like a
 * negative one and stall the connection. The window scale option is disabled
 * here, so that the window travels in the 16 bit field of the header.
 */
class TcpUnsignedWindowTestCase : public TestCase
{
  public:
    TcpUnsignedWindowTestCase()
        : TestCase("The window is treated as an unsigned number (MUST-1)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Trace sink of the segments the sender receives.
     * @param packet The payload.
     * @param header The TCP header.
     * @param socket The receiving socket.
     */
    void AckReceived(Ptr<const Packet> packet,
                     const TcpHeader& header,
                     Ptr<const TcpSocketBase> socket);

    /**
     * Trace sink of the data the receiver gets.
     * @param packet The payload.
     * @param header The TCP header.
     * @param socket The receiving socket.
     */
    void DataReceived(Ptr<const Packet> packet,
                      const TcpHeader& header,
                      Ptr<const TcpSocketBase> socket);

    /**
     * Accept callback, tracing the accepted socket.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    /**
     * Send the data.
     * @param socket The sending socket.
     * @param bytes The amount of data.
     */
    void SendData(Ptr<Socket> socket, uint32_t bytes);

    uint32_t m_largestWindow{0}; //!< Largest window the sender was told about
    uint32_t m_bytesReceived{0}; //!< Payload bytes which reached the receiver
};

void
TcpUnsignedWindowTestCase::AckReceived(Ptr<const Packet> packet,
                                       const TcpHeader& header,
                                       Ptr<const TcpSocketBase> socket)
{
    m_largestWindow = std::max<uint32_t>(m_largestWindow, header.GetWindowSize());
}

void
TcpUnsignedWindowTestCase::DataReceived(Ptr<const Packet> packet,
                                        const TcpHeader& header,
                                        Ptr<const TcpSocketBase> socket)
{
    m_bytesReceived += packet->GetSize();
}

void
TcpUnsignedWindowTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->TraceConnectWithoutContext(
        "Rx",
        MakeCallback(&TcpUnsignedWindowTestCase::DataReceived, this));
}

void
TcpUnsignedWindowTestCase::SendData(Ptr<Socket> socket, uint32_t bytes)
{
    socket->Send(Create<Packet>(bytes), 0);
}

void
TcpUnsignedWindowTestCase::DoRun()
{
    // A receive buffer whose free space does not fit in a signed 16 bit value
    const uint32_t bytes = 60000;
    Config::SetDefault("ns3::TcpSocketBase::WindowScaling", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(bytes));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(bytes));

    NodeContainer nodes;
    nodes.Create(2);

    SimpleNetDeviceHelper devHelper;
    devHelper.SetChannelAttribute("Delay", TimeValue(MilliSeconds(5)));
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
                              MakeCallback(&TcpUnsignedWindowTestCase::Accepted, this));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->TraceConnectWithoutContext("Rx",
                                       MakeCallback(&TcpUnsignedWindowTestCase::AckReceived, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));
    Simulator::Schedule(Seconds(1), &TcpUnsignedWindowTestCase::SendData, this, client, bytes);

    Simulator::Stop(Seconds(30));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_GT(m_largestWindow,
                          32767,
                          "The receiver never advertised a window above the signed 16 bit range");
    // A window read as a negative number would stall the transfer
    NS_TEST_ASSERT_MSG_EQ(m_bytesReceived, bytes, "The transfer did not complete");

    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the checksum is generated and checked
 *
 * @RFC{9293}, Section 3.1 requires the sender to generate the checksum
 * (MUST-2) and the receiver to check it (MUST-3). ns-3 computes checksums
 * only when the ChecksumEnabled global value asks for it, which this test
 * turns on.
 */
class TcpChecksumTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpChecksumTestCase()
        : TcpCraftedSegmentTestCase("The checksum is generated and checked (MUST-2, MUST-3)")
    {
    }

  private:
    void DoRun() override;
};

void
TcpChecksumTestCase::DoRun()
{
    GlobalValue::Bind("ChecksumEnabled", BooleanValue(true));

    SetupTopology();

    // A SYN whose checksum does not match must be dropped without an answer
    Simulator::Schedule(Seconds(1),
                        &TcpChecksumTestCase::SendSegmentFull,
                        this,
                        TcpHeader::SYN,
                        1,
                        0,
                        4096,
                        true);
    // The same SYN, with the checksum the receiver expects, is answered
    Simulator::Schedule(Seconds(3),
                        &TcpChecksumTestCase::SendSegmentFull,
                        this,
                        TcpHeader::SYN,
                        1,
                        0,
                        4096,
                        false);

    Simulator::Stop(Seconds(6));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(CountReplies(TcpHeader::SYN | TcpHeader::ACK),
                          1,
                          "The SYN with a valid checksum was not answered exactly once");
    NS_TEST_ASSERT_MSG_EQ(CountReplies(TcpHeader::RST),
                          0,
                          "A segment with a wrong checksum was answered with a reset");

    Simulator::Destroy();
    GlobalValue::Bind("ChecksumEnabled", BooleanValue(false));
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the sender withstands a window which shrinks
 *
 * @RFC{9293}, Section 3.8.6 asks a receiver not to shrink the window
 * (SHLD-14), but requires a sender to be robust against it (MUST-34), which
 * can make its usable window negative. The connection is driven from a raw
 * socket here, so that the right edge can be moved backwards, which the ns-3
 * receiver would not do.
 */
class TcpShrinkingWindowTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpShrinkingWindowTestCase()
        : TcpCraftedSegmentTestCase("The sender withstands a window which shrinks (MUST-34)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Complete the handshake and open a large window.
     */
    void CompleteHandshake();

    /**
     * Acknowledge nothing new while advertising a much smaller window.
     */
    void ShrinkWindow();

    /**
     * Accept callback, which sends data to the peer driving the connection.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    SequenceNumber32 m_targetIsn{0}; //!< Initial sequence number of the target
    uint32_t m_largeWindow{6000};    //!< Window advertised during the handshake
    uint32_t m_smallWindow{500};     //!< Window advertised afterwards
    SequenceNumber32 m_rightEdge{0}; //!< Right edge the large window opened
    uint32_t m_beyondRightEdge{0};   //!< Data segments sent beyond that edge
};

void
TcpShrinkingWindowTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->Send(Create<Packet>(20000), 0);
}

void
TcpShrinkingWindowTestCase::CompleteHandshake()
{
    NS_ASSERT_MSG(!GetReplies().empty(), "The SYN was not answered");
    m_targetIsn = GetReplies().front().GetSequenceNumber();

    // The ACK completing the handshake, opening a large window
    SendSegmentFull(TcpHeader::ACK, 2, (m_targetIsn + 1).GetValue(), m_largeWindow);
    m_rightEdge = m_targetIsn + 1 + m_largeWindow;
}

void
TcpShrinkingWindowTestCase::ShrinkWindow()
{
    // The same acknowledgment, with a window which moves the right edge back
    SendSegmentFull(TcpHeader::ACK, 2, (m_targetIsn + 1).GetValue(), m_smallWindow);
}

void
TcpShrinkingWindowTestCase::DoRun()
{
    SetupTopology();
    m_target->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                MakeCallback(&TcpShrinkingWindowTestCase::Accepted, this));

    Simulator::Schedule(Seconds(1),
                        &TcpShrinkingWindowTestCase::SendSegmentFull,
                        this,
                        TcpHeader::SYN,
                        1,
                        0,
                        m_largeWindow,
                        false);
    Simulator::Schedule(Seconds(2), &TcpShrinkingWindowTestCase::CompleteHandshake, this);
    Simulator::Schedule(Seconds(3), &TcpShrinkingWindowTestCase::ShrinkWindow, this);

    Simulator::Stop(Seconds(20));
    Simulator::Run();

    // The sender keeps the connection: shrinking the window is not an error
    NS_TEST_ASSERT_MSG_EQ(CountReplies(TcpHeader::RST),
                          0,
                          "The sender reset the connection when the window shrank");

    // and it does not send new data beyond the edge it was told about
    for (const auto& reply : GetReplies())
    {
        if (reply.GetSequenceNumber() > m_rightEdge)
        {
            m_beyondRightEdge++;
        }
    }
    NS_TEST_ASSERT_MSG_EQ(m_beyondRightEdge, 0, "The sender sent beyond the right edge");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the keep-alives of an idle connection
 *
 * @RFC{9293}, Section 3.8.4 requires the keep-alives to be turned on or off
 * for each connection (MUST-24) and to default to off (MUST-25), to be sent
 * only when no data is outstanding and nothing was received for the
 * configured interval (MUST-26), which must be configurable (MUST-27) and
 * default to no less than two hours (MUST-28), and requires a single
 * unanswered probe not to be read as a dead connection (MUST-29).
 */
class TcpKeepAliveTestCase : public TestCase
{
  public:
    TcpKeepAliveTestCase()
        : TestCase("Keep-alives of an idle connection (MUST-24 to MUST-29)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Run an idle connection and count the keep-alives it carries.
     *
     * @param keepAlive Whether the keep-alives are enabled.
     * @param busy Whether the connection keeps carrying data.
     * @return The number of keep-alives the peer received.
     */
    uint32_t CountKeepAlives(bool keepAlive, bool busy);

    /**
     * Trace sink of the segments the peer receives.
     * @param packet The payload.
     * @param header The TCP header.
     * @param socket The receiving socket.
     */
    void SegmentReceived(Ptr<const Packet> packet,
                         const TcpHeader& header,
                         Ptr<const TcpSocketBase> socket);

    /**
     * Accept callback, tracing the accepted socket.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    /**
     * Send a little data, keeping the connection busy.
     * @param socket The sending socket.
     */
    void SendData(Ptr<Socket> socket);

    /**
     * Run a connection whose peer stops answering, and record when it is given up on.
     *
     * @param retries The number of unanswered keep-alives to allow.
     * @return The instant the connection was closed, or zero if it stayed open.
     */
    Time RunUnansweredKeepAlives(uint32_t retries);

    /**
     * State trace sink of the probing socket.
     * @param oldState The previous state.
     * @param newState The new state.
     */
    void StateChanged(TcpSocket::TcpStates_t oldState, TcpSocket::TcpStates_t newState);

    /**
     * Stop the peer from answering, by dropping everything it receives.
     * @param device The device of the peer.
     */
    void BreakThePath(Ptr<NetDevice> device);

    SequenceNumber32 m_highestData{0}; //!< Highest sequence number of the data received
    uint32_t m_keepAlives{0};          //!< Segments repeating an acknowledged sequence number
    Time m_closed{Seconds(0)};         //!< Instant the probing socket gave up
    TcpSocket::TcpStates_t m_state{TcpSocket::CLOSED}; //!< State of the probing socket
};

void
TcpKeepAliveTestCase::StateChanged(TcpSocket::TcpStates_t oldState, TcpSocket::TcpStates_t newState)
{
    m_state = newState;
    if (newState == TcpSocket::CLOSED && m_closed == Seconds(0))
    {
        m_closed = Simulator::Now();
    }
}

void
TcpKeepAliveTestCase::BreakThePath(Ptr<NetDevice> device)
{
    Ptr<RateErrorModel> error = CreateObject<RateErrorModel>();
    error->SetAttribute("ErrorRate", DoubleValue(1.0));
    error->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
    device->SetAttribute("ReceiveErrorModel", PointerValue(error));
}

Time
TcpKeepAliveTestCase::RunUnansweredKeepAlives(uint32_t retries)
{
    m_closed = Seconds(0);
    m_state = TcpSocket::CLOSED;

    NodeContainer nodes;
    nodes.Create(2);

    SimpleNetDeviceHelper devHelper;
    NetDeviceContainer devices = devHelper.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t port = 9910;

    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->SetAttribute("KeepAlive", BooleanValue(true));
    client->SetAttribute("KeepAliveTime", TimeValue(Seconds(2)));
    client->SetAttribute("KeepAliveInterval", TimeValue(Seconds(1)));
    client->SetAttribute("KeepAliveRetries", UintegerValue(retries));
    client->TraceConnectWithoutContext("State",
                                       MakeCallback(&TcpKeepAliveTestCase::StateChanged, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));

    // The peer stops answering once the connection is up
    Simulator::Schedule(Seconds(1), &TcpKeepAliveTestCase::BreakThePath, this, devices.Get(1));

    Simulator::Stop(Seconds(60));
    Simulator::Run();
    Simulator::Destroy();

    return m_closed;
}

void
TcpKeepAliveTestCase::SegmentReceived(Ptr<const Packet> packet,
                                      const TcpHeader& header,
                                      Ptr<const TcpSocketBase> socket)
{
    if (packet->GetSize() > 0)
    {
        m_highestData = std::max(m_highestData, header.GetSequenceNumber() + packet->GetSize());
        return;
    }

    // A keep-alive holds no data and repeats the sequence number of the last
    // octet the receiver acknowledged
    if (m_highestData > SequenceNumber32(0) && header.GetSequenceNumber() < m_highestData)
    {
        m_keepAlives++;
    }
}

void
TcpKeepAliveTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->TraceConnectWithoutContext("Rx",
                                       MakeCallback(&TcpKeepAliveTestCase::SegmentReceived, this));
}

void
TcpKeepAliveTestCase::SendData(Ptr<Socket> socket)
{
    socket->Send(Create<Packet>(100), 0);
}

uint32_t
TcpKeepAliveTestCase::CountKeepAlives(bool keepAlive, bool busy)
{
    m_keepAlives = 0;
    m_highestData = SequenceNumber32(0);

    NodeContainer nodes;
    nodes.Create(2);

    SimpleNetDeviceHelper devHelper;
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
                              MakeCallback(&TcpKeepAliveTestCase::Accepted, this));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    // Turned on for this connection alone, with an interval short enough for
    // the test to run (MUST-24 and MUST-27)
    client->SetAttribute("KeepAlive", BooleanValue(keepAlive));
    client->SetAttribute("KeepAliveTime", TimeValue(Seconds(2)));
    client->SetAttribute("KeepAliveInterval", TimeValue(Seconds(1)));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));

    Simulator::Schedule(Seconds(1), &TcpKeepAliveTestCase::SendData, this, client);
    if (busy)
    {
        // Data flowing all along, so the connection is never idle
        for (uint32_t i = 0; i < 20; ++i)
        {
            Simulator::Schedule(Seconds(2) + MilliSeconds(500 * i),
                                &TcpKeepAliveTestCase::SendData,
                                this,
                                client);
        }
    }

    Simulator::Stop(Seconds(13));
    Simulator::Run();
    Simulator::Destroy();

    return m_keepAlives;
}

void
TcpKeepAliveTestCase::DoRun()
{
    // The defaults leave the keep-alives off (MUST-25), no less than two
    // hours apart (MUST-28)
    TypeId::AttributeInformation info;
    NS_TEST_ASSERT_MSG_EQ(
        TypeId::LookupByName("ns3::TcpSocket").LookupAttributeByName("KeepAlive", &info),
        true,
        "The keep-alives cannot be turned on or off");
    NS_TEST_ASSERT_MSG_EQ(DynamicCast<const BooleanValue>(info.initialValue)->Get(),
                          false,
                          "The keep-alives are enabled by default");

    NS_TEST_ASSERT_MSG_EQ(
        TypeId::LookupByName("ns3::TcpSocket").LookupAttributeByName("KeepAliveTime", &info),
        true,
        "The idle time before a keep-alive is not configurable");
    NS_TEST_ASSERT_MSG_GT_OR_EQ(DynamicCast<const TimeValue>(info.initialValue)->Get(),
                                Hours(2),
                                "The keep-alives default to less than two hours apart");

    uint32_t idle = CountKeepAlives(true, false);
    uint32_t idleWithoutKeepAlive = CountKeepAlives(false, false);
    uint32_t busy = CountKeepAlives(true, true);

    NS_TEST_ASSERT_MSG_GT(idle, 1, "An idle connection was not kept alive");
    // The connection outlives the probes it sends, since a peer which answers
    // none of them is only given up on after the whole series (MUST-29)
    NS_TEST_ASSERT_MSG_EQ(idleWithoutKeepAlive,
                          0,
                          "A connection was kept alive with the keep-alives disabled");
    NS_TEST_ASSERT_MSG_EQ(busy, 0, "A connection carrying data was probed");

    // A peer which answers nothing is given up on, but only after the whole
    // series of keep-alives, never upon a single one (MUST-29)
    Time fewRetries = RunUnansweredKeepAlives(1);
    Time manyRetries = RunUnansweredKeepAlives(5);

    NS_TEST_ASSERT_MSG_GT(fewRetries, Seconds(0), "The connection outlived every keep-alive");
    NS_TEST_ASSERT_MSG_GT(manyRetries, Seconds(0), "The connection outlived every keep-alive");

    // The first keep-alive leaves after the two idle seconds, and the next
    // ones follow it every second: a connection dropped upon the first one
    // would be gone by then
    NS_TEST_ASSERT_MSG_GT(fewRetries,
                          Seconds(3),
                          "A single unanswered keep-alive was read as a dead peer");
    // and allowing more of them keeps the connection alive for longer
    NS_TEST_ASSERT_MSG_GT(manyRetries,
                          fewRetries,
                          "The number of keep-alives allowed did not matter");
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the source route of a connection
 *
 * @RFC{9293}, Section 3.9.2.1 requires an application to be able to specify a
 * source route when it opens a connection (MUST-51), that route to take
 * precedence over the one a received datagram carries (MUST-52), and a
 * connection opened passively to answer along the route recorded by the
 * datagram which opened it (MUST-53).
 *
 * The three nodes are lined up, and the two ends are given a route to each
 * other through the one in the middle.
 */
class TcpSourceRouteTestCase : public TestCase
{
  public:
    TcpSourceRouteTestCase()
        : TestCase("The segments follow the source route (MUST-51 to MUST-53)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Record the headers of the datagrams reaching the peer.
     * @param socket The raw socket of the peer.
     */
    void ReceiveRaw(Ptr<Socket> socket);

    /**
     * Accept callback, which keeps the accepted socket and answers.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    uint32_t m_routedSegments{0};         //!< Segments which carried a source route
    uint32_t m_segments{0};               //!< Segments which reached the peer
    std::vector<Ipv4Address> m_seenRoute; //!< Route carried by the first of them
    Ptr<Socket> m_accepted;               //!< The socket of the passive open
};

void
TcpSourceRouteTestCase::ReceiveRaw(Ptr<Socket> socket)
{
    Address from;
    Ptr<Packet> packet = socket->RecvFrom(from);

    Ipv4Header ipv4;
    packet->RemoveHeader(ipv4);
    if (ipv4.GetProtocol() != 6)
    {
        return;
    }

    m_segments++;
    if (ipv4.HasLooseSourceRoute())
    {
        m_routedSegments++;
        if (m_seenRoute.empty())
        {
            m_seenRoute = ipv4.GetLooseSourceRoute();
        }
    }
}

void
TcpSourceRouteTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    m_accepted = socket;
}

void
TcpSourceRouteTestCase::DoRun()
{
    NodeContainer nodes;
    nodes.Create(3);

    NodeContainer left(nodes.Get(0), nodes.Get(1));
    NodeContainer right(nodes.Get(1), nodes.Get(2));

    SimpleNetDeviceHelper devHelper;
    NetDeviceContainer leftDevices = devHelper.Install(left);
    NetDeviceContainer rightDevices = devHelper.Install(right);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer leftIfs = address.Assign(leftDevices);
    address.SetBase("10.1.2.0", "255.255.255.0");
    Ipv4InterfaceContainer rightIfs = address.Assign(rightDevices);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    const uint16_t port = 9911;

    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(2), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpSourceRouteTestCase::Accepted, this));

    Ptr<Socket> sniffer = Socket::CreateSocket(nodes.Get(2), Ipv4RawSocketFactory::GetTypeId());
    sniffer->SetAttribute("Protocol", UintegerValue(6));
    sniffer->Bind();
    sniffer->SetRecvCallback(MakeCallback(&TcpSourceRouteTestCase::ReceiveRaw, this));

    // The route the application asks for: through the node in the middle,
    // then on to the peer
    std::vector<Ipv4Address> route = {leftIfs.GetAddress(1), rightIfs.GetAddress(1)};

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    Ptr<TcpSocketBase> tcpClient = DynamicCast<TcpSocketBase>(client);
    client->Bind();
    tcpClient->SetIpv4SourceRoute(route);
    client->Connect(InetSocketAddress(rightIfs.GetAddress(1), port));

    Simulator::Stop(Seconds(10));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_GT(m_segments, 0, "No segment reached the peer");
    NS_TEST_ASSERT_MSG_EQ(m_routedSegments,
                          m_segments,
                          "A segment of the connection carried no source route");

    // Every hop writes the address the datagram reached it at in the slot it
    // took the next hop from, so the option holds the way back when it arrives
    NS_TEST_ASSERT_MSG_EQ(m_seenRoute.size(), route.size(), "The route was not carried whole");
    NS_TEST_ASSERT_MSG_EQ(m_seenRoute.front(),
                          leftIfs.GetAddress(1),
                          "The hop in the middle did not record the address it was reached at");

    // The connection opened passively answers along the way back (MUST-53)
    NS_TEST_ASSERT_MSG_NE(m_accepted, nullptr, "The connection was not established");
    Ptr<TcpSocketBase> tcpServer = DynamicCast<TcpSocketBase>(m_accepted);
    NS_TEST_ASSERT_MSG_EQ(tcpServer->GetIpv4SourceRoute().empty(),
                          false,
                          "The passive open saved no return route");

    // and a route the application specifies takes precedence over it (MUST-52)
    std::vector<Ipv4Address> own = {rightIfs.GetAddress(0), leftIfs.GetAddress(0)};
    tcpServer->SetIpv4SourceRoute(own);
    NS_TEST_ASSERT_MSG_EQ(tcpServer->GetIpv4SourceRoute().front(),
                          own.front(),
                          "The route of the application did not take precedence");

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
        AddTestCase(new TcpLegacyOptionsTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpUnknownOptionTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpReservedFlagsTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpResetAckTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpAdvertisedMssTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpSimultaneousOpenTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpCloseNotificationTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpPassiveOpenTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpLocalAddressTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpTimeWaitTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpNagleTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpDelayedAckTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpTosAndTtlTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpZeroWindowProbeTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpPushFlagTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpAdvertisedMssBoundTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpBroadcastSynTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpUnsignedWindowTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpChecksumTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpShrinkingWindowTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpKeepAliveTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpSourceRouteTestCase(), TestCase::Duration::QUICK);
    }
};

static TcpRfc9293TestSuite g_tcpRfc9293TestSuite; //!< Static variable for test initialization
