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
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/ipv4-raw-socket-factory.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/ipv6-address-helper.h"
#include "ns3/ipv6-l3-protocol.h"
#include "ns3/ipv6-raw-socket-factory.h"
#include "ns3/log.h"
#include "ns3/map-scheduler.h"
#include "ns3/node-container.h"
#include "ns3/node.h"
#include "ns3/nstime.h"
#include "ns3/object-factory.h"
#include "ns3/pointer.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/simple-channel.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/simple-net-device.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-l4-protocol.h"
#include "ns3/tcp-option-rfc793.h"
#include "ns3/tcp-option-winscale.h"
#include "ns3/tcp-option.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/tcp-tx-buffer.h"
#include "ns3/test.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/uinteger.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <tuple>
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
     * @param removeProberTcp Whether to remove the TCP stack of the probing
     *                        node, which would otherwise reset the
     *                        connections the raw socket drives; keep it for
     *                        the tests running a real socket on that node.
     */
    void SetupTopology(bool listen = true, bool removeProberTcp = true);

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
     * @param urgentPointer The urgent pointer.
     * @param payload The number of data bytes to carry.
     */
    void SendSegmentFull(uint8_t flags,
                         uint32_t seq,
                         uint32_t ack,
                         uint16_t window,
                         bool corruptChecksum = false,
                         uint16_t urgentPointer = 0,
                         uint32_t payload = 0);

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
TcpCraftedSegmentTestCase::SetupTopology(bool listen, bool removeProberTcp)
{
    // Start from scratch, so that a test case can run several topologies
    m_nodes = NodeContainer();
    m_interfaces = Ipv4InterfaceContainer();
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

    if (removeProberTcp)
    {
        // The TCP stack of the probing node would reset the connections the
        // raw socket drives, since it has no endpoint for the replies: take
        // it out of the way, so that the probes alone speak for the node
        Ptr<Ipv4L3Protocol> proberIpv4 = m_nodes.Get(0)->GetObject<Ipv4L3Protocol>();
        proberIpv4->Remove(m_nodes.Get(0)->GetObject<TcpL4Protocol>());
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
                                           bool corruptChecksum,
                                           uint16_t urgentPointer,
                                           uint32_t payload)
{
    TcpHeader tcp;
    tcp.SetSourcePort(m_proberPort);
    tcp.SetDestinationPort(m_targetPort);
    tcp.SetSequenceNumber(SequenceNumber32(seq));
    tcp.SetAckNumber(SequenceNumber32(ack));
    tcp.SetFlags(flags);
    tcp.SetWindowSize(window);
    tcp.SetUrgentPointer(urgentPointer);

    Ptr<Packet> packet = Create<Packet>(payload);
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
    uint32_t m_icmpCount{0}; //!< Number of ICMP messages delivered to the socket
    uint8_t m_lastType{0};   //!< Type of the last ICMP message delivered to the socket
    NodeContainer m_nodes;   //!< The nodes hosting the sockets
    Ptr<Socket> m_socket;    //!< The socket under test
    Ptr<Node> m_senderNode;  //!< The node hosting the socket under test
    uint16_t m_localPort{0}; //!< Local port of the socket under test
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
 *
 * @brief Check that a passive opener with a large initial sequence number
 * keeps track of the acknowledgments it receives.
 *
 * The highest acknowledgment number received is compared with the ones of
 * the incoming segments in the wrap aware sequence number arithmetic. The
 * passive opener learns it from the ACK completing the handshake: if it
 * stayed at zero, an initial sequence number in the upper half of the space
 * would never update it, so that the window updates would be ignored and a
 * fast retransmission would start from sequence number zero.
 *
 * The clock driven initial sequence numbers are enabled and the run number
 * of the random number generator is changed until the target picks one in
 * the upper half; then the target sends data, the prober reports the first
 * segment lost with three duplicate ACKs, and the retransmission must carry
 * the sequence number of that segment.
 */
class TcpPassiveHighIsnTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpPassiveHighIsnTestCase()
        : TcpCraftedSegmentTestCase(
              "A passive opener with a high ISN retransmits the right segment")
    {
    }

  private:
    void DoRun() override;

    /**
     * Accept callback of the target.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    /**
     * Make the accepted socket send data.
     */
    void SendData();

    Ptr<Socket> m_accepted; //!< The accepted socket
};

void
TcpPassiveHighIsnTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    m_accepted = socket;
}

void
TcpPassiveHighIsnTestCase::SendData()
{
    NS_ABORT_MSG_IF(!m_accepted, "The connection was not accepted");
    m_accepted->Send(Create<Packet>(5000));
}

void
TcpPassiveHighIsnTestCase::DoRun()
{
    Config::SetDefault("ns3::TcpL4Protocol::ClockDrivenIsn", BooleanValue(true));

    const uint32_t upperHalf = 1U << 31;
    SequenceNumber32 isn;
    bool found = false;
    for (uint64_t run = 1; run <= 64 && !found; ++run)
    {
        RngSeedManager::SetRun(run);
        SetupTopology();
        m_target->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                    MakeCallback(&TcpPassiveHighIsnTestCase::Accepted, this));
        Simulator::Schedule(Seconds(1),
                            &TcpPassiveHighIsnTestCase::SendSegmentFull,
                            this,
                            TcpHeader::SYN,
                            1U,
                            0U,
                            static_cast<uint16_t>(65535),
                            false,
                            static_cast<uint16_t>(0),
                            0U);
        Simulator::Stop(Seconds(1.5));
        Simulator::Run();
        // The replies of every topology accumulate: the last one is the
        // SYN+ACK of this run
        NS_ABORT_MSG_IF(GetReplies().empty(), "The SYN was not answered");
        isn = GetReplies().back().GetSequenceNumber();
        if (isn.GetValue() >= upperHalf)
        {
            found = true;
            break;
        }
        Simulator::Destroy();
    }
    if (!found)
    {
        Config::Reset();
        NS_TEST_ASSERT_MSG_EQ(found, true, "No run produced an ISN in the upper half of the space");
        return;
    }

    // Complete the handshake, let the target send, then report the first
    // segment lost with three duplicate ACKs
    Simulator::Schedule(Seconds(2),
                        &TcpPassiveHighIsnTestCase::SendSegmentFull,
                        this,
                        TcpHeader::ACK,
                        2U,
                        (isn + 1).GetValue(),
                        static_cast<uint16_t>(65535),
                        false,
                        static_cast<uint16_t>(0),
                        0U);
    Simulator::Schedule(Seconds(2.5), &TcpPassiveHighIsnTestCase::SendData, this);
    for (uint32_t i = 0; i < 3; ++i)
    {
        Simulator::Schedule(Seconds(3) + MilliSeconds(i),
                            &TcpPassiveHighIsnTestCase::SendSegmentFull,
                            this,
                            TcpHeader::ACK,
                            2U,
                            (isn + 1).GetValue(),
                            static_cast<uint16_t>(65535),
                            false,
                            static_cast<uint16_t>(0),
                            0U);
    }
    Simulator::Stop(Seconds(4));
    Simulator::Run();

    uint32_t firstSegment = 0;
    for (const auto& reply : GetReplies())
    {
        if (reply.GetFlags() & TcpHeader::SYN)
        {
            continue;
        }
        NS_TEST_ASSERT_MSG_EQ(
            (reply.GetSequenceNumber() >= isn + 1 && reply.GetSequenceNumber() <= isn + 5001),
            true,
            "A segment was sent outside the sequence space of the connection: "
                << reply.GetSequenceNumber());
        if (reply.GetSequenceNumber() == isn + 1)
        {
            firstSegment++;
        }
    }
    NS_TEST_ASSERT_MSG_GT_OR_EQ(firstSegment,
                                2,
                                "The first segment was not retransmitted after the duplicate ACKs");

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
    // The ordinary data is read as it comes, so that the urgent data pending
    // is the urgent data alone
    socket->SetRecvCallback(Callback<void, Ptr<Socket>>([](Ptr<Socket> s) {
        while (s->Recv())
        {
        }
    }));
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

        // Nothing acknowledges the SYN+ACK, so the target may retransmit it
        NS_TEST_ASSERT_MSG_GT_OR_EQ(CountReplies(TcpHeader::SYN | TcpHeader::ACK),
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

        // Nothing acknowledges the SYN+ACK, so the target may retransmit it
        NS_TEST_ASSERT_MSG_GT_OR_EQ(CountReplies(TcpHeader::SYN | TcpHeader::ACK),
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

        // Nothing acknowledges the SYN+ACK, so the target may retransmit it
        NS_TEST_ASSERT_MSG_GT_OR_EQ(
            CountReplies(TcpHeader::SYN | TcpHeader::ACK),
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
 * The MSS option advertises the largest segment the interface receives
 * (MUST-67): the receiver opens the connection over an interface of small
 * MTU.
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
    constexpr uint32_t IPV4_TCP_HEADERS_SIZE = 40;
    DynamicCast<SimpleNetDevice>(devices.Get(0))->SetMtu(m_advertisedMss + IPV4_TCP_HEADERS_SIZE);

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
        SetupTopology(true, false);

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
 * @brief Test that an abort is told apart from a normal close
 *
 * @RFC{9293}, Section 3.6 (MUST-12) requires the application to be informed
 * whether the connection was closed normally or was aborted; this is the
 * abort half, in which the remote side closes with unread received data
 * pending, which resets the connection instead of closing it with a FIN.
 * The reset reaches the accepted socket while established, and once again
 * while it waits for the acknowledgment of its own FIN.
 */
class TcpAbortNotificationTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpAbortNotificationTestCase()
        : TcpCraftedSegmentTestCase("An abort is told apart from a normal close (MUST-12)")
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
     * Accept callback, which installs the close callbacks on the forked
     * socket and sends data the peer will never read.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    /**
     * Close the peer socket, with received data left unread.
     * @param socket The socket to close.
     */
    void CloseSocket(Ptr<Socket> socket);

    /**
     * Run the scenario, with the accepted socket closing first or not.
     * @param closeFirst Whether the accepted socket closes before the reset.
     */
    void Run(bool closeFirst);

    uint32_t m_normalCloses{0}; //!< Number of normal close notifications
    uint32_t m_errorCloses{0};  //!< Number of error close notifications
    Ptr<Socket> m_accepted;     //!< The accepted socket
};

void
TcpAbortNotificationTestCase::NormalClose(Ptr<Socket> socket)
{
    m_normalCloses++;
}

void
TcpAbortNotificationTestCase::ErrorClose(Ptr<Socket> socket)
{
    m_errorCloses++;
}

void
TcpAbortNotificationTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    m_accepted = socket;
    socket->SetCloseCallbacks(MakeCallback(&TcpAbortNotificationTestCase::NormalClose, this),
                              MakeCallback(&TcpAbortNotificationTestCase::ErrorClose, this));
    socket->Send(Create<Packet>(100), 0);
}

void
TcpAbortNotificationTestCase::CloseSocket(Ptr<Socket> socket)
{
    socket->Close();
}

void
TcpAbortNotificationTestCase::Run(bool closeFirst)
{
    m_normalCloses = 0;
    m_errorCloses = 0;
    m_accepted = nullptr;

    SetupTopology(true, false);

    m_target->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                MakeCallback(&TcpAbortNotificationTestCase::Accepted, this));

    // The peer never reads what the target sends, so closing it resets the
    // connection instead of closing it with a FIN
    Ptr<Socket> peer = Socket::CreateSocket(m_nodes.Get(0), TcpSocketFactory::GetTypeId());
    peer->Bind();
    peer->Connect(InetSocketAddress(m_interfaces.GetAddress(1), m_targetPort));
    if (closeFirst)
    {
        // The accepted socket closes its side first, and awaits the
        // acknowledgment of its FIN when the reset comes: the peer never
        // acknowledges, so that the socket is still waiting
        Ptr<RateErrorModel> error = CreateObject<RateErrorModel>();
        error->SetAttribute("ErrorRate", DoubleValue(1.0));
        error->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
        Simulator::Schedule(Seconds(2), [this, error] {
            m_nodes.Get(0)->GetDevice(0)->SetAttribute("ReceiveErrorModel", PointerValue(error));
            m_accepted->Close();
        });
        Simulator::Schedule(Seconds(3),
                            [error] { error->SetAttribute("ErrorRate", DoubleValue(0.0)); });
    }
    Simulator::Schedule(Seconds(3), &TcpAbortNotificationTestCase::CloseSocket, this, peer);

    Simulator::Stop(Seconds(10));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_errorCloses, 1, "A close with a RST was not notified as an abort");
    NS_TEST_ASSERT_MSG_EQ(m_normalCloses, 0, "A close with a RST was notified as normal");

    Simulator::Destroy();
}

void
TcpAbortNotificationTestCase::DoRun()
{
    Run(false);
    Run(true);
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
     * Read everything pending on the receiver.
     */
    void ReadAll();

    /**
     * Accept callback, holding the accepted socket without reading from it.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    Ptr<Socket> m_accepted;       //!< The socket of the receiver
    uint32_t m_probes{0};         //!< Segments sent while the peer window was zero
    uint32_t m_zeroWindows{0};    //!< Zero windows advertised by the receiver
    uint32_t m_tinyWindows{0};    //!< Non-zero windows below one segment
    uint32_t m_runtSegments{0};   //!< Data segments below one segment while more data waits
    bool m_windowIsZero{false};   //!< True while the last advertised window was zero
    uint32_t m_segmentSize{500};  //!< Segment size of the connection
    uint32_t m_lastWindow{0};     //!< Last window advertised by the receiver
    uint32_t m_reopenedWindow{0}; //!< First non-zero window advertised after a zero one
    uint32_t m_received{0};       //!< Bytes the receiving application read
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
    if (m_windowIsZero && window > 0 && m_reopenedWindow == 0)
    {
        m_reopenedWindow = window;
    }
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
        Ptr<Packet> packet = m_accepted->Recv(10, 0);
        m_received += packet ? packet->GetSize() : 0;
    }
}

void
TcpZeroWindowProbeTestCase::ReadAll()
{
    if (!m_accepted)
    {
        return;
    }
    while (Ptr<Packet> packet = m_accepted->Recv())
    {
        m_received += packet->GetSize();
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
    // then drains its buffer, which reopens the window by whole segments and
    // lets the transfer complete
    for (uint32_t i = 0; i < 40; ++i)
    {
        Simulator::Schedule(Seconds(10) + MilliSeconds(200 * i),
                            &TcpZeroWindowProbeTestCase::ReadAll,
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
    NS_TEST_ASSERT_MSG_GT_OR_EQ(m_reopenedWindow,
                                segmentSize,
                                "The window was not reopened by at least one segment");
    NS_TEST_ASSERT_MSG_EQ(m_received,
                          20000,
                          "The transfer did not complete once the window reopened");

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
        : TcpCraftedSegmentTestCase(
              "A SYN towards or from a broadcast address is dropped (MUST-57, MUST-63)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Inject a SYN towards the given destination.
     * @param destination The destination address.
     */
    void SendSynTo(Ipv4Address destination);

    /**
     * Inject a SYN claiming to come from the given source, the IP header
     * being built by the test.
     * @param source The source address.
     */
    void SendSynFrom(Ipv4Address source);

    /**
     * Count the TCP segments reaching the spoofing socket, which is bound
     * to the any address and hence sees the answers to a broadcast address.
     * @param socket The spoofing socket.
     */
    void ReceiveSpoofed(Ptr<Socket> socket);

    Ptr<Socket> m_spoofer;        //!< The raw socket sending the spoofed SYN
    uint32_t m_spoofedReplies{0}; //!< TCP segments the spoofing socket received
};

void
TcpBroadcastSynTestCase::ReceiveSpoofed(Ptr<Socket> socket)
{
    Address from;
    while (Ptr<Packet> packet = socket->RecvFrom(from))
    {
        Ipv4Header ipv4;
        packet->RemoveHeader(ipv4);
        if (ipv4.GetProtocol() == iana::internetprotocolnumbers::TCP)
        {
            m_spoofedReplies++;
        }
    }
}

void
TcpBroadcastSynTestCase::SendSynFrom(Ipv4Address source)
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
    Ipv4Header ipv4;
    ipv4.SetSource(source);
    ipv4.SetDestination(m_interfaces.GetAddress(1));
    ipv4.SetProtocol(iana::internetprotocolnumbers::TCP);
    ipv4.SetPayloadSize(packet->GetSize());
    ipv4.SetTtl(64);
    packet->AddHeader(ipv4);

    // The raw socket sends the header as built, through the device towards
    // the target, since the spoofed source belongs to no interface
    m_spoofer = Socket::CreateSocket(m_nodes.Get(0), Ipv4RawSocketFactory::GetTypeId());
    m_spoofer->SetAttribute("Protocol", UintegerValue(6));
    m_spoofer->SetAttribute("IpHeaderInclude", BooleanValue(true));
    m_spoofer->BindToNetDevice(m_nodes.Get(0)->GetDevice(0));
    m_spoofer->Bind();
    m_spoofer->SetRecvCallback(MakeCallback(&TcpBroadcastSynTestCase::ReceiveSpoofed, this));
    m_spoofer->SendTo(packet, 0, InetSocketAddress(m_interfaces.GetAddress(1), 0));
}

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
    // and so does a SYN claiming to come from it (MUST-63)
    Simulator::Schedule(Seconds(2),
                        &TcpBroadcastSynTestCase::SendSynFrom,
                        this,
                        Ipv4Address("10.1.1.255"));

    Simulator::Stop(Seconds(5));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(GetReplies().size(), 0, "A SYN towards a broadcast address was answered");
    NS_TEST_ASSERT_MSG_EQ(m_spoofedReplies, 0, "A SYN from a broadcast address was answered");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that a unicast address which looks like a broadcast is served
 *
 * The directed broadcast address of a subnet has its host part all ones
 * under the mask of that subnet, but so does one address in four of any
 * other subnet, under a /30 mask: a segment from such a remote address must
 * not be mistaken for one from a broadcast address (@RFC{9293},
 * Section 3.10.7.2, MUST-63 applies to broadcast addresses only).
 */
class TcpBroadcastLookalikeTestCase : public TestCase
{
  public:
    TcpBroadcastLookalikeTestCase()
        : TestCase("A remote address whose host bits are all ones is served")
    {
    }

  private:
    void DoRun() override;

    /**
     * Accept callback of the server.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    /**
     * Receive callback of the accepted socket, counting the data delivered.
     * @param socket The accepted socket.
     */
    void Received(Ptr<Socket> socket);

    /**
     * Send data from the client.
     * @param socket The client socket.
     */
    void SendData(Ptr<Socket> socket);

    uint32_t m_delivered{0}; //!< Bytes the server received
};

void
TcpBroadcastLookalikeTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->SetRecvCallback(MakeCallback(&TcpBroadcastLookalikeTestCase::Received, this));
}

void
TcpBroadcastLookalikeTestCase::Received(Ptr<Socket> socket)
{
    while (Ptr<Packet> packet = socket->Recv())
    {
        m_delivered += packet->GetSize();
    }
}

void
TcpBroadcastLookalikeTestCase::SendData(Ptr<Socket> socket)
{
    socket->Send(Create<Packet>(1000), 0);
}

void
TcpBroadcastLookalikeTestCase::DoRun()
{
    // The client (0) sits on a /24 LAN behind a router (1), which reaches
    // the server (2) over a /30 link
    NodeContainer nodes;
    nodes.Create(3);

    NodeContainer lan(nodes.Get(0), nodes.Get(1));
    NodeContainer link(nodes.Get(1), nodes.Get(2));

    SimpleNetDeviceHelper devHelper;
    NetDeviceContainer lanDevices = devHelper.Install(lan);
    NetDeviceContainer linkDevices = devHelper.Install(link);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    // The client takes 10.1.2.3, whose host part is all ones under the /30
    // mask of the link the server is on
    address.SetBase("10.1.2.0", "255.255.255.0", "0.0.0.3");
    Ipv4InterfaceContainer lanIfs = address.Assign(lanDevices);
    address.SetBase("10.1.3.0", "255.255.255.252");
    Ipv4InterfaceContainer linkIfs = address.Assign(linkDevices);
    NS_ASSERT_MSG(lanIfs.GetAddress(0) == Ipv4Address("10.1.2.3"), "Unexpected client address");

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    const uint16_t port = 9412;

    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(2), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpBroadcastLookalikeTestCase::Accepted, this));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->Connect(InetSocketAddress(linkIfs.GetAddress(1), port));
    Simulator::Schedule(Seconds(1), &TcpBroadcastLookalikeTestCase::SendData, this, client);

    Simulator::Stop(Seconds(5));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(
        m_delivered,
        1000,
        "The segments of a unicast address looking like a broadcast were dropped");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that a SYN towards an IPv6 multicast address is dropped
 *
 * @RFC{9293}, Section 3.10.7.2 requires an incoming SYN addressed to a
 * multicast address to be discarded (MUST-57), over IPv6 as over IPv4.
 */
class TcpIpv6MulticastSynTestCase : public TestCase
{
  public:
    TcpIpv6MulticastSynTestCase()
        : TestCase("A SYN towards an IPv6 multicast address is dropped (MUST-57)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Inject a SYN towards the all-nodes multicast address.
     */
    void SendSyn();

    /**
     * Count a TCP packet seen by one of the raw sockets.
     * @param counter The counter to increment.
     * @param socket The raw socket.
     */
    void ReceiveRaw(uint32_t* counter, Ptr<Socket> socket);

    Ptr<Socket> m_prober;        //!< The raw socket of the probing node
    uint32_t m_delivered{0};     //!< TCP packets the target node saw
    uint32_t m_replies{0};       //!< TCP packets the probing node received back
    uint16_t m_targetPort{9401}; //!< The port the target listens on
};

void
TcpIpv6MulticastSynTestCase::ReceiveRaw(uint32_t* counter, Ptr<Socket> socket)
{
    Address from;
    while (socket->RecvFrom(from))
    {
        (*counter)++;
    }
}

void
TcpIpv6MulticastSynTestCase::SendSyn()
{
    const uint8_t segment[20] = {
        0x24,
        0xba, // source port 9402
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
    m_prober->SendTo(packet, 0, Inet6SocketAddress(Ipv6Address::GetAllNodesMulticast(), 0));
}

void
TcpIpv6MulticastSynTestCase::DoRun()
{
    NodeContainer nodes;
    nodes.Create(2);

    SimpleNetDeviceHelper devHelper;
    NetDeviceContainer devices = devHelper.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv6AddressHelper address;
    address.SetBase("2001:db8::", Ipv6Prefix(64));
    Ipv6InterfaceContainer interfaces = address.Assign(devices);

    Ptr<Socket> target = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    target->Bind(Inet6SocketAddress(Ipv6Address::GetAny(), m_targetPort));
    target->Listen();

    // A raw socket on the target counts the TCP packets the IP layer
    // delivered, so that a SYN dropped before reaching the node cannot pass
    // the test either
    Ptr<Socket> watcher = Socket::CreateSocket(nodes.Get(1), Ipv6RawSocketFactory::GetTypeId());
    watcher->SetAttribute("Protocol", UintegerValue(6));
    watcher->Bind(Inet6SocketAddress(Ipv6Address::GetAny(), 0));
    watcher->SetRecvCallback(
        MakeCallback(&TcpIpv6MulticastSynTestCase::ReceiveRaw, this).Bind(&m_delivered));

    m_prober = Socket::CreateSocket(nodes.Get(0), Ipv6RawSocketFactory::GetTypeId());
    m_prober->SetAttribute("Protocol", UintegerValue(6));
    m_prober->Bind(Inet6SocketAddress(interfaces.GetAddress(0, 1), 0));
    m_prober->SetRecvCallback(
        MakeCallback(&TcpIpv6MulticastSynTestCase::ReceiveRaw, this).Bind(&m_replies));

    Simulator::Schedule(Seconds(1), &TcpIpv6MulticastSynTestCase::SendSyn, this);

    Simulator::Stop(Seconds(5));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_delivered, 1, "The multicast SYN did not reach the target node");
    NS_TEST_ASSERT_MSG_EQ(m_replies, 0, "A SYN towards a multicast address was answered");

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
    void DoTeardown() override;

    BooleanValue m_previousChecksumEnabled; //!< The global value before the test
};

void
TcpChecksumTestCase::DoTeardown()
{
    // Restored whatever the outcome of the test, for the suites which run
    // after this one in the same process
    GlobalValue::Bind("ChecksumEnabled", m_previousChecksumEnabled);
}

void
TcpChecksumTestCase::DoRun()
{
    GlobalValue::GetValueByName("ChecksumEnabled", m_previousChecksumEnabled);
    GlobalValue::Bind("ChecksumEnabled", BooleanValue(true));

    SetupTopology();

    // A SYN whose checksum does not match must be dropped without an answer
    Simulator::Schedule(Seconds(1),
                        &TcpChecksumTestCase::SendSegmentFull,
                        this,
                        TcpHeader::SYN,
                        1U,
                        0U,
                        static_cast<uint16_t>(4096),
                        true,
                        static_cast<uint16_t>(0),
                        0U);
    // The same SYN, with the checksum the receiver expects, is answered
    Simulator::Schedule(Seconds(3),
                        &TcpChecksumTestCase::SendSegmentFull,
                        this,
                        TcpHeader::SYN,
                        1U,
                        0U,
                        static_cast<uint16_t>(4096),
                        false,
                        static_cast<uint16_t>(0),
                        0U);

    Simulator::Stop(Seconds(6));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(CountReplies(TcpHeader::SYN | TcpHeader::ACK),
                          1,
                          "The SYN with a valid checksum was not answered exactly once");
    NS_TEST_ASSERT_MSG_EQ(CountReplies(TcpHeader::RST),
                          0,
                          "A segment with a wrong checksum was answered with a reset");

    Simulator::Destroy();
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
    NS_ABORT_MSG_IF(GetReplies().empty(), "The SYN was not answered");
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
                        1U,
                        0U,
                        m_largeWindow,
                        false,
                        static_cast<uint16_t>(0),
                        0U);
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
 * @brief Test that malformed options abort an established connection
 *
 * @RFC{9293}, Section 3.1 (MUST-7) resets the connection upon an illegal
 * option length: not only is the peer sent a RST, the connection on the
 * receiving end is torn down as well, and its application told of the
 * abort. The connection is driven from a raw socket, which injects a
 * malformed acknowledgment once the handshake is over.
 */
class TcpMalformedOptionsAbortTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpMalformedOptionsAbortTestCase()
        : TcpCraftedSegmentTestCase("Malformed options abort the established connection (MUST-7)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Complete the handshake driven from the raw socket.
     */
    void CompleteHandshake();

    /**
     * Inject an acknowledgment carrying an option with an illegal length.
     */
    void SendMalformedAck();

    /**
     * Accept callback, hooking the close callbacks and the state trace on
     * the accepted socket.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    /**
     * Error close callback of the accepted socket.
     * @param socket The aborted socket.
     */
    void ErrorClose(Ptr<Socket> socket);

    /**
     * Normal close callback of the accepted socket.
     * @param socket The closed socket.
     */
    void NormalClose(Ptr<Socket> socket);

    /**
     * State trace sink of the accepted socket.
     * @param oldState The previous state.
     * @param newState The new state.
     */
    void StateChanged(TcpSocket::TcpStates_t oldState, TcpSocket::TcpStates_t newState);

    SequenceNumber32 m_targetIsn{0};                   //!< Initial sequence number of the target
    uint32_t m_errorCloses{0};                         //!< Error close notifications
    uint32_t m_normalCloses{0};                        //!< Normal close notifications
    TcpSocket::TcpStates_t m_state{TcpSocket::CLOSED}; //!< Last state of the accepted socket
    bool m_accepted{false};                            //!< Whether a connection was accepted
};

void
TcpMalformedOptionsAbortTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    m_accepted = true;
    m_state = TcpSocket::ESTABLISHED;
    socket->SetCloseCallbacks(MakeCallback(&TcpMalformedOptionsAbortTestCase::NormalClose, this),
                              MakeCallback(&TcpMalformedOptionsAbortTestCase::ErrorClose, this));
    socket->TraceConnectWithoutContext(
        "State",
        MakeCallback(&TcpMalformedOptionsAbortTestCase::StateChanged, this));
}

void
TcpMalformedOptionsAbortTestCase::ErrorClose(Ptr<Socket> socket)
{
    m_errorCloses++;
}

void
TcpMalformedOptionsAbortTestCase::NormalClose(Ptr<Socket> socket)
{
    m_normalCloses++;
}

void
TcpMalformedOptionsAbortTestCase::StateChanged(TcpSocket::TcpStates_t oldState,
                                               TcpSocket::TcpStates_t newState)
{
    m_state = newState;
}

void
TcpMalformedOptionsAbortTestCase::CompleteHandshake()
{
    NS_ABORT_MSG_IF(GetReplies().empty(), "The SYN was not answered");
    m_targetIsn = GetReplies().front().GetSequenceNumber();
    SendSegmentFull(TcpHeader::ACK, 2, (m_targetIsn + 1).GetValue(), 4096);
}

void
TcpMalformedOptionsAbortTestCase::SendMalformedAck()
{
    // An acknowledgment carrying a timestamp option (kind 8) which announces
    // a length of 10 bytes, more than the word of option space it lies in
    const uint8_t segment[24] = {
        static_cast<uint8_t>(m_proberPort >> 8),
        static_cast<uint8_t>(m_proberPort & 0xff),
        static_cast<uint8_t>(m_targetPort >> 8),
        static_cast<uint8_t>(m_targetPort & 0xff),
        0x00,
        0x00,
        0x00,
        0x02, // sequence number
        static_cast<uint8_t>((m_targetIsn + 1).GetValue() >> 24),
        static_cast<uint8_t>((m_targetIsn + 1).GetValue() >> 16),
        static_cast<uint8_t>((m_targetIsn + 1).GetValue() >> 8),
        static_cast<uint8_t>((m_targetIsn + 1).GetValue()), // acknowledgment number
        0x60,                                               // data offset: 6 words
        TcpHeader::ACK,
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
    m_prober->SendTo(packet, 0, InetSocketAddress(m_interfaces.GetAddress(1), 0));
}

void
TcpMalformedOptionsAbortTestCase::DoRun()
{
    SetupTopology();
    m_target->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                MakeCallback(&TcpMalformedOptionsAbortTestCase::Accepted, this));

    Simulator::Schedule(Seconds(1),
                        &TcpMalformedOptionsAbortTestCase::SendSegmentFull,
                        this,
                        TcpHeader::SYN,
                        1U,
                        0U,
                        static_cast<uint16_t>(4096),
                        false,
                        static_cast<uint16_t>(0),
                        0U);
    Simulator::Schedule(Seconds(2), &TcpMalformedOptionsAbortTestCase::CompleteHandshake, this);
    Simulator::Schedule(Seconds(3), &TcpMalformedOptionsAbortTestCase::SendMalformedAck, this);

    Simulator::Stop(Seconds(6));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_accepted, true, "The connection was not established");
    NS_TEST_ASSERT_MSG_GT_OR_EQ(CountReplies(TcpHeader::RST),
                                1,
                                "The malformed options did not reset the peer");
    NS_TEST_ASSERT_MSG_EQ(m_state,
                          TcpSocket::CLOSED,
                          "The malformed options left the connection open on this end");
    NS_TEST_ASSERT_MSG_EQ(m_errorCloses, 1, "The abort was not notified to the application");
    NS_TEST_ASSERT_MSG_EQ(m_normalCloses, 0, "The abort was notified as a normal close");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that a refused simultaneous open is notified as a failure
 *
 * A connection which reached SYN-RECEIVED from SYN-SENT and is reset there
 * was refused (@RFC{9293}, Section 3.10.7.4): the application which called
 * Connect() is told through the connection failure callback, rather than
 * through a close callback it has no reason to have set. The peer is driven
 * from a raw socket, which answers the SYN with a SYN of its own and then
 * with a RST.
 */
class TcpSimultaneousOpenRefusedTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpSimultaneousOpenRefusedTestCase()
        : TcpCraftedSegmentTestCase("A refused simultaneous open is notified as a failure")
    {
    }

  private:
    void DoRun() override;

    /**
     * Answer the SYN of the target with a SYN, bringing it to SYN-RECEIVED.
     */
    void SendSyn();

    /**
     * Reset the connection.
     */
    void SendRst();

    /**
     * Connection success callback.
     * @param socket The connected socket.
     */
    void Succeeded(Ptr<Socket> socket);

    /**
     * Connection failure callback.
     * @param socket The refused socket.
     */
    void Failed(Ptr<Socket> socket);

    /**
     * Close callback, normal or error.
     * @param socket The closed socket.
     */
    void Closed(Ptr<Socket> socket);

    /**
     * State trace sink of the connecting socket.
     * @param oldState The previous state.
     * @param newState The new state.
     */
    void StateChanged(TcpSocket::TcpStates_t oldState, TcpSocket::TcpStates_t newState);

    SequenceNumber32 m_targetIsn{0}; //!< Initial sequence number of the target
    uint32_t m_successes{0};         //!< Connection success notifications
    uint32_t m_failures{0};          //!< Connection failure notifications
    uint32_t m_closes{0};            //!< Close notifications
    bool m_reachedSynRcvd{false};    //!< Whether the target reached SYN_RCVD
};

void
TcpSimultaneousOpenRefusedTestCase::Succeeded(Ptr<Socket> socket)
{
    m_successes++;
}

void
TcpSimultaneousOpenRefusedTestCase::Failed(Ptr<Socket> socket)
{
    m_failures++;
}

void
TcpSimultaneousOpenRefusedTestCase::Closed(Ptr<Socket> socket)
{
    m_closes++;
}

void
TcpSimultaneousOpenRefusedTestCase::StateChanged(TcpSocket::TcpStates_t oldState,
                                                 TcpSocket::TcpStates_t newState)
{
    if (newState == TcpSocket::SYN_RCVD)
    {
        m_reachedSynRcvd = true;
    }
}

void
TcpSimultaneousOpenRefusedTestCase::SendSyn()
{
    NS_ABORT_MSG_IF(GetReplies().empty(), "The target sent no SYN");
    m_targetIsn = GetReplies().front().GetSequenceNumber();
    // A SYN which acknowledges nothing: the target moves to SYN-RECEIVED
    SendSegmentFull(TcpHeader::SYN, 1, 0, 4096);
}

void
TcpSimultaneousOpenRefusedTestCase::SendRst()
{
    // A RST acceptable in SYN-RECEIVED, at the sequence number the target
    // expects
    SendSegmentFull(TcpHeader::RST, 2, (m_targetIsn + 1).GetValue(), 4096);
}

void
TcpSimultaneousOpenRefusedTestCase::DoRun()
{
    SetupTopology(false);

    // The target opens the connection actively, towards the raw socket
    m_target = Socket::CreateSocket(m_nodes.Get(1), TcpSocketFactory::GetTypeId());
    m_target->Bind(InetSocketAddress(m_interfaces.GetAddress(1), m_targetPort));
    m_target->SetConnectCallback(MakeCallback(&TcpSimultaneousOpenRefusedTestCase::Succeeded, this),
                                 MakeCallback(&TcpSimultaneousOpenRefusedTestCase::Failed, this));
    m_target->SetCloseCallbacks(MakeCallback(&TcpSimultaneousOpenRefusedTestCase::Closed, this),
                                MakeCallback(&TcpSimultaneousOpenRefusedTestCase::Closed, this));
    m_target->TraceConnectWithoutContext(
        "State",
        MakeCallback(&TcpSimultaneousOpenRefusedTestCase::StateChanged, this));
    Simulator::Schedule(Seconds(1),
                        &Socket::Connect,
                        m_target,
                        InetSocketAddress(m_interfaces.GetAddress(0), m_proberPort));

    Simulator::Schedule(Seconds(2), &TcpSimultaneousOpenRefusedTestCase::SendSyn, this);
    Simulator::Schedule(Seconds(3), &TcpSimultaneousOpenRefusedTestCase::SendRst, this);

    Simulator::Stop(Seconds(6));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_reachedSynRcvd, true, "The simultaneous open was not started");
    NS_TEST_ASSERT_MSG_EQ(m_failures, 1, "The refused connection was not notified as a failure");
    NS_TEST_ASSERT_MSG_EQ(m_successes, 0, "The refused connection was notified as a success");
    NS_TEST_ASSERT_MSG_EQ(m_closes, 0, "The refused connection was notified as a close");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the zero window probe of a connection with nothing to send
 *
 * A peer which advertises a zero window from the start puts the connection
 * in the persist state before it has anything to send: the probe then
 * carries no data, which @RFC{9293}, Section 3.8.6.1 allows, instead of
 * dereferencing the empty transmission buffer (see @issueid{1326}).
 */
class TcpEmptyZeroWindowProbeTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpEmptyZeroWindowProbeTestCase()
        : TcpCraftedSegmentTestCase("A zero window is probed with nothing to send")
    {
    }

  private:
    void DoRun() override;

    /**
     * Complete the handshake, advertising a zero window.
     */
    void CompleteHandshake();

    /**
     * Accept callback, recording the accepted socket and tracing its state.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    /**
     * State trace sink of the accepted socket.
     * @param oldState The previous state.
     * @param newState The new state.
     */
    void StateChanged(TcpSocket::TcpStates_t oldState, TcpSocket::TcpStates_t newState);

    SequenceNumber32 m_targetIsn{0};                   //!< Initial sequence number of the target
    TcpSocket::TcpStates_t m_state{TcpSocket::CLOSED}; //!< Last state of the accepted socket
};

void
TcpEmptyZeroWindowProbeTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    m_state = TcpSocket::ESTABLISHED;
    socket->TraceConnectWithoutContext(
        "State",
        MakeCallback(&TcpEmptyZeroWindowProbeTestCase::StateChanged, this));
}

void
TcpEmptyZeroWindowProbeTestCase::StateChanged(TcpSocket::TcpStates_t oldState,
                                              TcpSocket::TcpStates_t newState)
{
    m_state = newState;
}

void
TcpEmptyZeroWindowProbeTestCase::CompleteHandshake()
{
    NS_ABORT_MSG_IF(GetReplies().empty(), "The SYN was not answered");
    m_targetIsn = GetReplies().front().GetSequenceNumber();
    SendSegmentFull(TcpHeader::ACK, 2, (m_targetIsn + 1).GetValue(), 0);
}

void
TcpEmptyZeroWindowProbeTestCase::DoRun()
{
    Config::SetDefault("ns3::TcpSocketBase::PersistTimeout", TimeValue(Seconds(1)));
    SetupTopology();
    m_target->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                MakeCallback(&TcpEmptyZeroWindowProbeTestCase::Accepted, this));

    // A SYN, then an ACK, both advertising a zero window
    Simulator::Schedule(Seconds(1),
                        &TcpEmptyZeroWindowProbeTestCase::SendSegmentFull,
                        this,
                        TcpHeader::SYN,
                        1U,
                        0U,
                        static_cast<uint16_t>(0),
                        false,
                        static_cast<uint16_t>(0),
                        0U);
    Simulator::Schedule(Seconds(2), &TcpEmptyZeroWindowProbeTestCase::CompleteHandshake, this);

    Simulator::Stop(Seconds(8));
    Simulator::Run();

    // The probes hold no data and repeat the last acknowledged sequence
    // number (SND.NXT - 1, which is the ISN with nothing sent), so that the
    // peer answers them with an acknowledgment; the connection outlives them
    uint32_t probes = 0;
    for (const auto& reply : GetReplies())
    {
        if (reply.GetFlags() == TcpHeader::ACK && reply.GetSequenceNumber() == m_targetIsn)
        {
            probes++;
        }
    }
    NS_TEST_ASSERT_MSG_GT(probes, 0, "The zero window was not probed");
    NS_TEST_ASSERT_MSG_EQ(m_state,
                          TcpSocket::ESTABLISHED,
                          "The connection did not survive probing with nothing to send");

    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the urgent mechanism of a connection with a high sequence number
 *
 * The urgent points held by a socket are only comparable to the live sequence
 * numbers of the connection once they have been set from one: an urgent
 * pointer received on a connection whose sequence numbers lie in the upper
 * half of the sequence space used to be discarded by the circular comparison
 * against the initial value of zero, and the symmetric comparison used to
 * flag outgoing segments as urgent spuriously. Once the urgent data is
 * consumed, nothing is pending any longer and a later urgent pointer is
 * notified anew.
 */
class TcpUrgentHighIsnTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpUrgentHighIsnTestCase()
        : TcpCraftedSegmentTestCase("Urgent data is delivered on a connection with a high ISN")
    {
    }

  private:
    void DoRun() override;

    /**
     * Complete the handshake started with a high initial sequence number.
     */
    void CompleteHandshake();

    /**
     * Send a segment flagged as urgent.
     * @param seq The sequence number of the segment.
     */
    void SendUrgent(uint32_t seq);

    /**
     * Send the data the first urgent pointer announced.
     */
    void SendUrgentData();

    /**
     * Receive callback of the accepted socket, reading the data and
     * recording the urgent data pending afterwards.
     * @param socket The accepted socket.
     */
    void Received(Ptr<Socket> socket);

    /**
     * Accept callback, hooking the urgent data callback on the new socket.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    /**
     * Urgent data callback of the accepted socket.
     * @param socket The socket holding urgent data.
     */
    void UrgentDataPending(Ptr<Socket> socket);

    /// A sequence number in the upper half of the sequence space
    static constexpr uint32_t HIGH_ISN{0x80000000};

    SequenceNumber32 m_targetIsn{0}; //!< Initial sequence number of the target
    uint32_t m_notifications{0};     //!< Number of urgent data notifications
    uint32_t m_notifiedSize{0};      //!< Pending urgent bytes reported at the notification
    uint16_t m_urgentPointer{100};   //!< Urgent pointer of the injected segment
    uint32_t m_pendingAfterRead{1};  //!< Urgent bytes pending once the urgent data was read
};

void
TcpUrgentHighIsnTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->SetUrgentDataCallback(MakeCallback(&TcpUrgentHighIsnTestCase::UrgentDataPending, this));
    socket->SetRecvCallback(MakeCallback(&TcpUrgentHighIsnTestCase::Received, this));
}

void
TcpUrgentHighIsnTestCase::UrgentDataPending(Ptr<Socket> socket)
{
    m_notifications++;
    m_notifiedSize = socket->GetUrgentDataSize();
}

void
TcpUrgentHighIsnTestCase::CompleteHandshake()
{
    NS_ABORT_MSG_IF(GetReplies().empty(), "The SYN was not answered");
    m_targetIsn = GetReplies().front().GetSequenceNumber();
    SendSegmentFull(TcpHeader::ACK, HIGH_ISN + 1, (m_targetIsn + 1).GetValue(), 4096);
}

void
TcpUrgentHighIsnTestCase::SendUrgent(uint32_t seq)
{
    // A segment holding no data, whose urgent pointer reaches beyond its
    // sequence number, tells the receiver urgent data is on the way
    SendSegmentFull(TcpHeader::ACK | TcpHeader::URG,
                    seq,
                    (m_targetIsn + 1).GetValue(),
                    4096,
                    false,
                    m_urgentPointer);
}

void
TcpUrgentHighIsnTestCase::SendUrgentData()
{
    // The urgent data itself, which the application reads
    SendSegmentFull(TcpHeader::ACK | TcpHeader::URG | TcpHeader::PSH,
                    HIGH_ISN + 1,
                    (m_targetIsn + 1).GetValue(),
                    4096,
                    false,
                    m_urgentPointer,
                    m_urgentPointer);
}

void
TcpUrgentHighIsnTestCase::Received(Ptr<Socket> socket)
{
    while (socket->Recv())
    {
    }
    m_pendingAfterRead = socket->GetUrgentDataSize();
}

void
TcpUrgentHighIsnTestCase::DoRun()
{
    SetupTopology();
    m_target->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                MakeCallback(&TcpUrgentHighIsnTestCase::Accepted, this));

    Simulator::Schedule(Seconds(1),
                        &TcpUrgentHighIsnTestCase::SendSegmentFull,
                        this,
                        TcpHeader::SYN,
                        HIGH_ISN,
                        0U,
                        static_cast<uint16_t>(4096),
                        false,
                        static_cast<uint16_t>(0),
                        0U);
    Simulator::Schedule(Seconds(2), &TcpUrgentHighIsnTestCase::CompleteHandshake, this);
    Simulator::Schedule(Seconds(3), &TcpUrgentHighIsnTestCase::SendUrgent, this, HIGH_ISN + 1);
    Simulator::Schedule(Seconds(4), &TcpUrgentHighIsnTestCase::SendUrgentData, this);
    // A later urgent pointer, once the first urgent data was consumed
    Simulator::Schedule(Seconds(5),
                        &TcpUrgentHighIsnTestCase::SendUrgent,
                        this,
                        HIGH_ISN + 1 + m_urgentPointer);

    Simulator::Stop(Seconds(7));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_notifications,
                          2,
                          "The urgent pointers were not notified on a high ISN connection");
    NS_TEST_ASSERT_MSG_EQ(m_notifiedSize,
                          m_urgentPointer,
                          "Unexpected amount of pending urgent data");
    NS_TEST_ASSERT_MSG_EQ(m_pendingAfterRead, 0, "Urgent data was still pending once read");

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
     * @param passive Whether the passively opened socket probes, with the
     *                keep-alives enabled through the defaults.
     * @return The number of keep-alives sent.
     */
    uint32_t CountKeepAlives(bool keepAlive, bool busy, bool passive = false);

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

    /**
     * Tx trace sink of the probing socket, counting the keep-alive probes.
     * @param packet The segment sent.
     * @param header The TCP header of the segment.
     * @param socket The sending socket.
     */
    void ProbeSent(Ptr<const Packet> packet,
                   const TcpHeader& header,
                   Ptr<const TcpSocketBase> socket);

    /**
     * Close callback of the probing socket, counting the closes.
     * @param socket The closed socket.
     */
    void Closed(Ptr<Socket> socket);

    /**
     * Accept callback, keeping the accepted socket, which is the probing one
     * of the passive runs.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    /**
     * Hook the close callbacks and the traces on the probing socket.
     * @param socket The probing socket.
     */
    void WatchSocket(Ptr<Socket> socket);

    /**
     * Send data from the given socket, or from the accepted one if none.
     * @param socket The sending socket, or nullptr for the accepted one.
     */
    void SendDataFrom(Ptr<Socket> socket);

    Ptr<Socket> m_accepted;      //!< The accepted socket of the passive runs
    bool m_watchAccepted{false}; //!< Whether the accepted socket is the probing one

    uint32_t m_probesSent{0};                      //!< Keep-alive probes the probing socket sent
    std::optional<SequenceNumber32> m_highestSent; //!< Highest sequence number of the data sent
    uint32_t m_closeNotifications{0};              //!< Close notifications of the probing socket
    Time m_closed{Seconds(0)};                     //!< Instant the probing socket gave up
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
TcpKeepAliveTestCase::Closed(Ptr<Socket> socket)
{
    m_closeNotifications++;
}

void
TcpKeepAliveTestCase::WatchSocket(Ptr<Socket> socket)
{
    socket->SetCloseCallbacks(MakeCallback(&TcpKeepAliveTestCase::Closed, this),
                              MakeCallback(&TcpKeepAliveTestCase::Closed, this));
    socket->TraceConnectWithoutContext("State",
                                       MakeCallback(&TcpKeepAliveTestCase::StateChanged, this));
    socket->TraceConnectWithoutContext("Tx", MakeCallback(&TcpKeepAliveTestCase::ProbeSent, this));
}

void
TcpKeepAliveTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    m_accepted = socket;
    if (!m_watchAccepted)
    {
        return;
    }
    // The accepted socket is established already, so its state is not
    // traced from the start
    m_state = TcpSocket::ESTABLISHED;
    WatchSocket(socket);
}

void
TcpKeepAliveTestCase::SendDataFrom(Ptr<Socket> socket)
{
    SendData(socket ? socket : m_accepted);
}

void
TcpKeepAliveTestCase::BreakThePath(Ptr<NetDevice> device)
{
    Ptr<RateErrorModel> error = CreateObject<RateErrorModel>();
    error->SetAttribute("ErrorRate", DoubleValue(1.0));
    error->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
    device->SetAttribute("ReceiveErrorModel", PointerValue(error));
}

void
TcpKeepAliveTestCase::ProbeSent(Ptr<const Packet> packet,
                                const TcpHeader& header,
                                Ptr<const TcpSocketBase> socket)
{
    if (packet->GetSize() > 0)
    {
        SequenceNumber32 end = header.GetSequenceNumber() + packet->GetSize();
        m_highestSent = m_highestSent ? std::max(*m_highestSent, end) : end;
        return;
    }

    // A keep-alive holds no data and repeats the sequence number of the last
    // octet sent, which the peer acknowledged already
    if (header.GetFlags() == TcpHeader::ACK && m_highestSent &&
        header.GetSequenceNumber() + 1 == *m_highestSent)
    {
        m_probesSent++;
    }
}

Time
TcpKeepAliveTestCase::RunUnansweredKeepAlives(uint32_t retries)
{
    m_closed = Seconds(0);
    m_state = TcpSocket::CLOSED;
    m_probesSent = 0;
    m_highestSent.reset();

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
    client->TraceConnectWithoutContext("Tx", MakeCallback(&TcpKeepAliveTestCase::ProbeSent, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));

    // Some data first, which the probes repeat the last octet of; then the
    // peer stops answering once the connection is up
    Simulator::Schedule(MilliSeconds(500), &TcpKeepAliveTestCase::SendData, this, client);
    Simulator::Schedule(Seconds(1), &TcpKeepAliveTestCase::BreakThePath, this, devices.Get(1));

    Simulator::Stop(Seconds(60));
    Simulator::Run();
    Simulator::Destroy();

    return m_closed;
}

void
TcpKeepAliveTestCase::SendData(Ptr<Socket> socket)
{
    socket->Send(Create<Packet>(100), 0);
}

uint32_t
TcpKeepAliveTestCase::CountKeepAlives(bool keepAlive, bool busy, bool passive)
{
    m_probesSent = 0;
    m_highestSent.reset();
    m_closeNotifications = 0;
    m_closed = Seconds(0);
    m_state = TcpSocket::CLOSED;
    m_accepted = nullptr;
    m_watchAccepted = passive;

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

    if (passive)
    {
        // The accepted socket inherits the keep-alives from the defaults,
        // with an interval short enough for the test to run (MUST-24 and
        // MUST-27), and few enough retries for a peer which never answers to
        // be given up on well before the end of the run
        Config::SetDefault("ns3::TcpSocket::KeepAlive", BooleanValue(keepAlive));
        Config::SetDefault("ns3::TcpSocket::KeepAliveTime", TimeValue(Seconds(2)));
        Config::SetDefault("ns3::TcpSocketBase::KeepAliveInterval", TimeValue(Seconds(1)));
        Config::SetDefault("ns3::TcpSocketBase::KeepAliveRetries", UintegerValue(3));
    }

    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpKeepAliveTestCase::Accepted, this));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    if (!passive)
    {
        // Turned on for this connection alone (MUST-24 and MUST-27)
        client->SetAttribute("KeepAlive", BooleanValue(keepAlive));
        client->SetAttribute("KeepAliveTime", TimeValue(Seconds(2)));
        client->SetAttribute("KeepAliveInterval", TimeValue(Seconds(1)));
        client->SetAttribute("KeepAliveRetries", UintegerValue(3));
        WatchSocket(client);
    }
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));

    // The probing socket is the one sending the data, so that the probes
    // repeat the last octet it sent
    Ptr<Socket> sender = passive ? nullptr : client;
    Simulator::Schedule(Seconds(1), &TcpKeepAliveTestCase::SendDataFrom, this, sender);
    if (busy)
    {
        // Data flowing all along, so the connection is never idle
        for (uint32_t i = 0; i < 54; ++i)
        {
            Simulator::Schedule(Seconds(2) + MilliSeconds(500 * i),
                                &TcpKeepAliveTestCase::SendDataFrom,
                                this,
                                sender);
        }
    }

    Simulator::Stop(Seconds(30));
    Simulator::Run();
    Config::Reset();

    // The peer answers every probe, so the connection outlives them all: a
    // peer which did not would have been given up on after the retries
    NS_TEST_ASSERT_MSG_EQ_RETURNS_BOOL(
        m_closeNotifications,
        0,
        "A connection whose peer answers the keep-alives was closed");
    NS_TEST_ASSERT_MSG_EQ_RETURNS_BOOL(
        m_state,
        TcpSocket::ESTABLISHED,
        "A connection whose peer answers the keep-alives did not stay up");

    Simulator::Destroy();

    // A probe is dropped by the peer once acknowledged, so it is counted as
    // it leaves rather than as it arrives
    return m_probesSent;
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
    // The passively opened socket inherits the keep-alives from the defaults
    uint32_t passiveIdle = CountKeepAlives(true, false, true);

    NS_TEST_ASSERT_MSG_GT(idle, 1, "An idle connection was not kept alive");
    NS_TEST_ASSERT_MSG_GT(passiveIdle,
                          1,
                          "An idle connection was not kept alive by its passively opened end");
    // The connection outlives the probes it sends, since a peer which answers
    // none of them is only given up on after the whole series (MUST-29)
    NS_TEST_ASSERT_MSG_EQ(idleWithoutKeepAlive,
                          0,
                          "A connection was kept alive with the keep-alives disabled");
    NS_TEST_ASSERT_MSG_EQ(busy, 0, "A connection carrying data was probed");

    // A single unanswered keep-alive cannot be configured to drop the
    // connection (MUST-29)
    Ptr<TcpSocketBase> unconnected = CreateObject<TcpSocketBase>();
    NS_TEST_ASSERT_MSG_EQ(unconnected->SetAttributeFailSafe("KeepAliveRetries", UintegerValue(0)),
                          false,
                          "A connection can be dropped without any keep-alive");
    NS_TEST_ASSERT_MSG_EQ(unconnected->SetAttributeFailSafe("KeepAliveRetries", UintegerValue(1)),
                          false,
                          "A connection can be dropped upon a single keep-alive");

    // A peer which answers nothing is given up on, but only after the whole
    // series of keep-alives, never upon a single one (MUST-29)
    Time fewRetries = RunUnansweredKeepAlives(2);
    uint32_t fewProbes = m_probesSent;
    Time manyRetries = RunUnansweredKeepAlives(5);
    uint32_t manyProbes = m_probesSent;

    // The KeepAliveRetries attribute sets exactly how many unanswered probes
    // are sent before the connection is dropped
    NS_TEST_ASSERT_MSG_EQ(fewProbes, 2, "Unexpected number of keep-alive probes");
    NS_TEST_ASSERT_MSG_EQ(manyProbes, 5, "Unexpected number of keep-alive probes");

    NS_TEST_ASSERT_MSG_GT(fewRetries, Seconds(0), "The connection outlived every keep-alive");
    NS_TEST_ASSERT_MSG_GT(manyRetries, Seconds(0), "The connection outlived every keep-alive");

    // The first keep-alive leaves after the two idle seconds, and the next
    // ones follow it every second: a connection dropped upon the first one
    // would be gone by then
    NS_TEST_ASSERT_MSG_GT(fewRetries,
                          Seconds(4),
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
 * The client and the server are joined by two distinct nodes in the middle,
 * so that the way back a datagram records and the route the server
 * application prefers can be told apart. The full sized segments leave room
 * for the option, so that they fill the MTU of the links without being
 * fragmented (@RFC{9293}, Section 3.7.1), the checksums are enabled, so that
 * the option survives the checksum, and the client node has more than one
 * interface, so that the datagram must be routed towards its first hop rather
 * than its destination.
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
     * Record the route of the answers reaching the client.
     * @param socket The raw socket of the client.
     */
    void ReceiveAnswer(Ptr<Socket> socket);

    /**
     * Accept callback, which keeps the accepted socket, gives it the route
     * the server application prefers and answers with data.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    /**
     * Receive callback of the accepted socket, counting the data delivered.
     * @param socket The accepted socket.
     */
    void Received(Ptr<Socket> socket);

    /**
     * IP layer receive trace of the server, counting the fragments before
     * they are reassembled.
     * @param packet The received datagram, header included.
     * @param ipv4 The IP layer.
     * @param interface The incoming interface.
     */
    void DatagramReceived(Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface);

    /**
     * Send the data of the client.
     * @param socket The client socket.
     */
    void SendData(Ptr<Socket> socket);

    uint32_t m_routedSegments{0};           //!< Segments which carried a source route
    uint32_t m_segments{0};                 //!< Segments which reached the peer
    uint32_t m_delivered{0};                //!< Bytes the server application received
    uint32_t m_fragments{0};                //!< Fragments which reached the peer
    uint32_t m_largest{0};                  //!< Largest datagram which reached the peer
    std::vector<Ipv4Address> m_seenRoute;   //!< Route carried by the first of them
    std::vector<Ipv4Address> m_answerRoute; //!< Route carried by the first answer with data
    std::vector<Ipv4Address> m_synAckRoute; //!< Route carried by the SYN+ACK
    std::vector<Ipv4Address> m_ownRoute;    //!< Route the server application prefers
    Ptr<Socket> m_accepted;                 //!< The socket of the passive open
    uint32_t m_dataSize{20000};             //!< Bytes the client sends
    BooleanValue m_previousChecksumEnabled; //!< The global value before the test

    void DoTeardown() override;
};

void
TcpSourceRouteTestCase::DoTeardown()
{
    GlobalValue::Bind("ChecksumEnabled", m_previousChecksumEnabled);
}

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
TcpSourceRouteTestCase::ReceiveAnswer(Ptr<Socket> socket)
{
    Address from;
    Ptr<Packet> packet = socket->RecvFrom(from);

    Ipv4Header ipv4;
    packet->RemoveHeader(ipv4);
    if (ipv4.GetProtocol() != 6 || !ipv4.HasLooseSourceRoute())
    {
        return;
    }

    TcpHeader tcp;
    packet->RemoveHeader(tcp);
    if ((tcp.GetFlags() & TcpHeader::SYN) && m_synAckRoute.empty())
    {
        // The SYN+ACK leaves the server before its application is told of
        // the connection, along the recorded way back
        m_synAckRoute = ipv4.GetLooseSourceRoute();
    }
    if (packet->GetSize() > 0 && m_answerRoute.empty())
    {
        // The first answer carrying data left the server after its
        // application specified a route of its own
        m_answerRoute = ipv4.GetLooseSourceRoute();
    }
}

void
TcpSourceRouteTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    m_accepted = socket;
    socket->SetRecvCallback(MakeCallback(&TcpSourceRouteTestCase::Received, this));

    // The route the server application prefers takes precedence over the
    // recorded way back (MUST-52), which the data sent from here shows
    DynamicCast<TcpSocketBase>(socket)->SetIpv4SourceRoute(m_ownRoute);
    socket->Send(Create<Packet>(100), 0);
}

void
TcpSourceRouteTestCase::Received(Ptr<Socket> socket)
{
    while (Ptr<Packet> packet = socket->Recv())
    {
        m_delivered += packet->GetSize();
    }
}

void
TcpSourceRouteTestCase::SendData(Ptr<Socket> socket)
{
    socket->Send(Create<Packet>(m_dataSize), 0);
}

void
TcpSourceRouteTestCase::DatagramReceived(Ptr<const Packet> packet,
                                         Ptr<Ipv4> ipv4,
                                         uint32_t interface)
{
    Ipv4Header header;
    packet->PeekHeader(header);
    if (header.GetProtocol() == 6)
    {
        m_largest = std::max(m_largest, packet->GetSize());
    }
    if (header.GetProtocol() != 6 || (header.IsLastFragment() && header.GetFragmentOffset() == 0))
    {
        return;
    }
    // The option is copied to every fragment (RFC 791)
    m_fragments++;
    NS_TEST_ASSERT_MSG_EQ(header.HasLooseSourceRoute(),
                          true,
                          "A fragment of a source routed datagram lost the option");
}

void
TcpSourceRouteTestCase::DoRun()
{
    // The checksums cover the option, so they are verified along the way
    GlobalValue::GetValueByName("ChecksumEnabled", m_previousChecksumEnabled);
    GlobalValue::Bind("ChecksumEnabled", BooleanValue(true));

    // Client (0) and server (2), joined through two nodes in the middle: the
    // upper one (1) the client routes through, the lower one (3) the server
    // application prefers
    NodeContainer nodes;
    nodes.Create(4);

    NodeContainer left(nodes.Get(0), nodes.Get(1));
    NodeContainer right(nodes.Get(1), nodes.Get(2));
    NodeContainer lowerLeft(nodes.Get(0), nodes.Get(3));
    NodeContainer lowerRight(nodes.Get(3), nodes.Get(2));

    SimpleNetDeviceHelper devHelper;
    NetDeviceContainer leftDevices = devHelper.Install(left);
    NetDeviceContainer rightDevices = devHelper.Install(right);
    NetDeviceContainer lowerLeftDevices = devHelper.Install(lowerLeft);
    NetDeviceContainer lowerRightDevices = devHelper.Install(lowerRight);
    for (auto devices : {leftDevices, rightDevices, lowerLeftDevices, lowerRightDevices})
    {
        for (auto device = devices.Begin(); device != devices.End(); ++device)
        {
            (*device)->SetMtu(1500);
        }
    }

    InternetStackHelper stack;
    stack.Install(nodes);
    // The nodes in the middle act as the hops of the source routes
    for (uint32_t hop : {1, 3})
    {
        nodes.Get(hop)->GetObject<Ipv4L3Protocol>()->SetAttribute("SourceRouteForwarding",
                                                                  BooleanValue(true));
    }

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer leftIfs = address.Assign(leftDevices);
    address.SetBase("10.1.2.0", "255.255.255.0");
    Ipv4InterfaceContainer rightIfs = address.Assign(rightDevices);
    address.SetBase("10.1.3.0", "255.255.255.0");
    Ipv4InterfaceContainer lowerLeftIfs = address.Assign(lowerLeftDevices);
    address.SetBase("10.1.4.0", "255.255.255.0");
    Ipv4InterfaceContainer lowerRightIfs = address.Assign(lowerRightDevices);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    const uint16_t port = 9911;

    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(2), TcpSocketFactory::GetTypeId());
    // Full sized segments on both ends, which would exceed the MTU if the
    // option took no room from the data
    server->SetAttribute("SegmentSize", UintegerValue(1460));
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpSourceRouteTestCase::Accepted, this));

    Ptr<Socket> sniffer = Socket::CreateSocket(nodes.Get(2), Ipv4RawSocketFactory::GetTypeId());
    sniffer->SetAttribute("Protocol", UintegerValue(6));
    sniffer->Bind();
    sniffer->SetRecvCallback(MakeCallback(&TcpSourceRouteTestCase::ReceiveRaw, this));
    nodes.Get(2)->GetObject<Ipv4L3Protocol>()->TraceConnectWithoutContext(
        "Rx",
        MakeCallback(&TcpSourceRouteTestCase::DatagramReceived, this));

    Ptr<Socket> answerSniffer =
        Socket::CreateSocket(nodes.Get(0), Ipv4RawSocketFactory::GetTypeId());
    answerSniffer->SetAttribute("Protocol", UintegerValue(6));
    answerSniffer->Bind();
    answerSniffer->SetRecvCallback(MakeCallback(&TcpSourceRouteTestCase::ReceiveAnswer, this));

    // The route the application asks for: through the upper node in the
    // middle, then on to the peer
    std::vector<Ipv4Address> route = {leftIfs.GetAddress(1), rightIfs.GetAddress(1)};

    // The route the server application prefers for its answers: through the
    // lower node in the middle
    m_ownRoute = {lowerRightIfs.GetAddress(0), lowerLeftIfs.GetAddress(0)};

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    Ptr<TcpSocketBase> tcpClient = DynamicCast<TcpSocketBase>(client);
    client->SetAttribute("SegmentSize", UintegerValue(1460));
    client->Bind();
    tcpClient->SetIpv4SourceRoute(route);
    client->Connect(InetSocketAddress(rightIfs.GetAddress(1), port));
    Simulator::Schedule(Seconds(1), &TcpSourceRouteTestCase::SendData, this, client);

    Simulator::Stop(Seconds(10));
    Simulator::Run();

    // The full sized segments filled the MTU without being fragmented
    NS_TEST_ASSERT_MSG_EQ(m_fragments, 0, "A segment was fragmented");
    NS_TEST_ASSERT_MSG_EQ(m_largest, 1500, "The segments did not fill the MTU");
    NS_TEST_ASSERT_MSG_EQ(m_delivered, m_dataSize, "The data did not reach the server whole");

    NS_TEST_ASSERT_MSG_GT(m_segments, 0, "No segment reached the peer");
    NS_TEST_ASSERT_MSG_EQ(m_routedSegments,
                          m_segments,
                          "A segment of the connection carried no source route");

    // The first hop travels in the destination field, so the option holds one
    // slot less than the route, and every hop writes the address it forwards
    // the datagram from in the slot it took the next hop from (RFC 791): the
    // option holds the way back when it arrives
    NS_TEST_ASSERT_MSG_EQ(m_seenRoute.size(), route.size() - 1, "The route was not carried whole");
    NS_TEST_ASSERT_MSG_EQ(m_seenRoute.empty() ? Ipv4Address() : m_seenRoute.front(),
                          rightIfs.GetAddress(0),
                          "The hop in the middle did not record the address it forwarded from");

    // The connection opened passively answers along the way back (MUST-53):
    // the SYN+ACK reaches the client through the upper node in the middle,
    // whose address on the way back is the one facing the client
    NS_TEST_ASSERT_MSG_NE(m_accepted, nullptr, "The connection was not established");
    NS_TEST_ASSERT_MSG_EQ(m_synAckRoute.empty(), false, "The SYN+ACK carried no source route");
    NS_TEST_ASSERT_MSG_EQ(m_synAckRoute.empty() ? Ipv4Address() : m_synAckRoute.front(),
                          leftIfs.GetAddress(1),
                          "The SYN+ACK did not follow the recorded way back");

    // and the route its application specified took precedence over the
    // recorded one (MUST-52): the data went through the lower node in the
    // middle, which recorded the address it forwarded from
    NS_TEST_ASSERT_MSG_EQ(m_answerRoute.empty(), false, "No answer carried the preferred route");
    NS_TEST_ASSERT_MSG_EQ(m_answerRoute.empty() ? Ipv4Address() : m_answerRoute.front(),
                          lowerLeftIfs.GetAddress(1),
                          "The route of the application did not take precedence");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that urgent data sent after a shutdown is rejected whole
 *
 * Once the sending side of a connection is shut down, a Send() call fails
 * with ERROR_SHUTDOWN. Its data must not reach the transmission buffer, and an
 * urgent one must not move the urgent pointer either, even while the FIN
 * waits for the data queued before the shutdown.
 */
class TcpSendAfterShutdownTestCase : public TestCase
{
  public:
    TcpSendAfterShutdownTestCase()
        : TestCase("Urgent data sent after a shutdown is rejected whole")
    {
    }

  private:
    void DoRun() override;

    /**
     * Queue some data, shut the sending side down, then try to send urgent data.
     * @param socket The sending socket.
     */
    void SendAndShutdown(Ptr<Socket> socket);

    /**
     * Tx trace sink of the sending socket.
     * @param packet The payload.
     * @param header The TCP header.
     * @param socket The sending socket.
     */
    void SegmentSent(Ptr<const Packet> packet,
                     const TcpHeader& header,
                     Ptr<const TcpSocketBase> socket);

    /**
     * Rx trace sink of the accepted socket.
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

    int m_sendResult{0};                                 //!< Return value of the rejected Send()
    Socket::SocketErrno m_errno{Socket::ERROR_NOTERROR}; //!< Error of the rejected Send()
    uint32_t m_txAvailableBefore{0};                     //!< Free space before the rejected Send()
    uint32_t m_txAvailableAfter{0};                      //!< Free space after the rejected Send()
    uint32_t m_urgentSegments{0};                        //!< Segments carrying the URG flag
    uint32_t m_bytesReceived{0};                         //!< Payload bytes which reached the peer
    uint32_t m_queued{2000};                             //!< Data queued before the shutdown
};

void
TcpSendAfterShutdownTestCase::SendAndShutdown(Ptr<Socket> socket)
{
    socket->Send(Create<Packet>(m_queued), 0);
    // The data is still queued, so the FIN waits for it and the connection
    // stays established
    socket->ShutdownSend();
    m_txAvailableBefore = socket->GetTxAvailable();
    m_sendResult = socket->Send(Create<Packet>(100), Socket::MSG_FLAG_OOB);
    m_errno = socket->GetErrno();
    m_txAvailableAfter = socket->GetTxAvailable();
}

void
TcpSendAfterShutdownTestCase::SegmentSent(Ptr<const Packet> packet,
                                          const TcpHeader& header,
                                          Ptr<const TcpSocketBase> socket)
{
    if (header.GetFlags() & TcpHeader::URG)
    {
        m_urgentSegments++;
    }
}

void
TcpSendAfterShutdownTestCase::DataReceived(Ptr<const Packet> packet,
                                           const TcpHeader& header,
                                           Ptr<const TcpSocketBase> socket)
{
    m_bytesReceived += packet->GetSize();
}

void
TcpSendAfterShutdownTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->TraceConnectWithoutContext(
        "Rx",
        MakeCallback(&TcpSendAfterShutdownTestCase::DataReceived, this));
}

void
TcpSendAfterShutdownTestCase::DoRun()
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

    const uint16_t port = 9920;

    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpSendAfterShutdownTestCase::Accepted, this));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->TraceConnectWithoutContext(
        "Tx",
        MakeCallback(&TcpSendAfterShutdownTestCase::SegmentSent, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));

    Simulator::Schedule(Seconds(1), &TcpSendAfterShutdownTestCase::SendAndShutdown, this, client);

    Simulator::Stop(Seconds(10));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_sendResult, -1, "Data was accepted after the shutdown");
    NS_TEST_ASSERT_MSG_EQ(m_errno, Socket::ERROR_SHUTDOWN, "Unexpected error");
    NS_TEST_ASSERT_MSG_EQ(m_txAvailableAfter,
                          m_txAvailableBefore,
                          "The rejected data was queued for transmission");
    NS_TEST_ASSERT_MSG_EQ(m_urgentSegments, 0, "The rejected urgent data was pointed at");
    NS_TEST_ASSERT_MSG_EQ(m_bytesReceived, m_queued, "The rejected data reached the peer");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the pending urgent data is counted from the read point
 *
 * @RFC{9293}, Section 3.8.5 (MUST-33) requires a way for the application to
 * learn how much urgent data remains to be read. The urgent data is consumed
 * once the application reads it, not once it reaches the receive buffer: the
 * receive callback must still see all of it pending, and each read must take
 * its share off.
 */
class TcpUrgentReadPointTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpUrgentReadPointTestCase()
        : TcpCraftedSegmentTestCase("Pending urgent data is counted from the read point (MUST-33)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Complete the handshake.
     */
    void CompleteHandshake();

    /**
     * Receive callback of the accepted socket, reading the data in two steps.
     * @param socket The accepted socket.
     */
    void Received(Ptr<Socket> socket);

    /**
     * Accept callback, hooking the receive callback on the new socket.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    SequenceNumber32 m_targetIsn{0}; //!< Initial sequence number of the target
    uint32_t m_urgent{100};          //!< Size of the urgent data
    uint32_t m_firstRead{40};        //!< Size of the first read
    uint32_t m_beforeRead{0};        //!< Pending urgent bytes before reading
    uint32_t m_afterFirstRead{0};    //!< Pending urgent bytes after the first read
    uint32_t m_afterLastRead{1};     //!< Pending urgent bytes once all was read
};

void
TcpUrgentReadPointTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->SetRecvCallback(MakeCallback(&TcpUrgentReadPointTestCase::Received, this));
}

void
TcpUrgentReadPointTestCase::CompleteHandshake()
{
    NS_ABORT_MSG_IF(GetReplies().empty(), "The SYN was not answered");
    m_targetIsn = GetReplies().front().GetSequenceNumber();
    SendSegmentFull(TcpHeader::ACK, 1001, (m_targetIsn + 1).GetValue(), 4096);
}

void
TcpUrgentReadPointTestCase::Received(Ptr<Socket> socket)
{
    m_beforeRead = socket->GetUrgentDataSize();
    socket->Recv(m_firstRead, 0);
    m_afterFirstRead = socket->GetUrgentDataSize();
    while (socket->Recv())
    {
    }
    m_afterLastRead = socket->GetUrgentDataSize();
}

void
TcpUrgentReadPointTestCase::DoRun()
{
    SetupTopology();
    m_target->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                MakeCallback(&TcpUrgentReadPointTestCase::Accepted, this));

    Simulator::Schedule(Seconds(1),
                        &TcpUrgentReadPointTestCase::SendSegmentFull,
                        this,
                        TcpHeader::SYN,
                        1000U,
                        0U,
                        static_cast<uint16_t>(4096),
                        false,
                        static_cast<uint16_t>(0),
                        0U);
    Simulator::Schedule(Seconds(2), &TcpUrgentReadPointTestCase::CompleteHandshake, this);
    Simulator::Schedule(Seconds(3), [this]() {
        SendSegmentFull(TcpHeader::ACK | TcpHeader::URG | TcpHeader::PSH,
                        1001,
                        (m_targetIsn + 1).GetValue(),
                        4096,
                        false,
                        m_urgent,
                        m_urgent);
    });

    Simulator::Stop(Seconds(5));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_beforeRead,
                          m_urgent,
                          "The urgent data was taken as consumed before it was read");
    NS_TEST_ASSERT_MSG_EQ(m_afterFirstRead,
                          m_urgent - m_firstRead,
                          "The pending urgent data did not shrink by the bytes read");
    NS_TEST_ASSERT_MSG_EQ(m_afterLastRead, 0, "Urgent data was still pending once read");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that only an urgent pointer ahead of the data read is notified
 *
 * @RFC{9293}, Section 3.10.7.4 signals the application of urgent data only if
 * the urgent pointer is in advance of the data consumed. Neither the
 * retransmission of urgent data the application read already, nor an
 * acknowledgment flagged as urgent whose pointer stays behind the read point,
 * announces new urgent data; a pointer which does reach beyond it does.
 */
class TcpUrgentSpuriousNotificationTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpUrgentSpuriousNotificationTestCase()
        : TcpCraftedSegmentTestCase("Only an urgent pointer ahead of the data read is notified")
    {
    }

  private:
    void DoRun() override;

    /**
     * Complete the handshake.
     */
    void CompleteHandshake();

    /**
     * Send a segment flagged as urgent.
     * @param seq The sequence number.
     * @param urgentPointer The urgent pointer.
     * @param payload The number of data bytes.
     */
    void SendUrgent(uint32_t seq, uint16_t urgentPointer, uint32_t payload);

    /**
     * Accept callback, hooking the callbacks on the new socket.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    /**
     * Urgent data callback of the accepted socket.
     * @param socket The socket holding urgent data.
     */
    void UrgentDataPending(Ptr<Socket> socket);

    SequenceNumber32 m_targetIsn{0};    //!< Initial sequence number of the target
    std::vector<uint32_t> m_notifiedAt; //!< Seconds at which urgent data was notified
};

void
TcpUrgentSpuriousNotificationTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->SetUrgentDataCallback(
        MakeCallback(&TcpUrgentSpuriousNotificationTestCase::UrgentDataPending, this));
    socket->SetRecvCallback(Callback<void, Ptr<Socket>>([](Ptr<Socket> s) {
        while (s->Recv())
        {
        }
    }));
}

void
TcpUrgentSpuriousNotificationTestCase::UrgentDataPending(Ptr<Socket> socket)
{
    m_notifiedAt.push_back(static_cast<uint32_t>(Simulator::Now().GetSeconds()));
}

void
TcpUrgentSpuriousNotificationTestCase::CompleteHandshake()
{
    NS_ABORT_MSG_IF(GetReplies().empty(), "The SYN was not answered");
    m_targetIsn = GetReplies().front().GetSequenceNumber();
    SendSegmentFull(TcpHeader::ACK, 1001, (m_targetIsn + 1).GetValue(), 4096);
}

void
TcpUrgentSpuriousNotificationTestCase::SendUrgent(uint32_t seq,
                                                  uint16_t urgentPointer,
                                                  uint32_t payload)
{
    SendSegmentFull(TcpHeader::ACK | TcpHeader::URG,
                    seq,
                    (m_targetIsn + 1).GetValue(),
                    4096,
                    false,
                    urgentPointer,
                    payload);
}

void
TcpUrgentSpuriousNotificationTestCase::DoRun()
{
    SetupTopology();
    m_target->SetAcceptCallback(
        MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
        MakeCallback(&TcpUrgentSpuriousNotificationTestCase::Accepted, this));

    Simulator::Schedule(Seconds(1),
                        &TcpUrgentSpuriousNotificationTestCase::SendSegmentFull,
                        this,
                        TcpHeader::SYN,
                        1000U,
                        0U,
                        static_cast<uint16_t>(4096),
                        false,
                        static_cast<uint16_t>(0),
                        0U);
    Simulator::Schedule(Seconds(2),
                        &TcpUrgentSpuriousNotificationTestCase::CompleteHandshake,
                        this);
    // The urgent data, which the application reads at once
    Simulator::Schedule(Seconds(3),
                        &TcpUrgentSpuriousNotificationTestCase::SendUrgent,
                        this,
                        1001U,
                        static_cast<uint16_t>(100),
                        100U);
    // Its retransmission, once read
    Simulator::Schedule(Seconds(4),
                        &TcpUrgentSpuriousNotificationTestCase::SendUrgent,
                        this,
                        1001U,
                        static_cast<uint16_t>(100),
                        100U);
    // An acknowledgment flagged as urgent, pointing at the read point
    Simulator::Schedule(Seconds(5),
                        &TcpUrgentSpuriousNotificationTestCase::SendUrgent,
                        this,
                        1101U,
                        static_cast<uint16_t>(0),
                        0U);
    // New urgent data announced beyond the read point
    Simulator::Schedule(Seconds(6),
                        &TcpUrgentSpuriousNotificationTestCase::SendUrgent,
                        this,
                        1101U,
                        static_cast<uint16_t>(50),
                        0U);

    Simulator::Stop(Seconds(8));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_notifiedAt.size(), 2, "Unexpected number of urgent notifications");
    if (m_notifiedAt.size() == 2)
    {
        NS_TEST_ASSERT_MSG_EQ(m_notifiedAt[0], 3, "The urgent data was not notified");
        NS_TEST_ASSERT_MSG_EQ(m_notifiedAt[1], 6, "The new urgent pointer was not notified");
    }

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that urgent data read long ago stays consumed
 *
 * The urgent point is compared with the read point modulo 2^32: once the
 * application has read more than 2^31 octets past urgent data it consumed,
 * a stale urgent point would compare as ahead of the read point again. The
 * urgent data has to be taken as consumed as soon as it is read.
 */
class TcpUrgentWrapTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpUrgentWrapTestCase()
        : TcpCraftedSegmentTestCase("Urgent data read stays consumed past 2^31 octets")
    {
    }

  private:
    void DoRun() override;

    /**
     * Complete the handshake, then send the urgent octet.
     */
    void CompleteHandshake();

    /**
     * Send the next segment of ordinary data, and schedule the following one.
     * @param seq The sequence number of the segment.
     * @param left The number of segments left to send.
     */
    void SendData(uint32_t seq, uint32_t left);

    /**
     * Accept callback, hooking the receive callback on the new socket.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    /**
     * Receive callback of the accepted socket, reading everything.
     * @param socket The accepted socket.
     */
    void Received(Ptr<Socket> socket);

    /// The payload of each data segment, filling a 65535 byte datagram
    static constexpr uint32_t SEGMENT_SIZE{65535 - 40};

    Ptr<Socket> m_accepted;          //!< The accepted socket
    SequenceNumber32 m_targetIsn{0}; //!< Initial sequence number of the target
    uint64_t m_bytesRead{0};         //!< Bytes read by the application
};

void
TcpUrgentWrapTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    m_accepted = socket;
    socket->SetRecvCallback(MakeCallback(&TcpUrgentWrapTestCase::Received, this));
}

void
TcpUrgentWrapTestCase::Received(Ptr<Socket> socket)
{
    while (Ptr<Packet> p = socket->Recv())
    {
        m_bytesRead += p->GetSize();
    }
}

void
TcpUrgentWrapTestCase::CompleteHandshake()
{
    NS_ABORT_MSG_IF(GetReplies().empty(), "The SYN was not answered");
    m_targetIsn = GetReplies().front().GetSequenceNumber();
    SendSegmentFull(TcpHeader::ACK, 1001, (m_targetIsn + 1).GetValue(), 65535);
    // A single urgent octet, which the application reads at once
    SendSegmentFull(TcpHeader::ACK | TcpHeader::URG,
                    1001,
                    (m_targetIsn + 1).GetValue(),
                    65535,
                    false,
                    1,
                    1);
    // Enough ordinary data to move the read point more than 2^31 past it
    uint32_t segments = (1U << 31) / SEGMENT_SIZE + 2;
    Simulator::Schedule(MilliSeconds(1), &TcpUrgentWrapTestCase::SendData, this, 1002U, segments);
}

void
TcpUrgentWrapTestCase::SendData(uint32_t seq, uint32_t left)
{
    SendSegmentFull(TcpHeader::ACK,
                    seq,
                    (m_targetIsn + 1).GetValue(),
                    65535,
                    false,
                    0,
                    SEGMENT_SIZE);
    if (left > 1)
    {
        Simulator::Schedule(MicroSeconds(10),
                            &TcpUrgentWrapTestCase::SendData,
                            this,
                            seq + SEGMENT_SIZE,
                            left - 1);
    }
}

void
TcpUrgentWrapTestCase::DoRun()
{
    SetupTopology();
    // Datagrams as large as IPv4 allows, so that few segments carry the data
    for (uint32_t i = 0; i < m_nodes.GetN(); ++i)
    {
        m_nodes.Get(i)->GetDevice(1)->SetMtu(65535);
    }
    m_target->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                MakeCallback(&TcpUrgentWrapTestCase::Accepted, this));

    Simulator::Schedule(Seconds(1),
                        &TcpUrgentWrapTestCase::SendSegmentFull,
                        this,
                        TcpHeader::SYN,
                        1000U,
                        0U,
                        static_cast<uint16_t>(65535),
                        false,
                        static_cast<uint16_t>(0),
                        0U);
    Simulator::Schedule(Seconds(2), &TcpUrgentWrapTestCase::CompleteHandshake, this);

    Simulator::Stop(Seconds(10));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_GT(m_bytesRead,
                          (1ULL << 31) + 1,
                          "The application did not read past 2^31 octets");
    NS_TEST_ASSERT_MSG_EQ(m_accepted->GetUrgentDataSize(),
                          0,
                          "Urgent data read long ago is pending again");

    m_accepted = nullptr;
    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the urgent field is processed with a zero receive window
 *
 * @RFC{9293}, Section 3.4 (MUST-66) requires the URG field of the incoming
 * segments to be processed even when the receive window is zero: the data of
 * the segment at RCV.NXT is dropped, but its urgent pointer is not.
 */
class TcpUrgentZeroWindowTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpUrgentZeroWindowTestCase()
        : TcpCraftedSegmentTestCase("The urgent field is processed with a zero window (MUST-66)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Complete the handshake, then fill the receive buffer.
     */
    void CompleteHandshake();

    /**
     * Send urgent data, which the closed window drops.
     */
    void SendUrgent();

    /**
     * Accept callback, hooking the urgent data callback on the new socket.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    /**
     * Urgent data callback of the accepted socket.
     * @param socket The socket holding urgent data.
     */
    void UrgentDataPending(Ptr<Socket> socket);

    Ptr<Socket> m_accepted;          //!< The accepted socket
    SequenceNumber32 m_targetIsn{0}; //!< Initial sequence number of the target
    uint32_t m_buffer{1000};         //!< Size of the receive buffer of the target
    uint16_t m_urgent{50};           //!< Size of the urgent data
    uint32_t m_notifications{0};     //!< Number of urgent data notifications
    uint32_t m_notifiedSize{0};      //!< Pending urgent bytes reported at the notification
    uint32_t m_available{0};         //!< Bytes the application can read at the end
};

void
TcpUrgentZeroWindowTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    m_accepted = socket;
    socket->SetUrgentDataCallback(
        MakeCallback(&TcpUrgentZeroWindowTestCase::UrgentDataPending, this));
}

void
TcpUrgentZeroWindowTestCase::UrgentDataPending(Ptr<Socket> socket)
{
    m_notifications++;
    m_notifiedSize = socket->GetUrgentDataSize();
}

void
TcpUrgentZeroWindowTestCase::CompleteHandshake()
{
    NS_ABORT_MSG_IF(GetReplies().empty(), "The SYN was not answered");
    m_targetIsn = GetReplies().front().GetSequenceNumber();
    SendSegmentFull(TcpHeader::ACK, 1001, (m_targetIsn + 1).GetValue(), 4096);
    // The application reads nothing, so this closes the window
    SendSegmentFull(TcpHeader::ACK, 1001, (m_targetIsn + 1).GetValue(), 4096, false, 0, m_buffer);
}

void
TcpUrgentZeroWindowTestCase::SendUrgent()
{
    SendSegmentFull(TcpHeader::ACK | TcpHeader::URG,
                    1001 + m_buffer,
                    (m_targetIsn + 1).GetValue(),
                    4096,
                    false,
                    m_urgent,
                    m_urgent);
}

void
TcpUrgentZeroWindowTestCase::DoRun()
{
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(m_buffer));
    SetupTopology();
    m_target->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                MakeCallback(&TcpUrgentZeroWindowTestCase::Accepted, this));

    Simulator::Schedule(Seconds(1),
                        &TcpUrgentZeroWindowTestCase::SendSegmentFull,
                        this,
                        TcpHeader::SYN,
                        1000U,
                        0U,
                        static_cast<uint16_t>(4096),
                        false,
                        static_cast<uint16_t>(0),
                        0U);
    Simulator::Schedule(Seconds(2), &TcpUrgentZeroWindowTestCase::CompleteHandshake, this);
    Simulator::Schedule(Seconds(3), &TcpUrgentZeroWindowTestCase::SendUrgent, this);
    Simulator::Schedule(Seconds(4), [this]() { m_available = m_accepted->GetRxAvailable(); });

    Simulator::Stop(Seconds(5));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_available, m_buffer, "The receive window did not close");
    NS_TEST_ASSERT_MSG_EQ(m_notifications,
                          1,
                          "The urgent pointer was not processed with a zero window");
    // Everything up to the urgent pointer is still to be read
    NS_TEST_ASSERT_MSG_EQ(m_notifiedSize,
                          m_buffer + m_urgent,
                          "Unexpected amount of pending urgent data");

    m_accepted = nullptr;
    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the zero window probes carry the urgent field and PSH
 *
 * @RFC{9293}, Section 3.8.5 carries the urgent field in every segment sent
 * while the urgent pointer is ahead of it, and the PSH flag is set on the
 * segment holding the last octet queued (MUST-61). The one octet probing a
 * zero window is no exception.
 */
class TcpZeroWindowProbeUrgentTestCase : public TestCase
{
  public:
    TcpZeroWindowProbeUrgentTestCase()
        : TestCase("The zero window probes carry the urgent field and PSH")
    {
    }

  private:
    void DoRun() override;

    /**
     * Run a connection whose receiver reads nothing, sending urgent data
     * beyond its receive buffer.
     *
     * @param beyond The number of urgent octets beyond the receive buffer.
     */
    void Run(uint32_t beyond);

    /**
     * Tx trace sink of the sending socket, recording the probes.
     * @param packet The payload.
     * @param header The TCP header.
     * @param socket The sending socket.
     */
    void SegmentSent(Ptr<const Packet> packet,
                     const TcpHeader& header,
                     Ptr<const TcpSocketBase> socket);

    std::vector<TcpHeader> m_probes; //!< Headers of the one octet probes
    uint32_t m_buffer{1000};         //!< Receive buffer of the receiver
};

void
TcpZeroWindowProbeUrgentTestCase::SegmentSent(Ptr<const Packet> packet,
                                              const TcpHeader& header,
                                              Ptr<const TcpSocketBase> socket)
{
    if (packet->GetSize() == 1)
    {
        m_probes.push_back(header);
    }
}

void
TcpZeroWindowProbeUrgentTestCase::Run(uint32_t beyond)
{
    m_probes.clear();
    // The receive buffer holds a whole number of segments, so that the window
    // closes rather than leaving a fraction of a segment open
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(m_buffer));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(m_buffer / 2));
    Config::SetDefault("ns3::TcpSocketBase::Timestamp", BooleanValue(false));

    NodeContainer nodes;
    nodes.Create(2);

    SimpleNetDeviceHelper devHelper;
    NetDeviceContainer devices = devHelper.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t port = 9921;

    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->TraceConnectWithoutContext(
        "Tx",
        MakeCallback(&TcpZeroWindowProbeUrgentTestCase::SegmentSent, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));

    Simulator::Schedule(Seconds(1), [client, this, beyond]() {
        client->Send(Create<Packet>(m_buffer + beyond), Socket::MSG_FLAG_OOB);
    });

    Simulator::Stop(Seconds(20));
    Simulator::Run();
    Simulator::Destroy();
    Config::Reset();
}

void
TcpZeroWindowProbeUrgentTestCase::DoRun()
{
    // Urgent data reaching far beyond the probe
    const uint32_t beyond = 2000;
    Run(beyond);
    NS_TEST_ASSERT_MSG_GT(m_probes.size(), 0, "The zero window was not probed");
    for (const auto& probe : m_probes)
    {
        NS_TEST_ASSERT_MSG_EQ((probe.GetFlags() & TcpHeader::URG),
                              TcpHeader::URG,
                              "A probe within the urgent data was not flagged as urgent");
        NS_TEST_ASSERT_MSG_EQ(probe.GetUrgentPointer(),
                              beyond,
                              "A probe did not point past the urgent data");
        NS_TEST_ASSERT_MSG_EQ((probe.GetFlags() & TcpHeader::PSH),
                              0,
                              "A probe followed by queued data was pushed");
    }

    // The probe octet is the last urgent octet, and the last one queued
    Run(1);
    NS_TEST_ASSERT_MSG_GT(m_probes.size(), 0, "The zero window was not probed");
    for (const auto& probe : m_probes)
    {
        NS_TEST_ASSERT_MSG_EQ((probe.GetFlags() & TcpHeader::URG),
                              TcpHeader::URG,
                              "A probe holding urgent data was not flagged as urgent");
        NS_TEST_ASSERT_MSG_EQ(probe.GetUrgentPointer(),
                              1,
                              "A probe did not point past the urgent data");
        NS_TEST_ASSERT_MSG_EQ((probe.GetFlags() & TcpHeader::PSH),
                              TcpHeader::PSH,
                              "A probe holding the last octet queued was not pushed");
    }
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test urgent data reaching more than 65535 octets ahead
 *
 * @RFC{9293}, Section 3.8.5 supports a sequence of urgent data of any length
 * (MUST-31), while the urgent pointer is a 16 bit offset. Each segment
 * before the urgent point has to carry the URG flag, with its pointer
 * saturated while the urgent point lies further ahead, so that the urgent
 * point seen by the receiver only moves forward and ends at the true one;
 * the segments past the urgent point are not flagged.
 */
class TcpLongUrgentDataTestCase : public TestCase
{
  public:
    TcpLongUrgentDataTestCase()
        : TestCase("Urgent data more than 65535 octets long (MUST-31)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Tx trace sink of the sending socket.
     * @param packet The payload.
     * @param header The TCP header.
     * @param socket The sending socket.
     */
    void SegmentSent(Ptr<const Packet> packet,
                     const TcpHeader& header,
                     Ptr<const TcpSocketBase> socket);

    /**
     * Accept callback, keeping the accepted socket.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    Ptr<Socket> m_accepted;                //!< The accepted socket
    SequenceNumber32 m_isn{0};             //!< Initial sequence number of the sender
    bool m_isnKnown{false};                //!< Whether the SYN was seen
    uint32_t m_normal{1000};               //!< Ordinary data before and after the urgent data
    uint32_t m_urgent{200000};             //!< Size of the urgent data
    uint32_t m_unflaggedBefore{0};         //!< Segments before the urgent point without URG
    uint32_t m_flaggedAfter{0};            //!< Segments past the urgent point with URG
    uint32_t m_wrongPointer{0};            //!< Segments with a wrong urgent pointer
    uint32_t m_receded{0};                 //!< Segments whose urgent point receded
    uint32_t m_saturated{0};               //!< Segments with a saturated urgent pointer
    SequenceNumber32 m_lastUrgentPoint{0}; //!< Last urgent point sent
};

void
TcpLongUrgentDataTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    m_accepted = socket;
}

void
TcpLongUrgentDataTestCase::SegmentSent(Ptr<const Packet> packet,
                                       const TcpHeader& header,
                                       Ptr<const TcpSocketBase> socket)
{
    if (header.GetFlags() & TcpHeader::SYN)
    {
        m_isn = header.GetSequenceNumber();
        // The urgent points are compared in the wrap aware arithmetic, from
        // the initial sequence number
        m_lastUrgentPoint = m_isn;
        m_isnKnown = true;
        return;
    }
    if (packet->GetSize() == 0 || !m_isnKnown)
    {
        return;
    }

    SequenceNumber32 seq = header.GetSequenceNumber();
    SequenceNumber32 point = m_isn + 1 + m_normal + m_urgent;
    bool urgent = header.GetFlags() & TcpHeader::URG;
    if (seq >= point)
    {
        m_flaggedAfter += urgent ? 1 : 0;
        return;
    }
    if (!urgent)
    {
        m_unflaggedBefore++;
        return;
    }
    uint32_t expected = std::min<uint32_t>(point - seq, 65535);
    if (header.GetUrgentPointer() != expected)
    {
        m_wrongPointer++;
    }
    if (header.GetUrgentPointer() == 65535)
    {
        m_saturated++;
    }
    SequenceNumber32 urgentPoint = seq + header.GetUrgentPointer();
    if (urgentPoint < m_lastUrgentPoint)
    {
        m_receded++;
    }
    m_lastUrgentPoint = std::max(m_lastUrgentPoint, urgentPoint);
}

void
TcpLongUrgentDataTestCase::DoRun()
{
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(300000));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(300000));

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

    const uint16_t port = 9922;

    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpLongUrgentDataTestCase::Accepted, this));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->TraceConnectWithoutContext("Tx",
                                       MakeCallback(&TcpLongUrgentDataTestCase::SegmentSent, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));

    Simulator::Schedule(Seconds(1), [this, client]() {
        client->Send(Create<Packet>(m_normal), 0);
        client->Send(Create<Packet>(m_urgent), Socket::MSG_FLAG_OOB);
        client->Send(Create<Packet>(m_normal), 0);
    });

    Simulator::Stop(Seconds(30));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_unflaggedBefore, 0, "A segment before the urgent point lacks URG");
    NS_TEST_ASSERT_MSG_EQ(m_flaggedAfter, 0, "A segment past the urgent point carries URG");
    NS_TEST_ASSERT_MSG_EQ(m_wrongPointer, 0, "Unexpected urgent pointer");
    NS_TEST_ASSERT_MSG_GT(m_saturated, 0, "The urgent pointer was never saturated");
    NS_TEST_ASSERT_MSG_EQ(m_receded, 0, "The urgent point moved backwards");
    NS_TEST_ASSERT_MSG_EQ(m_lastUrgentPoint,
                          m_isn + 1 + m_normal + m_urgent,
                          "The true urgent point was never sent");
    // Nothing was read: everything up to the true urgent point is pending
    NS_TEST_ASSERT_MSG_EQ(m_accepted->GetUrgentDataSize(),
                          m_normal + m_urgent,
                          "The receiver did not learn the true urgent point");
    NS_TEST_ASSERT_MSG_EQ(m_accepted->GetRxAvailable(),
                          2 * m_normal + m_urgent,
                          "The transfer did not complete");

    m_accepted = nullptr;
    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Base class of the keep-alive test cases
 *
 * Builds a two node topology whose client probes the server, records the
 * keep-alives the client sends and how its connection ends. A keep-alive is
 * a segment holding no data whose sequence number is the one of the last
 * octet sent, SYN and FIN included.
 */
class TcpKeepAliveBaseTestCase : public TestCase
{
  public:
    /**
     * Constructor.
     * @param name The name of the test case.
     */
    TcpKeepAliveBaseTestCase(std::string name)
        : TestCase(name)
    {
    }

  protected:
    /**
     * Build the topology, with a server listening, and a client with the
     * keep-alives enabled which connects to it. The IPv6 topology turns the
     * duplicate address detection off through the defaults, which the
     * caller resets with Config::Reset() once done.
     *
     * @param ipv6 Whether to run over IPv6.
     * @param keepAliveTime The idle time before the first keep-alive.
     */
    void Setup(bool ipv6 = false, Time keepAliveTime = Seconds(2));

    /**
     * Stop the server from answering, by dropping everything it receives.
     */
    void BreakThePath();

    /**
     * Tx trace sink of the client, counting the keep-alives.
     * @param packet The segment sent.
     * @param header The TCP header of the segment.
     * @param socket The sending socket.
     */
    void SegmentSent(Ptr<const Packet> packet,
                     const TcpHeader& header,
                     Ptr<const TcpSocketBase> socket);

    /**
     * State trace sink of the client.
     * @param oldState The previous state.
     * @param newState The new state.
     */
    void StateChanged(TcpSocket::TcpStates_t oldState, TcpSocket::TcpStates_t newState);

    /**
     * Accept callback, keeping the accepted socket.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    NodeContainer m_nodes;                             //!< The client and the server node
    NetDeviceContainer m_devices;                      //!< The devices of the two nodes
    Ptr<Socket> m_client;                              //!< The probing socket
    Ptr<Socket> m_accepted;                            //!< The accepted socket of the server
    std::vector<Time> m_probes;                        //!< Instants of the keep-alives sent
    std::optional<SequenceNumber32> m_highestSent;     //!< Next sequence number of the client
    uint32_t m_normalCloses{0};                        //!< Normal close notifications
    uint32_t m_errorCloses{0};                         //!< Error close notifications
    TcpSocket::TcpStates_t m_state{TcpSocket::CLOSED}; //!< State of the client
    bool m_resetSent{false};                           //!< Whether the client sent a RST
};

void
TcpKeepAliveBaseTestCase::Setup(bool ipv6, Time keepAliveTime)
{
    m_nodes = NodeContainer();
    m_nodes.Create(2);
    m_probes.clear();
    m_highestSent.reset();
    m_normalCloses = 0;
    m_errorCloses = 0;
    m_state = TcpSocket::CLOSED;
    m_resetSent = false;
    m_accepted = nullptr;

    if (ipv6)
    {
        // The addresses are usable at once, without waiting for the
        // duplicate address detection
        Config::SetDefault("ns3::Icmpv6L4Protocol::DAD", BooleanValue(false));
    }

    SimpleNetDeviceHelper devHelper;
    m_devices = devHelper.Install(m_nodes);

    InternetStackHelper stack;
    stack.Install(m_nodes);

    const uint16_t port = 9923;
    Address serverAddress;
    Address anyAddress;
    if (ipv6)
    {
        Ipv6AddressHelper address;
        address.SetBase(Ipv6Address("2001:db8::"), Ipv6Prefix(64));
        Ipv6InterfaceContainer interfaces = address.Assign(m_devices);
        interfaces.SetForwarding(0, false);
        serverAddress = Inet6SocketAddress(interfaces.GetAddress(1, 1), port);
        anyAddress = Inet6SocketAddress(Ipv6Address::GetAny(), port);
    }
    else
    {
        Ipv4AddressHelper address;
        address.SetBase("10.1.1.0", "255.255.255.0");
        Ipv4InterfaceContainer interfaces = address.Assign(m_devices);
        serverAddress = InetSocketAddress(interfaces.GetAddress(1), port);
        anyAddress = InetSocketAddress(Ipv4Address::GetAny(), port);
    }

    Ptr<Socket> server = Socket::CreateSocket(m_nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(anyAddress);
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              MakeCallback(&TcpKeepAliveBaseTestCase::Accepted, this));

    m_client = Socket::CreateSocket(m_nodes.Get(0), TcpSocketFactory::GetTypeId());
    if (ipv6)
    {
        m_client->Bind6();
    }
    else
    {
        m_client->Bind();
    }
    m_client->SetAttribute("KeepAlive", BooleanValue(true));
    m_client->SetAttribute("KeepAliveTime", TimeValue(keepAliveTime));
    m_client->SetAttribute("KeepAliveInterval", TimeValue(Seconds(1)));
    m_client->SetAttribute("KeepAliveRetries", UintegerValue(3));
    m_client->SetCloseCallbacks(
        Callback<void, Ptr<Socket>>([this](Ptr<Socket>) { m_normalCloses++; }),
        Callback<void, Ptr<Socket>>([this](Ptr<Socket>) { m_errorCloses++; }));
    m_client->TraceConnectWithoutContext(
        "State",
        MakeCallback(&TcpKeepAliveBaseTestCase::StateChanged, this));
    m_client->TraceConnectWithoutContext(
        "Tx",
        MakeCallback(&TcpKeepAliveBaseTestCase::SegmentSent, this));
    m_client->Connect(serverAddress);
}

void
TcpKeepAliveBaseTestCase::BreakThePath()
{
    Ptr<RateErrorModel> error = CreateObject<RateErrorModel>();
    error->SetAttribute("ErrorRate", DoubleValue(1.0));
    error->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
    m_devices.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(error));
}

void
TcpKeepAliveBaseTestCase::SegmentSent(Ptr<const Packet> packet,
                                      const TcpHeader& header,
                                      Ptr<const TcpSocketBase> socket)
{
    if (header.GetFlags() & TcpHeader::RST)
    {
        m_resetSent = true;
        return;
    }
    uint32_t length = packet->GetSize();
    if (header.GetFlags() & (TcpHeader::SYN | TcpHeader::FIN))
    {
        length++;
    }
    if (length > 0)
    {
        SequenceNumber32 end = header.GetSequenceNumber() + length;
        m_highestSent = m_highestSent ? std::max(*m_highestSent, end) : end;
        return;
    }
    if (header.GetFlags() == TcpHeader::ACK && m_highestSent &&
        header.GetSequenceNumber() + 1 == *m_highestSent)
    {
        m_probes.push_back(Simulator::Now());
    }
}

void
TcpKeepAliveBaseTestCase::StateChanged(TcpSocket::TcpStates_t oldState,
                                       TcpSocket::TcpStates_t newState)
{
    m_state = newState;
}

void
TcpKeepAliveBaseTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    m_accepted = socket;
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the keep-alives of a half closed connection
 *
 * A connection whose peer closed its side waits for the application in
 * CLOSE_WAIT: if the peer is gone without a word, nothing ever tells. The
 * keep-alives probe it in CLOSE_WAIT, like Linux does.
 */
class TcpKeepAliveHalfClosedTestCase : public TcpKeepAliveBaseTestCase
{
  public:
    TcpKeepAliveHalfClosedTestCase()
        : TcpKeepAliveBaseTestCase("Keep-alives of a half closed connection")
    {
    }

  private:
    void DoRun() override;

    /**
     * Run a half closed connection, whose server stops answering later on.
     *
     * @param clientCloses Whether the client closes, reaching FIN_WAIT_2,
     *                     rather than the server, leaving the client in
     *                     CLOSE_WAIT.
     */
    void Run(bool clientCloses);
};

void
TcpKeepAliveHalfClosedTestCase::Run(bool clientCloses)
{
    Setup();
    TcpSocket::TcpStates_t halfClosed =
        clientCloses ? TcpSocket::FIN_WAIT_2 : TcpSocket::CLOSE_WAIT;
    std::string name = TcpSocket::TcpStateName[halfClosed];

    Simulator::Schedule(Seconds(1), [this, clientCloses]() {
        if (clientCloses)
        {
            m_client->Close();
        }
        else
        {
            m_accepted->Close();
        }
    });
    TcpSocket::TcpStates_t stateBeforeBreak = TcpSocket::CLOSED;
    size_t probesBeforeBreak = 0;
    Simulator::Schedule(Seconds(8), [this, &stateBeforeBreak, &probesBeforeBreak]() {
        stateBeforeBreak = m_state;
        probesBeforeBreak = m_probes.size();
        BreakThePath();
    });

    Simulator::Stop(Seconds(20));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_GT(probesBeforeBreak, 1, "No keep-alive was sent in " << name);
    NS_TEST_ASSERT_MSG_EQ(stateBeforeBreak,
                          halfClosed,
                          "The connection whose peer answers did not stay in " << name);
    NS_TEST_ASSERT_MSG_EQ(m_probes.size() - probesBeforeBreak,
                          3,
                          "Unexpected number of unanswered keep-alives in " << name);
    NS_TEST_ASSERT_MSG_EQ(m_state,
                          TcpSocket::CLOSED,
                          "The connection whose peer is gone stayed in " << name);
    NS_TEST_ASSERT_MSG_EQ(m_resetSent, true, "The dropped connection was not reset");

    m_client = nullptr;
    m_accepted = nullptr;
    Simulator::Destroy();
}

void
TcpKeepAliveHalfClosedTestCase::DoRun()
{
    Run(false);
    Run(true);
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that a new keep-alive idle time applies to the running timer
 *
 * The idle time before the first keep-alive is set through an attribute,
 * which an application may change on an established connection: the idle
 * period under way has to be timed with the new value, rather than the one
 * the timer was started with.
 */
class TcpKeepAliveTimeChangeTestCase : public TcpKeepAliveBaseTestCase
{
  public:
    TcpKeepAliveTimeChangeTestCase()
        : TcpKeepAliveBaseTestCase("A new keep-alive idle time applies to the running timer")
    {
    }

  private:
    void DoRun() override;
};

void
TcpKeepAliveTimeChangeTestCase::DoRun()
{
    Setup(false, Hours(2));
    Simulator::Schedule(Seconds(2), [this]() {
        m_client->SetAttribute("KeepAliveTime", TimeValue(Seconds(1)));
    });

    Simulator::Stop(Seconds(10));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_GT(m_probes.size(), 0, "The new idle time was not applied");
    if (!m_probes.empty())
    {
        NS_TEST_ASSERT_MSG_EQ(m_probes.front(),
                              Seconds(3),
                              "The first keep-alive did not follow the new idle time");
    }

    m_client = nullptr;
    m_accepted = nullptr;
    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that a keep-alive leaves the delayed acknowledgment pending
 *
 * A keep-alive carries an acknowledgment, but lies outside of the window of
 * the peer, which drops it without looking at that acknowledgment (@RFC{9293},
 * Section 3.10.7.4). An acknowledgment delayed when the keep-alive leaves must
 * still be sent, or the peer retransmits the data it acknowledges.
 */
class TcpKeepAliveDelayedAckTestCase : public TcpKeepAliveBaseTestCase
{
  public:
    TcpKeepAliveDelayedAckTestCase()
        : TcpKeepAliveBaseTestCase("A keep-alive leaves the delayed acknowledgment pending")
    {
    }

  private:
    void DoRun() override;

    uint32_t m_dataSegments{0}; //!< Data segments sent by the server
};

void
TcpKeepAliveDelayedAckTestCase::DoRun()
{
    // The keep-alive leaves while the acknowledgment of the data is delayed
    Setup(false, MilliSeconds(100));
    m_client->SetAttribute("DelAckTimeout", TimeValue(MilliSeconds(500)));
    m_client->SetAttribute("DelAckCount", UintegerValue(2));

    Simulator::Schedule(Seconds(1), [this]() {
        m_accepted->TraceConnectWithoutContext(
            "Tx",
            Callback<void, Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>>(
                [this](Ptr<const Packet> p, const TcpHeader&, Ptr<const TcpSocketBase>) {
                    m_dataSegments += (p->GetSize() > 0) ? 1 : 0;
                }));
        m_accepted->Send(Create<Packet>(100), 0);
    });

    Simulator::Stop(Seconds(3));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_GT(m_probes.size(), 0, "No keep-alive was sent");
    NS_TEST_ASSERT_MSG_EQ(m_dataSegments, 1, "The data was retransmitted");

    m_client = nullptr;
    m_accepted = nullptr;
    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the keep-alives over the life of a connection
 *
 * Covers the keep-alives of @RFC{9293}, Section 3.8.4 beyond an idle
 * connection: a peer gone silent is given up on after the configured number
 * of keep-alives with a RST and a single error notification, over IPv4 as
 * over IPv6; a peer which answers keeps the connection up through any number
 * of keep-alives; no keep-alive is sent while data is outstanding (MUST-26),
 * nor while the persist timer probes a zero window; the keep-alives follow
 * SetKeepAlive() on an established connection; and a socket released with a
 * keep-alive pending is torn down cleanly.
 */
class TcpKeepAliveScenariosTestCase : public TcpKeepAliveBaseTestCase
{
  public:
    TcpKeepAliveScenariosTestCase()
        : TcpKeepAliveBaseTestCase("Keep-alives over the life of a connection")
    {
    }

  private:
    void DoRun() override;

    /**
     * Run an idle connection whose peer answers, then stops answering.
     * @param ipv6 Whether to run over IPv6.
     */
    void RunSilentPeer(bool ipv6);

    /**
     * Run a connection whose data is never acknowledged.
     */
    void RunOutstandingData();

    /**
     * Run a connection whose peer advertises a zero window.
     */
    void RunZeroWindow();

    /**
     * Run a connection whose keep-alives are turned on and off once established.
     */
    void RunToggle();

    /**
     * Release a socket with a keep-alive pending.
     */
    void RunRelease();

    /**
     * Release the test references to the sockets and destroy the simulator.
     */
    void Teardown();
};

void
TcpKeepAliveScenariosTestCase::Teardown()
{
    m_client = nullptr;
    m_accepted = nullptr;
    Simulator::Destroy();
    Config::Reset();
}

void
TcpKeepAliveScenariosTestCase::RunSilentPeer(bool ipv6)
{
    std::string family = ipv6 ? " over IPv6" : " over IPv4";
    Setup(ipv6);
    size_t answered = 0;
    Simulator::Schedule(Seconds(15), [this, &answered]() {
        answered = m_probes.size();
        BreakThePath();
    });

    Simulator::Stop(Seconds(30));
    Simulator::Run();

    // Each answer starts the count anew, so the answered keep-alives outnumber
    // the retries without the connection being dropped
    NS_TEST_ASSERT_MSG_GT(answered, 3, "Too few answered keep-alives" << family);
    NS_TEST_ASSERT_MSG_EQ(m_probes.size() - answered,
                          3,
                          "Unexpected number of unanswered keep-alives" << family);
    NS_TEST_ASSERT_MSG_EQ(m_state, TcpSocket::CLOSED, "The silent peer was not given up on");
    NS_TEST_ASSERT_MSG_EQ(m_resetSent, true, "The connection was not reset" << family);
    NS_TEST_ASSERT_MSG_EQ(m_errorCloses, 1, "The abort was not notified once" << family);
    NS_TEST_ASSERT_MSG_EQ(m_normalCloses, 0, "The abort was notified as a close" << family);
    Teardown();
}

void
TcpKeepAliveScenariosTestCase::RunOutstandingData()
{
    Setup(false, Seconds(1));
    Simulator::Schedule(Seconds(1), [this]() {
        BreakThePath();
        m_client->Send(Create<Packet>(100), 0);
    });

    Simulator::Stop(Seconds(20));
    Simulator::Run();

    // The retransmissions probe the peer while the data is outstanding
    NS_TEST_ASSERT_MSG_EQ(m_probes.size(), 0, "A keep-alive was sent with data outstanding");
    Teardown();
}

void
TcpKeepAliveScenariosTestCase::RunZeroWindow()
{
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(1000));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(500));
    Config::SetDefault("ns3::TcpSocketBase::Timestamp", BooleanValue(false));
    Setup(false, Seconds(1));

    uint32_t windowProbes = 0;
    m_client->TraceConnectWithoutContext(
        "Tx",
        Callback<void, Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>>(
            [&windowProbes](Ptr<const Packet> p, const TcpHeader&, Ptr<const TcpSocketBase>) {
                windowProbes += (p->GetSize() == 1) ? 1 : 0;
            }));
    uint32_t received = 0;
    size_t whileWaiting = 0;
    Simulator::Schedule(Seconds(1), [this]() { m_client->Send(Create<Packet>(3000), 0); });
    // The server reads nothing for a while, then everything
    Simulator::Schedule(Seconds(30), [this, &received, &whileWaiting]() {
        whileWaiting = m_probes.size();
        m_accepted->SetRecvCallback(Callback<void, Ptr<Socket>>([&received](Ptr<Socket> s) {
            while (Ptr<Packet> p = s->Recv())
            {
                received += p->GetSize();
            }
        }));
        while (Ptr<Packet> p = m_accepted->Recv())
        {
            received += p->GetSize();
        }
    });

    Simulator::Stop(Seconds(60));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_GT(windowProbes, 0, "The zero window was not probed");
    NS_TEST_ASSERT_MSG_EQ(whileWaiting, 0, "A keep-alive was sent with data waiting");
    NS_TEST_ASSERT_MSG_EQ(received, 3000, "The transfer did not complete");
    NS_TEST_ASSERT_MSG_EQ(m_state, TcpSocket::ESTABLISHED, "The connection did not stay up");
    Teardown();
}

void
TcpKeepAliveScenariosTestCase::RunToggle()
{
    Setup(false, Seconds(1));
    m_client->SetAttribute("KeepAlive", BooleanValue(false));
    size_t whileOff = 0;
    size_t whileOn = 0;
    Simulator::Schedule(Seconds(5), [this, &whileOff]() {
        whileOff = m_probes.size();
        m_client->SetAttribute("KeepAlive", BooleanValue(true));
    });
    Simulator::Schedule(Seconds(10), [this, &whileOn]() {
        m_client->SetAttribute("KeepAlive", BooleanValue(false));
        whileOn = m_probes.size();
    });

    Simulator::Stop(Seconds(20));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(whileOff, 0, "A keep-alive was sent before they were turned on");
    NS_TEST_ASSERT_MSG_GT(whileOn, 0, "No keep-alive followed turning them on");
    NS_TEST_ASSERT_MSG_EQ(m_probes.size(), whileOn, "A keep-alive followed turning them off");
    Teardown();
}

void
TcpKeepAliveScenariosTestCase::RunRelease()
{
    Setup(false, Seconds(1));
    // The keep-alive timer is pending when the simulation ends and the
    // test lets go of the socket
    Simulator::Stop(Seconds(1.5));
    Simulator::Run();
    NS_TEST_ASSERT_MSG_EQ(m_state, TcpSocket::ESTABLISHED, "The connection did not open");
    Teardown();
}

void
TcpKeepAliveScenariosTestCase::DoRun()
{
    RunSilentPeer(false);
    RunSilentPeer(true);
    RunOutstandingData();
    RunZeroWindow();
    RunToggle();
    RunRelease();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test a keep-alive answered with a reset
 *
 * A peer which lost the connection, after a reboot for instance, answers a
 * keep-alive with a RST: the connection is aborted, and the application is
 * told of the abort exactly once.
 */
class TcpKeepAliveResetAnswerTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpKeepAliveResetAnswerTestCase()
        : TcpCraftedSegmentTestCase("A keep-alive answered with a reset aborts the connection")
    {
    }

  private:
    void DoRun() override;

    /**
     * Complete the handshake.
     */
    void CompleteHandshake();

    /**
     * Answer the keep-alive with a reset.
     */
    void Reset();

    /**
     * Accept callback, hooking the callbacks on the new socket.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    SequenceNumber32 m_targetIsn{0};                   //!< Initial sequence number of the target
    bool m_probed{false};                              //!< Whether a keep-alive came in
    uint32_t m_normalCloses{0};                        //!< Normal close notifications
    uint32_t m_errorCloses{0};                         //!< Error close notifications
    TcpSocket::TcpStates_t m_state{TcpSocket::CLOSED}; //!< State of the accepted socket
};

void
TcpKeepAliveResetAnswerTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    m_state = TcpSocket::ESTABLISHED;
    socket->SetCloseCallbacks(
        Callback<void, Ptr<Socket>>([this](Ptr<Socket>) { m_normalCloses++; }),
        Callback<void, Ptr<Socket>>([this](Ptr<Socket>) { m_errorCloses++; }));
    socket->TraceConnectWithoutContext(
        "State",
        Callback<void, TcpSocket::TcpStates_t, TcpSocket::TcpStates_t>(
            [this](TcpSocket::TcpStates_t, TcpSocket::TcpStates_t newState) {
                m_state = newState;
            }));
}

void
TcpKeepAliveResetAnswerTestCase::CompleteHandshake()
{
    NS_ABORT_MSG_IF(GetReplies().empty(), "The SYN was not answered");
    m_targetIsn = GetReplies().front().GetSequenceNumber();
    SendSegmentFull(TcpHeader::ACK, 1001, (m_targetIsn + 1).GetValue(), 4096);
}

void
TcpKeepAliveResetAnswerTestCase::Reset()
{
    for (const auto& reply : GetReplies())
    {
        if (reply.GetFlags() == TcpHeader::ACK && reply.GetSequenceNumber() == m_targetIsn)
        {
            m_probed = true;
        }
    }
    SendSegmentFull(TcpHeader::RST, 1001, 0, 0);
}

void
TcpKeepAliveResetAnswerTestCase::DoRun()
{
    Config::SetDefault("ns3::TcpSocket::KeepAlive", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocket::KeepAliveTime", TimeValue(Seconds(2)));
    SetupTopology();
    Config::Reset();
    m_target->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                MakeCallback(&TcpKeepAliveResetAnswerTestCase::Accepted, this));

    Simulator::Schedule(Seconds(1),
                        &TcpKeepAliveResetAnswerTestCase::SendSegmentFull,
                        this,
                        TcpHeader::SYN,
                        1000U,
                        0U,
                        static_cast<uint16_t>(4096),
                        false,
                        static_cast<uint16_t>(0),
                        0U);
    Simulator::Schedule(Seconds(2), &TcpKeepAliveResetAnswerTestCase::CompleteHandshake, this);
    // The keep-alive leaves two seconds after the handshake
    Simulator::Schedule(Seconds(4.5), &TcpKeepAliveResetAnswerTestCase::Reset, this);

    Simulator::Stop(Seconds(20));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_probed, true, "No keep-alive was sent");
    NS_TEST_ASSERT_MSG_EQ(m_state, TcpSocket::CLOSED, "The reset did not abort the connection");
    NS_TEST_ASSERT_MSG_EQ(m_errorCloses, 1, "The abort was not notified once");
    NS_TEST_ASSERT_MSG_EQ(m_normalCloses, 0, "The abort was notified as a close");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Object aggregated to a socket, telling when the socket is deleted
 */
class TcpSocketLifetimeProbe : public Object
{
  public:
    /**
     * @brief Get the type ID.
     * @return The object TypeId.
     */
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::TcpSocketLifetimeProbe")
                                .SetParent<Object>()
                                .SetGroupName("Internet")
                                .AddConstructor<TcpSocketLifetimeProbe>();
        return tid;
    }

    ~TcpSocketLifetimeProbe() override
    {
        if (m_deleted)
        {
            *m_deleted = true;
        }
    }

    bool* m_deleted{nullptr}; //!< Set when the socket, and this probe with it, is deleted
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test a socket let go of when its keep-alives give up on the peer
 *
 * The application may release its last reference to the socket when told of
 * the abort, as ThreeGppHttpServer does: the socket is then deleted as its
 * endpoint is deallocated, from within the keep-alive timer, which must not
 * touch it afterwards. Run under a memory checker to catch an access to the
 * deleted socket.
 */
class TcpKeepAliveReleaseOnAbortTestCase : public TcpKeepAliveBaseTestCase
{
  public:
    TcpKeepAliveReleaseOnAbortTestCase()
        : TcpKeepAliveBaseTestCase("A socket let go of when its keep-alives give up on the peer")
    {
    }

  private:
    void DoRun() override;
};

void
TcpKeepAliveReleaseOnAbortTestCase::DoRun()
{
    Setup(false, Seconds(1));
    bool deleted = false;
    Ptr<TcpSocketLifetimeProbe> probe = CreateObject<TcpSocketLifetimeProbe>();
    probe->m_deleted = &deleted;
    m_client->AggregateObject(probe);
    probe = nullptr;
    // The only reference of the application is let go of upon the abort
    m_client->SetCloseCallbacks(Callback<void, Ptr<Socket>>([this](Ptr<Socket>) {
                                    m_normalCloses++;
                                    m_client = nullptr;
                                }),
                                Callback<void, Ptr<Socket>>([this](Ptr<Socket>) {
                                    m_errorCloses++;
                                    m_client = nullptr;
                                }));
    Simulator::Schedule(Seconds(2), [this]() { BreakThePath(); });

    Simulator::Stop(Seconds(20));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_resetSent, true, "The silent peer was not given up on");
    NS_TEST_ASSERT_MSG_EQ(m_state, TcpSocket::CLOSED, "The connection was not closed");
    NS_TEST_ASSERT_MSG_EQ(m_errorCloses, 1, "The abort was not notified once");
    NS_TEST_ASSERT_MSG_EQ(m_normalCloses, 0, "The abort was notified as a close");
    NS_TEST_ASSERT_MSG_EQ(deleted, true, "The socket let go of was not deleted");

    m_accepted = nullptr;
    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Scheduler counting the events scheduled far ahead
 *
 * A cancelled event stays in the scheduler until its time comes, so that
 * the events scheduled far ahead and cancelled pile up.
 */
class TcpCountingScheduler : public MapScheduler
{
  public:
    /**
     * @brief Get the type ID.
     * @return The object TypeId.
     */
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::TcpCountingScheduler")
                                .SetParent<MapScheduler>()
                                .SetGroupName("Internet")
                                .AddConstructor<TcpCountingScheduler>();
        return tid;
    }

    TcpCountingScheduler()
    {
        s_farAhead = 0;
    }

    void Insert(const Scheduler::Event& ev) override
    {
        MapScheduler::Insert(ev);
        if (ev.key.m_ts > static_cast<uint64_t>((Simulator::Now() + Minutes(10)).GetTimeStep()))
        {
            ++s_farAhead;
        }
    }

    static inline uint64_t s_farAhead{0}; //!< Events scheduled more than ten minutes ahead
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the keep-alive timer is not scheduled anew for every segment
 *
 * Every segment received starts the idle period anew, but an event cancelled
 * in the scheduler stays there until its time comes, two hours later by
 * default: a timer scheduled anew for every segment would leave one dead
 * event per segment behind. The events held by the scheduler during a bulk
 * transfer are the same with the keep-alives on as with them off, but for
 * the timer of each end.
 */
class TcpKeepAliveTimerEventsTestCase : public TestCase
{
  public:
    TcpKeepAliveTimerEventsTestCase()
        : TestCase("The keep-alive timer is not scheduled anew for every segment")
    {
    }

  private:
    void DoRun() override;

    /**
     * Run a bulk transfer.
     * @param keepAlive Whether the keep-alives are on, on both ends.
     * @return The events scheduled more than ten minutes ahead.
     */
    uint64_t Run(bool keepAlive);
};

uint64_t
TcpKeepAliveTimerEventsTestCase::Run(bool keepAlive)
{
    ObjectFactory factory;
    factory.SetTypeId(TcpCountingScheduler::GetTypeId());
    Simulator::SetScheduler(factory);
    Config::SetDefault("ns3::TcpSocket::KeepAlive", BooleanValue(keepAlive));

    NodeContainer nodes;
    nodes.Create(2);
    SimpleNetDeviceHelper devHelper;
    devHelper.SetDeviceAttribute("DataRate", StringValue("20Mbps"));
    devHelper.SetChannelAttribute("Delay", StringValue("1ms"));
    NetDeviceContainer devices = devHelper.Install(nodes);
    InternetStackHelper stack;
    stack.Install(nodes);
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t port = 9924;
    uint64_t received = 0;
    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              Callback<void, Ptr<Socket>, const Address&>(
                                  [&received](Ptr<Socket> socket, const Address&) {
                                      socket->SetRecvCallback(
                                          Callback<void, Ptr<Socket>>([&received](Ptr<Socket> s) {
                                              while (Ptr<Packet> packet = s->Recv())
                                              {
                                                  received += packet->GetSize();
                                              }
                                          }));
                                  }));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    auto fill = [](Ptr<Socket> socket, uint32_t available) {
        while (available >= 1000 && socket->Send(Create<Packet>(1000), 0) > 0)
        {
            available -= 1000;
        }
    };
    client->SetSendCallback(Callback<void, Ptr<Socket>, uint32_t>(fill));
    client->SetConnectCallback(Callback<void, Ptr<Socket>>([fill](Ptr<Socket> socket) {
                                   fill(socket, socket->GetTxAvailable());
                               }),
                               MakeNullCallback<void, Ptr<Socket>>());
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));

    Simulator::Stop(Seconds(0.5));
    Simulator::Run();
    uint64_t farAhead = TcpCountingScheduler::s_farAhead;

    NS_TEST_EXPECT_MSG_GT(received, 500000, "The transfer did not run");
    client = nullptr;
    server = nullptr;
    Simulator::Destroy();
    Config::Reset();
    return farAhead;
}

void
TcpKeepAliveTimerEventsTestCase::DoRun()
{
    uint64_t without = Run(false);
    uint64_t with = Run(true);
    // The timer of each end is armed as the connection is established, and
    // goes on expiring no sooner than two hours later
    NS_TEST_ASSERT_MSG_LT_OR_EQ(with,
                                without + 2,
                                "The keep-alive timer was scheduled anew for the segments");
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the shortest keep-alive idle time and interval
 *
 * A zero idle time would have the timer, armed again with it while data is
 * outstanding, expire over and over at the same instant, so that the
 * simulated time stops; a zero interval would send all the keep-alives, and
 * the reset after them, at once. Both are refused, and a connection with the
 * shortest ones runs through outstanding data.
 */
class TcpKeepAliveShortestTimesTestCase : public TcpKeepAliveBaseTestCase
{
  public:
    TcpKeepAliveShortestTimesTestCase()
        : TcpKeepAliveBaseTestCase("The shortest keep-alive idle time and interval")
    {
    }

  private:
    void DoRun() override;
};

void
TcpKeepAliveShortestTimesTestCase::DoRun()
{
    for (const auto& [type, name] :
         {std::pair<std::string, std::string>{"ns3::TcpSocket", "KeepAliveTime"},
          {"ns3::TcpSocketBase", "KeepAliveInterval"}})
    {
        TypeId::AttributeInformation info;
        NS_TEST_ASSERT_MSG_EQ(TypeId::LookupByName(type).LookupAttributeByName(name, &info),
                              true,
                              "No " << name << " attribute");
        NS_TEST_ASSERT_MSG_EQ(info.checker->Check(TimeValue(Seconds(0))),
                              false,
                              "A zero " << name << " is accepted");
        NS_TEST_ASSERT_MSG_EQ(info.checker->Check(TimeValue(MilliSeconds(1))),
                              true,
                              "A " << name << " of one millisecond is refused");
    }

    Setup(false, MilliSeconds(1));
    m_client->SetAttribute("KeepAliveInterval", TimeValue(MilliSeconds(1)));
    size_t answered = 0;
    Simulator::Schedule(Seconds(1), [this, &answered]() {
        answered = m_probes.size();
        BreakThePath();
        m_client->Send(Create<Packet>(100), 0);
    });

    Simulator::Stop(Seconds(3));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(Simulator::Now(), Seconds(3), "The simulated time did not run");
    NS_TEST_ASSERT_MSG_GT(answered, 0, "No keep-alive was sent");
    NS_TEST_ASSERT_MSG_EQ(m_probes.size(), answered, "A keep-alive was sent with data outstanding");
    NS_TEST_ASSERT_MSG_LT(Simulator::GetEventCount(),
                          100000,
                          "The keep-alive timer expired too often");
    m_client = nullptr;
    m_accepted = nullptr;
    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the window of the SYN does not hold the right edge
 *
 * The receiver holds the right edge of the window it advertised to avoid the
 * silly window syndrome (@RFC{9293}, Section 3.8.6.2.2), but the window of a
 * SYN is advertised before RCV.NXT is known: its right edge is not to be
 * held once the SYN of the peer sets RCV.NXT, whatever the initial sequence
 * number of the peer.
 */
class TcpSynWindowEdgeTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpSynWindowEdgeTestCase()
        : TcpCraftedSegmentTestCase("The window of the SYN does not hold the right edge")
    {
    }

  private:
    void DoRun() override;

    /**
     * Open a connection from the target towards a peer with the given ISN.
     * @param peerIsn The initial sequence number of the peer.
     * @return The window the target advertises in the ACK of the handshake.
     */
    uint32_t Run(uint32_t peerIsn);

    /**
     * Answer the SYN of the target.
     * @param peerIsn The initial sequence number of the peer.
     */
    void AnswerSyn(uint32_t peerIsn);

    uint32_t m_buffer{10000}; //!< Receive buffer of the target
};

void
TcpSynWindowEdgeTestCase::AnswerSyn(uint32_t peerIsn)
{
    NS_ABORT_MSG_IF(GetReplies().empty(), "No SYN was sent");
    SequenceNumber32 targetIsn = GetReplies().front().GetSequenceNumber();
    SendSegmentFull(TcpHeader::SYN | TcpHeader::ACK, peerIsn, (targetIsn + 1).GetValue(), 4096);
}

uint32_t
TcpSynWindowEdgeTestCase::Run(uint32_t peerIsn)
{
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(m_buffer));
    SetupTopology(false);
    Ptr<Socket> client = Socket::CreateSocket(m_nodes.Get(1), TcpSocketFactory::GetTypeId());
    client->Bind(InetSocketAddress(Ipv4Address::GetAny(), m_targetPort));
    client->Connect(InetSocketAddress(m_interfaces.GetAddress(0), m_proberPort));

    Simulator::Schedule(Seconds(1), &TcpSynWindowEdgeTestCase::AnswerSyn, this, peerIsn);
    Simulator::Stop(Seconds(2));
    Simulator::Run();

    uint32_t window = 0;
    for (const auto& reply : GetReplies())
    {
        if (reply.GetFlags() == TcpHeader::ACK &&
            reply.GetAckNumber() == SequenceNumber32(peerIsn + 1))
        {
            window = reply.GetWindowSize();
            break;
        }
    }
    Simulator::Destroy();
    Config::Reset();
    return window;
}

void
TcpSynWindowEdgeTestCase::DoRun()
{
    // RCV.NXT a little past the right edge of the SYN window, seen from zero
    NS_TEST_ASSERT_MSG_EQ(Run(100), m_buffer, "The window of the SYN held the right edge");
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Base class of the test cases on the right edge of the receive window
 *
 * Runs a transfer towards a receiver whose application reads at its own
 * pace, and records the right edge (acknowledgment number plus scaled
 * window) of every segment the sender receives from it.
 */
class TcpRightEdgeBaseTestCase : public TestCase
{
  public:
    /**
     * Constructor.
     * @param name The name of the test case.
     */
    TcpRightEdgeBaseTestCase(std::string name)
        : TestCase(name)
    {
    }

  protected:
    /**
     * Run the transfer.
     *
     * @param buffer The receive buffer of the receiver.
     * @param segmentSize The segment size of both ends.
     * @param scaling Whether the window scale option is enabled.
     * @param writes The number of writes of the sender.
     * @param writeSize The size of each write.
     * @param writeInterval The time between the writes.
     * @param readSize The size of each read of the receiver.
     * @param readInterval The time between the reads.
     */
    void Run(uint32_t buffer,
             uint32_t segmentSize,
             bool scaling,
             uint32_t writes,
             uint32_t writeSize,
             Time writeInterval,
             uint32_t readSize,
             Time readInterval);

    /**
     * Rx trace sink of the sender, recording the segments of the receiver.
     * @param packet The segment received.
     * @param header The TCP header of the segment.
     * @param socket The receiving socket.
     */
    void SegmentReceived(Ptr<const Packet> packet,
                         const TcpHeader& header,
                         Ptr<const TcpSocketBase> socket);

    std::vector<SequenceNumber32> m_edges; //!< Right edges advertised by the receiver
    uint8_t m_shift{0};                    //!< Window scale of the receiver
    uint64_t m_bytesRead{0};               //!< Bytes read by the receiver

  private:
    /**
     * Read once, and schedule the next read.
     * @param socket The receiving socket.
     * @param size The size of the read.
     * @param interval The time until the next read.
     */
    void Read(Ptr<Socket> socket, uint32_t size, Time interval);
};

void
TcpRightEdgeBaseTestCase::SegmentReceived(Ptr<const Packet> packet,
                                          const TcpHeader& header,
                                          Ptr<const TcpSocketBase> socket)
{
    if (header.GetFlags() & TcpHeader::SYN)
    {
        if (header.HasOption(TcpOption::WINSCALE))
        {
            m_shift = DynamicCast<const TcpOptionWinScale>(header.GetOption(TcpOption::WINSCALE))
                          ->GetScale();
        }
        return;
    }
    if (header.GetFlags() & TcpHeader::ACK)
    {
        m_edges.push_back(header.GetAckNumber() +
                          (static_cast<uint32_t>(header.GetWindowSize()) << m_shift));
    }
}

void
TcpRightEdgeBaseTestCase::Read(Ptr<Socket> socket, uint32_t size, Time interval)
{
    Ptr<Packet> p = socket->Recv(size, 0);
    if (p)
    {
        m_bytesRead += p->GetSize();
    }
    Simulator::Schedule(interval, &TcpRightEdgeBaseTestCase::Read, this, socket, size, interval);
}

void
TcpRightEdgeBaseTestCase::Run(uint32_t buffer,
                              uint32_t segmentSize,
                              bool scaling,
                              uint32_t writes,
                              uint32_t writeSize,
                              Time writeInterval,
                              uint32_t readSize,
                              Time readInterval)
{
    m_edges.clear();
    m_shift = 0;
    m_bytesRead = 0;
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(buffer));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(writes * writeSize));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(segmentSize));
    Config::SetDefault("ns3::TcpSocket::TcpNoDelay", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocketBase::Timestamp", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocketBase::WindowScaling", BooleanValue(scaling));

    NodeContainer nodes;
    nodes.Create(2);

    SimpleNetDeviceHelper devHelper;
    devHelper.SetChannelAttribute("Delay", TimeValue(MilliSeconds(1)));
    NetDeviceContainer devices = devHelper.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t port = 9924;

    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              Callback<void, Ptr<Socket>, const Address&>(
                                  [this, readSize, readInterval](Ptr<Socket> s, const Address&) {
                                      // The reads start once the data flows
                                      Simulator::Schedule(Seconds(1),
                                                          &TcpRightEdgeBaseTestCase::Read,
                                                          this,
                                                          s,
                                                          readSize,
                                                          readInterval);
                                  }));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    // The segments of the receiver, as the sender sees them
    client->TraceConnectWithoutContext(
        "Rx",
        MakeCallback(&TcpRightEdgeBaseTestCase::SegmentReceived, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));
    for (uint32_t i = 0; i < writes; ++i)
    {
        Simulator::Schedule(Seconds(1) + writeInterval * i,
                            [client, writeSize]() { client->Send(Create<Packet>(writeSize), 0); });
    }

    Simulator::Stop(Seconds(30));
    Simulator::Run();
    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the receiver silly window syndrome avoidance
 *
 * @RFC{9293}, Section 3.8.6.2.2 (MUST-39) keeps the right edge RCV.NXT +
 * RCV.WND where it is unless it can move by at least min(Fr * RCV.BUFF,
 * Eff.snd.MSS), with Fr = 1/2, whatever the size of the window: an
 * application reading a little at a time must not move the edge a little at
 * a time, even with a window wide open.
 */
class TcpReceiverSwsTestCase : public TcpRightEdgeBaseTestCase
{
  public:
    TcpReceiverSwsTestCase()
        : TcpRightEdgeBaseTestCase("The receiver avoids the silly window syndrome (MUST-39)")
    {
    }

  private:
    void DoRun() override;
};

void
TcpReceiverSwsTestCase::DoRun()
{
    const uint32_t segmentSize = 500;
    // Small writes, each acknowledged with the window the small reads leave
    Run(10000, segmentSize, false, 100, 50, MilliSeconds(50), 120, MilliSeconds(40));

    NS_TEST_ASSERT_MSG_EQ(m_bytesRead, 5000, "The transfer did not complete");
    uint32_t small = 0;
    uint32_t receded = 0;
    for (size_t i = 1; i < m_edges.size(); ++i)
    {
        if (m_edges[i] < m_edges[i - 1])
        {
            receded++;
        }
        else if (m_edges[i] > m_edges[i - 1] &&
                 m_edges[i] - m_edges[i - 1] < static_cast<int32_t>(segmentSize))
        {
            small++;
        }
    }
    NS_TEST_ASSERT_MSG_GT(m_edges.size(), 10, "Too few segments from the receiver");
    NS_TEST_ASSERT_MSG_EQ(small, 0, "The right edge moved by less than a segment");
    NS_TEST_ASSERT_MSG_EQ(receded, 0, "The right edge moved left");
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the right edge holds with the window scale option
 *
 * @RFC{1122}, Section 4.2.2.16 asks a receiver not to move the right edge of
 * the window left, which @RFC{7323}, Section 2.4 extends to the window scaled
 * down to its unit: rounding the window down at each segment would move the
 * edge left by up to 2^shift - 1 octets whenever RCV.NXT moves by an amount
 * which is not a multiple of the unit. The edge seen by the sender, the
 * acknowledgment number plus the window scaled back, must only move forward,
 * however odd the sizes of the segments and of the reads.
 */
class TcpScaledRightEdgeTestCase : public TcpRightEdgeBaseTestCase
{
  public:
    TcpScaledRightEdgeTestCase()
        : TcpRightEdgeBaseTestCase("The right edge holds with the window scale option")
    {
    }

  private:
    void DoRun() override;
};

void
TcpScaledRightEdgeTestCase::DoRun()
{
    Run(200000, 333, true, 200, 1001, MilliSeconds(5), 777, MilliSeconds(2));

    NS_TEST_ASSERT_MSG_GT(m_shift, 0, "The window was not scaled");
    NS_TEST_ASSERT_MSG_EQ(m_bytesRead, 200 * 1001, "The transfer did not complete");
    uint32_t receded = 0;
    for (size_t i = 1; i < m_edges.size(); ++i)
    {
        if (m_edges[i] < m_edges[i - 1])
        {
            receded++;
        }
    }
    NS_TEST_ASSERT_MSG_GT(m_edges.size(), 10, "Too few segments from the receiver");
    NS_TEST_ASSERT_MSG_EQ(receded, 0, "The scaled right edge moved left");
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the window update sent when the application reads
 *
 * A window smaller than a segment is not probed by the sender, whose silly
 * window syndrome avoidance waits for it to open: the receiver has to
 * advertise the window the application reopens by reading, or the transfer
 * stalls.
 */
class TcpWindowUpdateAfterReadTestCase : public TestCase
{
  public:
    TcpWindowUpdateAfterReadTestCase()
        : TestCase("The window reopened by the application is advertised")
    {
    }

  private:
    void DoRun() override;
};

void
TcpWindowUpdateAfterReadTestCase::DoRun()
{
    // The first writes leave a window smaller than a segment
    const uint32_t buffer = 2000;
    const uint32_t bytes = 3000;
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(buffer));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(600));
    Config::SetDefault("ns3::TcpSocketBase::Timestamp", BooleanValue(false));

    NodeContainer nodes;
    nodes.Create(2);

    SimpleNetDeviceHelper devHelper;
    devHelper.SetChannelAttribute("Delay", TimeValue(MilliSeconds(1)));
    NetDeviceContainer devices = devHelper.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t port = 9925;

    Ptr<Socket> accepted;
    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              Callback<void, Ptr<Socket>, const Address&>(
                                  [&accepted](Ptr<Socket> s, const Address&) { accepted = s; }));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));
    Simulator::Schedule(Seconds(1), [client, bytes]() { client->Send(Create<Packet>(bytes), 0); });

    uint32_t freeBefore = 0;
    uint32_t received = 0;
    Time completed;
    Simulator::Schedule(Seconds(2), [&]() {
        freeBefore = buffer - accepted->GetRxAvailable();
        // Everything is read from now on, as it comes
        auto drain = [&](Ptr<Socket> s) {
            while (Ptr<Packet> p = s->Recv())
            {
                received += p->GetSize();
            }
            if (received == bytes && completed.IsZero())
            {
                completed = Simulator::Now();
            }
        };
        accepted->SetRecvCallback(Callback<void, Ptr<Socket>>(drain));
        drain(accepted);
    });

    Simulator::Stop(Seconds(10));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_GT(freeBefore, 0, "The window closed");
    NS_TEST_ASSERT_MSG_LT(freeBefore, 600, "The window left was not smaller than a segment");
    NS_TEST_ASSERT_MSG_EQ(received, bytes, "The transfer stalled");
    // Without the window update, the transfer would wait for a retransmission
    NS_TEST_ASSERT_MSG_LT(completed, Seconds(2.1), "The reopened window was not advertised");

    accepted = nullptr;
    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the delayed acknowledgments carry the held window
 *
 * The acknowledgments sent when the delayed acknowledgment timer expires
 * follow the receiver silly window syndrome avoidance like any other
 * segment (@RFC{9293}, Section 3.8.6.2.2): the reads done while the
 * acknowledgment was delayed do not move the right edge by a little.
 */
class TcpDelayedAckHeldWindowTestCase : public TcpRightEdgeBaseTestCase
{
  public:
    TcpDelayedAckHeldWindowTestCase()
        : TcpRightEdgeBaseTestCase("The delayed acknowledgments carry the held window")
    {
    }

  private:
    void DoRun() override;
};

void
TcpDelayedAckHeldWindowTestCase::DoRun()
{
    const uint32_t segmentSize = 500;
    // A single small segment at a time, acknowledged when the delayed
    // acknowledgment timer expires, after a few small reads
    Run(10000, segmentSize, false, 20, 50, MilliSeconds(300), 20, MilliSeconds(30));

    NS_TEST_ASSERT_MSG_EQ(m_bytesRead, 1000, "The transfer did not complete");
    uint32_t small = 0;
    for (size_t i = 1; i < m_edges.size(); ++i)
    {
        if (m_edges[i] > m_edges[i - 1] &&
            m_edges[i] - m_edges[i - 1] < static_cast<int32_t>(segmentSize))
        {
            small++;
        }
    }
    NS_TEST_ASSERT_MSG_GT(m_edges.size(), 10, "Too few segments from the receiver");
    NS_TEST_ASSERT_MSG_EQ(small, 0, "A delayed acknowledgment moved the right edge by a little");
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the PSH flag over the life of a connection
 *
 * @RFC{9293}, Section 3.9.1.3 (MUST-61) sets PSH on the segment after which
 * no data is queued. This holds when more data is written while the first
 * is in flight, when the last segment is retransmitted, when the FIN rides
 * on the last segment, and when Nagle coalesces small writes.
 */
class TcpPushScenariosTestCase : public TestCase
{
  public:
    TcpPushScenariosTestCase()
        : TestCase("PSH over the life of a connection (MUST-61)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Run a connection whose client writes as told.
     *
     * @param writes The instants and sizes of the writes.
     * @param closeAfterWrites Whether the client closes right after its writes.
     * @param noDelay Whether Nagle is disabled.
     * @param breakFrom Start of the interval in which the server receives nothing.
     * @param breakTo End of that interval.
     */
    void Run(const std::vector<std::pair<Time, uint32_t>>& writes,
             bool closeAfterWrites,
             bool noDelay,
             Time breakFrom = Seconds(0),
             Time breakTo = Seconds(0));

    /**
     * Tx trace sink of the client.
     * @param packet The segment sent.
     * @param header The TCP header of the segment.
     * @param socket The sending socket.
     */
    void SegmentSent(Ptr<const Packet> packet,
                     const TcpHeader& header,
                     Ptr<const TcpSocketBase> socket);

    /// The data segments sent: sequence number, size and flags
    std::vector<std::tuple<SequenceNumber32, uint32_t, uint8_t>> m_segments;
    SequenceNumber32 m_isn{0}; //!< Initial sequence number of the client
};

void
TcpPushScenariosTestCase::SegmentSent(Ptr<const Packet> packet,
                                      const TcpHeader& header,
                                      Ptr<const TcpSocketBase> socket)
{
    if (header.GetFlags() & TcpHeader::SYN)
    {
        m_isn = header.GetSequenceNumber();
    }
    else if (packet->GetSize() > 0)
    {
        m_segments.emplace_back(header.GetSequenceNumber(), packet->GetSize(), header.GetFlags());
    }
}

void
TcpPushScenariosTestCase::Run(const std::vector<std::pair<Time, uint32_t>>& writes,
                              bool closeAfterWrites,
                              bool noDelay,
                              Time breakFrom,
                              Time breakTo)
{
    m_segments.clear();
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(500));
    Config::SetDefault("ns3::TcpSocket::TcpNoDelay", BooleanValue(noDelay));

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

    const uint16_t port = 9926;

    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->TraceConnectWithoutContext("Tx",
                                       MakeCallback(&TcpPushScenariosTestCase::SegmentSent, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));
    for (size_t i = 0; i < writes.size(); ++i)
    {
        bool last = (i + 1 == writes.size());
        uint32_t size = writes[i].second;
        Simulator::Schedule(writes[i].first, [client, size, last, closeAfterWrites]() {
            client->Send(Create<Packet>(size), 0);
            if (last && closeAfterWrites)
            {
                client->Close();
            }
        });
    }
    if (breakTo > breakFrom)
    {
        Ptr<NetDevice> device = devices.Get(1);
        Ptr<RateErrorModel> error = CreateObject<RateErrorModel>();
        error->SetAttribute("ErrorRate", DoubleValue(1.0));
        error->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
        error->Disable();
        device->SetAttribute("ReceiveErrorModel", PointerValue(error));
        Simulator::Schedule(breakFrom, [error]() { error->Enable(); });
        Simulator::Schedule(breakTo, [error]() { error->Disable(); });
    }

    Simulator::Stop(Seconds(20));
    Simulator::Run();
    Simulator::Destroy();
    Config::Reset();
}

void
TcpPushScenariosTestCase::DoRun()
{
    auto ends = [this](const std::tuple<SequenceNumber32, uint32_t, uint8_t>& s) {
        return std::get<0>(s) + std::get<1>(s) - m_isn - 1;
    };
    auto pushed = [](const std::tuple<SequenceNumber32, uint32_t, uint8_t>& s) {
        return (std::get<2>(s) & TcpHeader::PSH) != 0;
    };

    // A second write while the first is in flight: the last segment of each
    // write empties the buffer when it leaves
    Run({{Seconds(1), 1200}, {Seconds(1) + MilliSeconds(5), 700}}, false, true);
    uint32_t pushes = 0;
    for (const auto& s : m_segments)
    {
        int32_t end = ends(s);
        bool emptiesBuffer = (end == 1200 || end == 1900);
        NS_TEST_ASSERT_MSG_EQ(pushed(s),
                              emptiesBuffer,
                              "Unexpected PSH on the segment ending at " << end);
        pushes += pushed(s) ? 1 : 0;
    }
    NS_TEST_ASSERT_MSG_EQ(pushes, 2, "Each write did not end with PSH");

    // The last segment lost, then retransmitted
    Run({{Seconds(1), 700}}, false, true, Seconds(1), Seconds(1.5));
    uint32_t lastSent = 0;
    for (const auto& s : m_segments)
    {
        if (ends(s) == 700)
        {
            lastSent++;
            NS_TEST_ASSERT_MSG_EQ(pushed(s), true, "The last segment was sent without PSH");
        }
    }
    NS_TEST_ASSERT_MSG_GT(lastSent, 1, "The last segment was not retransmitted");

    // The FIN riding on the last segment
    Run({{Seconds(1), 700}}, true, true);
    NS_TEST_ASSERT_MSG_EQ(m_segments.empty(), false, "No data was sent");
    if (!m_segments.empty())
    {
        uint8_t flags = std::get<2>(m_segments.back());
        NS_TEST_ASSERT_MSG_EQ((flags & (TcpHeader::PSH | TcpHeader::FIN)),
                              (TcpHeader::PSH | TcpHeader::FIN),
                              "The last segment carrying the FIN lacks PSH");
    }

    // Small writes coalesced by Nagle while the first one is unacknowledged
    std::vector<std::pair<Time, uint32_t>> smallWrites;
    for (uint32_t i = 0; i < 10; ++i)
    {
        smallWrites.emplace_back(Seconds(1) + MilliSeconds(i), 100);
    }
    Run(smallWrites, false, false);
    NS_TEST_ASSERT_MSG_LT(m_segments.size(), 10, "The small writes were not coalesced");
    NS_TEST_ASSERT_MSG_EQ(m_segments.empty(), false, "No data was sent");
    if (!m_segments.empty())
    {
        NS_TEST_ASSERT_MSG_EQ(ends(m_segments.back()), 1000, "Not all the data was sent");
        NS_TEST_ASSERT_MSG_EQ(pushed(m_segments.back()),
                              true,
                              "The coalesced segment emptying the buffer lacks PSH");
    }
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the sender reads the window as an unsigned number
 *
 * @RFC{9293}, Section 3.1 (MUST-1) requires the window to be treated as an
 * unsigned number. The peer, driven from a raw socket without the window
 * scale option, advertises a window of 0xFFFF, which read as a signed number
 * would be negative: the sender has to keep more than 32767 octets in
 * flight.
 */
class TcpUnsignedWindowSenderTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpUnsignedWindowSenderTestCase()
        : TcpCraftedSegmentTestCase("The sender reads the window as unsigned (MUST-1)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Complete the handshake.
     */
    void CompleteHandshake();

    /**
     * Acknowledge the data sent so far, and schedule the next acknowledgment.
     */
    void Acknowledge();

    /**
     * Accept callback, sending the data and tracing what is sent.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    SequenceNumber32 m_targetIsn{0}; //!< Initial sequence number of the target
    SequenceNumber32 m_highest{0};   //!< Highest sequence number sent by the target
    SequenceNumber32 m_acked{0};     //!< Highest acknowledgment sent to the target
    uint32_t m_maxInFlight{0};       //!< Largest amount of data in flight
    uint32_t m_bytes{200000};        //!< Data the target sends
};

void
TcpUnsignedWindowSenderTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->TraceConnectWithoutContext(
        "Tx",
        Callback<void, Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>>(
            [this](Ptr<const Packet> p, const TcpHeader& h, Ptr<const TcpSocketBase>) {
                if (p->GetSize() > 0)
                {
                    m_highest = std::max(m_highest, h.GetSequenceNumber() + p->GetSize());
                    m_maxInFlight =
                        std::max<uint32_t>(m_maxInFlight,
                                           m_highest - std::max(m_acked, m_targetIsn + 1));
                }
            }));
    socket->Send(Create<Packet>(m_bytes), 0);
}

void
TcpUnsignedWindowSenderTestCase::CompleteHandshake()
{
    NS_ABORT_MSG_IF(GetReplies().empty(), "The SYN was not answered");
    m_targetIsn = GetReplies().front().GetSequenceNumber();
    m_acked = m_targetIsn + 1;
    m_highest = m_acked;
    SendSegmentFull(TcpHeader::ACK, 1001, m_acked.GetValue(), 0xFFFF);
    Simulator::Schedule(MilliSeconds(10), &TcpUnsignedWindowSenderTestCase::Acknowledge, this);
}

void
TcpUnsignedWindowSenderTestCase::Acknowledge()
{
    if (m_highest > m_acked)
    {
        m_acked = m_highest;
        SendSegmentFull(TcpHeader::ACK, 1001, m_acked.GetValue(), 0xFFFF);
    }
    Simulator::Schedule(MilliSeconds(10), &TcpUnsignedWindowSenderTestCase::Acknowledge, this);
}

void
TcpUnsignedWindowSenderTestCase::DoRun()
{
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(m_bytes));
    SetupTopology();
    Config::Reset();
    m_target->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                MakeCallback(&TcpUnsignedWindowSenderTestCase::Accepted, this));

    Simulator::Schedule(Seconds(1),
                        &TcpUnsignedWindowSenderTestCase::SendSegmentFull,
                        this,
                        TcpHeader::SYN,
                        1000U,
                        0U,
                        static_cast<uint16_t>(0xFFFF),
                        false,
                        static_cast<uint16_t>(0),
                        0U);
    Simulator::Schedule(Seconds(2), &TcpUnsignedWindowSenderTestCase::CompleteHandshake, this);

    Simulator::Stop(Seconds(5));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_highest - (m_targetIsn + 1),
                          static_cast<int32_t>(m_bytes),
                          "The sender did not send all its data");
    NS_TEST_ASSERT_MSG_GT(m_maxInFlight,
                          32767,
                          "The sender kept the data in flight within a signed window");
    NS_TEST_ASSERT_MSG_LT_OR_EQ(m_maxInFlight, 0xFFFF, "The sender overran the window");

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the checksum is checked on an established connection
 *
 * @RFC{9293}, Section 3.1 (MUST-3) requires the checksum of every incoming
 * segment to be checked: a data segment whose checksum does not match is
 * dropped without an acknowledgment, and a RST whose checksum does not match
 * does not reset the connection. The same holds over IPv6, whose pseudo
 * header differs.
 */
class TcpChecksumEstablishedTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpChecksumEstablishedTestCase()
        : TcpCraftedSegmentTestCase("The checksum is checked on an established connection (MUST-3)")
    {
    }

  private:
    void DoRun() override;
    void DoTeardown() override;

    /**
     * Run the IPv6 part of the test.
     */
    void RunIpv6();

    /**
     * Complete the handshake.
     */
    void CompleteHandshake();

    /**
     * Accept callback, keeping the accepted socket.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    Ptr<Socket> m_accepted;                            //!< The accepted socket
    SequenceNumber32 m_targetIsn{0};                   //!< Initial sequence number of the target
    BooleanValue m_previousChecksumEnabled;            //!< The global value before the test
    TcpSocket::TcpStates_t m_state{TcpSocket::CLOSED}; //!< State of the accepted socket
};

void
TcpChecksumEstablishedTestCase::DoTeardown()
{
    GlobalValue::Bind("ChecksumEnabled", m_previousChecksumEnabled);
}

void
TcpChecksumEstablishedTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    m_accepted = socket;
    m_state = TcpSocket::ESTABLISHED;
    socket->TraceConnectWithoutContext(
        "State",
        Callback<void, TcpSocket::TcpStates_t, TcpSocket::TcpStates_t>(
            [this](TcpSocket::TcpStates_t, TcpSocket::TcpStates_t newState) {
                m_state = newState;
            }));
}

void
TcpChecksumEstablishedTestCase::CompleteHandshake()
{
    NS_ABORT_MSG_IF(GetReplies().empty(), "The SYN was not answered");
    m_targetIsn = GetReplies().front().GetSequenceNumber();
    SendSegmentFull(TcpHeader::ACK, 1001, (m_targetIsn + 1).GetValue(), 4096);
}

void
TcpChecksumEstablishedTestCase::RunIpv6()
{
    NodeContainer nodes;
    nodes.Create(2);

    SimpleNetDeviceHelper devHelper;
    NetDeviceContainer devices = devHelper.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv6AddressHelper address;
    address.SetBase("2001:db8::", Ipv6Prefix(64));
    Ipv6InterfaceContainer interfaces = address.Assign(devices);
    Ipv6Address source = interfaces.GetAddress(0, 1);
    Ipv6Address destination = interfaces.GetAddress(1, 1);

    Ptr<Socket> target = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
    target->Bind(Inet6SocketAddress(Ipv6Address::GetAny(), m_targetPort));
    target->Listen();

    uint32_t synAcks = 0;
    Ptr<Socket> prober = Socket::CreateSocket(nodes.Get(0), Ipv6RawSocketFactory::GetTypeId());
    prober->SetAttribute("Protocol", UintegerValue(6));
    prober->Bind(Inet6SocketAddress(source, 0));
    prober->SetRecvCallback(Callback<void, Ptr<Socket>>([&synAcks](Ptr<Socket> s) {
        Address from;
        while (Ptr<Packet> p = s->RecvFrom(from))
        {
            Ipv6Header ipv6;
            p->RemoveHeader(ipv6);
            TcpHeader tcp;
            p->RemoveHeader(tcp);
            if ((tcp.GetFlags() & (TcpHeader::SYN | TcpHeader::ACK)) ==
                (TcpHeader::SYN | TcpHeader::ACK))
            {
                synAcks++;
            }
        }
    }));

    auto sendSyn = [&](bool corrupt) {
        TcpHeader tcp;
        tcp.SetSourcePort(m_proberPort);
        tcp.SetDestinationPort(m_targetPort);
        tcp.SetSequenceNumber(SequenceNumber32(1000));
        tcp.SetFlags(TcpHeader::SYN);
        tcp.SetWindowSize(4096);
        tcp.EnableChecksums();
        tcp.InitializeChecksum(source,
                               corrupt ? Ipv6Address("2001:db8::99") : destination,
                               iana::internetprotocolnumbers::TCP);
        Ptr<Packet> packet = Create<Packet>();
        packet->AddHeader(tcp);
        prober->SendTo(packet, 0, Inet6SocketAddress(destination, 0));
    };
    uint32_t afterCorrupt = 0;
    Simulator::Schedule(Seconds(2), [&]() { sendSyn(true); });
    Simulator::Schedule(Seconds(3), [&]() {
        afterCorrupt = synAcks;
        sendSyn(false);
    });

    Simulator::Stop(Seconds(4));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(afterCorrupt, 0, "A SYN with a wrong checksum was answered over IPv6");
    NS_TEST_ASSERT_MSG_EQ(synAcks, 1, "The SYN with a valid checksum was not answered over IPv6");

    Simulator::Destroy();
}

void
TcpChecksumEstablishedTestCase::DoRun()
{
    GlobalValue::GetValueByName("ChecksumEnabled", m_previousChecksumEnabled);
    GlobalValue::Bind("ChecksumEnabled", BooleanValue(true));

    SetupTopology();
    m_target->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                MakeCallback(&TcpChecksumEstablishedTestCase::Accepted, this));

    Simulator::Schedule(Seconds(1),
                        &TcpChecksumEstablishedTestCase::SendSegmentFull,
                        this,
                        TcpHeader::SYN,
                        1000U,
                        0U,
                        static_cast<uint16_t>(4096),
                        false,
                        static_cast<uint16_t>(0),
                        0U);
    Simulator::Schedule(Seconds(2), &TcpChecksumEstablishedTestCase::CompleteHandshake, this);

    size_t beforeData = 0;
    size_t afterData = 0;
    size_t afterReset = 0;
    TcpSocket::TcpStates_t stateAfterReset = TcpSocket::CLOSED;
    uint32_t availableAfterReset = 0;
    Simulator::Schedule(Seconds(3), [&]() {
        beforeData = GetReplies().size();
        // Data with a wrong checksum
        SendSegmentFull(TcpHeader::ACK | TcpHeader::PSH,
                        1001,
                        (m_targetIsn + 1).GetValue(),
                        4096,
                        true,
                        0,
                        100);
    });
    Simulator::Schedule(Seconds(4), [&]() {
        afterData = GetReplies().size();
        // A reset with a wrong checksum
        SendSegmentFull(TcpHeader::RST, 1001, 0, 0, true);
    });
    Simulator::Schedule(Seconds(5), [&]() {
        afterReset = GetReplies().size();
        stateAfterReset = m_state;
        availableAfterReset = m_accepted->GetRxAvailable();
        // The same data, with a valid checksum
        SendSegmentFull(TcpHeader::ACK | TcpHeader::PSH,
                        1001,
                        (m_targetIsn + 1).GetValue(),
                        4096,
                        false,
                        0,
                        100);
    });

    Simulator::Stop(Seconds(6));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(afterData, beforeData, "Data with a wrong checksum was answered");
    NS_TEST_ASSERT_MSG_EQ(afterReset, afterData, "A reset with a wrong checksum was answered");
    NS_TEST_ASSERT_MSG_EQ(availableAfterReset, 0, "Data with a wrong checksum was delivered");
    NS_TEST_ASSERT_MSG_EQ(stateAfterReset,
                          TcpSocket::ESTABLISHED,
                          "A reset with a wrong checksum reset the connection");
    NS_TEST_ASSERT_MSG_EQ(m_accepted->GetRxAvailable(), 100, "The valid data was not delivered");
    bool acked = false;
    for (size_t i = afterReset; i < GetReplies().size(); ++i)
    {
        acked |= (GetReplies()[i].GetAckNumber() == SequenceNumber32(1101));
    }
    NS_TEST_ASSERT_MSG_EQ(acked, true, "The valid data was not acknowledged");

    m_accepted = nullptr;
    Simulator::Destroy();

    RunIpv6();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the retransmissions and the probes within a shrunk window
 *
 * @RFC{9293}, Section 3.8.6.1 asks a sender whose window shrank not to send
 * new data, but to retransmit the unacknowledged data which lies within the
 * window (SHLD-15, SHLD-16), and requires a window shrunk to zero to be
 * probed (MUST-35), keeping the connection open (MUST-36).
 */
class TcpShrinkingWindowProbeTestCase : public TcpCraftedSegmentTestCase
{
  public:
    TcpShrinkingWindowProbeTestCase()
        : TcpCraftedSegmentTestCase(
              "A shrunk window is retransmitted within and probed (SHLD-16, MUST-35)")
    {
    }

  private:
    void DoRun() override;

    /**
     * Run a connection whose peer shrinks the window without acknowledging
     * anything.
     * @param window The window the peer shrinks to.
     */
    void Run(uint16_t window);

    /**
     * Complete the handshake and open a large window.
     */
    void CompleteHandshake();

    /**
     * Accept callback, sending the data and tracing what is sent.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    Ptr<Socket> m_accepted;          //!< The accepted socket
    SequenceNumber32 m_targetIsn{0}; //!< Initial sequence number of the target
    uint32_t m_largeWindow{6000};    //!< Window advertised during the handshake
    Time m_shrunk{Seconds(3)};       //!< Instant the window shrinks
    std::vector<std::pair<SequenceNumber32, uint32_t>> m_after; //!< Data sent after the shrink
    TcpSocket::TcpStates_t m_state{TcpSocket::CLOSED};          //!< State of the accepted socket
    bool m_resetSent{false};                                    //!< Whether the target sent a RST
};

void
TcpShrinkingWindowProbeTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    m_accepted = socket;
    m_state = TcpSocket::ESTABLISHED;
    socket->TraceConnectWithoutContext(
        "State",
        Callback<void, TcpSocket::TcpStates_t, TcpSocket::TcpStates_t>(
            [this](TcpSocket::TcpStates_t, TcpSocket::TcpStates_t newState) {
                m_state = newState;
            }));
    socket->TraceConnectWithoutContext(
        "Tx",
        Callback<void, Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>>(
            [this](Ptr<const Packet> p, const TcpHeader& h, Ptr<const TcpSocketBase>) {
                m_resetSent |= (h.GetFlags() & TcpHeader::RST) != 0;
                if (Simulator::Now() > m_shrunk && p->GetSize() > 0)
                {
                    m_after.emplace_back(h.GetSequenceNumber(), p->GetSize());
                }
            }));
    socket->Send(Create<Packet>(20000), 0);
}

void
TcpShrinkingWindowProbeTestCase::CompleteHandshake()
{
    // The replies of the earlier runs come first
    NS_ABORT_MSG_IF(GetReplies().empty(), "The SYN was not answered");
    m_targetIsn = GetReplies().back().GetSequenceNumber();
    SendSegmentFull(TcpHeader::ACK, 1001, (m_targetIsn + 1).GetValue(), m_largeWindow);
}

void
TcpShrinkingWindowProbeTestCase::Run(uint16_t window)
{
    m_after.clear();
    m_accepted = nullptr;
    m_state = TcpSocket::CLOSED;
    m_resetSent = false;
    SetupTopology();
    m_target->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                MakeCallback(&TcpShrinkingWindowProbeTestCase::Accepted, this));

    Simulator::Schedule(Seconds(1),
                        &TcpShrinkingWindowProbeTestCase::SendSegmentFull,
                        this,
                        TcpHeader::SYN,
                        1000U,
                        0U,
                        static_cast<uint16_t>(m_largeWindow),
                        false,
                        static_cast<uint16_t>(0),
                        0U);
    Simulator::Schedule(Seconds(2), &TcpShrinkingWindowProbeTestCase::CompleteHandshake, this);
    // The same acknowledgment, with a window which moves the right edge back
    Simulator::Schedule(m_shrunk, [this, window]() {
        SendSegmentFull(TcpHeader::ACK, 1001, (m_targetIsn + 1).GetValue(), window);
    });

    Simulator::Stop(Seconds(40));
    Simulator::Run();
}

void
TcpShrinkingWindowProbeTestCase::DoRun()
{
    // Shrunk to a window holding the first segment
    const uint16_t window = 1000;
    Run(window);
    NS_TEST_ASSERT_MSG_GT(m_after.size(), 0, "Nothing was retransmitted in the shrunk window");
    for (const auto& [seq, size] : m_after)
    {
        NS_TEST_ASSERT_MSG_EQ(seq, m_targetIsn + 1, "Not the first unacknowledged octet");
        NS_TEST_ASSERT_MSG_LT_OR_EQ(size, window, "A retransmission overran the shrunk window");
    }
    NS_TEST_ASSERT_MSG_EQ(m_resetSent, false, "The connection was reset");
    NS_TEST_ASSERT_MSG_EQ(m_state, TcpSocket::ESTABLISHED, "The connection did not stay up");
    m_accepted = nullptr;
    Simulator::Destroy();

    // Shrunk to zero: probed, and kept open
    Run(0);
    NS_TEST_ASSERT_MSG_GT(m_after.size(), 1, "The zero window was not probed");
    for (const auto& [seq, size] : m_after)
    {
        NS_TEST_ASSERT_MSG_EQ(seq, m_targetIsn + 1, "Not the first unacknowledged octet");
    }
    NS_TEST_ASSERT_MSG_EQ(m_resetSent, false, "The connection with a zero window was reset");
    NS_TEST_ASSERT_MSG_EQ(m_state,
                          TcpSocket::ESTABLISHED,
                          "The connection with a zero window did not stay up");
    m_accepted = nullptr;
    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief A transport protocol absorbing the TCP segments of a node
 *
 * Takes the place of the TCP stack of the probing IPv6 node: the IPv6 layer
 * hands the raw sockets only the segments of a protocol it knows, but the
 * TCP stack would reset the connections the raw socket drives.
 */
class TcpSinkL4Protocol : public IpL4Protocol
{
  public:
    /**
     * Get the type ID.
     * @return The object TypeId.
     */
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::TcpSinkL4Protocol")
                                .SetParent<IpL4Protocol>()
                                .SetGroupName("Internet")
                                .AddConstructor<TcpSinkL4Protocol>();
        return tid;
    }

    int GetProtocolNumber() const override
    {
        return iana::internetprotocolnumbers::TCP;
    }

    RxStatus Receive(Ptr<Packet> p,
                     const Ipv4Header& header,
                     Ptr<Ipv4Interface> incomingInterface) override
    {
        return RX_OK;
    }

    RxStatus Receive(Ptr<Packet> p,
                     const Ipv6Header& header,
                     Ptr<Ipv6Interface> incomingInterface) override
    {
        return RX_OK;
    }

    void SetDownTarget(DownTargetCallback cb) override
    {
    }

    void SetDownTarget6(DownTargetCallback6 cb) override
    {
    }

    DownTargetCallback GetDownTarget() const override
    {
        return DownTargetCallback();
    }

    DownTargetCallback6 GetDownTarget6() const override
    {
        return DownTargetCallback6();
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Base class of the connection setup and segment validation tests
 *
 * Extends the crafted segment harness: every field of the injected segments,
 * the data offset and the raw option bytes included, is chosen by the test,
 * and the topology can be built over IPv6 as well, the probing node then
 * sending through an IPv6 raw socket. The connection the target accepts, if
 * any, is watched.
 */
class TcpRawSegmentTestCase : public TcpCraftedSegmentTestCase
{
  public:
    /**
     * Constructor.
     * @param name The name of the test case.
     * @param ipv6 Whether to run the test over IPv6.
     */
    TcpRawSegmentTestCase(std::string name, bool ipv6 = false);

  protected:
    /// A segment to inject
    struct Segment
    {
        uint8_t flags{0};             ///< The TCP flags
        uint32_t seq{0};              ///< The sequence number
        uint32_t ack{0};              ///< The acknowledgment number
        std::vector<uint8_t> options; ///< The raw option bytes
        uint32_t payload{0};          ///< The number of data bytes to carry
        uint16_t window{4096};        ///< The window
        int dataOffset{-1};           ///< The data offset, derived from the options if negative
        uint16_t sport{0};            ///< The source port, the prober port if zero
        uint16_t dport{0};            ///< The destination port, the target port if zero
        uint8_t tos{0};               ///< The TOS or traffic class, carrying the ECN bits
    };

    /**
     * Build the topology over IPv4 or IPv6, depending on the test case.
     * @param listen Whether the target listens on the target port, watching
     *               the connection it accepts.
     */
    void Setup(bool listen = true);

    /**
     * Inject a segment towards the target.
     * @param segment The segment.
     */
    void Send(Segment segment);

    /**
     * Inject a segment at the given time.
     * @param at The time.
     * @param segment The segment.
     */
    void SendAt(Time at, const Segment& segment);

    /**
     * Build a segment.
     * @param flags The TCP flags.
     * @param seq The sequence number.
     * @param ack The acknowledgment number.
     * @param options The raw option bytes.
     * @param payload The number of data bytes to carry.
     * @return The segment.
     */
    static Segment Make(uint8_t flags,
                        uint32_t seq,
                        uint32_t ack,
                        const std::vector<uint8_t>& options = {},
                        uint32_t payload = 0);

    /**
     * Inject a segment acknowledging the given offset from the initial
     * sequence number of the target, which is learned during the handshake,
     * at the given time.
     * @param at The time.
     * @param flags The TCP flags.
     * @param seq The sequence number.
     * @param ackOffset The acknowledgment number, relative to the initial
     *                  sequence number of the target.
     * @param options The raw option bytes.
     * @param payload The number of data bytes to carry.
     */
    void SendRelativeAt(Time at,
                        uint8_t flags,
                        uint32_t seq,
                        uint32_t ackOffset,
                        const std::vector<uint8_t>& options = {},
                        uint32_t payload = 0);

    /**
     * Open a connection from the raw socket: a SYN carrying the given
     * options at the given time, and the ACK completing the handshake half
     * a second later.
     * @param at The time of the SYN.
     * @param synOptions The raw option bytes of the SYN.
     */
    void Handshake(Time at, const std::vector<uint8_t>& synOptions = {});

    /**
     * Acknowledge the last SYN+ACK received, learning the initial sequence
     * number of the target from it.
     */
    void CompleteHandshake();

    /**
     * Open a connection from the raw socket, timing its SYN so that the
     * target picks the given initial sequence number, and run the simulation
     * until the handshake is over.
     *
     * The clock driven initial sequence numbers must be enabled: a first
     * connection, aborted right away, tells the target ISN at a known time,
     * from which the clock gives the time of the wanted one. The replies of
     * the first connection are forgotten.
     *
     * @param wantedIsn The initial sequence number the target has to pick.
     */
    void HandshakeAtIsn(uint32_t wantedIsn);

    /**
     * Get the address of the target.
     * @param port The port.
     * @return The address of the target.
     */
    Address TargetAddress(uint16_t port) const;

    /**
     * Get the address of the prober.
     * @param port The port.
     * @return The address of the prober.
     */
    Address ProberAddress(uint16_t port) const;

    /**
     * Count the segments received from the target carrying all the given
     * flags, from the given index of the replies on.
     * @param flags The flags to look for.
     * @param from The index of the first reply to consider.
     * @return The number of matching segments.
     */
    uint32_t Count(uint8_t flags, std::size_t from = 0) const;

    /**
     * Error close callback of the accepted socket.
     * @param socket The socket.
     */
    void ErrorClosed(Ptr<Socket> socket);

    /**
     * Normal close callback of the accepted socket.
     * @param socket The socket.
     */
    void NormalClosed(Ptr<Socket> socket);

    /**
     * State trace sink of the accepted socket.
     * @param oldState The previous state.
     * @param newState The new state.
     */
    void StateChanged(TcpSocket::TcpStates_t oldState, TcpSocket::TcpStates_t newState);

    /**
     * Accept callback of the target, watching the accepted socket.
     * @param socket The accepted socket.
     * @param from The address of the peer.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    bool m_ipv6{false};                     //!< Whether the test runs over IPv6
    Ipv6InterfaceContainer m_interfaces6;   //!< The IPv6 interfaces of the two nodes
    std::vector<TcpHeader> m_headers;       //!< The headers of the segments from the target
    std::vector<uint32_t> m_payloads;       //!< The data sizes of the segments from the target
    std::vector<Time> m_times;              //!< The arrival times of the segments from the target
    uint8_t m_synFlags{TcpHeader::SYN};     //!< The flags of the SYN of the handshake
    uint32_t m_proberIsn{1};                //!< The initial sequence number of the prober
    std::vector<uint8_t> m_ackOptions;      //!< The raw option bytes of the handshake ACK
    Time m_closedAt;                        //!< When the accepted socket reached CLOSED
    SequenceNumber32 m_targetIsn{0};        //!< The initial sequence number of the target
    Ptr<Socket> m_accepted;                 //!< The socket accepted by the target
    TcpSocket::TcpStates_t m_acceptedState; //!< The state of the accepted socket
    uint32_t m_errorCloses{0};              //!< Error close notifications of the accepted socket
    uint32_t m_normalCloses{0};             //!< Normal close notifications of the accepted socket

  private:
    void DoTeardown() override;

    /**
     * Receive a segment on the raw socket of the prober.
     * @param socket The raw socket.
     */
    void Receive(Ptr<Socket> socket);
};

TcpRawSegmentTestCase::TcpRawSegmentTestCase(std::string name, bool ipv6)
    : TcpCraftedSegmentTestCase(name + (ipv6 ? " (IPv6)" : "")),
      m_ipv6(ipv6),
      m_acceptedState(TcpSocket::CLOSED)
{
}

void
TcpRawSegmentTestCase::DoTeardown()
{
    Simulator::Destroy();
    Config::Reset();
}

void
TcpRawSegmentTestCase::Setup(bool listen)
{
    m_headers.clear();
    m_payloads.clear();
    m_times.clear();
    m_accepted = nullptr;
    m_acceptedState = TcpSocket::CLOSED;
    m_errorCloses = 0;
    m_normalCloses = 0;
    m_closedAt = Seconds(0);

    if (!m_ipv6)
    {
        SetupTopology(listen);
        m_prober->SetRecvCallback(MakeCallback(&TcpRawSegmentTestCase::Receive, this));
    }
    else
    {
        // No duplicate address detection, so that the addresses are usable
        // from the start
        Config::SetDefault("ns3::Icmpv6L4Protocol::DAD", BooleanValue(false));
        m_nodes = NodeContainer();
        m_nodes.Create(2);
        SimpleNetDeviceHelper devHelper;
        NetDeviceContainer devices = devHelper.Install(m_nodes);
        InternetStackHelper stack;
        stack.SetIpv4StackInstall(false);
        stack.Install(m_nodes);
        Ipv6AddressHelper address;
        address.SetBase(Ipv6Address("2001:db8::"), Ipv6Prefix(64));
        m_interfaces6 = address.Assign(devices);

        if (listen)
        {
            m_target = Socket::CreateSocket(m_nodes.Get(1), TcpSocketFactory::GetTypeId());
            m_target->Bind(Inet6SocketAddress(Ipv6Address::GetAny(), m_targetPort));
            m_target->Listen();
        }

        // See SetupTopology: the probes alone speak for the probing node
        Ptr<Ipv6L3Protocol> proberIpv6 = m_nodes.Get(0)->GetObject<Ipv6L3Protocol>();
        proberIpv6->Remove(m_nodes.Get(0)->GetObject<TcpL4Protocol>());
        proberIpv6->Insert(CreateObject<TcpSinkL4Protocol>());

        m_prober = Socket::CreateSocket(m_nodes.Get(0), Ipv6RawSocketFactory::GetTypeId());
        m_prober->SetAttribute("Protocol", UintegerValue(6));
        m_prober->Bind(Inet6SocketAddress(m_interfaces6.GetAddress(0, 1), 0));
        m_prober->SetRecvCallback(MakeCallback(&TcpRawSegmentTestCase::Receive, this));
    }

    if (listen)
    {
        m_target->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                    MakeCallback(&TcpRawSegmentTestCase::Accepted, this));
    }
}

Address
TcpRawSegmentTestCase::TargetAddress(uint16_t port) const
{
    if (m_ipv6)
    {
        return Inet6SocketAddress(m_interfaces6.GetAddress(1, 1), port);
    }
    return InetSocketAddress(m_interfaces.GetAddress(1), port);
}

Address
TcpRawSegmentTestCase::ProberAddress(uint16_t port) const
{
    if (m_ipv6)
    {
        return Inet6SocketAddress(m_interfaces6.GetAddress(0, 1), port);
    }
    return InetSocketAddress(m_interfaces.GetAddress(0), port);
}

void
TcpRawSegmentTestCase::Send(Segment segment)
{
    NS_ASSERT(segment.options.size() % 4 == 0);
    const uint16_t sport = segment.sport == 0 ? m_proberPort : segment.sport;
    const uint16_t dport = segment.dport == 0 ? m_targetPort : segment.dport;
    const uint8_t dataOffset =
        segment.dataOffset < 0 ? 5 + segment.options.size() / 4 : segment.dataOffset;

    std::vector<uint8_t> bytes = {
        static_cast<uint8_t>(sport >> 8),
        static_cast<uint8_t>(sport & 0xff),
        static_cast<uint8_t>(dport >> 8),
        static_cast<uint8_t>(dport & 0xff),
        static_cast<uint8_t>(segment.seq >> 24),
        static_cast<uint8_t>(segment.seq >> 16),
        static_cast<uint8_t>(segment.seq >> 8),
        static_cast<uint8_t>(segment.seq),
        static_cast<uint8_t>(segment.ack >> 24),
        static_cast<uint8_t>(segment.ack >> 16),
        static_cast<uint8_t>(segment.ack >> 8),
        static_cast<uint8_t>(segment.ack),
        static_cast<uint8_t>(dataOffset << 4),
        segment.flags,
        static_cast<uint8_t>(segment.window >> 8),
        static_cast<uint8_t>(segment.window & 0xff),
        0x00,
        0x00, // checksum, not verified unless checksums are enabled
        0x00,
        0x00 // urgent pointer
    };
    bytes.insert(bytes.end(), segment.options.begin(), segment.options.end());
    bytes.insert(bytes.end(), segment.payload, 0);

    Ptr<Packet> packet = Create<Packet>(bytes.data(), bytes.size());
    if (m_ipv6)
    {
        m_prober->SetIpv6Tclass(segment.tos);
        m_prober->SendTo(packet, 0, Inet6SocketAddress(m_interfaces6.GetAddress(1, 1), 0));
    }
    else
    {
        m_prober->SetIpTos(segment.tos);
        m_prober->SendTo(packet, 0, InetSocketAddress(m_interfaces.GetAddress(1), 0));
    }
}

void
TcpRawSegmentTestCase::SendAt(Time at, const Segment& segment)
{
    Simulator::Schedule(at, &TcpRawSegmentTestCase::Send, this, segment);
}

TcpRawSegmentTestCase::Segment
TcpRawSegmentTestCase::Make(uint8_t flags,
                            uint32_t seq,
                            uint32_t ack,
                            const std::vector<uint8_t>& options,
                            uint32_t payload)
{
    Segment segment;
    segment.flags = flags;
    segment.seq = seq;
    segment.ack = ack;
    segment.options = options;
    segment.payload = payload;
    segment.window = 65535;
    return segment;
}

void
TcpRawSegmentTestCase::SendRelativeAt(Time at,
                                      uint8_t flags,
                                      uint32_t seq,
                                      uint32_t ackOffset,
                                      const std::vector<uint8_t>& options,
                                      uint32_t payload)
{
    Simulator::Schedule(at, [=, this]() {
        Send(Make(flags, seq, (m_targetIsn + ackOffset).GetValue(), options, payload));
    });
}

void
TcpRawSegmentTestCase::Handshake(Time at, const std::vector<uint8_t>& synOptions)
{
    Segment syn;
    syn.flags = m_synFlags;
    syn.seq = m_proberIsn;
    syn.options = synOptions;
    syn.window = 65535;
    SendAt(at, syn);
    Simulator::Schedule(at + MilliSeconds(500), &TcpRawSegmentTestCase::CompleteHandshake, this);
}

void
TcpRawSegmentTestCase::CompleteHandshake()
{
    for (auto it = m_headers.rbegin(); it != m_headers.rend(); ++it)
    {
        if ((it->GetFlags() & (TcpHeader::SYN | TcpHeader::ACK)) ==
            (TcpHeader::SYN | TcpHeader::ACK))
        {
            m_targetIsn = it->GetSequenceNumber();
            Segment ack;
            ack.flags = TcpHeader::ACK;
            ack.seq = m_proberIsn + 1;
            ack.ack = (m_targetIsn + 1).GetValue();
            ack.window = 65535;
            ack.options = m_ackOptions;
            Send(ack);
            return;
        }
    }
    // Nothing to acknowledge: the test cases check the outcome
}

void
TcpRawSegmentTestCase::HandshakeAtIsn(uint32_t wantedIsn)
{
    // A segment towards a closed port, which the target resets, refreshes
    // the address resolution in both directions first, so that the SYN
    // reaches the target when it is sent, and the SYN+ACK the prober
    Segment refresh = Make(TcpHeader::ACK, 0, 0);
    refresh.dport = m_targetPort + 1;
    SendAt(Seconds(0.5), refresh);
    SendAt(Seconds(1), Make(m_synFlags, m_proberIsn, 0));
    Simulator::Stop(Seconds(1.5));
    Simulator::Run();
    NS_ABORT_MSG_IF(m_headers.empty() || !(m_headers.back().GetFlags() & TcpHeader::SYN),
                    "The first SYN was not answered");
    uint32_t firstIsn = m_headers.back().GetSequenceNumber().GetValue();
    // The SYN+ACK leaves the target when the SYN reaches it, over a link
    // without delay: the clock ticks every 4 microseconds (RFC 9293,
    // Section 3.4.1)
    int64_t firstTick = m_times.back().GetMicroSeconds() / 4;
    Send(Make(TcpHeader::RST, m_proberIsn + 1, 0));

    uint32_t ticks = wantedIsn - firstIsn;
    Time at = MicroSeconds(4 * (firstTick + ticks));
    while (at < Simulator::Now() + Seconds(1))
    {
        at += MicroSeconds(4 * (int64_t{1} << 32));
    }
    SendAt(at - Simulator::Now() - MilliSeconds(100), refresh);
    Simulator::Schedule(at - Simulator::Now() - MilliSeconds(1), [this]() {
        m_headers.clear();
        m_payloads.clear();
        m_times.clear();
    });
    Handshake(at - Simulator::Now());
    Simulator::Stop(at - Simulator::Now() + Seconds(1));
    Simulator::Run();
}

void
TcpRawSegmentTestCase::Receive(Ptr<Socket> socket)
{
    Address from;
    while (Ptr<Packet> packet = socket->RecvFrom(from))
    {
        if (m_ipv6)
        {
            Ipv6Header ipv6;
            packet->RemoveHeader(ipv6);
            if (ipv6.GetNextHeader() != iana::internetprotocolnumbers::TCP)
            {
                continue;
            }
        }
        else
        {
            Ipv4Header ipv4;
            packet->RemoveHeader(ipv4);
            if (ipv4.GetProtocol() != iana::internetprotocolnumbers::TCP)
            {
                continue;
            }
        }
        TcpHeader tcp;
        packet->RemoveHeader(tcp);
        m_headers.push_back(tcp);
        m_payloads.push_back(packet->GetSize());
        m_times.push_back(Simulator::Now());
    }
}

uint32_t
TcpRawSegmentTestCase::Count(uint8_t flags, std::size_t from) const
{
    uint32_t count = 0;
    for (std::size_t i = from; i < m_headers.size(); ++i)
    {
        if ((m_headers[i].GetFlags() & flags) == flags)
        {
            count++;
        }
    }
    return count;
}

void
TcpRawSegmentTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    m_accepted = socket;
    m_acceptedState = TcpSocket::ESTABLISHED;
    socket->SetCloseCallbacks(MakeCallback(&TcpRawSegmentTestCase::NormalClosed, this),
                              MakeCallback(&TcpRawSegmentTestCase::ErrorClosed, this));
    socket->TraceConnectWithoutContext("State",
                                       MakeCallback(&TcpRawSegmentTestCase::StateChanged, this));
}

void
TcpRawSegmentTestCase::ErrorClosed(Ptr<Socket> socket)
{
    m_errorCloses++;
}

void
TcpRawSegmentTestCase::NormalClosed(Ptr<Socket> socket)
{
    m_normalCloses++;
}

void
TcpRawSegmentTestCase::StateChanged(TcpSocket::TcpStates_t oldState,
                                    TcpSocket::TcpStates_t newState)
{
    m_acceptedState = newState;
    if (newState == TcpSocket::CLOSED)
    {
        m_closedAt = Simulator::Now();
    }
}

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the bytes after the End of Option List option are ignored
 *
 * @RFC{9293}, Section 3.1 (MUST-69) requires the sender to zero the header
 * padding after the End of Option List option; a receiver ignores those bytes
 * whatever they hold, so a SYN with non-zero bytes there opens a connection
 * rather than being reset as a malformed segment.
 */
class TcpEolPaddingTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpEolPaddingTestCase()
        : TcpRawSegmentTestCase("Non-zero bytes after the End of Option List are ignored")
    {
    }

  private:
    void DoRun() override
    {
        Setup();
        // NOP, End of Option List, and two words of non-zero bytes
        Handshake(Seconds(1), {0x01, 0x00, 0x42, 0x13, 0x37, 0x08, 0x0a, 0x02});
        Simulator::Stop(Seconds(3));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::RST), 0, "The SYN was reset");
        NS_TEST_ASSERT_MSG_EQ((m_accepted != nullptr), true, "The connection was not accepted");
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that a segment with an inconsistent data offset is dropped
 *
 * A data offset smaller than the five words of the fixed header, or pointing
 * past the end of the segment, leaves a header which cannot be parsed: the
 * segment is dropped like a corrupted one, without reading past its end and
 * without any reply, and the listener keeps answering valid SYNs.
 */
class TcpDataOffsetTestCase : public TcpRawSegmentTestCase
{
  public:
    /**
     * Constructor.
     * @param ipv6 Whether to run the test over IPv6.
     */
    TcpDataOffsetTestCase(bool ipv6)
        : TcpRawSegmentTestCase("A segment with an inconsistent data offset is dropped", ipv6)
    {
    }

  private:
    void DoRun() override
    {
        Setup();
        // The segments are spaced, so that the address resolution queue does
        // not drop any while the first one waits for the resolution
        Time at = Seconds(1);
        uint16_t sport = 20000;
        for (uint8_t offset : {0, 1, 2, 3, 4, 6, 10, 15})
        {
            Segment syn;
            syn.flags = TcpHeader::SYN;
            syn.seq = 1;
            syn.dataOffset = offset;
            syn.sport = sport++;
            SendAt(at, syn);
            at += MilliSeconds(20);
        }
        // Fifteen words of header, but only four bytes of options and two of
        // data follow the fixed header
        Segment truncated;
        truncated.flags = TcpHeader::SYN;
        truncated.seq = 1;
        truncated.options = {0x01, 0x01, 0x01, 0x00};
        truncated.payload = 2;
        truncated.dataOffset = 15;
        truncated.sport = sport++;
        SendAt(at, truncated);
        Simulator::Stop(Seconds(2));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_EQ(m_headers.size(),
                              0,
                              "A segment with an inconsistent data offset was answered");

        Segment valid;
        valid.flags = TcpHeader::SYN;
        valid.seq = 1;
        valid.sport = sport;
        SendAt(Seconds(0), valid);
        Simulator::Stop(Seconds(1));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::SYN | TcpHeader::ACK),
                              1,
                              "The listener does not answer a valid SYN any more");
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the acceptability of the RST segments of a connection
 *
 * @RFC{9293}, Section 3.10.7.4 accepts a RST only if its sequence number lies
 * in the receive window, and recommends the mitigation of @RFC{5961},
 * Section 3.2: a RST at exactly RCV.NXT resets the connection, one elsewhere
 * in the window is answered with a challenge ACK, and one outside of it is
 * dropped silently, so that a blind attacker cannot reset the connection
 * without guessing its exact sequence number.
 */
class TcpResetAcceptabilityTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpResetAcceptabilityTestCase()
        : TcpRawSegmentTestCase("A RST resets the connection only at RCV.NXT")
    {
    }

  private:
    void DoRun() override
    {
        Setup();
        Handshake(Seconds(1));
        // RCV.NXT is 2: below it, and far beyond the window
        SendRelativeAt(Seconds(2), TcpHeader::RST, 1, 1);
        SendRelativeAt(Seconds(2.1), TcpHeader::RST, 0x80000000, 1);
        SendRelativeAt(Seconds(2.2), TcpHeader::RST | TcpHeader::ACK, 0x80000000, 1);
        Simulator::Stop(Seconds(2.5));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_EQ((m_accepted != nullptr), true, "The connection was not accepted");
        std::size_t beforeChallenge = m_headers.size();
        NS_TEST_ASSERT_MSG_EQ(m_acceptedState,
                              TcpSocket::ESTABLISHED,
                              "A RST outside the window reset the connection");
        NS_TEST_ASSERT_MSG_EQ(m_errorCloses, 0, "A RST outside the window was notified");

        // In the window, but not at RCV.NXT: a challenge ACK
        SendRelativeAt(Seconds(0.5), TcpHeader::RST, 2 + 1000, 1);
        Simulator::Stop(Seconds(1));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_EQ(m_acceptedState,
                              TcpSocket::ESTABLISHED,
                              "A RST not at RCV.NXT reset the connection");
        NS_TEST_ASSERT_MSG_EQ(m_headers.size(),
                              beforeChallenge + 1,
                              "A RST in the window was not answered with a challenge ACK");
        if (m_headers.size() == beforeChallenge + 1)
        {
            NS_TEST_ASSERT_MSG_EQ(m_headers.back().GetFlags() & (TcpHeader::ACK | TcpHeader::RST),
                                  TcpHeader::ACK,
                                  "The challenge is not an ACK");
            NS_TEST_ASSERT_MSG_EQ(m_headers.back().GetAckNumber(),
                                  SequenceNumber32(2),
                                  "The challenge ACK does not acknowledge RCV.NXT");
        }

        // At RCV.NXT: the connection is reset
        SendRelativeAt(Seconds(0.5), TcpHeader::RST, 2, 1);
        Simulator::Stop(Seconds(1));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_EQ(m_acceptedState,
                              TcpSocket::CLOSED,
                              "A RST at RCV.NXT did not reset the connection");
        NS_TEST_ASSERT_MSG_EQ(m_errorCloses, 1, "The reset was not notified as an abort");
        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::RST), 0, "A RST was answered with a RST");
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that a RST without the timestamp option is processed
 *
 * Once the timestamp option is negotiated, the segments which do not carry
 * it are dropped, except the RST segments (@RFC{7323}, Section 3.2): a peer
 * which lost the state of the connection resets it without any option.
 */
class TcpResetWithoutTimestampTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpResetWithoutTimestampTestCase()
        : TcpRawSegmentTestCase("A RST without the timestamp option resets the connection")
    {
    }

  private:
    void DoRun() override
    {
        Setup();
        // NOP, NOP, and a timestamp option: TSval 1000, TSecr 0
        const std::vector<uint8_t> synTs =
            {0x01, 0x01, 0x08, 0x0a, 0x00, 0x00, 0x03, 0xe8, 0x00, 0x00, 0x00, 0x00};
        // NOP, NOP, and a timestamp option: TSval 1001, TSecr 1
        m_ackOptions = {0x01, 0x01, 0x08, 0x0a, 0x00, 0x00, 0x03, 0xe9, 0x00, 0x00, 0x00, 0x01};
        Handshake(Seconds(1), synTs);
        SendRelativeAt(Seconds(2), TcpHeader::RST | TcpHeader::ACK, 2, 1);
        Simulator::Stop(Seconds(3));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_EQ((m_accepted != nullptr), true, "The connection was not accepted");
        bool timestamps = false;
        for (const auto& header : m_headers)
        {
            timestamps |= header.HasOption(TcpOption::TS);
        }
        NS_TEST_ASSERT_MSG_EQ(timestamps, true, "The timestamp option was not negotiated");
        NS_TEST_ASSERT_MSG_EQ(m_acceptedState,
                              TcpSocket::CLOSED,
                              "A RST without the timestamp option was dropped");
        NS_TEST_ASSERT_MSG_EQ(m_errorCloses, 1, "The reset was not notified as an abort");
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the RST segments reaching a connection in SYN-SENT
 *
 * In SYN-SENT, a RST is acceptable only if it acknowledges the SYN, in which
 * case the connection was refused and the application which called Connect()
 * is told so; any other RST is dropped (@RFC{9293}, Section 3.10.7.3).
 */
class TcpSynSentResetTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpSynSentResetTestCase()
        : TcpRawSegmentTestCase("A RST in SYN-SENT is acceptable only if it acknowledges the SYN")
    {
    }

  private:
    void DoRun() override
    {
        Setup(false);
        m_target = Socket::CreateSocket(m_nodes.Get(1), TcpSocketFactory::GetTypeId());
        m_target->Bind(InetSocketAddress(m_interfaces.GetAddress(1), m_targetPort));
        m_target->SetConnectCallback(MakeCallback(&TcpSynSentResetTestCase::Succeeded, this),
                                     MakeCallback(&TcpSynSentResetTestCase::Failed, this));
        m_target->SetCloseCallbacks(MakeCallback(&TcpSynSentResetTestCase::Closed, this),
                                    MakeCallback(&TcpSynSentResetTestCase::Closed, this));
        m_target->TraceConnectWithoutContext(
            "State",
            MakeCallback(&TcpSynSentResetTestCase::TargetStateChanged, this));
        Simulator::Schedule(Seconds(1),
                            &Socket::Connect,
                            m_target,
                            InetSocketAddress(m_interfaces.GetAddress(0), m_proberPort));
        Simulator::Stop(Seconds(1.5));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_EQ(m_headers.empty(), false, "The target sent no SYN");
        if (m_headers.empty())
        {
            return;
        }
        m_targetIsn = m_headers.front().GetSequenceNumber();

        // No ACK, then an ACK of something else than the SYN
        SendRelativeAt(Seconds(0), TcpHeader::RST, 0, 0);
        SendRelativeAt(Seconds(0.1), TcpHeader::RST | TcpHeader::ACK, 0, 5);
        Simulator::Stop(Seconds(0.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_state,
                              TcpSocket::SYN_SENT,
                              "A RST not acknowledging the SYN was accepted");

        SendRelativeAt(Seconds(0), TcpHeader::RST | TcpHeader::ACK, 0, 1);
        Simulator::Stop(Seconds(0.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_state, TcpSocket::CLOSED, "The refused connection is not closed");
        NS_TEST_ASSERT_MSG_EQ(m_failures, 1, "The refusal was not notified as a failure");
        NS_TEST_ASSERT_MSG_EQ(m_successes, 0, "The refusal was notified as a success");
        NS_TEST_ASSERT_MSG_EQ(m_closes, 0, "The refusal was notified as a close");
    }

    /**
     * Connection success callback.
     * @param socket The socket.
     */
    void Succeeded(Ptr<Socket> socket)
    {
        m_successes++;
    }

    /**
     * Connection failure callback.
     * @param socket The socket.
     */
    void Failed(Ptr<Socket> socket)
    {
        m_failures++;
    }

    /**
     * Close callback.
     * @param socket The socket.
     */
    void Closed(Ptr<Socket> socket)
    {
        m_closes++;
    }

    /**
     * State trace sink of the connecting socket.
     * @param oldState The previous state.
     * @param newState The new state.
     */
    void TargetStateChanged(TcpSocket::TcpStates_t oldState, TcpSocket::TcpStates_t newState)
    {
        m_state = newState;
    }

    TcpSocket::TcpStates_t m_state{TcpSocket::CLOSED}; //!< The state of the connecting socket
    uint32_t m_successes{0};                           //!< Connection success notifications
    uint32_t m_failures{0};                            //!< Connection failure notifications
    uint32_t m_closes{0};                              //!< Close notifications
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that a FIN outside the receive window is acknowledged and dropped
 *
 * The FIN segments are subject to the acceptability test of @RFC{9293},
 * Section 3.10.7.4 like any other: one outside the window does not close the
 * connection, and is answered with an acknowledgment of RCV.NXT.
 */
class TcpOutOfWindowFinTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpOutOfWindowFinTestCase()
        : TcpRawSegmentTestCase("A FIN outside the window is acknowledged and dropped")
    {
    }

  private:
    void DoRun() override
    {
        Setup();
        Handshake(Seconds(1));
        Simulator::Stop(Seconds(2));
        Simulator::Run();
        std::size_t before = m_headers.size();

        SendRelativeAt(Seconds(0), TcpHeader::FIN | TcpHeader::ACK, 0x80000000, 1);
        Simulator::Stop(Seconds(0.5));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_EQ((m_accepted != nullptr), true, "The connection was not accepted");
        NS_TEST_ASSERT_MSG_EQ(m_acceptedState,
                              TcpSocket::ESTABLISHED,
                              "A FIN outside the window closed the connection");
        NS_TEST_ASSERT_MSG_EQ(m_headers.size(),
                              before + 1,
                              "A FIN outside the window was not acknowledged");
        if (m_headers.size() == before + 1)
        {
            NS_TEST_ASSERT_MSG_EQ(m_headers.back().GetAckNumber(),
                                  SequenceNumber32(2),
                                  "The acknowledgment is not of RCV.NXT");
        }
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that the closing states acknowledge the unacceptable segments
 *
 * The acceptability test of @RFC{9293}, Section 3.10.7.4 applies to every
 * synchronized state: once the FIN of the peer is in, RCV.NXT is the only
 * acceptable sequence number, so that a keep-alive of the peer reaching a
 * connection in CLOSE-WAIT is acknowledged. A FIN retransmitted to a
 * connection in TIME-WAIT is acknowledged as well, and restarts the 2 MSL
 * timeout.
 */
class TcpClosingStatesAckTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpClosingStatesAckTestCase()
        : TcpRawSegmentTestCase("The closing states acknowledge the unacceptable segments")
    {
    }

  private:
    void DoRun() override
    {
        // CLOSE-WAIT: the peer closes, the application does not
        Setup();
        Handshake(Seconds(1));
        SendRelativeAt(Seconds(2), TcpHeader::FIN | TcpHeader::ACK, 2, 1);
        Simulator::Stop(Seconds(2.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_acceptedState,
                              TcpSocket::CLOSE_WAIT,
                              "The connection did not reach CLOSE-WAIT");
        std::size_t before = m_headers.size();
        // A keep-alive of the peer repeats the sequence number of its FIN
        SendRelativeAt(Seconds(0), TcpHeader::ACK, 2, 1);
        Simulator::Stop(Seconds(0.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_headers.size(),
                              before + 1,
                              "A keep-alive reaching CLOSE-WAIT was not acknowledged");
        if (m_headers.size() == before + 1)
        {
            NS_TEST_ASSERT_MSG_EQ(m_headers.back().GetAckNumber(),
                                  SequenceNumber32(3),
                                  "The acknowledgment is not of RCV.NXT");
        }
        Simulator::Destroy();

        // TIME-WAIT: the application closes first, then the peer, whose FIN
        // is retransmitted
        Config::SetDefault("ns3::TcpSocketBase::MaxSegLifetime", DoubleValue(1));
        Setup();
        Handshake(Seconds(1));
        Simulator::Schedule(Seconds(2), &TcpClosingStatesAckTestCase::CloseAccepted, this);
        SendRelativeAt(Seconds(2.5), TcpHeader::FIN | TcpHeader::ACK, 2, 2);
        Simulator::Stop(Seconds(3));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_acceptedState,
                              TcpSocket::TIME_WAIT,
                              "The connection did not reach TIME-WAIT");
        before = m_headers.size();
        SendRelativeAt(Seconds(1), TcpHeader::FIN | TcpHeader::ACK, 2, 2);
        Simulator::Stop(Seconds(1.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_headers.size(),
                              before + 1,
                              "A retransmitted FIN reaching TIME-WAIT was not acknowledged");
        if (m_headers.size() == before + 1)
        {
            NS_TEST_ASSERT_MSG_EQ(m_headers.back().GetAckNumber(),
                                  SequenceNumber32(3),
                                  "The acknowledgment is not of RCV.NXT");
            NS_TEST_ASSERT_MSG_EQ(m_headers.back().GetSequenceNumber(),
                                  m_targetIsn + 2,
                                  "The acknowledgment does not come after the FIN");
        }
        // TIME-WAIT was entered at 2.5 s and restarted at 4 s, for 2 s
        Simulator::Stop(Seconds(5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_closedAt,
                              Seconds(6),
                              "The retransmitted FIN did not restart the TIME-WAIT timeout");
    }

    /**
     * Close the accepted socket.
     */
    void CloseAccepted()
    {
        NS_ABORT_MSG_IF(!m_accepted, "The connection was not accepted");
        m_accepted->Close();
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that an empty segment acknowledging new data is not lost
 *
 * The acknowledgment of new data carried by a segment without data is
 * processed even when the segment lies out of the window:
 *
 * - below RCV.NXT, as the pure ACKs of a peer which rewound its sequence
 *   numbers for a retransmission;
 * - at the sequence number following a FIN which arrived out of order, while
 *   the data before the FIN is missing: the window then ends at the FIN.
 *
 * Dropping them stalls a connection carrying data in both directions.
 */
class TcpOldSequenceAckTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpOldSequenceAckTestCase()
        : TcpRawSegmentTestCase("An empty segment acknowledging new data is processed")
    {
    }

  private:
    void DoRun() override
    {
        // Below RCV.NXT
        Run(false);
        Simulator::Destroy();
        // After an out of order FIN
        Run(true);
    }

    /**
     * Run a scenario.
     * @param outOfOrderFin Whether the peer sends an out of order FIN
     *                      rather than data before the ACK.
     */
    void Run(bool outOfOrderFin)
    {
        Setup();
        Handshake(Seconds(1));
        Simulator::Stop(Seconds(2));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ((m_accepted != nullptr), true, "The connection was not accepted");
        if (!m_accepted)
        {
            return;
        }
        m_accepted->Send(Create<Packet>(1000));

        uint32_t ackSeq;
        if (outOfOrderFin)
        {
            // The 500 bytes before the FIN are missing
            SendRelativeAt(MilliSeconds(100), TcpHeader::FIN | TcpHeader::ACK, 502, 1);
            ackSeq = 503;
        }
        else
        {
            SendRelativeAt(MilliSeconds(100), TcpHeader::ACK, 2, 1, {}, 500);
            ackSeq = 2;
        }
        SendRelativeAt(MilliSeconds(200), TcpHeader::ACK, ackSeq, 1001);
        Simulator::Stop(MilliSeconds(300));
        Simulator::Run();

        Ptr<TcpSocketBase> target = DynamicCast<TcpSocketBase>(m_accepted);
        NS_TEST_ASSERT_MSG_EQ(target->GetTxBuffer()->HeadSequence(),
                              m_targetIsn + 1001,
                              "The acknowledgment of the data was dropped"
                                  << (outOfOrderFin ? " after an out of order FIN" : ""));
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that a pure ACK carries the highest sequence number sent
 *
 * A retransmission timeout rewinds the next sequence number to send down
 * to the first unacknowledged byte, below the RCV.NXT of a peer which got
 * the data but whose acknowledgments were lost. A pure ACK sent then carries
 * SND.NXT, the highest sequence number sent, so that the peer finds it in
 * its window (@RFC{9293}, Section 3.10.7.4).
 */
class TcpPureAckSequenceTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpPureAckSequenceTestCase()
        : TcpRawSegmentTestCase("A pure ACK carries the highest sequence number sent")
    {
    }

  private:
    void DoRun() override
    {
        Setup();
        Handshake(Seconds(1));
        Simulator::Stop(Seconds(2));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ((m_accepted != nullptr), true, "The connection was not accepted");
        if (!m_accepted)
        {
            return;
        }
        // Never acknowledged: the retransmission timer expires after 1 s
        m_accepted->Send(Create<Packet>(3000));
        Simulator::Stop(Seconds(1.5));
        Simulator::Run();
        std::size_t before = m_headers.size();

        // The data of the prober is acknowledged by a pure ACK
        SendRelativeAt(Seconds(0), TcpHeader::ACK, 2, 1, {}, 100);
        Simulator::Stop(Seconds(0.5));
        Simulator::Run();

        uint32_t pureAcks = 0;
        for (std::size_t i = before; i < m_headers.size(); ++i)
        {
            if (m_payloads[i] == 0 && m_headers[i].GetFlags() == TcpHeader::ACK)
            {
                ++pureAcks;
                NS_TEST_ASSERT_MSG_EQ(m_headers[i].GetSequenceNumber(),
                                      m_targetIsn + 3001,
                                      "The pure ACK does not carry SND.NXT");
                NS_TEST_ASSERT_MSG_EQ(m_headers[i].GetAckNumber(),
                                      SequenceNumber32(102),
                                      "The pure ACK does not acknowledge the data");
            }
        }
        NS_TEST_ASSERT_MSG_EQ(pureAcks, 1, "The data was not acknowledged by a pure ACK");
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that a Close() from the error close callback is harmless
 *
 * An application may close its socket from the error close callback, as
 * ThreeGppHttpServer does. With data left unread in the receive buffer, a
 * Close() sends a RST and notifies an error close: it must not recurse
 * through the callback once the connection is aborted, whether by a RST of
 * the peer or by the Close() of the application itself. The connection is
 * CLOSED, the application is notified once, and no further RST is sent.
 */
class TcpCloseFromErrorCallbackTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpCloseFromErrorCallbackTestCase()
        : TcpRawSegmentTestCase("A Close() from the error close callback is harmless")
    {
    }

  private:
    void DoRun() override
    {
        // A RST of the peer, with out of order data held
        Run(true);
        Simulator::Destroy();
        // A Close() of the application, with data unread
        Run(false);
    }

    /**
     * Run a scenario.
     * @param peerReset Whether the peer resets the connection, rather than
     *                  the application closing it.
     */
    void Run(bool peerReset)
    {
        m_callbacks = 0;
        Setup();
        Handshake(Seconds(1));
        Simulator::Stop(Seconds(2));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ((m_accepted != nullptr), true, "The connection was not accepted");
        if (!m_accepted)
        {
            return;
        }
        m_accepted->SetCloseCallbacks(
            MakeNullCallback<void, Ptr<Socket>>(),
            MakeCallback(&TcpCloseFromErrorCallbackTestCase::CloseOnError, this));
        std::size_t before = m_headers.size();

        if (peerReset)
        {
            // Data after a hole is held, then the peer resets
            SendRelativeAt(MilliSeconds(100), TcpHeader::ACK, 102, 1, {}, 50);
            SendRelativeAt(MilliSeconds(200), TcpHeader::RST, 2, 1);
        }
        else
        {
            // Data in order, never read
            SendRelativeAt(MilliSeconds(100), TcpHeader::ACK, 2, 1, {}, 50);
            Simulator::Schedule(MilliSeconds(200), &Socket::Close, m_accepted);
        }
        Simulator::Stop(Seconds(1));
        Simulator::Run();

        std::string scenario = peerReset ? " (reset by the peer)" : " (closed with unread data)";
        NS_TEST_ASSERT_MSG_EQ(m_callbacks, 1, "The error close was not notified once" << scenario);
        NS_TEST_ASSERT_MSG_EQ(m_acceptedState,
                              TcpSocket::CLOSED,
                              "The connection is not CLOSED" << scenario);
        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::RST, before),
                              (peerReset ? 0 : 1),
                              "Wrong number of RSTs sent" << scenario);
    }

    /**
     * Error close callback closing the socket, as ThreeGppHttpServer does.
     * The recursion is bounded, so that a failure does not overflow the
     * stack.
     * @param socket The socket.
     */
    void CloseOnError(Ptr<Socket> socket)
    {
        if (++m_callbacks < 5)
        {
            socket->Close();
        }
    }

    uint32_t m_callbacks{0}; //!< Error close notifications
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the urgent data sent before the connection is established
 *
 * Urgent data sent in SYN-SENT is queued until the handshake gives the
 * sequence number of its first byte: the urgent pointer must still point
 * after the last urgent byte (@RFC{9293}, Section 3.8.5, MUST-62), so that the
 * peer is told of all of them, even of a single urgent byte.
 */
class TcpUrgentSynSentTestCase : public TestCase
{
  public:
    TcpUrgentSynSentTestCase()
        : TestCase("Urgent data sent in SYN-SENT is pointed at whole")
    {
    }

  private:
    void DoRun() override
    {
        for (uint32_t size : {1, 10})
        {
            Run(size);
        }
    }

    /**
     * Run the scenario.
     * @param size The number of urgent bytes.
     */
    void Run(uint32_t size)
    {
        m_urgentSizes.clear();
        m_urgentPointers.clear();

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

        const uint16_t port = 9700;
        Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
        server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
        server->Listen();
        server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                  MakeCallback(&TcpUrgentSynSentTestCase::Accepted, this));

        Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
        client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));
        client->Send(Create<Packet>(size), Socket::MSG_FLAG_OOB);

        Simulator::Stop(Seconds(5));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_EQ(m_urgentPointers.empty(),
                              false,
                              "No urgent pointer for " << size << " urgent bytes");
        for (uint16_t pointer : m_urgentPointers)
        {
            NS_TEST_ASSERT_MSG_EQ(pointer,
                                  size,
                                  "The urgent pointer misses urgent bytes of " << size);
        }
        NS_TEST_ASSERT_MSG_EQ(m_urgentSizes.size(),
                              1,
                              "The urgent data of " << size << " bytes was not notified once");
        if (!m_urgentSizes.empty())
        {
            NS_TEST_ASSERT_MSG_EQ(m_urgentSizes.front(),
                                  size,
                                  "Wrong amount of urgent data notified");
        }
        Simulator::Destroy();
    }

    /**
     * Accept callback, watching the accepted socket.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from)
    {
        socket->SetUrgentDataCallback(MakeCallback(&TcpUrgentSynSentTestCase::Urgent, this));
        socket->TraceConnectWithoutContext(
            "Rx",
            MakeCallback(&TcpUrgentSynSentTestCase::SegmentReceived, this));
    }

    /**
     * Urgent data callback.
     * @param socket The socket.
     */
    void Urgent(Ptr<Socket> socket)
    {
        m_urgentSizes.push_back(socket->GetUrgentDataSize());
    }

    /**
     * Rx trace of the accepted socket, recording the urgent pointers.
     * @param packet The payload.
     * @param header The TCP header.
     * @param socket The receiving socket.
     */
    void SegmentReceived(Ptr<const Packet> packet,
                         const TcpHeader& header,
                         Ptr<const TcpSocketBase> socket)
    {
        if (header.GetFlags() & TcpHeader::URG)
        {
            m_urgentPointers.push_back(header.GetUrgentPointer());
        }
    }

    std::vector<uint32_t> m_urgentSizes;    //!< Urgent data sizes notified
    std::vector<uint16_t> m_urgentPointers; //!< Urgent pointers received
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the urgent field of the segment completing a passive open
 *
 * A segment which completes the handshake moves the connection from
 * SYN-RECEIVED to ESTABLISHED, and its urgent field is processed then, before
 * its data (@RFC{9293}, Section 3.10.7.4, MUST-32): the application is told of
 * the urgent data the segment carries.
 */
class TcpUrgentHandshakeAckTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpUrgentHandshakeAckTestCase()
        : TcpRawSegmentTestCase("The urgent field of the segment completing the handshake")
    {
    }

  private:
    void DoRun() override
    {
        Setup();
        m_target->SetAcceptCallback(
            MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
            MakeCallback(&TcpUrgentHandshakeAckTestCase::AcceptedUrgent, this));
        Segment syn;
        syn.flags = TcpHeader::SYN;
        syn.seq = m_proberIsn;
        syn.window = 65535;
        SendAt(Seconds(1), syn);
        Simulator::Stop(Seconds(1.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::SYN | TcpHeader::ACK),
                              1,
                              "The SYN was not answered");
        if (m_headers.empty())
        {
            return;
        }
        m_targetIsn = m_headers.back().GetSequenceNumber();

        // The handshake is completed by a segment carrying urgent data
        const uint16_t urgentBytes = 10;
        SendSegmentFull(TcpHeader::ACK | TcpHeader::URG,
                        m_proberIsn + 1,
                        (m_targetIsn + 1).GetValue(),
                        65535,
                        false,
                        urgentBytes,
                        urgentBytes);
        Simulator::Stop(Seconds(0.5));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_EQ(m_acceptedState,
                              TcpSocket::ESTABLISHED,
                              "The connection was not established");
        NS_TEST_ASSERT_MSG_EQ(m_urgentSizes.size(), 1, "The urgent data was not notified once");
        if (!m_urgentSizes.empty())
        {
            NS_TEST_ASSERT_MSG_EQ(m_urgentSizes.front(),
                                  urgentBytes,
                                  "Wrong amount of urgent data notified");
        }
    }

    /**
     * Accept callback, watching the urgent data of the accepted socket.
     * @param socket The accepted socket.
     * @param from The address of the peer.
     */
    void AcceptedUrgent(Ptr<Socket> socket, const Address& from)
    {
        Accepted(socket, from);
        socket->SetUrgentDataCallback(MakeCallback(&TcpUrgentHandshakeAckTestCase::Urgent, this));
    }

    /**
     * Urgent data callback.
     * @param socket The socket.
     */
    void Urgent(Ptr<Socket> socket)
    {
        m_urgentSizes.push_back(socket->GetUrgentDataSize());
    }

    std::vector<uint32_t> m_urgentSizes; //!< Urgent data sizes notified
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that a SYN reaching a synchronized connection gets a challenge ACK
 *
 * @RFC{9293}, Section 3.10.7.4 answers a SYN reaching a synchronized
 * connection, whatever its sequence number, with a challenge ACK
 * (@RFC{5961}, Section 4.2), and leaves the connection untouched: a
 * retransmitted SYN+ACK reaching an established connection does not
 * renegotiate the options nor reset the congestion window to its initial
 * value.
 */
class TcpSynInSynchronizedStateTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpSynInSynchronizedStateTestCase()
        : TcpRawSegmentTestCase("A SYN reaching a synchronized connection gets a challenge ACK")
    {
    }

  private:
    void DoRun() override
    {
        Setup(false);
        m_target = Socket::CreateSocket(m_nodes.Get(1), TcpSocketFactory::GetTypeId());
        m_target->Bind(InetSocketAddress(m_interfaces.GetAddress(1), m_targetPort));
        m_target->TraceConnectWithoutContext(
            "CongestionWindow",
            MakeCallback(&TcpSynInSynchronizedStateTestCase::CwndChanged, this));
        m_target->TraceConnectWithoutContext(
            "State",
            MakeCallback(&TcpSynInSynchronizedStateTestCase::TargetStateChanged, this));
        Simulator::Schedule(Seconds(1),
                            &Socket::Connect,
                            m_target,
                            InetSocketAddress(m_interfaces.GetAddress(0), m_proberPort));
        Simulator::Stop(Seconds(1.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_headers.empty(), false, "The target sent no SYN");
        if (m_headers.empty())
        {
            return;
        }
        m_targetIsn = m_headers.front().GetSequenceNumber();

        // The SYN+ACK of the prober, whose initial sequence number is 1000
        SendRelativeAt(Seconds(0), TcpHeader::SYN | TcpHeader::ACK, 1000, 1);
        Simulator::Schedule(Seconds(0.2), [this]() { m_target->Send(Create<Packet>(20000)); });
        // Acknowledge whatever was received, so that the window grows
        uint32_t initialCwnd = 0;
        Simulator::Schedule(Seconds(0.3), [this, &initialCwnd]() { initialCwnd = m_cwnd; });
        Simulator::Schedule(Seconds(0.4), &TcpSynInSynchronizedStateTestCase::AckAll, this);
        Simulator::Stop(Seconds(0.8));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_state, TcpSocket::ESTABLISHED, "The connection is not established");
        NS_TEST_ASSERT_MSG_GT(m_cwnd, initialCwnd, "The congestion window did not grow");
        uint32_t cwnd = m_cwnd;

        // The SYN+ACK is retransmitted, and a SYN is sent
        std::size_t before = m_headers.size();
        SendRelativeAt(Seconds(0), TcpHeader::SYN | TcpHeader::ACK, 1000, 1);
        SendRelativeAt(Seconds(0.1), TcpHeader::SYN, 5000, 0);
        Simulator::Stop(Seconds(0.5));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_EQ(m_cwnd, cwnd, "A SYN+ACK changed the congestion window");
        NS_TEST_ASSERT_MSG_EQ(m_state, TcpSocket::ESTABLISHED, "A SYN changed the state");
        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::ACK, before),
                              2,
                              "The SYN segments were not answered with challenge ACKs");
        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::RST, before), 0, "A SYN was answered with a RST");
        for (std::size_t i = before; i < m_headers.size(); ++i)
        {
            NS_TEST_ASSERT_MSG_EQ(m_headers[i].GetAckNumber(),
                                  SequenceNumber32(1001),
                                  "The challenge ACK does not acknowledge RCV.NXT");
        }
    }

    /**
     * Acknowledge every data byte received from the target.
     */
    void AckAll()
    {
        SequenceNumber32 highest = m_targetIsn + 1;
        for (std::size_t i = 0; i < m_headers.size(); ++i)
        {
            SequenceNumber32 end = m_headers[i].GetSequenceNumber() + m_payloads[i];
            if (m_payloads[i] > 0 && end > highest)
            {
                highest = end;
            }
        }
        Send(Make(TcpHeader::ACK, 1001, highest.GetValue()));
    }

    /**
     * Congestion window trace sink of the target.
     * @param oldValue The previous value.
     * @param newValue The new value.
     */
    void CwndChanged(uint32_t oldValue, uint32_t newValue)
    {
        m_cwnd = newValue;
    }

    /**
     * State trace sink of the target.
     * @param oldState The previous state.
     * @param newState The new state.
     */
    void TargetStateChanged(TcpSocket::TcpStates_t oldState, TcpSocket::TcpStates_t newState)
    {
        m_state = newState;
    }

    uint32_t m_cwnd{0};                                //!< The congestion window of the target
    TcpSocket::TcpStates_t m_state{TcpSocket::CLOSED}; //!< The state of the target
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that malformed options reset only an acceptable segment
 *
 * A segment with an illegal option resets the connection (@RFC{9293},
 * Section 3.1, MUST-7), but only once it passed the acceptability test of
 * Section 3.10.7.4, as recommended by @RFC{5961}: a malformed segment lying
 * outside the window, which anybody guessing the ports can forge, is answered
 * with an acknowledgment and dropped like any unacceptable segment, while one
 * at RCV.NXT resets the connection.
 */
class TcpMalformedOutOfWindowTestCase : public TcpRawSegmentTestCase
{
  public:
    /**
     * Constructor.
     * @param ipv6 Whether to run the test over IPv6.
     */
    TcpMalformedOutOfWindowTestCase(bool ipv6)
        : TcpRawSegmentTestCase("Malformed options reset the connection only in the window", ipv6)
    {
    }

  private:
    void DoRun() override
    {
        // A timestamp option announcing 10 bytes in a word of option space
        const std::vector<uint8_t> malformed = {0x08, 0x0a, 0x00, 0x00};
        Setup();
        Handshake(Seconds(1));
        Simulator::Stop(Seconds(2));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ((m_accepted != nullptr), true, "The connection was not accepted");
        std::size_t before = m_headers.size();

        SendRelativeAt(Seconds(0), TcpHeader::ACK, 0x80000000, 1, malformed);
        Simulator::Stop(Seconds(0.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_acceptedState,
                              TcpSocket::ESTABLISHED,
                              "A malformed segment outside the window reset the connection");
        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::RST, before),
                              0,
                              "A malformed segment outside the window was answered with a RST");
        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::ACK, before),
                              1,
                              "A malformed segment outside the window was not acknowledged");

        before = m_headers.size();
        SendRelativeAt(Seconds(0), TcpHeader::ACK, 2, 1, malformed);
        Simulator::Stop(Seconds(0.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_acceptedState,
                              TcpSocket::CLOSED,
                              "A malformed segment at RCV.NXT did not reset the connection");
        NS_TEST_ASSERT_MSG_EQ(m_errorCloses, 1, "The reset was not notified as an abort");
        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::RST, before), 1, "The peer was not reset");
        if (Count(TcpHeader::RST, before) == 1)
        {
            NS_TEST_ASSERT_MSG_EQ(m_headers.back().GetSequenceNumber(),
                                  m_targetIsn + 1,
                                  "The RST does not start at SND.NXT");
        }
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that malformed options refuse a connection in SYN-SENT
 *
 * A SYN+ACK with an illegal option resets the connection (@RFC{9293},
 * Section 3.1, MUST-7) if it acknowledges the SYN, which makes it acceptable
 * in SYN-SENT (Section 3.10.7.3); the application, which only has the
 * connection callbacks at that point, is told of the failure through them.
 * A malformed SYN+ACK acknowledging something else is dropped.
 */
class TcpMalformedSynSentTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpMalformedSynSentTestCase()
        : TcpRawSegmentTestCase("Malformed options in SYN-SENT fail the connection")
    {
    }

  private:
    void DoRun() override
    {
        const std::vector<uint8_t> malformed = {0x08, 0x0a, 0x00, 0x00};
        Setup(false);
        m_target = Socket::CreateSocket(m_nodes.Get(1), TcpSocketFactory::GetTypeId());
        m_target->Bind(InetSocketAddress(m_interfaces.GetAddress(1), m_targetPort));
        m_target->SetConnectCallback(MakeCallback(&TcpMalformedSynSentTestCase::Succeeded, this),
                                     MakeCallback(&TcpMalformedSynSentTestCase::Failed, this));
        m_target->SetCloseCallbacks(MakeCallback(&TcpMalformedSynSentTestCase::Closed, this),
                                    MakeCallback(&TcpMalformedSynSentTestCase::Closed, this));
        m_target->TraceConnectWithoutContext(
            "State",
            MakeCallback(&TcpMalformedSynSentTestCase::TargetStateChanged, this));
        Simulator::Schedule(Seconds(1),
                            &Socket::Connect,
                            m_target,
                            InetSocketAddress(m_interfaces.GetAddress(0), m_proberPort));
        Simulator::Stop(Seconds(1.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_headers.empty(), false, "The target sent no SYN");
        if (m_headers.empty())
        {
            return;
        }
        m_targetIsn = m_headers.front().GetSequenceNumber();

        std::size_t before = m_headers.size();
        SendRelativeAt(Seconds(0), TcpHeader::SYN | TcpHeader::ACK, 1000, 7, malformed);
        Simulator::Stop(Seconds(0.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_state,
                              TcpSocket::SYN_SENT,
                              "A malformed SYN+ACK which does not acknowledge the SYN was taken");
        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::RST, before), 0, "The prober was reset");

        SendRelativeAt(Seconds(0), TcpHeader::SYN | TcpHeader::ACK, 1000, 1, malformed);
        Simulator::Stop(Seconds(0.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_state, TcpSocket::CLOSED, "The connection is not closed");
        NS_TEST_ASSERT_MSG_EQ(m_failures, 1, "The failure was not notified");
        NS_TEST_ASSERT_MSG_EQ(m_successes, 0, "The failure was notified as a success");
        NS_TEST_ASSERT_MSG_EQ(m_closes, 0, "The failure was notified as a close");
        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::RST, before), 1, "The prober was not reset");
        if (Count(TcpHeader::RST, before) == 1)
        {
            NS_TEST_ASSERT_MSG_EQ(m_headers.back().GetSequenceNumber(),
                                  m_targetIsn + 1,
                                  "The RST does not start at the acknowledgment number");
        }
    }

    /**
     * Connection success callback.
     * @param socket The socket.
     */
    void Succeeded(Ptr<Socket> socket)
    {
        m_successes++;
    }

    /**
     * Connection failure callback.
     * @param socket The socket.
     */
    void Failed(Ptr<Socket> socket)
    {
        m_failures++;
    }

    /**
     * Close callback.
     * @param socket The socket.
     */
    void Closed(Ptr<Socket> socket)
    {
        m_closes++;
    }

    /**
     * State trace sink of the connecting socket.
     * @param oldState The previous state.
     * @param newState The new state.
     */
    void TargetStateChanged(TcpSocket::TcpStates_t oldState, TcpSocket::TcpStates_t newState)
    {
        m_state = newState;
    }

    TcpSocket::TcpStates_t m_state{TcpSocket::CLOSED}; //!< The state of the connecting socket
    uint32_t m_successes{0};                           //!< Connection success notifications
    uint32_t m_failures{0};                            //!< Connection failure notifications
    uint32_t m_closes{0};                              //!< Close notifications
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the handling of malformed options on an established connection
 *
 * Every illegal option layout resets an established connection when it
 * reaches it at RCV.NXT (@RFC{9293}, Section 3.1, MUST-7 and MUST-68): an
 * option length of 1, a length running past the 40 bytes of option space, a
 * kind needing a length as the last byte of the option space, an option
 * appearing twice, and an option of a known kind whose length does not match
 * its kind: MSS of 4 bytes, window scale of 3, SACK permitted of 2, SACK of 2
 * plus 8 per block for 1 to 4 blocks, timestamps of 10. An option filling the
 * whole option space exactly is legal. A SYN with a malformed MSS option
 * towards a closed port is reset, the simulation going on, and a window scale
 * above 14 opens the connection.
 */
class TcpMalformedOptionMatrixTestCase : public TcpRawSegmentTestCase
{
  public:
    /**
     * Constructor.
     * @param ipv6 Whether to run the test over IPv6.
     */
    TcpMalformedOptionMatrixTestCase(bool ipv6)
        : TcpRawSegmentTestCase("Malformed options reset an established connection", ipv6)
    {
    }

  private:
    void DoRun() override
    {
        // An unknown option kind, whose content is not parsed
        constexpr uint8_t UNKNOWN_KIND = 0x22;
        std::vector<uint8_t> fills40(40, 0x00);
        fills40[0] = UNKNOWN_KIND;
        fills40[1] = 40;
        std::vector<uint8_t> exceeds40 = fills40;
        exceeds40[1] = 41;

        const std::vector<std::pair<std::string, std::vector<uint8_t>>> malformed = {
            {"an option length of 1", {UNKNOWN_KIND, 0x01, 0x00, 0x00}},
            {"an option length past the option space", exceeds40},
            {"a kind without length as the last byte", {0x01, 0x01, 0x01, UNKNOWN_KIND}},
            {"a duplicate option",
             {0x02, 0x04, 0x05, 0xb4, 0x01, 0x01, 0x01, 0x01, 0x02, 0x04, 0x05, 0xb4}},
            {"an MSS option of length 2", {0x02, 0x02, 0x01, 0x00}},
            {"an MSS option of length 3", {0x02, 0x03, 0x05, 0x01}},
            {"an MSS option of length 5", {0x02, 0x05, 0x05, 0xb4, 0x00, 0x00, 0x00, 0x00}},
            {"a window scale option of length 4", {0x03, 0x04, 0x07, 0x00}},
            {"a SACK permitted option of length 3", {0x04, 0x03, 0x00, 0x00}},
            {"a SACK option of length 3", {0x05, 0x03, 0x01, 0x00}},
            {"a SACK option of length 11", {0x05, 0x0b, 0, 0, 0, 1, 0, 0, 0, 2, 0x01, 0x00}},
            {"a timestamp option of length 9", {0x08, 0x09, 0, 0, 0, 1, 0, 0, 0, 0x00, 0x00, 0x00}},
        };
        for (const auto& [description, options] : malformed)
        {
            Setup();
            Handshake(Seconds(1));
            SendRelativeAt(Seconds(2), TcpHeader::ACK, 2, 1, options);
            Simulator::Stop(Seconds(3));
            Simulator::Run();
            NS_TEST_ASSERT_MSG_EQ((m_accepted != nullptr), true, "The connection was not accepted");
            NS_TEST_ASSERT_MSG_EQ(m_acceptedState,
                                  TcpSocket::CLOSED,
                                  "The connection was not reset by " << description);
            NS_TEST_ASSERT_MSG_EQ(m_errorCloses, 1, "The abort was not notified");
            NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::RST), 1, "The peer was not reset");
            Simulator::Destroy();
        }

        Setup();
        Handshake(Seconds(1));
        SendRelativeAt(Seconds(2), TcpHeader::ACK, 2, 1, fills40, 100);
        Simulator::Stop(Seconds(3));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_acceptedState,
                              TcpSocket::ESTABLISHED,
                              "An option filling the option space reset the connection");
        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::RST), 0, "A legal option was answered with a RST");
        NS_TEST_ASSERT_MSG_EQ(m_accepted->GetRxAvailable(),
                              100,
                              "The data of a segment with a legal option was not received");
        Simulator::Destroy();

        // Towards a closed port, the segment is reset without its options
        // being looked at further
        Setup(false);
        Segment syn = Make(TcpHeader::SYN, m_proberIsn, 0, {0x02, 0x03, 0x05, 0x01});
        SendAt(Seconds(1), syn);
        Simulator::Stop(Seconds(2));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::RST),
                              1,
                              "A SYN with a malformed MSS option to a closed port was not reset");
        Simulator::Destroy();

        // A window scale above 14 is well formed, and taken as 14 (RFC 7323,
        // Section 2.3)
        Setup();
        Handshake(Seconds(1), {0x03, 0x03, 0x0f, 0x01});
        Simulator::Stop(Seconds(2));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_acceptedState,
                              TcpSocket::ESTABLISHED,
                              "A window scale above 14 did not open the connection");
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test malformed options on an IPv4 connection to an IPv6 socket
 *
 * An IPv4 segment matching no IPv4 endpoint is handed to the IPv6 sockets,
 * as a v4-mapped address, whether its options are malformed or not: a
 * malformed segment reaching a connection accepted by a socket listening on
 * the IPv6 any address resets it like any other connection, and only when it
 * is acceptable (@RFC{9293}, Section 3.1, MUST-7, and Section 3.10.7.4).
 */
class TcpMalformedV4MappedTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpMalformedV4MappedTestCase()
        : TcpRawSegmentTestCase("Malformed options reset a v4-mapped connection")
    {
    }

  private:
    void DoRun() override
    {
        const std::vector<uint8_t> malformed = {0x08, 0x0a, 0x00, 0x00};
        Setup(false);
        m_target = Socket::CreateSocket(m_nodes.Get(1), TcpSocketFactory::GetTypeId());
        m_target->Bind(Inet6SocketAddress(Ipv6Address::GetAny(), m_targetPort));
        m_target->Listen();
        m_target->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                    MakeCallback(&TcpMalformedV4MappedTestCase::Accepted, this));
        Handshake(Seconds(1));
        SendRelativeAt(Seconds(2), TcpHeader::ACK, 0x80000000, 1, malformed);
        Simulator::Stop(Seconds(2.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ((m_accepted != nullptr), true, "The connection was not accepted");
        NS_TEST_ASSERT_MSG_EQ(m_acceptedState,
                              TcpSocket::ESTABLISHED,
                              "A malformed segment outside the window reset the connection");
        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::RST),
                              0,
                              "A malformed segment outside the window was answered with a RST");

        SendRelativeAt(Seconds(0), TcpHeader::ACK, 2, 1, malformed);
        Simulator::Stop(Seconds(0.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_acceptedState,
                              TcpSocket::CLOSED,
                              "A malformed segment at RCV.NXT did not reset the connection");
        NS_TEST_ASSERT_MSG_EQ(m_errorCloses, 1, "The abort was not notified");
        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::RST), 1, "The peer was not reset");
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Error model dropping the first SYN+ACK segment it sees
 *
 * The packets are inspected as raw bytes, so that the address resolution
 * and neighbor discovery packets crossing the device are told apart from the
 * IPv4 and IPv6 datagrams carrying TCP.
 */
class TcpSynAckDropModel : public ErrorModel
{
  public:
    /**
     * Get the type ID.
     * @return The object TypeId.
     */
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::TcpSynAckDropModel")
                                .SetParent<ErrorModel>()
                                .SetGroupName("Internet")
                                .AddConstructor<TcpSynAckDropModel>();
        return tid;
    }

    /**
     * Get the number of dropped segments.
     * @return The number of dropped segments.
     */
    uint32_t GetDropped() const
    {
        return m_dropped;
    }

  private:
    bool DoCorrupt(Ptr<Packet> p) override
    {
        if (m_dropped > 0)
        {
            return false;
        }
        uint8_t bytes[60];
        uint32_t size = p->CopyData(bytes, sizeof(bytes));
        uint32_t flagsOffset = 0;
        if (size >= 34 && (bytes[0] >> 4) == 4 && bytes[9] == iana::internetprotocolnumbers::TCP)
        {
            flagsOffset = (bytes[0] & 0x0f) * 4 + 13;
        }
        else if (size >= 54 && (bytes[0] >> 4) == 6 &&
                 bytes[6] == iana::internetprotocolnumbers::TCP)
        {
            flagsOffset = 40 + 13;
        }
        if (flagsOffset == 0 || flagsOffset >= size)
        {
            return false;
        }
        constexpr uint8_t synAck = TcpHeader::SYN | TcpHeader::ACK;
        if ((bytes[flagsOffset] & synAck) == synAck)
        {
            m_dropped++;
            return true;
        }
        return false;
    }

    void DoReset() override
    {
    }

    uint32_t m_dropped{0}; //!< The number of dropped segments
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test a simultaneous open whose first SYN+ACK is lost
 *
 * Both endpoints open the connection actively and cross their SYN segments,
 * then their SYN+ACK segments (@RFC{9293}, Figure 7). One SYN+ACK is lost:
 * the endpoint which sent it has to retransmit the SYN+ACK, not its original
 * SYN, and the endpoint already established answers it with an ACK, so that
 * both ends reach ESTABLISHED.
 */
class TcpSimultaneousOpenLossTestCase : public TestCase
{
  public:
    /**
     * Constructor.
     * @param ipv6 Whether to run the test over IPv6.
     */
    TcpSimultaneousOpenLossTestCase(bool ipv6)
        : TestCase(std::string("A simultaneous open survives the loss of a SYN+ACK") +
                   (ipv6 ? " (IPv6)" : "")),
          m_ipv6(ipv6)
    {
    }

  private:
    void DoRun() override
    {
        Config::SetDefault("ns3::Icmpv6L4Protocol::DAD", BooleanValue(false));
        NodeContainer nodes;
        nodes.Create(2);
        SimpleNetDeviceHelper devHelper;
        devHelper.SetChannelAttribute("Delay", TimeValue(MilliSeconds(10)));
        NetDeviceContainer devices = devHelper.Install(nodes);
        InternetStackHelper stack;
        stack.Install(nodes);
        Address addressA;
        Address addressB;
        const uint16_t portA = 9701;
        const uint16_t portB = 9702;
        if (m_ipv6)
        {
            Ipv6AddressHelper address;
            address.SetBase(Ipv6Address("2001:db8::"), Ipv6Prefix(64));
            Ipv6InterfaceContainer interfaces = address.Assign(devices);
            addressA = Inet6SocketAddress(interfaces.GetAddress(0, 1), portA);
            addressB = Inet6SocketAddress(interfaces.GetAddress(1, 1), portB);
        }
        else
        {
            Ipv4AddressHelper address;
            address.SetBase("10.1.1.0", "255.255.255.0");
            Ipv4InterfaceContainer interfaces = address.Assign(devices);
            addressA = InetSocketAddress(interfaces.GetAddress(0), portA);
            addressB = InetSocketAddress(interfaces.GetAddress(1), portB);
        }

        // The SYN+ACK of A is lost on its way to B
        Ptr<TcpSynAckDropModel> drop = CreateObject<TcpSynAckDropModel>();
        devices.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(drop));

        Ptr<Socket> a = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
        a->Bind(addressA);
        a->SetConnectCallback(MakeCallback(&TcpSimultaneousOpenLossTestCase::Connected, this),
                              MakeCallback(&TcpSimultaneousOpenLossTestCase::Failed, this));
        Ptr<Socket> b = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
        b->Bind(addressB);
        b->SetConnectCallback(MakeCallback(&TcpSimultaneousOpenLossTestCase::Connected, this),
                              MakeCallback(&TcpSimultaneousOpenLossTestCase::Failed, this));
        b->TraceConnectWithoutContext("Tx",
                                      MakeCallback(&TcpSimultaneousOpenLossTestCase::TxB, this));

        Simulator::Schedule(Seconds(1), &Socket::Connect, a, addressB);
        Simulator::Schedule(Seconds(1), &Socket::Connect, b, addressA);
        Simulator::Stop(Seconds(30));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_EQ(drop->GetDropped(), 1, "No SYN+ACK was lost");
        NS_TEST_ASSERT_MSG_EQ(m_failed, 0, "An endpoint of the simultaneous open gave up");
        NS_TEST_ASSERT_MSG_EQ(m_connected, 2, "The simultaneous open did not establish both ends");
        NS_TEST_ASSERT_MSG_GT_OR_EQ(m_synAcks,
                                    2,
                                    "The SYN+ACK left unacknowledged was not retransmitted");
        NS_TEST_ASSERT_MSG_EQ(m_bareSynsAfterSynAck,
                              0,
                              "A bare SYN was retransmitted after the SYN+ACK");

        Simulator::Destroy();
        Config::Reset();
    }

    /**
     * Connection succeeded callback.
     * @param socket The connected socket.
     */
    void Connected(Ptr<Socket> socket)
    {
        m_connected++;
    }

    /**
     * Connection failed callback.
     * @param socket The socket which failed to connect.
     */
    void Failed(Ptr<Socket> socket)
    {
        m_failed++;
    }

    /**
     * Transmission trace sink of the endpoint which does not receive the
     * SYN+ACK of its peer, and has to retransmit its own.
     * @param packet The packet.
     * @param header The TCP header.
     * @param socket The socket.
     */
    void TxB(Ptr<const Packet> packet, const TcpHeader& header, Ptr<const TcpSocketBase> socket)
    {
        uint8_t flags = header.GetFlags() & (TcpHeader::SYN | TcpHeader::ACK);
        if (flags == (TcpHeader::SYN | TcpHeader::ACK))
        {
            m_synAcks++;
        }
        else if (flags == TcpHeader::SYN && m_synAcks > 0)
        {
            m_bareSynsAfterSynAck++;
        }
    }

    bool m_ipv6;                       //!< Whether the test runs over IPv6
    uint32_t m_connected{0};           //!< Number of endpoints which reached ESTABLISHED
    uint32_t m_failed{0};              //!< Number of endpoints which gave up
    uint32_t m_synAcks{0};             //!< SYN+ACK segments sent by the endpoint left waiting
    uint32_t m_bareSynsAfterSynAck{0}; //!< SYN segments it sent after its first SYN+ACK
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the retransmissions of a simultaneous open which gets no answer
 *
 * Once a SYN reaches an endpoint in SYN-SENT, the SYN+ACK it answers with is
 * the segment retransmitted (@RFC{9293}, Section 3.10.7.3), and the
 * retransmissions draw from the retries the SYN left rather than starting
 * over, so that a peer which goes silent does not double the time the
 * connection takes to fail.
 */
class TcpSimultaneousOpenRetriesTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpSimultaneousOpenRetriesTestCase()
        : TcpRawSegmentTestCase("A simultaneous open keeps the retries of its SYN")
    {
    }

  private:
    void DoRun() override
    {
        constexpr uint32_t synRetries = 3;
        Setup(false);
        m_target = Socket::CreateSocket(m_nodes.Get(1), TcpSocketFactory::GetTypeId());
        m_target->SetAttribute("ConnCount", UintegerValue(synRetries));
        m_target->Bind(InetSocketAddress(m_interfaces.GetAddress(1), m_targetPort));
        m_target->SetConnectCallback(
            MakeCallback(&TcpSimultaneousOpenRetriesTestCase::Succeeded, this),
            MakeCallback(&TcpSimultaneousOpenRetriesTestCase::Failed, this));
        Simulator::Schedule(Seconds(1),
                            &Socket::Connect,
                            m_target,
                            InetSocketAddress(m_interfaces.GetAddress(0), m_proberPort));
        // A SYN crossing the one of the target, after which the prober is
        // silent
        SendAt(Seconds(1.5), Make(TcpHeader::SYN, 1000, 0));
        Simulator::Stop(Seconds(100));
        Simulator::Run();

        uint32_t syns = 0;
        uint32_t bareSynsAfterSyn = 0;
        for (std::size_t i = 0; i < m_headers.size(); ++i)
        {
            if (m_headers[i].GetFlags() & TcpHeader::SYN)
            {
                syns++;
                if (i > 0 && !(m_headers[i].GetFlags() & TcpHeader::ACK))
                {
                    bareSynsAfterSyn++;
                }
            }
        }
        NS_TEST_ASSERT_MSG_EQ(syns,
                              synRetries,
                              "The SYN and SYN+ACK segments exceed the retries of the SYN");
        NS_TEST_ASSERT_MSG_EQ(bareSynsAfterSyn,
                              0,
                              "A bare SYN was retransmitted after the SYN of the peer");
        NS_TEST_ASSERT_MSG_EQ(m_failures, 1, "The failure was not notified");
        NS_TEST_ASSERT_MSG_EQ(m_successes, 0, "The failure was notified as a success");
    }

    /**
     * Connection success callback.
     * @param socket The socket.
     */
    void Succeeded(Ptr<Socket> socket)
    {
        m_successes++;
    }

    /**
     * Connection failure callback.
     * @param socket The socket.
     */
    void Failed(Ptr<Socket> socket)
    {
        m_failures++;
    }

    uint32_t m_successes{0}; //!< Connection success notifications
    uint32_t m_failures{0};  //!< Connection failure notifications
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test a simultaneous open whose SYN+ACK carries other options than its SYN
 *
 * The options of a connection are negotiated by the SYN segments: in a
 * simultaneous open, the SYN+ACK completing the handshake repeats the SYN of
 * the peer, which was processed already, so that options it would drop do
 * not renegotiate the connection. The peer offers the timestamp and window
 * scale options on its SYN, not on its SYN+ACK, and the connection keeps the
 * timestamps.
 */
class TcpSimultaneousOpenAsymmetricOptionsTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpSimultaneousOpenAsymmetricOptionsTestCase()
        : TcpRawSegmentTestCase(
              "A simultaneous open negotiates the options of the SYN, not of the SYN+ACK")
    {
    }

  private:
    void DoRun() override
    {
        Setup(false);
        m_target = Socket::CreateSocket(m_nodes.Get(1), TcpSocketFactory::GetTypeId());
        m_target->Bind(InetSocketAddress(m_interfaces.GetAddress(1), m_targetPort));
        m_target->SetConnectCallback(
            MakeCallback(&TcpSimultaneousOpenAsymmetricOptionsTestCase::Succeeded, this),
            MakeNullCallback<void, Ptr<Socket>>());
        Simulator::Schedule(Seconds(1),
                            &Socket::Connect,
                            m_target,
                            InetSocketAddress(m_interfaces.GetAddress(0), m_proberPort));
        Simulator::Stop(Seconds(1.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_headers.empty(), false, "The target sent no SYN");
        if (m_headers.empty())
        {
            return;
        }
        m_targetIsn = m_headers.front().GetSequenceNumber();

        // NOP, window scale 2, NOP, NOP, timestamp (TSval 1000, TSecr 0)
        const std::vector<uint8_t> synOptions = {0x01,
                                                 0x03,
                                                 0x03,
                                                 0x02,
                                                 0x01,
                                                 0x01,
                                                 0x08,
                                                 0x0a,
                                                 0x00,
                                                 0x00,
                                                 0x03,
                                                 0xe8,
                                                 0x00,
                                                 0x00,
                                                 0x00,
                                                 0x00};
        SendRelativeAt(Seconds(0), TcpHeader::SYN, 1000, 0, synOptions);
        SendRelativeAt(Seconds(0.2), TcpHeader::SYN | TcpHeader::ACK, 1000, 1);
        Simulator::Schedule(Seconds(0.4), [this]() { m_target->Send(Create<Packet>(500)); });
        Simulator::Stop(Seconds(1));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_EQ(m_connected, true, "The simultaneous open did not succeed");
        bool sawData = false;
        for (std::size_t i = 0; i < m_headers.size(); ++i)
        {
            if (m_payloads[i] > 0)
            {
                sawData = true;
                NS_TEST_ASSERT_MSG_EQ(m_headers[i].HasOption(TcpOption::TS),
                                      true,
                                      "The SYN+ACK renegotiated the timestamp option away");
            }
        }
        NS_TEST_ASSERT_MSG_EQ(sawData, true, "The target sent no data");
    }

    /**
     * Connection success callback.
     * @param socket The socket.
     */
    void Succeeded(Ptr<Socket> socket)
    {
        m_connected = true;
    }

    bool m_connected{false}; //!< Whether the connection succeeded
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test a connection whose own initial sequence number is about to wrap
 *
 * With the clock driven initial sequence numbers, the target picks one a few
 * thousand bytes short of the end of the sequence space, so that its first
 * data segments wrap around zero: they must be sent, acknowledged and
 * accounted for in the wrap aware arithmetic, the highest acknowledgment
 * tracked for FACK included.
 */
class TcpOwnIsnWrapTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpOwnIsnWrapTestCase()
        : TcpRawSegmentTestCase("The first data segments wrap around the sequence space")
    {
    }

  private:
    void DoRun() override
    {
        Config::SetDefault("ns3::TcpL4Protocol::ClockDrivenIsn", BooleanValue(true));
        Config::SetDefault("ns3::TcpSocketBase::Fack", BooleanValue(true));
        const uint32_t wantedIsn = 0xffffffff - 2000;
        Setup();
        HandshakeAtIsn(wantedIsn);
        NS_TEST_ASSERT_MSG_EQ(m_targetIsn.GetValue(), wantedIsn, "The target picked another ISN");
        NS_TEST_ASSERT_MSG_EQ((m_accepted != nullptr), true, "The connection was not accepted");
        if (!m_accepted)
        {
            return;
        }

        const uint32_t bytes = 10000;
        m_accepted->Send(Create<Packet>(bytes));
        // The first segment alone, which does not wrap yet, then everything
        // received
        Simulator::Schedule(Seconds(0.1), [this]() {
            Send(Make(TcpHeader::ACK, m_proberIsn + 1, (m_targetIsn + 1 + 536).GetValue()));
        });
        for (uint32_t i = 1; i <= 10; ++i)
        {
            Simulator::Schedule(Seconds(0.1 * i + 0.05), &TcpOwnIsnWrapTestCase::AckAll, this);
        }
        Simulator::Stop(Seconds(2));
        Simulator::Run();

        SequenceNumber32 first = m_targetIsn + 1;
        SequenceNumber32 end = first + bytes;
        SequenceNumber32 highest = first;
        for (std::size_t i = 0; i < m_headers.size(); ++i)
        {
            if (m_payloads[i] == 0)
            {
                continue;
            }
            SequenceNumber32 seq = m_headers[i].GetSequenceNumber();
            NS_TEST_ASSERT_MSG_EQ((seq >= first && seq + m_payloads[i] <= end),
                                  true,
                                  "A segment was sent outside the data of the connection: " << seq);
            if (seq + m_payloads[i] > highest)
            {
                highest = seq + m_payloads[i];
            }
        }
        NS_TEST_ASSERT_MSG_EQ(highest, end, "Not every byte was sent");
        UintegerValue sndBufSize;
        m_accepted->GetAttribute("SndBufSize", sndBufSize);
        NS_TEST_ASSERT_MSG_EQ(m_accepted->GetTxAvailable(),
                              sndBufSize.Get(),
                              "Not every byte was acknowledged");
        NS_TEST_ASSERT_MSG_EQ(DynamicCast<TcpSocketBase>(m_accepted)->GetSndFack(),
                              end.GetValue(),
                              "The FACK mark did not follow the acknowledgments across zero");
    }

    /**
     * Acknowledge every data byte received from the target.
     */
    void AckAll()
    {
        SequenceNumber32 highest = m_targetIsn + 1;
        for (std::size_t i = 0; i < m_headers.size(); ++i)
        {
            SequenceNumber32 end = m_headers[i].GetSequenceNumber() + m_payloads[i];
            if (m_payloads[i] > 0 && end > highest)
            {
                highest = end;
            }
        }
        Send(Make(TcpHeader::ACK, m_proberIsn + 1, highest.GetValue()));
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the clock of the initial sequence numbers
 *
 * @RFC{9293}, Section 3.4.1 (MUST-8) drives the initial sequence numbers by a
 * clock ticking every 4 microseconds, offset by a hash of the connection
 * which does not change over time: two connections between the same sockets
 * opened one second apart pick initial sequence numbers 250000 apart.
 */
class TcpIsnClockTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpIsnClockTestCase()
        : TcpRawSegmentTestCase("The initial sequence numbers tick every 4 microseconds")
    {
    }

  private:
    void DoRun() override
    {
        Config::SetDefault("ns3::TcpL4Protocol::ClockDrivenIsn", BooleanValue(true));
        Setup();
        // Resolve the addresses first, so that the segments cross the link
        // without delay (see HandshakeAtIsn)
        Segment refresh = Make(TcpHeader::ACK, 0, 0);
        refresh.dport = m_targetPort + 1;
        SendAt(Seconds(0.5), refresh);
        SendAt(Seconds(1), Make(TcpHeader::SYN, m_proberIsn, 0));
        Simulator::Stop(Seconds(1.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::SYN | TcpHeader::ACK),
                              1,
                              "The SYN was not answered");
        if (m_headers.empty())
        {
            return;
        }
        SequenceNumber32 firstIsn = m_headers.back().GetSequenceNumber();
        Time firstAt = m_times.back();
        Send(Make(TcpHeader::RST, m_proberIsn + 1, 0));

        // The same connection, one second later
        SendAt(firstAt + Seconds(1) - Simulator::Now(), Make(TcpHeader::SYN, m_proberIsn, 0));
        Simulator::Stop(Seconds(1.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::SYN | TcpHeader::ACK),
                              2,
                              "The second SYN was not answered");
        NS_TEST_ASSERT_MSG_EQ(m_times.back(), firstAt + Seconds(1), "The second SYN was delayed");
        NS_TEST_ASSERT_MSG_EQ(m_headers.back().GetSequenceNumber() - firstIsn,
                              250000,
                              "The initial sequence number did not advance with the clock");
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that a retransmitted SYN keeps the initial sequence number
 *
 * The initial sequence number is picked when the connection is opened, not
 * when each SYN is sent: with the clock driven initial sequence numbers, the
 * retransmissions of the SYN repeat the sequence number of the first one.
 */
class TcpSynRetransmissionIsnTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpSynRetransmissionIsnTestCase()
        : TcpRawSegmentTestCase("A retransmitted SYN keeps the initial sequence number")
    {
    }

  private:
    void DoRun() override
    {
        Config::SetDefault("ns3::TcpL4Protocol::ClockDrivenIsn", BooleanValue(true));
        Setup(false);
        m_target = Socket::CreateSocket(m_nodes.Get(1), TcpSocketFactory::GetTypeId());
        m_target->Bind(InetSocketAddress(m_interfaces.GetAddress(1), m_targetPort));
        Simulator::Schedule(Seconds(1),
                            &Socket::Connect,
                            m_target,
                            InetSocketAddress(m_interfaces.GetAddress(0), m_proberPort));
        Simulator::Stop(Seconds(20));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_GT_OR_EQ(Count(TcpHeader::SYN), 3, "The SYN was not retransmitted");
        if (m_headers.empty())
        {
            return;
        }
        NS_TEST_ASSERT_MSG_NE(m_headers.front().GetSequenceNumber(),
                              SequenceNumber32(0),
                              "The initial sequence number is not clock driven");
        for (const auto& header : m_headers)
        {
            NS_TEST_ASSERT_MSG_EQ(header.GetSequenceNumber(),
                                  m_headers.front().GetSequenceNumber(),
                                  "A retransmitted SYN changed the initial sequence number");
        }
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the CE marks of a peer whose initial sequence number is high
 *
 * The sequence number of the last segment marked with CE is compared with
 * the ones of the incoming segments in the wrap aware arithmetic, so it has
 * to start from the initial sequence number of the peer: from zero, it would
 * be ahead of every sequence number in the upper half of the space, and the
 * CE marks of such a peer would never be echoed (@RFC{3168}, Section 6.1.3).
 */
class TcpEcnCeHighIsnTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpEcnCeHighIsnTestCase()
        : TcpRawSegmentTestCase("The CE marks of a peer with a high ISN are echoed")
    {
    }

  private:
    void DoRun() override
    {
        Config::SetDefault("ns3::TcpSocketBase::UseEcn", StringValue("On"));
        m_proberIsn = 0x90000000;
        m_synFlags = TcpHeader::SYN | TcpHeader::ECE | TcpHeader::CWR;
        Setup();
        Handshake(Seconds(1));
        Simulator::Stop(Seconds(2));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ((m_accepted != nullptr), true, "The connection was not accepted");
        std::size_t before = m_headers.size();

        Segment data = Make(TcpHeader::ACK, m_proberIsn + 1, (m_targetIsn + 1).GetValue(), {}, 100);
        data.tos = Ipv4Header::ECN_CE;
        Send(data);
        Simulator::Stop(Seconds(1));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_GT_OR_EQ(Count(TcpHeader::ACK | TcpHeader::ECE, before),
                                    1,
                                    "The CE mark was not echoed");
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the ECN echoes reaching a sender whose initial sequence number is high
 *
 * The acknowledgment number of the last ECN echo, and the sequence number of
 * the last segment sent with CWR, are compared in the wrap aware arithmetic,
 * so they have to start from the initial sequence number of the sender: an
 * echo in the upper half of the space must still reduce the congestion
 * window, which the next new segment tells the peer with CWR (@RFC{3168},
 * Section 6.1.2).
 */
class TcpEcnEchoHighIsnTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpEcnEchoHighIsnTestCase()
        : TcpRawSegmentTestCase("A sender with a high ISN reacts to the ECN echoes")
    {
    }

  private:
    void DoRun() override
    {
        Config::SetDefault("ns3::TcpL4Protocol::ClockDrivenIsn", BooleanValue(true));
        Config::SetDefault("ns3::TcpSocketBase::UseEcn", StringValue("On"));
        m_synFlags = TcpHeader::SYN | TcpHeader::ECE | TcpHeader::CWR;
        Setup();
        HandshakeAtIsn(0xf0000000);
        NS_TEST_ASSERT_MSG_EQ((m_accepted != nullptr), true, "The connection was not accepted");
        if (!m_accepted)
        {
            return;
        }
        m_accepted->Send(Create<Packet>(20000));
        Simulator::Schedule(Seconds(0.1), [this]() {
            Send(Make(TcpHeader::ACK | TcpHeader::ECE,
                      m_proberIsn + 1,
                      (m_targetIsn + 1 + 536).GetValue()));
        });
        Simulator::Stop(Seconds(0.5));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_GT_OR_EQ(Count(TcpHeader::CWR),
                                    1,
                                    "The ECN echo did not reduce the congestion window");
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the window updates from a peer whose initial sequence number is high
 *
 * The send window is taken from a segment carrying new data even when it
 * shrinks (@RFC{9293}, Section 3.10.7.4, SND.WL1 < SEG.SEQ), which the
 * highest sequence number received tells: on the active side, it has to start
 * from the initial sequence number of the peer, or the updates of a peer in
 * the upper half of the space would be ignored.
 */
class TcpWindowUpdateHighIsnTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpWindowUpdateHighIsnTestCase()
        : TcpRawSegmentTestCase("The window updates of a peer with a high ISN are taken")
    {
    }

  private:
    void DoRun() override
    {
        Setup(false);
        m_target = Socket::CreateSocket(m_nodes.Get(1), TcpSocketFactory::GetTypeId());
        m_target->Bind(InetSocketAddress(m_interfaces.GetAddress(1), m_targetPort));
        m_target->TraceConnectWithoutContext(
            "RWND",
            MakeCallback(&TcpWindowUpdateHighIsnTestCase::RwndChanged, this));
        Simulator::Schedule(Seconds(1),
                            &Socket::Connect,
                            m_target,
                            InetSocketAddress(m_interfaces.GetAddress(0), m_proberPort));
        Simulator::Stop(Seconds(1.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_headers.empty(), false, "The target sent no SYN");
        if (m_headers.empty())
        {
            return;
        }
        m_targetIsn = m_headers.front().GetSequenceNumber();

        const uint32_t peerIsn = 0x90000000;
        SendRelativeAt(Seconds(0), TcpHeader::SYN | TcpHeader::ACK, peerIsn, 1);
        Simulator::Stop(Seconds(0.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_rwnd, 65535, "The window of the SYN+ACK was not taken");

        Simulator::Schedule(Seconds(0), [this]() {
            Segment data = Make(TcpHeader::ACK, peerIsn + 1, (m_targetIsn + 1).GetValue(), {}, 100);
            data.window = 2000;
            Send(data);
        });
        Simulator::Stop(Seconds(0.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_rwnd, 2000, "The window of a segment with new data was ignored");
    }

    /**
     * Receive window trace sink of the target.
     * @param oldValue The previous value.
     * @param newValue The new value.
     */
    void RwndChanged(uint32_t oldValue, uint32_t newValue)
    {
        m_rwnd = newValue;
    }

    uint32_t m_rwnd{0}; //!< The send window of the target
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that a SYN towards the directed broadcast of another interface is dropped
 *
 * The IP layer of a multihomed host delivers locally a datagram towards the
 * directed broadcast address of any of its subnets, whichever interface it
 * came through: a SYN addressed to the one of another interface than the
 * arrival one is still a SYN towards a broadcast address, which @RFC{9293},
 * Section 3.10.7.2 (MUST-57) requires to be discarded.
 */
class TcpOtherInterfaceBroadcastSynTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpOtherInterfaceBroadcastSynTestCase()
        : TcpRawSegmentTestCase("A SYN towards the broadcast of another interface is dropped")
    {
    }

  private:
    void DoRun() override
    {
        Setup();

        // A second interface of the target, on a subnet of its own
        Ptr<SimpleNetDevice> device = CreateObject<SimpleNetDevice>();
        device->SetAddress(Mac48Address::Allocate());
        device->SetChannel(CreateObject<SimpleChannel>());
        m_nodes.Get(1)->AddDevice(device);
        Ptr<Ipv4> ipv4 = m_nodes.Get(1)->GetObject<Ipv4>();
        int32_t interface = ipv4->AddInterface(device);
        ipv4->AddAddress(interface, Ipv4InterfaceAddress("10.2.2.1", "255.255.255.0"));
        ipv4->SetUp(interface);

        // The prober reaches that subnet through the target
        Ipv4StaticRoutingHelper routing;
        Ptr<Ipv4StaticRouting> proberRouting =
            routing.GetStaticRouting(m_nodes.Get(0)->GetObject<Ipv4>());
        proberRouting->AddNetworkRouteTo("10.2.2.0",
                                         "255.255.255.0",
                                         m_interfaces.GetAddress(1),
                                         1);

        Simulator::Schedule(Seconds(1),
                            &TcpOtherInterfaceBroadcastSynTestCase::SendSynTo,
                            this,
                            Ipv4Address("10.2.2.255"));
        // The unicast address of the other interface is served
        Simulator::Schedule(Seconds(2),
                            &TcpOtherInterfaceBroadcastSynTestCase::SendSynTo,
                            this,
                            Ipv4Address("10.2.2.1"));
        Simulator::Stop(Seconds(1.5));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(m_headers.size(),
                              0,
                              "A SYN towards the broadcast of another interface was answered");

        Simulator::Stop(Seconds(1));
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::SYN | TcpHeader::ACK),
                              1,
                              "A SYN towards the address of another interface was not answered");
    }

    /**
     * Inject a SYN towards the given destination.
     * @param destination The destination address.
     */
    void SendSynTo(Ipv4Address destination)
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
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that an OPEN towards an unspecified or directed broadcast address is rejected
 *
 * @RFC{9293}, Section 3.9.1.5 (MUST-46) requires a local OPEN call for an
 * invalid remote IP address to be rejected as an error: besides the
 * broadcast and multicast addresses, the unspecified addresses of IPv4 and
 * IPv6 cannot name a peer, and neither can the directed broadcast address of
 * a subnet the node is attached to.
 */
class TcpInvalidRemoteAddressExtraTestCase : public TestCase
{
  public:
    TcpInvalidRemoteAddressExtraTestCase()
        : TestCase("Connect to an unspecified or directed broadcast address is rejected (MUST-46)")
    {
    }

  private:
    void DoRun() override
    {
        Config::SetDefault("ns3::Icmpv6L4Protocol::DAD", BooleanValue(false));
        NodeContainer nodes;
        nodes.Create(2);
        SimpleNetDeviceHelper devHelper;
        NetDeviceContainer devices = devHelper.Install(nodes);
        InternetStackHelper stack;
        stack.Install(nodes);
        Ipv4AddressHelper address;
        address.SetBase("10.1.1.0", "255.255.255.0");
        address.Assign(devices);
        Ipv6AddressHelper address6;
        address6.SetBase(Ipv6Address("2001:db8::"), Ipv6Prefix(64));
        address6.Assign(devices);

        const std::vector<Address> invalid = {
            InetSocketAddress(Ipv4Address::GetAny(), 9000),
            InetSocketAddress(Ipv4Address("10.1.1.255"), 9000),
            Inet6SocketAddress(Ipv6Address::GetAny(), 9000),
            Inet6SocketAddress(Ipv6Address::MakeIpv4MappedAddress(Ipv4Address::GetAny()), 9000),
        };
        for (const auto& peer : invalid)
        {
            Ptr<Socket> socket = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
            NS_TEST_ASSERT_MSG_EQ(socket->Connect(peer),
                                  -1,
                                  "Connect to an invalid remote address succeeded");
            NS_TEST_ASSERT_MSG_EQ(socket->GetErrno(),
                                  Socket::ERROR_INVAL,
                                  "Connect to an invalid remote address did not set ERROR_INVAL");
        }

        // A unicast address whose host part is all ones, in a subnet the node
        // is not attached to, is a valid peer
        Ptr<Socket> socket = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
        Ptr<Ipv4StaticRouting> routing =
            Ipv4StaticRoutingHelper().GetStaticRouting(nodes.Get(0)->GetObject<Ipv4>());
        routing->SetDefaultRoute("10.1.1.2", 1);
        NS_TEST_ASSERT_MSG_EQ(socket->Connect(InetSocketAddress(Ipv4Address("10.2.255.255"), 9000)),
                              0,
                              "Connect to a remote address was rejected");

        Simulator::Destroy();
        Config::Reset();
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the default MSS of an IPv4 peer reaching an IPv6 socket
 *
 * Without an MSS option, the send MSS defaults to 536 bytes over IPv4 and to
 * 1220 bytes over IPv6 (@RFC{9293}, Section 3.7.1, MUST-15). A connection
 * accepted by a socket listening on the IPv6 any address from an IPv4 peer,
 * through a v4-mapped address, runs over IPv4 and takes the IPv4 default.
 */
class TcpV4MappedDefaultMssTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpV4MappedDefaultMssTestCase()
        : TcpRawSegmentTestCase("A v4-mapped peer without MSS option gets the IPv4 default")
    {
    }

  private:
    void DoRun() override
    {
        Setup(false);
        m_target = Socket::CreateSocket(m_nodes.Get(1), TcpSocketFactory::GetTypeId());
        m_target->SetAttribute("SegmentSize", UintegerValue(1400));
        m_target->Bind(Inet6SocketAddress(Ipv6Address::GetAny(), m_targetPort));
        m_target->Listen();
        m_target->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                    MakeCallback(&TcpV4MappedDefaultMssTestCase::Accepted, this));
        Handshake(Seconds(1));
        Simulator::Schedule(Seconds(2), [this]() {
            NS_ABORT_MSG_IF(!m_accepted, "The connection was not accepted");
            m_accepted->Send(Create<Packet>(3000));
        });
        Simulator::Stop(Seconds(3));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_EQ((m_accepted != nullptr), true, "The connection was not accepted");
        uint32_t largest = 0;
        for (uint32_t payload : m_payloads)
        {
            largest = std::max(largest, payload);
        }
        NS_TEST_ASSERT_MSG_EQ(largest, 536, "The segments are not sized by the IPv4 default MSS");
    }
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test the end to end discard of an ICMP Source Quench by TCP
 *
 * An ICMP Source Quench message crossing the network is handed by the IP layer
 * to the transport protocol of the datagram it quotes: a UDP socket is told of
 * it, while TCP silently discards it (@RFC{9293}, Section 3.9.2.2, MUST-55),
 * leaving the connection untouched.
 */
class TcpIcmpSourceQuenchEndToEndTestCase : public TestCase
{
  public:
    TcpIcmpSourceQuenchEndToEndTestCase()
        : TestCase("An ICMP Source Quench crossing the network is discarded by TCP (MUST-55)")
    {
    }

  private:
    void DoRun() override
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
        const Ipv4Address local = interfaces.GetAddress(0);
        const Ipv4Address remote = interfaces.GetAddress(1);
        const uint16_t tcpPort = 9801;
        const uint16_t udpPort = 9802;
        const uint16_t localPort = 9803;

        Ptr<Socket> server = Socket::CreateSocket(nodes.Get(1), TcpSocketFactory::GetTypeId());
        server->Bind(InetSocketAddress(Ipv4Address::GetAny(), tcpPort));
        server->Listen();

        Ptr<Socket> tcp = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
        tcp->Bind(InetSocketAddress(local, localPort));
        tcp->SetAttribute(
            "IcmpCallback",
            CallbackValue(MakeCallback(&TcpIcmpSourceQuenchEndToEndTestCase::TcpIcmp, this)));
        tcp->TraceConnectWithoutContext(
            "State",
            MakeCallback(&TcpIcmpSourceQuenchEndToEndTestCase::StateChanged, this));
        tcp->Connect(InetSocketAddress(remote, tcpPort));

        Ptr<Socket> udp = Socket::CreateSocket(nodes.Get(0), UdpSocketFactory::GetTypeId());
        udp->Bind(InetSocketAddress(local, localPort));
        udp->Connect(InetSocketAddress(remote, udpPort));
        udp->SetAttribute(
            "IcmpCallback",
            CallbackValue(MakeCallback(&TcpIcmpSourceQuenchEndToEndTestCase::UdpIcmp, this)));

        // The remote node quenches a TCP segment and a UDP datagram of the
        // local node, quoting their headers
        Ptr<Socket> icmp = Socket::CreateSocket(nodes.Get(1), Ipv4RawSocketFactory::GetTypeId());
        icmp->SetAttribute("Protocol", UintegerValue(1));
        Simulator::Schedule(Seconds(1),
                            &TcpIcmpSourceQuenchEndToEndTestCase::SendQuench,
                            icmp,
                            local,
                            remote,
                            iana::internetprotocolnumbers::TCP,
                            localPort,
                            tcpPort);
        Simulator::Schedule(Seconds(1),
                            &TcpIcmpSourceQuenchEndToEndTestCase::SendQuench,
                            icmp,
                            local,
                            remote,
                            iana::internetprotocolnumbers::UDP,
                            localPort,
                            udpPort);
        Simulator::Stop(Seconds(2));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_EQ(m_udpQuenches,
                              1,
                              "The Source Quench was not handed to the transport protocol");
        NS_TEST_ASSERT_MSG_EQ(m_tcpIcmps, 0, "TCP did not discard the Source Quench");
        NS_TEST_ASSERT_MSG_EQ(m_state,
                              TcpSocket::ESTABLISHED,
                              "The Source Quench affected the connection");

        Simulator::Destroy();
    }

    /**
     * Send an ICMP Source Quench message quoting a datagram of the local node.
     * @param socket The raw ICMP socket of the remote node.
     * @param local The address of the local node.
     * @param remote The address of the remote node.
     * @param protocol The transport protocol of the quoted datagram.
     * @param localPort The source port of the quoted datagram.
     * @param remotePort The destination port of the quoted datagram.
     */
    static void SendQuench(Ptr<Socket> socket,
                           Ipv4Address local,
                           Ipv4Address remote,
                           uint8_t protocol,
                           uint16_t localPort,
                           uint16_t remotePort)
    {
        Ipv4Header quoted;
        quoted.SetSource(local);
        quoted.SetDestination(remote);
        quoted.SetProtocol(protocol);
        quoted.SetPayloadSize(20);
        quoted.SetTtl(64);
        const uint8_t ports[8] = {static_cast<uint8_t>(localPort >> 8),
                                  static_cast<uint8_t>(localPort & 0xff),
                                  static_cast<uint8_t>(remotePort >> 8),
                                  static_cast<uint8_t>(remotePort & 0xff),
                                  0,
                                  0,
                                  0,
                                  0};
        Icmpv4TimeExceeded body;
        body.SetHeader(quoted);
        body.SetData(Create<Packet>(ports, sizeof(ports)));
        Ptr<Packet> packet = Create<Packet>();
        packet->AddHeader(body);
        Icmpv4Header icmp;
        icmp.SetType(Icmpv4Header::ICMPV4_SOURCE_QUENCH);
        icmp.SetCode(0);
        packet->AddHeader(icmp);
        socket->SendTo(packet, 0, InetSocketAddress(local, 0));
    }

    /**
     * ICMP callback of the TCP socket.
     * @param source The ICMP source address.
     * @param ttl The ICMP TTL.
     * @param type The ICMP type.
     * @param code The ICMP code.
     * @param info The ICMP info.
     */
    void TcpIcmp(Ipv4Address source, uint8_t ttl, uint8_t type, uint8_t code, uint32_t info)
    {
        m_tcpIcmps++;
    }

    /**
     * ICMP callback of the UDP socket.
     * @param source The ICMP source address.
     * @param ttl The ICMP TTL.
     * @param type The ICMP type.
     * @param code The ICMP code.
     * @param info The ICMP info.
     */
    void UdpIcmp(Ipv4Address source, uint8_t ttl, uint8_t type, uint8_t code, uint32_t info)
    {
        if (type == Icmpv4Header::ICMPV4_SOURCE_QUENCH)
        {
            m_udpQuenches++;
        }
    }

    /**
     * State trace sink of the TCP socket.
     * @param oldState The previous state.
     * @param newState The new state.
     */
    void StateChanged(TcpSocket::TcpStates_t oldState, TcpSocket::TcpStates_t newState)
    {
        m_state = newState;
    }

    uint32_t m_tcpIcmps{0};                            //!< ICMP messages reaching the TCP socket
    uint32_t m_udpQuenches{0};                         //!< Source Quenches reaching the UDP socket
    TcpSocket::TcpStates_t m_state{TcpSocket::CLOSED}; //!< The state of the TCP socket
};

/**
 * @ingroup internet-test
 * @ingroup tests
 *
 * @brief Test that a SYN with a zero window does not make the listener probe
 *
 * A listening socket has no connection whose window could be probed: a SYN
 * advertising a zero window is answered with a SYN+ACK by the socket forked
 * for it, which retransmits it while the handshake is not completed, and
 * nothing else is sent (@RFC{9293}, Section 3.8.6.1); the listener keeps
 * answering the SYNs of other clients.
 */
class TcpListenerZeroWindowSynTestCase : public TcpRawSegmentTestCase
{
  public:
    TcpListenerZeroWindowSynTestCase()
        : TcpRawSegmentTestCase("A SYN with a zero window does not make the listener probe")
    {
    }

  private:
    void DoRun() override
    {
        Setup();
        Segment syn = Make(TcpHeader::SYN, m_proberIsn, 0);
        syn.window = 0;
        SendAt(Seconds(1), syn);
        // Another client opens its window right away, which a listener in
        // the persist state would take for the end of it
        Segment other = Make(TcpHeader::SYN, m_proberIsn, 0);
        other.sport = m_proberPort + 1;
        SendAt(Seconds(1.5), other);
        Simulator::Stop(Seconds(20));
        Simulator::Run();

        std::set<uint16_t> answered;
        for (const auto& header : m_headers)
        {
            answered.insert(header.GetDestinationPort());
        }
        NS_TEST_ASSERT_MSG_EQ(answered.size(), 2, "Not both SYNs were answered");
        NS_TEST_ASSERT_MSG_EQ(Count(TcpHeader::SYN | TcpHeader::ACK),
                              m_headers.size(),
                              "Something else than a SYN+ACK was sent");
    }
};

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
        AddTestCase(new TcpPassiveHighIsnTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpUrgentDataTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpLegacyOptionsTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpUnknownOptionTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpReservedFlagsTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpResetAckTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpAdvertisedMssTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpSimultaneousOpenTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpCloseNotificationTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpAbortNotificationTestCase(), TestCase::Duration::QUICK);
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
        AddTestCase(new TcpBroadcastLookalikeTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpIpv6MulticastSynTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpUnsignedWindowTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpChecksumTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpShrinkingWindowTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpMalformedOptionsAbortTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpSimultaneousOpenRefusedTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpEmptyZeroWindowProbeTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpUrgentHighIsnTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpKeepAliveTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpSourceRouteTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpSendAfterShutdownTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpUrgentReadPointTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpUrgentSpuriousNotificationTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpUrgentWrapTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpUrgentZeroWindowTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpZeroWindowProbeUrgentTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpLongUrgentDataTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpKeepAliveHalfClosedTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpKeepAliveTimeChangeTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpKeepAliveDelayedAckTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpKeepAliveScenariosTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpKeepAliveResetAnswerTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpKeepAliveReleaseOnAbortTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpKeepAliveTimerEventsTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpKeepAliveShortestTimesTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpSynWindowEdgeTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpReceiverSwsTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpScaledRightEdgeTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpWindowUpdateAfterReadTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpDelayedAckHeldWindowTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpPushScenariosTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpUnsignedWindowSenderTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpChecksumEstablishedTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpShrinkingWindowProbeTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpEolPaddingTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpDataOffsetTestCase(false), TestCase::Duration::QUICK);
        AddTestCase(new TcpDataOffsetTestCase(true), TestCase::Duration::QUICK);
        AddTestCase(new TcpResetAcceptabilityTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpResetWithoutTimestampTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpSynSentResetTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpOutOfWindowFinTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpClosingStatesAckTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpPureAckSequenceTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpOldSequenceAckTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpCloseFromErrorCallbackTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpUrgentSynSentTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpUrgentHandshakeAckTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpSynInSynchronizedStateTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpMalformedOutOfWindowTestCase(false), TestCase::Duration::QUICK);
        AddTestCase(new TcpMalformedOutOfWindowTestCase(true), TestCase::Duration::QUICK);
        AddTestCase(new TcpMalformedSynSentTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpMalformedOptionMatrixTestCase(false), TestCase::Duration::QUICK);
        AddTestCase(new TcpMalformedOptionMatrixTestCase(true), TestCase::Duration::QUICK);
        AddTestCase(new TcpMalformedV4MappedTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpSimultaneousOpenLossTestCase(false), TestCase::Duration::QUICK);
        AddTestCase(new TcpSimultaneousOpenLossTestCase(true), TestCase::Duration::QUICK);
        AddTestCase(new TcpSimultaneousOpenRetriesTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpSimultaneousOpenAsymmetricOptionsTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpOwnIsnWrapTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpIsnClockTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpSynRetransmissionIsnTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpEcnCeHighIsnTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpEcnEchoHighIsnTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpWindowUpdateHighIsnTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpOtherInterfaceBroadcastSynTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpInvalidRemoteAddressExtraTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpV4MappedDefaultMssTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpIcmpSourceQuenchEndToEndTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TcpListenerZeroWindowSynTestCase(), TestCase::Duration::QUICK);
    }
};

static TcpRfc9293TestSuite g_tcpRfc9293TestSuite; //!< Static variable for test initialization
