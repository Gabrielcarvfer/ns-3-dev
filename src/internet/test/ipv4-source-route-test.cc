/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ns3/boolean.h"
#include "ns3/global-value.h"
#include "ns3/iana-ieee802-numbers.h"
#include "ns3/iana-internet-protocol-numbers.h"
#include "ns3/icmpv4.h"
#include "ns3/inet-socket-address.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/ipv4-header.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/ipv4-raw-socket-factory.h"
#include "ns3/ipv4-source-route-tag.h"
#include "ns3/ipv6-address-helper.h"
#include "ns3/mac48-address.h"
#include "ns3/node-container.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/simulator.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/test.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/uinteger.h"

#include <vector>

/**
 * @file
 * @ingroup internet-test
 *
 * Tests of the IPv4 options, of the loose source route option in particular.
 */

using namespace ns3;

/**
 * @ingroup internet-test
 *
 * @brief Base of the IPv4 option tests, over a line of four nodes
 *
 * Node 0 hands crafted datagrams to the device of node 1 directly, and the
 * IP layer of every node is watched: the datagrams it drops, forwards and
 * delivers locally, and the ICMP messages node 0 receives.
 */
class Ipv4OptionsTestBase : public TestCase
{
  public:
    /**
     * Constructor.
     * @param name The name of the test.
     * @param checksum Whether the checksums are enabled.
     */
    Ipv4OptionsTestBase(const std::string& name, bool checksum)
        : TestCase(name + (checksum ? " (checksums enabled)" : " (checksums disabled)")),
          m_checksum(checksum)
    {
    }

  protected:
    /// What the IP layer of a node did
    struct Observed
    {
        std::vector<Ipv4L3Protocol::DropReason> drops; //!< Reasons of the drops
        std::vector<Ipv4Header> forwarded;             //!< Headers of the forwarded datagrams
        std::vector<Ipv4Header> delivered;             //!< Headers of the delivered datagrams
        std::vector<Ipv4Header> sent;                  //!< Headers of the datagrams sent
        std::vector<std::pair<uint8_t, uint8_t>> icmp; //!< Type and code of the ICMP received
    };

    /**
     * Build the line of nodes 0 - 1 - 2 - 3, on the subnets 10.0.1.0/24,
     * 10.0.2.0/24 and 10.0.3.0/24, with the global routing.
     */
    void BuildLine()
    {
        GlobalValue::GetValueByName("ChecksumEnabled", m_previousChecksum);
        GlobalValue::Bind("ChecksumEnabled", BooleanValue(m_checksum));

        m_nodes.Create(4);
        SimpleNetDeviceHelper devHelper;
        InternetStackHelper stack;
        stack.Install(m_nodes);
        Ipv4AddressHelper address;
        for (uint32_t link = 0; link < 3; ++link)
        {
            NetDeviceContainer devices =
                devHelper.Install(NodeContainer(m_nodes.Get(link), m_nodes.Get(link + 1)));
            m_devices.push_back(devices);
            std::string base = "10.0." + std::to_string(link + 1) + ".0";
            address.SetBase(base.c_str(), "255.255.255.0");
            m_links.push_back(address.Assign(devices));
        }
        Ipv4GlobalRoutingHelper::PopulateRoutingTables();

        m_observed.resize(m_nodes.GetN());
        for (uint32_t node = 0; node < m_nodes.GetN(); ++node)
        {
            Ptr<Ipv4L3Protocol> ipv4 = m_nodes.Get(node)->GetObject<Ipv4L3Protocol>();
            Observed* observed = &m_observed[node];
            ipv4->TraceConnectWithoutContext(
                "Drop",
                Callback<void,
                         const Ipv4Header&,
                         Ptr<const Packet>,
                         Ipv4L3Protocol::DropReason,
                         Ptr<Ipv4>,
                         uint32_t>([observed](const Ipv4Header&,
                                              Ptr<const Packet>,
                                              Ipv4L3Protocol::DropReason reason,
                                              Ptr<Ipv4>,
                                              uint32_t) { observed->drops.push_back(reason); }));
            ipv4->TraceConnectWithoutContext(
                "UnicastForward",
                Callback<void, const Ipv4Header&, Ptr<const Packet>, uint32_t>(
                    [observed](const Ipv4Header& header, Ptr<const Packet>, uint32_t) {
                        observed->forwarded.push_back(header);
                    }));
            ipv4->TraceConnectWithoutContext(
                "SendOutgoing",
                Callback<void, const Ipv4Header&, Ptr<const Packet>, uint32_t>(
                    [observed](const Ipv4Header& header, Ptr<const Packet>, uint32_t) {
                        observed->sent.push_back(header);
                    }));
            ipv4->TraceConnectWithoutContext(
                "LocalDeliver",
                Callback<void, const Ipv4Header&, Ptr<const Packet>, uint32_t>(
                    [observed](const Ipv4Header& header, Ptr<const Packet> packet, uint32_t) {
                        observed->delivered.push_back(header);
                        if (header.GetProtocol() == iana::internetprotocolnumbers::ICMP)
                        {
                            Icmpv4Header icmp;
                            packet->PeekHeader(icmp);
                            observed->icmp.emplace_back(icmp.GetType(), icmp.GetCode());
                        }
                    }));
        }
    }

    /**
     * Get the address of a node on a link of the line.
     * @param link The link, from 0 to 2.
     * @param side 0 for the node on the left of the link, 1 for the one on its right.
     * @return The address.
     */
    Ipv4Address Addr(uint32_t link, uint32_t side) const
    {
        return m_links[link].GetAddress(side);
    }

    /**
     * Build a header from node 0, holding a loose source route.
     * @param destination The destination of the header.
     * @param route The addresses of the option, none for no option.
     * @param pointer The pointer of the option.
     * @param ttl The TTL.
     * @return The header.
     */
    Ipv4Header MakeHeader(Ipv4Address destination,
                          const std::vector<Ipv4Address>& route = {},
                          uint8_t pointer = 4,
                          uint8_t ttl = 64) const
    {
        Ipv4Header header;
        header.SetSource(Addr(0, 0));
        header.SetDestination(destination);
        header.SetProtocol(253);
        header.SetTtl(ttl);
        header.SetPayloadSize(PAYLOAD);
        if (!route.empty())
        {
            header.SetLooseSourceRoute(route);
            header.SetSourceRoutePointer(pointer);
        }
        if (m_checksum)
        {
            header.EnableChecksum();
        }
        return header;
    }

    /**
     * Build a datagram from its header.
     * @param header The header.
     * @param payload The size of the payload.
     * @return The datagram.
     */
    static Ptr<Packet> MakeDatagram(const Ipv4Header& header, uint32_t payload = PAYLOAD)
    {
        Ptr<Packet> datagram = Create<Packet>(payload);
        datagram->AddHeader(header);
        return datagram;
    }

    /**
     * Build a datagram from raw header bytes.
     * @param bytes The bytes of the header, or of the whole datagram.
     * @return The datagram.
     */
    static Ptr<Packet> MakeDatagram(const std::vector<uint8_t>& bytes)
    {
        return Create<Packet>(bytes.data(), bytes.size());
    }

    /**
     * Build a datagram from node 0 holding the given option bytes.
     * @param destination The destination of the datagram.
     * @param options The option bytes, padded to a word by this function.
     * @return The datagram.
     */
    Ptr<Packet> MakeRawDatagram(Ipv4Address destination, std::vector<uint8_t> options) const
    {
        Ipv4Header header = MakeHeader(destination);
        Buffer buffer;
        buffer.AddAtStart(header.GetSerializedSize());
        header.Serialize(buffer.Begin());
        std::vector<uint8_t> bytes(buffer.GetSize());
        buffer.CopyData(bytes.data(), bytes.size());

        options.resize((options.size() + 3) & ~3, 0);
        bytes.insert(bytes.end(), options.begin(), options.end());
        uint16_t total = bytes.size() + PAYLOAD;
        bytes[0] = 0x40 | (bytes.size() / 4);
        bytes[2] = total >> 8;
        bytes[3] = total & 0xff;
        bytes[10] = 0;
        bytes[11] = 0;
        Buffer checksummed;
        checksummed.AddAtStart(bytes.size());
        checksummed.Begin().Write(bytes.data(), bytes.size());
        uint16_t checksum = checksummed.Begin().CalculateIpChecksum(bytes.size());
        Buffer::Iterator it = checksummed.Begin();
        it.Next(10);
        it.WriteU16(checksum);
        checksummed.CopyData(bytes.data(), bytes.size());
        bytes.resize(total, 0);
        return MakeDatagram(bytes);
    }

    /**
     * Get the bytes of an address, in network order.
     * @param address The address.
     * @return The bytes.
     */
    static std::vector<uint8_t> Bytes(Ipv4Address address)
    {
        uint8_t buf[4];
        address.Serialize(buf);
        return {buf, buf + 4};
    }

    /**
     * Hand a datagram from node 0 to the device of node 1.
     * @param datagram The datagram.
     */
    void Inject(Ptr<Packet> datagram)
    {
        Ptr<NetDevice> from = m_devices[0].Get(0);
        Ptr<NetDevice> to = m_devices[0].Get(1);
        from->Send(datagram, to->GetAddress(), iana::ieee802numbers::IPV4);
    }

    /**
     * Run the simulation until the datagrams settled.
     */
    void RunSimulation()
    {
        Simulator::Stop(Seconds(10));
        Simulator::Run();
    }

    void DoTeardown() override
    {
        Simulator::Destroy();
        if (!m_nodes.GetN())
        {
            return;
        }
        GlobalValue::Bind("ChecksumEnabled", m_previousChecksum);
    }

    static constexpr uint32_t PAYLOAD = 100; //!< Size of the payload of the datagrams

    bool m_checksum;                             //!< Whether the checksums are enabled
    BooleanValue m_previousChecksum;             //!< The global value before the test
    NodeContainer m_nodes;                       //!< The nodes of the line
    std::vector<NetDeviceContainer> m_devices;   //!< The devices of each link
    std::vector<Ipv4InterfaceContainer> m_links; //!< The interfaces of each link
    std::vector<Observed> m_observed;            //!< What the IP layer of each node did
};

/**
 * @ingroup internet-test
 *
 * @brief Test that a received header which does not hold together is dropped
 *
 * A datagram shorter than its header says, or whose header length is below
 * the 20 bytes of the fixed fields, must be dropped by the IP layer, without
 * the header being read beyond the bytes received.
 */
class Ipv4MalformedHeaderTest : public Ipv4OptionsTestBase
{
  public:
    /**
     * Constructor.
     * @param checksum Whether the checksums are enabled.
     */
    Ipv4MalformedHeaderTest(bool checksum)
        : Ipv4OptionsTestBase("A malformed IPv4 header is dropped", checksum)
    {
    }

  private:
    void DoRun() override
    {
        BuildLine();

        Ipv4Header header = MakeHeader(Addr(0, 1));
        Buffer buffer;
        buffer.AddAtStart(header.GetSerializedSize());
        header.Serialize(buffer.Begin());
        std::vector<uint8_t> bytes(buffer.GetSize());
        buffer.CopyData(bytes.data(), bytes.size());

        std::vector<std::vector<uint8_t>> datagrams;
        // A 20 byte datagram claiming a longer header
        for (uint8_t ihl = 6; ihl <= 15; ++ihl)
        {
            std::vector<uint8_t> claim = bytes;
            claim[0] = 0x40 | ihl;
            claim[2] = 0;
            claim[3] = 20;
            datagrams.push_back(claim);
        }
        // A header length below the fixed fields
        std::vector<uint8_t> shortIhl = bytes;
        shortIhl[0] = 0x44;
        shortIhl.resize(40, 0);
        datagrams.push_back(shortIhl);
        // A total length below the header length
        std::vector<uint8_t> shortTotal = bytes;
        shortTotal[2] = 0;
        shortTotal[3] = 16;
        datagrams.push_back(shortTotal);
        // A total length beyond the bytes received
        std::vector<uint8_t> longTotal = bytes;
        longTotal[2] = 0;
        longTotal[3] = 200;
        longTotal.resize(60, 0);
        datagrams.push_back(longTotal);
        // Fewer bytes than the fixed fields
        std::vector<uint8_t> tiny(bytes.begin(), bytes.begin() + 12);
        datagrams.push_back(tiny);

        for (const auto& datagram : datagrams)
        {
            Inject(MakeDatagram(datagram));
        }
        RunSimulation();

        NS_TEST_ASSERT_MSG_EQ(m_observed[1].delivered.size(),
                              0,
                              "A malformed header was delivered");
        NS_TEST_ASSERT_MSG_EQ(m_observed[1].drops.size(),
                              datagrams.size(),
                              "A malformed header was not dropped");
        for (auto reason : m_observed[1].drops)
        {
            NS_TEST_ASSERT_MSG_EQ(reason,
                                  Ipv4L3Protocol::DROP_BAD_CHECKSUM,
                                  "Unexpected drop reason");
        }
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Test that a malformed source route option is discarded
 *
 * A loose source route option whose pointer is below 4 or not on an address
 * boundary, a second loose source route and a strict source route make the
 * datagram be discarded (@RFC{791}, @RFC{1812} Section 5.2.4.1), whereas a
 * pointer past the option means that the route is exhausted.
 */
class Ipv4MalformedSourceRouteTest : public Ipv4OptionsTestBase
{
  public:
    /**
     * Constructor.
     * @param checksum Whether the checksums are enabled.
     */
    Ipv4MalformedSourceRouteTest(bool checksum)
        : Ipv4OptionsTestBase("A malformed source route is discarded", checksum)
    {
    }

  private:
    void DoRun() override
    {
        BuildLine();
        std::vector<uint8_t> next = Bytes(Addr(2, 1));
        std::vector<uint8_t> other = Bytes(Addr(1, 1));
        auto lsrr = [](uint8_t pointer, const std::vector<uint8_t>& address) {
            std::vector<uint8_t> option = {131, 7, pointer};
            option.insert(option.end(), address.begin(), address.end());
            return option;
        };
        auto concat = [](std::vector<uint8_t> first, const std::vector<uint8_t>& second) {
            first.insert(first.end(), second.begin(), second.end());
            return first;
        };
        std::vector<uint8_t> ssrr = lsrr(4, other);
        ssrr[0] = 137;

        const std::vector<std::vector<uint8_t>> malformed = {
            lsrr(3, next),
            lsrr(5, next),
            lsrr(7, next),
            concat(lsrr(4, next), lsrr(4, other)),
            concat(lsrr(4, next), ssrr),
            ssrr,
            {131, 5, 4, 10, 0},
        };
        for (const auto& options : malformed)
        {
            Inject(MakeRawDatagram(Addr(0, 1), options));
        }
        // A pointer past the option: the route is exhausted, and the datagram
        // has reached its final destination
        Simulator::Schedule(Seconds(1), [this, lsrr, next]() {
            Inject(MakeRawDatagram(Addr(0, 1), lsrr(12, next)));
        });
        RunSimulation();

        NS_TEST_ASSERT_MSG_EQ(m_observed[1].drops.size(),
                              malformed.size(),
                              "A malformed source route was not discarded");
        NS_TEST_ASSERT_MSG_EQ(m_observed[1].forwarded.size(),
                              0,
                              "A malformed source route was followed");
        NS_TEST_ASSERT_MSG_EQ(m_observed[1].delivered.size(),
                              1,
                              "The datagram with an exhausted route was not delivered");
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Test that a hop passes over its own addresses in the route
 *
 * When the next address of the route is one of the addresses of the node
 * the datagram reached, the node passes over it rather than sending the
 * datagram to itself again (@RFC{1812}, Section 5.2.3), and a route which
 * this leaves exhausted ends at the node, which delivers the datagram.
 */
class Ipv4SourceRouteOwnAddressTest : public Ipv4OptionsTestBase
{
  public:
    /**
     * Constructor.
     * @param checksum Whether the checksums are enabled.
     */
    Ipv4SourceRouteOwnAddressTest(bool checksum)
        : Ipv4OptionsTestBase("A hop passes over its own addresses in the route", checksum)
    {
    }

  private:
    void DoRun() override
    {
        BuildLine();
        Ptr<Ipv4L3Protocol> hop = m_nodes.Get(1)->GetObject<Ipv4L3Protocol>();
        hop->SetAttribute("SourceRouteForwarding", BooleanValue(true));

        // The address of the hop on its other link, then the end of the line
        Inject(MakeDatagram(MakeHeader(Addr(0, 1), {Addr(1, 0), Addr(2, 1)})));

        // A route whose remaining addresses are all the hop's own ends there,
        // even at a node which follows no source route
        Simulator::Schedule(Seconds(1), [this, hop]() {
            hop->SetAttribute("SourceRouteForwarding", BooleanValue(false));
            hop->SetAttribute("IpForward", BooleanValue(false));
            Inject(MakeDatagram(MakeHeader(Addr(0, 1), {Addr(1, 0)})));
        });
        RunSimulation();

        NS_TEST_ASSERT_MSG_EQ(m_observed[1].forwarded.size(), 1, "The route was not followed");
        if (!m_observed[1].forwarded.empty())
        {
            const Ipv4Header& forwarded = m_observed[1].forwarded[0];
            NS_TEST_ASSERT_MSG_EQ(forwarded.GetDestination(),
                                  Addr(2, 1),
                                  "The hop sent the datagram to itself");
            NS_TEST_ASSERT_MSG_EQ(+forwarded.GetSourceRoutePointer(),
                                  12,
                                  "The pointer was not moved past both addresses");
        }
        NS_TEST_ASSERT_MSG_EQ(m_observed[3].delivered.size(),
                              1,
                              "The datagram did not reach the end of its route");
        NS_TEST_ASSERT_MSG_EQ(m_observed[1].drops.size(), 0, "A datagram was dropped");
        NS_TEST_ASSERT_MSG_EQ(m_observed[1].delivered.size(),
                              1,
                              "The datagram whose route ended at the hop was not delivered");
        if (!m_observed[1].delivered.empty())
        {
            NS_TEST_ASSERT_MSG_EQ(m_observed[1].delivered[0].GetDestination(),
                                  Addr(1, 0),
                                  "The datagram was not addressed to the end of its route");
        }
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Test the switch of the source route forwarding
 *
 * A node acts as a hop of a source route only when configured to, whether
 * or not it forwards the other datagrams (@RFC{1122}, Section 3.3.5); it
 * otherwise discards the datagram and returns an ICMP Destination
 * Unreachable, Source Route Failed, unless the datagram is itself an ICMP
 * error message.
 */
class Ipv4SourceRouteForwardingSwitchTest : public Ipv4OptionsTestBase
{
  public:
    /**
     * Constructor.
     * @param checksum Whether the checksums are enabled.
     */
    Ipv4SourceRouteForwardingSwitchTest(bool checksum)
        : Ipv4OptionsTestBase("A node follows a source route only when configured to", checksum)
    {
    }

  private:
    void DoRun() override
    {
        BuildLine();
        Ptr<Ipv4L3Protocol> hop = m_nodes.Get(1)->GetObject<Ipv4L3Protocol>();
        Ipv4Header header = MakeHeader(Addr(0, 1), {Addr(2, 1)});

        // The default: the router does not follow the route
        Inject(MakeDatagram(header));

        // Nor does it answer an ICMP error message with another
        Simulator::Schedule(Seconds(1), [this, header]() {
            Ipv4Header icmpHeader = header;
            icmpHeader.SetProtocol(iana::internetprotocolnumbers::ICMP);
            Ptr<Packet> error = Create<Packet>(PAYLOAD - 4);
            Icmpv4Header icmp;
            icmp.SetType(Icmpv4Header::ICMPV4_DEST_UNREACH);
            icmp.SetCode(Icmpv4DestinationUnreachable::ICMPV4_PORT_UNREACHABLE);
            error->AddHeader(icmp);
            error->AddHeader(icmpHeader);
            Inject(error);
        });

        // Once enabled, a host which forwards nothing else follows it
        Simulator::Schedule(Seconds(2), [this, hop, header]() {
            hop->SetAttribute("SourceRouteForwarding", BooleanValue(true));
            hop->SetAttribute("IpForward", BooleanValue(false));
            Inject(MakeDatagram(header));
        });
        RunSimulation();

        NS_TEST_ASSERT_MSG_EQ(m_observed[1].drops.size(), 2, "The source routes were followed");
        NS_TEST_ASSERT_MSG_EQ(m_observed[0].icmp.size(), 1, "Unexpected ICMP messages");
        if (!m_observed[0].icmp.empty())
        {
            NS_TEST_ASSERT_MSG_EQ(+m_observed[0].icmp[0].first,
                                  +Icmpv4Header::ICMPV4_DEST_UNREACH,
                                  "Unexpected ICMP type");
            NS_TEST_ASSERT_MSG_EQ(+m_observed[0].icmp[0].second,
                                  +Icmpv4DestinationUnreachable::ICMPV4_SOURCE_ROUTE_FAILED,
                                  "Unexpected ICMP code");
        }
        NS_TEST_ASSERT_MSG_EQ(m_observed[1].forwarded.size(), 1, "The route was not followed");
        NS_TEST_ASSERT_MSG_EQ(m_observed[3].delivered.size(),
                              1,
                              "The datagram did not reach the end of its route");
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Test the limits of the relaying of a source routed datagram
 *
 * A datagram which reached a hop as a broadcast is never relayed, nor is one
 * sent towards a next hop which is a broadcast address, which a hop answers
 * like one it has no route to, with an ICMP Destination Unreachable, Source
 * Route Failed (@RFC{1812}, Section 4.3.3.1). A datagram whose TTL runs out
 * at a hop is answered with an ICMP Time Exceeded. A datagram sent to a
 * broadcast address does not follow the route its application asked for.
 */
class Ipv4SourceRouteRelayLimitsTest : public Ipv4OptionsTestBase
{
  public:
    /**
     * Constructor.
     * @param checksum Whether the checksums are enabled.
     */
    Ipv4SourceRouteRelayLimitsTest(bool checksum)
        : Ipv4OptionsTestBase("The relaying of a source routed datagram is bounded", checksum)
    {
    }

  private:
    void DoRun() override
    {
        BuildLine();
        m_nodes.Get(1)->GetObject<Ipv4L3Protocol>()->SetAttribute("SourceRouteForwarding",
                                                                  BooleanValue(true));

        // Received as a broadcast, of the subnet or of the link
        Inject(MakeDatagram(MakeHeader(Ipv4Address("10.0.1.255"), {Addr(2, 1)})));
        Inject(MakeDatagram(MakeHeader(Ipv4Address::GetBroadcast(), {Addr(2, 1)})));
        // Sent on to a broadcast address
        Simulator::Schedule(Seconds(1), [this]() {
            Inject(MakeDatagram(MakeHeader(Addr(0, 1), {Ipv4Address("10.0.2.255")})));
        });
        // Sent on with a TTL run out
        Simulator::Schedule(Seconds(2), [this]() {
            Inject(MakeDatagram(MakeHeader(Addr(0, 1), {Addr(2, 1)}, 4, 1)));
        });
        // Sent on to an address without a route
        Simulator::Schedule(Seconds(3), [this]() {
            Inject(MakeDatagram(MakeHeader(Addr(0, 1), {Ipv4Address("10.9.9.9")})));
        });
        // A broadcast sent by an application which asked for a route
        Simulator::Schedule(Seconds(4), [this]() {
            Ptr<Socket> socket =
                Socket::CreateSocket(m_nodes.Get(0), UdpSocketFactory::GetTypeId());
            socket->SetAllowBroadcast(true);
            socket->Bind();
            Ptr<Packet> packet = Create<Packet>(PAYLOAD);
            Ipv4SourceRouteTag route;
            route.SetRoute({Addr(0, 1), Addr(2, 1)});
            packet->AddPacketTag(route);
            socket->SendTo(packet, 0, InetSocketAddress(Ipv4Address::GetBroadcast(), 9));
        });
        RunSimulation();

        NS_TEST_ASSERT_MSG_EQ(m_observed[1].forwarded.size(), 0, "A datagram was relayed");
        const std::vector<std::pair<uint8_t, uint8_t>> expected = {
            {Icmpv4Header::ICMPV4_DEST_UNREACH,
             Icmpv4DestinationUnreachable::ICMPV4_SOURCE_ROUTE_FAILED},
            {Icmpv4Header::ICMPV4_TIME_EXCEEDED, Icmpv4TimeExceeded::ICMPV4_TIME_TO_LIVE},
            {Icmpv4Header::ICMPV4_DEST_UNREACH,
             Icmpv4DestinationUnreachable::ICMPV4_SOURCE_ROUTE_FAILED},
        };
        NS_TEST_ASSERT_MSG_EQ((m_observed[0].icmp == expected),
                              true,
                              "Unexpected ICMP messages received: " << m_observed[0].icmp.size());
        NS_TEST_ASSERT_MSG_EQ(m_observed[0].sent.empty(), false, "The broadcast was not sent");
        for (const auto& header : m_observed[0].sent)
        {
            NS_TEST_ASSERT_MSG_EQ(header.HasLooseSourceRoute(),
                                  false,
                                  "A broadcast was sent along a route");
            NS_TEST_ASSERT_MSG_EQ(header.GetDestination(),
                                  Ipv4Address::GetBroadcast(),
                                  "A broadcast was sent to a unicast address");
        }
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Test the number of hops of a source route
 *
 * The option holds 9 addresses beside the first hop, which travels in the
 * destination field: a route of 10 hops ending at the destination fills the
 * 60 bytes of the largest header and is followed, whereas a route which
 * does not fit is dropped by the IP layer rather than aborting the
 * simulation, and refused by a TCP socket whenever it can tell.
 */
class Ipv4SourceRouteHopLimitTest : public Ipv4OptionsTestBase
{
  public:
    /**
     * Constructor.
     * @param checksum Whether the checksums are enabled.
     */
    Ipv4SourceRouteHopLimitTest(bool checksum)
        : Ipv4OptionsTestBase("A source route holds at most 10 hops", checksum)
    {
    }

  private:
    /**
     * Send a UDP datagram from node 0 to node 3 along a route.
     * @param route The route.
     */
    void SendAlong(const std::vector<Ipv4Address>& route)
    {
        Ptr<Socket> socket = Socket::CreateSocket(m_nodes.Get(0), UdpSocketFactory::GetTypeId());
        socket->Bind();
        Ptr<Packet> packet = Create<Packet>(PAYLOAD);
        Ipv4SourceRouteTag tag;
        tag.SetRoute(route);
        packet->AddPacketTag(tag);
        socket->SendTo(packet, 0, InetSocketAddress(Addr(2, 1), 9));
    }

    void DoRun() override
    {
        BuildLine();
        for (uint32_t node : {1, 2})
        {
            m_nodes.Get(node)->GetObject<Ipv4L3Protocol>()->SetAttribute("SourceRouteForwarding",
                                                                         BooleanValue(true));
        }

        // Back and forth between the two nodes in the middle, then on to the
        // end of the line
        std::vector<Ipv4Address> longest = {Addr(0, 1)};
        while (longest.size() < Ipv4SourceRouteTag::MAX_HOPS - 1)
        {
            longest.push_back(longest.size() % 2 ? Addr(1, 1) : Addr(1, 0));
        }
        std::vector<Ipv4Address> tooLong = longest;
        tooLong.push_back(Addr(1, 1));
        longest.push_back(Addr(2, 1));

        SendAlong(longest);
        // The destination appended to a route of 10 hops makes 11
        Simulator::Schedule(Seconds(1), [this, tooLong]() { SendAlong(tooLong); });

        // A TCP socket refuses a route which cannot fit, or holding a hop
        // which is not a unicast address
        Ptr<TcpSocketBase> unbound = DynamicCast<TcpSocketBase>(
            Socket::CreateSocket(m_nodes.Get(0), TcpSocketFactory::GetTypeId()));
        std::vector<Ipv4Address> eleven = tooLong;
        eleven.push_back(Addr(2, 1));
        NS_TEST_ASSERT_MSG_EQ(unbound->SetIpv4SourceRoute(eleven), false, "11 hops accepted");
        NS_TEST_ASSERT_MSG_EQ(unbound->SetIpv4SourceRoute({Addr(0, 1), Ipv4Address("224.0.0.1")}),
                              false,
                              "A route through a multicast address was accepted");
        NS_TEST_ASSERT_MSG_EQ(unbound->SetIpv4SourceRoute({Ipv4Address::GetBroadcast()}),
                              false,
                              "A route through a broadcast address was accepted");

        Ptr<Socket> server = Socket::CreateSocket(m_nodes.Get(3), TcpSocketFactory::GetTypeId());
        server->Bind(InetSocketAddress(Ipv4Address::GetAny(), 80));
        server->Listen();
        Ptr<TcpSocketBase> connected = DynamicCast<TcpSocketBase>(
            Socket::CreateSocket(m_nodes.Get(0), TcpSocketFactory::GetTypeId()));
        connected->Bind();
        connected->Connect(InetSocketAddress(Addr(2, 1), 80));
        NS_TEST_ASSERT_MSG_EQ(connected->SetIpv4SourceRoute(tooLong),
                              false,
                              "A route of 10 hops not ending at the peer was accepted");
        NS_TEST_ASSERT_MSG_EQ(connected->SetIpv4SourceRoute(longest),
                              true,
                              "A route of 10 hops ending at the peer was refused");

        // Before the peer is known, such a route cannot be told apart: the
        // IP layer drops its segments instead
        Ptr<TcpSocketBase> early = DynamicCast<TcpSocketBase>(
            Socket::CreateSocket(m_nodes.Get(0), TcpSocketFactory::GetTypeId()));
        early->Bind();
        NS_TEST_ASSERT_MSG_EQ(early->SetIpv4SourceRoute(tooLong),
                              true,
                              "A route of 10 hops was refused before the peer is known");
        Simulator::Schedule(Seconds(2),
                            [early, this]() { early->Connect(InetSocketAddress(Addr(2, 1), 80)); });
        Simulator::Stop(Seconds(4));
        Simulator::Run();

        NS_TEST_ASSERT_MSG_GT_OR_EQ(m_observed[0].drops.size(), 2, "A route too long was sent");
        for (auto reason : m_observed[0].drops)
        {
            NS_TEST_ASSERT_MSG_EQ(reason, Ipv4L3Protocol::DROP_ROUTE_ERROR, "Unexpected drop");
        }
        bool reached = false;
        for (const auto& header : m_observed[3].delivered)
        {
            if (header.GetProtocol() == iana::internetprotocolnumbers::UDP)
            {
                reached = true;
                NS_TEST_ASSERT_MSG_EQ(header.GetSerializedSize(), 60, "The option was not full");
                NS_TEST_ASSERT_MSG_EQ(header.GetLooseSourceRoute().size(),
                                      Ipv4SourceRouteTag::MAX_HOPS - 1,
                                      "The option was not full");
            }
        }
        NS_TEST_ASSERT_MSG_EQ(reached, true, "The longest route was not followed");
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Test that a relayed datagram does not take the way back along
 *
 * The UDP and raw sockets deliver a source routed datagram without the way
 * back to its sender: an application which sends it on to a third party
 * sends it along no route.
 */
class Ipv4SourceRouteUdpRelayTest : public Ipv4OptionsTestBase
{
  public:
    /**
     * Constructor.
     * @param checksum Whether the checksums are enabled.
     */
    Ipv4SourceRouteUdpRelayTest(bool checksum)
        : Ipv4OptionsTestBase("A relayed datagram does not follow the way back", checksum)
    {
    }

  private:
    /**
     * Send the datagrams received by the UDP socket on to node 2.
     * @param socket The UDP socket.
     */
    void RelayUdp(Ptr<Socket> socket)
    {
        while (Ptr<Packet> packet = socket->Recv())
        {
            m_relay->SendTo(packet, 0, InetSocketAddress(Addr(1, 1), 10));
        }
    }

    /**
     * Send the datagrams received by the raw socket on to node 2.
     * @param socket The raw socket.
     */
    void RelayRaw(Ptr<Socket> socket)
    {
        while (Ptr<Packet> packet = socket->Recv())
        {
            Ipv4Header header;
            packet->RemoveHeader(header);
            socket->SendTo(packet, 0, InetSocketAddress(Addr(1, 1), 0));
        }
    }

    void DoRun() override
    {
        BuildLine();
        m_nodes.Get(1)->GetObject<Ipv4L3Protocol>()->SetAttribute("SourceRouteForwarding",
                                                                  BooleanValue(true));

        Ptr<Socket> receiver = Socket::CreateSocket(m_nodes.Get(3), UdpSocketFactory::GetTypeId());
        receiver->Bind(InetSocketAddress(Ipv4Address::GetAny(), 9));
        receiver->SetRecvCallback(MakeCallback(&Ipv4SourceRouteUdpRelayTest::RelayUdp, this));
        m_relay = Socket::CreateSocket(m_nodes.Get(3), UdpSocketFactory::GetTypeId());
        m_relay->Bind();
        Ptr<Socket> raw = Socket::CreateSocket(m_nodes.Get(3), Ipv4RawSocketFactory::GetTypeId());
        raw->SetAttribute("Protocol", UintegerValue(iana::internetprotocolnumbers::UDP));
        raw->Bind();
        raw->SetRecvCallback(MakeCallback(&Ipv4SourceRouteUdpRelayTest::RelayRaw, this));

        Ptr<Socket> sender = Socket::CreateSocket(m_nodes.Get(0), UdpSocketFactory::GetTypeId());
        sender->Bind();
        Ptr<Packet> packet = Create<Packet>(PAYLOAD);
        Ipv4SourceRouteTag route;
        route.SetRoute({Addr(0, 1)});
        packet->AddPacketTag(route);
        sender->SendTo(packet, 0, InetSocketAddress(Addr(2, 1), 9));
        RunSimulation();

        NS_TEST_ASSERT_MSG_EQ(m_observed[1].forwarded.size(), 1, "The route was not followed");
        NS_TEST_ASSERT_MSG_EQ(m_observed[3].sent.size(), 2, "The datagram was not relayed");
        for (const auto& header : m_observed[3].sent)
        {
            NS_TEST_ASSERT_MSG_EQ(header.HasLooseSourceRoute(),
                                  false,
                                  "The relayed datagram took the way back along");
            NS_TEST_ASSERT_MSG_EQ(header.GetDestination(),
                                  Addr(1, 1),
                                  "The relayed datagram was sent along the way back");
        }
        m_relay = nullptr;
    }

    Ptr<Socket> m_relay; //!< The socket sending the datagrams on
};

/**
 * @ingroup internet-test
 *
 * @brief Test that an ICMP Echo Reply follows the way back
 *
 * The reply to an echo request which followed a source route must follow the
 * reversed route (@RFC{1122}, Section 3.2.2.6).
 */
class Ipv4SourceRouteEchoTest : public Ipv4OptionsTestBase
{
  public:
    /**
     * Constructor.
     * @param checksum Whether the checksums are enabled.
     */
    Ipv4SourceRouteEchoTest(bool checksum)
        : Ipv4OptionsTestBase("An ICMP Echo Reply follows the way back", checksum)
    {
    }

  private:
    void DoRun() override
    {
        BuildLine();
        m_nodes.Get(1)->GetObject<Ipv4L3Protocol>()->SetAttribute("SourceRouteForwarding",
                                                                  BooleanValue(true));

        Ptr<Packet> request = Create<Packet>(PAYLOAD - 12);
        Icmpv4Echo echo;
        echo.SetIdentifier(1);
        echo.SetSequenceNumber(1);
        request->AddHeader(echo);
        Icmpv4Header icmp;
        icmp.SetType(Icmpv4Header::ICMPV4_ECHO);
        if (m_checksum)
        {
            icmp.EnableChecksum();
        }
        request->AddHeader(icmp);
        Ipv4Header header = MakeHeader(Addr(0, 1), {Addr(2, 1)});
        header.SetProtocol(iana::internetprotocolnumbers::ICMP);
        header.SetPayloadSize(request->GetSize());
        request->AddHeader(header);
        Inject(request);
        RunSimulation();

        std::vector<Ipv4Header> replies;
        for (const auto& sent : m_observed[3].sent)
        {
            if (sent.GetProtocol() == iana::internetprotocolnumbers::ICMP)
            {
                replies.push_back(sent);
            }
        }
        NS_TEST_ASSERT_MSG_EQ(replies.size(), 1, "The request was not answered");
        if (!replies.empty())
        {
            NS_TEST_ASSERT_MSG_EQ(replies[0].HasLooseSourceRoute(),
                                  true,
                                  "The reply did not follow the way back");
            NS_TEST_ASSERT_MSG_EQ(replies[0].GetDestination(),
                                  Addr(1, 0),
                                  "The reply did not go through the hop the request did");
        }
        NS_TEST_ASSERT_MSG_EQ(m_observed[1].forwarded.size(),
                              2,
                              "The request and the reply did not both go through the hop");
        NS_TEST_ASSERT_MSG_EQ(
            (m_observed[0].icmp ==
             std::vector<std::pair<uint8_t, uint8_t>>{{Icmpv4Header::ICMPV4_ECHO_REPLY, 0}}),
            true,
            "The reply did not come back");
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Test that a RST follows the way back of the segment it answers
 *
 * The RST sent for a segment which reached no socket, or a socket in the
 * CLOSED state, follows the way back the segment recorded (@RFC{9293},
 * Section 3.9.2.1, MUST-53).
 */
class Ipv4SourceRouteRstTest : public Ipv4OptionsTestBase
{
  public:
    /**
     * Constructor.
     * @param checksum Whether the checksums are enabled.
     */
    Ipv4SourceRouteRstTest(bool checksum)
        : Ipv4OptionsTestBase("A RST follows the way back", checksum)
    {
    }

  private:
    void DoRun() override
    {
        BuildLine();
        m_nodes.Get(1)->GetObject<Ipv4L3Protocol>()->SetAttribute("SourceRouteForwarding",
                                                                  BooleanValue(true));

        // A socket in the CLOSED state on one port, none on the other
        Ptr<Socket> closed = Socket::CreateSocket(m_nodes.Get(3), TcpSocketFactory::GetTypeId());
        closed->Bind(InetSocketAddress(Ipv4Address::GetAny(), 81));

        std::vector<Ptr<Socket>> clients;
        for (uint16_t port : {81, 82})
        {
            Ptr<TcpSocketBase> client = DynamicCast<TcpSocketBase>(
                Socket::CreateSocket(m_nodes.Get(0), TcpSocketFactory::GetTypeId()));
            client->Bind();
            client->SetIpv4SourceRoute({Addr(0, 1)});
            client->Connect(InetSocketAddress(Addr(2, 1), port));
            clients.emplace_back(client);
        }
        RunSimulation();

        uint32_t resets = 0;
        for (const auto& header : m_observed[3].sent)
        {
            if (header.GetProtocol() != iana::internetprotocolnumbers::TCP)
            {
                continue;
            }
            ++resets;
            NS_TEST_ASSERT_MSG_EQ(header.HasLooseSourceRoute(),
                                  true,
                                  "The RST did not follow the way back");
            NS_TEST_ASSERT_MSG_EQ(header.GetDestination(),
                                  Addr(1, 0),
                                  "The RST did not go through the hop the segment did");
        }
        // The RST of the CLOSED socket carries no ACK, so the SYN it answers
        // may be sent again
        NS_TEST_ASSERT_MSG_GT_OR_EQ(resets, 2, "A segment was not answered with a RST");
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Test that an IPv4 source route leaves an IPv6 connection alone
 *
 * The route set on a TCP socket which connects over IPv6 neither reduces its
 * segment size nor prevents the data from flowing.
 */
class Ipv4SourceRouteIpv6SocketTest : public Ipv4OptionsTestBase
{
  public:
    /**
     * Constructor.
     * @param checksum Whether the checksums are enabled.
     */
    Ipv4SourceRouteIpv6SocketTest(bool checksum)
        : Ipv4OptionsTestBase("An IPv4 source route leaves an IPv6 connection alone", checksum)
    {
    }

  private:
    void DoRun() override
    {
        BuildLine();
        Ipv6AddressHelper address;
        address.SetBase(Ipv6Address("2001:1::"), Ipv6Prefix(64));
        Ipv6InterfaceContainer interfaces = address.Assign(m_devices[0]);
        interfaces.SetForwarding(1, true);

        Ptr<Socket> server = Socket::CreateSocket(m_nodes.Get(1), TcpSocketFactory::GetTypeId());
        server->Bind(Inet6SocketAddress(Ipv6Address::GetAny(), 80));
        server->Listen();
        uint32_t received = 0;
        server->SetAcceptCallback(
            MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
            Callback<void, Ptr<Socket>, const Address&>(
                [&received](Ptr<Socket> socket, const Address&) {
                    socket->SetRecvCallback(Callback<void, Ptr<Socket>>([&received](Ptr<Socket> s) {
                        while (Ptr<Packet> packet = s->Recv())
                        {
                            received += packet->GetSize();
                        }
                    }));
                }));

        std::vector<Ptr<TcpSocketBase>> clients;
        for (bool routed : {false, true})
        {
            Ptr<TcpSocketBase> client = DynamicCast<TcpSocketBase>(
                Socket::CreateSocket(m_nodes.Get(0), TcpSocketFactory::GetTypeId()));
            client->Bind6();
            if (routed)
            {
                NS_TEST_ASSERT_MSG_EQ(client->SetIpv4SourceRoute({Addr(0, 1)}),
                                      true,
                                      "The route was refused");
            }
            client->Connect(Inet6SocketAddress(interfaces.GetAddress(1, 1), 80));
            Simulator::Schedule(Seconds(1), [client]() { client->Send(Create<Packet>(5000), 0); });
            clients.push_back(client);
        }
        RunSimulation();

        NS_TEST_ASSERT_MSG_EQ(received, 10000, "The data did not flow");
        UintegerValue plain;
        UintegerValue routed;
        clients[0]->GetAttribute("SegmentSize", plain);
        clients[1]->GetAttribute("SegmentSize", routed);
        NS_TEST_ASSERT_MSG_EQ(routed.Get(),
                              plain.Get(),
                              "The IPv4 source route reduced the segment size");
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Base of the tests of the way back of a TCP connection
 */
class Ipv4SourceRouteTcpTestBase : public Ipv4OptionsTestBase
{
  public:
    /**
     * Constructor.
     * @param name The name of the test.
     * @param checksum Whether the checksums are enabled.
     */
    Ipv4SourceRouteTcpTestBase(const std::string& name, bool checksum)
        : Ipv4OptionsTestBase(name, checksum)
    {
    }

  protected:
    /**
     * Inject a TCP segment from a source on the link of node 0 to node 3,
     * through node 1.
     * @param source The source of the segment.
     * @param tcp The TCP header of the segment.
     * @param payload The size of the data of the segment.
     */
    void InjectSegment(Ipv4Address source, TcpHeader tcp, uint32_t payload = 0)
    {
        Ptr<Packet> segment = Create<Packet>(payload);
        if (m_checksum)
        {
            tcp.EnableChecksums();
            tcp.InitializeChecksum(source, Addr(2, 1), iana::internetprotocolnumbers::TCP);
        }
        segment->AddHeader(tcp);
        Ipv4Header header = MakeHeader(Addr(0, 1), {Addr(2, 1)});
        header.SetSource(source);
        header.SetProtocol(iana::internetprotocolnumbers::TCP);
        header.SetPayloadSize(segment->GetSize());
        segment->AddHeader(header);
        Inject(segment);
    }

    /**
     * Enable the source route forwarding on the nodes in the middle.
     */
    void EnableHops()
    {
        for (uint32_t node : {1, 2})
        {
            m_nodes.Get(node)->GetObject<Ipv4L3Protocol>()->SetAttribute("SourceRouteForwarding",
                                                                         BooleanValue(true));
        }
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Test which segments update the way back of a connection
 *
 * A later segment recording a different route overrides the way back the
 * connection saved (@RFC{1122}, Section 4.2.3.8), but only if it passes the
 * acceptability checks: a segment out of the window, a SYN, an ACK of data
 * never sent or older than the last one, or a segment without an ACK on a
 * synchronized connection leave it alone.
 */
class Ipv4SourceRouteTcpUpdateTest : public Ipv4SourceRouteTcpTestBase
{
  public:
    /**
     * Constructor.
     * @param checksum Whether the checksums are enabled.
     */
    Ipv4SourceRouteTcpUpdateTest(bool checksum)
        : Ipv4SourceRouteTcpTestBase("Only an acceptable segment updates the way back", checksum)
    {
    }

  private:
    void DoRun() override
    {
        BuildLine();
        EnableHops();

        Ptr<TcpSocketBase> accepted;
        Ptr<Socket> server = Socket::CreateSocket(m_nodes.Get(3), TcpSocketFactory::GetTypeId());
        // Without timestamps, which would have the injected segments discarded
        // for lack of them
        server->SetAttribute("Timestamp", BooleanValue(false));
        server->Bind(InetSocketAddress(Ipv4Address::GetAny(), 80));
        server->Listen();
        server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                  Callback<void, Ptr<Socket>, const Address&>(
                                      [&accepted](Ptr<Socket> socket, const Address&) {
                                          accepted = DynamicCast<TcpSocketBase>(socket);
                                      }));

        Ptr<TcpSocketBase> client = DynamicCast<TcpSocketBase>(
            Socket::CreateSocket(m_nodes.Get(0), TcpSocketFactory::GetTypeId()));
        client->SetAttribute("Timestamp", BooleanValue(false));
        client->Bind();
        client->SetIpv4SourceRoute({Addr(0, 1)});
        client->Connect(InetSocketAddress(Addr(2, 1), 80));
        Simulator::Schedule(Seconds(1), [client]() { client->Send(Create<Packet>(1000), 0); });

        // The client switches to a route through the other node in the middle
        std::vector<Ipv4Address> first;
        Simulator::Schedule(Seconds(2), [this, client, &accepted, &first]() {
            if (accepted)
            {
                first = accepted->GetIpv4SourceRoute();
            }
            client->SetIpv4SourceRoute({Addr(1, 1)});
            client->Send(Create<Packet>(1000), 0);
        });

        // Segments recording the first route again, none acceptable
        std::vector<Ipv4Address> second;
        Simulator::Schedule(Seconds(3), [this, client, &accepted, &second]() {
            if (!accepted)
            {
                return;
            }
            second = accepted->GetIpv4SourceRoute();
            Address local;
            client->GetSockName(local);
            uint16_t port = InetSocketAddress::ConvertFrom(local).GetPort();
            SequenceNumber32 rcvNxt = accepted->GetRxBuffer()->NextRxSequence();
            SequenceNumber32 sndNxt = client->GetRxBuffer()->NextRxSequence();

            TcpHeader tcp;
            tcp.SetSourcePort(port);
            tcp.SetDestinationPort(80);
            tcp.SetWindowSize(65535);
            // A FIN out of the window
            tcp.SetFlags(TcpHeader::FIN | TcpHeader::ACK);
            tcp.SetSequenceNumber(rcvNxt + 1000000);
            tcp.SetAckNumber(sndNxt);
            InjectSegment(Addr(0, 0), tcp);
            // An ACK of data never sent
            tcp.SetFlags(TcpHeader::ACK);
            tcp.SetSequenceNumber(rcvNxt);
            tcp.SetAckNumber(sndNxt + 100000);
            InjectSegment(Addr(0, 0), tcp);
            // A SYN on the synchronized connection
            tcp.SetFlags(TcpHeader::SYN);
            tcp.SetSequenceNumber(rcvNxt);
            tcp.SetAckNumber(SequenceNumber32(0));
            InjectSegment(Addr(0, 0), tcp);
            // Data received already, which ends at the left edge of the window
            tcp.SetFlags(TcpHeader::ACK);
            tcp.SetSequenceNumber(rcvNxt - 100);
            tcp.SetAckNumber(sndNxt);
            InjectSegment(Addr(0, 0), tcp, 100);
            // An ACK older than the last one, sent before it
            tcp.SetSequenceNumber(rcvNxt);
            tcp.SetAckNumber(sndNxt - 1);
            InjectSegment(Addr(0, 0), tcp);
            // A segment without an ACK
            tcp.SetFlags(0);
            tcp.SetAckNumber(SequenceNumber32(0));
            InjectSegment(Addr(0, 0), tcp);
        });
        RunSimulation();

        NS_TEST_ASSERT_MSG_NE(accepted, nullptr, "The connection was not established");
        std::vector<Ipv4Address> viaFirst = {Addr(1, 0), Addr(0, 0)};
        std::vector<Ipv4Address> viaSecond = {Addr(2, 0), Addr(0, 0)};
        NS_TEST_ASSERT_MSG_EQ((first == viaFirst), true, "The way back was not saved");
        NS_TEST_ASSERT_MSG_EQ((second == viaSecond),
                              true,
                              "A later segment did not update the way back");
        NS_TEST_ASSERT_MSG_EQ((accepted && accepted->GetIpv4SourceRoute() == viaSecond),
                              true,
                              "An unacceptable segment updated the way back");
        accepted = nullptr;
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Test the route of the connections a listener accepts
 *
 * A route the application set on a listening socket is inherited by the
 * connections it accepts, whose SYN+ACK follows it in place of the way back
 * (@RFC{9293}, Section 3.9.2.1, MUST-52), and a SYN+ACK sent again follows
 * the way back the SYN recorded, like the first.
 */
class Ipv4SourceRouteTcpListenerTest : public Ipv4SourceRouteTcpTestBase
{
  public:
    /**
     * Constructor.
     * @param checksum Whether the checksums are enabled.
     */
    Ipv4SourceRouteTcpListenerTest(bool checksum)
        : Ipv4SourceRouteTcpTestBase("The connections of a listener follow its route", checksum)
    {
    }

  private:
    void DoRun() override
    {
        BuildLine();
        EnableHops();

        // A listener with a route of its own, through the node next to it
        std::vector<Ipv4Address> route = {Addr(2, 0)};
        Ptr<TcpSocketBase> accepted;
        Ptr<TcpSocketBase> server = DynamicCast<TcpSocketBase>(
            Socket::CreateSocket(m_nodes.Get(3), TcpSocketFactory::GetTypeId()));
        server->Bind(InetSocketAddress(Ipv4Address::GetAny(), 80));
        server->SetIpv4SourceRoute(route);
        server->Listen();
        server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                  Callback<void, Ptr<Socket>, const Address&>(
                                      [&accepted](Ptr<Socket> socket, const Address&) {
                                          accepted = DynamicCast<TcpSocketBase>(socket);
                                      }));
        Ptr<TcpSocketBase> client = DynamicCast<TcpSocketBase>(
            Socket::CreateSocket(m_nodes.Get(0), TcpSocketFactory::GetTypeId()));
        client->Bind();
        client->SetIpv4SourceRoute({Addr(0, 1)});
        client->Connect(InetSocketAddress(Addr(2, 1), 80));

        // A listener without a route of its own, which a SYN reaches from a
        // source which never answers
        Ptr<Socket> other = Socket::CreateSocket(m_nodes.Get(3), TcpSocketFactory::GetTypeId());
        other->Bind(InetSocketAddress(Ipv4Address::GetAny(), 81));
        other->Listen();
        Simulator::Schedule(Seconds(1), [this]() {
            TcpHeader syn;
            syn.SetSourcePort(1234);
            syn.SetDestinationPort(81);
            syn.SetFlags(TcpHeader::SYN);
            syn.SetSequenceNumber(SequenceNumber32(1000));
            syn.SetWindowSize(65535);
            InjectSegment(Ipv4Address("10.0.1.9"), syn);
        });
        RunSimulation();

        NS_TEST_ASSERT_MSG_NE(accepted, nullptr, "The connection was not established");
        NS_TEST_ASSERT_MSG_EQ((accepted && accepted->GetIpv4SourceRoute() == route),
                              true,
                              "The accepted connection did not inherit the route");
        uint32_t ownSynAcks = 0;
        uint32_t recordedSynAcks = 0;
        for (const auto& header : m_observed[3].sent)
        {
            if (header.GetProtocol() != iana::internetprotocolnumbers::TCP)
            {
                continue;
            }
            NS_TEST_ASSERT_MSG_EQ(header.HasLooseSourceRoute(),
                                  true,
                                  "A segment did not follow a route");
            if (header.GetDestination() == Addr(2, 0))
            {
                ++ownSynAcks;
            }
            else if (header.GetDestination() == Addr(1, 0) &&
                     header.GetLooseSourceRoute().back() == Ipv4Address("10.0.1.9"))
            {
                ++recordedSynAcks;
            }
        }
        NS_TEST_ASSERT_MSG_GT(ownSynAcks, 0, "No segment followed the route of the listener");
        NS_TEST_ASSERT_MSG_GT(recordedSynAcks,
                              1,
                              "The SYN+ACK sent again did not follow the way back");
        accepted = nullptr;
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Test that the zero window probes follow the route of the connection
 *
 * The probes of a closed window are segments of the connection like any
 * other, and follow its source route (@RFC{9293}, Section 3.9.2.1, MUST-51),
 * whether or not the connection uses ECN.
 */
class Ipv4SourceRouteTcpWindowProbeTest : public Ipv4SourceRouteTcpTestBase
{
  public:
    /**
     * Constructor.
     * @param checksum Whether the checksums are enabled.
     */
    Ipv4SourceRouteTcpWindowProbeTest(bool checksum)
        : Ipv4SourceRouteTcpTestBase("The zero window probes follow the route", checksum)
    {
    }

  private:
    void DoRun() override
    {
        BuildLine();
        EnableHops();

        // A server which reads nothing, so that its window closes once two
        // segments of the client, 8 bytes of route short of the 508 set, fill it
        Ptr<Socket> server = Socket::CreateSocket(m_nodes.Get(3), TcpSocketFactory::GetTypeId());
        server->SetAttribute("RcvBufSize", UintegerValue(1000));
        server->SetAttribute("Timestamp", BooleanValue(false));
        server->Bind(InetSocketAddress(Ipv4Address::GetAny(), 80));
        server->Listen();

        Ptr<TcpSocketBase> client = DynamicCast<TcpSocketBase>(
            Socket::CreateSocket(m_nodes.Get(0), TcpSocketFactory::GetTypeId()));
        client->SetAttribute("SegmentSize", UintegerValue(508));
        client->SetAttribute("Timestamp", BooleanValue(false));
        client->Bind();
        client->SetIpv4SourceRoute({Addr(0, 1)});
        uint32_t windowProbes = 0;
        client->TraceConnectWithoutContext(
            "Tx",
            Callback<void, Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>>(
                [&windowProbes](Ptr<const Packet> packet,
                                const TcpHeader&,
                                Ptr<const TcpSocketBase>) {
                    windowProbes += (packet->GetSize() == 1) ? 1 : 0;
                }));
        client->Connect(InetSocketAddress(Addr(2, 1), 80));
        Simulator::Schedule(Seconds(1), [client]() { client->Send(Create<Packet>(3000), 0); });
        RunSimulation();

        NS_TEST_ASSERT_MSG_GT(windowProbes, 0, "The zero window was not probed");
        uint32_t segments = 0;
        for (const auto& header : m_observed[0].sent)
        {
            if (header.GetProtocol() != iana::internetprotocolnumbers::TCP)
            {
                continue;
            }
            ++segments;
            NS_TEST_ASSERT_MSG_EQ(header.HasLooseSourceRoute(),
                                  true,
                                  "A segment did not follow the route");
        }
        NS_TEST_ASSERT_MSG_GT(segments, windowProbes, "Too few segments were sent");
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Test that a segment refused while opening leaves the way back alone
 *
 * A connection opening actively learns the way back from the SYN of its
 * peer: a segment without a SYN, which the connection drops in SYN-SENT
 * (@RFC{9293}, Section 3.10.7.3), must not send the SYN retransmissions
 * along the route it recorded.
 */
class Ipv4SourceRouteTcpSynSentTest : public Ipv4SourceRouteTcpTestBase
{
  public:
    /**
     * Constructor.
     * @param checksum Whether the checksums are enabled.
     */
    Ipv4SourceRouteTcpSynSentTest(bool checksum)
        : Ipv4SourceRouteTcpTestBase("A segment refused while opening leaves the way back alone",
                                     checksum)
    {
    }

  private:
    void DoRun() override
    {
        BuildLine();
        EnableHops();

        // A peer which never answers the SYN, on the link of node 0
        const Ipv4Address peer("10.0.1.9");
        Ptr<TcpSocketBase> client = DynamicCast<TcpSocketBase>(
            Socket::CreateSocket(m_nodes.Get(3), TcpSocketFactory::GetTypeId()));
        // Without timestamps, which would have the injected segment discarded
        // for lack of them
        client->SetAttribute("Timestamp", BooleanValue(false));
        client->Bind();
        client->Connect(InetSocketAddress(peer, 80));

        Simulator::Schedule(Seconds(1), [this, client, peer]() {
            Address local;
            client->GetSockName(local);
            TcpHeader tcp;
            tcp.SetSourcePort(80);
            tcp.SetDestinationPort(InetSocketAddress::ConvertFrom(local).GetPort());
            tcp.SetWindowSize(65535);
            tcp.SetFlags(TcpHeader::ACK);
            tcp.SetSequenceNumber(SequenceNumber32(1000));
            tcp.SetAckNumber(SequenceNumber32(12345));
            InjectSegment(peer, tcp);
        });
        RunSimulation();

        NS_TEST_ASSERT_MSG_EQ(client->GetIpv4SourceRoute().empty(),
                              true,
                              "A segment dropped in SYN-SENT saved the way back");
        uint32_t syns = 0;
        for (const auto& header : m_observed[3].sent)
        {
            if (header.GetProtocol() != iana::internetprotocolnumbers::TCP)
            {
                continue;
            }
            ++syns;
            NS_TEST_ASSERT_MSG_EQ(header.HasLooseSourceRoute(),
                                  false,
                                  "A SYN followed the way back of a dropped segment");
        }
        NS_TEST_ASSERT_MSG_GT(syns, 1, "The SYN was not sent again");
    }
};

/**
 * @ingroup internet-test
 *
 * @brief IPv4 options TestSuite
 */
class Ipv4SourceRouteTestSuite : public TestSuite
{
  public:
    Ipv4SourceRouteTestSuite()
        : TestSuite("ipv4-source-route", Type::UNIT)
    {
        for (bool checksum : {false, true})
        {
            AddTestCase(new Ipv4MalformedHeaderTest(checksum), TestCase::Duration::QUICK);
            AddTestCase(new Ipv4MalformedSourceRouteTest(checksum), TestCase::Duration::QUICK);
            AddTestCase(new Ipv4SourceRouteOwnAddressTest(checksum), TestCase::Duration::QUICK);
            AddTestCase(new Ipv4SourceRouteForwardingSwitchTest(checksum),
                        TestCase::Duration::QUICK);
            AddTestCase(new Ipv4SourceRouteRelayLimitsTest(checksum), TestCase::Duration::QUICK);
            AddTestCase(new Ipv4SourceRouteHopLimitTest(checksum), TestCase::Duration::QUICK);
            AddTestCase(new Ipv4SourceRouteUdpRelayTest(checksum), TestCase::Duration::QUICK);
            AddTestCase(new Ipv4SourceRouteEchoTest(checksum), TestCase::Duration::QUICK);
            AddTestCase(new Ipv4SourceRouteRstTest(checksum), TestCase::Duration::QUICK);
            AddTestCase(new Ipv4SourceRouteIpv6SocketTest(checksum), TestCase::Duration::QUICK);
            AddTestCase(new Ipv4SourceRouteTcpUpdateTest(checksum), TestCase::Duration::QUICK);
            AddTestCase(new Ipv4SourceRouteTcpListenerTest(checksum), TestCase::Duration::QUICK);
            AddTestCase(new Ipv4SourceRouteTcpWindowProbeTest(checksum), TestCase::Duration::QUICK);
            AddTestCase(new Ipv4SourceRouteTcpSynSentTest(checksum), TestCase::Duration::QUICK);
        }
    }
};

static Ipv4SourceRouteTestSuite
    g_ipv4SourceRouteTestSuite; //!< Static variable for test initialization
