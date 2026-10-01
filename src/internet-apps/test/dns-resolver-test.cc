/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ns3/boolean.h"
#include "ns3/dns-resolver-helper.h"
#include "ns3/dns-resolver.h"
#include "ns3/inet-socket-address.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4.h"
#include "ns3/ipv6-address-helper.h"
#include "ns3/log.h"
#include "ns3/node-container.h"
#include "ns3/packet.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/simple-ref-count.h"
#include "ns3/simulator.h"
#include "ns3/socket.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/test.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/uinteger.h"

#include <map>
#include <set>

/**
 * @file
 * @ingroup dns-resolver
 * DNS resolver tests.
 */

/**
 * @ingroup internet-apps
 * @defgroup dns-resolver-test DNS resolver tests
 */

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("DnsResolverTest");

namespace
{

const Ipv4Address SERVER("10.0.0.2");          ///< IPv4 address of the server
const Ipv4Address SERVFAIL_SERVER("10.0.0.3"); ///< address of a server answering SERVFAIL
const Ipv4Address DEAD_SERVER("10.0.0.99");    ///< address of a server not answering

/**
 * @ingroup dns-resolver-test
 * @brief Encode a domain name, without compression.
 * @param name the dot-separated name
 * @return the encoded name
 */
std::vector<uint8_t>
EncodeName(const std::string& name)
{
    std::vector<uint8_t> encoded;
    size_t start = 0;
    while (start < name.size())
    {
        size_t dot = name.find('.', start);
        dot = dot == std::string::npos ? name.size() : dot;
        encoded.push_back(static_cast<uint8_t>(dot - start));
        encoded.insert(encoded.end(), name.begin() + start, name.begin() + dot);
        start = dot + 1;
    }
    encoded.push_back(0);
    return encoded;
}

/**
 * @ingroup dns-resolver-test
 * @brief A compression pointer.
 * @param offset the offset of the name pointed to
 * @return the encoded pointer
 */
std::vector<uint8_t>
Pointer(size_t offset)
{
    return {static_cast<uint8_t>(0xc0 | (offset >> 8)), static_cast<uint8_t>(offset)};
}

/**
 * @ingroup dns-resolver-test
 * @brief Append a 32-bit big-endian value.
 * @param data the data
 * @param value the value
 */
void
AppendU32(std::vector<uint8_t>& data, uint32_t value)
{
    data.insert(data.end(),
                {static_cast<uint8_t>(value >> 24),
                 static_cast<uint8_t>(value >> 16),
                 static_cast<uint8_t>(value >> 8),
                 static_cast<uint8_t>(value)});
}

/**
 * @ingroup dns-resolver-test
 * @brief The data of an SOA record.
 * @param ttlMinimum the MINIMUM field
 * @return the data
 */
std::vector<uint8_t>
SoaData(uint32_t ttlMinimum)
{
    std::vector<uint8_t> soa = EncodeName("ns.nsnam.org");
    auto rname = EncodeName("admin.nsnam.org");
    soa.insert(soa.end(), rname.begin(), rname.end());
    for (uint32_t value : {1u, 3600u, 600u, 86400u, ttlMinimum})
    {
        AppendU32(soa, value);
    }
    return soa;
}

/**
 * @ingroup dns-resolver-test
 * @brief Builder of DNS messages.
 */
class MessageBuilder
{
  public:
    /// Offset of the name of the question
    static constexpr size_t QUESTION_NAME{12};

    /**
     * Start a response with a question.
     * @param id the identifier
     * @param flags the flags of the header (QR, opcode, AA, TC, RD, RA, RCODE)
     * @param name the name of the question
     * @param type the type of the question
     */
    MessageBuilder(uint16_t id, uint16_t flags, const std::string& name, uint16_t type)
        : m_message{static_cast<uint8_t>(id >> 8),
                    static_cast<uint8_t>(id),
                    static_cast<uint8_t>(flags >> 8),
                    static_cast<uint8_t>(flags),
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0}
    {
        auto encoded = EncodeName(name);
        m_message.insert(m_message.end(), encoded.begin(), encoded.end());
        m_message.insert(m_message.end(),
                         {static_cast<uint8_t>(type >> 8), static_cast<uint8_t>(type), 0, 1});
    }

    /**
     * Append a record; the records must be appended in the order of the sections.
     * @param section the section: 0 for answer, 1 for authority, 2 for additional
     * @param owner the encoded owner name
     * @param type the type
     * @param ttl the TTL field
     * @param rdata the data
     * @param rclass the class
     * @return the offset of the data
     */
    size_t Add(uint8_t section,
               const std::vector<uint8_t>& owner,
               uint16_t type,
               uint32_t ttl,
               const std::vector<uint8_t>& rdata,
               uint16_t rclass = 1)
    {
        m_message.insert(m_message.end(), owner.begin(), owner.end());
        m_message.insert(m_message.end(),
                         {static_cast<uint8_t>(type >> 8),
                          static_cast<uint8_t>(type),
                          static_cast<uint8_t>(rclass >> 8),
                          static_cast<uint8_t>(rclass)});
        AppendU32(m_message, ttl);
        m_message.insert(
            m_message.end(),
            {static_cast<uint8_t>(rdata.size() >> 8), static_cast<uint8_t>(rdata.size())});
        const size_t offset = m_message.size();
        m_message.insert(m_message.end(), rdata.begin(), rdata.end());
        m_message[7 + 2 * section]++;
        return offset;
    }

    /**
     * @return the message
     */
    const std::vector<uint8_t>& Get() const
    {
        return m_message;
    }

  private:
    std::vector<uint8_t> m_message; ///< the message
};

} // namespace

/**
 * @ingroup dns-resolver-test
 * @brief DNS server answering the queries of the tests.
 *
 * It answers on UDP and TCP at SERVER, on UDP at its IPv6 address, and with SERVFAIL on UDP at
 * SERVFAIL_SERVER:
 * - www.nsnam.org: an alias of web.nsnam.org (TTL 3600 s), with two A records (TTL 60 and 120 s)
 *   or an AAAA record (TTL 60 s);
 * - big.nsnam.org: a truncated response over UDP, and 20 A records over TCP;
 * - tcpclose.nsnam.org: a truncated response over UDP, and the TCP connection closed;
 * - spoof.nsnam.org: a response to another question;
 * - stray.nsnam.org: an A record of another name;
 * - referral.nsnam.org: a referral, without recursion;
 * - loop.nsnam.org: an alias of itself;
 * - noedns.nsnam.org: FORMERR without question to the queries with EDNS(0), an A record
 *   otherwise;
 * - ttlmsb.nsnam.org: an A record with the most significant bit of its TTL set;
 * - late.nsnam.org: an A record, sent after 1.5 s;
 * - lateerror.nsnam.org: SERVFAIL, sent after 1.5 s;
 * - tcptrunc.nsnam.org: a truncated response over UDP and over TCP;
 * - chain.nsnam.org: an alias of target.nsnam.org, without its records; target.nsnam.org: an A
 *   record;
 * - v6only.nsnam.org: an AAAA record, and NXDOMAIN for the other types;
 * - lateempty.nsnam.org: sent after 1.5 s, an A record over IPv6, and a response without record
 *   over IPv4;
 * - latenoedns.nsnam.org: as noedns.nsnam.org, with the FORMERR sent after 1.5 s;
 * - latechain.nsnam.org: as chain.nsnam.org, sent after 1.5 s;
 * - underscore.nsnam.org: an alias of _target.nsnam.org, without its records;
 *   _target.nsnam.org: an A record;
 * - other names: NXDOMAIN, with the SOA record of nsnam.org (negative TTL of 30 s).
 */
class DnsTestServer : public SimpleRefCount<DnsTestServer>
{
  public:
    /**
     * Start the server.
     * @param node the node of the server
     * @param server6 the IPv6 address of the server
     */
    void Start(Ptr<Node> node, Ipv6Address server6)
    {
        for (const Address& address : {Address(InetSocketAddress(SERVER, 53)),
                                       Address(InetSocketAddress(SERVFAIL_SERVER, 53)),
                                       Address(Inet6SocketAddress(server6, 53))})
        {
            auto socket = Socket::CreateSocket(node, UdpSocketFactory::GetTypeId());
            socket->Bind(address);
            socket->SetRecvCallback(MakeCallback(&DnsTestServer::ReceiveUdp, this));
            m_sockets.push_back(socket);
        }
        auto tcp = Socket::CreateSocket(node, TcpSocketFactory::GetTypeId());
        tcp->Bind(InetSocketAddress(SERVER, 53));
        tcp->Listen();
        tcp->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                               MakeCallback(&DnsTestServer::Accept, this));
        m_sockets.push_back(tcp);
    }

    /// Stop the server
    void Stop()
    {
        for (auto& socket : m_sockets)
        {
            socket->Close();
        }
        m_sockets.clear();
        m_tcpData.clear();
    }

    /// Number of queries received, by "name/type/transport" (transport: udp, udp6 or tcp)
    std::map<std::string, uint32_t> m_queries;
    uint32_t m_ednsQueries{0};  ///< number of queries with an EDNS(0) OPT record
    std::set<uint16_t> m_ports; ///< source ports of the queries received over UDP

  private:
    /**
     * Answer the queries received over UDP.
     * @param socket the socket
     */
    void ReceiveUdp(Ptr<Socket> socket)
    {
        Address local;
        socket->GetSockName(local);
        const bool ipv6 = Inet6SocketAddress::IsMatchingType(local);
        const bool servfail =
            !ipv6 && InetSocketAddress::ConvertFrom(local).GetIpv4() == SERVFAIL_SERVER;
        Address from;
        while (Ptr<Packet> packet = socket->RecvFrom(from))
        {
            m_ports.insert(ipv6 ? Inet6SocketAddress::ConvertFrom(from).GetPort()
                                : InetSocketAddress::ConvertFrom(from).GetPort());
            std::vector<uint8_t> query(packet->GetSize());
            packet->CopyData(query.data(), query.size());
            bool late = false;
            auto response = Answer(query, ipv6 ? "udp6" : "udp", servfail, late);
            auto out = Create<Packet>(response.data(), response.size());
            if (late)
            {
                Simulator::Schedule(MilliSeconds(1500),
                                    [socket, out, from]() { socket->SendTo(out, 0, from); });
            }
            else
            {
                socket->SendTo(out, 0, from);
            }
        }
    }

    /**
     * Accept a TCP connection.
     * @param socket the socket of the connection
     * @param from the address of the client
     */
    void Accept(Ptr<Socket> socket, const Address& from)
    {
        socket->SetRecvCallback(MakeCallback(&DnsTestServer::ReceiveTcp, this));
        m_sockets.push_back(socket);
    }

    /**
     * Answer a query received over TCP.
     * @param socket the socket of the connection
     */
    void ReceiveTcp(Ptr<Socket> socket)
    {
        auto& data = m_tcpData[socket];
        while (Ptr<Packet> packet = socket->Recv())
        {
            const size_t size = data.size();
            data.resize(size + packet->GetSize());
            packet->CopyData(data.data() + size, packet->GetSize());
        }
        if (data.size() < 2 || data.size() < 2u + (data[0] << 8 | data[1]))
        {
            return;
        }
        std::vector<uint8_t> query(data.begin() + 2, data.end());
        data.clear();
        bool late = false;
        auto response = Answer(query, "tcp", false, late);
        if (response.empty())
        {
            socket->Close();
            return;
        }
        std::vector<uint8_t> out{static_cast<uint8_t>(response.size() >> 8),
                                 static_cast<uint8_t>(response.size())};
        out.insert(out.end(), response.begin(), response.end());
        socket->Send(Create<Packet>(out.data(), out.size()));
    }

    /**
     * Build the response to a query.
     * @param query the query
     * @param transport the transport of the query
     * @param servfail whether to answer SERVFAIL
     * @param late set to true if the response must be delayed
     * @return the response, empty to close the TCP connection
     */
    std::vector<uint8_t> Answer(const std::vector<uint8_t>& query,
                                const std::string& transport,
                                bool servfail,
                                bool& late)
    {
        std::string name;
        size_t i = MessageBuilder::QUESTION_NAME;
        for (; i < query.size() && query[i] != 0; i += query[i] + 1)
        {
            name += (name.empty() ? "" : ".") +
                    std::string(query.begin() + i + 1, query.begin() + i + 1 + query[i]);
        }
        const uint16_t type = query[i + 1] << 8 | query[i + 2];
        const uint16_t id = query[0] << 8 | query[1];
        const bool edns = query[11] > 0;
        m_queries[name + "/" + std::to_string(type) + "/" + transport]++;
        m_ednsQueries += edns ? 1 : 0;

        // response, recursion desired and available
        constexpr uint16_t flags = 0x8180;
        const auto qname = Pointer(MessageBuilder::QUESTION_NAME);
        // "nsnam.org", in the name of the question
        const auto zone = Pointer(MessageBuilder::QUESTION_NAME + name.find("nsnam"));
        if (servfail)
        {
            return MessageBuilder(id, flags | 2, name, type).Get();
        }
        if (name == "www.nsnam.org")
        {
            MessageBuilder message(id, flags, name, type);
            const size_t alias =
                message.Add(0, qname, DnsResolver::TYPE_CNAME, 3600, EncodeName("web.nsnam.org"));
            if (type == DnsResolver::TYPE_A)
            {
                message.Add(0, Pointer(alias), DnsResolver::TYPE_A, 60, {1, 2, 3, 4});
                message.Add(0, Pointer(alias), DnsResolver::TYPE_A, 120, {5, 6, 7, 8});
            }
            else
            {
                uint8_t address[16];
                Ipv6Address("2001:db8::80").Serialize(address);
                message.Add(0,
                            Pointer(alias),
                            DnsResolver::TYPE_AAAA,
                            60,
                            std::vector<uint8_t>(address, address + 16));
            }
            return message.Get();
        }
        if (name == "lateerror.nsnam.org")
        {
            late = true;
            return MessageBuilder(id, flags | 2, name, type).Get();
        }
        if (name == "tcptrunc.nsnam.org")
        {
            return MessageBuilder(id, flags | 0x0200, name, type).Get();
        }
        if (name == "chain.nsnam.org" || name == "underscore.nsnam.org" ||
            name == "latechain.nsnam.org")
        {
            late = name == "latechain.nsnam.org";
            MessageBuilder message(id, flags, name, type);
            message.Add(0,
                        qname,
                        DnsResolver::TYPE_CNAME,
                        50,
                        EncodeName(name == "underscore.nsnam.org" ? "_target.nsnam.org"
                                                                  : "target.nsnam.org"));
            return message.Get();
        }
        if (name == "target.nsnam.org" || name == "_target.nsnam.org")
        {
            const uint8_t byte = name == "target.nsnam.org" ? 4 : 3;
            MessageBuilder message(id, flags, name, type);
            message.Add(0, qname, DnsResolver::TYPE_A, 300, {byte, byte, byte, byte});
            return message.Get();
        }
        if (name == "lateempty.nsnam.org")
        {
            late = true;
            MessageBuilder message(id, flags, name, type);
            if (transport == "udp6")
            {
                message.Add(0, qname, DnsResolver::TYPE_A, 60, {7, 7, 7, 7});
            }
            return message.Get();
        }
        if (name == "v6only.nsnam.org" && type == DnsResolver::TYPE_AAAA)
        {
            uint8_t address[16];
            Ipv6Address("2001:db8::66").Serialize(address);
            MessageBuilder message(id, flags, name, type);
            message.Add(0,
                        qname,
                        DnsResolver::TYPE_AAAA,
                        300,
                        std::vector<uint8_t>(address, address + 16));
            return message.Get();
        }
        if ((name == "big.nsnam.org" || name == "tcpclose.nsnam.org") && transport != "tcp")
        {
            return MessageBuilder(id, flags | 0x0200, name, type).Get();
        }
        if (name == "tcpclose.nsnam.org")
        {
            return {};
        }
        if (name == "big.nsnam.org")
        {
            MessageBuilder message(id, flags, name, type);
            for (uint8_t host = 1; host <= 20; host++)
            {
                message.Add(0, qname, DnsResolver::TYPE_A, 60, {10, 1, 0, host});
            }
            return message.Get();
        }
        if (name == "spoof.nsnam.org")
        {
            MessageBuilder message(id, flags, "evil.nsnam.org", type);
            message.Add(0, qname, DnsResolver::TYPE_A, 60, {6, 6, 6, 6});
            return message.Get();
        }
        if (name == "stray.nsnam.org")
        {
            MessageBuilder message(id, flags, name, type);
            message.Add(0, EncodeName("other.nsnam.org"), DnsResolver::TYPE_A, 60, {7, 7, 7, 7});
            return message.Get();
        }
        if (name == "referral.nsnam.org")
        {
            // no recursion available
            MessageBuilder message(id, 0x8100, name, type);
            message.Add(1, zone, DnsResolver::TYPE_NS, 3600, EncodeName("ns.nsnam.org"));
            return message.Get();
        }
        if (name == "loop.nsnam.org")
        {
            MessageBuilder message(id, flags, name, type);
            message.Add(0, qname, DnsResolver::TYPE_CNAME, 60, EncodeName("loop.nsnam.org"));
            return message.Get();
        }
        if ((name == "noedns.nsnam.org" || name == "latenoedns.nsnam.org") && edns)
        {
            late = name == "latenoedns.nsnam.org";
            // FORMERR, without question
            return {static_cast<uint8_t>(id >> 8),
                    static_cast<uint8_t>(id),
                    0x81,
                    0x81,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0};
        }
        if (name == "noedns.nsnam.org" || name == "latenoedns.nsnam.org" ||
            name == "ttlmsb.nsnam.org" || name == "late.nsnam.org")
        {
            late = name == "late.nsnam.org";
            MessageBuilder message(id, flags, name, type);
            message.Add(0,
                        qname,
                        DnsResolver::TYPE_A,
                        name == "ttlmsb.nsnam.org" ? 0xffffffff : 60,
                        {9, 9, 9, 9});
            return message.Get();
        }
        // NXDOMAIN, with the SOA record of the zone for the negative TTL
        MessageBuilder message(id, flags | 3, name, type);
        message.Add(1, zone, DnsResolver::TYPE_SOA, 300, SoaData(30));
        return message.Get();
    }

    std::vector<Ptr<Socket>> m_sockets;                    ///< sockets of the server
    std::map<Ptr<Socket>, std::vector<uint8_t>> m_tcpData; ///< data received over TCP
};

/**
 * @ingroup dns-resolver-test
 * @brief Base of the DNS resolver test cases: a client and a DnsTestServer.
 */
class DnsResolverTestBase : public TestCase
{
  public:
    /**
     * Constructor.
     * @param name the name of the test case
     */
    DnsResolverTestBase(const std::string& name)
        : TestCase(name)
    {
    }

  protected:
    /// Create the client and the server, and start the server
    void Setup()
    {
        NodeContainer nodes(2);
        InternetStackHelper internet;
        internet.Install(nodes);
        SimpleNetDeviceHelper devices;
        auto netDevices = devices.Install(nodes);
        Ipv4AddressHelper addresses("10.0.0.0", "255.255.255.0");
        addresses.Assign(netDevices);
        nodes.Get(1)->GetObject<Ipv4>()->AddAddress(
            1,
            Ipv4InterfaceAddress(SERVFAIL_SERVER, "255.255.255.0"));
        Ipv6AddressHelper addresses6(Ipv6Address("2001:db8::"), Ipv6Prefix(64));
        m_server6 = addresses6.Assign(netDevices).GetAddress(1, 1);

        m_server = Create<DnsTestServer>();
        m_server->Start(nodes.Get(1), m_server6);
        DnsResolverHelper resolverHelper(SERVER);
        resolverHelper.SetAttribute("Timeout", TimeValue(Seconds(1)));
        resolverHelper.SetAttribute("Retransmissions", UintegerValue(1));
        resolverHelper.Install(nodes.Get(0));
        DnsResolverHelper::AssignStreams(nodes.Get(0), 1);
        m_resolver = nodes.Get(0)->GetObject<DnsResolver>();
    }

    /// Stop the server and destroy the simulation
    void Teardown()
    {
        m_server->Stop();
        m_server = nullptr;
        m_resolver = nullptr;
        Simulator::Destroy();
    }

    /**
     * Schedule a resolution, whose result is stored under a key.
     * @param time the time of the resolution
     * @param key the key of the result
     * @param name the host name
     * @param family the address family
     */
    void ScheduleResolve(Time time,
                         const std::string& key,
                         const std::string& name,
                         DnsResolver::AddressFamily family = DnsResolver::IPV4)
    {
        Simulator::Schedule(time, [this, key, name, family]() {
            m_resolver->Resolve(
                name,
                DnsResolver::ResolveCallback(
                    [this, key](const std::string&, const std::vector<Address>& addresses) {
                        m_results[key] = addresses;
                        m_times[key] = Simulator::Now();
                    }),
                family);
        });
    }

    /**
     * Schedule the recording of the number of queries received by the server.
     * @param time the time of the recording
     * @param key the key of the recording
     * @param counter the counter of the server ("name/type/transport")
     */
    void ScheduleCount(Time time, const std::string& key, const std::string& counter)
    {
        Simulator::Schedule(time, [this, key, counter]() {
            m_counts[key] = m_server->m_queries[counter];
        });
    }

    /**
     * @param key the key of a result
     * @return the addresses of the result, as text separated by spaces
     */
    std::string Result(const std::string& key)
    {
        std::ostringstream text;
        for (const auto& address : m_results[key])
        {
            text << (text.tellp() > 0 ? " " : "");
            if (Ipv4Address::IsMatchingType(address))
            {
                text << Ipv4Address::ConvertFrom(address);
            }
            else
            {
                text << Ipv6Address::ConvertFrom(address);
            }
        }
        return text.str();
    }

    Ptr<DnsTestServer> m_server;                           ///< the server
    Ipv6Address m_server6;                                 ///< IPv6 address of the server
    Ptr<DnsResolver> m_resolver;                           ///< the resolver of the client
    std::map<std::string, std::vector<Address>> m_results; ///< results, by key
    std::map<std::string, Time> m_times;                   ///< times of the results, by key
    std::map<std::string, uint32_t> m_counts;              ///< numbers of queries, by key
};

/**
 * @ingroup dns-resolver-test
 * @brief Decoding of responses: header, question, aliases, EDNS(0), names, negative TTL.
 */
class DnsResolverDecodingTestCase : public TestCase
{
  public:
    DnsResolverDecodingTestCase()
        : TestCase("Decoding of responses")
    {
    }

  private:
    void DoRun() override
    {
        const std::string name = "www.nsnam.org";
        const auto qname = Pointer(MessageBuilder::QUESTION_NAME);
        const auto zone = Pointer(MessageBuilder::QUESTION_NAME + 4); // nsnam.org
        auto decode = [&](const MessageBuilder& message) {
            return DnsResolver::DecodeResponse(message.Get(),
                                               "WWW.nsnam.org.",
                                               DnsResolver::TYPE_A);
        };

        // valid response, with an alias whose addresses are the only ones taken
        MessageBuilder valid(1, 0x8180, "www.NSNAM.org", DnsResolver::TYPE_A);
        const size_t alias =
            valid.Add(0, qname, DnsResolver::TYPE_CNAME, 300, EncodeName("web.nsnam.org"));
        valid.Add(0, Pointer(alias), DnsResolver::TYPE_A, 0xffffffff, {1, 2, 3, 4});
        valid.Add(0, qname, DnsResolver::TYPE_A, 60, {6, 6, 6, 6});
        valid.Add(0, EncodeName("other.org"), DnsResolver::TYPE_A, 60, {7, 7, 7, 7});
        auto response = decode(valid);
        NS_TEST_ASSERT_MSG_EQ(response.valid, true, "Valid response rejected");
        NS_TEST_ASSERT_MSG_EQ(response.addresses.size(), 1, "Records of other names accepted");
        NS_TEST_EXPECT_MSG_EQ(Ipv4Address::ConvertFrom(response.addresses[0]),
                              Ipv4Address("1.2.3.4"),
                              "Wrong address");
        NS_TEST_EXPECT_MSG_EQ(response.ttl, 0, "A TTL with the most significant bit set is not 0");

        // header and question
        for (uint16_t flags : {0x0180, 0xa180})
        {
            MessageBuilder message(1, flags, name, DnsResolver::TYPE_A);
            NS_TEST_EXPECT_MSG_EQ(decode(message).valid,
                                  false,
                                  "Query or non-standard opcode accepted");
        }
        NS_TEST_EXPECT_MSG_EQ(
            decode(MessageBuilder(1, 0x8180, "evil.org", DnsResolver::TYPE_A)).valid,
            false,
            "Other question accepted");
        NS_TEST_EXPECT_MSG_EQ(decode(MessageBuilder(1, 0x8180, name, DnsResolver::TYPE_AAAA)).valid,
                              false,
                              "Other question type accepted");
        auto twoQuestions = MessageBuilder(1, 0x8180, name, DnsResolver::TYPE_A).Get();
        twoQuestions[5] = 2;
        NS_TEST_EXPECT_MSG_EQ(
            DnsResolver::DecodeResponse(twoQuestions, name, DnsResolver::TYPE_A).valid,
            false,
            "Two questions accepted");

        // EDNS(0): extended RCODE, and a single OPT record
        MessageBuilder badvers(1, 0x8180, name, DnsResolver::TYPE_A);
        badvers.Add(2, {0}, DnsResolver::TYPE_OPT, 0x01000000, {}, 1232);
        response = decode(badvers);
        NS_TEST_EXPECT_MSG_EQ(response.hasOpt, true, "OPT record not found");
        NS_TEST_EXPECT_MSG_EQ(response.rcode, DnsResolver::RCODE_BADVERS, "Wrong extended RCODE");
        badvers.Add(2, {0}, DnsResolver::TYPE_OPT, 0, {}, 1232);
        NS_TEST_EXPECT_MSG_EQ(decode(badvers).rcode,
                              DnsResolver::RCODE_SERVFAIL,
                              "Two OPT records accepted");
        // EDNS version 1: an error, unless BADVERS
        MessageBuilder version1(1, 0x8180, name, DnsResolver::TYPE_A);
        version1.Add(2, {0}, DnsResolver::TYPE_OPT, 0x00010000, {}, 1232);
        NS_TEST_EXPECT_MSG_EQ(decode(version1).rcode,
                              DnsResolver::RCODE_SERVFAIL,
                              "Response of another EDNS version accepted");
        MessageBuilder badversion1(1, 0x8180, name, DnsResolver::TYPE_A);
        badversion1.Add(2, {0}, DnsResolver::TYPE_OPT, 0x01010000, {}, 1232);
        NS_TEST_EXPECT_MSG_EQ(decode(badversion1).rcode,
                              DnsResolver::RCODE_BADVERS,
                              "BADVERS of another EDNS version rejected");

        // a single alias for a name, possibly repeated, and address records of the right length
        MessageBuilder twoAliases(1, 0x8180, name, DnsResolver::TYPE_A);
        twoAliases.Add(0, qname, DnsResolver::TYPE_CNAME, 60, EncodeName("web.nsnam.org"));
        twoAliases.Add(0, qname, DnsResolver::TYPE_CNAME, 60, EncodeName("ftp.nsnam.org"));
        NS_TEST_EXPECT_MSG_EQ(decode(twoAliases).rcode,
                              DnsResolver::RCODE_SERVFAIL,
                              "Two aliases for a name accepted");
        MessageBuilder repeated(1, 0x8180, name, DnsResolver::TYPE_A);
        repeated.Add(0, qname, DnsResolver::TYPE_CNAME, 60, EncodeName("web.nsnam.org"));
        const size_t web =
            repeated.Add(0, qname, DnsResolver::TYPE_CNAME, 30, EncodeName("web.nsnam.org"));
        repeated.Add(0, Pointer(web), DnsResolver::TYPE_A, 60, {1, 2, 3, 4});
        repeated.Add(0, Pointer(web), DnsResolver::TYPE_A, 60, {1, 2, 3, 4});
        response = decode(repeated);
        NS_TEST_EXPECT_MSG_EQ(response.rcode, DnsResolver::RCODE_NOERROR, "Repeated alias");
        NS_TEST_EXPECT_MSG_EQ(response.addresses.size(), 1, "Duplicate address not removed");
        NS_TEST_EXPECT_MSG_EQ(response.ttl, 30, "TTL of the repeated alias ignored");
        MessageBuilder shortAddress(1, 0x8180, name, DnsResolver::TYPE_A);
        shortAddress.Add(0, qname, DnsResolver::TYPE_A, 60, {1, 2, 3, 4});
        shortAddress.Add(0, qname, DnsResolver::TYPE_A, 60, {5, 6, 7});
        response = decode(shortAddress);
        NS_TEST_EXPECT_MSG_EQ(response.rcode,
                              DnsResolver::RCODE_SERVFAIL,
                              "Address record of the wrong length accepted");
        NS_TEST_EXPECT_MSG_EQ(response.addresses.empty(), true, "Addresses of a failure");

        // names: loops, forward pointers, and length
        MessageBuilder loop(1, 0x8180, name, DnsResolver::TYPE_A);
        auto selfPointer = loop.Get();
        selfPointer[7] = 1;
        const size_t owner = selfPointer.size();
        auto pointer = Pointer(owner);
        selfPointer.insert(selfPointer.end(), pointer.begin(), pointer.end());
        selfPointer.insert(selfPointer.end(), {0, 1, 0, 1, 0, 0, 0, 60, 0, 4, 1, 2, 3, 4});
        NS_TEST_EXPECT_MSG_EQ(
            DnsResolver::DecodeResponse(selfPointer, name, DnsResolver::TYPE_A).rcode,
            DnsResolver::RCODE_SERVFAIL,
            "Compression loop accepted");
        auto forward = selfPointer;
        auto forwardPointer = Pointer(owner + 2);
        std::copy(forwardPointer.begin(), forwardPointer.end(), forward.begin() + owner);
        NS_TEST_EXPECT_MSG_EQ(DnsResolver::DecodeResponse(forward, name, DnsResolver::TYPE_A).rcode,
                              DnsResolver::RCODE_SERVFAIL,
                              "Forward compression pointer accepted");
        MessageBuilder longName(1, 0x8180, name, DnsResolver::TYPE_A);
        longName.Add(0,
                     EncodeName(std::string(63, 'a') + "." + std::string(63, 'b') + "." +
                                std::string(63, 'c') + "." + std::string(63, 'd')),
                     DnsResolver::TYPE_A,
                     60,
                     {1, 2, 3, 4});
        NS_TEST_EXPECT_MSG_EQ(decode(longName).rcode,
                              DnsResolver::RCODE_SERVFAIL,
                              "Name longer than 255 octets");

        // OPT outside of the additional section
        MessageBuilder misplacedOpt(1, 0x8180, name, DnsResolver::TYPE_A);
        misplacedOpt.Add(0, {0}, DnsResolver::TYPE_OPT, 0, {}, 1232);
        NS_TEST_EXPECT_MSG_EQ(decode(misplacedOpt).rcode,
                              DnsResolver::RCODE_SERVFAIL,
                              "OPT record in the answer section accepted");

        // errors without question: only FORMERR and NOTIMP, which reject EDNS(0)
        for (uint8_t rcode : {DnsResolver::RCODE_FORMERR,
                              DnsResolver::RCODE_NOTIMP,
                              DnsResolver::RCODE_SERVFAIL,
                              DnsResolver::RCODE_REFUSED})
        {
            std::vector<uint8_t>
                error{0, 1, 0x81, static_cast<uint8_t>(0x80 | rcode), 0, 0, 0, 0, 0, 0, 0, 0};
            response = DnsResolver::DecodeResponse(error, name, DnsResolver::TYPE_A);
            const bool ednsRejection =
                rcode == DnsResolver::RCODE_FORMERR || rcode == DnsResolver::RCODE_NOTIMP;
            NS_TEST_EXPECT_MSG_EQ(response.valid,
                                  ednsRejection,
                                  "Wrong validity of an error without question");
            NS_TEST_EXPECT_MSG_EQ(response.hasQuestion, false, "Question found");
            NS_TEST_EXPECT_MSG_EQ(response.rcode, rcode, "Wrong RCODE");
        }
        auto answerWithoutQuestion = MessageBuilder(1, 0x8180, name, DnsResolver::TYPE_A).Get();
        answerWithoutQuestion[5] = 0;
        NS_TEST_EXPECT_MSG_EQ(
            DnsResolver::DecodeResponse(answerWithoutQuestion, name, DnsResolver::TYPE_A).valid,
            false,
            "Answer without question accepted");

        // a label containing a dot
        auto dotted = MessageBuilder(1, 0x8180, "x.org", DnsResolver::TYPE_A).Get();
        std::vector<uint8_t> dottedQuestion{9,
                                            'w',
                                            'w',
                                            'w',
                                            '.',
                                            'n',
                                            's',
                                            'n',
                                            'a',
                                            'm',
                                            3,
                                            'o',
                                            'r',
                                            'g',
                                            0,
                                            0,
                                            1,
                                            0,
                                            1};
        dotted.resize(12);
        dotted.insert(dotted.end(), dottedQuestion.begin(), dottedQuestion.end());
        NS_TEST_EXPECT_MSG_EQ(DnsResolver::DecodeResponse(dotted, name, DnsResolver::TYPE_A).valid,
                              false,
                              "Label containing a dot confused with two labels");

        // CNAME loop
        MessageBuilder cnameLoop(1, 0x8180, name, DnsResolver::TYPE_A);
        cnameLoop.Add(0, qname, DnsResolver::TYPE_CNAME, 60, EncodeName(name));
        NS_TEST_EXPECT_MSG_EQ(decode(cnameLoop).rcode,
                              DnsResolver::RCODE_SERVFAIL,
                              "CNAME loop not detected");

        // negative TTL: SOA of the zone, bounded by the alias
        MessageBuilder nodata(1, 0x8180, name, DnsResolver::TYPE_A);
        nodata.Add(0, qname, DnsResolver::TYPE_CNAME, 20, EncodeName("web.nsnam.org"));
        nodata.Add(1, EncodeName("other.org"), DnsResolver::TYPE_SOA, 300, SoaData(5));
        nodata.Add(1, zone, DnsResolver::TYPE_SOA, 300, SoaData(30));
        nodata.Add(1, zone, DnsResolver::TYPE_SOA, 300, SoaData(10));
        response = decode(nodata);
        NS_TEST_EXPECT_MSG_EQ(response.hasSoa, true, "SOA of the zone not found");
        NS_TEST_EXPECT_MSG_EQ(response.ttl, 20, "Negative TTL not bounded by the alias");
        MessageBuilder unrelated(1, 0x8180, name, DnsResolver::TYPE_A);
        unrelated.Add(1, EncodeName("other.org"), DnsResolver::TYPE_SOA, 300, SoaData(5));
        NS_TEST_EXPECT_MSG_EQ(decode(unrelated).hasSoa, false, "SOA of another zone accepted");
        // the zone is compared label by label: x\.nsnam.org is not in nsnam.org
        MessageBuilder escaped(1, 0x8180, name, DnsResolver::TYPE_A);
        escaped.Add(0,
                    qname,
                    DnsResolver::TYPE_CNAME,
                    20,
                    {7, 'x', '.', 'n', 's', 'n', 'a', 'm', 3, 'o', 'r', 'g', 0});
        escaped.Add(1, zone, DnsResolver::TYPE_SOA, 300, SoaData(30));
        response = decode(escaped);
        NS_TEST_EXPECT_MSG_EQ(response.canonicalName, "x\\.nsnam.org", "Dot not escaped");
        NS_TEST_EXPECT_MSG_EQ(response.hasSoa, false, "SOA of the parent of a label accepted");

        // encoding: host names and EDNS(0) payload size
        for (const auto& invalid : {"a b.org", "-bad.org", "bad-.org", "host..", "", "."})
        {
            NS_TEST_EXPECT_MSG_EQ(DnsResolver::IsValidHostName(invalid),
                                  false,
                                  "Invalid host name " << invalid << " accepted");
        }
        NS_TEST_EXPECT_MSG_EQ(DnsResolver::IsValidHostName("Www.ns-3.org."), true, "Valid name");
        NS_TEST_EXPECT_MSG_EQ(DnsResolver::IsValidHostName("192.0.2.1"), false, "Address accepted");
        NS_TEST_EXPECT_MSG_EQ(DnsResolver::IsValidHostName("host.123"),
                              false,
                              "All-numeric top-level label accepted");
        auto query = DnsResolver::EncodeQuery(1, name, DnsResolver::TYPE_A, 100);
        NS_TEST_EXPECT_MSG_EQ((query[query.size() - 8] << 8 | query[query.size() - 7]),
                              512,
                              "EDNS(0) payload size below 512");
    }
};

/**
 * @ingroup dns-resolver-test
 * @brief A and AAAA records, aliases, non-existent and invalid names, EDNS(0).
 */
class DnsResolverRecordsTestCase : public DnsResolverTestBase
{
  public:
    DnsResolverRecordsTestCase()
        : DnsResolverTestBase("A and AAAA records")
    {
    }

  private:
    void DoRun() override
    {
        Setup();
        ScheduleResolve(Seconds(1), "a", "www.nsnam.org");
        ScheduleResolve(Seconds(2), "aaaa", "WWW.nsnam.org.", DnsResolver::IPV6);
        ScheduleResolve(Seconds(3), "any", "www.nsnam.org", DnsResolver::ANY);
        ScheduleResolve(Seconds(4), "unknown", "unknown.nsnam.org", DnsResolver::ANY);
        ScheduleResolve(Seconds(5), "invalid", "bad..org");
        ScheduleResolve(Seconds(5), "double dot", "www.nsnam.org..");
        Simulator::Stop(Seconds(10));
        Simulator::Run();

        NS_TEST_EXPECT_MSG_EQ(Result("a"), "1.2.3.4 5.6.7.8", "Wrong A records");
        NS_TEST_EXPECT_MSG_EQ(Result("aaaa"), "2001:db8::80", "Wrong AAAA records");
        NS_TEST_EXPECT_MSG_EQ(Result("any"),
                              "2001:db8::80 1.2.3.4 5.6.7.8",
                              "Wrong addresses of both families");
        NS_TEST_EXPECT_MSG_EQ(m_results.contains("unknown"), true, "No result");
        NS_TEST_EXPECT_MSG_EQ(Result("unknown"), "", "Non-existent name resolved");
        // NXDOMAIN applies to both types: the AAAA query completes the A query
        NS_TEST_EXPECT_MSG_LT_OR_EQ(m_server->m_queries["unknown.nsnam.org/1/udp"] +
                                        m_server->m_queries["unknown.nsnam.org/28/udp"],
                                    2,
                                    "Too many queries");
        for (const auto& key : {"invalid", "double dot"})
        {
            NS_TEST_EXPECT_MSG_EQ(m_results.contains(key), true, "No result");
            NS_TEST_EXPECT_MSG_EQ(Result(key), "", "Invalid name resolved");
        }
        NS_TEST_EXPECT_MSG_GT(m_server->m_ednsQueries, 0, "No EDNS(0) in the queries");
        Teardown();
    }
};

/**
 * @ingroup dns-resolver-test
 * @brief Responses to other questions, records of other names, referrals and alias loops.
 */
class DnsResolverValidationTestCase : public DnsResolverTestBase
{
  public:
    DnsResolverValidationTestCase()
        : DnsResolverTestBase("Validation of the responses")
    {
    }

  private:
    void DoRun() override
    {
        Setup();
        ScheduleResolve(Seconds(1), "spoof", "spoof.nsnam.org");
        ScheduleResolve(Seconds(10), "stray", "stray.nsnam.org");
        ScheduleResolve(Seconds(11), "stray again", "stray.nsnam.org");
        ScheduleResolve(Seconds(20), "referral", "referral.nsnam.org");
        ScheduleResolve(Seconds(30), "loop", "loop.nsnam.org");
        ScheduleResolve(Seconds(35), "chain", "chain.nsnam.org");
        ScheduleResolve(Seconds(36), "chain again", "chain.nsnam.org");
        ScheduleResolve(Seconds(36), "literal", "192.0.2.7", DnsResolver::ANY);
        ScheduleResolve(Seconds(36), "literal dot", "192.0.2.7.");
        ScheduleResolve(Seconds(36), "leading zero", "010.0.2.7");
        Simulator::Stop(Seconds(40));
        Simulator::Run();

        // the responses to another question are ignored: the query times out
        NS_TEST_EXPECT_MSG_EQ(Result("spoof"), "", "Response to another question accepted");
        NS_TEST_EXPECT_MSG_EQ(m_server->m_queries["spoof.nsnam.org/1/udp"], 2, "No retransmission");
        NS_TEST_EXPECT_MSG_EQ_TOL(m_times["spoof"],
                                  Seconds(4),
                                  MilliSeconds(50),
                                  "Not completed by the timeouts");
        // a record of another name: no address, and nothing cached without SOA
        NS_TEST_EXPECT_MSG_EQ(Result("stray"), "", "Record of another name accepted");
        NS_TEST_EXPECT_MSG_EQ(m_server->m_queries["stray.nsnam.org/1/udp"], 2, "Cached");
        // a referral is not an answer: the query is retransmitted
        NS_TEST_EXPECT_MSG_EQ(Result("referral"), "", "Referral resolved");
        NS_TEST_EXPECT_MSG_EQ(m_server->m_queries["referral.nsnam.org/1/udp"],
                              2,
                              "Referral not retransmitted");
        // an alias loop is a failure of the server: the query is retransmitted
        NS_TEST_EXPECT_MSG_EQ(Result("loop"), "", "Alias loop resolved");
        NS_TEST_EXPECT_MSG_EQ(m_server->m_queries["loop.nsnam.org/1/udp"],
                              2,
                              "Alias loop not retransmitted");
        // a chain ending without the records of its target: the target is queried, and the
        // answer is cached for the alias
        NS_TEST_EXPECT_MSG_EQ(Result("chain"), "4.4.4.4", "Target of the alias not queried");
        NS_TEST_EXPECT_MSG_EQ(Result("chain again"), "4.4.4.4", "Wrong cached answer");
        NS_TEST_EXPECT_MSG_EQ(m_server->m_queries["chain.nsnam.org/1/udp"], 1, "Not cached");
        NS_TEST_EXPECT_MSG_EQ(m_server->m_queries["target.nsnam.org/1/udp"], 1, "Not cached");
        // an address is returned as is
        NS_TEST_EXPECT_MSG_EQ(Result("literal"), "192.0.2.7", "Address not returned as is");
        NS_TEST_EXPECT_MSG_EQ(Result("literal dot"), "192.0.2.7", "Address with a final dot");
        NS_TEST_EXPECT_MSG_EQ(m_results.contains("leading zero"), true, "No result");
        NS_TEST_EXPECT_MSG_EQ(Result("leading zero"), "", "Address with a leading zero accepted");
        Teardown();
    }
};

/**
 * @ingroup dns-resolver-test
 * @brief Retry without EDNS(0) for a server not supporting it.
 */
class DnsResolverEdnsTestCase : public DnsResolverTestBase
{
  public:
    DnsResolverEdnsTestCase()
        : DnsResolverTestBase("Fallback without EDNS(0)")
    {
    }

  private:
    void DoRun() override
    {
        Setup();
        m_resolver->SetAttribute("NoEdnsTime", TimeValue(Seconds(5)));
        ScheduleResolve(Seconds(1), "noedns", "noedns.nsnam.org");
        Simulator::Schedule(Seconds(2), [this]() { m_ednsQueries = m_server->m_ednsQueries; });
        ScheduleResolve(Seconds(3), "www", "www.nsnam.org");
        Simulator::Schedule(Seconds(4), [this]() { m_ednsQueriesAfter = m_server->m_ednsQueries; });
        // after NoEdnsTime, EDNS(0) is tried again
        ScheduleResolve(Seconds(10), "ttlmsb", "ttlmsb.nsnam.org");
        Simulator::Stop(Seconds(20));
        Simulator::Run();

        NS_TEST_EXPECT_MSG_EQ(Result("noedns"), "9.9.9.9", "No fallback without EDNS(0)");
        NS_TEST_EXPECT_MSG_EQ(m_server->m_queries["noedns.nsnam.org/1/udp"], 2, "Wrong queries");
        NS_TEST_EXPECT_MSG_EQ_TOL(m_times["noedns"],
                                  Seconds(1),
                                  MilliSeconds(50),
                                  "Fallback delayed");
        NS_TEST_EXPECT_MSG_EQ(Result("www"), "1.2.3.4 5.6.7.8", "Wrong A records");
        NS_TEST_EXPECT_MSG_EQ(m_ednsQueriesAfter,
                              m_ednsQueries,
                              "EDNS(0) used again with the server");
        NS_TEST_EXPECT_MSG_EQ(m_server->m_ednsQueries,
                              m_ednsQueries + 1,
                              "EDNS(0) not tried again after NoEdnsTime");
        Teardown();
    }

    uint32_t m_ednsQueries{0};      ///< number of queries with EDNS(0) after the fallback
    uint32_t m_ednsQueriesAfter{0}; ///< number of queries with EDNS(0) after another query
};

/**
 * @ingroup dns-resolver-test
 * @brief Failover to the next server, late responses, and changes of servers.
 */
class DnsResolverFailoverTestCase : public DnsResolverTestBase
{
  public:
    DnsResolverFailoverTestCase()
        : DnsResolverTestBase("Failover to the next server")
    {
    }

  private:
    void DoRun() override
    {
        Setup();
        Simulator::Schedule(Seconds(1),
                            [this]() { m_resolver->SetServers({DEAD_SERVER, SERVER}); });
        ScheduleResolve(Seconds(1), "timeout", "www.nsnam.org");
        Simulator::Schedule(Seconds(10), [this]() {
            m_resolver->FlushCache();
            m_resolver->SetServers({SERVFAIL_SERVER, SERVER});
        });
        ScheduleResolve(Seconds(10), "error", "www.nsnam.org");
        Simulator::Schedule(Seconds(20), [this]() { m_resolver->SetServers({DEAD_SERVER}); });
        ScheduleResolve(Seconds(20), "failure", "big.nsnam.org");
        // the response of the first server, after its timeout, is accepted
        Simulator::Schedule(Seconds(30),
                            [this]() { m_resolver->SetServers({SERVER, DEAD_SERVER}); });
        ScheduleResolve(Seconds(30), "late", "late.nsnam.org");
        // removing the servers fails the queries in progress
        Simulator::Schedule(Seconds(40), [this]() { m_resolver->SetServers({DEAD_SERVER}); });
        ScheduleResolve(Seconds(40), "removed", "www.nsnam.org", DnsResolver::IPV6);
        Simulator::Schedule(MilliSeconds(40500), [this]() { m_resolver->SetServers({}); });
        ScheduleResolve(MilliSeconds(40500), "no server", "ttlmsb.nsnam.org");
        // a TCP connection closed by the server is retried at once
        Simulator::Schedule(Seconds(50), [this]() { m_resolver->SetServers({SERVER}); });
        ScheduleResolve(Seconds(50), "tcpclose", "tcpclose.nsnam.org");
        // a truncated response over TCP is a failure of the server
        ScheduleResolve(Seconds(55), "tcptrunc", "tcptrunc.nsnam.org");
        // the late error of the previous server does not affect the current attempt
        Simulator::Schedule(Seconds(60),
                            [this]() { m_resolver->SetServers({SERVER, DEAD_SERVER}); });
        ScheduleResolve(Seconds(60), "lateerror", "lateerror.nsnam.org");
        // a server without route fails at once
        Simulator::Schedule(Seconds(70), [this]() {
            m_resolver->FlushCache();
            m_resolver->SetServers({Ipv4Address("192.0.2.1"), SERVER});
        });
        ScheduleResolve(Seconds(70), "unroutable", "www.nsnam.org");
        Simulator::Stop(Seconds(80));
        Simulator::Run();

        NS_TEST_EXPECT_MSG_EQ(Result("timeout"), "1.2.3.4 5.6.7.8", "No failover after timeout");
        NS_TEST_EXPECT_MSG_EQ_TOL(m_times["timeout"],
                                  Seconds(2),
                                  MilliSeconds(50),
                                  "Failover not at the timeout");
        NS_TEST_EXPECT_MSG_EQ(Result("error"), "1.2.3.4 5.6.7.8", "No failover after SERVFAIL");
        NS_TEST_EXPECT_MSG_EQ_TOL(m_times["error"],
                                  Seconds(10),
                                  MilliSeconds(50),
                                  "Failover after SERVFAIL delayed");
        NS_TEST_EXPECT_MSG_EQ(m_results.contains("failure"), true, "No failure reported");
        NS_TEST_EXPECT_MSG_EQ(Result("failure"), "", "Unanswered query resolved");
        // the timeout doubles after a round of the servers: 1 s, then 2 s
        NS_TEST_EXPECT_MSG_EQ_TOL(m_times["failure"],
                                  Seconds(23),
                                  MilliSeconds(50),
                                  "Wrong time of the failure");
        NS_TEST_EXPECT_MSG_EQ(Result("late"), "9.9.9.9", "Late response discarded");
        NS_TEST_EXPECT_MSG_EQ_TOL(m_times["late"],
                                  MilliSeconds(31500),
                                  MilliSeconds(50),
                                  "Wrong time of the late response");
        NS_TEST_EXPECT_MSG_EQ(Result("removed"), "", "Query resolved without server");
        NS_TEST_EXPECT_MSG_EQ(m_times["removed"], MilliSeconds(40500), "Query not failed");
        NS_TEST_EXPECT_MSG_EQ(m_results.contains("no server"), true, "No result without server");
        NS_TEST_EXPECT_MSG_EQ(Result("no server"), "", "Resolved without server");
        NS_TEST_EXPECT_MSG_EQ(Result("tcptrunc"), "", "Truncated TCP response resolved");
        NS_TEST_EXPECT_MSG_EQ(m_server->m_queries["tcptrunc.nsnam.org/1/tcp"],
                              2,
                              "Truncated TCP response not retried");
        NS_TEST_EXPECT_MSG_EQ(Result("lateerror"), "", "Late error resolved");
        NS_TEST_EXPECT_MSG_EQ_TOL(m_times["lateerror"],
                                  Seconds(62),
                                  MilliSeconds(50),
                                  "Late error of the previous server applied");
        NS_TEST_EXPECT_MSG_EQ(Result("unroutable"), "1.2.3.4 5.6.7.8", "No failover");
        NS_TEST_EXPECT_MSG_EQ_TOL(m_times["unroutable"],
                                  Seconds(70),
                                  MilliSeconds(50),
                                  "Unroutable server not abandoned at once");
        NS_TEST_EXPECT_MSG_EQ(Result("tcpclose"), "", "Closed connection resolved");
        NS_TEST_EXPECT_MSG_EQ(m_server->m_queries["tcpclose.nsnam.org/1/tcp"],
                              2,
                              "Closed connection not retried");
        NS_TEST_EXPECT_MSG_LT(m_times["tcpclose"], Seconds(51), "Closed connection detected late");
        Teardown();
    }
};

/**
 * @ingroup dns-resolver-test
 * @brief Caching of the answers for their TTL, and of the negative answers.
 */
class DnsResolverCacheTestCase : public DnsResolverTestBase
{
  public:
    DnsResolverCacheTestCase()
        : DnsResolverTestBase("Cache of the answers")
    {
    }

  private:
    void DoRun() override
    {
        Setup();
        const std::string www = "www.nsnam.org/1/udp";
        const std::string unknown = "unknown.nsnam.org/1/udp";
        ScheduleResolve(Seconds(1), "www1", "www.nsnam.org");
        ScheduleResolve(Seconds(1), "unknown1", "unknown.nsnam.org");
        ScheduleResolve(Seconds(1), "ttlmsb1", "ttlmsb.nsnam.org");
        // within the TTL of 60 s, and the negative TTL of 30 s, which applies to all the types
        ScheduleResolve(Seconds(20), "www2", "www.nsnam.org");
        ScheduleResolve(Seconds(20), "unknown2", "unknown.nsnam.org", DnsResolver::ANY);
        ScheduleResolve(Seconds(20), "ttlmsb2", "ttlmsb.nsnam.org");
        ScheduleCount(Seconds(21), "www2", www);
        ScheduleCount(Seconds(21), "unknown2", unknown);
        ScheduleCount(Seconds(21), "unknown2 AAAA", "unknown.nsnam.org/28/udp");
        ScheduleCount(Seconds(21), "ttlmsb2", "ttlmsb.nsnam.org/1/udp");
        // after the negative TTL
        ScheduleResolve(Seconds(40), "unknown3", "unknown.nsnam.org");
        ScheduleCount(Seconds(41), "unknown3", unknown);
        // after the TTL
        ScheduleResolve(Seconds(62), "www3", "www.nsnam.org");
        ScheduleCount(Seconds(63), "www3", www);
        // after flushing the cache
        Simulator::Schedule(Seconds(64), [this]() { m_resolver->FlushCache(); });
        ScheduleResolve(Seconds(64), "www4", "www.nsnam.org");
        ScheduleCount(Seconds(65), "www4", www);
        // disposing of the resolver before reporting a cached answer
        ScheduleResolve(Seconds(66), "disposed", "www.nsnam.org");
        Simulator::Schedule(Seconds(66), [this]() { m_resolver->Dispose(); });
        Simulator::Stop(Seconds(70));
        Simulator::Run();

        NS_TEST_EXPECT_MSG_EQ(m_counts["www2"], 1, "Answer not cached");
        NS_TEST_EXPECT_MSG_EQ(Result("www2"), "1.2.3.4 5.6.7.8", "Wrong cached answer");
        NS_TEST_EXPECT_MSG_EQ(m_times["www2"], Seconds(20), "Cached answer delayed");
        NS_TEST_EXPECT_MSG_EQ(m_counts["unknown2"], 1, "Negative answer not cached");
        NS_TEST_EXPECT_MSG_EQ(m_counts["unknown2 AAAA"], 0, "NXDOMAIN not cached for AAAA");
        NS_TEST_EXPECT_MSG_EQ(Result("unknown2"), "", "Wrong cached negative answer");
        NS_TEST_EXPECT_MSG_EQ(m_counts["ttlmsb2"], 2, "Answer with a TTL of 0 cached");
        NS_TEST_EXPECT_MSG_EQ(m_counts["unknown3"], 2, "Negative answer cached beyond its TTL");
        NS_TEST_EXPECT_MSG_EQ(m_counts["www3"], 2, "Answer cached beyond its TTL");
        NS_TEST_EXPECT_MSG_EQ(m_counts["www4"], 3, "Cache not flushed");
        NS_TEST_EXPECT_MSG_EQ(m_results.contains("disposed"), false, "Reported after disposal");
        Teardown();
    }
};

/**
 * @ingroup dns-resolver-test
 * @brief Limits of the cache: maximum negative TTL, and negative answers of the other type.
 */
class DnsResolverCacheLimitsTestCase : public DnsResolverTestBase
{
  public:
    DnsResolverCacheLimitsTestCase()
        : DnsResolverTestBase("Limits of the cache")
    {
    }

  private:
    void DoRun() override
    {
        Setup();
        m_resolver->SetAttribute("MaxNegativeTtl", TimeValue(Seconds(10)));
        const std::string unknown = "unknown.nsnam.org/1/udp";
        ScheduleResolve(Seconds(1), "unknown1", "unknown.nsnam.org");
        ScheduleResolve(Seconds(5), "unknown2", "unknown.nsnam.org");
        ScheduleCount(Seconds(6), "unknown2", unknown);
        ScheduleResolve(Seconds(12), "unknown3", "unknown.nsnam.org");
        ScheduleCount(Seconds(13), "unknown3", unknown);
        // NXDOMAIN for A does not replace the cached AAAA records
        ScheduleResolve(Seconds(20), "v6only AAAA", "v6only.nsnam.org", DnsResolver::IPV6);
        ScheduleResolve(Seconds(21), "v6only A", "v6only.nsnam.org");
        ScheduleResolve(Seconds(22), "v6only AAAA again", "v6only.nsnam.org", DnsResolver::IPV6);
        Simulator::Stop(Seconds(30));
        Simulator::Run();

        NS_TEST_EXPECT_MSG_EQ(m_counts["unknown2"], 1, "Negative answer not cached");
        NS_TEST_EXPECT_MSG_EQ(m_counts["unknown3"], 2, "Negative answer cached beyond the cap");
        NS_TEST_EXPECT_MSG_EQ(Result("v6only AAAA"), "2001:db8::66", "Wrong AAAA records");
        NS_TEST_EXPECT_MSG_EQ(Result("v6only A"), "", "Wrong A records");
        NS_TEST_EXPECT_MSG_EQ(Result("v6only AAAA again"),
                              "2001:db8::66",
                              "AAAA records replaced by NXDOMAIN");
        NS_TEST_EXPECT_MSG_EQ(m_server->m_queries["v6only.nsnam.org/28/udp"], 1, "Not cached");
        Teardown();
    }
};

/**
 * @ingroup dns-resolver-test
 * @brief Retry over TCP of a truncated response.
 */
class DnsResolverTcpTestCase : public DnsResolverTestBase
{
  public:
    DnsResolverTcpTestCase()
        : DnsResolverTestBase("TCP fallback for truncated responses")
    {
    }

  private:
    void DoRun() override
    {
        Setup();
        ScheduleResolve(Seconds(1), "big", "big.nsnam.org");
        Simulator::Stop(Seconds(10));
        Simulator::Run();

        NS_TEST_EXPECT_MSG_EQ(m_server->m_queries["big.nsnam.org/1/udp"], 1, "No UDP query");
        NS_TEST_EXPECT_MSG_EQ(m_server->m_queries["big.nsnam.org/1/tcp"], 1, "No TCP query");
        NS_TEST_EXPECT_MSG_EQ(m_results["big"].size(), 20, "Wrong number of addresses");
        Teardown();
    }
};

/**
 * @ingroup dns-resolver-test
 * @brief Server reached over IPv6.
 */
class DnsResolverIpv6ServerTestCase : public DnsResolverTestBase
{
  public:
    DnsResolverIpv6ServerTestCase()
        : DnsResolverTestBase("Server reached over IPv6")
    {
    }

  private:
    void DoRun() override
    {
        Setup();
        Simulator::Schedule(Seconds(1), [this]() { m_resolver->SetServers({m_server6}); });
        ScheduleResolve(Seconds(1), "www", "www.nsnam.org", DnsResolver::ANY);
        Simulator::Stop(Seconds(10));
        Simulator::Run();

        NS_TEST_EXPECT_MSG_EQ(m_server->m_queries["www.nsnam.org/28/udp6"], 1, "No IPv6 query");
        NS_TEST_EXPECT_MSG_EQ(Result("www"), "2001:db8::80 1.2.3.4 5.6.7.8", "Wrong addresses");
        Teardown();
    }
};

/**
 * @ingroup dns-resolver-test
 * @brief Cached failures, changes of servers, late responses, aliases to non-host names, EDNS(0)
 * without memory, and the cache size and switch.
 */
class DnsResolverRobustnessTestCase : public DnsResolverTestBase
{
  public:
    DnsResolverRobustnessTestCase()
        : DnsResolverTestBase("Robustness")
    {
    }

  private:
    void DoRun() override
    {
        Setup();
        m_resolver->SetAttribute("Retransmissions", UintegerValue(5));
        const std::string www = "www.nsnam.org/1/udp";
        // at most three attempts to a server: 1 s, 2 s and 4 s
        Simulator::Schedule(Seconds(1), [this]() { m_resolver->SetServers({DEAD_SERVER}); });
        ScheduleResolve(Seconds(1), "dead", "www.nsnam.org");
        // the failure is cached until the servers change
        ScheduleResolve(Seconds(10), "cached failure", "www.nsnam.org");
        Simulator::Schedule(Seconds(11), [this]() { m_resolver->SetServers({SERVER}); });
        ScheduleResolve(Seconds(11), "new servers", "www.nsnam.org");
        // removing the current server: the next server is still tried
        Simulator::Schedule(Seconds(20), [this]() {
            m_resolver->FlushCache();
            m_resolver->SetServers({DEAD_SERVER, Ipv4Address("10.0.0.98"), SERVER});
        });
        ScheduleResolve(Seconds(20), "removed current", "www.nsnam.org");
        Simulator::Schedule(MilliSeconds(20500), [this]() {
            m_resolver->SetServers({Ipv4Address("10.0.0.98"), SERVER});
        });
        // a late response without record from the previous server does not end the query
        Simulator::Schedule(Seconds(30), [this]() { m_resolver->SetServers({SERVER, m_server6}); });
        ScheduleResolve(Seconds(30), "late empty", "lateempty.nsnam.org");
        // without callback
        Simulator::Schedule(Seconds(40), [this]() {
            m_resolver->SetServers({SERVER});
            m_resolver->Resolve("chain.nsnam.org", DnsResolver::ResolveCallback());
            m_resolver->Resolve("www.nsnam.org", DnsResolver::ResolveCallback());
        });
        // the cache is used without server, and not when disabled
        Simulator::Schedule(Seconds(45), [this]() { m_resolver->SetServers({}); });
        ScheduleResolve(Seconds(45), "cache without server", "www.nsnam.org");
        Simulator::Schedule(Seconds(46), [this]() {
            m_resolver->SetAttribute("CacheEnabled", BooleanValue(false));
        });
        ScheduleResolve(Seconds(46), "cache disabled", "www.nsnam.org");
        // a fallback without EDNS(0) for the query, even if not remembered for the server
        Simulator::Schedule(Seconds(50), [this]() {
            m_resolver->SetAttribute("CacheEnabled", BooleanValue(true));
            m_resolver->SetAttribute("NoEdnsTime", TimeValue(Seconds(0)));
            m_resolver->SetServers({SERVER});
        });
        ScheduleResolve(Seconds(50), "noedns", "noedns.nsnam.org");
        // an alias to a name which is not a host name
        ScheduleResolve(Seconds(52), "underscore", "underscore.nsnam.org");
        // a full cache removes the answer closest to its expiry
        Simulator::Schedule(Seconds(60), [this]() {
            m_resolver->FlushCache();
            m_resolver->SetAttribute("MaxCacheEntries", UintegerValue(1));
        });
        ScheduleResolve(Seconds(60), "evicted", "www.nsnam.org");
        ScheduleResolve(Seconds(61), "evicting", "v6only.nsnam.org", DnsResolver::IPV6);
        ScheduleCount(MilliSeconds(61500), "before eviction", www);
        ScheduleResolve(Seconds(62), "after eviction", "www.nsnam.org");
        ScheduleCount(MilliSeconds(62500), "after eviction", www);
        // removing the current server before its rejection of EDNS(0), or its incomplete chain:
        // the next server is queried
        Simulator::Schedule(Seconds(70), [this]() {
            m_resolver->SetAttribute("MaxCacheEntries", UintegerValue(10000));
            m_resolver->SetAttribute("Timeout", TimeValue(Seconds(3)));
            m_resolver->SetServers({SERVER, m_server6});
        });
        ScheduleResolve(Seconds(70), "latenoedns", "latenoedns.nsnam.org");
        Simulator::Schedule(MilliSeconds(70500), [this]() { m_resolver->SetServers({m_server6}); });
        Simulator::Schedule(Seconds(80), [this]() { m_resolver->SetServers({SERVER, m_server6}); });
        ScheduleResolve(Seconds(80), "latechain", "latechain.nsnam.org");
        Simulator::Schedule(MilliSeconds(80500), [this]() { m_resolver->SetServers({m_server6}); });
        // concurrent resolutions of a name share their query
        Simulator::Schedule(Seconds(90), [this]() { m_resolver->SetServers({SERVER}); });
        ScheduleResolve(Seconds(90), "shared A", "late.nsnam.org");
        ScheduleResolve(Seconds(90), "shared ANY", "late.nsnam.org", DnsResolver::ANY);
        // the caching of consecutive failures doubles: 5 s, then 10 s
        Simulator::Schedule(Seconds(100), [this]() {
            m_resolver->SetAttribute("Timeout", TimeValue(Seconds(1)));
            m_resolver->FlushCache();
            m_resolver->SetServers({DEAD_SERVER});
        });
        ScheduleResolve(Seconds(100), "backoff1", "www.nsnam.org");
        ScheduleResolve(Seconds(113), "backoff2", "www.nsnam.org");
        ScheduleResolve(Seconds(128), "backoff3", "www.nsnam.org");
        Simulator::Stop(Seconds(140));
        Simulator::Run();

        NS_TEST_EXPECT_MSG_EQ(Result("dead"), "", "Resolved without answer");
        NS_TEST_EXPECT_MSG_EQ_TOL(m_times["dead"],
                                  Seconds(8),
                                  MilliSeconds(50),
                                  "More than three attempts to the server");
        NS_TEST_EXPECT_MSG_EQ(m_results.contains("cached failure"), true, "No result");
        NS_TEST_EXPECT_MSG_EQ(m_times["cached failure"], Seconds(10), "Failure not cached");
        NS_TEST_EXPECT_MSG_EQ(Result("new servers"), "1.2.3.4 5.6.7.8", "Failure kept");
        NS_TEST_EXPECT_MSG_EQ(Result("removed current"), "1.2.3.4 5.6.7.8", "Not resolved");
        NS_TEST_EXPECT_MSG_EQ_TOL(m_times["removed current"],
                                  Seconds(22),
                                  MilliSeconds(50),
                                  "Next server skipped");
        NS_TEST_EXPECT_MSG_EQ(Result("late empty"), "7.7.7.7", "Ended by the previous server");
        NS_TEST_EXPECT_MSG_EQ_TOL(m_times["late empty"],
                                  MilliSeconds(32500),
                                  MilliSeconds(50),
                                  "Wrong time of the late response");
        NS_TEST_EXPECT_MSG_EQ(Result("cache without server"),
                              "1.2.3.4 5.6.7.8",
                              "Cache not used without server");
        NS_TEST_EXPECT_MSG_EQ(m_results.contains("cache disabled"), true, "No result");
        NS_TEST_EXPECT_MSG_EQ(Result("cache disabled"), "", "Disabled cache used");
        NS_TEST_EXPECT_MSG_EQ(Result("noedns"), "9.9.9.9", "No fallback without EDNS(0)");
        NS_TEST_EXPECT_MSG_EQ(m_server->m_queries["noedns.nsnam.org/1/udp"], 2, "Wrong queries");
        NS_TEST_EXPECT_MSG_EQ(Result("underscore"), "3.3.3.3", "Target of the alias not queried");
        NS_TEST_EXPECT_MSG_EQ(Result("evicting"), "2001:db8::66", "Wrong AAAA records");
        NS_TEST_EXPECT_MSG_EQ(m_counts["after eviction"],
                              m_counts["before eviction"] + 1,
                              "Answer not removed from the full cache");
        NS_TEST_EXPECT_MSG_EQ(Result("latenoedns"), "9.9.9.9", "Not resolved");
        NS_TEST_EXPECT_MSG_EQ(m_server->m_queries["latenoedns.nsnam.org/1/udp"],
                              1,
                              "Removed server queried again");
        NS_TEST_EXPECT_MSG_EQ(m_server->m_queries["latenoedns.nsnam.org/1/udp6"],
                              1,
                              "Next server not queried");
        NS_TEST_EXPECT_MSG_EQ(Result("latechain"), "4.4.4.4", "Not resolved");
        NS_TEST_EXPECT_MSG_EQ(m_server->m_queries["target.nsnam.org/1/udp6"],
                              1,
                              "Target queried from the removed server");
        NS_TEST_EXPECT_MSG_EQ(Result("shared A"), "9.9.9.9", "Wrong addresses");
        NS_TEST_EXPECT_MSG_EQ(Result("shared ANY"), "9.9.9.9", "Wrong addresses");
        NS_TEST_EXPECT_MSG_EQ(m_server->m_queries["late.nsnam.org/1/udp"], 1, "Query not shared");
        NS_TEST_EXPECT_MSG_EQ_TOL(m_times["backoff1"], Seconds(107), MilliSeconds(50), "Failed");
        NS_TEST_EXPECT_MSG_EQ_TOL(m_times["backoff2"], Seconds(120), MilliSeconds(50), "Failed");
        NS_TEST_EXPECT_MSG_EQ(m_times["backoff3"], Seconds(128), "Consecutive failure not cached");
        // a random source port in the dynamic range for each query
        NS_TEST_EXPECT_MSG_GT(m_server->m_ports.size(), 10, "Source ports reused");
        NS_TEST_EXPECT_MSG_GT_OR_EQ(*m_server->m_ports.begin(),
                                    49152,
                                    "Source port out of the dynamic range");
        Teardown();
    }
};

/**
 * @ingroup dns-resolver-test
 * @brief DNS resolver test suite.
 */
class DnsResolverTestSuite : public TestSuite
{
  public:
    DnsResolverTestSuite()
        : TestSuite("dns-resolver", Type::UNIT)
    {
        AddTestCase(new DnsResolverDecodingTestCase, TestCase::Duration::QUICK);
        AddTestCase(new DnsResolverRecordsTestCase, TestCase::Duration::QUICK);
        AddTestCase(new DnsResolverValidationTestCase, TestCase::Duration::QUICK);
        AddTestCase(new DnsResolverEdnsTestCase, TestCase::Duration::QUICK);
        AddTestCase(new DnsResolverFailoverTestCase, TestCase::Duration::QUICK);
        AddTestCase(new DnsResolverCacheTestCase, TestCase::Duration::QUICK);
        AddTestCase(new DnsResolverCacheLimitsTestCase, TestCase::Duration::QUICK);
        AddTestCase(new DnsResolverTcpTestCase, TestCase::Duration::QUICK);
        AddTestCase(new DnsResolverIpv6ServerTestCase, TestCase::Duration::QUICK);
        AddTestCase(new DnsResolverRobustnessTestCase, TestCase::Duration::QUICK);
    }
};

static DnsResolverTestSuite g_dnsResolverTestSuite; //!< Static variable for test initialization
