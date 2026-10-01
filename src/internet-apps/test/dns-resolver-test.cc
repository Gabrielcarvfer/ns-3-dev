/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/boolean.h"
#include "ns3/dns-header.h"
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
#include <optional>
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
constexpr uint16_t DNS_PORT{53};               ///< port of the server
constexpr uint32_t TCP_LENGTH_PREFIX_SIZE{2};  ///< size of the length prefix over TCP

/**
 * @ingroup dns-resolver-test
 * @brief Start a response to a question, with recursion desired and available.
 * @param id the identifier
 * @param name the name of the question
 * @param type the type of the question
 * @return the response
 */
DnsHeader
Response(uint16_t id, const std::string& name, uint16_t type)
{
    DnsHeader response;
    response.SetId(id);
    response.SetResponse(true);
    response.SetRecursionDesired(true);
    response.SetRecursionAvailable(true);
    response.SetQuestion(name, type);
    return response;
}

/**
 * @ingroup dns-resolver-test
 * @brief An A or AAAA record.
 * @param owner the owner name
 * @param ttl the TTL
 * @param address the address (an Ipv4Address or an Ipv6Address)
 * @return the record
 */
DnsResourceRecord
AddressRecord(const std::string& owner, uint32_t ttl, const Address& address)
{
    DnsResourceRecord record;
    record.name = owner;
    record.type = Ipv4Address::IsMatchingType(address) ? DnsHeader::TYPE_A : DnsHeader::TYPE_AAAA;
    record.ttl = ttl;
    record.address = address;
    return record;
}

/**
 * @ingroup dns-resolver-test
 * @brief A CNAME or NS record.
 * @param type the type (TYPE_CNAME or TYPE_NS)
 * @param owner the owner name
 * @param ttl the TTL
 * @param target the target name
 * @return the record
 */
DnsResourceRecord
NameRecord(uint16_t type, const std::string& owner, uint32_t ttl, const std::string& target)
{
    DnsResourceRecord record;
    record.name = owner;
    record.type = type;
    record.ttl = ttl;
    record.target = target;
    return record;
}

/**
 * @ingroup dns-resolver-test
 * @brief An SOA record.
 * @param zone the zone
 * @param ttl the TTL
 * @param minimum the MINIMUM field
 * @return the record
 */
DnsResourceRecord
SoaRecord(const std::string& zone, uint32_t ttl, uint32_t minimum)
{
    DnsResourceRecord record;
    record.name = zone;
    record.type = DnsHeader::TYPE_SOA;
    record.ttl = ttl;
    record.soa = {"ns.nsnam.org", "admin.nsnam.org", 1, 3600, 600, 86400, minimum};
    return record;
}

/**
 * @ingroup dns-resolver-test
 * @brief An OPT record.
 * @param udpPayloadSize the UDP payload size
 * @param ttl the TTL field (extended RCODE, version and flags)
 * @param owner the owner name
 * @return the record
 */
DnsResourceRecord
OptRecord(uint16_t udpPayloadSize, uint32_t ttl = 0, const std::string& owner = "")
{
    DnsResourceRecord record;
    record.name = owner;
    record.type = DnsHeader::TYPE_OPT;
    record.rclass = udpPayloadSize;
    record.ttl = ttl;
    return record;
}

/**
 * @ingroup dns-resolver-test
 * @brief Serialize a message.
 * @param message the message
 * @return the serialized message
 */
std::vector<uint8_t>
Serialize(const DnsHeader& message)
{
    Ptr<Packet> packet = Create<Packet>();
    packet->AddHeader(message);
    std::vector<uint8_t> data(packet->GetSize());
    packet->CopyData(data.data(), data.size());
    return data;
}

/**
 * @ingroup dns-resolver-test
 * @brief Deserialize a message.
 * @param data the serialized message
 * @param message the message
 * @return the number of bytes deserialized, 0 if the message is malformed
 */
uint32_t
Deserialize(const std::vector<uint8_t>& data, DnsHeader& message)
{
    return Create<Packet>(data.data(), data.size())->PeekHeader(message);
}

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
    for (size_t start = 0; start < name.size();)
    {
        size_t dot = name.find('.', start);
        dot = (dot == std::string::npos) ? name.size() : dot;
        encoded.push_back(dot - start);
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
 * @brief Append a record to the answer section of a serialized message.
 * @param message the serialized message
 * @param owner the encoded owner name
 * @param type the type
 * @param rdata the data
 */
void
AppendAnswer(std::vector<uint8_t>& message,
             const std::vector<uint8_t>& owner,
             uint16_t type,
             const std::vector<uint8_t>& rdata)
{
    message.insert(message.end(), owner.begin(), owner.end());
    // type, class IN, TTL of 60 s, and data length
    message.insert(message.end(),
                   {static_cast<uint8_t>(type >> 8),
                    static_cast<uint8_t>(type),
                    0,
                    1,
                    0,
                    0,
                    0,
                    60,
                    static_cast<uint8_t>(rdata.size() >> 8),
                    static_cast<uint8_t>(rdata.size())});
    message.insert(message.end(), rdata.begin(), rdata.end());
    message[7]++;
}

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
        for (const Address& address : {Address(InetSocketAddress(SERVER, DNS_PORT)),
                                       Address(InetSocketAddress(SERVFAIL_SERVER, DNS_PORT)),
                                       Address(Inet6SocketAddress(server6, DNS_PORT))})
        {
            auto socket = Socket::CreateSocket(node, UdpSocketFactory::GetTypeId());
            socket->Bind(address);
            socket->SetRecvCallback(MakeCallback(&DnsTestServer::ReceiveUdp, this));
            m_sockets.push_back(socket);
        }
        auto tcp = Socket::CreateSocket(node, TcpSocketFactory::GetTypeId());
        tcp->Bind(InetSocketAddress(SERVER, DNS_PORT));
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
    uint32_t m_ednsQueries{0};        ///< number of queries with an EDNS(0) OPT record
    uint16_t m_ednsUdpPayloadSize{0}; ///< EDNS(0) UDP payload size of the last query with one
    std::set<uint16_t> m_ports;       ///< source ports of the queries received over UDP

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
            DnsHeader query;
            if (packet->PeekHeader(query) == 0)
            {
                continue;
            }
            bool late = false;
            auto response = Answer(query, ipv6 ? "udp6" : "udp", servfail, late);
            Ptr<Packet> out = Create<Packet>();
            out->AddHeader(*response);
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
        m_tcpData[socket] = Create<Packet>();
    }

    /**
     * Answer a query received over TCP; the messages are prefixed with their length.
     * @param socket the socket of the connection
     */
    void ReceiveTcp(Ptr<Socket> socket)
    {
        Ptr<Packet>& data = m_tcpData[socket];
        while (Ptr<Packet> packet = socket->Recv())
        {
            data->AddAtEnd(packet);
        }
        uint8_t prefix[TCP_LENGTH_PREFIX_SIZE];
        if (data->GetSize() < sizeof(prefix))
        {
            return;
        }
        data->CopyData(prefix, sizeof(prefix));
        const uint32_t length = prefix[0] << 8 | prefix[1];
        if (data->GetSize() < sizeof(prefix) + length)
        {
            return;
        }
        DnsHeader query;
        data->RemoveAtStart(sizeof(prefix));
        data->RemoveHeader(query);
        bool late = false;
        auto response = Answer(query, "tcp", false, late);
        if (!response)
        {
            socket->Close();
            return;
        }
        Ptr<Packet> message = Create<Packet>();
        message->AddHeader(*response);
        prefix[0] = message->GetSize() >> 8;
        prefix[1] = message->GetSize();
        Ptr<Packet> out = Create<Packet>(prefix, sizeof(prefix));
        out->AddAtEnd(message);
        socket->Send(out);
    }

    /**
     * Build the response to a query.
     * @param query the query
     * @param transport the transport of the query
     * @param servfail whether to answer SERVFAIL
     * @param late set to true if the response must be delayed
     * @return the response, or nothing to close the TCP connection
     */
    std::optional<DnsHeader> Answer(const DnsHeader& query,
                                    const std::string& transport,
                                    bool servfail,
                                    bool& late)
    {
        const std::string name = query.GetQuestionName();
        const uint16_t type = query.GetQuestionType();
        const auto opt = query.GetOptRecord();
        m_queries[name + "/" + std::to_string(type) + "/" + transport]++;
        if (opt)
        {
            m_ednsQueries++;
            m_ednsUdpPayloadSize = opt->rclass;
        }
        const std::string zone = "nsnam.org";
        auto a = [](const std::string& owner, uint32_t ttl, const char* address) {
            return AddressRecord(owner, ttl, Ipv4Address(address));
        };
        auto aaaa = [](const std::string& owner, uint32_t ttl, const char* address) {
            return AddressRecord(owner, ttl, Ipv6Address(address));
        };
        auto cname = [](const std::string& owner, uint32_t ttl, const std::string& target) {
            return NameRecord(DnsHeader::TYPE_CNAME, owner, ttl, target);
        };

        DnsHeader response = Response(query.GetId(), name, type);
        if (servfail)
        {
            response.SetRcode(DnsHeader::RCODE_SERVFAIL);
            return response;
        }
        if (name == "www.nsnam.org")
        {
            response.AddAnswer(cname(name, 3600, "web.nsnam.org"));
            if (type == DnsHeader::TYPE_A)
            {
                response.AddAnswer(a("web.nsnam.org", 60, "1.2.3.4"));
                response.AddAnswer(a("web.nsnam.org", 120, "5.6.7.8"));
            }
            else
            {
                response.AddAnswer(aaaa("web.nsnam.org", 60, "2001:db8::80"));
            }
            return response;
        }
        if (name == "lateerror.nsnam.org")
        {
            late = true;
            response.SetRcode(DnsHeader::RCODE_SERVFAIL);
            return response;
        }
        if (name == "tcptrunc.nsnam.org")
        {
            response.SetTruncated(true);
            return response;
        }
        if (name == "chain.nsnam.org" || name == "underscore.nsnam.org" ||
            name == "latechain.nsnam.org")
        {
            late = name == "latechain.nsnam.org";
            const std::string target =
                name == "underscore.nsnam.org" ? "_target.nsnam.org" : "target.nsnam.org";
            response.AddAnswer(cname(name, 50, target));
            return response;
        }
        if (name == "target.nsnam.org" || name == "_target.nsnam.org")
        {
            response.AddAnswer(a(name, 300, name == "target.nsnam.org" ? "4.4.4.4" : "3.3.3.3"));
            return response;
        }
        if (name == "lateempty.nsnam.org")
        {
            late = true;
            if (transport == "udp6")
            {
                response.AddAnswer(a(name, 60, "7.7.7.7"));
            }
            return response;
        }
        if (name == "v6only.nsnam.org" && type == DnsHeader::TYPE_AAAA)
        {
            response.AddAnswer(aaaa(name, 300, "2001:db8::66"));
            return response;
        }
        if ((name == "big.nsnam.org" || name == "tcpclose.nsnam.org") && transport != "tcp")
        {
            response.SetTruncated(true);
            return response;
        }
        if (name == "tcpclose.nsnam.org")
        {
            return std::nullopt;
        }
        if (name == "big.nsnam.org")
        {
            for (int host = 1; host <= 20; host++)
            {
                response.AddAnswer(a(name, 60, ("10.1.0." + std::to_string(host)).c_str()));
            }
            return response;
        }
        if (name == "spoof.nsnam.org")
        {
            response.SetQuestion("evil.nsnam.org", type);
            response.AddAnswer(a(name, 60, "6.6.6.6"));
            return response;
        }
        if (name == "stray.nsnam.org")
        {
            response.AddAnswer(a("other.nsnam.org", 60, "7.7.7.7"));
            return response;
        }
        if (name == "referral.nsnam.org")
        {
            response.SetRecursionAvailable(false);
            response.AddAuthority(NameRecord(DnsHeader::TYPE_NS, zone, 3600, "ns.nsnam.org"));
            return response;
        }
        if (name == "loop.nsnam.org")
        {
            response.AddAnswer(cname(name, 60, name));
            return response;
        }
        if ((name == "noedns.nsnam.org" || name == "latenoedns.nsnam.org") && opt)
        {
            late = name == "latenoedns.nsnam.org";
            // FORMERR, without question
            DnsHeader formerr;
            formerr.SetId(query.GetId());
            formerr.SetResponse(true);
            formerr.SetRecursionDesired(true);
            formerr.SetRecursionAvailable(true);
            formerr.SetRcode(DnsHeader::RCODE_FORMERR);
            return formerr;
        }
        if (name == "noedns.nsnam.org" || name == "latenoedns.nsnam.org" ||
            name == "ttlmsb.nsnam.org" || name == "late.nsnam.org")
        {
            late = name == "late.nsnam.org";
            response.AddAnswer(a(name, name == "ttlmsb.nsnam.org" ? 0xffffffff : 60, "9.9.9.9"));
            return response;
        }
        // NXDOMAIN, with the SOA record of the zone for the negative TTL
        response.SetRcode(DnsHeader::RCODE_NXDOMAIN);
        response.AddAuthority(SoaRecord(zone, 300, 30));
        return response;
    }

    std::vector<Ptr<Socket>> m_sockets;           ///< sockets of the server
    std::map<Ptr<Socket>, Ptr<Packet>> m_tcpData; ///< data received over TCP, by connection
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
 * @brief Wire format of the messages: round trip, names, compression pointers, OPT records.
 */
class DnsHeaderTestCase : public TestCase
{
  public:
    DnsHeaderTestCase()
        : TestCase("DNS message format")
    {
    }

  private:
    void DoRun() override
    {
        const std::string name = "www.nsnam.org";

        // round trip of a response with records of every decoded type, and escaped names
        DnsHeader message = Response(0x1234, "WWW.nsnam.org", DnsHeader::TYPE_A);
        message.SetAuthoritativeAnswer(true);
        message.AddAnswer(NameRecord(DnsHeader::TYPE_CNAME, name, 300, "x\\.y\\\\z.nsnam.org"));
        message.AddAnswer(AddressRecord("x\\.y\\\\z.nsnam.org", 60, Ipv4Address("1.2.3.4")));
        message.AddAnswer(AddressRecord(name, 60, Ipv6Address("2001:db8::1")));
        message.AddAuthority(NameRecord(DnsHeader::TYPE_NS, "nsnam.org", 3600, "ns.nsnam.org"));
        message.AddAuthority(SoaRecord("nsnam.org", 300, 30));
        message.AddOptRecord(1232);
        message.SetRcode(DnsHeader::RCODE_BADVERS);
        DnsResourceRecord txt;
        txt.type = 16;
        txt.name = name;
        txt.rdata = {5, 'h', 'e', 'l', 'l', 'o'};
        message.AddAdditional(txt);
        const auto data = Serialize(message);
        NS_TEST_ASSERT_MSG_EQ(data.size(), message.GetSerializedSize(), "Wrong serialized size");
        DnsHeader decoded;
        NS_TEST_ASSERT_MSG_EQ(Deserialize(data, decoded), data.size(), "Round trip failed");
        NS_TEST_EXPECT_MSG_EQ(decoded.GetId(), 0x1234, "Wrong id");
        NS_TEST_EXPECT_MSG_EQ(decoded.IsResponse(), true, "Not a response");
        NS_TEST_EXPECT_MSG_EQ(decoded.IsAuthoritativeAnswer(), true, "Not authoritative");
        NS_TEST_EXPECT_MSG_EQ(decoded.IsRecursionDesired(), true, "Recursion not desired");
        NS_TEST_EXPECT_MSG_EQ(decoded.IsRecursionAvailable(), true, "Recursion not available");
        NS_TEST_EXPECT_MSG_EQ(decoded.GetRcode(), DnsHeader::RCODE_BADVERS, "Wrong extended RCODE");
        NS_TEST_EXPECT_MSG_EQ(decoded.GetQuestionName(), "WWW.nsnam.org", "Case not preserved");
        NS_TEST_EXPECT_MSG_EQ(decoded.GetQuestionType(), DnsHeader::TYPE_A, "Wrong question type");
        NS_TEST_ASSERT_MSG_EQ(decoded.GetAnswers().size(), 3, "Wrong number of answers");
        NS_TEST_EXPECT_MSG_EQ(decoded.GetAnswers()[0].target,
                              "x\\.y\\\\z.nsnam.org",
                              "Escaped alias target");
        NS_TEST_EXPECT_MSG_EQ(decoded.GetAnswers()[1].name,
                              "x\\.y\\\\z.nsnam.org",
                              "Escaped owner name");
        NS_TEST_EXPECT_MSG_EQ(Ipv4Address::ConvertFrom(decoded.GetAnswers()[1].address),
                              Ipv4Address("1.2.3.4"),
                              "Wrong IPv4 address");
        NS_TEST_EXPECT_MSG_EQ(Ipv6Address::ConvertFrom(decoded.GetAnswers()[2].address),
                              Ipv6Address("2001:db8::1"),
                              "Wrong IPv6 address");
        NS_TEST_ASSERT_MSG_EQ(decoded.GetAuthorities().size(), 2, "Wrong number of authorities");
        NS_TEST_EXPECT_MSG_EQ(decoded.GetAuthorities()[0].target, "ns.nsnam.org", "Wrong NS");
        NS_TEST_EXPECT_MSG_EQ(decoded.GetAuthorities()[1].soa.mname, "ns.nsnam.org", "Wrong SOA");
        NS_TEST_EXPECT_MSG_EQ(decoded.GetAuthorities()[1].soa.minimum, 30, "Wrong SOA minimum");
        NS_TEST_ASSERT_MSG_EQ(decoded.GetAdditionalRecords().size(),
                              2,
                              "Wrong number of additional records");
        NS_TEST_ASSERT_MSG_EQ(decoded.GetOptRecord().has_value(), true, "OPT record not found");
        NS_TEST_EXPECT_MSG_EQ(decoded.GetOptRecord()->rclass, 1232, "Wrong UDP payload size");
        NS_TEST_EXPECT_MSG_EQ((decoded.GetAdditionalRecords()[1].rdata == txt.rdata),
                              true,
                              "Raw data of an unknown type");
        std::ostringstream printed;
        decoded.Print(printed);
        NS_TEST_EXPECT_MSG_NE(printed.str().find("rcode 16"), std::string::npos, "Not printed");

        // a message without question (an error of a server not supporting EDNS(0))
        DnsHeader formerr;
        formerr.SetId(1);
        formerr.SetResponse(true);
        formerr.SetRcode(DnsHeader::RCODE_FORMERR);
        NS_TEST_ASSERT_MSG_EQ(Deserialize(Serialize(formerr), decoded),
                              DnsHeader::HEADER_SIZE,
                              "Message without question rejected");
        NS_TEST_EXPECT_MSG_EQ(decoded.HasQuestion(), false, "Question found");
        NS_TEST_EXPECT_MSG_EQ(decoded.GetRcode(), DnsHeader::RCODE_FORMERR, "Wrong RCODE");

        // too short, and two questions
        auto response = Serialize(Response(1, name, DnsHeader::TYPE_A));
        std::vector<uint8_t> truncated(response.begin(), response.begin() + 20);
        NS_TEST_EXPECT_MSG_EQ(Deserialize(truncated, decoded), 0, "Truncated question accepted");
        auto twoQuestions = response;
        twoQuestions[5] = 2;
        NS_TEST_EXPECT_MSG_EQ(Deserialize(twoQuestions, decoded), 0, "Two questions accepted");

        // names: compression loop, forward pointer, and length
        auto selfPointer = response;
        const size_t owner = selfPointer.size();
        AppendAnswer(selfPointer, Pointer(owner), DnsHeader::TYPE_A, {1, 2, 3, 4});
        NS_TEST_EXPECT_MSG_EQ(Deserialize(selfPointer, decoded), 0, "Compression loop accepted");
        auto forward = response;
        AppendAnswer(forward, Pointer(owner + 2), DnsHeader::TYPE_A, {1, 2, 3, 4});
        NS_TEST_EXPECT_MSG_EQ(Deserialize(forward, decoded), 0, "Forward pointer accepted");
        auto backward = response;
        AppendAnswer(backward, Pointer(DnsHeader::HEADER_SIZE), DnsHeader::TYPE_A, {1, 2, 3, 4});
        NS_TEST_ASSERT_MSG_EQ(Deserialize(backward, decoded),
                              backward.size(),
                              "Backward pointer rejected");
        NS_TEST_EXPECT_MSG_EQ(decoded.GetAnswers()[0].name, name, "Compressed name not followed");
        auto longName = response;
        AppendAnswer(longName,
                     EncodeName(std::string(63, 'a') + "." + std::string(63, 'b') + "." +
                                std::string(63, 'c') + "." + std::string(63, 'd')),
                     DnsHeader::TYPE_A,
                     {1, 2, 3, 4});
        NS_TEST_EXPECT_MSG_EQ(Deserialize(longName, decoded), 0, "Name longer than 255 octets");
        const std::string longLabel(64, 'a');
        for (const auto& invalid : {"a..b", ".a", "a.", longLabel.c_str(), "a\\"})
        {
            NS_TEST_EXPECT_MSG_EQ(DnsHeader::SplitName(invalid).has_value(),
                                  false,
                                  "Invalid name " << invalid << " split");
        }
        NS_TEST_EXPECT_MSG_EQ(DnsHeader::SplitName("")->size(), 0, "Root not split");
        NS_TEST_EXPECT_MSG_EQ(DnsHeader::SplitName("a\\.b.c")->size(), 2, "Escaped dot split");

        // records: address of the wrong length, OPT misplaced, repeated or with an owner
        auto shortAddress = response;
        AppendAnswer(shortAddress, EncodeName(name), DnsHeader::TYPE_A, {5, 6, 7});
        NS_TEST_EXPECT_MSG_EQ(Deserialize(shortAddress, decoded),
                              0,
                              "Address record of the wrong length accepted");
        auto misplacedOpt = Response(1, name, DnsHeader::TYPE_A);
        misplacedOpt.AddAnswer(OptRecord(1232));
        NS_TEST_EXPECT_MSG_EQ(Deserialize(Serialize(misplacedOpt), decoded),
                              0,
                              "OPT record in the answer section accepted");
        auto twoOpts = Response(1, name, DnsHeader::TYPE_A);
        twoOpts.AddOptRecord(1232);
        twoOpts.AddOptRecord(1232);
        NS_TEST_EXPECT_MSG_EQ(Deserialize(Serialize(twoOpts), decoded),
                              0,
                              "Two OPT records accepted");
        auto ownedOpt = Response(1, name, DnsHeader::TYPE_A);
        ownedOpt.AddAdditional(OptRecord(1232, 0, "nsnam.org"));
        NS_TEST_EXPECT_MSG_EQ(Deserialize(Serialize(ownedOpt), decoded),
                              0,
                              "OPT record with an owner accepted");
    }
};

/**
 * @ingroup dns-resolver-test
 * @brief Interpretation of the responses: header, question, aliases, EDNS(0), negative TTL.
 */
class DnsResolverAnswerTestCase : public TestCase
{
  public:
    DnsResolverAnswerTestCase()
        : TestCase("Interpretation of the responses")
    {
    }

  private:
    void DoRun() override
    {
        const std::string name = "www.nsnam.org";
        auto a = [](const std::string& owner, uint32_t ttl, const char* address) {
            return AddressRecord(owner, ttl, Ipv4Address(address));
        };
        auto cname = [](const std::string& owner, uint32_t ttl, const std::string& target) {
            return NameRecord(DnsHeader::TYPE_CNAME, owner, ttl, target);
        };
        auto isResponse = [&](const DnsHeader& response) {
            return DnsResolver::IsResponseTo(response, 1, "WWW.nsnam.org.", DnsHeader::TYPE_A);
        };
        auto read = [&](const DnsHeader& response) {
            return DnsResolver::ReadAnswer(response, "WWW.nsnam.org.", DnsHeader::TYPE_A);
        };

        // valid response, with an alias whose addresses are the only ones taken
        DnsHeader valid = Response(1, "www.NSNAM.org", DnsHeader::TYPE_A);
        valid.AddAnswer(cname(name, 300, "web.nsnam.org"));
        valid.AddAnswer(a("web.nsnam.org", 0xffffffff, "1.2.3.4"));
        valid.AddAnswer(a(name, 60, "6.6.6.6"));
        valid.AddAnswer(a("other.org", 60, "7.7.7.7"));
        NS_TEST_ASSERT_MSG_EQ(isResponse(valid), true, "Valid response rejected");
        auto answer = read(valid);
        NS_TEST_EXPECT_MSG_EQ(answer.rcode, DnsHeader::RCODE_NOERROR, "Wrong RCODE");
        NS_TEST_EXPECT_MSG_EQ(answer.canonicalName, "web.nsnam.org", "Alias not followed");
        NS_TEST_ASSERT_MSG_EQ(answer.addresses.size(), 1, "Records of other names accepted");
        NS_TEST_EXPECT_MSG_EQ(Ipv4Address::ConvertFrom(answer.addresses[0]),
                              Ipv4Address("1.2.3.4"),
                              "Wrong address");
        NS_TEST_EXPECT_MSG_EQ(answer.chainTtl, Seconds(300), "Wrong TTL of the chain");
        NS_TEST_EXPECT_MSG_EQ(answer.ttl,
                              Time(0),
                              "A TTL with the most significant bit set is not 0");

        // header and question
        DnsHeader query = Response(1, name, DnsHeader::TYPE_A);
        query.SetResponse(false);
        NS_TEST_EXPECT_MSG_EQ(isResponse(query), false, "Query accepted");
        DnsHeader opcode = Response(1, name, DnsHeader::TYPE_A);
        opcode.SetOpcode(4);
        NS_TEST_EXPECT_MSG_EQ(isResponse(opcode), false, "Non-standard opcode accepted");
        NS_TEST_EXPECT_MSG_EQ(isResponse(Response(2, name, DnsHeader::TYPE_A)),
                              false,
                              "Other identifier accepted");
        NS_TEST_EXPECT_MSG_EQ(isResponse(Response(1, "evil.org", DnsHeader::TYPE_A)),
                              false,
                              "Other question accepted");
        NS_TEST_EXPECT_MSG_EQ(isResponse(Response(1, name, DnsHeader::TYPE_AAAA)),
                              false,
                              "Other question type accepted");
        NS_TEST_EXPECT_MSG_EQ(isResponse(Response(1, name, DnsHeader::TYPE_A)),
                              true,
                              "Question compared with case");
        // a label containing a dot is not two labels
        NS_TEST_EXPECT_MSG_EQ(isResponse(Response(1, "www\\.nsnam.org", DnsHeader::TYPE_A)),
                              false,
                              "Label containing a dot confused with two labels");

        // errors without question: only FORMERR and NOTIMP, which reject EDNS(0)
        for (uint16_t rcode : {DnsHeader::RCODE_FORMERR,
                               DnsHeader::RCODE_NOTIMP,
                               DnsHeader::RCODE_SERVFAIL,
                               DnsHeader::RCODE_REFUSED})
        {
            DnsHeader error;
            error.SetId(1);
            error.SetResponse(true);
            error.SetRcode(rcode);
            const bool ednsRejection =
                rcode == DnsHeader::RCODE_FORMERR || rcode == DnsHeader::RCODE_NOTIMP;
            NS_TEST_EXPECT_MSG_EQ(isResponse(error),
                                  ednsRejection,
                                  "Wrong validity of the error " << rcode << " without question");
        }
        DnsHeader answerWithoutQuestion;
        answerWithoutQuestion.SetId(1);
        answerWithoutQuestion.SetResponse(true);
        answerWithoutQuestion.SetRcode(DnsHeader::RCODE_FORMERR);
        answerWithoutQuestion.AddAnswer(a(name, 60, "1.2.3.4"));
        NS_TEST_EXPECT_MSG_EQ(isResponse(answerWithoutQuestion),
                              false,
                              "Answer without question accepted");

        // EDNS(0): extended RCODE, and version
        DnsHeader badvers = Response(1, name, DnsHeader::TYPE_A);
        badvers.AddOptRecord(1232);
        badvers.SetRcode(DnsHeader::RCODE_BADVERS);
        NS_TEST_EXPECT_MSG_EQ(read(badvers).rcode,
                              DnsHeader::RCODE_BADVERS,
                              "Wrong extended RCODE");
        // EDNS version 1: an error, unless BADVERS
        DnsHeader version1 = Response(1, name, DnsHeader::TYPE_A);
        version1.AddAdditional(OptRecord(1232, 0x00010000));
        NS_TEST_EXPECT_MSG_EQ(read(version1).rcode,
                              DnsHeader::RCODE_SERVFAIL,
                              "Response of another EDNS version accepted");
        DnsHeader badversion1 = Response(1, name, DnsHeader::TYPE_A);
        badversion1.AddAdditional(OptRecord(1232, 0x01010000));
        NS_TEST_EXPECT_MSG_EQ(read(badversion1).rcode,
                              DnsHeader::RCODE_BADVERS,
                              "BADVERS of another EDNS version rejected");

        // a single alias for a name, possibly repeated, and no duplicate address
        DnsHeader twoAliases = Response(1, name, DnsHeader::TYPE_A);
        twoAliases.AddAnswer(cname(name, 60, "web.nsnam.org"));
        twoAliases.AddAnswer(cname(name, 60, "ftp.nsnam.org"));
        NS_TEST_EXPECT_MSG_EQ(read(twoAliases).rcode,
                              DnsHeader::RCODE_SERVFAIL,
                              "Two aliases for a name accepted");
        DnsHeader repeated = Response(1, name, DnsHeader::TYPE_A);
        repeated.AddAnswer(cname(name, 60, "web.nsnam.org"));
        repeated.AddAnswer(cname(name, 30, "WEB.nsnam.org"));
        repeated.AddAnswer(a("web.nsnam.org", 60, "1.2.3.4"));
        repeated.AddAnswer(a("web.nsnam.org", 60, "1.2.3.4"));
        answer = read(repeated);
        NS_TEST_EXPECT_MSG_EQ(answer.rcode, DnsHeader::RCODE_NOERROR, "Repeated alias rejected");
        NS_TEST_EXPECT_MSG_EQ(answer.addresses.size(), 1, "Duplicate address not removed");
        NS_TEST_EXPECT_MSG_EQ(answer.ttl, Seconds(30), "TTL of the repeated alias ignored");
        DnsHeader loop = Response(1, name, DnsHeader::TYPE_A);
        loop.AddAnswer(cname(name, 60, name));
        NS_TEST_EXPECT_MSG_EQ(read(loop).rcode,
                              DnsHeader::RCODE_SERVFAIL,
                              "CNAME loop not detected");

        // negative TTL: SOA of the zone, bounded by the alias
        DnsHeader nodata = Response(1, name, DnsHeader::TYPE_A);
        nodata.AddAnswer(cname(name, 20, "web.nsnam.org"));
        nodata.AddAuthority(SoaRecord("other.org", 300, 5));
        nodata.AddAuthority(SoaRecord("nsnam.org", 300, 30));
        nodata.AddAuthority(SoaRecord("nsnam.org", 300, 10));
        answer = read(nodata);
        NS_TEST_EXPECT_MSG_EQ(answer.hasSoa, true, "SOA of the zone not found");
        NS_TEST_EXPECT_MSG_EQ(answer.ttl, Seconds(20), "Negative TTL not bounded by the alias");
        DnsHeader unrelated = Response(1, name, DnsHeader::TYPE_A);
        unrelated.AddAuthority(SoaRecord("other.org", 300, 5));
        NS_TEST_EXPECT_MSG_EQ(read(unrelated).hasSoa, false, "SOA of another zone accepted");
        // the zone is compared label by label: x\.nsnam.org is not in nsnam.org
        DnsHeader escaped = Response(1, name, DnsHeader::TYPE_A);
        escaped.AddAnswer(cname(name, 20, "x\\.nsnam.org"));
        escaped.AddAuthority(SoaRecord("nsnam.org", 300, 30));
        answer = read(escaped);
        NS_TEST_EXPECT_MSG_EQ(answer.canonicalName, "x\\.nsnam.org", "Dot not escaped");
        NS_TEST_EXPECT_MSG_EQ(answer.hasSoa, false, "SOA of the parent of a label accepted");
        DnsHeader referral = Response(1, name, DnsHeader::TYPE_A);
        referral.AddAuthority(NameRecord(DnsHeader::TYPE_NS, "nsnam.org", 3600, "ns.nsnam.org"));
        NS_TEST_EXPECT_MSG_EQ(read(referral).hasNs, true, "NS records not found");

        // host names
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
        // an EDNS(0) UDP payload size below 512 is advertised as 512
        Simulator::Schedule(Seconds(6), [this]() {
            m_resolver->SetAttribute("EdnsUdpPayloadSize", UintegerValue(100));
        });
        ScheduleResolve(Seconds(6), "small edns", "ttlmsb.nsnam.org");
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
        NS_TEST_EXPECT_MSG_EQ(Result("small edns"), "9.9.9.9", "Wrong A record");
        NS_TEST_EXPECT_MSG_EQ(m_server->m_ednsUdpPayloadSize,
                              512,
                              "EDNS(0) payload size below 512");
        Teardown();
    }
};

/**
 * @ingroup dns-resolver-test
 * @brief Responses to other questions, records of other names, referrals, alias loops and
 * address literals.
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
        ScheduleResolve(Seconds(36), "literal v6 only", "192.0.2.7", DnsResolver::IPV6);
        ScheduleResolve(Seconds(36), "leading zero", "010.0.2.7");
        ScheduleResolve(Seconds(36), "literal6", "2001:DB8::7", DnsResolver::ANY);
        ScheduleResolve(Seconds(36), "literal6 v4 only", "2001:db8::7");
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
        // an address is returned as is, if it belongs to the requested family
        NS_TEST_EXPECT_MSG_EQ(Result("literal"), "192.0.2.7", "Address not returned as is");
        NS_TEST_EXPECT_MSG_EQ(Result("literal dot"), "192.0.2.7", "Address with a final dot");
        NS_TEST_EXPECT_MSG_EQ(m_results.contains("literal v6 only"), true, "No result");
        NS_TEST_EXPECT_MSG_EQ(Result("literal v6 only"), "", "IPv4 address returned for IPv6");
        NS_TEST_EXPECT_MSG_EQ(m_results.contains("leading zero"), true, "No result");
        NS_TEST_EXPECT_MSG_EQ(Result("leading zero"), "", "Address with a leading zero accepted");
        NS_TEST_EXPECT_MSG_EQ(Result("literal6"), "2001:db8::7", "IPv6 address not returned as is");
        NS_TEST_EXPECT_MSG_EQ(m_results.contains("literal6 v4 only"), true, "No result");
        NS_TEST_EXPECT_MSG_EQ(Result("literal6 v4 only"), "", "IPv6 address returned for IPv4");
        NS_TEST_EXPECT_MSG_EQ(m_server->m_queries.contains("010.0.2.7/1/udp"),
                              false,
                              "Address queried");
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
        AddTestCase(new DnsHeaderTestCase, TestCase::Duration::QUICK);
        AddTestCase(new DnsResolverAnswerTestCase, TestCase::Duration::QUICK);
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
