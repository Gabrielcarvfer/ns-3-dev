/*
 * Copyright (c) 2010 Hajime Tazaki
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: John Abraham <john.abraham@gatech.edu>
 * Adapted from: ipv4-raw-test.cc
 */

#include "ns3/arp-l3-protocol.h"
#include "ns3/boolean.h"
#include "ns3/icmpv4-l4-protocol.h"
#include "ns3/inet-socket-address.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/ipv4-list-routing.h"
#include "ns3/ipv4-raw-socket-factory.h"
#include "ns3/ipv4-static-routing.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/simple-channel.h"
#include "ns3/simple-net-device.h"
#include "ns3/simulator.h"
#include "ns3/socket-factory.h"
#include "ns3/socket.h"
#include "ns3/test.h"
#include "ns3/traffic-control-layer.h"

#ifdef __WIN32__
#include "ns3/win32-internet.h"
#else
#include <netinet/in.h>
#include <sys/socket.h>
#endif

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <vector>

using namespace ns3;

/**
 * @ingroup internet-test
 *
 * @brief IPv4 Header Test
 */
class Ipv4HeaderTest : public TestCase
{
    Ptr<Packet> m_receivedPacket; //!< Received packet.
    Ipv4Header m_receivedHeader;  //!< Received header.

    /**
     * @brief Send a packet with specific DSCP and ECN fields.
     * @param socket The source socket.
     * @param to The destination address.
     * @param dscp The DSCP field.
     * @param ecn The ECN field.
     */
    void DoSendData_IpHdr_Dscp(Ptr<Socket> socket,
                               std::string to,
                               Ipv4Header::DscpType dscp,
                               Ipv4Header::EcnType ecn);

    /**
     * @brief Send a packet with specific DSCP and ECN fields.
     * @param socket The source socket.
     * @param to The destination address.
     * @param dscp The DSCP field.
     * @param ecn The ECN field.
     */
    void SendData_IpHdr_Dscp(Ptr<Socket> socket,
                             std::string to,
                             Ipv4Header::DscpType dscp,
                             Ipv4Header::EcnType ecn);

  public:
    void DoRun() override;
    Ipv4HeaderTest();

    /**
     * @brief Receives a packet.
     * @param socket The receiving socket.
     * @param packet The packet.
     * @param from The source address.
     */
    void ReceivePacket(Ptr<Socket> socket, Ptr<Packet> packet, const Address& from);
    /**
     * @brief Receives a packet.
     * @param socket The receiving socket.
     */
    void ReceivePkt(Ptr<Socket> socket);
};

Ipv4HeaderTest::Ipv4HeaderTest()
    : TestCase("IPv4 Header Test")
{
}

void
Ipv4HeaderTest::ReceivePacket(Ptr<Socket> socket, Ptr<Packet> packet, const Address& from)
{
    m_receivedPacket = packet;
}

void
Ipv4HeaderTest::ReceivePkt(Ptr<Socket> socket)
{
    uint32_t availableData;
    availableData = socket->GetRxAvailable();
    m_receivedPacket = socket->Recv(2, MSG_PEEK);
    NS_TEST_ASSERT_MSG_EQ(m_receivedPacket->GetSize(),
                          2,
                          "Received packet size is  not equal to 2");
    m_receivedPacket = socket->Recv(std::numeric_limits<uint32_t>::max(), 0);
    NS_TEST_ASSERT_MSG_EQ(availableData,
                          m_receivedPacket->GetSize(),
                          "Received packet size is not equal to Rx buffer size");
    m_receivedPacket->PeekHeader(m_receivedHeader);
}

void
Ipv4HeaderTest::DoSendData_IpHdr_Dscp(Ptr<Socket> socket,
                                      std::string to,
                                      Ipv4Header::DscpType dscp,
                                      Ipv4Header::EcnType ecn)
{
    Address realTo = InetSocketAddress(Ipv4Address(to.c_str()), 0);
    socket->SetAttribute("IpHeaderInclude", BooleanValue(true));
    Ptr<Packet> p = Create<Packet>(123);
    Ipv4Header ipHeader;
    ipHeader.SetSource(Ipv4Address("10.0.0.2"));
    ipHeader.SetDestination(Ipv4Address(to.c_str()));
    ipHeader.SetProtocol(0);
    ipHeader.SetPayloadSize(p->GetSize());
    ipHeader.SetTtl(255);
    ipHeader.SetDscp(dscp);
    ipHeader.SetEcn(ecn);
    p->AddHeader(ipHeader);

    NS_TEST_EXPECT_MSG_EQ(socket->SendTo(p, 0, realTo), 143, to);
    socket->SetAttribute("IpHeaderInclude", BooleanValue(false));
}

void
Ipv4HeaderTest::SendData_IpHdr_Dscp(Ptr<Socket> socket,
                                    std::string to,
                                    Ipv4Header::DscpType dscp,
                                    Ipv4Header::EcnType ecn)
{
    m_receivedPacket = Create<Packet>();
    Simulator::ScheduleWithContext(socket->GetNode()->GetId(),
                                   Seconds(0),
                                   &Ipv4HeaderTest::DoSendData_IpHdr_Dscp,
                                   this,
                                   socket,
                                   to,
                                   dscp,
                                   ecn);
    Simulator::Run();
}

void
Ipv4HeaderTest::DoRun()
{
    // Create topology

    InternetStackHelper internet;
    internet.SetIpv6StackInstall(false);

    // Receiver Node
    Ptr<Node> rxNode = CreateObject<Node>();
    internet.Install(rxNode);

    Ptr<SimpleNetDevice> rxDev1;
    Ptr<SimpleNetDevice> rxDev2;
    { // first interface
        rxDev1 = CreateObject<SimpleNetDevice>();
        rxDev1->SetAddress(Mac48Address::ConvertFrom(Mac48Address::Allocate()));
        rxNode->AddDevice(rxDev1);
        Ptr<Ipv4> ipv4 = rxNode->GetObject<Ipv4>();
        uint32_t netdev_idx = ipv4->AddInterface(rxDev1);
        Ipv4InterfaceAddress ipv4Addr =
            Ipv4InterfaceAddress(Ipv4Address("10.0.0.1"), Ipv4Mask(0xffff0000U));
        ipv4->AddAddress(netdev_idx, ipv4Addr);
        ipv4->SetUp(netdev_idx);
    }

    // Sender Node
    Ptr<Node> txNode = CreateObject<Node>();
    internet.Install(txNode);
    Ptr<SimpleNetDevice> txDev1;
    {
        txDev1 = CreateObject<SimpleNetDevice>();
        txDev1->SetAddress(Mac48Address::ConvertFrom(Mac48Address::Allocate()));
        txNode->AddDevice(txDev1);
        Ptr<Ipv4> ipv4 = txNode->GetObject<Ipv4>();
        uint32_t netdev_idx = ipv4->AddInterface(txDev1);
        Ipv4InterfaceAddress ipv4Addr =
            Ipv4InterfaceAddress(Ipv4Address("10.0.0.2"), Ipv4Mask(0xffff0000U));
        ipv4->AddAddress(netdev_idx, ipv4Addr);
        ipv4->SetUp(netdev_idx);
    }

    // link the two nodes
    Ptr<SimpleChannel> channel1 = CreateObject<SimpleChannel>();
    rxDev1->SetChannel(channel1);
    txDev1->SetChannel(channel1);

    // Create the IPv4 Raw sockets
    Ptr<SocketFactory> rxSocketFactory = rxNode->GetObject<Ipv4RawSocketFactory>();
    Ptr<Socket> rxSocket = rxSocketFactory->CreateSocket();
    NS_TEST_EXPECT_MSG_EQ(rxSocket->Bind(InetSocketAddress(Ipv4Address("0.0.0.0"), 0)),
                          0,
                          "trivial");
    rxSocket->SetRecvCallback(MakeCallback(&Ipv4HeaderTest::ReceivePkt, this));

    Ptr<SocketFactory> txSocketFactory = txNode->GetObject<Ipv4RawSocketFactory>();
    Ptr<Socket> txSocket = txSocketFactory->CreateSocket();

    // ------ Now the tests ------------

    // Dscp Tests
    std::cout << "Dscp Test\n";

    std::vector<Ipv4Header::DscpType> vDscpTypes;
    vDscpTypes.push_back(Ipv4Header::DscpDefault);
    vDscpTypes.push_back(Ipv4Header::DSCP_CS1);
    vDscpTypes.push_back(Ipv4Header::DSCP_AF11);
    vDscpTypes.push_back(Ipv4Header::DSCP_AF12);
    vDscpTypes.push_back(Ipv4Header::DSCP_AF13);
    vDscpTypes.push_back(Ipv4Header::DSCP_CS2);
    vDscpTypes.push_back(Ipv4Header::DSCP_AF21);
    vDscpTypes.push_back(Ipv4Header::DSCP_AF22);
    vDscpTypes.push_back(Ipv4Header::DSCP_AF23);
    vDscpTypes.push_back(Ipv4Header::DSCP_CS3);
    vDscpTypes.push_back(Ipv4Header::DSCP_AF31);
    vDscpTypes.push_back(Ipv4Header::DSCP_AF32);
    vDscpTypes.push_back(Ipv4Header::DSCP_AF33);
    vDscpTypes.push_back(Ipv4Header::DSCP_CS4);
    vDscpTypes.push_back(Ipv4Header::DSCP_AF41);
    vDscpTypes.push_back(Ipv4Header::DSCP_AF42);
    vDscpTypes.push_back(Ipv4Header::DSCP_AF43);
    vDscpTypes.push_back(Ipv4Header::DSCP_CS5);
    vDscpTypes.push_back(Ipv4Header::DSCP_EF);
    vDscpTypes.push_back(Ipv4Header::DSCP_CS6);
    vDscpTypes.push_back(Ipv4Header::DSCP_CS7);

    for (uint32_t i = 0; i < vDscpTypes.size(); i++)
    {
        SendData_IpHdr_Dscp(txSocket, "10.0.0.1", vDscpTypes[i], Ipv4Header::ECN_ECT1);
        NS_TEST_EXPECT_MSG_EQ(m_receivedPacket->GetSize(), 143, "recv(hdrincl): 10.0.0.1");
        NS_TEST_EXPECT_MSG_EQ(m_receivedHeader.GetDscp(), vDscpTypes[i], "recv(hdrincl): 10.0.0.1");
        m_receivedHeader.Print(std::cout);
        std::cout << std::endl;
        m_receivedPacket->RemoveAllByteTags();
        m_receivedPacket = nullptr;
    }

    // Ecn tests
    std::cout << "Ecn Test\n";
    std::vector<Ipv4Header::EcnType> vEcnTypes;
    vEcnTypes.push_back(Ipv4Header::ECN_NotECT);
    vEcnTypes.push_back(Ipv4Header::ECN_ECT1);
    vEcnTypes.push_back(Ipv4Header::ECN_ECT0);
    vEcnTypes.push_back(Ipv4Header::ECN_CE);

    for (uint32_t i = 0; i < vEcnTypes.size(); i++)
    {
        SendData_IpHdr_Dscp(txSocket, "10.0.0.1", Ipv4Header::DscpDefault, vEcnTypes[i]);
        NS_TEST_EXPECT_MSG_EQ(m_receivedPacket->GetSize(), 143, "recv(hdrincl): 10.0.0.1");
        NS_TEST_EXPECT_MSG_EQ(m_receivedHeader.GetEcn(), vEcnTypes[i], "recv(hdrincl): 10.0.0.1");
        m_receivedHeader.Print(std::cout);
        std::cout << std::endl;
        m_receivedPacket->RemoveAllByteTags();
        m_receivedPacket = nullptr;
    }

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 *
 * @brief Test the serialization of the loose source route option
 *
 * The option of @RFC{791} is carried beside the fixed header fields, so a
 * header holding one must survive a serialization round trip, and the parser
 * must withstand padding and ill-formed options.
 */
class Ipv4HeaderLsrrTest : public TestCase
{
  public:
    Ipv4HeaderLsrrTest()
        : TestCase("The loose source route option survives a round trip")
    {
    }

  private:
    void DoRun() override
    {
        std::vector<Ipv4Address> route = {Ipv4Address("10.1.1.2"),
                                          Ipv4Address("10.1.2.2"),
                                          Ipv4Address("10.1.3.2")};

        Ipv4Header header;
        header.SetPayloadSize(100);
        header.SetSource(Ipv4Address("10.1.1.1"));
        header.SetDestination(route.back());
        header.SetLooseSourceRoute(route);
        header.SetSourceRoutePointer(8);
        // The checksum covers the option bytes as well as the fixed fields
        header.EnableChecksum();

        // Type, length and pointer, plus three addresses, padded to a word
        NS_TEST_ASSERT_MSG_EQ(header.GetSerializedSize(), 20 + 16, "Unexpected header size");

        Buffer buffer;
        buffer.AddAtStart(header.GetSerializedSize());
        header.Serialize(buffer.Begin());

        Ipv4Header read;
        read.EnableChecksum();
        NS_TEST_ASSERT_MSG_EQ(read.Deserialize(buffer.Begin()),
                              header.GetSerializedSize(),
                              "The option was not deserialized whole");
        NS_TEST_ASSERT_MSG_EQ(read.IsChecksumOk(),
                              true,
                              "The checksum of a header carrying the option did not verify");
        NS_TEST_ASSERT_MSG_EQ(read.HasLooseSourceRoute(), true, "The option was lost");
        NS_TEST_ASSERT_MSG_EQ((read.GetLooseSourceRoute() == route),
                              true,
                              "The route did not survive the round trip");
        NS_TEST_ASSERT_MSG_EQ(read.GetSourceRoutePointer(), 8, "The pointer was lost");

        // A header without the option deserializes to one without a route
        Ipv4Header plain;
        plain.SetPayloadSize(100);
        Buffer plainBuffer;
        plainBuffer.AddAtStart(plain.GetSerializedSize());
        plain.Serialize(plainBuffer.Begin());
        Ipv4Header readPlain;
        readPlain.Deserialize(plainBuffer.Begin());
        NS_TEST_ASSERT_MSG_EQ(readPlain.HasLooseSourceRoute(),
                              false,
                              "A route appeared out of nowhere");

        // An option with an illegal length must not be read past the header:
        // hand craft an option space claiming more than it holds
        Buffer malformed;
        malformed.AddAtStart(24);
        Buffer::Iterator it = malformed.Begin();
        it.WriteU8(0x46); // version 4, IHL 6 words
        it.WriteU8(0);
        it.WriteHtonU16(24);
        it.WriteHtonU16(0);
        it.WriteHtonU16(0);
        it.WriteU8(64);
        it.WriteU8(6);
        it.WriteU16(0);
        it.WriteHtonU32(Ipv4Address("10.1.1.1").Get());
        it.WriteHtonU32(Ipv4Address("10.1.3.2").Get());
        it.WriteU8(131); // loose source route
        it.WriteU8(39);  // length beyond the option space
        it.WriteU8(4);
        it.WriteU8(0);
        Ipv4Header readMalformed;
        readMalformed.Deserialize(malformed.Begin());
        NS_TEST_ASSERT_MSG_EQ(readMalformed.HasLooseSourceRoute(),
                              false,
                              "An option with an illegal length was accepted");
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Build the bytes of an IPv4 header around the given options
 *
 * @param options The option bytes, which the header length covers.
 * @param ihl The IHL field, or -1 to derive it from the options.
 * @param totalLength The total length field, or 0 for the header length.
 * @return The bytes of the header, checksum included.
 */
static std::vector<uint8_t>
BuildRawIpv4Header(const std::vector<uint8_t>& options, int ihl = -1, uint16_t totalLength = 0)
{
    std::vector<uint8_t> bytes = {0, 0, 0, 0, 0, 0, 0, 0, 64, 17, 0, 0, 10, 1, 1, 1, 10, 1, 2, 2};
    bytes.insert(bytes.end(), options.begin(), options.end());
    uint8_t headerWords = ihl >= 0 ? ihl : bytes.size() / 4;
    uint16_t length = totalLength ? totalLength : headerWords * 4;
    bytes[0] = 0x40 | (headerWords & 0x0f);
    bytes[2] = length >> 8;
    bytes[3] = length & 0xff;

    Buffer buffer;
    buffer.AddAtStart(bytes.size());
    buffer.Begin().Write(bytes.data(), bytes.size());
    uint16_t checksum =
        buffer.Begin().CalculateIpChecksum(std::min<uint32_t>(headerWords * 4, bytes.size()));
    // The checksum is written in network order by WriteU16 of the complement
    Buffer::Iterator it = buffer.Begin();
    it.Next(10);
    it.WriteU16(checksum);
    buffer.CopyData(bytes.data(), bytes.size());
    return bytes;
}

/**
 * @ingroup internet-test
 *
 * @brief Deserialize the given bytes into an IPv4 header
 *
 * @param bytes The bytes of the header.
 * @param header The header to fill, with its checksum enabled.
 * @return The number of bytes Deserialize consumed.
 */
static uint32_t
ReadRawIpv4Header(const std::vector<uint8_t>& bytes, Ipv4Header& header)
{
    Buffer buffer;
    buffer.AddAtStart(bytes.size());
    buffer.Begin().Write(bytes.data(), bytes.size());
    header.EnableChecksum();
    return header.Deserialize(buffer.Begin());
}

/**
 * @ingroup internet-test
 *
 * @brief Serialize an IPv4 header into bytes
 *
 * @param header The header.
 * @return The bytes of the header.
 */
static std::vector<uint8_t>
WriteRawIpv4Header(const Ipv4Header& header)
{
    Buffer buffer;
    buffer.AddAtStart(header.GetSerializedSize());
    header.Serialize(buffer.Begin());
    std::vector<uint8_t> bytes(buffer.GetSize());
    buffer.CopyData(bytes.data(), bytes.size());
    return bytes;
}

/**
 * @ingroup internet-test
 *
 * @brief Test the options the IPv4 header does not model and its robustness
 *
 * A received header holding options other than the loose source route must
 * serialize back to the very bytes it was read from, so that a forwarded
 * datagram or the header an ICMP error quotes carries them unchanged
 * (@RFC{1812}, Section 5.3.13). A header whose length fields do not hold
 * together must be flagged as malformed, without reading beyond the bytes
 * received.
 */
class Ipv4HeaderOptionsTest : public TestCase
{
  public:
    Ipv4HeaderOptionsTest()
        : TestCase("The IPv4 header keeps the options it does not model and withstands bad "
                   "lengths")
    {
    }

  private:
    void DoRun() override
    {
        // Record Route (type 7) with room for one address, a NOP, and a loose
        // source route of one address, padded by an End of Option List
        std::vector<uint8_t> options = {7, 7, 4, 0, 0, 0, 0, 1, 131, 7, 4, 10, 1, 3, 3, 0};
        std::vector<uint8_t> bytes = BuildRawIpv4Header(options);
        Ipv4Header header;
        NS_TEST_ASSERT_MSG_EQ(ReadRawIpv4Header(bytes, header), 36, "Wrong header length read");
        NS_TEST_ASSERT_MSG_EQ(header.IsMalformed(), false, "A sound header was flagged");
        NS_TEST_ASSERT_MSG_EQ(header.IsChecksumOk(), true, "The checksum did not verify");
        NS_TEST_ASSERT_MSG_EQ(header.HasLooseSourceRoute(), true, "The route was lost");
        NS_TEST_ASSERT_MSG_EQ(header.GetSerializedSize(), 36, "Inconsistent serialized size");
        NS_TEST_ASSERT_MSG_EQ((WriteRawIpv4Header(header) == bytes),
                              true,
                              "The options did not serialize back unchanged");

        // A hop updating the route leaves the other options untouched
        header.SetLooseSourceRoute({Ipv4Address("10.1.2.1")});
        header.SetSourceRoutePointer(8);
        std::vector<uint8_t> forwarded = WriteRawIpv4Header(header);
        NS_TEST_ASSERT_MSG_EQ(forwarded.size(), 36, "The forwarded header changed size");
        NS_TEST_ASSERT_MSG_EQ(
            (std::vector<uint8_t>(forwarded.begin() + 20, forwarded.begin() + 28) ==
             std::vector<uint8_t>(options.begin(), options.begin() + 8)),
            true,
            "The options before the route were not kept");
        NS_TEST_ASSERT_MSG_EQ(+forwarded[30], 8, "The pointer was not updated");
        NS_TEST_ASSERT_MSG_EQ(+forwarded[33], 2, "The route was not updated in place");

        // Only the options with the copied flag go to the later fragments
        Ipv4Header fragment = header;
        fragment.RemoveOptionsNotCopied();
        NS_TEST_ASSERT_MSG_EQ(fragment.GetSerializedSize(),
                              28,
                              "The options without the copied flag were not removed");
        NS_TEST_ASSERT_MSG_EQ(fragment.HasLooseSourceRoute(), true, "The route was removed");

        // A header with an option only, which the class does not model at all
        std::vector<uint8_t> timestamp = {68, 8, 5, 0, 0, 0, 0, 0};
        bytes = BuildRawIpv4Header(timestamp);
        Ipv4Header plain;
        ReadRawIpv4Header(bytes, plain);
        NS_TEST_ASSERT_MSG_EQ(plain.GetSerializedSize(), 28, "The option was dropped");
        NS_TEST_ASSERT_MSG_EQ((WriteRawIpv4Header(plain) == bytes),
                              true,
                              "The option did not serialize back unchanged");

        // Bytes following an End of Option List are padding, kept as is but
        // never parsed: no route appears from them
        bytes = BuildRawIpv4Header({1, 0, 131, 7, 4, 10, 1, 3, 3, 0, 0, 0});
        Ipv4Header afterEnd;
        ReadRawIpv4Header(bytes, afterEnd);
        NS_TEST_ASSERT_MSG_EQ(afterEnd.IsMalformed(), false, "Padding was taken as malformed");
        NS_TEST_ASSERT_MSG_EQ(afterEnd.HasLooseSourceRoute(),
                              false,
                              "An option was parsed after the end of the list");
        NS_TEST_ASSERT_MSG_EQ((WriteRawIpv4Header(afterEnd) == bytes),
                              true,
                              "The padding did not serialize back unchanged");

        // An unknown option of length 0 or 1 cannot be skipped
        for (uint8_t length : {0, 1})
        {
            Ipv4Header bad;
            ReadRawIpv4Header(BuildRawIpv4Header({68, length, 0, 0}), bad);
            NS_TEST_ASSERT_MSG_EQ(bad.IsMalformed(),
                                  true,
                                  "An option of length " << +length << " was accepted");
        }
        // An option whose length runs past the option space
        Ipv4Header overrun;
        ReadRawIpv4Header(BuildRawIpv4Header({68, 9, 5, 0, 0, 0, 0, 0}), overrun);
        NS_TEST_ASSERT_MSG_EQ(overrun.IsMalformed(), true, "An overrunning option was accepted");
        // An option type in the last byte, without room for its length
        Ipv4Header noLength;
        ReadRawIpv4Header(BuildRawIpv4Header({1, 1, 1, 68}), noLength);
        NS_TEST_ASSERT_MSG_EQ(noLength.IsMalformed(),
                              true,
                              "An option without length was accepted");

        // A 20 byte datagram claiming a longer header is never read past its
        // end, whatever IHL it claims
        for (uint8_t ihl = 6; ihl <= 15; ++ihl)
        {
            Ipv4Header truncated;
            uint32_t read = ReadRawIpv4Header(BuildRawIpv4Header({}, ihl, 20), truncated);
            NS_TEST_ASSERT_MSG_EQ(truncated.IsMalformed(),
                                  true,
                                  "IHL " << +ihl << " beyond the bytes received was accepted");
            NS_TEST_ASSERT_MSG_LT_OR_EQ(read, 20, "Read beyond the bytes received");
        }
        // An IHL below the fixed fields
        for (uint8_t ihl = 0; ihl < 5; ++ihl)
        {
            Ipv4Header shortHeader;
            ReadRawIpv4Header(BuildRawIpv4Header({}, ihl, 40), shortHeader);
            NS_TEST_ASSERT_MSG_EQ(shortHeader.IsMalformed(),
                                  true,
                                  "IHL " << +ihl << " was accepted");
            NS_TEST_ASSERT_MSG_EQ(shortHeader.GetPayloadSize(),
                                  20,
                                  "The payload size is not counted past the fixed fields");
        }
        // A total length below the header length
        Ipv4Header shortTotal;
        ReadRawIpv4Header(BuildRawIpv4Header({1, 1, 1, 1}, -1, 20), shortTotal);
        NS_TEST_ASSERT_MSG_EQ(shortTotal.IsMalformed(),
                              true,
                              "A total length below the header length was accepted");
        // Fewer bytes than the fixed fields
        Ipv4Header tiny;
        std::vector<uint8_t> tinyBytes = BuildRawIpv4Header({});
        tinyBytes.resize(12);
        NS_TEST_ASSERT_MSG_EQ(ReadRawIpv4Header(tinyBytes, tiny), 0, "A truncated header was read");
        NS_TEST_ASSERT_MSG_EQ(tiny.IsMalformed(), true, "A truncated header was accepted");
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Test the validity checks of the loose source route option
 *
 * An option whose length or pointer is illegal, a second loose source route
 * and a strict source route make the header malformed (@RFC{791}, @RFC{1812}
 * Section 5.2.4.1), whereas a pointer past the option only means that the
 * route is exhausted.
 */
class Ipv4HeaderSourceRouteValidityTest : public TestCase
{
  public:
    Ipv4HeaderSourceRouteValidityTest()
        : TestCase("The loose source route option is checked for validity")
    {
    }

  private:
    /// A case of the matrix
    struct Case
    {
        std::string name;             //!< Name of the case
        std::vector<uint8_t> options; //!< Option bytes, padded to a word
        bool malformed;               //!< Whether the header is malformed
        bool hasRoute;                //!< Whether a route is parsed
        uint32_t hops;                //!< Number of addresses of the route
    };

    void DoRun() override
    {
        const std::vector<Case> cases = {
            {"length 2", {131, 2, 1, 0}, true, false, 0},
            {"length 3", {131, 3, 4, 0}, false, true, 0},
            {"length 5", {131, 5, 4, 10, 1, 0, 0, 0}, true, false, 0},
            {"pointer 3", {131, 7, 3, 10, 1, 3, 3, 0}, true, false, 0},
            {"pointer 5", {131, 7, 5, 10, 1, 3, 3, 0}, true, false, 0},
            {"pointer 0", {131, 7, 0, 10, 1, 3, 3, 0}, true, false, 0},
            {"pointer past the option", {131, 7, 12, 10, 1, 3, 3, 0}, false, true, 1},
            {"NOP first", {1, 131, 7, 4, 10, 1, 3, 3}, false, true, 1},
            {"strict source route", {137, 7, 4, 10, 1, 3, 3, 0}, true, false, 0},
            {"two loose source routes",
             {131, 7, 4, 10, 1, 3, 3, 131, 7, 4, 10, 1, 4, 4, 0, 0},
             true,
             true,
             1},
            {"loose and strict source routes",
             {131, 7, 4, 10, 1, 3, 3, 137, 7, 4, 10, 1, 4, 4, 0, 0},
             true,
             true,
             1},
        };
        for (const auto& c : cases)
        {
            std::vector<uint8_t> bytes = BuildRawIpv4Header(c.options);
            Ipv4Header header;
            ReadRawIpv4Header(bytes, header);
            NS_TEST_ASSERT_MSG_EQ(header.IsMalformed(), c.malformed, "Case " << c.name);
            NS_TEST_ASSERT_MSG_EQ(header.HasLooseSourceRoute(), c.hasRoute, "Case " << c.name);
            NS_TEST_ASSERT_MSG_EQ(header.GetLooseSourceRoute().size(), c.hops, "Case " << c.name);
            NS_TEST_ASSERT_MSG_EQ((WriteRawIpv4Header(header) == bytes),
                                  true,
                                  "Case " << c.name << " did not serialize back unchanged");
        }
    }
};

/**
 * @ingroup internet-test
 *
 * @brief IPv4 Header TestSuite
 */
class Ipv4HeaderTestSuite : public TestSuite
{
  public:
    Ipv4HeaderTestSuite()
        : TestSuite("ipv4-header", Type::UNIT)
    {
        AddTestCase(new Ipv4HeaderTest, TestCase::Duration::QUICK);
        AddTestCase(new Ipv4HeaderLsrrTest, TestCase::Duration::QUICK);
        AddTestCase(new Ipv4HeaderOptionsTest, TestCase::Duration::QUICK);
        AddTestCase(new Ipv4HeaderSourceRouteValidityTest, TestCase::Duration::QUICK);
    }
};

static Ipv4HeaderTestSuite g_ipv4HeaderTestSuite; //!< Static variable for test initialization
