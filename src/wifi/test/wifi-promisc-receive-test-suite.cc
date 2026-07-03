/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/boolean.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/mobility-helper.h"
#include "ns3/packet.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/ssid.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/udp-client-server-helper.h"
#include "ns3/uinteger.h"
#include "ns3/wifi-net-device.h"
#include "ns3/yans-wifi-helper.h"

#include <map>
#include <string>
#include <tuple>
#include <vector>

using namespace ns3;

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief Test case for @issueid{424}: devices in promiscuous mode deliver the data frames
 *        they overhear, whatever their type of MAC.
 *
 * On one channel:
 *
 *   - BSS "bss": AP A, STA B, STA C
 *   - BSS "other": AP D
 *   - an ad hoc network: E, F, G
 *
 * A sends to B and to C (in step, so that the sequence numbers of the two flows are the
 * same), B sends to A and E sends to F. C (a STA), D (an AP) and G (an ad hoc device) are
 * in promiscuous mode: each must deliver every data frame it overhears, once, as
 * PACKET_OTHERHOST, with its source and destination, whatever the To DS and From DS
 * fields of the frame. C must also receive all the packets A sends to it.
 */
class WifiPromiscRxTest : public TestCase
{
  public:
    WifiPromiscRxTest();

  private:
    void DoSetup() override;
    void DoRun() override;

    /**
     * Promiscuous callback of an observer.
     *
     * @param observer the name of the observer
     * @param device the net device that received the packet
     * @param packet the received packet
     * @param protocol the protocol number associated with the packet
     * @param from the source address
     * @param to the destination address
     * @param packetType the type of packet
     * @return true
     */
    bool PromiscRx(std::string observer,
                   Ptr<NetDevice> device,
                   Ptr<const Packet> packet,
                   uint16_t protocol,
                   const Address& from,
                   const Address& to,
                   NetDevice::PacketType packetType);

    /**
     * Install a UDP flow.
     *
     * @param from the node sending
     * @param to the address the packets are sent to
     */
    void InstallFlow(Ptr<Node> from, Ipv4Address to);

    static constexpr uint32_t m_payloadSize{500}; ///< UDP payload size
    static constexpr uint32_t m_nPackets{10};     ///< packets sent by each flow

    NodeContainer m_nodes;                ///< A, B, C, D, E, F, G
    std::map<char, Mac48Address> m_addrs; ///< the addresses of the nodes, by name
    /// overheard data frames, by observer, source and destination
    std::map<std::tuple<std::string, Mac48Address, Mac48Address>, uint32_t> m_overheard;
    uint32_t m_notOtherHost{0};           ///< overheard frames not delivered as OTHERHOST
    Ptr<UdpServer> m_serverC;             ///< the UDP server of C
};

WifiPromiscRxTest::WifiPromiscRxTest()
    : TestCase("Devices in promiscuous mode deliver the data frames they overhear")
{
}

bool
WifiPromiscRxTest::PromiscRx(std::string observer,
                             Ptr<NetDevice> device,
                             Ptr<const Packet> packet,
                             uint16_t protocol,
                             const Address& from,
                             const Address& to,
                             NetDevice::PacketType packetType)
{
    // count only the UDP packets of the flows (not ARP)
    if (packet->GetSize() < m_payloadSize)
    {
        return true;
    }
    const auto dst = Mac48Address::ConvertFrom(to);
    if (dst == Mac48Address::ConvertFrom(device->GetAddress()))
    {
        return true; // received, not overheard
    }
    m_notOtherHost += (packetType == NetDevice::PACKET_OTHERHOST) ? 0 : 1;
    ++m_overheard[{observer, Mac48Address::ConvertFrom(from), dst}];
    return true;
}

void
WifiPromiscRxTest::InstallFlow(Ptr<Node> from, Ipv4Address to)
{
    UdpClientHelper client(to, 9);
    client.SetAttribute("MaxPackets", UintegerValue(m_nPackets));
    client.SetAttribute("Interval", TimeValue(MilliSeconds(50)));
    client.SetAttribute("PacketSize", UintegerValue(m_payloadSize));
    auto apps = client.Install(from);
    apps.Start(Seconds(1));
}

void
WifiPromiscRxTest::DoSetup()
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    m_nodes.Create(7);
    const auto node = [this](char name) { return m_nodes.Get(name - 'A'); };

    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(m_nodes);

    YansWifiPhyHelper phy;
    phy.SetChannel(YansWifiChannelHelper::Default().Create());
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue("HtMcs0"),
                                 "ControlMode",
                                 StringValue("HtMcs0"));

    WifiMacHelper mac;
    NetDeviceContainer bss;
    NetDeviceContainer adhoc;
    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(Ssid("bss")));
    bss.Add(wifi.Install(phy, mac, node('A')));
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(Ssid("bss")));
    bss.Add(wifi.Install(phy, mac, NodeContainer(node('B'), node('C'))));
    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(Ssid("other")));
    const auto devD = wifi.Install(phy, mac, node('D'));
    mac.SetType("ns3::AdhocWifiMac");
    adhoc.Add(wifi.Install(phy, mac, NodeContainer(node('E'), node('F'), node('G'))));
    NetDeviceContainer all(bss, devD);
    all.Add(adhoc);
    WifiHelper::AssignStreams(all, 100);

    for (auto name = 'A'; name <= 'G'; ++name)
    {
        m_addrs[name] = Mac48Address::ConvertFrom(node(name)->GetDevice(0)->GetAddress());
    }

    InternetStackHelper stack;
    stack.Install(m_nodes);
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    const auto bssIfs = address.Assign(bss); // A, B, C
    address.Assign(devD);
    address.SetBase("10.2.2.0", "255.255.255.0");
    const auto adhocIfs = address.Assign(adhoc); // E, F, G

    UdpServerHelper server(9);
    server.Install(NodeContainer(node('A'), node('B'), node('F'))).Start(Seconds(0));
    m_serverC = DynamicCast<UdpServer>(server.Install(node('C')).Get(0));
    m_serverC->SetStartTime(Seconds(0));
    InstallFlow(node('A'), bssIfs.GetAddress(1));   // A to B
    InstallFlow(node('A'), bssIfs.GetAddress(2));   // A to C
    InstallFlow(node('B'), bssIfs.GetAddress(0));   // B to A
    InstallFlow(node('E'), adhocIfs.GetAddress(1)); // E to F

    for (const auto observer : {'C', 'D', 'G'})
    {
        node(observer)->GetDevice(0)->SetPromiscReceiveCallback(
            MakeCallback(&WifiPromiscRxTest::PromiscRx, this).Bind(std::string(1, observer)));
    }
}

void
WifiPromiscRxTest::DoRun()
{
    Simulator::Stop(Seconds(3));
    Simulator::Run();
    Simulator::Destroy();

    // the flows each observer overhears (C receives A to C, it does not overhear it)
    const std::map<std::string, std::vector<std::pair<char, char>>> expected{
        {"C", {{'A', 'B'}, {'B', 'A'}, {'E', 'F'}}},
        {"D", {{'A', 'B'}, {'A', 'C'}, {'B', 'A'}, {'E', 'F'}}},
        {"G", {{'A', 'B'}, {'A', 'C'}, {'B', 'A'}, {'E', 'F'}}}};
    uint32_t expectedTotal = 0;
    for (const auto& [observer, flows] : expected)
    {
        for (const auto& [src, dst] : flows)
        {
            const auto key = std::make_tuple(observer, m_addrs.at(src), m_addrs.at(dst));
            NS_TEST_EXPECT_MSG_EQ(m_overheard[key],
                                  m_nPackets,
                                  observer << " overheard " << src << " to " << dst);
            expectedTotal += m_nPackets;
        }
    }
    uint32_t total = 0;
    for (const auto& [key, count] : m_overheard)
    {
        total += count;
    }
    NS_TEST_EXPECT_MSG_EQ(total, expectedTotal, "Overheard frames with wrong addresses");
    NS_TEST_EXPECT_MSG_EQ(m_notOtherHost, 0, "Overheard frames not delivered as OTHERHOST");
    NS_TEST_EXPECT_MSG_EQ(m_serverC->GetReceived(),
                          m_nPackets,
                          "C lost packets sent to it while overhearing");
}

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief Wi-Fi promiscuous receive test suite (@issueid{424}).
 */
class WifiPromiscRxTestSuite : public TestSuite
{
  public:
    WifiPromiscRxTestSuite();
};

WifiPromiscRxTestSuite::WifiPromiscRxTestSuite()
    : TestSuite("wifi-promisc-rx", Type::UNIT)
{
    AddTestCase(new WifiPromiscRxTest, TestCase::Duration::QUICK);
}

static WifiPromiscRxTestSuite g_wifiPromiscRxTestSuite; ///< the test suite
