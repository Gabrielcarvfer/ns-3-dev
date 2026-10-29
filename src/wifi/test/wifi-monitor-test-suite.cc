/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/boolean.h"
#include "ns3/config.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/mobility-helper.h"
#include "ns3/radiotap-header.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/ssid.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/udp-client-server-helper.h"
#include "ns3/uinteger.h"
#include "ns3/wifi-mac-header.h"
#include "ns3/wifi-monitor-net-device.h"
#include "ns3/wifi-net-device.h"
#include "ns3/yans-wifi-helper.h"

using namespace ns3;

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief Test of WifiMonitorNetDevice, a monitor interface on the radio of a STA.
 *
 * An HT AP (A) sends UDP packets to a STA (B), aggregated in A-MPDUs; a STA (C)
 * of the same BSS has a monitor interface and a Radiotap pcap trace. The
 * monitor interface must deliver every frame on the air, prefixed by a
 * Radiotap header: beacons, the data frames from A to B without A-MPDU
 * delimiter, padding nor FCS, control frames, and the frames transmitted by C,
 * whose WifiNetDevice keeps working (it associates).
 */
class WifiMonitorTest : public TestCase
{
  public:
    WifiMonitorTest();

    void DoRun() override;

  private:
    /**
     * Promiscuous callback of the monitor interface.
     *
     * @param device the monitor interface
     * @param packet the frame
     * @param protocol the protocol
     * @param from the transmitter
     * @param to the receiver
     * @param packetType the packet type
     * @return true
     */
    bool MonitorRx(Ptr<NetDevice> device,
                   Ptr<const Packet> packet,
                   uint16_t protocol,
                   const Address& from,
                   const Address& to,
                   NetDevice::PacketType packetType);

    uint32_t m_payloadSize;  ///< UDP payload size
    uint32_t m_nPackets;     ///< packets sent from A to B
    Mac48Address m_apAddr;   ///< the address of A
    Mac48Address m_peerAddr; ///< the address of B
    Mac48Address m_selfAddr; ///< the address of C
    uint32_t m_frames;       ///< frames delivered
    uint32_t m_wrong;        ///< frames with a wrong protocol, packet type or header
    uint32_t m_beacons;      ///< beacons
    uint32_t m_data;         ///< data frames from A to B
    uint32_t m_badDataSize;  ///< data frames from A to B of a wrong size
    uint32_t m_control;      ///< control frames
    uint32_t m_own;          ///< frames transmitted by C
};

WifiMonitorTest::WifiMonitorTest()
    : TestCase("Monitor interface delivers every frame on the air"),
      m_payloadSize(500),
      m_nPackets(20),
      m_frames(0),
      m_wrong(0),
      m_beacons(0),
      m_data(0),
      m_badDataSize(0),
      m_control(0),
      m_own(0)
{
}

bool
WifiMonitorTest::MonitorRx(Ptr<NetDevice> device,
                           Ptr<const Packet> packet,
                           uint16_t protocol,
                           const Address& from,
                           const Address& to,
                           NetDevice::PacketType packetType)
{
    m_frames++;
    if (protocol != WifiMonitorNetDevice::PROTOCOL || packetType != NetDevice::PACKET_OTHERHOST)
    {
        m_wrong++;
        return true;
    }
    auto copy = packet->Copy();
    RadiotapHeader radiotap;
    copy->RemoveHeader(radiotap);
    WifiMacHeader hdr;
    copy->PeekHeader(hdr);
    if (Mac48Address::ConvertFrom(to) != hdr.GetAddr1())
    {
        m_wrong++;
    }
    if (hdr.IsBeacon())
    {
        m_beacons++;
    }
    if (hdr.IsCtl())
    {
        m_control++;
    }
    else if (hdr.GetAddr2() == m_selfAddr)
    {
        m_own++;
    }
    if (hdr.IsQosData() && hdr.GetAddr1() == m_peerAddr && hdr.GetAddr2() == m_apAddr &&
        hdr.HasData())
    {
        m_data++;
        // QoS data header, LLC/SNAP, IPv4 and UDP headers, payload: no FCS
        if (copy->GetSize() != hdr.GetSerializedSize() + 8 + 20 + 8 + m_payloadSize)
        {
            m_badDataSize++;
        }
    }
    return true;
}

void
WifiMonitorTest::DoRun()
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);
    // keep the packets queued while the address of B is resolved
    Config::SetDefault("ns3::ArpCache::PendingQueueSize", UintegerValue(m_nPackets));

    NodeContainer apNode(1);
    NodeContainer staNodes(2); // B, C

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue("HtMcs7"),
                                 "ControlMode",
                                 StringValue("HtMcs0"));
    Ssid ssid("wifi-monitor");
    WifiMacHelper mac;
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer staDevices = wifi.Install(phy, mac, staNodes);
    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevices = wifi.Install(phy, mac, apNode);
    WifiHelper::AssignStreams(apDevices, 100);
    WifiHelper::AssignStreams(staDevices, 200);

    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(apNode);
    mobility.Install(staNodes);

    InternetStackHelper stack;
    stack.Install(apNode);
    stack.Install(staNodes);
    Ipv4AddressHelper address;
    address.SetBase("192.168.1.0", "255.255.255.0");
    address.Assign(apDevices);
    auto staInterfaces = address.Assign(staDevices);

    UdpServerHelper server(9);
    server.Install(staNodes.Get(0)).Start(Seconds(0));
    UdpClientHelper client(staInterfaces.GetAddress(0), 9);
    client.SetAttribute("MaxPackets", UintegerValue(m_nPackets));
    client.SetAttribute("Interval", TimeValue(MicroSeconds(100))); // queued: A-MPDUs
    client.SetAttribute("PacketSize", UintegerValue(m_payloadSize));
    client.Install(apNode.Get(0)).Start(Seconds(1));

    m_apAddr = Mac48Address::ConvertFrom(apDevices.Get(0)->GetAddress());
    m_peerAddr = Mac48Address::ConvertFrom(staDevices.Get(0)->GetAddress());
    m_selfAddr = Mac48Address::ConvertFrom(staDevices.Get(1)->GetAddress());
    NetDeviceContainer monitors = WifiHelper::InstallMonitor(NetDeviceContainer(staDevices.Get(1)));
    monitors.Get(0)->SetPromiscReceiveCallback(MakeCallback(&WifiMonitorTest::MonitorRx, this));
    NS_TEST_EXPECT_MSG_EQ(monitors.Get(0)->Send(Create<Packet>(10), m_apAddr, 0),
                          false,
                          "A monitor interface does not transmit");
    // the frames of A-MPDUs used to abort the Radiotap pcap traces
    phy.EnablePcap(CreateTempDirFilename("wifi-monitor"), staDevices.Get(1));

    Simulator::Stop(Seconds(2));
    Simulator::Run();
    Simulator::Destroy();

    NS_TEST_EXPECT_MSG_GT(m_frames, 0, "No frame delivered");
    NS_TEST_EXPECT_MSG_EQ(m_wrong, 0, "Frames with a wrong protocol, type or header");
    NS_TEST_EXPECT_MSG_GT(m_beacons, 0, "No beacon delivered");
    NS_TEST_EXPECT_MSG_GT(m_control, 0, "No control frame delivered");
    NS_TEST_EXPECT_MSG_GT(m_own, 0, "No frame transmitted by C delivered");
    NS_TEST_EXPECT_MSG_GT_OR_EQ(m_data, m_nPackets, "Data frames from A to B missing");
    NS_TEST_EXPECT_MSG_EQ(m_badDataSize, 0, "Data frames with delimiter, padding or FCS");
}

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief WifiMonitorNetDevice test suite
 */
class WifiMonitorTestSuite : public TestSuite
{
  public:
    WifiMonitorTestSuite();
};

WifiMonitorTestSuite::WifiMonitorTestSuite()
    : TestSuite("wifi-monitor", Type::UNIT)
{
    AddTestCase(new WifiMonitorTest, TestCase::Duration::QUICK);
}

static WifiMonitorTestSuite g_wifiMonitorTestSuite; ///< the test suite
