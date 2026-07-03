/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/boolean.h"
#include "ns3/config.h"
#include "ns3/double.h"
#include "ns3/mobility-helper.h"
#include "ns3/packet-socket-client.h"
#include "ns3/packet-socket-helper.h"
#include "ns3/packet-socket-server.h"
#include "ns3/packet.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/ssid.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"
#include "ns3/wifi-net-device.h"
#include "ns3/yans-wifi-helper.h"

using namespace ns3;

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief Regression test for @issueid{1199}.
 *
 * Reproduces the double-dequeue abort that occurred with a VHT (802.11ac)
 * configuration when A-MPDU aggregation is disabled (BE_MaxAmpduSize=0) and
 * MSDU/MPDU fragmentation is enabled (a small FragmentationThreshold). The test
 * also forces an originator Block Ack agreement to be established by lowering
 * BE_BlockAckThreshold so that QoS data frames are tracked in the Block Ack
 * window even though they are sent with the normal ack policy (one MPDU at a
 * time, because aggregation is disabled and fragments cannot be aggregated).
 *
 * Acknowledged non-final fragments must remain available for the normal
 * fragment-replacement path even though a Block Ack agreement tracks the
 * MSDU, and the received fragments must be reassembled by the regular
 * reassembly machinery.
 *
 * The test checks that Simulator::Run() returns normally and at least one
 * packet is received by the server.
 *
 * Deterministic: fixed RngSeed/RngRun, ConstantPositionMobilityModel,
 * IdealWifiManager (no random rate selection), short fixed-duration run.
 */
class WifiFragmentationBlockAckTestCase : public TestCase
{
  public:
    WifiFragmentationBlockAckTestCase();
    ~WifiFragmentationBlockAckTestCase() override;

    void DoRun() override;

  private:
    /**
     * Callback invoked when the server application receives a packet.
     *
     * @param context the trace context
     * @param p the received packet
     * @param addr the address of the sender
     */
    void L7Receive(std::string context, Ptr<const Packet> p, const Address& addr);

    uint32_t m_received; ///< number of packets received by the server application
};

WifiFragmentationBlockAckTestCase::WifiFragmentationBlockAckTestCase()
    : TestCase("Issue #1199: VHT, A-MPDU disabled, fragmentation + Block Ack "
               "double-dequeue abort"),
      m_received(0)
{
}

WifiFragmentationBlockAckTestCase::~WifiFragmentationBlockAckTestCase()
{
}

void
WifiFragmentationBlockAckTestCase::L7Receive(std::string context,
                                             Ptr<const Packet> p,
                                             const Address& addr)
{
    m_received++;
}

void
WifiFragmentationBlockAckTestCase::DoRun()
{
    // Deterministic RNG configuration.
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    NodeContainer wifiStaNode;
    wifiStaNode.Create(1);

    NodeContainer wifiApNode;
    wifiApNode.Create(1);

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());

    WifiHelper wifi;
    // VHT (802.11ac), as in the issue report.
    wifi.SetStandard(WIFI_STANDARD_80211ac);
    // IdealWifiManager keeps rate selection deterministic and, together with the
    // small FragmentationThreshold, guarantees the data PSDU is fragmented.
    wifi.SetRemoteStationManager("ns3::IdealWifiManager",
                                 "FragmentationThreshold",
                                 UintegerValue(400));

    Ssid ssid = Ssid("ns-3-ssid");
    WifiMacHelper mac;

    // A-MPDU aggregation disabled (BE_MaxAmpduSize=0) and A-MSDU disabled, so
    // each MSDU is transmitted as a single MPDU and is fragmented when larger
    // than the fragmentation threshold. BE_BlockAckThreshold=2 forces a Block
    // Ack agreement to be established once more than one frame is queued.
    mac.SetType("ns3::StaWifiMac",
                "Ssid",
                SsidValue(ssid),
                "BE_MaxAmsduSize",
                UintegerValue(0),
                "BE_MaxAmpduSize",
                UintegerValue(0),
                "BE_BlockAckThreshold",
                UintegerValue(2),
                "ActiveProbing",
                BooleanValue(false));

    NetDeviceContainer staDevices = wifi.Install(phy, mac, wifiStaNode);

    mac.SetType("ns3::ApWifiMac",
                "Ssid",
                SsidValue(ssid),
                "BE_MaxAmsduSize",
                UintegerValue(0),
                "BE_MaxAmpduSize",
                UintegerValue(0),
                "BeaconGeneration",
                BooleanValue(true));

    NetDeviceContainer apDevices = wifi.Install(phy, mac, wifiApNode);

    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    positionAlloc->Add(Vector(0.0, 0.0, 0.0));
    positionAlloc->Add(Vector(1.0, 0.0, 0.0));
    mobility.SetPositionAllocator(positionAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(wifiApNode);
    mobility.Install(wifiStaNode);

    Ptr<WifiNetDevice> apDevice = DynamicCast<WifiNetDevice>(apDevices.Get(0));
    Ptr<WifiNetDevice> staDevice = DynamicCast<WifiNetDevice>(staDevices.Get(0));

    PacketSocketAddress socket;
    socket.SetSingleDevice(staDevice->GetIfIndex());
    socket.SetPhysicalAddress(apDevice->GetAddress());
    socket.SetProtocol(1);

    PacketSocketHelper packetSocket;
    packetSocket.Install(wifiStaNode);
    packetSocket.Install(wifiApNode);

    // Generate several large packets (each larger than the fragmentation
    // threshold, so each is fragmented into multiple MPDUs). Having more than
    // one packet queued triggers the Block Ack agreement (BE_BlockAckThreshold).
    Ptr<PacketSocketClient> client = CreateObject<PacketSocketClient>();
    client->SetAttribute("PacketSize", UintegerValue(1400));
    client->SetAttribute("MaxPackets", UintegerValue(5));
    client->SetAttribute("Interval", TimeValue(MicroSeconds(0)));
    client->SetRemote(socket);
    wifiStaNode.Get(0)->AddApplication(client);
    client->SetStartTime(Seconds(1.0));
    client->SetStopTime(Seconds(3.0));

    Ptr<PacketSocketServer> server = CreateObject<PacketSocketServer>();
    server->SetLocal(socket);
    wifiApNode.Get(0)->AddApplication(server);
    server->SetStartTime(Seconds(0.0));
    server->SetStopTime(Seconds(4.0));

    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::PacketSocketServer/Rx",
                    MakeCallback(&WifiFragmentationBlockAckTestCase::L7Receive, this));

    Simulator::Stop(Seconds(4.0));

    Simulator::Run();
    Simulator::Destroy();

    // If we get here, the double-dequeue abort did not happen.
    NS_TEST_EXPECT_MSG_GT_OR_EQ(m_received,
                                1,
                                "Expected the simulation to complete and deliver at least one "
                                "packet (issue #1199 double-dequeue abort)");
}

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief Test suite for the @issueid{1199} fragmentation + Block Ack
 * double-dequeue regression. See WifiFragmentationBlockAckTestCase.
 */
class WifiFragmentationBlockAckTestSuite : public TestSuite
{
  public:
    WifiFragmentationBlockAckTestSuite();
};

WifiFragmentationBlockAckTestSuite::WifiFragmentationBlockAckTestSuite()
    : TestSuite("wifi-fragmentation-blockack", Type::UNIT)
{
    AddTestCase(new WifiFragmentationBlockAckTestCase, TestCase::Duration::QUICK);
}

/// the test suite
static WifiFragmentationBlockAckTestSuite g_wifiFragmentationBlockAckTestSuite;
