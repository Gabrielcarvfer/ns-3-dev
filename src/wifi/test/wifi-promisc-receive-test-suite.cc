/*
 * Copyright (c) 2026
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

/**
 * @file
 * @ingroup wifi-test
 *
 * Regression test for issue 424
 * (https://gitlab.com/nsnam/ns-3-dev/-/issues/424).
 *
 * KNOWN-FAILING REGRESSION TEST: as of the time of writing there is no fix
 * for issue 424. This test is expected to FAIL on the current master branch.
 * It documents the observable correct behavior so that, once the underlying
 * defect is fixed, the test will start to pass and guard against regressions.
 *
 * Issue 424 summary: a Wi-Fi STA put in promiscuous mode via
 * SetPromiscReceiveCallback only delivers ONE packet to the promiscuous
 * callback, instead of every overheard frame. The triage notes
 * (utils/issue_triage/INVEST_shortlist2.md) point to the MAC RX
 * deduplication logic (mac-rx-middle.cc) as the likely culprit, while the
 * forwarding path (frame-exchange-manager.cc -> m_rxMiddle->Receive ->
 * wifi-net-device.cc) appears correct. This test deliberately does NOT assume
 * a fix location; it only asserts the externally observable behavior: a
 * promiscuous STA must see MORE THAN ONE overheard data frame when several
 * frames are exchanged between two other stations within its range.
 */

#include "ns3/boolean.h"
#include "ns3/config.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/mobility-helper.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/packet.h"
#include "ns3/propagation-delay-model.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/udp-client-server-helper.h"
#include "ns3/uinteger.h"
#include "ns3/wifi-net-device.h"

using namespace ns3;

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief Test case for issue 424: promiscuous Wi-Fi STA must overhear all
 *        frames, not just one.
 *
 * Topology (all nodes are at fixed positions, very close together so that
 * every frame is received by every node):
 *
 *   - Node A: AP
 *   - Node B: STA, exchanges several data packets with A
 *   - Node C: STA, placed in promiscuous mode; it should overhear the A<->B
 *     data frames.
 *
 * The test counts how many times the promiscuous callback installed on C is
 * invoked and asserts that it is invoked for MORE THAN ONE overheard frame.
 *
 * On the current (unfixed) master branch this assertion is expected to FAIL
 * because only a single packet is delivered to the promiscuous callback.
 */
class WifiPromiscReceiveTest : public TestCase
{
  public:
    WifiPromiscReceiveTest();
    ~WifiPromiscReceiveTest() override;

    void DoRun() override;

  private:
    /**
     * Callback installed on node C in promiscuous mode. Counts every
     * invocation that carries a data payload (i.e. an overheard data frame).
     *
     * @param device the net device that received the packet
     * @param packet the received packet
     * @param protocol the protocol number associated with the packet
     * @param from the source address
     * @param to the destination address
     * @param packetType the type of packet (unicast/broadcast/otherhost/...)
     * @return always true (the packet is consumed by the test)
     */
    bool PromiscRx(Ptr<NetDevice> device,
                   Ptr<const Packet> packet,
                   uint16_t protocol,
                   const Address& from,
                   const Address& to,
                   NetDevice::PacketType packetType);

    uint32_t m_promiscRxCount; ///< number of overheard data frames seen by C
    uint32_t m_payloadSize;    ///< UDP payload size in bytes
    uint32_t m_nPackets;       ///< number of data packets sent from A to B
};

WifiPromiscReceiveTest::WifiPromiscReceiveTest()
    : TestCase("Promiscuous STA must overhear all frames (issue 424)"),
      m_promiscRxCount(0),
      m_payloadSize(500),
      m_nPackets(10)
{
}

WifiPromiscReceiveTest::~WifiPromiscReceiveTest()
{
}

bool
WifiPromiscReceiveTest::PromiscRx(Ptr<NetDevice> device,
                                  Ptr<const Packet> packet,
                                  uint16_t protocol,
                                  const Address& from,
                                  const Address& to,
                                  NetDevice::PacketType packetType)
{
    // Count only frames that actually carry a payload. This filters out
    // control/management frames (beacons, ACKs, association, etc.) and keeps
    // the assertion focused on overheard data frames.
    if (packet->GetSize() >= m_payloadSize)
    {
        m_promiscRxCount++;
    }
    return true;
}

void
WifiPromiscReceiveTest::DoRun()
{
    // Make the run fully deterministic.
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);
    int64_t streamNumber = 100;

    Time warmup(Seconds(1)); // to account for association
    Time appDuration(Seconds(2));

    // Node A: AP. Node B and Node C: STAs.
    NodeContainer wifiApNode;
    wifiApNode.Create(1);
    NodeContainer wifiStaNodes;
    wifiStaNodes.Create(2); // B = index 0 (peer), C = index 1 (promiscuous)

    Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel>();
    Ptr<FriisPropagationLossModel> lossModel = CreateObject<FriisPropagationLossModel>();
    spectrumChannel->AddPropagationLossModel(lossModel);
    Ptr<ConstantSpeedPropagationDelayModel> delayModel =
        CreateObject<ConstantSpeedPropagationDelayModel>();
    spectrumChannel->SetPropagationDelayModel(delayModel);

    SpectrumWifiPhyHelper phy;
    phy.SetChannel(spectrumChannel);

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue("HtMcs0"),
                                 "ControlMode",
                                 StringValue("HtMcs0"));

    Ssid ssid = Ssid("wifi-promisc-424");

    WifiMacHelper mac;
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer staDevices = wifi.Install(phy, mac, wifiStaNodes);

    mac.SetType("ns3::ApWifiMac",
                "Ssid",
                SsidValue(ssid),
                "EnableBeaconJitter",
                BooleanValue(false));
    NetDeviceContainer apDevices = wifi.Install(phy, mac, wifiApNode);

    // Assign fixed streams to random variables in use, for determinism.
    WifiHelper::AssignStreams(apDevices, streamNumber);
    WifiHelper::AssignStreams(staDevices, streamNumber);

    // Place all nodes at fixed positions, very close together, so that node C
    // is guaranteed to overhear every A<->B frame.
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    positionAlloc->Add(Vector(0.0, 0.0, 0.0)); // A (AP)
    positionAlloc->Add(Vector(1.0, 0.0, 0.0)); // B (peer STA)
    positionAlloc->Add(Vector(2.0, 0.0, 0.0)); // C (promiscuous STA)
    mobility.SetPositionAllocator(positionAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(wifiApNode);
    mobility.Install(wifiStaNodes);

    // Internet stack.
    InternetStackHelper stack;
    stack.Install(wifiApNode);
    stack.Install(wifiStaNodes);

    Ipv4AddressHelper address;
    address.SetBase("192.168.1.0", "255.255.255.0");
    Ipv4InterfaceContainer apInterface = address.Assign(apDevices);
    Ipv4InterfaceContainer staInterfaces = address.Assign(staDevices);

    // Generate several data packets from A (AP) to B (peer STA). Node C only
    // overhears them; it is not a destination.
    uint16_t port = 9;
    UdpServerHelper server(port);
    ApplicationContainer serverApp = server.Install(wifiStaNodes.Get(0)); // B
    serverApp.Start(Seconds(0));
    serverApp.Stop(warmup + appDuration);

    UdpClientHelper client(staInterfaces.GetAddress(0), port); // to B
    client.SetAttribute("MaxPackets", UintegerValue(m_nPackets));
    client.SetAttribute("Interval", TimeValue(MilliSeconds(100)));
    client.SetAttribute("PacketSize", UintegerValue(m_payloadSize));
    ApplicationContainer clientApp = client.Install(wifiApNode.Get(0)); // from A
    clientApp.Start(warmup);
    clientApp.Stop(warmup + appDuration);

    // Put node C into promiscuous mode and count overheard data frames.
    Ptr<WifiNetDevice> cDevice = DynamicCast<WifiNetDevice>(staDevices.Get(1));
    NS_TEST_ASSERT_MSG_NE(cDevice, nullptr, "Node C does not have a WifiNetDevice");
    cDevice->SetPromiscReceiveCallback(
        MakeCallback(&WifiPromiscReceiveTest::PromiscRx, this));

    Simulator::Stop(warmup + appDuration);
    Simulator::Run();
    Simulator::Destroy();

    // Correct behavior (post-fix): the promiscuous STA C must overhear MORE
    // than one of the data frames exchanged between A and B. On the current
    // (unfixed) master branch only a single packet is delivered to the
    // promiscuous callback, so this assertion is EXPECTED TO FAIL until issue
    // 424 is resolved.
    NS_TEST_EXPECT_MSG_GT(m_promiscRxCount,
                          1,
                          "Promiscuous STA C should overhear more than one A<->B "
                          "data frame, but the promiscuous callback only fired for "
                          << m_promiscRxCount << " data frame(s) (issue 424)");
}

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief Wi-Fi promiscuous receive test suite (issue 424).
 */
class WifiPromiscReceiveTestSuite : public TestSuite
{
  public:
    WifiPromiscReceiveTestSuite();
};

WifiPromiscReceiveTestSuite::WifiPromiscReceiveTestSuite()
    : TestSuite("wifi-promisc-receive", Type::UNIT)
{
    AddTestCase(new WifiPromiscReceiveTest, TestCase::Duration::QUICK);
}

static WifiPromiscReceiveTestSuite g_wifiPromiscReceiveTestSuite; ///< the test suite
