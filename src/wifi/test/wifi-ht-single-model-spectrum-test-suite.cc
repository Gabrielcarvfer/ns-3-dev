/*
 * Copyright (c) 2026 ns-3 contributors
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

/**
 * @file
 *
 * KNOWN-FAILING regression test for ns-3 GitLab issue #1090
 * ("wifi: Ht simulations cannot use SingleModelSpectrumChannel in 2.4 GHz").
 *
 * This suite reproduces the bug deterministically: an 802.11n (HT) Wi-Fi
 * configuration on a SingleModelSpectrumChannel in the 2.4 GHz band aborts
 * during reception. The transmit power spectral density produced by
 * HtPhy::GetTxPowerSpectralDensity() (built via
 * WifiSpectrumValueHelper::CreateHtOfdmTxPowerSpectralDensity, e.g. 135 bands
 * for a 20 MHz channel) does not match the receive SpectrumModel installed by
 * SpectrumWifiPhy::ResetSpectrumModel() through
 * WifiSpectrumValueHelper::GetSpectrumModel() (e.g. 193 bands). Because a
 * SingleModelSpectrumChannel performs no spectrum conversion between TX and RX,
 * the NS_ASSERT_MSG in SpectrumWifiPhy::StartRx() (see
 * src/wifi/model/spectrum-wifi-phy.cc) fires when the two band counts differ.
 *
 * As of the current master there is NO fix for issue #1090, so this test is
 * EXPECTED TO ABORT (fail/crash) when run. It is intended to start passing
 * once the underlying band-count mismatch is resolved.
 */

#include "ns3/boolean.h"
#include "ns3/config.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/propagation-delay-model.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/mobility-helper.h"
#include "ns3/nstime.h"
#include "ns3/packet-socket-address.h"
#include "ns3/packet-socket-client.h"
#include "ns3/packet-socket-helper.h"
#include "ns3/packet-socket-server.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/single-model-spectrum-channel.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"
#include "ns3/wifi-mac-helper.h"
#include "ns3/wifi-net-device.h"

using namespace ns3;

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief KNOWN-FAILING regression test for issue #1090.
 *
 * Sets up a minimal two-node 802.11n network (one AP, one STA) on a
 * SingleModelSpectrumChannel in the 2.4 GHz band, sends a single packet, and
 * asserts that the simulation completes with at least one received packet.
 *
 * On the current master this aborts inside SpectrumWifiPhy::StartRx() because
 * the HT TX PSD and the RX SpectrumModel are built with different numbers of
 * spectrum bands; see the file-level comment for details. There is no fix yet,
 * so this test currently fails by design.
 */
class WifiHtSingleModelSpectrumTestCase : public TestCase
{
  public:
    WifiHtSingleModelSpectrumTestCase();

  private:
    void DoRun() override;

    /**
     * Callback invoked by the packet socket server upon receiving a packet.
     *
     * @param packet The received packet.
     * @param from The address the packet was received from.
     */
    void ReceivePacket(Ptr<const Packet> packet, const Address& from);

    uint32_t m_received; ///< number of packets received by the server
};

WifiHtSingleModelSpectrumTestCase::WifiHtSingleModelSpectrumTestCase()
    : TestCase("KNOWN-FAILING (issue #1090): 802.11n on SingleModelSpectrumChannel at 2.4 GHz"),
      m_received(0)
{
}

void
WifiHtSingleModelSpectrumTestCase::ReceivePacket(Ptr<const Packet> /* packet */,
                                                 const Address& /* from */)
{
    ++m_received;
}

void
WifiHtSingleModelSpectrumTestCase::DoRun()
{
    // Make the scenario fully deterministic.
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);
    int64_t streamNumber = 100;

    NodeContainer apNode(1);
    NodeContainer staNode(1);

    // Single-model spectrum channel: it performs no TX->RX spectrum conversion,
    // which is precisely what triggers the band-count mismatch in issue #1090.
    auto spectrumChannel = CreateObject<SingleModelSpectrumChannel>();
    auto lossModel = CreateObject<FriisPropagationLossModel>();
    spectrumChannel->AddPropagationLossModel(lossModel);
    auto delayModel = CreateObject<ConstantSpeedPropagationDelayModel>();
    spectrumChannel->SetPropagationDelayModel(delayModel);

    SpectrumWifiPhyHelper phy;
    phy.SetChannel(spectrumChannel);
    // 2.4 GHz, 20 MHz channel: ChannelSettings {channel, width, band, primary20}.
    phy.Set("ChannelSettings", StringValue("{1, 20, BAND_2_4GHZ, 0}"));

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue("HtMcs0"),
                                 "ControlMode",
                                 StringValue("HtMcs0"));

    const Ssid ssid("wifi-ht-single-model-spectrum");

    WifiMacHelper apMac;
    apMac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid), "BeaconGeneration", BooleanValue(true));

    WifiMacHelper staMac;
    staMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "ActiveProbing", BooleanValue(false));

    NetDeviceContainer apDevice = wifi.Install(phy, apMac, apNode);
    NetDeviceContainer staDevice = wifi.Install(phy, staMac, staNode);

    // Assign fixed streams to random variables in use for determinism.
    streamNumber += WifiHelper::AssignStreams(apDevice, streamNumber);
    streamNumber += WifiHelper::AssignStreams(staDevice, streamNumber);

    // Fixed positions, nodes close together.
    MobilityHelper mobility;
    auto positionAlloc = CreateObject<ListPositionAllocator>();
    positionAlloc->Add(Vector(0.0, 0.0, 0.0));
    positionAlloc->Add(Vector(1.0, 0.0, 0.0));
    mobility.SetPositionAllocator(positionAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(apNode);
    mobility.Install(staNode);

    // Packet socket traffic from STA to AP.
    PacketSocketHelper packetSocket;
    packetSocket.Install(apNode);
    packetSocket.Install(staNode);

    PacketSocketAddress socketAddr;
    socketAddr.SetSingleDevice(staDevice.Get(0)->GetIfIndex());
    socketAddr.SetPhysicalAddress(apDevice.Get(0)->GetAddress());
    socketAddr.SetProtocol(1);

    auto client = CreateObject<PacketSocketClient>();
    client->SetAttribute("PacketSize", UintegerValue(500));
    client->SetAttribute("MaxPackets", UintegerValue(1));
    client->SetAttribute("Interval", TimeValue(Seconds(1)));
    client->SetRemote(socketAddr);
    staNode.Get(0)->AddApplication(client);
    client->SetStartTime(Seconds(1.0));
    client->SetStopTime(Seconds(2.0));

    auto server = CreateObject<PacketSocketServer>();
    server->SetLocal(socketAddr);
    server->TraceConnectWithoutContext(
        "Rx",
        MakeCallback(&WifiHtSingleModelSpectrumTestCase::ReceivePacket, this));
    apNode.Get(0)->AddApplication(server);
    server->SetStartTime(Seconds(0.0));
    server->SetStopTime(Seconds(3.0));

    Simulator::Stop(Seconds(3.0));
    // On current master, Simulator::Run() aborts here due to the band-count
    // mismatch assert in SpectrumWifiPhy::StartRx() (issue #1090).
    Simulator::Run();
    Simulator::Destroy();

    // Once issue #1090 is fixed, the single packet should be delivered.
    NS_TEST_ASSERT_MSG_GT_OR_EQ(m_received,
                                1,
                                "Expected at least one received packet on the "
                                "SingleModelSpectrumChannel 802.11n scenario (issue #1090)");
}

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief KNOWN-FAILING regression test suite for issue #1090.
 *
 * Verifies that an 802.11n configuration can use a SingleModelSpectrumChannel
 * in the 2.4 GHz band. This currently aborts and is expected to pass only once
 * issue #1090 is fixed.
 */
class WifiHtSingleModelSpectrumTestSuite : public TestSuite
{
  public:
    WifiHtSingleModelSpectrumTestSuite();
};

WifiHtSingleModelSpectrumTestSuite::WifiHtSingleModelSpectrumTestSuite()
    : TestSuite("wifi-ht-single-model-spectrum", Type::UNIT)
{
    AddTestCase(new WifiHtSingleModelSpectrumTestCase, TestCase::Duration::QUICK);
}

/// the test suite
static WifiHtSingleModelSpectrumTestSuite g_wifiHtSingleModelSpectrumTestSuite;
