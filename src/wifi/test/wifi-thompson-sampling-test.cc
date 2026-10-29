/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/boolean.h"
#include "ns3/config.h"
#include "ns3/mobility-helper.h"
#include "ns3/double.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/packet-socket-client.h"
#include "ns3/packet-socket-helper.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-phy.h"
#include "ns3/wifi-psdu.h"
#include "ns3/yans-wifi-helper.h"

#include <set>

using namespace ns3;

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief ThompsonSamplingWifiManager uses the non-HT rates of a station that
 * probed before associating.
 *
 * An 802.11a AP with ThompsonSamplingWifiManager sends data to a STA a few
 * meters away, which scans actively, like Linux does: the AP acknowledges its
 * Probe Request with a Probe Response before it knows its supported rates.
 * The AP must use other rates than the 6 Mb/s it starts with.
 */
class ThompsonSamplingProbingStaTest : public TestCase
{
  public:
    ThompsonSamplingProbingStaTest();

  private:
    void DoRun() override;

    /**
     * Callback invoked when the AP PHY starts transmitting a PSDU.
     *
     * @param psduMap the PSDU map
     * @param txVector the TX vector
     * @param txPowerW the TX power in Watts
     */
    void ApTx(WifiConstPsduMap psduMap, WifiTxVector txVector, double txPowerW);

    std::set<std::string> m_dataModes; ///< the modes of the data frames sent by the AP
};

ThompsonSamplingProbingStaTest::ThompsonSamplingProbingStaTest()
    : TestCase("ThompsonSamplingWifiManager uses the rates of a probing station")
{
}

void
ThompsonSamplingProbingStaTest::ApTx(WifiConstPsduMap psduMap,
                                     WifiTxVector txVector,
                                     double txPowerW)
{
    for (const auto& [staId, psdu] : psduMap)
    {
        if (psdu->GetHeader(0).IsData())
        {
            m_dataModes.insert(txVector.GetMode().GetUniqueName());
        }
    }
}

void
ThompsonSamplingProbingStaTest::DoRun()
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    NodeContainer nodes(2); // AP, STA
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);

    YansWifiPhyHelper phy;
    phy.SetChannel(YansWifiChannelHelper::Default().Create());
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211a);
    wifi.SetRemoteStationManager("ns3::ThompsonSamplingWifiManager");
    WifiMacHelper mac;
    Ssid ssid("thompson-sampling");
    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevice = wifi.Install(phy, mac, nodes.Get(0));
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "ActiveProbing", BooleanValue(true));
    NetDeviceContainer staDevice = wifi.Install(phy, mac, nodes.Get(1));
    WifiHelper::AssignStreams(apDevice, 100);
    WifiHelper::AssignStreams(staDevice, 200);

    PacketSocketHelper packetSocket;
    packetSocket.Install(nodes);
    PacketSocketAddress socket;
    socket.SetSingleDevice(apDevice.Get(0)->GetIfIndex());
    socket.SetPhysicalAddress(staDevice.Get(0)->GetAddress());
    socket.SetProtocol(1);
    auto client = CreateObject<PacketSocketClient>();
    client->SetAttribute("PacketSize", UintegerValue(1000));
    client->SetAttribute("MaxPackets", UintegerValue(500));
    client->SetAttribute("Interval", TimeValue(MilliSeconds(1)));
    client->SetRemote(socket);
    nodes.Get(0)->AddApplication(client);
    client->SetStartTime(Seconds(0.5));

    DynamicCast<WifiNetDevice>(apDevice.Get(0))
        ->GetPhy()
        ->TraceConnectWithoutContext(
            "PhyTxPsduBegin",
            MakeCallback(&ThompsonSamplingProbingStaTest::ApTx, this));

    Simulator::Stop(Seconds(1.5));
    Simulator::Run();
    Simulator::Destroy();

    NS_TEST_EXPECT_MSG_GT(m_dataModes.size(),
                          1,
                          "The data frames used " << m_dataModes.size() << " rate(s)");
}

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief ThompsonSamplingWifiManager at a reduced channel width (@issueid{1194}).
 *
 * A VHT AP on a 40 MHz channel sends to a STA a few meters away, with a single
 * spatial stream and RTS/CTS, while two other devices, which the AP hears but which do
 * not hear the AP (they transmit with a higher power, farther away), keep the secondary
 * 20 MHz channel busy: the AP then sends at 20 MHz, where the MCS chosen for 40 MHz may
 * be forbidden (VHT MCS 9 with one spatial stream). The manager must use an MCS allowed
 * at 20 MHz instead of aborting.
 */
class ThompsonSamplingReducedWidthTest : public TestCase
{
  public:
    ThompsonSamplingReducedWidthTest();

  private:
    void DoRun() override;

    /**
     * Callback invoked when the AP PHY starts transmitting a PSDU.
     *
     * @param psduMap the PSDU map
     * @param txVector the TX vector
     * @param txPowerW the TX power in Watts
     */
    void ApTx(WifiConstPsduMap psduMap, WifiTxVector txVector, double txPowerW);

    uint32_t m_narrow{0};  ///< data frames sent by the AP at 20 MHz
    uint32_t m_invalid{0}; ///< data frames sent with a forbidden MCS
};

ThompsonSamplingReducedWidthTest::ThompsonSamplingReducedWidthTest()
    : TestCase("ThompsonSamplingWifiManager at a reduced channel width")
{
}

void
ThompsonSamplingReducedWidthTest::ApTx(WifiConstPsduMap psduMap,
                                       WifiTxVector txVector,
                                       double txPowerW)
{
    if (!psduMap.begin()->second->GetHeader(0).IsQosData() ||
        txVector.GetModulationClass() != WIFI_MOD_CLASS_VHT)
    {
        return;
    }
    if (txVector.GetChannelWidth() == MHz_u{20})
    {
        ++m_narrow;
    }
    if (!txVector.GetMode().IsAllowed(txVector.GetChannelWidth(), txVector.GetNss()))
    {
        ++m_invalid;
    }
}

void
ThompsonSamplingReducedWidthTest::DoRun()
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    NodeContainer nodes(4); // AP, STA, two devices on the secondary channel
    MobilityHelper mobility;
    auto positions = CreateObject<ListPositionAllocator>();
    positions->Add(Vector(0, 0, 0));   // AP
    positions->Add(Vector(5, 0, 0));   // STA
    positions->Add(Vector(100, 0, 0)); // the AP hears these two, they do not hear it
    positions->Add(Vector(101, 0, 0));
    mobility.SetPositionAllocator(positions);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);

    auto channel = CreateObject<MultiModelSpectrumChannel>();
    channel->AddPropagationLossModel(CreateObject<FriisPropagationLossModel>());
    SpectrumWifiPhyHelper phy;
    phy.SetChannel(channel);
    phy.Set("TxPowerStart", DoubleValue(0));
    phy.Set("TxPowerEnd", DoubleValue(0));

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211ac);
    // RTS/CTS on every frame: protected at 20 MHz while the secondary channel is busy
    wifi.SetRemoteStationManager("ns3::ThompsonSamplingWifiManager",
                                 "RtsCtsThreshold",
                                 UintegerValue(0));
    WifiMacHelper mac;
    const Ssid ssid("thompson-sampling-width");
    phy.Set("ChannelSettings", StringValue("{38, 40, BAND_5GHZ, 0}"));
    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    auto apDevice = wifi.Install(phy, mac, nodes.Get(0));
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    auto staDevice = wifi.Install(phy, mac, nodes.Get(1));

    // the secondary 20 MHz channel, kept busy
    WifiHelper busyWifi;
    busyWifi.SetStandard(WIFI_STANDARD_80211a);
    // long frames, so that the secondary channel is often busy when the AP gets access
    busyWifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                     "DataMode",
                                     StringValue("OfdmRate6Mbps"));
    phy.Set("ChannelSettings", StringValue("{40, 20, BAND_5GHZ, 0}"));
    phy.Set("TxPowerStart", DoubleValue(30));
    phy.Set("TxPowerEnd", DoubleValue(30));
    mac.SetType("ns3::AdhocWifiMac");
    auto busyDevices = busyWifi.Install(phy, mac, NodeContainer(nodes.Get(2), nodes.Get(3)));

    WifiHelper::AssignStreams(apDevice, 100);
    WifiHelper::AssignStreams(staDevice, 200);
    WifiHelper::AssignStreams(busyDevices, 300);

    PacketSocketHelper packetSocket;
    packetSocket.Install(nodes);
    const auto flow = [&nodes](Ptr<NetDevice> from, Ptr<NetDevice> to, Time interval) {
        PacketSocketAddress socket;
        socket.SetSingleDevice(from->GetIfIndex());
        socket.SetPhysicalAddress(to->GetAddress());
        socket.SetProtocol(1);
        auto client = CreateObject<PacketSocketClient>();
        client->SetAttribute("PacketSize", UintegerValue(1000));
        client->SetAttribute("MaxPackets", UintegerValue(0));
        client->SetAttribute("Interval", TimeValue(interval));
        client->SetRemote(socket);
        from->GetNode()->AddApplication(client);
        client->SetStartTime(Seconds(0.5));
        client->SetStopTime(Seconds(1.5));
    };
    flow(apDevice.Get(0), staDevice.Get(0), MicroSeconds(100));
    flow(busyDevices.Get(0), busyDevices.Get(1), MicroSeconds(50));

    DynamicCast<WifiNetDevice>(apDevice.Get(0))
        ->GetPhy()
        ->TraceConnectWithoutContext(
            "PhyTxPsduBegin",
            MakeCallback(&ThompsonSamplingReducedWidthTest::ApTx, this));

    Simulator::Stop(Seconds(1.5));
    Simulator::Run();
    Simulator::Destroy();

    NS_TEST_EXPECT_MSG_GT(m_narrow, 0, "The AP never sent at 20 MHz");
    NS_TEST_EXPECT_MSG_EQ(m_invalid, 0, "Data frames sent with a forbidden MCS");
}

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief ThompsonSamplingWifiManager test suite
 */
class ThompsonSamplingTestSuite : public TestSuite
{
  public:
    ThompsonSamplingTestSuite();
};

ThompsonSamplingTestSuite::ThompsonSamplingTestSuite()
    : TestSuite("wifi-thompson-sampling", Type::UNIT)
{
    AddTestCase(new ThompsonSamplingProbingStaTest, TestCase::Duration::QUICK);
    AddTestCase(new ThompsonSamplingReducedWidthTest, TestCase::Duration::QUICK);
}

static ThompsonSamplingTestSuite g_thompsonSamplingTestSuite; ///< the test suite
