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
#include "ns3/packet-socket-client.h"
#include "ns3/packet-socket-helper.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/ssid.h"
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
}

static ThompsonSamplingTestSuite g_thompsonSamplingTestSuite; ///< the test suite
