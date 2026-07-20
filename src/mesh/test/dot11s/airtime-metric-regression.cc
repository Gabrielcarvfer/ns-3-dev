/*
 * Copyright (c) 2026 Centre Tecnològic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/airtime-metric.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/mesh-helper.h"
#include "ns3/mesh-point-device.h"
#include "ns3/mesh-wifi-interface-mac.h"
#include "ns3/mobility-helper.h"
#include "ns3/node-container.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-utils.h"
#include "ns3/yans-wifi-helper.h"

using namespace ns3;
using namespace ns3::dot11s;

/**
 * @ingroup dot11s-test
 *
 * @brief Airtime link metric regression test (see issue #1345)
 *
 * Checks that AirtimeLinkMetricCalculator::CalculateMetric() computes the
 * airtime using the WifiTxVector resolved by the remote station manager
 * (carrying the actual preamble type, channel width, number of spatial
 * streams and guard interval) instead of rebuilding a tx vector with a
 * hardcoded long preamble, which is inconsistent with HT and later modes.
 */
class AirtimeMetricTest : public TestCase
{
  public:
    AirtimeMetricTest()
        : TestCase("Airtime link metric uses the resolved WifiTxVector")
    {
    }

    void DoRun() override;
};

void
AirtimeMetricTest::DoRun()
{
    // Build a two-node 802.11n mesh on a 40 MHz channel with a fixed HT rate;
    // the 40 MHz width makes the airtime differ from the one computed with the
    // default-constructed (20 MHz wide, long preamble) tx vector
    NodeContainer nodes;
    nodes.Create(2);
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);

    YansWifiPhyHelper phy;
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    phy.SetChannel(channel.Create());
    phy.Set("ChannelSettings", StringValue("{0, 40, BAND_5GHZ, 0}"));

    MeshHelper mesh = MeshHelper::Default();
    mesh.SetStackInstaller("ns3::Dot11sStack");
    mesh.SetStandard(WIFI_STANDARD_80211n);
    mesh.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue("HtMcs7"),
                                 "ControlMode",
                                 StringValue("HtMcs0"));
    NetDeviceContainer meshDevices = mesh.Install(phy, nodes);

    auto getMac = [](Ptr<NetDevice> dev) {
        Ptr<MeshPointDevice> mp = DynamicCast<MeshPointDevice>(dev);
        NS_ASSERT(mp);
        Ptr<WifiNetDevice> ifdev = DynamicCast<WifiNetDevice>(mp->GetInterfaces().at(0));
        NS_ASSERT(ifdev);
        return DynamicCast<MeshWifiInterfaceMac>(ifdev->GetMac());
    };
    Ptr<MeshWifiInterfaceMac> mac = getMac(meshDevices.Get(0));
    Ptr<MeshWifiInterfaceMac> peerMac = getMac(meshDevices.Get(1));
    NS_TEST_ASSERT_MSG_NE(mac, nullptr, "Mesh interface MAC not found");
    Mac48Address peerAddress = Mac48Address::ConvertFrom(peerMac->GetAddress());

    // Resolve the tx vector the same way the metric calculator does
    WifiMacHeader testHeader;
    testHeader.SetType(WIFI_MAC_DATA);
    testHeader.SetDsFrom();
    testHeader.SetDsTo();
    testHeader.SetQosTid(0);
    testHeader.SetAddr1(peerAddress);
    const WifiTxVector txVector =
        mac->GetWifiRemoteStationManager()->GetDataTxVector(testHeader,
                                                            mac->GetWifiPhy()->GetChannelWidth());

    // The resolved vector must carry an HT mode with an HT preamble
    NS_TEST_ASSERT_MSG_EQ(txVector.GetMode().GetUniqueName(),
                          "HtMcs7",
                          "Unexpected data mode resolved by the station manager");
    NS_TEST_ASSERT_MSG_EQ(+txVector.GetPreambleType(),
                          +WIFI_PREAMBLE_HT_MF,
                          "Unexpected preamble type resolved by the station manager");

    // Expected airtime metric per IEEE 802.11-2012 Section 13.9, computed from
    // the resolved tx vector (frame error rate is zero at this point)
    const uint32_t testFrameSize = 1024 + 6 /*Mesh header*/ + 36 /*802.11 header*/;
    const Time airtime =
        2 * mac->GetWifiPhy()->GetSifs() + 2 * mac->GetWifiPhy()->GetSlot() +
        GetEstimatedAckTxTime(txVector) +
        WifiPhy::CalculateTxDuration(testFrameSize, txVector, mac->GetWifiPhy()->GetPhyBand());
    const auto expectedMetric = static_cast<uint32_t>(airtime.GetMicroSeconds() / 10.24);

    Ptr<AirtimeLinkMetricCalculator> calculator = CreateObject<AirtimeLinkMetricCalculator>();
    const uint32_t metric = calculator->CalculateMetric(peerAddress, mac);
    NS_TEST_ASSERT_MSG_EQ(metric,
                          expectedMetric,
                          "Airtime metric not computed from the resolved WifiTxVector");

    Simulator::Destroy();
}

/**
 * @ingroup dot11s-test
 *
 * @brief Airtime link metric test suite
 */
class AirtimeMetricTestSuite : public TestSuite
{
  public:
    AirtimeMetricTestSuite()
        : TestSuite("devices-mesh-dot11s-airtime-metric", Type::UNIT)
    {
        AddTestCase(new AirtimeMetricTest, TestCase::Duration::QUICK);
    }
};

static AirtimeMetricTestSuite g_airtimeMetricTestSuite; ///< the test suite
