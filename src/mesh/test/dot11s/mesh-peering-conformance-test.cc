/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/buffer.h"
#include "ns3/double.h"
#include "ns3/ie-dot11s-peer-management.h"
#include "ns3/ie-dot11s-perr.h"
#include "ns3/ie-dot11s-prep.h"
#include "ns3/mesh-helper.h"
#include "ns3/mesh-point-device.h"
#include "ns3/mobility-helper.h"
#include "ns3/node-container.h"
#include "ns3/packet.h"
#include "ns3/peer-link-frame.h"
#include "ns3/peer-management-protocol.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/wifi-mac-header.h"
#include "ns3/wifi-mac-trailer.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-phy.h"
#include "ns3/yans-wifi-helper.h"

#include <algorithm>
#include <vector>

using namespace ns3;
using namespace ns3::dot11s;

/**
 * @ingroup dot11s-test
 *
 * @brief Peering with a mesh STA whose beacons have no Beacon Timing element
 *
 * The Beacon Timing element is optional in beacons (IEEE 802.11-2012, Table 8-20): a mesh STA
 * that does not perform beacon collision avoidance, such as Linux's mac80211, does not send it.
 * The test sets up two mesh STAs, of which only one performs beacon collision avoidance, and
 * checks that the other one's beacons are received and that the two establish a peering.
 */
class BeaconWithoutTimingTest : public TestCase
{
  public:
    BeaconWithoutTimingTest()
        : TestCase("Peering with a mesh STA that sends no Beacon Timing element")
    {
    }

    void DoRun() override;

  private:
    /**
     * LinkOpen trace sink.
     *
     * @param src the local mesh point address
     * @param dst the peer mesh point address
     */
    void LinkOpen(Mac48Address src, Mac48Address dst);

    uint32_t m_linksOpen{0}; ///< peerings established, as seen by the STA with collision avoidance
};

void
BeaconWithoutTimingTest::LinkOpen(Mac48Address src, Mac48Address dst)
{
    m_linksOpen++;
}

void
BeaconWithoutTimingTest::DoRun()
{
    NodeContainer nodes;
    nodes.Create(2);

    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "DeltaX",
                                  DoubleValue(50),
                                  "GridWidth",
                                  UintegerValue(2));
    mobility.Install(nodes);

    YansWifiPhyHelper phy;
    phy.SetChannel(YansWifiChannelHelper::Default().Create());
    MeshHelper mesh = MeshHelper::Default();
    mesh.SetStackInstaller("ns3::Dot11sStack");
    NetDeviceContainer devices = mesh.Install(phy, nodes);
    mesh.AssignStreams(devices, 1);

    auto withTiming = devices.Get(0)->GetObject<PeerManagementProtocol>();
    auto withoutTiming = devices.Get(1)->GetObject<PeerManagementProtocol>();
    NS_TEST_ASSERT_MSG_NE(withTiming, nullptr, "No peer management protocol");
    NS_TEST_ASSERT_MSG_NE(withoutTiming, nullptr, "No peer management protocol");
    withoutTiming->SetBeaconCollisionAvoidance(false);
    withTiming->TraceConnectWithoutContext("LinkOpen",
                                           MakeCallback(&BeaconWithoutTimingTest::LinkOpen, this));

    Simulator::Stop(Seconds(5));
    Simulator::Run();
    Simulator::Destroy();

    NS_TEST_EXPECT_MSG_EQ(m_linksOpen, 1, "The two mesh STAs should have established a peering");
}

/**
 * @ingroup dot11s-test
 *
 * @brief Peering conformance test suite
 */
class MeshPeeringConformanceSuite : public TestSuite
{
  public:
    MeshPeeringConformanceSuite()
        : TestSuite("devices-mesh-dot11s-peering-conformance", Type::UNIT)
    {
        AddTestCase(new BeaconWithoutTimingTest, TestCase::Duration::QUICK);
    }
};

static MeshPeeringConformanceSuite g_meshPeeringConformanceSuite; ///< the test suite
