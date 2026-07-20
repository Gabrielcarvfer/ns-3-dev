/*
 * Copyright (c) 2026 Centre Tecnològic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/mesh-helper.h"
#include "ns3/mesh-point-device.h"
#include "ns3/mobility-helper.h"
#include "ns3/node-container.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/yans-wifi-helper.h"

using namespace ns3;

/**
 * @ingroup dot11s-test
 *
 * @brief HWMP multicast forwarding regression test (see issue #485)
 *
 * Multicast frames are group addressed frames and must be flooded through the
 * mesh exactly like broadcast frames (IEEE 802.11-2012, Section 10.35.4).
 * Prior to the fix, HWMP treated any non-broadcast destination as unicast and
 * tried to discover a path towards the multicast group address, which never
 * gets a path reply, so multicast frames were never forwarded.
 *
 * The test builds a three-node chain topology where the outer nodes are out of
 * radio range of each other and checks that a multicast frame originated by
 * the first node is forwarded by the middle node and delivered to the last
 * one.
 */
class HwmpMulticastRegressionTest : public TestCase
{
  public:
    HwmpMulticastRegressionTest()
        : TestCase("HWMP floods multicast frames like broadcast ones"),
          m_received(0)
    {
    }

    void DoRun() override;

  private:
    /**
     * Receive callback registered on the last node of the chain.
     *
     * @param device The receiving device.
     * @param packet The received packet.
     * @param protocol The protocol number.
     * @param source The source address.
     * @return true (packet accepted)
     */
    bool Receive(Ptr<NetDevice> device,
                 Ptr<const Packet> packet,
                 uint16_t protocol,
                 const Address& source);

    uint32_t m_received; ///< number of packets received by the last node
};

bool
HwmpMulticastRegressionTest::Receive(Ptr<NetDevice> device,
                                     Ptr<const Packet> packet,
                                     uint16_t protocol,
                                     const Address& source)
{
    m_received++;
    return true;
}

void
HwmpMulticastRegressionTest::DoRun()
{
    // Three nodes in a chain; only adjacent nodes are within radio range
    NodeContainer nodes;
    nodes.Create(3);
    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX",
                                  DoubleValue(0.0),
                                  "MinY",
                                  DoubleValue(0.0),
                                  "DeltaX",
                                  DoubleValue(100.0),
                                  "GridWidth",
                                  UintegerValue(3),
                                  "LayoutType",
                                  StringValue("RowFirst"));
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);

    YansWifiChannelHelper channel;
    channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    channel.AddPropagationLoss("ns3::RangePropagationLossModel", "MaxRange", DoubleValue(120.0));
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());

    MeshHelper mesh = MeshHelper::Default();
    mesh.SetStackInstaller("ns3::Dot11sStack");
    NetDeviceContainer meshDevices = mesh.Install(phy, nodes);

    Ptr<MeshPointDevice> src = DynamicCast<MeshPointDevice>(meshDevices.Get(0));
    Ptr<MeshPointDevice> dst = DynamicCast<MeshPointDevice>(meshDevices.Get(2));
    dst->SetReceiveCallback(MakeCallback(&HwmpMulticastRegressionTest::Receive, this));

    // Send a multicast frame after the peer links have been established
    const Mac48Address multicastGroup("01:00:5e:00:00:01");
    Simulator::Schedule(Seconds(3), [src, multicastGroup]() {
        src->Send(Create<Packet>(100), multicastGroup, 0x0800);
    });

    Simulator::Stop(Seconds(6));
    Simulator::Run();
    Simulator::Destroy();

    NS_TEST_ASSERT_MSG_GT(m_received, 0, "Multicast frame was not forwarded through the mesh");
}

/**
 * @ingroup dot11s-test
 *
 * @brief HWMP multicast forwarding test suite
 */
class HwmpMulticastRegressionSuite : public TestSuite
{
  public:
    HwmpMulticastRegressionSuite()
        : TestSuite("devices-mesh-dot11s-multicast", Type::UNIT)
    {
        AddTestCase(new HwmpMulticastRegressionTest, TestCase::Duration::QUICK);
    }
};

static HwmpMulticastRegressionSuite g_hwmpMulticastRegressionSuite; ///< the test suite
