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

namespace
{

/**
 * Serialize an information element.
 *
 * @param element the element
 * @return its octets, Element ID and Length included
 */
std::vector<uint8_t>
Serialize(const WifiInformationElement& element)
{
    Buffer buffer;
    buffer.AddAtStart(element.GetSerializedSize());
    element.Serialize(buffer.Begin());
    std::vector<uint8_t> octets(buffer.GetSize());
    buffer.CopyData(octets.data(), octets.size());
    return octets;
}

/**
 * Deserialize an information element.
 *
 * @param element the element
 * @param octets its octets, Element ID and Length included
 */
void
Deserialize(WifiInformationElement& element, const std::vector<uint8_t>& octets)
{
    Buffer buffer;
    buffer.AddAtStart(octets.size());
    buffer.Begin().Write(octets.data(), octets.size());
    element.Deserialize(buffer.Begin());
}

} // namespace

/**
 * @ingroup dot11s-test
 *
 * @brief Mesh Peering Management element format
 *
 * The element carries the Mesh Peering Protocol Identifier, the Local Link ID, then the Peer Link
 * ID in Confirm frames, and the Peer Link ID (if known) and the Reason Code in Close frames, all
 * little endian (IEEE 802.11-2012, 8.4.2.104); it has no subtype, which is the frame's. The
 * expected octets are those of the elements sent by Linux's mac80211 (Linux 6.12).
 */
class PeeringManagementElementTest : public TestCase
{
  public:
    PeeringManagementElementTest()
        : TestCase("Mesh Peering Management element format")
    {
    }

    void DoRun() override;
};

void
PeeringManagementElementTest::DoRun()
{
    // Open from a mesh STA with local link ID 0xa99f
    const std::vector<uint8_t> open{0x75, 0x04, 0x00, 0x00, 0x9f, 0xa9};
    // Confirm from a mesh STA with local link ID 0xc373, peering with link ID 0xa99f
    const std::vector<uint8_t> confirm{0x75, 0x06, 0x00, 0x00, 0x73, 0xc3, 0x9f, 0xa9};
    // Close, with and without the peer link ID, for MESH-CLOSE-RCVD (55)
    const std::vector<uint8_t> close{0x75, 0x08, 0x00, 0x00, 0x73, 0xc3, 0x9f, 0xa9, 0x37, 0x00};
    const std::vector<uint8_t> closeNoPeer{0x75, 0x06, 0x00, 0x00, 0x73, 0xc3, 0x37, 0x00};

    IePeerManagement element;
    element.SetPeerOpen(0xa99f);
    NS_TEST_EXPECT_MSG_EQ((Serialize(element) == open), true, "Wrong Open element");
    element.SetPeerConfirm(0xc373, 0xa99f);
    NS_TEST_EXPECT_MSG_EQ((Serialize(element) == confirm), true, "Wrong Confirm element");
    element.SetPeerClose(0xc373, 0xa99f, REASON11S_MESH_CLOSE_RCVD);
    NS_TEST_EXPECT_MSG_EQ((Serialize(element) == close), true, "Wrong Close element");

    IePeerManagement received;
    Deserialize(received, open);
    received.SetSubtype(IePeerManagement::PEER_OPEN);
    NS_TEST_EXPECT_MSG_EQ(received.GetLocalLinkId(), 0xa99f, "Wrong local link ID of Open");

    Deserialize(received, confirm);
    received.SetSubtype(IePeerManagement::PEER_CONFIRM);
    NS_TEST_EXPECT_MSG_EQ(received.GetLocalLinkId(), 0xc373, "Wrong local link ID of Confirm");
    NS_TEST_EXPECT_MSG_EQ(received.GetPeerLinkId(), 0xa99f, "Wrong peer link ID of Confirm");

    Deserialize(received, close);
    received.SetSubtype(IePeerManagement::PEER_CLOSE);
    NS_TEST_EXPECT_MSG_EQ(received.GetPeerLinkId(), 0xa99f, "Wrong peer link ID of Close");
    NS_TEST_EXPECT_MSG_EQ(received.GetReasonCode(),
                          REASON11S_MESH_CLOSE_RCVD,
                          "Wrong reason code of Close");

    Deserialize(received, closeNoPeer);
    received.SetSubtype(IePeerManagement::PEER_CLOSE);
    NS_TEST_EXPECT_MSG_EQ(received.GetLocalLinkId(), 0xc373, "Wrong local link ID of Close");
    NS_TEST_EXPECT_MSG_EQ(received.GetReasonCode(),
                          REASON11S_MESH_CLOSE_RCVD,
                          "Wrong reason code of Close without peer link ID");
    NS_TEST_EXPECT_MSG_EQ((Serialize(received) == closeNoPeer),
                          true,
                          "Close without peer link ID not serialized back as received");
}

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
        AddTestCase(new PeeringManagementElementTest, TestCase::Duration::QUICK);
    }
};

static MeshPeeringConformanceSuite g_meshPeeringConformanceSuite; ///< the test suite
