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
 * @brief Mesh Peering Open and Confirm frame bodies
 *
 * The bodies of Mesh Peering Open and Confirm frames sent by Linux's mac80211 (Linux 6.12),
 * after the Category and Self-protected Action fields, up to the Mesh Configuration element
 * (IEEE 802.11-2012, 8.6.16.2.2 and 8.6.16.3.2). They are deserialized, checked and serialized
 * back to the same octets. Unlike the Open frame, the Confirm frame carries an AID, and both carry
 * a Mesh ID element before the Mesh Configuration element.
 */
class PeeringFrameTest : public TestCase
{
  public:
    PeeringFrameTest()
        : TestCase("Mesh Peering Open and Confirm frame bodies")
    {
    }

    void DoRun() override;

  private:
    /**
     * Deserialize a frame body, check it is all read, and serialize it back.
     *
     * @tparam T the frame body header type
     * @param octets the frame body
     * @param frame the deserialized frame body
     * @return the octets of the frame body serialized back
     */
    template <typename T>
    std::vector<uint8_t> RoundTrip(const std::vector<uint8_t>& octets, T& frame);
};

template <typename T>
std::vector<uint8_t>
PeeringFrameTest::RoundTrip(const std::vector<uint8_t>& octets, T& frame)
{
    Ptr<Packet> packet = Create<Packet>(octets.data(), octets.size());
    uint32_t read = packet->RemoveHeader(frame);
    NS_TEST_EXPECT_MSG_EQ(read, octets.size(), "Frame body not read entirely");
    Ptr<Packet> back = Create<Packet>();
    back->AddHeader(frame);
    std::vector<uint8_t> serialized(back->GetSize());
    back->CopyData(serialized.data(), serialized.size());
    return serialized;
}

void
PeeringFrameTest::DoRun()
{
    // Capability, Supported Rates (6, 12 and 24 Mb/s basic), Mesh ID "mesh", Mesh Configuration
    const std::vector<uint8_t> open{0x00, 0x00, 0x01, 0x08, 0x8c, 0x12, 0x98, 0x24, 0xb0,
                                    0x48, 0x60, 0x6c, 0x72, 0x04, 0x6d, 0x65, 0x73, 0x68,
                                    0x71, 0x07, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x09};
    // Capability, AID 1, then as the Open frame
    const std::vector<uint8_t> confirm{0x00, 0x00, 0x01, 0x00, 0x01, 0x08, 0x8c, 0x12, 0x98, 0x24,
                                       0xb0, 0x48, 0x60, 0x6c, 0x72, 0x04, 0x6d, 0x65, 0x73, 0x68,
                                       0x71, 0x07, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x09};

    PeerLinkOpenStart openFrame;
    NS_TEST_EXPECT_MSG_EQ((RoundTrip(open, openFrame) == open),
                          true,
                          "Open frame body not serialized back as received");
    NS_TEST_EXPECT_MSG_EQ(openFrame.GetFields().meshId.IsEqual(IeMeshId("mesh")),
                          true,
                          "Wrong Mesh ID of Open frame");

    PeerLinkConfirmStart confirmFrame;
    NS_TEST_EXPECT_MSG_EQ((RoundTrip(confirm, confirmFrame) == confirm),
                          true,
                          "Confirm frame body not serialized back as received");
    auto fields = confirmFrame.GetFields();
    NS_TEST_EXPECT_MSG_EQ(fields.aid, 1, "Wrong AID of Confirm frame");
    NS_TEST_EXPECT_MSG_EQ(fields.meshId.IsEqual(IeMeshId("mesh")),
                          true,
                          "Wrong Mesh ID of Confirm frame");
    NS_TEST_EXPECT_MSG_EQ(fields.config.IsHWMP(), true, "Wrong path selection protocol");
    NS_TEST_EXPECT_MSG_EQ(fields.config.IsAirtime(), true, "Wrong path selection metric");
}

/**
 * @ingroup dot11s-test
 *
 * @brief Mesh beacon contents
 *
 * A mesh STA peers only with mesh STAs whose beacons carry a Mesh ID and a Mesh Configuration
 * element matching its own, and the same BSS basic rate set (IEEE 802.11-2012, 13.2.4); Linux's
 * mac80211 ignores the beacons of the others. The test checks that the beacons of a mesh STA carry
 * the Mesh ID, then the Mesh Configuration element (Table 8-20), and that 802.11a's mandatory rates
 * (6, 12 and 24 Mb/s) are marked basic in its Supported Rates element.
 */
class BeaconContentTest : public TestCase
{
  public:
    BeaconContentTest()
        : TestCase("Mesh beacons carry the Mesh Configuration element and the basic rates")
    {
    }

    void DoRun() override;

  private:
    /**
     * MonitorSnifferTx trace sink: keep the first beacon.
     *
     * @param packet the frame
     * @param channelFreqMhz the frequency
     * @param txVector the TX vector
     * @param aMpdu the A-MPDU information
     * @param staId the station ID
     */
    void Sniff(Ptr<const Packet> packet,
               uint16_t channelFreqMhz,
               WifiTxVector txVector,
               MpduInfo aMpdu,
               uint16_t staId);

    std::vector<uint8_t> m_beacon; ///< body of the first beacon sent
};

void
BeaconContentTest::Sniff(Ptr<const Packet> packet,
                         uint16_t channelFreqMhz,
                         WifiTxVector txVector,
                         MpduInfo aMpdu,
                         uint16_t staId)
{
    Ptr<Packet> frame = packet->Copy();
    WifiMacHeader header;
    frame->RemoveHeader(header);
    if (!header.IsBeacon() || !m_beacon.empty())
    {
        return;
    }
    WifiMacTrailer fcs;
    frame->RemoveTrailer(fcs);
    m_beacon.resize(frame->GetSize());
    frame->CopyData(m_beacon.data(), m_beacon.size());
}

void
BeaconContentTest::DoRun()
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

    auto mp = DynamicCast<MeshPointDevice>(devices.Get(0));
    auto iface = DynamicCast<WifiNetDevice>(mp->GetInterfaces().at(0));
    iface->GetPhy()->TraceConnectWithoutContext("MonitorSnifferTx",
                                                MakeCallback(&BeaconContentTest::Sniff, this));

    Simulator::Stop(Seconds(3));
    Simulator::Run();
    Simulator::Destroy();

    NS_TEST_ASSERT_MSG_EQ(m_beacon.empty(), false, "No beacon sent");
    // Timestamp, Beacon Interval and Capability, then the elements
    std::vector<std::pair<uint8_t, std::vector<uint8_t>>> elements;
    for (std::size_t i = 12; i + 2 <= m_beacon.size(); i += 2 + m_beacon[i + 1])
    {
        elements.emplace_back(m_beacon[i],
                              std::vector<uint8_t>(m_beacon.begin() + i + 2,
                                                   m_beacon.begin() + i + 2 + m_beacon[i + 1]));
    }
    auto find = [&elements](uint8_t id) {
        return std::find_if(elements.begin(), elements.end(), [id](const auto& e) {
            return e.first == id;
        });
    };

    auto rates = find(IE_SUPPORTED_RATES);
    NS_TEST_ASSERT_MSG_EQ((rates != elements.end()), true, "No Supported Rates element");
    const std::vector<uint8_t> expectedRates{0x8c, 0x12, 0x98, 0x24, 0xb0, 0x48, 0x60, 0x6c};
    NS_TEST_EXPECT_MSG_EQ((rates->second == expectedRates),
                          true,
                          "6, 12 and 24 Mb/s should be the basic rates");

    auto meshId = find(IE_MESH_ID);
    auto config = find(IE_MESH_CONFIGURATION);
    NS_TEST_ASSERT_MSG_EQ((meshId != elements.end()), true, "No Mesh ID element");
    NS_TEST_ASSERT_MSG_EQ((config != elements.end()), true, "No Mesh Configuration element");
    NS_TEST_EXPECT_MSG_LT(meshId - elements.begin(),
                          config - elements.begin(),
                          "The Mesh ID element should precede the Mesh Configuration element");
    // HWMP, airtime metric, no congestion control, neighbor offset synchronization, no
    // authentication (Table 8-174 to 8-178)
    NS_TEST_ASSERT_MSG_EQ(config->second.size(), 7, "Wrong Mesh Configuration element length");
    const std::vector<uint8_t> expectedIds{0x01, 0x01, 0x00, 0x01, 0x00};
    NS_TEST_EXPECT_MSG_EQ(
        (std::vector<uint8_t>(config->second.begin(), config->second.begin() + 5) == expectedIds),
        true,
        "Wrong protocol identifiers in the Mesh Configuration element");
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
        AddTestCase(new PeeringFrameTest, TestCase::Duration::QUICK);
        AddTestCase(new BeaconContentTest, TestCase::Duration::QUICK);
    }
};

static MeshPeeringConformanceSuite g_meshPeeringConformanceSuite; ///< the test suite
