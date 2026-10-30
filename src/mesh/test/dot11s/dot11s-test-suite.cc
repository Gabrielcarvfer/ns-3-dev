/*
 * Copyright (c) 2009 IITP RAS
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Pavel Boyko <boyko@iitp.ru>
 */
#include "ns3/dot11s-mac-header.h"
#include "ns3/hwmp-protocol-mac.h"
#include "ns3/hwmp-protocol.h"
#include "ns3/hwmp-rtable.h"
#include "ns3/ie-dot11s-id.h"
#include "ns3/ie-dot11s-peer-management.h"
#include "ns3/mesh-helper.h"
#include "ns3/mesh-information-element-vector.h"
#include "ns3/mesh-point-device.h"
#include "ns3/mgt-action-headers.h"
#include "ns3/mgt-headers.h"
#include "ns3/mobility-helper.h"
#include "ns3/packet.h"
#include "ns3/peer-link-frame.h"
#include "ns3/peer-management-protocol.h"
#include "ns3/simulator.h"
#include "ns3/test.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-phy.h"
#include "ns3/wifi-psdu.h"
#include "ns3/yans-wifi-helper.h"

#include <iomanip>
#include <sstream>

using namespace ns3;
using namespace dot11s;

/**
 * @ingroup mesh-test
 * @defgroup dot11s-test dot11s sub-module tests
 */

/**
 * @ingroup dot11s-test
 *
 * @brief Built-in self test for MeshHeader
 */
struct MeshHeaderTest : public TestCase
{
    MeshHeaderTest()
        : TestCase("Dot11sMeshHeader roundtrip serialization")
    {
    }

    void DoRun() override;
};

void
MeshHeaderTest::DoRun()
{
    {
        MeshHeader a;
        a.SetAddressExt(3);
        a.SetAddr4(Mac48Address("11:22:33:44:55:66"));
        a.SetAddr5(Mac48Address("11:00:33:00:55:00"));
        a.SetAddr6(Mac48Address("00:22:00:44:00:66"));
        a.SetMeshTtl(122);
        a.SetMeshSeqno(321);
        Ptr<Packet> packet = Create<Packet>();
        packet->AddHeader(a);
        MeshHeader b;
        packet->RemoveHeader(b);
        NS_TEST_ASSERT_MSG_EQ(a, b, "Mesh header roundtrip serialization works, 3 addresses");
    }
    {
        MeshHeader a;
        a.SetAddressExt(2);
        a.SetAddr5(Mac48Address("11:00:33:00:55:00"));
        a.SetAddr6(Mac48Address("00:22:00:44:00:66"));
        a.SetMeshTtl(122);
        a.SetMeshSeqno(321);
        Ptr<Packet> packet = Create<Packet>();
        packet->AddHeader(a);
        MeshHeader b;
        packet->RemoveHeader(b);
        NS_TEST_ASSERT_MSG_EQ(a, b, "Mesh header roundtrip serialization works, 2 addresses");
    }
    {
        MeshHeader a;
        a.SetAddressExt(1);
        a.SetAddr4(Mac48Address("11:22:33:44:55:66"));
        a.SetMeshTtl(122);
        a.SetMeshSeqno(321);
        Ptr<Packet> packet = Create<Packet>();
        packet->AddHeader(a);
        MeshHeader b;
        packet->RemoveHeader(b);
        NS_TEST_ASSERT_MSG_EQ(a, b, "Mesh header roundtrip serialization works, 1 address");
    }
}

/**
 * @ingroup mesh-test
 *
 * @brief Unit test for HwmpRtable
 */
class HwmpRtableTest : public TestCase
{
  public:
    HwmpRtableTest();
    void DoRun() override;

  private:
    /// Test Add apth and lookup path;
    void TestLookup();

    /// Test add path and try to lookup after entry has expired
    void TestAddPath();
    /// Test add path and try to lookup after entry has expired
    void TestExpire();

    /// Test add precursors and find precursor list in rtable
    void TestPrecursorAdd();
    /// Test add precursors and find precursor list in rtable
    void TestPrecursorFind();

  private:
    Mac48Address dst;                     ///< destination address
    Mac48Address hop;                     ///< hop address
    uint32_t iface;                       ///< interface
    uint32_t metric;                      ///< metric
    uint32_t seqnum;                      ///< sequence number
    Time expire;                          ///< expiration time
    Ptr<HwmpRtable> table;                ///< tab;e
    std::vector<Mac48Address> precursors; ///< precursors
};

HwmpRtableTest::HwmpRtableTest()
    : TestCase("HWMP routing table"),
      dst("01:00:00:01:00:01"),
      hop("01:00:00:01:00:03"),
      iface(8010),
      metric(10),
      seqnum(1),
      expire(Seconds(10))
{
    precursors.emplace_back("00:10:20:30:40:50");
    precursors.emplace_back("00:11:22:33:44:55");
    precursors.emplace_back("00:01:02:03:04:05");
}

void
HwmpRtableTest::TestLookup()
{
    HwmpRtable::LookupResult correct(hop, iface, metric, seqnum);

    // Reactive path
    table->AddReactivePath(dst, hop, iface, metric, expire, seqnum);
    NS_TEST_EXPECT_MSG_EQ((table->LookupReactive(dst) == correct), true, "Reactive lookup works");
    table->DeleteReactivePath(dst);
    NS_TEST_EXPECT_MSG_EQ(table->LookupReactive(dst).IsValid(), false, "Reactive lookup works");

    // Proactive
    table->AddProactivePath(metric, dst, hop, iface, expire, seqnum);
    NS_TEST_EXPECT_MSG_EQ((table->LookupProactive() == correct), true, "Proactive lookup works");
    table->DeleteProactivePath(dst);
    NS_TEST_EXPECT_MSG_EQ(table->LookupProactive().IsValid(), false, "Proactive lookup works");
}

void
HwmpRtableTest::TestAddPath()
{
    table->AddReactivePath(dst, hop, iface, metric, expire, seqnum);
    table->AddProactivePath(metric, dst, hop, iface, expire, seqnum);
}

void
HwmpRtableTest::TestExpire()
{
    // this is assumed to be called when path records are already expired
    HwmpRtable::LookupResult correct(hop, iface, metric, seqnum);
    NS_TEST_EXPECT_MSG_EQ((table->LookupReactiveExpired(dst) == correct),
                          true,
                          "Reactive expiration works");
    NS_TEST_EXPECT_MSG_EQ((table->LookupProactiveExpired() == correct),
                          true,
                          "Proactive expiration works");

    NS_TEST_EXPECT_MSG_EQ(table->LookupReactive(dst).IsValid(), false, "Reactive expiration works");
    NS_TEST_EXPECT_MSG_EQ(table->LookupProactive().IsValid(), false, "Proactive expiration works");
}

void
HwmpRtableTest::TestPrecursorAdd()
{
    for (auto i = precursors.begin(); i != precursors.end(); i++)
    {
        table->AddPrecursor(dst, iface, *i, Seconds(100));
        // Check that duplicates are filtered
        table->AddPrecursor(dst, iface, *i, Seconds(100));
    }
}

void
HwmpRtableTest::TestPrecursorFind()
{
    HwmpRtable::PrecursorList precursorList = table->GetPrecursors(dst);
    NS_TEST_EXPECT_MSG_EQ(precursors.size(), precursorList.size(), "Precursors size works");
    for (unsigned i = 0; i < precursors.size(); i++)
    {
        NS_TEST_EXPECT_MSG_EQ(precursorList[i].first, iface, "Precursors lookup works");
        NS_TEST_EXPECT_MSG_EQ(precursorList[i].second, precursors[i], "Precursors lookup works");
    }
}

void
HwmpRtableTest::DoRun()
{
    table = CreateObject<HwmpRtable>();

    Simulator::Schedule(Seconds(0), &HwmpRtableTest::TestLookup, this);
    Simulator::Schedule(Seconds(1), &HwmpRtableTest::TestAddPath, this);
    Simulator::Schedule(Seconds(2), &HwmpRtableTest::TestPrecursorAdd, this);
    Simulator::Schedule(expire + Seconds(2), &HwmpRtableTest::TestExpire, this);
    Simulator::Schedule(expire + Seconds(3), &HwmpRtableTest::TestPrecursorFind, this);

    Simulator::Run();
    Simulator::Destroy();
}

//-----------------------------------------------------------------------------
/// Built-in self test for PeerLinkFrameStart
struct PeerLinkFrameStartTest : public TestCase
{
    PeerLinkFrameStartTest()
        : TestCase("PeerLinkFrames (open, confirm, close) unit tests")
    {
    }

    void DoRun() override;
};

void
PeerLinkFrameStartTest::DoRun()
{
    {
        PeerLinkOpenStart a;
        PeerLinkOpenStart::PlinkOpenStartFields fields;
        fields.capability = 0;
        fields.meshId = IeMeshId("qwertyuiop");
        a.SetPlinkOpenStart(fields);
        Ptr<Packet> packet = Create<Packet>();
        packet->AddHeader(a);
        PeerLinkOpenStart b;
        packet->RemoveHeader(b);
        NS_TEST_EXPECT_MSG_EQ(a, b, "PEER_LINK_OPEN works");
    }
    {
        PeerLinkConfirmStart a;
        PeerLinkConfirmStart::PlinkConfirmStartFields fields;
        fields.capability = 0;
        fields.aid = 1234;
        a.SetPlinkConfirmStart(fields);
        Ptr<Packet> packet = Create<Packet>();
        packet->AddHeader(a);
        PeerLinkConfirmStart b;
        packet->RemoveHeader(b);
        NS_TEST_EXPECT_MSG_EQ(a, b, "PEER_LINK_CONFIRM works");
    }
    {
        PeerLinkCloseStart a;
        PeerLinkCloseStart::PlinkCloseStartFields fields;
        fields.meshId = IeMeshId("qqq");
        a.SetPlinkCloseStart(fields);
        Ptr<Packet> packet = Create<Packet>();
        packet->AddHeader(a);
        PeerLinkCloseStart b;
        packet->RemoveHeader(b);
        NS_TEST_EXPECT_MSG_EQ(a, b, "PEER_LINK_CLOSE works");
    }
}

/**
 * @ingroup mesh-test
 *
 * @brief Mesh Peering Open and Confirm frames of Linux (mac80211), whose Mesh
 * Peering Management element has the format of IEEE 802.11-2020 9.4.2.102, and the
 * elements of ns-3 in that format.
 */
struct PeerLinkFrameLinuxTest : public TestCase
{
    PeerLinkFrameLinuxTest()
        : TestCase("Mesh peering frames of Linux and IEEE 802.11-2020 format")
    {
    }

    /**
     * Deserialize the body of a Self-protected Action frame.
     *
     * @param hex the frame body, from the Category field
     * @return the packet
     */
    static Ptr<Packet> FromHex(std::string hex)
    {
        std::vector<uint8_t> bytes;
        for (std::size_t k = 0; k < hex.size(); k += 2)
        {
            bytes.push_back(static_cast<uint8_t>(std::stoi(hex.substr(k, 2), nullptr, 16)));
        }
        return Create<Packet>(bytes.data(), bytes.size());
    }

    /**
     * Serialize an element.
     *
     * @param element the element
     * @return its octets, as hex
     */
    static std::string ToHex(const WifiInformationElement& element)
    {
        Buffer buffer;
        buffer.AddAtStart(element.GetSerializedSize());
        element.Serialize(buffer.Begin());
        std::vector<uint8_t> bytes(buffer.GetSize());
        buffer.CopyData(bytes.data(), bytes.size());
        std::ostringstream os;
        for (auto b : bytes)
        {
            os << std::hex << std::setw(2) << std::setfill('0') << +b;
        }
        return os.str();
    }

    void DoRun() override
    {
        // a Mesh Peering Open of Linux 6.12: Capability, Supported Rates, Mesh ID,
        // Mesh Configuration and Mesh Peering Management (protocol 0, local link 0x453e)
        auto open = FromHex("0f01000001088c129824b048606c72046d657368710701010001000009"
                            "750400003e45");
        WifiActionHeader action;
        open->RemoveHeader(action);
        NS_TEST_EXPECT_MSG_EQ(action.GetAction().selfProtectedAction,
                              WifiActionHeader::PEER_LINK_OPEN,
                              "Not a Mesh Peering Open");
        PeerLinkOpenStart openStart;
        open->RemoveHeader(openStart);
        NS_TEST_EXPECT_MSG_EQ(openStart.GetFields().meshId.IsEqual(IeMeshId("mesh")),
                              true,
                              "Wrong Mesh ID");
        MeshInformationElementVector elements;
        open->RemoveHeader(elements, open->GetSize());
        auto peering =
            DynamicCast<IePeerManagement>(elements.FindFirst(IE_MESH_PEERING_MANAGEMENT));
        NS_TEST_ASSERT_MSG_NE(peering, nullptr, "No Mesh Peering Management element");
        NS_TEST_EXPECT_MSG_EQ(peering->GetLocalLinkId(), 0x453e, "Wrong local link ID");

        // a Mesh Peering Confirm of Linux: AID 1, Mesh ID before Mesh Configuration,
        // local link 0x453e, peer link 0xbeea
        auto confirm = FromHex("0f020000010001088c129824b048606c72046d657368710701010001000009"
                               "750600003e45eabe");
        confirm->RemoveHeader(action);
        PeerLinkConfirmStart confirmStart;
        confirm->RemoveHeader(confirmStart);
        NS_TEST_EXPECT_MSG_EQ(confirmStart.GetFields().aid, 1, "Wrong AID");
        MeshInformationElementVector confirmElements;
        confirm->RemoveHeader(confirmElements, confirm->GetSize());
        peering = DynamicCast<IePeerManagement>(
            confirmElements.FindFirst(IE_MESH_PEERING_MANAGEMENT));
        NS_TEST_ASSERT_MSG_NE(peering, nullptr, "No Mesh Peering Management element");
        NS_TEST_EXPECT_MSG_EQ(peering->GetLocalLinkId(), 0x453e, "Wrong local link ID");
        NS_TEST_EXPECT_MSG_EQ(peering->GetPeerLinkId(), 0xbeea, "Wrong peer link ID");

        // the elements of ns-3, in the format of Linux
        IePeerManagement element;
        element.SetPeerOpen(0x453e);
        NS_TEST_EXPECT_MSG_EQ(ToHex(element), "750400003e45", "Wrong Open element");
        element.SetPeerConfirm(0x453e, 0xbeea);
        NS_TEST_EXPECT_MSG_EQ(ToHex(element), "750600003e45eabe", "Wrong Confirm element");
        element.SetPeerClose(0x453e, 0xbeea, REASON11S_MESH_CAPABILITY_POLICY_VIOLATION);
        NS_TEST_EXPECT_MSG_EQ(ToHex(element).substr(0, 16),
                              "750800003e45eabe",
                              "Wrong Close element");
    }
};

/**
 * @ingroup mesh-test
 *
 * @brief A mesh point receives beacons without Beacon Timing element, which is
 * optional (e.g., Linux does not send it): those of a mesh point whose beacon
 * collision avoidance is disabled.
 */
struct BeaconWithoutTimingTest : public TestCase
{
    BeaconWithoutTimingTest()
        : TestCase("Beacons without Beacon Timing element")
    {
    }

    void DoRun() override
    {
        NodeContainer nodes(2);
        MobilityHelper mobility;
        mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        mobility.Install(nodes);
        YansWifiPhyHelper phy;
        phy.SetChannel(YansWifiChannelHelper::Default().Create());
        MeshHelper mesh = MeshHelper::Default();
        mesh.SetStackInstaller("ns3::Dot11sStack");
        NetDeviceContainer devices = mesh.Install(phy, nodes);
        devices.Get(0)->GetObject<PeerManagementProtocol>()->SetBeaconCollisionAvoidance(false);
        auto pmp = devices.Get(1)->GetObject<PeerManagementProtocol>();
        NS_TEST_ASSERT_MSG_EQ(pmp->GetBeaconCollisionAvoidance(), true, "Wrong configuration");
        Simulator::Stop(Seconds(2));
        Simulator::Run();
        NS_TEST_EXPECT_MSG_EQ(pmp->GetNumberOfLinks(), 1, "The mesh points did not peer");
        Simulator::Destroy();
    }
};

/**
 * @ingroup mesh-test
 *
 * @brief The beacons of a mesh point carry the Mesh Configuration element (IEEE
 * 802.11-2020 9.3.3.2), without which Linux (mac80211) does not peer: HWMP, airtime
 * metric, no congestion control, neighbor offset synchronization, no authentication.
 */
struct BeaconMeshConfigurationTest : public TestCase
{
    BeaconMeshConfigurationTest()
        : TestCase("Mesh Configuration element in beacons")
    {
    }

    /**
     * Callback invoked when the PHY of the mesh point starts transmitting a PSDU.
     *
     * @param psduMap the PSDU map
     * @param txVector the TX vector
     * @param txPowerW the TX power in Watts
     */
    void Tx(WifiConstPsduMap psduMap, WifiTxVector txVector, double txPowerW)
    {
        auto psdu = psduMap.begin()->second;
        if (!psdu->GetHeader(0).IsBeacon())
        {
            return;
        }
        auto packet = psdu->GetPayload(0);
        std::vector<uint8_t> body(packet->GetSize());
        packet->CopyData(body.data(), body.size());
        m_beacons++;
        // the elements follow the Timestamp, Beacon Interval and Capability fields
        for (std::size_t k = 12; k + 2 <= body.size(); k += 2 + body[k + 1])
        {
            if (body[k] == IE_MESH_CONFIGURATION && body[k + 1] == 7)
            {
                m_meshConfig = std::vector<uint8_t>(body.begin() + k + 2, body.begin() + k + 7);
            }
        }
    }

    void DoRun() override
    {
        NodeContainer nodes(2);
        MobilityHelper mobility;
        mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        mobility.Install(nodes);
        YansWifiPhyHelper phy;
        phy.SetChannel(YansWifiChannelHelper::Default().Create());
        MeshHelper mesh = MeshHelper::Default();
        mesh.SetStackInstaller("ns3::Dot11sStack");
        NetDeviceContainer devices = mesh.Install(phy, nodes);
        auto interface = DynamicCast<WifiNetDevice>(
            DynamicCast<MeshPointDevice>(devices.Get(0))->GetInterfaces().front());
        interface->GetPhy()->TraceConnectWithoutContext(
            "PhyTxPsduBegin",
            MakeCallback(&BeaconMeshConfigurationTest::Tx, this));
        Simulator::Stop(Seconds(1));
        Simulator::Run();
        Simulator::Destroy();
        NS_TEST_ASSERT_MSG_GT(m_beacons, 0, "No beacon sent");
        // Active Path Selection Protocol, Metric, Congestion Control, Synchronization
        // and Authentication Protocol identifiers
        NS_TEST_EXPECT_MSG_EQ((m_meshConfig == std::vector<uint8_t>{1, 1, 0, 1, 0}),
                              true,
                              "No or wrong Mesh Configuration element in beacons");
    }

    uint32_t m_beacons{0};             ///< beacons sent
    std::vector<uint8_t> m_meshConfig; ///< identifiers of the Mesh Configuration element
};

/**
 * @ingroup mesh-test
 *
 * @brief HWMP receives (QoS) Null frames, which have no Mesh Control field: Linux
 * (mac80211) sends them to its mesh peers.
 */
struct HwmpNullFrameTest : public TestCase
{
    HwmpNullFrameTest()
        : TestCase("HWMP and QoS Null frames of mesh peers")
    {
    }

    void DoRun() override
    {
        auto hwmp = Create<HwmpProtocolMac>(1, CreateObject<HwmpProtocol>());
        WifiMacHeader header(WIFI_MAC_QOSDATA_NULL);
        header.SetDsFrom();
        header.SetDsTo();
        header.SetAddr1(Mac48Address("00:00:00:00:00:04"));
        header.SetAddr2(Mac48Address("00:00:00:00:00:01"));
        header.SetAddr3(Mac48Address("00:00:00:00:00:04"));
        header.SetAddr4(Mac48Address("00:00:00:00:00:01"));
        NS_TEST_EXPECT_MSG_EQ(hwmp->Receive(Create<Packet>(), header),
                              false,
                              "QoS Null frame not dropped");
    }
};

/**
 * @ingroup mesh-test
 *
 * @brief Dot11s Test Suite
 */
class Dot11sTestSuite : public TestSuite
{
  public:
    Dot11sTestSuite();
};

Dot11sTestSuite::Dot11sTestSuite()
    : TestSuite("devices-mesh-dot11s", Type::UNIT)
{
    AddTestCase(new MeshHeaderTest, TestCase::Duration::QUICK);
    AddTestCase(new HwmpRtableTest, TestCase::Duration::QUICK);
    AddTestCase(new PeerLinkFrameStartTest, TestCase::Duration::QUICK);
    AddTestCase(new PeerLinkFrameLinuxTest, TestCase::Duration::QUICK);
    AddTestCase(new BeaconWithoutTimingTest, TestCase::Duration::QUICK);
    AddTestCase(new BeaconMeshConfigurationTest, TestCase::Duration::QUICK);
    AddTestCase(new HwmpNullFrameTest, TestCase::Duration::QUICK);
}

static Dot11sTestSuite g_dot11sTestSuite; ///< the test suite
