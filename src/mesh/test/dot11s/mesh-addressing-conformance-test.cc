/*
 * Copyright (c) 2026 Centre Tecnològic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/config.h"
#include "ns3/dot11s-mac-header.h"
#include "ns3/double.h"
#include "ns3/mesh-helper.h"
#include "ns3/mesh-point-device.h"
#include "ns3/mgt-action-headers.h"
#include "ns3/mobility-helper.h"
#include "ns3/node-container.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"
#include "ns3/wifi-mac-header.h"
#include "ns3/yans-wifi-helper.h"

using namespace ns3;
using namespace ns3::dot11s;

/**
 * @ingroup dot11s-test
 *
 * @brief Address field usage of mesh frames transmitted over the air
 *
 * IEEE 802.11-2012 Table 8-19 (Table 9-15 in IEEE 802.11-2016 and later)
 * defines which address each field of a mesh frame carries. This test drives a
 * three-node chain mesh, captures the frames it transmits and checks the two
 * Mesh Data rows that the model is able to produce, plus the individually and
 * group addressed Multihop Action frames:
 *
 * | Description         | To DS | From DS | Addr 1  | Addr 2 | Addr 3  | Addr 4 |
 * | Mesh Data unicast   |   1   |    1    | RA      | TA     | DA      | SA     |
 * | Mesh Data multicast |   0   |    1    | RA = DA | TA     | Mesh SA | -      |
 *
 * The proxied rows of the same table are covered by ProxiedAddressingHeaderTest
 * at the level of the header encoding, and over the air by the mesh gate test
 * suite, which drives traffic to and from a station outside the mesh.
 */
class MeshDataAddressingTest : public TestCase
{
  public:
    MeshDataAddressingTest()
        : TestCase("Mesh frames use the address fields of Table 8-19")
    {
    }

    void DoRun() override;

  private:
    /**
     * Trace sink recording the header of every frame transmitted in the mesh.
     *
     * @param packet The frame about to be transmitted, including its MAC header.
     * @param txPowerW The transmission power, unused.
     */
    void Transmitted(Ptr<const Packet> packet, double txPowerW);

    /// A captured frame header, kept alongside whether it was seen at all
    struct Captured
    {
        WifiMacHeader header; ///< the captured MAC header
        bool seen{false};     ///< whether a frame of this kind was captured
    };

    Captured m_dataUnicast;   ///< first individually addressed Mesh Data frame
    Captured m_dataGroup;     ///< first group addressed Mesh Data frame
    Captured m_actionUnicast; ///< first individually addressed Multihop Action frame
    Captured m_actionGroup;   ///< first group addressed Multihop Action frame
};

void
MeshDataAddressingTest::Transmitted(Ptr<const Packet> packet, double txPowerW)
{
    Ptr<Packet> copy = packet->Copy();
    WifiMacHeader header;
    copy->RemoveHeader(header);

    if (header.IsData())
    {
        // Peer link management traffic is carried in Action frames, so any Data
        // frame seen here is a Mesh Data frame
        Captured& slot = header.GetAddr1().IsGroup() ? m_dataGroup : m_dataUnicast;
        if (!slot.seen)
        {
            slot.header = header;
            slot.seen = true;
        }
    }
    else if (header.IsAction())
    {
        WifiActionHeader actionHdr;
        copy->RemoveHeader(actionHdr);
        // Peer link management uses the SELF_PROTECTED category; only the MESH
        // category carries the Multihop Action frames of the path selection
        // protocol
        if (actionHdr.GetCategory() != WifiActionHeader::MESH)
        {
            return;
        }
        Captured& slot = header.GetAddr1().IsGroup() ? m_actionGroup : m_actionUnicast;
        if (!slot.seen)
        {
            slot.header = header;
            slot.seen = true;
        }
    }
}

void
MeshDataAddressingTest::DoRun()
{
    // Three nodes in a chain; only adjacent nodes are within radio range, so
    // the middle node has to forward and a path discovery has to take place
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
    const auto srcAddress = Mac48Address::ConvertFrom(src->GetAddress());
    const auto dstAddress = Mac48Address::ConvertFrom(dst->GetAddress());

    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxBegin",
                                  MakeCallback(&MeshDataAddressingTest::Transmitted, this));

    // Send both kinds of Mesh Data frame once the peer links are up
    const Mac48Address multicastGroup("01:00:5e:00:00:01");
    Simulator::Schedule(Seconds(3), [src, dstAddress]() {
        src->Send(Create<Packet>(100), dstAddress, 0x0800);
    });
    Simulator::Schedule(Seconds(4), [src, multicastGroup]() {
        src->Send(Create<Packet>(100), multicastGroup, 0x0800);
    });

    Simulator::Stop(Seconds(6));
    Simulator::Run();
    Simulator::Destroy();

    // Mesh Data, individually addressed: four-address format
    NS_TEST_ASSERT_MSG_EQ(m_dataUnicast.seen,
                          true,
                          "No individually addressed Mesh Data frame was transmitted");
    NS_TEST_EXPECT_MSG_EQ(m_dataUnicast.header.IsToDs(),
                          true,
                          "Individually addressed Mesh Data frame must have To DS set");
    NS_TEST_EXPECT_MSG_EQ(m_dataUnicast.header.IsFromDs(),
                          true,
                          "Individually addressed Mesh Data frame must have From DS set");
    NS_TEST_EXPECT_MSG_EQ(m_dataUnicast.header.GetAddr1().IsGroup(),
                          false,
                          "Address 1 of an individually addressed frame must be the unicast RA");
    NS_TEST_EXPECT_MSG_EQ(m_dataUnicast.header.GetAddr3(),
                          dstAddress,
                          "Address 3 of an individually addressed Mesh Data frame must be the DA");
    NS_TEST_EXPECT_MSG_EQ(m_dataUnicast.header.GetAddr4(),
                          srcAddress,
                          "Address 4 of an individually addressed Mesh Data frame must be the SA");

    // Mesh Data, group addressed: three-address format
    NS_TEST_ASSERT_MSG_EQ(m_dataGroup.seen,
                          true,
                          "No group addressed Mesh Data frame was transmitted");
    NS_TEST_EXPECT_MSG_EQ(m_dataGroup.header.IsToDs(),
                          false,
                          "Group addressed Mesh Data frame must have To DS clear");
    NS_TEST_EXPECT_MSG_EQ(m_dataGroup.header.IsFromDs(),
                          true,
                          "Group addressed Mesh Data frame must have From DS set");
    NS_TEST_EXPECT_MSG_EQ(m_dataGroup.header.GetAddr1(),
                          multicastGroup,
                          "Address 1 of a group addressed Mesh Data frame must be the group DA");
    NS_TEST_EXPECT_MSG_EQ(m_dataGroup.header.GetAddr3(),
                          srcAddress,
                          "Address 3 of a group addressed Mesh Data frame must be the mesh SA");

    // Multihop Action frames keep the mesh SA in Address 3 in both forms
    NS_TEST_ASSERT_MSG_EQ(m_actionGroup.seen,
                          true,
                          "No group addressed Multihop Action frame was transmitted");
    NS_TEST_EXPECT_MSG_EQ(m_actionGroup.header.IsToDs(),
                          false,
                          "Multihop Action frame must have To DS clear");
    NS_TEST_EXPECT_MSG_EQ(m_actionGroup.header.IsFromDs(),
                          false,
                          "Multihop Action frame must have From DS clear");
    NS_TEST_EXPECT_MSG_EQ(m_actionGroup.header.GetAddr2(),
                          m_actionGroup.header.GetAddr3(),
                          "A Multihop Action frame originated locally has the TA as its mesh SA");

    NS_TEST_ASSERT_MSG_EQ(m_actionUnicast.seen,
                          true,
                          "No individually addressed Multihop Action frame was transmitted");
    NS_TEST_EXPECT_MSG_EQ(m_actionUnicast.header.IsToDs(),
                          false,
                          "Multihop Action frame must have To DS clear");
    NS_TEST_EXPECT_MSG_EQ(m_actionUnicast.header.IsFromDs(),
                          false,
                          "Multihop Action frame must have From DS clear");
    NS_TEST_EXPECT_MSG_EQ(m_actionUnicast.header.GetAddr1().IsGroup(),
                          false,
                          "Address 1 of an individually addressed frame must be the unicast RA");
}

/**
 * @ingroup dot11s-test
 *
 * @brief Address field usage of proxied mesh frames
 *
 * The proxied rows of IEEE 802.11-2012 Table 8-19 carry the addresses of the
 * stations outside the mesh in the Mesh Address Extension subfield of the Mesh
 * Control field, while the 802.11 header carries the addresses of the mesh
 * stations that proxy for them:
 *
 * | Description | To DS | From DS | Ext   | Addr 1  | Addr 3  | Addr 4  | Addr 5 | Addr 6 |
 * | unicast     |   1   |    1    | 5 & 6 | RA      | Mesh DA | Mesh SA | DA     | SA     |
 * | multicast   |   0   |    1    | 4     | RA = DA | Mesh SA | SA      | -      | -      |
 *
 * This test covers the encoding of the header pair in isolation; the mesh gate
 * test suite checks the same rows on frames actually exchanged over the air.
 */
class ProxiedAddressingHeaderTest : public TestCase
{
  public:
    ProxiedAddressingHeaderTest()
        : TestCase("Proxied mesh frames encode the address extension of Table 8-19")
    {
    }

    void DoRun() override;
};

void
ProxiedAddressingHeaderTest::DoRun()
{
    const Mac48Address meshSa("00:00:00:00:00:01");
    const Mac48Address meshDa("00:00:00:00:00:02");
    const Mac48Address ra("00:00:00:00:00:03");
    const Mac48Address externalSa("00:00:00:00:00:04");
    const Mac48Address externalDa("00:00:00:00:00:05");
    const Mac48Address multicastGroup("01:00:5e:00:00:01");

    // Proxied Mesh Data, individually addressed: the external DA and SA go to
    // Address 5 and Address 6 of the Mesh Address Extension
    {
        WifiMacHeader hdr;
        hdr.SetType(WIFI_MAC_QOSDATA);
        hdr.SetDsTo();
        hdr.SetDsFrom();
        hdr.SetAddr1(ra);
        hdr.SetAddr2(meshSa);
        hdr.SetAddr3(meshDa);
        hdr.SetAddr4(meshSa);

        MeshHeader meshHdr;
        meshHdr.SetAddressExt(2);
        meshHdr.SetAddr5(externalDa);
        meshHdr.SetAddr6(externalSa);

        NS_TEST_EXPECT_MSG_EQ(meshHdr.GetSerializedSize(),
                              6 + 2 * 6,
                              "A two-address extension adds two addresses to the Mesh Control "
                              "field");

        Ptr<Packet> packet = Create<Packet>(10);
        packet->AddHeader(meshHdr);
        packet->AddHeader(hdr);

        WifiMacHeader readHdr;
        MeshHeader readMeshHdr;
        packet->RemoveHeader(readHdr);
        packet->RemoveHeader(readMeshHdr);

        NS_TEST_EXPECT_MSG_EQ(readHdr.IsToDs(), true, "Proxied unicast must have To DS set");
        NS_TEST_EXPECT_MSG_EQ(readHdr.IsFromDs(), true, "Proxied unicast must have From DS set");
        NS_TEST_EXPECT_MSG_EQ(readHdr.GetAddr3(), meshDa, "Address 3 must be the mesh DA");
        NS_TEST_EXPECT_MSG_EQ(readHdr.GetAddr4(), meshSa, "Address 4 must be the mesh SA");
        NS_TEST_EXPECT_MSG_EQ(readMeshHdr.GetAddressExt(),
                              2,
                              "Proxied unicast uses the Address 5 and 6 extension");
        NS_TEST_EXPECT_MSG_EQ(readMeshHdr.GetAddr5(), externalDa, "Address 5 must be the DA");
        NS_TEST_EXPECT_MSG_EQ(readMeshHdr.GetAddr6(), externalSa, "Address 6 must be the SA");
    }

    // Proxied Mesh Data, group addressed: only the external SA is carried, in
    // Address 4 of the Mesh Address Extension
    {
        WifiMacHeader hdr;
        hdr.SetType(WIFI_MAC_QOSDATA);
        hdr.SetDsFrom();
        hdr.SetDsNotTo();
        hdr.SetAddr1(multicastGroup);
        hdr.SetAddr2(meshSa);
        hdr.SetAddr3(meshSa);

        MeshHeader meshHdr;
        meshHdr.SetAddressExt(1);
        meshHdr.SetAddr4(externalSa);

        NS_TEST_EXPECT_MSG_EQ(meshHdr.GetSerializedSize(),
                              6 + 6,
                              "A one-address extension adds one address to the Mesh Control field");

        Ptr<Packet> packet = Create<Packet>(10);
        packet->AddHeader(meshHdr);
        packet->AddHeader(hdr);

        WifiMacHeader readHdr;
        MeshHeader readMeshHdr;
        packet->RemoveHeader(readHdr);
        packet->RemoveHeader(readMeshHdr);

        NS_TEST_EXPECT_MSG_EQ(readHdr.IsToDs(), false, "Proxied multicast must have To DS clear");
        NS_TEST_EXPECT_MSG_EQ(readHdr.IsFromDs(), true, "Proxied multicast must have From DS set");
        NS_TEST_EXPECT_MSG_EQ(readHdr.GetAddr1(), multicastGroup, "Address 1 must be the group DA");
        NS_TEST_EXPECT_MSG_EQ(readHdr.GetAddr3(), meshSa, "Address 3 must be the mesh SA");
        NS_TEST_EXPECT_MSG_EQ(readMeshHdr.GetAddressExt(),
                              1,
                              "Proxied multicast uses the Address 4 extension");
        NS_TEST_EXPECT_MSG_EQ(readMeshHdr.GetAddr4(), externalSa, "Address 4 must be the SA");
    }

    // The extension mode is a field, not a set of flags to accumulate
    {
        MeshHeader meshHdr;
        meshHdr.SetAddressExt(1);
        meshHdr.SetAddressExt(2);
        NS_TEST_EXPECT_MSG_EQ(meshHdr.GetAddressExt(),
                              2,
                              "Setting the address extension must replace the previous value");
    }
}

/**
 * @ingroup dot11s-test
 *
 * @brief Mesh addressing conformance test suite
 */
class MeshAddressingConformanceSuite : public TestSuite
{
  public:
    MeshAddressingConformanceSuite()
        : TestSuite("devices-mesh-dot11s-addressing", Type::UNIT)
    {
        AddTestCase(new MeshDataAddressingTest, TestCase::Duration::QUICK);
        AddTestCase(new ProxiedAddressingHeaderTest, TestCase::Duration::QUICK);
    }
};

static MeshAddressingConformanceSuite g_meshAddressingConformanceSuite; ///< the test suite
