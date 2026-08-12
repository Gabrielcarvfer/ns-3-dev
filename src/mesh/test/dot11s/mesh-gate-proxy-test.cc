/*
 * Copyright (c) 2026 Centre Tecnològic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/bridge-helper.h"
#include "ns3/csma-helper.h"
#include "ns3/double.h"
#include "ns3/mesh-helper.h"
#include "ns3/mesh-point-device.h"
#include "ns3/mobility-helper.h"
#include "ns3/node-container.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/yans-wifi-helper.h"

using namespace ns3;

/**
 * @ingroup dot11s-test
 *
 * @brief Frames carried by a mesh gate on behalf of a station outside the mesh
 *
 * A mesh STA bridged to a network outside the mesh is a mesh gate: it carries frames for the
 * stations behind it, putting their addresses in the Mesh Address Extension subfield of the Mesh
 * Control field and its own address in the 802.11 header (IEEE 802.11-2012, Table 8-19).
 *
 * The test attaches a CSMA segment to one of two mesh STAs through a bridge and checks that:
 *  - a group addressed frame from the station outside the mesh reaches the other mesh STA
 *    carrying the address of the originating station, not that of the gate;
 *  - the same holds for an individually addressed frame;
 *  - the mesh STA, having learnt which gate proxies for that station, can reach it back.
 */
class MeshGateProxyTest : public TestCase
{
  public:
    MeshGateProxyTest()
        : TestCase("A mesh gate carries frames for stations outside the mesh")
    {
    }

    void DoRun() override;

  private:
    /**
     * Receive callback registered on the mesh STA that is not a gate.
     *
     * @param device the receiving device
     * @param packet the received packet
     * @param protocol the protocol number
     * @param source the source address
     * @return true (packet accepted)
     */
    bool ReceiveAtMesh(Ptr<NetDevice> device,
                       Ptr<const Packet> packet,
                       uint16_t protocol,
                       const Address& source);
    /**
     * Receive callback registered on the station outside the mesh.
     *
     * @param device the receiving device
     * @param packet the received packet
     * @param protocol the protocol number
     * @param source the source address
     * @return true (packet accepted)
     */
    bool ReceiveAtExternal(Ptr<NetDevice> device,
                           Ptr<const Packet> packet,
                           uint16_t protocol,
                           const Address& source);

    std::vector<Mac48Address> m_sourcesAtMesh; ///< sources seen by the mesh STA
    uint32_t m_receivedAtExternal{0};          ///< frames delivered back to the outside station
};

bool
MeshGateProxyTest::ReceiveAtMesh(Ptr<NetDevice> device,
                                 Ptr<const Packet> packet,
                                 uint16_t protocol,
                                 const Address& source)
{
    m_sourcesAtMesh.push_back(Mac48Address::ConvertFrom(source));
    return true;
}

bool
MeshGateProxyTest::ReceiveAtExternal(Ptr<NetDevice> device,
                                     Ptr<const Packet> packet,
                                     uint16_t protocol,
                                     const Address& source)
{
    m_receivedAtExternal++;
    return true;
}

void
MeshGateProxyTest::DoRun()
{
    // Two mesh STAs within range of each other; the first one is also bridged to a CSMA segment
    // carrying a single station outside the mesh
    NodeContainer meshNodes;
    meshNodes.Create(2);
    NodeContainer externalNode;
    externalNode.Create(1);

    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(meshNodes);

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());

    MeshHelper mesh = MeshHelper::Default();
    mesh.SetStackInstaller("ns3::Dot11sStack");
    NetDeviceContainer meshDevices = mesh.Install(phy, meshNodes);

    CsmaHelper csma;
    NetDeviceContainer csmaDevices =
        csma.Install(NodeContainer(meshNodes.Get(0), externalNode.Get(0)));

    // Bridging the mesh point to the CSMA segment turns the first mesh STA into a mesh gate
    BridgeHelper bridge;
    bridge.Install(meshNodes.Get(0), NetDeviceContainer(meshDevices.Get(0), csmaDevices.Get(0)));

    Ptr<MeshPointDevice> plainMesh = DynamicCast<MeshPointDevice>(meshDevices.Get(1));
    Ptr<NetDevice> external = csmaDevices.Get(1);
    plainMesh->SetReceiveCallback(MakeCallback(&MeshGateProxyTest::ReceiveAtMesh, this));
    external->SetReceiveCallback(MakeCallback(&MeshGateProxyTest::ReceiveAtExternal, this));

    const auto externalAddress = Mac48Address::ConvertFrom(external->GetAddress());
    const auto plainMeshAddress = Mac48Address::ConvertFrom(plainMesh->GetAddress());
    const Mac48Address multicastGroup("01:00:5e:00:00:01");

    // Group addressed, then individually addressed, from outside the mesh into it
    Simulator::Schedule(Seconds(3), [external, multicastGroup]() {
        external->Send(Create<Packet>(100), multicastGroup, 0x0800);
    });
    Simulator::Schedule(Seconds(4), [external, plainMeshAddress]() {
        external->Send(Create<Packet>(100), plainMeshAddress, 0x0800);
    });
    // Back out of the mesh, which requires the proxy to have been learnt
    Simulator::Schedule(Seconds(5), [plainMesh, externalAddress]() {
        plainMesh->Send(Create<Packet>(100), externalAddress, 0x0800);
    });

    Simulator::Stop(Seconds(8));
    Simulator::Run();
    Simulator::Destroy();

    NS_TEST_ASSERT_MSG_EQ(m_sourcesAtMesh.size(),
                          2,
                          "The mesh STA did not receive both frames from outside the mesh");
    NS_TEST_EXPECT_MSG_EQ(m_sourcesAtMesh.at(0),
                          externalAddress,
                          "A group addressed frame must reach the mesh STA with the address of "
                          "the originating station, not that of the gate");
    NS_TEST_EXPECT_MSG_EQ(m_sourcesAtMesh.at(1),
                          externalAddress,
                          "An individually addressed frame must reach the mesh STA with the "
                          "address of the originating station, not that of the gate");
    NS_TEST_EXPECT_MSG_GT(m_receivedAtExternal,
                          0,
                          "The mesh STA could not reach the station behind the gate");
}

/**
 * @ingroup dot11s-test
 *
 * @brief Mesh gate test suite
 */
class MeshGateProxySuite : public TestSuite
{
  public:
    MeshGateProxySuite()
        : TestSuite("devices-mesh-dot11s-gate", Type::UNIT)
    {
        AddTestCase(new MeshGateProxyTest, TestCase::Duration::QUICK);
    }
};

static MeshGateProxySuite g_meshGateProxySuite; ///< the test suite
