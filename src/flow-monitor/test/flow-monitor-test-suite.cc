/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/flow-monitor-helper.h"
#include "ns3/inet-socket-address.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/simulator.h"
#include "ns3/socket.h"
#include "ns3/test.h"

using namespace ns3;

/**
 * @ingroup flow-monitor
 * @defgroup flow-monitor-test FlowMonitor module tests
 */

/**
 * @ingroup flow-monitor-test
 * @ingroup tests
 *
 * Check that the FlowMonitorHelpers of a simulation share one FlowMonitor,
 * count each packet once even when installed twice on the same nodes, and
 * that the next simulation gets a new FlowMonitor.
 */
class FlowMonitorSharedMonitorTestCase : public TestCase
{
  public:
    FlowMonitorSharedMonitorTestCase();

  private:
    void DoRun() override;

    /**
     * Run one simulation sending packets over UDP between two nodes, with two
     * FlowMonitorHelpers installed on both nodes.
     *
     * @param packets Number of packets to send.
     * @return The FlowMonitor of the simulation.
     */
    Ptr<FlowMonitor> RunSimulation(uint32_t packets);
};

FlowMonitorSharedMonitorTestCase::FlowMonitorSharedMonitorTestCase()
    : TestCase("FlowMonitorHelpers share one FlowMonitor per simulation")
{
}

Ptr<FlowMonitor>
FlowMonitorSharedMonitorTestCase::RunSimulation(uint32_t packets)
{
    const uint16_t port = 9;

    NodeContainer nodes;
    nodes.Create(2);
    NetDeviceContainer devices = SimpleNetDeviceHelper().Install(nodes);
    InternetStackHelper().Install(nodes);
    Ipv4AddressHelper addresses;
    addresses.SetBase(Ipv4Address("10.0.0.0"), Ipv4Mask("255.255.255.0"));
    Ipv4InterfaceContainer interfaces = addresses.Assign(devices);

    NS_TEST_EXPECT_MSG_EQ(FlowMonitorHelper::HasMonitor(),
                          false,
                          "No FlowMonitor should exist before a helper creates one");

    FlowMonitorHelper first;
    Ptr<FlowMonitor> monitor = first.InstallAll();
    NS_TEST_EXPECT_MSG_EQ(FlowMonitorHelper::HasMonitor(), true, "The FlowMonitor should exist");

    // Installing again on the same nodes must not add probes counting the
    // packets a second time.
    FlowMonitorHelper second;
    NS_TEST_EXPECT_MSG_EQ(second.InstallAll(), monitor, "Helpers should share the FlowMonitor");
    NS_TEST_EXPECT_MSG_EQ(second.GetClassifier(),
                          first.GetClassifier(),
                          "Helpers should share the IPv4 classifier");
    NS_TEST_EXPECT_MSG_EQ(second.GetClassifier6(),
                          first.GetClassifier6(),
                          "Helpers should share the IPv6 classifier");

    TypeId udp = TypeId::LookupByName("ns3::UdpSocketFactory");
    Ptr<Socket> sink = Socket::CreateSocket(nodes.Get(1), udp);
    sink->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    Ptr<Socket> source = Socket::CreateSocket(nodes.Get(0), udp);
    source->Bind();
    InetSocketAddress destination(interfaces.GetAddress(1), port);
    for (uint32_t i = 0; i < packets; i++)
    {
        Simulator::Schedule(Seconds(1 + i * 0.1),
                            [=]() { source->SendTo(Create<Packet>(100), 0, destination); });
    }

    Simulator::Stop(Seconds(3));
    Simulator::Run();

    const FlowMonitor::FlowStatsContainer& stats = monitor->GetFlowStats();
    NS_TEST_EXPECT_MSG_EQ(stats.size(), 1, "The FlowMonitor should see one flow");
    if (!stats.empty())
    {
        NS_TEST_EXPECT_MSG_EQ(stats.begin()->second.txPackets,
                              packets,
                              "Each transmitted packet should be counted once");
        NS_TEST_EXPECT_MSG_EQ(stats.begin()->second.rxPackets,
                              packets,
                              "Each received packet should be counted once");
    }

    Simulator::Destroy();
    NS_TEST_EXPECT_MSG_EQ(FlowMonitorHelper::HasMonitor(),
                          false,
                          "Simulator::Destroy() should release the shared FlowMonitor");
    return monitor;
}

void
FlowMonitorSharedMonitorTestCase::DoRun()
{
    Ptr<FlowMonitor> firstRun = RunSimulation(5);
    Ptr<FlowMonitor> secondRun = RunSimulation(3);
    NS_TEST_EXPECT_MSG_NE(firstRun, secondRun, "A new simulation should get a new FlowMonitor");
}

/**
 * @ingroup flow-monitor-test
 * @ingroup tests
 *
 * FlowMonitor test suite.
 */
class FlowMonitorTestSuite : public TestSuite
{
  public:
    FlowMonitorTestSuite();
};

FlowMonitorTestSuite::FlowMonitorTestSuite()
    : TestSuite("flow-monitor", Type::UNIT)
{
    AddTestCase(new FlowMonitorSharedMonitorTestCase(), TestCase::Duration::QUICK);
}

static FlowMonitorTestSuite g_flowMonitorTestSuite; //!< Static variable for test initialization
