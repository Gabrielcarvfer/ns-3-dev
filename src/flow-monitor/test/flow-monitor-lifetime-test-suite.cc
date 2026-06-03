/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ns3/flow-monitor-helper.h"
#include "ns3/flow-monitor.h"
#include "ns3/inet-socket-address.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/node-container.h"
#include "ns3/packet.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/simulator.h"
#include "ns3/socket.h"
#include "ns3/test.h"
#include "ns3/udp-socket-factory.h"

using namespace ns3;

/**
 * @ingroup flow-monitor-test
 *
 * @brief Regression test for FlowMonitorHelper object lifetime (issue #665).
 *
 * When a FlowMonitorHelper is destroyed (e.g. it goes out of scope) while the
 * caller still holds the Ptr<FlowMonitor> it returned, the helper disposes the
 * monitor and its probes. The probes remain connected to the IP-layer trace
 * sources, so a packet flowing afterwards used to dereference the now-null
 * probe members and crash. This test reproduces that pattern and checks that
 * the simulation runs to completion.
 */
class FlowMonitorHelperLifetimeTestCase : public TestCase
{
  public:
    FlowMonitorHelperLifetimeTestCase()
        : TestCase("FlowMonitorHelper destroyed before Simulator::Run does not crash")
    {
    }

  private:
    void DoRun() override;

    /// Receive callback: drains the socket and counts received bytes.
    /// @param socket the receiving socket
    void Receive(Ptr<Socket> socket);

    uint32_t m_received{0}; //!< Total bytes received.
};

void
FlowMonitorHelperLifetimeTestCase::Receive(Ptr<Socket> socket)
{
    Ptr<Packet> packet;
    while ((packet = socket->Recv()))
    {
        m_received += packet->GetSize();
    }
}

void
FlowMonitorHelperLifetimeTestCase::DoRun()
{
    NodeContainer nodes;
    nodes.Create(2);

    SimpleNetDeviceHelper simpleNet;
    NetDeviceContainer devices = simpleNet.Install(nodes);

    InternetStackHelper internet;
    internet.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    // Install the flow monitor through a helper that is destroyed immediately,
    // while keeping the returned Ptr<FlowMonitor> alive (the issue #665 pattern).
    Ptr<FlowMonitor> monitor;
    {
        FlowMonitorHelper flowmonHelper;
        monitor = flowmonHelper.InstallAll();
    }
    NS_TEST_ASSERT_MSG_NE(monitor, nullptr, "InstallAll should return a FlowMonitor");

    // Set up a UDP receiver and sender.
    auto rxSocket = Socket::CreateSocket(nodes.Get(1), UdpSocketFactory::GetTypeId());
    rxSocket->Bind(InetSocketAddress(Ipv4Address::GetAny(), 9));
    rxSocket->SetRecvCallback(
        MakeCallback(&FlowMonitorHelperLifetimeTestCase::Receive, this));

    auto txSocket = Socket::CreateSocket(nodes.Get(0), UdpSocketFactory::GetTypeId());
    InetSocketAddress dst(interfaces.GetAddress(1), 9);
    Simulator::Schedule(Seconds(1.0), [txSocket, dst]() {
        txSocket->SendTo(Create<Packet>(128), 0, dst);
    });

    Simulator::Stop(Seconds(2.0));
    // Without the fix, a packet traversing the IP layer triggers the disposed
    // probe's trace callbacks and crashes here.
    Simulator::Run();
    Simulator::Destroy();

    // Reaching this point means no crash occurred. The packet must have been
    // delivered, proving the simulation ran normally with the disposed monitor.
    NS_TEST_ASSERT_MSG_EQ(m_received, 128, "Packet was not delivered (simulation did not run)");
}

/**
 * @ingroup flow-monitor-test
 *
 * @brief Sanity check that normal FlowMonitor usage (helper kept in scope)
 * still collects flow statistics, i.e. the trace disconnection added for issue
 * #665 does not affect a probe that is not disposed during the simulation.
 */
class FlowMonitorNormalCollectionTestCase : public TestCase
{
  public:
    FlowMonitorNormalCollectionTestCase()
        : TestCase("FlowMonitor collects statistics during normal usage")
    {
    }

  private:
    void DoRun() override;
};

void
FlowMonitorNormalCollectionTestCase::DoRun()
{
    NodeContainer nodes;
    nodes.Create(2);

    SimpleNetDeviceHelper simpleNet;
    NetDeviceContainer devices = simpleNet.Install(nodes);

    InternetStackHelper internet;
    internet.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    // The helper is kept in scope for the whole simulation (the normal pattern).
    FlowMonitorHelper flowmonHelper;
    Ptr<FlowMonitor> monitor = flowmonHelper.InstallAll();

    auto rxSocket = Socket::CreateSocket(nodes.Get(1), UdpSocketFactory::GetTypeId());
    rxSocket->Bind(InetSocketAddress(Ipv4Address::GetAny(), 9));

    auto txSocket = Socket::CreateSocket(nodes.Get(0), UdpSocketFactory::GetTypeId());
    InetSocketAddress dst(interfaces.GetAddress(1), 9);
    Simulator::Schedule(Seconds(1.0), [txSocket, dst]() {
        txSocket->SendTo(Create<Packet>(128), 0, dst);
    });

    Simulator::Stop(Seconds(2.0));
    Simulator::Run();

    monitor->CheckForLostPackets();
    auto stats = monitor->GetFlowStats();
    NS_TEST_ASSERT_MSG_EQ(stats.empty(), false, "No flow statistics were collected");

    uint64_t totalTxPackets = 0;
    for (const auto& [flowId, flowStats] : stats)
    {
        totalTxPackets += flowStats.txPackets;
    }
    NS_TEST_ASSERT_MSG_GT_OR_EQ(totalTxPackets, 1, "FlowMonitor recorded no transmitted packets");

    Simulator::Destroy();
}

/**
 * @ingroup flow-monitor-test
 *
 * @brief FlowMonitorHelper lifetime test suite.
 */
class FlowMonitorHelperLifetimeTestSuite : public TestSuite
{
  public:
    FlowMonitorHelperLifetimeTestSuite()
        : TestSuite("flow-monitor-helper-lifetime", Type::UNIT)
    {
        AddTestCase(new FlowMonitorHelperLifetimeTestCase, TestCase::Duration::QUICK);
        AddTestCase(new FlowMonitorNormalCollectionTestCase, TestCase::Duration::QUICK);
    }
};

static FlowMonitorHelperLifetimeTestSuite
    g_flowMonitorHelperLifetimeTestSuite; //!< Static variable for test initialization
