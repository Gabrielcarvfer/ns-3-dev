/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ns3/boolean.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/flow-monitor.h"
#include "ns3/inet-socket-address.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-flow-classifier.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/node-container.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/test.h"

/**
 * @file
 * @ingroup flow-monitor-test
 *
 * Tests of the flow monitor on source routed flows.
 */

/**
 * @ingroup flow-monitor
 * @defgroup flow-monitor-test flow-monitor module tests
 */

using namespace ns3;

/**
 * @ingroup flow-monitor-test
 * @ingroup tests
 *
 * @brief Test the flows of a source routed TCP connection
 *
 * A datagram following a loose source route is addressed to the next hop of
 * the route, and readdressed at each hop: its flow is classified by its final
 * destination, so that the datagrams the destination receives, and the ones
 * the hops forward, are counted with the ones sent, in both directions of
 * the connection.
 */
class FlowMonitorSourceRouteTestCase : public TestCase
{
  public:
    FlowMonitorSourceRouteTestCase()
        : TestCase("The flows of a source routed TCP connection")
    {
    }

  private:
    void DoRun() override;
};

void
FlowMonitorSourceRouteTestCase::DoRun()
{
    // The line 0 - 1 - 2, on the subnets 10.1.1.0/24 and 10.1.2.0/24
    NodeContainer nodes;
    nodes.Create(3);
    InternetStackHelper stack;
    stack.Install(nodes);
    SimpleNetDeviceHelper devHelper;
    devHelper.SetDeviceAttribute("DataRate", StringValue("10Mbps"));
    devHelper.SetChannelAttribute("Delay", StringValue("1ms"));
    Ipv4AddressHelper address;
    std::vector<Ipv4InterfaceContainer> links;
    for (uint32_t link = 0; link < 2; ++link)
    {
        NetDeviceContainer devices =
            devHelper.Install(NodeContainer(nodes.Get(link), nodes.Get(link + 1)));
        std::string base = "10.1." + std::to_string(link + 1) + ".0";
        address.SetBase(base.c_str(), "255.255.255.0");
        links.push_back(address.Assign(devices));
    }
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
    nodes.Get(1)->GetObject<Ipv4L3Protocol>()->SetAttribute("SourceRouteForwarding",
                                                            BooleanValue(true));

    FlowMonitorHelper flowHelper;
    Ptr<FlowMonitor> monitor = flowHelper.InstallAll();

    const uint16_t port = 5000;
    const uint32_t bytes = 100000;
    uint32_t received = 0;
    Ptr<Socket> server = Socket::CreateSocket(nodes.Get(2), TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                              Callback<void, Ptr<Socket>, const Address&>(
                                  [&received](Ptr<Socket> socket, const Address&) {
                                      socket->SetRecvCallback(
                                          Callback<void, Ptr<Socket>>([&received](Ptr<Socket> s) {
                                              while (Ptr<Packet> packet = s->Recv())
                                              {
                                                  received += packet->GetSize();
                                              }
                                          }));
                                  }));

    Ptr<TcpSocketBase> client = DynamicCast<TcpSocketBase>(
        Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId()));
    client->Bind();
    NS_TEST_ASSERT_MSG_EQ(client->SetIpv4SourceRoute({links[0].GetAddress(1)}),
                          true,
                          "The route was refused");
    client->Connect(InetSocketAddress(links[1].GetAddress(1), port));
    Simulator::Schedule(Seconds(1), [client, bytes]() { client->Send(Create<Packet>(bytes), 0); });

    Simulator::Stop(Seconds(10));
    Simulator::Run();
    monitor->CheckForLostPackets();

    NS_TEST_ASSERT_MSG_EQ(received, bytes, "The data did not flow");
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();
    NS_TEST_ASSERT_MSG_EQ(stats.size(), 2, "Not one flow in each direction");
    for (const auto& [flowId, flow] : stats)
    {
        Ipv4FlowClassifier::FiveTuple tuple = classifier->FindFlow(flowId);
        bool forward = tuple.sourceAddress == links[0].GetAddress(0);
        Ipv4Address destination = forward ? links[1].GetAddress(1) : links[0].GetAddress(0);
        NS_TEST_ASSERT_MSG_EQ(tuple.destinationAddress,
                              destination,
                              "The flow was not classified by its final destination");
        NS_TEST_ASSERT_MSG_GT(flow.txPackets, 0, "No packet was sent");
        NS_TEST_ASSERT_MSG_EQ(flow.rxPackets,
                              flow.txPackets,
                              "The packets received were not counted");
        NS_TEST_ASSERT_MSG_EQ(flow.lostPackets, 0, "Packets were counted as lost");
        NS_TEST_ASSERT_MSG_EQ(flow.timesForwarded,
                              flow.rxPackets,
                              "The forwarding by the hop was not counted");
    }

    client = nullptr;
    server = nullptr;
    Simulator::Destroy();
}

/**
 * @ingroup flow-monitor-test
 * @ingroup tests
 *
 * @brief Test suite of the flow monitor on source routed flows
 */
class FlowMonitorSourceRouteTestSuite : public TestSuite
{
  public:
    FlowMonitorSourceRouteTestSuite()
        : TestSuite("flow-monitor-source-route", Type::UNIT)
    {
        AddTestCase(new FlowMonitorSourceRouteTestCase(), TestCase::Duration::QUICK);
    }
};

static FlowMonitorSourceRouteTestSuite
    g_flowMonitorSourceRouteTestSuite; //!< Static variable for test initialization
