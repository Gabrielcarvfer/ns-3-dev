/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

/**
 * @file
 * Many concurrent TCP flows through a single bottleneck.
 *
 * Topology: nInputNodes origin nodes --1Gb/s--> LinkNode --340Mb/s--> DestNode.
 * Workload: nLargeApps unlimited bulk-send flows plus nMedApps/nSmallApps
 * bounded flows, all TCP, with one PacketSink per destination port.
 *
 * Every flow adds a TCP endpoint on both end nodes, so this stresses the
 * receive-side endpoint demultiplexing and socket bookkeeping.
 *
 * One origin node can source at most ~16k concurrent flows (ephemeral port
 * range); use --nInputNodes to go beyond (e.g. 100k flows with
 * --nLargeApps=12500 --nInputNodes=8). Destination ports are 16-bit, so above
 * 60000 flows several flows share one PacketSink.
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-helper.h"

#include <iostream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpManyFlows");

/**
 * Bulk-send volume of a medium flow is this many times the base volume,
 * a small flow this many times less; see appBytesBase below.
 */
constexpr uint32_t MEDIUM_APP_FACTOR = 25;
constexpr uint32_t SMALL_APP_FACTOR = 7; //!< @see MEDIUM_APP_FACTOR

int
main(int argc, char* argv[])
{
    QueueSize txQueueSize("210p");
    Time simTime = Seconds(200);
    uint32_t nLargeApps = 0;
    uint32_t nSmallApps = 0;
    uint32_t nMedApps = 0;
    Time lowLinkDelay = MilliSeconds(10);
    bool useCubic = false;
    uint32_t nInputNodes = 1;
    bool flowMonitorEnabled = true;
    bool pfifoFastQd = false;
    uint32_t sendBufSize = 131072;

    CommandLine cmd(__FILE__);
    cmd.Usage("Many concurrent TCP flows through a single bottleneck link.\n"
              "\n"
              "Flows come in three sizes, differing only in how many bytes they send:\n"
              "large flows are unlimited (they run for the whole simulation), medium\n"
              "flows send 25 times the base volume and small flows 7 times, so they\n"
              "complete during the run and exercise endpoint setup and teardown.\n"
              "\n"
              "One origin node can source at most ~16k concurrent flows, so raise\n"
              "--nInputNodes to reach higher flow counts.");
    cmd.AddValue("SndBuf", "TCP send/receive buffer size per socket", sendBufSize);
    cmd.AddValue("PfifoFastQd",
                 "Install pfifo-fast instead of the default FqCoDel queue disc",
                 pfifoFastQd);
    cmd.AddValue("TxQueueSize", "TX queue size", txQueueSize);
    cmd.AddValue("SimTime", "Simulation execution time", simTime);
    cmd.AddValue("nLargeApps", "Number of active large apps", nLargeApps);
    cmd.AddValue("nMedApps", "Number of active medium apps", nMedApps);
    cmd.AddValue("nSmallApps", "Number of active small apps", nSmallApps);
    cmd.AddValue("Delay", "One way delay of the slow link", lowLinkDelay);
    cmd.AddValue("Cubic", "Use TCP Cubic instead of TCP NewReno", useCubic);
    cmd.AddValue("nInputNodes", "Number of input nodes", nInputNodes);
    cmd.AddValue("FlowMonitor", "Enable flow monitor", flowMonitorEnabled);
    cmd.Parse(argc, argv);

    if (useCubic)
    {
        Config::SetDefault("ns3::TcpL4Protocol::SocketType", TypeIdValue(TcpCubic::GetTypeId()));
    }
    else
    {
        Config::SetDefault("ns3::TcpL4Protocol::SocketType", TypeIdValue(TcpNewReno::GetTypeId()));
    }

    Config::SetDefault("ns3::Ipv4L3Protocol::FragmentExpirationTimeout", TimeValue(Seconds(0.2)));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(sendBufSize));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(sendBufSize));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1200));

    NodeContainer nodes(2 + nInputNodes);
    Ptr<Node> linkNode = nodes.Get(1);
    Ptr<Node> destNode = nodes.Get(0);

    PointToPointHelper p2ph;
    p2ph.SetQueue("ns3::DropTailQueue", "MaxSize", QueueSizeValue(txQueueSize));
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("340Mb/s")));
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(1600));
    p2ph.SetChannelAttribute("Delay", TimeValue(lowLinkDelay));
    NetDeviceContainer internetDevices = p2ph.Install(linkNode, destNode);
    std::cout << "Link delay set to " << lowLinkDelay.As(Time::MS) << std::endl;

    NetDeviceContainer ipNodes(internetDevices);

    for (uint32_t i = 0; i < nInputNodes; i++)
    {
        PointToPointHelper p2phFast;
        p2phFast.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("20000p"));
        p2phFast.SetDeviceAttribute("DataRate", DataRateValue(DataRate("1Gb/s")));
        p2phFast.SetDeviceAttribute("Mtu", UintegerValue(1600));
        p2phFast.SetChannelAttribute("Delay", TimeValue(MilliSeconds(1)));
        ipNodes.Add(p2phFast.Install(nodes.Get(2 + i), linkNode));
    }

    InternetStackHelper internet;
    Ipv4GlobalRoutingHelper ipv4RoutingHelper;
    internet.SetRoutingHelper(ipv4RoutingHelper);
    internet.Install(nodes);

    if (pfifoFastQd)
    {
        TrafficControlHelper tch;
        tch.SetRootQueueDisc("ns3::PfifoFastQueueDisc");
        tch.Install(ipNodes);
    }

    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(ipNodes);
    Ipv4Address destNodeAddr = destNode->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
    std::cout << "Assigned " << internetIpIfaces.GetN() << " IP interfaces, destination is "
              << destNodeAddr << std::endl;
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // Ports are 16 bit, so above maxDistinctPorts flows the destination
    // ports wrap around and several flows share one PacketSink (flows stay
    // distinct through their source port).
    const uint32_t firstPort = 10;
    const uint32_t maxDistinctPorts = 60000;
    uint32_t nextPort = firstPort;
    uint32_t distinctPorts = 0;

    /**
     * Install one bulk-send flow towards the destination node, and the
     * matching sink if its port has not been used yet.
     *
     * @param srcNode Node sourcing the flow.
     * @param maxBytes Bytes to send; 0 means unlimited, i.e. the flow sends
     *                 for the whole simulation.
     */
    auto makeFlow = [&](Ptr<Node> srcNode, uint64_t maxBytes) {
        uint32_t p = firstPort + (nextPort - firstPort) % maxDistinctPorts;
        nextPort++;
        BulkSendHelper sourceHelper("ns3::TcpSocketFactory", InetSocketAddress(destNodeAddr, p));
        sourceHelper.SetAttribute("MaxBytes", UintegerValue(maxBytes));
        sourceHelper.SetAttribute("SendSize", UintegerValue(1200));
        ApplicationContainer sourceApp = sourceHelper.Install(srcNode);
        sourceApp.Start(Seconds(0));
        sourceApp.Stop(simTime);
        if (distinctPorts < maxDistinctPorts)
        {
            distinctPorts++;
            PacketSinkHelper sinkHelper("ns3::TcpSocketFactory",
                                        InetSocketAddress(Ipv4Address::GetAny(), p));
            ApplicationContainer sinkApp = sinkHelper.Install(destNode);
            sinkApp.Start(Seconds(0));
            sinkApp.Stop(simTime);
        }
    };

    for (uint32_t j = 0; j < nInputNodes; j++)
    {
        for (uint32_t i = 0; i < nLargeApps; i++)
        {
            makeFlow(nodes.Get(2 + j), 0);
        }
    }

    constexpr uint64_t appBytesBase = 50ULL * 1000 * 1000;
    for (uint32_t i = 0; i < nMedApps; i++)
    {
        makeFlow(nodes.Get(2), appBytesBase * MEDIUM_APP_FACTOR);
    }

    for (uint32_t i = 0; i < nSmallApps; i++)
    {
        makeFlow(nodes.Get(2), appBytesBase * SMALL_APP_FACTOR);
    }

    Ptr<FlowMonitor> flowMonitor;
    FlowMonitorHelper flowHelper;
    if (flowMonitorEnabled)
    {
        flowMonitor = flowHelper.InstallAll();
    }

    ShowProgress progress(MilliSeconds(200), std::cout);

    Simulator::Stop(simTime);
    Simulator::Run();

    if (flowMonitorEnabled)
    {
        flowMonitor->SerializeToXmlFile("FlowMonitor.xml", true, true);
    }

    Simulator::Destroy();

    return 0;
}
