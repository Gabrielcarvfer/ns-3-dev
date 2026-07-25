/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

/**
 * @file
 * AODV route discovery and forwarding in a dense broadcast domain with
 * many concurrent flows, stressing the routing table lookup and purge
 * path (the routing table is purged on every route lookup).
 *
 * Scaling reference (default build, 300 nodes, 300 flows, 10 s):
 *   before the lazy routing table purge: 87 s wall
 *   after:                               41 s wall
 *
 * Example: ./ns3 run "aodv-dense-scaling --nodes=300 --flows=300"
 */

#include "ns3/aodv-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"

using namespace ns3;

int
main(int argc, char* argv[])
{
    uint32_t nNodes = 200;
    uint32_t flows = 200;
    Time simTime = Seconds(20);
    CommandLine cmd(__FILE__);
    cmd.Usage("AODV in a dense broadcast domain: every node shares one CSMA segment,\n"
              "so each route request reaches every other node and the routing table\n"
              "grows with the node count.");
    cmd.AddValue("nodes", "Number of nodes", nNodes);
    cmd.AddValue("flows", "Number of UDP flows", flows);
    cmd.AddValue("simTime", "Simulation time", simTime);
    cmd.Parse(argc, argv);

    NodeContainer nodes(nNodes);

    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", StringValue("1Gbps"));
    csma.SetChannelAttribute("Delay", TimeValue(MicroSeconds(1)));
    NetDeviceContainer devices = csma.Install(nodes);

    AodvHelper aodv;
    InternetStackHelper stack;
    stack.SetRoutingHelper(aodv);
    stack.Install(nodes);

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.0.0.0", "255.255.0.0");
    Ipv4InterfaceContainer ifs = ipv4.Assign(devices);

    uint16_t port = 9000;
    Ptr<UniformRandomVariable> rng = CreateObject<UniformRandomVariable>();
    ApplicationContainer apps;
    OnOffHelper onoff("ns3::UdpSocketFactory", Address());
    onoff.SetConstantRate(DataRate("500kbps"), 1200);
    PacketSinkHelper sink("ns3::UdpSocketFactory", Address());
    for (uint32_t f = 0; f < flows; f++)
    {
        uint32_t src = rng->GetInteger(0, nNodes - 1);
        // Pick any node other than the source, without a retry loop.
        uint32_t dst = (src + 1 + rng->GetInteger(0, nNodes - 2)) % nNodes;
        onoff.SetAttribute("Remote",
                           AddressValue(InetSocketAddress(ifs.GetAddress(dst), port + f)));
        apps.Add(onoff.Install(nodes.Get(src)));
        sink.SetAttribute("Local",
                          AddressValue(InetSocketAddress(Ipv4Address::GetAny(), port + f)));
        apps.Add(sink.Install(nodes.Get(dst)));
    }
    apps.Start(Seconds(0.1));

    std::cout << "nodes=" << nNodes << " flows=" << flows << " simTime=" << simTime.As(Time::S)
              << std::endl;
    ShowProgress progress(Seconds(1), std::cout);
    Simulator::Stop(simTime);
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
