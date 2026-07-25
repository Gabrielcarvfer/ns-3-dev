/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

/**
 * @file
 * DSR route discovery and forwarding in a dense broadcast domain with
 * many concurrent flows, stressing the promiscuous receive path (which
 * maps MAC to IP addresses per packet) and the route cache purges.
 *
 * Scaling reference (default build, 80 nodes, 80 flows, 10 s):
 *   before caching the MAC-to-IP mapping: 45 s wall
 *   after:                                26 s wall
 *
 * Example: ./ns3 run "dsr-dense-scaling --nodes=80 --flows=80"
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/dsr-module.h"
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
    cmd.Usage("DSR in a dense broadcast domain: every node shares one CSMA segment,\n"
              "so every node overhears every transmission and the route caches grow\n"
              "with the node count.");
    cmd.AddValue("nodes", "Number of nodes", nNodes);
    cmd.AddValue("flows", "Number of UDP flows", flows);
    cmd.AddValue("simTime", "Simulation time", simTime);
    cmd.Parse(argc, argv);

    NodeContainer nodes(nNodes);

    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", StringValue("1Gbps"));
    csma.SetChannelAttribute("Delay", TimeValue(MicroSeconds(1)));
    NetDeviceContainer devices = csma.Install(nodes);

    DsrHelper dsr;
    DsrMainHelper dsrMain;
    InternetStackHelper stack;
    stack.Install(nodes);
    dsrMain.Install(dsr, nodes);

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.0.0.0", "255.255.0.0");
    Ipv4InterfaceContainer ifs = ipv4.Assign(devices);

    uint16_t port = 9000;
    Ptr<UniformRandomVariable> rng = CreateObject<UniformRandomVariable>();
    ApplicationContainer apps;
    for (uint32_t f = 0; f < flows; f++)
    {
        uint32_t src = rng->GetInteger(0, nNodes - 1);
        uint32_t dst = rng->GetInteger(0, nNodes - 1);
        if (src == dst)
        {
            dst = (dst + 1) % nNodes;
        }
        OnOffHelper onoff("ns3::UdpSocketFactory",
                          InetSocketAddress(ifs.GetAddress(dst), port + f));
        onoff.SetConstantRate(DataRate("500kbps"), 1200);
        apps.Add(onoff.Install(nodes.Get(src)));
        PacketSinkHelper sink("ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), port + f));
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
