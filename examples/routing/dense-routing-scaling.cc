/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

/**
 * @file
 * Ad hoc routing protocols in a large, fully dense broadcast domain.
 *
 * Every node shares one CSMA segment, so each node has N-1 neighbors and
 * overhears every transmission. This is the worst case for the per-node
 * routing state of each protocol:
 *  - AODV: every route request reaches every node, and the routing table
 *    (purged on every route lookup) grows with the node count.
 *  - DSDV: every periodic and triggered update carries N routes.
 *  - DSR: the promiscuous receive path maps MAC to IP addresses per packet,
 *    and the route caches grow with the node count.
 *  - OLSR: roughly (N-1)*(N-2) two-hop entries, stressing HELLO/TC
 *    processing, MPR selection and routing table computation.
 *
 * Scaling reference (default build):
 *   AODV, 300 nodes, 300 flows, 10 s:
 *     before the lazy routing table purge: 87 s wall
 *     after:                               41 s wall
 *   DSR, 80 nodes, 80 flows, 10 s:
 *     before caching the MAC-to-IP mapping: 45 s wall
 *     after:                                26 s wall
 *   OLSR, 150 nodes, no flows, 20 s:
 *     before the hash-index/recomputation-skip work: 511 s wall
 *     after:                                          47 s wall
 *
 * Examples:
 *   ./ns3 run "dense-routing-scaling --protocol=aodv --nodes=300 --flows=300 --simTime=10s"
 *   ./ns3 run "dense-routing-scaling --protocol=dsr --nodes=80 --flows=80 --simTime=10s"
 *   ./ns3 run "dense-routing-scaling --protocol=olsr --nodes=150 --flows=0 --simTime=20s"
 */

#include "ns3/aodv-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/dsdv-module.h"
#include "ns3/dsr-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/olsr-module.h"

using namespace ns3;

int
main(int argc, char* argv[])
{
    std::string protocol = "aodv";
    uint32_t nNodes = 200;
    uint32_t flows = 200;
    Time simTime = Seconds(20);
    CommandLine cmd(__FILE__);
    cmd.Usage("Ad hoc routing protocol in a dense broadcast domain: every node shares\n"
              "one CSMA segment, so every node hears every other node and the routing\n"
              "state grows with the node count.");
    cmd.AddValue("protocol", "Routing protocol: aodv, dsdv, dsr or olsr", protocol);
    cmd.AddValue("nodes", "Number of nodes", nNodes);
    cmd.AddValue("flows", "Number of UDP flows", flows);
    cmd.AddValue("simTime", "Simulation time", simTime);
    cmd.Parse(argc, argv);

    NodeContainer nodes(nNodes);

    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", StringValue("1Gbps"));
    csma.SetChannelAttribute("Delay", TimeValue(MicroSeconds(1)));
    NetDeviceContainer devices = csma.Install(nodes);

    InternetStackHelper stack;
    AodvHelper aodv;
    DsdvHelper dsdv;
    OlsrHelper olsr;
    if (protocol == "aodv")
    {
        stack.SetRoutingHelper(aodv);
    }
    else if (protocol == "dsdv")
    {
        stack.SetRoutingHelper(dsdv);
    }
    else if (protocol == "olsr")
    {
        stack.SetRoutingHelper(olsr);
    }
    else if (protocol != "dsr")
    {
        NS_FATAL_ERROR("Unknown protocol: " << protocol);
    }
    stack.Install(nodes);
    if (protocol == "dsr")
    {
        DsrHelper dsr;
        DsrMainHelper dsrMain;
        dsrMain.Install(dsr, nodes);
    }

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

    std::cout << "protocol=" << protocol << " nodes=" << nNodes << " flows=" << flows
              << " simTime=" << simTime.As(Time::S) << std::endl;
    ShowProgress progress(Seconds(1), std::cout);
    Simulator::Stop(simTime);
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
