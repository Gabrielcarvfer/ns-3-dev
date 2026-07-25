/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

/**
 * @file
 * OLSR route discovery in a large, fully dense broadcast domain.
 *
 * Every node shares one CSMA segment, so each node has N-1 neighbors and
 * roughly (N-1)*(N-2) two-hop entries: the worst case for OLSR HELLO/TC
 * processing, MPR selection and routing table computation.
 *
 * Scaling reference (default build, 150 nodes, 20 s):
 *   before the OLSR hash-index/recomputation-skip work: 511 s wall
 *   after:                                               47 s wall
 *
 * Example: ./ns3 run "olsr-dense-scaling --nodes=150 --simTime=20"
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/olsr-module.h"

using namespace ns3;

int
main(int argc, char* argv[])
{
    uint32_t nNodes = 200;
    Time simTime = Seconds(30);
    CommandLine cmd(__FILE__);
    cmd.Usage("OLSR in a fully dense broadcast domain, the worst case for HELLO/TC\n"
              "processing, MPR selection and routing table computation.");
    cmd.AddValue("nodes", "Number of nodes", nNodes);
    cmd.AddValue("simTime", "Simulation time", simTime);
    cmd.Parse(argc, argv);

    NodeContainer nodes(nNodes);

    // Single shared CSMA segment: every node hears every other node,
    // maximizing OLSR neighbor density (dense datacenter-like broadcast domain).
    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", StringValue("1Gbps"));
    csma.SetChannelAttribute("Delay", TimeValue(MicroSeconds(1)));
    NetDeviceContainer devices = csma.Install(nodes);

    OlsrHelper olsr;
    InternetStackHelper stack;
    stack.SetRoutingHelper(olsr);
    stack.Install(nodes);

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.0.0.0", "255.255.0.0");
    ipv4.Assign(devices);

    std::cout << "nodes=" << nNodes << " simTime=" << simTime.As(Time::S) << std::endl;
    ShowProgress progress(Seconds(1), std::cout);
    Simulator::Stop(simTime);
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
