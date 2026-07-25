/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

/**
 * @file
 * Global routing at scale: an N x N grid of routers with point-to-point
 * links, global routing tables and random UDP flows, stressing the SPF
 * computation, routing table population and the per-packet route lookup.
 *
 * Scaling reference (default build, 20x20 grid, 400 flows, 5 s):
 *   before the global routing indexing work: 21 s route computation,
 *                                            440 s total wall
 *   after:                                   0.6 s route computation,
 *                                            49 s total wall
 *
 * Example: ./ns3 run "global-routing-grid --grid=20 --flows=400"
 * Use --randomMetrics=1 to exercise the SPF candidate queue harder.
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

#include <chrono>

using namespace ns3;

int
main(int argc, char* argv[])
{
    uint32_t grid = 12; // grid side; nodes = grid^2
    uint32_t flows = 200;
    Time simTime = Seconds(10);
    bool randomMetrics = false;
    CommandLine cmd(__FILE__);
    cmd.Usage("Global routing over a square grid of point-to-point routers.\n"
              "Route computation is timed separately from the simulation, since\n"
              "populating the routing tables dominates at large grid sizes.");
    cmd.AddValue("grid", "Grid side length (nodes = grid^2)", grid);
    cmd.AddValue("flows", "Number of UDP flows", flows);
    cmd.AddValue("simTime", "Simulation time", simTime);
    cmd.AddValue("randomMetrics", "Assign random link metrics (stresses SPF)", randomMetrics);
    cmd.Parse(argc, argv);

    NodeContainer nodes(grid * grid);

    InternetStackHelper stack;
    stack.Install(nodes);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("1Gbps"));
    p2p.SetChannelAttribute("Delay", TimeValue(MicroSeconds(50)));

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.0.0.0", "255.255.255.252");
    std::vector<Ipv4Address> nodeAddr(nodes.GetN());
    auto connect = [&](uint32_t a, uint32_t b) {
        auto d = p2p.Install(nodes.Get(a), nodes.Get(b));
        auto ifs = ipv4.Assign(d);
        nodeAddr[a] = ifs.GetAddress(0);
        nodeAddr[b] = ifs.GetAddress(1);
        ipv4.NewNetwork();
    };
    for (uint32_t r = 0; r < grid; r++)
    {
        for (uint32_t c = 0; c < grid; c++)
        {
            uint32_t n = r * grid + c;
            if (c + 1 < grid)
            {
                connect(n, n + 1);
            }
            if (r + 1 < grid)
            {
                connect(n, n + grid);
            }
        }
    }

    if (randomMetrics)
    {
        Ptr<UniformRandomVariable> metricRng = CreateObject<UniformRandomVariable>();
        for (uint32_t n = 0; n < nodes.GetN(); n++)
        {
            Ptr<Ipv4> ip = nodes.Get(n)->GetObject<Ipv4>();
            for (uint32_t i = 1; i < ip->GetNInterfaces(); i++)
            {
                ip->SetMetric(i, metricRng->GetInteger(1, 16));
            }
        }
    }

    auto tr0 = std::chrono::steady_clock::now();
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
    auto tr1 = std::chrono::steady_clock::now();
    std::cout << "route computation: " << std::chrono::duration<double>(tr1 - tr0).count() << "s"
              << std::endl;

    uint16_t port = 9000;
    Ptr<UniformRandomVariable> rng = CreateObject<UniformRandomVariable>();
    ApplicationContainer apps;
    OnOffHelper onoff("ns3::UdpSocketFactory", Address());
    onoff.SetConstantRate(DataRate("2Mbps"), 1200);
    for (uint32_t f = 0; f < flows; f++)
    {
        uint32_t src = rng->GetInteger(0, nodes.GetN() - 1);
        // Pick any node other than the source, without a retry loop.
        uint32_t dst = (src + 1 + rng->GetInteger(0, nodes.GetN() - 2)) % nodes.GetN();
        onoff.SetAttribute("Remote", AddressValue(InetSocketAddress(nodeAddr[dst], port + f)));
        apps.Add(onoff.Install(nodes.Get(src)));
        PacketSinkHelper sink("ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), port + f));
        apps.Add(sink.Install(nodes.Get(dst)));
    }
    apps.Start(Seconds(0.1));

    std::cout << "nodes=" << nodes.GetN() << " flows=" << flows
              << " simTime=" << simTime.As(Time::S) << std::endl;
    ShowProgress progress(Seconds(1), std::cout);
    Simulator::Stop(simTime);
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
