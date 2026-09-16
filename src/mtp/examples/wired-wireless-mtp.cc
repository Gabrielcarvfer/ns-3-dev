/*
 * Copyright (c) 2026 Universidade de Brasilia
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

/**
 * @file
 * @ingroup mtp
 *
 * A mixed wired and wireless topology used to check that the multithreaded
 * simulator reproduces the results of the sequential simulator.
 *
 * Several Wi-Fi BSSs (an AP and a few STAs each) and a CSMA LAN are connected
 * through point-to-point links to a router, which is connected to a server.
 * Every STA and LAN host runs a TCP flow and a UDP flow towards the server, and
 * the server sends a UDP flow back to every STA.
 *
 * The automatic partition puts every wireless (or CSMA) segment in its own
 * logical process, since shared channels are never cut; the point-to-point
 * links carry the traffic between the logical processes.
 *
 *   STA ... STA        STA ... STA        host ... host
 *      \   /              \   /              |   |
 *       AP0                AP1              CSMA LAN
 *        |                  |                  |
 *        +-------- p2p --- router --- p2p -----+
 *                            |
 *                          server
 *
 * The flow statistics are printed keyed by the flow five-tuple, since the flow
 * ids of the flow monitor depend on the order in which flows are first seen,
 * which is not reproducible when the packets of different LPs are seen at the
 * same simulation time.
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/mtp-interface.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/wifi-module.h"

#include <iomanip>
#include <map>
#include <sstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("WiredWirelessMtp");

int
main(int argc, char* argv[])
{
    uint32_t thread = 4;
    uint32_t nBss = 2;
    uint32_t nSta = 3;
    uint32_t nLan = 3;
    double duration = 5;
    bool verbose = false;

    CommandLine cmd(__FILE__);
    cmd.AddValue("thread",
                 "Maximum number of threads (0 uses the default sequential simulator)",
                 thread);
    cmd.AddValue("nBss", "Number of Wi-Fi BSSs", nBss);
    cmd.AddValue("nSta", "Number of STAs per BSS", nSta);
    cmd.AddValue("nLan", "Number of hosts in the CSMA LAN", nLan);
    cmd.AddValue("duration", "Simulation duration in seconds", duration);
    cmd.AddValue("verbose", "Print the packets received by the sinks", verbose);
    cmd.Parse(argc, argv);

    if (thread > 0)
    {
        MtpInterface::Enable(thread);
    }

    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    NodeContainer serverNode;
    serverNode.Create(1);
    NodeContainer routerNode;
    routerNode.Create(1);
    NodeContainer apNodes;
    apNodes.Create(nBss);
    std::vector<NodeContainer> staNodes(nBss);
    for (uint32_t i = 0; i < nBss; i++)
    {
        staNodes[i].Create(nSta);
    }
    NodeContainer lanNodes;
    lanNodes.Create(nLan);

    // Wired part
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));
    NetDeviceContainer serverDevices = p2p.Install(serverNode.Get(0), routerNode.Get(0));
    std::vector<NetDeviceContainer> apLinkDevices(nBss);
    for (uint32_t i = 0; i < nBss; i++)
    {
        apLinkDevices[i] = p2p.Install(apNodes.Get(i), routerNode.Get(0));
    }
    NetDeviceContainer lanGwDevices;
    NodeContainer lanGwNode;
    lanGwNode.Create(1);
    NetDeviceContainer lanLinkDevices = p2p.Install(lanGwNode.Get(0), routerNode.Get(0));

    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", StringValue("100Mbps"));
    csma.SetChannelAttribute("Delay", TimeValue(NanoSeconds(6560)));
    NodeContainer lan(lanGwNode, lanNodes);
    NetDeviceContainer lanDevices = csma.Install(lan);

    // Wireless part
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    wifi.SetRemoteStationManager("ns3::MinstrelHtWifiManager");
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    std::vector<NetDeviceContainer> apDevices(nBss);
    std::vector<NetDeviceContainer> staDevices(nBss);
    for (uint32_t i = 0; i < nBss; i++)
    {
        YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
        YansWifiPhyHelper phy;
        phy.SetChannel(channel.Create());
        WifiMacHelper mac;
        std::ostringstream ssid;
        ssid << "bss-" << i;
        mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(Ssid(ssid.str())));
        staDevices[i] = wifi.Install(phy, mac, staNodes[i]);
        mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(Ssid(ssid.str())));
        apDevices[i] = wifi.Install(phy, mac, apNodes.Get(i));

        Ptr<ListPositionAllocator> positions = CreateObject<ListPositionAllocator>();
        positions->Add(Vector(1000.0 * i, 0.0, 0.0));
        for (uint32_t j = 0; j < nSta; j++)
        {
            positions->Add(Vector(1000.0 * i + 5.0 * (j + 1), 3.0 * j, 0.0));
        }
        mobility.SetPositionAllocator(positions);
        mobility.Install(NodeContainer(apNodes.Get(i), staNodes[i]));
    }

    InternetStackHelper internet;
    internet.InstallAll();

    Ipv4AddressHelper address;
    address.SetBase("10.0.0.0", "255.255.255.0");
    Ipv4InterfaceContainer serverIfaces = address.Assign(serverDevices);
    std::vector<Ipv4InterfaceContainer> staIfaces(nBss);
    for (uint32_t i = 0; i < nBss; i++)
    {
        address.NewNetwork();
        address.Assign(apLinkDevices[i]);
        address.NewNetwork();
        address.Assign(apDevices[i]);
        staIfaces[i] = address.Assign(staDevices[i]);
    }
    address.NewNetwork();
    address.Assign(lanLinkDevices);
    address.NewNetwork();
    Ipv4InterfaceContainer lanIfaces = address.Assign(lanDevices);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // Applications
    const Ipv4Address serverAddress = serverIfaces.GetAddress(0);
    const uint16_t tcpPort = 5000;
    const uint16_t udpPort = 6000;
    const uint16_t downlinkPort = 7000;
    ApplicationContainer sinks;
    ApplicationContainer sources;

    PacketSinkHelper tcpSink("ns3::TcpSocketFactory",
                             InetSocketAddress(Ipv4Address::GetAny(), tcpPort));
    PacketSinkHelper udpSink("ns3::UdpSocketFactory",
                             InetSocketAddress(Ipv4Address::GetAny(), udpPort));
    sinks.Add(tcpSink.Install(serverNode));
    sinks.Add(udpSink.Install(serverNode));

    NodeContainer clients;
    for (uint32_t i = 0; i < nBss; i++)
    {
        clients.Add(staNodes[i]);
    }
    clients.Add(lanNodes);

    Ptr<UniformRandomVariable> startRv = CreateObject<UniformRandomVariable>();
    startRv->SetStream(100);
    for (uint32_t i = 0; i < clients.GetN(); i++)
    {
        BulkSendHelper bulk("ns3::TcpSocketFactory", InetSocketAddress(serverAddress, tcpPort));
        bulk.SetAttribute("MaxBytes", UintegerValue(200000));
        ApplicationContainer app = bulk.Install(clients.Get(i));
        app.Start(Seconds(1.0 + startRv->GetValue(0.0, 0.5)));
        sources.Add(app);

        OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress(serverAddress, udpPort));
        onoff.SetConstantRate(DataRate("500kbps"), 500);
        app = onoff.Install(clients.Get(i));
        app.Start(Seconds(1.0 + startRv->GetValue(0.0, 0.5)));
        sources.Add(app);
    }
    for (uint32_t i = 0; i < nBss; i++)
    {
        for (uint32_t j = 0; j < nSta; j++)
        {
            PacketSinkHelper dlSink("ns3::UdpSocketFactory",
                                    InetSocketAddress(Ipv4Address::GetAny(), downlinkPort));
            sinks.Add(dlSink.Install(staNodes[i].Get(j)));
            OnOffHelper onoff("ns3::UdpSocketFactory",
                              InetSocketAddress(staIfaces[i].GetAddress(j), downlinkPort));
            onoff.SetConstantRate(DataRate("1Mbps"), 1000);
            ApplicationContainer app = onoff.Install(serverNode);
            app.Start(Seconds(1.0 + startRv->GetValue(0.0, 0.5)));
            sources.Add(app);
        }
    }
    sinks.Start(Seconds(0.5));
    sources.Stop(Seconds(duration));

    if (verbose)
    {
        LogComponentEnable("PacketSink",
                           (LogLevel)(LOG_LEVEL_INFO | LOG_PREFIX_NODE | LOG_PREFIX_TIME));
    }

    FlowMonitorHelper flowHelper;
    Ptr<FlowMonitor> monitor = flowHelper.InstallAll();

    Simulator::Stop(Seconds(duration + 1));
    Simulator::Run();

    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowHelper.GetClassifier());
    std::map<std::string, std::string> results;
    for (const auto& [flowId, stats] : monitor->GetFlowStats())
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(flowId);
        std::ostringstream key;
        key << t.sourceAddress << ":" << t.sourcePort << " -> " << t.destinationAddress << ":"
            << t.destinationPort << " proto " << +t.protocol;
        std::ostringstream value;
        value << "tx " << stats.txPackets << "/" << stats.txBytes << " rx " << stats.rxPackets
              << "/" << stats.rxBytes << " lost " << stats.lostPackets << " delay "
              << stats.delaySum.GetMicroSeconds() << "us jitter "
              << stats.jitterSum.GetMicroSeconds() << "us last-rx "
              << stats.timeLastRxPacket.GetMicroSeconds() << "us";
        results[key.str()] = value.str();
    }
    for (const auto& [key, value] : results)
    {
        std::cout << key << " " << value << std::endl;
    }

    uint64_t sinkBytes = 0;
    for (uint32_t i = 0; i < sinks.GetN(); i++)
    {
        sinkBytes += DynamicCast<PacketSink>(sinks.Get(i))->GetTotalRx();
    }
    std::cout << "Total bytes received by the sinks: " << sinkBytes << std::endl;
    std::cout << "Event count: " << Simulator::GetEventCount() << std::endl;

    Simulator::Destroy();
    return 0;
}
