/*
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Lalith Suresh <suresh.lalith@gmail.com>
 *
 */

//
// This script, adapted from examples/wireless/wifi-simple-adhoc illustrates
// the use of OLSR HNA.
//
// Network Topology:
//
//             |------ OLSR ------|   |---- non-OLSR ----|
//           A ))))            (((( B ------------------- C
//           10.1.1.1     10.1.1.2   172.16.1.2     172.16.1.1
//
// Node A needs to send a UDP packet to node C. This can be done only after
// A receives an HNA message from B, in which B announces 172.16.1.0/24
// as an associated network.
//
// If no HNA message is generated by B, a will not be able to form a route to C.
// This can be verified as follows:
//
// ./ns3 run olsr-hna
//
// There are two ways to make a node to generate HNA messages.
//
// One way is to use olsr::RoutingProtocol::SetRoutingTableAssociation ()
// to use which you may run:
//
// ./ns3 run "olsr-hna --assocMethod=SetRoutingTable"
//
// The other way is to use olsr::RoutingProtocol::AddHostNetworkAssociation ()
// to use which you may run:
//
// ./ns3 run "olsr-hna --assocMethod=AddHostNetwork"
//

#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/olsr-helper.h"
#include "ns3/olsr-routing-protocol.h"
#include "ns3/yans-wifi-helper.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("OlsrHna");

void
ReceivePacket(Ptr<Socket> socket)
{
    NS_LOG_UNCOND("Received one packet!");
}

static void
GenerateTraffic(Ptr<Socket> socket, uint32_t pktSize, uint32_t pktCount, Time pktInterval)
{
    if (pktCount > 0)
    {
        socket->Send(Create<Packet>(pktSize));
        Simulator::Schedule(pktInterval,
                            &GenerateTraffic,
                            socket,
                            pktSize,
                            pktCount - 1,
                            pktInterval);
    }
    else
    {
        socket->Close();
    }
}

int
main(int argc, char* argv[])
{
    std::string phyMode("DsssRate1Mbps");
    double rss = -80;           // -dBm
    uint32_t packetSize = 1000; // bytes
    uint32_t numPackets = 1;
    double interval = 1.0; // seconds
    bool verbose = false;
    std::string assocMethod("SetRoutingTable");
    bool reverse = false;

    CommandLine cmd(__FILE__);

    cmd.AddValue("phyMode", "Wifi Phy mode", phyMode);
    cmd.AddValue("rss", "received signal strength", rss);
    cmd.AddValue("packetSize", "size of application packet sent", packetSize);
    cmd.AddValue("numPackets", "number of packets generated", numPackets);
    cmd.AddValue("interval", "interval (seconds) between packets", interval);
    cmd.AddValue("verbose", "turn on all WifiNetDevice log components", verbose);
    cmd.AddValue("assocMethod",
                 "How to add the host to the HNA table (SetRoutingTable, "
                 "AddHostNetwork)",
                 assocMethod);
    cmd.AddValue("reverse", "Send packets from CSMA to WiFi if set to true", reverse);

    cmd.Parse(argc, argv);
    // Convert to time object
    Time interPacketInterval = Seconds(interval);

    // disable fragmentation for frames below 2200 bytes
    Config::SetDefault("ns3::WifiRemoteStationManager::FragmentationThreshold",
                       StringValue("2200"));
    // turn off RTS/CTS for frames below 2200 bytes
    Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold", StringValue("2200"));
    // Fix non-unicast data rate to be the same as that of unicast
    Config::SetDefault("ns3::WifiRemoteStationManager::NonUnicastMode", StringValue(phyMode));

    NodeContainer olsrNodes;
    olsrNodes.Create(2);

    NodeContainer csmaNodes;
    csmaNodes.Create(1);

    // The below set of helpers will help us to put together the wifi NICs we want
    WifiHelper wifi;
    if (verbose)
    {
        WifiHelper::EnableLogComponents(); // Turn on all Wifi logging
    }
    wifi.SetStandard(WIFI_STANDARD_80211b);

    YansWifiPhyHelper wifiPhy;
    // This is one parameter that matters when using FixedRssLossModel
    // set it to zero; otherwise, gain will be added
    wifiPhy.Set("RxGain", DoubleValue(0));
    // ns-3 supports RadioTap and Prism tracing extensions for 802.11b
    wifiPhy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);

    YansWifiChannelHelper wifiChannel;
    wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    // The below FixedRssLossModel will cause the rss to be fixed regardless
    // of the distance between the two stations, and the transmit power
    wifiChannel.AddPropagationLoss("ns3::FixedRssLossModel", "Rss", DoubleValue(rss));
    wifiPhy.SetChannel(wifiChannel.Create());

    // Add a mac and disable rate control
    WifiMacHelper wifiMac;
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue(phyMode),
                                 "ControlMode",
                                 StringValue(phyMode));
    // Set it to adhoc mode
    wifiMac.SetType("ns3::AdhocWifiMac");
    NetDeviceContainer devices = wifi.Install(wifiPhy, wifiMac, olsrNodes);

    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", DataRateValue(DataRate(5000000)));
    csma.SetChannelAttribute("Delay", TimeValue(MilliSeconds(2)));
    NetDeviceContainer csmaDevices =
        csma.Install(NodeContainer(csmaNodes.Get(0), olsrNodes.Get(1)));

    // Note that with FixedRssLossModel, the positions below are not
    // used for received signal strength.
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    positionAlloc->Add(Vector(0.0, 0.0, 0.0));
    positionAlloc->Add(Vector(5.0, 0.0, 0.0));
    mobility.SetPositionAllocator(positionAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(olsrNodes);

    OlsrHelper olsr;

    // Specify Node B's csma device as a non-OLSR device.
    olsr.ExcludeInterface(olsrNodes.Get(1), 2);

    Ipv4StaticRoutingHelper staticRouting;

    Ipv4ListRoutingHelper list;
    list.Add(staticRouting, 0);
    list.Add(olsr, 10);

    InternetStackHelper internet_olsr;
    internet_olsr.SetRoutingHelper(list); // has effect on the next Install ()
    internet_olsr.Install(olsrNodes);

    InternetStackHelper internet_csma;
    internet_csma.Install(csmaNodes);

    Ipv4AddressHelper ipv4;
    NS_LOG_INFO("Assign IP Addresses.");
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    ipv4.Assign(devices);

    ipv4.SetBase("172.16.1.0", "255.255.255.0");
    ipv4.Assign(csmaDevices);

    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
    Ptr<Socket> recvSink;
    if (!reverse)
    {
        recvSink = Socket::CreateSocket(csmaNodes.Get(0), tid);
    }
    else
    {
        recvSink = Socket::CreateSocket(olsrNodes.Get(0), tid);
    }
    InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), 80);
    recvSink->Bind(local);
    recvSink->SetRecvCallback(MakeCallback(&ReceivePacket));

    Ptr<Socket> source;
    if (!reverse)
    {
        source = Socket::CreateSocket(olsrNodes.Get(0), tid);
        InetSocketAddress remote = InetSocketAddress(Ipv4Address("172.16.1.1"), 80);
        source->Connect(remote);
    }
    else
    {
        source = Socket::CreateSocket(csmaNodes.Get(0), tid);
        InetSocketAddress remote = InetSocketAddress(Ipv4Address("10.1.1.1"), 80);
        source->Connect(remote);
    }

    // Obtain olsr::RoutingProtocol instance of gateway node
    // (namely, node B) and add the required association
    Ptr<olsr::RoutingProtocol> olsrrp_Gw = Ipv4RoutingHelper::GetRouting<olsr::RoutingProtocol>(
        olsrNodes.Get(1)->GetObject<Ipv4>()->GetRoutingProtocol());

    if (assocMethod == "SetRoutingTable")
    {
        // Create a special Ipv4StaticRouting instance for RoutingTableAssociation
        // Even the Ipv4StaticRouting instance added to list may be used
        Ptr<Ipv4StaticRouting> hnaEntries = CreateObject<Ipv4StaticRouting>();

        // Add the required routes into the Ipv4StaticRouting Protocol instance
        // and have the node generate HNA messages for all these routes
        // which are associated with non-OLSR interfaces specified above.
        hnaEntries->AddNetworkRouteTo(Ipv4Address("172.16.1.0"),
                                      Ipv4Mask("255.255.255.0"),
                                      uint32_t(2),
                                      uint32_t(1));
        olsrrp_Gw->SetRoutingTableAssociation(hnaEntries);
    }
    else if (assocMethod == "AddHostNetwork")
    {
        // Specify the required associations directly.
        olsrrp_Gw->AddHostNetworkAssociation(Ipv4Address("172.16.1.0"), Ipv4Mask("255.255.255.0"));
    }
    else
    {
        std::cout << "invalid HnaMethod option (" << assocMethod << ")" << std::endl;
        exit(0);
    }

    // Add a default route to the CSMA node (to enable replies)
    Ptr<Ipv4StaticRouting> staticRoutingProt;
    staticRoutingProt = Ipv4RoutingHelper::GetRouting<Ipv4StaticRouting>(
        csmaNodes.Get(0)->GetObject<Ipv4>()->GetRoutingProtocol());
    staticRoutingProt->AddNetworkRouteTo("10.1.1.0", "255.255.255.0", "172.16.1.2", 1);

    // Tracing
    wifiPhy.EnablePcap("olsr-hna-wifi", devices);
    csma.EnablePcap("olsr-hna-csma", csmaDevices, false);
    AsciiTraceHelper ascii;
    wifiPhy.EnableAsciiAll(ascii.CreateFileStream("olsr-hna-wifi.tr"));
    csma.EnableAsciiAll(ascii.CreateFileStream("olsr-hna-csma.tr"));

    Simulator::ScheduleWithContext(source->GetNode()->GetId(),
                                   Seconds(15),
                                   &GenerateTraffic,
                                   source,
                                   packetSize,
                                   numPackets,
                                   interPacketInterval);

    Ptr<OutputStreamWrapper> routingStream =
        Create<OutputStreamWrapper>("olsr-hna.routes", std::ios::out);
    Ipv4RoutingHelper::PrintRoutingTableAllAt(Seconds(15), routingStream);

    Simulator::Stop(Seconds(20));
    Simulator::Run();
    Simulator::Destroy();

    return 0;
}
