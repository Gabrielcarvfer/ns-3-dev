/*
 * Copyright (c) 2012 University of Washington, 2012 INRIA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Alina Quereilhac <alina.quereilhac@inria.fr>
 *
 */

//
//        node 0                          node 1
//  +----------------+              +----------------+
//  |    ns-3 TCP    |              |    ns-3 TCP    |
//  +----------------+              +----------------+
//  |    10.1.1.1    |              |    10.1.1.2    |
//  +----------------+  socketpair  +----------------+
//  |  fd-net-device |--------------|  fd-net-device |
//  +----------------+              +----------------+
//
// This example is aimed at measuring the throughput of the FdNetDevice
// in a pure simulation. For this purpose two FdNetDevices, attached to
// different nodes but in a same simulation, are connected using a socket pair.
// TCP traffic is sent at a saturating data rate. Then the throughput can
// be obtained from the generated .pcap files.
//
// Steps to run the experiment:
//
// $ ./ns3 run "fd2fd-onoff"
//

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/fd-net-device-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"

#include <errno.h>
#include <sys/socket.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("RealtimeFdNetDeviceSaturationExample");

int
main(int argc, char* argv[])
{
    CommandLine cmd(__FILE__);
    cmd.Parse(argc, argv);

    uint16_t sinkPort = 8000;
    uint32_t packetSize = 10000; // bytes
    std::string dataRate("1000Mb/s");

    GlobalValue::Bind("SimulatorImplementationType", StringValue("ns3::RealtimeSimulatorImpl"));
    GlobalValue::Bind("ChecksumEnabled", BooleanValue(true));

    NS_LOG_INFO("Create Node");
    NodeContainer nodes;
    nodes.Create(2);

    NS_LOG_INFO("Create Device");
    FdNetDeviceHelper fd;
    NetDeviceContainer devices = fd.Install(nodes);

    int sv[2];
    if (socketpair(AF_UNIX, SOCK_DGRAM, 0, sv) < 0)
    {
        NS_FATAL_ERROR("Error creating pipe=" << strerror(errno));
    }

    Ptr<NetDevice> d1 = devices.Get(0);
    Ptr<FdNetDevice> clientDevice = d1->GetObject<FdNetDevice>();
    clientDevice->SetFileDescriptor(sv[0]);

    Ptr<NetDevice> d2 = devices.Get(1);
    Ptr<FdNetDevice> serverDevice = d2->GetObject<FdNetDevice>();
    serverDevice->SetFileDescriptor(sv[1]);

    NS_LOG_INFO("Add Internet Stack");
    InternetStackHelper internetStackHelper;
    internetStackHelper.SetIpv4StackInstall(true);
    internetStackHelper.Install(nodes);

    NS_LOG_INFO("Create IPv4 Interface");
    Ipv4AddressHelper addresses;
    addresses.SetBase("10.0.0.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = addresses.Assign(devices);

    Ptr<Node> clientNode = nodes.Get(0);
    Ipv4Address serverIp = interfaces.GetAddress(1);
    Ptr<Node> serverNode = nodes.Get(1);

    // server
    Address sinkLocalAddress(InetSocketAddress(serverIp, sinkPort));
    PacketSinkHelper sinkHelper("ns3::TcpSocketFactory", sinkLocalAddress);
    ApplicationContainer sinkApp = sinkHelper.Install(serverNode);
    sinkApp.Start(Seconds(0));
    sinkApp.Stop(Seconds(40));
    fd.EnablePcap("rt-fd2fd-onoff-server", serverDevice);

    // client
    AddressValue serverAddress(InetSocketAddress(serverIp, sinkPort));
    OnOffHelper onoff("ns3::TcpSocketFactory", Address());
    onoff.SetAttribute("Remote", serverAddress);
    onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    onoff.SetAttribute("DataRate", DataRateValue(dataRate));
    onoff.SetAttribute("PacketSize", UintegerValue(packetSize));
    ApplicationContainer clientApps = onoff.Install(clientNode);
    clientApps.Start(Seconds(1));
    clientApps.Stop(Seconds(39));
    fd.EnablePcap("rt-fd2fd-onoff-client", clientDevice);

    Simulator::Stop(Seconds(40));
    Simulator::Run();
    Simulator::Destroy();

    return 0;
}
