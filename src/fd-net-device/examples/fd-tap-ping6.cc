/*
 * Copyright (c) 2012 University of Washington, 2012 INRIA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

// Allow ns-3 to ping a TAP device in the host machine.
//
//   -------------------------------------------------
//   | ns-3 simulation                               |
//   |                                               |
//   |  -------                        --------      |
//   | | node  |                      |  node  |     |
//   | | (r)   |                      |  (n)   |     |
//   | |       |                      |        |     |
//   |  ------- --------               --------      |
//   | | fd-   | csma-  |             | csma-  |     |
//   | | net-  | net-   |             | net-   |     |
//   | | device| device |             | device |     |
//   |  ------- --------               --------      |
//   |   |          |____csma channel_____|          |
//   |   |                                           |
//   ----|------------------------------------------
//   |  ---            |
//   | |   |           |
//   | |TAP|           |
//   | |   |           |
//   |  ---            |
//   |                 |
//   |  host           |
//   ------------------
//
//

#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/fd-net-device-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/internet-module.h"

#include <sstream>
#include <string>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TAPPing6Example");

int
main(int argc, char* argv[])
{
    CommandLine cmd(__FILE__);
    cmd.Parse(argc, argv);

    NS_LOG_INFO("Ping6 Emulation Example with TAP");

    //
    // Since we are using a real piece of hardware we need to use the realtime
    // simulator.
    //
    GlobalValue::Bind("SimulatorImplementationType", StringValue("ns3::RealtimeSimulatorImpl"));

    //
    // Since we are going to be talking to real-world machines, we need to enable
    // calculation of checksums in our protocols.
    //
    GlobalValue::Bind("ChecksumEnabled", BooleanValue(true));

    //
    // Create the two nodes.
    //
    Ptr<Node> n = CreateObject<Node>();
    Ptr<Node> r = CreateObject<Node>();
    NodeContainer net(n, r);

    //
    // Install IPv6 stack.
    //
    InternetStackHelper internetv6;
    internetv6.Install(net);

    //
    // Create CSMA channel.
    //
    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", DataRateValue(5000000));
    csma.SetChannelAttribute("Delay", TimeValue(MilliSeconds(2)));
    NetDeviceContainer devs = csma.Install(net);

    //
    // Assign IPv6 addresses.
    //
    Ipv6AddressHelper ipv6;

    ipv6.SetBase(Ipv6Address("4001:beef:1::"), Ipv6Prefix(64));
    Ipv6InterfaceContainer i1 = ipv6.Assign(devs);
    i1.SetForwarding(1, true);
    i1.SetDefaultRouteInAllNodes(1);

    ipv6.SetBase(Ipv6Address("4001:beef:2::"), Ipv6Prefix(64));
    Ipv6Address tapAddr = ipv6.NewAddress();
    std::stringstream ss;
    std::string tapIp;
    tapAddr.Print(ss);
    ss >> tapIp;

    //
    // Create FdNetDevice.
    //
    TapFdNetDeviceHelper helper;
    helper.SetDeviceName("tap0");
    helper.SetTapIpv6Address(tapIp.c_str());
    helper.SetTapIpv6Prefix(64);

    NetDeviceContainer fdevs = helper.Install(r);
    Ptr<NetDevice> device = fdevs.Get(0);
    Ptr<FdNetDevice> fdevice = device->GetObject<FdNetDevice>();
    fdevice->SetIsMulticast(true);
    Ipv6InterfaceContainer i2 = ipv6.Assign(fdevs);
    i2.SetForwarding(0, true);
    i2.SetDefaultRouteInAllNodes(0);

    //
    // Create the Ping6 application.
    //
    uint32_t packetSize = 1024;
    uint32_t maxPacketCount = 1;
    Time interPacketInterval = Seconds(1);

    PingHelper ping(Ipv6Address(tapIp.c_str()));
    ping.SetAttribute("Count", UintegerValue(maxPacketCount));
    ping.SetAttribute("Interval", TimeValue(interPacketInterval));
    ping.SetAttribute("Size", UintegerValue(packetSize));
    ApplicationContainer apps = ping.Install(n);

    // Ping6Helper ping6;

    // ping6.SetRemote(tapIp.c_str());

    // ping6.SetAttribute("MaxPackets", UintegerValue(maxPacketCount));
    // ping6.SetAttribute("Interval", TimeValue(interPacketInterval));
    // ping6.SetAttribute("PacketSize", UintegerValue(packetSize));
    // ApplicationContainer apps = ping6.Install(n);
    apps.Start(Seconds(2));
    apps.Stop(Seconds(20));

    AsciiTraceHelper ascii;
    csma.EnableAsciiAll(ascii.CreateFileStream("csma-ping6.tr"));
    csma.EnablePcapAll("csma-ping6", true);

    //
    // Enable a promiscuous pcap trace to see what is coming and going on in the fd-net-device.
    //
    helper.EnablePcap("fd-ping6", fdevice, true);

    //
    // Run the experiment.
    //
    NS_LOG_INFO("Run Emulation.");
    Simulator::Stop(Seconds(200));
    Simulator::Run();
    Simulator::Destroy();
    NS_LOG_INFO("Done.");

    return 0;
}
