/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#ifndef TCP_TEST_PAIR_H
#define TCP_TEST_PAIR_H

#include "ns3/data-rate.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-interface-container.h"
#include "ns3/net-device-container.h"
#include "ns3/node-container.h"
#include "ns3/nstime.h"
#include "ns3/simple-net-device-helper.h"

#include <string>

namespace ns3
{

/**
 * @ingroup internet-test
 *
 * Create two nodes linked by a SimpleNetDevice channel, with the Internet
 * stack and the 10.1.1.0/24 addresses, as most TCP socket tests need.
 *
 * @param nodes the container the two nodes are created in
 * @param dataRate the data rate of the devices, or empty for the default
 * @param delay the delay of the channel, or zero for the default
 * @param devices if not null, the container receiving the devices
 * @return the interfaces of the two nodes
 */
inline Ipv4InterfaceContainer
InstallTcpTestPair(NodeContainer& nodes,
                   const std::string& dataRate = "",
                   Time delay = Time(0),
                   NetDeviceContainer* devices = nullptr)
{
    nodes.Create(2);
    SimpleNetDeviceHelper devHelper;
    if (!dataRate.empty())
    {
        devHelper.SetDeviceAttribute("DataRate", DataRateValue(DataRate(dataRate)));
    }
    if (!delay.IsZero())
    {
        devHelper.SetChannelAttribute("Delay", TimeValue(delay));
    }
    NetDeviceContainer installed = devHelper.Install(nodes);
    if (devices != nullptr)
    {
        *devices = installed;
    }
    InternetStackHelper stack;
    stack.Install(nodes);
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    return address.Assign(installed);
}

} // namespace ns3

#endif /* TCP_TEST_PAIR_H */
