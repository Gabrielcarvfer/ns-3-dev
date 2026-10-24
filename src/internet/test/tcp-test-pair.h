/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#ifndef TCP_TEST_PAIR_H
#define TCP_TEST_PAIR_H

#include "ns3/callback.h"
#include "ns3/data-rate.h"
#include "ns3/inet-socket-address.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-interface-container.h"
#include "ns3/net-device-container.h"
#include "ns3/node-container.h"
#include "ns3/nstime.h"
#include "ns3/packet.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/socket.h"
#include "ns3/tcp-socket-factory.h"

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

/**
 * @ingroup internet-test
 *
 * Create a TCP socket listening on a port of a node.
 *
 * @param node the node
 * @param port the port
 * @param accepted the callback told of each new connection, if not null
 * @return the listening socket
 */
inline Ptr<Socket>
InstallTcpListener(Ptr<Node> node,
                   uint16_t port,
                   Callback<void, Ptr<Socket>, const Address&> accepted =
                       MakeNullCallback<void, Ptr<Socket>, const Address&>())
{
    Ptr<Socket> server = Socket::CreateSocket(node, TcpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    server->Listen();
    if (!accepted.IsNull())
    {
        server->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(), accepted);
    }
    return server;
}

/**
 * @ingroup internet-test
 *
 * Create a TCP socket listening on a port of a node, whose connections read
 * all the data they receive.
 *
 * @param node the node
 * @param port the port
 * @param received the counter of the bytes read, which must outlive the simulation
 * @return the listening socket
 */
inline Ptr<Socket>
InstallTcpByteSink(Ptr<Node> node, uint16_t port, uint32_t& received)
{
    auto read = [&received](Ptr<Socket> socket) {
        while (Ptr<Packet> packet = socket->Recv())
        {
            received += packet->GetSize();
        }
    };
    auto accepted = [read](Ptr<Socket> socket, const Address&) {
        socket->SetRecvCallback(Callback<void, Ptr<Socket>>(read));
    };
    return InstallTcpListener(node, port, Callback<void, Ptr<Socket>, const Address&>(accepted));
}

} // namespace ns3

#endif /* TCP_TEST_PAIR_H */
