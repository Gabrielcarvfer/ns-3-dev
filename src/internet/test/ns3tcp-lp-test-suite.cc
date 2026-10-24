/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

// Bulk transfers through a bottleneck of 10 Mbit/s, 20 ms and a queue of 100
// packets, behind a router:
//
//            100 Mbit/s              10 Mbit/s
//  Sender A ------------+            20 ms, 100 packets
//              1 ms      Router ------------------------ Receiver
//  Sender B ------------+
//
// A single TCP-LP flow uses the bottleneck as a NewReno flow does, and
// TCP-LP yields it to a competing NewReno flow.

#include "ns3/bulk-send-helper.h"
#include "ns3/config.h"
#include "ns3/inet-socket-address.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/log.h"
#include "ns3/node-container.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/tcp-l4-protocol.h"
#include "ns3/tcp-socket.h"
#include "ns3/test.h"
#include "ns3/traffic-control-helper.h"
#include "ns3/uinteger.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("Ns3TcpLpTest");

namespace
{

/**
 * @ingroup internet-test
 *
 * The flows through the bottleneck
 */
class LpBottleneck
{
  public:
    /**
     * Build the topology, with a flow from each sender
     * @param ccA congestion control of the flow of sender A
     * @param ccB congestion control of the flow of sender B
     */
    LpBottleneck(const std::string& ccA, const std::string& ccB)
    {
        Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1448));
        Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(256 << 10));
        Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(4 << 20));
        // The paper's algorithm collapses with a delayed-ACK receiver; these
        // tests enable the two safeguards against it
        Config::SetDefault("ns3::TcpLp::AckHoldFilter", BooleanValue(true));
        Config::SetDefault("ns3::TcpLp::RestartSmoothing", BooleanValue(true));

        NodeContainer senders;
        senders.Create(2);
        Ptr<Node> router = CreateObject<Node>();
        m_receiver = CreateObject<Node>();
        PointToPointHelper access;
        access.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
        access.SetChannelAttribute("Delay", StringValue("1ms"));
        PointToPointHelper bottleneck;
        bottleneck.SetDeviceAttribute("DataRate", StringValue("10Mbps"));
        bottleneck.SetChannelAttribute("Delay", StringValue("20ms"));
        bottleneck.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("100p"));
        NetDeviceContainer a0 = access.Install(senders.Get(0), router);
        NetDeviceContainer a1 = access.Install(senders.Get(1), router);
        NetDeviceContainer b = bottleneck.Install(router, m_receiver);

        InternetStackHelper internet;
        internet.Install(senders);
        internet.Install(router);
        internet.Install(m_receiver);
        const std::string cc[2] = {ccA, ccB};
        for (uint32_t i = 0; i < 2; ++i)
        {
            senders.Get(i)->GetObject<TcpL4Protocol>()->SetAttribute(
                "SocketType",
                TypeIdValue(TypeId::LookupByName(cc[i])));
        }

        Ipv4AddressHelper address;
        address.SetBase("10.0.0.0", "255.255.255.0");
        address.Assign(a0);
        address.SetBase("10.0.1.0", "255.255.255.0");
        address.Assign(a1);
        address.SetBase("10.0.2.0", "255.255.255.0");
        m_receiverAddress = address.Assign(b).GetAddress(1);
        Ipv4GlobalRoutingHelper::PopulateRoutingTables();
        // The bottleneck queues only in its device
        TrafficControlHelper tch;
        tch.Uninstall(b.Get(0));
        m_senders = senders;
    }

    /**
     * Start a bulk transfer
     * @param i the sender
     * @param start the start time
     * @param stop the stop time
     * @param bytes the bytes to transfer (0: unlimited)
     */
    void Flow(uint32_t i, Time start, Time stop, uint64_t bytes)
    {
        uint16_t port = 5001 + i;
        BulkSendHelper source("ns3::TcpSocketFactory", InetSocketAddress(m_receiverAddress, port));
        source.SetAttribute("MaxBytes", UintegerValue(bytes));
        ApplicationContainer app = source.Install(m_senders.Get(i));
        app.Start(start);
        app.Stop(stop);
        PacketSinkHelper sink("ns3::TcpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer sinkApp = sink.Install(m_receiver);
        sinkApp.Start(Seconds(0.5));
        sinkApp.Get(0)->TraceConnectWithoutContext("Rx", MakeBoundCallback(&Rx, &m_rx[i]));
    }

    /**
     * Count the bytes received
     * @param rx the counter
     * @param p the packet
     */
    static void Rx(uint64_t* rx, Ptr<const Packet> p, const Address&)
    {
        *rx += p->GetSize();
    }

    uint64_t m_rx[2]{0, 0}; //!< Bytes received of each flow

  private:
    NodeContainer m_senders;       //!< Senders
    Ptr<Node> m_receiver;          //!< Receiver
    Ipv4Address m_receiverAddress; //!< Address of the receiver
};

} // namespace

/**
 * @ingroup internet-test
 *
 * A single TCP-LP flow transfers 20 MB through the bottleneck about as
 * quickly as a NewReno flow, although its receiver delays its ACKs.
 */
class Ns3TcpLpSingleFlowTest : public TestCase
{
  public:
    Ns3TcpLpSingleFlowTest()
        : TestCase("A single TCP-LP flow uses the bottleneck as NewReno")
    {
    }

  private:
    void DoRun() override;

    /**
     * Transfer the data
     * @param cc the congestion control
     * @return the duration of the transfer
     */
    Time Transfer(const std::string& cc);

    /**
     * Record the end of the transfer
     * @param net the topology
     */
    void Poll(LpBottleneck* net);

    Time m_end; //!< End of the transfer
};

void
Ns3TcpLpSingleFlowTest::Poll(LpBottleneck* net)
{
    if (net->m_rx[0] >= 20000000)
    {
        m_end = Simulator::Now();
        Simulator::Stop();
        return;
    }
    Simulator::Schedule(MilliSeconds(10), &Ns3TcpLpSingleFlowTest::Poll, this, net);
}

Time
Ns3TcpLpSingleFlowTest::Transfer(const std::string& cc)
{
    LpBottleneck net(cc, "ns3::TcpNewReno");
    net.Flow(0, Seconds(1), Seconds(60), 20000000);
    m_end = Seconds(61);
    Simulator::Schedule(Seconds(1), &Ns3TcpLpSingleFlowTest::Poll, this, &net);
    Simulator::Stop(Seconds(61));
    Simulator::Run();
    Simulator::Destroy();
    Config::Reset();
    return m_end - Seconds(1);
}

void
Ns3TcpLpSingleFlowTest::DoRun()
{
    Time reno = Transfer("ns3::TcpNewReno");
    Time lp = Transfer("ns3::TcpLp");
    NS_LOG_DEBUG("NewReno " << reno.As(Time::S) << " TCP-LP " << lp.As(Time::S));
    NS_TEST_ASSERT_MSG_LT_OR_EQ(lp,
                                reno * 1.25,
                                "The TCP-LP transfer lasts much longer than the NewReno one");
}

/**
 * @ingroup internet-test
 *
 * A TCP-LP flow uses the bottleneck alone, yields it to a NewReno flow
 * while it lasts, and uses it again afterwards.
 */
class Ns3TcpLpYieldTest : public TestCase
{
  public:
    Ns3TcpLpYieldTest()
        : TestCase("TCP-LP yields the bottleneck to NewReno")
    {
    }

  private:
    void DoRun() override;

    /**
     * Record the bytes received so far
     * @param net the topology
     */
    void Sample(LpBottleneck* net);

    std::vector<std::pair<uint64_t, uint64_t>> m_samples; //!< Bytes received, every second
};

void
Ns3TcpLpYieldTest::Sample(LpBottleneck* net)
{
    m_samples.emplace_back(net->m_rx[0], net->m_rx[1]);
}

void
Ns3TcpLpYieldTest::DoRun()
{
    LpBottleneck net("ns3::TcpLp", "ns3::TcpNewReno");
    net.Flow(0, Seconds(1), Seconds(40), 0);
    net.Flow(1, Seconds(10), Seconds(30), 0);
    for (uint32_t t = 0; t <= 40; ++t)
    {
        Simulator::Schedule(Seconds(t) + MilliSeconds(1), &Ns3TcpLpYieldTest::Sample, this, &net);
    }
    Simulator::Stop(Seconds(40.5));
    Simulator::Run();
    Simulator::Destroy();
    Config::Reset();

    // Rate of each flow, in Mbit/s, between two times in s
    auto rate = [this](uint32_t from, uint32_t to) {
        double lp = (m_samples[to].first - m_samples[from].first) * 8e-6 / (to - from);
        double reno = (m_samples[to].second - m_samples[from].second) * 8e-6 / (to - from);
        NS_LOG_DEBUG(from << "-" << to << " s: TCP-LP " << lp << " NewReno " << reno);
        return std::make_pair(lp, reno);
    };
    auto alone = rate(2, 10);
    auto shared = rate(12, 30);
    auto after = rate(32, 40);
    NS_TEST_EXPECT_MSG_GT(alone.first, 9.0, "TCP-LP alone does not use the bottleneck");
    NS_TEST_EXPECT_MSG_LT(shared.first, 0.5, "TCP-LP does not yield to NewReno");
    NS_TEST_EXPECT_MSG_GT(shared.second, 9.0, "NewReno does not get the bottleneck");
    NS_TEST_EXPECT_MSG_GT(after.first, 9.0, "TCP-LP does not use the bottleneck again");
}

/**
 * @ingroup internet-test
 *
 * TCP-LP over a bottleneck
 */
class Ns3TcpLpTestSuite : public TestSuite
{
  public:
    Ns3TcpLpTestSuite()
        : TestSuite("ns3-tcp-lp", Type::SYSTEM)
    {
        AddTestCase(new Ns3TcpLpSingleFlowTest(), TestCase::Duration::QUICK);
        AddTestCase(new Ns3TcpLpYieldTest(), TestCase::Duration::QUICK);
    }
};

/// Static variable for test initialization
static Ns3TcpLpTestSuite g_ns3TcpLpTestSuite;
