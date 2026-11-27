/*
 * Copyright (c) 2004 Francisco J. Ros
 * Copyright (c) 2007 INESC Porto
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Francisco J. Ros  <fjrm@dif.um.es>
 *          Gustavo J. A. M. Carneiro <gjc@inescporto.pt>
 */

#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-header.h"
#include "ns3/olsr-helper.h"
#include "ns3/olsr-repositories.h"
#include "ns3/olsr-routing-protocol.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/simulator.h"
#include "ns3/test.h"

#include <optional>
#include <vector>

/**
 * @ingroup olsr
 * @defgroup olsr-test olsr module tests
 */

using namespace ns3;
using namespace olsr;

/**
 * @ingroup olsr-test
 * @ingroup tests
 *
 * Testcase for MPR computation mechanism
 */
class OlsrMprTestCase : public TestCase
{
  public:
    OlsrMprTestCase();
    ~OlsrMprTestCase() override;
    void DoRun() override;
};

OlsrMprTestCase::OlsrMprTestCase()
    : TestCase("Check OLSR MPR computing mechanism")
{
}

OlsrMprTestCase::~OlsrMprTestCase()
{
}

void
OlsrMprTestCase::DoRun()
{
    Ptr<RoutingProtocol> protocol = CreateObject<RoutingProtocol>();
    protocol->m_mainAddress = Ipv4Address("10.0.0.1");
    OlsrState& state = protocol->m_state;

    /*
     *  1 -- 2
     *  |    |
     *  3 -- 4
     *
     * Node 1 must select only one MPR (2 or 3, doesn't matter)
     */
    NeighborTuple neighbor;
    neighbor.status = NeighborTuple::STATUS_SYM;
    neighbor.willingness = Willingness::DEFAULT;
    neighbor.neighborMainAddr = Ipv4Address("10.0.0.2");
    protocol->m_state.InsertNeighborTuple(neighbor);
    neighbor.neighborMainAddr = Ipv4Address("10.0.0.3");
    protocol->m_state.InsertNeighborTuple(neighbor);
    TwoHopNeighborTuple tuple;
    tuple.expirationTime = Seconds(3600);
    tuple.neighborMainAddr = Ipv4Address("10.0.0.2");
    tuple.twoHopNeighborAddr = Ipv4Address("10.0.0.4");
    protocol->m_state.InsertTwoHopNeighborTuple(tuple);
    tuple.neighborMainAddr = Ipv4Address("10.0.0.3");
    tuple.twoHopNeighborAddr = Ipv4Address("10.0.0.4");
    protocol->m_state.InsertTwoHopNeighborTuple(tuple);

    protocol->MprComputation();
    NS_TEST_EXPECT_MSG_EQ(state.GetMprSet().size(), 1, "An only address must be chosen.");
    /*
     *  1 -- 2 -- 5
     *  |    |
     *  3 -- 4
     *
     * Node 1 must select node 2 as MPR.
     */
    tuple.neighborMainAddr = Ipv4Address("10.0.0.2");
    tuple.twoHopNeighborAddr = Ipv4Address("10.0.0.5");
    protocol->m_state.InsertTwoHopNeighborTuple(tuple);

    protocol->MprComputation();
    MprSet mpr = state.GetMprSet();
    NS_TEST_EXPECT_MSG_EQ(mpr.size(), 1, "An only address must be chosen.");
    NS_TEST_EXPECT_MSG_EQ((mpr.find("10.0.0.2") != mpr.end()),
                          true,
                          "Node 1 must select node 2 as MPR");
    /*
     *  1 -- 2 -- 5
     *  |    |
     *  3 -- 4
     *  |
     *  6
     *
     * Node 1 must select nodes 2 and 3 as MPRs.
     */
    tuple.neighborMainAddr = Ipv4Address("10.0.0.3");
    tuple.twoHopNeighborAddr = Ipv4Address("10.0.0.6");
    protocol->m_state.InsertTwoHopNeighborTuple(tuple);

    protocol->MprComputation();
    mpr = state.GetMprSet();
    NS_TEST_EXPECT_MSG_EQ(mpr.size(), 2, "An only address must be chosen.");
    NS_TEST_EXPECT_MSG_EQ((mpr.find("10.0.0.2") != mpr.end()),
                          true,
                          "Node 1 must select node 2 as MPR");
    NS_TEST_EXPECT_MSG_EQ((mpr.find("10.0.0.3") != mpr.end()),
                          true,
                          "Node 1 must select node 3 as MPR");
    /*
     *  7 (Willingness::ALWAYS)
     *  |
     *  1 -- 2 -- 5
     *  |    |
     *  3 -- 4
     *  |
     *  6
     *
     * Node 1 must select nodes 2, 3 and 7 (since it is Willingness::ALWAYS) as MPRs.
     */
    neighbor.willingness = olsr::Willingness::ALWAYS;
    neighbor.neighborMainAddr = Ipv4Address("10.0.0.7");
    protocol->m_state.InsertNeighborTuple(neighbor);

    protocol->MprComputation();
    mpr = state.GetMprSet();
    NS_TEST_EXPECT_MSG_EQ(mpr.size(), 3, "An only address must be chosen.");
    NS_TEST_EXPECT_MSG_EQ((mpr.find("10.0.0.7") != mpr.end()),
                          true,
                          "Node 1 must select node 7 as MPR");
    /*
     *                        7 <- Willingness::ALWAYS
     *                        |
     *              9 -- 8 -- 1 -- 2 -- 5
     *                        |    |
     *                   ^    3 -- 4
     *                   |    |
     *   Willingness::NEVER   6
     *
     * Node 1 must select nodes 2, 3 and 7 (since it is Willingness::ALWAYS) as MPRs.
     * Node 1 must NOT select node 8 as MPR since it is Willingness::NEVER
     */
    neighbor.willingness = Willingness::NEVER;
    neighbor.neighborMainAddr = Ipv4Address("10.0.0.8");
    protocol->m_state.InsertNeighborTuple(neighbor);
    tuple.neighborMainAddr = Ipv4Address("10.0.0.8");
    tuple.twoHopNeighborAddr = Ipv4Address("10.0.0.9");
    protocol->m_state.InsertTwoHopNeighborTuple(tuple);

    protocol->MprComputation();
    mpr = state.GetMprSet();
    NS_TEST_EXPECT_MSG_EQ(mpr.size(), 3, "An only address must be chosen.");
    NS_TEST_EXPECT_MSG_EQ((mpr.find("10.0.0.9") == mpr.end()),
                          true,
                          "Node 1 must NOT select node 8 as MPR");
}

/**
 * @ingroup olsr-test
 * @ingroup tests
 *
 * Check that the route to a network is removed after the gateway announcing it
 * through HNA messages withdraws it, while the network is otherwise static.
 */
class OlsrHnaWithdrawalTestCase : public TestCase
{
  public:
    OlsrHnaWithdrawalTestCase();
    void DoRun() override;

  private:
    /**
     * Run a simulation where the gateway withdraws one of the networks.
     *
     * @param withdrawn The network the gateway withdraws.
     */
    void RunWithdrawal(const Ipv4Address& withdrawn);

    /**
     * Check which of the networks have a route.
     *
     * @param withdrawn The network the gateway has withdrawn, if any.
     */
    void CheckRoutes(std::optional<Ipv4Address> withdrawn);

    /// Networks announced by the gateway
    const std::vector<Ipv4Address> m_networks{"172.16.1.0", "172.16.2.0"};
    const Ipv4Mask m_mask{"255.255.255.0"};   ///< Mask of the networks announced by the gateway
    const Time m_withdrawalTime{Seconds(40)}; ///< Time the gateway withdraws the network
    const Time m_checkBefore{Seconds(39)};    ///< Time of the check before the withdrawal
    const Time m_checkAfter{Seconds(80)};     ///< Time of the check after the withdrawal
    Ptr<RoutingProtocol> m_protocol;          ///< OLSR of the node the routes are checked on
};

OlsrHnaWithdrawalTestCase::OlsrHnaWithdrawalTestCase()
    : TestCase("Check the removal of the route to a withdrawn HNA network")
{
}

void
OlsrHnaWithdrawalTestCase::CheckRoutes(std::optional<Ipv4Address> withdrawn)
{
    for (const auto& network : m_networks)
    {
        Ipv4Header header;
        header.SetDestination(network);
        Socket::SocketErrno error;
        Ptr<Ipv4Route> route = m_protocol->RouteOutput(nullptr, header, nullptr, error);
        NS_TEST_EXPECT_MSG_EQ((route != nullptr),
                              (network != withdrawn),
                              "Unexpected route to " << network << " at " << Simulator::Now());
    }
}

void
OlsrHnaWithdrawalTestCase::DoRun()
{
    for (const auto& network : m_networks)
    {
        RunWithdrawal(network);
    }
}

void
OlsrHnaWithdrawalTestCase::RunWithdrawal(const Ipv4Address& withdrawn)
{
    NodeContainer nodes(3);
    SimpleNetDeviceHelper simple;
    NetDeviceContainer devices = simple.Install(nodes);

    InternetStackHelper internet;
    OlsrHelper olsrHelper;
    internet.SetRoutingHelper(olsrHelper);
    internet.Install(nodes);
    olsrHelper.AssignStreams(nodes, 1);

    Ipv4AddressHelper address;
    address.SetBase("10.0.0.0", "255.255.255.0");
    address.Assign(devices);

    m_protocol = nodes.Get(0)->GetObject<RoutingProtocol>();
    Ptr<RoutingProtocol> gateway = nodes.Get(2)->GetObject<RoutingProtocol>();
    NS_TEST_ASSERT_MSG_NE(m_protocol, nullptr, "OLSR is not installed");
    NS_TEST_ASSERT_MSG_NE(gateway, nullptr, "OLSR is not installed on the gateway");

    for (const auto& network : m_networks)
    {
        gateway->AddHostNetworkAssociation(network, m_mask);
    }
    Simulator::Schedule(m_withdrawalTime,
                        &RoutingProtocol::RemoveHostNetworkAssociation,
                        gateway,
                        withdrawn,
                        m_mask);
    Simulator::Schedule(m_checkBefore,
                        &OlsrHnaWithdrawalTestCase::CheckRoutes,
                        this,
                        std::optional<Ipv4Address>());
    Simulator::Schedule(m_checkAfter, &OlsrHnaWithdrawalTestCase::CheckRoutes, this, withdrawn);

    Simulator::Stop(m_checkAfter);
    Simulator::Run();
    m_protocol = nullptr;
    Simulator::Destroy();
}

/**
 * @ingroup olsr-test
 * @ingroup tests
 *
 * OLSR protocol test suite
 */
class OlsrProtocolTestSuite : public TestSuite
{
  public:
    OlsrProtocolTestSuite();
};

OlsrProtocolTestSuite::OlsrProtocolTestSuite()
    : TestSuite("routing-olsr", Type::UNIT)
{
    AddTestCase(new OlsrMprTestCase(), TestCase::Duration::QUICK);
    AddTestCase(new OlsrHnaWithdrawalTestCase(), TestCase::Duration::QUICK);
}

static OlsrProtocolTestSuite g_olsrProtocolTestSuite; //!< Static variable for test initialization
