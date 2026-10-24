/*
 * Copyright (c) 2010 University of Washington (trace writing)
 * Copyright (c) 2018-20 NITK Surathkal (topology setup)
 * Copyright (c) 2024 Tom Henderson (test definition)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

// Test suite based on tcp-bbr-example.cc modified topology setup:
//
//           1 Gbps            50/100Mbps         1 Gbps
//  Sender -------------- R1 -------------- R2 -------------- Receiver
//            5us               baseRtt/2          5us
//
// This is a test suite written initially to test the addition of
// TCP friendliness but can also be extended for other Cubic tests.  By
// changing some compile-time constants below, it can be used to also produce
// gnuplots and pcaps that demonstrate the behavior.  See the WRITE_PCAP and
// WRITE_GNUPLOT booleans below to enable this output.
//
// There are four cases configured presently.
// 1. A Cubic TCP on a 5 ms RTT bottleneck, with no fast convergence nor
//    TCP friendliness configured
// 2. A Cubic TCP on a 5 ms RTT bottleneck, with fast convergence but no
//    TCP friendliness configured.  This exhibits a different cwnd plot from 1.
// 3. A Cubic TCP on a 20 ms RTT bottleneck, with fast convergence but no
//    TCP friendliness configured.  There is a step change increase in
//    bottleneck link capacity but the TCP is slow to exploit it.
// 4. A Cubic TCP on a 20 ms RTT bottleneck, with fast convergence and
//    TCP friendliness configured.  There is a step change increase in
//    bottleneck link capacity and the TCP responds more quickly than in 3.

#include "ns3/abort.h"
#include "ns3/boolean.h"
#include "ns3/bulk-send-helper.h"
#include "ns3/config.h"
#include "ns3/data-rate.h"
#include "ns3/enum.h"
#include "ns3/error-model.h"
#include "ns3/gnuplot.h"
#include "ns3/inet-socket-address.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/log.h"
#include "ns3/node-container.h"
#include "ns3/node-list.h"
#include "ns3/output-stream-wrapper.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/pcap-file.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/point-to-point-net-device.h"
#include "ns3/pointer.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/tcp-cubic.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-l4-protocol.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/test.h"
#include "ns3/trace-helper.h"
#include "ns3/traffic-control-helper.h"
#include "ns3/uinteger.h"

#include <iomanip>
#include <optional>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("Ns3TcpCubicTest");

static constexpr bool WRITE_PCAP = false;    //!< Set to true to write out pcap.
static constexpr bool WRITE_GNUPLOT = false; //!< Set to true to write out gnuplot.

/**
 * @ingroup internet-test
 *
 * Add sample trace values to data structures
 * @param gnuplotTimeSeries Gnuplot data structure
 * @param timeSeries time series of cwnd changes
 * @param oldval old value of cwnd
 * @param newval new value of cwnd
 */
void
CubicCwndTracer(Gnuplot2dDataset* gnuplotTimeSeries,
                std::map<Time, double>* timeSeries,
                uint32_t oldval,
                uint32_t newval)
{
    // We store data in two structures because Gnuplot2DDataset data is not exportable
    NS_LOG_DEBUG("cwnd " << newval / 1448.0);
    gnuplotTimeSeries->Add(Simulator::Now().GetSeconds(), newval / 1448.0);
    timeSeries->insert_or_assign(Simulator::Now(), newval / 1448.0);
}

/**
 * @ingroup internet-test
 * Test Cubic response
 */
class Ns3TcpCubicTestCase : public TestCase
{
  public:
    /**
     * Constructor.
     *
     * @param testCase testcase name
     * @param prefix filename prefix if writing output files
     * @param fastConvergence whether to enable fast convergence
     * @param tcpFriendliness whether to enable TCP friendliness
     * @param baseRtt base RTT to use for test case
     * @param capacityIncrease whether to trigger a sudden capacity increase
     * @param clockDrivenIsn whether the initial sequence numbers are clock
     *        driven, with the one of the sender in the upper half of the
     *        sequence space
     */
    Ns3TcpCubicTestCase(std::string testCase,
                        std::string prefix,
                        bool fastConvergence,
                        bool tcpFriendliness,
                        Time baseRtt,
                        bool capacityIncrease,
                        bool clockDrivenIsn = false);
    ~Ns3TcpCubicTestCase() override;

  private:
    void DoRun() override;

    /**
     * Connect TCP cwnd trace after socket is instantiated
     *
     * @param nodeId node ID to connect to
     * @param socketId socket ID to connect to
     */
    void ConnectCwndTrace(uint32_t nodeId, uint32_t socketId);

    /**
     * Record the sequence number of the first data segment sent
     *
     * @param packet the packet sent
     * @param header the TCP header
     * @param socket the sending socket
     */
    void FirstSegmentSent(Ptr<const Packet> packet,
                          const TcpHeader& header,
                          Ptr<const TcpSocketBase> socket);

    /**
     * Increases the device bandwidth to 100 Mbps
     * @param device device to modify
     */
    void IncreaseBandwidth(Ptr<PointToPointNetDevice> device);

    /**
     * Check that time series values within a time range are within a value range.
     * @param start start of time range
     * @param end end of time range
     * @param lowerBound lower bound of acceptable values
     * @param upperBound upper bound of acceptable values
     * @return true if values are within range
     */
    bool CheckValues(Time start, Time end, double lowerBound, double upperBound);

    bool m_writeResults{WRITE_PCAP};     //!< Whether to write pcaps
    bool m_writeGnuplot{WRITE_GNUPLOT};  //!< Whether to write gnuplot files
    Gnuplot2dDataset m_cwndTimeSeries;   //!< cwnd time series
    std::map<Time, double> m_timeSeries; //!< time series to check
    std::string m_prefix;                //!< filename prefix if writing files
    bool m_fastConvergence;              //!< whether to enable fast convergence
    bool m_tcpFriendliness;              //!< whether to enable TCP friendliness
    Time m_baseRtt;                      //!< the base RTT to use
    bool m_capacityIncrease;             //!< whether to trigger a capacity increase
    bool m_clockDrivenIsn;               //!< whether the initial sequence numbers are clock driven
    std::optional<SequenceNumber32> m_firstSeq; //!< sequence number of the first data segment
};

Ns3TcpCubicTestCase::Ns3TcpCubicTestCase(std::string testCase,
                                         std::string prefix,
                                         bool fastConvergence,
                                         bool tcpFriendliness,
                                         Time baseRtt,
                                         bool capacityIncrease,
                                         bool clockDrivenIsn)
    : TestCase(testCase),
      m_prefix(prefix),
      m_fastConvergence(fastConvergence),
      m_tcpFriendliness(tcpFriendliness),
      m_baseRtt(baseRtt),
      m_capacityIncrease(capacityIncrease),
      m_clockDrivenIsn(clockDrivenIsn)
{
}

Ns3TcpCubicTestCase::~Ns3TcpCubicTestCase()
{
}

void
Ns3TcpCubicTestCase::IncreaseBandwidth(Ptr<PointToPointNetDevice> device)
{
    device->SetDataRate(DataRate("100Mbps"));
}

void
Ns3TcpCubicTestCase::ConnectCwndTrace(uint32_t nodeId, uint32_t socketId)
{
    Config::ConnectWithoutContext(
        "/NodeList/" + std::to_string(nodeId) + "/$ns3::TcpL4Protocol/SocketList/" +
            std::to_string(socketId) + "/CongestionWindow",
        MakeBoundCallback(&CubicCwndTracer, &m_cwndTimeSeries, &m_timeSeries));
    Config::ConnectWithoutContext("/NodeList/" + std::to_string(nodeId) +
                                      "/$ns3::TcpL4Protocol/SocketList/" +
                                      std::to_string(socketId) + "/Tx",
                                  MakeCallback(&Ns3TcpCubicTestCase::FirstSegmentSent, this));
}

void
Ns3TcpCubicTestCase::FirstSegmentSent(Ptr<const Packet> packet,
                                      const TcpHeader& header,
                                      Ptr<const TcpSocketBase> socket)
{
    if (packet->GetSize() > 0 && !m_firstSeq)
    {
        m_firstSeq = header.GetSequenceNumber();
    }
}

bool
Ns3TcpCubicTestCase::CheckValues(Time start, Time end, double lowerBound, double upperBound)
{
    auto itStart = m_timeSeries.lower_bound(start);
    auto itEnd = m_timeSeries.upper_bound(end);
    for (auto it = itStart; it != itEnd; it++)
    {
        if (it->second < lowerBound || it->second > upperBound)
        {
            NS_LOG_DEBUG("Value " << it->second << " at time " << it->first.GetSeconds()
                                  << " outside range (" << lowerBound << "," << upperBound << ")");
            return false;
        }
    }
    return true;
}

void
Ns3TcpCubicTestCase::DoRun()
{
    NS_LOG_DEBUG("Running " << m_prefix);
    // The maximum bandwidth delay product is 20 ms (base RTT) times 100 Mbps
    // which works out to 250KB.  Setting the socket buffer sizes to four times
    // that value (1MB) should be more than enough.
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(1000000));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(1000000));
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(10));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(2));
    // 1460 byte MSS minus the 12 byte timestamp option: 1448 byte payload, as
    // Linux with a 1500 byte MTU (cwnd counted in 1448 byte units)
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1460));
    Config::SetDefault("ns3::TcpSocketState::EnablePacing", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocketState::PaceInitialWindow", BooleanValue(true));
    Config::SetDefault("ns3::TcpCubic::FastConvergence", BooleanValue(m_fastConvergence));
    Config::SetDefault("ns3::TcpCubic::TcpFriendliness", BooleanValue(m_tcpFriendliness));

    Time stopTime = Seconds(20);

    NodeContainer sender;
    NodeContainer receiver;
    NodeContainer routers;
    sender.Create(1);
    receiver.Create(1);
    routers.Create(2);

    // Create the point-to-point link helpers
    PointToPointHelper bottleneckLink;
    // With the DynamicQueueLimits setting below, the following statement
    // will minimize the buffering in the device and force most buffering
    // into the AQM
    bottleneckLink.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("2p"));
    if (m_capacityIncrease)
    {
        // Start at a lower capacity, and increase later
        bottleneckLink.SetDeviceAttribute("DataRate", StringValue("50Mbps"));
    }
    else
    {
        bottleneckLink.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    }
    bottleneckLink.SetChannelAttribute("Delay", TimeValue(m_baseRtt / 2));

    PointToPointHelper edgeLink;
    edgeLink.SetDeviceAttribute("DataRate", StringValue("1000Mbps"));
    edgeLink.SetChannelAttribute("Delay", StringValue("5us"));

    // Create NetDevice containers
    NetDeviceContainer senderEdge = edgeLink.Install(sender.Get(0), routers.Get(0));
    NetDeviceContainer r1r2 = bottleneckLink.Install(routers.Get(0), routers.Get(1));
    NetDeviceContainer receiverEdge = edgeLink.Install(routers.Get(1), receiver.Get(0));

    // Install Stack
    InternetStackHelper internet;
    internet.Install(sender);
    internet.Install(receiver);
    internet.Install(routers);

    if (m_clockDrivenIsn)
    {
        // Seed 1, run 1 and stream 0 put the sender ISN in the upper half of
        // the sequence space, where wrap aware comparisons with a mark left at
        // 0 (e.g., the end of the HyStart round) never hold
        uint32_t seed = RngSeedManager::GetSeed();
        uint64_t run = RngSeedManager::GetRun();
        RngSeedManager::SetSeed(1);
        RngSeedManager::SetRun(1);
        for (auto it = NodeList::Begin(); it != NodeList::End(); ++it)
        {
            (*it)->GetObject<TcpL4Protocol>()->SetAttribute("ClockDrivenIsn", BooleanValue(true));
        }
        sender.Get(0)->GetObject<TcpL4Protocol>()->AssignStreams(0);
        RngSeedManager::SetSeed(seed);
        RngSeedManager::SetRun(run);
    }

    // Configure the root queue discipline
    TrafficControlHelper tch;
    tch.SetRootQueueDisc("ns3::FqCoDelQueueDisc");
    tch.SetQueueLimits("ns3::DynamicQueueLimits");

    tch.Install(senderEdge);
    tch.Install(receiverEdge);

    // Assign IP addresses
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.0.0.0", "255.255.255.0");

    Ipv4InterfaceContainer i1i2 = ipv4.Assign(r1r2);
    ipv4.NewNetwork();
    Ipv4InterfaceContainer is1 = ipv4.Assign(senderEdge);
    ipv4.NewNetwork();
    Ipv4InterfaceContainer ir1 = ipv4.Assign(receiverEdge);

    // Populate routing tables
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // Select sender side port
    uint16_t port = 50001;

    // Install application on the sender
    BulkSendHelper source("ns3::TcpSocketFactory", InetSocketAddress(ir1.GetAddress(1), port));
    source.SetAttribute("MaxBytes", UintegerValue(0));
    ApplicationContainer sourceApps = source.Install(sender.Get(0));
    sourceApps.Start(Seconds(0.1));
    // Hook trace source after application starts
    Simulator::Schedule(Seconds(0.1) + MilliSeconds(1),
                        &Ns3TcpCubicTestCase::ConnectCwndTrace,
                        this,
                        0,
                        0);
    sourceApps.Stop(stopTime);

    // Install application on the receiver
    PacketSinkHelper sink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApps = sink.Install(receiver.Get(0));
    sinkApps.Start(Seconds(0));
    sinkApps.Stop(stopTime);

    if (m_capacityIncrease)
    {
        // Double the capacity of the bottleneck link at 10 seconds
        Ptr<PointToPointNetDevice> device = r1r2.Get(0)->GetObject<PointToPointNetDevice>();
        Simulator::Schedule(Seconds(10), &Ns3TcpCubicTestCase::IncreaseBandwidth, this, device);
    }

    std::ostringstream oss;
    oss << m_prefix;
    if (m_writeResults)
    {
        edgeLink.EnablePcap(oss.str(), senderEdge.Get(0));
        edgeLink.EnablePcap(oss.str(), receiverEdge.Get(0));
        edgeLink.EnableAscii(oss.str(), senderEdge.Get(0));
        edgeLink.EnableAscii(oss.str(), receiverEdge.Get(0));
    }
    Simulator::Stop(Seconds(30));
    Simulator::Run();

    if (m_writeGnuplot)
    {
        std::ofstream outfile(oss.str() + ".plt");

        std::string capacityIncreaseString;
        if (m_capacityIncrease)
        {
            capacityIncreaseString = " capacity increase = 1";
        }
        Gnuplot gnuplot = Gnuplot(
            oss.str() + ".eps",
            "Cubic concave/convex response (" + std::to_string(m_baseRtt.GetMilliSeconds()) +
                " ms RTT, 1448 byte segs, 100 Mbps bottleneck)\\nFast convergence = " +
                std::to_string(m_fastConvergence) +
                " friendliness = " + std::to_string(m_tcpFriendliness) + capacityIncreaseString);
        gnuplot.SetTerminal("post eps color enhanced");
        gnuplot.SetLegend("Time (seconds)", "Cwnd (segments)");
        m_cwndTimeSeries.SetTitle("");
        gnuplot.AddDataset(m_cwndTimeSeries);
        gnuplot.GenerateOutput(outfile);
    }

    if (m_clockDrivenIsn)
    {
        NS_TEST_ASSERT_MSG_EQ(m_firstSeq.has_value(), true, "No data segment was sent");
        NS_TEST_ASSERT_MSG_GT_OR_EQ((*m_firstSeq - 1).GetValue(),
                                    1U << 31,
                                    "The initial sequence number of the sender is not in the "
                                    "upper half of the sequence space: pick another stream");
    }

    // Check that cwnd values are within tolerance of expected values
    // The expected values were determined by inspecting the plots generated
    if (m_prefix == "ns3-tcp-cubic-no-heuristic")
    {
        // Check overall min and max
        NS_TEST_ASSERT_MSG_EQ(CheckValues(Seconds(1), Seconds(19), 60, 110),
                              true,
                              "cwnd outside range");
        // Time just before a reduction does not have much variation
        NS_TEST_ASSERT_MSG_EQ(CheckValues(Seconds(8.6), Seconds(9.1), 84, 89),
                              true,
                              "cwnd outside range");
    }
    else if (m_prefix == "ns3-tcp-cubic-fast-conv")
    {
        // Check overall min and max
        NS_TEST_ASSERT_MSG_EQ(CheckValues(Seconds(1), Seconds(19), 60, 110),
                              true,
                              "cwnd outside range");
        // Flat plateau around the maximum reduced by fast convergence
        NS_TEST_ASSERT_MSG_EQ(CheckValues(Seconds(4), Seconds(7), 73, 79),
                              true,
                              "cwnd outside range");
    }
    else if (m_prefix == "ns3-tcp-cubic-no-friendly")
    {
        // Between time 11 and 15, cwnd should be fairly constant
        // because without TCP friendliness, Cubic does not respond quickly
        NS_TEST_ASSERT_MSG_EQ(CheckValues(Seconds(11), Seconds(15), 85, 95),
                              true,
                              "cwnd outside range");
        // After time 17.5, cwnd should have grown much higher
        NS_TEST_ASSERT_MSG_EQ(CheckValues(Seconds(17.5), Seconds(18.5), 114, 155),
                              true,
                              "cwnd outside range");
    }
    else if (m_prefix == "ns3-tcp-cubic-friendly")
    {
        // In contrast to previous case, cwnd should grow above 150 much sooner
        NS_TEST_ASSERT_MSG_EQ(CheckValues(Seconds(13), Seconds(15), 150, 220),
                              true,
                              "cwnd outside range");
    }

    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 *
 * @brief Check the thresholds of the HyStart exit from slow start.
 *
 * With a minimum RTT of 100 ms, as in Linux (hystart_update()):
 * - the delay detection exits the slow start once the minimum RTT of a round
 *   exceeds the minimum RTT by one eighth of it (12.5 ms), clamped between
 *   HyStartDelayMin and HyStartDelayMax (16 ms);
 * - the ACK train detection exits the slow start once closely spaced ACKs
 *   have arrived for half the minimum RTT (50 ms) since the start of the
 *   round, i.e., the forward one way delay, when the flow is not paced.
 */
class Ns3TcpCubicHystartTestCase : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param mode The HyStart detection mechanism to check
     * @param desc Test description.
     */
    Ns3TcpCubicHystartTestCase(TcpCubic::HybridSSDetectionMode mode, const std::string& desc)
        : TestCase(desc),
          m_mode(mode)
    {
    }

  private:
    void DoRun() override;

    /**
     * @brief Process an ACK of one segment, as the socket does.
     * @param rtt The RTT sample of the ACK
     */
    void Ack(Time rtt);

    TcpCubic::HybridSSDetectionMode m_mode; //!< HyStart detection mechanism
    Ptr<TcpSocketState> m_state;            //!< TCP socket state
    Ptr<TcpCubic> m_cubic;                  //!< The congestion control
};

void
Ns3TcpCubicHystartTestCase::Ack(Time rtt)
{
    m_cubic->PktsAcked(m_state, 1, rtt);
    m_cubic->IncreaseWindow(m_state, 1);
}

void
Ns3TcpCubicHystartTestCase::DoRun()
{
    const uint32_t segmentSize = 1000;
    m_state = CreateObject<TcpSocketState>();
    m_state->m_segmentSize = segmentSize;
    m_state->m_cWnd = 20 * segmentSize;
    m_state->m_ssThresh = UINT32_MAX;
    m_state->m_isCwndLimited = true;
    m_state->m_lastAckedSeq = SequenceNumber32(1);
    m_state->m_highTxMark = SequenceNumber32(20 * segmentSize + 1);

    m_cubic = CreateObject<TcpCubic>();
    m_cubic->SetAttribute("HyStartDetect", EnumValue(m_mode));
    m_cubic->Init(m_state);

    // The first sample sets the minimum RTT
    Ack(MilliSeconds(100));

    if (m_mode == TcpCubic::HybridSSDetectionMode::DELAY)
    {
        // A new round, whose RTT samples are 15 ms above the minimum
        m_state->m_lastAckedSeq = SequenceNumber32(20 * segmentSize + 2);
        m_state->m_highTxMark = SequenceNumber32(60 * segmentSize + 1);
        for (uint32_t i = 0; i < 12; i++)
        {
            Ack(MilliSeconds(115));
        }
    }
    else
    {
        // ACKs 1 ms apart for 60 ms, within the round
        for (uint32_t i = 1; i <= 60; i++)
        {
            Simulator::Schedule(MilliSeconds(i),
                                &Ns3TcpCubicHystartTestCase::Ack,
                                this,
                                MilliSeconds(100));
        }
        Simulator::Run();
    }

    NS_TEST_ASSERT_MSG_NE(m_state->m_ssThresh.Get(),
                          UINT32_MAX,
                          "HyStart did not exit the slow start");
    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 *
 * @brief Check that the ACKs counted towards the next window increase are
 * discarded when the window is reduced.
 *
 * Linux clears tp->snd_cwnd_cnt in tcp_init_cwnd_reduction() and
 * tcp_enter_loss(): ACKs counted at the pre-loss window must not hasten the
 * first increase of the reduced window.
 */
class Ns3TcpCubicCwndCntResetTestCase : public TestCase
{
  public:
    Ns3TcpCubicCwndCntResetTestCase()
        : TestCase("ACK count discarded on a window reduction")
    {
    }

  private:
    void DoRun() override;
};

void
Ns3TcpCubicCwndCntResetTestCase::DoRun()
{
    const uint32_t segmentSize = 1000;
    Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
    state->m_segmentSize = segmentSize;
    state->m_cWnd = 100 * segmentSize;
    state->m_ssThresh = 50 * segmentSize;
    state->m_isCwndLimited = true;

    Ptr<TcpCubic> cubic = CreateObject<TcpCubic>();
    cubic->Init(state);

    // Minimum delay 4.2 s: one segment every 3 ACKs; count two ACKs
    cubic->PktsAcked(state, 1, Seconds(4.2));
    cubic->IncreaseWindow(state, 1);
    cubic->IncreaseWindow(state, 1);
    NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(), 100 * segmentSize, "Unexpected window increase");

    // Loss: cwnd 70 segments; the new epoch (origin 100 segments, 4.2 s
    // away) grows one segment every 2 ACKs
    cubic->CongestionStateSet(state, TcpSocketState::CA_RECOVERY);
    state->m_ssThresh = cubic->GetSsThresh(state, 100 * segmentSize);
    NS_TEST_ASSERT_MSG_EQ(state->m_ssThresh.Get(), 70 * segmentSize, "Unexpected ssThresh");
    state->m_cWnd = state->m_ssThresh;
    cubic->CongestionStateSet(state, TcpSocketState::CA_OPEN);

    cubic->IncreaseWindow(state, 1);
    NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(),
                          70 * segmentSize,
                          "The ACKs counted before the loss hastened the increase");
    cubic->IncreaseWindow(state, 1);
    NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(), 71 * segmentSize, "Unexpected window increase");
}

/**
 * @ingroup internet-test
 *
 * @brief Check the CntUpdateInterval attribute of TcpCubic.
 *
 * cwnd 100 segments, no previous maximum: the first ACK finds the target at
 * cwnd and the count is clamped to 20 ACKs per increase. 3 s later the target
 * is 110 segments and the count becomes 10, unless kept because cwnd did not
 * change within CntUpdateInterval (Linux bictcp_update()).
 */
class Ns3TcpCubicCntUpdateIntervalTestCase : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param interval The CntUpdateInterval attribute.
     * @param expectedCwnd The expected window, in segments.
     * @param desc Test description.
     */
    Ns3TcpCubicCntUpdateIntervalTestCase(Time interval,
                                         uint32_t expectedCwnd,
                                         const std::string& desc)
        : TestCase(desc),
          m_interval(interval),
          m_expectedCwnd(expectedCwnd)
    {
    }

  private:
    void DoRun() override;

    Time m_interval;         //!< The CntUpdateInterval attribute
    uint32_t m_expectedCwnd; //!< Expected window, in segments
};

void
Ns3TcpCubicCntUpdateIntervalTestCase::DoRun()
{
    const uint32_t segmentSize = 1000;
    Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
    state->m_segmentSize = segmentSize;
    state->m_cWnd = 100 * segmentSize;
    state->m_ssThresh = 50 * segmentSize;
    state->m_isCwndLimited = true;

    Ptr<TcpCubic> cubic = CreateObject<TcpCubic>();
    cubic->SetAttribute("CntUpdateInterval", TimeValue(m_interval));
    cubic->Init(state);
    cubic->PktsAcked(state, 1, MilliSeconds(1));
    cubic->IncreaseWindow(state, 1);

    Simulator::Schedule(Seconds(3), [cubic, state]() {
        for (uint32_t i = 0; i < 10; ++i)
        {
            cubic->IncreaseWindow(state, 1);
        }
    });
    Simulator::Run();
    Simulator::Destroy();

    NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(), m_expectedCwnd * segmentSize, "Unexpected window");
}

/**
 * @ingroup internet-test
 *
 * @brief Check the Linux arithmetic of TcpCubic (LinuxHz attribute).
 *
 * A loss at 1427 segments gives ssThresh 999 with beta = 717/1024 (998 with
 * 0.7). From 999 segments with a 95 ms minimum delay (10 jiffies at HZ = 100),
 * the Linux fixed point cube root and cubic function give a target of 1013
 * segments, one segment per 71 ACKs; the published function (K = 10.23 s)
 * gives one per 83.
 */
class Ns3TcpCubicLinuxHzTestCase : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param linuxHz The LinuxHz attribute.
     * @param expectedSsThresh The expected ssThresh, in segments.
     * @param expectedCwnd The expected window after 71 ACKs, in segments.
     * @param desc Test description.
     */
    Ns3TcpCubicLinuxHzTestCase(uint32_t linuxHz,
                               uint32_t expectedSsThresh,
                               uint32_t expectedCwnd,
                               const std::string& desc)
        : TestCase(desc),
          m_linuxHz(linuxHz),
          m_expectedSsThresh(expectedSsThresh),
          m_expectedCwnd(expectedCwnd)
    {
    }

  private:
    void DoRun() override;

    uint32_t m_linuxHz;          //!< The LinuxHz attribute
    uint32_t m_expectedSsThresh; //!< Expected ssThresh, in segments
    uint32_t m_expectedCwnd;     //!< Expected window, in segments
};

void
Ns3TcpCubicLinuxHzTestCase::DoRun()
{
    const uint32_t segmentSize = 1000;
    Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
    state->m_segmentSize = segmentSize;
    state->m_cWnd = 1427 * segmentSize;
    state->m_ssThresh = 1000 * segmentSize;
    state->m_isCwndLimited = true;

    Ptr<TcpCubic> cubic = CreateObject<TcpCubic>();
    cubic->SetAttribute("LinuxHz", UintegerValue(m_linuxHz));
    cubic->SetAttribute("HyStart", BooleanValue(false));
    cubic->Init(state);

    // Start away from time 0, as Linux jiffies do
    Simulator::Schedule(Seconds(1), [cubic, state]() {
        cubic->PktsAcked(state, 1, MilliSeconds(95));
        state->m_ssThresh = cubic->GetSsThresh(state, 1427 * segmentSize);
        state->m_cWnd = state->m_ssThresh;
        for (uint32_t i = 0; i < 71; ++i)
        {
            cubic->IncreaseWindow(state, 1);
        }
    });
    Simulator::Run();
    Simulator::Destroy();

    NS_TEST_ASSERT_MSG_EQ(state->m_ssThresh.Get(),
                          m_expectedSsThresh * segmentSize,
                          "Unexpected ssThresh");
    NS_TEST_ASSERT_MSG_EQ(state->m_cWnd.Get(), m_expectedCwnd * segmentSize, "Unexpected window");
}

/**
 * @ingroup internet-test
 * TestSuite for module tcp-cubic
 */
class Ns3TcpCubicTestSuite : public TestSuite
{
  public:
    Ns3TcpCubicTestSuite();
};

Ns3TcpCubicTestSuite::Ns3TcpCubicTestSuite()
    : TestSuite("ns3-tcp-cubic", Type::UNIT)
{
    // Test Cubic with no fast convergence or TCP friendliness enabled
    // This results in a cwnd plot that has only the concave portion of
    // returning cwnd to Wmax
    AddTestCase(new Ns3TcpCubicTestCase("no heuristic",
                                        "ns3-tcp-cubic-no-heuristic",
                                        false,
                                        false,
                                        MilliSeconds(5),
                                        false),
                TestCase::Duration::QUICK);

    // Clock driven ISN: HyStart must exit slow start as with an ISN of 0
    AddTestCase(new Ns3TcpCubicTestCase("no heuristic, clock driven ISN",
                                        "ns3-tcp-cubic-no-heuristic",
                                        false,
                                        false,
                                        MilliSeconds(5),
                                        false,
                                        true),
                TestCase::Duration::QUICK);

    // Test Cubic with fast convergence but no TCP friendliness enabled
    // This results in a cwnd plot that has concave and convex regions, as
    // well as convex-only regions; also observed on Linux testbeds
    AddTestCase(new Ns3TcpCubicTestCase("fast convergence on",
                                        "ns3-tcp-cubic-fast-conv",
                                        true,
                                        false,
                                        MilliSeconds(5),
                                        false),
                TestCase::Duration::QUICK);

    // Test Cubic with fast convergence but no TCP friendliness enabled
    // with a higher RTT (20ms) and a step change in capacity at time 10s.
    // This results in a sluggish response by TCP Cubic to use the new capacity.
    AddTestCase(new Ns3TcpCubicTestCase("fast convergence on, Reno friendliness off",
                                        "ns3-tcp-cubic-no-friendly",
                                        true,
                                        false,
                                        MilliSeconds(20),
                                        true),
                TestCase::Duration::QUICK);

    // Test Cubic with fast convergence but with TCP friendliness enabled
    // with a higher RTT (20ms) and a step change in capacity at time 10s.
    // This results in a faster response by TCP Cubic to use the new capacity.
    AddTestCase(new Ns3TcpCubicTestCase("fast convergence on, Reno friendliness on",
                                        "ns3-tcp-cubic-friendly",
                                        true,
                                        true,
                                        MilliSeconds(20),
                                        true),
                TestCase::Duration::QUICK);

    AddTestCase(new Ns3TcpCubicHystartTestCase(TcpCubic::HybridSSDetectionMode::DELAY,
                                               "HyStart delay threshold"),
                TestCase::Duration::QUICK);
    AddTestCase(new Ns3TcpCubicHystartTestCase(TcpCubic::HybridSSDetectionMode::PACKET_TRAIN,
                                               "HyStart ACK train threshold"),
                TestCase::Duration::QUICK);
    AddTestCase(new Ns3TcpCubicCwndCntResetTestCase(), TestCase::Duration::QUICK);
    AddTestCase(
        new Ns3TcpCubicCntUpdateIntervalTestCase(Time(0), 101, "Count computed on every ACK"),
        TestCase::Duration::QUICK);
    AddTestCase(new Ns3TcpCubicCntUpdateIntervalTestCase(Seconds(10),
                                                         100,
                                                         "Count kept within CntUpdateInterval"),
                TestCase::Duration::QUICK);
    AddTestCase(new Ns3TcpCubicLinuxHzTestCase(100, 999, 1000, "Linux arithmetic of cubic"),
                TestCase::Duration::QUICK);
    AddTestCase(new Ns3TcpCubicLinuxHzTestCase(0, 998, 998, "Published arithmetic of cubic"),
                TestCase::Duration::QUICK);
}

/**
 * Static variable for test initialization
 */
static Ns3TcpCubicTestSuite ns3TcpCubicTestSuite;
