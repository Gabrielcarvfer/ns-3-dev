/*
 * Copyright (c) 2026 Centre Tecnològic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/bulk-send-helper.h"
#include "ns3/config.h"
#include "ns3/error-model.h"
#include "ns3/inet-socket-address.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/ipv4-header.h"
#include "ns3/log.h"
#include "ns3/node-container.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/packet-sink.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/point-to-point-net-device.h"
#include "ns3/pointer.h"
#include "ns3/ppp-header.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"

#include <map>
#include <set>
#include <sstream>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("Ns3TcpRecoveryFsmTest");

/**
 * @ingroup internet-test
 *
 * Error model dropping data segments by sequence number, once per listed
 * occurrence (a repeated number also drops the retransmission). Sequence
 * numbers are relative to the sender ISN seen in its SYN: the first data
 * byte is 1.
 */
class TcpSeqMultiDropErrorModel : public ErrorModel
{
  public:
    /**
     * Register this type.
     * @return The TypeId.
     */
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::TcpSeqMultiDropErrorModel")
                                .SetParent<ErrorModel>()
                                .SetGroupName("Internet")
                                .AddConstructor<TcpSeqMultiDropErrorModel>();
        return tid;
    }

    /**
     * Add a data sequence number to drop (may be repeated).
     * @param seq The sequence number.
     */
    void AddSeqToDrop(uint32_t seq)
    {
        m_seqsToDrop.insert(seq);
    }

    /**
     * Delay the first transmission of a data segment until the given number
     * of later data segments have been received, reordering it behind them.
     * @param seq The sequence number.
     * @param segments The number of later segments delivered before it.
     * @param device The device on which the delayed segment is delivered.
     */
    void AddSeqToHold(uint32_t seq, uint32_t segments, Ptr<PointToPointNetDevice> device)
    {
        m_seqToHold = seq;
        m_holdSegments = segments;
        m_device = device;
    }

    /**
     * Get the initial sequence number of the sender.
     * @return The sequence number of the SYN received, 0 if none was.
     */
    SequenceNumber32 GetIsn() const
    {
        return m_isn;
    }

  private:
    bool DoCorrupt(Ptr<Packet> p) override
    {
        Ptr<Packet> copy = p->Copy();
        PppHeader ppp;
        copy->RemoveHeader(ppp);
        Ipv4Header ip;
        copy->RemoveHeader(ip);
        if (ip.GetProtocol() != 6)
        {
            return false;
        }
        TcpHeader tcp;
        copy->RemoveHeader(tcp);
        if (tcp.GetFlags() & TcpHeader::SYN)
        {
            m_isn = tcp.GetSequenceNumber();
        }
        if (copy->GetSize() == 0)
        {
            return false;
        }
        uint32_t seq = tcp.GetSequenceNumber().GetValue() - m_isn.GetValue();
        auto it = m_seqsToDrop.find(seq);
        if (it != m_seqsToDrop.end())
        {
            m_seqsToDrop.erase(it);
            return true;
        }
        if (m_held)
        {
            if (--m_holdSegments == 0)
            {
                // Delivered right after the segment being received
                Simulator::ScheduleNow(&PointToPointNetDevice::Receive, m_device, m_held);
                m_held = nullptr;
            }
        }
        else if (m_seqToHold != 0 && seq == m_seqToHold)
        {
            m_seqToHold = 0;
            m_held = p->Copy();
            return true;
        }
        return false;
    }

    void DoReset() override
    {
    }

    std::multiset<uint32_t> m_seqsToDrop; //!< Sequence numbers to drop.
    uint32_t m_seqToHold{0};              //!< Sequence number to delay (0 for none).
    uint32_t m_holdSegments{0};           //!< Later segments to deliver before the held one.
    Ptr<Packet> m_held;                   //!< The delayed segment, while held.
    Ptr<PointToPointNetDevice> m_device;  //!< Device on which the held segment is delivered.
    SequenceNumber32 m_isn{0};            //!< Initial sequence number of the sender.
};

/**
 * @ingroup internet-test
 *
 * @brief Check the congestion state machine transitions during loss recovery.
 *
 * Drops (a repeated sequence number also drops the retransmission) or
 * reorders data segments on a point-to-point link and compares the observed
 * state transitions and retransmissions with the expected ones. A new
 * recovery must never start before the cumulative ACK reaches the previous
 * recovery point (@RFC{6582}, Section 3.2, step 4; @RFC{6675}, Section 5.1).
 * The default expected sequence is CA_OPEN -> CA_DISORDER -> CA_RECOVERY ->
 * CA_OPEN without timeouts; the cases are listed in the test suite.
 */
class Ns3TcpRecoveryFsmTestCase : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     * @param sack Whether to enable SACK.
     * @param drops Data sequence numbers to drop (repeats allowed).
     * @param expectedRetx Expected retransmissions (sequence number -> count).
     * @param minRto Minimum retransmission timeout.
     * @param expectedTransitions Expected congestion state transitions, as
     *        "OLD->NEW" strings (empty for the single fast recovery episode
     *        CA_OPEN -> CA_DISORDER -> CA_RECOVERY -> CA_OPEN).
     */
    Ns3TcpRecoveryFsmTestCase(const std::string& name,
                              bool sack,
                              const std::vector<uint32_t>& drops,
                              const std::map<uint32_t, uint32_t>& expectedRetx,
                              Time minRto,
                              const std::vector<std::string>& expectedTransitions = {})
        : TestCase(name),
          m_sack(sack),
          m_drops(drops),
          m_expectedRetx(expectedRetx),
          m_minRto(minRto),
          m_expectedTransitions(expectedTransitions)
    {
        if (m_expectedTransitions.empty())
        {
            m_expectedTransitions = {"CA_OPEN->CA_DISORDER",
                                     "CA_DISORDER->CA_RECOVERY",
                                     "CA_RECOVERY->CA_OPEN"};
        }
    }

    /**
     * Delay the first transmission of a data segment behind later segments.
     * @param seq The sequence number of the delayed segment.
     * @param segments The number of later segments delivered before it.
     */
    void SetHold(uint32_t seq, uint32_t segments)
    {
        m_holdSeq = seq;
        m_holdSegments = segments;
    }

  private:
    void DoRun() override;

    /**
     * Trace the congestion state machine transitions.
     * @param oldValue Old congestion state.
     * @param newValue New congestion state.
     */
    void CongStateTrace(TcpSocketState::TcpCongState_t oldValue,
                        TcpSocketState::TcpCongState_t newValue)
    {
        std::ostringstream oss;
        oss << TcpSocketState::TcpCongStateName[oldValue] << "->"
            << TcpSocketState::TcpCongStateName[newValue];
        m_transitions.push_back(oss.str());

        if (newValue == TcpSocketState::CA_RECOVERY)
        {
            NS_TEST_EXPECT_MSG_GT_OR_EQ(m_highestAck,
                                        m_recoveryPoint,
                                        "Recovery entered before the cumulative ACK reached the "
                                        "recovery point of the previous episode");
        }
        if (newValue == TcpSocketState::CA_RECOVERY || newValue == TcpSocketState::CA_LOSS)
        {
            m_recoveryPoint = m_highestTx;
        }
    }

    /**
     * Trace the transmitted segments and count retransmissions.
     * @param packet The transmitted packet.
     * @param header The TCP header.
     * @param socket The transmitting socket.
     */
    void TxTrace(Ptr<const Packet> packet, const TcpHeader& header, Ptr<const TcpSocketBase> socket)
    {
        if (packet->GetSize() == 0)
        {
            return;
        }
        uint32_t seq = ToRelative(header.GetSequenceNumber()).GetValue();
        if (!m_seqsSent.insert(seq).second)
        {
            m_retx[seq]++;
        }
    }

    /**
     * Trace the received segments, keeping the highest cumulative ACK.
     * @param packet The received packet.
     * @param header The TCP header.
     * @param socket The receiving socket.
     */
    void RxTrace(Ptr<const Packet> packet, const TcpHeader& header, Ptr<const TcpSocketBase> socket)
    {
        if ((header.GetFlags() & TcpHeader::ACK) &&
            ToRelative(header.GetAckNumber()) > m_highestAck)
        {
            m_highestAck = ToRelative(header.GetAckNumber());
        }
    }

    /**
     * Trace the highest sequence number sent (HighData).
     * @param oldValue Old value.
     * @param newValue New value.
     */
    void HighestTxTrace(SequenceNumber32 oldValue, SequenceNumber32 newValue)
    {
        m_highestTx = ToRelative(newValue);
    }

    /**
     * Express a sequence number of the sender relative to its initial one.
     * @param seq The absolute sequence number.
     * @return The sequence number, the first data byte being numbered 1.
     */
    SequenceNumber32 ToRelative(SequenceNumber32 seq) const
    {
        return SequenceNumber32(seq.GetValue() - m_errorModel->GetIsn().GetValue());
    }

    /// Connect the traces to the sender socket (created at application start).
    void ConnectTraces()
    {
        const std::string socket = "/NodeList/0/$ns3::TcpL4Protocol/SocketList/0/";
        Config::ConnectWithoutContext(
            socket + "CongState",
            MakeCallback(&Ns3TcpRecoveryFsmTestCase::CongStateTrace, this));
        Config::ConnectWithoutContext(socket + "Tx",
                                      MakeCallback(&Ns3TcpRecoveryFsmTestCase::TxTrace, this));
        Config::ConnectWithoutContext(socket + "Rx",
                                      MakeCallback(&Ns3TcpRecoveryFsmTestCase::RxTrace, this));
        Config::ConnectWithoutContext(
            socket + "HighestSequence",
            MakeCallback(&Ns3TcpRecoveryFsmTestCase::HighestTxTrace, this));
    }

    bool m_sack;                                    //!< Whether SACK is enabled.
    std::vector<uint32_t> m_drops;                  //!< Sequence numbers to drop.
    std::map<uint32_t, uint32_t> m_expectedRetx;    //!< Expected retransmission counts.
    Time m_minRto;                                  //!< Minimum retransmission timeout.
    std::vector<std::string> m_expectedTransitions; //!< Expected state transitions.
    uint32_t m_holdSeq{0};                          //!< Sequence number to delay (0 for none).
    uint32_t m_holdSegments{0};                     //!< Later segments delivered before it.
    std::vector<std::string> m_transitions;         //!< Observed state transitions.
    std::set<uint32_t> m_seqsSent;                  //!< Sequence numbers transmitted so far.
    std::map<uint32_t, uint32_t> m_retx;            //!< Observed retransmission counts.
    SequenceNumber32 m_highestAck{0};               //!< Highest cumulative ACK received.
    SequenceNumber32 m_highestTx{0};                //!< Highest sequence number sent.
    SequenceNumber32 m_recoveryPoint{0};            //!< HighData at the last recovery or timeout.
    Ptr<TcpSeqMultiDropErrorModel> m_errorModel;    //!< Error model, which knows the ISN.
};

void
Ns3TcpRecoveryFsmTestCase::DoRun()
{
    Config::SetDefault("ns3::TcpSocketBase::Sack", BooleanValue(m_sack));
    Config::SetDefault("ns3::TcpSocketBase::MinRto", TimeValue(m_minRto));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(512));
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(10));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(1));
    Config::SetDefault("ns3::TcpL4Protocol::SocketType", StringValue("ns3::TcpNewReno"));
    Config::SetDefault("ns3::TcpL4Protocol::RecoveryType",
                       TypeIdValue(TypeId::LookupByName("ns3::TcpClassicRecovery")));

    NodeContainer nodes;
    nodes.Create(2);
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("10Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("50ms"));
    NetDeviceContainer devices = p2p.Install(nodes);
    InternetStackHelper stack;
    stack.Install(nodes);
    Ipv4AddressHelper address;
    address.SetBase("10.0.0.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    Ptr<TcpSeqMultiDropErrorModel> errorModel = CreateObject<TcpSeqMultiDropErrorModel>();
    m_errorModel = errorModel;
    for (auto seq : m_drops)
    {
        errorModel->AddSeqToDrop(seq);
    }
    if (m_holdSeq != 0)
    {
        errorModel->AddSeqToHold(m_holdSeq,
                                 m_holdSegments,
                                 DynamicCast<PointToPointNetDevice>(devices.Get(1)));
    }
    devices.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(errorModel));

    uint16_t port = 5000;
    PacketSinkHelper sinkHelper("ns3::TcpSocketFactory",
                                InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApp = sinkHelper.Install(nodes.Get(1));
    sinkApp.Start(Seconds(0));
    BulkSendHelper bulkHelper("ns3::TcpSocketFactory",
                              InetSocketAddress(interfaces.GetAddress(1), port));
    bulkHelper.SetAttribute("MaxBytes", UintegerValue(40000));
    ApplicationContainer sourceApp = bulkHelper.Install(nodes.Get(0));
    sourceApp.Start(Seconds(0.1));

    Simulator::Schedule(Seconds(0.1001), &Ns3TcpRecoveryFsmTestCase::ConnectTraces, this);

    Simulator::Stop(Seconds(3));
    Simulator::Run();

    uint64_t rxBytes = DynamicCast<PacketSink>(sinkApp.Get(0))->GetTotalRx();
    Simulator::Destroy();
    m_errorModel = nullptr;

    NS_TEST_ASSERT_MSG_EQ(rxBytes, 40000, "Not all data was delivered");

    auto join = [](const std::vector<std::string>& v) {
        std::string out;
        for (const auto& t : v)
        {
            out += (out.empty() ? "" : ", ") + t;
        }
        return out;
    };
    NS_TEST_ASSERT_MSG_EQ(join(m_transitions),
                          join(m_expectedTransitions),
                          "Unexpected congestion state transitions");

    NS_TEST_ASSERT_MSG_EQ(m_retx.size(),
                          m_expectedRetx.size(),
                          "Unexpected number of retransmitted segments");
    for (const auto& [seq, count] : m_expectedRetx)
    {
        auto it = m_retx.find(seq);
        NS_TEST_ASSERT_MSG_EQ((it != m_retx.end()),
                              true,
                              "Segment " << seq << " was not retransmitted");
        if (it != m_retx.end())
        {
            NS_TEST_ASSERT_MSG_EQ(it->second,
                                  count,
                                  "Segment " << seq
                                             << " retransmitted an unexpected number of times");
        }
    }
}

/**
 * @ingroup internet-test
 *
 * @brief Congestion state machine loss recovery TestSuite.
 */
class Ns3TcpRecoveryFsmTestSuite : public TestSuite
{
  public:
    Ns3TcpRecoveryFsmTestSuite()
        : TestSuite("ns3-tcp-recovery-fsm", Type::SYSTEM)
    {
        // 500 byte payload (512 byte MSS minus 12 byte timestamp option),
        // first data byte at sequence number 1

        // RFC 6675: each loss retransmitted once, no forced retransmission on
        // the partial ACK
        AddTestCase(new Ns3TcpRecoveryFsmTestCase("SACK recovery with two separated losses",
                                                  true,
                                                  {20001, 22001},
                                                  {{20001, 1}, {22001, 1}},
                                                  Seconds(1)),
                    TestCase::Duration::QUICK);

        // RFC 6582: one retransmission per partial ACK. The first partial ACK
        // comes ~200 ms after the last timer restart, partial ACKs ~100 ms
        // apart, the full ACK ~400 ms after the first. With a 300 ms RTO,
        // restarting on every partial ACK (RFC 6298, Section 5.3) keeps
        // >= 100 ms of margin; restarting on the first only (Impatient
        // variant, RFC 6582, Section 3.2) times out ~100 ms before the full ACK
        AddTestCase(new Ns3TcpRecoveryFsmTestCase(
                        "NewReno recovery with five losses and multiple partial ACKs",
                        false,
                        {5001, 6001, 7001, 8001, 9001},
                        {{5001, 1}, {6001, 1}, {7001, 1}, {8001, 1}, {9001, 1}},
                        MilliSeconds(300)),
                    TestCase::Duration::QUICK);

        // The retransmission is deemed lost once a later segment is SACKed and
        // is retransmitted again in the same recovery
        AddTestCase(new Ns3TcpRecoveryFsmTestCase("SACK recovery with a lost retransmission",
                                                  true,
                                                  {5001, 5001},
                                                  {{5001, 2}},
                                                  Seconds(1)),
                    TestCase::Duration::QUICK);

        // Without SACK only the timer recovers a lost retransmission; later
        // duplicate ACKs start no recovery below the recovery point
        // (RFC 6582, Section 3.2, step 4; RFC 6675, Section 5.1)
        AddTestCase(new Ns3TcpRecoveryFsmTestCase("NewReno recovery with a lost retransmission",
                                                  false,
                                                  {5001, 5001},
                                                  {{5001, 2}},
                                                  Seconds(1),
                                                  {"CA_OPEN->CA_DISORDER",
                                                   "CA_DISORDER->CA_RECOVERY",
                                                   "CA_RECOVERY->CA_LOSS",
                                                   "CA_LOSS->CA_OPEN"}),
                    TestCase::Duration::QUICK);

        // Last segment (bytes 39501-40000): no duplicate ACK, timer recovers it
        AddTestCase(new Ns3TcpRecoveryFsmTestCase("Tail loss recovered by a timeout",
                                                  true,
                                                  {39501},
                                                  {{39501, 1}},
                                                  Seconds(1),
                                                  {"CA_OPEN->CA_LOSS", "CA_LOSS->CA_OPEN"}),
                    TestCase::Duration::QUICK);

        // 16001 (the recovery point) is the first new segment sent in the
        // recovery of 5001; its loss starts a second recovery once the first
        // ends (RFC 6675, Section 5.1; RFC 6582, Section 3.2, step 2)
        AddTestCase(new Ns3TcpRecoveryFsmTestCase("Loss of a segment sent during the recovery",
                                                  true,
                                                  {5001, 16001},
                                                  {{5001, 1}, {16001, 1}},
                                                  Seconds(1),
                                                  {"CA_OPEN->CA_DISORDER",
                                                   "CA_DISORDER->CA_RECOVERY",
                                                   "CA_RECOVERY->CA_OPEN",
                                                   "CA_OPEN->CA_DISORDER",
                                                   "CA_DISORDER->CA_RECOVERY",
                                                   "CA_RECOVERY->CA_OPEN"}),
                    TestCase::Duration::QUICK);

        // A segment held behind two later ones: two duplicate ACKs, below the
        // threshold, nothing retransmitted
        auto reordering =
            new Ns3TcpRecoveryFsmTestCase("Reordering without loss",
                                          true,
                                          {},
                                          {},
                                          Seconds(1),
                                          {"CA_OPEN->CA_DISORDER", "CA_DISORDER->CA_OPEN"});
        reordering->SetHold(5001, 2);
        AddTestCase(reordering, TestCase::Duration::QUICK);
    }
};

/// Static variable for test initialization
static Ns3TcpRecoveryFsmTestSuite g_ns3TcpRecoveryFsmTestSuite;
