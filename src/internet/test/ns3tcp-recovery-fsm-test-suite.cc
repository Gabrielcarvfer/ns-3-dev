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
 * @ingroup system-tests-tcp
 *
 * Error model dropping the data segments with the given sequence numbers
 * (each listed occurrence once, so a repeated sequence number drops the
 * original transmission and the retransmission).
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
        if (copy->GetSize() == 0)
        {
            return false;
        }
        auto it = m_seqsToDrop.find(tcp.GetSequenceNumber().GetValue());
        if (it != m_seqsToDrop.end())
        {
            m_seqsToDrop.erase(it);
            return true;
        }
        return false;
    }

    void DoReset() override
    {
    }

    std::multiset<uint32_t> m_seqsToDrop; //!< Sequence numbers to drop.
};

/**
 * @ingroup system-tests-tcp
 *
 * @brief Check the congestion state machine transitions during loss recovery.
 *
 * The test transfers data over a point-to-point link, drops the configured
 * data segments (dropping a sequence number twice drops the original
 * transmission and the retransmission) and records the congestion state
 * machine transitions and the retransmitted segments. It then compares them
 * with the expected ones:
 *
 * - Two separated losses with SACK (@RFC{6675}): CA_OPEN -> CA_DISORDER on the
 *   first duplicate ACK, CA_DISORDER -> CA_RECOVERY on the third, each lost
 *   segment retransmitted exactly once (no NewReno-style forced
 *   retransmission on partial ACKs) and CA_RECOVERY -> CA_OPEN on the full
 *   ACK, without retransmission timeouts.
 * - Five alternating losses without SACK (@RFC{6582}): one retransmission per
 *   partial ACK per RTT, with the recovery lasting longer than the minimum
 *   RTO, so the recovery completes without timeouts only if the
 *   retransmission timer is reset on every partial ACK (@RFC{6675},
 *   Section 6).
 * - A lost retransmission with SACK: the retransmission is deemed lost when
 *   a segment sent after it is SACKed, and it is retransmitted again during
 *   the same recovery, without retransmission timeouts.
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
     */
    Ns3TcpRecoveryFsmTestCase(const std::string& name,
                              bool sack,
                              const std::vector<uint32_t>& drops,
                              const std::map<uint32_t, uint32_t>& expectedRetx,
                              Time minRto)
        : TestCase(name),
          m_sack(sack),
          m_drops(drops),
          m_expectedRetx(expectedRetx),
          m_minRto(minRto)
    {
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
        uint32_t seq = header.GetSequenceNumber().GetValue();
        if (!m_seqsSent.insert(seq).second)
        {
            m_retx[seq]++;
        }
    }

    /// Connect the traces to the sender socket (created at application start).
    void ConnectTraces()
    {
        Config::ConnectWithoutContext(
            "/NodeList/0/$ns3::TcpL4Protocol/SocketList/0/CongState",
            MakeCallback(&Ns3TcpRecoveryFsmTestCase::CongStateTrace, this));
        Config::ConnectWithoutContext("/NodeList/0/$ns3::TcpL4Protocol/SocketList/0/Tx",
                                      MakeCallback(&Ns3TcpRecoveryFsmTestCase::TxTrace, this));
    }

    bool m_sack;                                 //!< Whether SACK is enabled.
    std::vector<uint32_t> m_drops;               //!< Sequence numbers to drop.
    std::map<uint32_t, uint32_t> m_expectedRetx; //!< Expected retransmission counts.
    Time m_minRto;                               //!< Minimum retransmission timeout.
    std::vector<std::string> m_transitions;      //!< Observed state transitions.
    std::set<uint32_t> m_seqsSent;               //!< Sequence numbers transmitted so far.
    std::map<uint32_t, uint32_t> m_retx;         //!< Observed retransmission counts.
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
    for (auto seq : m_drops)
    {
        errorModel->AddSeqToDrop(seq);
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

    NS_TEST_ASSERT_MSG_EQ(rxBytes, 40000, "Not all data was delivered");

    // The recovery episode must follow CA_OPEN -> CA_DISORDER (first dupack),
    // CA_DISORDER -> CA_RECOVERY (loss inferred) and CA_RECOVERY -> CA_OPEN
    // (full ACK), without ever entering CA_LOSS (no retransmission timeout)
    std::vector<std::string> expectedTransitions{"CA_OPEN->CA_DISORDER",
                                                 "CA_DISORDER->CA_RECOVERY",
                                                 "CA_RECOVERY->CA_OPEN"};
    NS_TEST_ASSERT_MSG_EQ(m_transitions.size(),
                          expectedTransitions.size(),
                          "Unexpected number of congestion state transitions");
    for (std::size_t i = 0; i < expectedTransitions.size() && i < m_transitions.size(); i++)
    {
        NS_TEST_ASSERT_MSG_EQ(m_transitions[i],
                              expectedTransitions[i],
                              "Unexpected congestion state transition");
    }

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
 * @ingroup system-tests-tcp
 *
 * @brief Congestion state machine loss recovery TestSuite.
 */
class Ns3TcpRecoveryFsmTestSuite : public TestSuite
{
  public:
    Ns3TcpRecoveryFsmTestSuite()
        : TestSuite("ns3-tcp-recovery-fsm", Type::SYSTEM)
    {
        // The segment payload is 500 bytes (512 bytes MSS minus 12 bytes of
        // timestamp option) and the first data byte has sequence number 1

        // RFC 6675: two separated losses, each retransmitted exactly once (no
        // NewReno-style forced retransmission on the partial ACK), recovery
        // completes without timeouts
        AddTestCase(new Ns3TcpRecoveryFsmTestCase("SACK recovery with two separated losses",
                                                  true,
                                                  {20001, 22001},
                                                  {{20001, 1}, {22001, 1}},
                                                  Seconds(1)),
                    TestCase::Duration::QUICK);

        // RFC 6582: five alternating losses, one retransmission per partial
        // ACK per RTT. The recovery spans five RTTs (~500 ms), longer than
        // the minimum RTO (400 ms), so it completes without timeouts only if
        // the retransmission timer is reset on every partial ACK (RFC 6675,
        // Section 6)
        AddTestCase(new Ns3TcpRecoveryFsmTestCase(
                        "NewReno recovery with five losses and multiple partial ACKs",
                        false,
                        {5001, 6001, 7001, 8001, 9001},
                        {{5001, 1}, {6001, 1}, {7001, 1}, {8001, 1}, {9001, 1}},
                        MilliSeconds(400)),
                    TestCase::Duration::QUICK);

        // Lost retransmission: the retransmission is deemed lost when a
        // segment sent after it is SACKed, and retransmitted again during the
        // same recovery, without timeouts
        AddTestCase(new Ns3TcpRecoveryFsmTestCase("SACK recovery with a lost retransmission",
                                                  true,
                                                  {5001, 5001},
                                                  {{5001, 2}},
                                                  Seconds(1)),
                    TestCase::Duration::QUICK);
    }
};

/// Static variable for test initialization
static Ns3TcpRecoveryFsmTestSuite g_ns3TcpRecoveryFsmTestSuite;
