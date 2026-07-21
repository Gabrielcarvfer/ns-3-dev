/**
 * Copyright (c) 2026 Centre Tecnològic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 *
 * Trace the TCP loss recovery behavior: the program transfers data over a
 * point-to-point link (100 ms RTT), drops the data segments with the given
 * sequence numbers (a repeated sequence number drops the retransmission too)
 * and prints every signal (transmissions, retransmissions, ACKs, duplicate
 * ACKs, SACK blocks) together with the congestion state machine transitions.
 *
 * The output can be compared against a real Linux TCP stack driven through
 * the same scenario, and plotted, with tcp-recovery-fsm-compare.py:
 *
 *   ./ns3 run 'tcp-recovery-fsm-trace --sack=1 --drops=20001:22001' > ns3.log
 *   python3 examples/tcp/tcp-recovery-fsm-compare.py linux 1 20001:22001 1000 linux.json
 *   python3 examples/tcp/tcp-recovery-fsm-compare.py plot ns3.log linux.json out.png
 *
 * The scenarios of the ns3-tcp-recovery-fsm test suite correspond to:
 *   --sack=1 --drops=20001:22001                                  (@RFC{6675})
 *   --sack=0 --drops=5001:6001:7001:8001:9001
 *            --ns3::TcpSocketBase::MinRto=400ms       (@RFC{6582} + @RFC{6675})
 *   --sack=1 --drops=5001:5001               (lost retransmission detection)
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

#include <map>
#include <set>

using namespace ns3;

// Deep-dive tracer: logs TCP signals (ACKs, dupacks, SACK blocks, RTO) and the
// congestion state machine transitions they trigger, plus every (re)transmission.

std::multiset<uint32_t> g_seqsToDropMulti; //!< Data sequence numbers to drop
std::set<uint32_t> g_seqsSent;             //!< Data sequence numbers transmitted so far
uint32_t g_lastAck = 0;                    //!< Highest cumulative ACK received
uint32_t g_dupCount = 0;                   //!< Current duplicate ACK count

/**
 * @return The current simulation time in seconds.
 */
double
TNow()
{
    return Simulator::Now().GetSeconds();
}

/**
 * Print a congestion state machine transition.
 *
 * @param o The old congestion state.
 * @param n The new congestion state.
 */
void
CongStateTrace(TcpSocketState::TcpCongState_t o, TcpSocketState::TcpCongState_t n)
{
    std::cout << TNow() << " STATE   " << TcpSocketState::TcpCongStateName[o] << " -> "
              << TcpSocketState::TcpCongStateName[n] << std::endl;
}

/**
 * Print a congestion window change.
 *
 * @param o The old congestion window.
 * @param n The new congestion window.
 */
void
CwndTrace(uint32_t o, uint32_t n)
{
    std::cout << TNow() << " CWND    " << o << " -> " << n << std::endl;
}

/**
 * Print a slow start threshold change.
 *
 * @param o The old slow start threshold.
 * @param n The new slow start threshold.
 */
void
SsThreshTrace(uint32_t o, uint32_t n)
{
    std::cout << TNow() << " SSTH    " << o << " -> " << n << std::endl;
}

/**
 * Print a retransmission timeout estimate change.
 *
 * @param o The old RTO.
 * @param n The new RTO.
 */
void
RtoTrace(Time o, Time n)
{
    std::cout << TNow() << " RTO-VAL " << o.GetSeconds() << " -> " << n.GetSeconds() << std::endl;
}

/**
 * Print a transmitted data segment, flagging retransmissions.
 *
 * @param p The transmitted packet.
 * @param h The TCP header.
 * @param s The transmitting socket.
 */
void
TxTrace(Ptr<const Packet> p, const TcpHeader& h, Ptr<const TcpSocketBase> s)
{
    if (p->GetSize() == 0)
    {
        return;
    }
    uint32_t seq = h.GetSequenceNumber().GetValue();
    bool retx = !g_seqsSent.insert(seq).second;
    std::cout << TNow() << " TX      seq=" << seq << " len=" << p->GetSize()
              << (retx ? "  [RETX]" : "") << std::endl;
}

/**
 * Print a received ACK, annotated with the duplicate ACK count and the SACK blocks.
 *
 * @param p The received packet.
 * @param h The TCP header.
 * @param s The receiving socket.
 */
void
RxTrace(Ptr<const Packet> p, const TcpHeader& h, Ptr<const TcpSocketBase> s)
{
    if (!(h.GetFlags() & TcpHeader::ACK) || (h.GetFlags() & TcpHeader::SYN))
    {
        return;
    }
    uint32_t ack = h.GetAckNumber().GetValue();
    std::string kind;
    if (ack > g_lastAck)
    {
        kind = (g_lastAck != 0 && g_dupCount > 0) ? "CUMACK(partial-or-full)" : "CUMACK";
        g_lastAck = ack;
        g_dupCount = 0;
    }
    else if (ack == g_lastAck && p->GetSize() == 0)
    {
        g_dupCount++;
        kind = "DUPACK#" + std::to_string(g_dupCount);
    }
    else
    {
        kind = "ACK";
    }
    std::string sackStr;
    if (h.HasOption(TcpOption::SACK))
    {
        auto sack = DynamicCast<const TcpOptionSack>(h.GetOption(TcpOption::SACK));
        sackStr = " SACK={";
        for (const auto& b : sack->GetSackList())
        {
            sackStr += "[" + std::to_string(b.first.GetValue()) + "," +
                       std::to_string(b.second.GetValue()) + ")";
        }
        sackStr += "}";
    }
    std::cout << TNow() << " RX-ACK  ack=" << ack << " " << kind << sackStr << std::endl;
}

/**
 * Print a SYN segment with its MSS option.
 *
 * @param dir The direction (TX or RX).
 * @param p The packet.
 * @param h The TCP header.
 * @param s The socket.
 */
void
SynTrace(std::string dir, Ptr<const Packet> p, const TcpHeader& h, Ptr<const TcpSocketBase> s)
{
    if (!(h.GetFlags() & TcpHeader::SYN))
    {
        return;
    }
    std::cout << TNow() << " " << dir << "-SYN flags=" << TcpHeader::FlagsToString(h.GetFlags());
    if (h.HasOption(TcpOption::MSS))
    {
        auto mss = DynamicCast<const TcpOptionMSS>(h.GetOption(TcpOption::MSS));
        std::cout << " MSS=" << mss->GetMSS();
    }
    std::cout << " optlen=" << (h.GetLength() * 4 - 20) << std::endl;
}

/// Connect the traces to the sender socket (created at application start).
void
ConnectTraces()
{
    Config::ConnectWithoutContextFailSafe("/NodeList/0/$ns3::TcpL4Protocol/SocketList/0/CongState",
                                          MakeCallback(&CongStateTrace));
    Config::ConnectWithoutContextFailSafe(
        "/NodeList/0/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow",
        MakeCallback(&CwndTrace));
    Config::ConnectWithoutContextFailSafe(
        "/NodeList/0/$ns3::TcpL4Protocol/SocketList/0/SlowStartThreshold",
        MakeCallback(&SsThreshTrace));
    Config::ConnectWithoutContextFailSafe("/NodeList/0/$ns3::TcpL4Protocol/SocketList/0/RTO",
                                          MakeCallback(&RtoTrace));
    Config::ConnectWithoutContextFailSafe("/NodeList/0/$ns3::TcpL4Protocol/SocketList/0/Tx",
                                          MakeCallback(&TxTrace));
    Config::ConnectWithoutContextFailSafe("/NodeList/0/$ns3::TcpL4Protocol/SocketList/0/Rx",
                                          MakeCallback(&RxTrace));
    Config::ConnectWithoutContextFailSafe("/NodeList/0/$ns3::TcpL4Protocol/SocketList/0/Tx",
                                          MakeCallback(&SynTrace).Bind(std::string("TX")));
    Config::ConnectWithoutContextFailSafe("/NodeList/0/$ns3::TcpL4Protocol/SocketList/0/Rx",
                                          MakeCallback(&SynTrace).Bind(std::string("RX")));
    // Report the effective segment size after the handshake
    Ptr<TcpSocketBase> sock;
    Config::MatchContainer m =
        Config::LookupMatches("/NodeList/0/$ns3::TcpL4Protocol/SocketList/0");
    if (m.GetN())
    {
        sock = DynamicCast<TcpSocketBase>(m.Get(0));
        UintegerValue segSize;
        sock->GetAttribute("SegmentSize", segSize);
        std::cout << TNow() << " SEGSIZE attribute=" << segSize.Get() << std::endl;
    }
}

/**
 * Error model dropping the data segments with the given sequence numbers
 * (each listed occurrence once).
 */
class SeqDropModel : public ErrorModel
{
  public:
    /**
     * Register this type.
     * @return The TypeId.
     */
    static TypeId GetTypeId()
    {
        static TypeId tid =
            TypeId("SeqDropModel").SetParent<ErrorModel>().AddConstructor<SeqDropModel>();
        return tid;
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
        uint32_t seq = tcp.GetSequenceNumber().GetValue();
        auto it = g_seqsToDropMulti.find(seq);
        if (it != g_seqsToDropMulti.end())
        {
            g_seqsToDropMulti.erase(it);
            std::cout << TNow() << " DROP    seq=" << seq << " len=" << copy->GetSize()
                      << std::endl;
            return true;
        }
        return false;
    }

    void DoReset() override
    {
    }
};

int
main(int argc, char* argv[])
{
    bool sack = true;
    std::string drops = "";
    double stopTime = 3;
    CommandLine cmd;
    cmd.AddValue("sack", "Enable SACK", sack);
    cmd.AddValue("drops", "colon separated data seq numbers to drop (repeats allowed)", drops);
    cmd.AddValue("stop", "stop time", stopTime);
    cmd.Parse(argc, argv);

    Config::SetDefault("ns3::TcpSocketBase::Sack", BooleanValue(sack));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(512));
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(10));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(1));
    Config::SetDefault("ns3::TcpL4Protocol::SocketType", StringValue("ns3::TcpNewReno"));
    Config::SetDefault("ns3::TcpL4Protocol::RecoveryType",
                       TypeIdValue(TypeId::LookupByName("ns3::TcpClassicRecovery")));

    std::stringstream ss(drops);
    std::string tok;
    while (std::getline(ss, tok, ':'))
    {
        g_seqsToDropMulti.insert(std::stoul(tok));
    }

    NodeContainer n;
    n.Create(2);
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("10Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("50ms"));
    auto d = p2p.Install(n);
    InternetStackHelper stack;
    stack.Install(n);
    Ipv4AddressHelper a;
    a.SetBase("10.0.0.0", "255.255.255.0");
    auto i = a.Assign(d);

    // Drop data segments on the receiver side device
    Ptr<SeqDropModel> em = CreateObject<SeqDropModel>();
    d.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(em));

    uint16_t port = 5000;
    PacketSinkHelper sink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    auto sapp = sink.Install(n.Get(1));
    sapp.Start(Seconds(0));
    BulkSendHelper bulk("ns3::TcpSocketFactory", InetSocketAddress(i.GetAddress(1), port));
    bulk.SetAttribute("MaxBytes", UintegerValue(40000));
    auto capp = bulk.Install(n.Get(0));
    capp.Start(Seconds(0.1));

    Simulator::Schedule(Seconds(0.1001), &ConnectTraces);
    Simulator::Schedule(Seconds(0.5), [] {
        Ptr<TcpSocketBase> sock;
        Config::MatchContainer m =
            Config::LookupMatches("/NodeList/0/$ns3::TcpL4Protocol/SocketList/0");
        if (m.GetN())
        {
            sock = DynamicCast<TcpSocketBase>(m.Get(0));
            UintegerValue segSize;
            sock->GetAttribute("SegmentSize", segSize);
            std::cout << TNow() << " SEGSIZE attribute=" << segSize.Get() << std::endl;
        }
    });

    Simulator::Stop(Seconds(stopTime));
    Simulator::Run();
    auto rx = DynamicCast<PacketSink>(sapp.Get(0))->GetTotalRx();
    std::cout << TNow() << " DONE    rx=" << rx << std::endl;
    Simulator::Destroy();
    return 0;
}
