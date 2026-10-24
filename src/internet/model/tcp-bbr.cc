/*
 * Copyright (c) 2018 NITK Surathkal
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Vivek Jain <jain.vivek.anand@gmail.com>
 *          Viyom Mittal <viyommittal@gmail.com>
 *          Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 */

#include "tcp-bbr.h"

#include "ns3/abort.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/string.h"

#include <algorithm>
#include <sstream>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TcpBbr");
NS_OBJECT_ENSURE_REGISTERED(TcpBbr);

const double TcpBbr::PACING_GAIN_CYCLE[] = {5.0 / 4, 3.0 / 4, 1, 1, 1, 1, 1, 1};

TypeId
TcpBbr::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::TcpBbr")
            .SetParent<TcpCongestionOps>()
            .AddConstructor<TcpBbr>()
            .SetGroupName("Internet")
            .AddAttribute("Stream",
                          "Random number stream (default is set to 4 to align with Linux results)",
                          UintegerValue(4),
                          MakeUintegerAccessor(&TcpBbr::SetStream),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute(
                "LinuxRecovery",
                "Handle the loss recovery as Linux (bbr_set_cwnd_to_recover_or_restore()): "
                "on each ACK, from the congestion state seen on the previous one, "
                "so that a recovery left and entered again on one ACK stays one "
                "recovery; deduct the newly lost data from the window in any state; "
                "and save the window of before the recovery from that previous state",
                BooleanValue(false),
                MakeBooleanAccessor(&TcpBbr::m_linuxRecovery),
                MakeBooleanChecker())
            .AddAttribute("LinuxHz",
                          "If not zero, the rate of the jiffies of Linux (CONFIG_HZ) on which "
                          "the minimum RTT and the PROBE_RTT duration are timed, as Linux "
                          "(tcp_jiffies32), rather than to the nanosecond",
                          UintegerValue(0),
                          MakeUintegerAccessor(&TcpBbr::m_linuxHz),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute(
                "ProbeBwStartPhases",
                "The phases of the pacing gain cycle in which PROBE_BW starts, one "
                "per entry into PROBE_BW, separated by spaces or commas, each from 0 (gain "
                "5/4) to 7, except 1 (gain 3/4); past them, or with none, the phase "
                "is drawn at random among these seven phases, as Linux does",
                StringValue(""),
                MakeStringAccessor(&TcpBbr::SetProbeBwStartPhases),
                MakeStringChecker())
            .AddAttribute("HighGain",
                          "Value of high gain",
                          DoubleValue(2.89),
                          MakeDoubleAccessor(&TcpBbr::m_highGain),
                          MakeDoubleChecker<double>())
            .AddAttribute("LinuxArithmetic",
                          "Compute the pacing rate and the gains as Linux (tcp_bbr.c): the "
                          "bandwidth in packets per microsecond and the gains in fixed point "
                          "(a high gain of 739/256 and a drain gain of 88/256, whatever "
                          "HighGain), the initial bandwidth from the smoothed RTT in "
                          "microseconds, the pacing rate rounded down to whole bytes per "
                          "second, a minimum RTT renewed before it expires only by a smaller "
                          "sample in whole microseconds, and the data in flight counted at the "
                          "departure time of the next packet (bbr_packets_in_net_at_edt())",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpBbr::m_linuxArithmetic),
                          MakeBooleanChecker())
            .AddAttribute("BwWindowLength",
                          "Length of bandwidth windowed filter",
                          UintegerValue(10),
                          MakeUintegerAccessor(&TcpBbr::m_bandwidthWindowLength),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("RttWindowLength",
                          "Length of RTT windowed filter",
                          TimeValue(Seconds(10)),
                          MakeTimeAccessor(&TcpBbr::m_minRttFilterLen),
                          MakeTimeChecker())
            .AddAttribute("ProbeRttDuration",
                          "Time to be spent in PROBE_RTT phase",
                          TimeValue(MilliSeconds(200)),
                          MakeTimeAccessor(&TcpBbr::m_probeRttDuration),
                          MakeTimeChecker())
            .AddAttribute("ExtraAckedRttWindowLength",
                          "Window length of extra acked window",
                          UintegerValue(5),
                          MakeUintegerAccessor(&TcpBbr::m_extraAckedWinRttLength),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("AckEpochAckedResetThresh",
                          "Number of segments acknowledged in an ACK aggregation epoch past "
                          "which the epoch is reset, the count saturating one below (Linux "
                          "bbr_ack_epoch_acked_reset_thresh, 2^20 packets)",
                          UintegerValue(1 << 20),
                          MakeUintegerAccessor(&TcpBbr::m_ackEpochAckedResetThresh),
                          MakeUintegerChecker<uint32_t>(1))
            .AddTraceSource("MinRtt",
                            "Estimated two-way round-trip propagation delay of the path, estimated "
                            "from the windowed minimum recent round-trip delay sample",
                            MakeTraceSourceAccessor(&TcpBbr::m_minRtt),
                            "ns3::TracedValueCallback::Time")
            .AddTraceSource("PacingGain",
                            "The dynamic pacing gain factor",
                            MakeTraceSourceAccessor(&TcpBbr::m_pacingGain),
                            "ns3::TracedValueCallback::Double")
            .AddTraceSource("CwndGain",
                            "The dynamic congestion window gain factor",
                            MakeTraceSourceAccessor(&TcpBbr::m_cWndGain),
                            "ns3::TracedValueCallback::Double");
    return tid;
}

TcpBbr::TcpBbr()
    : TcpCongestionOps()
{
    NS_LOG_FUNCTION(this);
    m_uv = CreateObject<UniformRandomVariable>();
}

TcpBbr::TcpBbr(const TcpBbr& sock)
    : TcpCongestionOps(sock),
      m_bandwidthWindowLength(sock.m_bandwidthWindowLength),
      m_pacingGain(sock.m_pacingGain),
      m_cWndGain(sock.m_cWndGain),
      m_highGain(sock.m_highGain),
      m_isPipeFilled(sock.m_isPipeFilled),
      m_minPipeCwnd(sock.m_minPipeCwnd),
      m_roundCount(sock.m_roundCount),
      m_roundStart(sock.m_roundStart),
      m_nextRoundDelivered(sock.m_nextRoundDelivered),
      m_probeRttDuration(sock.m_probeRttDuration),
      m_probeRtPropStamp(sock.m_probeRtPropStamp),
      m_probeRttDoneStamp(sock.m_probeRttDoneStamp),
      m_probeRttRoundDone(sock.m_probeRttRoundDone),
      m_packetConservation(sock.m_packetConservation),
      m_priorCwnd(sock.m_priorCwnd),
      m_idleRestart(sock.m_idleRestart),
      m_targetCWnd(sock.m_targetCWnd),
      m_fullBandwidth(sock.m_fullBandwidth),
      m_fullBandwidthCount(sock.m_fullBandwidthCount),
      m_minRtt(Time::Max()),
      m_sendQuantum(sock.m_sendQuantum),
      m_cycleStamp(sock.m_cycleStamp),
      m_cycleIndex(sock.m_cycleIndex),
      m_minRttExpired(sock.m_minRttExpired),
      m_minRttFilterLen(sock.m_minRttFilterLen),
      m_minRttStamp(sock.m_minRttStamp),
      m_isInitialized(sock.m_isInitialized),
      m_uv(sock.m_uv),
      m_delivered(sock.m_delivered),
      m_extraAckedGain(sock.m_extraAckedGain),
      m_extraAckedWinRtt(sock.m_extraAckedWinRtt),
      m_extraAckedWinRttLength(sock.m_extraAckedWinRttLength),
      m_ackEpochAckedResetThresh(sock.m_ackEpochAckedResetThresh),
      m_extraAckedIdx(sock.m_extraAckedIdx),
      m_ackEpochTime(sock.m_ackEpochTime),
      m_ackEpochAcked(sock.m_ackEpochAcked),
      m_hasSeenRtt(sock.m_hasSeenRtt),
      m_linuxArithmetic(sock.m_linuxArithmetic),
      m_probeBwStartPhases(sock.m_probeBwStartPhases),
      m_probeBwEntries(sock.m_probeBwEntries),
      m_linuxRecovery(sock.m_linuxRecovery),
      m_prevCaState(sock.m_prevCaState),
      m_linuxHz(sock.m_linuxHz)
{
    NS_LOG_FUNCTION(this);
}

void
TcpBbr::SetRateOps(Ptr<TcpRateOps> rateOps)
{
    m_rateOps = rateOps;
}

const char* const TcpBbr::BbrModeName[BBR_PROBE_RTT + 1] = {
    "BBR_STARTUP",
    "BBR_DRAIN",
    "BBR_PROBE_BW",
    "BBR_PROBE_RTT",
};

void
TcpBbr::SetStream(uint32_t stream)
{
    NS_LOG_FUNCTION(this << stream);
    m_uv->SetStream(stream);
}

void
TcpBbr::InitRoundCounting()
{
    NS_LOG_FUNCTION(this);
    m_nextRoundDelivered = 0;
    m_roundStart = false;
    m_roundCount = 0;
}

void
TcpBbr::InitFullPipe()
{
    NS_LOG_FUNCTION(this);
    m_isPipeFilled = false;
    m_fullBandwidth = 0;
    m_fullBandwidthCount = 0;
}

void
TcpBbr::InitPacingRate(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);

    if (!tcb->m_pacing)
    {
        NS_LOG_WARN("BBR must use pacing");
        tcb->m_pacing = true;
    }

    // As bbr_init_pacing_rate_from_rtt() of Linux, the nominal bandwidth is
    // the window over the smoothed RTT, or over 1 ms without an RTT sample
    Time rtt;
    if (tcb->m_srtt.Get() != Time::Max() && tcb->m_srtt.Get().IsStrictlyPositive())
    {
        rtt = std::max(MicroSeconds(tcb->m_srtt.Get().GetMicroSeconds()), MicroSeconds(1));
        m_hasSeenRtt = true;
    }
    else
    {
        rtt = MilliSeconds(1);
    }

    DataRate nominalBandwidth(tcb->m_cWnd * 8 / rtt.GetSeconds());
    if (m_linuxArithmetic)
    {
        // In whole packets per microsecond, scaled by 2^24 (BW_UNIT)
        uint64_t bw = (static_cast<uint64_t>(tcb->m_cWnd / tcb->m_segmentSize) << 24) /
                      static_cast<uint64_t>(rtt.GetMicroSeconds());
        nominalBandwidth =
            DataRate(static_cast<double>(bw) * tcb->m_segmentSize * 8 * 1e6 / (1ULL << 24));
    }
    tcb->m_pacingRate = BwToPacingRate(tcb, nominalBandwidth, m_pacingGain);
    NS_LOG_DEBUG("Initial pacing rate " << tcb->m_pacingRate << " from an RTT of "
                                        << rtt.As(Time::US));
    // Linux starts the bandwidth filter empty (bbr_init())
    m_maxBwFilter = MaxBandwidthFilter_t(
        m_bandwidthWindowLength,
        m_linuxArithmetic ? DataRate(0) : DataRate(tcb->m_cWnd * 8 / rtt.GetSeconds()),
        0);
}

void
TcpBbr::EnterStartup()
{
    NS_LOG_FUNCTION(this);
    SetBbrState(BbrMode_t::BBR_STARTUP);
    m_pacingGain = GetHighGain();
    m_cWndGain = GetHighGain();
}

void
TcpBbr::HandleRestartFromIdle(Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << tcb << rs);
    if (tcb->m_bytesInFlight.Get() == 0U && rs.m_isAppLimited)
    {
        m_idleRestart = true;
        if (m_state == BbrMode_t::BBR_PROBE_BW)
        {
            SetPacingRate(tcb, 1);
        }
    }
}

void
TcpBbr::SetPacingRate(Ptr<TcpSocketState> tcb, double gain)
{
    NS_LOG_FUNCTION(this << tcb << gain);
    DataRate rate = BwToPacingRate(tcb, m_maxBwFilter.GetBest(), gain);

    if (!m_hasSeenRtt && tcb->m_srtt.Get() != Time::Max() && tcb->m_srtt.Get().IsStrictlyPositive())
    {
        InitPacingRate(tcb);
    }

    if (m_isPipeFilled || rate > tcb->m_pacingRate)
    {
        tcb->m_pacingRate = rate;
        NS_LOG_DEBUG("Pacing rate updated. New value: " << tcb->m_pacingRate);
    }
}

DataRate
TcpBbr::BwToPacingRate(Ptr<const TcpSocketState> tcb, DataRate bw, double gain) const
{
    NS_LOG_FUNCTION(this << tcb << bw << gain);
    if (!m_linuxArithmetic)
    {
        DataRate rate(gain * bw.GetBitRate());
        rate *= (1.f - m_pacingMargin);
        return std::min(rate, tcb->m_maxPacingRate);
    }
    // bbr_rate_bytes_per_sec(): the bandwidth in packets per microsecond
    // scaled by 2^24, the gain scaled by 2^8, 99% of the rate in bytes per
    // second, all in integers
    const uint64_t mss = tcb->m_segmentSize;
    uint64_t bwUnits = BwUnits(tcb, bw);
    auto gainUnits = static_cast<uint64_t>(gain * 256 + 1e-9);
    uint64_t rate = bwUnits * mss;
    rate *= gainUnits;
    rate >>= 8;
    rate *= 1000000 / 100 * (100 - static_cast<uint64_t>(m_pacingMargin * 100 + 0.5));
    rate >>= 24;
    return std::min(DataRate(rate * 8), tcb->m_maxPacingRate);
}

double
TcpBbr::GetHighGain() const
{
    // bbr_high_gain of Linux: 2/ln(2) rounded up to 739/256
    return m_linuxArithmetic ? 739.0 / 256 : m_highGain;
}

void
TcpBbr::SetProbeBwStartPhases(std::string phases)
{
    NS_LOG_FUNCTION(this << phases);
    m_probeBwStartPhases.clear();
    std::replace(phases.begin(), phases.end(), ',', ' ');
    std::istringstream iss(phases);
    int32_t phase;
    while (iss >> phase)
    {
        NS_ABORT_MSG_IF(phase < 0 || phase >= GAIN_CYCLE_LENGTH || phase == 1,
                        "Invalid PROBE_BW start phase " << phase);
        m_probeBwStartPhases.push_back(static_cast<uint32_t>(phase));
    }
}

Time
TcpBbr::JiffiesNow() const
{
    Time now = Simulator::Now();
    if (m_linuxHz == 0)
    {
        return now;
    }
    int64_t jiffy = 1000000000LL / m_linuxHz;
    return NanoSeconds(now.GetNanoSeconds() / jiffy * jiffy);
}

uint64_t
TcpBbr::BwUnits(Ptr<const TcpSocketState> tcb, DataRate bw) const
{
    // DataRate truncates to whole bits per second: compensate for the up to
    // one bit per second lost, so as not to round the units down wrongly
    return static_cast<uint64_t>((bw.GetBitRate() + 1.0) / 8.0 / tcb->m_segmentSize / 1e6 *
                                 (1ULL << 24));
}

uint32_t
TcpBbr::TsoSegsGoal(Ptr<const TcpSocketState> tcb) const
{
    const uint64_t mss = tcb->m_segmentSize;
    uint64_t rate = tcb->m_pacingRate.Get().GetBitRate() / 8;
    uint64_t bytes = std::min<uint64_t>(rate >> 10, 65536 - 1 - 320);
    uint64_t minTso = rate < 1200000 / 8 ? 1 : 2;
    return static_cast<uint32_t>(std::min<uint64_t>(std::max(bytes / mss, minTso), 0x7F));
}

uint32_t
TcpBbr::InNetAtEdt(Ptr<const TcpSocketState> tcb, uint32_t inFlight) const
{
    const uint64_t mss = tcb->m_segmentSize;
    Time now = Simulator::Now();
    uint64_t intervalUs =
        tcb->m_pacingDeparture > now ? (tcb->m_pacingDeparture - now).GetMicroSeconds() : 0;
    uint64_t delivered = (BwUnits(tcb, m_maxBwFilter.GetBest()) * intervalUs) >> 24;
    uint64_t atEdt = inFlight / mss;
    if (m_pacingGain > 1)
    {
        atEdt += TsoSegsGoal(tcb);
    }
    return delivered >= atEdt ? 0 : static_cast<uint32_t>((atEdt - delivered) * mss);
}

uint32_t
TcpBbr::LinuxInFlight(Ptr<TcpSocketState> tcb, double gain, bool aggregation)
{
    NS_LOG_FUNCTION(this << tcb << gain << aggregation);
    const uint64_t mss = tcb->m_segmentSize;
    uint64_t bw = BwUnits(tcb, m_maxBwFilter.GetBest());
    uint64_t cwnd = tcb->m_initialCWnd;
    if (m_minRtt != Time::Max())
    {
        // bbr_bdp()
        uint64_t w = bw * static_cast<uint64_t>(m_minRtt.Get().GetMicroSeconds());
        auto gainUnits = static_cast<uint64_t>(gain * 256 + 1e-9);
        cwnd = (((w * gainUnits) >> 8) + (1ULL << 24) - 1) >> 24;
    }
    if (aggregation && m_extraAckedGain && m_isPipeFilled)
    {
        // bbr_ack_aggregation_cwnd(), with at most 100 ms of data
        uint64_t maxAggr = (bw * 100000) >> 24;
        uint64_t aggr = std::max(m_extraAcked[0], m_extraAcked[1]) / mss;
        cwnd += std::min(aggr, maxAggr);
    }
    // bbr_quantization_budget()
    cwnd += 3 * TsoSegsGoal(tcb);
    cwnd = (cwnd + 1) & ~1ULL;
    if (m_state == BbrMode_t::BBR_PROBE_BW && m_cycleIndex == 0)
    {
        cwnd += 2;
    }
    return static_cast<uint32_t>(cwnd * mss);
}

uint32_t
TcpBbr::InFlight(Ptr<TcpSocketState> tcb, double gain)
{
    NS_LOG_FUNCTION(this << tcb << gain);
    if (m_linuxArithmetic)
    {
        return LinuxInFlight(tcb, gain, false);
    }
    if (m_minRtt == Time::Max())
    {
        return tcb->m_initialCWnd * tcb->m_segmentSize;
    }
    double quanta = 3 * m_sendQuantum;
    double estimatedBdp = m_maxBwFilter.GetBest() * m_minRtt / 8.0;

    if (m_state == BbrMode_t::BBR_PROBE_BW && m_cycleIndex == 0)
    {
        return (gain * estimatedBdp) + quanta + (2 * tcb->m_segmentSize);
    }
    return (gain * estimatedBdp) + quanta;
}

void
TcpBbr::AdvanceCyclePhase()
{
    NS_LOG_FUNCTION(this);
    m_cycleStamp = Simulator::Now();
    m_cycleIndex = (m_cycleIndex + 1) % GAIN_CYCLE_LENGTH;
    m_pacingGain = PACING_GAIN_CYCLE[m_cycleIndex];
    NS_LOG_DEBUG("Pacing gain cycle phase " << m_cycleIndex << " (gain " << m_pacingGain
                                            << "), min RTT " << m_minRtt.Get().As(Time::US));
}

bool
TcpBbr::IsNextCyclePhase(Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << tcb << rs);
    bool isFullLength = (Simulator::Now() - m_cycleStamp) > m_minRtt;
    if (m_pacingGain == 1)
    {
        return isFullLength;
    }
    // With LinuxArithmetic, the data in flight is counted at the departure
    // time of the next packet, as Linux
    uint32_t inFlight =
        m_linuxArithmetic ? InNetAtEdt(tcb, rs.m_priorInFlight) : rs.m_priorInFlight;
    if (m_pacingGain > 1)
    {
        return isFullLength && (rs.m_bytesLoss > 0 || inFlight >= InFlight(tcb, m_pacingGain));
    }
    else
    {
        return isFullLength || inFlight <= InFlight(tcb, 1);
    }
}

void
TcpBbr::CheckCyclePhase(Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << tcb << rs);
    if (m_state == BbrMode_t::BBR_PROBE_BW && IsNextCyclePhase(tcb, rs))
    {
        AdvanceCyclePhase();
    }
}

void
TcpBbr::CheckFullPipe(const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << rs);
    if (m_roundStart)
    {
        NS_LOG_DEBUG("Round start " << m_roundCount << " app limited " << rs.m_isAppLimited);
    }
    if (m_isPipeFilled || !m_roundStart || rs.m_isAppLimited)
    {
        return;
    }

    NS_LOG_DEBUG("Round " << m_roundCount << " max bw " << m_maxBwFilter.GetBest() << " full bw "
                          << m_fullBandwidth << " count " << m_fullBandwidthCount);
    /* Check if Bottleneck bandwidth is still growing*/
    if (m_maxBwFilter.GetBest().GetBitRate() >= m_fullBandwidth.GetBitRate() * 1.25)
    {
        m_fullBandwidth = m_maxBwFilter.GetBest();
        m_fullBandwidthCount = 0;
        return;
    }

    m_fullBandwidthCount++;
    if (m_fullBandwidthCount >= 3)
    {
        NS_LOG_DEBUG("Pipe filled");
        m_isPipeFilled = true;
    }
}

void
TcpBbr::EnterDrain()
{
    NS_LOG_FUNCTION(this);
    SetBbrState(BbrMode_t::BBR_DRAIN);
    // bbr_drain_gain of Linux is rounded down to 88/256
    m_pacingGain = m_linuxArithmetic ? 88.0 / 256 : 1.0 / m_highGain;
    m_cWndGain = GetHighGain();
}

void
TcpBbr::EnterProbeBW()
{
    NS_LOG_FUNCTION(this);
    SetBbrState(BbrMode_t::BBR_PROBE_BW);
    m_pacingGain = 1;
    m_cWndGain = 2;
    if (m_probeBwEntries < m_probeBwStartPhases.size())
    {
        uint32_t phase = m_probeBwStartPhases[m_probeBwEntries++];
        m_cycleIndex = (phase + GAIN_CYCLE_LENGTH - 1) % GAIN_CYCLE_LENGTH;
    }
    else
    {
        // The phase is drawn among the seven phases but the draining one
        // (bbr_cycle_rand of Linux)
        m_cycleIndex = GAIN_CYCLE_LENGTH - 1 - m_uv->GetInteger(0, 6);
    }
    NS_LOG_DEBUG("Probe bandwidth starting at phase " << (m_cycleIndex + 1) % GAIN_CYCLE_LENGTH);
    AdvanceCyclePhase();
}

void
TcpBbr::CheckDrain(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);
    if (m_state == BbrMode_t::BBR_STARTUP && m_isPipeFilled)
    {
        EnterDrain();
        tcb->m_ssThresh = InFlight(tcb, 1);
    }

    uint32_t inFlight =
        m_linuxArithmetic ? InNetAtEdt(tcb, tcb->m_bytesInFlight) : tcb->m_bytesInFlight.Get();
    if (m_state == BbrMode_t::BBR_DRAIN && inFlight <= InFlight(tcb, 1))
    {
        EnterProbeBW();
    }
}

void
TcpBbr::UpdateRTprop(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);
    m_minRttExpired = JiffiesNow() > (m_minRttStamp + m_minRttFilterLen);
    // With LinuxArithmetic, as Linux, only a sample below the minimum, in
    // whole microseconds, renews it before it expires
    bool lower = m_linuxArithmetic
                     ? tcb->m_lastRtt.Get().GetMicroSeconds() < m_minRtt.Get().GetMicroSeconds()
                     : tcb->m_lastRtt <= m_minRtt;
    if (tcb->m_lastRtt.Get().IsPositive() && (lower || m_minRttExpired))
    {
        NS_LOG_DEBUG("Min RTT " << tcb->m_lastRtt.Get().As(Time::US) << " was "
                                << m_minRtt.Get().As(Time::US) << " expired " << m_minRttExpired);
        m_minRtt = tcb->m_lastRtt;
        m_minRttStamp = JiffiesNow();
    }
}

void
TcpBbr::EnterProbeRTT()
{
    NS_LOG_FUNCTION(this);
    SetBbrState(BbrMode_t::BBR_PROBE_RTT);
    m_pacingGain = 1;
    m_cWndGain = 1;
}

void
TcpBbr::SaveCwnd(Ptr<const TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);
    // Linux tests the state BBR saw on the last ACK (bbr_save_cwnd())
    bool reduced = m_linuxRecovery ? m_prevCaState >= TcpSocketState::CA_RECOVERY
                                   : tcb->m_congState == TcpSocketState::CA_RECOVERY;
    if (!reduced && m_state != BbrMode_t::BBR_PROBE_RTT)
    {
        m_priorCwnd = tcb->m_cWnd;
    }
    else
    {
        m_priorCwnd = std::max(m_priorCwnd, tcb->m_cWnd.Get());
    }
}

void
TcpBbr::RestoreCwnd(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);
    tcb->m_cWnd = std::max(m_priorCwnd, tcb->m_cWnd.Get());
}

void
TcpBbr::ExitProbeRTT()
{
    NS_LOG_FUNCTION(this);
    if (m_isPipeFilled)
    {
        EnterProbeBW();
    }
    else
    {
        EnterStartup();
    }
}

void
TcpBbr::HandleProbeRTT(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);

    m_rateOps->SetAppLimited(tcb->m_bytesInFlight.Get());

    if (m_probeRttDoneStamp.IsZero() && tcb->m_bytesInFlight <= m_minPipeCwnd)
    {
        m_probeRttDoneStamp = JiffiesNow() + m_probeRttDuration;
        m_probeRttRoundDone = false;
        m_nextRoundDelivered = m_delivered;
    }
    else if (!m_probeRttDoneStamp.IsZero())
    {
        if (m_roundStart)
        {
            m_probeRttRoundDone = true;
        }
        if (m_probeRttRoundDone && JiffiesNow() > m_probeRttDoneStamp)
        {
            m_minRttStamp = JiffiesNow();
            RestoreCwnd(tcb);
            ExitProbeRTT();
        }
    }
}

void
TcpBbr::CheckProbeRTT(Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << tcb);
    if (m_state != BbrMode_t::BBR_PROBE_RTT && m_minRttExpired && !m_idleRestart)
    {
        EnterProbeRTT();
        SaveCwnd(tcb);
        m_probeRttDoneStamp = Seconds(0);
    }

    if (m_state == BbrMode_t::BBR_PROBE_RTT)
    {
        HandleProbeRTT(tcb);
    }

    if (rs.m_delivered)
    {
        m_idleRestart = false;
    }
}

void
TcpBbr::SetSendQuantum(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);
    m_sendQuantum = 1 * tcb->m_segmentSize;
}

void
TcpBbr::UpdateTargetCwnd(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);
    m_targetCWnd = m_linuxArithmetic ? LinuxInFlight(tcb, m_cWndGain, true)
                                     : InFlight(tcb, m_cWndGain) + AckAggregationCwnd();
}

uint32_t
TcpBbr::AckAggregationCwnd()
{
    NS_LOG_FUNCTION(this);
    uint32_t maxAggrBytes; // MaxBW * 0.1 secs
    uint32_t aggrCwndBytes = 0;

    if (m_extraAckedGain && m_isPipeFilled)
    {
        maxAggrBytes = m_maxBwFilter.GetBest().GetBitRate() / (10 * 8);
        aggrCwndBytes = m_extraAckedGain * std::max(m_extraAcked[0], m_extraAcked[1]);
        aggrCwndBytes = std::min(aggrCwndBytes, maxAggrBytes);
    }
    return aggrCwndBytes;
}

void
TcpBbr::UpdateAckAggregation(Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << tcb << rs);
    uint32_t expectedAcked;
    uint32_t extraAck;
    uint32_t epochProp;

    if (!m_extraAckedGain || rs.m_ackedSacked <= 0 || rs.m_delivered < 0 ||
        (m_linuxArithmetic && !rs.m_interval.IsStrictlyPositive()))
    {
        return;
    }

    if (m_roundStart)
    {
        m_extraAckedWinRtt = std::min<uint32_t>(31, m_extraAckedWinRtt + 1);
        if (m_extraAckedWinRtt >= m_extraAckedWinRttLength)
        {
            m_extraAckedWinRtt = 0;
            m_extraAckedIdx = m_extraAckedIdx ? 0 : 1;
            m_extraAcked[m_extraAckedIdx] = 0;
        }
    }

    epochProp = Simulator::Now().GetSeconds() - m_ackEpochTime.GetSeconds();
    expectedAcked = m_maxBwFilter.GetBest().GetBitRate() * epochProp / 8;
    if (m_linuxArithmetic)
    {
        // In whole packets over the epoch in microseconds, as Linux
        uint64_t epochUs = (Simulator::Now() - m_ackEpochTime).GetMicroSeconds();
        expectedAcked = static_cast<uint32_t>(
            ((BwUnits(tcb, m_maxBwFilter.GetBest()) * epochUs) >> 24) * tcb->m_segmentSize);
    }

    // As Linux (bbr_ack_epoch_acked_reset_thresh), the epoch is reset past
    // a number of segments, and the data acknowledged in it saturates one
    // segment below
    const uint64_t resetThresh =
        static_cast<uint64_t>(m_ackEpochAckedResetThresh) * tcb->m_segmentSize;
    if (m_ackEpochAcked <= expectedAcked ||
        (static_cast<uint64_t>(m_ackEpochAcked) + rs.m_ackedSacked >= resetThresh))
    {
        m_ackEpochAcked = 0;
        m_ackEpochTime = Simulator::Now();
        expectedAcked = 0;
    }

    m_ackEpochAcked = static_cast<uint32_t>(
        std::min<uint64_t>(static_cast<uint64_t>(m_ackEpochAcked) + rs.m_ackedSacked,
                           resetThresh - tcb->m_segmentSize));
    extraAck = m_ackEpochAcked - expectedAcked;
    extraAck = std::min(extraAck, tcb->m_cWnd.Get());

    if (extraAck > m_extraAcked[m_extraAckedIdx])
    {
        m_extraAcked[m_extraAckedIdx] = extraAck;
    }
}

bool
TcpBbr::ModulateCwndForRecovery(Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << tcb << rs);
    if (rs.m_bytesLoss > 0)
    {
        tcb->m_cWnd =
            std::max((int)tcb->m_cWnd.Get() - (int)rs.m_bytesLoss, (int)tcb->m_segmentSize);
    }

    if (m_packetConservation)
    {
        tcb->m_cWnd = std::max(tcb->m_cWnd.Get(), tcb->m_bytesInFlight.Get() + rs.m_ackedSacked);
        return true;
    }
    return false;
}

bool
TcpBbr::LinuxRecoverOrRestore(Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << tcb << rs);
    const uint32_t mss = tcb->m_segmentSize;
    TcpSocketState::TcpCongState_t state = tcb->m_congState;
    uint32_t cwnd = tcb->m_cWnd.Get();
    if (rs.m_bytesLoss > 0)
    {
        cwnd = cwnd > rs.m_bytesLoss + mss ? cwnd - rs.m_bytesLoss : mss;
    }
    NS_LOG_DEBUG("Linux recovery: state " << state << " previous " << m_prevCaState);
    if (state == TcpSocketState::CA_RECOVERY && m_prevCaState != TcpSocketState::CA_RECOVERY)
    {
        // The first round of the recovery conserves packets, from now
        m_packetConservation = true;
        m_nextRoundDelivered = m_rateOps->GetConnectionRate().m_delivered;
        cwnd = tcb->m_bytesInFlight.Get() + rs.m_ackedSacked;
    }
    else if (m_prevCaState >= TcpSocketState::CA_RECOVERY && state < TcpSocketState::CA_RECOVERY)
    {
        // The window of before the recovery is restored
        cwnd = std::max(cwnd, m_priorCwnd);
        m_packetConservation = false;
    }
    m_prevCaState = state;
    if (m_packetConservation)
    {
        tcb->m_cWnd = std::max(cwnd, tcb->m_bytesInFlight.Get() + rs.m_ackedSacked);
        return true;
    }
    tcb->m_cWnd = cwnd;
    return false;
}

void
TcpBbr::ModulateCwndForProbeRTT(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);
    if (m_state == BbrMode_t::BBR_PROBE_RTT)
    {
        tcb->m_cWnd = std::min(tcb->m_cWnd.Get(), m_minPipeCwnd);
    }
}

void
TcpBbr::SetCwnd(Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << tcb << rs);

    if (!rs.m_ackedSacked)
    {
        goto done;
    }

    NS_LOG_DEBUG("SetCwnd cwnd " << tcb->m_cWnd << " inflight " << tcb->m_bytesInFlight << " acked "
                                 << rs.m_ackedSacked << " loss " << rs.m_bytesLoss
                                 << " conservation " << m_packetConservation << " filled "
                                 << m_isPipeFilled << " bw " << m_maxBwFilter.GetBest()
                                 << " minRtt " << m_minRtt.Get().As(Time::US));
    if (m_linuxRecovery)
    {
        if (LinuxRecoverOrRestore(tcb, rs))
        {
            goto done;
        }
    }
    else if (tcb->m_congState == TcpSocketState::CA_RECOVERY)
    {
        if (ModulateCwndForRecovery(tcb, rs))
        {
            goto done;
        }
    }

    UpdateTargetCwnd(tcb);

    if (m_isPipeFilled)
    {
        tcb->m_cWnd = std::min(tcb->m_cWnd.Get() + (uint32_t)rs.m_ackedSacked, m_targetCWnd);
    }
    else if (tcb->m_cWnd < m_targetCWnd || m_delivered < tcb->m_initialCWnd * tcb->m_segmentSize)
    {
        tcb->m_cWnd = tcb->m_cWnd.Get() + rs.m_ackedSacked;
    }
    tcb->m_cWnd = std::max(tcb->m_cWnd.Get(), m_minPipeCwnd);
    NS_LOG_DEBUG("Congestion window updated. New value:" << tcb->m_cWnd << " target "
                                                         << m_targetCWnd);

done:
    ModulateCwndForProbeRTT(tcb);
}

void
TcpBbr::UpdateRound(Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << tcb << rs);
    if (rs.m_priorDelivered >= m_nextRoundDelivered)
    {
        m_nextRoundDelivered = m_delivered;
        m_roundCount++;
        m_roundStart = true;
        m_packetConservation = false;
    }
    else
    {
        m_roundStart = false;
    }
}

void
TcpBbr::UpdateBottleneckBandwidth(Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << tcb << rs);

    if (rs.m_delivered < 0 || rs.m_interval.IsZero())
    {
        return;
    }

    UpdateRound(tcb, rs);
    NS_LOG_DEBUG("Rate sample " << rs.m_deliveryRate << " delivered " << rs.m_delivered
                                << " interval " << rs.m_interval.As(Time::US) << " app limited "
                                << rs.m_isAppLimited << " max " << m_maxBwFilter.GetBest());

    DataRate bw = rs.m_deliveryRate;
    if (m_linuxArithmetic)
    {
        // bbr_update_bw(): whole packets delivered per microsecond, scaled by
        // 2^24 (BW_UNIT), rounded down
        uint64_t units = (static_cast<uint64_t>(rs.m_delivered / tcb->m_segmentSize) << 24) /
                         static_cast<uint64_t>(rs.m_interval.GetMicroSeconds());
        bw = DataRate(static_cast<double>(units) * tcb->m_segmentSize * 8 * 1e6 / (1ULL << 24));
    }
    if (bw >= m_maxBwFilter.GetBest() || !rs.m_isAppLimited)
    {
        m_maxBwFilter.Update(bw, m_roundCount);
    }
}

void
TcpBbr::UpdateModelAndState(Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << tcb << rs);
    UpdateBottleneckBandwidth(tcb, rs);
    UpdateAckAggregation(tcb, rs);
    CheckCyclePhase(tcb, rs);
    CheckFullPipe(rs);
    CheckDrain(tcb);
    UpdateRTprop(tcb);
    CheckProbeRTT(tcb, rs);
}

void
TcpBbr::UpdateControlParameters(Ptr<TcpSocketState> tcb, const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << tcb << rs);
    SetPacingRate(tcb, m_pacingGain);
    SetSendQuantum(tcb);
    SetCwnd(tcb, rs);
}

void
TcpBbr::SetBbrState(BbrMode_t mode)
{
    NS_LOG_FUNCTION(this << mode);
    NS_LOG_DEBUG(Simulator::Now() << " Changing from " << BbrModeName[m_state] << " to "
                                  << BbrModeName[mode]);
    m_state = mode;
}

uint32_t
TcpBbr::GetBbrState()
{
    NS_LOG_FUNCTION(this);
    return m_state;
}

double
TcpBbr::GetCwndGain()
{
    NS_LOG_FUNCTION(this);
    return m_cWndGain;
}

double
TcpBbr::GetPacingGain()
{
    NS_LOG_FUNCTION(this);
    return m_pacingGain;
}

std::string
TcpBbr::GetName() const
{
    return "TcpBbr";
}

bool
TcpBbr::HasCongControl() const
{
    NS_LOG_FUNCTION(this);
    return true;
}

uint32_t
TcpBbr::GetMinTsoSegs(Ptr<const TcpSocketState> tcb) const
{
    return tcb->m_pacingRate.Get().GetBitRate() < 1200000 ? 1 : 2;
}

void
TcpBbr::CongControl(Ptr<TcpSocketState> tcb,
                    const TcpRateOps::TcpRateConnection& rc,
                    const TcpRateOps::TcpRateSample& rs)
{
    NS_LOG_FUNCTION(this << tcb << rs);
    m_delivered = rc.m_delivered;
    UpdateModelAndState(tcb, rs);
    UpdateControlParameters(tcb, rs);
}

void
TcpBbr::CongestionStateSet(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCongState_t newState)
{
    NS_LOG_FUNCTION(this << tcb << newState);
    if (newState == TcpSocketState::CA_OPEN && !m_isInitialized)
    {
        NS_LOG_DEBUG("CongestionStateSet triggered to CA_OPEN :: " << newState);
        m_minRtt = tcb->m_srtt.Get() != Time::Max() ? tcb->m_srtt.Get() : Time::Max();
        m_minRttStamp = JiffiesNow();
        m_priorCwnd = tcb->m_cWnd;
        tcb->m_ssThresh = tcb->m_initialSsThresh;
        m_targetCWnd = tcb->m_cWnd;
        m_minPipeCwnd = 4 * tcb->m_segmentSize;
        m_sendQuantum = 1 * tcb->m_segmentSize;

        InitRoundCounting();
        InitFullPipe();
        EnterStartup();
        InitPacingRate(tcb);
        m_ackEpochTime = Simulator::Now();
        m_extraAckedWinRtt = 0;
        m_extraAckedIdx = 0;
        m_ackEpochAcked = 0;
        m_extraAcked[0] = 0;
        m_extraAcked[1] = 0;
        m_isInitialized = true;
    }
    else if (newState == TcpSocketState::CA_LOSS)
    {
        NS_LOG_DEBUG("CongestionStateSet triggered to CA_LOSS :: " << newState);
        SaveCwnd(tcb);
        m_roundStart = true;
        if (m_linuxRecovery)
        {
            // bbr_set_state() of Linux
            m_prevCaState = TcpSocketState::CA_LOSS;
            m_fullBandwidth = 0;
        }
    }
    else if (newState == TcpSocketState::CA_RECOVERY && m_linuxRecovery)
    {
        // Linux handles the recovery on the next ACK (bbr_main()), and saves
        // the window when the slow start threshold is asked (bbr_ssthresh())
    }
    else if (newState == TcpSocketState::CA_RECOVERY)
    {
        NS_LOG_DEBUG("CongestionStateSet triggered to CA_RECOVERY :: " << newState);
        SaveCwnd(tcb);
        tcb->m_cWnd =
            tcb->m_bytesInFlight.Get() + std::max(tcb->m_lastAckedSackedBytes, tcb->m_segmentSize);
        m_packetConservation = true;
        // The packet conservation lasts one round trip from now: a round
        // starts (bbr_set_cwnd_to_recover_or_restore() of Linux)
        if (m_rateOps)
        {
            m_nextRoundDelivered = m_rateOps->GetConnectionRate().m_delivered;
        }
    }
}

void
TcpBbr::CwndEvent(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event)
{
    NS_LOG_FUNCTION(this << tcb << event);
    if (event == TcpSocketState::CA_EVENT_COMPLETE_CWR && !m_linuxRecovery)
    {
        NS_LOG_DEBUG("CwndEvent triggered to CA_EVENT_COMPLETE_CWR :: " << event);
        m_packetConservation = false;
        RestoreCwnd(tcb);
    }
    else if (event == TcpSocketState::CA_EVENT_TX_START &&
             m_rateOps->GetConnectionRate().m_appLimited)
    {
        NS_LOG_DEBUG("CwndEvent triggered to CA_EVENT_TX_START :: " << event);
        m_idleRestart = true;
        m_ackEpochTime = Simulator::Now();
        m_ackEpochAcked = 0;
        if (m_state == BbrMode_t::BBR_PROBE_BW)
        {
            SetPacingRate(tcb, 1);
        }
        else if (m_state == BbrMode_t::BBR_PROBE_RTT)
        {
            if (m_probeRttRoundDone && JiffiesNow() > m_probeRttDoneStamp)
            {
                m_minRttStamp = JiffiesNow();
                RestoreCwnd(tcb);
                ExitProbeRTT();
            }
        }
    }
}

uint32_t
TcpBbr::GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
    NS_LOG_FUNCTION(this << tcb << bytesInFlight);
    SaveCwnd(tcb);
    return tcb->m_ssThresh;
}

Ptr<TcpCongestionOps>
TcpBbr::Fork()
{
    return CopyObject<TcpBbr>(this);
}

} // namespace ns3
