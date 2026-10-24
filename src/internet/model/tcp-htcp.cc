/*
 * Copyright (c) 2015 ResiliNets, ITTC, University of Kansas
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * by: Amir Modarresi <amodarresi@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  https://resilinets.org/
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 */

#include "tcp-htcp.h"

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/uinteger.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TcpHtcp");

NS_OBJECT_ENSURE_REGISTERED(TcpHtcp);

TypeId
TcpHtcp::GetTypeId()
{
    static TypeId tid = TypeId("ns3::TcpHtcp")
                            .SetParent<TcpNewReno>()
                            .AddConstructor<TcpHtcp>()
                            .SetGroupName("Internet")
                            .AddAttribute("DefaultBackoff",
                                          "The default AIMD backoff factor",
                                          DoubleValue(0.5),
                                          MakeDoubleAccessor(&TcpHtcp::m_defaultBackoff),
                                          MakeDoubleChecker<double>(0, 1))
                            .AddAttribute("ThroughputRatio",
                                          "Threshold value for updating beta",
                                          DoubleValue(0.2),
                                          MakeDoubleAccessor(&TcpHtcp::m_throughputRatio),
                                          MakeDoubleChecker<double>())
                            .AddAttribute("DeltaL",
                                          "Delta_L parameter in increase function",
                                          TimeValue(Seconds(1)),
                                          MakeTimeAccessor(&TcpHtcp::m_deltaL),
                                          MakeTimeChecker())
                            .AddAttribute("RttScaling",
                                          "Scale the increase function by the ratio of the "
                                          "minimum RTT to a reference RTT of 100 ms, bounded "
                                          "between 0.5 and 10, as Linux does by default",
                                          BooleanValue(false),
                                          MakeBooleanAccessor(&TcpHtcp::m_rttScaling),
                                          MakeBooleanChecker())
                            .AddAttribute("LinuxHz",
                                          "Rate of the jiffies (CONFIG_HZ) of Linux. When "
                                          "nonzero, the window follows the Linux tcp_htcp.c, "
                                          "with its fixed point alpha and beta, its RTTs and "
                                          "times in jiffies at this rate, its throughput "
                                          "estimate for the adaptive backoff and its per-ACK "
                                          "increase; zero (the default) follows the H-TCP "
                                          "paper. RttScaling applies as well",
                                          UintegerValue(0),
                                          MakeUintegerAccessor(&TcpHtcp::m_linuxHz),
                                          MakeUintegerChecker<uint32_t>());
    return tid;
}

std::string
TcpHtcp::GetName() const
{
    return "TcpHtcp";
}

TcpHtcp::TcpHtcp()
    : TcpNewReno(),
      m_alpha(0),
      m_beta(0),
      m_delta(0),
      m_lastCon(0),
      m_minRtt(Time::Max()),
      m_maxRtt(Time::Min()),
      m_throughput(0),
      m_lastThroughput(0),
      m_dataSent(0)
{
    NS_LOG_FUNCTION(this);
}

TcpHtcp::TcpHtcp(const TcpHtcp& sock)
    : TcpNewReno(sock),
      m_alpha(sock.m_alpha),
      m_beta(sock.m_beta),
      m_defaultBackoff(sock.m_defaultBackoff),
      m_throughputRatio(sock.m_throughputRatio),
      m_delta(sock.m_delta),
      m_deltaL(sock.m_deltaL),
      m_rttScaling(sock.m_rttScaling),
      m_lowestRtt(sock.m_lowestRtt),
      m_lastCon(sock.m_lastCon),
      m_minRtt(sock.m_minRtt),
      m_maxRtt(sock.m_maxRtt),
      m_throughput(sock.m_throughput),
      m_lastThroughput(sock.m_lastThroughput),
      m_dataSent(sock.m_dataSent),
      m_linuxHz(sock.m_linuxHz),
      m_lx(sock.m_lx)
{
    NS_LOG_FUNCTION(this);
}

TcpHtcp::~TcpHtcp()
{
    NS_LOG_FUNCTION(this);
}

Ptr<TcpCongestionOps>
TcpHtcp::Fork()
{
    NS_LOG_FUNCTION(this);
    return CopyObject<TcpHtcp>(this);
}

uint32_t
TcpHtcp::Jiffies() const
{
    return static_cast<uint32_t>(Simulator::Now().GetMicroSeconds() * m_linuxHz / 1000000);
}

void
TcpHtcp::Init(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);
    TcpNewReno::Init(tcb);
    if (m_linuxHz > 0)
    {
        m_lx = LinuxState();
        m_lx.lastCong = Jiffies();
    }
}

void
TcpHtcp::LinuxReset(uint32_t jiffies)
{
    m_lx.undoLastCong = m_lx.lastCong;
    m_lx.undoMaxRtt = m_lx.maxRtt;
    m_lx.undoOldMaxB = m_lx.oldMaxB;
    m_lx.lastCong = jiffies;
}

void
TcpHtcp::LinuxApplyPendingReset()
{
    if (m_lx.resetPending && Simulator::Now() != m_lx.resetTime)
    {
        LinuxReset(m_lx.resetJiffies);
        m_lx.resetPending = false;
    }
}

void
TcpHtcp::LinuxAlphaUpdate()
{
    // htcp_alpha_update()
    const uint32_t hz = m_linuxHz;
    uint32_t factor = 1;
    uint32_t diff = Jiffies() - m_lx.lastCong;
    if (diff > hz)
    {
        diff -= hz;
        factor = 1 + (10 * diff + ((diff / 2) * (diff / 2) / hz)) / hz;
    }
    if (m_rttScaling && m_lx.minRtt != 0)
    {
        uint32_t scale = std::clamp((hz << 3) / (10 * m_lx.minRtt), 1U << 2, 10U << 3);
        factor = std::max((factor << 3) / scale, 1U);
    }
    m_lx.alpha = 2 * factor * ((1 << 7) - m_lx.beta);
    if (m_lx.alpha == 0)
    {
        m_lx.alpha = 1 << 7;
    }
    NS_LOG_DEBUG("alpha " << m_lx.alpha << " beta " << +m_lx.beta);
}

void
TcpHtcp::IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);
    if (m_linuxHz == 0)
    {
        TcpNewReno::IncreaseWindow(tcb, segmentsAcked);
        return;
    }
    LinuxApplyPendingReset();
    uint32_t cwnd = tcb->GetCwndInSegments();
    if (tcb->m_cWnd < tcb->m_ssThresh)
    {
        // tcp_slow_start(), the segments beyond ssThresh being dropped
        cwnd = std::min(cwnd + segmentsAcked, tcb->m_ssThresh.Get() / tcb->m_segmentSize);
    }
    else
    {
        // htcp_cong_avoid(): one segment once cwnd / alpha ACKed segments
        if (((m_lx.cwndCnt * m_lx.alpha) >> 7) >= cwnd)
        {
            ++cwnd;
            m_lx.cwndCnt = 0;
            LinuxAlphaUpdate();
        }
        else
        {
            m_lx.cwndCnt += m_lx.pktsAcked;
        }
        m_lx.pktsAcked = 1;
    }
    tcb->m_cWnd = cwnd * tcb->m_segmentSize;
}

void
TcpHtcp::CongestionStateSet(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCongState_t newState)
{
    NS_LOG_FUNCTION(this << tcb << newState);
    TcpNewReno::CongestionStateSet(tcb, newState);
    if (m_linuxHz == 0)
    {
        return;
    }
    LinuxApplyPendingReset();
    // htcp_state()
    if (newState == TcpSocketState::CA_OPEN)
    {
        if (m_lx.undoLastCong != 0)
        {
            m_lx.lastCong = Jiffies();
            m_lx.undoLastCong = 0;
        }
    }
    else if (newState == TcpSocketState::CA_CWR || newState == TcpSocketState::CA_RECOVERY ||
             newState == TcpSocketState::CA_LOSS)
    {
        m_lx.cwndCnt = 0;
        // Linux computes the threshold first: htcp_reset() waits for it
        // when it is not computed yet
        if (m_lx.ssThreshTime == Simulator::Now())
        {
            LinuxReset(Jiffies());
        }
        else
        {
            m_lx.resetPending = true;
            m_lx.resetJiffies = Jiffies();
            m_lx.resetTime = Simulator::Now();
        }
    }
}

void
TcpHtcp::CongestionAvoidance(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);
    // cwnd grows by alpha segments per RTT, i.e., by alpha / cwnd segments
    // per ACK, or per acknowledged segment as Linux does
    uint32_t increments = CountsAckedSegments() ? segmentsAcked : std::min(segmentsAcked, 1U);
    for (uint32_t i = 0; i < increments; ++i)
    {
        double adder = m_alpha * tcb->m_segmentSize * tcb->m_segmentSize / tcb->m_cWnd;
        adder = std::max(1.0, adder);
        tcb->m_cWnd += static_cast<uint32_t>(adder);
    }
    if (increments > 0)
    {
        NS_LOG_INFO("In CongAvoid, updated to cwnd " << tcb->m_cWnd << " ssthresh "
                                                     << tcb->m_ssThresh);
    }
}

void
TcpHtcp::UpdateAlpha()
{
    NS_LOG_FUNCTION(this);

    m_delta = (Simulator::Now() - m_lastCon);
    if (m_delta <= m_deltaL)
    {
        m_alpha = 1;
    }
    else
    {
        Time diff = m_delta - m_deltaL;
        double diffSec = diff.GetSeconds();
        // alpha=1+10(Delta-Delta_L)+[0.5(Delta-Delta_L)]^2  (seconds)
        // from Leith and Shorten H-TCP paper
        m_alpha = (1 + 10 * diffSec + 0.25 * (diffSec * diffSec));
    }
    if (m_rttScaling && m_lowestRtt != Time::Max())
    {
        // Linux htcp_alpha_update(): the factor is scaled by RTT / 100 ms,
        // with the ratio bounded to [0.5, 10], and is at least 1
        double ratio = std::clamp(m_lowestRtt.GetSeconds() / 0.1, 0.5, 10.0);
        m_alpha = std::max(m_alpha * ratio, 1.0);
    }
    m_alpha = 2 * (1 - m_beta) * m_alpha;
    if (m_alpha < 1)
    {
        m_alpha = 1;
    }
    NS_LOG_DEBUG("Updated m_alpha: " << m_alpha);
}

void
TcpHtcp::UpdateBeta()
{
    NS_LOG_FUNCTION(this);

    // Default value for m_beta
    m_beta = m_defaultBackoff;

    if (m_lastThroughput > 0)
    {
        double diff = std::abs(static_cast<double>(m_throughput) - m_lastThroughput);
        if (diff / m_lastThroughput <= m_throughputRatio)
        {
            // The adaptive backoff is bounded between 0.5 and 0.8 (H-TCP
            // paper, Linux BETA_MIN and BETA_MAX)
            m_beta = std::clamp(m_minRtt.GetDouble() / m_maxRtt.GetDouble(), 0.5, 0.8);
        }
    }
    NS_LOG_DEBUG("Updated m_beta: " << m_beta);
}

uint32_t
TcpHtcp::GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
    NS_LOG_FUNCTION(this << tcb << bytesInFlight);

    if (m_linuxHz > 0)
    {
        // htcp_param_update(): htcp_beta_update() and htcp_alpha_update()
        const uint32_t maxB = m_lx.maxB;
        const uint32_t oldMaxB = m_lx.oldMaxB;
        m_lx.oldMaxB = maxB;
        // between(5 * maxB, 4 * oldMaxB, 6 * oldMaxB)
        if (6 * oldMaxB - 4 * oldMaxB < 5 * maxB - 4 * oldMaxB)
        {
            m_lx.beta = 1 << 6;
            m_lx.modeswitch = false;
        }
        else if (m_lx.modeswitch && m_lx.minRtt > (10 * m_linuxHz + 999) / 1000 && m_lx.maxRtt != 0)
        {
            m_lx.beta = static_cast<uint8_t>((m_lx.minRtt << 7) / m_lx.maxRtt);
            m_lx.beta = std::clamp<uint8_t>(m_lx.beta, 1 << 6, 102);
        }
        else
        {
            m_lx.beta = 1 << 6;
            m_lx.modeswitch = true;
        }
        LinuxAlphaUpdate();
        if (m_lx.minRtt > 0 && m_lx.maxRtt > m_lx.minRtt)
        {
            m_lx.maxRtt = m_lx.minRtt + ((m_lx.maxRtt - m_lx.minRtt) * 95) / 100;
        }
        uint32_t ssThresh = std::max((tcb->GetCwndInSegments() * m_lx.beta) >> 7, 2U);
        if (m_lx.resetPending)
        {
            LinuxReset(m_lx.resetJiffies);
            m_lx.resetPending = false;
        }
        m_lx.ssThreshTime = Simulator::Now();
        return ssThresh * tcb->m_segmentSize;
    }

    m_lastCon = Simulator::Now();

    UpdateBeta();
    UpdateAlpha();

    uint32_t segWin = 2 * tcb->m_segmentSize;
    auto bFlight = static_cast<uint32_t>(bytesInFlight * m_beta);
    uint32_t ssThresh = std::max(segWin, bFlight);
    m_minRtt = Time::Max();
    m_maxRtt = Time::Min();
    m_lastThroughput = m_throughput;
    m_throughput = 0;
    m_dataSent = 0;
    NS_LOG_DEBUG(this << " ssThresh: " << ssThresh << " m_beta: " << m_beta);
    return ssThresh;
}

void
TcpHtcp::PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked << rtt);
    NS_LOG_DEBUG("TcpSocketState: " << tcb->m_congState);
    if (m_linuxHz > 0)
    {
        // measure_achieved_throughput(), with the segments cumulatively
        // acknowledged (see TcpSocketBase::LinuxPktsAcked)
        LinuxApplyPendingReset();
        const auto state = tcb->m_congState.Get();
        const uint32_t pktsAcked = segmentsAcked;
        const uint32_t now = Jiffies();
        if (state == TcpSocketState::CA_OPEN)
        {
            m_lx.pktsAcked = static_cast<uint16_t>(pktsAcked);
        }
        if (rtt.IsStrictlyPositive())
        {
            // measure_rtt(), usecs_to_jiffies() rounding up
            auto srtt =
                static_cast<uint32_t>((rtt.GetMicroSeconds() * m_linuxHz + 999999) / 1000000);
            if (m_lx.minRtt > srtt || m_lx.minRtt == 0)
            {
                m_lx.minRtt = srtt;
            }
            if (state == TcpSocketState::CA_OPEN)
            {
                m_lx.maxRtt = std::max(m_lx.maxRtt, m_lx.minRtt);
                if (m_lx.maxRtt < srtt && srtt <= m_lx.maxRtt + (20 * m_linuxHz + 999) / 1000)
                {
                    m_lx.maxRtt = srtt;
                }
            }
        }
        if (state != TcpSocketState::CA_OPEN && state != TcpSocketState::CA_DISORDER)
        {
            m_lx.packetCount = 0;
            m_lx.lastTime = now;
            return;
        }
        m_lx.packetCount += pktsAcked;
        uint32_t alphaSegs = std::max(m_lx.alpha >> 7, 1U);
        if (m_lx.packetCount >= tcb->GetCwndInSegments() - alphaSegs &&
            now - m_lx.lastTime >= m_lx.minRtt && m_lx.minRtt > 0)
        {
            uint32_t curBi = m_lx.packetCount * m_linuxHz / (now - m_lx.lastTime);
            if ((now - m_lx.lastCong) / m_lx.minRtt <= 3)
            {
                m_lx.minB = m_lx.maxB = m_lx.bi = curBi;
            }
            else
            {
                m_lx.bi = (3 * m_lx.bi + curBi) / 4;
                m_lx.maxB = std::max(m_lx.maxB, m_lx.bi);
                m_lx.minB = std::min(m_lx.minB, m_lx.maxB);
            }
            m_lx.packetCount = 0;
            m_lx.lastTime = now;
        }
        return;
    }
    if (tcb->m_congState == TcpSocketState::CA_OPEN)
    {
        m_dataSent += segmentsAcked * tcb->m_segmentSize;
    }

    m_throughput = static_cast<uint32_t>(m_dataSent /
                                         (Simulator::Now().GetSeconds() - m_lastCon.GetSeconds()));

    if (rtt.IsStrictlyPositive())
    {
        m_lowestRtt = std::min(m_lowestRtt, rtt);
    }
    UpdateAlpha();
    // An ACK without RTT sample does not change the RTT range (Linux
    // measure_achieved_throughput())
    if (!rtt.IsStrictlyPositive())
    {
        return;
    }
    if (rtt < m_minRtt)
    {
        m_minRtt = rtt;
        NS_LOG_DEBUG("Updated m_minRtt=" << m_minRtt);
    }
    if (rtt > m_maxRtt)
    {
        m_maxRtt = rtt;
        NS_LOG_DEBUG("Updated m_maxRtt=" << m_maxRtt);
    }
}

} // namespace ns3
