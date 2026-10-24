/*
 * Copyright (c) 2016 ResiliNets, ITTC, University of Kansas
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Keerthi Ganta <keerthiganta@ku.edu>
 *         Truc Anh N. Nguyen <annguyen@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  https://resilinets.org/
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 */

#include "tcp-illinois.h"

#include "tcp-socket-state.h"

#include "ns3/log.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TcpIllinois");
NS_OBJECT_ENSURE_REGISTERED(TcpIllinois);

TypeId
TcpIllinois::GetTypeId()
{
    static TypeId tid = TypeId("ns3::TcpIllinois")
                            .SetParent<TcpNewReno>()
                            .AddConstructor<TcpIllinois>()
                            .SetGroupName("Internet")
                            .AddAttribute("AlphaMin",
                                          "Minimum alpha threshold",
                                          DoubleValue(0.3),
                                          MakeDoubleAccessor(&TcpIllinois::m_alphaMin),
                                          MakeDoubleChecker<double>())
                            .AddAttribute("AlphaMax",
                                          "Maximum alpha threshold",
                                          DoubleValue(10.0),
                                          MakeDoubleAccessor(&TcpIllinois::m_alphaMax),
                                          MakeDoubleChecker<double>())
                            .AddAttribute("AlphaBase",
                                          "Alpha base threshold",
                                          DoubleValue(1.0),
                                          MakeDoubleAccessor(&TcpIllinois::m_alphaBase),
                                          MakeDoubleChecker<double>())
                            .AddAttribute("BetaMin",
                                          "Minimum beta threshold",
                                          DoubleValue(0.125),
                                          MakeDoubleAccessor(&TcpIllinois::m_betaMin),
                                          MakeDoubleChecker<double>())
                            .AddAttribute("BetaMax",
                                          "Maximum beta threshold",
                                          DoubleValue(0.5),
                                          MakeDoubleAccessor(&TcpIllinois::m_betaMax),
                                          MakeDoubleChecker<double>())
                            .AddAttribute("BetaBase",
                                          "Beta base threshold",
                                          DoubleValue(0.5),
                                          MakeDoubleAccessor(&TcpIllinois::m_betaBase),
                                          MakeDoubleChecker<double>())
                            .AddAttribute("WinThresh",
                                          "Window threshold",
                                          UintegerValue(15),
                                          MakeUintegerAccessor(&TcpIllinois::m_winThresh),
                                          MakeUintegerChecker<uint32_t>())
                            .AddAttribute("Theta",
                                          "Theta threshold",
                                          UintegerValue(5),
                                          MakeUintegerAccessor(&TcpIllinois::m_theta),
                                          MakeUintegerChecker<uint32_t>())
                            .AddAttribute("LinuxArithmetic",
                                          "Compute alpha, beta and the slow start threshold "
                                          "as the tcp_illinois.c of Linux does: from RTT "
                                          "samples in microseconds, in integer arithmetic "
                                          "(alpha in 1/128, beta in 1/64), updating them once "
                                          "the ACK passes the right edge of the RTT, and "
                                          "keeping alpha below its maximum for Theta RTTs "
                                          "only once the delay rose above d1",
                                          BooleanValue(false),
                                          MakeBooleanAccessor(&TcpIllinois::m_linuxArithmetic),
                                          MakeBooleanChecker());
    return tid;
}

TcpIllinois::TcpIllinois()
    : TcpNewReno(),
      m_sumRtt(Time(0)),
      m_cntRtt(0),
      m_baseRtt(Time::Max()),
      m_maxRtt(Time::Min()),
      m_endSeq(0),
      m_rttAbove(false),
      m_rttLow(0),
      m_alphaMin(0.3),
      m_alphaMax(10.0),
      m_alphaBase(1.0),
      m_alpha(m_alphaMax),
      m_betaMin(0.125),
      m_betaMax(0.5),
      m_betaBase(0.5),
      m_beta(m_betaBase),
      m_winThresh(15),
      m_theta(5),
      m_ackCnt(0)
{
    NS_LOG_FUNCTION(this);
}

TcpIllinois::TcpIllinois(const TcpIllinois& sock)
    : TcpNewReno(sock),
      m_sumRtt(sock.m_sumRtt),
      m_cntRtt(sock.m_cntRtt),
      m_baseRtt(sock.m_baseRtt),
      m_maxRtt(sock.m_maxRtt),
      m_endSeq(sock.m_endSeq),
      m_rttAbove(sock.m_rttAbove),
      m_rttLow(sock.m_rttLow),
      m_alphaMin(sock.m_alphaMin),
      m_alphaMax(sock.m_alphaMax),
      m_alphaBase(sock.m_alphaBase),
      m_alpha(sock.m_alpha),
      m_betaMin(sock.m_betaMin),
      m_betaMax(sock.m_betaMax),
      m_betaBase(sock.m_betaBase),
      m_beta(sock.m_beta),
      m_winThresh(sock.m_winThresh),
      m_theta(sock.m_theta),
      m_ackCnt(sock.m_ackCnt),
      m_linuxCnt(sock.m_linuxCnt),
      m_pktsAcked(sock.m_pktsAcked),
      m_linuxArithmetic(sock.m_linuxArithmetic)
{
    NS_LOG_FUNCTION(this);
}

TcpIllinois::~TcpIllinois()
{
    NS_LOG_FUNCTION(this);
}

void
TcpIllinois::RecalcParam(uint32_t cWnd)
{
    NS_LOG_FUNCTION(this << cWnd);

    if (cWnd < m_winThresh)
    {
        NS_LOG_INFO("cWnd < winThresh, set alpha & beta to base values");

        m_alpha = m_alphaBase;
        m_beta = m_betaBase;
    }
    else if (m_cntRtt > 0 && m_linuxArithmetic)
    {
        // update_params(), with the delays in microseconds
        auto base = static_cast<uint32_t>(m_baseRtt.GetMicroSeconds());
        auto dm = static_cast<uint32_t>(m_maxRtt.GetMicroSeconds()) - base;
        auto da = static_cast<uint32_t>(m_sumRtt.GetMicroSeconds() / m_cntRtt) - base;
        m_alpha = LinuxAlpha(da, dm) / 128.0;
        m_beta = LinuxBeta(da, dm) / 64.0;
        NS_LOG_INFO("Updated to dm = " << dm << "us da = " << da << "us alpha = " << m_alpha
                                       << " beta = " << m_beta);
    }
    else if (m_cntRtt > 0)
    {
        double dm = CalculateMaxDelay().GetSeconds() * 1000;
        double da = CalculateAvgDelay().GetSeconds() * 1000;

        NS_LOG_INFO("Updated to dm = " << dm << " da = " << da);

        CalculateAlpha(da, dm);
        CalculateBeta(da, dm);
    }
}

void
TcpIllinois::CongestionStateSet(Ptr<TcpSocketState> tcb,
                                const TcpSocketState::TcpCongState_t newState)
{
    NS_LOG_FUNCTION(this << tcb << newState);
    TcpNewReno::CongestionStateSet(tcb, newState);

    // As tp->snd_cwnd_cnt, which tcp_init_cwnd_reduction() and
    // tcp_enter_loss() clear in Linux
    if (newState == TcpSocketState::CA_RECOVERY || newState == TcpSocketState::CA_CWR ||
        newState == TcpSocketState::CA_LOSS)
    {
        m_linuxCnt = 0;
    }

    if (newState == TcpSocketState::CA_LOSS)
    {
        m_alpha = m_alphaBase;
        m_beta = m_betaBase;
        m_rttLow = 0;
        m_rttAbove = false;
        Reset(m_linuxArithmetic ? tcb->m_highTxMark.Get() : tcb->m_nextTxSequence.Get());
    }
}

void
TcpIllinois::IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << segmentsAcked);

    // Linux updates the parameters once the ACK is after the right edge of
    // the RTT, snd_nxt when it was set
    if (m_linuxArithmetic ? tcb->m_lastAckedSeq > m_endSeq : tcb->m_lastAckedSeq >= m_endSeq)
    {
        RecalcParam(tcb->GetCwndInSegments());
        Reset(m_linuxArithmetic ? tcb->m_highTxMark.Get() : tcb->m_nextTxSequence.Get());
    }

    if (tcb->m_cWnd < tcb->m_ssThresh)
    {
        TcpNewReno::SlowStart(tcb, segmentsAcked);
        NS_LOG_INFO("In SlowStart, updated to cwnd " << tcb->m_cWnd << " ssthresh "
                                                     << tcb->m_ssThresh);
    }
    else
    {
        uint32_t segCwnd = tcb->GetCwndInSegments();
        uint32_t oldCwnd = segCwnd;

        if (CountsAckedSegments())
        {
            // Linux tcp_illinois_cong_avoid(): the count is the segments the
            // ACK removed from the retransmission queue (ca->acked, set by
            // pkts_acked), including those SACKed before, as on the ACK which
            // ends a recovery; alpha is in 1/128, and the increase leaves no
            // remainder
            m_linuxCnt += m_pktsAcked;
            m_pktsAcked = 1;
            auto alpha = static_cast<uint64_t>(m_alpha * 128);
            uint64_t delta = (static_cast<uint64_t>(m_linuxCnt) * alpha) >> 7;
            if (delta >= segCwnd)
            {
                segCwnd += static_cast<uint32_t>(delta / segCwnd);
                m_linuxCnt = 0;
            }
        }
        else
        {
            // alpha / cwnd segments per acknowledged segment, the fractions
            // of segment carried to the next ACKs
            m_ackCnt += segmentsAcked * m_alpha;
            while (m_ackCnt >= segCwnd)
            {
                m_ackCnt -= segCwnd;
                segCwnd += 1;
            }
        }

        if (segCwnd != oldCwnd)
        {
            tcb->m_cWnd = segCwnd * tcb->m_segmentSize;
            NS_LOG_INFO("In CongAvoid, updated to cwnd " << tcb->m_cWnd << " ssthresh "
                                                         << tcb->m_ssThresh);
        }
    }
}

void
TcpIllinois::PktsAcked(Ptr<TcpSocketState> tcb, uint32_t packetsAcked, const Time& rtt)
{
    NS_LOG_FUNCTION(this << tcb << packetsAcked << rtt);

    m_pktsAcked = packetsAcked;

    if (rtt.IsZero())
    {
        return;
    }

    Time sample = rtt;
    if (m_linuxArithmetic)
    {
        // tcp_illinois_acked(): microseconds, at most RTT_MAX (U32_MAX / ALPHA_MAX)
        sample = MicroSeconds(std::min<int64_t>(rtt.GetMicroSeconds(), UINT32_MAX / (10 * 128)));
    }

    // Keep track of minimum RTT
    m_baseRtt = std::min(m_baseRtt, sample);

    // Keep track of maximum RTT
    m_maxRtt = std::max(sample, m_maxRtt);

    ++m_cntRtt;
    m_sumRtt += sample;

    NS_LOG_INFO("Updated baseRtt = " << m_baseRtt << " maxRtt = " << m_maxRtt
                                     << " cntRtt = " << m_cntRtt << " sumRtt = " << m_sumRtt);
}

uint32_t
TcpIllinois::GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
    NS_LOG_FUNCTION(this << tcb << bytesInFlight);

    // The window, in segments, is reduced by beta, as in the Illinois paper
    // and in Linux, not the data in flight, which the lost and SACKed
    // segments reduce
    uint32_t segCwnd = tcb->GetCwndInSegments();
    uint32_t ssThresh;
    if (m_linuxArithmetic)
    {
        // tcp_illinois_ssthresh(): the decrease is rounded down
        auto decr = static_cast<uint32_t>(
            (static_cast<uint64_t>(segCwnd) * static_cast<uint32_t>(m_beta * 64)) >> 6);
        ssThresh = std::max(segCwnd - decr, 2U);
    }
    else
    {
        ssThresh = static_cast<uint32_t>(std::max(2.0, (1.0 - m_beta) * segCwnd));
    }

    NS_LOG_DEBUG("Calculated ssThresh (in segments) = " << ssThresh);

    return ssThresh * tcb->m_segmentSize;
}

void
TcpIllinois::CalculateAlpha(double da, double dm)
{
    NS_LOG_FUNCTION(this << da << dm);

    double d1 = dm / 100;

    if (da <= d1)
    {
        NS_LOG_INFO("da <= d1");

        if (!m_rttAbove)
        { // In case we can't get out of this low delay zone, we use alphaMax
            m_alpha = m_alphaMax;
        }
        if (++m_rttLow >= m_theta)
        {
            /*
             * da needs to stay below d1 for theta times RTT amount of time
             * before we can increase alpha to alphaMax
             */
            NS_LOG_INFO("da stays below d1 for theta times RTT amount of time, "
                        "increase alpha to alphaMax");

            m_rttLow = 0;
            m_rttAbove = false;
            m_alpha = m_alphaMax;
        }
    }
    else
    {
        NS_LOG_INFO("da > d1");

        m_rttAbove = true;
        /*
         * alpha = k1 / (k2 + da), where
         * k1 = ((dm - d1) * alphaMin * alphaMax) / (alphaMax - alphaMin)
         * k2 = (((dm - d1) * alphaMin) / (alphaMax - alphaMin)) - d1
         */
        dm -= d1;
        da -= d1;
        m_alpha = (dm * m_alphaMax) / (dm + (da * (m_alphaMax - m_alphaMin)) / m_alphaMin);
    }

    NS_LOG_INFO("Updated to alpha = " << m_alpha);
}

void
TcpIllinois::CalculateBeta(double da, double dm)
{
    NS_LOG_FUNCTION(this << da << dm);

    double d2;
    double d3;

    d2 = dm / 10;
    d3 = (8 * dm) / 10;

    if (da <= d2)
    {
        NS_LOG_INFO("da <= d2");

        m_beta = m_betaMin;
    }

    else if (da > d2 && da < d3)
    {
        NS_LOG_INFO("da > d2 && da < d3");

        /*
         * beta = k3 + k4 * da, where
         * k3 = (betaMin * d3 - betaMax * d2) / (d3 - d2)
         * k4 = (betaMax - betaMin) / (d3 - d2)
         */
        m_beta = (m_betaMin * d3 - m_betaMax * d2 + (m_betaMax - m_betaMin) * da) / (d3 - d2);
    }

    else if (da >= d3 || d3 <= d2)
    {
        NS_LOG_INFO("da >= d3 || d3 <= d2");

        m_beta = m_betaMax;
    }

    NS_LOG_INFO("Updated to beta = " << m_beta);
}

uint32_t
TcpIllinois::LinuxAlpha(uint32_t da, uint32_t dm)
{
    NS_LOG_FUNCTION(this << da << dm);

    const auto alphaMax = static_cast<uint32_t>(m_alphaMax * 128);
    const auto alphaMin = static_cast<uint32_t>(m_alphaMin * 128);
    const auto alpha = static_cast<uint32_t>(m_alpha * 128);
    uint32_t d1 = dm / 100;

    if (da <= d1)
    {
        // The count of low delay RTTs starts only once the delay was above d1
        if (!m_rttAbove)
        {
            return alphaMax;
        }
        if (++m_rttLow < m_theta)
        {
            return alpha;
        }
        m_rttLow = 0;
        m_rttAbove = false;
        return alphaMax;
    }

    m_rttAbove = true;
    dm -= d1;
    da -= d1;
    return (dm * alphaMax) / (dm + (da * (alphaMax - alphaMin)) / alphaMin);
}

uint32_t
TcpIllinois::LinuxBeta(uint32_t da, uint32_t dm) const
{
    NS_LOG_FUNCTION(this << da << dm);

    const auto betaMin = static_cast<uint32_t>(m_betaMin * 64);
    const auto betaMax = static_cast<uint32_t>(m_betaMax * 64);
    uint32_t d2 = dm / 10;
    if (da <= d2)
    {
        return betaMin;
    }
    uint32_t d3 = (8 * dm) / 10;
    if (da >= d3 || d3 <= d2)
    {
        return betaMax;
    }
    return (betaMin * d3 - betaMax * d2 + (betaMax - betaMin) * da) / (d3 - d2);
}

Time
TcpIllinois::CalculateAvgDelay() const
{
    NS_LOG_FUNCTION(this);

    return (m_sumRtt / m_cntRtt - m_baseRtt);
}

Time
TcpIllinois::CalculateMaxDelay() const
{
    NS_LOG_FUNCTION(this);

    return (m_maxRtt - m_baseRtt);
}

void
TcpIllinois::Init(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);
    Reset(tcb->m_nextTxSequence);
}

void
TcpIllinois::Reset(const SequenceNumber32& nextTxSequence)
{
    NS_LOG_FUNCTION(this << nextTxSequence);

    m_endSeq = nextTxSequence;
    m_cntRtt = 0;
    m_sumRtt = Time(0);
}

Ptr<TcpCongestionOps>
TcpIllinois::Fork()
{
    NS_LOG_FUNCTION(this);

    return CopyObject<TcpIllinois>(this);
}

std::string
TcpIllinois::GetName() const
{
    NS_LOG_FUNCTION(this);

    return "TcpIllinois";
}

} // namespace ns3
