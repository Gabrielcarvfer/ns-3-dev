/*
 * Copyright (c) 2014 Natale Patriciello <natale.patriciello@gmail.com>
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#include "tcp-hybla.h"

#include "tcp-socket-state.h"

#include "ns3/boolean.h"
#include "ns3/log.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TcpHybla");
NS_OBJECT_ENSURE_REGISTERED(TcpHybla);

TypeId
TcpHybla::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::TcpHybla")
            .SetParent<TcpNewReno>()
            .AddConstructor<TcpHybla>()
            .SetGroupName("Internet")
            .AddAttribute("RRTT",
                          "Reference RTT",
                          TimeValue(MilliSeconds(25)),
                          MakeTimeAccessor(&TcpHybla::m_rRtt),
                          MakeTimeChecker())
            .AddAttribute("RenoOutsideOpen",
                          "Apply the Hybla increase only in the Open state, and grow the "
                          "window as NewReno in the other states, as Linux does",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpHybla::m_renoOutsideOpen),
                          MakeBooleanChecker())
            .AddAttribute("RhoInitialWindow",
                          "Set the initial window to the integer part of rho, computed from "
                          "the RTT of the handshake, as Linux does",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpHybla::m_rhoInitialWindow),
                          MakeBooleanChecker())
            .AddAttribute("FixedPoint",
                          "Compute rho in 1/8 and the increments in 1/128 of a segment, "
                          "truncating them, and the slow start threshold in whole segments "
                          "from the window (max(cwnd >> 1, 2), tcp_reno_ssthresh()), as the "
                          "fixed-point arithmetic of Linux does",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpHybla::m_fixedPoint),
                          MakeBooleanChecker())
            .AddTraceSource("Rho",
                            "Rho parameter of Hybla",
                            MakeTraceSourceAccessor(&TcpHybla::m_rho),
                            "ns3::TracedValueCallback::Double");
    return tid;
}

TcpHybla::TcpHybla()
    : TcpNewReno(),
      m_rho(1.0),
      m_cWndCnt(0),
      m_rhoRtt(Time::Max())
{
    NS_LOG_FUNCTION(this);
}

TcpHybla::TcpHybla(const TcpHybla& sock)
    : TcpNewReno(sock),
      m_rho(sock.m_rho),
      m_rRtt(sock.m_rRtt),
      m_cWndCnt(sock.m_cWndCnt),
      m_rhoRtt(sock.m_rhoRtt),
      m_renoOutsideOpen(sock.m_renoOutsideOpen),
      m_rhoInitialWindow(sock.m_rhoInitialWindow),
      m_fixedPoint(sock.m_fixedPoint),
      m_rho3ls(sock.m_rho3ls),
      m_ackCnt(sock.m_ackCnt),
      m_open(sock.m_open),
      m_initialWindowSet(sock.m_initialWindowSet)
{
    NS_LOG_FUNCTION(this);
}

TcpHybla::~TcpHybla()
{
    NS_LOG_FUNCTION(this);
}

void
TcpHybla::RecalcParam(const Ptr<TcpSocketState>& tcb)
{
    NS_LOG_FUNCTION(this);

    if (m_fixedPoint)
    {
        // Linux hybla_recalc_param(): rho in 1/8, truncated, at least 1
        m_rho3ls = std::max<uint64_t>(tcb->m_minRtt.GetMicroSeconds() * 8 /
                                          std::max<int64_t>(m_rRtt.GetMicroSeconds(), 1),
                                      8);
        m_rho = m_rho3ls / 8.0;
    }
    else
    {
        m_rho = std::max(tcb->m_minRtt.GetSeconds() / m_rRtt.GetSeconds(), 1.0);
    }

    NS_ASSERT(m_rho > 0.0);
    NS_LOG_DEBUG("Calculated rho=" << m_rho);
}

void
TcpHybla::PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked << rtt);

    // As in Linux, rho follows the minimum of the smoothed RTT, not the
    // minimum of the RTT samples
    if (tcb->m_minRtt != m_rhoRtt)
    {
        m_rhoRtt = tcb->m_minRtt;
        RecalcParam(tcb);
    }
}

void
TcpHybla::CongestionStateSet(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCongState_t newState)
{
    NS_LOG_FUNCTION(this << tcb << newState);
    TcpNewReno::CongestionStateSet(tcb, newState);

    m_open = (newState == TcpSocketState::CA_OPEN);
    if (newState == TcpSocketState::CA_RECOVERY || newState == TcpSocketState::CA_CWR ||
        newState == TcpSocketState::CA_LOSS)
    {
        // As tp->snd_cwnd_cnt in Linux; the fraction of segment carried
        // between the increases is kept
        m_ackCnt = 0;
    }
}

void
TcpHybla::CwndEvent(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event)
{
    NS_LOG_FUNCTION(this << tcb << event);

    // Linux hybla_init() runs once the connection is established, with the
    // smoothed RTT of the handshake: the first transmission is the first
    // occasion the congestion control is given after it
    if (m_rhoInitialWindow && !m_initialWindowSet && event == TcpSocketState::CA_EVENT_TX_START &&
        tcb->m_minRtt != Time::Max())
    {
        m_initialWindowSet = true;
        m_rhoRtt = tcb->m_minRtt;
        RecalcParam(tcb);
        tcb->m_cWnd = static_cast<uint32_t>(m_rho) * tcb->m_segmentSize;
        tcb->m_cWndInfl = tcb->m_cWnd;
        NS_LOG_INFO("Initial window set to " << tcb->m_cWnd << " (rho=" << m_rho << ")");
    }
}

void
TcpHybla::IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);

    if (m_renoOutsideOpen && !m_open)
    {
        // Linux hybla_cong_avoid() calls tcp_reno_cong_avoid() outside the
        // Open state (hybla_en false)
        if (tcb->m_cWnd < tcb->m_ssThresh)
        {
            segmentsAcked = TcpNewReno::SlowStart(tcb, segmentsAcked);
        }
        if (tcb->m_cWnd >= tcb->m_ssThresh)
        {
            TcpNewReno::CongestionAvoidance(tcb, segmentsAcked);
        }
        return;
    }

    // An ACK grows the window either in slow start or in congestion
    // avoidance, never both
    if (tcb->m_cWnd < tcb->m_ssThresh)
    {
        SlowStart(tcb, segmentsAcked);
    }
    else
    {
        CongestionAvoidance(tcb, segmentsAcked);
    }
}

void
TcpHybla::AddFixedPointIncrement(Ptr<TcpSocketState> tcb, uint32_t increment)
{
    NS_LOG_FUNCTION(this << tcb << increment);

    // Linux hybla_cong_avoid(): the whole segments are added at once, the
    // fractions (in 1/128) accumulate until they make a segment
    uint32_t segCwnd = tcb->GetCwndInSegments() + (increment >> 7);
    m_cWndCnt += (increment % 128) / 128.0;
    while (m_cWndCnt >= 1.0)
    {
        ++segCwnd;
        m_cWndCnt -= 1.0;
        m_ackCnt = 0;
    }
    // A window that has not grown for a window of ACKs gains a segment
    if (increment == 0 && m_ackCnt >= segCwnd)
    {
        ++segCwnd;
        m_ackCnt = 0;
    }
    tcb->m_cWnd = segCwnd * tcb->m_segmentSize;
}

uint32_t
TcpHybla::SlowStart(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);

    NS_ASSERT(tcb->m_cWnd <= tcb->m_ssThresh);

    if (segmentsAcked == 0)
    {
        return 0;
    }

    /*
     * slow start
     * INC = 2^RHO - 1, once per ACK whatever the number of segments it
     * acknowledges, as in the Hybla paper and in Linux
     */
    if (m_fixedPoint)
    {
        // 2^fraction in 1/128, for the fraction of rho in 1/8 (Linux
        // hybla_fraction())
        static const uint32_t fractions[] = {128, 139, 152, 165, 181, 197, 215, 234};
        uint32_t rhoInt = m_rho3ls >> 3;
        uint32_t increment =
            ((1U << std::min(rhoInt, 16U)) * fractions[m_rho3ls - (rhoInt << 3)]) - 128;
        AddFixedPointIncrement(tcb, increment);
        tcb->m_cWnd = std::min(tcb->m_cWnd.Get(), tcb->m_ssThresh.Get());
    }
    else
    {
        // As in Linux, the integer part of the exponent is bounded to 16, so
        // that the increment does not overflow for very long RTTs
        double exponent = std::min(std::floor(m_rho), 16.0) + (m_rho - std::floor(m_rho));
        double increment = std::pow(2, exponent) - 1.0;
        auto incr = static_cast<uint32_t>(increment * tcb->m_segmentSize);
        tcb->m_cWnd = std::min(tcb->m_cWnd.Get() + incr, tcb->m_ssThresh.Get());
    }

    NS_LOG_INFO("In SlowStart, updated to cwnd " << tcb->m_cWnd << " ssthresh " << tcb->m_ssThresh);
    return 0;
}

void
TcpHybla::CongestionAvoidance(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);

    if (segmentsAcked == 0)
    {
        return;
    }

    /*
     * congestion avoidance
     * INC = RHO^2 / W, once per ACK whatever the number of segments it
     * acknowledges, as in the Hybla paper and in Linux
     */
    uint32_t segCwnd = tcb->GetCwndInSegments();
    if (m_fixedPoint)
    {
        uint32_t increment = static_cast<uint32_t>((m_rho3ls * m_rho3ls) << 1) / segCwnd;
        if (increment < 128)
        {
            ++m_ackCnt;
        }
        AddFixedPointIncrement(tcb, increment);
    }
    else
    {
        m_cWndCnt += std::pow(m_rho, 2) / static_cast<double>(segCwnd);
        if (m_cWndCnt >= 1.0)
        {
            // double to int truncates every time.
            auto inc = static_cast<uint32_t>(m_cWndCnt);
            m_cWndCnt -= inc;
            NS_ASSERT(m_cWndCnt >= 0.0);
            tcb->m_cWnd += inc * tcb->m_segmentSize;
        }
    }

    NS_LOG_INFO("In CongAvoid, updated to cwnd " << tcb->m_cWnd << " ssthresh " << tcb->m_ssThresh);
}

uint32_t
TcpHybla::GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
    NS_LOG_FUNCTION(this << tcb << bytesInFlight);

    if (m_fixedPoint)
    {
        return std::max(tcb->GetCwndInSegments() >> 1, 2U) * tcb->m_segmentSize;
    }
    return TcpNewReno::GetSsThresh(tcb, bytesInFlight);
}

Ptr<TcpCongestionOps>
TcpHybla::Fork()
{
    return CopyObject<TcpHybla>(this);
}

std::string
TcpHybla::GetName() const
{
    return "TcpHybla";
}

} // namespace ns3
