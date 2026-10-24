/*
 * Copyright (c) 2014 Natale Patriciello <natale.patriciello@gmail.com>
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */
#include "tcp-bic.h"

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/simulator.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TcpBic");
NS_OBJECT_ENSURE_REGISTERED(TcpBic);

TypeId
TcpBic::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::TcpBic")
            .SetParent<TcpCongestionOps>()
            .AddConstructor<TcpBic>()
            .SetGroupName("Internet")
            .AddAttribute("FastConvergence",
                          "Turn on/off fast convergence.",
                          BooleanValue(true),
                          MakeBooleanAccessor(&TcpBic::m_fastConvergence),
                          MakeBooleanChecker())
            .AddAttribute("Beta",
                          "Beta for multiplicative decrease",
                          DoubleValue(0.8),
                          MakeDoubleAccessor(&TcpBic::m_beta),
                          MakeDoubleChecker<double>(0.0))
            .AddAttribute("MaxIncr",
                          "Limit on increment allowed during binary search",
                          UintegerValue(16),
                          MakeUintegerAccessor(&TcpBic::m_maxIncr),
                          MakeUintegerChecker<uint32_t>(1))
            .AddAttribute("LowWnd",
                          "Threshold window size (in segments) for engaging BIC response",
                          UintegerValue(14),
                          MakeUintegerAccessor(&TcpBic::m_lowWnd),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("SmoothPart",
                          "Number of RTT needed to approach cWnd_max from "
                          "cWnd_max-BinarySearchCoefficient. It can be viewed as the gradient "
                          "of the slow start AIM phase: less this value is, "
                          "more steep the increment will be.",
                          UintegerValue(5),
                          MakeUintegerAccessor(&TcpBic::m_smoothPart),
                          MakeUintegerChecker<uint32_t>(1))
            .AddAttribute("BinarySearchCoefficient",
                          "Inverse of the coefficient for the "
                          "binary search. Default 4, as in Linux",
                          UintegerValue(4),
                          MakeUintegerAccessor(&TcpBic::m_b),
                          MakeUintegerChecker<uint8_t>(2))
            .AddAttribute("AckRatioCompensation",
                          "Divide the number of acknowledged segments needed for a window "
                          "increase by the average number of segments an ACK acknowledges "
                          "(the delayed ACK ratio), as Linux does",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpBic::m_ackRatioCompensation),
                          MakeBooleanChecker())
            .AddAttribute("LinuxHz",
                          "Rate of the jiffies (CONFIG_HZ) of Linux. When nonzero, the window "
                          "follows the integer arithmetic of the Linux tcp_bic.c, with its "
                          "times in jiffies at this rate (the count kept for HZ/32 while the "
                          "window does not change), its fixed point beta, the delayed ACK "
                          "ratio compensation, and grows as tcp_cong_avoid_ai(); zero (the "
                          "default) follows the published algorithm. Linux uses a SmoothPart "
                          "of 20",
                          UintegerValue(0),
                          MakeUintegerAccessor(&TcpBic::m_linuxHz),
                          MakeUintegerChecker<uint32_t>());
    return tid;
}

TcpBic::TcpBic()
    : TcpCongestionOps(),
      m_cWndCnt(0),
      m_lastMaxCwnd(0),
      m_lastCwnd(0),
      m_epochStart(Time::Min())
{
    NS_LOG_FUNCTION(this);
}

TcpBic::TcpBic(const TcpBic& sock)
    : TcpCongestionOps(sock),
      m_fastConvergence(sock.m_fastConvergence),
      m_beta(sock.m_beta),
      m_maxIncr(sock.m_maxIncr),
      m_lowWnd(sock.m_lowWnd),
      m_smoothPart(sock.m_smoothPart),
      m_cWndCnt(sock.m_cWndCnt),
      m_lastMaxCwnd(sock.m_lastMaxCwnd),
      m_lastCwnd(sock.m_lastCwnd),
      m_epochStart(sock.m_epochStart),
      m_b(sock.m_b),
      m_ackRatioCompensation(sock.m_ackRatioCompensation),
      m_delayedAck(sock.m_delayedAck),
      m_linuxHz(sock.m_linuxHz),
      m_cnt(sock.m_cnt),
      m_jLastTime(sock.m_jLastTime),
      m_jEpochStart(sock.m_jEpochStart)
{
    NS_LOG_FUNCTION(this);
}

void
TcpBic::IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);

    if (m_linuxHz > 0)
    {
        LinuxIncreaseWindow(tcb, segmentsAcked);
        return;
    }

    if (tcb->m_cWnd < tcb->m_ssThresh)
    {
        // Count every segment ACKed, up to the slow start threshold, as Linux
        // tcp_slow_start() does: a delayed ACK grows the window by two segments
        uint32_t sndCwnd = tcb->m_cWnd;
        tcb->m_cWnd = std::min(sndCwnd + segmentsAcked * tcb->m_segmentSize, tcb->m_ssThresh.Get());
        segmentsAcked -=
            std::min(segmentsAcked, (tcb->m_cWnd.Get() - sndCwnd) / tcb->m_segmentSize);

        NS_LOG_INFO("In SlowStart, updated to cwnd " << tcb->m_cWnd << " ssthresh "
                                                     << tcb->m_ssThresh);
    }

    if (tcb->m_cWnd >= tcb->m_ssThresh && segmentsAcked > 0)
    {
        m_cWndCnt += segmentsAcked;
        uint32_t cnt = Update(tcb);

        /* According to the BIC paper and RFC 6356 even once the new cwnd is
         * calculated you must compare this to the number of ACKs received since
         * the last cwnd update. If not enough ACKs have been received then cwnd
         * cannot be updated.
         */
        if (m_cWndCnt >= cnt)
        {
            tcb->m_cWnd += tcb->m_segmentSize;
            m_cWndCnt -= cnt;
            NS_LOG_INFO("In CongAvoid, updated to cwnd " << tcb->m_cWnd);
        }
        else
        {
            NS_LOG_INFO("Not enough segments have been ACKed to increment cwnd."
                        "Until now "
                        << m_cWndCnt);
        }
    }
}

uint32_t
TcpBic::Jiffies() const
{
    return static_cast<uint32_t>(Simulator::Now().GetMicroSeconds() * m_linuxHz / 1000000);
}

void
TcpBic::LinuxIncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);
    uint32_t cwnd = tcb->GetCwndInSegments();
    uint32_t acked = segmentsAcked;
    if (tcb->m_cWnd < tcb->m_ssThresh)
    {
        // tcp_slow_start()
        uint32_t newCwnd = std::min(cwnd + acked, tcb->m_ssThresh.Get() / tcb->m_segmentSize);
        acked -= newCwnd - cwnd;
        cwnd = newCwnd;
        tcb->m_cWnd = cwnd * tcb->m_segmentSize;
        if (acked == 0)
        {
            return;
        }
    }

    // bictcp_update(): the count is kept for HZ/32 while the window does
    // not change
    const uint32_t now = Jiffies();
    if (m_lastCwnd != cwnd ||
        static_cast<int32_t>(now - m_jLastTime) > static_cast<int32_t>(m_linuxHz / 32))
    {
        m_jLastTime = now;
        if (m_jEpochStart == 0)
        {
            m_jEpochStart = now;
        }
        m_delayedAck = std::max(m_delayedAck, 1U);
        m_cnt = Update(tcb);
    }

    // tcp_cong_avoid_ai()
    if (m_cWndCnt >= m_cnt)
    {
        m_cWndCnt = 0;
        ++cwnd;
    }
    m_cWndCnt += acked;
    if (m_cWndCnt >= m_cnt)
    {
        uint32_t delta = m_cWndCnt / m_cnt;
        m_cWndCnt -= delta * m_cnt;
        cwnd += delta;
    }
    tcb->m_cWnd = cwnd * tcb->m_segmentSize;
    NS_LOG_INFO("In CongAvoid, cwnd " << cwnd << " cnt " << m_cnt << " count " << m_cWndCnt);
}

uint32_t
TcpBic::Update(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);

    uint32_t segCwnd = tcb->GetCwndInSegments();
    uint32_t cnt;

    m_lastCwnd = segCwnd;

    if (m_epochStart == Time::Min())
    {
        m_epochStart = Simulator::Now(); /* record the beginning of an epoch */
    }

    if (segCwnd <= m_lowWnd)
    {
        NS_LOG_INFO("Under lowWnd, compatibility mode. Behaving as NewReno");
        cnt = segCwnd;
        return cnt;
    }

    if (segCwnd < m_lastMaxCwnd)
    {
        double dist = (m_lastMaxCwnd - segCwnd) / m_b;

        NS_LOG_INFO("cWnd = " << segCwnd << " under lastMax, " << m_lastMaxCwnd
                              << " and dist=" << dist);
        if (dist > m_maxIncr)
        {
            /* Linear increase */
            cnt = segCwnd / m_maxIncr;
            NS_LOG_INFO("Linear increase (maxIncr=" << m_maxIncr << "), cnt=" << cnt);
        }
        else if (dist <= 1)
        {
            /* smoothed binary search increase: when our window is really
             * close to the last maximum, we parameterize in m_smoothPart the number
             * of RTT needed to reach that window.
             */
            cnt = (segCwnd * m_smoothPart) / m_b;

            NS_LOG_INFO("Binary search increase (smoothPart=" << m_smoothPart << "), cnt=" << cnt);
        }
        else
        {
            /* binary search increase */
            cnt = static_cast<uint32_t>(segCwnd / dist);

            NS_LOG_INFO("Binary search increase, cnt=" << cnt);
        }
    }
    else
    {
        NS_LOG_INFO("cWnd = " << segCwnd << " above last max, " << m_lastMaxCwnd);
        if (segCwnd < m_lastMaxCwnd + m_b)
        {
            /* slow start AMD linear increase */
            cnt = (segCwnd * m_smoothPart) / m_b;
            NS_LOG_INFO("Slow start AMD, cnt=" << cnt);
        }
        else if (segCwnd < m_lastMaxCwnd + m_maxIncr * (m_b - 1))
        {
            /* slow start */
            cnt = (segCwnd * (m_b - 1)) / (segCwnd - m_lastMaxCwnd);

            NS_LOG_INFO("Slow start, cnt=" << cnt);
        }
        else
        {
            /* linear increase */
            cnt = segCwnd / m_maxIncr;

            NS_LOG_INFO("Linear, cnt=" << cnt);
        }
    }

    /* if in slow start or link utilization is very low. Code taken from Linux
     * kernel, not sure of the source they take it. Usually, it is not reached,
     * since if m_lastMaxCwnd is 0, we are (hopefully) in slow start.
     */
    if (m_lastMaxCwnd == 0)
    {
        if (cnt > 20) /* increase cwnd 5% per RTT */
        {
            cnt = 20;
        }
    }

    if (m_ackRatioCompensation || m_linuxHz > 0)
    {
        // Linux bictcp_update(): the count is in ACKs of the average size
        cnt = (cnt << ACK_RATIO_SHIFT) / m_delayedAck;
    }

    if (cnt == 0)
    {
        cnt = 1;
    }

    return cnt;
}

void
TcpBic::PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked << rtt);

    // Linux bictcp_acked(): moving average, over 16 ACKs, of the segments
    // acknowledged by an ACK, in 1/16 of a segment
    if (m_linuxHz > 0)
    {
        // bictcp_acked(), with the segments cumulatively acknowledged (see
        // TcpSocketBase::LinuxPktsAcked)
        if (tcb->m_congState == TcpSocketState::CA_OPEN)
        {
            // delayed_ack += pkts_acked - (delayed_ack >> 4)
            m_delayedAck += segmentsAcked - (m_delayedAck >> ACK_RATIO_SHIFT);
        }
        NS_LOG_LOGIC("Delayed ACK ratio " << m_delayedAck);
    }
    else if (tcb->m_congState == TcpSocketState::CA_OPEN)
    {
        m_delayedAck += segmentsAcked;
        m_delayedAck -= m_delayedAck >> ACK_RATIO_SHIFT;
        m_delayedAck = std::max<uint32_t>(m_delayedAck, 1);
    }
}

void
TcpBic::CongestionStateSet(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCongState_t newState)
{
    NS_LOG_FUNCTION(this << tcb << newState);

    // The ACKs counted towards the next increase of the window are
    // discarded when it is reduced, as tp->snd_cwnd_cnt in Linux
    // (tcp_init_cwnd_reduction() and tcp_enter_loss())
    if (newState == TcpSocketState::CA_RECOVERY || newState == TcpSocketState::CA_CWR ||
        newState == TcpSocketState::CA_LOSS)
    {
        m_cWndCnt = 0;
    }

    if (m_linuxHz > 0 && newState == TcpSocketState::CA_LOSS)
    {
        // bictcp_reset()
        m_cnt = 0;
        m_lastMaxCwnd = 0;
        m_lastCwnd = 0;
        m_jLastTime = 0;
        m_jEpochStart = 0;
        m_delayedAck = 2 << ACK_RATIO_SHIFT;
    }
}

std::string
TcpBic::GetName() const
{
    return "TcpBic";
}

uint32_t
TcpBic::GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
    NS_LOG_FUNCTION(this);

    uint32_t segCwnd = tcb->GetCwndInSegments();
    uint32_t ssThresh = 0;

    m_epochStart = Time::Min();

    if (m_linuxHz > 0)
    {
        // bictcp_recalc_ssthresh(), with beta in 1/1024
        const auto beta = static_cast<uint32_t>(std::lround(m_beta * 1024));
        m_jEpochStart = 0;
        if (segCwnd < m_lastMaxCwnd && m_fastConvergence)
        {
            m_lastMaxCwnd = (segCwnd * (1024 + beta)) / (2 * 1024);
        }
        else
        {
            m_lastMaxCwnd = segCwnd;
        }
        ssThresh = segCwnd <= m_lowWnd ? segCwnd >> 1 : (segCwnd * beta) / 1024;
        return std::max(ssThresh, 2U) * tcb->m_segmentSize;
    }

    /* Wmax and fast convergence */
    if (segCwnd < m_lastMaxCwnd && m_fastConvergence)
    {
        // The new maximum is the midpoint between the window and the reduced
        // window, as in the BIC paper (Wmax = cwnd * (2 - beta) / 2, with beta
        // the decrease factor) and in Linux bictcp_recalc_ssthresh()
        auto lastMaxCwnd = static_cast<uint32_t>(segCwnd * (1 + m_beta) / 2);
        NS_LOG_INFO("Fast Convergence. Last max cwnd: " << m_lastMaxCwnd << " updated to "
                                                        << lastMaxCwnd);
        m_lastMaxCwnd = lastMaxCwnd;
    }
    else
    {
        NS_LOG_INFO("Last max cwnd: " << m_lastMaxCwnd << " updated to " << segCwnd);
        m_lastMaxCwnd = segCwnd;
    }

    if (segCwnd <= m_lowWnd)
    {
        ssThresh = std::max(2 * tcb->m_segmentSize, bytesInFlight / 2);
        NS_LOG_INFO("Less than lowWindow, ssTh= " << ssThresh);
    }
    else
    {
        ssThresh = static_cast<uint32_t>(std::max(segCwnd * m_beta, 2.0) * tcb->m_segmentSize);
        NS_LOG_INFO("More than lowWindow, ssTh= " << ssThresh);
    }

    return ssThresh;
}

Ptr<TcpCongestionOps>
TcpBic::Fork()
{
    return CopyObject<TcpBic>(this);
}

} // namespace ns3
