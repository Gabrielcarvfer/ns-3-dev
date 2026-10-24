/*
 * Copyright (c) 2016 ResiliNets, ITTC, University of Kansas
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Truc Anh N. Nguyen <annguyen@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  https://resilinets.org/
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 */

#include "tcp-veno.h"

#include "tcp-socket-state.h"

#include "ns3/log.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TcpVeno");
NS_OBJECT_ENSURE_REGISTERED(TcpVeno);

TypeId
TcpVeno::GetTypeId()
{
    static TypeId tid = TypeId("ns3::TcpVeno")
                            .SetParent<TcpNewReno>()
                            .AddConstructor<TcpVeno>()
                            .SetGroupName("Internet")
                            .AddAttribute("Beta",
                                          "Threshold for congestion detection",
                                          UintegerValue(3),
                                          MakeUintegerAccessor(&TcpVeno::m_beta),
                                          MakeUintegerChecker<uint32_t>())
                            .AddAttribute("LinuxArithmetic",
                                          "Compute as the tcp_veno.c of Linux does: the RTT "
                                          "samples in microseconds plus one, the difference in "
                                          "half segments, computed only while Veno is on with "
                                          "enough samples, the slow start leftover given to the "
                                          "congestion avoidance, the congestive increase of a "
                                          "segment every other window of acknowledged segments, "
                                          "the halving of the window in whole segments, and the "
                                          "restart of Veno on the first transmission",
                                          BooleanValue(false),
                                          MakeBooleanAccessor(&TcpVeno::m_linuxArithmetic),
                                          MakeBooleanChecker());
    return tid;
}

TcpVeno::TcpVeno()
    : TcpNewReno(),
      m_baseRtt(Time::Max()),
      m_minRtt(Time::Max()),
      m_cntRtt(0),
      m_doingVenoNow(true),
      m_diff(0),
      m_inc(true),
      m_ackCnt(0),
      m_beta(6)
{
    NS_LOG_FUNCTION(this);
}

TcpVeno::TcpVeno(const TcpVeno& sock)
    : TcpNewReno(sock),
      m_baseRtt(sock.m_baseRtt),
      m_minRtt(sock.m_minRtt),
      m_cntRtt(sock.m_cntRtt),
      m_doingVenoNow(true),
      m_diff(0),
      m_inc(true),
      m_ackCnt(sock.m_ackCnt),
      m_beta(sock.m_beta),
      m_linuxArithmetic(sock.m_linuxArithmetic)
{
    NS_LOG_FUNCTION(this);
}

TcpVeno::~TcpVeno()
{
    NS_LOG_FUNCTION(this);
}

Ptr<TcpCongestionOps>
TcpVeno::Fork()
{
    return CopyObject<TcpVeno>(this);
}

void
TcpVeno::PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked << rtt);

    if (rtt.IsZero())
    {
        return;
    }

    // Linux tcp_veno_pkts_acked(): microseconds, plus one so that no RTT is zero
    Time sample = m_linuxArithmetic ? MicroSeconds(rtt.GetMicroSeconds() + 1) : rtt;

    m_minRtt = std::min(m_minRtt, sample);
    NS_LOG_DEBUG("Updated m_minRtt= " << m_minRtt);

    m_baseRtt = std::min(m_baseRtt, sample);
    NS_LOG_DEBUG("Updated m_baseRtt= " << m_baseRtt);

    // Update RTT counter
    m_cntRtt++;
    NS_LOG_DEBUG("Updated m_cntRtt= " << m_cntRtt);
}

void
TcpVeno::EnableVeno()
{
    NS_LOG_FUNCTION(this);

    m_doingVenoNow = true;
    m_minRtt = Time::Max();
}

void
TcpVeno::DisableVeno()
{
    NS_LOG_FUNCTION(this);

    m_doingVenoNow = false;
}

void
TcpVeno::CongestionStateSet(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCongState_t newState)
{
    NS_LOG_FUNCTION(this << tcb << newState);
    TcpNewReno::CongestionStateSet(tcb, newState);
    if (newState == TcpSocketState::CA_OPEN)
    {
        EnableVeno();
        NS_LOG_LOGIC("Veno is now on.");
    }
    else
    {
        DisableVeno();
        NS_LOG_LOGIC("Veno is turned off.");
    }
}

void
TcpVeno::CwndEvent(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event)
{
    NS_LOG_FUNCTION(this << tcb << event);

    // Linux tcp_veno_cwnd_event(): tcp_veno_init() on a (re)start of the
    // transmissions
    if (m_linuxArithmetic && (event == TcpSocketState::CA_EVENT_TX_START ||
                              event == TcpSocketState::CA_EVENT_CWND_RESTART))
    {
        m_baseRtt = Time::Max();
        m_inc = true;
        EnableVeno();
    }
}

void
TcpVeno::LinuxIncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);

    if (!m_doingVenoNow)
    {
        TcpNewReno::IncreaseWindow(tcb, segmentsAcked);
        return;
    }

    if (m_cntRtt <= 2)
    {
        TcpNewReno::IncreaseWindow(tcb, segmentsAcked);
    }
    else
    {
        // The difference is in half segments (V_PARAM_SHIFT)
        uint32_t segCwnd = tcb->GetCwndInSegments();
        uint64_t targetCwnd =
            (static_cast<uint64_t>(segCwnd) * static_cast<uint64_t>(m_baseRtt.GetMicroSeconds()))
            << 1;
        targetCwnd /= static_cast<uint64_t>(m_minRtt.GetMicroSeconds());
        m_diff = (segCwnd << 1) - static_cast<uint32_t>(targetCwnd);
        NS_LOG_DEBUG("Calculated m_diff = " << m_diff << " half segments");

        uint32_t acked = segmentsAcked;
        bool done = false;
        if (tcb->m_cWnd < tcb->m_ssThresh)
        {
            acked = TcpNewReno::SlowStart(tcb, acked);
            done = (acked == 0);
        }
        if (!done)
        {
            if (m_diff < (m_beta << 1))
            {
                TcpNewReno::CongestionAvoidance(tcb, acked);
            }
            else
            {
                // A segment every other window of acknowledged segments,
                // the ACK which completes the window being not counted
                segCwnd = tcb->GetCwndInSegments();
                if (GetCwndCount() >= segCwnd)
                {
                    if (m_inc)
                    {
                        tcb->m_cWnd += tcb->m_segmentSize;
                        m_inc = false;
                    }
                    else
                    {
                        m_inc = true;
                    }
                    SetCwndCount(0);
                }
                else
                {
                    SetCwndCount(GetCwndCount() + acked);
                }
            }
        }
        tcb->m_cWnd = std::max(tcb->m_cWnd.Get(), 2 * tcb->m_segmentSize);
    }
    m_minRtt = Time::Max();
}

void
TcpVeno::IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);

    if (m_linuxArithmetic)
    {
        LinuxIncreaseWindow(tcb, segmentsAcked);
        return;
    }

    // Always calculate m_diff, even if we are not doing Veno now
    uint32_t targetCwnd;
    uint32_t segCwnd = tcb->GetCwndInSegments();

    /*
     * Calculate the cwnd we should have. baseRtt is the minimum RTT
     * per-connection, minRtt is the minimum RTT in this window
     *
     * little trick:
     * desidered throughput is currentCwnd * baseRtt
     * target cwnd is throughput / minRtt
     */
    double tmp = m_baseRtt.GetSeconds() / m_minRtt.GetSeconds();
    targetCwnd = static_cast<uint32_t>(segCwnd * tmp);
    NS_LOG_DEBUG("Calculated targetCwnd = " << targetCwnd);
    NS_ASSERT(segCwnd >= targetCwnd); // implies baseRtt <= minRtt

    // Calculate the difference between actual and target cwnd
    m_diff = segCwnd - targetCwnd;
    NS_LOG_DEBUG("Calculated m_diff = " << m_diff);

    if (!m_doingVenoNow)
    {
        // If Veno is not on, we follow NewReno algorithm
        NS_LOG_LOGIC("Veno is not turned on, we follow NewReno algorithm.");
        TcpNewReno::IncreaseWindow(tcb, segmentsAcked);
        return;
    }

    // We do the Veno calculations only if we got enough RTT samples
    if (m_cntRtt <= 2)
    { // We do not have enough RTT samples, so we should behave like NewReno
        NS_LOG_LOGIC("We do not have enough RTT samples to perform Veno "
                     "calculations, we behave like NewReno.");
        TcpNewReno::IncreaseWindow(tcb, segmentsAcked);
    }
    else
    {
        NS_LOG_LOGIC("We have enough RTT samples to perform Veno calculations.");

        if (tcb->m_cWnd < tcb->m_ssThresh)
        { // Slow start mode. Veno employs same slow start algorithm as NewReno's.
            NS_LOG_LOGIC("We are in slow start, behave like NewReno.");
            TcpNewReno::SlowStart(tcb, segmentsAcked);
        }
        else
        { // Congestion avoidance mode
            NS_LOG_LOGIC("We are in congestion avoidance, execute Veno additive "
                         "increase algo.");

            if (m_diff < m_beta)
            {
                // Available bandwidth is not fully utilized,
                // increase cwnd by 1 every RTT
                NS_LOG_LOGIC("Available bandwidth not fully utilized, increase "
                             "cwnd by 1 every RTT");
                TcpNewReno::CongestionAvoidance(tcb, segmentsAcked);
            }
            else
            {
                // Available bandwidth is fully utilized,
                // increase cwnd by 1 every other RTT
                NS_LOG_LOGIC("Available bandwidth fully utilized, increase cwnd "
                             "by 1 every other RTT");
                if (m_inc)
                {
                    TcpNewReno::CongestionAvoidance(tcb, segmentsAcked);
                    m_inc = false;
                }
                else
                {
                    m_inc = true;
                }
            }
        }
    }

    // As in Linux tcp_veno_cong_avoid(), minRtt restarts with every ACK, while
    // cntRtt keeps counting the samples taken since the start of the
    // connection: resetting it here would leave it below the three samples
    // Veno needs, so that the window would always grow as in NewReno
    m_minRtt = Time::Max();
}

std::string
TcpVeno::GetName() const
{
    return "TcpVeno";
}

uint32_t
TcpVeno::GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
    NS_LOG_FUNCTION(this << tcb << bytesInFlight);

    if (m_linuxArithmetic)
    {
        // tcp_veno_ssthresh(): the window in whole segments
        uint32_t segCwnd = tcb->GetCwndInSegments();
        uint32_t ssThresh = m_diff < (m_beta << 1) ? segCwnd * 4 / 5 : segCwnd >> 1;
        return std::max(ssThresh, 2U) * tcb->m_segmentSize;
    }

    if (m_diff < m_beta)
    {
        // random loss due to bit errors is most likely to have occurred,
        // we cut cwnd by 1/5: the window, in segments, as in the Veno paper
        // and in Linux, not the data in flight, which the lost and SACKed
        // segments reduce
        NS_LOG_LOGIC("Random loss is most likely to have occurred, "
                     "cwnd is reduced by 1/5");
        return std::max(tcb->GetCwndInSegments() * 4 / 5, 2U) * tcb->m_segmentSize;
    }
    else
    {
        // congestion-based loss is most likely to have occurred,
        // we reduce cwnd by 1/2 as in NewReno
        NS_LOG_LOGIC("Congestive loss is most likely to have occurred, "
                     "cwnd is halved");
        return TcpNewReno::GetSsThresh(tcb, bytesInFlight);
    }
}

} // namespace ns3
