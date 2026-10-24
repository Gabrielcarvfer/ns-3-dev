/*
 * Copyright (c) 2013 ResiliNets, ITTC, University of Kansas
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Siddharth Gangadhar <siddharth@ittc.ku.edu>,
 *          Truc Anh N. Nguyen <annguyen@ittc.ku.edu>,
 *          Greeshma Umapathi
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  https://resilinets.org/
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 *
 * Work supported in part by NSF FIND (Future Internet Design) Program
 * under grant CNS-0626918 (Postmodern Internet Architecture),
 * NSF grant CNS-1050226 (Multilayer Network Resilience Analysis and Experimentation on GENI),
 * US Department of Defense (DoD), and ITTC at The University of Kansas.
 */

#include "tcp-westwood-plus.h"

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/uinteger.h"

NS_LOG_COMPONENT_DEFINE("TcpWestwoodPlus");

namespace ns3
{

NS_OBJECT_ENSURE_REGISTERED(TcpWestwoodPlus);

TypeId
TcpWestwoodPlus::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::TcpWestwoodPlus")
            .SetParent<TcpNewReno>()
            .SetGroupName("Internet")
            .AddConstructor<TcpWestwoodPlus>()
            .AddAttribute("FilterType",
                          "Use this to choose no filter, Tustin's approximation filter, or the "
                          "estimator of Linux (samples over windows of an RTT, and of at least "
                          "50 ms, with the RTTs in jiffies, filtered by two cascaded moving "
                          "averages of gain 1/8)",
                          EnumValue(TcpWestwoodPlus::TUSTIN),
                          MakeEnumAccessor<FilterType>(&TcpWestwoodPlus::m_fType),
                          MakeEnumChecker(TcpWestwoodPlus::NONE,
                                          "None",
                                          TcpWestwoodPlus::TUSTIN,
                                          "Tustin",
                                          TcpWestwoodPlus::LINUX,
                                          "Linux"))
            .AddAttribute("LinuxRecovery",
                          "Set the slow start threshold from the bandwidth estimate as Linux "
                          "does: halve the window when a fast recovery starts, and set the "
                          "threshold and the window from the estimate when it ends",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpWestwoodPlus::m_linuxRecovery),
                          MakeBooleanChecker())
            .AddAttribute("LinuxHz",
                          "Rate of the jiffies of the Linux estimator (FilterType Linux), "
                          "the HZ of the Linux kernel",
                          UintegerValue(1000),
                          MakeUintegerAccessor(&TcpWestwoodPlus::m_linuxHz),
                          MakeUintegerChecker<uint32_t>(20))
            .AddTraceSource("EstimatedBW",
                            "The estimated bandwidth",
                            MakeTraceSourceAccessor(&TcpWestwoodPlus::m_currentBW),
                            "ns3::TracedValueCallback::DataRate");
    return tid;
}

TcpWestwoodPlus::TcpWestwoodPlus()
    : TcpNewReno(),
      m_currentBW(0),
      m_lastSampleBW(0),
      m_lastBW(0),
      m_ackedSegments(0),
      m_IsCount(false),
      m_lastAck(0)
{
    NS_LOG_FUNCTION(this);
}

TcpWestwoodPlus::TcpWestwoodPlus(const TcpWestwoodPlus& sock)
    : TcpNewReno(sock),
      m_currentBW(sock.m_currentBW),
      m_lastSampleBW(sock.m_lastSampleBW),
      m_lastBW(sock.m_lastBW),
      m_fType(sock.m_fType),
      m_linuxRecovery(sock.m_linuxRecovery),
      m_ackedSegments(sock.m_ackedSegments),
      m_IsCount(sock.m_IsCount),
      m_lastAck(sock.m_lastAck),
      m_linuxHz(sock.m_linuxHz),
      m_bwNsEst(sock.m_bwNsEst),
      m_bwEst(sock.m_bwEst),
      m_rttWinSx(sock.m_rttWinSx),
      m_bk(sock.m_bk),
      m_sndUna(sock.m_sndUna),
      m_accounted(sock.m_accounted),
      m_rttJiffies(sock.m_rttJiffies),
      m_rttMinJiffies(sock.m_rttMinJiffies),
      m_firstAck(sock.m_firstAck),
      m_resetRttMin(sock.m_resetRttMin),
      m_linuxStarted(sock.m_linuxStarted)
{
    NS_LOG_FUNCTION(this);
    NS_LOG_LOGIC("Invoked the copy constructor");
}

TcpWestwoodPlus::~TcpWestwoodPlus()
{
}

uint32_t
TcpWestwoodPlus::Jiffies() const
{
    return static_cast<uint32_t>(Simulator::Now().GetMicroSeconds() * m_linuxHz / 1000000);
}

void
TcpWestwoodPlus::LinuxStart()
{
    if (!m_linuxStarted)
    {
        // tcp_westwood_init(): the initial RTT is 20 s, and the first
        // bandwidth window starts with the connection
        m_linuxStarted = true;
        m_rttJiffies = 20 * m_linuxHz;
        m_rttMinJiffies = m_rttJiffies;
        m_rttWinSx = Jiffies();
    }
}

void
TcpWestwoodPlus::LinuxPktsAcked(Ptr<TcpSocketState> tcb, const Time& rtt)
{
    LinuxStart();

    // tcp_westwood_ack(), on the slow path: westwood_update_window()
    if (m_firstAck)
    {
        m_sndUna = tcb->m_lastAckedSeq;
        m_firstAck = false;
    }
    auto delta = static_cast<int32_t>(Jiffies() - m_rttWinSx);
    if (m_rttJiffies != 0 && delta > static_cast<int32_t>(std::max(m_rttJiffies, m_linuxHz / 20)))
    {
        // westwood_filter()
        uint32_t sample = m_bk / static_cast<uint32_t>(delta);
        if (m_bwNsEst == 0 && m_bwEst == 0)
        {
            m_bwNsEst = sample;
            m_bwEst = m_bwNsEst;
        }
        else
        {
            m_bwNsEst = ((7 * m_bwNsEst) + sample) >> 3;
            m_bwEst = ((7 * m_bwEst) + m_bwNsEst) >> 3;
        }
        m_currentBW = DataRate(static_cast<uint64_t>(m_bwEst) * 8 * m_linuxHz);
        NS_LOG_LOGIC("Sample " << sample << " bytes/jiffy, estimate " << m_bwEst);
        m_bk = 0;
        m_rttWinSx = Jiffies();
    }

    // westwood_acked_count(): a duplicate ACK counts a segment, which the
    // next cumulative ACK does not count again
    uint32_t mss = tcb->m_segmentSize;
    auto cumulAck = static_cast<uint32_t>(tcb->m_lastAckedSeq - m_sndUna);
    if (cumulAck == 0)
    {
        m_accounted += mss;
        cumulAck = mss;
    }
    if (cumulAck > mss)
    {
        if (m_accounted >= cumulAck)
        {
            m_accounted -= cumulAck;
            cumulAck = mss;
        }
        else
        {
            cumulAck -= m_accounted;
            m_accounted = 0;
        }
    }
    m_sndUna = tcb->m_lastAckedSeq;
    m_bk += cumulAck;

    // update_rtt_min(), with the RTT of the previous ACK
    if (m_resetRttMin)
    {
        m_rttMinJiffies = m_rttJiffies;
        m_resetRttMin = false;
    }
    else
    {
        m_rttMinJiffies = std::min(m_rttJiffies, m_rttMinJiffies);
    }

    // tcp_westwood_pkts_acked(): usecs_to_jiffies() rounds up
    if (rtt.IsStrictlyPositive())
    {
        m_rttJiffies =
            static_cast<uint32_t>((rtt.GetMicroSeconds() * m_linuxHz + 999999) / 1000000);
    }
}

void
TcpWestwoodPlus::PktsAcked(Ptr<TcpSocketState> tcb, uint32_t packetsAcked, const Time& rtt)
{
    NS_LOG_FUNCTION(this << tcb << packetsAcked << rtt);

    if (m_fType == TcpWestwoodPlus::LINUX)
    {
        LinuxPktsAcked(tcb, rtt);
        return;
    }

    // The acknowledged segments count towards the bandwidth estimate even
    // when the ACK gives no RTT sample, as the duplicate ACKs do (Linux
    // westwood_acked_count())
    m_ackedSegments += packetsAcked;

    if (!(rtt.IsZero() || m_IsCount))
    {
        m_IsCount = true;
        m_bwEstimateEvent.Cancel();
        m_bwEstimateEvent = Simulator::Schedule(rtt, &TcpWestwoodPlus::EstimateBW, this, rtt, tcb);
    }
}

void
TcpWestwoodPlus::EstimateBW(const Time& rtt, Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this);

    NS_ASSERT(!rtt.IsZero());

    m_currentBW = DataRate(m_ackedSegments * tcb->m_segmentSize * 8.0 / rtt.GetSeconds());
    m_IsCount = false;

    m_ackedSegments = 0;

    NS_LOG_LOGIC("Estimated BW: " << m_currentBW);

    // Filter the BW sample

    constexpr double ALPHA = 0.9;

    if (m_fType != TcpWestwoodPlus::NONE && m_lastBW.GetBitRate() == 0)
    {
        // The filter starts from the first sample, not from zero
        m_lastSampleBW = m_currentBW;
        m_lastBW = m_currentBW;
    }
    else if (m_fType == TcpWestwoodPlus::TUSTIN)
    {
        DataRate sample_bwe = m_currentBW;
        m_currentBW = (m_lastBW * ALPHA) + (((sample_bwe + m_lastSampleBW) * 0.5) * (1 - ALPHA));
        m_lastSampleBW = sample_bwe;
        m_lastBW = m_currentBW;
    }

    NS_LOG_LOGIC("Estimated BW after filtering: " << m_currentBW);
}

uint32_t
TcpWestwoodPlus::GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight [[maybe_unused]])
{
    NS_LOG_FUNCTION(this << tcb << bytesInFlight);

    if (m_linuxRecovery)
    {
        // As tcp_reno_ssthresh() of Linux; the estimate applies when the
        // recovery ends, or right away after a retransmission timeout (see
        // CwndEvent)
        return std::max(2 * tcb->m_segmentSize, tcb->m_cWnd.Get() / 2);
    }
    return EstimatedSsThresh(tcb);
}

void
TcpWestwoodPlus::CwndEvent(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event)
{
    NS_LOG_FUNCTION(this << tcb << event);

    // Linux tcp_westwood_init() runs when the connection is established: the
    // first transmission of data is the first event the congestion control
    // is given after it, while the first ACK comes a round trip later
    if (event == TcpSocketState::CA_EVENT_TX_START && m_fType == TcpWestwoodPlus::LINUX)
    {
        LinuxStart();
    }

    if (event == TcpSocketState::CA_EVENT_LOSS && m_fType == TcpWestwoodPlus::LINUX)
    {
        // The minimum RTT restarts from the next sample
        m_resetRttMin = true;
    }

    if (!m_linuxRecovery)
    {
        return;
    }

    if (event == TcpSocketState::CA_EVENT_LOSS)
    {
        tcb->m_ssThresh = EstimatedSsThresh(tcb);
    }
    else if (event == TcpSocketState::CA_EVENT_COMPLETE_CWR)
    {
        tcb->m_ssThresh = EstimatedSsThresh(tcb);
        tcb->m_cWnd = tcb->m_ssThresh.Get();
    }
}

uint32_t
TcpWestwoodPlus::EstimatedSsThresh(Ptr<const TcpSocketState> tcb) const
{
    if (m_fType == TcpWestwoodPlus::LINUX)
    {
        // tcp_westwood_bw_rttmin()
        return std::max(m_bwEst * m_rttMinJiffies / tcb->m_segmentSize, 2U) * tcb->m_segmentSize;
    }

    uint32_t ssThresh = static_cast<uint32_t>((m_currentBW * tcb->m_minRtt) / 8.0);

    NS_LOG_LOGIC("CurrentBW: " << m_currentBW << " minRtt: " << tcb->m_minRtt
                               << " ssThresh: " << ssThresh);

    return std::max(2 * tcb->m_segmentSize, ssThresh);
}

Ptr<TcpCongestionOps>
TcpWestwoodPlus::Fork()
{
    return CopyObject<TcpWestwoodPlus>(this);
}

} // namespace ns3
