/*
 * Copyright (c) 2016 NITK Surathkal
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Charitha Sangaraju <charitha29193@gmail.com>
 *          Nandita G <gm.nandita@gmail.com>
 *          Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 *
 */

#include "tcp-lp.h"

#include "tcp-option-ts.h"

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/simulator.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TcpLp");
NS_OBJECT_ENSURE_REGISTERED(TcpLp);

TypeId
TcpLp::GetTypeId()
{
    static TypeId tid = TypeId("ns3::TcpLp")
                            .SetParent<TcpNewReno>()
                            .AddConstructor<TcpLp>()
                            .SetGroupName("Internet")
                            .AddAttribute("LinuxArithmetic",
                                          "Compute the one way delays, the thresholds and the "
                                          "inference as the tcp_lp.c of Linux does, with its "
                                          "estimate of the clock rate of the receiver and its "
                                          "32-bit arithmetic",
                                          BooleanValue(false),
                                          MakeBooleanAccessor(&TcpLp::m_linuxArithmetic),
                                          MakeBooleanChecker())
                            .AddAttribute("AckHoldFilter",
                                          "Exclude from the one way delays the time the receiver "
                                          "holds a segment before acknowledging it: the delay is "
                                          "measured from the send time of the most recently sent "
                                          "segment that the ACK acknowledges, not from the echoed "
                                          "timestamp, and the ACKs of a single segment, sent by "
                                          "the delayed ACK timer of a receiver which delays its "
                                          "ACKs, are ignored. Not part of the TCP-LP paper, nor "
                                          "of Linux",
                                          BooleanValue(false),
                                          MakeBooleanAccessor(&TcpLp::m_ackHoldFilter),
                                          MakeBooleanChecker())
                            .AddAttribute("RestartSmoothing",
                                          "Restart the smoothed one way delay at the first "
                                          "sample of the data sent after a window reduction, "
                                          "so that the delays of the larger window before it "
                                          "do not indicate congestion again. Not part of the "
                                          "TCP-LP paper, nor of Linux",
                                          BooleanValue(false),
                                          MakeBooleanAccessor(&TcpLp::m_restartSmoothing),
                                          MakeBooleanChecker());
    return tid;
}

TcpLp::TcpLp()
    : TcpNewReno(),
      m_flag(0),
      m_sOwd(0),
      m_owdMin(0xffffffff),
      m_owdMax(0),
      m_owdMaxRsv(0),
      m_lastDrop(Time(0)),
      m_inference(Time(0))
{
    NS_LOG_FUNCTION(this);
}

TcpLp::TcpLp(const TcpLp& sock)
    : TcpNewReno(sock),
      m_flag(sock.m_flag),
      m_sOwd(sock.m_sOwd),
      m_owdMin(sock.m_owdMin),
      m_owdMax(sock.m_owdMax),
      m_owdMaxRsv(sock.m_owdMaxRsv),
      m_lastDrop(sock.m_lastDrop),
      m_inference(sock.m_inference),
      m_linuxArithmetic(sock.m_linuxArithmetic),
      m_remoteHz(sock.m_remoteHz),
      m_remoteRefTime(sock.m_remoteRefTime),
      m_localRefTime(sock.m_localRefTime),
      m_lastDropTs(sock.m_lastDropTs),
      m_inferenceTs(sock.m_inferenceTs),
      m_ackHoldFilter(sock.m_ackHoldFilter),
      m_restartSmoothing(sock.m_restartSmoothing),
      m_haveOwd(sock.m_haveOwd),
      m_owdRef(sock.m_owdRef),
      m_dMin(sock.m_dMin),
      m_dMax(sock.m_dMax),
      m_sd8(sock.m_sd8),
      m_lossSeen(sock.m_lossSeen),
      m_reactionSeq(sock.m_reactionSeq),
      m_reacted(sock.m_reacted),
      m_restartPending(sock.m_restartPending),
      m_lastAck(sock.m_lastAck),
      m_haveAck(sock.m_haveAck),
      m_delayedAcks(sock.m_delayedAcks),
      m_reducing(sock.m_reducing)
{
    NS_LOG_FUNCTION(this);
}

TcpLp::~TcpLp()
{
    NS_LOG_FUNCTION(this);
}

Ptr<TcpCongestionOps>
TcpLp::Fork()
{
    return CopyObject<TcpLp>(this);
}

void
TcpLp::CongestionAvoidance(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);

    if (!(m_flag & LP_WITHIN_INF))
    {
        TcpNewReno::CongestionAvoidance(tcb, segmentsAcked);
    }
}

void
TcpLp::SetRateOps(Ptr<TcpRateOps> rateOps)
{
    m_rateOps = rateOps;
}

void
TcpLp::CongestionStateSet(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCongState_t newState)
{
    NS_LOG_FUNCTION(this << tcb << newState);

    TcpNewReno::CongestionStateSet(tcb, newState);
    bool reducing = newState == TcpSocketState::CA_RECOVERY ||
                    newState == TcpSocketState::CA_LOSS || newState == TcpSocketState::CA_CWR;
    if (m_reducing && !reducing)
    {
        // The reaction to the loss ends with the reduction: the delays of
        // the data sent until then are those of the larger window
        m_reactionSeq = tcb->m_highTxMark;
        m_reacted = true;
        m_restartPending = true;
    }
    m_reducing = reducing;
}

int64_t
TcpLp::OwdCalculator(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);

    // The timestamp of the receiver minus the send time on the local clock.
    // The offset between the two clocks (and the random offsets of the
    // timestamps) is constant: only the difference to the first sample,
    // which may be negative, is kept, in microseconds.
    uint32_t sentTs = tcb->m_rcvTimestampEchoReply;
    int64_t sentUs = 0;
    if (m_ackHoldFilter && m_rateOps)
    {
        // The most recently sent segment this ACK delivers: the receiver
        // sends the ACK on its arrival, or on the expiry of its delayed ACK
        // timer
        Time sent = m_rateOps->GetConnectionRate().m_firstSentTime;
        sentTs = static_cast<uint32_t>(sent.GetMilliSeconds());
        sentUs = sent.GetMicroSeconds() % 1000;
    }
    uint32_t raw = tcb->m_rcvTimestampValue - sentTs;
    if (!m_haveOwd)
    {
        m_owdRef = raw;
    }
    return static_cast<int64_t>(static_cast<int32_t>(raw - m_owdRef)) * 1000 - sentUs;
}

bool
TcpLp::RttSample(Ptr<TcpSocketState> tcb, uint32_t bytesAcked)
{
    NS_LOG_FUNCTION(this << tcb << bytesAcked);

    if (m_ackHoldFilter)
    {
        // A receiver which delays its ACKs acknowledges a single segment
        // only when its delayed ACK timer expires: the one way delay then
        // includes the time the segment waited for the timer
        if (bytesAcked > tcb->m_segmentSize)
        {
            m_delayedAcks = true;
        }
        else if (m_delayedAcks)
        {
            NS_LOG_DEBUG("Ignore the one way delay of the ACK of a single segment");
            return false;
        }
    }

    int64_t owd = OwdCalculator(tcb);

    if (!m_haveOwd)
    {
        m_haveOwd = true;
        m_dMin = owd;
        m_dMax = owd;
        m_sd8 = owd * 8;
        return true;
    }

    m_dMin = std::min(m_dMin, owd);
    m_dMax = std::max(m_dMax, owd);
    if (m_restartSmoothing && m_restartPending && tcb->m_lastAckedSeq > m_reactionSeq)
    {
        // The first sample of the data sent after the last reaction
        m_restartPending = false;
        m_sd8 = owd * 8;
        return true;
    }
    // sd = 7/8 sd + 1/8 owd
    m_sd8 += owd - m_sd8 / 8;
    return true;
}

uint32_t
TcpLp::GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
    NS_LOG_FUNCTION(this << tcb << bytesInFlight);

    if (m_linuxArithmetic)
    {
        // tcp_reno_ssthresh(): half the window, not the bytes in flight,
        // which differ once TCP-LP has cut the window
        return std::max(tcb->GetCwndInSegments() >> 1U, 2U) * tcb->m_segmentSize;
    }
    // The delay indications are used from the first loss on
    m_lossSeen = true;
    m_reactionSeq = tcb->m_highTxMark;
    m_reacted = true;
    m_restartPending = true;
    return TcpNewReno::GetSsThresh(tcb, bytesInFlight);
}

void
TcpLp::IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);

    // No increase within the inference, in slow start too (as
    // tcp_lp_cong_avoid() of Linux)
    if (m_flag & LP_WITHIN_INF)
    {
        return;
    }
    TcpNewReno::IncreaseWindow(tcb, segmentsAcked);
}

/*
 * The functions below follow tcp_lp.c of Linux 6.12 statement by statement,
 * including the types of its operands: the products and differences of
 * 32-bit unsigned values wrap around, as they do in C. The timestamps are the
 * raw ones of the segments, the echoed one including the offset of this end,
 * and the local clock (tcp_time_stamp_ts()) does not include it.
 */
uint32_t
TcpLp::LinuxRemoteHzEstimator(Ptr<TcpSocketState> tcb)
{
    const uint32_t tsHz = 1000; // TCP_TS_HZ
    uint32_t rcvTsval = tcb->m_rcvTimestampValue;
    uint32_t rcvTsecr = tcb->m_rcvTimestampEchoReply + tcb->m_timestampOffset;
    auto rhz = static_cast<int64_t>(static_cast<uint32_t>(m_remoteHz << 6));
    int64_t m = 0;

    if (m_remoteRefTime != 0 && m_localRefTime != 0 && rcvTsval != m_remoteRefTime &&
        rcvTsecr != m_localRefTime)
    {
        m = static_cast<uint32_t>(tsHz * (rcvTsval - m_remoteRefTime)) /
            static_cast<uint32_t>(rcvTsecr - m_localRefTime);
        if (rhz > 0)
        {
            m -= rhz >> 6;
            rhz += m;
        }
        else
        {
            rhz = m << 6;
        }
    }

    if ((rhz >> 6) > 0)
    {
        m_flag |= LP_VALID_RHZ;
    }
    else
    {
        m_flag &= ~LP_VALID_RHZ;
    }

    m_remoteRefTime = rcvTsval;
    m_localRefTime = rcvTsecr;

    return static_cast<uint32_t>(rhz >> 6);
}

uint32_t
TcpLp::LinuxOwdCalculator(Ptr<TcpSocketState> tcb)
{
    const uint32_t lpResol = 1000; // LP_RESOL, TCP_TS_HZ
    uint32_t rcvTsval = tcb->m_rcvTimestampValue;
    uint32_t rcvTsecr = tcb->m_rcvTimestampEchoReply + tcb->m_timestampOffset;
    int64_t owd = 0;

    m_remoteHz = LinuxRemoteHzEstimator(tcb);

    // A null rate after the 32-bit truncation would make Linux divide by zero
    if ((m_flag & LP_VALID_RHZ) && m_remoteHz != 0)
    {
        owd =
            static_cast<uint32_t>(rcvTsval * (lpResol / m_remoteHz) - rcvTsecr * (lpResol / 1000));
    }

    if (owd > 0)
    {
        m_flag |= LP_VALID_OWD;
    }
    else
    {
        m_flag &= ~LP_VALID_OWD;
    }
    return static_cast<uint32_t>(owd);
}

void
TcpLp::LinuxRttSample(Ptr<TcpSocketState> tcb)
{
    int64_t mowd = LinuxOwdCalculator(tcb);

    if (!(m_flag & LP_VALID_RHZ) || !(m_flag & LP_VALID_OWD))
    {
        return;
    }

    if (mowd < m_owdMin)
    {
        m_owdMin = static_cast<uint32_t>(mowd);
    }

    if (mowd > m_owdMax)
    {
        if (mowd > m_owdMaxRsv)
        {
            if (m_owdMaxRsv == 0)
            {
                m_owdMax = static_cast<uint32_t>(mowd);
            }
            else
            {
                m_owdMax = m_owdMaxRsv;
            }
            m_owdMaxRsv = static_cast<uint32_t>(mowd);
        }
        else
        {
            m_owdMax = static_cast<uint32_t>(mowd);
        }
    }

    if (m_sOwd != 0)
    {
        mowd -= m_sOwd >> 3;
        m_sOwd += static_cast<uint32_t>(mowd);
    }
    else
    {
        m_sOwd = static_cast<uint32_t>(mowd << 3);
    }
}

void
TcpLp::LinuxPktsAcked(Ptr<TcpSocketState> tcb, const Time& rtt)
{
    uint32_t now = TcpOptionTS::NowToTsValue();
    uint32_t rcvTsecr = tcb->m_rcvTimestampEchoReply + tcb->m_timestampOffset;

    if (rtt.IsStrictlyPositive())
    {
        LinuxRttSample(tcb);
    }

    uint32_t delta = now - rcvTsecr;
    if (static_cast<int32_t>(delta) > 0)
    {
        m_inferenceTs = 3 * delta;
    }

    if (m_lastDropTs != 0 && (now - m_lastDropTs < m_inferenceTs))
    {
        m_flag |= LP_WITHIN_INF;
    }
    else
    {
        m_flag &= ~LP_WITHIN_INF;
    }

    if ((m_sOwd >> 3) < m_owdMin + 15 * (m_owdMax - m_owdMin) / 100)
    {
        m_flag |= LP_WITHIN_THR;
    }
    else
    {
        m_flag &= ~LP_WITHIN_THR;
    }

    NS_LOG_DEBUG("flag " << m_flag << " cwnd " << tcb->GetCwndInSegments() << " remoteHz "
                         << m_remoteHz << " owdMin " << m_owdMin << " owdMax " << m_owdMax
                         << " sowd " << (m_sOwd >> 3));

    if (m_flag & LP_WITHIN_THR)
    {
        return;
    }

    m_owdMin = m_sOwd >> 3;
    m_owdMax = m_sOwd >> 2;
    m_owdMaxRsv = m_sOwd >> 2;

    if (m_flag & LP_WITHIN_INF)
    {
        tcb->m_cWnd = tcb->m_segmentSize;
    }
    else
    {
        tcb->m_cWnd = std::max(tcb->GetCwndInSegments() >> 1U, 1U) * tcb->m_segmentSize;
    }

    m_lastDropTs = now;
}

void
TcpLp::PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked << rtt);

    if (m_linuxArithmetic)
    {
        LinuxPktsAcked(tcb, rtt);
        return;
    }

    Time now = Simulator::Now();

    // Only the ACKs of new data are used: a duplicate ACK echoes the
    // timestamp of the last segment received in order
    if (m_haveAck && tcb->m_lastAckedSeq <= m_lastAck)
    {
        return;
    }
    uint32_t bytesAcked =
        m_haveAck ? tcb->m_lastAckedSeq - m_lastAck : segmentsAcked * tcb->m_segmentSize;
    m_lastAck = tcb->m_lastAckedSeq;
    m_haveAck = true;

    // The ACKs of a loss recovery acknowledge retransmissions, and the
    // duplicate ACKs before it echo old timestamps
    bool sampled = false;
    if (!rtt.IsZero() && (tcb->m_congState == TcpSocketState::CA_OPEN ||
                          tcb->m_congState == TcpSocketState::CA_DISORDER))
    {
        sampled = RttSample(tcb, bytesAcked);
    }

    Time rttEst = tcb->m_srtt.Get().IsZero() ? rtt : tcb->m_srtt.Get();

    // The inference time-out is three round trip times
    m_inference = 3 * rttEst;

    if (!m_lastDrop.IsZero() && (now - m_lastDrop < m_inference))
    {
        m_flag |= LP_WITHIN_INF;
    }
    else
    {
        m_flag &= ~LP_WITHIN_INF;
    }

    // Early congestion indication: sd > dMin + 0.15 (dMax - dMin), where the
    // margin above the minimum is at least the granularity of the
    // timestamps of the receiver (1 ms), below which the delays are not
    // measured
    if (!m_haveOwd || m_sd8 <= 8 * m_dMin + std::max<int64_t>(120 * (m_dMax - m_dMin) / 100, 8000))
    {
        m_flag |= LP_WITHIN_THR;
    }
    else
    {
        m_flag &= ~LP_WITHIN_THR;
    }

    NS_LOG_DEBUG("owd min " << m_dMin << " max " << m_dMax << " smoothed " << m_sd8 / 8.0
                            << " flag " << m_flag << " cwnd " << tcb->GetCwndInSegments());

    // The minimum and maximum delays measured in slow start are used from
    // the first loss on, and only an ACK with a new delay sample indicates
    // congestion
    if ((m_flag & LP_WITHIN_THR) || !m_lossSeen || !sampled)
    {
        return;
    }

    // The loss recovery sets the window; TCP-LP reacts at most once per round
    // trip time, to a delay indication or to a loss: as ECN (RFC 3168), once
    // per window of data, until the data sent before the last reaction is
    // acknowledged
    if (tcb->m_congState == TcpSocketState::CA_RECOVERY ||
        tcb->m_congState == TcpSocketState::CA_LOSS ||
        (m_reacted && tcb->m_lastAckedSeq <= m_reactionSeq))
    {
        return;
    }

    if (m_flag & LP_WITHIN_INF)
    {
        // A second indication within the inference: window of one segment
        tcb->m_cWnd = tcb->m_segmentSize;
    }
    else
    {
        tcb->m_cWnd = std::max(tcb->m_cWnd.Get() >> 1U, tcb->m_segmentSize);
    }
    NS_LOG_DEBUG("Delay indication, cwnd " << tcb->GetCwndInSegments());

    m_lastDrop = now;
    m_reactionSeq = tcb->m_highTxMark;
    m_reacted = true;
    m_restartPending = true;
    m_flag |= LP_WITHIN_INF;
}

std::string
TcpLp::GetName() const
{
    return "TcpLp";
}
} // namespace ns3
