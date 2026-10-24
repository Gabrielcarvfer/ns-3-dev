/*
 * Copyright (c) 2017 NITK Surathkal
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Shravya K.S. <shravya.ks0@gmail.com>
 *
 */

#include "tcp-dctcp.h"

#include "tcp-socket-state.h"

#include "ns3/abort.h"
#include "ns3/log.h"

#include <cmath>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TcpDctcp");

NS_OBJECT_ENSURE_REGISTERED(TcpDctcp);

TypeId
TcpDctcp::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::TcpDctcp")
            .SetParent<TcpLinuxReno>()
            .AddConstructor<TcpDctcp>()
            .SetGroupName("Internet")
            .AddAttribute("DctcpShiftG",
                          "Parameter G for updating dctcp_alpha",
                          DoubleValue(0.0625),
                          MakeDoubleAccessor(&TcpDctcp::m_g),
                          MakeDoubleChecker<double>(0, 1))
            .AddAttribute("DctcpAlphaOnInit",
                          "Initial alpha value",
                          DoubleValue(1.0),
                          MakeDoubleAccessor(&TcpDctcp::InitializeDctcpAlpha),
                          MakeDoubleChecker<double>(0, 1))
            .AddAttribute("UseEct0",
                          "Use ECT(0) for ECN codepoint, if false use ECT(1)",
                          BooleanValue(true),
                          MakeBooleanAccessor(&TcpDctcp::m_useEct0),
                          MakeBooleanChecker())
            .AddAttribute("LinuxArithmetic",
                          "Compute as the tcp_dctcp.c of Linux does: alpha in 1/1024, its "
                          "decay by a shift of DctcpShiftG (down to zero once smaller than "
                          "the shift), the window reductions in whole segments, the one on a "
                          "loss from the window (max(cwnd >> 1, 2)) rather than from the "
                          "FlightSize, and the observation window ending at the highest "
                          "sequence number sent",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpDctcp::m_linuxArithmetic),
                          MakeBooleanChecker())
            .AddTraceSource("CongestionEstimate",
                            "Update sender-side congestion estimate state",
                            MakeTraceSourceAccessor(&TcpDctcp::m_traceCongestionEstimate),
                            "ns3::TcpDctcp::CongestionEstimateTracedCallback");
    return tid;
}

std::string
TcpDctcp::GetName() const
{
    return "TcpDctcp";
}

TcpDctcp::TcpDctcp()
    : TcpLinuxReno(),
      m_ackedBytesEcn(0),
      m_ackedBytesTotal(0),
      m_priorRcvNxt(SequenceNumber32(0)),
      m_priorRcvNxtFlag(false),
      m_nextSeq(SequenceNumber32(0)),
      m_nextSeqFlag(false),
      m_ceState(false),
      m_delayedAckReserved(false),
      m_initialized(false)
{
    NS_LOG_FUNCTION(this);
}

TcpDctcp::TcpDctcp(const TcpDctcp& sock)
    : TcpLinuxReno(sock),
      m_ackedBytesEcn(sock.m_ackedBytesEcn),
      m_ackedBytesTotal(sock.m_ackedBytesTotal),
      m_priorRcvNxt(sock.m_priorRcvNxt),
      m_priorRcvNxtFlag(sock.m_priorRcvNxtFlag),
      m_alpha(sock.m_alpha),
      m_nextSeq(sock.m_nextSeq),
      m_nextSeqFlag(sock.m_nextSeqFlag),
      m_ceState(sock.m_ceState),
      m_delayedAckReserved(sock.m_delayedAckReserved),
      m_g(sock.m_g),
      m_useEct0(sock.m_useEct0),
      m_initialized(sock.m_initialized),
      m_linuxArithmetic(sock.m_linuxArithmetic)
{
    NS_LOG_FUNCTION(this);
}

TcpDctcp::~TcpDctcp()
{
    NS_LOG_FUNCTION(this);
}

Ptr<TcpCongestionOps>
TcpDctcp::Fork()
{
    NS_LOG_FUNCTION(this);
    return CopyObject<TcpDctcp>(this);
}

void
TcpDctcp::Init(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);
    NS_LOG_INFO(this << "Enabling DctcpEcn for DCTCP");
    tcb->m_useEcn = TcpSocketState::On;
    tcb->m_ecnMode = TcpSocketState::DctcpEcn;
    tcb->m_ectCodePoint = m_useEct0 ? TcpSocketState::Ect0 : TcpSocketState::Ect1;
    SetSuppressIncreaseIfCwndLimited(false);
    m_initialized = true;
}

// Step 9, Section 3.3 of RFC 8257.  GetSsThresh() is called upon
// entering the CWR state, and then later, when CWR is exited,
// cwnd is set to ssthresh (this value).
//
// @RFC{8257} Section 3.5 ("Handling of Packet Loss"): a DCTCP sender MUST react
// to a loss episode (a loss inferred implicitly, not signalled by ECN) in the same
// way as a conventional TCP, i.e. reduce ssthresh as @RFC{5681} does. With non-ECN
// routers, m_alpha decays toward zero, so the DCTCP reduction below would leave
// ssthresh at ~cwnd on a packet loss (no back-off). Distinguish the congestion
// event type: the ECN-driven DCTCP reduction (in which case bytesInFlight is
// ignored) applies only to the ECN Echo which makes the socket enter CA_CWR;
// a fast retransmission (the socket is already in CA_RECOVERY) or a
// retransmission timeout (the socket does not hold an ECN Echo) applies the
// @RFC{5681} reduction, as Linux dctcp_react_to_loss() does.
uint32_t
TcpDctcp::GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
    NS_LOG_FUNCTION(this << tcb << bytesInFlight);
    bool ecnEcho = tcb->m_ecnState == TcpSocketState::ECN_ECE_RCVD &&
                   tcb->m_congState != TcpSocketState::CA_RECOVERY &&
                   tcb->m_congState != TcpSocketState::CA_LOSS;
    if (m_linuxArithmetic)
    {
        // dctcp_react_to_loss() and dctcp_ssthresh(), in whole segments
        uint32_t segCwnd = tcb->GetCwndInSegments();
        uint32_t ssThresh =
            ecnEcho ? segCwnd - ((segCwnd * static_cast<uint32_t>(m_alpha * 1024)) >> 11)
                    : segCwnd >> 1;
        return std::max(ssThresh, 2U) * tcb->m_segmentSize;
    }
    if (!ecnEcho)
    {
        return std::max(bytesInFlight / 2, 2 * tcb->m_segmentSize);
    }
    return static_cast<uint32_t>((1 - m_alpha / 2.0) * tcb->m_cWnd);
}

void
TcpDctcp::PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked << rtt);
    m_ackedBytesTotal += segmentsAcked * tcb->m_segmentSize;
    if (tcb->m_ecnState == TcpSocketState::ECN_ECE_RCVD)
    {
        m_ackedBytesEcn += segmentsAcked * tcb->m_segmentSize;
    }
    if (!m_nextSeqFlag)
    {
        m_nextSeq = m_linuxArithmetic ? tcb->m_highTxMark.Get() : tcb->m_nextTxSequence.Get();
        m_nextSeqFlag = true;
    }
    if (m_linuxArithmetic && tcb->m_lastAckedSeq >= m_nextSeq)
    {
        // dctcp_update_alpha(), alpha in 1/1024
        auto alpha = static_cast<uint32_t>(m_alpha * 1024);
        auto shiftG = static_cast<uint32_t>(std::lround(-std::log2(m_g)));
        uint32_t decay = alpha >> shiftG;
        alpha -= (decay == 0 ? alpha : decay);
        uint32_t delivered = m_ackedBytesTotal / tcb->m_segmentSize;
        uint32_t deliveredCe = m_ackedBytesEcn / tcb->m_segmentSize;
        if (deliveredCe > 0)
        {
            deliveredCe <<= (10 - shiftG);
            deliveredCe /= std::max(1U, delivered);
            alpha = std::min(alpha + deliveredCe, 1024U);
        }
        m_alpha = alpha / 1024.0;
        m_traceCongestionEstimate(m_ackedBytesEcn, m_ackedBytesTotal, m_alpha);
        NS_LOG_INFO(this << " m_alpha " << m_alpha);
        Reset(tcb);
    }
    else if (tcb->m_lastAckedSeq >= m_nextSeq)
    {
        double bytesEcn = 0.0; // Corresponds to variable M in RFC 8257
        if (m_ackedBytesTotal > 0)
        {
            bytesEcn = static_cast<double>(m_ackedBytesEcn * 1.0 / m_ackedBytesTotal);
        }
        m_alpha = (1.0 - m_g) * m_alpha + m_g * bytesEcn;
        m_traceCongestionEstimate(m_ackedBytesEcn, m_ackedBytesTotal, m_alpha);
        NS_LOG_INFO(this << "bytesEcn " << bytesEcn << ", m_alpha " << m_alpha);
        Reset(tcb);
    }
}

void
TcpDctcp::InitializeDctcpAlpha(double alpha)
{
    NS_LOG_FUNCTION(this << alpha);
    NS_ABORT_MSG_IF(m_initialized, "DCTCP has already been initialized");
    m_alpha = alpha;
}

void
TcpDctcp::Reset(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);
    m_nextSeq = m_linuxArithmetic ? tcb->m_highTxMark.Get() : tcb->m_nextTxSequence.Get();
    m_ackedBytesEcn = 0;
    m_ackedBytesTotal = 0;
}

void
TcpDctcp::CeState0to1(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);
    if (!m_ceState && m_delayedAckReserved && m_priorRcvNxtFlag)
    {
        SequenceNumber32 tmpRcvNxt;
        /* Save current NextRxSequence. */
        tmpRcvNxt = tcb->m_rxBuffer->NextRxSequence();

        /* Generate previous ACK without ECE */
        tcb->m_rxBuffer->SetNextRxSequence(m_priorRcvNxt);
        tcb->m_sendEmptyPacketCallback(TcpHeader::ACK);

        /* Recover current RcvNxt. */
        tcb->m_rxBuffer->SetNextRxSequence(tmpRcvNxt);
    }

    if (!m_priorRcvNxtFlag)
    {
        m_priorRcvNxtFlag = true;
    }
    m_priorRcvNxt = tcb->m_rxBuffer->NextRxSequence();
    m_ceState = true;
    tcb->m_ecnState = TcpSocketState::ECN_CE_RCVD;
}

void
TcpDctcp::CeState1to0(Ptr<TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);
    if (m_ceState && m_delayedAckReserved && m_priorRcvNxtFlag)
    {
        SequenceNumber32 tmpRcvNxt;
        /* Save current NextRxSequence. */
        tmpRcvNxt = tcb->m_rxBuffer->NextRxSequence();

        /* Generate previous ACK with ECE */
        tcb->m_rxBuffer->SetNextRxSequence(m_priorRcvNxt);
        tcb->m_sendEmptyPacketCallback(TcpHeader::ACK | TcpHeader::ECE);

        /* Recover current RcvNxt. */
        tcb->m_rxBuffer->SetNextRxSequence(tmpRcvNxt);
    }

    if (!m_priorRcvNxtFlag)
    {
        m_priorRcvNxtFlag = true;
    }
    m_priorRcvNxt = tcb->m_rxBuffer->NextRxSequence();
    m_ceState = false;

    if (tcb->m_ecnState.Get() == TcpSocketState::ECN_CE_RCVD ||
        tcb->m_ecnState.Get() == TcpSocketState::ECN_SENDING_ECE)
    {
        tcb->m_ecnState = TcpSocketState::ECN_IDLE;
    }
}

void
TcpDctcp::UpdateAckReserved(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event)
{
    NS_LOG_FUNCTION(this << tcb << event);
    switch (event)
    {
    case TcpSocketState::CA_EVENT_DELAYED_ACK:
        if (!m_delayedAckReserved)
        {
            m_delayedAckReserved = true;
        }
        break;
    case TcpSocketState::CA_EVENT_NON_DELAYED_ACK:
        if (m_delayedAckReserved)
        {
            m_delayedAckReserved = false;
        }
        break;
    default:
        /* Don't care for the rest. */
        break;
    }
}

void
TcpDctcp::CwndEvent(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event)
{
    NS_LOG_FUNCTION(this << tcb << event);
    switch (event)
    {
    case TcpSocketState::CA_EVENT_ECN_IS_CE:
        CeState0to1(tcb);
        break;
    case TcpSocketState::CA_EVENT_ECN_NO_CE:
        CeState1to0(tcb);
        break;
    case TcpSocketState::CA_EVENT_DELAYED_ACK:
    case TcpSocketState::CA_EVENT_NON_DELAYED_ACK:
        UpdateAckReserved(tcb, event);
        break;
    default:
        /* Don't care for the rest. */
        break;
    }
}

} // namespace ns3
