/*
 * Copyright (c) 2016 ResiliNets, ITTC, University of Kansas
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Truc Anh N. Nguyen <annguyen@ittc.ku.edu>
 *          Keerthi Ganta <keerthig@ittc.ku.edu>
 *          Md. Moshfequr Rahman <moshfequr@ittc.ku.edu>
 *          Amir Modarresi <amodarresi@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  https://resilinets.org/
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 */

#include "tcp-scalable.h"

#include "tcp-socket-state.h"

#include "ns3/log.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TcpScalable");
NS_OBJECT_ENSURE_REGISTERED(TcpScalable);

TypeId
TcpScalable::GetTypeId()
{
    static TypeId tid = TypeId("ns3::TcpScalable")
                            .SetParent<TcpNewReno>()
                            .AddConstructor<TcpScalable>()
                            .SetGroupName("Internet")
                            .AddAttribute("AIFactor",
                                          "Additive Increase Factor",
                                          UintegerValue(100),
                                          MakeUintegerAccessor(&TcpScalable::m_aiFactor),
                                          MakeUintegerChecker<uint32_t>())
                            .AddAttribute("MDFactor",
                                          "Multiplicative Decrease Factor",
                                          DoubleValue(0.125),
                                          MakeDoubleAccessor(&TcpScalable::m_mdFactor),
                                          MakeDoubleChecker<double>())
                            .AddAttribute("LinuxDecrease",
                                          "Compute the multiplicative decrease as Linux does: "
                                          "the decrease, rounded down, is subtracted from cWnd, "
                                          "instead of scaling the bytes in flight",
                                          BooleanValue(false),
                                          MakeBooleanAccessor(&TcpScalable::m_linuxDecrease),
                                          MakeBooleanChecker());
    return tid;
}

TcpScalable::TcpScalable()
    : TcpNewReno(),
      m_aiFactor(50),
      m_mdFactor(0.125)
{
    NS_LOG_FUNCTION(this);
}

TcpScalable::TcpScalable(const TcpScalable& sock)
    : TcpNewReno(sock),
      m_aiFactor(sock.m_aiFactor),
      m_mdFactor(sock.m_mdFactor),
      m_linuxDecrease(sock.m_linuxDecrease)
{
    NS_LOG_FUNCTION(this);
}

TcpScalable::~TcpScalable()
{
    NS_LOG_FUNCTION(this);
}

Ptr<TcpCongestionOps>
TcpScalable::Fork()
{
    return CopyObject<TcpScalable>(this);
}

void
TcpScalable::CongestionAvoidance(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);

    uint32_t segCwnd = tcb->GetCwndInSegments();
    NS_ASSERT(segCwnd >= 1);

    // One segment of growth every min(cwnd, AIFactor) acknowledged segments,
    // with the count cleared when the window is reduced, as Linux
    // tcp_cong_avoid_ai() on snd_cwnd_cnt
    CountedIncrease(tcb, segmentsAcked, std::min(segCwnd, m_aiFactor));
}

std::string
TcpScalable::GetName() const
{
    return "TcpScalable";
}

uint32_t
TcpScalable::GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
    NS_LOG_FUNCTION(this << tcb << bytesInFlight);

    uint32_t segCwnd = bytesInFlight / tcb->m_segmentSize;

    if (m_linuxDecrease)
    {
        // tcp_scalable_ssthresh(): cwnd - (cwnd >> 3)
        segCwnd = tcb->GetCwndInSegments();
        auto decrease = static_cast<uint32_t>(segCwnd * m_mdFactor);
        return std::max(segCwnd - decrease, 2U) * tcb->m_segmentSize;
    }

    double b = 1.0 - m_mdFactor;
    uint32_t ssThresh = static_cast<uint32_t>(std::max(2.0, segCwnd * b));

    NS_LOG_DEBUG("Calculated b(w) = " << b << " resulting (in segment) ssThresh=" << ssThresh);

    return ssThresh * tcb->m_segmentSize;
}

} // namespace ns3
