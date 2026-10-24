/*
 * Copyright (c) 2014 Natale Patriciello, <natale.patriciello@gmail.com>
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#include "tcp-highspeed.h"

#include "tcp-socket-state.h"

#include "ns3/boolean.h"
#include "ns3/log.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TcpHighSpeed");
NS_OBJECT_ENSURE_REGISTERED(TcpHighSpeed);

TypeId
TcpHighSpeed::GetTypeId()
{
    static TypeId tid = TypeId("ns3::TcpHighSpeed")
                            .SetParent<TcpNewReno>()
                            .AddConstructor<TcpHighSpeed>()
                            .SetGroupName("Internet")
                            .AddAttribute("LinuxDecrease",
                                          "Compute the multiplicative decrease as Linux does: "
                                          "b(w) of the window at the last congestion avoidance "
                                          "increase (0.5 when the loss ends slow start), applied "
                                          "to cWnd instead of the bytes in flight",
                                          BooleanValue(false),
                                          MakeBooleanAccessor(&TcpHighSpeed::m_linuxDecrease),
                                          MakeBooleanChecker());
    return tid;
}

TcpHighSpeed::TcpHighSpeed()
    : TcpNewReno(),
      m_ackCnt(0)
{
    NS_LOG_FUNCTION(this);
}

TcpHighSpeed::TcpHighSpeed(const TcpHighSpeed& sock)
    : TcpNewReno(sock),
      m_ackCnt(sock.m_ackCnt),
      m_caCwnd(sock.m_caCwnd),
      m_linuxDecrease(sock.m_linuxDecrease)
{
    NS_LOG_FUNCTION(this);
}

TcpHighSpeed::~TcpHighSpeed()
{
    NS_LOG_FUNCTION(this);
}

Ptr<TcpCongestionOps>
TcpHighSpeed::Fork()
{
    return CopyObject<TcpHighSpeed>(this);
}

/**
 * @brief Congestion avoidance of TcpHighSpeed
 *
 * RFC 3649 increases the congestion window, in response to a single
 * acknowledgement, as follows:
 *
 *               w = w + a(w)/w
 *
 * As implementation choice, we increment cWnd only by MSS, when the right
 * number of ACKs has been received. Each ACK adds a(w) to a counter (m_ackCnt),
 * and the window grows by one segment each time the counter reaches w: with
 * a(w) equal to 1 this is the classical NewReno increase of 1/w per ACK, and
 * with a(w) equal to 2 one ACK weighs as much as two NewReno ACKs.
 *
 * The increase is per ACK, whatever the number of segments the ACK covers,
 * as in RFC 3649 and in Linux (tcp_highspeed.c), also with the
 * CountAckedSegments attribute set: with delayed ACKs, the window grows by
 * about a(w)/2 segments per round-trip time.
 *
 * @param tcb internal congestion state
 * @param segmentsAcked count of segments acked
 */
void
TcpHighSpeed::CongestionAvoidance(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);

    uint32_t segCwnd = tcb->GetCwndInSegments();
    uint32_t oldCwnd = segCwnd;

    if (segmentsAcked > 0)
    {
        m_ackCnt += TableLookupA(segCwnd);
        m_caCwnd = segCwnd;
    }

    while (m_ackCnt >= segCwnd)
    {
        m_ackCnt -= segCwnd;
        segCwnd += 1;
    }

    if (segCwnd != oldCwnd)
    {
        tcb->m_cWnd = segCwnd * tcb->m_segmentSize;
        NS_LOG_INFO("In CongAvoid, updated to cwnd " << tcb->m_cWnd << " ssthresh "
                                                     << tcb->m_ssThresh);
    }
}

void
TcpHighSpeed::CongestionStateSet(Ptr<TcpSocketState> tcb,
                                 const TcpSocketState::TcpCongState_t newState)
{
    NS_LOG_FUNCTION(this << tcb << newState);
    TcpNewReno::CongestionStateSet(tcb, newState);
    // The increments counted towards the next segment of growth are
    // discarded when the window is reduced, as tp->snd_cwnd_cnt in Linux
    // (tcp_init_cwnd_reduction() and tcp_enter_loss())
    if (newState == TcpSocketState::CA_RECOVERY || newState == TcpSocketState::CA_CWR ||
        newState == TcpSocketState::CA_LOSS)
    {
        m_ackCnt = 0;
    }
}

std::string
TcpHighSpeed::GetName() const
{
    return "TcpHighSpeed";
}

/**
 * @brief Get slow start threshold following HighSpeed principles
 *
 * @param tcb internal congestion state
 * @param bytesInFlight Bytes in flight
 *
 * @return the slow start threshold value
 */
uint32_t
TcpHighSpeed::GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
    NS_LOG_FUNCTION(this << tcb << bytesInFlight);

    uint32_t segCwnd = bytesInFlight / tcb->m_segmentSize;
    uint32_t bWindow = segCwnd;

    if (m_linuxDecrease)
    {
        // The window that sets b(w) is the one of the last increase in
        // congestion avoidance (none when the loss ends slow start, so
        // b(w) = 0.5), and the decrease applies to cWnd
        segCwnd = tcb->GetCwndInSegments();
        bWindow = m_caCwnd;
    }

    if (m_linuxDecrease)
    {
        // hstcp_ssthresh(): cwnd - ((cwnd * md) >> 8)
        return std::max(segCwnd - ((segCwnd * TableLookupLinuxB(bWindow)) >> 8), 2U) *
               tcb->m_segmentSize;
    }

    double b = 1.0 - TableLookupB(bWindow);
    uint32_t ssThresh = static_cast<uint32_t>(std::max(2.0, segCwnd * b));

    NS_LOG_DEBUG("Calculated b(w) = " << b << " resulting (in segment) ssThresh=" << ssThresh);

    return ssThresh * tcb->m_segmentSize;
}

uint32_t
TcpHighSpeed::TableLookupA(uint32_t w)
{
    if (w <= 38)
    {
        return 1;
    }
    else if (w <= 118)
    {
        return 2;
    }
    else if (w <= 221)
    {
        return 3;
    }
    else if (w <= 347)
    {
        return 4;
    }
    else if (w <= 495)
    {
        return 5;
    }
    else if (w <= 663)
    {
        return 6;
    }
    else if (w <= 851)
    {
        return 7;
    }
    else if (w <= 1058)
    {
        return 8;
    }
    else if (w <= 1284)
    {
        return 9;
    }
    else if (w <= 1529)
    {
        return 10;
    }
    else if (w <= 1793)
    {
        return 11;
    }
    else if (w <= 2076)
    {
        return 12;
    }
    else if (w <= 2378)
    {
        return 13;
    }
    else if (w <= 2699)
    {
        return 14;
    }
    else if (w <= 3039)
    {
        return 15;
    }
    else if (w <= 3399)
    {
        return 16;
    }
    else if (w <= 3778)
    {
        return 17;
    }
    else if (w <= 4177)
    {
        return 18;
    }
    else if (w <= 4596)
    {
        return 19;
    }
    else if (w <= 5036)
    {
        return 20;
    }
    else if (w <= 5497)
    {
        return 21;
    }
    else if (w <= 5979)
    {
        return 22;
    }
    else if (w <= 6483)
    {
        return 23;
    }
    else if (w <= 7009)
    {
        return 24;
    }
    else if (w <= 7558)
    {
        return 25;
    }
    else if (w <= 8130)
    {
        return 26;
    }
    else if (w <= 8726)
    {
        return 27;
    }
    else if (w <= 9346)
    {
        return 28;
    }
    else if (w <= 9991)
    {
        return 29;
    }
    else if (w <= 10661)
    {
        return 30;
    }
    else if (w <= 11358)
    {
        return 31;
    }
    else if (w <= 12082)
    {
        return 32;
    }
    else if (w <= 12834)
    {
        return 33;
    }
    else if (w <= 13614)
    {
        return 34;
    }
    else if (w <= 14424)
    {
        return 35;
    }
    else if (w <= 15265)
    {
        return 36;
    }
    else if (w <= 16137)
    {
        return 37;
    }
    else if (w <= 17042)
    {
        return 38;
    }
    else if (w <= 17981)
    {
        return 39;
    }
    else if (w <= 18955)
    {
        return 40;
    }
    else if (w <= 19965)
    {
        return 41;
    }
    else if (w <= 21013)
    {
        return 42;
    }
    else if (w <= 22101)
    {
        return 43;
    }
    else if (w <= 23230)
    {
        return 44;
    }
    else if (w <= 24402)
    {
        return 45;
    }
    else if (w <= 25618)
    {
        return 46;
    }
    else if (w <= 26881)
    {
        return 47;
    }
    else if (w <= 28193)
    {
        return 48;
    }
    else if (w <= 29557)
    {
        return 49;
    }
    else if (w <= 30975)
    {
        return 50;
    }
    else if (w <= 32450)
    {
        return 51;
    }
    else if (w <= 33986)
    {
        return 52;
    }
    else if (w <= 35586)
    {
        return 53;
    }
    else if (w <= 37253)
    {
        return 54;
    }
    else if (w <= 38992)
    {
        return 55;
    }
    else if (w <= 40808)
    {
        return 56;
    }
    else if (w <= 42707)
    {
        return 57;
    }
    else if (w <= 44694)
    {
        return 58;
    }
    else if (w <= 46776)
    {
        return 59;
    }
    else if (w <= 48961)
    {
        return 60;
    }
    else if (w <= 51258)
    {
        return 61;
    }
    else if (w <= 53667)
    {
        return 62;
    }
    else if (w <= 56230)
    {
        return 63;
    }
    else if (w <= 58932)
    {
        return 64;
    }
    else if (w <= 61799)
    {
        return 65;
    }
    else if (w <= 64851)
    {
        return 66;
    }
    else if (w <= 68113)
    {
        return 67;
    }
    else if (w <= 71617)
    {
        return 68;
    }
    else if (w <= 75401)
    {
        return 69;
    }
    else if (w <= 79517)
    {
        return 70;
    }
    else if (w <= 84035)
    {
        return 71;
    }
    else if (w <= 89053)
    {
        return 72;
    }
    else if (w <= 94717)
    {
        return 73;
    }

    return 73;
}

double
TcpHighSpeed::TableLookupB(uint32_t w)
{
    // NOLINTBEGIN(bugprone-branch-clone)
    if (w <= 38)
    {
        return 0.50;
    }
    else if (w <= 118)
    {
        return 0.44;
    }
    else if (w <= 221)
    {
        return 0.41;
    }
    else if (w <= 347)
    {
        return 0.38;
    }
    else if (w <= 495)
    {
        return 0.37;
    }
    else if (w <= 663)
    {
        return 0.35;
    }
    else if (w <= 851)
    {
        return 0.34;
    }
    else if (w <= 1058)
    {
        return 0.33;
    }
    else if (w <= 1284)
    {
        return 0.32;
    }
    else if (w <= 1529)
    {
        return 0.31;
    }
    else if (w <= 1793)
    {
        return 0.30;
    }
    else if (w <= 2076)
    {
        return 0.29;
    }
    else if (w <= 2378)
    {
        return 0.28;
    }
    else if (w <= 2699)
    {
        return 0.28;
    }
    else if (w <= 3039)
    {
        return 0.27;
    }
    else if (w <= 3399)
    {
        return 0.27;
    }
    else if (w <= 3778)
    {
        return 0.26;
    }
    else if (w <= 4177)
    {
        return 0.26;
    }
    else if (w <= 4596)
    {
        return 0.25;
    }
    else if (w <= 5036)
    {
        return 0.25;
    }
    else if (w <= 5497)
    {
        return 0.24;
    }
    else if (w <= 5979)
    {
        return 0.24;
    }
    else if (w <= 6483)
    {
        return 0.23;
    }
    else if (w <= 7009)
    {
        return 0.23;
    }
    else if (w <= 7558)
    {
        return 0.22;
    }
    else if (w <= 8130)
    {
        return 0.22;
    }
    else if (w <= 8726)
    {
        return 0.22;
    }
    else if (w <= 9346)
    {
        return 0.21;
    }
    else if (w <= 9991)
    {
        return 0.21;
    }
    else if (w <= 10661)
    {
        return 0.21;
    }
    else if (w <= 11358)
    {
        return 0.20;
    }
    else if (w <= 12082)
    {
        return 0.20;
    }
    else if (w <= 12834)
    {
        return 0.20;
    }
    else if (w <= 13614)
    {
        return 0.19;
    }
    else if (w <= 14424)
    {
        return 0.19;
    }
    else if (w <= 15265)
    {
        return 0.19;
    }
    else if (w <= 16137)
    {
        return 0.19;
    }
    else if (w <= 17042)
    {
        return 0.18;
    }
    else if (w <= 17981)
    {
        return 0.18;
    }
    else if (w <= 18955)
    {
        return 0.18;
    }
    else if (w <= 19965)
    {
        return 0.17;
    }
    else if (w <= 21013)
    {
        return 0.17;
    }
    else if (w <= 22101)
    {
        return 0.17;
    }
    else if (w <= 23230)
    {
        return 0.17;
    }
    else if (w <= 24402)
    {
        return 0.16;
    }
    else if (w <= 25618)
    {
        return 0.16;
    }
    else if (w <= 26881)
    {
        return 0.16;
    }
    else if (w <= 28193)
    {
        return 0.16;
    }
    else if (w <= 29557)
    {
        return 0.15;
    }
    else if (w <= 30975)
    {
        return 0.15;
    }
    else if (w <= 32450)
    {
        return 0.15;
    }
    else if (w <= 33986)
    {
        return 0.15;
    }
    else if (w <= 35586)
    {
        return 0.14;
    }
    else if (w <= 37253)
    {
        return 0.14;
    }
    else if (w <= 38992)
    {
        return 0.14;
    }
    else if (w <= 40808)
    {
        return 0.14;
    }
    else if (w <= 42707)
    {
        return 0.13;
    }
    else if (w <= 44694)
    {
        return 0.13;
    }
    else if (w <= 46776)
    {
        return 0.13;
    }
    else if (w <= 48961)
    {
        return 0.13;
    }
    else if (w <= 51258)
    {
        return 0.13;
    }
    else if (w <= 53667)
    {
        return 0.12;
    }
    else if (w <= 56230)
    {
        return 0.12;
    }
    else if (w <= 58932)
    {
        return 0.12;
    }
    else if (w <= 61799)
    {
        return 0.12;
    }
    else if (w <= 64851)
    {
        return 0.11;
    }
    else if (w <= 68113)
    {
        return 0.11;
    }
    else if (w <= 71617)
    {
        return 0.11;
    }
    else if (w <= 75401)
    {
        return 0.10;
    }
    else if (w <= 79517)
    {
        return 0.10;
    }
    else if (w <= 84035)
    {
        return 0.10;
    }
    else if (w <= 89053)
    {
        return 0.10;
    }
    else if (w <= 94717)
    {
        return 0.09;
    }
    else
    {
        return 0.09;
    }
    // NOLINTEND(bugprone-branch-clone)
}

uint32_t
TcpHighSpeed::TableLookupLinuxB(uint32_t w)
{
    // hstcp_aimd_vals[] of Linux: the upper window of each row and its b
    static const std::pair<uint32_t, uint32_t> table[] = {
        {38, 128},   {118, 112},  {221, 104},  {347, 98},   {495, 93},   {663, 89},   {851, 86},
        {1058, 83},  {1284, 81},  {1529, 78},  {1793, 76},  {2076, 74},  {2378, 72},  {2699, 71},
        {3039, 69},  {3399, 68},  {3778, 66},  {4177, 65},  {4596, 64},  {5036, 62},  {5497, 61},
        {5979, 60},  {6483, 59},  {7009, 58},  {7558, 57},  {8130, 56},  {8726, 55},  {9346, 54},
        {9991, 53},  {10661, 52}, {11358, 52}, {12082, 51}, {12834, 50}, {13614, 49}, {14424, 48},
        {15265, 48}, {16137, 47}, {17042, 46}, {17981, 45}, {18955, 45}, {19965, 44}, {21013, 43},
        {22101, 43}, {23230, 42}, {24402, 41}, {25618, 41}, {26881, 40}, {28193, 39}, {29557, 39},
        {30975, 38}, {32450, 38}, {33986, 37}, {35586, 36}, {37253, 36}, {38992, 35}, {40808, 35},
        {42707, 34}, {44694, 33}, {46776, 33}, {48961, 32}, {51258, 32}, {53677, 31}, {56230, 30},
        {58932, 30}, {61799, 29}, {64851, 28}, {68113, 28}, {71617, 27}, {75401, 26}, {79517, 26},
        {84035, 25}, {89053, 24},
    };
    for (const auto& [cwnd, md] : table)
    {
        if (w <= cwnd)
        {
            return md;
        }
    }
    return std::rbegin(table)->second;
}

} // namespace ns3
