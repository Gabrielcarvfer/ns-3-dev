/*
 * Copyright (c) 2014 Natale Patriciello <natale.patriciello@gmail.com>
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#include "tcp-cubic.h"

#include "ns3/log.h"

NS_LOG_COMPONENT_DEFINE("TcpCubic");

namespace ns3
{

NS_OBJECT_ENSURE_REGISTERED(TcpCubic);

TypeId
TcpCubic::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::TcpCubic")
            .SetParent<TcpCongestionOps>()
            .AddConstructor<TcpCubic>()
            .SetGroupName("Internet")
            .AddAttribute("FastConvergence",
                          "Enable (true) or disable (false) fast convergence",
                          BooleanValue(true),
                          MakeBooleanAccessor(&TcpCubic::m_fastConvergence),
                          MakeBooleanChecker())
            .AddAttribute("TcpFriendliness",
                          "Enable (true) or disable (false) TCP friendliness",
                          BooleanValue(true),
                          MakeBooleanAccessor(&TcpCubic::m_tcpFriendliness),
                          MakeBooleanChecker())
            .AddAttribute("Beta",
                          "Beta for multiplicative decrease",
                          DoubleValue(0.7),
                          MakeDoubleAccessor(&TcpCubic::m_beta),
                          MakeDoubleChecker<double>(0.0, 1.0))
            .AddAttribute("BetaEcn",
                          "Beta for multiplicative decrease for ABE",
                          DoubleValue(0.85), // According to RFC 8511 (ABE)
                          MakeDoubleAccessor(&TcpCubic::m_betaEcn),
                          MakeDoubleChecker<double>(0.0, 1.0))
            .AddAttribute("HyStart",
                          "Enable (true) or disable (false) hybrid slow start algorithm",
                          BooleanValue(true),
                          MakeBooleanAccessor(&TcpCubic::m_hystart),
                          MakeBooleanChecker())
            .AddAttribute("HyStartLowWindow",
                          "Lower bound cWnd for hybrid slow start (segments)",
                          UintegerValue(16),
                          MakeUintegerAccessor(&TcpCubic::m_hystartLowWindow),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("HyStartDetect",
                          "Hybrid Slow Start detection mechanisms:"
                          "packet train, delay, both",
                          EnumValue(HybridSSDetectionMode::BOTH),
                          MakeEnumAccessor<HybridSSDetectionMode>(&TcpCubic::m_hystartDetect),
                          MakeEnumChecker(HybridSSDetectionMode::PACKET_TRAIN,
                                          "PACKET_TRAIN",
                                          HybridSSDetectionMode::DELAY,
                                          "DELAY",
                                          HybridSSDetectionMode::BOTH,
                                          "BOTH"))
            .AddAttribute("HyStartMinSamples",
                          "Number of delay samples for detecting the increase of delay",
                          UintegerValue(8),
                          MakeUintegerAccessor(&TcpCubic::m_hystartMinSamples),
                          MakeUintegerChecker<uint8_t>())
            .AddAttribute("HyStartAckDelta",
                          "Spacing between ack's indicating train",
                          TimeValue(MilliSeconds(2)),
                          MakeTimeAccessor(&TcpCubic::m_hystartAckDelta),
                          MakeTimeChecker())
            .AddAttribute("HyStartDelayMin",
                          "Minimum time for hystart algorithm",
                          TimeValue(MilliSeconds(4)),
                          MakeTimeAccessor(&TcpCubic::m_hystartDelayMin),
                          MakeTimeChecker())
            .AddAttribute("HyStartDelayMax",
                          "Maximum time for hystart algorithm",
                          TimeValue(MilliSeconds(16)),
                          MakeTimeAccessor(&TcpCubic::m_hystartDelayMax),
                          MakeTimeChecker())
            .AddAttribute("CubicDelta",
                          "Delta Time to wait after fast recovery before adjusting param",
                          TimeValue(MilliSeconds(10)),
                          MakeTimeAccessor(&TcpCubic::m_cubicDelta),
                          MakeTimeChecker())
            .AddAttribute("CntUpdateInterval",
                          "Minimum time between two computations of the number of ACKs per "
                          "window increase while the window does not change, as Linux "
                          "bictcp_update() does with HZ/32 jiffies (1/32 s); zero (the default) "
                          "computes it on every ACK",
                          TimeValue(Time(0)),
                          MakeTimeAccessor(&TcpCubic::m_cntUpdateInterval),
                          MakeTimeChecker())
            .AddAttribute("LinuxHz",
                          "Rate of the jiffies (CONFIG_HZ) of Linux. When nonzero, the window "
                          "follows the integer arithmetic of the Linux tcp_cubic.c, with its "
                          "times in jiffies at this rate (the epoch, the count update interval "
                          "of HZ/32, the minimum delay rounded up to a jiffy and the discarded "
                          "delay samples within HZ of an epoch start), its fixed point beta and "
                          "C and its cube root, and grows as tcp_cong_avoid_ai(); zero (the "
                          "default) follows the published algorithm",
                          UintegerValue(0),
                          MakeUintegerAccessor(&TcpCubic::m_linuxHz),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("CntClamp",
                          "Counter value when no losses are detected (counter is used"
                          " when incrementing cWnd in congestion avoidance, to avoid"
                          " floating point arithmetic). It is the modulo of the (avoided)"
                          " division",
                          UintegerValue(20),
                          MakeUintegerAccessor(&TcpCubic::m_cntClamp),
                          MakeUintegerChecker<uint8_t>())
            .AddAttribute("C",
                          "Cubic Scaling factor",
                          DoubleValue(0.4),
                          MakeDoubleAccessor(&TcpCubic::m_c),
                          MakeDoubleChecker<double>(0.0));
    return tid;
}

TcpCubic::TcpCubic()
    : TcpCongestionOps(),
      m_cWndCnt(0),
      m_lastMaxCwnd(0),
      m_bicOriginPoint(0),
      m_bicK(0.0),
      m_delayMin(Time::Min()),
      m_epochStart(Time::Min()),
      m_found(false),
      m_roundStart(Time::Min()),
      m_endSeq(0),
      m_lastAck(Time::Min()),
      m_cubicDelta(Time::Min()),
      m_currRtt(Time::Min()),
      m_sampleCnt(0)
{
    NS_LOG_FUNCTION(this);
}

TcpCubic::TcpCubic(const TcpCubic& sock)
    : TcpCongestionOps(sock),
      m_fastConvergence(sock.m_fastConvergence),
      m_beta(sock.m_beta),
      m_betaEcn(sock.m_betaEcn),
      m_hystart(sock.m_hystart),
      m_hystartDetect(sock.m_hystartDetect),
      m_hystartLowWindow(sock.m_hystartLowWindow),
      m_hystartAckDelta(sock.m_hystartAckDelta),
      m_hystartDelayMin(sock.m_hystartDelayMin),
      m_hystartDelayMax(sock.m_hystartDelayMax),
      m_hystartMinSamples(sock.m_hystartMinSamples),
      m_initialCwnd(sock.m_initialCwnd),
      m_cntClamp(sock.m_cntClamp),
      m_c(sock.m_c),
      m_cWndCnt(sock.m_cWndCnt),
      m_lastMaxCwnd(sock.m_lastMaxCwnd),
      m_bicOriginPoint(sock.m_bicOriginPoint),
      m_bicK(sock.m_bicK),
      m_delayMin(sock.m_delayMin),
      m_epochStart(sock.m_epochStart),
      m_found(sock.m_found),
      m_roundStart(sock.m_roundStart),
      m_endSeq(sock.m_endSeq),
      m_lastAck(sock.m_lastAck),
      m_cubicDelta(sock.m_cubicDelta),
      m_cntUpdateInterval(sock.m_cntUpdateInterval),
      m_lastUpdateCwnd(sock.m_lastUpdateCwnd),
      m_lastUpdateTime(sock.m_lastUpdateTime),
      m_cnt(sock.m_cnt),
      m_linuxHz(sock.m_linuxHz),
      m_jEpochStart(sock.m_jEpochStart),
      m_jLastTime(sock.m_jLastTime),
      m_linuxK(sock.m_linuxK),
      m_delayMinUs(sock.m_delayMinUs),
      m_currRtt(sock.m_currRtt),
      m_sampleCnt(sock.m_sampleCnt)
{
    NS_LOG_FUNCTION(this);
}

std::string
TcpCubic::GetName() const
{
    return "TcpCubic";
}

void
TcpCubic::Init(Ptr<TcpSocketState> tcb)
{
    HystartReset(tcb);
}

void
TcpCubic::HystartReset(Ptr<const TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this);

    m_roundStart = m_lastAck = Simulator::Now();
    m_endSeq = tcb->m_highTxMark;
    m_currRtt = Time::Min();
    m_sampleCnt = 0;
}

void
TcpCubic::IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);

    if (!tcb->m_isCwndLimited)
    {
        NS_LOG_DEBUG("No increase because current cwnd " << tcb->m_cWnd
                                                         << " is not limiting the flow");
        return;
    }

    if (m_linuxHz > 0)
    {
        LinuxIncreaseWindow(tcb, segmentsAcked);
        return;
    }

    if (tcb->m_cWnd < tcb->m_ssThresh)
    {
        // In Linux, the QUICKACK socket option enables the receiver to send
        // immediate acks initially (during slow start) and then transition
        // to delayed acks.  ns-3 does not implement QUICKACK, and if ack
        // counting instead of byte counting is used during slow start window
        // growth, when TcpSocket::DelAckCount==2, then the slow start will
        // not reach as large of an initial window as in Linux.  Therefore,
        // we can approximate the effect of QUICKACK by making this slow
        // start phase perform Appropriate Byte Counting (RFC 3465)
        tcb->m_cWnd += segmentsAcked * tcb->m_segmentSize;
        segmentsAcked = 0;

        NS_LOG_INFO("In SlowStart, updated to cwnd " << tcb->m_cWnd << " ssthresh "
                                                     << tcb->m_ssThresh);
    }

    if (tcb->m_cWnd >= tcb->m_ssThresh && segmentsAcked > 0)
    {
        m_cWndCnt += segmentsAcked;
        uint32_t cnt = Update(tcb, segmentsAcked);

        /* According to RFC 6356 even once the new cwnd is
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
                        << m_cWndCnt << " cnd " << cnt);
        }
    }
}

uint32_t
TcpCubic::Jiffies() const
{
    return static_cast<uint32_t>(Simulator::Now().GetMicroSeconds() * m_linuxHz / 1000000);
}

uint32_t
TcpCubic::CubicRoot(uint64_t a)
{
    // Table lookup and one Newton-Raphson iteration, as Linux cubic_root()
    static const uint8_t v[] = {
        0,   54,  54,  54,  118, 118, 118, 118, 123, 129, 134, 138, 143, 147, 151, 156,
        157, 161, 164, 168, 170, 173, 176, 179, 181, 185, 187, 190, 192, 194, 197, 199,
        200, 202, 204, 206, 209, 211, 213, 215, 217, 219, 221, 222, 224, 225, 227, 229,
        231, 232, 234, 236, 237, 239, 240, 242, 244, 245, 246, 248, 250, 251, 252, 254,
    };
    uint32_t b = 0;
    for (uint64_t x = a; x != 0; x >>= 1)
    {
        ++b; // fls64()
    }
    if (b < 7)
    {
        return (static_cast<uint32_t>(v[a]) + 35) >> 6;
    }
    b = ((b * 84) >> 8) - 1;
    auto shift = static_cast<uint32_t>(a >> (b * 3));
    uint32_t x = static_cast<uint32_t>((static_cast<uint32_t>(v[shift]) + 10) << b) >> 6;
    x = 2 * x + static_cast<uint32_t>(a / (static_cast<uint64_t>(x) * (x - 1)));
    x = (x * 341) >> 10;
    return x;
}

void
TcpCubic::LinuxIncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);
    uint32_t cwnd = tcb->GetCwndInSegments();
    uint32_t acked = segmentsAcked;
    if (tcb->m_cWnd < tcb->m_ssThresh)
    {
        acked = LinuxSlowStart(tcb, cwnd, acked);
        if (acked == 0)
        {
            return;
        }
    }
    LinuxUpdate(cwnd, acked);
    LinuxCongAvoidAi(cwnd, m_cWndCnt, m_cnt, acked);
    tcb->m_cWnd = cwnd * tcb->m_segmentSize;
    NS_LOG_INFO("In CongAvoid, cwnd " << cwnd << " cnt " << m_cnt << " count " << m_cWndCnt);
}

void
TcpCubic::LinuxUpdate(uint32_t cwnd, uint32_t acked)
{
    NS_LOG_FUNCTION(this << cwnd << acked);
    const uint32_t beta = static_cast<uint32_t>(std::lround(m_beta * 1024));
    const uint64_t cubeRttScale = static_cast<uint64_t>(std::lround(m_c * 1024 / 10)) * 10;
    const uint32_t now = Jiffies();

    m_ackCnt += acked;

    if (m_lastUpdateCwnd == cwnd &&
        static_cast<int32_t>(now - m_jLastTime) <= static_cast<int32_t>(m_linuxHz / 32))
    {
        return;
    }

    // The count is computed at most once per jiffy
    if (m_jEpochStart == 0 || now != m_jLastTime)
    {
        m_lastUpdateCwnd = cwnd;
        m_jLastTime = now;

        if (m_jEpochStart == 0)
        {
            m_jEpochStart = now;
            m_ackCnt = acked;
            m_tcpCwnd = cwnd;
            if (m_lastMaxCwnd <= cwnd)
            {
                m_linuxK = 0;
                m_bicOriginPoint = cwnd;
            }
            else
            {
                const uint64_t cubeFactor = (1ULL << 40) / cubeRttScale;
                m_linuxK = CubicRoot(cubeFactor * (m_lastMaxCwnd - cwnd));
                m_bicOriginPoint = m_lastMaxCwnd;
            }
        }

        // usecs_to_jiffies() rounds up
        uint64_t t = static_cast<int32_t>(now - m_jEpochStart);
        t += (static_cast<uint64_t>(m_delayMinUs) * m_linuxHz + 999999) / 1000000;
        t = (t << 10) / m_linuxHz;
        uint64_t offs = t < m_linuxK ? m_linuxK - t : t - m_linuxK;
        auto delta = static_cast<uint32_t>((cubeRttScale * offs * offs * offs) >> 40);
        uint32_t bicTarget = t < m_linuxK ? m_bicOriginPoint - delta : m_bicOriginPoint + delta;
        m_cnt = bicTarget > cwnd ? cwnd / (bicTarget - cwnd) : 100 * cwnd;
        if (m_lastMaxCwnd == 0 && m_cnt > m_cntClamp)
        {
            m_cnt = m_cntClamp;
        }
    }

    if (m_tcpFriendliness)
    {
        const uint32_t betaScale = 8 * (1024 + beta) / 3 / (1024 - beta);
        uint32_t delta = (cwnd * betaScale) >> 3;
        while (m_ackCnt > delta)
        {
            m_ackCnt -= delta;
            m_tcpCwnd++;
        }
        if (m_tcpCwnd > cwnd)
        {
            uint32_t maxCnt = cwnd / (m_tcpCwnd - cwnd);
            m_cnt = std::min(m_cnt, maxCnt);
        }
    }
    m_cnt = std::max(m_cnt, 2U);
}

uint32_t
TcpCubic::Update(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this);
    Time t;
    uint32_t delta;
    uint32_t bicTarget;
    uint32_t cnt = 0;
    uint32_t maxCnt;
    double offs;
    uint32_t segCwnd = tcb->GetCwndInSegments();

    m_ackCnt += segmentsAcked;

    // Linux bictcp_update(): while the window does not change, the count is
    // kept for HZ/32 jiffies; a reduction, which starts a new epoch, always
    // recomputes it
    if (m_cntUpdateInterval.IsStrictlyPositive())
    {
        if (m_epochStart != Time::Min() && m_lastUpdateCwnd == segCwnd &&
            Simulator::Now() - m_lastUpdateTime <= m_cntUpdateInterval)
        {
            return m_cnt;
        }
        m_lastUpdateCwnd = segCwnd;
        m_lastUpdateTime = Simulator::Now();
    }

    if (m_epochStart == Time::Min())
    {
        m_epochStart = Simulator::Now(); // record the beginning of an epoch
        m_ackCnt = segmentsAcked;
        m_tcpCwnd = segCwnd;

        if (m_lastMaxCwnd <= segCwnd)
        {
            NS_LOG_DEBUG("lastMaxCwnd <= m_cWnd. K=0 and origin=" << segCwnd);
            m_bicK = 0.0;
            m_bicOriginPoint = segCwnd;
        }
        else
        {
            m_bicK = std::pow((m_lastMaxCwnd - segCwnd) / m_c, 1 / 3.);
            m_bicOriginPoint = m_lastMaxCwnd;
            NS_LOG_DEBUG("lastMaxCwnd > m_cWnd. K=" << m_bicK << " and origin=" << m_lastMaxCwnd);
        }
    }

    t = Simulator::Now() + m_delayMin - m_epochStart;

    if (t.GetSeconds() < m_bicK) /* t - K */
    {
        offs = m_bicK - t.GetSeconds();
        NS_LOG_DEBUG("t=" << t.GetSeconds() << " <k: offs=" << offs);
    }
    else
    {
        offs = t.GetSeconds() - m_bicK;
        NS_LOG_DEBUG("t=" << t.GetSeconds() << " >= k: offs=" << offs);
    }

    /* Constant value taken from Experimental Evaluation of Cubic Tcp, available at
     * eprints.nuim.ie/1716/1/Hamiltonpfldnet2007_cubic_final.pdf */
    delta = m_c * std::pow(offs, 3);

    NS_LOG_DEBUG("delta: " << delta);

    if (t.GetSeconds() < m_bicK)
    {
        // below origin
        bicTarget = m_bicOriginPoint - delta;
        NS_LOG_DEBUG("t < k: Bic Target: " << bicTarget);
    }
    else
    {
        // above origin
        bicTarget = m_bicOriginPoint + delta;
        NS_LOG_DEBUG("t >= k: Bic Target: " << bicTarget);
    }

    // Next the window target is converted into a cnt or count value. CUBIC will
    // wait until enough new ACKs have arrived that a counter meets or exceeds
    // this cnt value. This is how the CUBIC implementation simulates growing
    // cwnd by values other than 1 segment size.
    if (bicTarget > segCwnd)
    {
        cnt = segCwnd / (bicTarget - segCwnd);
        NS_LOG_DEBUG("target>cwnd. cnt=" << cnt);
    }
    else
    {
        cnt = 100 * segCwnd;
    }

    if (m_lastMaxCwnd == 0 && cnt > m_cntClamp)
    {
        cnt = m_cntClamp;
    }

    if (m_tcpFriendliness)
    {
        auto scale = static_cast<uint32_t>(8 * (1024 + m_beta * 1024) / 3 / (1024 - m_beta * 1024));
        delta = (segCwnd * scale) >> 3;
        while (m_ackCnt > delta)
        {
            m_ackCnt -= delta;
            m_tcpCwnd++;
        }
        if (m_tcpCwnd > segCwnd)
        {
            delta = m_tcpCwnd - segCwnd;
            maxCnt = segCwnd / delta;
            if (cnt > maxCnt)
            {
                cnt = maxCnt;
            }
        }
    }

    // The maximum rate of cwnd increase CUBIC allows is 1 packet per
    // 2 packets ACKed, meaning cwnd grows at 1.5x per RTT.
    m_cnt = std::max(cnt, 2U);
    return m_cnt;
}

void
TcpCubic::PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked << rtt);

    // An ACK without RTT sample (Linux cubictcp_acked(), rtt_us < 0)
    if (!rtt.IsStrictlyPositive())
    {
        return;
    }

    if (m_linuxHz > 0)
    {
        // The samples within a second (HZ jiffies) of the start of an epoch
        // are discarded, and the delay is in whole microseconds
        if (m_jEpochStart != 0 &&
            static_cast<int32_t>(Jiffies() - m_jEpochStart) < static_cast<int32_t>(m_linuxHz))
        {
            return;
        }
        auto delay = std::max<uint32_t>(static_cast<uint32_t>(rtt.GetMicroSeconds()), 1);
        if (m_delayMinUs == 0 || m_delayMinUs > delay)
        {
            m_delayMinUs = delay;
            m_delayMin = MicroSeconds(delay);
        }
    }
    else
    {
        /* Discard delay samples right after fast recovery */
        if (m_epochStart != Time::Min() && (Simulator::Now() - m_epochStart) < m_cubicDelta)
        {
            return;
        }

        /* first time call or link delay decreases */
        if (m_delayMin == Time::Min() || m_delayMin > rtt)
        {
            m_delayMin = rtt;
        }
    }

    /* hystart triggers when cwnd is larger than some threshold */
    if (m_hystart && tcb->m_cWnd <= tcb->m_ssThresh &&
        tcb->m_cWnd >= m_hystartLowWindow * tcb->m_segmentSize)
    {
        HystartUpdate(tcb, rtt);
    }
}

void
TcpCubic::HystartUpdate(Ptr<TcpSocketState> tcb, const Time& delay)
{
    NS_LOG_FUNCTION(this << delay);

    if (!m_found)
    {
        Time now = Simulator::Now();

        // A new round starts once the end of the previous one is ACKed
        if (tcb->m_lastAckedSeq > m_endSeq)
        {
            HystartReset(tcb);
        }

        /* first detection parameter - ack-train detection */
        if ((now - m_lastAck) <= m_hystartAckDelta)
        {
            m_lastAck = now;

            // The train has to last the forward one way delay, i.e., half the
            // minimum RTT (Linux hystart_update()), unless pacing spreads the
            // segments of a round
            Time threshold = tcb->m_pacing ? m_delayMin : m_delayMin / 2;
            if ((now - m_roundStart) > threshold)
            {
                if (m_hystartDetect == HybridSSDetectionMode::PACKET_TRAIN ||
                    m_hystartDetect == HybridSSDetectionMode::BOTH)
                {
                    m_found = true;
                }
            }
        }

        /* obtain the minimum delay of more than sampling packets */
        if (m_currRtt == Time::Min() || m_currRtt > delay)
        {
            m_currRtt = delay;
        }
        if (m_sampleCnt < m_hystartMinSamples)
        {
            ++m_sampleCnt;
        }
        else if (m_currRtt > m_delayMin + HystartDelayThresh(m_delayMin / 8))
        {
            if (m_hystartDetect == HybridSSDetectionMode::DELAY ||
                m_hystartDetect == HybridSSDetectionMode::BOTH)
            {
                m_found = true;
            }
        }

        /*
         * Either one of two conditions are met,
         * we exit from slow start immediately.
         */
        if (m_found)
        {
            NS_LOG_DEBUG("Exit from SS, immediately :-)");
            tcb->m_ssThresh = tcb->m_cWnd;
        }
    }
}

Time
TcpCubic::HystartDelayThresh(const Time& t) const
{
    NS_LOG_FUNCTION(this << t);

    Time ret = t;
    if (t > m_hystartDelayMax)
    {
        ret = m_hystartDelayMax;
    }
    else if (t < m_hystartDelayMin)
    {
        ret = m_hystartDelayMin;
    }

    return ret;
}

uint32_t
TcpCubic::GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
    NS_LOG_FUNCTION(this << tcb << bytesInFlight);

    uint32_t segCwnd = tcb->GetCwndInSegments();
    NS_LOG_DEBUG("Loss at cWnd=" << segCwnd
                                 << " segments in flight=" << bytesInFlight / tcb->m_segmentSize);

    if (m_linuxHz > 0 && !(tcb->m_abeEnabled && tcb->m_ecnState == TcpSocketState::ECN_ECE_RCVD))
    {
        // cubictcp_recalc_ssthresh(), with beta in 1/1024
        const auto beta = static_cast<uint32_t>(std::lround(m_beta * 1024));
        m_jEpochStart = 0;
        m_lastMaxCwnd = LinuxLastMaxCwnd(segCwnd, m_lastMaxCwnd, beta, m_fastConvergence);
        return std::max((segCwnd * beta) / 1024, 2U) * tcb->m_segmentSize;
    }

    /* Wmax and fast convergence */
    if (segCwnd < m_lastMaxCwnd && m_fastConvergence)
    {
        m_lastMaxCwnd = (segCwnd * (1 + m_beta)) / 2; // Section 4.6 in RFC 8312
    }
    else
    {
        m_lastMaxCwnd = segCwnd;
    }

    m_epochStart = Time::Min(); // end of epoch

    uint32_t ssThresh;
    if (tcb->m_abeEnabled && tcb->m_ecnState == TcpSocketState::ECN_ECE_RCVD)
    {
        ssThresh = std::max(static_cast<uint32_t>(segCwnd * m_betaEcn), 2U) *
                   tcb->m_segmentSize; // According to RFC 8511 (ABE)
    }
    else
    { /* Formula taken from the Linux kernel */
        ssThresh = std::max(static_cast<uint32_t>(segCwnd * m_beta), 2U) * tcb->m_segmentSize;
    }

    NS_LOG_DEBUG("SsThresh = " << ssThresh);

    return ssThresh;
}

void
TcpCubic::CongestionStateSet(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCongState_t newState)
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

    if (newState == TcpSocketState::CA_LOSS)
    {
        CubicReset(tcb);
        HystartReset(tcb);
    }
}

void
TcpCubic::CubicReset(Ptr<const TcpSocketState> tcb)
{
    NS_LOG_FUNCTION(this << tcb);

    m_bicOriginPoint = 0;
    m_bicK = 0;
    m_ackCnt = 0;
    m_tcpCwnd = 0;
    m_delayMin = Time::Min();
    m_found = false;
    if (m_linuxHz > 0)
    {
        // bictcp_reset() clears the whole state
        m_cnt = 0;
        m_lastMaxCwnd = 0;
        m_lastUpdateCwnd = 0;
        m_jLastTime = 0;
        m_linuxK = 0;
        m_delayMinUs = 0;
        m_jEpochStart = 0;
    }
}

Ptr<TcpCongestionOps>
TcpCubic::Fork()
{
    NS_LOG_FUNCTION(this);
    return CopyObject<TcpCubic>(this);
}

} // namespace ns3
