/*
 * Copyright (c) 2015 Natale Patriciello <natale.patriciello@gmail.com>
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */
#ifndef TCPCONGESTIONOPS_H
#define TCPCONGESTIONOPS_H

#include "tcp-rate-ops.h"
#include "tcp-socket-state.h"

namespace ns3
{

/**
 * @ingroup tcp
 * @defgroup congestionOps Congestion Control Algorithms.
 *
 * The various congestion control algorithms, also known as "TCP flavors".
 */

/**
 * @ingroup congestionOps
 *
 * @brief Congestion control abstract class
 *
 * The design is inspired by what Linux v4.0 does (but it has been
 * in place for years). The congestion control is split from the main
 * socket code, and it is a pluggable component. An interface has been defined;
 * variables are maintained in the TcpSocketState class, while subclasses of
 * TcpCongestionOps operate over an instance of that class.
 *
 * Only three methods have been implemented right now; however, Linux has many others,
 * which can be added later in ns-3.
 *
 * @see IncreaseWindow
 * @see PktsAcked
 */
class TcpCongestionOps : public Object
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    TcpCongestionOps();

    /**
     * @brief Copy constructor.
     * @param other object to copy.
     */
    TcpCongestionOps(const TcpCongestionOps& other);

    ~TcpCongestionOps() override;

    /**
     * @brief Get the name of the congestion control algorithm
     *
     * @return A string identifying the name
     */
    virtual std::string GetName() const = 0;

    /**
     * @brief Set configuration required by congestion control algorithm
     *
     * @param tcb internal congestion state
     */
    virtual void Init(Ptr<TcpSocketState> tcb [[maybe_unused]])
    {
    }

    /**
     * @brief Set rate operation required by the congestion control algorithm
     *
     * @param rateOps a pointer to the rate operation object used to manage rate control
     */
    virtual void SetRateOps(Ptr<TcpRateOps> rateOps [[maybe_unused]])
    {
    }

    /**
     * @brief Get the slow start threshold after a loss event
     *
     * Is guaranteed that the congestion control state (\p TcpAckState_t) is
     * changed BEFORE the invocation of this method.
     * The implementer should return the slow start threshold (and not change
     * it directly) because, in the future, the TCP implementation may require to
     * instantly recover from a loss event (e.g. when there is a network with an high
     * reordering factor).
     *
     * @param tcb internal congestion state
     * @param bytesInFlight the FlightSize of RFC 5681 (the data sent and not
     *        cumulatively acknowledged, including the segments SACKed or
     *        deemed lost), bounded by the congestion window
     * @return Slow start threshold
     */
    virtual uint32_t GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) = 0;

    /**
     * @brief Congestion avoidance algorithm implementation
     *
     * Mimic the function \pname{cong_avoid} in Linux. New segments have been ACKed,
     * and the congestion control duty is to update the window.
     *
     * The function is allowed to change directly cWnd and/or ssThresh.
     *
     * @param tcb internal congestion state
     * @param segmentsAcked count of segments acked
     */
    virtual void IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked);

    /**
     * @brief Timing information on received ACK
     *
     * The function is called every time an ACK is received (only one time
     * also for cumulative ACKs) and contains timing information. It is
     * optional (congestion controls need not implement it) and the default
     * implementation does nothing.
     *
     * @param tcb internal congestion state
     * @param segmentsAcked count of segments acked
     * @param rtt RTT sample of the last ACKed segment (not the smoothed RTT),
     *        or zero when the ACK gives no new sample (e.g., a duplicate ACK)
     */
    virtual void PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt);

    /**
     * @brief Trigger events/calculations specific to a congestion state
     *
     * This function mimics the notification function \pname{set_state} in Linux.
     * The function does not change the congestion state in the tcb; it notifies
     * the congestion control algorithm that this state is about to be changed.
     * The tcb->m_congState variable must be separately set; for example:
     *
     * @code
     *   m_congestionControl->CongestionStateSet (m_tcb, TcpSocketState::CA_RECOVERY);
     *   m_tcb->m_congState = TcpSocketState::CA_RECOVERY;
     * @endcode
     *
     * @param tcb internal congestion state
     * @param newState new congestion state to which the TCP is going to switch
     */
    virtual void CongestionStateSet(Ptr<TcpSocketState> tcb,
                                    const TcpSocketState::TcpCongState_t newState);

    /**
     * @brief Trigger events/calculations on occurrence of congestion window event
     *
     * This function mimics the function \pname{cwnd_event} in Linux.
     * The function is called in case of congestion window events.
     *
     * @param tcb internal congestion state
     * @param event the event which triggered this function
     */
    virtual void CwndEvent(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event);

    /**
     * @brief Returns true when Congestion Control Algorithm implements CongControl
     *
     * @return true if CC implements CongControl function
     *
     * This function is the equivalent in C++ of the C checks that are used
     * in the Linux kernel to see if an optional function has been defined.
     * Since CongControl is optional, not all congestion controls have it. But,
     * from the perspective of TcpSocketBase, the behavior is different if
     * CongControl is present. Therefore, this check should return true for any
     * congestion controls that implements the CongControl optional function.
     */
    virtual bool HasCongControl() const;

    /**
     * @brief Called when packets are delivered to update cwnd and pacing rate
     *
     * This function mimics the function cong_control in Linux. It is allowed to
     * change directly cWnd and pacing rate.
     *
     * @param tcb internal congestion state
     * @param rc Rate information for the connection
     * @param rs Rate sample (over a period of time) information
     */
    virtual void CongControl(Ptr<TcpSocketState> tcb,
                             const TcpRateOps::TcpRateConnection& rc,
                             const TcpRateOps::TcpRateSample& rs);

    /**
     * @brief Get the least number of segments of a GSO packet of Linux
     * (min_tso_segs(), or the tcp_min_tso_segs sysctl if the congestion
     * control does not define it)
     *
     * @param tcb internal congestion state
     * @return the number of segments
     */
    virtual uint32_t GetMinTsoSegs(Ptr<const TcpSocketState> tcb) const;

    // Present in Linux but not in ns-3 yet:
    /* call when ack arrives (optional) */
    //     void (*in_ack_event)(struct sock *sk, u32 flags);
    /* new value of cwnd after loss (optional) */
    //     u32  (*undo_cwnd)(struct sock *sk);
    /* hook for packet ack accounting (optional) */
    //     void (*pkts_acked)(struct sock *sk, u32 ext, int *attr, union tcp_cc_info *info);

    /**
     * @brief Copy the congestion control algorithm across sockets
     *
     * @return a pointer of the copied object
     */
    virtual Ptr<TcpCongestionOps> Fork() = 0;

  protected:
    /**
     * @brief Grow the window in segments as Linux tcp_slow_start(), the
     * segments beyond ssThresh being left for the congestion avoidance
     *
     * @param tcb internal congestion state, whose cWnd is set to the new window
     * @param cwnd the window in segments, updated
     * @param acked count of segments acked
     * @return the acknowledged segments left for the congestion avoidance
     */
    static uint32_t LinuxSlowStart(Ptr<TcpSocketState> tcb, uint32_t& cwnd, uint32_t acked);

    /**
     * @brief Grow the window in segments by one segment for every w
     * acknowledged segments, as Linux tcp_cong_avoid_ai()
     *
     * @param cwnd the window in segments, updated
     * @param count the acknowledged segments not yet counted (snd_cwnd_cnt), updated
     * @param w segments to acknowledge per segment of growth
     * @param acked count of segments acked
     */
    static void LinuxCongAvoidAi(uint32_t& cwnd, uint32_t& count, uint32_t w, uint32_t acked);

    /**
     * @brief Get the last maximum window of BIC and CUBIC after a loss, as
     * Linux bictcp_recalc_ssthresh() and cubictcp_recalc_ssthresh()
     *
     * @param segCwnd the window in segments at the loss
     * @param lastMaxCwnd the last maximum window in segments
     * @param beta the multiplicative decrease factor in 1/1024
     * @param fastConvergence whether fast convergence is enabled
     * @return the new last maximum window in segments
     */
    static uint32_t LinuxLastMaxCwnd(uint32_t segCwnd,
                                     uint32_t lastMaxCwnd,
                                     uint32_t beta,
                                     bool fastConvergence);
};

/**
 * @brief The NewReno implementation
 *
 * New Reno introduces partial ACKs inside the well-established Reno algorithm.
 * This and other modifications are described in RFC 6582.
 *
 * @see IncreaseWindow
 */
class TcpNewReno : public TcpCongestionOps
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    TcpNewReno();

    /**
     * @brief Copy constructor.
     * @param sock object to copy.
     */
    TcpNewReno(const TcpNewReno& sock);

    ~TcpNewReno() override;

    std::string GetName() const override;

    void IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) override;
    uint32_t GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) override;
    void CongestionStateSet(Ptr<TcpSocketState> tcb,
                            const TcpSocketState::TcpCongState_t newState) override;
    Ptr<TcpCongestionOps> Fork() override;

  protected:
    virtual uint32_t SlowStart(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked);
    virtual void CongestionAvoidance(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked);

    /**
     * @brief Tell whether the window grows per acknowledged segment
     *
     * When false (the default), slow start and congestion avoidance grow the
     * window once per ACK that acknowledges new data, as RFC 5681 equations
     * 2 and 3 state, whatever the number of segments it acknowledges. When
     * true, every acknowledged segment counts, as the byte counting of
     * RFC 3465 and the tcp_slow_start() and tcp_cong_avoid_ai() functions
     * of Linux do: with delayed ACKs, the window then grows as it would with
     * an ACK per segment. Derived models with their own per-ACK increase
     * rules honour this flag as well.
     *
     * @return true if the window grows per acknowledged segment
     */
    bool CountsAckedSegments() const
    {
        return m_countAckedSegments;
    }

    /**
     * @brief Grow the window by one segment for every w acknowledged
     * segments, as Linux tcp_cong_avoid_ai()
     *
     * The count of acknowledged segments is the one of the congestion
     * avoidance of TcpNewReno (Linux snd_cwnd_cnt), cleared when the window
     * is reduced, so that derived models whose Linux implementation shares
     * snd_cwnd_cnt between several increase rules (such as YeAH) share it too.
     *
     * @param tcb internal congestion state
     * @param segmentsAcked count of segments acked
     * @param w segments to acknowledge per segment of growth (at least 1)
     */
    void CountedIncrease(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, uint32_t w);

    /**
     * @brief Get the count of acknowledged segments of the congestion
     * avoidance (Linux snd_cwnd_cnt)
     *
     * @return the count of acknowledged segments not yet counted in the window growth
     */
    uint32_t GetCwndCount() const
    {
        return m_cWndCnt;
    }

    /**
     * @brief Set the count of acknowledged segments of the congestion
     * avoidance (Linux snd_cwnd_cnt), for derived models whose Linux
     * implementation changes it with their own rules
     *
     * @param count the new count
     */
    void SetCwndCount(uint32_t count)
    {
        m_cWndCnt = count;
    }

  private:
    bool m_countAckedSegments{false}; //!< Grow the window per acknowledged segment
    uint32_t m_cWndCnt{0}; //!< Acknowledged segments not yet counted in the window growth
    double m_betaLoss{0};  //!< Beta for New-Reno multiplicative decrease
    double m_betaEcn{0};   //!< Beta for New-Reno multiplicative decrease with ABE
};

} // namespace ns3

#endif // TCPCONGESTIONOPS_H
