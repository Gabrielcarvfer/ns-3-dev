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

#ifndef TCPLP_H
#define TCPLP_H

#include "tcp-congestion-ops.h"
#include "tcp-rate-ops.h"

#include "ns3/sequence-number.h"
#include "ns3/traced-value.h"

namespace ns3
{

class TcpSocketState;

/**
 * @ingroup congestionOps
 *
 * @brief TCP-LP (Low Priority) congestion control algorithm
 */
class TcpLp : public TcpNewReno
{
  public:
    /**
     * @brief Get the type ID.
     *
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    /**
     * @brief Creates an unbound tcp socket.
     *
     */
    TcpLp();

    /**
     * @brief Copy constructor
     *
     * @param sock the object to copy
     */
    TcpLp(const TcpLp& sock);

    ~TcpLp() override;

    /**
     * @brief Timing information on received ACK
     *
     * The function is called every time an ACK is received.
     * It determines the state of TcpLp and adjusts the congestion window accordingly.
     *
     * @param tcb internal congestion state
     * @param segmentsAcked count of segments acked
     * @param rtt last rtt
     */
    void PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt) override;

    std::string GetName() const override;

    Ptr<TcpCongestionOps> Fork() override;

    void IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) override;

    uint32_t GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) override;
    void SetRateOps(Ptr<TcpRateOps> rateOps) override;
    void CongestionStateSet(Ptr<TcpSocketState> tcb,
                            const TcpSocketState::TcpCongState_t newState) override;

  protected:
    /**
     * @brief Invokes Congestion Avoidance of TcpNewReno if TcpLp is not within inference.
     *
     * @param tcb internal congestion state
     * @param segmentsAcked count of segments acked
     */
    void CongestionAvoidance(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) override;

  private:
    /**
     * @brief Describes the state of TcpLp.
     *
     */
    enum State
    {
        LP_VALID_RHZ = (1 << 0),  /**< Estimated remote clock rate is valid (Linux) */
        LP_VALID_OWD = (1 << 1),  /**< Calculated One-Way Delay is valid */
        LP_WITHIN_THR = (1 << 3), /**< TcpLp is within Threshold */
        LP_WITHIN_INF = (1 << 4), /**< TcpLp is within Inference */
    };

    uint32_t m_flag;      //!< TcpLp state flag
    uint32_t m_sOwd;      //!< Smoothed One-Way Delay (Linux)
    uint32_t m_owdMin;    //!< Minimum One-Way Delay (Linux)
    uint32_t m_owdMax;    //!< Maximum One-Way Delay (Linux)
    uint32_t m_owdMaxRsv; //!< Reserved Maximum One-Way Delay (Linux)
    Time m_lastDrop;      //!< Last time when cwnd was reduced on a delay indication
    Time m_inference;     //!< Current inference period

    bool m_linuxArithmetic{false}; //!< Compute as the tcp_lp.c of Linux does
    uint32_t m_remoteHz{0};        //!< Estimated remote timestamp clock rate (Linux)
    uint32_t m_remoteRefTime{0};   //!< Remote timestamp of the last rate estimate (Linux)
    uint32_t m_localRefTime{0};    //!< Echoed timestamp of the last rate estimate (Linux)
    uint32_t m_lastDropTs{0};      //!< Local timestamp of the last reduction (Linux)
    uint32_t m_inferenceTs{0};     //!< Inference period, in timestamp units (Linux)

    bool m_ackHoldFilter{false};       //!< Discard the one way delays biased by the ACK hold
    bool m_restartSmoothing{false};    //!< Restart the smoothed delay after a reaction
    bool m_haveOwd{false};             //!< At least a one way delay sample was taken
    uint32_t m_owdRef{0};              //!< Raw one way delay of the first sample (ms)
    int64_t m_dMin{0};                 //!< Minimum one way delay, relative to m_owdRef (us)
    int64_t m_dMax{0};                 //!< Maximum one way delay, relative to m_owdRef (us)
    int64_t m_sd8{0};                  //!< Smoothed one way delay, times 8 (us)
    bool m_lossSeen{false};            //!< A loss was detected (the delay inference is active)
    SequenceNumber32 m_reactionSeq{0}; //!< Highest data sent at the last reaction
    bool m_reacted{false};             //!< m_reactionSeq is valid
    bool m_restartPending{false};      //!< Restart the smoothing at the next sample
    SequenceNumber32 m_lastAck{0};     //!< Highest cumulative ACK seen
    bool m_haveAck{false};             //!< m_lastAck is valid
    bool m_delayedAcks{false};         //!< The receiver was seen to delay its ACKs
    bool m_reducing{false};            //!< In a window reduction (recovery, loss, CWR)
    Ptr<TcpRateOps> m_rateOps; //!< Rate operations, with the send time of the data delivered

  private:
    /**
     * @brief Calculate the one way delay of an ACK, as the timestamp of the
     * receiver minus the send time of a segment it acknowledges, relative to
     * the first sample
     *
     * @param tcb internal congestion state
     * @return One-Way Delay, in microseconds, relative to the first sample
     */
    int64_t OwdCalculator(Ptr<TcpSocketState> tcb);

    /**
     * @brief Take a one way delay sample: update the minimum, the maximum
     * and the smoothed one way delays
     *
     * @param tcb internal congestion state
     * @param bytesAcked the bytes newly acknowledged by the ACK
     * @return whether a sample was taken
     */
    bool RttSample(Ptr<TcpSocketState> tcb, uint32_t bytesAcked);

    /**
     * @brief Estimate the timestamp clock rate of the receiver, as
     * tcp_lp_remote_hz_estimator() of Linux
     *
     * @param tcb internal congestion state
     * @return the estimated rate, in Hz
     */
    uint32_t LinuxRemoteHzEstimator(Ptr<TcpSocketState> tcb);

    /**
     * @brief Calculate the one way delay, as tcp_lp_owd_calculator() of Linux
     *
     * @param tcb internal congestion state
     * @return the one way delay
     */
    uint32_t LinuxOwdCalculator(Ptr<TcpSocketState> tcb);

    /**
     * @brief Take a one way delay sample, as tcp_lp_rtt_sample() of Linux
     *
     * @param tcb internal congestion state
     */
    void LinuxRttSample(Ptr<TcpSocketState> tcb);

    /**
     * @brief Process an ACK, as tcp_lp_pkts_acked() of Linux
     *
     * @param tcb internal congestion state
     * @param rtt last rtt
     */
    void LinuxPktsAcked(Ptr<TcpSocketState> tcb, const Time& rtt);
};

} // namespace ns3

#endif // TCPLP_H
