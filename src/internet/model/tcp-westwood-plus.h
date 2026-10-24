/*
 * Copyright (c) 2013 ResiliNets, ITTC, University of Kansas
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Siddharth Gangadhar <siddharth@ittc.ku.edu>, Truc Anh N. Nguyen <annguyen@ittc.ku.edu>,
 * and Greeshma Umapathi
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

#ifndef TCP_WESTWOOD_H
#define TCP_WESTWOOD_H

#include "tcp-congestion-ops.h"
#include "tcp-recovery-ops.h"

#include "ns3/data-rate.h"
#include "ns3/event-id.h"
#include "ns3/traced-value.h"

namespace ns3
{

class Time;

/**
 * @ingroup congestionOps
 *
 * @brief An implementation of TCP Westwood+.
 *
 * Westwood+ employ the AIAD (Additive Increase/Adaptive Decrease)
 * congestion control paradigm. When a congestion episode happens,
 * instead of halving the cwnd, these protocols try to estimate the network's
 * bandwidth and use the estimated value to adjust the cwnd.
 * While Westwood performs the bandwidth sampling every ACK reception,
 * Westwood+ samples the bandwidth every RTT.
 *
 * The two main methods in the implementation are the CountAck (const TCPHeader&)
 * and the EstimateBW (int, const, Time). The CountAck method calculates
 * the number of acknowledged segments on the receipt of an ACK.
 * The EstimateBW estimates the bandwidth based on the value returned by CountAck
 * and the sampling interval (last RTT).
 *
 * WARNING: this TCP model lacks validation and regression tests; use with caution.
 */
class TcpWestwoodPlus : public TcpNewReno
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    TcpWestwoodPlus();
    /**
     * @brief Copy constructor
     * @param sock the object to copy
     */
    TcpWestwoodPlus(const TcpWestwoodPlus& sock);
    ~TcpWestwoodPlus() override;

    /**
     * @brief Filter type (None or Tustin)
     */
    enum FilterType
    {
        NONE,
        TUSTIN,
        LINUX
    };

    uint32_t GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) override;

    void CwndEvent(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event) override;

    void PktsAcked(Ptr<TcpSocketState> tcb, uint32_t packetsAcked, const Time& rtt) override;

    Ptr<TcpCongestionOps> Fork() override;

  private:
    /**
     * Update the total number of acknowledged packets during the current RTT
     *
     * @param [in] acked the number of packets the currently received ACK acknowledges
     */
    void UpdateAckedSegments(int acked);

    /**
     * Estimate the network's bandwidth
     *
     * @param [in] rtt the RTT estimation.
     * @param [in] tcb the socket state.
     */
    void EstimateBW(const Time& rtt, Ptr<TcpSocketState> tcb);

    /**
     * Compute the slow start threshold from the bandwidth estimate and the
     * minimum RTT
     *
     * @param [in] tcb the socket state.
     * @return the slow start threshold, in bytes
     */
    uint32_t EstimatedSsThresh(Ptr<const TcpSocketState> tcb) const;

    /**
     * @brief Initialize the Linux estimator (tcp_westwood_init()), once
     */
    void LinuxStart();

    /**
     * Update the Linux estimator on an ACK, as tcp_westwood_ack() and
     * tcp_westwood_pkts_acked() of Linux do
     *
     * @param [in] tcb the socket state.
     * @param [in] rtt the RTT sample of the ACK.
     */
    void LinuxPktsAcked(Ptr<TcpSocketState> tcb, const Time& rtt);

    /**
     * @return the current time in jiffies of the Linux estimator
     */
    uint32_t Jiffies() const;

  protected:
    TracedValue<DataRate> m_currentBW; //!< Current value of the estimated BW
    DataRate m_lastSampleBW;           //!< Last bandwidth sample
    DataRate m_lastBW;                 //!< Last bandwidth sample after being filtered
    FilterType m_fType;                //!< 0 for none, 1 for Tustin, 2 for Linux
    bool m_linuxRecovery{false};       //!< Apply the estimate when the recovery ends

    uint32_t m_ackedSegments;  //!< The number of segments ACKed between RTTs
    bool m_IsCount;            //!< Start keeping track of m_ackedSegments for Westwood+ if TRUE
    EventId m_bwEstimateEvent; //!< The BW estimation event for Westwood+
    Time m_lastAck;            //!< The last ACK time

    uint32_t m_linuxHz{1000};     //!< Rate of the jiffies of the Linux estimator
    uint32_t m_bwNsEst{0};        //!< First filtered estimate, bytes per jiffy (Linux)
    uint32_t m_bwEst{0};          //!< Bandwidth estimate, bytes per jiffy (Linux)
    uint32_t m_rttWinSx{0};       //!< Start of the sampling window, in jiffies (Linux)
    uint32_t m_bk{0};             //!< Bytes acknowledged in the sampling window (Linux)
    SequenceNumber32 m_sndUna{0}; //!< Highest acknowledged sequence counted (Linux)
    uint32_t m_accounted{0};      //!< Bytes counted by duplicate ACKs (Linux)
    uint32_t m_rttJiffies{0};     //!< Last RTT sample, in jiffies (Linux)
    uint32_t m_rttMinJiffies{0};  //!< Minimum RTT, in jiffies (Linux)
    bool m_firstAck{true};        //!< No ACK processed yet (Linux)
    bool m_resetRttMin{true};     //!< Restart the minimum RTT from the next sample (Linux)
    bool m_linuxStarted{false};   //!< The Linux estimator has been initialized
};

} // namespace ns3

#endif /* TCP_WESTWOOD_H */
