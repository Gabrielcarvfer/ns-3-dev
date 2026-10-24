/*
 * Copyright (c) 2015 ResiliNets, ITTC, University of Kansas
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * by: Amir Modarresi <amodarresi@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  https://resilinets.org/
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 */

#ifndef TCP_HTCP_H
#define TCP_HTCP_H

#include "tcp-congestion-ops.h"

namespace ns3
{

class TcpSocketState;

/**
 * @ingroup congestionOps
 *
 * @brief An implementation of the H-TCP variant of TCP.
 *
 * This class contains the H-TCP implementation of TCP, according to
 * Internet-Draft draft-leith-tcp-htcp-03 and its related paper,
 * "H-TCP: TCP for high-speed and long-distance networks"
 * H-TCP is a congestion control protocol suitable for high bandwidth-delay
 * product networks. It is fair to similar flows present in the network and
 * also friendly with conventional TCP. It also makes use of free
 * bandwidth when it is available.
 */
class TcpHtcp : public TcpNewReno
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();
    /**
     * Create an unbound tcp socket.
     */
    TcpHtcp();
    /**
     * @brief Copy constructor
     * @param sock the object to copy
     */
    TcpHtcp(const TcpHtcp& sock);
    ~TcpHtcp() override;
    std::string GetName() const override;
    Ptr<TcpCongestionOps> Fork() override;
    uint32_t GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) override;

    void PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt) override;
    void IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) override;
    void CongestionStateSet(Ptr<TcpSocketState> tcb,
                            const TcpSocketState::TcpCongState_t newState) override;
    void Init(Ptr<TcpSocketState> tcb) override;

  protected:
    void CongestionAvoidance(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) override;

  private:
    /**
     * @brief State of the Linux tcp_htcp.c (struct htcp), times in jiffies
     */
    struct LinuxState
    {
        uint32_t alpha{1 << 7};   //!< Additive increase, << 7
        uint8_t beta{1 << 6};     //!< Multiplicative decrease, << 7
        bool modeswitch{false};   //!< Adaptive backoff enabled after a congestion event
        uint16_t pktsAcked{1};    //!< Segments acknowledged by the last ACK in CA_OPEN
        uint32_t packetCount{0};  //!< Segments acknowledged since lastTime
        uint32_t minRtt{0};       //!< Minimum RTT
        uint32_t maxRtt{0};       //!< Maximum RTT
        uint32_t lastCong{0};     //!< Time of the end of the last congestion event
        uint32_t undoLastCong{0}; //!< lastCong before the last congestion event
        uint32_t undoMaxRtt{0};   //!< maxRtt before the last congestion event
        uint32_t undoOldMaxB{0};  //!< oldMaxB before the last congestion event
        uint32_t minB{0};         //!< Minimum throughput
        uint32_t maxB{0};         //!< Maximum throughput
        uint32_t oldMaxB{0};      //!< maxB at the last congestion event
        uint32_t bi{0};           //!< Throughput estimate
        uint32_t lastTime{0};     //!< Start of the throughput sample
        uint32_t cwndCnt{0};      //!< Linux snd_cwnd_cnt
        bool resetPending{false}; //!< A congestion event awaits its htcp_reset()
        uint32_t resetJiffies{0}; //!< Time of that congestion event, in jiffies
        Time resetTime{-1};       //!< Time of that congestion event
        Time ssThreshTime{-1};    //!< Time of the last GetSsThresh()
    };

    /**
     * @brief Current time in jiffies of Linux, at LinuxHz
     * @return the jiffies
     */
    uint32_t Jiffies() const;

    /**
     * @brief Linux htcp_alpha_update()
     */
    void LinuxAlphaUpdate();

    /**
     * @brief Linux htcp_reset(), at the given time
     * @param jiffies the time of the congestion event
     */
    void LinuxReset(uint32_t jiffies);

    /**
     * @brief Apply a pending htcp_reset() of a congestion event whose
     * threshold was not computed
     */
    void LinuxApplyPendingReset();

    /**
     * @brief Updates the additive increase parameter for H-TCP
     */
    void UpdateAlpha();

    /**
     * @brief Updates the multiplicative decrease factor beta for H-TCP
     */
    void UpdateBeta();

    // h-tcp variables
    double m_alpha;           //!< AIMD additive increase parameter
    double m_beta;            //!< AIMD multiplicative decrease factor
    double m_defaultBackoff;  //!< default value when throughput ratio less than default
    double m_throughputRatio; //!< ratio of two consequence throughput
    Time m_delta;             //!< Time in second that has elapsed since the
                              // last congestion event experienced by a flow
    Time m_deltaL; //!< Threshold for switching between standard and new increase function
    bool m_rttScaling{false};      //!< Scale the increase function by the minimum RTT
    Time m_lowestRtt{Time::Max()}; //!< Minimum RTT of the connection
    Time m_lastCon;                //!< Time of the last congestion for the flow
    Time m_minRtt;                 //!< Minimum RTT in each congestion period
    Time m_maxRtt;                 //!< Maximum RTT in each congestion period
    uint32_t m_throughput;         //!< Current throughput since last congestion
    uint32_t m_lastThroughput;     //!< Throughput in last congestion period
    uint32_t m_dataSent;           //!< Current amount of data sent since last congestion
    uint32_t m_linuxHz{0};         //!< Rate of the Linux jiffies, zero without Linux arithmetic
    LinuxState m_lx;               //!< State of the Linux arithmetic
};

} // namespace ns3

#endif /* TCP_HTCP_H */
