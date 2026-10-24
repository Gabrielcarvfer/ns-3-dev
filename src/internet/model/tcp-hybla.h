/*
 * Copyright (c) 2014 Natale Patriciello <natale.patriciello@gmail.com>
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */
#ifndef TCPHYBLA_H
#define TCPHYBLA_H

#include "tcp-congestion-ops.h"

#include "ns3/traced-value.h"

namespace ns3
{

class TcpSocketState;

/**
 * @ingroup congestionOps
 *
 * @brief Implementation of the TCP Hybla algorithm
 *
 * The key idea behind TCP Hybla is to obtain for long RTT connections the same
 * instantaneous transmission rate of a reference TCP connection with lower RTT.
 * With analytical steps, it is shown that this goal can be achieved by
 * modifying the time scale, in order for the throughput to be independent from
 * the RTT. This independence is obtained through the use of a coefficient rho.
 *
 * This coefficient is used to calculate both the slow start threshold
 * and the congestion window when in slow start and in congestion avoidance,
 * respectively.
 *
 * Both increases apply once per ACK, as in the Hybla paper and in Linux,
 * whatever the number of segments the ACK acknowledges. The attributes
 * RenoOutsideOpen, RhoInitialWindow and FixedPoint (all false by default)
 * reproduce the details of the Linux implementation.
 *
 * More information: http://dl.acm.org/citation.cfm?id=2756518
 */
class TcpHybla : public TcpNewReno
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
    TcpHybla();

    /**
     * @brief Copy constructor
     * @param sock the object to copy
     */
    TcpHybla(const TcpHybla& sock);

    ~TcpHybla() override;

    // Inherited
    void PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt) override;
    void IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) override;
    uint32_t GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) override;
    void CongestionStateSet(Ptr<TcpSocketState> tcb,
                            const TcpSocketState::TcpCongState_t newState) override;
    void CwndEvent(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event) override;
    std::string GetName() const override;
    Ptr<TcpCongestionOps> Fork() override;

  protected:
    uint32_t SlowStart(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) override;
    void CongestionAvoidance(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) override;

  private:
    TracedValue<double> m_rho;      //!< Rho parameter
    Time m_rRtt;                    //!< Reference RTT
    double m_cWndCnt;               //!< cWnd integer-to-float counter
    Time m_rhoRtt;                  //!< Minimum smoothed RTT rho was computed from
    bool m_renoOutsideOpen{false};  //!< Grow as NewReno outside the Open state
    bool m_rhoInitialWindow{false}; //!< Set the initial window to rho segments
    bool m_fixedPoint{false};       //!< Use the fixed-point arithmetic of Linux
    uint32_t m_rho3ls{8};           //!< Rho in 1/8, with FixedPoint
    uint32_t m_ackCnt{0};           //!< ACKs without window increase, with FixedPoint
    bool m_open{true};              //!< Whether the connection is in the Open state
    bool m_initialWindowSet{false}; //!< Whether the initial window was set to rho

  private:
    /**
     * @brief Recalculate algorithm parameters
     * @param tcb the socket state.
     */
    void RecalcParam(const Ptr<TcpSocketState>& tcb);

    /**
     * @brief Add an increment in 1/128 of a segment to the window, carrying
     * the fraction to the next increments, as the FixedPoint attribute sets
     * @param tcb the socket state.
     * @param increment the increment, in 1/128 of a segment
     */
    void AddFixedPointIncrement(Ptr<TcpSocketState> tcb, uint32_t increment);
};

} // namespace ns3

#endif // TCPHYBLA_H
