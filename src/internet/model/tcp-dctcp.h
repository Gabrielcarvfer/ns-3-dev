/*
 * Copyright (c) 2017 NITK Surathkal
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Shravya K.S. <shravya.ks0@gmail.com>
 *
 */

#ifndef TCP_DCTCP_H
#define TCP_DCTCP_H

#include "tcp-congestion-ops.h"
#include "tcp-linux-reno.h"

#include "ns3/traced-callback.h"

namespace ns3
{

/**
 * @ingroup tcp
 *
 * @brief An implementation of DCTCP. This model implements all of the
 * endpoint capabilities mentioned in the DCTCP SIGCOMM paper.  Note that
 * this model presently will not do any fallback to RFC 5681 congestion
 * avoidance as specified in RFC 8257 Section 3.5, so it should only be
 * used in a simulation that does not involve packet loss.
 */

class TcpDctcp : public TcpLinuxReno
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
    TcpDctcp();

    /**
     * @brief Copy constructor
     * @param sock the object to copy
     */
    TcpDctcp(const TcpDctcp& sock);

    /**
     * @brief Destructor
     */
    ~TcpDctcp() override;

    // Documented in base class
    std::string GetName() const override;

    /**
     * @brief Set configuration required by congestion control algorithm,
     *        This method will force DctcpEcn mode and will force usage of
     *        either ECT(0) or ECT(1) (depending on the 'UseEct0' attribute),
     *        despite any other configuration in the base classes.
     *
     * @param tcb internal congestion state
     */
    void Init(Ptr<TcpSocketState> tcb) override;

    /**
     * TracedCallback signature for DCTCP update of congestion state
     *
     * @param [in] bytesAcked Bytes acked in this observation window
     * @param [in] bytesMarked Bytes marked in this observation window
     * @param [in] alpha New alpha (congestion estimate) value
     */
    typedef void (*CongestionEstimateTracedCallback)(uint32_t bytesAcked,
                                                     uint32_t bytesMarked,
                                                     double alpha);

    // Documented in base class
    uint32_t GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) override;
    Ptr<TcpCongestionOps> Fork() override;
    void PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt) override;
    void CwndEvent(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event) override;

  private:
    /**
     * @brief Changes state of m_ceState to true
     *
     * @param tcb internal congestion state
     */
    void CeState0to1(Ptr<TcpSocketState> tcb);

    /**
     * @brief Changes state of m_ceState to false
     *
     * @param tcb internal congestion state
     */
    void CeState1to0(Ptr<TcpSocketState> tcb);

    /**
     * @brief Updates the value of m_delayedAckReserved
     *
     * @param tcb internal congestion state
     * @param event the congestion window event
     */
    void UpdateAckReserved(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event);

    /**
     * @brief Resets the value of m_ackedBytesEcn, m_ackedBytesTotal and m_nextSeq
     *
     * @param tcb internal congestion state
     */
    void Reset(Ptr<TcpSocketState> tcb);

    /**
     * @brief Initialize the value of m_alpha
     *
     * @param alpha DCTCP alpha parameter
     */
    void InitializeDctcpAlpha(double alpha);

    uint32_t m_ackedBytesEcn;       //!< Number of acked bytes which are marked
    uint32_t m_ackedBytesTotal;     //!< Total number of acked bytes
    SequenceNumber32 m_priorRcvNxt; //!< Sequence number of the first missing byte in data
    bool m_priorRcvNxtFlag; //!< Variable used in setting the value of m_priorRcvNxt for first time
    double m_alpha;         //!< Parameter used to estimate the amount of network congestion
    SequenceNumber32
        m_nextSeq;      //!< TCP sequence number threshold for beginning a new observation window
    bool m_nextSeqFlag; //!< Variable used in setting the value of m_nextSeq for first time
    bool m_ceState;     //!< DCTCP Congestion Experienced state
    bool m_delayedAckReserved; //!< Delayed Ack state
    double m_g;                //!< Estimation gain
    bool m_useEct0;            //!< Use ECT(0) for ECN codepoint
    bool m_initialized;        //!< Whether DCTCP has been initialized
    /**
     * @brief Callback pointer for congestion state update
     */
    TracedCallback<uint32_t, uint32_t, double> m_traceCongestionEstimate;
};

} // namespace ns3

#endif /* TCP_DCTCP_H */
