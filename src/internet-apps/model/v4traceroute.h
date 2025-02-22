/*
 * Copyright (c) 2019 Ritsumeikan University, Shiga, Japan
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Alberto Gallegos Ramonet <ramonet@fc.ritsumei.ac.jp>
 */

#ifndef V4TRACEROUTE_H
#define V4TRACEROUTE_H

#include "ns3/application.h"
#include "ns3/average.h"
#include "ns3/nstime.h"
#include "ns3/output-stream-wrapper.h"
#include "ns3/simulator.h"
#include "ns3/traced-callback.h"

#include <map>

namespace ns3
{

class Socket;

/**
 * @ingroup internet-apps
 * @defgroup v4traceroute V4Traceroute
 */

/**
 * @ingroup v4traceroute
 * @brief Traceroute application sends one ICMP ECHO request with TTL=1,
 *        and after receiving an ICMP TIME EXCEED reply, it increases the
 *        TTL and repeat the process to reveal all the intermediate hops to
 *        the destination.
 *
 */
class V4TraceRoute : public Application
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();
    V4TraceRoute();
    ~V4TraceRoute() override;
    /**
     * @brief Prints the application traced routes into a given OutputStream.
     * @param stream the output stream
     */
    void Print(Ptr<OutputStreamWrapper> stream);

  protected:
    void DoDispose() override;

  private:
    void StartApplication() override;
    void StopApplication() override;

    /**
     * @brief Return the application ID in the node.
     * @returns the application id
     */
    uint32_t GetApplicationId() const;
    /**
     * @brief Receive an ICMP Echo
     * @param socket the receiving socket
     *
     * This function is called by lower layers through a callback.
     */
    void Receive(Ptr<Socket> socket);

    /** @brief Send one (ICMP ECHO) to the destination.*/
    void Send();

    /** @brief Starts a timer after sending an ICMP ECHO.*/
    void StartWaitReplyTimer();

    /** @brief Triggers an action if an ICMP TIME EXCEED have not being received
     *         in the time defined by StartWaitReplyTimer.
     */
    void HandleWaitReplyTimeout();

    /// Remote address
    Ipv4Address m_remote;

    /// Wait  interval  seconds between sending each packet
    Time m_interval;
    /**
     * Specifies  the number of data bytes to be sent.
     * The default is 56, which translates into 64 ICMP data bytes when
     * combined with the 8 bytes of ICMP header data.
     */
    uint32_t m_size;
    /// The socket we send packets from
    Ptr<Socket> m_socket;
    /// ICMP ECHO sequence number
    uint16_t m_seq;
    /// produce traceroute style output if true
    bool m_verbose;
    /// Start time to report total ping time
    Time m_started;
    /// Next packet will be sent
    EventId m_next;
    /// The Current probe value
    uint32_t m_probeCount;
    /// The maximum number of probe packets per hop
    uint16_t m_maxProbes;
    /// The current TTL value
    uint16_t m_ttl;
    /// The packets Type of Service
    uint8_t m_tos;
    /// The maximum Ttl (Max number of hops to trace)
    uint32_t m_maxTtl;
    /// The wait time until the response is considered lost.
    Time m_waitIcmpReplyTimeout;
    /// The timer used to wait for the probes ICMP replies
    EventId m_waitIcmpReplyTimer;
    /// All sent but not answered packets. Map icmp seqno -> when sent
    std::map<uint16_t, Time> m_sent;

    /// Stream of characters used for printing a single route
    std::ostringstream m_osRoute;
    /// The Ipv4 address of the latest hop found
    std::ostringstream m_routeIpv4;
    /// Stream of the traceroute used for the output file
    Ptr<OutputStreamWrapper> m_printStream;
};

} // namespace ns3

#endif /*V4TRACEROUTE_H*/
