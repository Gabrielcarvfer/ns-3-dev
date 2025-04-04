/*
 * Copyright (c) 2009 IITP RAS
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pavel Boyko <boyko@iitp.ru>
 */

#ifndef TC_REGRESSION_TEST_H
#define TC_REGRESSION_TEST_H

#include "ns3/ipv4-raw-socket-impl.h"
#include "ns3/node-container.h"
#include "ns3/nstime.h"
#include "ns3/socket.h"
#include "ns3/test.h"

namespace ns3
{
namespace olsr
{
/**
 * @ingroup olsr-test
 * @ingroup tests
 *
 * @brief Less trivial test of OLSR Topology Control message generation
 *
 * This test simulates 3 Wi-Fi stations with chain topology and runs OLSR without any extra traffic.
 * It is expected that only second station will send TC messages.
 *
 * Expected trace (20 seconds, note random b-cast jitter):
 */
// clang-format off
/**
 * \verbatim
          1       2       3
          |<------|------>|         HELLO (empty) src = 10.1.1.2
          |       |<------|------>  HELLO (empty) src = 10.1.1.3
   <------|------>|       |         HELLO (empty) src = 10.1.1.1
   <------|------>|       |         HELLO (Link Type: Asymmetric, Neighbor: 10.1.1.2) src = 10.1.1.1
          |       |<------|------>  HELLO (Link Type: Asymmetric, Neighbor: 10.1.1.2) src = 10.1.1.3
          |<------|------>|         HELLO (Link Type: Asymmetric, Neighbor: 10.1.1.3; Link Type: Asymmetric, Neighbor: 10.1.1.1) src = 10.1.1.2
          |<------|------>|         HELLO (Link Type: Asymmetric, Neighbor: 10.1.1.3; Link Type: Asymmetric, Neighbor: 10.1.1.1) src = 10.1.1.2
   <------|------>|       |         HELLO (Link Type: Symmetric, Neighbor: 10.1.1.2) src = 10.1.1.1
          |       |<------|------>  HELLO (Link Type: Symmetric, Neighbor: 10.1.1.2) src = 10.1.1.3
          |<------|------>|         HELLO (Link Type: Symmetric, Neighbor: 10.1.1.3; Link Type: Symmetric, Neighbor: 10.1.1.1) src = 10.1.1.2
   <------|------>|       |         HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.1
          |       |<------|------>  HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.3
          |<------|------>|         HELLO (Link Type: Symmetric, Neighbor: 10.1.1.3; Link Type: Symmetric, Neighbor: 10.1.1.1) src = 10.1.1.2
   <------|------>|       |         HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.1
          |       |<------|------>  HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.3
          |<======|======>|         TC (10.1.1.3; 10.1.1.1) + HELLO (Link Type: Symmetric, Neighbor: 10.1.1.3; Link Type: Symmetric, Neighbor: 10.1.1.1) src = 10.1.1.2
          |       |<------|------>  HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.3
   <------|------>|       |         HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.1
          |<------|------>|         HELLO (Link Type: Symmetric, Neighbor: 10.1.1.3; Link Type: Symmetric, Neighbor: 10.1.1.1) src = 10.1.1.2
   <------|------>|       |         HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.1
          |       |<------|------>  HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.3
   <------|------>|       |         HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.1
          |       |<------|------>  HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.3
          |<------|------>|         HELLO (Link Type: Symmetric, Neighbor: 10.1.1.3; Link Type: Symmetric, Neighbor: 10.1.1.1) src = 10.1.1.2
          |<======|======>|         TC (10.1.1.3; 10.1.1.1) src = 10.1.1.2
          |       |<------|------>  HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.3
   <------|------>|       |         HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.1
          |<------|------>|         HELLO (Link Type: Symmetric, Neighbor: 10.1.1.3; Link Type: Symmetric, Neighbor: 10.1.1.1) src = 10.1.1.2
   <------|------>|       |         HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.1
          |       |<------|------>  HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.3
          |<------|------>|         HELLO (Link Type: Symmetric, Neighbor: 10.1.1.3; Link Type: Symmetric, Neighbor: 10.1.1.1) src = 10.1.1.2
   \endverbatim
 */
// clang-format on

class TcRegressionTest : public TestCase
{
  public:
    TcRegressionTest();
    ~TcRegressionTest() override;

  private:
    /// Total simulation time
    const Time m_time;
    /// Create & configure test network
    void CreateNodes();
    void DoRun() override;

    /**
     * Receive raw data on node A
     * @param socket receiving socket
     */
    void ReceivePktProbeA(Ptr<Socket> socket);
    /// Packet counter on node A
    uint8_t m_countA;
    /// Receiving socket on node A
    Ptr<Ipv4RawSocketImpl> m_rxSocketA;

    /**
     * Receive raw data on node B
     * @param socket receiving socket
     */
    void ReceivePktProbeB(Ptr<Socket> socket);
    /// Packet counter on node B
    uint8_t m_countB;
    /// Receiving socket on node B
    Ptr<Ipv4RawSocketImpl> m_rxSocketB;

    /**
     * Receive raw data on node C
     * @param socket receiving socket
     */
    void ReceivePktProbeC(Ptr<Socket> socket);
    /// Packet counter on node C
    uint8_t m_countC;
    /// Receiving socket on node C
    Ptr<Ipv4RawSocketImpl> m_rxSocketC;
};

} // namespace olsr
} // namespace ns3

#endif /* TC_REGRESSION_TEST_H */
