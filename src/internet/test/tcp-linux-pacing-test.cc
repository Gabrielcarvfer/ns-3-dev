/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "tcp-test-pair.h"

#include "ns3/boolean.h"
#include "ns3/config.h"
#include "ns3/data-rate.h"
#include "ns3/inet-socket-address.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/log.h"
#include "ns3/node-container.h"
#include "ns3/packet.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/simulator.h"
#include "ns3/socket.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"

#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpLinuxPacingTest");

/**
 * @ingroup internet-test
 *
 * @brief Test the pacing of whole GSO packets, as the internal pacing of Linux
 *
 * A bulk transfer with pacing at 2 Mbit/s. With
 * ForcedPush, which models the GSO packets of Linux, the sender paces as
 * Linux (tcp_pacing_check(), tcp_update_skb_after_send()): after the first
 * ten segments, which are not paced, it sends whole GSO packets of several
 * segments at once, each delaying the next one by its transmission time at
 * the pacing rate, less the delay with which it was sent, up to half of it.
 * Without ForcedPush, the sender paces segment by segment.
 */
class TcpLinuxPacingTestCase : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param linux Value of the ForcedPush attribute of the sender.
     * @param desc The test description.
     */
    TcpLinuxPacingTestCase(bool linux, const std::string& desc)
        : TestCase(desc),
          m_linux(linux)
    {
    }

  private:
    void DoRun() override;

    /**
     * Trace sink of the segments the sender sends.
     * @param packet The payload.
     * @param header The TCP header.
     * @param socket The sending socket.
     */
    void SegmentSent(Ptr<const Packet> packet,
                     const TcpHeader& header,
                     Ptr<const TcpSocketBase> socket);

    /**
     * Accept callback: read everything as it arrives.
     * @param socket The accepted socket.
     * @param from The peer address.
     */
    void Accepted(Ptr<Socket> socket, const Address& from);

    /**
     * Receive callback of the receiver.
     * @param socket The receiving socket.
     */
    void Read(Ptr<Socket> socket);

    bool m_linux;                  //!< ForcedPush attribute of the sender
    uint32_t m_first{0};           //!< Sequence number of the first data byte
    uint32_t m_highest{0};         //!< Highest end of the data segments sent
    uint32_t m_segments{0};        //!< New data segments sent
    std::vector<Time> m_times;     //!< Transmission times of the bursts after the first ten
    std::vector<uint32_t> m_bytes; //!< Bytes of these bursts
};

void
TcpLinuxPacingTestCase::SegmentSent(Ptr<const Packet> packet,
                                    const TcpHeader& header,
                                    Ptr<const TcpSocketBase> socket)
{
    if (header.GetFlags() & TcpHeader::SYN)
    {
        m_first = header.GetSequenceNumber().GetValue() + 1;
        return;
    }
    uint32_t end = header.GetSequenceNumber().GetValue() + packet->GetSize() - m_first;
    if (packet->GetSize() == 0 || end <= m_highest)
    {
        return;
    }
    m_highest = end;
    if (++m_segments <= 10)
    {
        return;
    }
    if (!m_times.empty() && m_times.back() == Simulator::Now())
    {
        m_bytes.back() += packet->GetSize();
    }
    else
    {
        m_times.push_back(Simulator::Now());
        m_bytes.push_back(packet->GetSize());
    }
}

void
TcpLinuxPacingTestCase::Accepted(Ptr<Socket> socket, const Address& from)
{
    socket->SetRecvCallback(MakeCallback(&TcpLinuxPacingTestCase::Read, this));
}

void
TcpLinuxPacingTestCase::Read(Ptr<Socket> socket)
{
    while (socket->Recv())
    {
    }
}

void
TcpLinuxPacingTestCase::DoRun()
{
    const DataRate rate("2Mbps");
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(500));
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(10));
    Config::SetDefault("ns3::TcpSocketState::EnablePacing", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocketState::PaceInitialWindow", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocketState::MaxPacingRate", DataRateValue(rate));
    Config::SetDefault("ns3::TcpSocketBase::ForcedPush", BooleanValue(m_linux));

    NodeContainer nodes;
    Ipv4InterfaceContainer interfaces = InstallTcpTestPair(nodes, "", MilliSeconds(10));

    const uint16_t port = 9909;
    Ptr<Socket> server = InstallTcpListener(nodes.Get(1),
                                            port,
                                            MakeCallback(&TcpLinuxPacingTestCase::Accepted, this));

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->TraceConnectWithoutContext("Tx",
                                       MakeCallback(&TcpLinuxPacingTestCase::SegmentSent, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));
    Simulator::Schedule(Seconds(1), [client]() { client->Send(Create<Packet>(40000), 0); });

    Simulator::Stop(Seconds(30));
    Simulator::Run();

    NS_TEST_ASSERT_MSG_EQ(m_highest, 40000, "The transfer did not complete");
    NS_TEST_ASSERT_MSG_GT(m_times.size(), 10, "Too few paced bursts");
    uint32_t multi = 0;
    for (size_t i = 0; i + 1 < m_times.size(); ++i)
    {
        multi += m_bytes[i] > 500 ? 1 : 0;
        Time gap = m_times[i + 1] - m_times[i];
        Time len = rate.CalculateBytesTxTime(m_bytes[i]);
        if (m_linux)
        {
            NS_TEST_ASSERT_MSG_GT_OR_EQ(gap + NanoSeconds(1),
                                        len / 2,
                                        "Burst " << i << " paced less than half its length");
        }
        else
        {
            NS_TEST_ASSERT_MSG_GT_OR_EQ(gap + NanoSeconds(1),
                                        len,
                                        "Burst " << i << " paced less than its length");
        }
    }
    if (m_linux)
    {
        NS_TEST_ASSERT_MSG_GT(multi, m_times.size() / 2, "The GSO packets are not paced whole");
    }
    else
    {
        NS_TEST_ASSERT_MSG_EQ(multi, 0, "Segments paced together");
    }
    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 *
 * @brief Test suite of the pacing of whole GSO packets
 */
class TcpLinuxPacingTestSuite : public TestSuite
{
  public:
    TcpLinuxPacingTestSuite()
        : TestSuite("tcp-linux-pacing", Type::UNIT)
    {
        AddTestCase(new TcpLinuxPacingTestCase(false, "Pacing segment by segment"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpLinuxPacingTestCase(true, "Pacing whole GSO packets as Linux"),
                    TestCase::Duration::QUICK);
    }
};

static TcpLinuxPacingTestSuite
    g_tcpLinuxPacingTestSuite; //!< Static variable for test initialization
