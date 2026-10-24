/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "tcp-test-pair.h"

#include "ns3/boolean.h"
#include "ns3/config.h"
#include "ns3/data-rate.h"
#include "ns3/error-model.h"
#include "ns3/inet-socket-address.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-header.h"
#include "ns3/log.h"
#include "ns3/node-container.h"
#include "ns3/packet.h"
#include "ns3/pointer.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/simulator.h"
#include "ns3/socket.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/test.h"
#include "ns3/type-id.h"
#include "ns3/uinteger.h"

#include <set>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpLinuxRecoveryTest");

/**
 * @ingroup internet-test
 *
 * @brief Drop, once each, the TCP data segments starting at the given
 * offsets from the first data byte
 */
class TcpOffsetsErrorModel : public ErrorModel
{
  public:
    /**
     * @brief Add an offset of a segment to drop.
     * @param offset Bytes from the first data byte.
     */
    void AddOffset(uint32_t offset)
    {
        m_offsets.insert(offset);
    }

  private:
    bool DoCorrupt(Ptr<Packet> p) override
    {
        Ptr<Packet> copy = p->Copy();
        Ipv4Header ip;
        if (copy->PeekHeader(ip) == 0 || ip.GetProtocol() != 6)
        {
            return false;
        }
        copy->RemoveHeader(ip);
        TcpHeader tcp;
        copy->RemoveHeader(tcp);
        if (tcp.GetFlags() & TcpHeader::SYN)
        {
            m_first = tcp.GetSequenceNumber().GetValue() + 1;
            return false;
        }
        if (copy->GetSize() == 0)
        {
            return false;
        }
        return m_offsets.erase(tcp.GetSequenceNumber().GetValue() - m_first) > 0;
    }

    void DoReset() override
    {
    }

    std::set<uint32_t> m_offsets; //!< Offsets of the segments still to drop
    uint32_t m_first{0};          //!< Sequence number of the first data byte
};

/**
 * @ingroup internet-test
 *
 * @brief Base of the transfers of these tests: 20 segments of 1000 bytes,
 * written at once, with SACK, over a 10 Mbit/s link with an RTT of 20 ms
 */
class TcpLinuxRecoveryTransfer : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param desc The test description.
     */
    TcpLinuxRecoveryTransfer(const std::string& desc)
        : TestCase(desc)
    {
    }

  protected:
    /**
     * @brief Run the transfer.
     * @param drops Offsets of the segments to drop once.
     */
    void Transfer(const std::set<uint32_t>& drops);

    /**
     * Trace sink of the segments the sender sends.
     * @param packet The payload.
     * @param header The TCP header.
     * @param socket The sending socket.
     */
    virtual void SegmentSent(Ptr<const Packet> packet,
                             const TcpHeader& header,
                             Ptr<const TcpSocketBase> socket);

    /**
     * Trace sink of the segments the sender receives.
     * @param packet The payload.
     * @param header The TCP header.
     * @param socket The sending socket.
     */
    virtual void AckReceived(Ptr<const Packet> packet,
                             const TcpHeader& header,
                             Ptr<const TcpSocketBase> socket);

    /**
     * Trace sink of the smoothed RTT of the sender.
     * @param oldValue Previous value.
     * @param newValue New value.
     */
    virtual void SrttChange(Time oldValue, Time newValue);

    /**
     * Write the data.
     * @param socket The sending socket.
     */
    void Write(Ptr<Socket> socket);

    static constexpr uint32_t SEGMENT = 1000; //!< Segment size
    static constexpr uint32_t BYTES = 20000;  //!< Bytes to transfer

    uint32_t m_received{0}; //!< Bytes the receiving application read
    uint32_t m_first{0};    //!< Sequence number of the first data byte
};

void
TcpLinuxRecoveryTransfer::SegmentSent(Ptr<const Packet> packet,
                                      const TcpHeader& header,
                                      Ptr<const TcpSocketBase> socket)
{
    if (header.GetFlags() & TcpHeader::SYN)
    {
        m_first = header.GetSequenceNumber().GetValue() + 1;
    }
}

void
TcpLinuxRecoveryTransfer::AckReceived(Ptr<const Packet> packet,
                                      const TcpHeader& header,
                                      Ptr<const TcpSocketBase> socket)
{
}

void
TcpLinuxRecoveryTransfer::SrttChange(Time oldValue, Time newValue)
{
}

void
TcpLinuxRecoveryTransfer::Write(Ptr<Socket> socket)
{
    socket->Send(Create<Packet>(BYTES), 0);
}

void
TcpLinuxRecoveryTransfer::Transfer(const std::set<uint32_t>& drops)
{
    // The timestamps take 12 bytes of the MSS
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(SEGMENT + 12));
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(20));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(BYTES));

    NodeContainer nodes;
    nodes.Create(2);
    SimpleNetDeviceHelper devHelper;
    devHelper.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Mbps")));
    devHelper.SetChannelAttribute("Delay", TimeValue(MilliSeconds(10)));
    NetDeviceContainer devices = devHelper.Install(nodes);
    Ptr<TcpOffsetsErrorModel> errors = CreateObject<TcpOffsetsErrorModel>();
    for (uint32_t offset : drops)
    {
        errors->AddOffset(offset);
    }
    devices.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(errors));
    InternetStackHelper stack;
    stack.Install(nodes);
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    const uint16_t port = 9915;
    Ptr<Socket> server = InstallTcpByteSink(nodes.Get(1), port, m_received);

    Ptr<Socket> client = Socket::CreateSocket(nodes.Get(0), TcpSocketFactory::GetTypeId());
    client->Bind();
    client->TraceConnectWithoutContext("Tx",
                                       MakeCallback(&TcpLinuxRecoveryTransfer::SegmentSent, this));
    client->TraceConnectWithoutContext("Rx",
                                       MakeCallback(&TcpLinuxRecoveryTransfer::AckReceived, this));
    client->TraceConnectWithoutContext("RTT",
                                       MakeCallback(&TcpLinuxRecoveryTransfer::SrttChange, this));
    client->Connect(InetSocketAddress(interfaces.GetAddress(1), port));
    Simulator::Schedule(Seconds(1), &TcpLinuxRecoveryTransfer::Write, this, client);

    Simulator::Stop(Seconds(30));
    Simulator::Run();
    NS_TEST_ASSERT_MSG_EQ(m_received, BYTES, "The transfer did not complete");
    Simulator::Destroy();
    Config::Reset();
}

/**
 * @ingroup internet-test
 *
 * @brief Check that ForwardRetransmit controls the retransmission of a segment not deemed lost
 *
 * The 5th and 18th segments are lost. The 5th starts a SACK recovery; the 18th
 * has only two segments SACKed above it, so it is not deemed lost, and no new
 * data is available. With ForwardRetransmit (RFC 6675 NextSeg() rule (3)) the
 * 18th is retransmitted before the retransmitted 5th is ACKed; without it, as
 * in Linux, only segments deemed lost are retransmitted.
 */
class TcpForwardRetransmitTest : public TcpLinuxRecoveryTransfer
{
  public:
    /**
     * @brief Constructor.
     * @param forward Value of the ForwardRetransmit attribute.
     * @param desc The test description.
     */
    TcpForwardRetransmitTest(bool forward, const std::string& desc)
        : TcpLinuxRecoveryTransfer(desc),
          m_forward(forward)
    {
    }

  private:
    void DoRun() override;
    void SegmentSent(Ptr<const Packet> packet,
                     const TcpHeader& header,
                     Ptr<const TcpSocketBase> socket) override;
    void AckReceived(Ptr<const Packet> packet,
                     const TcpHeader& header,
                     Ptr<const TcpSocketBase> socket) override;

    bool m_forward;                //!< ForwardRetransmit attribute of the sender
    bool m_fifthAcked{false};      //!< The retransmission of the 5th was acknowledged
    bool m_earlyRetransmit{false}; //!< The 18th was retransmitted before that
    uint32_t m_sent18{0};          //!< Transmissions of the 18th segment
};

void
TcpForwardRetransmitTest::SegmentSent(Ptr<const Packet> packet,
                                      const TcpHeader& header,
                                      Ptr<const TcpSocketBase> socket)
{
    TcpLinuxRecoveryTransfer::SegmentSent(packet, header, socket);
    if (packet->GetSize() > 0 && header.GetSequenceNumber().GetValue() - m_first == 17 * SEGMENT)
    {
        if (++m_sent18 == 2 && !m_fifthAcked)
        {
            m_earlyRetransmit = true;
        }
    }
}

void
TcpForwardRetransmitTest::AckReceived(Ptr<const Packet> packet,
                                      const TcpHeader& header,
                                      Ptr<const TcpSocketBase> socket)
{
    if (m_first > 0 && header.GetAckNumber().GetValue() - m_first > 4 * SEGMENT)
    {
        m_fifthAcked = true;
    }
}

void
TcpForwardRetransmitTest::DoRun()
{
    Config::SetDefault("ns3::TcpSocketBase::ForwardRetransmit", BooleanValue(m_forward));
    // Classic recovery inflates the window with duplicate ACKs, opening room for a send while no
    // new data exists
    Config::SetDefault("ns3::TcpL4Protocol::RecoveryType",
                       TypeIdValue(TypeId::LookupByName("ns3::TcpClassicRecovery")));
    Transfer({4 * SEGMENT, 17 * SEGMENT});
    NS_TEST_ASSERT_MSG_EQ(m_earlyRetransmit,
                          m_forward,
                          "Unexpected retransmission of a segment not deemed lost");
}

/**
 * @ingroup internet-test
 *
 * @brief Check that LinuxCaRtt takes a timestamp RTT sample from the ACK of an RTO retransmission
 *
 * The last segment is lost and retransmitted on RTO. Its ACK covers only
 * retransmitted data, so Karn's rule gives no sample; with LinuxCaRtt the
 * echoed timestamp gives one (as tcp_ack_update_rtt()) and the smoothed RTT
 * changes.
 */
class TcpTimeoutRttSampleTest : public TcpLinuxRecoveryTransfer
{
  public:
    /**
     * @brief Constructor.
     * @param linuxCaRtt Value of the LinuxCaRtt attribute.
     * @param desc The test description.
     */
    TcpTimeoutRttSampleTest(bool linuxCaRtt, const std::string& desc)
        : TcpLinuxRecoveryTransfer(desc),
          m_linuxCaRtt(linuxCaRtt)
    {
    }

  private:
    void DoRun() override;
    void SegmentSent(Ptr<const Packet> packet,
                     const TcpHeader& header,
                     Ptr<const TcpSocketBase> socket) override;
    void SrttChange(Time oldValue, Time newValue) override;

    bool m_linuxCaRtt;             //!< LinuxCaRtt attribute of the sender
    uint32_t m_sentLast{0};        //!< Transmissions of the last segment
    bool m_updatedAfterRto{false}; //!< The smoothed RTT changed after the timeout
};

void
TcpTimeoutRttSampleTest::SegmentSent(Ptr<const Packet> packet,
                                     const TcpHeader& header,
                                     Ptr<const TcpSocketBase> socket)
{
    TcpLinuxRecoveryTransfer::SegmentSent(packet, header, socket);
    if (packet->GetSize() > 0 && header.GetSequenceNumber().GetValue() - m_first == 19 * SEGMENT)
    {
        ++m_sentLast;
    }
}

void
TcpTimeoutRttSampleTest::SrttChange(Time oldValue, Time newValue)
{
    if (m_sentLast >= 2 && oldValue != newValue)
    {
        m_updatedAfterRto = true;
    }
}

void
TcpTimeoutRttSampleTest::DoRun()
{
    Config::SetDefault("ns3::TcpSocketBase::LinuxCaRtt", BooleanValue(m_linuxCaRtt));
    Transfer({19 * SEGMENT});
    NS_TEST_ASSERT_MSG_EQ(m_sentLast, 2, "The last segment was not retransmitted once");
    NS_TEST_ASSERT_MSG_EQ(m_updatedAfterRto,
                          m_linuxCaRtt,
                          "Unexpected RTT sample after the timeout");
}

/**
 * @ingroup internet-test
 *
 * @brief Test suite of the Linux behaviors in loss recovery
 */
class TcpLinuxRecoveryTestSuite : public TestSuite
{
  public:
    TcpLinuxRecoveryTestSuite()
        : TestSuite("tcp-linux-recovery", Type::UNIT)
    {
        AddTestCase(new TcpForwardRetransmitTest(true, "Rule (3) retransmits as RFC 6675"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpForwardRetransmitTest(false, "Only lost segments retransmitted"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpTimeoutRttSampleTest(false, "No RTT sample after the timeout"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpTimeoutRttSampleTest(true, "Timestamp RTT sample after the timeout"),
                    TestCase::Duration::QUICK);
    }
};

static TcpLinuxRecoveryTestSuite g_tcpLinuxRecoveryTestSuite; //!< Static variable for test init
