/*
 * Copyright (c) 2026 Centre Tecnològic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "tcp-general-test.h"

#include "ns3/config.h"
#include "ns3/ipv4-header.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/simple-channel.h"
#include "ns3/simple-net-device.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-option-sack.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/test.h"

#include <set>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpDisorderAckTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief A channel which reorders, drops and hides segments so that the
 * sender receives, in CA_DISORDER, an ACK which advances SND.UNA and SACKs
 * new data at once.
 *
 * With the data segments numbered from the first one (X = first + 4), the
 * channel holds X until X + 3 has been sent, drops X + 2, and drops the
 * duplicate ACK which first SACKs X + 3. The receiver gets X + 1, X + 3 and
 * X: the ACK of X advances SND.UNA to X + 2 and SACKs X + 3, which the
 * sender had not seen.
 */
class TcpDisorderAckChannel : public SimpleChannel
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::TcpDisorderAckChannel")
                                .SetParent<SimpleChannel>()
                                .SetGroupName("Internet")
                                .AddConstructor<TcpDisorderAckChannel>();
        return tid;
    }

    /**
     * @brief Set the segment size and the first sequence number reordered.
     * @param segSize The segment size.
     * @param seq The sequence number of X, relative to the ISN of the sender.
     */
    void Configure(uint32_t segSize, uint32_t seq)
    {
        m_segSize = segSize;
        m_x = seq;
    }

    void Send(Ptr<Packet> p,
              uint16_t protocol,
              Mac48Address to,
              Mac48Address from,
              Ptr<SimpleNetDevice> sender) override
    {
        Ptr<Packet> copy = p->Copy();
        Ipv4Header ipHeader;
        TcpHeader tcpHeader;
        copy->RemoveHeader(ipHeader);
        copy->RemoveHeader(tcpHeader);
        if (tcpHeader.GetFlags() == TcpHeader::SYN)
        {
            m_isn = tcpHeader.GetSequenceNumber();
            m_dataSrc = ipHeader.GetSource();
        }
        bool isData = copy->GetSize() > 0 && ipHeader.GetSource() == m_dataSrc;
        if (isData)
        {
            uint32_t seq = tcpHeader.GetSequenceNumber().GetValue() - m_isn.GetValue();
            bool first = m_sent.insert(seq).second;
            if (first && seq == m_x)
            {
                m_held = p;
                m_heldProtocol = protocol;
                m_heldTo = to;
                m_heldFrom = from;
                m_heldSender = sender;
                return;
            }
            if (first && seq == m_x + 2 * m_segSize)
            {
                return;
            }
            SimpleChannel::Send(p, protocol, to, from, sender);
            if (m_held && seq == m_x + 3 * m_segSize)
            {
                SimpleChannel::Send(m_held, m_heldProtocol, m_heldTo, m_heldFrom, m_heldSender);
                m_held = nullptr;
                m_heldSender = nullptr;
            }
            return;
        }
        if (!m_ackDropped && tcpHeader.HasOption(TcpOption::SACK))
        {
            auto sack = DynamicCast<const TcpOptionSack>(tcpHeader.GetOption(TcpOption::SACK));
            for (const auto& [begin, end] : sack->GetSackList())
            {
                if (end.GetValue() - m_isn.GetValue() == m_x + 4 * m_segSize)
                {
                    m_ackDropped = true;
                    return;
                }
            }
        }
        SimpleChannel::Send(p, protocol, to, from, sender);
    }

  private:
    uint32_t m_segSize{0};             //!< Segment size
    uint32_t m_x{0};                   //!< Sequence number of X
    SequenceNumber32 m_isn{0};         //!< Initial sequence number of the data sender
    Ipv4Address m_dataSrc;             //!< Address of the data sender
    std::set<uint32_t> m_sent;         //!< Data sequence numbers sent so far
    bool m_ackDropped{false};          //!< Whether the ACK SACKing X + 3 was dropped
    Ptr<Packet> m_held;                //!< The held segment X
    uint16_t m_heldProtocol{0};        //!< Protocol of X
    Mac48Address m_heldTo;             //!< Destination of X
    Mac48Address m_heldFrom;           //!< Source of X
    Ptr<SimpleNetDevice> m_heldSender; //!< Sender of X
};

NS_OBJECT_ENSURE_REGISTERED(TcpDisorderAckChannel);

/**
 * @ingroup internet-test
 *
 * @brief Test an ACK which advances SND.UNA and SACKs new data in
 * CA_DISORDER.
 *
 * Such an ACK is a duplicate ACK for the loss detection of @RFC{6675}, and
 * the sender stays in CA_DISORDER, but it acknowledges new data: the window
 * grows (@RFC{5681}, Section 3.1). During slow start, it grows by one
 * segment, or, with TcpNewReno::CountAckedSegments, by the two segments
 * delivered (the one acknowledged and the one newly SACKed, as Linux
 * acked_sacked).
 */
class TcpDisorderAckTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     * @param countAcked Whether TcpNewReno::CountAckedSegments is enabled.
     */
    TcpDisorderAckTest(const std::string& name, bool countAcked)
        : TcpGeneralTest(name),
          m_countAcked(countAcked)
    {
    }

  protected:
    void ConfigureEnvironment() override;
    void DoTeardown() override;
    void ConfigureProperties() override;
    Ptr<SimpleChannel> CreateChannel() override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    void RcvAck(const Ptr<const TcpSocketState> tcb, const TcpHeader& h, SocketWho who) override;
    void ProcessedAck(const Ptr<const TcpSocketState> tcb,
                      const TcpHeader& h,
                      SocketWho who) override;
    void FinalChecks() override;

  private:
    static constexpr uint32_t X = 2001; //!< Relative sequence number of X

    bool m_countAcked;        //!< Whether CountAckedSegments is enabled
    bool m_isTarget{false};   //!< Whether the ACK being processed is the target one
    uint32_t m_cwndBefore{0}; //!< Window before the target ACK
    bool m_seen{false};       //!< Whether the target ACK was processed in CA_DISORDER
    uint32_t m_cwndAfter{0};  //!< Window after the target ACK
};

void
TcpDisorderAckTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(40);
    SetPropagationDelay(MilliSeconds(50));
    Config::SetDefault("ns3::TcpNewReno::CountAckedSegments", BooleanValue(m_countAcked));
}

void
TcpDisorderAckTest::DoTeardown()
{
    Config::SetDefault("ns3::TcpNewReno::CountAckedSegments", BooleanValue(false));
    TcpGeneralTest::DoTeardown();
}

void
TcpDisorderAckTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetSegmentSize(SENDER, GetSegSize(SENDER) + TcpSocketBase::GetSegmentOptionsSize());
}

Ptr<SimpleChannel>
TcpDisorderAckTest::CreateChannel()
{
    Ptr<TcpDisorderAckChannel> ch = CreateObject<TcpDisorderAckChannel>();
    ch->SetAttribute("Delay", TimeValue(MilliSeconds(50)));
    ch->Configure(500, X);
    return ch;
}

Ptr<TcpSocketMsgBase>
TcpDisorderAckTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("InitialCwnd", UintegerValue(10));
    socket->SetAttribute("Sack", BooleanValue(true));
    return socket;
}

void
TcpDisorderAckTest::RcvAck(const Ptr<const TcpSocketState> tcb, const TcpHeader& h, SocketWho who)
{
    m_isTarget = who == SENDER && ToRelativeSeq(SENDER, h.GetAckNumber()).GetValue() == X + 1000 &&
                 h.HasOption(TcpOption::SACK) && !m_seen;
    if (m_isTarget)
    {
        m_cwndBefore = tcb->m_cWnd;
    }
}

void
TcpDisorderAckTest::ProcessedAck(const Ptr<const TcpSocketState> tcb,
                                 const TcpHeader& h,
                                 SocketWho who)
{
    if (m_isTarget && who == SENDER)
    {
        m_isTarget = false;
        if (tcb->m_congState == TcpSocketState::CA_DISORDER)
        {
            m_seen = true;
            m_cwndAfter = tcb->m_cWnd;
        }
    }
}

void
TcpDisorderAckTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_seen, true, "No advancing ACK with new SACK in CA_DISORDER");
    NS_TEST_ASSERT_MSG_EQ(m_cwndAfter,
                          m_cwndBefore + (m_countAcked ? 2 : 1) * 500,
                          "The window did not grow as expected");
}

/**
 * @ingroup internet-test
 *
 * @brief CA_DISORDER ACK TestSuite
 */
class TcpDisorderAckTestSuite : public TestSuite
{
  public:
    TcpDisorderAckTestSuite()
        : TestSuite("tcp-disorder-ack", Type::UNIT)
    {
        AddTestCase(new TcpDisorderAckTest("Advancing ACK with new SACK in CA_DISORDER", false),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpDisorderAckTest("Advancing ACK with new SACK in CA_DISORDER, "
                                           "counting the segments delivered",
                                           true),
                    TestCase::Duration::QUICK);
    }
};

static TcpDisorderAckTestSuite g_tcpDisorderAckTestSuite; //!< static var for init
