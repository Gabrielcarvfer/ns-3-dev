/*
 * Copyright (c) 2026 Centre Tecnològic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "tcp-error-model.h"
#include "tcp-general-test.h"

#include "ns3/ipv4-header.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/simple-channel.h"
#include "ns3/simple-net-device.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-l4-protocol.h"
#include "ns3/test.h"

#include <map>
#include <set>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpLostRetransmissionTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief Test the detection of lost retransmissions (see @issueid{979})
 *
 * When a retransmitted segment is lost again, @RFC{6675} based recovery would
 * stall until the retransmission timer expires, because retransmitted
 * segments are not selected again by NextSeg(). Like Linux, ns-3 detects a
 * lost retransmission when a segment transmitted after the retransmission is
 * SACKed, and marks it for retransmission again.
 *
 * The test drops a segment and its retransmission, and checks that the
 * transfer recovers without a retransmission timeout.
 */
class TcpLostRetransmissionTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     */
    TcpLostRetransmissionTest(const std::string& name)
        : TcpGeneralTest(name)
    {
    }

  protected:
    void ConfigureEnvironment() override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                        const TcpSocketState::TcpCongState_t newValue) override;
    void AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who) override;
    void FinalChecks() override;

  private:
    bool m_recoveryEntered{false}; //!< True if fast recovery was entered.
    uint32_t m_rtoCount{0};        //!< Number of RTO expirations at the sender.
};

void
TcpLostRetransmissionTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(60);
    SetPropagationDelay(MilliSeconds(50));
}

Ptr<TcpSocketMsgBase>
TcpLostRetransmissionTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("InitialCwnd", UintegerValue(10));
    // The dropped sequence numbers assume that the segment payload equals the
    // configured segment size: disable the timestamp option, which would
    // decrease the payload by its size
    socket->SetAttribute("Timestamp", BooleanValue(false));
    socket->SetAttribute("Sack", BooleanValue(true));
    return socket;
}

Ptr<ErrorModel>
TcpLostRetransmissionTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    // Drop a segment in the middle of the transfer and then its
    // retransmission (segment size is 500 bytes, first byte has sequence
    // number 1)
    errorModel->AddSeqToKill(SequenceNumber32(10001));
    errorModel->AddSeqToKill(SequenceNumber32(10001));
    return errorModel;
}

void
TcpLostRetransmissionTest::CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                                          const TcpSocketState::TcpCongState_t newValue)
{
    if (newValue == TcpSocketState::CA_RECOVERY)
    {
        m_recoveryEntered = true;
    }
}

void
TcpLostRetransmissionTest::AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who)
{
    if (who == SENDER)
    {
        m_rtoCount++;
    }
}

void
TcpLostRetransmissionTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_recoveryEntered, true, "Fast recovery was not entered");
    NS_TEST_ASSERT_MSG_EQ(m_rtoCount,
                          0,
                          "RTO expired: the lost retransmission was not detected and "
                          "retransmitted again during fast recovery");
}

/**
 * @ingroup internet-test
 *
 * @brief A channel delaying a retransmitted data segment behind the next
 * data segment sent after it.
 */
class TcpRetransmissionReorderingChannel : public SimpleChannel
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::TcpRetransmissionReorderingChannel")
                                .SetParent<SimpleChannel>()
                                .SetGroupName("Internet")
                                .AddConstructor<TcpRetransmissionReorderingChannel>();
        return tid;
    }

    /**
     * @brief Set the sequence number whose first retransmission is delayed.
     *
     * The sequence numbers are relative to the initial sequence number of the
     * data sender, taken from the SYN it sent.
     *
     * @param seq The sequence number.
     */
    void SetSeqToReorder(uint32_t seq)
    {
        m_seqToReorder = seq;
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
        bool isData = false;
        uint32_t seq = 0;
        if (ipHeader.GetProtocol() == TcpL4Protocol::PROT_NUMBER)
        {
            copy->RemoveHeader(tcpHeader);
            if (tcpHeader.GetFlags() == TcpHeader::SYN)
            {
                m_isn = tcpHeader.GetSequenceNumber();
            }
            isData = copy->GetSize() > 0;
            seq = tcpHeader.GetSequenceNumber().GetValue() - m_isn.GetValue();
        }
        if (!isData)
        {
            SimpleChannel::Send(p, protocol, to, from, sender);
            return;
        }

        bool retransmission = !m_seqsSent.insert(seq).second;
        if (retransmission && seq == m_seqToReorder && !m_held)
        {
            NS_LOG_DEBUG("Delaying the retransmission of " << seq);
            m_seqToReorder = 0;
            m_held = p;
            m_heldProtocol = protocol;
            m_heldTo = to;
            m_heldFrom = from;
            m_heldSender = sender;
            return;
        }

        SimpleChannel::Send(p, protocol, to, from, sender);
        if (m_held)
        {
            SimpleChannel::Send(m_held, m_heldProtocol, m_heldTo, m_heldFrom, m_heldSender);
            m_held = nullptr;
            m_heldSender = nullptr;
        }
    }

  private:
    uint32_t m_seqToReorder{0};        //!< Sequence number whose retransmission is delayed
    SequenceNumber32 m_isn{0};         //!< Initial sequence number of the data sender
    std::set<uint32_t> m_seqsSent;     //!< Data sequence numbers sent so far
    Ptr<Packet> m_held;                //!< The delayed retransmission, while held
    uint16_t m_heldProtocol{0};        //!< Protocol of the delayed retransmission
    Mac48Address m_heldTo;             //!< Destination of the delayed retransmission
    Mac48Address m_heldFrom;           //!< Source of the delayed retransmission
    Ptr<SimpleNetDevice> m_heldSender; //!< Sender of the delayed retransmission
};

NS_OBJECT_ENSURE_REGISTERED(TcpRetransmissionReorderingChannel);

/**
 * @ingroup internet-test
 *
 * @brief Document the zero reordering window of the lost retransmission
 * detection.
 *
 * A segment is dropped and its fast retransmission is delayed behind the
 * next data segment sent, so that the receiver SACKs that segment before
 * receiving the retransmission. The lost retransmission detection has no
 * reordering window: a SACK of any segment sent after a retransmission
 * marks the retransmission as lost, so the delayed retransmission is resent
 * once more, spuriously, before the cumulative ACK covering it arrives.
 * @RFC{8985} (RACK) also uses a zero reordering window during loss recovery
 * as long as no reordering has been observed on the connection, but it then
 * widens the window when reordering is detected (e.g., via DSACK), which
 * ns-3 does not do. The test checks that the transfer completes without
 * retransmission timeouts and within one recovery episode, and documents
 * the spurious retransmission: should a reordering tolerance be
 * implemented, the expected number of retransmissions becomes one.
 */
class TcpReorderedRetransmissionTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     */
    TcpReorderedRetransmissionTest(const std::string& name)
        : TcpGeneralTest(name)
    {
    }

  protected:
    void ConfigureEnvironment() override;
    Ptr<SimpleChannel> CreateChannel() override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                        const TcpSocketState::TcpCongState_t newValue) override;
    void AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who) override;
    void FinalChecks() override;

  private:
    static constexpr uint32_t LOST_SEQ = 10001; //!< Sequence number of the dropped segment

    std::map<uint32_t, uint32_t> m_txCount; //!< Transmissions per data sequence number
    uint32_t m_recoveries{0};               //!< Number of fast recovery entries
    uint32_t m_rtoCount{0};                 //!< Number of RTO expirations at the sender
};

void
TcpReorderedRetransmissionTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(60);
    SetPropagationDelay(MilliSeconds(50));
}

Ptr<SimpleChannel>
TcpReorderedRetransmissionTest::CreateChannel()
{
    Ptr<TcpRetransmissionReorderingChannel> ch = CreateObject<TcpRetransmissionReorderingChannel>();
    ch->SetAttribute("Delay", TimeValue(MilliSeconds(50)));
    ch->SetSeqToReorder(LOST_SEQ);
    return ch;
}

Ptr<TcpSocketMsgBase>
TcpReorderedRetransmissionTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("InitialCwnd", UintegerValue(10));
    socket->SetAttribute("Timestamp", BooleanValue(false));
    socket->SetAttribute("Sack", BooleanValue(true));
    return socket;
}

Ptr<ErrorModel>
TcpReorderedRetransmissionTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    errorModel->AddSeqToKill(SequenceNumber32(LOST_SEQ));
    return errorModel;
}

void
TcpReorderedRetransmissionTest::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who == SENDER && p->GetSize() > 0)
    {
        m_txCount[ToRelativeSeq(SENDER, h.GetSequenceNumber()).GetValue()]++;
    }
}

void
TcpReorderedRetransmissionTest::CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                                               const TcpSocketState::TcpCongState_t newValue)
{
    if (newValue == TcpSocketState::CA_RECOVERY)
    {
        m_recoveries++;
    }
}

void
TcpReorderedRetransmissionTest::AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who)
{
    if (who == SENDER)
    {
        m_rtoCount++;
    }
}

void
TcpReorderedRetransmissionTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_recoveries, 1, "Fast recovery not entered exactly once");
    NS_TEST_ASSERT_MSG_EQ(m_rtoCount, 0, "Retransmission timeout");
    uint32_t retransmissions = 0;
    for (const auto& [seq, count] : m_txCount)
    {
        retransmissions += count - 1;
    }
    // The original transmission, the delayed retransmission and the
    // spurious one, due to the zero reordering window
    NS_TEST_ASSERT_MSG_EQ(m_txCount[LOST_SEQ], 3, "Unexpected transmissions of the lost segment");
    NS_TEST_ASSERT_MSG_EQ(retransmissions, 2, "Other segments retransmitted");
}

/**
 * @ingroup internet-test
 *
 * @brief Lost retransmission detection TestSuite
 */
class TcpLostRetransmissionTestSuite : public TestSuite
{
  public:
    TcpLostRetransmissionTestSuite()
        : TestSuite("tcp-lost-retransmission", Type::UNIT)
    {
        AddTestCase(new TcpLostRetransmissionTest("Lost retransmission detected and recovered"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpReorderedRetransmissionTest(
                        "Reordered retransmission resent with no reordering window"),
                    TestCase::Duration::QUICK);
    }
};

static TcpLostRetransmissionTestSuite g_tcpLostRetransmissionTestSuite; //!< static var for init
