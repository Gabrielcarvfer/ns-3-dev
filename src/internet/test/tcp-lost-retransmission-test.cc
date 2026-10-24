/*
 * Copyright (c) 2026 Centre Tecnològic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "tcp-error-model.h"
#include "tcp-general-test.h"

#include "ns3/config.h"
#include "ns3/data-rate.h"
#include "ns3/iana-internet-protocol-numbers.h"
#include "ns3/ipv4-header.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/simple-channel.h"
#include "ns3/simple-net-device.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-option-sack.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-tx-buffer.h"
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
 * transfer recovers without a retransmission timeout. With the detection
 * disabled (TcpTxBuffer::LostRetransmissionDetection), as in strict @RFC{6675}
 * recovery and Linux without RACK, it checks instead that the lost
 * retransmission is not resent before the retransmission timeout repairs it.
 */
class TcpLostRetransmissionTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     * @param detect Whether the lost retransmission detection is enabled.
     */
    TcpLostRetransmissionTest(const std::string& name, bool detect)
        : TcpGeneralTest(name),
          m_detect(detect)
    {
    }

  protected:
    void ConfigureProperties() override;
    void ConfigureEnvironment() override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                        const TcpSocketState::TcpCongState_t newValue) override;
    void BeforeRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who) override;
    void FinalChecks() override;

  private:
    static constexpr uint32_t LOST_SEQ = 10001; //!< Sequence number of the dropped segment

    bool m_detect;                 //!< Whether the lost retransmission detection is enabled.
    bool m_recoveryEntered{false}; //!< True if fast recovery was entered.
    uint32_t m_rtoCount{0};        //!< Number of RTO expirations at the sender.
    uint32_t m_lostTx{0};          //!< Transmissions of the dropped segment.
    uint32_t m_lostTxBeforeRto{0}; //!< Transmissions of the dropped segment before the RTO.
};

void
TcpLostRetransmissionTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(60);
    SetPropagationDelay(MilliSeconds(50));
}

void
TcpLostRetransmissionTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    // The segments carry the timestamp option on top of the data size the
    // test expects
    SetSegmentSize(SENDER, GetSegSize(SENDER) + TcpSocketBase::GetSegmentOptionsSize());
}

Ptr<TcpSocketMsgBase>
TcpLostRetransmissionTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("InitialCwnd", UintegerValue(10));
    socket->SetAttribute("Sack", BooleanValue(true));
    socket->GetTxBuffer()->SetAttribute("LostRetransmissionDetection", BooleanValue(m_detect));
    return socket;
}

void
TcpLostRetransmissionTest::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who == SENDER && p->GetSize() > 0 &&
        ToRelativeSeq(SENDER, h.GetSequenceNumber()).GetValue() == LOST_SEQ)
    {
        m_lostTx++;
    }
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
TcpLostRetransmissionTest::BeforeRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who)
{
    if (who == SENDER)
    {
        if (m_rtoCount == 0)
        {
            m_lostTxBeforeRto = m_lostTx;
        }
        m_rtoCount++;
    }
}

void
TcpLostRetransmissionTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_recoveryEntered, true, "Fast recovery was not entered");
    if (m_detect)
    {
        NS_TEST_ASSERT_MSG_EQ(m_rtoCount,
                              0,
                              "RTO expired: the lost retransmission was not detected and "
                              "retransmitted again during fast recovery");
        NS_TEST_ASSERT_MSG_EQ(m_lostTx, 3, "Unexpected transmissions of the dropped segment");
    }
    else
    {
        NS_TEST_ASSERT_MSG_EQ(m_rtoCount, 1, "The RTO did not repair the lost retransmission");
        NS_TEST_ASSERT_MSG_EQ(m_lostTxBeforeRto,
                              2,
                              "The lost retransmission was resent before the RTO");
        NS_TEST_ASSERT_MSG_EQ(m_lostTx, 3, "Unexpected transmissions of the dropped segment");
    }
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
        if (ipHeader.GetProtocol() == iana::internetprotocolnumbers::TCP)
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
    void ConfigureProperties() override;
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

void
TcpReorderedRetransmissionTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    // The segments carry the timestamp option on top of the data size the
    // test expects
    SetSegmentSize(SENDER, GetSegSize(SENDER) + TcpSocketBase::GetSegmentOptionsSize());
}

Ptr<TcpSocketMsgBase>
TcpReorderedRetransmissionTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("InitialCwnd", UintegerValue(10));
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
 * @brief Test the Linux behaviours of TcpSocketBase which change the
 * retransmission timeout ending a recovery whose retransmission is lost.
 *
 * As in TcpLostRetransmissionTest without the lost retransmission detection,
 * a segment and its fast retransmission are dropped, and the retransmission
 * timeout repairs the loss. The cumulative ACK stays on the dropped segment
 * until then, while the receiver SACKs the segments sent around the
 * retransmission. Each test case enables (or not) one attribute and checks:
 *
 * - KeepRecoverySsThresh: the slow start threshold after the timeout is the
 *   one the recovery set, instead of half of the bytes in flight.
 * - SackRestartsRto: the timeout expires one RTO after the last ACK which
 *   SACKs a segment, never retransmitted, sent before the retransmission of
 *   the first unacknowledged segment, instead of earlier.
 * - SackRttSamples: the smoothed RTT changes on the ACKs SACKing new data,
 *   whereas it does not change during the recovery without the attribute.
 * - RetransmitRestartsRto: the timeout expires one RTO after the fast
 *   retransmission, instead of one RTO after the last cumulative ACK.
 * - RtoTimerGranularity (with SackRestartsRto): the timeout expires on the
 *   tick of the Linux timer wheel, instead of exactly one RTO after the
 *   last restart of the timer.
 */
class TcpLinuxRtoRecoveryTest : public TcpGeneralTest
{
  public:
    /// The attribute under test
    enum Option
    {
        KEEP_SSTHRESH,     //!< KeepRecoverySsThresh
        SACK_RESTARTS_RTO, //!< SackRestartsRto
        SACK_RTT_SAMPLES,  //!< SackRttSamples
        RETX_RESTARTS_RTO, //!< RetransmitRestartsRto
        TIMER_GRANULARITY, //!< RtoTimerGranularity, with SackRestartsRto
    };

    /**
     * @brief Constructor.
     * @param name Test description.
     * @param option The attribute under test.
     * @param enabled Whether the attribute is enabled.
     * @param granularity The RtoTimerGranularity, when enabled.
     */
    TcpLinuxRtoRecoveryTest(const std::string& name,
                            Option option,
                            bool enabled,
                            Time granularity = Time(0))
        : TcpGeneralTest(name),
          m_option(option),
          m_enabled(enabled),
          m_granularity(granularity)
    {
    }

  protected:
    void ConfigureProperties() override;
    void ConfigureEnvironment() override;
    void DoTeardown() override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                        const TcpSocketState::TcpCongState_t newValue) override;
    void RttTrace(Time oldTime, Time newTime) override;
    void BeforeRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who) override;
    void AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who) override;
    void FinalChecks() override;

  private:
    static constexpr uint32_t LOST_SEQ = 10001; //!< Sequence number of the dropped segment

    /**
     * @brief Get the expiry of a Linux timer armed at a given time
     * @param armed The time the timer is armed at.
     * @param rto The timeout.
     * @return The time the timer expires at, on a tick of m_granularity.
     */
    Time GetTickExpiry(Time armed, Time rto) const;

    Option m_option;    //!< The attribute under test
    bool m_enabled;     //!< Whether the attribute is enabled
    Time m_granularity; //!< RtoTimerGranularity, when enabled

    std::map<uint32_t, Time> m_firstTx;     //!< First transmission time per data sequence
    Time m_headLastTx;                      //!< Last transmission of the dropped segment
    Time m_headRetx;                        //!< Fast retransmission of the dropped segment
    uint32_t m_sackRight{0};                //!< Highest SACKed sequence number so far
    Time m_lastRestartingSack{Time::Min()}; //!< Last ACK SACKing data sent before m_headLastTx
    bool m_recoveryEntered{false};          //!< True once fast recovery was entered
    uint32_t m_rtoCount{0};                 //!< Number of RTO expirations at the sender
    uint32_t m_srttChanges{0};              //!< Changes of the smoothed RTT in the recovery
    Time m_rtoTime;                         //!< Time of the first RTO expiration
    Time m_rtoValue;                        //!< RTO at the first expiration
    uint32_t m_ssThreshBefore{0};           //!< ssthresh before the first RTO
    uint32_t m_ssThreshAfter{0};            //!< ssthresh after the first RTO
};

void
TcpLinuxRtoRecoveryTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(60);
    SetPropagationDelay(MilliSeconds(50));
    // The segments of a burst leave one after the other, so that those
    // sent with the fast retransmission are sent after it
    Config::SetDefault("ns3::SimpleNetDevice::DataRate", DataRateValue(DataRate("10Mb/s")));
}

void
TcpLinuxRtoRecoveryTest::DoTeardown()
{
    Config::SetDefault("ns3::SimpleNetDevice::DataRate", DataRateValue(DataRate("0b/s")));
    TcpGeneralTest::DoTeardown();
}

void
TcpLinuxRtoRecoveryTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    // The segments carry the timestamp option on top of the data size the
    // test expects
    SetSegmentSize(SENDER, GetSegSize(SENDER) + TcpSocketBase::GetSegmentOptionsSize());
}

Ptr<TcpSocketMsgBase>
TcpLinuxRtoRecoveryTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("InitialCwnd", UintegerValue(10));
    socket->SetAttribute("Sack", BooleanValue(true));
    socket->GetTxBuffer()->SetAttribute("LostRetransmissionDetection", BooleanValue(false));
    socket->SetAttribute("KeepRecoverySsThresh",
                         BooleanValue(m_option == KEEP_SSTHRESH && m_enabled));
    socket->SetAttribute("SackRestartsRto",
                         BooleanValue((m_option == SACK_RESTARTS_RTO && m_enabled) ||
                                      m_option == TIMER_GRANULARITY));
    socket->SetAttribute("SackRttSamples", BooleanValue(m_option == SACK_RTT_SAMPLES && m_enabled));
    socket->SetAttribute("RetransmitRestartsRto",
                         BooleanValue(m_option == RETX_RESTARTS_RTO && m_enabled));
    socket->SetAttribute(
        "RtoTimerGranularity",
        TimeValue(m_option == TIMER_GRANULARITY && m_enabled ? m_granularity : Time(0)));
    return socket;
}

Ptr<ErrorModel>
TcpLinuxRtoRecoveryTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    errorModel->AddSeqToKill(SequenceNumber32(LOST_SEQ));
    errorModel->AddSeqToKill(SequenceNumber32(LOST_SEQ));
    return errorModel;
}

void
TcpLinuxRtoRecoveryTest::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who != SENDER || p->GetSize() == 0)
    {
        return;
    }
    uint32_t seq = ToRelativeSeq(SENDER, h.GetSequenceNumber()).GetValue();
    if (seq == LOST_SEQ && m_firstTx.contains(seq) && m_headRetx.IsZero())
    {
        m_headRetx = Simulator::Now();
    }
    m_firstTx.emplace(seq, Simulator::Now());
    if (seq == LOST_SEQ)
    {
        m_headLastTx = Simulator::Now();
    }
}

void
TcpLinuxRtoRecoveryTest::Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who != SENDER || !h.HasOption(TcpOption::SACK) ||
        ToRelativeSeq(SENDER, h.GetAckNumber()).GetValue() != LOST_SEQ)
    {
        return;
    }
    auto sack = DynamicCast<const TcpOptionSack>(h.GetOption(TcpOption::SACK));
    uint32_t right = 0;
    uint32_t left = 0;
    for (const auto& [begin, end] : sack->GetSackList())
    {
        if (ToRelativeSeq(SENDER, end).GetValue() > right)
        {
            right = ToRelativeSeq(SENDER, end).GetValue();
            left = ToRelativeSeq(SENDER, begin).GetValue();
        }
    }
    if (right <= m_sackRight)
    {
        return;
    }
    // The segments are SACKed in order: the oldest newly SACKed one starts
    // at the previous right edge
    uint32_t oldest = m_sackRight == 0 ? left : m_sackRight;
    m_sackRight = right;
    if (m_rtoCount == 0 && m_firstTx.contains(oldest) && m_firstTx[oldest] < m_headLastTx)
    {
        m_lastRestartingSack = Simulator::Now();
    }
}

void
TcpLinuxRtoRecoveryTest::CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                                        const TcpSocketState::TcpCongState_t newValue)
{
    if (newValue == TcpSocketState::CA_RECOVERY)
    {
        m_recoveryEntered = true;
    }
}

void
TcpLinuxRtoRecoveryTest::RttTrace(Time oldTime, Time newTime)
{
    if (m_recoveryEntered && m_rtoCount == 0)
    {
        m_srttChanges++;
    }
}

void
TcpLinuxRtoRecoveryTest::BeforeRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who)
{
    if (who == SENDER && m_rtoCount == 0)
    {
        m_rtoTime = Simulator::Now();
        m_rtoValue = GetRto(SENDER);
        m_ssThreshBefore = tcb->m_ssThresh;
    }
}

void
TcpLinuxRtoRecoveryTest::AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who)
{
    if (who == SENDER)
    {
        if (m_rtoCount == 0)
        {
            m_ssThreshAfter = tcb->m_ssThresh;
        }
        m_rtoCount++;
    }
}

Time
TcpLinuxRtoRecoveryTest::GetTickExpiry(Time armed, Time rto) const
{
    int64_t g = m_granularity.GetTimeStep();
    int64_t now = armed.GetTimeStep() / g;
    int64_t ticks = (rto.GetTimeStep() + g - 1) / g;
    // Levels of the Linux timer wheel: 1 tick below 63 ticks, 8 below 504
    int64_t step = ticks < 63 ? 1 : 8;
    NS_ASSERT(ticks < 504);
    return m_granularity * (((now + ticks) / step + 1) * step);
}

void
TcpLinuxRtoRecoveryTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_recoveryEntered, true, "Fast recovery was not entered");
    NS_TEST_ASSERT_MSG_EQ(m_rtoCount, 1, "The RTO did not repair the lost retransmission");

    switch (m_option)
    {
    case KEEP_SSTHRESH:
        if (m_enabled)
        {
            NS_TEST_ASSERT_MSG_EQ(m_ssThreshAfter,
                                  m_ssThreshBefore,
                                  "The RTO changed the ssthresh of the recovery");
        }
        else
        {
            NS_TEST_ASSERT_MSG_LT(m_ssThreshAfter,
                                  m_ssThreshBefore,
                                  "The RTO did not compute the ssthresh again");
        }
        break;
    case SACK_RESTARTS_RTO:
        NS_TEST_ASSERT_MSG_EQ(m_lastRestartingSack.IsStrictlyPositive(),
                              true,
                              "No SACK of the data sent before the retransmission");
        if (m_enabled)
        {
            NS_TEST_ASSERT_MSG_EQ(m_rtoTime,
                                  m_lastRestartingSack + m_rtoValue,
                                  "The RTO did not expire one RTO after the last SACK "
                                  "of the data sent before the retransmission");
        }
        else
        {
            NS_TEST_ASSERT_MSG_LT(m_rtoTime,
                                  m_lastRestartingSack + m_rtoValue,
                                  "The SACKs restarted the RTO");
        }
        break;
    case SACK_RTT_SAMPLES:
        if (m_enabled)
        {
            NS_TEST_ASSERT_MSG_GT(m_srttChanges, 0, "No RTT sample from the SACKs");
        }
        else
        {
            NS_TEST_ASSERT_MSG_EQ(m_srttChanges, 0, "RTT samples from the SACKs");
        }
        break;
    case RETX_RESTARTS_RTO:
        NS_TEST_ASSERT_MSG_EQ(m_headRetx.IsStrictlyPositive(), true, "No fast retransmission");
        if (m_enabled)
        {
            NS_TEST_ASSERT_MSG_EQ(m_rtoTime,
                                  m_headRetx + m_rtoValue,
                                  "The RTO did not expire one RTO after the retransmission");
        }
        else
        {
            NS_TEST_ASSERT_MSG_LT(m_rtoTime,
                                  m_headRetx + m_rtoValue,
                                  "The retransmission restarted the RTO");
        }
        break;
    case TIMER_GRANULARITY:
        if (m_enabled)
        {
            NS_TEST_ASSERT_MSG_EQ(m_rtoTime,
                                  GetTickExpiry(m_lastRestartingSack, m_rtoValue),
                                  "The RTO did not expire on the tick of the timer wheel");
        }
        else
        {
            NS_TEST_ASSERT_MSG_EQ(m_rtoTime,
                                  m_lastRestartingSack + m_rtoValue,
                                  "The RTO did not expire one RTO after its restart");
        }
        break;
    }
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
        AddTestCase(
            new TcpLostRetransmissionTest("Lost retransmission detected and recovered", true),
            TestCase::Duration::QUICK);
        AddTestCase(new TcpLostRetransmissionTest(
                        "Lost retransmission repaired by the RTO without the detection",
                        false),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpReorderedRetransmissionTest(
                        "Reordered retransmission resent with no reordering window"),
                    TestCase::Duration::QUICK);
        for (bool enabled : {false, true})
        {
            std::string onOff = enabled ? " on" : " off";
            AddTestCase(new TcpLinuxRtoRecoveryTest("KeepRecoverySsThresh" + onOff,
                                                    TcpLinuxRtoRecoveryTest::KEEP_SSTHRESH,
                                                    enabled),
                        TestCase::Duration::QUICK);
            AddTestCase(new TcpLinuxRtoRecoveryTest("SackRestartsRto" + onOff,
                                                    TcpLinuxRtoRecoveryTest::SACK_RESTARTS_RTO,
                                                    enabled),
                        TestCase::Duration::QUICK);
            AddTestCase(new TcpLinuxRtoRecoveryTest("SackRttSamples" + onOff,
                                                    TcpLinuxRtoRecoveryTest::SACK_RTT_SAMPLES,
                                                    enabled),
                        TestCase::Duration::QUICK);
            AddTestCase(new TcpLinuxRtoRecoveryTest("RetransmitRestartsRto" + onOff,
                                                    TcpLinuxRtoRecoveryTest::RETX_RESTARTS_RTO,
                                                    enabled),
                        TestCase::Duration::QUICK);
        }
        AddTestCase(new TcpLinuxRtoRecoveryTest("RtoTimerGranularity off",
                                                TcpLinuxRtoRecoveryTest::TIMER_GRANULARITY,
                                                false),
                    TestCase::Duration::QUICK);
        // A timeout of fewer than 63 ticks, and one of more
        for (Time granularity : {MilliSeconds(100), MilliSeconds(10)})
        {
            AddTestCase(
                new TcpLinuxRtoRecoveryTest(
                    "RtoTimerGranularity " + std::to_string(granularity.GetMilliSeconds()) + " ms",
                    TcpLinuxRtoRecoveryTest::TIMER_GRANULARITY,
                    true,
                    granularity),
                TestCase::Duration::QUICK);
        }
    }
};

static TcpLostRetransmissionTestSuite g_tcpLostRetransmissionTestSuite; //!< static var for init
