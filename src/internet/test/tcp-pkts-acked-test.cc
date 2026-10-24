/*
 * Copyright (c) 2016 Natale Patriciello <natale.patriciello@gmail.com>
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#include "tcp-error-model.h"
#include "tcp-general-test.h"

#include "ns3/boolean.h"
#include "ns3/data-rate.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/simple-channel.h"
#include "ns3/simulator.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-westwood-plus.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpPktsAckedTestSuite");

class DummyCongControl;

/**
 * @ingroup internet-test
 *
 * @brief Check the number of times that PktsAcked is called
 *
 * Set a custom congestion control class, which calls PktsAckedCalled
 * each time the TCP implementation calls PktsAcked.
 *
 * The checks are performed in FinalChecks: the number of bytes acked divided
 * by segment size should be the same as the number of segments passed through
 * PktsAcked in the congestion control.
 *
 * @see DummyCongControl
 * @see FinalChecks
 */
class TcpPktsAckedOpenTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param desc Test description.
     */
    TcpPktsAckedOpenTest(const std::string& desc);

    /**
     * @brief Called when an ACK is received.
     * @param segmentsAcked The segment ACKed.
     */
    void PktsAckedCalled(uint32_t segmentsAcked);

  protected:
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;

    void ConfigureEnvironment() override;

    void FinalChecks() override;

  private:
    uint32_t m_segmentsAcked;    //!< Contains the number of times PktsAcked is called
    uint32_t m_segmentsReceived; //!< Contains the ack number received

    Ptr<DummyCongControl> m_congCtl; //!< Dummy congestion control.
};

/**
 * @ingroup internet-test
 *
 * @brief Behaves as NewReno, except that each time PktsAcked is called,
 * a notification is sent to TcpPktsAckedOpenTest.
 */
class DummyCongControl : public TcpNewReno
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    DummyCongControl()
    {
    }

    /**
     * @brief Set the callback to be used when an ACK is received.
     * @param test The callback.
     */
    void SetCallback(Callback<void, uint32_t> test)
    {
        m_test = test;
    }

    /**
     * @brief Set the callback to be used with the RTT sample of an ACK.
     * @param test The callback.
     */
    void SetRttCallback(Callback<void, Time> test)
    {
        m_rttTest = test;
    }

    /**
     * @brief Set the callback to be used with the segments and the state.
     * @param test The callback.
     */
    void SetStateCallback(Callback<void, uint32_t, TcpSocketState::TcpCongState_t> test)
    {
        m_stateTest = test;
    }

    void PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt) override
    {
        if (!m_stateTest.IsNull())
        {
            m_stateTest(segmentsAcked, tcb->m_congState);
        }
        if (!m_test.IsNull())
        {
            m_test(segmentsAcked);
        }
        if (!m_rttTest.IsNull())
        {
            m_rttTest(rtt);
        }
    }

  private:
    Callback<void, uint32_t> m_test; //!< Callback to be used when an ACK is received.
    Callback<void, Time> m_rttTest;  //!< Callback receiving the RTT sample of an ACK.
    Callback<void, uint32_t, TcpSocketState::TcpCongState_t>
        m_stateTest; //!< Callback receiving the segments and the state.
};

TypeId
DummyCongControl::GetTypeId()
{
    static TypeId tid = TypeId("ns3::DummyCongControl")
                            .SetParent<TcpNewReno>()
                            .AddConstructor<DummyCongControl>()
                            .SetGroupName("Internet");
    return tid;
}

TcpPktsAckedOpenTest::TcpPktsAckedOpenTest(const std::string& desc)
    : TcpGeneralTest(desc),
      m_segmentsAcked(0),
      m_segmentsReceived(0)
{
}

void
TcpPktsAckedOpenTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktCount(20);
    SetMTU(500);
}

Ptr<TcpSocketMsgBase>
TcpPktsAckedOpenTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> s = TcpGeneralTest::CreateSenderSocket(node);
    m_congCtl = CreateObject<DummyCongControl>();
    m_congCtl->SetCallback(MakeCallback(&TcpPktsAckedOpenTest::PktsAckedCalled, this));
    s->SetCongestionControlAlgorithm(m_congCtl);

    return s;
}

void
TcpPktsAckedOpenTest::PktsAckedCalled(uint32_t segmentsAcked)
{
    m_segmentsAcked += segmentsAcked;
}

void
TcpPktsAckedOpenTest::Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who == SENDER && (!(h.GetFlags() & TcpHeader::SYN)))
    {
        m_segmentsReceived = ToRelativeSeq(SENDER, h.GetAckNumber()).GetValue();
    }
}

void
TcpPktsAckedOpenTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_segmentsReceived / GetSegSize(SENDER),
                          m_segmentsAcked,
                          "Not all acked segments have been passed to PktsAcked method");
}

/**
 * @ingroup internet-test
 *
 * @brief Check that PktsAcked receives a zero RTT on duplicate ACKs
 *
 * A duplicate ACK carries no new RTT sample, so the congestion control gets
 * zero (Linux passes rtt_us = -1 to pkts_acked()) instead of the last sample.
 * ACKs of new data give a positive sample.
 */
class TcpPktsAckedDupAckRttTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param desc Test description.
     */
    TcpPktsAckedDupAckRttTest(const std::string& desc)
        : TcpGeneralTest(desc)
    {
    }

    /**
     * @brief Called when PktsAcked is invoked.
     * @param rtt The RTT sample passed to PktsAcked.
     */
    void PktsAckedCalled(Time rtt);

  protected:
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void ConfigureEnvironment() override;
    void ConfigureProperties() override;
    void FinalChecks() override;

  private:
    Ptr<DummyCongControl> m_congCtl; //!< Dummy congestion control
    SequenceNumber32 m_highAck{0};   //!< Highest ACK number received by the sender
    bool m_dupAck{false};            //!< Whether the ACK being processed is a duplicate
    uint32_t m_dupAckCalls{0};       //!< PktsAcked calls on duplicate ACKs
    uint32_t m_dupAckSamples{0};     //!< Non-zero RTT samples on duplicate ACKs
    uint32_t m_newAckSamples{0};     //!< Positive RTT samples on new ACKs
};

void
TcpPktsAckedDupAckRttTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(40);
    SetPropagationDelay(MilliSeconds(50));
}

void
TcpPktsAckedDupAckRttTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetSegmentSize(SENDER, GetSegSize(SENDER) + TcpSocketBase::GetSegmentOptionsSize());
}

Ptr<TcpSocketMsgBase>
TcpPktsAckedDupAckRttTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> s = TcpGeneralTest::CreateSenderSocket(node);
    s->SetAttribute("InitialCwnd", UintegerValue(10));
    m_congCtl = CreateObject<DummyCongControl>();
    m_congCtl->SetRttCallback(MakeCallback(&TcpPktsAckedDupAckRttTest::PktsAckedCalled, this));
    s->SetCongestionControlAlgorithm(m_congCtl);
    return s;
}

Ptr<ErrorModel>
TcpPktsAckedDupAckRttTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    // The third segment (500 bytes of data each)
    errorModel->AddSeqToKill(SequenceNumber32(1001));
    return errorModel;
}

void
TcpPktsAckedDupAckRttTest::Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who == SENDER && !(h.GetFlags() & TcpHeader::SYN) && (h.GetFlags() & TcpHeader::ACK))
    {
        m_dupAck = (h.GetAckNumber() == m_highAck) && p->GetSize() == 0;
        m_highAck = std::max(m_highAck, h.GetAckNumber());
    }
}

void
TcpPktsAckedDupAckRttTest::PktsAckedCalled(Time rtt)
{
    if (m_dupAck)
    {
        ++m_dupAckCalls;
        if (!rtt.IsZero())
        {
            ++m_dupAckSamples;
        }
    }
    else if (rtt.IsStrictlyPositive())
    {
        ++m_newAckSamples;
    }
}

void
TcpPktsAckedDupAckRttTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_GT(m_dupAckCalls, 0, "No duplicate ACK reached PktsAcked");
    NS_TEST_ASSERT_MSG_EQ(m_dupAckSamples, 0, "A duplicate ACK passed a stale RTT sample");
    NS_TEST_ASSERT_MSG_GT(m_newAckSamples, 0, "The new ACKs passed no RTT sample");
}

/**
 * @ingroup internet-test
 *
 * @brief Check that segments ACKed without an RTT sample count in the TcpWestwoodPlus bandwidth
 * estimate
 *
 * Four segments ACKed by duplicate ACKs (no RTT sample) and one by an ACK
 * with a 100 ms RTT give a first estimate of five 1000-byte segments in
 * 100 ms, as Linux westwood_acked_count() counts duplicate ACKs.
 */
class TcpWestwoodPlusNoRttCountTest : public TestCase
{
  public:
    TcpWestwoodPlusNoRttCountTest()
        : TestCase("Westwood+ counts the segments of ACKs without RTT sample")
    {
    }

  private:
    void DoRun() override
    {
        Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
        state->m_segmentSize = 1000;
        Ptr<TcpWestwoodPlus> cong = CreateObject<TcpWestwoodPlus>();
        DataRate bw;
        cong->TraceConnectWithoutContext(
            "EstimatedBW",
            MakeBoundCallback(+[](DataRate* bw, DataRate, DataRate newVal) { *bw = newVal; }, &bw));
        cong->PktsAcked(state, 4, Time(0));
        cong->PktsAcked(state, 1, MilliSeconds(100));
        Simulator::Run();
        Simulator::Destroy();
        NS_TEST_ASSERT_MSG_EQ(bw.GetBitRate(), 400000, "Unexpected bandwidth estimate");
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Check that LinuxCaRtt makes the minimum RTT the smallest raw sample
 *
 * The path delay drops briefly, giving a few samples far below the smoothed
 * RTT. With LinuxCaRtt (as tcp_update_rtt_min()) the minimum RTT is the
 * smallest sample; without it, it is the smallest smoothed RTT, which the
 * short drop barely moves.
 */
class TcpLinuxMinRttTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param desc Test description.
     * @param linux Value of the LinuxCaRtt attribute.
     */
    TcpLinuxMinRttTest(const std::string& desc, bool linux)
        : TcpGeneralTest(desc),
          m_linux(linux)
    {
    }

  protected:
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<SimpleChannel> CreateChannel() override;
    void ConfigureEnvironment() override;
    void FinalChecks() override;

  private:
    /**
     * @brief Trace sink of the RTT samples.
     * @param oldValue The previous sample.
     * @param newValue The new sample.
     */
    void LastRtt(Time oldValue, Time newValue);

    /**
     * @brief Set the delay of the path.
     * @param delay The delay.
     */
    void SetDelay(Time delay);

    bool m_linux;                  //!< Value of the LinuxCaRtt attribute
    Ptr<SimpleChannel> m_channel;  //!< The channel
    Time m_minSample{Time::Max()}; //!< Smallest RTT sample
};

void
TcpLinuxMinRttTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(200);
    SetAppPktInterval(MilliSeconds(2));
    SetPropagationDelay(MilliSeconds(50));
    Simulator::Schedule(Seconds(10.5), &TcpLinuxMinRttTest::SetDelay, this, MilliSeconds(10));
    Simulator::Schedule(Seconds(10.52), &TcpLinuxMinRttTest::SetDelay, this, MilliSeconds(50));
}

Ptr<SimpleChannel>
TcpLinuxMinRttTest::CreateChannel()
{
    m_channel = TcpGeneralTest::CreateChannel();
    return m_channel;
}

void
TcpLinuxMinRttTest::SetDelay(Time delay)
{
    m_channel->SetAttribute("Delay", TimeValue(delay));
}

Ptr<TcpSocketMsgBase>
TcpLinuxMinRttTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> s = TcpGeneralTest::CreateSenderSocket(node);
    s->SetAttribute("LinuxCaRtt", BooleanValue(m_linux));
    s->TraceConnectWithoutContext("LastRTT", MakeCallback(&TcpLinuxMinRttTest::LastRtt, this));
    return s;
}

void
TcpLinuxMinRttTest::LastRtt(Time oldValue, Time newValue)
{
    if (newValue.IsStrictlyPositive())
    {
        m_minSample = std::min(m_minSample, newValue);
    }
}

void
TcpLinuxMinRttTest::FinalChecks()
{
    Time minRtt = GetTcb(SENDER)->m_minRtt;
    NS_TEST_ASSERT_MSG_LT(m_minSample, MilliSeconds(90), "The delay drop gave no small sample");
    if (m_linux)
    {
        NS_TEST_ASSERT_MSG_EQ(minRtt, m_minSample, "The minimum RTT is not the smallest sample");
    }
    else
    {
        NS_TEST_ASSERT_MSG_GT(minRtt, m_minSample, "The minimum RTT is not a smoothed RTT");
    }
}

/**
 * @ingroup internet-test
 *
 * @brief Check that LinuxCaRtt gives the RTT estimator microsecond samples
 *
 * With timestamps, RFC 7323 (Section 4) samples have the 1 ms granularity of
 * the timestamp clock. With LinuxCaRtt (as Linux seq_rtt_us) the sample is
 * the time since sending the SYN or the oldest newly ACKed, never
 * retransmitted segment, in microseconds. The path RTT, checked on the SYN,
 * is not a whole number of milliseconds.
 */
class TcpLinuxRttEstimatorTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param desc Test description.
     * @param linux Value of the LinuxCaRtt attribute.
     */
    TcpLinuxRttEstimatorTest(const std::string& desc, bool linux)
        : TcpGeneralTest(desc),
          m_linux(linux)
    {
    }

  protected:
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    void ConfigureEnvironment() override;
    void RttTrace(Time oldTime, Time newTime) override;
    void FinalChecks() override;

  private:
    bool m_linux;              //!< Value of the LinuxCaRtt attribute
    uint32_t m_samples{0};     //!< Number of updates of the estimate
    Time m_first{Time::Max()}; //!< First estimate (from the SYN)
};

void
TcpLinuxRttEstimatorTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(20);
    SetPropagationDelay(MicroSeconds(50250));
}

Ptr<TcpSocketMsgBase>
TcpLinuxRttEstimatorTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> s = TcpGeneralTest::CreateSenderSocket(node);
    s->SetAttribute("LinuxCaRtt", BooleanValue(m_linux));
    return s;
}

void
TcpLinuxRttEstimatorTest::RttTrace(Time oldTime, Time newTime)
{
    ++m_samples;
    if (m_first == Time::Max())
    {
        m_first = newTime;
    }
}

void
TcpLinuxRttEstimatorTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_GT(m_samples, 1, "Too few RTT samples");
    if (m_linux)
    {
        NS_TEST_ASSERT_MSG_EQ(m_first, MicroSeconds(100500), "The SYN RTT is not exact");
    }
    else
    {
        NS_TEST_ASSERT_MSG_EQ(m_first.GetNanoSeconds() % 1000000,
                              0,
                              "The SYN RTT does not come from the timestamps");
    }
}

/**
 * @ingroup internet-test
 *
 * @brief Check that LinuxCaRtt samples exclude the time SACKed segments wait for a retransmission
 *
 * As Linux ca_rtt_us, a duplicate ACK that newly SACKs segments gives the
 * time since the newest of them was sent, and the ACK of the retransmitted
 * segment gives only the timestamp sample (Karn). Expected: every sample
 * from the loss on is about one RTT.
 */
class TcpLinuxCaRttTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param desc Test description.
     */
    TcpLinuxCaRttTest(const std::string& desc)
        : TcpGeneralTest(desc)
    {
    }

    /**
     * @brief Called when PktsAcked is invoked.
     * @param rtt The RTT sample passed to PktsAcked.
     */
    void PktsAckedCalled(Time rtt);

  protected:
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void ConfigureEnvironment() override;
    void ConfigureProperties() override;
    void FinalChecks() override;

  private:
    Ptr<DummyCongControl> m_congCtl; //!< Dummy congestion control
    SequenceNumber32 m_highAck{0};   //!< Highest ACK number received by the sender
    bool m_dupAck{false};            //!< Whether the ACK being processed is a duplicate
    uint32_t m_dupAckSamples{0};     //!< Positive RTT samples on duplicate ACKs
    Time m_minSample{Time::Max()};   //!< Smallest RTT sample
    Time m_maxSample{0};             //!< Largest RTT sample
};

void
TcpLinuxCaRttTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(40);
    SetPropagationDelay(MilliSeconds(50));
}

void
TcpLinuxCaRttTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetSegmentSize(SENDER, GetSegSize(SENDER) + TcpSocketBase::GetSegmentOptionsSize());
}

Ptr<TcpSocketMsgBase>
TcpLinuxCaRttTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> s = TcpGeneralTest::CreateSenderSocket(node);
    s->SetAttribute("InitialCwnd", UintegerValue(10));
    s->SetAttribute("Sack", BooleanValue(true));
    s->SetAttribute("LinuxCaRtt", BooleanValue(true));
    m_congCtl = CreateObject<DummyCongControl>();
    m_congCtl->SetRttCallback(MakeCallback(&TcpLinuxCaRttTest::PktsAckedCalled, this));
    s->SetCongestionControlAlgorithm(m_congCtl);
    return s;
}

Ptr<ErrorModel>
TcpLinuxCaRttTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    // The third segment (500 bytes of data each)
    errorModel->AddSeqToKill(SequenceNumber32(1001));
    return errorModel;
}

void
TcpLinuxCaRttTest::Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who == SENDER && !(h.GetFlags() & TcpHeader::SYN) && (h.GetFlags() & TcpHeader::ACK))
    {
        m_dupAck = (h.GetAckNumber() == m_highAck) && p->GetSize() == 0;
        m_highAck = std::max(m_highAck, h.GetAckNumber());
    }
}

void
TcpLinuxCaRttTest::PktsAckedCalled(Time rtt)
{
    if (!rtt.IsStrictlyPositive())
    {
        return;
    }
    if (m_dupAck)
    {
        ++m_dupAckSamples;
    }
    // Start at the loss: earlier samples of a lone segment include the delayed ACK timeout
    if (m_dupAckSamples == 0)
    {
        return;
    }
    m_minSample = std::min(m_minSample, rtt);
    m_maxSample = std::max(m_maxSample, rtt);
}

void
TcpLinuxCaRttTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_GT(m_dupAckSamples, 0, "The duplicate ACKs gave no SACK RTT sample");
    NS_TEST_ASSERT_MSG_GT_OR_EQ(m_minSample, MilliSeconds(100), "RTT sample below the path RTT");
    NS_TEST_ASSERT_MSG_LT(m_maxSample,
                          m_minSample + MilliSeconds(50),
                          "An RTT sample counts the wait for a retransmission");
}

/**
 * @ingroup internet-test
 *
 * @brief Check the PktsAcked calls with the LinuxPktsAcked attribute
 *
 * The third segment is lost. As Linux pkts_acked(), each ACK is passed once
 * with the segments it cumulatively ACKs: zero for duplicate ACKs (the first
 * in CA_OPEN, before the switch to CA_DISORDER), and all segments covered by
 * the retransmission for the ACK ending recovery, in CA_RECOVERY. Without
 * the attribute, each duplicate ACK passes one segment.
 */
class TcpLinuxPktsAckedTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param desc Test description.
     * @param linux Value of the LinuxPktsAcked attribute.
     */
    TcpLinuxPktsAckedTest(const std::string& desc, bool linux)
        : TcpGeneralTest(desc),
          m_linux(linux)
    {
    }

    /**
     * @brief Called when PktsAcked is invoked.
     * @param segmentsAcked The segments passed to PktsAcked.
     * @param state The congestion state at the call.
     */
    void PktsAckedCalled(uint32_t segmentsAcked, TcpSocketState::TcpCongState_t state);

  protected:
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void ConfigureEnvironment() override;
    void ConfigureProperties() override;
    void FinalChecks() override;

  private:
    bool m_linux;                    //!< Value of the LinuxPktsAcked attribute
    Ptr<DummyCongControl> m_congCtl; //!< Dummy congestion control
    SequenceNumber32 m_highAck{0};   //!< Highest ACK number received by the sender
    bool m_dupAck{false};            //!< Whether the ACK being processed is a duplicate
    uint32_t m_acks{0};              //!< ACKs received by the sender, after the handshake
    uint32_t m_calls{0};             //!< PktsAcked calls
    uint32_t m_segments{0};          //!< Segments passed to PktsAcked
    uint32_t m_dupAckSegments{0};    //!< Segments passed on duplicate ACKs
    uint32_t m_openDupAcks{0};       //!< Duplicate ACKs passed in CA_OPEN
    uint32_t m_recoverySegments{0};  //!< Largest number of segments passed in CA_RECOVERY
};

void
TcpLinuxPktsAckedTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(40);
    SetPropagationDelay(MilliSeconds(50));
}

void
TcpLinuxPktsAckedTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetSegmentSize(SENDER, GetSegSize(SENDER) + TcpSocketBase::GetSegmentOptionsSize());
}

Ptr<TcpSocketMsgBase>
TcpLinuxPktsAckedTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> s = TcpGeneralTest::CreateSenderSocket(node);
    s->SetAttribute("InitialCwnd", UintegerValue(10));
    s->SetAttribute("LinuxPktsAcked", BooleanValue(m_linux));
    m_congCtl = CreateObject<DummyCongControl>();
    m_congCtl->SetStateCallback(MakeCallback(&TcpLinuxPktsAckedTest::PktsAckedCalled, this));
    s->SetCongestionControlAlgorithm(m_congCtl);
    return s;
}

Ptr<ErrorModel>
TcpLinuxPktsAckedTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    errorModel->AddSeqToKill(SequenceNumber32(1001));
    return errorModel;
}

void
TcpLinuxPktsAckedTest::Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who == SENDER && !(h.GetFlags() & TcpHeader::SYN) && (h.GetFlags() & TcpHeader::ACK))
    {
        m_dupAck = (h.GetAckNumber() == m_highAck) && p->GetSize() == 0;
        m_highAck = std::max(m_highAck, h.GetAckNumber());
        ++m_acks;
    }
}

void
TcpLinuxPktsAckedTest::PktsAckedCalled(uint32_t segmentsAcked, TcpSocketState::TcpCongState_t state)
{
    ++m_calls;
    m_segments += segmentsAcked;
    if (m_dupAck)
    {
        m_dupAckSegments += segmentsAcked;
        if (state == TcpSocketState::CA_OPEN)
        {
            ++m_openDupAcks;
        }
    }
    if (state == TcpSocketState::CA_RECOVERY)
    {
        m_recoverySegments = std::max(m_recoverySegments, segmentsAcked);
    }
}

void
TcpLinuxPktsAckedTest::FinalChecks()
{
    if (m_linux)
    {
        NS_TEST_ASSERT_MSG_EQ(m_dupAckSegments, 0, "A duplicate ACK passed segments");
        NS_TEST_ASSERT_MSG_EQ(m_openDupAcks, 1, "The first duplicate ACK is not in CA_OPEN");
        NS_TEST_ASSERT_MSG_GT(m_recoverySegments, 1, "The end of the recovery passed no hole");
        // The 40 data segments and the FIN, as a packet of Linux
        NS_TEST_ASSERT_MSG_EQ(m_segments, 41, "The segments passed are not those acknowledged");
        // Every ACK is passed once except the one received with nothing outstanding (the receiver's
        // FIN), as Linux tcp_ack()
        NS_TEST_ASSERT_MSG_EQ(m_calls + 1, m_acks, "An ACK is not passed once");
    }
    else
    {
        NS_TEST_ASSERT_MSG_GT(m_dupAckSegments, 0, "The duplicate ACKs passed no segment");
    }
}

/**
 * @ingroup internet-test
 *
 * @brief PktsAcked is calls TestSuite.
 */
class TcpPktsAckedTestSuite : public TestSuite
{
  public:
    TcpPktsAckedTestSuite()
        : TestSuite("tcp-pkts-acked-test", Type::UNIT)
    {
        AddTestCase(new TcpPktsAckedOpenTest("PktsAcked check while in OPEN state"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpLinuxCaRttTest("RTT samples of LinuxCaRtt"), TestCase::Duration::QUICK);
        AddTestCase(new TcpLinuxMinRttTest("Minimum RTT of LinuxCaRtt", true),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpLinuxMinRttTest("Minimum RTT of the smoothed RTT", false),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpLinuxRttEstimatorTest("RTT estimator samples of LinuxCaRtt", true),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpLinuxRttEstimatorTest("RTT estimator samples of the timestamps", false),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpPktsAckedDupAckRttTest("No RTT sample on duplicate ACKs"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpWestwoodPlusNoRttCountTest(), TestCase::Duration::QUICK);
        AddTestCase(new TcpLinuxPktsAckedTest("PktsAcked once per ACK as Linux", true),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpLinuxPktsAckedTest("PktsAcked from the state machine", false),
                    TestCase::Duration::QUICK);
        // Add DISORDER, RECOVERY and LOSS state check
    }
};

static TcpPktsAckedTestSuite g_TcpPktsAckedTestSuite; //!< Static variable for test initialization
