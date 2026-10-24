/*
 * Copyright (c) 2013 Natale Patriciello <natale.patriciello@gmail.com>
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#include "tcp-general-test.h"

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/rtt-estimator.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-option-ts.h"

#include <optional>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TimestampTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief TCP TimeStamp enabling Test.
 */
class TimestampTestCase : public TcpGeneralTest
{
  public:
    /**
     * TimeStamp configuration.
     */
    enum Configuration
    {
        DISABLED,
        ENABLED_RECEIVER,
        ENABLED_SENDER,
        ENABLED
    };

    /**
     * @brief Constructor.
     * @param conf Test configuration.
     */
    TimestampTestCase(TimestampTestCase::Configuration conf);

  protected:
    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;

    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;

    Configuration m_configuration; //!< Test configuration.
};

TimestampTestCase::TimestampTestCase(TimestampTestCase::Configuration conf)
    : TcpGeneralTest("Testing the TCP Timestamp option")
{
    m_configuration = conf;
}

Ptr<TcpSocketMsgBase>
TimestampTestCase::CreateReceiverSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateReceiverSocket(node);

    switch (m_configuration)
    {
    case DISABLED:
        socket->SetAttribute("Timestamp", BooleanValue(false));
        break;

    case ENABLED_RECEIVER:
        socket->SetAttribute("Timestamp", BooleanValue(true));
        break;

    case ENABLED_SENDER:
        socket->SetAttribute("Timestamp", BooleanValue(false));
        break;

    case ENABLED:
        socket->SetAttribute("Timestamp", BooleanValue(true));
        break;
    }

    return socket;
}

Ptr<TcpSocketMsgBase>
TimestampTestCase::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);

    switch (m_configuration)
    {
    case DISABLED:
    case ENABLED_RECEIVER:
        socket->SetAttribute("Timestamp", BooleanValue(false));
        break;

    case ENABLED_SENDER:
    case ENABLED:
        socket->SetAttribute("Timestamp", BooleanValue(true));
        break;
    }

    return socket;
}

void
TimestampTestCase::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (m_configuration == DISABLED)
    {
        NS_TEST_ASSERT_MSG_EQ(h.HasOption(TcpOption::TS),
                              false,
                              "timestamp disabled but option enabled");
    }
    else if (m_configuration == ENABLED)
    {
        NS_TEST_ASSERT_MSG_EQ(h.HasOption(TcpOption::TS),
                              true,
                              "timestamp enabled but option disabled");
    }

    NS_LOG_INFO(h);
    if (who == SENDER)
    {
        if (h.GetFlags() & TcpHeader::SYN)
        {
            if (m_configuration == ENABLED_RECEIVER)
            {
                NS_TEST_ASSERT_MSG_EQ(h.HasOption(TcpOption::TS),
                                      false,
                                      "timestamp disabled but option enabled");
            }
            else if (m_configuration == ENABLED_SENDER)
            {
                NS_TEST_ASSERT_MSG_EQ(h.HasOption(TcpOption::TS),
                                      true,
                                      "timestamp enabled but option disabled");
            }
        }
        else
        {
            if (m_configuration != ENABLED)
            {
                NS_TEST_ASSERT_MSG_EQ(h.HasOption(TcpOption::TS),
                                      false,
                                      "timestamp disabled but option enabled");
            }
        }
    }
    else if (who == RECEIVER)
    {
        if (h.GetFlags() & TcpHeader::SYN)
        {
            // Sender has not sent timestamp, so implementation should disable ts
            if (m_configuration == ENABLED_RECEIVER)
            {
                NS_TEST_ASSERT_MSG_EQ(h.HasOption(TcpOption::TS),
                                      false,
                                      "sender has not ts, but receiver sent anyway");
            }
            else if (m_configuration == ENABLED_SENDER)
            {
                NS_TEST_ASSERT_MSG_EQ(h.HasOption(TcpOption::TS),
                                      false,
                                      "receiver has not ts enabled but sent anyway");
            }
        }
        else
        {
            if (m_configuration != ENABLED)
            {
                NS_TEST_ASSERT_MSG_EQ(h.HasOption(TcpOption::TS),
                                      false,
                                      "timestamp disabled but option enabled");
            }
        }
    }
}

void
TimestampTestCase::Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    // if (who == SENDER)
    // {
    // }
    // else if (who == RECEIVER)
    // {
    // }
}

/**
 * @ingroup internet-test
 *
 * @brief TCP TimeStamp values Test.
 */
class TimestampValueTestCase : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param startTime Start time (Seconds).
     * @param timeToWait Time to wait (Seconds).
     * @param name Test description.
     */
    TimestampValueTestCase(double startTime, double timeToWait, std::string name);

  private:
    void DoRun() override;
    void DoTeardown() override;

    /**
     * @brief Perform the test checks.
     */
    void Check();
    /**
     * @brief Test initialization.
     */
    void Init();

    double m_startTime;  //!< Start time (Seconds).
    double m_timeToWait; //!< Time to wait (Seconds).
    double m_initValue;  //!< Initialization value (Seconds).
};

TimestampValueTestCase::TimestampValueTestCase(double startTime,
                                               double timeToWait,
                                               std::string name)
    : TestCase(name)
{
    m_startTime = startTime;
    m_timeToWait = timeToWait;
}

void
TimestampValueTestCase::DoRun()
{
    Simulator::Schedule(Seconds(m_startTime + m_timeToWait), &TimestampValueTestCase::Check, this);
    Simulator::Schedule(Seconds(m_startTime), &TimestampValueTestCase::Init, this);

    Simulator::Run();
}

void
TimestampValueTestCase::DoTeardown()
{
    Simulator::Destroy();
}

void
TimestampValueTestCase::Init()
{
    m_initValue = TcpOptionTS::NowToTsValue();
}

void
TimestampValueTestCase::Check()
{
    uint32_t lastValue = TcpOptionTS::NowToTsValue();

    NS_TEST_ASSERT_MSG_EQ_TOL(MilliSeconds(lastValue - m_initValue),
                              Seconds(m_timeToWait),
                              MilliSeconds(1),
                              "Different TS values");

    NS_TEST_ASSERT_MSG_EQ_TOL(TcpOptionTS::ElapsedTimeFromTsValue(m_initValue),
                              Seconds(m_timeToWait),
                              MilliSeconds(1),
                              "Estimating Wrong RTT");
}

/**
 * @ingroup internet-test
 *
 * @brief Check that a delayed ACK echoes the timestamp of the earliest segment it covers
 *
 * @RFC{7323}, Section 4.3: only a segment starting at or below the last ACK
 * sent updates the timestamp to echo. A segment goes out every 2 ms and every
 * other one is ACKed, so the two segments an ACK covers differ in timestamp.
 * The clock-driven ISN variant puts the sender ISN in the upper half of the
 * sequence space, where a last ACK sent left at 0 would compare wrongly and
 * echo the latest segment.
 */
class TimestampEchoTestCase : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param clockDrivenIsn Whether the initial sequence numbers are clock driven.
     */
    TimestampEchoTestCase(bool clockDrivenIsn);

  protected:
    void ConfigureEnvironment() override;
    void ConfigureProperties() override;
    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void FinalChecks() override;

  private:
    std::optional<uint32_t> m_earliestTs; //!< Timestamp of the earliest segment not acknowledged
    uint32_t m_checkedAcks{0};            //!< ACKs whose echoed timestamp was checked
};

TimestampEchoTestCase::TimestampEchoTestCase(bool clockDrivenIsn)
    : TcpGeneralTest(std::string("Timestamp echoed by delayed ACKs") +
                     (clockDrivenIsn ? ", clock driven ISN" : ""))
{
    if (clockDrivenIsn)
    {
        SetClockDrivenIsn(true, UPPER_HALF_ISN_STREAM);
    }
}

void
TimestampEchoTestCase::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktCount(20);
    SetAppPktInterval(MilliSeconds(2));
}

void
TimestampEchoTestCase::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetInitialCwnd(SENDER, 40);
    SetDelAckMaxCount(RECEIVER, 2);
}

void
TimestampEchoTestCase::Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who == RECEIVER && p->GetSize() > 0 && !m_earliestTs)
    {
        m_earliestTs = DynamicCast<const TcpOptionTS>(h.GetOption(TcpOption::TS))->GetTimestamp();
    }
}

void
TimestampEchoTestCase::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who == RECEIVER && h.GetFlags() == TcpHeader::ACK && m_earliestTs)
    {
        NS_TEST_ASSERT_MSG_EQ(DynamicCast<const TcpOptionTS>(h.GetOption(TcpOption::TS))->GetEcho(),
                              *m_earliestTs,
                              "The ACK does not echo the earliest segment it acknowledges");
        m_earliestTs.reset();
        ++m_checkedAcks;
    }
}

void
TimestampEchoTestCase::FinalChecks()
{
    NS_TEST_ASSERT_MSG_GT(m_checkedAcks, 0, "No ACK was checked");
}

/**
 * @ingroup internet-test
 *
 * @brief Random per-connection timestamp offsets (RFC 7323, Section 5.4)
 *
 * With the RandomTimestampOffset attribute set on both ends, every segment
 * of a connection carries the timestamp clock plus the same offset, which
 * differs between the two ends (the receiver end is a fork of the listening
 * socket), while the echoed timestamps are taken back to the local clock:
 * the RTT samples and the echoed timestamp kept for the congestion control
 * do not see the offset.
 */
class TimestampOffsetTestCase : public TcpGeneralTest
{
  public:
    TimestampOffsetTestCase()
        : TcpGeneralTest("Random timestamp offsets")
    {
    }

  protected:
    void ConfigureEnvironment() override;
    void ConfigureProperties() override;
    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void ProcessedAck(const Ptr<const TcpSocketState> tcb,
                      const TcpHeader& h,
                      SocketWho who) override;
    void FinalChecks() override;

  private:
    std::optional<uint32_t> m_offset[2]; //!< Offset of the timestamps of each end
    uint32_t m_checkedAcks{0};           //!< ACKs whose echoed timestamp was checked
};

void
TimestampOffsetTestCase::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktCount(20);
    SetPropagationDelay(MilliSeconds(50));
}

void
TimestampOffsetTestCase::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    GetSenderSocket()->SetAttribute("RandomTimestampOffset", BooleanValue(true));
    GetReceiverSocket()->SetAttribute("RandomTimestampOffset", BooleanValue(true));
}

void
TimestampOffsetTestCase::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (!h.HasOption(TcpOption::TS))
    {
        return;
    }
    uint32_t offset = DynamicCast<const TcpOptionTS>(h.GetOption(TcpOption::TS))->GetTimestamp() -
                      TcpOptionTS::NowToTsValue();
    if (!m_offset[who])
    {
        m_offset[who] = offset;
    }
    NS_TEST_ASSERT_MSG_EQ(offset, *m_offset[who], "The timestamp offset changed");
}

void
TimestampOffsetTestCase::ProcessedAck(const Ptr<const TcpSocketState> tcb,
                                      const TcpHeader& h,
                                      SocketWho who)
{
    if (who != SENDER)
    {
        return;
    }
    uint32_t now = TcpOptionTS::NowToTsValue();
    NS_TEST_ASSERT_MSG_LT_OR_EQ(tcb->m_rcvTimestampEchoReply,
                                now,
                                "The echoed timestamp is not on the local clock");
    NS_TEST_ASSERT_MSG_GT_OR_EQ(tcb->m_rcvTimestampEchoReply + 1000,
                                now,
                                "The echoed timestamp is not on the local clock");
    ++m_checkedAcks;
}

void
TimestampOffsetTestCase::FinalChecks()
{
    NS_TEST_ASSERT_MSG_GT(m_checkedAcks, 0, "No ACK was checked");
    NS_TEST_ASSERT_MSG_EQ((m_offset[SENDER] && m_offset[RECEIVER]), true, "No timestamp sent");
    NS_TEST_ASSERT_MSG_NE(*m_offset[SENDER], 0, "No offset on the sender timestamps");
    NS_TEST_ASSERT_MSG_NE(*m_offset[SENDER],
                          *m_offset[RECEIVER],
                          "The two ends share the same offset");
    Time rtt = GetRttEstimator(SENDER)->GetEstimate();
    NS_TEST_ASSERT_MSG_GT_OR_EQ(rtt, MilliSeconds(100), "RTT estimate too small");
    NS_TEST_ASSERT_MSG_LT(rtt, MilliSeconds(200), "RTT estimate too large");
}

/**
 * @ingroup internet-test
 *
 * @brief TCP TimeStamp TestSuite.
 */
class TcpTimestampTestSuite : public TestSuite
{
  public:
    TcpTimestampTestSuite()
        : TestSuite("tcp-timestamp", Type::UNIT)
    {
        AddTestCase(new TimestampTestCase(TimestampTestCase::DISABLED), TestCase::Duration::QUICK);
        AddTestCase(new TimestampTestCase(TimestampTestCase::ENABLED_RECEIVER),
                    TestCase::Duration::QUICK);
        AddTestCase(new TimestampTestCase(TimestampTestCase::ENABLED_SENDER),
                    TestCase::Duration::QUICK);
        AddTestCase(new TimestampTestCase(TimestampTestCase::ENABLED), TestCase::Duration::QUICK);
        AddTestCase(new TimestampEchoTestCase(false), TestCase::Duration::QUICK);
        AddTestCase(new TimestampEchoTestCase(true), TestCase::Duration::QUICK);
        AddTestCase(new TimestampOffsetTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new TimestampValueTestCase(0.0, 0.01, "Value Check"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TimestampValueTestCase(3.0, 0.5, "Value Check"), TestCase::Duration::QUICK);
        AddTestCase(new TimestampValueTestCase(5.5, 1.0, "Value Check"), TestCase::Duration::QUICK);
        AddTestCase(new TimestampValueTestCase(6.0, 2.0, "Value Check"), TestCase::Duration::QUICK);
        AddTestCase(new TimestampValueTestCase(2.4, 0.7, "Value Check"), TestCase::Duration::QUICK);
    }
};

static TcpTimestampTestSuite g_tcpTimestampTestSuite; //!< Static variable for test initialization
