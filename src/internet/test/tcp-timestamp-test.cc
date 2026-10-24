/*
 * Copyright (c) 2013 Natale Patriciello <natale.patriciello@gmail.com>
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#include "tcp-general-test.h"

#include "ns3/log.h"
#include "ns3/node.h"
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
 * @brief TCP TimeStamp echo Test.
 *
 * With delayed ACKs, an ACK acknowledging several segments echoes the
 * timestamp of the earliest of them (@RFC{7323}, Section 4.3): the timestamp
 * to echo is updated only by a segment which starts at or below the last
 * acknowledgment sent. The sender sends a segment every 2 ms and the
 * receiver acknowledges every other segment, so the two segments an ACK
 * covers carry different timestamps.
 *
 * Run with clock driven initial sequence numbers, the initial sequence
 * number of the sender is placed in the upper half of the sequence space, in
 * which the wrap aware comparisons with a last acknowledgment sent left at 0
 * would make the receiver echo the latest segment instead.
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
        AddTestCase(new TimestampValueTestCase(0.0, 0.01, "Value Check"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TimestampValueTestCase(3.0, 0.5, "Value Check"), TestCase::Duration::QUICK);
        AddTestCase(new TimestampValueTestCase(5.5, 1.0, "Value Check"), TestCase::Duration::QUICK);
        AddTestCase(new TimestampValueTestCase(6.0, 2.0, "Value Check"), TestCase::Duration::QUICK);
        AddTestCase(new TimestampValueTestCase(2.4, 0.7, "Value Check"), TestCase::Duration::QUICK);
    }
};

static TcpTimestampTestSuite g_tcpTimestampTestSuite; //!< Static variable for test initialization
