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
#include "ns3/tcp-rx-buffer.h"
#include "ns3/tcp-tx-buffer.h"
#include "ns3/test.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("WScalingTestSuite");

// TODO: Check the buffer size and scaling option value
/**
 * @ingroup internet-test
 *
 * @brief TCP Window Scaling enabling Test.
 */
class WScalingTestCase : public TcpGeneralTest
{
  public:
    /**
     * Window Scaling configuration.
     */
    enum Configuration
    {
        DISABLED,
        ENABLED_SENDER,
        ENABLED_RECEIVER,
        ENABLED
    };

    /**
     * @brief Constructor.
     * @param conf Test configuration.
     * @param maxRcvBufferSize Maximum receiver buffer size.
     * @param maxSndBufferSize Maximum sender buffer size.
     * @param name Test description.
     */
    WScalingTestCase(WScalingTestCase::Configuration conf,
                     uint32_t maxRcvBufferSize,
                     uint32_t maxSndBufferSize,
                     std::string name);

  protected:
    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;

    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;

    Configuration m_configuration; //!< Test configuration.
    uint32_t m_maxRcvBufferSize;   //!< Maximum receiver buffer size.
    uint32_t m_maxSndBufferSize;   //!< Maximum sender buffer size.
};

WScalingTestCase::WScalingTestCase(WScalingTestCase::Configuration conf,
                                   uint32_t maxRcvBufferSize,
                                   uint32_t maxSndBufferSize,
                                   std::string name)
    : TcpGeneralTest(name)
{
    m_configuration = conf;
    m_maxRcvBufferSize = maxRcvBufferSize;
    m_maxSndBufferSize = maxSndBufferSize;
}

Ptr<TcpSocketMsgBase>
WScalingTestCase::CreateReceiverSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateReceiverSocket(node);

    switch (m_configuration)
    {
    case DISABLED:
        socket->SetAttribute("WindowScaling", BooleanValue(false));
        break;

    case ENABLED_RECEIVER:
        socket->SetAttribute("WindowScaling", BooleanValue(true));
        break;

    case ENABLED_SENDER:
        socket->SetAttribute("WindowScaling", BooleanValue(false));
        break;

    case ENABLED:
        socket->SetAttribute("WindowScaling", BooleanValue(true));
        break;
    }

    return socket;
}

Ptr<TcpSocketMsgBase>
WScalingTestCase::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);

    switch (m_configuration)
    {
    case DISABLED:
    case ENABLED_RECEIVER:
        socket->SetAttribute("WindowScaling", BooleanValue(false));
        break;

    case ENABLED_SENDER:
    case ENABLED:
        socket->SetAttribute("WindowScaling", BooleanValue(true));
        break;
    }

    return socket;
}

void
WScalingTestCase::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    NS_LOG_INFO(h);

    if (!(h.GetFlags() & TcpHeader::SYN))
    {
        NS_TEST_ASSERT_MSG_EQ(h.HasOption(TcpOption::WINSCALE),
                              false,
                              "wscale present in non-SYN packets");
    }
    else
    {
        if (m_configuration == DISABLED)
        {
            NS_TEST_ASSERT_MSG_EQ(h.HasOption(TcpOption::WINSCALE),
                                  false,
                                  "wscale disabled but option enabled");
        }
        else if (m_configuration == ENABLED)
        {
            NS_TEST_ASSERT_MSG_EQ(h.HasOption(TcpOption::WINSCALE),
                                  true,
                                  "wscale enabled but option disabled");

            if (who == RECEIVER)
            {
                uint16_t advWin = h.GetWindowSize();
                uint32_t maxSize = GetRxBuffer(RECEIVER)->MaxBufferSize();

                if (maxSize > 65535)
                {
                    NS_TEST_ASSERT_MSG_EQ(advWin, 65535, "Scaling SYN segment");
                }
                else
                {
                    NS_TEST_ASSERT_MSG_EQ(advWin, maxSize, "Not advertising all window");
                }
            }
        }

        if (who == SENDER)
        {
            if (m_configuration == ENABLED_RECEIVER)
            {
                NS_TEST_ASSERT_MSG_EQ(h.HasOption(TcpOption::WINSCALE),
                                      false,
                                      "wscale disabled but option enabled");
            }
            else if (m_configuration == ENABLED_SENDER)
            {
                NS_TEST_ASSERT_MSG_EQ(h.HasOption(TcpOption::WINSCALE),
                                      true,
                                      "wscale enabled but option disabled");

                uint16_t advWin = h.GetWindowSize();
                uint32_t maxSize = GetRxBuffer(SENDER)->MaxBufferSize();

                if (maxSize > 65535)
                {
                    NS_TEST_ASSERT_MSG_EQ(advWin, 65535, "Scaling SYN segment");
                }
                else
                {
                    NS_TEST_ASSERT_MSG_EQ(advWin, maxSize, "Not advertising all window");
                }
            }
        }
        else if (who == RECEIVER)
        {
            if (m_configuration == ENABLED_RECEIVER)
            {
                NS_TEST_ASSERT_MSG_EQ(h.HasOption(TcpOption::WINSCALE),
                                      false,
                                      "sender has not ws, but receiver sent anyway");
            }
            else if (m_configuration == ENABLED_SENDER)
            {
                NS_TEST_ASSERT_MSG_EQ(h.HasOption(TcpOption::WINSCALE),
                                      false,
                                      "receiver has not ws enabled but sent anyway");
            }
        }
    }
}

/**
 * @ingroup internet-test
 *
 * @brief TCP Window Scaling TestSuite.
 */
class TcpWScalingTestSuite : public TestSuite
{
  public:
    TcpWScalingTestSuite()
        : TestSuite("tcp-wscaling", Type::UNIT)
    {
        AddTestCase(
            new WScalingTestCase(WScalingTestCase::ENABLED, 200000, 65535, "WS only server"),
            TestCase::Duration::QUICK);
        AddTestCase(new WScalingTestCase(WScalingTestCase::ENABLED,
                                         65535,
                                         65535,
                                         "Window scaling not used, all enabled"),
                    TestCase::Duration::QUICK);
        AddTestCase(new WScalingTestCase(WScalingTestCase::DISABLED, 65535, 65535, "WS disabled"),
                    TestCase::Duration::QUICK);
        AddTestCase(new WScalingTestCase(WScalingTestCase::ENABLED_SENDER,
                                         65535,
                                         65535,
                                         "WS enabled client"),
                    TestCase::Duration::QUICK);
        AddTestCase(new WScalingTestCase(WScalingTestCase::ENABLED_RECEIVER,
                                         65535,
                                         65535,
                                         "WS disabled client"),
                    TestCase::Duration::QUICK);

        AddTestCase(
            new WScalingTestCase(WScalingTestCase::ENABLED, 65535, 200000, "WS only client"),
            TestCase::Duration::QUICK);
        AddTestCase(
            new WScalingTestCase(WScalingTestCase::ENABLED, 131072, 65535, "WS only server"),
            TestCase::Duration::QUICK);
        AddTestCase(
            new WScalingTestCase(WScalingTestCase::ENABLED, 65535, 131072, "WS only client"),
            TestCase::Duration::QUICK);
        AddTestCase(
            new WScalingTestCase(WScalingTestCase::ENABLED, 4000, 4000, "WS small window, all"),
            TestCase::Duration::QUICK);
        AddTestCase(new WScalingTestCase(WScalingTestCase::ENABLED_SENDER,
                                         4000,
                                         4000,
                                         "WS small window, sender"),
                    TestCase::Duration::QUICK);
    }
};

static TcpWScalingTestSuite g_tcpWScalingTestSuite; //!< Static variable for test initialization
