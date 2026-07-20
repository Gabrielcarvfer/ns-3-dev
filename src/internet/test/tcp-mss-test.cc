/*
 * Copyright (c) 2026 Centre Tecnològic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "tcp-general-test.h"

#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-option-rfc793.h"
#include "ns3/test.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpMssTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief TCP MSS option test (see @issueid{946})
 *
 * Checks that the MSS option is sent in every SYN segment (@RFC{9293},
 * Section 3.7.1, SHLD-5 and MAY-3), that it advertises the configured segment
 * size, and that the segment size used by the sender is clamped to the value
 * advertised by the peer (MUST-16).
 */
class TcpMssOptionTestCase : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param senderSegSize Segment size configured on the sender.
     * @param receiverSegSize Segment size configured on the receiver.
     * @param name Test description.
     */
    TcpMssOptionTestCase(uint32_t senderSegSize, uint32_t receiverSegSize, std::string name)
        : TcpGeneralTest(name),
          m_senderSegSize(senderSegSize),
          m_receiverSegSize(receiverSegSize)
    {
    }

  protected:
    void ConfigureProperties() override;
    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;

  private:
    uint32_t m_senderSegSize;   //!< Segment size configured on the sender.
    uint32_t m_receiverSegSize; //!< Segment size configured on the receiver.
};

void
TcpMssOptionTestCase::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetSegmentSize(SENDER, m_senderSegSize);
    SetSegmentSize(RECEIVER, m_receiverSegSize);
}

void
TcpMssOptionTestCase::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    NS_LOG_INFO(h);

    if (h.GetFlags() & TcpHeader::SYN)
    {
        NS_TEST_ASSERT_MSG_EQ(h.HasOption(TcpOption::MSS), true, "MSS option missing in SYN");
        Ptr<const TcpOptionMSS> mss = DynamicCast<const TcpOptionMSS>(h.GetOption(TcpOption::MSS));
        if (who == SENDER)
        {
            NS_TEST_ASSERT_MSG_EQ(mss->GetMSS(),
                                  m_senderSegSize,
                                  "Sender advertised an unexpected MSS");
        }
        else
        {
            // The receiver processes the sender MSS before sending the SYN+ACK,
            // hence it advertises the minimum of the two configured values
            NS_TEST_ASSERT_MSG_EQ(mss->GetMSS(),
                                  std::min(m_senderSegSize, m_receiverSegSize),
                                  "Receiver advertised an unexpected MSS");
        }
    }
    else
    {
        NS_TEST_ASSERT_MSG_EQ(h.HasOption(TcpOption::MSS),
                              false,
                              "MSS option present in non-SYN segment");
        if (who == SENDER && p->GetSize() > 0)
        {
            NS_TEST_ASSERT_MSG_EQ(GetSegSize(SENDER),
                                  std::min(m_senderSegSize, m_receiverSegSize),
                                  "Sender segment size not clamped to the advertised MSS");
        }
    }
}

/**
 * @ingroup internet-test
 *
 * @brief TCP MSS option TestSuite
 */
class TcpMssTestSuite : public TestSuite
{
  public:
    TcpMssTestSuite()
        : TestSuite("tcp-mss", Type::UNIT)
    {
        AddTestCase(new TcpMssOptionTestCase(1400, 800, "MSS option, sender larger than receiver"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpMssOptionTestCase(800, 1400, "MSS option, receiver larger than sender"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpMssOptionTestCase(536, 536, "MSS option, default segment size"),
                    TestCase::Duration::QUICK);
    }
};

static TcpMssTestSuite g_tcpMssTestSuite; //!< static var for test initialization
