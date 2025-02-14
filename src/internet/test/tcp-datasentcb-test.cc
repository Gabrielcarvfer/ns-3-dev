/*
 * Copyright (c) 2015 Natale Patriciello <natale.patriciello@gmail.com>
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#include "tcp-general-test.h"

#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/tcp-header.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpDatSentCbTest");

/**
 * @ingroup internet-test
 *
 * @brief Socket that the 50% of the times saves the entire packet in the buffer,
 * while in the other 50% saves only half the packet.
 */
class TcpSocketHalfAck : public TcpSocketMsgBase
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    TcpSocketHalfAck()
        : TcpSocketMsgBase()
    {
    }

  protected:
    Ptr<TcpSocketBase> Fork() override;
    void ReceivedData(Ptr<Packet> packet, const TcpHeader& tcpHeader) override;
};

NS_OBJECT_ENSURE_REGISTERED(TcpSocketHalfAck);

TypeId
TcpSocketHalfAck::GetTypeId()
{
    static TypeId tid = TypeId("ns3::TcpSocketHalfAck")
                            .SetParent<TcpSocketMsgBase>()
                            .SetGroupName("Internet")
                            .AddConstructor<TcpSocketHalfAck>();
    return tid;
}

Ptr<TcpSocketBase>
TcpSocketHalfAck::Fork()
{
    return CopyObject<TcpSocketHalfAck>(this);
}

void
TcpSocketHalfAck::ReceivedData(Ptr<Packet> packet, const TcpHeader& tcpHeader)
{
    NS_LOG_FUNCTION(this << packet << tcpHeader);
    static uint32_t times = 1;

    Ptr<Packet> halved = packet->Copy();

    if (times % 2 == 0)
    {
        halved->RemoveAtEnd(packet->GetSize() / 2);
    }

    times++;

    TcpSocketMsgBase::ReceivedData(halved, tcpHeader);
}

/**
 * @ingroup internet-test
 *
 * @brief Data Sent callback test
 *
 * The rationale of this test is to check if the dataSent callback advertises
 * to the application all the transmitted bytes. We know in advance how many
 * bytes are being transmitted, and we check if the amount of data notified
 * equals this value.
 *
 */
class TcpDataSentCbTestCase : public TcpGeneralTest
{
  public:
    /**
     * Constructor.
     * @param desc Test description.
     * @param size Packet size.
     * @param packets Number of packets.
     */
    TcpDataSentCbTestCase(const std::string& desc, uint32_t size, uint32_t packets)
        : TcpGeneralTest(desc),
          m_pktSize(size),
          m_pktCount(packets),
          m_notifiedData(0)
    {
    }

  protected:
    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override;

    void DataSent(uint32_t size, SocketWho who) override;
    void ConfigureEnvironment() override;
    void FinalChecks() override;

  private:
    uint32_t m_pktSize;      //!< Packet size.
    uint32_t m_pktCount;     //!< Number of packets sent.
    uint32_t m_notifiedData; //!< Amount of data notified.
};

void
TcpDataSentCbTestCase::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktCount(m_pktCount);
    SetAppPktSize(m_pktSize);
}

void
TcpDataSentCbTestCase::DataSent(uint32_t size, SocketWho who)
{
    NS_LOG_FUNCTION(this << who << size);

    m_notifiedData += size;
}

void
TcpDataSentCbTestCase::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_notifiedData,
                          GetPktSize() * GetPktCount(),
                          "Notified more data than application sent");
}

Ptr<TcpSocketMsgBase>
TcpDataSentCbTestCase::CreateReceiverSocket(Ptr<Node> node)
{
    NS_LOG_FUNCTION(this);

    return CreateSocket(node, TcpSocketHalfAck::GetTypeId(), m_congControlTypeId);
}

/**
 * @ingroup internet-test
 *
 * @brief TestSuite: Data Sent callback
 */
class TcpDataSentCbTestSuite : public TestSuite
{
  public:
    TcpDataSentCbTestSuite()
        : TestSuite("tcp-datasentcb", Type::UNIT)
    {
        AddTestCase(new TcpDataSentCbTestCase("Check the data sent callback", 500, 10),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpDataSentCbTestCase("Check the data sent callback", 100, 100),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpDataSentCbTestCase("Check the data sent callback", 1000, 50),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpDataSentCbTestCase("Check the data sent callback", 855, 18),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpDataSentCbTestCase("Check the data sent callback", 1243, 59),
                    TestCase::Duration::QUICK);
    }
};

static TcpDataSentCbTestSuite g_tcpDataSentCbTestSuite; //!< Static variable for test initialization
