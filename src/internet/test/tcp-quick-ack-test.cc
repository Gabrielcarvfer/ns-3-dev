/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "tcp-error-model.h"
#include "tcp-general-test.h"

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/nstime.h"
#include "ns3/simulator.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"

#include <algorithm>
#include <optional>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpQuickAckTest");

namespace
{
constexpr uint32_t SEGMENT_SIZE = 500; //!< Segment size of the tests
} // namespace

/**
 * @ingroup internet-test
 *
 * @brief Base of the quick ACK tests: records the ACKs of the receiver
 */
class TcpQuickAckTestBase : public TcpGeneralTest
{
  protected:
    /**
     * @brief Constructor.
     * @param quickAck Value of the QuickAck attribute of the receiver.
     * @param desc The test description.
     */
    TcpQuickAckTestBase(bool quickAck, const std::string& desc);

    /// An ACK of new data sent by the receiver
    struct AckRecord
    {
        uint32_t segments; //!< Number of segments the ACK acknowledges
        bool immediate;    //!< True if sent upon the arrival of a data segment
    };

    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override;
    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;

    /**
     * @brief Tell if an ACK acknowledges a single segment, upon its arrival.
     * @param ack The ACK.
     * @return true if the ACK is a quick ACK.
     */
    static bool IsQuick(const AckRecord& ack)
    {
        return ack.segments == 1 && ack.immediate;
    }

    bool m_quickAck;                    //!< QuickAck attribute of the receiver
    std::vector<AckRecord> m_acks;      //!< ACKs of new data sent by the receiver
    SequenceNumber32 m_lastAck{0};      //!< Highest ACK number sent by the receiver
    bool m_synAckSent{false};           //!< True once the receiver sent its SYN+ACK
    Time m_lastDataRx{Time::Min()};     //!< Arrival time of the last data segment
    uint32_t m_dataRx{0};               //!< Data segments received by the receiver
    SequenceNumber32 m_highestTxSeq{0}; //!< Highest sequence number sent by the sender
};

TcpQuickAckTestBase::TcpQuickAckTestBase(bool quickAck, const std::string& desc)
    : TcpGeneralTest(desc),
      m_quickAck(quickAck)
{
}

Ptr<TcpSocketMsgBase>
TcpQuickAckTestBase::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    // Without the option, every application write fills a segment
    socket->SetAttribute("Timestamp", BooleanValue(false));
    return socket;
}

Ptr<TcpSocketMsgBase>
TcpQuickAckTestBase::CreateReceiverSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateReceiverSocket(node);
    socket->SetAttribute("Timestamp", BooleanValue(false));
    socket->SetAttribute("QuickAck", BooleanValue(m_quickAck));
    return socket;
}

void
TcpQuickAckTestBase::Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who == RECEIVER && p->GetSize() > 0)
    {
        m_lastDataRx = Simulator::Now();
        ++m_dataRx;
    }
}

void
TcpQuickAckTestBase::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who != RECEIVER || !(h.GetFlags() & TcpHeader::ACK))
    {
        return;
    }
    if (h.GetFlags() & TcpHeader::SYN)
    {
        m_lastAck = h.GetAckNumber();
        m_synAckSent = true;
        return;
    }
    if (m_synAckSent && h.GetAckNumber() > m_lastAck)
    {
        uint32_t acked = h.GetAckNumber() - m_lastAck;
        m_acks.push_back(
            {(acked + SEGMENT_SIZE - 1) / SEGMENT_SIZE, Simulator::Now() == m_lastDataRx});
        NS_LOG_INFO("ACK " << m_acks.size() << " of " << m_acks.back().segments
                           << " segment(s), immediate " << m_acks.back().immediate);
        m_lastAck = h.GetAckNumber();
    }
}

/**
 * @ingroup internet-test
 *
 * @brief Test the quick ACKs at the start of a connection
 *
 * A burst of segments reaches the receiver. With the QuickAck attribute set,
 * the first min(RcvBufSize / (2 * MSS), 16) segments are each acknowledged
 * upon arrival, and the ACKs are then delayed to every second segment. Without
 * it, only the first segment is acknowledged alone.
 */
class TcpQuickAckStartTest : public TcpQuickAckTestBase
{
  public:
    /**
     * @brief Constructor.
     * @param quickAck Value of the QuickAck attribute of the receiver.
     * @param rcvBufSize Receive buffer size of the receiver.
     * @param desc The test description.
     */
    TcpQuickAckStartTest(bool quickAck, uint32_t rcvBufSize, const std::string& desc);

  protected:
    void ConfigureEnvironment() override;
    void ConfigureProperties() override;
    void FinalChecks() override;

  private:
    uint32_t m_rcvBufSize; //!< Receive buffer size of the receiver
};

TcpQuickAckStartTest::TcpQuickAckStartTest(bool quickAck,
                                           uint32_t rcvBufSize,
                                           const std::string& desc)
    : TcpQuickAckTestBase(quickAck, desc),
      m_rcvBufSize(rcvBufSize)
{
}

void
TcpQuickAckStartTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    // A single write, so that only its last segment carries PSH
    SetAppPktSize(60 * SEGMENT_SIZE);
    SetAppPktCount(1);
    SetAppPktInterval(MilliSeconds(1));
}

void
TcpQuickAckStartTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetInitialCwnd(SENDER, 100);
    SetRcvBufSize(RECEIVER, m_rcvBufSize);
}

void
TcpQuickAckStartTest::FinalChecks()
{
    uint32_t quickAcks = m_quickAck ? std::min(m_rcvBufSize / (2 * SEGMENT_SIZE), 16U) : 0;
    // The first segment is acknowledged alone in any case
    uint32_t aloneAcks = std::max(quickAcks, 1U);
    NS_TEST_ASSERT_MSG_GT(m_acks.size(), aloneAcks + 5, "Too few ACKs");

    for (uint32_t i = 0; i < aloneAcks; ++i)
    {
        NS_TEST_ASSERT_MSG_EQ(IsQuick(m_acks[i]),
                              true,
                              "ACK " << i + 1 << " is not sent upon a single segment");
    }
    for (uint32_t i = aloneAcks; i < aloneAcks + 5; ++i)
    {
        NS_TEST_ASSERT_MSG_EQ(m_acks[i].segments,
                              2,
                              "ACK " << i + 1 << " is not delayed to the second segment");
    }
}

/**
 * @ingroup internet-test
 *
 * @brief Error model dropping the pure ACKs while enabled
 */
class TcpQuickAckDropModel : public TcpGeneralErrorModel
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    /**
     * @brief Start or stop dropping the pure ACKs.
     * @param drop True to drop them.
     */
    void SetDropping(bool drop)
    {
        m_drop = drop;
    }

  protected:
    bool ShouldDrop(const Ipv4Header& ipHeader,
                    const TcpHeader& tcpHeader,
                    uint32_t packetSize) override
    {
        return m_drop && packetSize == 0 &&
               !(tcpHeader.GetFlags() & (TcpHeader::SYN | TcpHeader::FIN));
    }

  private:
    void DoReset() override
    {
    }

    bool m_drop{false}; //!< True to drop the pure ACKs
};

NS_OBJECT_ENSURE_REGISTERED(TcpQuickAckDropModel);

TypeId
TcpQuickAckDropModel::GetTypeId()
{
    static TypeId tid = TypeId("ns3::TcpQuickAckDropModel")
                            .SetParent<TcpGeneralErrorModel>()
                            .AddConstructor<TcpQuickAckDropModel>();
    return tid;
}

/**
 * @ingroup internet-test
 *
 * @brief Test the return to the quick ACK mode upon duplicate data
 *
 * The ACKs of the receiver are dropped once the quick ACKs of the start of
 * the connection are used up, until the sender retransmits, on timeout,
 * data the receiver already has. With the QuickAck attribute set, the
 * receiver then acknowledges again every segment that follows, as Linux does
 * after duplicate data. The receiver gets a large minimum RTO, so that the
 * idle period the timeout makes does not by itself start the quick ACK mode
 * again.
 */
class TcpQuickAckDuplicateTest : public TcpQuickAckTestBase
{
  public:
    /**
     * @brief Constructor.
     * @param quickAck Value of the QuickAck attribute of the receiver.
     * @param desc The test description.
     */
    TcpQuickAckDuplicateTest(bool quickAck, const std::string& desc);

  protected:
    void ConfigureEnvironment() override;
    void ConfigureProperties() override;
    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateSenderErrorModel() override;
    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void FinalChecks() override;

  private:
    Ptr<TcpQuickAckDropModel> m_dropModel;    //!< Drops the ACKs reaching the sender
    std::optional<size_t> m_firstAckAfterDup; //!< Index of the first ACK after the duplicate
};

TcpQuickAckDuplicateTest::TcpQuickAckDuplicateTest(bool quickAck, const std::string& desc)
    : TcpQuickAckTestBase(quickAck, desc)
{
}

void
TcpQuickAckDuplicateTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(SEGMENT_SIZE);
    SetAppPktCount(200);
    SetAppPktInterval(MilliSeconds(1));
}

void
TcpQuickAckDuplicateTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetInitialCwnd(SENDER, 100);
}

Ptr<TcpSocketMsgBase>
TcpQuickAckDuplicateTest::CreateReceiverSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpQuickAckTestBase::CreateReceiverSocket(node);
    socket->SetAttribute("MinRto", TimeValue(Seconds(60)));
    return socket;
}

Ptr<ErrorModel>
TcpQuickAckDuplicateTest::CreateSenderErrorModel()
{
    m_dropModel = CreateObject<TcpQuickAckDropModel>();
    return m_dropModel;
}

void
TcpQuickAckDuplicateTest::Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    TcpQuickAckTestBase::Rx(p, h, who);
    if (who == RECEIVER && m_dataRx == 40 && !m_firstAckAfterDup)
    {
        m_dropModel->SetDropping(true);
    }
}

void
TcpQuickAckDuplicateTest::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    TcpQuickAckTestBase::Tx(p, h, who);
    if (who != SENDER || p->GetSize() == 0)
    {
        return;
    }
    if (h.GetSequenceNumber() < m_highestTxSeq && !m_firstAckAfterDup)
    {
        // The retransmission on timeout of data already received
        m_dropModel->SetDropping(false);
        m_firstAckAfterDup = m_acks.size();
    }
    m_highestTxSeq = std::max(m_highestTxSeq, h.GetSequenceNumber());
}

void
TcpQuickAckDuplicateTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_firstAckAfterDup.has_value(), true, "No retransmission happened");
    // The ACK of the duplicate acknowledges all the data received
    size_t first = *m_firstAckAfterDup + 1;
    NS_TEST_ASSERT_MSG_GT(m_acks.size(), first + 10, "Too few ACKs after the duplicate");
    bool allQuick = std::all_of(m_acks.begin() + first,
                                m_acks.begin() + first + 10,
                                [](const AckRecord& ack) { return IsQuick(ack); });
    NS_TEST_ASSERT_MSG_EQ(allQuick,
                          m_quickAck,
                          "The segments after duplicate data are not acknowledged as expected");
}

/**
 * @ingroup internet-test
 *
 * @brief Test the immediate ACK of a pushed segment
 *
 * The application of the sender writes three segments every 3 s, the last
 * of which carries PSH. Once the quick ACKs of the start of the connection
 * are used up, the first two segments of each write are acknowledged
 * together. With the QuickAck attribute set, the third one is acknowledged
 * at once as well, since the application of the receiver has read all the
 * data, as Linux does (tcp_cleanup_rbuf()). Without it, its ACK waits for
 * the delayed ACK timeout.
 */
class TcpQuickAckPushTest : public TcpQuickAckTestBase
{
  public:
    /**
     * @brief Constructor.
     * @param quickAck Value of the QuickAck attribute of the receiver.
     * @param desc The test description.
     */
    TcpQuickAckPushTest(bool quickAck, const std::string& desc)
        : TcpQuickAckTestBase(quickAck, desc)
    {
    }

  protected:
    void ConfigureEnvironment() override
    {
        TcpGeneralTest::ConfigureEnvironment();
        SetAppPktSize(3 * SEGMENT_SIZE);
        SetAppPktCount(25);
        SetAppPktInterval(Seconds(3));
    }

    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override
    {
        Ptr<TcpSocketMsgBase> socket = TcpQuickAckTestBase::CreateReceiverSocket(node);
        // The pause between the writes does not start the quick ACK mode again
        socket->SetAttribute("MinRto", TimeValue(Seconds(60)));
        return socket;
    }

    void FinalChecks() override
    {
        // Past the 16 quick ACKs of the start
        const size_t first = 20;
        NS_TEST_ASSERT_MSG_GT(m_acks.size(), first + 20, "Too few ACKs");
        auto alone = std::count_if(m_acks.begin() + first, m_acks.end(), IsQuick);
        if (m_quickAck)
        {
            NS_TEST_ASSERT_MSG_GT(alone, 10, "The pushed segments are not acknowledged at once");
        }
        else
        {
            NS_TEST_ASSERT_MSG_EQ(alone, 0, "A segment was acknowledged alone");
        }
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Test the SACK compression of the receiver
 *
 * The receiver loses the 10th of a burst of segments, which arrive 100 us
 * apart, so that the 90 that follow are out of order until the
 * retransmission comes, one RTT later. Without the SackCompression attribute,
 * each of them is acknowledged at once (RFC 5681, Section 4.2). With it, the
 * first three are, and the ACKs of the following ones wait up to 1 ms
 * (SackCompressionDelay; 5% of the RTT of 1 s is longer), so that one ACK
 * covers about 10 segments.
 */
class TcpSackCompressionTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param compression Value of the SackCompression attribute of the receiver.
     * @param desc The test description.
     */
    TcpSackCompressionTest(bool compression, const std::string& desc);

  protected:
    void ConfigureEnvironment() override;
    void ConfigureProperties() override;
    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void FinalChecks() override;

  private:
    bool m_compression;                       //!< SackCompression attribute of the receiver
    std::optional<SequenceNumber32> m_rcvNxt; //!< Next in-sequence byte expected
    std::vector<Time> m_oooRx;                //!< Arrival times of the out-of-order segments
    std::vector<Time> m_dupAcks;              //!< Times of the ACKs of out-of-order data
    std::vector<bool> m_dupAckAtRx;           //!< True for the ACKs sent upon an arrival
    Time m_lastRx{Time::Min()};               //!< Arrival time of the last segment
    uint32_t m_maxUncovered{0};               //!< Most out-of-order segments left unacked
    uint32_t m_uncovered{0};                  //!< Out-of-order segments not yet acknowledged
    Time m_maxWait{0};                        //!< Longest wait of out-of-order data for its ACK
    Time m_firstUncovered{0};                 //!< Arrival of the oldest unacknowledged segment
};

TcpSackCompressionTest::TcpSackCompressionTest(bool compression, const std::string& desc)
    : TcpGeneralTest(desc),
      m_compression(compression)
{
}

void
TcpSackCompressionTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(SEGMENT_SIZE);
    SetAppPktCount(100);
    SetAppPktInterval(MicroSeconds(100));
}

void
TcpSackCompressionTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetInitialCwnd(SENDER, 200);
}

Ptr<TcpSocketMsgBase>
TcpSackCompressionTest::CreateReceiverSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateReceiverSocket(node);
    socket->SetAttribute("Timestamp", BooleanValue(false));
    socket->SetAttribute("Sack", BooleanValue(true));
    socket->SetAttribute("SackCompression", BooleanValue(m_compression));
    return socket;
}

Ptr<ErrorModel>
TcpSackCompressionTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    errorModel->AddSeqToKill(SequenceNumber32(1 + 9 * SEGMENT_SIZE));
    return errorModel;
}

void
TcpSackCompressionTest::Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who != RECEIVER || p->GetSize() == 0 || !m_rcvNxt)
    {
        return;
    }
    m_lastRx = Simulator::Now();
    if (h.GetSequenceNumber() == *m_rcvNxt)
    {
        *m_rcvNxt += p->GetSize();
    }
    else if (h.GetSequenceNumber() > *m_rcvNxt)
    {
        m_oooRx.push_back(Simulator::Now());
        if (m_uncovered++ == 0)
        {
            m_firstUncovered = Simulator::Now();
        }
        m_maxUncovered = std::max(m_maxUncovered, m_uncovered);
    }
}

void
TcpSackCompressionTest::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who != RECEIVER || !(h.GetFlags() & TcpHeader::ACK))
    {
        return;
    }
    if (h.GetFlags() & TcpHeader::SYN)
    {
        m_rcvNxt = h.GetAckNumber();
        return;
    }
    if (m_rcvNxt && h.GetAckNumber() == *m_rcvNxt && h.HasOption(TcpOption::SACK) &&
        m_uncovered > 0)
    {
        m_dupAcks.push_back(Simulator::Now());
        m_dupAckAtRx.push_back(Simulator::Now() == m_lastRx);
        m_maxWait = std::max(m_maxWait, Simulator::Now() - m_firstUncovered);
        m_uncovered = 0;
    }
}

void
TcpSackCompressionTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_oooRx.size(), 90, "Unexpected number of out-of-order segments");
    NS_TEST_ASSERT_MSG_EQ(m_uncovered, 0, "Out-of-order data left unacknowledged");
    if (!m_compression)
    {
        NS_TEST_ASSERT_MSG_EQ(m_dupAcks.size(), m_oooRx.size(), "Out-of-order data not ACKed");
        NS_TEST_ASSERT_MSG_EQ(m_maxWait, Time(0), "Out-of-order data not ACKed at once");
        return;
    }
    NS_TEST_ASSERT_MSG_GT(m_dupAcks.size(), 3, "Too few ACKs of out-of-order data");
    for (uint32_t i = 0; i < 3; ++i)
    {
        NS_TEST_ASSERT_MSG_EQ(m_dupAckAtRx[i], true, "ACK " << i + 1 << " is not sent at once");
        NS_TEST_ASSERT_MSG_EQ(m_dupAcks[i], m_oooRx[i], "ACK " << i + 1 << " is not sent at once");
    }
    NS_TEST_ASSERT_MSG_EQ(m_maxWait, MilliSeconds(1), "ACK not held back for 1 ms");
    // The 87 segments that follow the first three arrive within 8.7 ms
    NS_TEST_ASSERT_MSG_LT_OR_EQ(m_dupAcks.size(), 3 + 9, "ACKs not compressed");
    NS_TEST_ASSERT_MSG_GT_OR_EQ(m_maxUncovered, 10, "ACKs not compressed");
}

/**
 * @ingroup internet-test
 *
 * @brief Test the timeout of the delayed ACKs
 *
 * Single segments reach the receiver at a given interval, and all but the
 * first one are acknowledged on the delayed ACK timeout, whose values are
 * checked. Without the AdaptiveDelAck attribute, the timeout is always
 * DelAckTimeout. With it, as Linux, it starts at 40 ms and doubles after each
 * expiry, when the segments arrive further apart than the timeout, up to
 * DelAckTimeout.
 */
class TcpAdaptiveDelAckTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param adaptive Value of the AdaptiveDelAck attribute of the receiver.
     * @param delAckTimeout Value of the DelAckTimeout attribute of the receiver.
     * @param interval Interval between the segments.
     * @param count Number of segments.
     * @param expected Expected timeouts of the delayed ACKs, in order.
     * @param desc The test description.
     */
    TcpAdaptiveDelAckTest(bool adaptive,
                          Time delAckTimeout,
                          Time interval,
                          uint32_t count,
                          const std::vector<Time>& expected,
                          const std::string& desc);

  protected:
    void ConfigureEnvironment() override;
    void ConfigureProperties() override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override;
    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void FinalChecks() override;

  private:
    bool m_adaptive;                //!< AdaptiveDelAck attribute of the receiver
    Time m_delAckTimeout;           //!< DelAckTimeout attribute of the receiver
    Time m_interval;                //!< Interval between the segments
    uint32_t m_count;               //!< Number of segments
    std::vector<Time> m_expected;   //!< Expected timeouts of the delayed ACKs
    std::vector<Time> m_timeouts;   //!< Timeouts of the delayed ACKs
    Time m_lastDataRx{Time::Min()}; //!< Arrival time of the last data segment
    Time m_lastRx{Time::Min()};     //!< Arrival time of the last segment
    SequenceNumber32 m_lastAck{0};  //!< Highest ACK number sent by the receiver
    bool m_synAckSent{false};       //!< True once the receiver sent its SYN+ACK
};

TcpAdaptiveDelAckTest::TcpAdaptiveDelAckTest(bool adaptive,
                                             Time delAckTimeout,
                                             Time interval,
                                             uint32_t count,
                                             const std::vector<Time>& expected,
                                             const std::string& desc)
    : TcpGeneralTest(desc),
      m_adaptive(adaptive),
      m_delAckTimeout(delAckTimeout),
      m_interval(interval),
      m_count(count),
      m_expected(expected)
{
}

void
TcpAdaptiveDelAckTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(SEGMENT_SIZE);
    SetAppPktCount(m_count);
    SetAppPktInterval(m_interval);
}

void
TcpAdaptiveDelAckTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetInitialCwnd(SENDER, 100);
}

Ptr<TcpSocketMsgBase>
TcpAdaptiveDelAckTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    // Without the option, every application write fills a segment
    socket->SetAttribute("Timestamp", BooleanValue(false));
    return socket;
}

Ptr<TcpSocketMsgBase>
TcpAdaptiveDelAckTest::CreateReceiverSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateReceiverSocket(node);
    socket->SetAttribute("Timestamp", BooleanValue(false));
    socket->SetAttribute("AdaptiveDelAck", BooleanValue(m_adaptive));
    socket->SetAttribute("DelAckTimeout", TimeValue(m_delAckTimeout));
    return socket;
}

void
TcpAdaptiveDelAckTest::Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who != RECEIVER)
    {
        return;
    }
    m_lastRx = Simulator::Now();
    if (p->GetSize() > 0)
    {
        m_lastDataRx = Simulator::Now();
    }
}

void
TcpAdaptiveDelAckTest::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who != RECEIVER || !(h.GetFlags() & TcpHeader::ACK))
    {
        return;
    }
    if (h.GetFlags() & TcpHeader::SYN)
    {
        m_lastAck = h.GetAckNumber();
        m_synAckSent = true;
        return;
    }
    if (m_synAckSent && h.GetAckNumber() > m_lastAck)
    {
        m_lastAck = h.GetAckNumber();
        if (Simulator::Now() != m_lastRx)
        {
            m_timeouts.push_back(Simulator::Now() - m_lastDataRx);
            NS_LOG_INFO("Delayed ACK after " << m_timeouts.back().As(Time::MS));
        }
    }
}

void
TcpAdaptiveDelAckTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_timeouts.size(),
                          m_expected.size(),
                          "Unexpected number of delayed ACKs");
    for (size_t i = 0; i < std::min(m_timeouts.size(), m_expected.size()); ++i)
    {
        NS_TEST_ASSERT_MSG_EQ(m_timeouts[i],
                              m_expected[i],
                              "Unexpected timeout of delayed ACK " << i + 1);
    }
}

/**
 * @ingroup internet-test
 *
 * @brief TestSuite for the quick ACK mode of the receiver
 */
class TcpQuickAckTestSuite : public TestSuite
{
  public:
    TcpQuickAckTestSuite()
        : TestSuite("tcp-quick-ack", Type::UNIT)
    {
        AddTestCase(new TcpQuickAckStartTest(false, 131072, "Delayed ACKs from the start"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpQuickAckStartTest(true, 131072, "16 quick ACKs at the start"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpQuickAckStartTest(true, 6000, "Quick ACKs bounded by the window"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpQuickAckDuplicateTest(false, "Delayed ACKs after duplicate data"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpQuickAckDuplicateTest(true, "Quick ACKs after duplicate data"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpQuickAckPushTest(false, "Pushed segments ACKed as the others"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpQuickAckPushTest(true, "Pushed segments ACKed at once"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpSackCompressionTest(false, "Out-of-order data ACKed at once"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpSackCompressionTest(true, "ACKs of out-of-order data compressed"),
                    TestCase::Duration::QUICK);
        const Time ms = MilliSeconds(1);
        AddTestCase(new TcpAdaptiveDelAckTest(false,
                                              200 * ms,
                                              300 * ms,
                                              5,
                                              {200 * ms, 200 * ms, 200 * ms, 200 * ms},
                                              "Fixed delayed ACK timeout"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpAdaptiveDelAckTest(true,
                                              200 * ms,
                                              300 * ms,
                                              5,
                                              {40 * ms, 80 * ms, 160 * ms, 200 * ms},
                                              "Delayed ACK timeout growing on expiry"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpAdaptiveDelAckTest(true,
                                              500 * ms,
                                              600 * ms,
                                              6,
                                              {40 * ms, 80 * ms, 160 * ms, 320 * ms, 500 * ms},
                                              "Delayed ACK timeout bounded by 500 ms"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpAdaptiveDelAckTest(true,
                                              10 * ms,
                                              300 * ms,
                                              3,
                                              {10 * ms, 10 * ms},
                                              "Delayed ACK timeout bounded by DelAckTimeout"),
                    TestCase::Duration::QUICK);
    }
};

static TcpQuickAckTestSuite g_tcpQuickAckTestSuite; //!< Static variable for test initialization
