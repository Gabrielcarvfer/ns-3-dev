/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "tcp-error-model.h"
#include "tcp-general-test.h"

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/nstime.h"
#include "ns3/simulator.h"
#include "ns3/tcp-option-sack.h"
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
 * @brief Check the SACK compression of the receiver.
 *
 * The 10th of a burst of segments 100 us apart is lost, so the next 90 are
 * out of order for one RTT. Without SackCompression each is acknowledged at
 * once (RFC 5681, Section 4.2). With it, the first three are, then ACKs wait
 * 1.1 ms (SackCompressionDelay, shorter than 5% of the 1 s RTT, plus the
 * 100 us SackCompressionSlack of the Linux range timer), so one ACK covers
 * about 11 segments.
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
    NS_TEST_ASSERT_MSG_EQ(m_maxWait, MicroSeconds(1100), "ACK not held back for 1.1 ms");
    // The 87 segments that follow the first three arrive within 8.7 ms
    NS_TEST_ASSERT_MSG_LT_OR_EQ(m_dupAcks.size(), 3 + 8, "ACKs not compressed");
    NS_TEST_ASSERT_MSG_GT_OR_EQ(m_maxUncovered, 11, "ACKs not compressed");
}

/**
 * @ingroup internet-test
 *
 * @brief Test the ACK held back by SACK compression upon a third SACK block
 *
 * The receiver loses the 10th, 20th and 30th segments of a burst, which
 * arrive 100 us apart, with SACK compression. As tcp_sack_new_ofo_skb() of
 * Linux, the 31st segment, which starts a third SACK block, sends the ACK held
 * back at once, with the two SACK blocks of before its arrival, and is
 * itself left to the next ACK. The 21st segment, which starts the second
 * block only, does not.
 */
class TcpSackCompressionFlushTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param desc The test description.
     */
    TcpSackCompressionFlushTest(const std::string& desc)
        : TcpGeneralTest(desc)
    {
    }

  protected:
    void ConfigureEnvironment() override
    {
        TcpGeneralTest::ConfigureEnvironment();
        SetAppPktSize(SEGMENT_SIZE);
        SetAppPktCount(40);
        SetAppPktInterval(MicroSeconds(100));
    }

    void ConfigureProperties() override
    {
        TcpGeneralTest::ConfigureProperties();
        SetInitialCwnd(SENDER, 100);
    }

    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override
    {
        Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateReceiverSocket(node);
        socket->SetAttribute("Timestamp", BooleanValue(false));
        socket->SetAttribute("Sack", BooleanValue(true));
        socket->SetAttribute("SackCompression", BooleanValue(true));
        return socket;
    }

    Ptr<ErrorModel> CreateReceiverErrorModel() override
    {
        Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
        for (uint32_t lost : {9, 19, 29})
        {
            errorModel->AddSeqToKill(SequenceNumber32(1 + lost * SEGMENT_SIZE));
        }
        return errorModel;
    }

    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override
    {
        if (who == RECEIVER && p->GetSize() > 0)
        {
            m_lastRxSeq = h.GetSequenceNumber();
            m_lastRx = Simulator::Now();
        }
    }

    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override
    {
        if (who != RECEIVER || !h.HasOption(TcpOption::SACK) || Simulator::Now() != m_lastRx)
        {
            return;
        }
        auto sack = DynamicCast<const TcpOptionSack>(h.GetOption(TcpOption::SACK));
        bool covered = m_lastRxSeq < h.GetAckNumber();
        for (const auto& block : sack->GetSackList())
        {
            covered |= block.first <= m_lastRxSeq && m_lastRxSeq < block.second;
        }
        uint32_t segment = static_cast<uint32_t>(m_lastRxSeq - m_isn) / SEGMENT_SIZE + 1;
        NS_LOG_INFO("ACK upon segment " << segment << " with " << sack->GetNumSackBlocks()
                                        << " SACK block(s), covering it " << covered);
        if (!covered)
        {
            m_flushedAt.push_back(segment);
            NS_TEST_EXPECT_MSG_EQ(sack->GetNumSackBlocks(),
                                  2,
                                  "The ACK held back does not carry the blocks of before");
        }
    }

    void FinalChecks() override
    {
        NS_TEST_ASSERT_MSG_EQ(m_flushedAt.size(), 1, "Unexpected ACKs held back sent at once");
        if (!m_flushedAt.empty())
        {
            NS_TEST_ASSERT_MSG_EQ(m_flushedAt[0], 31, "Unexpected segment sending the held ACK");
        }
    }

  private:
    SequenceNumber32 m_isn{1};         //!< Sequence number of the first data byte
    SequenceNumber32 m_lastRxSeq{0};   //!< Sequence number of the last data received
    Time m_lastRx{Time::Min()};        //!< Arrival time of the last data received
    std::vector<uint32_t> m_flushedAt; //!< Segments upon which a held ACK was sent
};

/**
 * @ingroup internet-test
 *
 * @brief Check the ACKs of out-of-order data after a gap is partly filled.
 *
 * With SackCompression and LinuxRcvWindow, the receiver loses the 10th and
 * 20th 1448-byte segments and the first retransmission of the 20th. The
 * retransmission of the 10th fills only the first gap; as in Linux
 * (__tcp_ack_snd_check()) it is acknowledged at once as the first
 * out-of-order ACK at the new RCV.NXT. Expected: the next two out-of-order
 * segments are acknowledged at once and the ACK of the following one is held.
 */
class TcpSackCompressionGapFillTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param desc The test description.
     */
    TcpSackCompressionGapFillTest(const std::string& desc)
        : TcpGeneralTest(desc)
    {
    }

  protected:
    void ConfigureEnvironment() override
    {
        TcpGeneralTest::ConfigureEnvironment();
        SetAppPktSize(SEGMENT);
        SetAppPktCount(2000);
        SetAppPktInterval(MicroSeconds(100));
        SetPropagationDelay(MilliSeconds(50));
    }

    void ConfigureProperties() override
    {
        TcpGeneralTest::ConfigureProperties();
        SetInitialCwnd(SENDER, 40);
        SetSegmentSize(SENDER, SEGMENT);
        SetSegmentSize(RECEIVER, SEGMENT);
        SetRcvBufSize(RECEIVER, 262144);
    }

    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override
    {
        Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateReceiverSocket(node);
        socket->SetAttribute("Timestamp", BooleanValue(false));
        socket->SetAttribute("Sack", BooleanValue(true));
        socket->SetAttribute("SackCompression", BooleanValue(true));
        socket->SetAttribute("LinuxRcvWindow", BooleanValue(true));
        return socket;
    }

    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override
    {
        Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
        socket->SetAttribute("Timestamp", BooleanValue(false));
        socket->SetAttribute("SndBufSize", UintegerValue(4194304));
        return socket;
    }

    Ptr<ErrorModel> CreateReceiverErrorModel() override
    {
        Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
        errorModel->AddSeqToKill(SequenceNumber32(GAP1));
        // Also drop the first retransmission of the 20th so the second gap persists
        errorModel->AddSeqToKill(SequenceNumber32(GAP2));
        errorModel->AddSeqToKill(SequenceNumber32(GAP2));
        return errorModel;
    }

    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override
    {
        if (who == RECEIVER && p->GetSize() > 0)
        {
            m_lastRx = Simulator::Now();
        }
    }

    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override
    {
        if (who != RECEIVER || !(h.GetFlags() & TcpHeader::ACK) || (h.GetFlags() & TcpHeader::SYN))
        {
            return;
        }
        if (h.GetAckNumber() != SequenceNumber32(GAP2))
        {
            return;
        }
        bool atRx = Simulator::Now() == m_lastRx;
        NS_LOG_INFO("ACK of the second gap, sent at an arrival " << atRx);
        if (!m_heldSeen)
        {
            if (atRx)
            {
                ++m_immediate;
            }
            else
            {
                m_heldSeen = true;
            }
        }
    }

    void FinalChecks() override
    {
        NS_TEST_ASSERT_MSG_EQ(m_heldSeen, true, "No ACK held back after the gap filled");
        // The ACK of the retransmission, and two out-of-order segments
        NS_TEST_ASSERT_MSG_EQ(m_immediate, 3, "Unexpected number of immediate ACKs");
    }

  private:
    static constexpr uint32_t SEGMENT = 1448;          //!< Segment size of the test
    static constexpr uint32_t GAP1 = 1 + 9 * SEGMENT;  //!< First segment lost
    static constexpr uint32_t GAP2 = 1 + 19 * SEGMENT; //!< Second segment lost
    Time m_lastRx{Time::Min()};                        //!< Arrival of the last data
    uint32_t m_immediate{0}; //!< ACKs of the second gap sent at once before one is held back
    bool m_heldSeen{false};  //!< An ACK of the second gap was held back
};

/**
 * @ingroup internet-test
 *
 * @brief Check that the delayed ACK timer stays armed with SACK compression.
 *
 * Segments arrive 300 us apart and the 10th is lost. The 8th waits for the
 * delayed ACK timer (DelAckTimeout 20 ms) and is acknowledged by the ACK of
 * the 9th. As Linux leaves the timer running after an ACK
 * (inet_csk_clear_xmit_timer()), it expires 20 ms after the 8th arrived and
 * sends the ACK held by SACK compression (tcp_delack_timer_handler()),
 * before the 50 ms compression delay (5% of the 1 s RTT) elapses.
 */
class TcpSackCompressionDelAckTimerTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param desc The test description.
     */
    TcpSackCompressionDelAckTimerTest(const std::string& desc)
        : TcpGeneralTest(desc)
    {
    }

  protected:
    void ConfigureEnvironment() override
    {
        TcpGeneralTest::ConfigureEnvironment();
        SetAppPktSize(SEGMENT_SIZE);
        SetAppPktCount(120);
        SetAppPktInterval(MicroSeconds(300));
    }

    void ConfigureProperties() override
    {
        TcpGeneralTest::ConfigureProperties();
        SetInitialCwnd(SENDER, 200);
    }

    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override
    {
        Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateReceiverSocket(node);
        socket->SetAttribute("Timestamp", BooleanValue(false));
        socket->SetAttribute("Sack", BooleanValue(true));
        socket->SetAttribute("SackCompression", BooleanValue(true));
        socket->SetAttribute("SackCompressionDelay", TimeValue(MilliSeconds(100)));
        socket->SetAttribute("DelAckTimeout", TimeValue(MilliSeconds(20)));
        return socket;
    }

    Ptr<ErrorModel> CreateReceiverErrorModel() override
    {
        Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
        errorModel->AddSeqToKill(SequenceNumber32(1 + 9 * SEGMENT_SIZE));
        return errorModel;
    }

    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override
    {
        if (who == RECEIVER && p->GetSize() > 0)
        {
            m_lastRx = Simulator::Now();
            if (h.GetSequenceNumber() == SequenceNumber32(1 + 7 * SEGMENT_SIZE))
            {
                m_eighthRx = Simulator::Now();
            }
        }
    }

    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override
    {
        if (who == RECEIVER && h.HasOption(TcpOption::SACK) && Simulator::Now() != m_lastRx &&
            !m_firstHeld)
        {
            m_firstHeld = Simulator::Now();
        }
    }

    void FinalChecks() override
    {
        NS_TEST_ASSERT_MSG_EQ(m_firstHeld.has_value(), true, "No ACK held back");
        NS_TEST_ASSERT_MSG_EQ(m_firstHeld.value_or(Time(0)),
                              m_eighthRx + MilliSeconds(20),
                              "The held ACK is not sent on the delayed ACK timer");
    }

  private:
    Time m_lastRx{Time::Min()};      //!< Arrival of the last data
    Time m_eighthRx{0};              //!< Arrival of the 8th segment
    std::optional<Time> m_firstHeld; //!< Time of the first ACK of out-of-order data held back
};

/**
 * @ingroup internet-test
 *
 * @brief Receiver socket that sets CWR on the data segment of a given
 * sequence number upon its arrival
 */
class TcpCwrMarkingSocket : public TcpSocketMsgBase
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::TcpCwrMarkingSocket")
                                .SetParent<TcpSocketMsgBase>()
                                .SetGroupName("Internet")
                                .AddConstructor<TcpCwrMarkingSocket>();
        return tid;
    }

    static SequenceNumber32 s_cwrSeq; //!< Sequence number of the segment marked with CWR

  protected:
    Ptr<TcpSocketBase> Fork() override
    {
        return CopyObject<TcpCwrMarkingSocket>(this);
    }

    void DoForwardUp(Ptr<Packet> packet,
                     const Address& fromAddress,
                     const Address& toAddress) override
    {
        TcpHeader header;
        packet->PeekHeader(header);
        if (header.GetSequenceNumber() == s_cwrSeq &&
            packet->GetSize() > header.GetSerializedSize())
        {
            packet->RemoveHeader(header);
            header.SetFlags(header.GetFlags() | TcpHeader::CWR);
            packet->AddHeader(header);
        }
        TcpSocketMsgBase::DoForwardUp(packet, fromAddress, toAddress);
    }
};

SequenceNumber32 TcpCwrMarkingSocket::s_cwrSeq{0};

/**
 * @ingroup internet-test
 *
 * @brief Test the immediate ACK of a segment with CWR
 *
 * With the QuickAck and SackCompression attributes, the receiver loses the
 * 10th segment of a burst of segments 130 us apart. The 40th, out of order,
 * whose ACK SACK compression would hold back, carries CWR: as Linux
 * (tcp_ecn_accept_cwr()), it is acknowledged at once.
 */
class TcpQuickAckCwrTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param desc The test description.
     */
    TcpQuickAckCwrTest(const std::string& desc)
        : TcpGeneralTest(desc)
    {
    }

  protected:
    void ConfigureEnvironment() override
    {
        TcpGeneralTest::ConfigureEnvironment();
        SetAppPktSize(SEGMENT_SIZE);
        SetAppPktCount(60);
        SetAppPktInterval(MicroSeconds(130));
    }

    void ConfigureProperties() override
    {
        TcpGeneralTest::ConfigureProperties();
        SetInitialCwnd(SENDER, 200);
    }

    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override
    {
        TcpCwrMarkingSocket::s_cwrSeq = SequenceNumber32(CWR_SEQ);
        Ptr<TcpSocketMsgBase> socket = CreateSocket(node,
                                                    TcpCwrMarkingSocket::GetTypeId(),
                                                    m_congControlTypeId,
                                                    m_recoveryTypeId);
        socket->SetAttribute("Timestamp", BooleanValue(false));
        socket->SetAttribute("Sack", BooleanValue(true));
        socket->SetAttribute("QuickAck", BooleanValue(true));
        socket->SetAttribute("SackCompression", BooleanValue(true));
        return socket;
    }

    Ptr<ErrorModel> CreateReceiverErrorModel() override
    {
        Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
        errorModel->AddSeqToKill(SequenceNumber32(1 + 9 * SEGMENT_SIZE));
        return errorModel;
    }

    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override
    {
        if (who == RECEIVER && h.GetSequenceNumber() == SequenceNumber32(CWR_SEQ) &&
            p->GetSize() > 0)
        {
            m_cwrRx = Simulator::Now();
        }
    }

    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override
    {
        if (who != RECEIVER || !m_cwrRx || m_cwrAcked || !h.HasOption(TcpOption::SACK))
        {
            return;
        }
        auto sack = DynamicCast<const TcpOptionSack>(h.GetOption(TcpOption::SACK));
        for (const auto& block : sack->GetSackList())
        {
            if (block.first <= SequenceNumber32(CWR_SEQ) &&
                SequenceNumber32(CWR_SEQ) < block.second)
            {
                m_cwrAcked = Simulator::Now();
            }
        }
    }

    void FinalChecks() override
    {
        NS_TEST_ASSERT_MSG_EQ(m_cwrAcked.has_value(), true, "The CWR segment was not SACKed");
        NS_TEST_ASSERT_MSG_EQ(m_cwrAcked.value_or(Time(0)),
                              m_cwrRx.value_or(Time(0)),
                              "The segment with CWR is not acknowledged at once");
    }

  private:
    static constexpr uint32_t CWR_SEQ = 1 + 39 * SEGMENT_SIZE; //!< Segment carrying CWR
    std::optional<Time> m_cwrRx;                               //!< Arrival of the CWR segment
    std::optional<Time> m_cwrAcked;                            //!< First ACK covering it
};

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
 * DelAckTimeout. With a RtoTimerGranularity as well, as Linux, the timer of
 * a timeout of n ticks armed during tick k expires on tick k + n + 1.
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
     * @param granularity Value of the RtoTimerGranularity attribute of the receiver.
     */
    TcpAdaptiveDelAckTest(bool adaptive,
                          Time delAckTimeout,
                          Time interval,
                          uint32_t count,
                          const std::vector<Time>& expected,
                          const std::string& desc,
                          Time granularity = Seconds(0));

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
    std::vector<Time> m_arrivals;   //!< Arrival times of the data delayed ACKs acknowledge
    Time m_granularity;             //!< RtoTimerGranularity attribute of the receiver
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
                                             const std::string& desc,
                                             Time granularity)
    : TcpGeneralTest(desc),
      m_adaptive(adaptive),
      m_delAckTimeout(delAckTimeout),
      m_interval(interval),
      m_count(count),
      m_expected(expected),
      m_granularity(granularity)
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
    socket->SetAttribute("RtoTimerGranularity", TimeValue(m_granularity));
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
            m_arrivals.push_back(m_lastDataRx);
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
        Time expected = m_expected[i];
        if (m_granularity.IsStrictlyPositive())
        {
            const int64_t tick = m_granularity.GetTimeStep();
            const int64_t armed = m_arrivals[i].GetTimeStep() / tick;
            expected =
                TimeStep((armed + m_expected[i].GetTimeStep() / tick + 1) * tick) - m_arrivals[i];
        }
        NS_TEST_ASSERT_MSG_EQ(m_timeouts[i],
                              expected,
                              "Unexpected timeout of delayed ACK " << i + 1);
    }
}

/**
 * @ingroup internet-test
 *
 * @brief Listening socket on which ShutdownSend() has no effect, as on Linux
 *
 * The sockets it accepts thus stay open after the FIN of the peer, until the
 * application closes them.
 */
class TcpListenerNoShutdown : public TcpSocketMsgBase
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::TcpListenerNoShutdown")
                                .SetParent<TcpSocketMsgBase>()
                                .SetGroupName("Internet")
                                .AddConstructor<TcpListenerNoShutdown>();
        return tid;
    }

    int ShutdownSend() override
    {
        return 0;
    }
};

/**
 * @ingroup internet-test
 *
 * @brief Check the ACK of a peer FIN that the application does not answer.
 *
 * Without AdaptiveDelAck the FIN is acknowledged at once. With it, as
 * tcp_fin() in Linux, the interactive mode starts and the FIN (all data
 * already acknowledged) is acknowledged by the delayed ACK timer 40 ms later,
 * or, with a 10 ms RtoTimerGranularity, on the fifth tick after its arrival tick.
 */
class TcpFinDelayedAckTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param adaptive Value of the AdaptiveDelAck attribute of the receiver.
     * @param granularity Value of the RtoTimerGranularity attribute of the receiver.
     * @param desc The test description.
     */
    TcpFinDelayedAckTest(bool adaptive, Time granularity, const std::string& desc)
        : TcpGeneralTest(desc),
          m_adaptive(adaptive),
          m_granularity(granularity)
    {
    }

  protected:
    void ConfigureEnvironment() override;
    void ConfigureProperties() override;
    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override;
    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void FinalChecks() override;

  private:
    bool m_adaptive;               //!< AdaptiveDelAck attribute of the receiver
    Time m_granularity;            //!< RtoTimerGranularity attribute of the receiver
    std::optional<Time> m_finRx;   //!< Arrival of the FIN at the receiver
    std::optional<Time> m_finAck;  //!< Departure of the ACK of the FIN
    SequenceNumber32 m_finSeq{0};  //!< Sequence number of the FIN
    bool m_receiverFinSent{false}; //!< Whether the receiver sent a FIN
};

void
TcpFinDelayedAckTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(SEGMENT_SIZE / 2);
    // The last segment is acknowledged at once, as the second of a pair
    SetAppPktCount(3);
    SetAppPktInterval(MilliSeconds(1));
}

void
TcpFinDelayedAckTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    // The data is sent as it is written, and the FIN alone after it
    SetInitialCwnd(SENDER, 10);
}

Ptr<TcpSocketMsgBase>
TcpFinDelayedAckTest::CreateReceiverSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = CreateSocket(node,
                                                TcpListenerNoShutdown::GetTypeId(),
                                                m_congControlTypeId,
                                                m_recoveryTypeId);
    socket->SetAttribute("AdaptiveDelAck", BooleanValue(m_adaptive));
    socket->SetAttribute("RtoTimerGranularity", TimeValue(m_granularity));
    return socket;
}

void
TcpFinDelayedAckTest::Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who == RECEIVER && (h.GetFlags() & TcpHeader::FIN) && !m_finRx)
    {
        NS_TEST_ASSERT_MSG_EQ(p->GetSize(), 0, "The FIN carries data");
        m_finRx = Simulator::Now();
        m_finSeq = h.GetSequenceNumber() + p->GetSize();
    }
}

void
TcpFinDelayedAckTest::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who != RECEIVER)
    {
        return;
    }
    if (h.GetFlags() & TcpHeader::FIN)
    {
        m_receiverFinSent = true;
    }
    if (m_finRx && !m_finAck && (h.GetFlags() & TcpHeader::ACK) && h.GetAckNumber() == m_finSeq + 1)
    {
        m_finAck = Simulator::Now();
    }
}

void
TcpFinDelayedAckTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_EQ(m_finRx.has_value(), true, "The FIN did not arrive");
    NS_TEST_ASSERT_MSG_EQ(m_finAck.has_value(), true, "The FIN was not acknowledged");
    NS_TEST_ASSERT_MSG_EQ(m_receiverFinSent, false, "The receiver closed");
    if (!m_finRx || !m_finAck)
    {
        return;
    }
    Time expected = Seconds(0);
    if (m_adaptive && m_granularity.IsStrictlyPositive())
    {
        const int64_t tick = m_granularity.GetTimeStep();
        expected = TimeStep((m_finRx->GetTimeStep() / tick + 4 + 1) * tick) - *m_finRx;
    }
    else if (m_adaptive)
    {
        expected = MilliSeconds(40);
    }
    NS_TEST_ASSERT_MSG_EQ(*m_finAck - *m_finRx, expected, "Unexpected delay of the ACK of the FIN");
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
        AddTestCase(new TcpSackCompressionFlushTest("ACK held back sent upon a third SACK block"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpSackCompressionGapFillTest("ACKs held back after a gap partly filled"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpSackCompressionDelAckTimerTest("Held ACK sent on the delayed ACK timer"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpQuickAckCwrTest("Segment with CWR acknowledged at once"),
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
        AddTestCase(new TcpAdaptiveDelAckTest(true,
                                              200 * ms,
                                              300 * ms + MicroSeconds(3700),
                                              5,
                                              {40 * ms, 80 * ms, 160 * ms, 200 * ms},
                                              "Delayed ACK timer expiring on ticks",
                                              10 * ms),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpFinDelayedAckTest(false, Seconds(0), "FIN acknowledged at once"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpFinDelayedAckTest(true, Seconds(0), "FIN acknowledged after 40 ms"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpFinDelayedAckTest(true, 10 * ms, "FIN acknowledged on a tick"),
                    TestCase::Duration::QUICK);
    }
};

static TcpQuickAckTestSuite g_tcpQuickAckTestSuite; //!< Static variable for test initialization
