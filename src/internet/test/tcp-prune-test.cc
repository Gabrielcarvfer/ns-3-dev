/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "tcp-error-model.h"
#include "tcp-general-test.h"

#include "ns3/boolean.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/tcp-option-sack.h"
#include "ns3/tcp-tx-buffer.h"
#include "ns3/test.h"

#include <optional>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpPruneTest");

/**
 * @ingroup internet-test
 *
 * @brief Test the pruning of the out-of-order queue of the receiver
 *
 * The receive buffer holds 20000 bytes, and the sender sends 100 segments of
 * 500 bytes at once. The 6th and the 11th segments are lost, and the first
 * retransmission of the 6th as well, so that the out-of-order data fill the
 * buffer. With the PruneOverhead attribute set to 3, the receiver accounts
 * the data at three times their size against twice the buffer, so that the
 * retransmission of the 11th segment, out of order, finds the memory full:
 * as Linux (tcp_prune_ofo_queue()), the receiver drops the out-of-order
 * segments above it and forgets its SACK blocks, so that it then
 * acknowledges, below the data it had SACKed, without SACK blocks: it
 * reneges. The sender detects it when the cumulative ACK reaches SACKed
 * data, and the transfer completes. Without the attribute the receiver never
 * reneges. By default the sender then discards all the SACK information at
 * once and retransmits from the left edge, without a timeout; with
 * SackRenegingTimeout, as Linux, its retransmission timer expires half an RTT
 * later, on the ticks of the retransmission timer if it has some, and the
 * timeout discards the SACK information. Until then, as Linux
 * (tcp_check_sack_reneging(), on which tcp_fastretrans_alert() returns), the
 * ACK that reveals the reneging retransmits nothing, and so does the
 * application that writes more data then (tcp_push() sends new data only).
 */
class TcpPruneTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param overhead Value of the PruneOverhead attribute of the receiver.
     * @param renegingTimeout Value of the SackRenegingTimeout attribute of
     * the sender.
     * @param desc The test description.
     * @param tick Tick of the retransmission timer (RtoTimerGranularity), 0
     * for none.
     */
    TcpPruneTest(double overhead,
                 bool renegingTimeout,
                 const std::string& desc,
                 Time tick = Time(0))
        : TcpGeneralTest(desc),
          m_overhead(overhead),
          m_renegingTimeout(renegingTimeout),
          m_tick(tick)
    {
    }

  protected:
    void ConfigureEnvironment() override
    {
        TcpGeneralTest::ConfigureEnvironment();
        SetAppPktSize(500);
        SetAppPktCount(100);
        SetAppPktInterval(MicroSeconds(1));
    }

    void ConfigureProperties() override
    {
        TcpGeneralTest::ConfigureProperties();
        SetInitialCwnd(SENDER, 100);
        SetRcvBufSize(RECEIVER, 20000);
    }

    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override
    {
        Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
        socket->SetAttribute("Timestamp", BooleanValue(false));
        socket->SetAttribute("SackRenegingTimeout", BooleanValue(m_renegingTimeout));
        socket->SetAttribute("RtoTimerGranularity", TimeValue(m_tick));
        return socket;
    }

    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override
    {
        Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateReceiverSocket(node);
        socket->SetAttribute("Timestamp", BooleanValue(false));
        socket->SetAttribute("PruneOverhead", DoubleValue(m_overhead));
        return socket;
    }

    Ptr<ErrorModel> CreateReceiverErrorModel() override
    {
        Ptr<TcpSeqErrorModel> model = CreateObject<TcpSeqErrorModel>();
        model->AddSeqToKill(SequenceNumber32(2501));
        model->AddSeqToKill(SequenceNumber32(5001));
        model->AddSeqToKill(SequenceNumber32(2501));
        return model;
    }

    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override
    {
        if (who == SENDER && p->GetSize() > 0 && m_detectTime &&
            Simulator::Now() - *m_detectTime < MicroSeconds(1) &&
            h.GetSequenceNumber() < m_detectHighTx)
        {
            // A retransmission on the ACK that revealed the reneging, or on
            // the write that follows
            ++m_earlyRetransmissions;
        }
        if (who != RECEIVER || !(h.GetFlags() & TcpHeader::ACK) || (h.GetFlags() & TcpHeader::SYN))
        {
            return;
        }
        SequenceNumber32 highSack(0);
        if (h.HasOption(TcpOption::SACK))
        {
            auto sack = DynamicCast<const TcpOptionSack>(h.GetOption(TcpOption::SACK));
            for (const auto& block : sack->GetSackList())
            {
                highSack = std::max(highSack, block.second);
            }
        }
        if (m_highSack > h.GetAckNumber() && highSack < m_highSack)
        {
            // The ACK no longer reports data it SACKed
            m_reneged = true;
        }
        m_highSack = highSack > h.GetAckNumber() ? highSack : SequenceNumber32(0);
        m_lastAck = h.GetAckNumber();
    }

    void FinalChecks() override
    {
        NS_TEST_ASSERT_MSG_EQ(m_reneged, m_overhead > 0, "Unexpected reneging of the receiver");
        uint32_t events = GetTxBuffer(SENDER)->GetRenegingEvents();
        if (m_overhead == 0)
        {
            NS_TEST_ASSERT_MSG_EQ(events, 0, "Reneging detected without reneging");
        }
        else if (m_renegingTimeout)
        {
            // The timeout that follows the first detection, after half the
            // RTT, discards the SACK information
            NS_TEST_ASSERT_MSG_EQ(events, 1, "The SACK information was not discarded at once");
            NS_TEST_ASSERT_MSG_EQ(m_detectTime.has_value() && m_rtoTime.has_value(),
                                  true,
                                  "No timeout after reneging");
            if (m_tick.IsZero())
            {
                NS_TEST_ASSERT_MSG_EQ(*m_rtoTime - *m_detectTime,
                                      m_detectSrtt / 2,
                                      "The timeout after reneging is not half the RTT");
            }
            else
            {
                // On the ticks of the timer, as the retransmission timer
                // (the jiffies of Linux)
                NS_TEST_ASSERT_MSG_GT(*m_rtoTime - *m_detectTime,
                                      m_detectSrtt / 2,
                                      "The timeout after reneging is not on a tick");
                NS_TEST_ASSERT_MSG_LT(*m_rtoTime - *m_detectTime,
                                      m_detectSrtt / 2 + 2 * m_tick,
                                      "The timeout after reneging is too late");
            }
            NS_TEST_ASSERT_MSG_EQ(m_earlyRetransmissions,
                                  0,
                                  "Retransmission on the ACK revealing the reneging");
        }
        else
        {
            // The SACK information is discarded at the first detection, so
            // that the reneged data is repaired without a timeout
            NS_TEST_ASSERT_MSG_EQ(events, 1, "The SACK information was not discarded at once");
            NS_TEST_ASSERT_MSG_EQ(m_rtoTime.has_value(), false, "Timeout after reneging");
        }
        // The SYN and the FIN take one sequence number each
        NS_TEST_ASSERT_MSG_EQ(m_lastAck.GetValue() - m_isn.GetValue(),
                              100 * 500 + (m_detectTime ? 1000 : 0) + 2,
                              "The transfer did not complete");
    }

    void ProcessedAck(const Ptr<const TcpSocketState> tcb,
                      const TcpHeader& h,
                      SocketWho who) override
    {
        if (who == SENDER && !m_detectTime && GetTxBuffer(SENDER)->GetRenegingEvents() > 0)
        {
            m_detectTime = Simulator::Now();
            m_detectSrtt = tcb->m_srtt.Get();
            m_detectHighTx = tcb->m_highTxMark.Get();
            // The application writes more data at once
            Simulator::ScheduleNow(&TcpPruneTest::WriteMore, this);
        }
    }

    void AfterRTOExpired(const Ptr<const TcpSocketState> tcb, SocketWho who) override
    {
        if (who == SENDER && m_detectTime && !m_rtoTime)
        {
            m_rtoTime = Simulator::Now();
        }
    }

    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override
    {
        if (who == RECEIVER && (h.GetFlags() & TcpHeader::SYN))
        {
            m_isn = h.GetSequenceNumber();
        }
    }

    /**
     * @brief Write 1000 more bytes.
     */
    void WriteMore()
    {
        GetSenderSocket()->Send(Create<Packet>(1000), 0);
    }

  private:
    double m_overhead;                  //!< PruneOverhead attribute of the receiver
    bool m_renegingTimeout;             //!< SackRenegingTimeout attribute of the sender
    Time m_tick;                        //!< RtoTimerGranularity attribute of the sender
    std::optional<Time> m_detectTime;   //!< Time the sender detected the reneging
    std::optional<Time> m_rtoTime;      //!< Time of the first timeout after it
    Time m_detectSrtt;                  //!< Smoothed RTT when the reneging was detected
    SequenceNumber32 m_detectHighTx{0}; //!< Highest data sent when it was detected
    uint32_t m_earlyRetransmissions{0}; //!< Retransmissions on the ACK that revealed it
    SequenceNumber32 m_highSack{0};     //!< Highest SACKed sequence of the last ACK
    SequenceNumber32 m_lastAck{0};      //!< Last ACK number of the receiver
    SequenceNumber32 m_isn{0};          //!< Initial sequence number of the sender
    bool m_reneged{false};              //!< True if the receiver reneged on its SACKs
};

/**
 * @ingroup internet-test
 *
 * @brief Test suite of the pruning of the out-of-order queue
 */
class TcpPruneTestSuite : public TestSuite
{
  public:
    TcpPruneTestSuite()
        : TestSuite("tcp-prune", Type::UNIT)
    {
        AddTestCase(new TcpPruneTest(0, false, "No pruning by default"), TestCase::Duration::QUICK);
        AddTestCase(new TcpPruneTest(3, false, "Out-of-order queue pruned, SACKs reneged"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpPruneTest(3, true, "Timeout on SACK reneging as Linux"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpPruneTest(3,
                                     true,
                                     "Timeout on SACK reneging on the timer ticks",
                                     MilliSeconds(10)),
                    TestCase::Duration::QUICK);
    }
};

static TcpPruneTestSuite g_tcpPruneTestSuite; //!< Static variable for test initialization
