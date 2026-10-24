/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ns3/boolean.h"
#include "ns3/data-rate.h"
#include "ns3/enum.h"
#include "ns3/simulator.h"
#include "ns3/tcp-socket-state.h"
#include "ns3/tcp-westwood-plus.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"

using namespace ns3;

/**
 * @ingroup internet-test
 *
 * @brief Test the bandwidth filters of TcpWestwoodPlus
 *
 * The filters start from the first bandwidth sample, not from zero.
 */
class TcpWestwoodPlusFilterTest : public TestCase
{
  public:
    /**
     * Constructor
     * @param filter the filter type
     * @param name the test name
     */
    TcpWestwoodPlusFilterTest(TcpWestwoodPlus::FilterType filter, const std::string& name)
        : TestCase(name),
          m_filter(filter)
    {
    }

  private:
    void DoRun() override;

    /**
     * Record the estimated bandwidth
     * @param oldValue the old value
     * @param newValue the new value
     */
    void Estimate(DataRate oldValue, DataRate newValue)
    {
        m_estimate = newValue;
    }

    /**
     * Record the current estimate
     */
    void Record()
    {
        m_estimates.push_back(m_estimate);
    }

    /**
     * Acknowledge segments, with an RTT of 100 ms
     * @param cong the congestion control
     * @param segments the number of segments acknowledged
     */
    void Ack(Ptr<TcpCongestionOps> cong, uint32_t segments)
    {
        cong->PktsAcked(m_tcb, segments, MilliSeconds(100));
    }

    TcpWestwoodPlus::FilterType m_filter; //!< filter type
    Ptr<TcpSocketState> m_tcb;            //!< socket state
    DataRate m_estimate;                  //!< current estimated bandwidth
    std::vector<DataRate> m_estimates;    //!< recorded estimated bandwidths
};

void
TcpWestwoodPlusFilterTest::DoRun()
{
    m_tcb = CreateObject<TcpSocketState>();
    m_tcb->m_segmentSize = 1000;

    Ptr<TcpWestwoodPlus> proto = CreateObject<TcpWestwoodPlus>();
    proto->SetAttribute("FilterType", EnumValue(m_filter));
    // The socket uses a copy of the congestion control
    Ptr<TcpCongestionOps> cong = proto->Fork();
    cong->TraceConnectWithoutContext("EstimatedBW",
                                     MakeCallback(&TcpWestwoodPlusFilterTest::Estimate, this));

    // 10 segments of 1000 bytes in 100 ms (800 kbit/s), then 20 segments
    Simulator::Schedule(Seconds(1), &TcpWestwoodPlusFilterTest::Ack, this, cong, 10);
    Simulator::Schedule(Seconds(1.5), &TcpWestwoodPlusFilterTest::Record, this);
    Simulator::Schedule(Seconds(2), &TcpWestwoodPlusFilterTest::Ack, this, cong, 20);
    Simulator::Schedule(Seconds(2.5), &TcpWestwoodPlusFilterTest::Record, this);
    Simulator::Run();
    Simulator::Destroy();

    NS_TEST_ASSERT_MSG_EQ(m_estimates[0].GetBitRate(),
                          800000,
                          "The filter does not start from the first sample");
    // 0.9 * 800 + 0.1 * (1600 + 800) / 2
    NS_TEST_ASSERT_MSG_EQ_TOL(m_estimates[1].GetBitRate(), 840000, 1, "Wrong Tustin filter output");
}

/**
 * @ingroup internet-test
 *
 * @brief Test the Linux estimator of TcpWestwoodPlus (FilterType Linux)
 *
 * ACKs arrive every 10 ms, with an RTT of 100 ms and jiffies of 1 ms: a
 * sampling window closes with the first ACK more than 100 jiffies after its
 * start, every 110 ms. The ACKs of the first window acknowledge 1000 bytes
 * each, a sample of 100 bytes per jiffy; those of the second window 2000
 * bytes each, but four of them are duplicate ACKs, each counting a segment,
 * which the next cumulative ACK does not count again: a sample of 200 bytes
 * per jiffy, which the two moving averages of gain 1/8 take to 112, then 101
 * bytes per jiffy. The slow start threshold is the estimate times the minimum
 * RTT (100 jiffies).
 */
class TcpWestwoodPlusLinuxEstimatorTest : public TestCase
{
  public:
    TcpWestwoodPlusLinuxEstimatorTest()
        : TestCase("Westwood+ Linux estimator")
    {
    }

  private:
    void DoRun() override;

    /**
     * Acknowledge bytes
     * @param bytes the bytes acknowledged (0 for a duplicate ACK)
     */
    void Ack(uint32_t bytes)
    {
        m_tcb->m_lastAckedSeq += bytes;
        m_cong->PktsAcked(m_tcb, std::max(bytes / 1000, 1U), MilliSeconds(100));
    }

    /**
     * Record the estimated bandwidth
     * @param oldValue the old value
     * @param newValue the new value
     */
    void Estimate(DataRate oldValue, DataRate newValue)
    {
        m_estimates.push_back(newValue);
    }

    Ptr<TcpSocketState> m_tcb;         //!< socket state
    Ptr<TcpWestwoodPlus> m_cong;       //!< congestion control
    std::vector<DataRate> m_estimates; //!< estimated bandwidths
};

void
TcpWestwoodPlusLinuxEstimatorTest::DoRun()
{
    m_tcb = CreateObject<TcpSocketState>();
    m_tcb->m_segmentSize = 1000;
    m_tcb->m_lastAckedSeq = SequenceNumber32(10000);
    m_cong = CreateObject<TcpWestwoodPlus>();
    m_cong->SetAttribute("FilterType", EnumValue(TcpWestwoodPlus::LINUX));
    m_cong->SetAttribute("LinuxHz", UintegerValue(1000));
    m_cong->TraceConnectWithoutContext(
        "EstimatedBW",
        MakeCallback(&TcpWestwoodPlusLinuxEstimatorTest::Estimate, this));

    for (uint32_t k = 0; k <= 33; ++k)
    {
        uint32_t bytes;
        if (k == 0)
        {
            bytes = 0;
        }
        else if (k <= 21)
        {
            bytes = 1000;
        }
        else if (k >= 23 && k <= 26)
        {
            bytes = 0;
        }
        else if (k == 27)
        {
            bytes = 10000;
        }
        else
        {
            bytes = 2000;
        }
        Simulator::Schedule(MilliSeconds(1000 + 10 * k),
                            &TcpWestwoodPlusLinuxEstimatorTest::Ack,
                            this,
                            bytes);
    }
    Simulator::Run();
    Simulator::Destroy();

    NS_TEST_ASSERT_MSG_EQ(m_estimates.size(), 2, "Unexpected number of estimates");
    NS_TEST_ASSERT_MSG_EQ(m_estimates[0].GetBitRate(), 800000, "Wrong first estimate");
    NS_TEST_ASSERT_MSG_EQ(m_estimates[1].GetBitRate(), 808000, "Wrong second estimate");
    NS_TEST_ASSERT_MSG_EQ(m_cong->GetSsThresh(m_tcb, 0), 10000, "Wrong slow start threshold");
}

/**
 * @ingroup internet-test
 *
 * @brief Test the LinuxRecovery attribute of TcpWestwoodPlus
 *
 * With the attribute set, a fast recovery halves the window, and the
 * bandwidth estimate sets the slow start threshold and the window when it
 * ends, or right away after a retransmission timeout; without it, the
 * estimate sets the threshold when the recovery starts.
 */
class TcpWestwoodPlusLinuxRecoveryTest : public TestCase
{
  public:
    TcpWestwoodPlusLinuxRecoveryTest()
        : TestCase("Westwood+ slow start threshold with LinuxRecovery")
    {
    }

  private:
    void DoRun() override;
};

void
TcpWestwoodPlusLinuxRecoveryTest::DoRun()
{
    Ptr<TcpSocketState> tcb = CreateObject<TcpSocketState>();
    tcb->m_segmentSize = 1000;
    tcb->m_minRtt = MilliSeconds(100);
    tcb->m_cWnd = 100000;

    Ptr<TcpWestwoodPlus> rfc = CreateObject<TcpWestwoodPlus>();
    Ptr<TcpWestwoodPlus> lnx = CreateObject<TcpWestwoodPlus>();
    lnx->SetAttribute("LinuxRecovery", BooleanValue(true));

    // 30 segments acknowledged in 100 ms: 30000 bytes in the minimum RTT
    rfc->PktsAcked(tcb, 30, MilliSeconds(100));
    lnx->PktsAcked(tcb, 30, MilliSeconds(100));
    Simulator::Run();
    Simulator::Destroy();

    NS_TEST_ASSERT_MSG_EQ(rfc->GetSsThresh(tcb, 100000), 30000, "Estimate not applied");
    NS_TEST_ASSERT_MSG_EQ(lnx->GetSsThresh(tcb, 100000), 50000, "Window not halved");

    tcb->m_ssThresh = 50000;
    tcb->m_cWnd = 52000;
    lnx->CwndEvent(tcb, TcpSocketState::CA_EVENT_COMPLETE_CWR);
    NS_TEST_ASSERT_MSG_EQ(tcb->m_ssThresh.Get(), 30000, "Estimate not applied at the end");
    NS_TEST_ASSERT_MSG_EQ(tcb->m_cWnd.Get(), 30000, "Window not set at the end");

    tcb->m_ssThresh = 15000;
    lnx->CwndEvent(tcb, TcpSocketState::CA_EVENT_LOSS);
    NS_TEST_ASSERT_MSG_EQ(tcb->m_ssThresh.Get(), 30000, "Estimate not applied at the timeout");

    tcb->m_ssThresh = 15000;
    rfc->CwndEvent(tcb, TcpSocketState::CA_EVENT_COMPLETE_CWR);
    NS_TEST_ASSERT_MSG_EQ(tcb->m_ssThresh.Get(), 15000, "Threshold changed without the attribute");
}

/**
 * @ingroup internet-test
 *
 * @brief Test the start of the first sampling window of the Linux estimator
 * of TcpWestwoodPlus
 *
 * As tcp_westwood_init() of Linux, which runs when the connection is
 * established, the first window starts with the first transmission
 * (CA_EVENT_TX_START), not with the first ACK, a round trip later. The first
 * ACK, 100 ms after the transmission, gives the RTT (100 jiffies) and counts
 * one segment (it only sets the reference SND.UNA, as a duplicate ACK); the
 * ACK 110 ms after the transmission closes the window, before counting its
 * own data, with a sample of 1000 bytes over 110 jiffies, 9 bytes per jiffy.
 * With the window starting at the first ACK, it would not close yet.
 */
class TcpWestwoodPlusLinuxStartTest : public TestCase
{
  public:
    TcpWestwoodPlusLinuxStartTest()
        : TestCase("Westwood+ Linux estimator starts with the transmission")
    {
    }

  private:
    void DoRun() override
    {
        Ptr<TcpSocketState> tcb = CreateObject<TcpSocketState>();
        tcb->m_segmentSize = 1000;
        tcb->m_lastAckedSeq = SequenceNumber32(1);
        Ptr<TcpWestwoodPlus> cong = CreateObject<TcpWestwoodPlus>();
        cong->SetAttribute("FilterType", EnumValue(TcpWestwoodPlus::LINUX));
        cong->SetAttribute("LinuxHz", UintegerValue(1000));
        std::vector<DataRate> estimates;
        cong->TraceConnectWithoutContext(
            "EstimatedBW",
            MakeBoundCallback(
                +[](std::vector<DataRate>* v, DataRate, DataRate newVal) { v->push_back(newVal); },
                &estimates));

        Simulator::Schedule(Seconds(1), [cong, tcb]() {
            cong->CwndEvent(tcb, TcpSocketState::CA_EVENT_TX_START);
        });
        Simulator::Schedule(Seconds(1) + MilliSeconds(100), [cong, tcb]() {
            tcb->m_lastAckedSeq += 2000;
            cong->PktsAcked(tcb, 2, MilliSeconds(100));
        });
        Simulator::Schedule(Seconds(1) + MilliSeconds(110), [cong, tcb]() {
            tcb->m_lastAckedSeq += 2000;
            cong->PktsAcked(tcb, 2, MilliSeconds(100));
        });
        Simulator::Run();
        Simulator::Destroy();

        NS_TEST_ASSERT_MSG_EQ(estimates.size(), 1, "The first window did not close");
        if (!estimates.empty())
        {
            NS_TEST_ASSERT_MSG_EQ(estimates[0].GetBitRate(), 9 * 8 * 1000, "Wrong first estimate");
        }
    }
};

/**
 * @ingroup internet-test
 *
 * @brief TcpWestwoodPlus TestSuite
 */
class TcpWestwoodPlusTestSuite : public TestSuite
{
  public:
    TcpWestwoodPlusTestSuite()
        : TestSuite("tcp-westwood-plus-test", Type::UNIT)
    {
        AddTestCase(
            new TcpWestwoodPlusFilterTest(TcpWestwoodPlus::TUSTIN, "Westwood+ Tustin filter"),
            TestCase::Duration::QUICK);
        AddTestCase(new TcpWestwoodPlusLinuxEstimatorTest(), TestCase::Duration::QUICK);
        AddTestCase(new TcpWestwoodPlusLinuxRecoveryTest(), TestCase::Duration::QUICK);
        AddTestCase(new TcpWestwoodPlusLinuxStartTest(), TestCase::Duration::QUICK);
    }
};

static TcpWestwoodPlusTestSuite g_tcpWestwoodPlusTest; //!< Static variable for test initialization
