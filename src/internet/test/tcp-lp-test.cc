/*
 * Copyright (c) 2016 NITK Surathkal
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Charitha Sangaraju <charitha29193@gmail.com>
 *          Nandita G <gm.nandita@gmail.com>
 *          Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 *
 */

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-lp.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/test.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TcpLpTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief Testing the behaviour common to New Reno
 */
class TcpLpToNewReno : public TestCase
{
  public:
    /**
     * Constructor
     * @param cWnd Congestion window size
     * @param segmentSize Segment size
     * @param segmentsAcked Segments acked
     * @param ssThresh Slow start threshold
     * @param rtt RTT
     * @param name Test case name
     */
    TcpLpToNewReno(uint32_t cWnd,
                   uint32_t segmentSize,
                   uint32_t segmentsAcked,
                   uint32_t ssThresh,
                   Time rtt,
                   const std::string& name);

  private:
    void DoRun() override;
    uint32_t m_cWnd;             //!< Congestion window size
    uint32_t m_segmentSize;      //!< Segment size
    uint32_t m_ssThresh;         //!< Slow start threshold
    uint32_t m_segmentsAcked;    //!< Segments acked
    Time m_rtt;                  //!< RTT
    Ptr<TcpSocketState> m_state; //!< TCP socket state
};

TcpLpToNewReno::TcpLpToNewReno(uint32_t cWnd,
                               uint32_t segmentSize,
                               uint32_t segmentsAcked,
                               uint32_t ssThresh,
                               Time rtt,
                               const std::string& name)
    : TestCase(name),
      m_cWnd(cWnd),
      m_segmentSize(segmentSize),
      m_ssThresh(ssThresh),
      m_segmentsAcked(segmentsAcked),
      m_rtt(rtt)
{
}

void
TcpLpToNewReno::DoRun()
{
    m_state = CreateObject<TcpSocketState>();
    m_state->m_cWnd = m_cWnd;
    m_state->m_ssThresh = m_ssThresh;
    m_state->m_segmentSize = m_segmentSize;

    Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
    state->m_cWnd = m_cWnd;
    state->m_ssThresh = m_ssThresh;
    state->m_segmentSize = m_segmentSize;

    Ptr<TcpLp> cong = CreateObject<TcpLp>();

    m_state->m_rcvTimestampValue = 2;
    m_state->m_rcvTimestampEchoReply = 1;

    cong->PktsAcked(m_state, m_segmentsAcked, m_rtt);
    cong->IncreaseWindow(m_state, m_segmentsAcked);

    Ptr<TcpNewReno> NewRenoCong = CreateObject<TcpNewReno>();
    NewRenoCong->IncreaseWindow(state, m_segmentsAcked);

    NS_TEST_ASSERT_MSG_EQ(m_state->m_cWnd.Get(),
                          state->m_cWnd.Get(),
                          "cWnd has not updated correctly");
    Simulator::Run();
    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 *
 * @brief Test the delay inference of TcpLp, ACK by ACK
 *
 * A sequence of events is fed to TCP-LP, one every 10 ms, with a smoothed
 * RTT of 100 ms (an inference time-out of 300 ms): an ACK of two segments,
 * whose one way delay (the timestamp of the receiver minus the echoed one) is
 * given, an ACK of a single segment, or a loss. The sender keeps a window of
 * 20 segments in flight, so that the data sent before a reaction is
 * acknowledged ten ACKs later. The window is set to 20 segments before each
 * ACK, and compared after it: '.' when it is kept, 'h' when it is halved,
 * '1' when it is set to one segment.
 */
class TcpLpSequenceTest : public TestCase
{
  public:
    /**
     * An event of the sequence
     */
    struct Event
    {
        char type;      //!< 'a': ACK of two segments, 's': ACK of one segment, 'l': loss
        int32_t owd;    //!< one way delay of the ACK, in ms
        uint32_t count; //!< number of such events
    };

    /**
     * Constructor
     * @param events the events
     * @param expected the expected window after each ACK
     * @param name the test name
     * @param tsOffset the offset of the clock of the receiver, in ms
     * @param safeguards whether the AckHoldFilter and RestartSmoothing
     *        attributes are set
     */
    TcpLpSequenceTest(const std::vector<Event>& events,
                      const std::string& expected,
                      const std::string& name,
                      int32_t tsOffset = 0,
                      bool safeguards = false)
        : TestCase(name),
          m_events(events),
          m_expected(expected),
          m_tsOffset(tsOffset),
          m_safeguards(safeguards)
    {
    }

  private:
    void DoRun() override;

    /**
     * Feed an event to TCP-LP
     * @param type the event type
     * @param owd the one way delay, in ms
     */
    void Feed(char type, int32_t owd);

    std::vector<Event> m_events; //!< the events
    std::string m_expected;      //!< expected windows
    int32_t m_tsOffset;          //!< offset of the clock of the receiver
    bool m_safeguards;           //!< whether the safeguards against the ACK hold are set
    std::string m_result;        //!< windows after each ACK
    Ptr<TcpSocketState> m_tcb;   //!< socket state
    Ptr<TcpLp> m_cong;           //!< congestion control
};

void
TcpLpSequenceTest::Feed(char type, int32_t owd)
{
    const uint32_t window = 20 * m_tcb->m_segmentSize;
    m_tcb->m_cWnd = window;
    if (type == 'l')
    {
        m_cong->CongestionStateSet(m_tcb, TcpSocketState::CA_RECOVERY);
        m_tcb->m_ssThresh = m_cong->GetSsThresh(m_tcb, window);
        m_cong->CongestionStateSet(m_tcb, TcpSocketState::CA_OPEN);
        return;
    }
    m_tcb->m_lastAckedSeq += (type == 's' ? 1 : 2) * m_tcb->m_segmentSize;
    uint32_t now = Simulator::Now().GetMilliSeconds();
    m_tcb->m_rcvTimestampEchoReply = now - 100;
    m_tcb->m_rcvTimestampValue = now - 100 + owd + m_tsOffset;
    m_cong->PktsAcked(m_tcb, type == 's' ? 1 : 2, MilliSeconds(100));
    m_tcb->m_highTxMark = m_tcb->m_lastAckedSeq + window;
    uint32_t cwnd = m_tcb->GetCwndInSegments();
    m_result += (cwnd == 20 ? '.' : (cwnd == 10 ? 'h' : (cwnd == 1 ? '1' : '?')));
}

void
TcpLpSequenceTest::DoRun()
{
    m_tcb = CreateObject<TcpSocketState>();
    m_tcb->m_segmentSize = 1000;
    m_tcb->m_ssThresh = 10000;
    m_tcb->m_srtt = MilliSeconds(100);
    m_tcb->m_lastAckedSeq = SequenceNumber32(1);
    m_tcb->m_highTxMark = SequenceNumber32(20001);
    m_cong = CreateObject<TcpLp>();
    m_cong->SetAttribute("AckHoldFilter", BooleanValue(m_safeguards));
    m_cong->SetAttribute("RestartSmoothing", BooleanValue(m_safeguards));

    Time t = Seconds(1);
    for (const auto& e : m_events)
    {
        for (uint32_t i = 0; i < e.count; ++i)
        {
            Simulator::Schedule(t, &TcpLpSequenceTest::Feed, this, e.type, e.owd);
            t += MilliSeconds(10);
        }
    }
    Simulator::Run();
    Simulator::Destroy();

    NS_TEST_ASSERT_MSG_EQ(m_result, m_expected, "The windows differ from the expected ones");
}

/**
 * @ingroup internet-test
 *
 * @brief Test that TcpLp does not increase its window in slow start within
 * the inference
 */
class TcpLpInferenceSlowStartTest : public TestCase
{
  public:
    TcpLpInferenceSlowStartTest()
        : TestCase("LP holds slow start within the inference")
    {
    }

  private:
    void DoRun() override;

    /// Feed the ACKs and check the window
    void Check();
};

void
TcpLpInferenceSlowStartTest::Check()
{
    Ptr<TcpSocketState> tcb = CreateObject<TcpSocketState>();
    tcb->m_segmentSize = 1000;
    tcb->m_cWnd = 20000;
    tcb->m_srtt = MilliSeconds(100);
    tcb->m_lastAckedSeq = SequenceNumber32(1);
    tcb->m_highTxMark = SequenceNumber32(20001);
    Ptr<TcpLp> cong = CreateObject<TcpLp>();

    // A loss, from which on the delays are used
    cong->GetSsThresh(tcb, 20000);
    tcb->m_ssThresh = 100000;
    // ACKs with a delay of 20 ms, then of 60 ms, until the window is halved
    for (uint32_t i = 0; i < 20 && tcb->m_cWnd.Get() == 20000; ++i)
    {
        tcb->m_lastAckedSeq += 2000;
        tcb->m_rcvTimestampEchoReply = 1000 + 10 * i;
        tcb->m_rcvTimestampValue = tcb->m_rcvTimestampEchoReply + (i < 10 ? 20 : 60);
        cong->PktsAcked(tcb, 2, MilliSeconds(100));
        tcb->m_highTxMark = tcb->m_lastAckedSeq + 20000;
    }
    NS_TEST_EXPECT_MSG_EQ(tcb->m_cWnd.Get(), 10000, "The window was not halved");

    cong->IncreaseWindow(tcb, 2);
    NS_TEST_EXPECT_MSG_EQ(tcb->m_cWnd.Get(), 10000, "The window grew within the inference");
}

void
TcpLpInferenceSlowStartTest::DoRun()
{
    Simulator::Schedule(Seconds(1), &TcpLpInferenceSlowStartTest::Check, this);
    Simulator::Run();
    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 *
 * @brief Test the LinuxArithmetic attribute of TcpLp
 *
 * A sequence of ACKs is fed to TCP-LP with the attribute set, with a window
 * of 20 segments before each one, and the window after each ACK is compared
 * with the one that tcp_lp.c of Linux 6.12 gives for the same timestamps
 * (reproduced with its C integer semantics): '.' when the window is kept, 'h'
 * when it is halved, '1' when it is set to one segment. The sequences cover
 * the estimate of the clock rate of the receiver (an estimate above 1000 Hz
 * makes the integer factor of the remote timestamps null), and the 32-bit
 * wraparound of the smoothed one way delay when the timestamps carry a large
 * offset.
 */
class TcpLpLinuxArithmeticTest : public TestCase
{
  public:
    /**
     * Constructor
     * @param sequence the ACK sequence (0: random delays, 1: fast remote clock)
     * @param offset the offset of the timestamps of the sender
     * @param expected the expected window after each ACK
     * @param name the test name
     */
    TcpLpLinuxArithmeticTest(uint32_t sequence,
                             uint32_t offset,
                             const std::string& expected,
                             const std::string& name)
        : TestCase(name),
          m_sequence(sequence),
          m_offset(offset),
          m_expected(expected)
    {
    }

  private:
    void DoRun() override;

    /**
     * Feed an ACK to TCP-LP and record the window
     * @param tsval the timestamp of the receiver
     * @param tsecr the echoed timestamp, on the local clock
     */
    void Ack(uint32_t tsval, uint32_t tsecr);

    uint32_t m_sequence;       //!< ACK sequence
    uint32_t m_offset;         //!< timestamp offset of the sender
    std::string m_expected;    //!< expected windows
    std::string m_result;      //!< windows after each ACK
    Ptr<TcpSocketState> m_tcb; //!< socket state
    Ptr<TcpLp> m_cong;         //!< congestion control
};

void
TcpLpLinuxArithmeticTest::Ack(uint32_t tsval, uint32_t tsecr)
{
    m_tcb->m_cWnd = 20 * m_tcb->m_segmentSize;
    m_tcb->m_rcvTimestampValue = tsval;
    m_tcb->m_rcvTimestampEchoReply = tsecr;
    m_cong->PktsAcked(m_tcb, 1, MilliSeconds(100));
    uint32_t cwnd = m_tcb->GetCwndInSegments();
    m_result += (cwnd == 20 ? '.' : (cwnd == 10 ? 'h' : (cwnd == 1 ? '1' : '?')));
}

void
TcpLpLinuxArithmeticTest::DoRun()
{
    m_tcb = CreateObject<TcpSocketState>();
    m_tcb->m_segmentSize = 1000;
    m_tcb->m_ssThresh = UINT32_MAX;
    m_tcb->m_timestampOffset = m_offset;
    m_cong = CreateObject<TcpLp>();
    m_cong->SetAttribute("LinuxArithmetic", BooleanValue(true));

    uint32_t x = 1;
    for (uint32_t i = 1; i <= m_expected.size(); ++i)
    {
        uint32_t now;
        uint32_t tsval;
        uint32_t tsecr;
        if (m_sequence == 0)
        {
            // Pseudo-random delays, larger every other 30 ACKs
            x = (x * 1103515245U + 12345U) & 0x7fffffff;
            now = 1000 + 7 * i;
            uint32_t q = ((i / 30) % 2) ? (x >> 8) % 40 : (x >> 8) % 4;
            tsval = now - 40 + q;
            tsecr = now - 85 - ((x >> 4) % 3);
        }
        else
        {
            // The timestamps of the receiver run 10% faster, with a delay ramp
            now = 1000 + 10 * i;
            tsval = 900 + 11 * i + (i < 15 ? 0 : 3 * (i - 15));
            tsecr = 900 + 10 * i;
        }
        Simulator::Schedule(MilliSeconds(now), &TcpLpLinuxArithmeticTest::Ack, this, tsval, tsecr);
    }
    Simulator::Run();
    Simulator::Destroy();

    NS_TEST_ASSERT_MSG_EQ(m_result, m_expected, "The windows differ from those of Linux");
}

/**
 * @ingroup internet-test
 *
 * Test the behaviour common to New Reno
 */
class TcpLpTestSuite : public TestSuite
{
  public:
    TcpLpTestSuite()
        : TestSuite("tcp-lp-test", Type::UNIT)
    {
        AddTestCase(new TcpLpToNewReno(4 * 1446,
                                       1446,
                                       2,
                                       2 * 1446,
                                       MilliSeconds(100),
                                       "LP falls to New Reno if the cwd is within threshold"),
                    TestCase::Duration::QUICK);

        AddTestCase(new TcpLpSequenceTest({{'a', 20, 10}, {'a', 60, 10}},
                                          "....................",
                                          "LP infers no congestion before the first loss"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpLpSequenceTest({{'l', 0, 1}, {'a', 20, 12}, {'a', 60, 40}},
                                          ".............h.........1.........1.........1........",
                                          "LP reacts once per window, and sets the window to "
                                          "one segment on an indication within the inference"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpLpSequenceTest({{'l', 0, 1}, {'a', 20, 12}, {'a', 60, 5}},
                                          ".............h...",
                                          "LP measures the delays whatever the clock offset",
                                          -1000000),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpLpSequenceTest({{'l', 0, 1}, {'a', 10, 1}, {'a', 11, 20}},
                                          ".....................",
                                          "LP ignores the delays below the timestamp granularity"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpLpInferenceSlowStartTest(), TestCase::Duration::QUICK);
        AddTestCase(new TcpLpSequenceTest({{'l', 0, 1}, {'a', 20, 12}, {'s', 60, 2}, {'a', 20, 5}},
                                          "...................",
                                          "LP ignores the delays of the ACKs of a single segment",
                                          0,
                                          true),
                    TestCase::Duration::QUICK);
        AddTestCase(
            new TcpLpSequenceTest({{'l', 0, 1}, {'a', 20, 12}, {'a', 60, 11}, {'a', 22, 10}},
                                  ".............h...................",
                                  "LP restarts the smoothed delay after a reaction",
                                  0,
                                  true),
            TestCase::Duration::QUICK);

        AddTestCase(
            new TcpLpLinuxArithmeticTest(
                0,
                0,
                ".h.1.1......................................................................."
                "......................h....................",
                "LP with the Linux arithmetic, no timestamp offset"),
            TestCase::Duration::QUICK);
        AddTestCase(new TcpLpLinuxArithmeticTest(
                        0,
                        4000000000U,
                        ".h......................................................................."
                        "...............................................",
                        "LP with the Linux arithmetic, large timestamp offset"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpLpLinuxArithmeticTest(1,
                                                 0,
                                                 "........................................",
                                                 "LP with the Linux arithmetic, fast remote clock"),
                    TestCase::Duration::QUICK);
    }
};

static TcpLpTestSuite g_tcplpTest; //!< static var for test initialization

} // namespace ns3
