/*
 * Copyright (c) 2018 NITK Surathkal
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Vivek Jain <jain.vivek.anand@gmail.com>
 *          Viyom Mittal <viyommittal@gmail.com>
 *          Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 *
 */

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/string.h"
#include "ns3/tcp-bbr.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TcpBbrTestSuite");

/**
 * @brief Testing whether BBR enables pacing
 */
class TcpBbrPacingEnableTest : public TestCase
{
  public:
    /**
     * @brief constructor
     * @param pacing pacing configuration
     * @param name description of the test
     */
    TcpBbrPacingEnableTest(bool pacing, const std::string& name);

  private:
    void DoRun() override;
    /**
     * @brief Execute the test.
     */
    void ExecuteTest();
    bool m_pacing; //!< Initial pacing configuration.
};

TcpBbrPacingEnableTest::TcpBbrPacingEnableTest(bool pacing, const std::string& name)
    : TestCase(name),
      m_pacing(pacing)
{
}

void
TcpBbrPacingEnableTest::DoRun()
{
    Simulator::Schedule(Seconds(0), &TcpBbrPacingEnableTest::ExecuteTest, this);
    Simulator::Run();
    Simulator::Destroy();
}

void
TcpBbrPacingEnableTest::ExecuteTest()
{
    Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
    state->m_pacing = m_pacing;

    Ptr<TcpBbr> cong = CreateObject<TcpBbr>();

    cong->CongestionStateSet(state, TcpSocketState::CA_OPEN);

    NS_TEST_ASSERT_MSG_EQ(state->m_pacing, true, "BBR has not updated pacing value");
}

/**
 * @brief Tests whether BBR sets correct value of pacing and cwnd gain based on different state.
 */
class TcpBbrCheckGainValuesTest : public TestCase
{
  public:
    /**
     * @brief constructor
     * @param state BBR state/mode under test
     * @param highGain value of pacing and cwnd gain
     * @param name description of the test
     */
    TcpBbrCheckGainValuesTest(TcpBbr::BbrMode_t state, double highGain, const std::string& name);

  private:
    void DoRun() override;
    /**
     * @brief Execute the test.
     */
    void ExecuteTest();
    TcpBbr::BbrMode_t m_mode; //!< BBR mode under test
    double m_highGain;        //!< Value of BBR high gain
};

TcpBbrCheckGainValuesTest::TcpBbrCheckGainValuesTest(TcpBbr::BbrMode_t state,
                                                     double highGain,
                                                     const std::string& name)
    : TestCase(name),
      m_mode(state),
      m_highGain(highGain)
{
}

void
TcpBbrCheckGainValuesTest::DoRun()
{
    Simulator::Schedule(Seconds(0), &TcpBbrCheckGainValuesTest::ExecuteTest, this);
    Simulator::Run();
    Simulator::Destroy();
}

void
TcpBbrCheckGainValuesTest::ExecuteTest()
{
    Ptr<TcpBbr> cong = CreateObject<TcpBbr>();
    cong->SetAttribute("HighGain", DoubleValue(m_highGain));
    double actualPacingGain;
    double actualCwndGain;
    double desiredPacingGain = m_highGain;
    double desiredCwndGain = m_highGain;
    TcpBbr::BbrMode_t desiredMode = TcpBbr::BBR_STARTUP;
    switch (m_mode)
    {
    case TcpBbr::BBR_STARTUP:
        cong->EnterStartup();
        desiredPacingGain = m_highGain;
        desiredCwndGain = m_highGain;
        actualPacingGain = cong->GetPacingGain();
        actualCwndGain = cong->GetCwndGain();
        desiredMode = TcpBbr::BBR_STARTUP;
        break;
    case TcpBbr::BBR_DRAIN:
        cong->EnterDrain();
        desiredPacingGain = 1 / m_highGain;
        desiredCwndGain = m_highGain;
        desiredMode = TcpBbr::BBR_DRAIN;
        break;
    case TcpBbr::BBR_PROBE_BW:
        // The phase in which PROBE_BW starts is otherwise drawn at random
        cong->SetAttribute("ProbeBwStartPhases", StringValue("0"));
        cong->EnterProbeBW();
        desiredPacingGain = 1.25;
        desiredCwndGain = 2;
        desiredMode = TcpBbr::BBR_PROBE_BW;
        break;
    case TcpBbr::BBR_PROBE_RTT:
        cong->EnterProbeRTT();
        desiredPacingGain = 1;
        desiredCwndGain = 1;
        desiredMode = TcpBbr::BBR_PROBE_RTT;
        break;
    default:
        NS_ASSERT(false);
    }

    actualPacingGain = cong->GetPacingGain();
    actualCwndGain = cong->GetCwndGain();
    NS_TEST_ASSERT_MSG_EQ(m_mode, desiredMode, "BBR has not entered into desired state");
    NS_TEST_ASSERT_MSG_EQ(actualPacingGain,
                          desiredPacingGain,
                          "BBR has not updated into desired pacing gain");
    NS_TEST_ASSERT_MSG_EQ(actualCwndGain,
                          desiredCwndGain,
                          "BBR has not updated into desired cwnd gain");
}

/**
 * @ingroup internet-test
 *
 * @brief Check the LinuxArithmetic attribute of TcpBbr.
 *
 * As Linux (bbr_init_pacing_rate_from_rtt()), the initial pacing rate is the
 * window over the smoothed RTT, in whole packets per microsecond scaled by
 * 2^24, times the high gain of 739/256, less 1%, rounded down to whole bytes
 * per second; the drain gain is 88/256. Here, ten segments of 1448 bytes over
 * 82059 us make 504160 bytes per second.
 */
class TcpBbrLinuxArithmeticTest : public TestCase
{
  public:
    /**
     * @brief Constructor
     * @param linux Value of the LinuxArithmetic attribute
     * @param name Description of the test
     */
    TcpBbrLinuxArithmeticTest(bool linux, const std::string& name)
        : TestCase(name),
          m_linux(linux)
    {
    }

  private:
    void DoRun() override;
    bool m_linux; //!< Value of the LinuxArithmetic attribute
};

void
TcpBbrLinuxArithmeticTest::DoRun()
{
    Ptr<TcpSocketState> state = CreateObject<TcpSocketState>();
    state->m_pacing = true;
    state->m_segmentSize = 1448;
    state->m_cWnd = 10 * 1448;
    state->m_srtt = MicroSeconds(82059);
    state->m_minRtt = MicroSeconds(82059);

    Ptr<TcpBbr> cong = CreateObject<TcpBbr>();
    cong->SetAttribute("LinuxArithmetic", BooleanValue(m_linux));
    cong->CongestionStateSet(state, TcpSocketState::CA_OPEN);

    if (m_linux)
    {
        NS_TEST_ASSERT_MSG_EQ(cong->GetPacingGain(), 739.0 / 256, "Wrong high gain");
        NS_TEST_ASSERT_MSG_EQ(state->m_pacingRate.Get().GetBitRate(),
                              504160 * 8,
                              "Wrong initial pacing rate");
        cong->EnterDrain();
        NS_TEST_ASSERT_MSG_EQ(cong->GetPacingGain(), 88.0 / 256, "Wrong drain gain");
    }
    else
    {
        NS_TEST_ASSERT_MSG_NE(state->m_pacingRate.Get().GetBitRate(),
                              504160 * 8,
                              "The pacing rate is computed as Linux");
    }
    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 *
 * @brief Check the phase in which PROBE_BW starts.
 *
 * As Linux (bbr_reset_probe_bw_mode()), the phase is drawn among the seven
 * phases of the pacing gain cycle but the draining one (gain 3/4), all of
 * which are drawn over many draws, or are the ones ProbeBwStartPhases sets
 * in turn. The
 * phase is found from the number of phases until the gain of 5/4.
 */
class TcpBbrProbeBwStartPhaseTest : public TestCase
{
  public:
    TcpBbrProbeBwStartPhaseTest()
        : TestCase("BBR starts PROBE_BW in any phase but the draining one")
    {
    }

  private:
    void DoRun() override;

    /**
     * @brief Enter PROBE_BW and return the phase it starts in
     * @param cong The congestion control
     * @return the phase, from 0 to 7
     */
    uint32_t StartPhase(Ptr<TcpBbr> cong);
};

uint32_t
TcpBbrProbeBwStartPhaseTest::StartPhase(Ptr<TcpBbr> cong)
{
    cong->EnterProbeBW();
    uint32_t advances = 0;
    while (cong->GetPacingGain() != 1.25 && advances < 8)
    {
        cong->AdvanceCyclePhase();
        ++advances;
    }
    if (advances > 0 && cong->GetPacingGain() == 1.25)
    {
        // The phase before the one of gain 5/4 has gain 1, never 3/4
        cong->AdvanceCyclePhase();
    }
    return (8 - advances) % 8;
}

void
TcpBbrProbeBwStartPhaseTest::DoRun()
{
    for (uint32_t phase : {0U, 2U, 3U, 4U, 5U, 6U, 7U})
    {
        Ptr<TcpBbr> cong = CreateObject<TcpBbr>();
        cong->SetAttribute("ProbeBwStartPhases", StringValue(std::to_string(phase)));
        NS_TEST_ASSERT_MSG_EQ(StartPhase(cong), phase, "PROBE_BW not started in the phase set");
    }
    Ptr<TcpBbr> listed = CreateObject<TcpBbr>();
    listed->SetAttribute("ProbeBwStartPhases", StringValue("3 0"));
    NS_TEST_ASSERT_MSG_EQ(StartPhase(listed), 3, "First phase listed not used");
    NS_TEST_ASSERT_MSG_EQ(StartPhase(listed), 0, "Second phase listed not used");
    NS_TEST_ASSERT_MSG_NE(StartPhase(listed), 1, "PROBE_BW started in the draining phase");
    Ptr<TcpBbr> commas = CreateObject<TcpBbr>();
    commas->SetAttribute("ProbeBwStartPhases", StringValue("5,0"));
    NS_TEST_ASSERT_MSG_EQ(StartPhase(commas), 5, "First phase of a comma list not used");
    NS_TEST_ASSERT_MSG_EQ(StartPhase(commas), 0, "Second phase of a comma list not used");
    std::set<uint32_t> drawn;
    for (uint32_t stream = 0; stream < 200; ++stream)
    {
        Ptr<TcpBbr> cong = CreateObject<TcpBbr>();
        cong->SetAttribute("Stream", UintegerValue(stream));
        drawn.insert(StartPhase(cong));
    }
    NS_TEST_ASSERT_MSG_EQ(drawn.count(1), 0, "PROBE_BW started in the draining phase");
    NS_TEST_ASSERT_MSG_EQ(drawn.size(), 7, "Some phases are never drawn");
    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 *
 * @brief Check the timing of TcpBbr as Linux.
 *
 * - With LinuxHz, the minimum RTT is timed in jiffies (tcp_jiffies32): it
 *   expires only once a whole jiffy past the window, here at 11.05 s for a
 *   connection set up at 1.0421 s, whereas it expires at 11.0421 s otherwise.
 * - With LinuxArithmetic, only an RTT sample below the minimum, in whole
 *   microseconds, renews it before it expires (bbr_update_min_rtt()); an
 *   equal sample renews it otherwise.
 * - The ACK aggregation epoch is reset past AckEpochAckedResetThresh
 *   segments, 2^20 by default as Linux (bbr_ack_epoch_acked_reset_thresh), so
 *   that a burst of 20 ACKs gives an allowance of the whole burst, with or
 *   without LinuxArithmetic, whereas a threshold of 8 segments resets it
 *   before it reaches 8 segments.
 * - With LinuxArithmetic, the data in flight is counted at the departure time
 *   of the next packet (bbr_packets_in_net_at_edt()): less the data that the
 *   bottleneck delivers until then, plus a GSO packet when the pacing gain is
 *   above 1.
 */
class TcpBbrLinuxTimingTest : public TestCase
{
  public:
    TcpBbrLinuxTimingTest()
        : TestCase("BBR timing as Linux")
    {
    }

  private:
    void DoRun() override;

    /**
     * @brief Create a BBR set up at the current time.
     * @param hz Value of the LinuxHz attribute.
     * @param linux Value of the LinuxArithmetic attribute.
     * @param tcb The socket state.
     * @return the congestion control.
     */
    Ptr<TcpBbr> Create(uint32_t hz, bool linux, Ptr<TcpSocketState> tcb);

    /**
     * @brief Check the expiry of the minimum RTT.
     */
    void CheckExpiry();

    /**
     * @brief Check the renewal of the minimum RTT by an equal sample.
     */
    void CheckRenewal();

    /**
     * @brief Check the data in flight at the departure time.
     */
    void CheckInNetAtEdt();

    /**
     * @brief Check the ACK aggregation allowance.
     */
    void CheckAckAggregation();

    std::vector<Ptr<TcpBbr>> m_bbr;         //!< Congestion controls (with and without)
    std::vector<Ptr<TcpSocketState>> m_tcb; //!< Their socket states
};

Ptr<TcpBbr>
TcpBbrLinuxTimingTest::Create(uint32_t hz, bool linux, Ptr<TcpSocketState> tcb)
{
    tcb->m_pacing = true;
    tcb->m_segmentSize = 1448;
    tcb->m_cWnd = 10 * 1448;
    tcb->m_srtt = MicroSeconds(40000);
    tcb->m_minRtt = MicroSeconds(40000);
    Ptr<TcpBbr> cong = CreateObject<TcpBbr>();
    cong->SetAttribute("LinuxHz", UintegerValue(hz));
    cong->SetAttribute("LinuxArithmetic", BooleanValue(linux));
    cong->CongestionStateSet(tcb, TcpSocketState::CA_OPEN);
    return cong;
}

void
TcpBbrLinuxTimingTest::DoRun()
{
    Simulator::Schedule(MicroSeconds(1042100), [this]() {
        // With and without the options, to check the expiry, then the renewal
        for (bool linux : {true, false, true, false})
        {
            Ptr<TcpSocketState> tcb = CreateObject<TcpSocketState>();
            m_bbr.push_back(Create(linux ? 100 : 0, linux, tcb));
            m_tcb.push_back(tcb);
        }
    });
    Simulator::Schedule(Seconds(5), &TcpBbrLinuxTimingTest::CheckRenewal, this);
    Simulator::Schedule(MicroSeconds(11045000), &TcpBbrLinuxTimingTest::CheckExpiry, this);
    Simulator::Schedule(Seconds(12), &TcpBbrLinuxTimingTest::CheckInNetAtEdt, this);
    Simulator::Schedule(Seconds(13), &TcpBbrLinuxTimingTest::CheckAckAggregation, this);
    Simulator::Run();
    Simulator::Destroy();
}

void
TcpBbrLinuxTimingTest::CheckRenewal()
{
    for (size_t i = 2; i < 4; ++i)
    {
        Time stamp = m_bbr[i]->m_minRttStamp;
        m_tcb[i]->m_lastRtt = m_bbr[i]->m_minRtt.Get();
        m_bbr[i]->UpdateRTprop(m_tcb[i]);
        if (i == 2)
        {
            NS_TEST_ASSERT_MSG_EQ(m_bbr[i]->m_minRttStamp, stamp, "An equal sample renewed it");
            NS_TEST_ASSERT_MSG_EQ(stamp, Seconds(1.04), "The stamp is not on a jiffy");
        }
        else
        {
            NS_TEST_ASSERT_MSG_EQ(m_bbr[i]->m_minRttStamp,
                                  Simulator::Now(),
                                  "An equal sample did not renew it");
        }
    }
}

void
TcpBbrLinuxTimingTest::CheckExpiry()
{
    for (size_t i = 0; i < 2; ++i)
    {
        // A sample above the minimum renews it only once it expired
        m_tcb[i]->m_lastRtt = MicroSeconds(50000);
        m_bbr[i]->UpdateRTprop(m_tcb[i]);
    }
    NS_TEST_ASSERT_MSG_EQ(m_bbr[0]->m_minRttExpired, false, "Expired before the next jiffy");
    NS_TEST_ASSERT_MSG_EQ(m_bbr[1]->m_minRttExpired, true, "Not expired after 10 s");
}

void
TcpBbrLinuxTimingTest::CheckAckAggregation()
{
    TcpRateOps::TcpRateSample rs;
    rs.m_ackedSacked = 1448;
    rs.m_delivered = 1448;
    rs.m_interval = MilliSeconds(40);
    // With and without LinuxArithmetic, then with a small threshold
    Ptr<TcpSocketState> smallTcb = CreateObject<TcpSocketState>();
    Ptr<TcpBbr> small = Create(0, false, smallTcb);
    small->SetAttribute("AckEpochAckedResetThresh", UintegerValue(8));
    std::vector<Ptr<TcpBbr>> bbrs{m_bbr[0], m_bbr[1], small};
    std::vector<Ptr<TcpSocketState>> tcbs{m_tcb[0], m_tcb[1], smallTcb};
    for (size_t i = 0; i < bbrs.size(); ++i)
    {
        Ptr<TcpBbr> bbr = bbrs[i];
        Ptr<TcpSocketState> tcb = tcbs[i];
        tcb->m_cWnd = 100 * 1448;
        // One segment per millisecond at the bottleneck, and 20 ACKs at once
        bbr->m_maxBwFilter.Reset(DataRate(1448 * 8 * 1000), 0);
        bbr->m_roundStart = false;
        bbr->m_extraAcked[0] = 0;
        bbr->m_extraAcked[1] = 0;
        for (uint32_t ack = 0; ack < 20; ++ack)
        {
            bbr->UpdateAckAggregation(tcb, rs);
        }
    }
    for (size_t i = 0; i < 2; ++i)
    {
        NS_TEST_ASSERT_MSG_EQ(std::max(bbrs[i]->m_extraAcked[0], bbrs[i]->m_extraAcked[1]),
                              20 * 1448,
                              "The whole burst does not count as aggregation");
    }
    NS_TEST_ASSERT_MSG_EQ(std::max(small->m_extraAcked[0], small->m_extraAcked[1]),
                          7 * 1448,
                          "The epoch did not restart before reaching 8 segments");
}

void
TcpBbrLinuxTimingTest::CheckInNetAtEdt()
{
    Ptr<TcpBbr> bbr = m_bbr[0];
    Ptr<TcpSocketState> tcb = m_tcb[0];
    // One segment per millisecond at the bottleneck
    bbr->m_maxBwFilter.Reset(DataRate(1448 * 8 * 1000), 0);
    tcb->m_pacingRate = DataRate(1448 * 8 * 1000);
    tcb->m_pacingDeparture = Simulator::Now() + MicroSeconds(3500);
    bbr->m_pacingGain = 1;
    NS_TEST_ASSERT_MSG_EQ(bbr->InNetAtEdt(tcb, 10 * 1448),
                          7 * 1448,
                          "The data delivered until the departure is still counted");
    bbr->m_pacingGain = 1.25;
    NS_TEST_ASSERT_MSG_EQ(bbr->InNetAtEdt(tcb, 10 * 1448),
                          (7 + bbr->TsoSegsGoal(tcb)) * 1448,
                          "The GSO packet to send is not counted");
    tcb->m_pacingDeparture = Simulator::Now() - MilliSeconds(3);
    bbr->m_pacingGain = 1;
    NS_TEST_ASSERT_MSG_EQ(bbr->InNetAtEdt(tcb, 10 * 1448),
                          10 * 1448,
                          "A past departure time is counted");
}

/**
 * @ingroup internet-test
 *
 * @brief TCP BBR TestSuite
 */
class TcpBbrTestSuite : public TestSuite
{
  public:
    /**
     * @brief constructor
     */
    TcpBbrTestSuite()
        : TestSuite("tcp-bbr-test", Type::UNIT)
    {
        AddTestCase(new TcpBbrPacingEnableTest(true, "BBR must keep pacing feature on"),
                    TestCase::Duration::QUICK);

        AddTestCase(new TcpBbrPacingEnableTest(false, "BBR must turn on pacing feature"),
                    TestCase::Duration::QUICK);

        AddTestCase(
            new TcpBbrCheckGainValuesTest(
                TcpBbr::BBR_STARTUP,
                4,
                "BBR should enter to STARTUP phase and set cwnd and pacing gain accordingly"),
            TestCase::Duration::QUICK);

        AddTestCase(new TcpBbrCheckGainValuesTest(
                        TcpBbr::BBR_DRAIN,
                        4,
                        "BBR should enter to DRAIN phase and set cwnd and pacing gain accordingly"),
                    TestCase::Duration::QUICK);

        AddTestCase(
            new TcpBbrCheckGainValuesTest(
                TcpBbr::BBR_PROBE_BW,
                4,
                "BBR should enter to BBR_PROBE_BW phase and set cwnd and pacing gain accordingly"),
            TestCase::Duration::QUICK);

        AddTestCase(
            new TcpBbrCheckGainValuesTest(
                TcpBbr::BBR_PROBE_RTT,
                4,
                "BBR should enter to BBR_PROBE_RTT phase and set cwnd and pacing gain accordingly"),
            TestCase::Duration::QUICK);
        AddTestCase(new TcpBbrLinuxArithmeticTest(true, "BBR pacing rate computed as Linux"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpBbrLinuxArithmeticTest(false, "BBR pacing rate in floating point"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpBbrProbeBwStartPhaseTest(), TestCase::Duration::QUICK);
        AddTestCase(new TcpBbrLinuxTimingTest(), TestCase::Duration::QUICK);
    }
};

static TcpBbrTestSuite g_tcpBbrTest; //!< static variable for test initialization
} // namespace ns3
