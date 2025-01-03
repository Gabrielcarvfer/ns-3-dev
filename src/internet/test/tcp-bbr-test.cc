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

#include "ns3/log.h"
#include "ns3/tcp-bbr.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/test.h"

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
        cong->EnterProbeBW();
        // The value of desiredPacingGain is sensitive to the setting of random
        // variable stream. The value of 1.25 has been used in this test with a
        // stream value of 4 (default for TCP BBR). Note that if the stream value
        // is changed, this test might fail because when BBR enters the PROBE_BW
        // phase, the value of actualPacingGain is chosen randomly from 1.25,
        // 0.75, 1, 1, 1, 1, 1, 1.
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
    }
};

static TcpBbrTestSuite g_tcpBbrTest; //!< static variable for test initialization
} // namespace ns3
