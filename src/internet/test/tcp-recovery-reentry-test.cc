/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "tcp-error-model.h"
#include "tcp-general-test.h"

#include "ns3/boolean.h"
#include "ns3/config.h"
#include "ns3/data-rate.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/simulator.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/test.h"

#include <algorithm>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpRecoveryReentryTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief Check the RecoveryReentry attribute of TcpSocketBase.
 *
 * With SACK, the second and the last segment of the first window are lost,
 * and so is a segment sent in the fast recovery, which the SACKs reveal before
 * the ACK ending the recovery. With RecoveryReentry, as Linux
 * tcp_time_to_recover(), that ACK starts a new recovery at once because data
 * is still deemed lost; without it the next duplicate ACK does (RFC 6675,
 * Section 5). The window does not grow on that ACK (Linux tcp_cong_control()).
 */
class TcpRecoveryReentryTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param desc Test description.
     * @param reentry Value of the RecoveryReentry attribute.
     */
    TcpRecoveryReentryTest(const std::string& desc, bool reentry)
        : TcpGeneralTest(desc),
          m_reentry(reentry)
    {
    }

  protected:
    void ConfigureEnvironment() override;
    void ConfigureProperties() override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                        const TcpSocketState::TcpCongState_t newValue) override;
    void CWndTrace(uint32_t oldValue, uint32_t newValue) override;
    void FinalChecks() override;

  private:
    std::vector<Time> m_growths; //!< Times at which the window grew
    bool m_reentry;              //!< Value of the RecoveryReentry attribute
    std::vector<Time> m_exits;   //!< Ends of the fast recoveries
    std::vector<Time> m_starts;  //!< Starts of the fast recoveries
};

void
TcpRecoveryReentryTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(500);
    SetAppPktCount(200);
    SetAppPktInterval(MicroSeconds(10));
    SetPropagationDelay(MilliSeconds(50));
    // The segments and the ACKs arrive one after the other
    Config::SetDefault("ns3::SimpleNetDevice::DataRate", DataRateValue(DataRate("10Mbps")));
}

void
TcpRecoveryReentryTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetInitialCwnd(SENDER, 20);
    // Room for the timestamp option on top of the data size
    SetSegmentSize(SENDER, GetSegSize(SENDER) + TcpSocketBase::GetSegmentOptionsSize());
}

Ptr<TcpSocketMsgBase>
TcpRecoveryReentryTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("RecoveryReentry", BooleanValue(m_reentry));
    socket->SetAttribute("MinRto", TimeValue(Seconds(2)));
    return socket;
}

Ptr<ErrorModel>
TcpRecoveryReentryTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    // The second segment, the last of the initial window, and one sent in the fast recovery
    errorModel->AddSeqToKill(SequenceNumber32(501));
    errorModel->AddSeqToKill(SequenceNumber32(9501));
    errorModel->AddSeqToKill(SequenceNumber32(13001));
    return errorModel;
}

void
TcpRecoveryReentryTest::CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                                       const TcpSocketState::TcpCongState_t newValue)
{
    NS_LOG_DEBUG(Simulator::Now().As(Time::MS)
                 << " " << TcpSocketState::TcpCongStateName[oldValue] << " -> "
                 << TcpSocketState::TcpCongStateName[newValue]);
    if (newValue == TcpSocketState::CA_RECOVERY)
    {
        m_starts.push_back(Simulator::Now());
    }
    if (oldValue == TcpSocketState::CA_RECOVERY)
    {
        m_exits.push_back(Simulator::Now());
    }
}

void
TcpRecoveryReentryTest::CWndTrace(uint32_t oldValue, uint32_t newValue)
{
    if (newValue > oldValue)
    {
        m_growths.push_back(Simulator::Now());
    }
}

void
TcpRecoveryReentryTest::FinalChecks()
{
    Config::SetDefault("ns3::SimpleNetDevice::DataRate", DataRateValue(DataRate(0)));
    NS_TEST_ASSERT_MSG_GT_OR_EQ(m_starts.size(), 2, "No second fast recovery");
    NS_TEST_ASSERT_MSG_GT_OR_EQ(m_exits.size(), 1, "The first fast recovery did not end");
    if (m_starts.size() >= 2 && !m_exits.empty())
    {
        if (m_reentry)
        {
            NS_TEST_ASSERT_MSG_EQ(m_starts[1],
                                  m_exits[0],
                                  "The second recovery did not start on the ACK ending the first");
            NS_TEST_ASSERT_MSG_EQ(std::count(m_growths.begin(), m_growths.end(), m_exits[0]),
                                  0,
                                  "The window grew on the ACK ending the first recovery");
        }
        else
        {
            NS_TEST_ASSERT_MSG_GT(m_starts[1],
                                  m_exits[0],
                                  "The second recovery started on the ACK ending the first");
        }
    }
}

/**
 * @ingroup internet-test
 *
 * @brief TestSuite for the RecoveryReentry attribute
 */
class TcpRecoveryReentryTestSuite : public TestSuite
{
  public:
    TcpRecoveryReentryTestSuite()
        : TestSuite("tcp-recovery-reentry", Type::UNIT)
    {
        AddTestCase(new TcpRecoveryReentryTest("A recovery starts on the ACK ending one", true),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpRecoveryReentryTest("A recovery starts on a duplicate ACK", false),
                    TestCase::Duration::QUICK);
    }
};

static TcpRecoveryReentryTestSuite g_tcpRecoveryReentryTestSuite; //!< Static variable for test init
