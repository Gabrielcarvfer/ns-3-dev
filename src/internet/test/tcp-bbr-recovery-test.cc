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
#include "ns3/tcp-bbr.h"
#include "ns3/tcp-header.h"
#include "ns3/test.h"

#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpBbrRecoveryTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief TcpBbr which reports what each ACK gives it.
 */
class TcpBbrRecorder : public TcpBbr
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    /// What an ACK gave to the congestion control
    struct Record
    {
        Time time;                            //!< Time of the ACK
        TcpSocketState::TcpCongState_t state; //!< Congestion state
        uint32_t bytesLoss;                   //!< Bytes newly lost
        uint32_t ackedSacked;                 //!< Bytes newly delivered
        uint32_t inFlight;                    //!< Bytes in flight
        uint32_t cWnd;                        //!< Window set
        bool conservation;                    //!< Packet conservation on
    };

    /// Callback receiving the records
    using RecordCb = Callback<void, const Record&>;

    /**
     * @brief Set the callback receiving the records of all the instances.
     * @param cb The callback.
     */
    static void SetRecordCallback(RecordCb cb)
    {
        m_cb = cb;
    }

    void CongControl(Ptr<TcpSocketState> tcb,
                     const TcpRateOps::TcpRateConnection& rc,
                     const TcpRateOps::TcpRateSample& rs) override
    {
        TcpBbr::CongControl(tcb, rc, rs);
        if (!m_cb.IsNull())
        {
            m_cb(Record{Simulator::Now(),
                        tcb->m_congState.Get(),
                        rs.m_bytesLoss,
                        rs.m_ackedSacked,
                        tcb->m_bytesInFlight.Get(),
                        tcb->m_cWnd.Get(),
                        m_packetConservation});
        }
    }

    Ptr<TcpCongestionOps> Fork() override
    {
        return CopyObject<TcpBbrRecorder>(this);
    }

  private:
    static RecordCb m_cb; //!< Callback receiving the records
};

TcpBbrRecorder::RecordCb TcpBbrRecorder::m_cb;

NS_OBJECT_ENSURE_REGISTERED(TcpBbrRecorder);

TypeId
TcpBbrRecorder::GetTypeId()
{
    static TypeId tid = TypeId("ns3::TcpBbrRecorder")
                            .SetParent<TcpBbr>()
                            .AddConstructor<TcpBbrRecorder>()
                            .SetGroupName("Internet");
    return tid;
}

/**
 * @ingroup internet-test
 *
 * @brief Check the window of TcpBbr through a fast recovery.
 *
 * One segment is lost in slow start. As in Linux (bbr_set_cwnd_to_recover_or_restore())
 * and the BBR Internet Draft:
 * - the ACK which starts the recovery sets the window to the data in flight
 *   plus the data it delivers before the retransmission is sent, so that the
 *   retransmission is the only segment sent on that ACK;
 * - the packet conservation lasts one round trip from the start of the
 *   recovery: a new round starts then, so that the window does not grow
 *   before the data sent in the recovery is delivered;
 * - the ACK ending the recovery acknowledges the data deemed lost, but deems
 *   none lost: the rate sample reports no loss;
 * - the end of the recovery leaves the window that BBR sets, instead of the
 *   slow start threshold, which BBR leaves unset in STARTUP.
 */
class TcpBbrRecoveryTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param desc Test description.
     */
    TcpBbrRecoveryTest(const std::string& desc)
        : TcpGeneralTest(desc)
    {
    }

  protected:
    void ConfigureEnvironment() override;
    void ConfigureProperties() override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                        const TcpSocketState::TcpCongState_t newValue) override;
    void FinalChecks() override;

  private:
    /**
     * @brief Receive a record of the congestion control.
     * @param r The record.
     */
    void Recorded(const TcpBbrRecorder::Record& r);

    std::vector<TcpBbrRecorder::Record> m_records; //!< Records of the ACKs
    Time m_entry{Time::Max()};                     //!< Start of the recovery
    Time m_exit{Time::Max()};                      //!< End of the recovery
    uint32_t m_sentOnEntry{0};                     //!< Segments sent on the entry ACK
    uint32_t m_cWndBefore{0};                      //!< Largest window before the recovery
};

void
TcpBbrRecoveryTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    TcpBbrRecorder::SetRecordCallback(MakeCallback(&TcpBbrRecoveryTest::Recorded, this));
    SetCongestionControl(TcpBbrRecorder::GetTypeId());
    // The connection is established long before the data flows: no PROBE_RTT
    Config::SetDefault("ns3::TcpBbr::RttWindowLength", TimeValue(Seconds(100)));
    SetAppPktSize(500);
    SetAppPktCount(400);
    SetAppPktInterval(MicroSeconds(10));
    SetPropagationDelay(MilliSeconds(50));
}

void
TcpBbrRecoveryTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetInitialCwnd(SENDER, 10);
    // The segments carry the timestamp option on top of the data size the
    // test expects
    SetSegmentSize(SENDER, GetSegSize(SENDER) + TcpSocketBase::GetSegmentOptionsSize());
}

Ptr<TcpSocketMsgBase>
TcpBbrRecoveryTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("MinRto", TimeValue(Seconds(1)));
    return socket;
}

Ptr<ErrorModel>
TcpBbrRecoveryTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    errorModel->AddSeqToKill(SequenceNumber32(10001));
    return errorModel;
}

void
TcpBbrRecoveryTest::Recorded(const TcpBbrRecorder::Record& r)
{
    m_records.push_back(r);
}

void
TcpBbrRecoveryTest::Tx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who != SENDER || p->GetSize() == 0)
    {
        return;
    }
    if (Simulator::Now() == m_entry)
    {
        m_sentOnEntry++;
    }
    if (m_entry == Time::Max())
    {
        m_cWndBefore = std::max(m_cWndBefore, GetTcb(SENDER)->m_cWnd.Get());
    }
}

void
TcpBbrRecoveryTest::CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                                   const TcpSocketState::TcpCongState_t newValue)
{
    if (newValue == TcpSocketState::CA_RECOVERY && m_entry == Time::Max())
    {
        m_entry = Simulator::Now();
    }
    if (oldValue == TcpSocketState::CA_RECOVERY && newValue == TcpSocketState::CA_OPEN)
    {
        m_exit = Simulator::Now();
    }
}

void
TcpBbrRecoveryTest::FinalChecks()
{
    TcpBbrRecorder::SetRecordCallback(TcpBbrRecorder::RecordCb());
    Config::SetDefault("ns3::TcpBbr::RttWindowLength", TimeValue(Seconds(10)));
    NS_TEST_ASSERT_MSG_NE(m_entry, Time::Max(), "No fast recovery");
    NS_TEST_ASSERT_MSG_NE(m_exit, Time::Max(), "The fast recovery did not end");
    NS_TEST_ASSERT_MSG_EQ(m_sentOnEntry, 1, "Not only the retransmission sent on the entry ACK");
    Time minRtt = MilliSeconds(100);
    uint32_t prevCwnd = 0;
    for (const auto& r : m_records)
    {
        if (r.time > m_entry && r.time < m_entry + minRtt && r.state == TcpSocketState::CA_RECOVERY)
        {
            NS_TEST_ASSERT_MSG_LT_OR_EQ(r.cWnd,
                                        r.inFlight + r.ackedSacked,
                                        "The window grew in the first round of the recovery");
        }
        if (r.time == m_exit)
        {
            NS_TEST_ASSERT_MSG_EQ(r.bytesLoss, 0, "The end of the recovery reported a loss");
            // The pre-recovery window may be restored, plus at most the
            // data delivered
            NS_TEST_ASSERT_MSG_LT_OR_EQ(r.cWnd,
                                        std::max(prevCwnd, m_cWndBefore) + r.ackedSacked,
                                        "The window jumped at the end of the recovery");
        }
        prevCwnd = r.cWnd;
    }
}

/**
 * @ingroup internet-test
 *
 * @brief Check the LinuxRecovery attribute of TcpBbr.
 *
 * Three segments are lost so that, with the RecoveryReentry attribute of
 * TcpSocketBase, the ACK which ends a fast recovery starts another one (see
 * TcpRecoveryReentryTest). With LinuxRecovery, as Linux, TcpBbr sees the
 * congestion state once per ACK (bbr_set_cwnd_to_recover_or_restore()): it
 * sees a single recovery, whose packet conservation ended long before.
 * Without it, TcpBbr sees the recovery end and another start, with a new
 * packet conservation round.
 */
class TcpBbrLinuxRecoveryTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param desc Test description.
     * @param linux Value of the LinuxRecovery attribute.
     */
    TcpBbrLinuxRecoveryTest(const std::string& desc, bool linux)
        : TcpGeneralTest(desc),
          m_linux(linux)
    {
    }

  protected:
    void ConfigureEnvironment() override;
    void ConfigureProperties() override;
    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                        const TcpSocketState::TcpCongState_t newValue) override;
    void FinalChecks() override;

  private:
    /**
     * @brief Receive a record of the congestion control.
     * @param r The record.
     */
    void Recorded(const TcpBbrRecorder::Record& r);

    bool m_linux;                                  //!< Value of the LinuxRecovery attribute
    std::vector<TcpBbrRecorder::Record> m_records; //!< Records of the ACKs
    Time m_firstExit{Time::Max()};                 //!< End of the first recovery
    Time m_reentry{Time::Max()};                   //!< Start of the second recovery
};

void
TcpBbrLinuxRecoveryTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    TcpBbrRecorder::SetRecordCallback(MakeCallback(&TcpBbrLinuxRecoveryTest::Recorded, this));
    SetCongestionControl(TcpBbrRecorder::GetTypeId());
    // The connection is established long before the data flows: no PROBE_RTT
    Config::SetDefault("ns3::TcpBbr::RttWindowLength", TimeValue(Seconds(100)));
    SetAppPktSize(500);
    SetAppPktCount(200);
    SetAppPktInterval(MicroSeconds(10));
    SetPropagationDelay(MilliSeconds(50));
    // The segments and the ACKs arrive one after the other
    Config::SetDefault("ns3::SimpleNetDevice::DataRate", DataRateValue(DataRate("10Mbps")));
    Config::SetDefault("ns3::TcpBbr::LinuxRecovery", BooleanValue(m_linux));
}

void
TcpBbrLinuxRecoveryTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetInitialCwnd(SENDER, 20);
    // The segments carry the timestamp option on top of the data size the
    // test expects
    SetSegmentSize(SENDER, GetSegSize(SENDER) + TcpSocketBase::GetSegmentOptionsSize());
}

Ptr<TcpSocketMsgBase>
TcpBbrLinuxRecoveryTest::CreateSenderSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
    socket->SetAttribute("RecoveryReentry", BooleanValue(true));
    socket->SetAttribute("MinRto", TimeValue(Seconds(2)));
    return socket;
}

Ptr<ErrorModel>
TcpBbrLinuxRecoveryTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    errorModel->AddSeqToKill(SequenceNumber32(501));
    errorModel->AddSeqToKill(SequenceNumber32(9501));
    errorModel->AddSeqToKill(SequenceNumber32(12001));
    return errorModel;
}

void
TcpBbrLinuxRecoveryTest::Recorded(const TcpBbrRecorder::Record& r)
{
    m_records.push_back(r);
}

void
TcpBbrLinuxRecoveryTest::CongStateTrace(const TcpSocketState::TcpCongState_t oldValue,
                                        const TcpSocketState::TcpCongState_t newValue)
{
    NS_LOG_DEBUG(Simulator::Now().As(Time::MS)
                 << " " << TcpSocketState::TcpCongStateName[oldValue] << " -> "
                 << TcpSocketState::TcpCongStateName[newValue]);
    if (oldValue == TcpSocketState::CA_RECOVERY && m_firstExit == Time::Max())
    {
        m_firstExit = Simulator::Now();
    }
    if (newValue == TcpSocketState::CA_RECOVERY && m_firstExit != Time::Max() &&
        m_reentry == Time::Max())
    {
        m_reentry = Simulator::Now();
    }
}

void
TcpBbrLinuxRecoveryTest::FinalChecks()
{
    TcpBbrRecorder::SetRecordCallback(TcpBbrRecorder::RecordCb());
    Config::SetDefault("ns3::TcpBbr::RttWindowLength", TimeValue(Seconds(10)));
    Config::SetDefault("ns3::TcpBbr::LinuxRecovery", BooleanValue(false));
    Config::SetDefault("ns3::SimpleNetDevice::DataRate", DataRateValue(DataRate(0)));
    NS_TEST_ASSERT_MSG_NE(m_reentry, Time::Max(), "No second recovery");
    NS_TEST_ASSERT_MSG_EQ(m_reentry, m_firstExit, "The recovery did not start again at once");
    bool conservationAtReentry = false;
    for (const auto& r : m_records)
    {
        NS_LOG_DEBUG("rec " << m_linux << " " << r.time.As(Time::MS) << " state " << r.state
                            << " cwnd " << r.cWnd << " inflight " << r.inFlight << " acked "
                            << r.ackedSacked << " cons " << r.conservation);
        if (r.time == m_reentry && r.inFlight > 0)
        {
            conservationAtReentry = r.conservation;
        }
    }
    if (m_linux)
    {
        NS_TEST_ASSERT_MSG_EQ(conservationAtReentry, false, "A new packet conservation started");
    }
    else
    {
        NS_TEST_ASSERT_MSG_EQ(conservationAtReentry, true, "No new packet conservation started");
    }
}

/**
 * @ingroup internet-test
 *
 * @brief TestSuite for the recovery of TcpBbr
 */
class TcpBbrRecoveryTestSuite : public TestSuite
{
  public:
    TcpBbrRecoveryTestSuite()
        : TestSuite("tcp-bbr-recovery", Type::UNIT)
    {
        AddTestCase(new TcpBbrRecoveryTest("BBR window through a fast recovery"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpBbrLinuxRecoveryTest("BBR sees one recovery with LinuxRecovery", true),
                    TestCase::Duration::QUICK);
        AddTestCase(
            new TcpBbrLinuxRecoveryTest("BBR sees two recoveries without LinuxRecovery", false),
            TestCase::Duration::QUICK);
    }
};

static TcpBbrRecoveryTestSuite
    g_tcpBbrRecoveryTestSuite; //!< Static variable for test initialization
