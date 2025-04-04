/*
 * Copyright (c) 2011 The Boeing Company
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Tom Henderson <thomas.r.henderson@boeing.com>
 */
#include "ns3/callback.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/log.h"
#include "ns3/lr-wpan-error-model.h"
#include "ns3/lr-wpan-mac.h"
#include "ns3/lr-wpan-net-device.h"
#include "ns3/mac16-address.h"
#include "ns3/net-device.h"
#include "ns3/node.h"
#include "ns3/packet.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/simulator.h"
#include "ns3/single-model-spectrum-channel.h"
#include "ns3/test.h"

using namespace ns3;
using namespace ns3::lrwpan;

NS_LOG_COMPONENT_DEFINE("lr-wpan-error-model-test");

/**
 * @ingroup lr-wpan-test
 * @ingroup tests
 *
 * @brief LrWpan Error Vs Distance Test
 */
class LrWpanErrorDistanceTestCase : public TestCase
{
  public:
    LrWpanErrorDistanceTestCase();
    ~LrWpanErrorDistanceTestCase() override;

    /**
     * @brief Get the number of received packets.
     * @returns The number of received packets.
     */
    uint32_t GetReceived() const
    {
        return m_received;
    }

  private:
    void DoRun() override;

    /**
     * @brief Function to be called when a packet is received.
     * @param params MCPS params.
     * @param p The packet.
     */
    void Callback(McpsDataIndicationParams params, Ptr<Packet> p);
    uint32_t m_received; //!< The number of received packets.
};

/**
 * @ingroup lr-wpan-test
 * @ingroup tests
 *
 * @brief LrWpan Error model Test
 */
class LrWpanErrorModelTestCase : public TestCase
{
  public:
    LrWpanErrorModelTestCase();
    ~LrWpanErrorModelTestCase() override;

  private:
    void DoRun() override;
};

LrWpanErrorDistanceTestCase::LrWpanErrorDistanceTestCase()
    : TestCase("Test the 802.15.4 error model vs distance"),
      m_received(0)
{
}

LrWpanErrorDistanceTestCase::~LrWpanErrorDistanceTestCase()
{
}

void
LrWpanErrorDistanceTestCase::Callback(McpsDataIndicationParams params, Ptr<Packet> p)
{
    m_received++;
}

void
LrWpanErrorDistanceTestCase::DoRun()
{
    // Set the random seed and run number for this test
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(6);

    Ptr<Node> n0 = CreateObject<Node>();
    Ptr<Node> n1 = CreateObject<Node>();
    Ptr<LrWpanNetDevice> dev0 = CreateObject<LrWpanNetDevice>();
    Ptr<LrWpanNetDevice> dev1 = CreateObject<LrWpanNetDevice>();

    // Make random variable stream assignment deterministic
    dev0->AssignStreams(0);
    dev1->AssignStreams(10);

    dev0->SetAddress(Mac16Address("00:01"));
    dev1->SetAddress(Mac16Address("00:02"));
    Ptr<SingleModelSpectrumChannel> channel = CreateObject<SingleModelSpectrumChannel>();
    Ptr<LogDistancePropagationLossModel> model = CreateObject<LogDistancePropagationLossModel>();
    channel->AddPropagationLossModel(model);
    dev0->SetChannel(channel);
    dev1->SetChannel(channel);
    n0->AddDevice(dev0);
    n1->AddDevice(dev1);
    Ptr<ConstantPositionMobilityModel> mob0 = CreateObject<ConstantPositionMobilityModel>();
    dev0->GetPhy()->SetMobility(mob0);
    Ptr<ConstantPositionMobilityModel> mob1 = CreateObject<ConstantPositionMobilityModel>();
    dev1->GetPhy()->SetMobility(mob1);

    McpsDataIndicationCallback cb0;
    cb0 = MakeCallback(&LrWpanErrorDistanceTestCase::Callback, this);
    dev1->GetMac()->SetMcpsDataIndicationCallback(cb0);

    McpsDataRequestParams params;
    params.m_srcAddrMode = SHORT_ADDR;
    params.m_dstAddrMode = SHORT_ADDR;
    params.m_dstPanId = 0;
    params.m_dstAddr = Mac16Address("00:02");
    params.m_msduHandle = 0;
    params.m_txOptions = 0;

    Ptr<Packet> p;
    mob0->SetPosition(Vector(0, 0, 0));
    mob1->SetPosition(Vector(100, 0, 0));
    for (int i = 0; i < 1000; i++)
    {
        p = Create<Packet>(20);
        Simulator::Schedule(Seconds(i), &LrWpanMac::McpsDataRequest, dev0->GetMac(), params, p);
    }

    Simulator::Run();

    // Test that we received 977 packets out of 1000, at distance of 100 m
    // with default power of 0
    NS_TEST_ASSERT_MSG_EQ(GetReceived(), 977, "Model fails");

    Simulator::Destroy();
}

// ==============================================================================
LrWpanErrorModelTestCase::LrWpanErrorModelTestCase()
    : TestCase("Test the 802.15.4 error model")
{
}

LrWpanErrorModelTestCase::~LrWpanErrorModelTestCase()
{
}

void
LrWpanErrorModelTestCase::DoRun()
{
    Ptr<LrWpanErrorModel> model = CreateObject<LrWpanErrorModel>();

    // Test a few values
    double snr = 5;
    double ber = 1.0 - model->GetChunkSuccessRate(pow(10.0, snr / 10.0), 1);
    NS_TEST_ASSERT_MSG_EQ_TOL(ber, 7.38e-14, 0.01e-14, "Model fails for SNR = " << snr);
    snr = 2;
    ber = 1.0 - model->GetChunkSuccessRate(pow(10.0, snr / 10.0), 1);
    NS_TEST_ASSERT_MSG_EQ_TOL(ber, 5.13e-7, 0.01e-7, "Model fails for SNR = " << snr);
    snr = -1;
    ber = 1.0 - model->GetChunkSuccessRate(pow(10.0, snr / 10.0), 1);
    NS_TEST_ASSERT_MSG_EQ_TOL(ber, 0.00114, 0.00001, "Model fails for SNR = " << snr);
    snr = -4;
    ber = 1.0 - model->GetChunkSuccessRate(pow(10.0, snr / 10.0), 1);
    NS_TEST_ASSERT_MSG_EQ_TOL(ber, 0.0391, 0.0001, "Model fails for SNR = " << snr);
    snr = -7;
    ber = 1.0 - model->GetChunkSuccessRate(pow(10.0, snr / 10.0), 1);
    NS_TEST_ASSERT_MSG_EQ_TOL(ber, 0.175, 0.001, "Model fails for SNR = " << snr);
}

/**
 * @ingroup lr-wpan-test
 * @ingroup tests
 *
 * @brief LrWpan Error model TestSuite
 */
class LrWpanErrorModelTestSuite : public TestSuite
{
  public:
    LrWpanErrorModelTestSuite();
};

LrWpanErrorModelTestSuite::LrWpanErrorModelTestSuite()
    : TestSuite("lr-wpan-error-model", Type::UNIT)
{
    AddTestCase(new LrWpanErrorModelTestCase, TestCase::Duration::QUICK);
    AddTestCase(new LrWpanErrorDistanceTestCase, TestCase::Duration::QUICK);
}

static LrWpanErrorModelTestSuite
    g_lrWpanErrorModelTestSuite; //!< Static variable for test initialization
