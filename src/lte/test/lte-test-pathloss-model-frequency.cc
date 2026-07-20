/*
 * Copyright (c) 2026 Centre Tecnològic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/boolean.h"
#include "ns3/config.h"
#include "ns3/double.h"
#include "ns3/lte-helper.h"
#include "ns3/lte-spectrum-value-helper.h"
#include "ns3/node-container.h"
#include "ns3/mobility-helper.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/simulator.h"
#include "ns3/spectrum-channel.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"

using namespace ns3;

/**
 * @ingroup lte-test
 *
 * @brief Test that the pathloss model frequency is set to the primary
 * component carrier frequency (see issue #1293)
 *
 * All the component carriers share a single downlink (and uplink) spectrum
 * channel and hence a single pathloss model instance. When a
 * frequency-dependent (non spectrum aware) pathloss model is configured in a
 * carrier aggregation scenario, its Frequency attribute must be set to the
 * frequency of the primary component carrier, not to the frequency of
 * whichever carrier is configured last.
 */
class LtePathlossModelFrequencyTestCase : public TestCase
{
  public:
    LtePathlossModelFrequencyTestCase()
        : TestCase("Pathloss model frequency matches the primary component carrier")
    {
    }

    void DoRun() override;
};

void
LtePathlossModelFrequencyTestCase::DoRun()
{
    Config::SetDefault("ns3::LteHelper::UseCa", BooleanValue(true));
    Config::SetDefault("ns3::LteHelper::NumberOfComponentCarriers", UintegerValue(2));
    Config::SetDefault("ns3::LteHelper::EnbComponentCarrierManager",
                       StringValue("ns3::RrComponentCarrierManager"));

    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    // A frequency-dependent pathloss model which is not spectrum aware
    lteHelper->SetPathlossModelType(TypeId::LookupByName("ns3::Cost231PropagationLossModel"));

    NodeContainer enbNodes;
    enbNodes.Create(1);
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(enbNodes);
    lteHelper->InstallEnbDevice(enbNodes);

    // The primary component carrier uses the default DL EARFCN (100)
    const double primaryDlFreq = LteSpectrumValueHelper::GetCarrierFrequency(100);

    Ptr<PropagationLossModel> model =
        lteHelper->GetDownlinkSpectrumChannel()->GetPropagationLossModel();
    NS_TEST_ASSERT_MSG_NE(model, nullptr, "No propagation loss model found");
    DoubleValue frequency;
    model->GetAttribute("Frequency", frequency);
    NS_TEST_ASSERT_MSG_EQ_TOL(frequency.Get(),
                              primaryDlFreq,
                              1.0,
                              "Pathloss model frequency is not the primary carrier frequency");

    Simulator::Destroy();
}

/**
 * @ingroup lte-test
 *
 * @brief Pathloss model frequency in carrier aggregation TestSuite
 */
class LtePathlossModelFrequencyTestSuite : public TestSuite
{
  public:
    LtePathlossModelFrequencyTestSuite()
        : TestSuite("lte-pathloss-model-frequency", Type::UNIT)
    {
        AddTestCase(new LtePathlossModelFrequencyTestCase, TestCase::Duration::QUICK);
    }
};

/// Static variable for test initialization
static LtePathlossModelFrequencyTestSuite g_ltePathlossModelFrequencyTestSuite;
