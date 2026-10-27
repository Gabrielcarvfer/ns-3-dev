/*
 * Copyright (c) 2026, Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/boolean.h"
#include "ns3/channel-condition-model.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/mobility-building-info.h"
#include "ns3/node-container.h"
#include "ns3/node.h"
#include "ns3/object-factory.h"
#include "ns3/pointer.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/three-gpp-channel-model.h"
#include "ns3/three-gpp-v2v-channel-condition-model.h"
#include "ns3/three-gpp-v2v-propagation-loss-model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

/**
 * @file
 * @ingroup spectrum-tests
 *
 * Calibration of the V2V channel of ThreeGppChannelModel and of the V2V
 * propagation loss and channel condition models against 3GPP TR 37.885.
 */

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ThreeGppV2vChannelCalibrationTest");

namespace
{

/// Canonical LSP order of TR 38.901 Sec. 7.5 step 4: [SF, K, DS, ASD, ASA, ZSD, ZSA].
enum Lsp : uint8_t
{
    SF = 0,
    K = 1,
    DS = 2,
    ASD = 3,
    ASA = 4,
    ZSD = 5,
    ZSA = 6,
    NUM_LSPS = 7
};

/// Names of the LSPs in canonical order.
const std::array<std::string, NUM_LSPS> kLspNames{"SF", "K", "DS", "ASD", "ASA", "ZSD", "ZSA"};

/// Link state of a V2V link (TR 37.885 Sec. 6.2).
enum class V2vState
{
    LOS,
    NLOS, ///< LOS path blocked by buildings
    NLOSV ///< LOS path blocked by vehicles
};

/**
 * @brief Name of a V2V link state.
 * @param state the link state
 * @return the name
 */
std::string
StateName(V2vState state)
{
    switch (state)
    {
    case V2vState::LOS:
        return "LOS";
    case V2vState::NLOS:
        return "NLOS";
    default:
        return "NLOSv";
    }
}

/**
 * Reference distribution of the LSPs of a V2V link, from TR 37.885 Table
 * 6.2.3-1: mean and standard deviation of log10(DS/1s), log10(ASD/1deg),
 * log10(ASA/1deg), log10(ZSD/1deg), log10(ZSA/1deg) and of K in dB, the
 * cross-correlations between them (SF excluded, since it is drawn by the
 * propagation loss model) and their correlation distances.
 */
struct LspReference
{
    std::array<double, NUM_LSPS> mu{};    ///< mean in the log (dB for K) domain
    std::array<double, NUM_LSPS> sigma{}; ///< standard deviation in the log domain
    bool hasK{false};                     ///< whether the K-factor is defined
    /// cross-correlations {lsp a, lsp b, value}, SF included
    std::vector<std::tuple<Lsp, Lsp, double>> corr;
    /// correlation distances in the horizontal plane in m (0 when N/A)
    std::array<double, NUM_LSPS> corrDist{};
};

/// Cross-correlations of the LOS and NLOSv columns of TR 37.885 Table 6.2.3-1.
const std::vector<std::tuple<Lsp, Lsp, double>> kLosCorr{
    {ASD, DS, 0.5},  {ASA, DS, 0.8},  {ASA, SF, -0.4}, {ASD, SF, -0.5}, {DS, SF, -0.4},
    {ASD, ASA, 0.4}, {ASD, K, -0.2},  {ASA, K, -0.3},  {DS, K, -0.7},   {SF, K, 0.5},
    {ZSD, SF, 0},    {ZSA, SF, 0},    {ZSD, K, 0},     {ZSA, K, 0},     {ZSD, DS, 0},
    {ZSA, DS, 0.2},  {ZSD, ASD, 0.5}, {ZSA, ASD, 0.3}, {ZSD, ASA, 0},   {ZSA, ASA, 0},
    {ZSD, ZSA, 0},
};

/// Cross-correlations of the NLOS column of TR 37.885 Table 6.2.3-1.
const std::vector<std::tuple<Lsp, Lsp, double>> kNlosCorr{
    {ASD, DS, 0},
    {ASA, DS, 0.4},
    {ASA, SF, -0.4},
    {ASD, SF, 0},
    {DS, SF, -0.7},
    {ASD, ASA, 0},
    {ZSD, SF, 0},
    {ZSA, SF, 0},
    {ZSD, DS, -0.5},
    {ZSA, DS, 0},
    {ZSD, ASD, 0.5},
    {ZSA, ASD, 0.5},
    {ZSD, ASA, 0},
    {ZSA, ASA, 0.2},
    {ZSD, ZSA, 0},
};

/**
 * @brief Reference LSP distribution of TR 37.885 Table 6.2.3-1 for a scenario
 *        and link state.
 *
 * The values are typed from the TR, not read from the model under test.
 *
 * @param scenario "V2V-Urban" or "V2V-Highway"
 * @param state the link state
 * @param fcGHz the carrier frequency in GHz
 * @return the reference distribution
 */
LspReference
GetSpecLsps(const std::string& scenario, V2vState state, double fcGHz)
{
    const double lf = std::log10(1 + fcGHz);
    LspReference r;
    // [SF, K, DS, ASD, ASA, ZSD, ZSA]; the SF entries are not used
    if (state == V2vState::NLOS)
    {
        NS_ABORT_MSG_IF(scenario != "V2V-Urban", "TR 37.885 has no NLOS state for the highway");
        r.mu = {0,
                0,
                -0.3 * lf - 7,
                -0.08 * lf + 1.81,
                -0.08 * lf + 1.81,
                -0.04 * lf + 0.92,
                -0.04 * lf + 0.92};
        r.sigma =
            {0, 0, 0.28, 0.05 * lf + 0.3, 0.05 * lf + 0.3, -0.07 * lf + 0.41, -0.07 * lf + 0.41};
        r.corr = kNlosCorr;
        r.corrDist = {13, 0, 10, 10, 9, 10, 10};
        return r;
    }
    r.hasK = true;
    r.corr = kLosCorr;
    if (state == V2vState::LOS)
    {
        r.corrDist = {10, 15, 7, 8, 8, 12, 12};
        r.mu[ZSD] = -0.1 * lf + 0.73;
        r.sigma[ZSD] = -0.04 * lf + 0.34;
        if (scenario == "V2V-Urban")
        {
            r.mu[DS] = -0.2 * lf - 7.5;
            r.sigma[DS] = 0.1;
            r.mu[ASD] = -0.1 * lf + 1.6;
            r.sigma[ASD] = 0.1;
            r.mu[K] = 3.48;
            r.sigma[K] = 2;
        }
        else
        {
            r.mu[DS] = -8.3;
            r.sigma[DS] = 0.2;
            r.mu[ASD] = 1.4;
            r.sigma[ASD] = 0.1;
            r.mu[K] = 9;
            r.sigma[K] = 3.5;
        }
    }
    else
    {
        r.corrDist = {13, 0, 10, 10, 9, 10, 10};
        r.mu[ZSD] = -0.04 * lf + 0.92;
        r.sigma[ZSD] = -0.07 * lf + 0.41;
        r.mu[K] = 0;
        r.sigma[K] = 4.5;
        if (scenario == "V2V-Urban")
        {
            r.mu[DS] = -0.4 * lf - 7;
            r.sigma[DS] = 0.1;
            r.mu[ASD] = -0.1 * lf + 1.7;
            r.sigma[ASD] = 0.1;
        }
        else
        {
            r.mu[DS] = -8.3;
            r.sigma[DS] = 0.3;
            r.mu[ASD] = 1.5;
            r.sigma[ASD] = 0.1;
        }
    }
    // ASA follows ASD, and ZSA follows ZSD, in every column of the table
    r.mu[ASA] = r.mu[ASD];
    r.sigma[ASA] = r.sigma[ASD];
    r.mu[ZSA] = r.mu[ZSD];
    r.sigma[ZSA] = r.sigma[ZSD];
    return r;
}

/**
 * Upper clip of the angular spreads of TR 38.901 Sec. 7.5 step 4.
 *
 * @param lsp the large scale parameter
 * @return the clip of the parameter in log10 of degrees
 */
double
LogClip(Lsp lsp)
{
    switch (lsp)
    {
    case ASD:
    case ASA:
        return std::log10(104.0);
    case ZSD:
    case ZSA:
        return std::log10(52.0);
    default:
        return std::numeric_limits<double>::infinity();
    }
}

/**
 * @brief Kolmogorov-Smirnov statistic of samples against a normal distribution
 *        censored at an upper clip (all the mass above the clip sits on it).
 *
 * @param samples the samples, sorted in place
 * @param mu the mean of the normal distribution
 * @param sigma the standard deviation of the normal distribution
 * @param clip the upper clip
 * @return the maximum distance between the empirical and the reference CDFs
 */
double
KsStatistic(std::vector<double>& samples, double mu, double sigma, double clip)
{
    std::sort(samples.begin(), samples.end());
    const auto n = static_cast<double>(samples.size());
    double d = 0;
    for (size_t i = 0; i < samples.size(); i++)
    {
        const double x = samples[i];
        const double ref = x >= clip - 1e-12 ? 0.5 * std::erfc(-(clip - mu) / (sigma * M_SQRT2))
                                             : 0.5 * std::erfc(-(x - mu) / (sigma * M_SQRT2));
        d = std::max({d, std::abs((i + 1) / n - ref), std::abs(i / n - ref)});
        if (x >= clip - 1e-12)
        {
            break;
        }
    }
    return d;
}

/**
 * @brief Pearson correlation of two sample vectors.
 * @param a the first samples
 * @param b the second samples
 * @return the sample correlation
 */
double
Correlation(const std::vector<double>& a, const std::vector<double>& b)
{
    const auto n = static_cast<double>(a.size());
    double ma = 0;
    double mb = 0;
    for (size_t i = 0; i < a.size(); i++)
    {
        ma += a[i] / n;
        mb += b[i] / n;
    }
    double sab = 0;
    double saa = 0;
    double sbb = 0;
    for (size_t i = 0; i < a.size(); i++)
    {
        sab += (a[i] - ma) * (b[i] - mb);
        saa += (a[i] - ma) * (a[i] - ma);
        sbb += (b[i] - mb) * (b[i] - mb);
    }
    return sab / std::sqrt(saa * sbb);
}

/**
 * @brief Mean and sample standard deviation.
 * @param x the samples
 * @return the mean and the standard deviation
 */
std::pair<double, double>
MeanStd(const std::vector<double>& x)
{
    double mean = 0;
    for (double v : x)
    {
        mean += v / x.size();
    }
    double var = 0;
    for (double v : x)
    {
        var += (v - mean) * (v - mean) / (x.size() - 1);
    }
    return {mean, std::sqrt(var)};
}

/**
 * @brief Correlation of two jointly normal variables after clipping each at an
 *        upper bound, estimated by Monte Carlo with a fixed-seed generator.
 *
 * @param muA mean of the first variable
 * @param sigmaA standard deviation of the first variable
 * @param clipA upper clip of the first variable
 * @param muB mean of the second variable
 * @param sigmaB standard deviation of the second variable
 * @param clipB upper clip of the second variable
 * @param rho correlation of the unclipped variables
 * @return the correlation of the clipped variables
 */
double
ClippedCorrelation(double muA,
                   double sigmaA,
                   double clipA,
                   double muB,
                   double sigmaB,
                   double clipB,
                   double rho)
{
    constexpr uint32_t numDraws = 400000;
    std::mt19937_64 gen(12345);
    std::normal_distribution<double> normal;
    std::vector<double> a(numDraws);
    std::vector<double> b(numDraws);
    for (uint32_t i = 0; i < numDraws; i++)
    {
        const double z1 = normal(gen);
        const double z2 = normal(gen);
        a[i] = std::min(muA + sigmaA * z1, clipA);
        b[i] = std::min(muB + sigmaB * (rho * z1 + std::sqrt(1 - rho * rho) * z2), clipB);
    }
    return Correlation(a, b);
}

/**
 * Channel model exposing the LSP generation of TR 38.901 Sec. 7.5 step 4.
 */
class LspProbeChannelModel : public ThreeGppChannelModel
{
  public:
    using ThreeGppChannelModel::LargeScaleParameters;

    /**
     * @brief Draw the LSPs of a link.
     * @param cond the channel condition
     * @param table the parameter table of the link
     * @param site the site mobility model
     * @param terminal the terminal mobility model
     * @return the LSPs
     */
    LargeScaleParameters Draw(Ptr<const ChannelCondition> cond,
                              Ptr<const ParamsTable> table,
                              Ptr<const MobilityModel> site,
                              Ptr<const MobilityModel> terminal) const
    {
        return GenerateLSPs(cond, table, site, terminal);
    }
};

/**
 * Channel condition model returning the same condition for every link.
 */
class FixedConditionModel : public ChannelConditionModel
{
  public:
    Ptr<ChannelCondition> m_cond{CreateObject<ChannelCondition>()}; ///< the condition

    Ptr<ChannelCondition> GetChannelCondition(Ptr<const MobilityModel> a,
                                              Ptr<const MobilityModel> b) const override
    {
        return m_cond;
    }

    int64_t AssignStreams(int64_t stream) override
    {
        return 0;
    }
};

/**
 * @brief The ns-3 channel condition value of a V2V link state.
 * @param state the link state
 * @return the LOS condition value
 */
ChannelCondition::LosConditionValue
LosValue(V2vState state)
{
    switch (state)
    {
    case V2vState::LOS:
        return ChannelCondition::LOS;
    case V2vState::NLOS:
        return ChannelCondition::NLOS;
    default:
        return ChannelCondition::NLOSv;
    }
}

/**
 * The link states of a scenario: the highway has no NLOS state (TR 37.885 Sec. 6.2).
 *
 * @param scenario the ThreeGppChannelModel scenario
 * @return the link states of the scenario
 */
std::vector<V2vState>
States(const std::string& scenario)
{
    if (scenario == "V2V-Urban")
    {
        return {V2vState::LOS, V2vState::NLOS, V2vState::NLOSV};
    }
    return {V2vState::LOS, V2vState::NLOSV};
}

} // namespace

/**
 * @ingroup spectrum-tests
 *
 * Test that the large-scale parameters drawn by ThreeGppChannelModel for V2V
 * links follow TR 37.885 Table 6.2.3-1: log-normal DS and angular spreads and
 * normal K-factor with the tabulated means and standard deviations (typed from
 * the TR, not read from the model), the angular spreads clipped at 104 and 52
 * degrees as in TR 38.901, and the tabulated cross-correlations. NLOSv links
 * follow the LOS procedure (TR 37.885 Sec. 6.2.3), hence their K-factor. The
 * marginals are checked with a Kolmogorov-Smirnov test and the
 * cross-correlations against the sampling uncertainty, for every link state of
 * the urban and highway scenarios, at the 5.9, 30 and 63 GHz carriers of
 * Tables 6.1.1-1 and 6.1.1-2, with and without the inter-UE spatial
 * consistency.
 */
class ThreeGppV2vLspDistributionTestCase : public TestCase
{
  public:
    ThreeGppV2vLspDistributionTestCase()
        : TestCase("TR 37.885 V2V LSP distributions")
    {
    }

  private:
    void DoRun() override;

    /**
     * @brief Check the LSPs of one scenario, frequency and link state.
     * @param scenario the ThreeGppChannelModel scenario
     * @param fcGHz the carrier frequency in GHz
     * @param spatialConsistency whether the inter-UE spatial consistency is enabled
     * @param state the link state
     */
    void CheckState(const std::string& scenario,
                    double fcGHz,
                    bool spatialConsistency,
                    V2vState state);

    NodeContainer m_sites;        ///< one site per sample, for independent fields
    Ptr<MobilityModel> m_termMob; ///< the terminal mobility model

    static constexpr uint32_t NUM_SAMPLES = 10000; ///< number of links per case
};

void
ThreeGppV2vLspDistributionTestCase::CheckState(const std::string& scenario,
                                               double fcGHz,
                                               bool spatialConsistency,
                                               V2vState state)
{
    auto cond = CreateObject<ChannelCondition>(LosValue(state), ChannelCondition::O2O);
    std::ostringstream label;
    label << scenario << " " << fcGHz << " GHz " << StateName(state)
          << (spatialConsistency ? ", spatially consistent" : "");

    Ptr<ChannelConditionModel> condModel = CreateObject<AlwaysLosChannelConditionModel>();
    condModel->SetAttribute("InterUeSpatialConsistency", BooleanValue(spatialConsistency));
    auto model = CreateObject<LspProbeChannelModel>();
    model->SetAttribute("Scenario", StringValue(scenario));
    model->SetAttribute("Frequency", DoubleValue(fcGHz * 1e9));
    model->SetAttribute("ChannelConditionModel", PointerValue(condModel));
    model->AssignStreams(1);

    const Ptr<const MobilityModel> site0 = m_sites.Get(0)->GetObject<MobilityModel>();
    const auto table = model->GetThreeGppTable(site0, m_termMob, cond);

    std::array<std::vector<double>, NUM_LSPS> samples;
    for (uint32_t i = 0; i < NUM_SAMPLES; i++)
    {
        // With the spatial consistency every sample needs its own site, so that
        // the fields of the samples are independent.
        const auto lsps =
            model->Draw(cond,
                        table,
                        m_sites.Get(spatialConsistency ? i : 0)->GetObject<MobilityModel>(),
                        m_termMob);
        samples[K].push_back(lsps.kFactor);
        samples[DS].push_back(std::log10(lsps.DS));
        samples[ASD].push_back(std::log10(lsps.ASD));
        samples[ASA].push_back(std::log10(lsps.ASA));
        samples[ZSD].push_back(std::log10(lsps.ZSD));
        samples[ZSA].push_back(std::log10(lsps.ZSA));
    }

    const LspReference ref = GetSpecLsps(scenario, state, fcGHz);

    // Cross-correlations first, as the KS test sorts the samples. The standard
    // error of a sample correlation is (1 - r^2) / sqrt(N).
    for (const auto& [a, b, rho] : ref.corr)
    {
        if (a == SF || b == SF || ((a == K || b == K) && !ref.hasK))
        {
            continue;
        }
        const double expected = ClippedCorrelation(ref.mu[a],
                                                   ref.sigma[a],
                                                   LogClip(a),
                                                   ref.mu[b],
                                                   ref.sigma[b],
                                                   LogClip(b),
                                                   rho);
        const double r = Correlation(samples[a], samples[b]);
        const double tol = 5 * (1 - expected * expected) / std::sqrt(NUM_SAMPLES) + 0.005;
        NS_TEST_EXPECT_MSG_EQ_TOL(r,
                                  expected,
                                  tol,
                                  label.str() << ": correlation " << kLspNames[a] << "-"
                                              << kLspNames[b] << " (Table 6.2.3-1: " << rho
                                              << ", after clipping: " << expected << ")");
    }

    // Kolmogorov-Smirnov critical value for a significance of 1e-6 per test.
    const double ksCritical = std::sqrt(-0.5 * std::log(0.5e-6)) / std::sqrt(NUM_SAMPLES);
    for (Lsp lsp : {K, DS, ASD, ASA, ZSD, ZSA})
    {
        if (lsp == K && !ref.hasK)
        {
            continue;
        }
        auto& x = samples[lsp];
        const auto [mean, std] = MeanStd(x);
        const double d = KsStatistic(x, ref.mu[lsp], ref.sigma[lsp], LogClip(lsp));
        NS_LOG_INFO(label.str() << " " << kLspNames[lsp] << " KS " << d);
        NS_TEST_EXPECT_MSG_LT(d,
                              ksCritical,
                              label.str()
                                  << ": " << kLspNames[lsp] << " sample mean " << mean << " std "
                                  << std << ", TR 37.885 mean " << ref.mu[lsp] << " std "
                                  << ref.sigma[lsp] << " (log10, dB for K), KS distance " << d);
    }
}

void
ThreeGppV2vLspDistributionTestCase::DoRun()
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    m_sites.Create(NUM_SAMPLES);
    for (auto it = m_sites.Begin(); it != m_sites.End(); ++it)
    {
        auto mob = CreateObject<ConstantPositionMobilityModel>();
        mob->SetPosition(Vector(0, 0, 1.6));
        (*it)->AggregateObject(mob);
    }
    Ptr<Node> terminal = CreateObject<Node>();
    m_termMob = CreateObject<ConstantPositionMobilityModel>();
    m_termMob->SetPosition(Vector(100, 0, 1.6));
    terminal->AggregateObject(m_termMob);

    for (bool spatialConsistency : {false, true})
    {
        for (const std::string scenario : {"V2V-Urban", "V2V-Highway"})
        {
            for (double fcGHz : {5.9, 30.0, 63.0})
            {
                for (auto state : States(scenario))
                {
                    CheckState(scenario, fcGHz, spatialConsistency, state);
                }
            }
        }
    }
    Simulator::Destroy();
}

/**
 * @ingroup spectrum-tests
 *
 * Test the V2V propagation loss models against TR 37.885 Sec. 6.2.1 and 6.2.2:
 * the path loss of Table 6.2.1-1 per scenario and state, the shadow fading
 * standard deviation (3 dB for LOS and NLOSv, 4 dB for NLOS), and the
 * additional vehicle blockage loss of NLOSv links, whose mean and standard
 * deviation are 9 + max(0, 15 log10(d) - 41) and 4.5 dB when both antennas are
 * below the blocker, and 5 + max(0, 15 log10(d) - 41) and 4 dB when only one
 * is.
 */
class ThreeGppV2vPathlossTestCase : public TestCase
{
  public:
    ThreeGppV2vPathlossTestCase()
        : TestCase("TR 37.885 V2V path loss, shadow fading and NLOSv blockage loss")
    {
    }

  private:
    void DoRun() override;
};

void
ThreeGppV2vPathlossTestCase::DoRun()
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    const uint32_t numLinks = 4000;
    for (const std::string scenario : {"V2V-Urban", "V2V-Highway"})
    {
        const bool urban = scenario == "V2V-Urban";
        for (double fcGHz : {5.9, 63.0})
        {
            for (auto state : States(scenario))
            {
                for (double d : {50.0, 800.0})
                {
                    // NLOSv blockage: both antennas below the blocker (case 2) with
                    // type 1/2 vehicles only, or one above it (case 3) with trucks only
                    for (bool bothBelow : {true, false})
                    {
                        if (state != V2vState::NLOSV && !bothBelow)
                        {
                            continue;
                        }
                        auto condModel = CreateObject<FixedConditionModel>();
                        condModel->m_cond->SetLosCondition(LosValue(state));
                        condModel->m_cond->SetO2iCondition(ChannelCondition::O2O);
                        ObjectFactory factory(urban
                                                  ? "ns3::ThreeGppV2vUrbanPropagationLossModel"
                                                  : "ns3::ThreeGppV2vHighwayPropagationLossModel");
                        factory.Set("Frequency", DoubleValue(fcGHz * 1e9));
                        factory.Set("ChannelConditionModel", PointerValue(condModel));
                        factory.Set("PercType3Vehicles", DoubleValue(bothBelow ? 0 : 100));
                        auto withSf = factory.Create<ThreeGppPropagationLossModel>();
                        withSf->AssignStreams(1);
                        factory.Set("ShadowingEnabled", BooleanValue(false));
                        auto noSf = factory.Create<ThreeGppPropagationLossModel>();
                        // the same streams, so that both models draw the same
                        // NLOSv blockage losses and their difference is the SF
                        noSf->AssignStreams(1);

                        const double hA = 1.5;
                        const double hB = bothBelow ? 1.5 : 3.5;
                        const double d3D = std::sqrt(d * d + (hA - hB) * (hA - hB));
                        const double lf = std::log10(fcGHz);
                        const double losPl = urban ? 38.77 + 16.7 * std::log10(d3D) + 18.2 * lf
                                                   : 32.4 + 20 * std::log10(d3D) + 20 * lf;
                        const double pl = state == V2vState::NLOS
                                              ? 36.85 + 30 * std::log10(d3D) + 18.9 * lf
                                              : losPl;
                        const double sfStd = state == V2vState::NLOS ? 4 : 3;

                        NodeContainer nodes;
                        nodes.Create(2 * numLinks);
                        std::vector<double> sf(numLinks);
                        std::vector<double> excess(numLinks);
                        for (uint32_t i = 0; i < numLinks; i++)
                        {
                            auto a = CreateObject<ConstantPositionMobilityModel>();
                            auto b = CreateObject<ConstantPositionMobilityModel>();
                            nodes.Get(2 * i)->AggregateObject(a);
                            nodes.Get(2 * i + 1)->AggregateObject(b);
                            a->SetPosition(Vector(0, 0, hA));
                            b->SetPosition(Vector(d, 0, hB));
                            const double lossNoSf = -noSf->CalcRxPower(0, a, b);
                            sf[i] = withSf->CalcRxPower(0, a, b) + lossNoSf;
                            excess[i] = lossNoSf - pl;
                        }
                        std::ostringstream label;
                        label << scenario << " " << fcGHz << " GHz " << StateName(state) << " " << d
                              << " m";
                        const auto [sfMean, sfStdEst] = MeanStd(sf);
                        // The standard error of a sample standard deviation is
                        // sigma / sqrt(2 n); tolerances are 4 standard errors.
                        NS_TEST_EXPECT_MSG_EQ_TOL(sfStdEst,
                                                  sfStd,
                                                  4 * sfStd / std::sqrt(2.0 * numLinks),
                                                  label.str() << ": shadow fading std");
                        const auto [exMean, exStd] = MeanStd(excess);
                        if (state != V2vState::NLOSV)
                        {
                            NS_TEST_EXPECT_MSG_EQ_TOL(exMean,
                                                      0,
                                                      1e-9,
                                                      label.str() << ": path loss");
                            continue;
                        }
                        // NLOSv blockage of TR 37.885 Sec. 6.2.1; the max(0, .)
                        // has a negligible effect on the mean and deviation.
                        const double extra = std::max(0.0, 15 * std::log10(d) - 41);
                        const double mu = (bothBelow ? 9 : 5) + extra;
                        const double sigma = bothBelow ? 4.5 : 4;
                        label << (bothBelow ? ", both antennas below the blocker"
                                            : ", one antenna above the blocker");
                        NS_TEST_EXPECT_MSG_EQ_TOL(exMean,
                                                  mu,
                                                  4 * sigma / std::sqrt(numLinks) + 0.1,
                                                  label.str() << ": NLOSv blockage loss mean");
                        NS_TEST_EXPECT_MSG_EQ_TOL(exStd,
                                                  sigma,
                                                  4 * sigma / std::sqrt(2.0 * numLinks) + 0.1,
                                                  label.str() << ": NLOSv blockage loss std");
                    }
                }
            }
        }
    }
    Simulator::Destroy();
}

/**
 * @ingroup spectrum-tests
 *
 * Test the empirical LOS probability of the V2V channel condition models,
 * without buildings, against TR 37.885 Table 6.2-1; links that are not LOS are
 * NLOSv, since without buildings both vehicles are in the same street.
 */
class ThreeGppV2vLosProbabilityTestCase : public TestCase
{
  public:
    ThreeGppV2vLosProbabilityTestCase()
        : TestCase("TR 37.885 V2V LOS probability")
    {
    }

  private:
    void DoRun() override;
};

void
ThreeGppV2vLosProbabilityTestCase::DoRun()
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    const uint32_t numLinks = 4000;
    for (bool urban : {true, false})
    {
        for (double d : {10.0, 50.0, 100.0, 200.0, 400.0, 475.0, 600.0, 900.0})
        {
            double pLos;
            if (urban)
            {
                pLos = std::min(1.0, 1.05 * std::exp(-0.0114 * d));
            }
            else if (d <= 475)
            {
                pLos = std::min(1.0, 2.1013e-6 * d * d - 0.002 * d + 1.0193);
            }
            else
            {
                pLos = std::max(0.0, 0.54 - 0.001 * (d - 475));
            }
            Ptr<ChannelConditionModel> condModel;
            if (urban)
            {
                condModel = CreateObject<ThreeGppV2vUrbanChannelConditionModel>();
            }
            else
            {
                condModel = CreateObject<ThreeGppV2vHighwayChannelConditionModel>();
            }
            condModel->AssignStreams(static_cast<int64_t>(d) * 10 + (urban ? 1 : 0));
            NodeContainer nodes;
            nodes.Create(2 * numLinks);
            uint32_t los = 0;
            uint32_t nlosv = 0;
            for (uint32_t i = 0; i < numLinks; i++)
            {
                auto a = CreateObject<ConstantPositionMobilityModel>();
                auto b = CreateObject<ConstantPositionMobilityModel>();
                nodes.Get(2 * i)->AggregateObject(a);
                nodes.Get(2 * i + 1)->AggregateObject(b);
                a->SetPosition(Vector(0, 0, 1.6));
                b->SetPosition(Vector(d, 0, 1.6));
                a->AggregateObject(CreateObject<MobilityBuildingInfo>());
                b->AggregateObject(CreateObject<MobilityBuildingInfo>());
                const auto cond = condModel->GetChannelCondition(a, b);
                los += cond->IsLos();
                nlosv += cond->IsNlosv();
            }
            const double tol = 4 * std::sqrt(pLos * (1 - pLos) / numLinks) + 1e-9;
            NS_TEST_EXPECT_MSG_EQ_TOL(static_cast<double>(los) / numLinks,
                                      pLos,
                                      tol,
                                      (urban ? "Urban" : "Highway")
                                          << " LOS probability at " << d << " m");
            NS_TEST_EXPECT_MSG_EQ(los + nlosv,
                                  numLinks,
                                  (urban ? "Urban" : "Highway")
                                      << " links neither LOS nor NLOSv at " << d << " m");
        }
    }
    Simulator::Destroy();
}

/**
 * @ingroup spectrum-tests
 *
 * TR 37.885 V2V channel calibration test suite.
 */
class ThreeGppV2vChannelCalibrationTestSuite : public TestSuite
{
  public:
    ThreeGppV2vChannelCalibrationTestSuite();
};

ThreeGppV2vChannelCalibrationTestSuite::ThreeGppV2vChannelCalibrationTestSuite()
    : TestSuite("three-gpp-v2v-channel-calibration", Type::SYSTEM)
{
    AddTestCase(new ThreeGppV2vLosProbabilityTestCase, Duration::QUICK);
    AddTestCase(new ThreeGppV2vPathlossTestCase, Duration::QUICK);
    AddTestCase(new ThreeGppV2vLspDistributionTestCase, Duration::EXTENSIVE);
}

/// Static variable for test initialization
static ThreeGppV2vChannelCalibrationTestSuite g_threeGppV2vChannelCalibrationTestSuite;
