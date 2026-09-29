/*
 * Copyright (c) 2026, Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/abort.h"
#include "ns3/boolean.h"
#include "ns3/channel-condition-model.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/node-container.h"
#include "ns3/node.h"
#include "ns3/pointer.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/three-gpp-channel-model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

/**
 * @file
 * @ingroup spectrum-tests
 *
 * Calibration of ThreeGppChannelModel against 3GPP TR 38.901.
 */

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ThreeGppChannelCalibrationTest");

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

/// Link state of a reference parameter set.
enum class LinkState
{
    LOS,
    NLOS,
    O2I_LOS, ///< O2I link with a LOS outdoor part
    O2I_NLOS ///< O2I link with a NLOS outdoor part
};

/**
 * Reference distribution of the LSPs of a link, from TR 38.901 Tables 7.5-6 to
 * 7.5-10: mean and standard deviation of log10(DS/1s), log10(ASD/1deg),
 * log10(ASA/1deg), log10(ZSD/1deg), log10(ZSA/1deg) and of K in dB, and the
 * cross-correlations between them (SF excluded, since it is drawn by the
 * propagation loss model).
 */
struct LspReference
{
    std::array<double, NUM_LSPS> mu{};    ///< mean in the log (dB for K) domain
    std::array<double, NUM_LSPS> sigma{}; ///< standard deviation in the log domain
    bool hasK{false};                     ///< whether the K-factor is defined
    /// cross-correlations {lsp a, lsp b, value} of TR 38.901 Table 7.5-6
    std::vector<std::tuple<Lsp, Lsp, double>> corr;
};

/**
 * Geometry and frequency of a reference case.
 */
struct LinkGeometry
{
    double fcGHz; ///< carrier frequency in GHz
    double d2D;   ///< 2D BS-UT distance in m
    double hBs;   ///< BS height in m
    double hUt;   ///< UT height in m
};

/// Cross-correlations of TR 38.901 Table 7.5-6 Part-1 for the O2I columns of UMi and UMa.
const std::vector<std::tuple<Lsp, Lsp, double>> kUmO2iCorr{
    {ASD, DS, 0.4},
    {ASA, DS, 0.4},
    {ASD, ASA, 0},
    {ZSD, DS, -0.6},
    {ZSA, DS, -0.2},
    {ZSD, ASD, -0.2},
    {ZSA, ASD, 0},
    {ZSD, ASA, 0},
    {ZSA, ASA, 0.5},
    {ZSD, ZSA, 0.5},
};

/**
 * @brief Reference LSP distribution of TR 38.901 for a scenario, link state and geometry.
 *
 * @param scenario "UMa", "UMi-StreetCanyon", "RMa" or "InH-OfficeMixed"
 * @param state the link state
 * @param g the link geometry
 * @return the reference distribution
 */
LspReference
GetSpecLsps(const std::string& scenario, LinkState state, const LinkGeometry& g)
{
    LspReference r;
    const bool o2i = state == LinkState::O2I_LOS || state == LinkState::O2I_NLOS;
    const bool losOutdoor = state == LinkState::LOS || state == LinkState::O2I_LOS;
    const double dKm = g.d2D / 1000.0;
    if (scenario == "UMa")
    {
        // Table 7.5-6 Part-1 note 6 and Table 7.5-7 note 4: fc = 6 GHz below 6 GHz.
        const double lf = std::log10(std::max(g.fcGHz, 6.0));
        if (o2i)
        {
            r.mu = {0, 0, -6.62, 1.25, 1.76, 0, 1.01};
            r.sigma = {0, 0, 0.32, 0.42, 0.16, 0, 0.43};
            r.corr = kUmO2iCorr;
        }
        else if (losOutdoor)
        {
            r.mu = {0, 9, -6.955 - 0.0963 * lf, 1.06 + 0.1114 * lf, 1.81, 0, 0.95};
            r.sigma = {0, 3.5, 0.66, 0.28, 0.20, 0, 0.16};
            r.hasK = true;
            r.corr = {{ASD, DS, 0.4},
                      {ASA, DS, 0.8},
                      {ASD, ASA, 0},
                      {ASD, K, 0},
                      {ASA, K, -0.2},
                      {DS, K, -0.4},
                      {ZSD, K, 0},
                      {ZSA, K, 0},
                      {ZSD, DS, -0.2},
                      {ZSA, DS, 0},
                      {ZSD, ASD, 0.5},
                      {ZSA, ASD, 0},
                      {ZSD, ASA, -0.3},
                      {ZSA, ASA, 0.4},
                      {ZSD, ZSA, 0}};
        }
        else
        {
            r.mu = {0,
                    0,
                    -6.28 - 0.204 * lf,
                    1.5 - 0.1144 * lf,
                    2.08 - 0.27 * lf,
                    0,
                    -0.3236 * lf + 1.512};
            r.sigma = {0, 0, 0.39, 0.28, 0.11, 0, 0.16};
            r.corr = {{ASD, DS, 0.4},
                      {ASA, DS, 0.6},
                      {ASD, ASA, 0.4},
                      {ZSD, DS, -0.5},
                      {ZSA, DS, 0},
                      {ZSD, ASD, 0.5},
                      {ZSA, ASD, -0.1},
                      {ZSD, ASA, 0},
                      {ZSA, ASA, 0},
                      {ZSD, ZSA, 0}};
        }
        // Table 7.5-7; note 5: O2I links take the ZSD of their outdoor LOS state.
        if (losOutdoor)
        {
            r.mu[ZSD] = std::max(-0.5, -2.1 * dKm - 0.01 * (g.hUt - 1.5) + 0.75);
            r.sigma[ZSD] = 0.40;
        }
        else
        {
            r.mu[ZSD] = std::max(-0.5, -2.1 * dKm - 0.01 * (g.hUt - 1.5) + 0.9);
            r.sigma[ZSD] = 0.49;
        }
    }
    else if (scenario == "UMi-StreetCanyon")
    {
        // Table 7.5-6 Part-1 note 7: fc = 2 GHz below 2 GHz.
        const double lf1 = std::log10(1 + std::max(g.fcGHz, 2.0));
        if (o2i)
        {
            r.mu = {0, 0, -6.62, 1.25, 1.76, 0, 1.01};
            r.sigma = {0, 0, 0.32, 0.42, 0.16, 0, 0.43};
            r.corr = kUmO2iCorr;
        }
        else if (losOutdoor)
        {
            r.mu = {0,
                    9,
                    -0.24 * lf1 - 7.14,
                    -0.05 * lf1 + 1.21,
                    -0.08 * lf1 + 1.73,
                    0,
                    -0.1 * lf1 + 0.73};
            r.sigma = {0, 5, 0.38, 0.41, 0.014 * lf1 + 0.28, 0, -0.04 * lf1 + 0.34};
            r.hasK = true;
            r.corr = {{ASD, DS, 0.5},
                      {ASA, DS, 0.8},
                      {ASD, ASA, 0.4},
                      {ASD, K, -0.2},
                      {ASA, K, -0.3},
                      {DS, K, -0.7},
                      {ZSD, K, 0},
                      {ZSA, K, 0},
                      {ZSD, DS, 0},
                      {ZSA, DS, 0.2},
                      {ZSD, ASD, 0.5},
                      {ZSA, ASD, 0.3},
                      {ZSD, ASA, 0},
                      {ZSA, ASA, 0},
                      {ZSD, ZSA, 0}};
        }
        else
        {
            r.mu = {0,
                    0,
                    -0.24 * lf1 - 6.83,
                    -0.23 * lf1 + 1.53,
                    -0.08 * lf1 + 1.81,
                    0,
                    -0.04 * lf1 + 0.92};
            r.sigma = {0,
                       0,
                       0.16 * lf1 + 0.28,
                       0.11 * lf1 + 0.33,
                       0.05 * lf1 + 0.3,
                       0,
                       -0.07 * lf1 + 0.41};
            r.corr = {{ASD, DS, 0},
                      {ASA, DS, 0.4},
                      {ASD, ASA, 0},
                      {ZSD, DS, -0.5},
                      {ZSA, DS, 0},
                      {ZSD, ASD, 0.5},
                      {ZSA, ASD, 0.5},
                      {ZSD, ASA, 0},
                      {ZSA, ASA, 0.2},
                      {ZSD, ZSA, 0}};
        }
        // Table 7.5-8; note 5: O2I links take the ZSD of their outdoor LOS state.
        if (losOutdoor)
        {
            r.mu[ZSD] = std::max(-0.21, -14.8 * dKm + 0.01 * std::abs(g.hUt - g.hBs) + 0.83);
        }
        else
        {
            r.mu[ZSD] = std::max(-0.5, -3.1 * dKm + 0.01 * std::max(g.hUt - g.hBs, 0.0) + 0.2);
        }
        r.sigma[ZSD] = 0.35;
    }
    else if (scenario == "RMa")
    {
        if (o2i)
        {
            r.mu = {0, 0, -7.47, 0.67, 1.66, 0, 0.93};
            r.sigma = {0, 0, 0.24, 0.18, 0.21, 0, 0.22};
            r.corr = {{ASD, DS, 0},
                      {ASA, DS, 0},
                      {ASD, ASA, -0.7},
                      {ZSD, DS, 0},
                      {ZSA, DS, 0},
                      {ZSD, ASD, 0.66},
                      {ZSA, ASD, 0.47},
                      {ZSD, ASA, -0.55},
                      {ZSA, ASA, -0.22},
                      {ZSD, ZSA, 0}};
        }
        else if (losOutdoor)
        {
            r.mu = {0, 7, -7.49, 0.90, 1.52, 0, 0.47};
            r.sigma = {0, 4, 0.55, 0.38, 0.24, 0, 0.40};
            r.hasK = true;
            r.corr = {{ASD, DS, 0},
                      {ASA, DS, 0},
                      {ASD, ASA, 0},
                      {ASD, K, 0},
                      {ASA, K, 0},
                      {DS, K, 0},
                      {ZSD, K, 0},
                      {ZSA, K, -0.02},
                      {ZSD, DS, -0.05},
                      {ZSA, DS, 0.27},
                      {ZSD, ASD, 0.73},
                      {ZSA, ASD, -0.14},
                      {ZSD, ASA, -0.20},
                      {ZSA, ASA, 0.24},
                      {ZSD, ZSA, -0.07}};
        }
        else
        {
            r.mu = {0, 0, -7.43, 0.95, 1.52, 0, 0.58};
            r.sigma = {0, 0, 0.48, 0.45, 0.13, 0, 0.37};
            r.corr = {{ASD, DS, -0.4},
                      {ASA, DS, 0},
                      {ASD, ASA, 0},
                      {ZSD, DS, -0.10},
                      {ZSA, DS, -0.40},
                      {ZSD, ASD, 0.42},
                      {ZSA, ASD, -0.27},
                      {ZSD, ASA, -0.18},
                      {ZSA, ASA, 0.26},
                      {ZSD, ZSA, -0.27}};
        }
        // Table 7.5-9: the O2I column equals the NLOS one.
        if (state == LinkState::LOS)
        {
            r.mu[ZSD] = std::max(-1.0, -0.17 * dKm - 0.01 * (g.hUt - 1.5) + 0.22);
            r.sigma[ZSD] = 0.34;
        }
        else
        {
            r.mu[ZSD] = std::max(-1.0, -0.19 * dKm - 0.01 * (g.hUt - 1.5) + 0.28);
            r.sigma[ZSD] = 0.30;
        }
    }
    else
    {
        NS_ABORT_MSG_UNLESS(scenario == "InH-OfficeMixed", "Unknown scenario " << scenario);
        NS_ABORT_MSG_IF(o2i, "Indoor-Office has no O2I links");
        // Table 7.5-6 Part-2 note 6 and Table 7.5-10 note 4: fc = 6 GHz below 6 GHz.
        const double lf1 = std::log10(1 + std::max(g.fcGHz, 6.0));
        if (losOutdoor)
        {
            r.mu = {0,
                    7,
                    -0.01 * lf1 - 7.692,
                    1.60,
                    -0.19 * lf1 + 1.781,
                    -1.43 * lf1 + 2.228,
                    -0.26 * lf1 + 1.44};
            r.sigma =
                {0, 4, 0.18, 0.18, 0.12 * lf1 + 0.119, 0.13 * lf1 + 0.30, -0.04 * lf1 + 0.264};
            r.hasK = true;
            r.corr = {{ASD, DS, 0.6},
                      {ASA, DS, 0.8},
                      {ASD, ASA, 0.4},
                      {ASD, K, 0},
                      {ASA, K, 0},
                      {DS, K, -0.5},
                      {ZSD, K, 0},
                      {ZSA, K, 0.1},
                      {ZSD, DS, 0.1},
                      {ZSA, DS, 0.2},
                      {ZSD, ASD, 0.5},
                      {ZSA, ASD, 0},
                      {ZSD, ASA, 0},
                      {ZSA, ASA, 0.5},
                      {ZSD, ZSA, 0}};
        }
        else
        {
            r.mu =
                {0, 0, -0.28 * lf1 - 7.173, 1.62, -0.11 * lf1 + 1.863, 1.08, -0.15 * lf1 + 1.387};
            r.sigma =
                {0, 0, 0.10 * lf1 + 0.055, 0.25, 0.12 * lf1 + 0.059, 0.36, -0.09 * lf1 + 0.746};
            r.corr = {{ASD, DS, 0.4},
                      {ASA, DS, 0},
                      {ASD, ASA, 0},
                      {ZSD, DS, -0.27},
                      {ZSA, DS, -0.06},
                      {ZSD, ASD, 0.35},
                      {ZSA, ASD, 0.23},
                      {ZSD, ASA, -0.08},
                      {ZSA, ASA, 0.43},
                      {ZSD, ZSA, 0.42}};
        }
    }
    return r;
}

/// Upper clip of the angular spreads of TR 38.901 Sec. 7.5 step 4, in log10 of degrees.
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
        // A sample at the clip carries the censored mass: compare below it only.
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
 * @brief Correlation of two jointly normal variables after clipping each at an
 *        upper bound, estimated by Monte Carlo with a fixed-seed generator.
 *
 * The angular spreads are clipped at 104 and 52 degrees (TR 38.901 Sec. 7.5
 * step 4), which lowers their correlation with the other LSPs below the value
 * of Table 7.5-6 whenever a noticeable fraction of the distribution is clipped.
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

} // namespace

/**
 * @ingroup spectrum-tests
 *
 * Test that the large-scale parameters drawn by ThreeGppChannelModel follow
 * the distributions of TR 38.901 Sec. 7.5 step 4: log-normal DS and angular
 * spreads and normal K-factor, with the means and standard deviations of
 * Tables 7.5-6 to 7.5-10 (typed from the TR, not read from the model), the
 * angular spreads clipped at 104 and 52 degrees, and the cross-correlations of
 * Table 7.5-6. The marginals are checked with a Kolmogorov-Smirnov test and
 * the cross-correlations against the sampling uncertainty, for LOS, NLOS and
 * O2I links (with LOS and NLOS outdoor parts), with and without the inter-UE
 * spatial consistency, whose fields must preserve the distributions.
 */
class ThreeGppLspDistributionTestCase : public TestCase
{
  public:
    /// A scenario and the geometry of its links.
    using Case = std::pair<std::string, LinkGeometry>;

    /**
     * Constructor.
     * @param cases the scenarios and link geometries to check
     */
    ThreeGppLspDistributionTestCase(const std::vector<Case>& cases);

  private:
    void DoRun() override;

    /**
     * @brief Check the LSPs of one scenario, geometry and link state.
     * @param scenario the ThreeGppChannelModel scenario
     * @param geometry the link geometry
     * @param spatialConsistency whether the inter-UE spatial consistency is enabled
     * @param state the link state
     */
    void CheckState(const std::string& scenario,
                    const LinkGeometry& geometry,
                    bool spatialConsistency,
                    LinkState state);

    std::vector<Case> m_cases;    ///< the scenarios and link geometries to check
    NodeContainer m_sites;        ///< one site per sample, for independent fields
    Ptr<MobilityModel> m_termMob; ///< the terminal mobility model

    static constexpr uint32_t NUM_SAMPLES = 20000; ///< number of links per link state
};

ThreeGppLspDistributionTestCase::ThreeGppLspDistributionTestCase(const std::vector<Case>& cases)
    : TestCase("TR 38.901 LSP distributions"),
      m_cases(cases)
{
}

void
ThreeGppLspDistributionTestCase::CheckState(const std::string& scenario,
                                            const LinkGeometry& geometry,
                                            bool spatialConsistency,
                                            LinkState state)
{
    const bool o2i = state == LinkState::O2I_LOS || state == LinkState::O2I_NLOS;
    const bool los = state == LinkState::LOS || state == LinkState::O2I_LOS;
    auto cond = CreateObject<ChannelCondition>(los ? ChannelCondition::LOS : ChannelCondition::NLOS,
                                               o2i ? ChannelCondition::O2I : ChannelCondition::O2O);
    std::ostringstream label;
    label << scenario << " " << geometry.fcGHz << " GHz "
          << (o2i ? (los ? "O2I (LOS outdoor)" : "O2I (NLOS outdoor)") : (los ? "LOS" : "NLOS"))
          << (spatialConsistency ? ", spatially consistent" : "");

    Ptr<ChannelConditionModel> condModel = CreateObject<AlwaysLosChannelConditionModel>();
    condModel->SetAttribute("InterUeSpatialConsistency", BooleanValue(spatialConsistency));
    auto model = CreateObject<LspProbeChannelModel>();
    model->SetAttribute("Scenario", StringValue(scenario));
    model->SetAttribute("Frequency", DoubleValue(geometry.fcGHz * 1e9));
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

    const LspReference ref = GetSpecLsps(scenario, state, geometry);

    // Cross-correlations first, as the KS test sorts the samples. The standard
    // error of a sample correlation is (1 - r^2) / sqrt(N).
    for (const auto& [a, b, rho] : ref.corr)
    {
        if ((a == K || b == K) && !ref.hasK)
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
                                              << kLspNames[b] << " (Table 7.5-6: " << rho
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
        double mean = 0;
        double var = 0;
        for (double v : x)
        {
            mean += v / NUM_SAMPLES;
        }
        for (double v : x)
        {
            var += (v - mean) * (v - mean) / (NUM_SAMPLES - 1);
        }
        const double d = KsStatistic(x, ref.mu[lsp], ref.sigma[lsp], LogClip(lsp));
        NS_TEST_EXPECT_MSG_LT(d,
                              ksCritical,
                              label.str()
                                  << ": " << kLspNames[lsp] << " sample mean " << mean << " std "
                                  << std::sqrt(var) << ", TR 38.901 mean " << ref.mu[lsp] << " std "
                                  << ref.sigma[lsp] << " (log10, dB for K), KS distance " << d);
    }
}

void
ThreeGppLspDistributionTestCase::DoRun()
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    m_sites.Create(NUM_SAMPLES);
    for (auto it = m_sites.Begin(); it != m_sites.End(); ++it)
    {
        (*it)->AggregateObject(CreateObject<ConstantPositionMobilityModel>());
    }
    Ptr<Node> terminal = CreateObject<Node>();
    m_termMob = CreateObject<ConstantPositionMobilityModel>();
    terminal->AggregateObject(m_termMob);

    for (bool spatialConsistency : {false, true})
    {
        for (const auto& [scenario, geometry] : m_cases)
        {
            for (auto it = m_sites.Begin(); it != m_sites.End(); ++it)
            {
                (*it)->GetObject<MobilityModel>()->SetPosition(Vector(0, 0, geometry.hBs));
            }
            m_termMob->SetPosition(Vector(geometry.d2D, 0, geometry.hUt));
            std::vector<LinkState> states{LinkState::LOS, LinkState::NLOS};
            if (scenario != "InH-OfficeMixed")
            {
                states.push_back(LinkState::O2I_LOS);
                states.push_back(LinkState::O2I_NLOS);
            }
            for (auto state : states)
            {
                CheckState(scenario, geometry, spatialConsistency, state);
            }
        }
    }
    Simulator::Destroy();
}

/**
 * @ingroup spectrum-tests
 *
 * Calibration test suite of ThreeGppChannelModel against TR 38.901.
 */
class ThreeGppChannelCalibrationTestSuite : public TestSuite
{
  public:
    ThreeGppChannelCalibrationTestSuite();
};

ThreeGppChannelCalibrationTestSuite::ThreeGppChannelCalibrationTestSuite()
    : TestSuite("three-gpp-channel-calibration", Type::SYSTEM)
{
    AddTestCase(new ThreeGppLspDistributionTestCase({
                    {"UMa", {3.5, 200, 25, 4.5}},
                    {"UMa", {28, 200, 25, 4.5}},
                    {"UMi-StreetCanyon", {1.8, 100, 10, 1.5}},
                    {"UMi-StreetCanyon", {3.5, 100, 10, 1.5}},
                    {"UMi-StreetCanyon", {28, 100, 10, 1.5}},
                    {"RMa", {3.5, 500, 35, 1.5}},
                    {"InH-OfficeMixed", {3.5, 20, 3, 1}},
                    {"InH-OfficeMixed", {28, 20, 3, 1}},
                }),
                Duration::EXTENSIVE);
}

/// Static variable for test initialization
static ThreeGppChannelCalibrationTestSuite g_threeGppChannelCalibrationTestSuite;
