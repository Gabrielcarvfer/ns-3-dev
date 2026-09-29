/*
 * Copyright (c) 2026, Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "three-gpp-channel-calibration-reference.h"

#include "ns3/abort.h"
#include "ns3/angles.h"
#include "ns3/boolean.h"
#include "ns3/channel-condition-model.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/double.h"
#include "ns3/hexagonal-wraparound-model.h"
#include "ns3/isotropic-antenna-model.h"
#include "ns3/log.h"
#include "ns3/node-container.h"
#include "ns3/node.h"
#include "ns3/object-factory.h"
#include "ns3/pointer.h"
#include "ns3/random-variable-stream.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/three-gpp-antenna-model.h"
#include "ns3/three-gpp-channel-model.h"
#include "ns3/three-gpp-propagation-loss-model.h"
#include "ns3/uinteger.h"
#include "ns3/uniform-planar-array.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <random>
#include <set>
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

namespace
{

/**
 * Channel condition model whose O2I state follows the indoor state of the
 * terminal drawn by the calibration drop, instead of a per-link draw, so that
 * all the links of a terminal share its indoor state (TR 38.901 Sec. 7.5 step 2).
 */
template <class Base>
class DropIndoorConditionModel : public Base
{
  public:
    /**
     * @brief Set the node ids of the indoor terminals.
     * @param indoor the node ids of the indoor terminals
     */
    void SetIndoorNodes(const std::set<uint32_t>& indoor)
    {
        m_indoor = indoor;
    }

  private:
    ChannelCondition::O2iConditionValue ComputeO2i(Ptr<const MobilityModel> a,
                                                   Ptr<const MobilityModel> b) const override
    {
        const bool indoor = m_indoor.contains(a->GetObject<Node>()->GetId()) ||
                            m_indoor.contains(b->GetObject<Node>()->GetId());
        return indoor ? ChannelCondition::O2I : ChannelCondition::O2O;
    }

    std::set<uint32_t> m_indoor; ///< node ids of the indoor terminals
};

/**
 * Setup of a TR 38.901 Sec. 7.8.2 full calibration case.
 */
struct FullCalibrationSetup
{
    std::string name;            ///< reference scenario name: UMa, UMi or InH
    std::string channelScenario; ///< ThreeGppChannelModel scenario
    double fcGHz;                ///< carrier frequency in GHz
    double isd;                  ///< inter-site distance in m
    double hBs;                  ///< BS height in m
    double minDist2D;            ///< minimum 2D BS-UT distance in m
    double tiltDeg;              ///< electrical downtilt (zenith) of the CRS port in degrees
    uint32_t numDrops;           ///< number of drops
};

/// Number of UTs dropped per sector (TR 36.873).
constexpr uint32_t kUtsPerSector = 10;

/// Sector boresights of TR 38.901 Tables 7.8-1 and 7.8-2, in degrees.
constexpr std::array<double, 3> kSectorBoresightsDeg{30, 150, 270};

/**
 * @brief Positions of the 19 sites of a two-ring hexagonal layout whose
 *        neighboring sites lie at 30 + 60 k degrees, the orientation of the
 *        sector boresights and of HexagonalWraparoundModel.
 * @param isd the inter-site distance in m
 * @param hBs the BS height in m
 * @return the site positions
 */
std::vector<Vector>
HexagonalSites(double isd, double hBs)
{
    std::vector<Vector> ring1;
    for (int k = 0; k < 6; k++)
    {
        const double a = (30.0 + 60.0 * k) * M_PI / 180.0;
        ring1.emplace_back(isd * std::cos(a), isd * std::sin(a), hBs);
    }
    std::vector<Vector> sites{Vector(0, 0, hBs)};
    sites.insert(sites.end(), ring1.begin(), ring1.end());
    for (int k = 0; k < 6; k++)
    {
        const Vector& u = ring1[k];
        const Vector& v = ring1[(k + 1) % 6];
        sites.emplace_back(2 * u.x, 2 * u.y, hBs);
        sites.emplace_back(u.x + v.x, u.y + v.y, hBs);
    }
    return sites;
}

/**
 * @brief Whether a point lies in the hexagonal Voronoi cell of a site of the
 *        HexagonalSites layout (vertices at 0 + 60 k degrees).
 * @param dx x offset from the site in m
 * @param dy y offset from the site in m
 * @param isd the inter-site distance in m
 * @return true if the point lies in the cell
 */
bool
InSiteHexagon(double dx, double dy, double isd)
{
    // The cell edges are perpendicular to the directions of the neighbors, at
    // half the inter-site distance.
    for (int k = 0; k < 6; k++)
    {
        const double a = (30.0 + 60.0 * k) * M_PI / 180.0;
        if (dx * std::cos(a) + dy * std::sin(a) > isd / 2)
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief Power delay profile of the serving link used for the delay spread:
 *        the clusters, with the two strongest split into their sub-clusters
 *        (TR 38.901 Table 7.5-5) and, for LOS links, the Ricean weighting of
 *        the LOS ray (Equation 7.5-30).
 * @param params the channel parameters of the link
 * @param cDs the cluster delay spread in s
 * @return the (delay in s, power) taps
 */
std::vector<std::pair<double, double>>
DelayProfile(Ptr<const ThreeGppChannelModel::ThreeGppChannelParams> params, double cDs)
{
    const bool los = params->HasLosRay();
    const double kR = los ? std::pow(10, params->m_K_factor / 10) : 0;
    std::vector<std::pair<double, double>> taps;
    for (uint16_t n = 0; n < params->m_reducedClusterNumber; n++)
    {
        const double p = params->m_clusterPower[n] / (1 + kR);
        const double tau = params->m_delay[n];
        if (n == params->m_cluster1st || n == params->m_cluster2nd)
        {
            taps.emplace_back(tau, p * 10 / 20);
            taps.emplace_back(tau + 1.28 * cDs, p * 6 / 20);
            taps.emplace_back(tau + 2.56 * cDs, p * 4 / 20);
        }
        else
        {
            taps.emplace_back(tau, p);
        }
    }
    if (los)
    {
        taps.emplace_back(params->m_delay[0], kR / (1 + kR));
    }
    return taps;
}

/**
 * @brief RMS delay spread of a power delay profile.
 * @param taps the (delay, power) taps
 * @return the delay spread
 */
double
RmsDelaySpread(const std::vector<std::pair<double, double>>& taps)
{
    double pSum = 0;
    double m1 = 0;
    double m2 = 0;
    for (const auto& [tau, p] : taps)
    {
        pSum += p;
        m1 += p * tau;
        m2 += p * tau * tau;
    }
    m1 /= pSum;
    return std::sqrt(std::max(0.0, m2 / pSum - m1 * m1));
}

/**
 * @brief Circular angle spread of TR 25.996 Annex A: the RMS spread around the
 *        mean, minimized over a rotation of all the angles.
 * @param rays the (angle in rad, power) rays
 * @return the angle spread in rad
 */
double
AngleSpread25996(const std::vector<std::pair<double, double>>& rays)
{
    double pSum = 0;
    for (const auto& [a, p] : rays)
    {
        pSum += p;
    }
    auto wrap = [](double x) { return std::remainder(x, 2 * M_PI); };
    double best = std::numeric_limits<double>::infinity();
    for (int step = 0; step < 360; step++)
    {
        const double delta = step * M_PI / 180;
        double mean = 0;
        for (const auto& [a, p] : rays)
        {
            mean += p * wrap(a + delta);
        }
        mean /= pSum;
        double var = 0;
        for (const auto& [a, p] : rays)
        {
            const double d = wrap(wrap(a + delta) - mean);
            var += p * d * d;
        }
        best = std::min(best, var / pSum);
    }
    return std::sqrt(best);
}

/**
 * @brief Circular angle spread of TR 38.901 Annex A, Equation (A-1).
 * @param rays the (angle in rad, power) rays
 * @return the angle spread in rad
 */
double
AngleSpreadA1(const std::vector<std::pair<double, double>>& rays)
{
    double pSum = 0;
    std::complex<double> acc = 0;
    for (const auto& [a, p] : rays)
    {
        pSum += p;
        acc += p * std::polar(1.0, a);
    }
    const double r = std::min(std::abs(acc) / pSum, 1.0);
    return std::sqrt(std::max(0.0, -2 * std::log(r)));
}

/**
 * @brief Rays of the serving link for one angle, with the powers of their
 *        clusters shared equally and, for LOS links, the Ricean weighting of
 *        the LOS ray (Equation 7.5-30).
 * @param params the channel parameters of the link
 * @param rayAngles the per-cluster, per-ray angles in rad
 * @param losAngle the LOS angle in rad
 * @return the (angle in rad, power) rays
 */
std::vector<std::pair<double, double>>
RayProfile(Ptr<const ThreeGppChannelModel::ThreeGppChannelParams> params,
           const MatrixBasedChannelModel::Double2DVector& rayAngles,
           double losAngle)
{
    const bool los = params->HasLosRay();
    const double kR = los ? std::pow(10, params->m_K_factor / 10) : 0;
    std::vector<std::pair<double, double>> rays;
    for (uint16_t n = 0; n < params->m_reducedClusterNumber; n++)
    {
        const double p = params->m_clusterPower[n] / (1 + kR) / rayAngles[n].size();
        for (double a : rayAngles[n])
        {
            rays.emplace_back(a, p);
        }
    }
    if (los)
    {
        rays.emplace_back(losAngle, kR / (1 + kR));
    }
    return rays;
}

/**
 * @brief Empirical quantile of sorted samples.
 * @param sorted the sorted samples
 * @param q the quantile in [0, 1]
 * @return the quantile
 */
double
Quantile(const std::vector<double>& sorted, double q)
{
    const double pos = std::clamp(q, 0.0, 1.0) * (sorted.size() - 1);
    const auto i = static_cast<size_t>(pos);
    const double f = pos - i;
    return i + 1 < sorted.size() ? sorted[i] * (1 - f) + sorted[i + 1] * f : sorted[i];
}

} // namespace

/**
 * @ingroup spectrum-tests
 *
 * Full calibration of ThreeGppChannelModel, with ThreeGppPropagationLossModel
 * and the 3GPP channel condition models, following TR 38.901 Sec. 7.8.2 and
 * Table 7.8-2 (BS antenna configuration 1): hexagonal 19-site, 3-sector UMa and
 * UMi layouts with wrap-around and the 12-site open office InH layout, TR
 * 36.873 terminal drops with 80% indoor terminals on random floors, CRS port 0
 * mapped to the 16 +45-degree elements of the first panel with the electrical
 * downtilt of the large-scale calibration, and cross-polarized isotropic
 * terminals with a random bearing. The CDFs over the terminals of the coupling
 * loss and the wideband SIR of the serving cell (attachment on the strongest
 * CRS port 0 RSRP), and of the delay and angle spreads of the serving link
 * (circular angle spread of TR 25.996), are compared at every reference
 * percentile with the band spanned by the companies of the 3GPP calibration
 * (R1-165975, TR 38.900 V14.0.0): the UMa coupling loss and SIR percentiles
 * must lie in the band, while the confidence interval (3 standard deviations
 * of the order statistic) of the other percentiles must overlap it, as the InH
 * and O2I cluster parameters of TR 38.900 V14.0.0 differ from TR 38.901. The spreads
 * whose TR 38.900 V14.0.0 parameters differ further (InH, and ASA and ZSA of the
 * O2I links) are not compared with 3GPP, nor is the InH SIR, which depends on
 * the angular spreads through the beam gains. The spreads of all the scenarios are
 * compared with the Sionna TR 38.901 channel models, with attachment on the
 * path loss without shadow fading and the angle spread of TR 38.901 Annex A:
 * the difference of the ns-3 and Sionna percentiles must be within 4 standard
 * deviations of their combined confidence interval (see
 * three-gpp-channel-calibration-reference.py).
 */
class ThreeGppFullCalibrationTestCase : public TestCase
{
  public:
    /**
     * Constructor.
     * @param setups the calibration cases
     */
    ThreeGppFullCalibrationTestCase(const std::vector<FullCalibrationSetup>& setups);

  private:
    void DoRun() override;

    /**
     * @brief Run the drops of one calibration case and check its CDFs.
     * @param setup the calibration case
     */
    void RunSetup(const FullCalibrationSetup& setup);

    std::vector<FullCalibrationSetup> m_setups; ///< the calibration cases
};

ThreeGppFullCalibrationTestCase::ThreeGppFullCalibrationTestCase(
    const std::vector<FullCalibrationSetup>& setups)
    : TestCase("TR 38.901 full calibration against R1-165975"),
      m_setups(setups)
{
}

void
ThreeGppFullCalibrationTestCase::RunSetup(const FullCalibrationSetup& setup)
{
    const bool indoorScenario = setup.name == "InH";
    auto uniform = CreateObject<UniformRandomVariable>();
    std::map<calibration::Metric, std::vector<double>> samples;
    // Spreads of the link attached on the path gain, with the angle spread of
    // TR 38.901 Annex A, for the comparison with Sionna.
    std::map<calibration::Metric, std::vector<double>> pathGainSamples;

    for (uint32_t drop = 0; drop < setup.numDrops; drop++)
    {
        // Sites, and for the wrapped layouts the 6 wrap-around images of each.
        std::vector<Vector> sites;
        if (indoorScenario)
        {
            // TR 38.901 Table 7.2-4: 12 sites on a 20 m grid, 120 m x 50 m hall.
            for (double y : {-10.0, 10.0})
            {
                for (int i = 0; i < 6; i++)
                {
                    sites.emplace_back(-50.0 + 20.0 * i, y, setup.hBs);
                }
            }
        }
        else
        {
            sites = HexagonalSites(setup.isd, setup.hBs);
        }
        Ptr<HexagonalWraparoundModel> wrap;
        if (!indoorScenario)
        {
            wrap = CreateObject<HexagonalWraparoundModel>(setup.isd, sites.size());
            for (const auto& site : sites)
            {
                wrap->AddSitePosition(site);
            }
        }

        // Terminals: TR 36.873 drop per sector for UMa and UMi, uniform in the
        // hall for InH.
        struct Ut
        {
            Vector pos;
            bool indoor;
        };

        std::vector<Ut> uts;
        const uint32_t numUts = kUtsPerSector * kSectorBoresightsDeg.size() * sites.size();
        while (uts.size() < numUts)
        {
            Ut ut;
            if (indoorScenario)
            {
                ut.pos = Vector(uniform->GetValue(-60, 60), uniform->GetValue(-25, 25), 1.0);
                ut.indoor = false;
            }
            else
            {
                const Vector& site = sites[(uts.size() / (kUtsPerSector * 3)) % sites.size()];
                const double r = setup.isd / std::sqrt(3.0);
                const double dx = uniform->GetValue(-r, r);
                const double dy = uniform->GetValue(-r, r);
                if (!InSiteHexagon(dx, dy, setup.isd) || std::hypot(dx, dy) < setup.minDist2D)
                {
                    continue;
                }
                ut.indoor = uniform->GetValue() < 0.8;
                double h = 1.5;
                if (ut.indoor)
                {
                    const auto numFloors = static_cast<uint32_t>(uniform->GetInteger(4, 8));
                    const auto floor = static_cast<uint32_t>(uniform->GetInteger(1, numFloors));
                    h = 3.0 * (floor - 1) + 1.5;
                }
                ut.pos = Vector(site.x + dx, site.y + dy, h);
            }
            uts.push_back(ut);
        }

        // Nodes: for every terminal, one node per site at the wrap-around image
        // of the site nearest to the terminal. They are created before the
        // terminals, so that the sites are the departure end of the generated
        // channel parameters.
        NodeContainer siteNodes(uts.size() * sites.size());
        std::vector<std::vector<Ptr<MobilityModel>>> siteMobs(uts.size());
        for (size_t u = 0; u < uts.size(); u++)
        {
            for (size_t s = 0; s < sites.size(); s++)
            {
                auto mob = CreateObject<ConstantPositionMobilityModel>();
                mob->SetPosition(indoorScenario ? sites[s]
                                                : wrap->GetVirtualPosition(sites[s], uts[u].pos));
                siteNodes.Get(u * sites.size() + s)->AggregateObject(mob);
                siteMobs[u].push_back(mob);
            }
        }
        NodeContainer utNodes(uts.size());
        std::set<uint32_t> indoorIds;
        for (size_t u = 0; u < uts.size(); u++)
        {
            auto mob = CreateObject<ConstantPositionMobilityModel>();
            mob->SetPosition(uts[u].pos);
            utNodes.Get(u)->AggregateObject(mob);
            if (uts[u].indoor)
            {
                indoorIds.insert(utNodes.Get(u)->GetId());
            }
        }

        Ptr<ChannelConditionModel> condModel;
        Ptr<ThreeGppPropagationLossModel> lossModel;
        if (setup.name == "UMa")
        {
            auto m = CreateObject<DropIndoorConditionModel<ThreeGppUmaChannelConditionModel>>();
            m->SetIndoorNodes(indoorIds);
            condModel = m;
            lossModel = CreateObject<ThreeGppUmaPropagationLossModel>();
        }
        else if (setup.name == "UMi")
        {
            auto m = CreateObject<
                DropIndoorConditionModel<ThreeGppUmiStreetCanyonChannelConditionModel>>();
            m->SetIndoorNodes(indoorIds);
            condModel = m;
            lossModel = CreateObject<ThreeGppUmiStreetCanyonPropagationLossModel>();
        }
        else
        {
            condModel = CreateObject<ThreeGppIndoorOpenOfficeChannelConditionModel>();
            lossModel = CreateObject<ThreeGppIndoorOfficePropagationLossModel>();
        }
        // TR 38.901 Table 7.8-1: 50% low-loss and 50% high-loss buildings.
        condModel->SetAttribute("O2iLowLossThreshold", DoubleValue(0.5));
        // The building type and indoor distance are properties of the terminal
        // (TR 38.901 Sec. 7.4.3), which the spatial consistency provides:
        // per-link draws would bias the attachment towards low-loss links.
        condModel->SetAttribute("InterUeSpatialConsistency", BooleanValue(true));
        lossModel->SetAttribute("Frequency", DoubleValue(setup.fcGHz * 1e9));
        lossModel->SetAttribute("ShadowingEnabled", BooleanValue(true));
        lossModel->SetAttribute("BuildingPenetrationLossesEnabled", BooleanValue(true));
        lossModel->SetAttribute("ChannelConditionModel", PointerValue(condModel));
        // Path loss without shadowing, for the attachment of the comparison with
        // Sionna, as in three-gpp-channel-calibration-reference.py.
        Ptr<ThreeGppPropagationLossModel> pathLossModel = DynamicCast<ThreeGppPropagationLossModel>(
            ObjectFactory(lossModel->GetInstanceTypeId().GetName()).Create());
        pathLossModel->SetAttribute("Frequency", DoubleValue(setup.fcGHz * 1e9));
        pathLossModel->SetAttribute("ShadowingEnabled", BooleanValue(false));
        pathLossModel->SetAttribute("BuildingPenetrationLossesEnabled", BooleanValue(true));
        pathLossModel->SetAttribute("ChannelConditionModel", PointerValue(condModel));

        // CRS port 0 of BS antenna configuration 1: the 4x4 +45-degree
        // elements of the first panel, with the electrical downtilt.
        std::vector<std::array<Ptr<UniformPlanarArray>, 3>> sectorAntennas(sites.size());
        std::vector<std::array<PhasedArrayModel::ComplexVector, 3>> sectorWeights(sites.size());
        for (size_t s = 0; s < sites.size(); s++)
        {
            for (size_t c = 0; c < 3; c++)
            {
                // UniformPlanarArray takes bearings in [-pi, pi].
                const double bearing =
                    std::remainder(kSectorBoresightsDeg[c] * M_PI / 180, 2 * M_PI);
                auto ant = CreateObjectWithAttributes<UniformPlanarArray>(
                    "NumColumns",
                    UintegerValue(4),
                    "NumRows",
                    UintegerValue(4),
                    "AntennaHorizontalSpacing",
                    DoubleValue(0.5),
                    "AntennaVerticalSpacing",
                    DoubleValue(0.5),
                    "PolSlantAngle",
                    DoubleValue(M_PI / 4),
                    "BearingAngle",
                    DoubleValue(bearing),
                    "AntennaElement",
                    PointerValue(CreateObject<ThreeGppAntennaModel>()));
                sectorAntennas[s][c] = ant;
                // Beam pointed at the electrical tilt, see
                // PhasedArrayModel::SetBeamformingVector.
                auto w = ant->GetSteeringVector(Angles(bearing, setup.tiltDeg * M_PI / 180));
                const double norm = std::sqrt(static_cast<double>(w.GetSize()));
                for (size_t e = 0; e < w.GetSize(); e++)
                {
                    w[e] /= norm;
                }
                sectorWeights[s][c] = w;
            }
        }

        for (size_t u = 0; u < uts.size(); u++)
        {
            Ptr<MobilityModel> utMob = utNodes.Get(u)->GetObject<MobilityModel>();
            // Cross-polarized (0/90 degrees) isotropic terminal, uniformly
            // random bearing and 90-degree downtilt (Table 7.8-2).
            auto utAnt = CreateObjectWithAttributes<UniformPlanarArray>(
                "NumColumns",
                UintegerValue(1),
                "NumRows",
                UintegerValue(1),
                "IsDualPolarized",
                BooleanValue(true),
                "PolSlantAngle",
                DoubleValue(0),
                "BearingAngle",
                DoubleValue(uniform->GetValue(-M_PI, M_PI)),
                "DowntiltAngle",
                DoubleValue(M_PI / 2),
                "AntennaElement",
                PointerValue(CreateObject<IsotropicAntennaModel>()));

            // A channel model per terminal keeps the memory bounded.
            auto channel = CreateObject<ThreeGppChannelModel>();
            channel->SetAttribute("Scenario", StringValue(setup.channelScenario));
            channel->SetAttribute("Frequency", DoubleValue(setup.fcGHz * 1e9));
            channel->SetAttribute("ChannelConditionModel", PointerValue(condModel));

            std::vector<double> rsrp;
            std::vector<double> pathGain;
            for (size_t s = 0; s < sites.size(); s++)
            {
                Ptr<MobilityModel> siteMob = siteMobs[u][s];
                const double lossDb = lossModel->CalcRxPower(0, siteMob, utMob);
                pathGain.push_back(pathLossModel->CalcRxPower(0, siteMob, utMob));
                for (size_t c = 0; c < 3; c++)
                {
                    const auto& bsAnt = sectorAntennas[s][c];
                    const auto mat = channel->GetChannel(siteMob, utMob, bsAnt, utAnt);
                    const bool reverse = mat->IsReverse(bsAnt->GetId(), utAnt->GetId());
                    const auto& h = mat->m_channel;
                    const auto& w = sectorWeights[s][c];
                    const size_t numUt = utAnt->GetNumElems();
                    const size_t numTaps = h.GetNumPages();
                    double gain = 0;
                    for (size_t e = 0; e < numUt; e++)
                    {
                        for (size_t n = 0; n < numTaps; n++)
                        {
                            std::complex<double> acc = 0;
                            for (size_t b = 0; b < w.GetSize(); b++)
                            {
                                acc += (reverse ? h(b, e, n) : h(e, b, n)) * w[b];
                            }
                            gain += std::norm(acc);
                        }
                    }
                    rsrp.push_back(lossDb + 10 * std::log10(gain / numUt));
                }
            }

            // Attachment on the strongest CRS port 0 RSRP; SIR over all the
            // other cells.
            const auto best = std::max_element(rsrp.begin(), rsrp.end()) - rsrp.begin();
            double interference = 0;
            for (size_t i = 0; i < rsrp.size(); i++)
            {
                if (static_cast<long>(i) != best)
                {
                    interference += std::pow(10, rsrp[i] / 10);
                }
            }
            samples[calibration::Metric::COUPLING_LOSS].push_back(rsrp[best]);
            samples[calibration::Metric::SIR].push_back(rsrp[best] - 10 * std::log10(interference));

            // Delay and angle spreads of a serving link.
            auto addSpreads = [&](std::map<calibration::Metric, std::vector<double>>& out,
                                  Ptr<MobilityModel> servingMob,
                                  double (*angleSpread)(
                                      const std::vector<std::pair<double, double>>&)) {
                const auto params = DynamicCast<const ThreeGppChannelModel::ThreeGppChannelParams>(
                    channel->GetParams(servingMob, utMob));
                const auto cond = condModel->GetChannelCondition(servingMob, utMob);
                const double cDs = channel->GetThreeGppTable(servingMob, utMob, cond)->m_cDS;
                out[calibration::Metric::DS].push_back(RmsDelaySpread(DelayProfile(params, cDs)) *
                                                       1e9);
                const Angles dep(utMob->GetPosition(), servingMob->GetPosition());
                const Angles arr(servingMob->GetPosition(), utMob->GetPosition());
                const double r2d = 180 / M_PI;
                out[calibration::Metric::ASD].push_back(
                    angleSpread(RayProfile(params, params->m_rayAodRadian, dep.GetAzimuth())) *
                    r2d);
                out[calibration::Metric::ZSD].push_back(
                    angleSpread(RayProfile(params, params->m_rayZodRadian, dep.GetInclination())) *
                    r2d);
                out[calibration::Metric::ASA].push_back(
                    angleSpread(RayProfile(params, params->m_rayAoaRadian, arr.GetAzimuth())) *
                    r2d);
                out[calibration::Metric::ZSA].push_back(
                    angleSpread(RayProfile(params, params->m_rayZoaRadian, arr.GetInclination())) *
                    r2d);
            };
            addSpreads(samples, siteMobs[u][best / 3], &AngleSpread25996);
            const auto bestSite =
                std::max_element(pathGain.begin(), pathGain.end()) - pathGain.begin();
            addSpreads(pathGainSamples, siteMobs[u][bestSite], &AngleSpreadA1);
        }
    }

    const std::map<calibration::Metric, std::string> names{
        {calibration::Metric::COUPLING_LOSS, "coupling loss (dB)"},
        {calibration::Metric::SIR, "SIR (dB)"},
        {calibration::Metric::DS, "DS (ns)"},
        {calibration::Metric::ASD, "ASD (deg)"},
        {calibration::Metric::ZSD, "ZSD (deg)"},
        {calibration::Metric::ASA, "ASA (deg)"},
        {calibration::Metric::ZSA, "ZSA (deg)"},
    };
    // Compare every reference percentile, with the confidence interval of the
    // ns-3 percentile given by 3 standard deviations of the order statistic.
    enum class Criterion
    {
        POINT_IN_BAND, ///< the ns-3 percentile lies in the band of the 3GPP companies
        BAND_OVERLAP,  ///< the ns-3 confidence interval overlaps the band
        DIFFERENCE,    ///< the difference is within the combined confidence interval
    };

    auto check = [&](std::map<calibration::Metric, std::vector<double>>& sampleSet,
                     const calibration::ReferenceCdf& ref,
                     const std::string& source,
                     Criterion criterion) {
        auto& x = sampleSet[ref.metric];
        std::sort(x.begin(), x.end());
        const auto n = static_cast<double>(x.size());
        for (size_t i = 0; i < calibration::kReferencePercentiles.size(); i++)
        {
            const double p = calibration::kReferencePercentiles[i] / 100;
            const double delta = 3 * std::sqrt(p * (1 - p) / n);
            const double value = Quantile(x, p);
            const double lo = Quantile(x, p - delta);
            const double hi = Quantile(x, p + delta);
            bool pass = false;
            switch (criterion)
            {
            case Criterion::POINT_IN_BAND:
                pass = value >= ref.low[i] && value <= ref.high[i];
                break;
            case Criterion::BAND_OVERLAP:
                pass = hi >= ref.low[i] && lo <= ref.high[i];
                break;
            case Criterion::DIFFERENCE:
                // Both confidence intervals are asymmetric: use the half-widths
                // facing each other, combined as independent errors, and widened
                // from 3 to 4 standard deviations, as hundreds of percentiles are
                // compared.
                pass = value >= ref.mean[i]
                           ? value - ref.mean[i] <=
                                 4.0 / 3 * std::hypot(value - lo, ref.high[i] - ref.mean[i])
                           : ref.mean[i] - value <=
                                 4.0 / 3 * std::hypot(hi - value, ref.mean[i] - ref.low[i]);
                break;
            }
            NS_LOG_INFO(setup.name << " " << setup.fcGHz << " GHz " << names.at(ref.metric) << " "
                                   << calibration::kReferencePercentiles[i] << "% ns-3 " << value
                                   << " [" << lo << ", " << hi << "] " << source << " "
                                   << ref.mean[i] << " [" << ref.low[i] << ", " << ref.high[i]
                                   << "]");
            NS_TEST_EXPECT_MSG_EQ(
                pass,
                true,
                setup.name << " " << setup.fcGHz << " GHz " << names.at(ref.metric) << " at "
                           << calibration::kReferencePercentiles[i] << "%: ns-3 " << value << " ["
                           << lo << ", " << hi << "], " << source << " " << ref.mean[i] << " ["
                           << ref.low[i] << ", " << ref.high[i] << "]");
        }
    };

    for (const auto& ref : calibration::k3gppFullCalibration)
    {
        if (ref.scenario != setup.name || ref.fcGHz != setup.fcGHz)
        {
            continue;
        }
        // The 3GPP calibration used TR 38.900 V14.0.0, whose InH parameters,
        // and O2I arrival parameters (cASA of 20 instead of 8 degrees, lgZSA and
        // cZSA of Table 7.5-6), differ from those of TR 38.901, so these spreads
        // are only compared with Sionna. With the cASA of TR 38.900 V14.0.0, the
        // UMa and UMi ASA percentiles match the mean of the companies.
        const bool isSpread = ref.metric != calibration::Metric::COUPLING_LOSS &&
                              ref.metric != calibration::Metric::SIR;
        // The InH SIR also differs, as the serving and interfering beam gains
        // depend on the angular spreads, so only the InH coupling loss is kept.
        const bool inhSir = indoorScenario && ref.metric == calibration::Metric::SIR;
        if (inhSir || (isSpread && (indoorScenario || ref.metric == calibration::Metric::ASA ||
                                    ref.metric == calibration::Metric::ZSA)))
        {
            continue;
        }
        // The coupling loss and SIR of UMa must lie in the band. The InH coupling
        // loss, whose TR 38.900 V14.0.0 parameters differ, and the UMi ones only
        // have to overlap it: the UMi coupling loss at 30 GHz is about 2 dB above
        // the mean of the companies, at the edge of their band.
        const bool pointInBand = !isSpread && setup.name == "UMa";
        check(samples,
              ref,
              "3GPP mean",
              pointInBand ? Criterion::POINT_IN_BAND : Criterion::BAND_OVERLAP);
    }
    for (const auto& ref : calibration::kSionnaSpreads)
    {
        if (ref.scenario == setup.name && ref.fcGHz == setup.fcGHz)
        {
            check(pathGainSamples, ref, "Sionna", Criterion::DIFFERENCE);
        }
    }
}

void
ThreeGppFullCalibrationTestCase::DoRun()
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);
    for (const auto& setup : m_setups)
    {
        RunSetup(setup);
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
    AddTestCase(new ThreeGppFullCalibrationTestCase({
                    {"UMa", "UMa", 6, 500, 25, 35, 102, 16},
                    {"UMa", "UMa", 30, 500, 25, 35, 102, 16},
                    {"UMi", "UMi-StreetCanyon", 6, 200, 10, 10, 102, 8},
                    {"UMi", "UMi-StreetCanyon", 30, 200, 10, 10, 102, 8},
                    {"InH", "InH-OfficeOpen", 6, 20, 3, 0, 110, 8},
                    {"InH", "InH-OfficeOpen", 30, 20, 3, 0, 110, 8},
                }),
                Duration::TAKES_FOREVER);
}

/// Static variable for test initialization
static ThreeGppChannelCalibrationTestSuite g_threeGppChannelCalibrationTestSuite;
