/*
 * Copyright (c) 2026, Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "three-gpp-ntn-channel-calibration-reference.h"

#include "ns3/boolean.h"
#include "ns3/channel-condition-model.h"
#include "ns3/double.h"
#include "ns3/geocentric-constant-position-mobility-model.h"
#include "ns3/log.h"
#include "ns3/node-container.h"
#include "ns3/node.h"
#include "ns3/object-factory.h"
#include "ns3/pointer.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/three-gpp-channel-model.h"
#include "ns3/three-gpp-propagation-loss-model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

/**
 * @file
 * @ingroup spectrum-tests
 *
 * Calibration of the NTN channel of ThreeGppChannelModel,
 * ThreeGppNTN*PropagationLossModel and ThreeGppNTN*ChannelConditionModel
 * against the tables of 3GPP TR 38.811 V15.4.0.
 */

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ThreeGppNtnChannelCalibrationTest");

namespace
{

/// LSP order of TR 38.901 Sec. 7.5 step 4 and of the reference: [SF, K, DS, ASD, ASA, ZSD, ZSA].
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

/// Names of the LSPs.
const std::array<std::string, NUM_LSPS> kLspNames{"SF", "K", "DS", "ASD", "ASA", "ZSD", "ZSA"};

/// Altitude of the HAPS, below the 50 km above which a platform is a satellite, in meters.
constexpr double kHapsAltitude = 20e3;
/// Altitude of the LEO satellite, in meters.
constexpr double kLeoAltitude = 600e3;

/**
 * @brief Place a ground terminal and an NTN platform seen at a given elevation.
 * @param elevationDeg the elevation angle of the platform seen from the terminal, in degrees
 * @param altitude the altitude of the platform, in meters
 * @param ut the terminal mobility model to place
 * @param platform the platform mobility model to place
 */
void
PlaceNtnLink(double elevationDeg,
             double altitude,
             Ptr<GeocentricConstantPositionMobilityModel> ut,
             Ptr<GeocentricConstantPositionMobilityModel> platform)
{
    // Central angle between the terminal and the sub-platform point on a
    // spherical Earth, refined by secant steps so that the elevation of the
    // geodetic model is the requested one (the models apply the atmospheric
    // absorption below 10 degrees).
    const double re = 6371e3;
    const double el = elevationDeg * M_PI / 180;
    ut->SetGeographicPosition(Vector(45, 10, 0));
    auto elevationAt = [&](double psiDeg) {
        platform->SetGeographicPosition(Vector(45 + psiDeg, 10, altitude));
        return ut->GetElevationAngle(platform);
    };
    double psi0 = (std::acos(re / (re + altitude) * std::cos(el)) - el) * 180 / M_PI;
    double psi1 = psi0 + 0.01;
    double e0 = elevationAt(psi0) - elevationDeg;
    for (int i = 0; i < 20 && std::abs(e0) > 1e-9; i++)
    {
        const double e1 = elevationAt(psi1) - elevationDeg;
        if (e1 == e0)
        {
            break;
        }
        const double next = psi1 - e1 * (psi1 - psi0) / (e1 - e0);
        psi0 = psi1;
        e0 = e1;
        psi1 = next;
    }
    elevationAt(psi0);
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
 * The large deviations of some TR 38.811 tables clip a noticeable fraction of
 * the angular spreads at 104 and 52 degrees, which lowers their correlations.
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
    constexpr uint32_t numDraws = 100000;
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
 * @brief Frequency of a TR 38.811 band used by the tests.
 * @param band S or Ka
 * @return the carrier frequency in Hz
 */
double
BandFrequency(const std::string& band)
{
    return band == "S" ? 2e9 : 20e9;
}

} // namespace

/**
 * @ingroup spectrum-tests
 *
 * Test that the LSPs drawn by ThreeGppChannelModel for the NTN scenarios follow
 * the distributions of TR 38.811 Tables 6.7.2-1a to 6.7.2-8b (read from the TR,
 * see three-gpp-ntn-channel-calibration-reference.py), for every scenario, LOS
 * state, band and reference elevation of a HAPS link: log-normal DS and angular
 * spreads clipped at 104 and 52 degrees and normal K-factor, checked with a
 * Kolmogorov-Smirnov test, and the cross-correlations of the tables, checked
 * against the sampling uncertainty. The departure spreads of a satellite link
 * must be zero (NOTE 8 of the tables).
 */
class ThreeGppNtnLspDistributionTestCase : public TestCase
{
  public:
    ThreeGppNtnLspDistributionTestCase()
        : TestCase("TR 38.811 LSP distributions of the NTN scenarios")
    {
    }

  private:
    void DoRun() override;

    static constexpr uint32_t NUM_SAMPLES = 10000; ///< number of LSP draws per table and elevation
};

void
ThreeGppNtnLspDistributionTestCase::DoRun()
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    NodeContainer nodes(2);
    auto ut = CreateObject<GeocentricConstantPositionMobilityModel>();
    auto platform = CreateObject<GeocentricConstantPositionMobilityModel>();
    nodes.Get(0)->AggregateObject(ut);
    nodes.Get(1)->AggregateObject(platform);

    // Kolmogorov-Smirnov critical value for a significance of 1e-6 per test.
    const double ksCritical = std::sqrt(-0.5 * std::log(0.5e-6)) / std::sqrt(NUM_SAMPLES);
    double maxKs = 0;
    double maxCorrError = 0;
    for (const auto& ref : ntncalibration::kLsp)
    {
        auto cond =
            CreateObject<ChannelCondition>(ref.los ? ChannelCondition::LOS : ChannelCondition::NLOS,
                                           ChannelCondition::O2O);
        auto model = CreateObject<LspProbeChannelModel>();
        model->SetAttribute("Scenario", StringValue(std::string("NTN-") + ref.scenario));
        model->SetAttribute("Frequency", DoubleValue(BandFrequency(ref.band)));
        model->SetAttribute("ChannelConditionModel",
                            PointerValue(CreateObject<AlwaysLosChannelConditionModel>()));
        model->AssignStreams(1);
        std::ostringstream label;
        label << "NTN-" << ref.scenario << " " << (ref.los ? "LOS" : "NLOS") << " " << ref.band
              << " band, " << ref.elevation << " deg";

        PlaceNtnLink(ref.elevation, kHapsAltitude, ut, platform);
        const auto table = model->GetThreeGppTable(ut, platform, cond);
        std::array<std::vector<double>, NUM_LSPS> samples;
        for (uint32_t i = 0; i < NUM_SAMPLES; i++)
        {
            const auto lsps = model->Draw(cond, table, platform, ut);
            samples[K].push_back(lsps.kFactor);
            samples[DS].push_back(std::log10(lsps.DS));
            samples[ASD].push_back(std::log10(lsps.ASD));
            samples[ASA].push_back(std::log10(lsps.ASA));
            samples[ZSD].push_back(std::log10(lsps.ZSD));
            samples[ZSA].push_back(std::log10(lsps.ZSA));
        }

        // Cross-correlations first, as the KS test sorts the samples. The
        // standard error of a sample correlation is (1 - r^2) / sqrt(N).
        for (uint8_t a = K; a < NUM_LSPS; a++)
        {
            for (uint8_t b = a + 1; b < NUM_LSPS; b++)
            {
                if (a == K && !ref.hasK)
                {
                    continue;
                }
                const double expected = ClippedCorrelation(ref.mu[a],
                                                           ref.sigma[a],
                                                           LogClip(Lsp(a)),
                                                           ref.mu[b],
                                                           ref.sigma[b],
                                                           LogClip(Lsp(b)),
                                                           ref.corr[a][b]);
                const double r = Correlation(samples[a], samples[b]);
                maxCorrError = std::max(maxCorrError, std::abs(r - expected));
                NS_TEST_EXPECT_MSG_EQ_TOL(
                    r,
                    expected,
                    5 * (1 - expected * expected) / std::sqrt(NUM_SAMPLES) + 0.005,
                    label.str() << ": correlation " << kLspNames[a] << "-" << kLspNames[b]
                                << " (TR 38.811: " << ref.corr[a][b]
                                << ", after clipping: " << expected << ")");
            }
        }
        for (uint8_t lsp = K; lsp < NUM_LSPS; lsp++)
        {
            if (lsp == K && !ref.hasK)
            {
                continue;
            }
            const double d =
                KsStatistic(samples[lsp], ref.mu[lsp], ref.sigma[lsp], LogClip(Lsp(lsp)));
            maxKs = std::max(maxKs, d);
            NS_TEST_EXPECT_MSG_LT(d,
                                  ksCritical,
                                  label.str()
                                      << ": " << kLspNames[lsp] << " TR 38.811 mean " << ref.mu[lsp]
                                      << " std " << ref.sigma[lsp] << ", KS distance " << d);
        }

        // NOTE 8: no departure spreads for a satellite
        PlaceNtnLink(ref.elevation, kLeoAltitude, ut, platform);
        const auto satTable = model->GetThreeGppTable(ut, platform, cond);
        const auto satLsps = model->Draw(cond, satTable, platform, ut);
        NS_TEST_EXPECT_MSG_EQ(satLsps.ASD, 0, label.str() << ": ASD of a satellite");
        NS_TEST_EXPECT_MSG_EQ(satLsps.ZSD, 0, label.str() << ": ZSD of a satellite");
    }
    NS_LOG_INFO("largest KS distance " << maxKs << " (critical " << ksCritical
                                       << "), largest correlation error " << maxCorrError);
    Simulator::Destroy();
}

/**
 * @ingroup spectrum-tests
 *
 * Test the LOS probability of the ThreeGppNTN*ChannelConditionModel classes
 * against TR 38.811 Table 6.6.1-1, at each reference elevation and 4 degrees
 * away from it, which must take the probability of the nearest reference
 * elevation (Sec. 6.6.1).
 */
class ThreeGppNtnLosProbabilityTestCase : public TestCase
{
  public:
    ThreeGppNtnLosProbabilityTestCase()
        : TestCase("TR 38.811 LOS probability of the NTN scenarios")
    {
    }

  private:
    void DoRun() override;

    static constexpr uint32_t NUM_LINKS = 4000; ///< number of links per elevation
};

void
ThreeGppNtnLosProbabilityTestCase::DoRun()
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    // the terminals share one position and see the platform at the same
    // elevation, each with its own, independent, condition
    NodeContainer uts(NUM_LINKS);
    NodeContainer platform(1);
    std::vector<Ptr<GeocentricConstantPositionMobilityModel>> utMobs;
    for (uint32_t i = 0; i < NUM_LINKS; i++)
    {
        utMobs.push_back(CreateObject<GeocentricConstantPositionMobilityModel>());
        uts.Get(i)->AggregateObject(utMobs.back());
    }
    auto platformMob = CreateObject<GeocentricConstantPositionMobilityModel>();
    platform.Get(0)->AggregateObject(platformMob);

    double maxError = 0;
    for (const auto& ref : ntncalibration::kLos)
    {
        for (double offset : {-4.0, 0.0, 4.0})
        {
            const double elevation = ref.elevation + offset;
            if (elevation > 90)
            {
                continue;
            }
            ObjectFactory factory(std::string("ns3::ThreeGppNTN") + ref.scenario +
                                  "ChannelConditionModel");
            auto condModel = factory.Create<ChannelConditionModel>();
            condModel->AssignStreams(1 + ref.elevation);
            for (auto& mob : utMobs)
            {
                PlaceNtnLink(elevation, kLeoAltitude, mob, platformMob);
            }
            uint32_t numLos = 0;
            for (auto& mob : utMobs)
            {
                numLos += condModel->GetChannelCondition(mob, platformMob)->IsLos();
            }
            const double p = static_cast<double>(numLos) / NUM_LINKS;
            maxError = std::max(maxError, std::abs(p - ref.pLos));
            // 5 standard errors of the binomial proportion
            NS_TEST_EXPECT_MSG_EQ_TOL(p,
                                      ref.pLos,
                                      5 * std::sqrt(ref.pLos * (1 - ref.pLos) / NUM_LINKS) + 1e-3,
                                      "NTN-" << ref.scenario << " LOS probability at " << elevation
                                             << " deg");
        }
    }
    NS_LOG_INFO("largest LOS probability error " << maxError);
    Simulator::Destroy();
}

/**
 * @ingroup spectrum-tests
 *
 * Test the basic path loss of the ThreeGppNTN*PropagationLossModel classes,
 * TR 38.811 Equation 6.6-4, for a LEO satellite at every reference elevation:
 * the shadow fading deviation of LOS and NLOS links and the clutter loss of
 * NLOS links of Tables 6.6.2-1 to 6.6.2-3 in the S and Ka bands, and the free
 * space path loss of Equation 6.6-2 in the S band, where the model applies no
 * atmospheric absorption above 10 degrees. In the Ka band, the clutter loss is
 * checked as the difference between the NLOS and LOS losses, in which the
 * atmospheric absorption cancels. The tropospheric scintillation loss in the Ka
 * band must be the one of Table 6.6.6.2.1-1.
 */
class ThreeGppNtnPathLossTestCase : public TestCase
{
  public:
    ThreeGppNtnPathLossTestCase()
        : TestCase("TR 38.811 path loss, shadow fading and clutter loss of the NTN scenarios")
    {
    }

  private:
    void DoRun() override;

    static constexpr uint32_t NUM_LINKS = 5000; ///< number of links per elevation and state
};

void
ThreeGppNtnPathLossTestCase::DoRun()
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    NodeContainer uts(NUM_LINKS);
    NodeContainer platform(1);
    std::vector<Ptr<GeocentricConstantPositionMobilityModel>> utMobs;
    for (uint32_t i = 0; i < NUM_LINKS; i++)
    {
        utMobs.push_back(CreateObject<GeocentricConstantPositionMobilityModel>());
        uts.Get(i)->AggregateObject(utMobs.back());
    }
    auto platformMob = CreateObject<GeocentricConstantPositionMobilityModel>();
    platform.Get(0)->AggregateObject(platformMob);

    auto makeModel = [](const std::string& scenario,
                        double frequency,
                        bool los,
                        bool shadowing,
                        bool troposphere) {
        ObjectFactory factory(std::string("ns3::ThreeGppNTN") + scenario + "PropagationLossModel");
        factory.Set("Frequency", DoubleValue(frequency));
        factory.Set("ShadowingEnabled", BooleanValue(shadowing));
        factory.Set("IonosphericScintillationLossEnabled", BooleanValue(false));
        factory.Set("TroposphericScintillationLossEnabled", BooleanValue(troposphere));
        Ptr<ChannelConditionModel> cond;
        if (los)
        {
            cond = CreateObject<AlwaysLosChannelConditionModel>();
        }
        else
        {
            cond = CreateObject<NeverLosChannelConditionModel>();
        }
        factory.Set("ChannelConditionModel", PointerValue(cond));
        auto model = factory.Create<ThreeGppPropagationLossModel>();
        model->AssignStreams(1);
        return model;
    };

    // mean and standard deviation of the losses of the links
    auto stats = [&](Ptr<ThreeGppPropagationLossModel> model) {
        double mean = 0;
        double var = 0;
        std::vector<double> loss;
        for (auto& mob : utMobs)
        {
            loss.push_back(-model->CalcRxPower(0, mob, platformMob));
            mean += loss.back() / NUM_LINKS;
        }
        for (double l : loss)
        {
            var += (l - mean) * (l - mean) / (NUM_LINKS - 1);
        }
        return std::make_pair(mean, std::sqrt(var));
    };

    double maxStdError = 0;
    double maxMeanError = 0;
    for (const auto& ref : ntncalibration::kSfCl)
    {
        const double fc = BandFrequency(ref.band);
        for (auto& mob : utMobs)
        {
            // just above the reference elevation, as at 10 degrees the
            // atmospheric absorption applies only below it
            PlaceNtnLink(ref.elevation + 1e-6, kLeoAltitude, mob, platformMob);
        }
        const double d = CalculateDistance(utMobs[0]->GetPosition(), platformMob->GetPosition());
        const double fspl = 32.45 + 20 * std::log10(fc / 1e9) + 20 * std::log10(d); // (6.6-2)
        std::ostringstream label;
        label << "NTN-" << ref.scenario << " " << ref.band << " band, " << ref.elevation << " deg";

        const auto [meanLos, stdLos] = stats(makeModel(ref.scenario, fc, true, true, false));
        const auto [meanNlos, stdNlos] = stats(makeModel(ref.scenario, fc, false, true, false));
        // The standard error of a sample standard deviation is sigma / sqrt(2 n),
        // and that of a sample mean sigma / sqrt(n); the tolerances are 5 of them.
        for (const auto& [sample, expected, state] :
             {std::tuple{stdLos, ref.sfLos, "LOS"}, std::tuple{stdNlos, ref.sfNlos, "NLOS"}})
        {
            maxStdError = std::max(maxStdError, std::abs(sample - expected));
            NS_TEST_EXPECT_MSG_EQ_TOL(sample,
                                      expected,
                                      5 * expected / std::sqrt(2.0 * NUM_LINKS) + 1e-6,
                                      label.str() << ": " << state << " shadow fading deviation");
        }
        const double meanTol = 5 * std::hypot(ref.sfLos, ref.sfNlos) / std::sqrt(NUM_LINKS);
        maxMeanError = std::max(maxMeanError, std::abs(meanNlos - meanLos - ref.clNlos));
        NS_TEST_EXPECT_MSG_EQ_TOL(meanNlos - meanLos,
                                  ref.clNlos,
                                  meanTol,
                                  label.str() << ": NLOS clutter loss");
        if (std::string(ref.band) == "S")
        {
            NS_TEST_EXPECT_MSG_EQ_TOL(meanLos,
                                      fspl,
                                      5 * ref.sfLos / std::sqrt(NUM_LINKS),
                                      label.str() << ": LOS free space path loss");
        }
        else
        {
            const auto [withTropo, unused1] = stats(makeModel(ref.scenario, fc, true, false, true));
            const auto [withoutTropo, unused2] =
                stats(makeModel(ref.scenario, fc, true, false, false));
            const auto it =
                std::find_if(ntncalibration::kTroposphere.begin(),
                             ntncalibration::kTroposphere.end(),
                             [&](const auto& t) { return t.elevation == ref.elevation; });
            NS_TEST_EXPECT_MSG_EQ_TOL(withTropo - withoutTropo,
                                      it->loss,
                                      1e-6,
                                      label.str() << ": tropospheric scintillation loss");
        }
    }
    NS_LOG_INFO("largest shadow fading deviation error "
                << maxStdError << " dB, largest clutter loss error " << maxMeanError << " dB");
    Simulator::Destroy();
}

/**
 * @ingroup spectrum-tests
 *
 * Calibration of the NTN channel against 3GPP TR 38.811.
 */
class ThreeGppNtnChannelCalibrationTestSuite : public TestSuite
{
  public:
    ThreeGppNtnChannelCalibrationTestSuite();
};

ThreeGppNtnChannelCalibrationTestSuite::ThreeGppNtnChannelCalibrationTestSuite()
    : TestSuite("three-gpp-ntn-channel-calibration", Type::SYSTEM)
{
    AddTestCase(new ThreeGppNtnLosProbabilityTestCase(), TestCase::Duration::QUICK);
    AddTestCase(new ThreeGppNtnPathLossTestCase(), TestCase::Duration::QUICK);
    AddTestCase(new ThreeGppNtnLspDistributionTestCase(), TestCase::Duration::EXTENSIVE);
}

/// Static variable for test initialization
static ThreeGppNtnChannelCalibrationTestSuite g_threeGppNtnChannelCalibrationTestSuite;
