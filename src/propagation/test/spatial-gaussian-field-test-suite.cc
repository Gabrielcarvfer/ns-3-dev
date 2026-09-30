/*
 * Copyright (c) 2026, Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/rng-seed-manager.h"
#include "ns3/spatial-gaussian-field.h"
#include "ns3/test.h"

#include <cmath>

/**
 * @file
 * @ingroup propagation-tests
 * SpatialGaussianField test suite.
 */

namespace ns3
{

namespace tests
{

/**
 * @ingroup propagation-tests
 * The field must be a pure function of (seed, run, salt, key, position):
 * repeated evaluations agree, and changing any input changes the value.
 */
class SpatialGaussianFieldDeterminismTestCase : public TestCase
{
  public:
    SpatialGaussianFieldDeterminismTestCase()
        : TestCase("SpatialGaussianField determinism")
    {
    }

  private:
    void DoRun() override;
};

void
SpatialGaussianFieldDeterminismTestCase::DoRun()
{
    const uint32_t savedSeed = RngSeedManager::GetSeed();
    const uint64_t savedRun = RngSeedManager::GetRun();
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    const SpatialGaussianField f{SpatialGaussianField::Salt::SHADOW_FADING};
    const SpatialGaussianField g{SpatialGaussianField::Salt::O2I_PENETRATION};
    const SpatialGaussianField h{SpatialGaussianField::Salt::SHADOW_FADING,
                                 SpatialGaussianField::CellGenerator::IrwinHall};
    const Vector pos(17.3, -4.9, 1.5);
    const double corrDist = 25.0;

    const double ref = f.Sample(7, pos, corrDist);
    NS_TEST_ASSERT_MSG_EQ(f.Sample(7, pos, corrDist), ref, "Sample is not repeatable");
    NS_TEST_ASSERT_MSG_EQ(
        SpatialGaussianField{SpatialGaussianField::Salt::SHADOW_FADING}.Sample(7, pos, corrDist),
        ref,
        "Sample differs across instances of the same field");
    NS_TEST_ASSERT_MSG_EQ(f.Sample(7, Vector(pos.x, pos.y, 30.0), corrDist),
                          ref,
                          "Sample depends on the z coordinate");
    NS_TEST_ASSERT_MSG_NE(f.Sample(8, pos, corrDist), ref, "Sample ignores the field key");
    NS_TEST_ASSERT_MSG_NE(g.Sample(7, pos, corrDist), ref, "Sample ignores the salt");
    NS_TEST_ASSERT_MSG_NE(h.Sample(7, pos, corrDist), ref, "Sample ignores the cell generator");
    NS_TEST_ASSERT_MSG_NE(f.Sample(7, pos, 2 * corrDist),
                          ref,
                          "Sample ignores the correlation distance");

    NS_TEST_ASSERT_MSG_EQ(f.SampleWindow(7, SpatialGaussianField::ComputeWindow(pos, corrDist)),
                          ref,
                          "SampleWindow disagrees with Sample");
    NS_TEST_ASSERT_MSG_EQ(f.SampleUniform(7, pos, corrDist),
                          0.5 * std::erfc(-ref * M_SQRT1_2),
                          "SampleUniform is not the normal CDF of Sample");

    // Degenerate correlation distance: repeatable, but uncorrelated.
    const double d = f.Sample(7, pos, 0.0);
    NS_TEST_ASSERT_MSG_EQ(f.Sample(7, pos, 0.0), d, "Degenerate sample is not repeatable");
    NS_TEST_ASSERT_MSG_NE(f.Sample(7, Vector(pos.x + 1e-9, pos.y, pos.z), 0.0),
                          d,
                          "Degenerate sample ignores the position");
    // Positions differing only in the exponent or high mantissa bits of a
    // coordinate must not alias.
    NS_TEST_ASSERT_MSG_NE(f.Sample(7, Vector(1.0, 0.0, 0.0), 0.0),
                          f.Sample(7, Vector(2.0, 0.0, 0.0), 0.0),
                          "Degenerate sample ignores the high bits of x");
    NS_TEST_ASSERT_MSG_NE(f.Sample(7, Vector(0.0, 1.0, 0.0), 0.0),
                          f.Sample(7, Vector(0.0, 2.0, 0.0), 0.0),
                          "Degenerate sample ignores the high bits of y");

    RngSeedManager::SetSeed(2);
    NS_TEST_ASSERT_MSG_NE(f.Sample(7, pos, corrDist), ref, "Sample ignores the seed");
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(2);
    NS_TEST_ASSERT_MSG_NE(f.Sample(7, pos, corrDist), ref, "Sample ignores the run");

    RngSeedManager::SetSeed(savedSeed);
    RngSeedManager::SetRun(savedRun);
}

/**
 * @ingroup propagation-tests
 * The marginal of the field must be N(0,1), the sampled autocorrelation must
 * match the exact autocorrelation of the discrete filter, and the exact
 * autocorrelation must follow exp(-d/dcor) in every direction, for both cell
 * generators.
 */
class SpatialGaussianFieldStatisticsTestCase : public TestCase
{
  public:
    /**
     * Constructor.
     * @param generator The per-cell generator under test.
     */
    SpatialGaussianFieldStatisticsTestCase(SpatialGaussianField::CellGenerator generator)
        : TestCase(generator == SpatialGaussianField::CellGenerator::BoxMuller
                       ? "SpatialGaussianField statistics (Box-Muller cells)"
                       : "SpatialGaussianField statistics (Irwin-Hall cells)"),
          m_field(SpatialGaussianField::Salt::CHANNEL_PARAMETERS, generator)
    {
    }

  private:
    void DoRun() override;

    /**
     * Exact autocorrelation of the discrete filter between two positions,
     * i.e. the normalized inner product of the two filter windows.
     * @param a The first position.
     * @param b The second position.
     * @param corrDist The correlation distance.
     * @return The autocorrelation.
     */
    static double ExactAutocorrelation(const Vector& a, const Vector& b, double corrDist);

    /**
     * Second position at a separation and angle from a first position.
     * @param a The first position.
     * @param separation The separation.
     * @param theta The angle of the separation vector.
     * @return The second position.
     */
    static Vector Offset(const Vector& a, double separation, double theta);

    SpatialGaussianField m_field; ///< the field under test
};

double
SpatialGaussianFieldStatisticsTestCase::ExactAutocorrelation(const Vector& a,
                                                             const Vector& b,
                                                             double corrDist)
{
    const auto wa = SpatialGaussianField::ComputeWindow(a, corrDist);
    const auto wb = SpatialGaussianField::ComputeWindow(b, corrDist);
    // The components are independent, so their covariances add. Within a
    // component, the windows share the cells whose grid coordinates coincide,
    // and the separable weights factorize the inner product into x and y sums.
    double acf = 0.0;
    for (std::size_t s = 0; s < SpatialGaussianField::NUM_SCALES; s++)
    {
        const auto& sa = wa.scales[s];
        const auto& sb = wb.scales[s];
        double sx = 0.0;
        double sy = 0.0;
        for (std::size_t i = 0; i < SpatialGaussianField::WINDOW_CELLS; i++)
        {
            for (std::size_t k = 0; k < SpatialGaussianField::WINDOW_CELLS; k++)
            {
                if (sa.ix + static_cast<int64_t>(i) == sb.ix + static_cast<int64_t>(k))
                {
                    sx += sa.wx[i] * sb.wx[k];
                }
                if (sa.iy + static_cast<int64_t>(i) == sb.iy + static_cast<int64_t>(k))
                {
                    sy += sa.wy[i] * sb.wy[k];
                }
            }
        }
        acf += sx * sy * sa.gain * sb.gain;
    }
    return acf;
}

Vector
SpatialGaussianFieldStatisticsTestCase::Offset(const Vector& a, double separation, double theta)
{
    return Vector(a.x + separation * std::cos(theta), a.y + separation * std::sin(theta), a.z);
}

void
SpatialGaussianFieldStatisticsTestCase::DoRun()
{
    const uint32_t savedSeed = RngSeedManager::GetSeed();
    const uint64_t savedRun = RngSeedManager::GetRun();
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    const uint64_t nFields = 4000;
    const double corrDist = 20.0;
    const Vector origin(31.7, -12.3, 0);

    // Marginal moments over independent fields at one position.
    double sum = 0.0;
    double sum2 = 0.0;
    for (uint64_t key = 0; key < nFields; key++)
    {
        const double v = m_field.Sample(key, origin, corrDist);
        sum += v;
        sum2 += v * v;
    }
    const double mean = sum / nFields;
    const double var = sum2 / nFields - mean * mean;
    // Standard errors of the mean and variance estimates are 1/sqrt(n) and
    // sqrt(2/n); the tolerances are about 4 standard errors.
    NS_TEST_ASSERT_MSG_EQ_TOL(mean, 0.0, 0.07, "Field marginal is not zero-mean");
    NS_TEST_ASSERT_MSG_EQ_TOL(var, 1.0, 0.1, "Field marginal is not unit-variance");

    // Sampled autocorrelation over independent fields against the exact
    // autocorrelation of the discrete filter. The separation vector is
    // rotated per field so the estimate is not tied to the grid axes.
    for (double ratio : {0.25, 0.5, 1.0, 2.0})
    {
        double sumAb = 0.0;
        double sumA2 = 0.0;
        double sumB2 = 0.0;
        double exact = 0.0;
        for (uint64_t key = 0; key < nFields; key++)
        {
            const Vector b = Offset(origin, ratio * corrDist, 2.0 * M_PI * key / nFields);
            const double va = m_field.Sample(key, origin, corrDist);
            const double vb = m_field.Sample(key, b, corrDist);
            sumAb += va * vb;
            sumA2 += va * va;
            sumB2 += vb * vb;
            exact += ExactAutocorrelation(origin, b, corrDist);
        }
        const double measured = sumAb / std::sqrt(sumA2 * sumB2);
        exact /= nFields;
        // The standard error of the correlation estimate is below 1/sqrt(n).
        NS_TEST_ASSERT_MSG_EQ_TOL(measured,
                                  exact,
                                  0.05,
                                  "Sampled autocorrelation at " << ratio
                                                                << " correlation distances");
    }

    // The exact autocorrelation, averaged over grid offsets, must follow the
    // exp(-d/dcor) of TR 38.901 in every direction.
    const uint32_t nOffsets = 200;
    for (double theta : {0.0, M_PI / 8, M_PI / 4, M_PI / 3})
    {
        for (double ratio : {0.02, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0})
        {
            double exact = 0.0;
            for (uint32_t k = 0; k < nOffsets; k++)
            {
                const Vector a(origin.x + 0.173 * k, origin.y + 0.311 * k, 0);
                exact += ExactAutocorrelation(a, Offset(a, ratio * corrDist, theta), corrDist);
            }
            exact /= nOffsets;
            NS_TEST_ASSERT_MSG_EQ_TOL(exact,
                                      std::exp(-ratio),
                                      0.01,
                                      "Autocorrelation at "
                                          << ratio << " correlation distances, angle " << theta);
        }
    }

    RngSeedManager::SetSeed(savedSeed);
    RngSeedManager::SetRun(savedRun);
}

/**
 * @ingroup propagation-tests
 * SpatialGaussianField test suite.
 */
class SpatialGaussianFieldTestSuite : public TestSuite
{
  public:
    SpatialGaussianFieldTestSuite()
        : TestSuite("propagation-spatial-gaussian-field", Type::UNIT)
    {
        AddTestCase(new SpatialGaussianFieldDeterminismTestCase, Duration::QUICK);
        AddTestCase(new SpatialGaussianFieldStatisticsTestCase(
                        SpatialGaussianField::CellGenerator::BoxMuller),
                    Duration::QUICK);
        AddTestCase(new SpatialGaussianFieldStatisticsTestCase(
                        SpatialGaussianField::CellGenerator::IrwinHall),
                    Duration::QUICK);
    }
};

/// Static variable for test initialization.
static SpatialGaussianFieldTestSuite g_spatialGaussianFieldTestSuite;

} // namespace tests

} // namespace ns3
