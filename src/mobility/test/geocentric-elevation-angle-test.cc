/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ns3/geocentric-constant-position-mobility-model.h"
#include "ns3/test.h"

using namespace ns3;

/**
 * @ingroup mobility-test
 *
 * @brief Tests GeocentricConstantPositionMobilityModel::GetElevationAngle (issue #1308).
 *
 * The elevation angle is measured at the lower terminal, with the local
 * vertical given by the geocentric radial direction (the model uses a perfect
 * sphere). The previous implementation (a) selected the lower terminal using
 * the ECEF z-coordinate instead of the distance from the Earth's centre and
 * (b) took the absolute value of the result, so it returned a positive angle
 * even for a terminal below the local horizon and depended on the argument
 * order. This test pins the corrected, signed behavior.
 */
class GeocentricElevationAngleTestCase : public TestCase
{
  public:
    GeocentricElevationAngleTestCase()
        : TestCase("GeocentricConstantPositionMobilityModel elevation angle")
    {
    }

  private:
    void DoRun() override;

    /// Build a model at the given geographic position.
    /// @param lat latitude (deg)
    /// @param lon longitude (deg)
    /// @param alt altitude (m)
    /// @return the configured mobility model
    static Ptr<GeocentricConstantPositionMobilityModel> Make(double lat, double lon, double alt)
    {
        auto m = CreateObject<GeocentricConstantPositionMobilityModel>();
        m->SetGeographicPosition(Vector(lat, lon, alt));
        return m;
    }
};

void
GeocentricElevationAngleTestCase::DoRun()
{
    constexpr double TOL = 1e-2; // degrees
    const double geoAlt = 35786e3;
    const double leoAlt = 600e3;

    // A satellite directly above the ground station is at zenith: 90 degrees.
    auto gs = Make(45.0, 0.0, 0.0);
    auto overhead = Make(45.0, 0.0, geoAlt);
    NS_TEST_ASSERT_MSG_EQ_TOL(gs->GetElevationAngle(overhead),
                              90.0,
                              TOL,
                              "Satellite at zenith should have 90 degrees elevation");

    // GEO satellite over the equator seen from a ground station at 45 N.
    auto geoSat = Make(0.0, 0.0, geoAlt);
    double elevFromGs = gs->GetElevationAngle(geoSat);
    NS_TEST_ASSERT_MSG_EQ_TOL(elevFromGs, 38.1771, TOL, "Wrong elevation for GEO over the equator");
    // The result must be independent of the argument order (the old z-based
    // lower-terminal selection broke this symmetry).
    NS_TEST_ASSERT_MSG_EQ_TOL(geoSat->GetElevationAngle(gs),
                              elevFromGs,
                              TOL,
                              "Elevation angle must not depend on the argument order");

    // A terminal below the local horizon must yield a negative elevation (the
    // old std::abs forced a positive angle here).
    auto equatorGs = Make(0.0, 0.0, 0.0);
    auto farSat = Make(0.0, 120.0, leoAlt);
    NS_TEST_ASSERT_MSG_EQ_TOL(equatorGs->GetElevationAngle(farSat),
                              -58.5127,
                              TOL,
                              "Below-horizon terminal should have a negative elevation");
}

/**
 * @ingroup mobility-test
 *
 * @brief Geocentric elevation angle test suite.
 */
class GeocentricElevationAngleTestSuite : public TestSuite
{
  public:
    GeocentricElevationAngleTestSuite()
        : TestSuite("geocentric-elevation-angle", Type::UNIT)
    {
        AddTestCase(new GeocentricElevationAngleTestCase, TestCase::Duration::QUICK);
    }
};

static GeocentricElevationAngleTestSuite
    g_geocentricElevationAngleTestSuite; //!< Static variable for test initialization
