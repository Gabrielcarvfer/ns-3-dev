// Copyright (c) Tim Schubert
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Tim Schubert <ns-3-leo@timschubert.net>
// Porting: Thiago Miyazaki <miyathiago@gmail.com> <t.miyazaki@unesp.br>

#ifndef LEO_CIRCULAR_ORBIT_MOBILITY_MODEL_H
#define LEO_CIRCULAR_ORBIT_MOBILITY_MODEL_H

#include "geocentric-constant-position-mobility-model.h"
#include "mobility-model.h"

#include "ns3/nstime.h"
#include "ns3/object.h"
#include "ns3/vector.h"

/**
 * @file
 * @ingroup leo
 *
 * Declaration of LeoCircularOrbitMobilityModel
 */

namespace ns3
{

/**
 * @ingroup leo
 * @brief Keep track of the orbital position and velocity of a satellite.
 *
 * This uses simple circular orbits based on the inclination of the orbital
 * plane and the altitude of the satellite.
 */
class LeoCircularOrbitMobilityModel : public GeocentricConstantPositionMobilityModel
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();
    /// constructor
    LeoCircularOrbitMobilityModel();

    /**
     * @brief Gets the speed of the node
     * @return the speed in m/s
     */
    double GetSpeed() const;

    /**
     * @brief Gets the altitude in km
     * @return the altitude in km
     */
    double GetAltitude() const;
    /**
     * @brief Sets the altitude in km
     * @param h the altitude km
     */
    void SetAltitude(double h);

    /**
     * @brief Gets the inclination
     * @return the inclination in degrees
     */
    double GetInclination() const;

    /**
     * @brief Sets the inclination
     * @param incl the inclination in degrees
     */
    void SetInclination(double incl);

    /**
     * @brief Sets the starting index of this node in the progress vector.
     *
     * Different satellites in the same orbital plane are assigned different
     * indices so that they are spaced along the orbit.  The index advances
     * by one at each resolution time step and wraps around when a full
     * orbital period has elapsed.
     *
     * @param index zero-based index into the progress vector
     */
    void SetNodeIndexAtProgressVector(uint64_t index);

    /**
     * @brief Associates this node with a shared progress vector.
     *
     * A progress vector is a precomputed table of angular offsets (in
     * degrees) representing equally spaced positions around a circular
     * orbit.  Each entry is the true anomaly relative to the ascending
     * node.  The vector is shared among all satellites in orbital planes
     * of the same altitude and inclination (to save memory).
     *
     * @param ptr shared pointer to a vector of angular offsets in degrees
     *
     * @see LeoOrbitNodeHelper::GenerateProgressVector
     */
    void SetProgressVectorPointer(const std::shared_ptr<std::vector<double>>& ptr);

    /**
     * @brief Orders the calculation of the node position, notifies course change, advances
     * the node index at the Progress Vector, and schedules the next update event.
     * @return ECEF position in meters
     */
    Vector UpdateNodePositionAndScheduleEvent();

    /**
     * @brief Returns the Geocentric Position of the Node in ECEF (cartesian)
     * @return ECEF position in meters
     */
    Vector DoGetGeocentricPosition() const override;

    // Inherited from MobilityModel
    Ptr<MobilityModel> Copy() const override
    {
        return CreateObject<LeoCircularOrbitMobilityModel>(*this);
    }

  private:
    /// Orbit height in km
    double m_orbitHeight;

    /// Inclination in rad
    double m_inclination;

    /// Longitudinal offset in rad
    double m_longitude;

    /// Offset on the orbital plane in rad
    double m_offset;

    /// Current position
    Vector3D m_position;

    /// Time resolution step between precomputed orbital positions
    Time m_resolutionTimeStep;

    /// The index of the node in the Progress Vector
    uint16_t m_nodeIndexAtProgressVector{0};

    /// Shared progress vector of angular offsets (degrees) around the orbit
    std::shared_ptr<std::vector<double>> m_progressVector;

    /**
     * @brief Returns the node current position.
     * @return ECEF position in meters
     */
    Vector DoGetPosition() const override;

    /**
     * @brief Sets the node position via argument.
     *
     * @param position position.x is the longitude of the ascending node
     *        in degrees; position.y is an offset on the orbital plane
     *        in degrees.  Both are converted to radians internally.
     */
    void DoSetPosition(const Vector& position) override;

    /**
     * @brief Returns the current velocity of the node.
     * @return velocity vector in m/s (ECEF)
     */
    Vector DoGetVelocity() const override;

    /**
     * @brief Get the normal vector of the orbital plane
     * @param t the amount of time passed since the start of the simulation
     * @return the normal vector.
     */
    Vector3D PlaneNorm(Time t) const;

    /**
     * @brief Rotates a position vector by angle 'a' around the orbital plane
     * normal, using the Rodrigues rotation formula
     * (see https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula ).
     *
     * @param a angle by which to rotate, in radians
     * @param x position vector to rotate, in meters (ECEF)
     * @param t time passed since simulation start (used to compute the
     *        current orbital plane normal)
     * @return rotated position vector in meters (ECEF)
     */
    Vector3D RotatePlane(double a, const Vector3D& x, Time t) const;

    /**
     * @brief Calculate the position at time t
     *
     * The returned Vector contains Cartesian ECEF coordinates in meters.
     * Note that this differs from DoSetPosition(), where the Vector
     * encodes angular orbital parameters (longitude and offset in
     * degrees); that discrepancy is a consequence of the MobilityModel
     * base class interface, which uses a single Vector type for both
     * input and output.
     *
     * @param t simulation time
     * @return ECEF position in meters
     */
    Vector CalcPosition(Time t) const;

    /**
     * @brief Calculate the ECEF longitude of the ascending node at simulation time t
     *
     * The orbital plane drifts westward in the ECEF frame because the Earth
     * rotates eastward, so this returns m_longitude minus the Earth-rotation
     * angle accumulated over time t.
     *
     * @param t time
     * @return longitude of the ascending node in radians (ECEF)
     */
    double CalcLongitude(Time t) const;
};
} // namespace ns3

#endif
