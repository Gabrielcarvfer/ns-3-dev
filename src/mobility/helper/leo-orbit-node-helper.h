// Copyright (c) Tim Schubert
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Tim Schubert <ns-3-leo@timschubert.net>
// Porting: Thiago Miyazaki <miyathiago@gmail.com> <t.miyazaki@unesp.br>

#ifndef LEO_ORBIT_NODE_HELPER_H
#define LEO_ORBIT_NODE_HELPER_H

#include "ns3/leo-circular-orbit-mobility-model.h"
#include "ns3/leo-orbit.h"
#include "ns3/node-container.h"
#include "ns3/object-factory.h"

#include <string>

/**
 * @file
 * @ingroup leo
 */

namespace ns3
{

/**
 * @ingroup leo
 * @brief Builds a node container of nodes with LEO positions using a list of
 * orbit definitions.
 */
class LeoOrbitNodeHelper
{
  public:
    /**
     * @brief Construct a LEO Orbit Node Helper.
     *
     * @param timeStep time resolution for the progress vector.  Smaller
     *        values produce finer-grained orbital positions but use more
     *        memory.  Higher orbital speeds require smaller time steps.
     */
    LeoOrbitNodeHelper(const Time& timeStep);

    /// destructor
    virtual ~LeoOrbitNodeHelper();

    /**
     * @brief Create satellite nodes and install orbital mobility from a CSV file.
     *
     * The CSV file contains one or more lines, each with 4 comma-separated columns:
     *   - altitude (km, from earth surface)
     *   - inclination (degrees relative to equator)
     *   - number of orbital planes
     *   - number of satellites per plane
     *
     * Planes are evenly spaced by 360 degrees divided by the number of planes, starting from the
     * prime meridian.
     *
     * One example: 1150.0,53.0,32,50, represents 32 orbits with 50 satellites each, with 53 degrees
     * of inclination in respect to the equator, and with an altitude of 1150 km.
     *
     * @param orbitFile path to orbit definitions file
     * @returns a node container containing the created nodes
     */
    NodeContainer CreateNodesAndInstallMobility(const std::string& orbitFile);

    /**
     * @brief Create satellite nodes and install orbital mobility from a vector
     *        of orbital parameters.
     * @param orbits orbit definitions
     * @returns a node container containing the created nodes
     */
    NodeContainer CreateNodesAndInstallMobility(const std::vector<LeoOrbit>& orbits);

    /**
     * @brief Create satellite nodes and install orbital mobility from a single
     *        orbit definition.
     * @param orbit orbit definition
     * @returns a node container containing the created nodes
     */
    NodeContainer CreateNodesAndInstallMobility(const LeoOrbit& orbit);

    /**
     * Set an attribute for each node
     *
     * @param name name of the attribute
     * @param value value of the attribute
     */
    void SetAttribute(std::string name, const AttributeValue& value);

    /**
     * @brief Generate a progress vector for the given orbit.
     *
     * A progress vector is a precomputed table of angular offsets (in
     * degrees) representing equally spaced positions around a circular
     * orbit.  Each entry is the true anomaly relative to the ascending
     * node, sampled at intervals of the helper's time step.  The number
     * of entries equals one full orbital period divided by the time step.
     *
     * The vector is shared among all satellites at the same altitude
     * (and thus the same orbital speed) to save memory.
     *
     * @param orbit orbital parameters (altitude, inclination, etc.)
     * @return shared pointer to a vector of angular offsets in degrees
     */
    std::shared_ptr<std::vector<double>> GenerateProgressVector(const LeoOrbit& orbit) const;

  private:
    ObjectFactory m_nodeFactory; ///< Node factory

    /// The period at which the mobility model is updated. Higher orbital speeds require smaller
    /// steps.
    Time m_timeStep;
};

}; // namespace ns3

#endif
