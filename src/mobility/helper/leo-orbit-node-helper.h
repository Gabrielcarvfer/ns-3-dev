// Copyright (c) Tim Schubert
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Tim Schubert <ns-3-leo@timschubert.net>
// Porting: Thiago Miyazaki <miyathiago@gmail.com> <t.miyazaki@unesp.br>

#ifndef LEO_ORBIT_NODE_HELPER_H
#define LEO_ORBIT_NODE_HELPER_H

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
     * @param resolution time interval between CourseChange notifications
     *        on the installed mobility models.  Smaller values produce
     *        more frequent notifications.  Zero disables periodic
     *        notifications.
     */
    LeoOrbitNodeHelper(const Time& resolution);

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

  private:
    ObjectFactory m_nodeFactory; ///< Node factory

    /// Time interval between CourseChange notifications
    Time m_resolution;
};

}; // namespace ns3

#endif
