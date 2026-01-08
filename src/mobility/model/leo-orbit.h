// Copyright (c) Tim Schubert
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Tim Schubert <ns-3-leo@timschubert.net>
// Porting: Thiago Miyazaki <miyathiago@gmail.com> <t.miyazaki@unesp.br>

#ifndef LEO_ORBIT_H
#define LEO_ORBIT_H

#include "ns3/uinteger.h"

/**
 * @file
 * @ingroup leo
 */

namespace ns3
{
class LeoOrbit;

/**
 * @brief Serialize an orbit parameters in csv format
 *
 * Serialized orbit parameters as a comma-separated list in this order:
 *   - altitude (km, from earth surface)
 *   - inclination (degrees relative to equator)
 *   - number of orbital planes
 *   - number of satellites per plane
 *
 * Planes are evenly spaced by 360 degrees divided by the number of planes, starting from the prime
 * meridian.
 *
 * Example: "700,98,2,12" (700 km altitude, 98 degrees of inclination, 2 planes, 12
 * satellites/plane).
 *
 * @return the stream
 * @param os the stream
 * @param [in] orbit is an Orbit
 */
std::ostream& operator<<(std::ostream& os, const LeoOrbit& orbit);

/**
 * @brief Deserialize an orbit in csv format
 *
 * Deserialized orbit parameters as a comma-separated list in this order:
 *   - altitude (km, from earth surface)
 *   - inclination (degrees relative to equator)
 *   - number of orbital planes
 *   - number of satellites per plane
 *
 * Planes are evenly spaced by 360 degrees divided by the number of planes, starting from the prime
 * meridian.
 *
 * Example: "700,98,2,12" (700 km altitude, 98 degrees of inclination, 2 planes, 12
 * satellites/plane).
 *
 * If deserialization fails due to unexpected fields/comments/text, the orbit is initialized with
 * all zero fields. Input stream error flags are cleared to continue parsing, so comments can be
 * added, except in lines with orbital parameters.
 *
 * @return the stream
 * @param is the stream
 * @param [out] orbit is the Orbit
 */
std::istream& operator>>(std::istream& is, LeoOrbit& orbit);

/**
 * @ingroup leo
 * @brief Orbit definition
 */
class LeoOrbit
{
  public:
    /// constructor
    LeoOrbit() = default;

    /**
     * @brief Constructor
     * @param a altitude (km, from earth surface)
     * @param i inclination (degrees)
     * @param p number planes
     * @param s number of satellites in plane
     */
    LeoOrbit(double a, double i, double p, double s)
        : alt(a),
          inc(i),
          planes(p),
          sats(s)
    {
    }

    /// Altitude of orbit (km, from earth surface)
    double alt;
    /// Inclination of orbit (degrees)
    double inc;
    /// Number of planes with that altitude and inclination
    uint16_t planes;
    /// Number of satellites in those planes
    uint16_t sats;
};

}; // namespace ns3

#endif
