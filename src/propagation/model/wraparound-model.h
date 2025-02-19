/*
 * Copyright (c) 2025 CTTC
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#ifndef WRAPAROUND_H
#define WRAPAROUND_H

#include "ns3/object.h"
#include "ns3/vector.h"

namespace ns3
{
/**
 * @ingroup mobility
 * @brief Wrap nodes around in space
 *
 * It is used by MobilityModel to determine a virtual position
 * after wraparound and calculate the distance between a point
 * of reference and the virtual position.
 *
 * All space coordinates in this class and its subclasses are
 * understood to be meters or meters/s. i.e., they are all
 * metric international units.
 *
 * This is a base class for all specific wraparound models.
 */
class WraparoundModel : public Object
{
  public:
    /**
     * Register this type with the TypeId system.
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    /**
     * @brief Default constructor
     */
    WraparoundModel();

    /**
     * @brief Calculate distance after wraparound between two points
     * @param a position of point a
     * @param b position of point b
     * @return wraparound distance between a and b
     */
    virtual double CalculateDistance(const Vector3D& a, const Vector3D& b) const = 0;

    /**
     * @brief Get virtual position of absPos2 with respect to absPos1
     * @param absPos1
     * @param absPos2
     * @return virtual position of absPos2
     */
    virtual Vector3D GetRelativeVirtualPosition(const Vector3D& absPos1,
                                                const Vector3D& absPos2) const = 0;
};
} // namespace ns3

#endif /* WRAPAROUND_H */
