/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Marco Miozzo  <marco.miozzo@cttc.es>
 *
 */
#ifndef MOBILITY_BUILDING_INFO_H
#define MOBILITY_BUILDING_INFO_H

#include "building.h"

#include "ns3/box.h"
#include "ns3/constant-velocity-helper.h"
#include "ns3/mobility-model.h"
#include "ns3/object.h"
#include "ns3/ptr.h"
#include "ns3/simple-ref-count.h"

#include <map>

namespace ns3
{

/**
 * @ingroup buildings
 * @ingroup mobility

 * @brief mobility buildings information (to be used by mobility models)
 *
 * This model implements the management of scenarios where users might be
 * either indoor (e.g., houses, offices, etc.) and outdoor.
 *
 */
class MobilityBuildingInfo : public Object
{
  public:
    /**
     * @brief Get the type ID.
     *
     * @return the object TypeId
     */
    static TypeId GetTypeId();
    MobilityBuildingInfo();

    /**
     * @brief Parameterized constructor
     *
     * @param building The building in which the MobilityBuildingInfo instance would be placed
     */
    MobilityBuildingInfo(Ptr<Building> building);

    /**
     * @brief Is indoor method.
     *
     * @return true if the MobilityBuildingInfo instance is indoor, false otherwise
     */
    bool IsIndoor();

    /**
     * @brief Mark this MobilityBuildingInfo instance as indoor
     *
     * @param building the building into which the MobilityBuildingInfo instance is located
     * @param nfloor the floor number 1...nFloors at which the MobilityBuildingInfo instance
     * is located
     * @param nroomx the X room number 1...nRoomsX at which the MobilityBuildingInfo instance
     * is located
     * @param nroomy the Y room number 1...nRoomsY at which the MobilityBuildingInfo instance
     * is located
     */
    void SetIndoor(Ptr<Building> building, uint8_t nfloor, uint8_t nroomx, uint8_t nroomy);

    /**
     * @brief Mark this MobilityBuildingInfo instance as indoor
     *
     * @param nfloor the floor number 1...nFloors at which the MobilityBuildingInfo instance
     * is located
     * @param nroomx the X room number 1...nRoomsX at which the MobilityBuildingInfo instance
     * is located
     * @param nroomy the Y room number 1...nRoomsY at which the MobilityBuildingInfo instance
     * is located
     */

    void SetIndoor(uint8_t nfloor, uint8_t nroomx, uint8_t nroomy);

    /**
     * @brief Mark this MobilityBuildingInfo instance as outdoor
     */
    void SetOutdoor();

    /**
     * @brief Get the floor number at which the MobilityBuildingInfo instance is located
     *
     * @return The floor number
     */
    uint8_t GetFloorNumber();

    /**
     * @brief Get the room number along x-axis at which the MobilityBuildingInfo instance is located
     *
     * @return The room number
     */
    uint8_t GetRoomNumberX();

    /**
     * @brief Get the room number along y-axis at which the MobilityBuildingInfo instance is located
     *
     * @return The room number
     */
    uint8_t GetRoomNumberY();

    /**
     * @brief Get the building in which the MobilityBuildingInfo instance is located
     *
     * @return The building in which the MobilityBuildingInfo instance is located
     */
    Ptr<Building> GetBuilding();
    /**
     * @brief Make the given mobility model consistent, by determining whether
     * its position falls inside any of the building in BuildingList, and
     * updating accordingly the BuildingInfo aggregated with the MobilityModel.
     *
     * @param mm the mobility model to be made consistent
     */
    void MakeConsistent(Ptr<MobilityModel> mm);

  protected:
    // inherited from Object
    void DoInitialize() override;

  private:
    Ptr<Building> m_myBuilding; ///< Building
    bool m_indoor;              ///< Node position (indoor/outdoor) ?
    uint8_t m_nFloor; ///< The floor number at which the MobilityBuildingInfo instance is located
    uint8_t m_roomX; ///< The room number along x-axis at which the MobilityBuildingInfo instance is
                     ///< located
    uint8_t m_roomY; ///< The room number along y-axis at which the MobilityBuildingInfo instance is
                     ///< located
    Vector
        m_cachedPosition; ///< The node position cached after making its mobility model consistent
};

} // namespace ns3

#endif // MOBILITY_BUILDING_INFO_H
