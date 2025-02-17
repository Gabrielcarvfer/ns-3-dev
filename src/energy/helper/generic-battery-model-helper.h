/*
 * Copyright (c) 2023 Tokushima University, Japan.
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Alberto Gallegos Ramonet <alramonet@is.tokushima-u.ac.jp>
 */

#ifndef GENERIC_BATTERY_MODEL_HELPER_H_
#define GENERIC_BATTERY_MODEL_HELPER_H_

#include "energy-model-helper.h"

#include "ns3/generic-battery-model.h"
#include "ns3/node.h"

namespace ns3
{

/**
 * @ingroup energy
 * @brief Creates and assign an assortment of BatteryModels to Nodes.
 *
 */
class GenericBatteryModelHelper : public EnergySourceHelper
{
  public:
    GenericBatteryModelHelper();
    ~GenericBatteryModelHelper() override;

    /**
     * Sets one of the attributes of underlying EnergySource.
     *
     * @param name Name of attribute to set.
     * @param v Value of the attribute.
     */
    void Set(std::string name, const AttributeValue& v) override;

    /**
     * This function installs energy sources in a group of nodes in a
     * node container. An energy source (Li-Ion battery) with default values
     * is used on each node.
     *
     * @param c The node container
     * @returns An EnergySourceContainer which contains all the EnergySources.
     */
    Ptr<energy::EnergySourceContainer> Install(NodeContainer c) const;

    /**
     * This function installs an energy source (battery) into a node.
     *
     * @param node The node object.
     * @param bm The battery model that will be install to the node.
     * @returns A pointer to the energy source object used.
     */
    Ptr<energy::EnergySource> Install(Ptr<Node> node, energy::BatteryModel bm) const;

    /**
     * This function installs energy sources in a group of nodes in a
     * node container.
     *
     * @param c The node container.
     * @param bm The battery model that will be install to the nodes in the node container.
     * @returns An EnergySourceContainer which contains all the EnergySources.
     */
    energy::EnergySourceContainer Install(NodeContainer c, energy::BatteryModel bm) const;

    /**
     * This function takes an existing energy source and transform its values to form
     * a group of connected identical cells. The values of the newly formed cell block
     * depends on the connection of the cells defined by the user
     * (number of cells connected in series, number of cells connected in parallel).
     *
     * @param energySource The energy source used.
     * @param series The number of cells connected in series.
     * @param parallel The number of cells connected in parallel.
     */
    void SetCellPack(Ptr<energy::EnergySource> energySource,
                     uint8_t series,
                     uint8_t parallel) const;

    /**
     * This function takes an existing energy source container and transform the values
     * of each of its containing energy sources to form groups of connected identical cells.
     * The values of the newly formed cell blocks for each energy source
     * depends on the connection of the cells defined by the user
     * (number of cells connected in series, number of cells connected in parallel).
     *
     * @param energySourceContainer The energy source container used.
     * @param series The number of cells connected in series.
     * @param parallel The number of cells connected in parallel.
     */
    void SetCellPack(energy::EnergySourceContainer energySourceContainer,
                     uint8_t series,
                     uint8_t parallel) const;

  private:
    /**
     * Child classes of EnergySourceHelper only have to implement this function,
     * to create and aggregate an EnergySource object onto a single node. Rest of
     * the installation process (eg. installing EnergySource on set of nodes) is
     * implemented in the EnergySourceHelper base class.
     *
     * @param node Pointer to node where the energy source is to be installed.
     * @returns Pointer to the created EnergySource.
     */
    Ptr<energy::EnergySource> DoInstall(Ptr<Node> node) const override;

  private:
    ObjectFactory m_batteryModel; //!< The energy source (battery) used by this helper.
};

} // namespace ns3

#endif /* GENERIC_BATTERY_MODEL_HELPER_H_ */
