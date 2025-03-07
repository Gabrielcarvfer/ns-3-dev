/*
 * Copyright (c) 2010 Network Security Lab, University of Washington, Seattle.
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Sidharth Nabar <snabar@uw.edu>, He Wu <mdzz@u.washington.edu>
 */

#ifndef ENERGY_MODEL_HELPER_H
#define ENERGY_MODEL_HELPER_H

#include "energy-source-container.h"

#include "ns3/attribute.h"
#include "ns3/device-energy-model-container.h"
#include "ns3/device-energy-model.h"
#include "ns3/energy-source.h"
#include "ns3/net-device-container.h"
#include "ns3/net-device.h"
#include "ns3/node-container.h"
#include "ns3/object-factory.h"
#include "ns3/ptr.h"

namespace ns3
{

/**
 * @ingroup energy
 * @brief Creates EnergySource objects.
 *
 * This class creates and installs an energy source onto network nodes.
 * Multiple sources can exist on a network node.
 *
 */
class EnergySourceHelper
{
  public:
    virtual ~EnergySourceHelper();

    /**
     * @param name Name of attribute to set.
     * @param v Value of the attribute.
     *
     * Sets one of the attributes of underlying EnergySource.
     */
    virtual void Set(std::string name, const AttributeValue& v) = 0;

    /**
     * @param node Pointer to the node where EnergySource will be installed.
     * @returns An EnergySourceContainer which contains all the EnergySources.
     *
     * This function installs an EnergySource onto a node.
     */
    energy::EnergySourceContainer Install(Ptr<Node> node) const;

    /**
     * @param c List of nodes where EnergySource will be installed.
     * @returns An EnergySourceContainer which contains all the EnergySources.
     *
     * This function installs an EnergySource onto a list of nodes.
     */
    energy::EnergySourceContainer Install(NodeContainer c) const;

    /**
     * @param nodeName Name of node where EnergySource will be installed.
     * @returns An EnergySourceContainer which contains all the EnergySources.
     *
     * This function installs an EnergySource onto a node.
     */
    energy::EnergySourceContainer Install(std::string nodeName) const;

    /**
     * @brief This function installs an EnergySource on all nodes in simulation.
     *
     * @returns An EnergySourceContainer which contains all the EnergySources.
     */
    energy::EnergySourceContainer InstallAll() const;

  private:
    /**
     * @param node Pointer to node where the energy source is to be installed.
     * @returns Pointer to the created EnergySource.
     *
     * Child classes of EnergySourceHelper only have to implement this function,
     * to create and aggregate an EnergySource object onto a single node. Rest of
     * the installation process (eg. installing EnergySource on set of nodes) is
     * implemented in the EnergySourceHelper base class.
     */
    virtual Ptr<energy::EnergySource> DoInstall(Ptr<Node> node) const = 0;
};

/**
 * @ingroup energy
 * @brief Creates DeviceEnergyModel objects.
 *
 * This class helps to create and install DeviceEnergyModel onto NetDevice. A
 * DeviceEnergyModel is connected to a NetDevice (or PHY object) by callbacks.
 * Note that DeviceEnergyModel objects are *not* aggregated onto the node. They
 * can be accessed through the EnergySource object, which *is* aggregated onto
 * the node.
 *
 */
class DeviceEnergyModelHelper
{
  public:
    virtual ~DeviceEnergyModelHelper();

    /**
     * @param name Name of attribute to set.
     * @param v Value of the attribute.
     *
     * Sets one of the attributes of underlying DeviceEnergyModel.
     */
    virtual void Set(std::string name, const AttributeValue& v) = 0;

    /**
     * @param device Pointer to the NetDevice to install DeviceEnergyModel.
     * @param source The EnergySource the DeviceEnergyModel will be using.
     * @returns An DeviceEnergyModelContainer contains all the DeviceEnergyModels.
     *
     * Installs an DeviceEnergyModel with a specified EnergySource onto a
     * xNetDevice.
     */
    energy::DeviceEnergyModelContainer Install(Ptr<NetDevice> device,
                                               Ptr<energy::EnergySource> source) const;

    /**
     * @param deviceContainer List of NetDevices to be install DeviceEnergyModel
     * objects.
     * @param sourceContainer List of EnergySource the DeviceEnergyModel will be
     * using.
     * @returns An DeviceEnergyModelContainer contains all the DeviceEnergyModels.
     *
     * Installs DeviceEnergyModels with specified EnergySources onto a list of
     * NetDevices.
     */
    energy::DeviceEnergyModelContainer Install(NetDeviceContainer deviceContainer,
                                               energy::EnergySourceContainer sourceContainer) const;

  private:
    /**
     * @param device The net device corresponding to DeviceEnergyModel object.
     * @param source The EnergySource the DeviceEnergyModel will be using.
     * @returns Pointer to the created DeviceEnergyModel.
     *
     * Child classes of DeviceEnergyModelHelper only have to implement this
     * function, to create and aggregate an DeviceEnergyModel object onto a single
     * node. The rest of the installation process (eg. installing EnergySource on
     * set of nodes) is implemented in the DeviceEnergyModelHelper base class.
     */
    virtual Ptr<energy::DeviceEnergyModel> DoInstall(Ptr<NetDevice> device,
                                                     Ptr<energy::EnergySource> source) const = 0;
};

} // namespace ns3

#endif /* ENERGY_MODEL_HELPER_H */
