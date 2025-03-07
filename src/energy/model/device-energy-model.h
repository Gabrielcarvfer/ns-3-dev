/*
 * Copyright (c) 2010 Network Security Lab, University of Washington, Seattle.
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Sidharth Nabar <snabar@uw.edu>, He Wu <mdzz@u.washington.edu>
 */

#ifndef DEVICE_ENERGY_MODEL_H
#define DEVICE_ENERGY_MODEL_H

#include "ns3/node.h"
#include "ns3/object.h"
#include "ns3/ptr.h"
#include "ns3/type-id.h"

namespace ns3
{
namespace energy
{

class EnergySource;

/**
 * @ingroup energy
 * @brief Base class for device energy models.
 *
 * A device energy model should represent the energy consumption behavior of a
 * specific device. It will update remaining energy stored in the EnergySource
 * object installed on node. When energy is depleted, each DeviceEnergyModel
 * object installed on the same node will be informed by the EnergySource.
 *
 */
class DeviceEnergyModel : public Object
{
  public:
    /**
     * Callback type for ChangeState function. Devices uses this state to notify
     * DeviceEnergyModel of a state change.
     */
    typedef Callback<void, int> ChangeStateCallback;

  public:
    /**
     * @brief Get the type ID.
     * @return The object TypeId.
     */
    static TypeId GetTypeId();
    DeviceEnergyModel();
    ~DeviceEnergyModel() override;

    /**
     * @param source Pointer to energy source installed on node.
     *
     * This function sets the pointer to energy source installed on node. Should
     * be called only by DeviceEnergyModel helper classes.
     */
    virtual void SetEnergySource(Ptr<EnergySource> source) = 0;

    /**
     * @returns Total energy consumption of the device.
     *
     * DeviceEnergyModel records its own energy consumption during simulation.
     */
    virtual double GetTotalEnergyConsumption() const = 0;

    /**
     * @param newState New state the device is in.
     *
     * DeviceEnergyModel is a state based model. This function is implemented by
     * all child of DeviceEnergyModel to change the model's state. States are to
     * be defined by each child using an enum (int).
     */
    virtual void ChangeState(int newState) = 0;

    /**
     * @returns Current draw of the device, in Ampere.
     *
     * This function returns the current draw at the device in its current state.
     * This function is called from the EnergySource to obtain the total current
     * draw at any given time during simulation.
     */
    double GetCurrentA() const;

    /**
     * This function is called by the EnergySource object when energy stored in
     * the energy source is depleted. Should be implemented by child classes.
     */
    virtual void HandleEnergyDepletion() = 0;

    /**
     * This function is called by the EnergySource object when energy stored in
     * the energy source is recharged. Should be implemented by child classes.
     */
    virtual void HandleEnergyRecharged() = 0;

    /**
     * This function is called by the EnergySource object when energy stored in
     * the energy source is changed. Should be implemented by child classes.
     */
    virtual void HandleEnergyChanged() = 0;

  private:
    /**
     * @returns 0.0 as the current value, in Ampere.
     *
     * Child class does not have to implement this method if current draw for its
     * states are not know. This default method will always return 0.0A. For the
     * devices who does know the current draw of its states, this method must be
     * overwritten.
     */
    virtual double DoGetCurrentA() const;
};

} // namespace energy
} // namespace ns3

#endif /* DEVICE_ENERGY_MODEL_H */
