/*
 * Copyright (c) 2025 CTTC
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#ifndef WRAPAROUNDMODEL_H
#define WRAPAROUNDMODEL_H

#include "ns3/mobility-model.h"

#include <map>
#include <utility>

namespace ns3
{
class WraparoundModel : public Object
{
  public:
    /**
     * @brief Default constructor
     */
    WraparoundModel() = default;

    /**
     * Register this type with the TypeId system.
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    /**
     * @brief Get the virtual mobility model of tx with respect to rx, placed by the wraparound
     * model
     *
     * The virtual mobility model of a pair of transmitter and receiver is created once and then
     * moved to the current virtual position of the transmitter, so it is shared by the signals
     * of the pair; it is created again when the velocity of the transmitter changes, which a
     * copy cannot follow.
     *
     * @param tx Transmitter mobility model
     * @param rx Receiver Mobility model
     * @return virtual mobility model for transmitter
     */
    Ptr<MobilityModel> GetVirtualMobilityModel(Ptr<const MobilityModel> tx,
                                               Ptr<const MobilityModel> rx) const;
    /**
     * @brief Get virtual position of txPos with respect to rxPos. Each wraparound model
     * should override this function with their own implementation
     * @param tx Transmitter position
     * @param rx Receiver position
     * @return virtual position of transmitter in respect to receiver position
     */
    virtual Vector3D GetVirtualPosition(const Vector3D tx, const Vector3D rx) const;

  protected:
    void DoDispose() override;

  private:
    /// Virtual mobility model of each pair of transmitter and receiver mobility models
    mutable std::map<std::pair<Ptr<const MobilityModel>, Ptr<const MobilityModel>>,
                     Ptr<MobilityModel>>
        m_virtualMobilityModels;
};
} // namespace ns3

#endif // WRAPAROUNDMODEL_H
