/*
 * Copyright (c) 2011, 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Marco Miozzo  <marco.miozzo@cttc.es>
 *         Nicola Baldo <nbaldo@cttc.es>
 *
 */

#ifndef OKUMURA_HATA_PROPAGATION_LOSS_MODEL_H
#define OKUMURA_HATA_PROPAGATION_LOSS_MODEL_H

#include "propagation-environment.h"
#include "propagation-loss-model.h"

namespace ns3
{

/**
 * @ingroup propagation
 *
 * @brief this class implements the Okumura Hata propagation loss model
 *
 * this class implements the Okumura Hata propagation loss model,
 * which is used to model open area pathloss for distances > 1 Km
 * and frequencies ranging from 150 MHz to 2.0 GHz.
 * For more information about the model, please see
 * the propagation module documentation in .rst format.
 */
class OkumuraHataPropagationLossModel : public PropagationLossModel
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    OkumuraHataPropagationLossModel();
    ~OkumuraHataPropagationLossModel() override;

    // Delete copy constructor and assignment operator to avoid misuse
    OkumuraHataPropagationLossModel(const OkumuraHataPropagationLossModel&) = delete;
    OkumuraHataPropagationLossModel& operator=(const OkumuraHataPropagationLossModel&) = delete;

    /**
     * @param a the first mobility model
     * @param b the second mobility model
     *
     * @return the loss in dBm for the propagation between
     * the two given mobility models
     */
    double GetLoss(Ptr<MobilityModel> a, Ptr<MobilityModel> b) const;

  private:
    double DoCalcRxPower(double txPowerDbm,
                         Ptr<MobilityModel> a,
                         Ptr<MobilityModel> b) const override;
    int64_t DoAssignStreams(int64_t stream) override;

    EnvironmentType m_environment; //!< Environment Scenario
    CitySize m_citySize;           //!< Size of the city
    double m_frequency;            //!< frequency in Hz
};

} // namespace ns3

#endif // OKUMURA_HATA_PROPAGATION_LOSS_MODEL_H
