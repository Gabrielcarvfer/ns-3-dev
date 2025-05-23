/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Marco Miozzo  <marco.miozzo@cttc.es>
 *         Nicola Baldo <nbaldo@cttc.es>
 *
 */

#ifndef HYBRID_BUILDINGS_PROPAGATION_LOSS_MODEL_H_
#define HYBRID_BUILDINGS_PROPAGATION_LOSS_MODEL_H_

#include "buildings-propagation-loss-model.h"

#include "ns3/propagation-environment.h"

namespace ns3
{

class OkumuraHataPropagationLossModel;
class ItuR1411LosPropagationLossModel;
class ItuR1411NlosOverRooftopPropagationLossModel;
class ItuR1238PropagationLossModel;
class Kun2600MhzPropagationLossModel;

/**
 * @ingroup buildings
 * @ingroup propagation
 *
 * @brief The HybridBuildingsPropagationModel is a compound of different models able to evaluate
 * the pathloss from 200 to 2600 MHz, in different environments and with buildings (i.e., indoor and
 * outdoor communications).
 *
 *  This model includes Hata model, COST231, ITU-R P.1411 (short range
 *  communications), ITU-R P.1238 (indoor communications), which are combined in order
 *  to be able to evaluate the pathloss under different scenarios, in detail:
 *  - Environments: urban, suburban, open-areas;
 *  - frequency: from 200 uo to 2600 MHz
 *  - short range communications vs long range communications
 *  - Node position respect to buildings: indoor, outdoor and hybrid (indoor <-> outdoor)
 *  - Building penetration loss
 *  - floors, etc...
 *
 *  @warning This model works only with MobilityBuildingInfo
 *
 */

class HybridBuildingsPropagationLossModel : public BuildingsPropagationLossModel
{
  public:
    /**
     * @brief Get the type ID.
     * @return The object TypeId.
     */
    static TypeId GetTypeId();
    HybridBuildingsPropagationLossModel();
    ~HybridBuildingsPropagationLossModel() override;

    /**
     * set the environment type
     *
     * @param env
     */
    void SetEnvironment(EnvironmentType env);

    /**
     * set the size of the city
     *
     * @param size
     */
    void SetCitySize(CitySize size);

    /**
     * set the propagation frequency
     *
     * @param freq
     */
    void SetFrequency(double freq);

    /**
     * set the rooftop height
     *
     * @param rooftopHeight
     */
    void SetRooftopHeight(double rooftopHeight);

    /**
     * @brief Compute the path loss according to the nodes position
     * using the appropriate model.
     *
     * @param a the mobility model of the source
     * @param b the mobility model of the destination
     * @returns the propagation loss (in dBm)
     */
    double GetLoss(Ptr<MobilityModel> a, Ptr<MobilityModel> b) const override;

  private:
    /**
     * Compute the path loss using either OkumuraHataPropagationLossModel
     * or Kun2600MhzPropagationLossModel.
     *
     * @param a The mobility model of the source.
     * @param b The mobility model of the destination.
     * @returns the propagation loss (in dBm).
     */
    double OkumuraHata(Ptr<MobilityModel> a, Ptr<MobilityModel> b) const;
    /**
     * Compute the path loss using either ItuR1411LosPropagationLossModel or
     * ItuR1411NlosOverRooftopPropagationLossModel.
     *
     * @param a The mobility model of the source.
     * @param b The mobility model of the destination.
     * @returns the propagation loss (in dBm).
     */
    double ItuR1411(Ptr<MobilityModel> a, Ptr<MobilityModel> b) const;
    /**
     * Compute the path loss using ItuR1238PropagationLossModel.
     *
     * @param a The mobility model of the source.
     * @param b The mobility model of the destination.
     * @returns the propagation loss (in dBm).
     */
    double ItuR1238(Ptr<MobilityModel> a, Ptr<MobilityModel> b) const;

    /// OkumuraHataPropagationLossModel
    Ptr<OkumuraHataPropagationLossModel> m_okumuraHata;
    /// ItuR1411LosPropagationLossModel
    Ptr<ItuR1411LosPropagationLossModel> m_ituR1411Los;
    /// ItuR1411NlosOverRooftopPropagationLossModel
    Ptr<ItuR1411NlosOverRooftopPropagationLossModel> m_ituR1411NlosOverRooftop;
    /// ItuR1238PropagationLossModel
    Ptr<ItuR1238PropagationLossModel> m_ituR1238;
    /// Kun2600MhzPropagationLossModel
    Ptr<Kun2600MhzPropagationLossModel> m_kun2600Mhz;

    double m_itu1411NlosThreshold; ///< in meters (switch Los -> NLoS)
    double m_rooftopHeight;        ///< Roof Height (in meters)
    double m_frequency;            ///< Operation frequency
};

} // namespace ns3

#endif /* HYBRID_BUILDINGS_PROPAGATION_LOSS_MODEL_H_ */
