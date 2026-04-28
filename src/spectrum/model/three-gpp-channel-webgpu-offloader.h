/*
 * Copyright (c) 2026, CTTC, Centre Tecnologic de Telecomunicacions de Catalunya
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

#ifndef THREE_GPP_CHANNEL_WEBGPU_OFFLOADER_H
#define THREE_GPP_CHANNEL_WEBGPU_OFFLOADER_H

#include <ns3/phased-array-model.h>
#include <ns3/simple-ref-count.h>
#include <ns3/matrix-array.h>

#include <complex>
#include <vector>
#include <map>

namespace ns3
{

/**
 * \ingroup spectrum
 * \brief A class to offload ThreeGppChannelModel calculations to GPU using WebGPU.
 */
class ThreeGppChannelWebGpuOffloader : public SimpleRefCount<ThreeGppChannelWebGpuOffloader>
{
  public:
    ThreeGppChannelWebGpuOffloader();
    ~ThreeGppChannelWebGpuOffloader();

    /**
     * \brief Offload the channel matrix computation to the GPU.
     * \param uAntenna RX antenna array
     * \param sAntenna TX antenna array
     * \param numClusters Number of clusters
     * \param raysPerCluster Number of rays per cluster
     * \param clusterPower Cluster powers
     * \param sinCosA sin(AoA) * cos(ZoA) etc.
     * \param sinSinA
     * \param cosZoA
     * \param sinCosD
     * \param sinSinD
     * \param cosZoD
     * \param raysPreComp Precomputed rays (complex values)
     * \param cluster1st Index of the first cluster (for sub-clustering)
     * \param cluster2nd Index of the second cluster (for sub-clustering)
     * \param hUsn [out] The computed channel matrix coefficients
     */
    void ComputeChannelMatrix(
        Ptr<const PhasedArrayModel> uAntenna,
        Ptr<const PhasedArrayModel> sAntenna,
        uint8_t numClusters,
        uint8_t raysPerCluster,
        const std::vector<double>& clusterPower,
        const std::vector<std::vector<double>>& sinCosA,
        const std::vector<std::vector<double>>& sinSinA,
        const std::vector<std::vector<double>>& cosZoA,
        const std::vector<std::vector<double>>& sinCosD,
        const std::vector<std::vector<double>>& sinSinD,
        const std::vector<std::vector<double>>& cosZoD,
        const std::map<std::pair<uint8_t, uint8_t>, ComplexMatrixArray>& raysPreComp,
        uint8_t cluster1st,
        uint8_t cluster2nd,
        ComplexMatrixArray& hUsn);

    /**
     * \brief Check if WebGPU is available and initialized.
     * \return true if available
     */
    bool IsAvailable() const;

  private:
    bool InitializeWebGpu();

    bool m_initialized;
};

} // namespace ns3

#endif // THREE_GPP_CHANNEL_WEBGPU_OFFLOADER_H
