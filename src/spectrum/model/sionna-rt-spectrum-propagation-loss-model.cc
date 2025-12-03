// Copyright (c) 2025 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
// author: Amir Ashtari aashtari@cttc.es
// SPDX-License-Identifier: GPL-2.0-only

#include "sionna-rt-spectrum-propagation-loss-model.h"

#include "sionna-rt-channel-model.h"
#include "spectrum-signal-parameters.h"

#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/net-device.h"
#include "ns3/node.h"
#include "ns3/pointer.h"
#include "ns3/simulator.h"
#include "ns3/string.h"

#include <map>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("SionnaRtSpectrumPropagationLossModel");

NS_OBJECT_ENSURE_REGISTERED(SionnaRtSpectrumPropagationLossModel);

SionnaRtSpectrumPropagationLossModel::SionnaRtSpectrumPropagationLossModel()
{
    NS_LOG_FUNCTION(this);
}

SionnaRtSpectrumPropagationLossModel::~SionnaRtSpectrumPropagationLossModel()
{
    NS_LOG_FUNCTION(this);
}

void
SionnaRtSpectrumPropagationLossModel::DoDispose()
{
    m_longTermMap.clear();
    m_channelModel = nullptr;
}

TypeId
SionnaRtSpectrumPropagationLossModel::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::SionnaRtSpectrumPropagationLossModel")
            .SetParent<PhasedArraySpectrumPropagationLossModel>()
            .SetGroupName("Spectrum")
            .AddConstructor<SionnaRtSpectrumPropagationLossModel>()
            .AddAttribute(
                "ChannelModel",
                "The channel model. It needs to implement the MatrixBasedChannelModel interface",
                StringValue("ns3::SionnaRtChannelModel"),
                MakePointerAccessor(&SionnaRtSpectrumPropagationLossModel::SetChannelModel,
                                    &SionnaRtSpectrumPropagationLossModel::GetChannelModel),
                MakePointerChecker<MatrixBasedChannelModel>());
    return tid;
}

void
SionnaRtSpectrumPropagationLossModel::SetChannelModel(Ptr<MatrixBasedChannelModel> channel)
{
    m_channelModel = channel;
}

Ptr<MatrixBasedChannelModel>
SionnaRtSpectrumPropagationLossModel::GetChannelModel() const
{
    return m_channelModel;
}

double
SionnaRtSpectrumPropagationLossModel::GetFrequency() const
{
    DoubleValue freq;
    m_channelModel->GetAttribute("Frequency", freq);
    return freq.Get();
}

void
SionnaRtSpectrumPropagationLossModel::SetChannelModelAttribute(const std::string& name,
                                                               const AttributeValue& value)
{
    m_channelModel->SetAttribute(name, value);
}

void
SionnaRtSpectrumPropagationLossModel::GetChannelModelAttribute(const std::string& name,
                                                               AttributeValue& value) const
{
    m_channelModel->GetAttribute(name, value);
}

Ptr<const MatrixBasedChannelModel::Complex3DVector>
SionnaRtSpectrumPropagationLossModel::CalcLongTerm(
    Ptr<const MatrixBasedChannelModel::ChannelMatrix> params,
    Ptr<const PhasedArrayModel> sAnt,
    Ptr<const PhasedArrayModel> uAnt) const
{
    NS_LOG_FUNCTION(this);

    const PhasedArrayModel::ComplexVector& sW = sAnt->GetBeamformingVectorRef();
    const PhasedArrayModel::ComplexVector& uW = uAnt->GetBeamformingVectorRef();
    size_t sAntNumElems = sW.GetSize();
    size_t uAntNumElems = uW.GetSize();
    NS_ASSERT(uAntNumElems == params->m_channel.GetNumRows());
    NS_ASSERT(sAntNumElems == params->m_channel.GetNumCols());
    NS_LOG_DEBUG("CalcLongTerm with " << uW.GetSize() << " u antenna elements and " << sW.GetSize()
                                      << " s antenna elements, and with "
                                      << " s ports: " << sAnt->GetNumPorts()
                                      << " u ports: " << uAnt->GetNumPorts());
    NS_ASSERT_MSG((sAnt != nullptr) && (uAnt != nullptr), "Improper call to the method");
    size_t numClusters = params->m_channel.GetNumPages();
    // create and initialize the size of the longTerm 3D matrix
    Ptr<MatrixBasedChannelModel::Complex3DVector> longTerm =
        Create<MatrixBasedChannelModel::Complex3DVector>(uAnt->GetNumPorts(),
                                                         sAnt->GetNumPorts(),
                                                         numClusters);
    // Calculate long term uW * Husn * sW, the result is a matrix
    // with the dimensions #uPorts, #sPorts, #cluster
    for (auto sPortIdx = 0; sPortIdx < sAnt->GetNumPorts(); sPortIdx++)
    {
        for (auto uPortIdx = 0; uPortIdx < uAnt->GetNumPorts(); uPortIdx++)
        {
            for (size_t cIndex = 0; cIndex < numClusters; cIndex++)
            {
                longTerm->Elem(uPortIdx, sPortIdx, cIndex) =
                    CalculateLongTermComponent(params, sAnt, uAnt, sPortIdx, uPortIdx, cIndex);
            }
        }
    }
    return longTerm;
}

std::complex<double>
SionnaRtSpectrumPropagationLossModel::CalculateLongTermComponent(
    Ptr<const MatrixBasedChannelModel::ChannelMatrix> params,
    Ptr<const PhasedArrayModel> sAnt,
    Ptr<const PhasedArrayModel> uAnt,
    uint16_t sPortIdx,
    uint16_t uPortIdx,
    uint16_t cIndex) const
{
    NS_LOG_FUNCTION(this);
    const PhasedArrayModel::ComplexVector& sW = sAnt->GetBeamformingVectorRef();
    const PhasedArrayModel::ComplexVector& uW = uAnt->GetBeamformingVectorRef();
    auto sPortElems = sAnt->GetNumElemsPerPort();
    auto uPortElems = uAnt->GetNumElemsPerPort();
    auto startS = sAnt->ArrayIndexFromPortIndex(sPortIdx, 0);
    auto startU = uAnt->ArrayIndexFromPortIndex(uPortIdx, 0);
    std::complex<double> txSum(0, 0);
    // limiting multiplication operations to the port location
    auto sIndex = startS;
    // The sub-array partition model is adopted for TXRU virtualization,
    // as described in Section 5.2.2 of 3GPP TR 36.897,
    // and so equal beam weights are used for all the ports.
    // Support of the full-connection model for TXRU virtualization would need extensions.
    const auto uElemsPerPort = uAnt->GetHElemsPerPort();
    const auto sElemsPerPort = sAnt->GetHElemsPerPort();
    for (size_t tIndex = 0; tIndex < sPortElems; tIndex++, sIndex++)
    {
        std::complex<double> rxSum(0, 0);
        auto uIndex = startU;
        for (size_t rIndex = 0; rIndex < uPortElems; rIndex++, uIndex++)
        {
            rxSum += uW[uIndex - startU] * params->m_channel(uIndex, sIndex, cIndex);
            auto testV = (rIndex % uElemsPerPort);
            auto ptInc = uElemsPerPort - 1;
            if (testV == ptInc)
            {
                auto incVal = uAnt->GetNumColumns() - uElemsPerPort;
                uIndex += incVal; // Increment by a factor to reach next column in a port
            }
        }

        txSum += sW[sIndex - startS] * rxSum;
        auto testV = (tIndex % sElemsPerPort);
        auto ptInc = sElemsPerPort - 1;
        if (testV == ptInc)
        {
            size_t incVal = sAnt->GetNumColumns() - sElemsPerPort;
            sIndex += incVal; // Increment by a factor to reach next column in a port
        }
    }
    return txSum;
}

Ptr<SpectrumSignalParameters>
SionnaRtSpectrumPropagationLossModel::CalcBeamformingGain(
    Ptr<const SpectrumSignalParameters> params,
    Ptr<const MatrixBasedChannelModel::Complex3DVector> longTerm,
    Ptr<const MatrixBasedChannelModel::ChannelMatrix> channelMatrix,
    Ptr<const MatrixBasedChannelModel::ChannelParams> channelParams,
    const Vector& sSpeed,
    const ns3::Vector& uSpeed,
    uint8_t numTxPorts,
    uint8_t numRxPorts,
    bool isReverse) const

{
    Ptr<const SionnaRtChannelModel::SionnaRtChannelParams> sionnaRtChannelParams =
        DynamicCast<const SionnaRtChannelModel::SionnaRtChannelParams>(channelParams);

    NS_LOG_FUNCTION(this);
    Ptr<SpectrumSignalParameters> rxParams = params->Copy();

    size_t numCluster = channelMatrix->m_channel.GetNumPages();

    PhasedArrayModel::ComplexVector doppler(numCluster);

    // Make sure that all the structures that are passed to this function
    // are of the correct dimensions before using the operator [].
    NS_ASSERT(numCluster <=
              sionnaRtChannelParams->m_angle[MatrixBasedChannelModel::ZOA_INDEX].size());

    NS_ASSERT(numCluster <=
              sionnaRtChannelParams->m_angle[MatrixBasedChannelModel::ZOD_INDEX].size());

    NS_ASSERT(numCluster <=
              sionnaRtChannelParams->m_angle[MatrixBasedChannelModel::AOA_INDEX].size());

    NS_ASSERT(numCluster <=
              sionnaRtChannelParams->m_angle[MatrixBasedChannelModel::AOD_INDEX].size());

    NS_ASSERT(numCluster <= longTerm->GetNumPages());

    double slotTime = Simulator::Now().GetSeconds();
    double factor = 2 * M_PI * slotTime * GetFrequency() / 3e8;
    for (size_t cIndex = 0; cIndex < numCluster; cIndex++)
    {
        doppler[cIndex] =
            std::complex<double>(cos(sionnaRtChannelParams->m_doppler[cIndex] * factor),
                                 sin(sionnaRtChannelParams->m_doppler[cIndex] * factor));
        ;
    }

    // set the channel matrix
    rxParams->spectrumChannelMatrix = GenSpectrumChannelMatrix(rxParams->psd,
                                                               longTerm,
                                                               channelMatrix,
                                                               channelParams,
                                                               doppler,
                                                               numTxPorts,
                                                               numRxPorts,
                                                               isReverse);
    NS_ASSERT_MSG(rxParams->psd->GetValuesN() == rxParams->spectrumChannelMatrix->GetNumPages(),
                  "RX PSD and the spectrum channel matrix should have the same number of RBs ");

    // Calculate RX PSD from the spectrum channel matrix H and
    // the precoding matrix P as: PSD = (H*P)^h * (H*P)
    Ptr<const ComplexMatrixArray> p;
    if (!rxParams->precodingMatrix)
    {
        // When the precoding matrix P is not set, we create one with a single column
        ComplexMatrixArray page =
            ComplexMatrixArray(rxParams->spectrumChannelMatrix->GetNumCols(), 1, 1);
        // Initialize it to the inverse square of the number of txPorts
        page.Elem(0, 0, 0) = 1.0 / sqrt(rxParams->spectrumChannelMatrix->GetNumCols());
        for (size_t rowI = 0; rowI < rxParams->spectrumChannelMatrix->GetNumCols(); rowI++)
        {
            page.Elem(rowI, 0, 0) = page.Elem(0, 0, 0);
        }
        // Replicate vector to match the number of RBGs
        p = Create<const ComplexMatrixArray>(
            page.MakeNCopies(rxParams->spectrumChannelMatrix->GetNumPages()));
    }
    else
    {
        p = rxParams->precodingMatrix;
    }
    // When we have the precoding matrix P, we first do
    // H(rxPorts,txPorts,numRbs) x P(txPorts,txStreams,numRbs) = HxP(rxPorts,txStreams,numRbs)
    MatrixBasedChannelModel::Complex3DVector hP = *rxParams->spectrumChannelMatrix * *p;

    // Then (HxP)^h dimensions are (txStreams, rxPorts, numRbs)
    // MatrixBasedChannelModel::Complex3DVector hPHerm = hP.HermitianTranspose();

    // Finally, (HxP)^h x (HxP) = PSD(txStreams, txStreams, numRbs)
    // MatrixBasedChannelModel::Complex3DVector psd = hPHerm * hP;

    // And the received psd is the Trace(PSD).
    // To avoid wasting computations, we only compute the main diagonal of hPHerm*hP
    for (uint32_t rbIdx = 0; rbIdx < rxParams->psd->GetValuesN(); ++rbIdx)
    {
        (*rxParams->psd)[rbIdx] = 0.0;
        for (size_t rxPort = 0; rxPort < hP.GetNumRows(); ++rxPort)
        {
            for (size_t txStream = 0; txStream < hP.GetNumCols(); ++txStream)
            {
                (*rxParams->psd)[rbIdx] +=
                    std::real(std::conj(hP(rxPort, txStream, rbIdx)) * hP(rxPort, txStream, rbIdx));
            }
        }
    }
    return rxParams;
}

Ptr<MatrixBasedChannelModel::Complex3DVector>
SionnaRtSpectrumPropagationLossModel::GenSpectrumChannelMatrix(
    Ptr<SpectrumValue> inPsd,
    Ptr<const MatrixBasedChannelModel::Complex3DVector> longTerm,
    Ptr<const MatrixBasedChannelModel::ChannelMatrix> channelMatrix,
    Ptr<const MatrixBasedChannelModel::ChannelParams> channelParams,
    PhasedArrayModel::ComplexVector doppler,
    uint8_t numTxPorts,
    uint8_t numRxPorts,
    bool isReverse) const
{
    size_t numCluster = channelMatrix->m_channel.GetNumPages();
    auto numRb = inPsd->GetValuesN();

    auto directionalLongTerm = isReverse ? longTerm->Transpose() : (*longTerm);

    Ptr<MatrixBasedChannelModel::Complex3DVector> chanSpct =
        Create<MatrixBasedChannelModel::Complex3DVector>(numRxPorts, numTxPorts, (uint16_t)numRb);

    // Precompute the delay until numRb, numCluster or RB width changes
    // Whenever the channelParams is updated, the number of numRbs, numClusters
    // and RB width (12*SCS) are reset, ensuring these values are updated too
    double rbWidth = inPsd->ConstBandsBegin()->fh - inPsd->ConstBandsBegin()->fl;

    if (channelParams->m_cachedDelaySincos.GetNumRows() != numRb ||
        channelParams->m_cachedDelaySincos.GetNumCols() != numCluster ||
        channelParams->m_cachedRbWidth != rbWidth)
    {
        channelParams->m_cachedRbWidth = rbWidth;
        channelParams->m_cachedDelaySincos = ComplexMatrixArray(numRb, numCluster);
        auto sbit = inPsd->ConstBandsBegin(); // band iterator
        for (unsigned i = 0; i < numRb; i++)
        {
            double fsb = (*sbit).fc; // center frequency of the sub-band
            for (std::size_t cIndex = 0; cIndex < numCluster; cIndex++)
            {
                double delay = -2 * M_PI * fsb * (channelParams->m_delay[cIndex]);
                channelParams->m_cachedDelaySincos(i, cIndex) =
                    std::complex<double>(cos(delay), sin(delay));
            }
            sbit++;
        }
    }

    // Compute the product between the doppler and the delay sincos
    auto delaySincosCopy = channelParams->m_cachedDelaySincos;
    for (size_t iRb = 0; iRb < inPsd->GetValuesN(); iRb++)
    {
        for (std::size_t cIndex = 0; cIndex < numCluster; cIndex++)
        {
            delaySincosCopy(iRb, cIndex) *= doppler[cIndex];
        }
    }

    // If "params" (ChannelMatrix) and longTerm were computed for the reverse direction (e.g. this
    // is a DL transmission but params and longTerm were last updated during UL), then the elements
    // in longTerm start from different offsets.

    auto vit = inPsd->ValuesBegin(); // psd iterator
    size_t iRb = 0;
    // Compute the frequency-domain channel matrix
    while (vit != inPsd->ValuesEnd())
    {
        if ((*vit) != 0.00)
        {
            auto sqrtVit = sqrt(*vit);
            for (auto rxPortIdx = 0; rxPortIdx < numRxPorts; rxPortIdx++)
            {
                for (auto txPortIdx = 0; txPortIdx < numTxPorts; txPortIdx++)
                {
                    std::complex<double> subsbandGain(0.0, 0.0);
                    for (size_t cIndex = 0; cIndex < numCluster; cIndex++)
                    {
                        subsbandGain += directionalLongTerm(rxPortIdx, txPortIdx, cIndex) *
                                        delaySincosCopy(iRb, cIndex);
                    }
                    // Multiply with the square root of the input PSD so that the norm (absolute
                    // value squared) of chanSpct will be the output PSD
                    chanSpct->Elem(rxPortIdx, txPortIdx, iRb) = sqrtVit * subsbandGain;
                }
            }
        }
        vit++;
        iRb++;
    }
    return chanSpct;
}

Ptr<const MatrixBasedChannelModel::Complex3DVector>
SionnaRtSpectrumPropagationLossModel::GetLongTerm(
    Ptr<const MatrixBasedChannelModel::ChannelMatrix> channelMatrix,
    Ptr<const PhasedArrayModel> aPhasedArrayModel,
    Ptr<const PhasedArrayModel> bPhasedArrayModel) const
{
    Ptr<const MatrixBasedChannelModel::Complex3DVector>
        longTerm; // vector containing the long term component for each cluster

    // check if the channel matrix was generated considering a as the s-node and
    // b as the u-node or vice-versa
    auto isReverse =
        channelMatrix->IsReverse(aPhasedArrayModel->GetId(), bPhasedArrayModel->GetId());
    auto sAntenna = isReverse ? bPhasedArrayModel : aPhasedArrayModel;
    auto uAntenna = isReverse ? aPhasedArrayModel : bPhasedArrayModel;

    PhasedArrayModel::ComplexVector sW;
    PhasedArrayModel::ComplexVector uW;
    if (!isReverse)
    {
        sW = aPhasedArrayModel->GetBeamformingVector();
        uW = bPhasedArrayModel->GetBeamformingVector();
    }
    else
    {
        sW = bPhasedArrayModel->GetBeamformingVector();
        uW = aPhasedArrayModel->GetBeamformingVector();
    }

    bool update = false;   // indicates whether the long term has to be updated
    bool notFound = false; // indicates if the long term has not been computed yet

    // compute the long term key, the key is unique for each tx-rx pair
    uint64_t longTermId =
        MatrixBasedChannelModel::GetKey(aPhasedArrayModel->GetId(), bPhasedArrayModel->GetId());

    // look for the long term in the map and check if it is valid
    if (m_longTermMap.find(longTermId) != m_longTermMap.end())
    {
        NS_LOG_DEBUG("found the long term component in the map");
        longTerm = m_longTermMap[longTermId]->m_longTerm;

        // check if the channel matrix has been updated
        // or the s beam has been changed
        // or the u beam has been changed
        update = (m_longTermMap[longTermId]->m_channel->m_generatedTime !=
                      channelMatrix->m_generatedTime ||
                  m_longTermMap[longTermId]->m_sW != sW || m_longTermMap[longTermId]->m_uW != uW);
    }
    else
    {
        NS_LOG_DEBUG("long term component NOT found");
        notFound = true;
    }

    if (update || notFound)
    {
        NS_LOG_DEBUG("compute the long term");
        // compute the long term component
        longTerm = CalcLongTerm(channelMatrix, sAntenna, uAntenna);
        Ptr<LongTerm> longTermItem = Create<LongTerm>();
        longTermItem->m_longTerm = longTerm;
        longTermItem->m_channel = channelMatrix;
        longTermItem->m_sW = std::move(sW);
        longTermItem->m_uW = std::move(uW);
        // store the long term to reduce computation load
        // only the small scale fading needs to be updated if the large scale parameters and antenna
        // weights remain unchanged.
        m_longTermMap[longTermId] = longTermItem;
    }

    return longTerm;
}

Ptr<SpectrumSignalParameters>
SionnaRtSpectrumPropagationLossModel::DoCalcRxPowerSpectralDensity(
    Ptr<const SpectrumSignalParameters> spectrumSignalParams,
    Ptr<const MobilityModel> a,
    Ptr<const MobilityModel> b,
    Ptr<const PhasedArrayModel> aPhasedArrayModel,
    Ptr<const PhasedArrayModel> bPhasedArrayModel) const
{
    NS_LOG_FUNCTION(this << spectrumSignalParams << a << b << aPhasedArrayModel
                         << bPhasedArrayModel);

    uint32_t aId = a->GetObject<Node>()->GetId(); // id of the node a
    uint32_t bId = b->GetObject<Node>()->GetId(); // id of the node b
    NS_ASSERT_MSG(aPhasedArrayModel, "Antenna not found for node " << aId);
    NS_LOG_DEBUG("a node " << aId << " antenna " << aPhasedArrayModel);
    NS_ASSERT_MSG(bPhasedArrayModel, "Antenna not found for node " << bId);
    NS_LOG_DEBUG("b node " << bId << " antenna " << bPhasedArrayModel);

    Ptr<const MatrixBasedChannelModel::ChannelMatrix> channelMatrix =
        m_channelModel->GetChannel(a, b, aPhasedArrayModel, bPhasedArrayModel);
    Ptr<const MatrixBasedChannelModel::ChannelParams> channelParams =
        m_channelModel->GetParams(a, b);

    // retrieve the long term component
    Ptr<const MatrixBasedChannelModel::Complex3DVector> longTerm =
        GetLongTerm(channelMatrix, aPhasedArrayModel, bPhasedArrayModel);

    auto isReverse =
        channelMatrix->IsReverse(aPhasedArrayModel->GetId(), bPhasedArrayModel->GetId());

    // apply the beamforming gain
    return CalcBeamformingGain(spectrumSignalParams,
                               longTerm,
                               channelMatrix,
                               channelParams,
                               a->GetVelocity(),
                               b->GetVelocity(),
                               aPhasedArrayModel->GetNumPorts(),
                               bPhasedArrayModel->GetNumPorts(),
                               isReverse);
}

int64_t
SionnaRtSpectrumPropagationLossModel::DoAssignStreams(int64_t stream)
{
    return 0;
}

void
SionnaRtSpectrumPropagationLossModel::SetRtPathSolverConfig(
    SionnaRtChannelModel::RtPathSolverConfig configs)
{
    Ptr<SionnaRtChannelModel> sionnaRtChannelParams =
        DynamicCast<SionnaRtChannelModel>(m_channelModel);

    if (sionnaRtChannelParams)
    {
        sionnaRtChannelParams->SetRtPathSolverConfig(configs);
    }
}

} // namespace ns3
