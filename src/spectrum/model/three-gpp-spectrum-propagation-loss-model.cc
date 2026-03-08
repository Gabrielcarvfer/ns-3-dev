/*
 * Copyright (c) 2015, NYU WIRELESS, Tandon School of Engineering,
 * New York University
 * Copyright (c) 2019 SIGNET Lab, Department of Information Engineering,
 * University of Padova
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#include "three-gpp-spectrum-propagation-loss-model.h"

#include "spectrum-signal-parameters.h"
#include "three-gpp-channel-model.h"

#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/pointer.h"
#include "ns3/simulator.h"
#include "ns3/string.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("ThreeGppSpectrumPropagationLossModel");

NS_OBJECT_ENSURE_REGISTERED(ThreeGppSpectrumPropagationLossModel);

ThreeGppSpectrumPropagationLossModel::ThreeGppSpectrumPropagationLossModel()
{
    NS_LOG_FUNCTION(this);
}

ThreeGppSpectrumPropagationLossModel::~ThreeGppSpectrumPropagationLossModel()
{
    NS_LOG_FUNCTION(this);
}

void
ThreeGppSpectrumPropagationLossModel::DoDispose()
{
    m_longTermMap.clear();
    m_channelModel = nullptr;
}

TypeId
ThreeGppSpectrumPropagationLossModel::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::ThreeGppSpectrumPropagationLossModel")
            .SetParent<PhasedArraySpectrumPropagationLossModel>()
            .SetGroupName("Spectrum")
            .AddConstructor<ThreeGppSpectrumPropagationLossModel>()
            .AddAttribute(
                "ChannelModel",
                "The channel model. It needs to implement the MatrixBasedChannelModel interface",
                StringValue("ns3::ThreeGppChannelModel"),
                MakePointerAccessor(&ThreeGppSpectrumPropagationLossModel::SetChannelModel,
                                    &ThreeGppSpectrumPropagationLossModel::GetChannelModel),
                MakePointerChecker<MatrixBasedChannelModel>());
    return tid;
}

void
ThreeGppSpectrumPropagationLossModel::SetChannelModel(Ptr<MatrixBasedChannelModel> channel)
{
    m_channelModel = channel;
}

Ptr<MatrixBasedChannelModel>
ThreeGppSpectrumPropagationLossModel::GetChannelModel() const
{
    return m_channelModel;
}

double
ThreeGppSpectrumPropagationLossModel::GetFrequency() const
{
    NS_ASSERT_MSG(m_channelModel != nullptr, "Channel model is not set");
    DoubleValue freq;
    m_channelModel->GetAttribute("Frequency", freq);
    return freq.Get();
}

void
ThreeGppSpectrumPropagationLossModel::SetChannelModelAttribute(const std::string& name,
                                                               const AttributeValue& value)
{
    NS_ASSERT_MSG(m_channelModel != nullptr, "Channel model is not set");
    m_channelModel->SetAttribute(name, value);
}

void
ThreeGppSpectrumPropagationLossModel::GetChannelModelAttribute(const std::string& name,
                                                               AttributeValue& value) const
{
    NS_ASSERT_MSG(m_channelModel != nullptr, "Channel model is not set");
    m_channelModel->GetAttribute(name, value);
}

Ptr<const MatrixBasedChannelModel::Complex3DVector>
ThreeGppSpectrumPropagationLossModel::CalcLongTerm(
    Ptr<const MatrixBasedChannelModel::ChannelMatrix> channelMatrix,
    Ptr<const PhasedArrayModel> sAnt,
    Ptr<const PhasedArrayModel> uAnt) const
{
    NS_LOG_FUNCTION(this);

     const auto& sW = sAnt->GetBeamformingVectorRef();
    const auto& uW = uAnt->GetBeamformingVectorRef();
    const size_t numClusters  = channelMatrix->m_channel.GetNumPages();
    const size_t sAntElems    = sW.GetSize();
    const size_t uAntElems    = uW.GetSize();
    const size_t nUPorts      = uAnt->GetNumPorts();
    const size_t nSPorts      = sAnt->GetNumPorts();
    const auto   sPortElems   = sAnt->GetNumElemsPerPort();
    const auto   uPortElems   = uAnt->GetNumElemsPerPort();
    const auto   sElemsPerPort = sAnt->GetHElemsPerPort();
    const auto   uElemsPerPort = uAnt->GetHElemsPerPort();
    const auto   sNumCols     = sAnt->GetNumColumns();
    const auto   uNumCols     = uAnt->GetNumColumns();

    // Precompute port start indices — avoid per-call ArrayIndexFromPortIndex
    std::vector<size_t> sPortStart(nSPorts), uPortStart(nUPorts);
    for (size_t p = 0; p < nSPorts; ++p) sPortStart[p] = sAnt->ArrayIndexFromPortIndex(p, 0);
    for (size_t p = 0; p < nUPorts; ++p) uPortStart[p] = uAnt->ArrayIndexFromPortIndex(p, 0);

    // Precompute flat index arrays per port (avoids branch inside hot loop)
    auto buildPortIndices = [](size_t start, size_t portElems,
                               size_t elemsPerPort, size_t numCols) {
        std::vector<size_t> idx;
        idx.reserve(portElems);
        size_t i = start;
        for (size_t t = 0; t < portElems; ++t, ++i) {
            idx.push_back(i);
            if ((t % elemsPerPort) == (elemsPerPort - 1))
                i += numCols - elemsPerPort;  // skip to next column in port
        }
        return idx;
    };

    std::vector<std::vector<size_t>> sIdx(nSPorts), uIdx(nUPorts);
    for (size_t p = 0; p < nSPorts; ++p)
        sIdx[p] = buildPortIndices(sPortStart[p], sPortElems, sElemsPerPort, sNumCols);
    for (size_t p = 0; p < nUPorts; ++p)
        uIdx[p] = buildPortIndices(uPortStart[p], uPortElems, uElemsPerPort, uNumCols);

    auto longTerm = Create<MatrixBasedChannelModel::Complex3DVector>(nUPorts, nSPorts, numClusters);

    // Intermediate buffer: H[:,sPort_elems,c] * sW  →  hs[uAntElems]
    std::vector<std::complex<double>> hs(uAntElems);

    for (size_t cIndex = 0; cIndex < numClusters; ++cIndex)
    {
        for (size_t sPortIdx = 0; sPortIdx < nSPorts; ++sPortIdx)
        {
            // Phase 1: compute hs = H[:, sPort, c] * sW  (once per sPort × cluster)
            std::fill(hs.begin(), hs.end(), std::complex<double>(0.0, 0.0));
            for (size_t uIndex = 0; uIndex < uAntElems; ++uIndex)
                for (size_t t = 0; t < sPortElems; ++t)
                    hs[uIndex] += channelMatrix->m_channel(uIndex, sIdx[sPortIdx][t], cIndex)
                                  * sW[t];  // sW indexed relative to port start

            // Phase 2: dot conj(uW) with hs for each uPort
            for (size_t uPortIdx = 0; uPortIdx < nUPorts; ++uPortIdx)
            {
                std::complex<double> val(0.0, 0.0);
                for (size_t r = 0; r < uPortElems; ++r)
                    val += std::conj(uW[r]) * hs[uIdx[uPortIdx][r]];
                longTerm->Elem(uPortIdx, sPortIdx, cIndex) = val;
            }
        }
    }
    return longTerm;
}

Ptr<SpectrumSignalParameters>
ThreeGppSpectrumPropagationLossModel::CalcBeamformingGain(
    Ptr<const SpectrumSignalParameters> params,
    Ptr<const MatrixBasedChannelModel::Complex3DVector> longTerm,
    Ptr<const MatrixBasedChannelModel::ChannelMatrix> channelMatrix,
    Ptr<const MatrixBasedChannelModel::ChannelParams> channelParams,
    const Vector& sSpeed,
    const Vector& uSpeed,
    const uint8_t numTxPorts,
    const uint8_t numRxPorts,
    const bool isReverse) const

{
    NS_LOG_FUNCTION(this);
    Ptr<SpectrumSignalParameters> rxParams = params->Copy();
    const size_t numCluster = channelMatrix->m_channel.GetNumPages();
    // compute the doppler term
    // NOTE the update of Doppler is simplified by only taking the center angle of
    // each cluster in to consideration.
    const double slotTime = Simulator::Now().GetSeconds();
    const double factor = 2 * M_PI * slotTime * GetFrequency() / 3e8;
    PhasedArrayModel::ComplexVector doppler(numCluster);

    // Make sure that all the structures that are passed to this function
    // are of the correct dimensions before using the operator [].
    NS_ASSERT(numCluster <= channelParams->m_alpha.size());
    NS_ASSERT(numCluster <= channelParams->m_D.size());
    NS_ASSERT(numCluster <= channelParams->m_angle[MatrixBasedChannelModel::ZOA_INDEX].size());
    NS_ASSERT(numCluster <= channelParams->m_angle[MatrixBasedChannelModel::ZOD_INDEX].size());
    NS_ASSERT(numCluster <= channelParams->m_angle[MatrixBasedChannelModel::AOA_INDEX].size());
    NS_ASSERT(numCluster <= channelParams->m_angle[MatrixBasedChannelModel::AOD_INDEX].size());
    NS_ASSERT(numCluster <= longTerm->GetNumPages());

    // check if channelParams structure is generated in direction s-to-u or u-to-s
    const bool isSameDir = channelParams->m_nodeIds == channelMatrix->m_nodeIds;

    // if channel params is generated in the same direction in which we
    // generate the channel matrix, angles and zenith of departure and arrival are ok,
    // just set them to corresponding variable that will be used for the generation
    // of channel matrix, otherwise we need to flip angles and zeniths of departure and arrival
    using DPV = std::vector<std::pair<double, double>>;
    const auto& cachedAngleSincos = channelParams->m_cachedAngleSincos;
    NS_ASSERT_MSG(cachedAngleSincos.size() > MatrixBasedChannelModel::ZOD_INDEX,
                  "Cached angle sin/cos not initialized");
    const DPV& zoa = cachedAngleSincos[isSameDir ? MatrixBasedChannelModel::ZOA_INDEX
                                                 : MatrixBasedChannelModel::ZOD_INDEX];
    const DPV& zod = cachedAngleSincos[isSameDir ? MatrixBasedChannelModel::ZOD_INDEX
                                                 : MatrixBasedChannelModel::ZOA_INDEX];
    const DPV& aoa = cachedAngleSincos[isSameDir ? MatrixBasedChannelModel::AOA_INDEX
                                                 : MatrixBasedChannelModel::AOD_INDEX];
    const DPV& aod = cachedAngleSincos[isSameDir ? MatrixBasedChannelModel::AOD_INDEX
                                                 : MatrixBasedChannelModel::AOA_INDEX];
    NS_ASSERT(numCluster <= zoa.size());
    NS_ASSERT(numCluster <= zod.size());
    NS_ASSERT(numCluster <= aoa.size());
    NS_ASSERT(numCluster <= aod.size());

    for (size_t cIndex = 0; cIndex < numCluster; cIndex++)
    {
        // Compute alpha and D as described in 3GPP TR 37.885 v15.3.0, Sec. 6.2.3
        // These terms account for an additional Doppler contribution due to the
        // presence of moving objects in the surrounding environment, such as in
        // vehicular scenarios.
        // This contribution is applied only to the delayed (reflected) paths and
        // must be properly configured by setting the value of
        // m_vScatt, which is defined as "maximum speed of the vehicle in the
        // layout".
        // By default, m_vScatt is set to 0, so there is no additional Doppler
        // contribution.

        const double alpha = channelParams->m_alpha[cIndex];
        const double D = channelParams->m_D[cIndex];

        // cluster angle angle[direction][n], where direction = 0(aoa), 1(zoa).
        const double tempDoppler =
            factor *
            (zoa[cIndex].first * aoa[cIndex].second * uSpeed.x +
             zoa[cIndex].first * aoa[cIndex].first * uSpeed.y + zoa[cIndex].second * uSpeed.z +
             (zod[cIndex].first * aod[cIndex].second * sSpeed.x +
              zod[cIndex].first * aod[cIndex].first * sSpeed.y + zod[cIndex].second * sSpeed.z) +
             2 * alpha * D);
        doppler[cIndex] = std::complex(cos(tempDoppler), sin(tempDoppler));
    }

    NS_ASSERT(numCluster <= doppler.GetSize());

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
        auto page = ComplexMatrixArray(rxParams->spectrumChannelMatrix->GetNumCols(), 1, 1);
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
ThreeGppSpectrumPropagationLossModel::GenSpectrumChannelMatrix(
    Ptr<SpectrumValue> inPsd,
    Ptr<const MatrixBasedChannelModel::Complex3DVector> longTerm,
    Ptr<const MatrixBasedChannelModel::ChannelMatrix> channelMatrix,
    Ptr<const MatrixBasedChannelModel::ChannelParams> channelParams,
    PhasedArrayModel::ComplexVector doppler,
    uint8_t numTxPorts,
    uint8_t numRxPorts,
    const bool isReverse)
{
    const size_t numCluster = channelMatrix->m_channel.GetNumPages();
    const auto numRb = inPsd->GetValuesN();
    NS_ASSERT_MSG(numCluster <= channelParams->m_delay.size(),
                  "Channel params delays size is smaller than number of clusters");

    auto directionalLongTerm = isReverse ? longTerm->Transpose() : *longTerm;

    Ptr<MatrixBasedChannelModel::Complex3DVector> chanSpct =
        Create<MatrixBasedChannelModel::Complex3DVector>(numRxPorts,
                                                         numTxPorts,
                                                         static_cast<uint16_t>(numRb));

    // Precompute the delay until numRb, numCluster or RB width changes
    // Whenever the channelParams is updated, the number of numRbs, numClusters
    // and RB width (12*SCS) are reset, ensuring these values are updated too

    if (const double rbWidth = inPsd->ConstBandsBegin()->fh - inPsd->ConstBandsBegin()->fl;
        channelParams->m_cachedDelaySincos.GetNumRows() != numRb ||
        channelParams->m_cachedDelaySincos.GetNumCols() != numCluster ||
        channelParams->m_cachedRbWidth != rbWidth)
    {
        channelParams->m_cachedRbWidth = rbWidth;
        channelParams->m_cachedDelaySincos = ComplexMatrixArray(numRb, numCluster);
        auto sbit = inPsd->ConstBandsBegin(); // band iterator
        for (unsigned i = 0; i < numRb; i++)
        {
            const double fsb = sbit->fc; // center frequency of the sub-band
            for (std::size_t cIndex = 0; cIndex < numCluster; cIndex++)
            {
                const double delay = -2 * M_PI * fsb * channelParams->m_delay[cIndex];
                channelParams->m_cachedDelaySincos(i, cIndex) =
                    std::complex(cos(delay), sin(delay));
            }
            ++sbit;
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
        if (*vit != 0.00)
        {
            auto sqrtVit = sqrt(*vit);
            for (auto rxPortIdx = 0; rxPortIdx < numRxPorts; rxPortIdx++)
            {
                for (auto txPortIdx = 0; txPortIdx < numTxPorts; txPortIdx++)
                {
                    std::complex subsbandGain(0.0, 0.0);
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
        ++vit;
        ++iRb;
    }
    return chanSpct;
}

Ptr<const MatrixBasedChannelModel::Complex3DVector>
ThreeGppSpectrumPropagationLossModel::GetLongTerm(
    Ptr<const MatrixBasedChannelModel::ChannelMatrix> channelMatrix,
    Ptr<const PhasedArrayModel> aPhasedArrayModel,
    Ptr<const PhasedArrayModel> bPhasedArrayModel) const
{
    Ptr<const MatrixBasedChannelModel::Complex3DVector>
        longTerm; // vector containing the long term component for each cluster

    // check if the channel matrix was generated considering a as the s-node and
    // b as the u-node or vice-versa
    const auto isReverse =
        channelMatrix->IsReverse(aPhasedArrayModel->GetId(), bPhasedArrayModel->GetId());
    const auto sAntenna = isReverse ? bPhasedArrayModel : aPhasedArrayModel;
    const auto uAntenna = isReverse ? aPhasedArrayModel : bPhasedArrayModel;

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
    const uint64_t longTermId =
        MatrixBasedChannelModel::GetKey(aPhasedArrayModel->GetId(), bPhasedArrayModel->GetId());

    // look for the long term in the map and check if it is valid
    if (m_longTermMap.contains(longTermId))
    {
        NS_LOG_DEBUG("found the long term component in the map");
        longTerm = m_longTermMap[longTermId]->m_longTerm;

        // check if the channel matrix has been updated
        // or the s beam has been changed
        // or the u beam has been changed
        update = m_longTermMap[longTermId]->m_channel->m_generatedTime !=
                     channelMatrix->m_generatedTime ||
                 m_longTermMap[longTermId]->m_sW != sW || m_longTermMap[longTermId]->m_uW != uW;
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
ThreeGppSpectrumPropagationLossModel::DoCalcRxPowerSpectralDensity(
    Ptr<const SpectrumSignalParameters> spectrumSignalParams,
    Ptr<const MobilityModel> a,
    Ptr<const MobilityModel> b,
    Ptr<const PhasedArrayModel> aPhasedArrayModel,
    Ptr<const PhasedArrayModel> bPhasedArrayModel) const
{
    NS_LOG_FUNCTION(this << spectrumSignalParams << a << b << aPhasedArrayModel
                         << bPhasedArrayModel);
    NS_ASSERT_MSG(m_channelModel != nullptr, "Channel model is not set");

    const uint32_t aId = a->GetObject<Node>()->GetId(); // id of the node a
    const uint32_t bId = b->GetObject<Node>()->GetId(); // id of the node b
    NS_ASSERT_MSG(aPhasedArrayModel, "Antenna not found for node " << aId);
    NS_LOG_DEBUG("a node " << aId << " antenna " << aPhasedArrayModel);
    NS_ASSERT_MSG(bPhasedArrayModel, "Antenna not found for node " << bId);
    NS_LOG_DEBUG("b node " << bId << " antenna " << bPhasedArrayModel);

    Ptr<const MatrixBasedChannelModel::ChannelMatrix> channelMatrix =
        m_channelModel->GetChannel(a, b, aPhasedArrayModel, bPhasedArrayModel);
    const Ptr<const MatrixBasedChannelModel::ChannelParams> channelParams =
        m_channelModel->GetParams(a, b);
    NS_ASSERT_MSG(channelMatrix != nullptr, "Channel matrix is null");
    NS_ASSERT_MSG(channelParams != nullptr, "Channel params are null");

    // retrieve the long term component
    const Ptr<const MatrixBasedChannelModel::Complex3DVector> longTerm =
        GetLongTerm(channelMatrix, aPhasedArrayModel, bPhasedArrayModel);

    const auto isReverse =
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
ThreeGppSpectrumPropagationLossModel::DoAssignStreams(int64_t stream)
{
    return 0;
}

} // namespace ns3
