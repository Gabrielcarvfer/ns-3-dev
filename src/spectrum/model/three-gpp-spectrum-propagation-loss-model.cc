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

#include "sls-phase-timer.h"
#include "spectrum-signal-parameters.h"
#include "three-gpp-channel-model.h"

#include <cstring>

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
    SLS_PHASE_SCOPE("PRX::CalcLongTerm");
    NS_LOG_FUNCTION(this);

    NS_ASSERT_MSG(sAnt != nullptr && uAnt != nullptr, "Improper call to the method");
    const PhasedArrayModel::ComplexVector& sW = sAnt->GetBeamformingVectorRef();
    const PhasedArrayModel::ComplexVector& uW = uAnt->GetBeamformingVectorRef();
    [[maybe_unused]] const size_t sAntNumElems = sW.GetSize();
    [[maybe_unused]] const size_t uAntNumElems = uW.GetSize();
    NS_ASSERT(uAntNumElems == channelMatrix->m_channel.GetNumRows());
    NS_ASSERT(sAntNumElems == channelMatrix->m_channel.GetNumCols());
    const size_t numClusters = channelMatrix->m_channel.GetNumPages();
    const auto sPorts = static_cast<uint16_t>(sAnt->GetNumPorts());
    const auto uPorts = static_cast<uint16_t>(uAnt->GetNumPorts());
    Ptr<MatrixBasedChannelModel::Complex3DVector> longTerm =
        Create<MatrixBasedChannelModel::Complex3DVector>(uPorts, sPorts, numClusters);

    // ── Optimization: hoist invariants and the inner-loop body of
    // CalculateLongTermComponent out of the triple-nested call. The
    // original called the helper once per (sPort, uPort, cluster) cell;
    // for a 16x16x17 configuration that's 4352 function calls per
    // CalcLongTerm invocation, each with its own asserts +
    // sAnt/uAnt vtable dispatches. In Debug builds that overhead
    // ballooned to ~5 ms per call; inlining lets the compiler
    // (and humans) keep the per-port port-element loop tight.
    //
    // sub-array partition model from 3GPP TR 36.897 5.2.2: each port
    // has the same beam weights and a contiguous block of array
    // elements. With one element per port (the typical
    // dual-polarised 2x8-port / 2x8-element layout in NR) the inner
    // body collapses to a single conj(uW)*channel*sW per cell.
    const auto sPortElems = sAnt->GetNumElemsPerPort();
    const auto uPortElems = uAnt->GetNumElemsPerPort();
    const auto uElemsPerPort = uAnt->GetHElemsPerPort();
    const auto sElemsPerPort = sAnt->GetHElemsPerPort();
    const auto uIncVal =
        (uElemsPerPort > 0) ? uAnt->GetNumColumns() - uElemsPerPort : 0;
    const auto sIncVal =
        (sElemsPerPort > 0) ? sAnt->GetNumColumns() - sElemsPerPort : 0;

    // Hot-path optimizations beyond the prior call-inlining pass:
    //  1. Precompute conj(uW) and copy sW into raw arrays so the inner
    //     loops do plain indexed loads instead of repeated std::conj
    //     calls on every (sPort, uPort, cluster, tIndex, rIndex) tuple.
    //     For a 16x4 port / 8x1 element-per-port / 20-cluster shape
    //     that drops roughly 100k std::conj invocations per CalcLongTerm
    //     call to ~16 (one pass over uW).
    //  2. Cache the channel matrix's column-major page pointer once per
    //     cluster (`pagePtr = channel.GetPagePtr(cIndex)`). The matrix
    //     element at (row=uIndex, col=sIndex) is then pagePtr[uIndex +
    //     numRows * sIndex], which bypasses the operator()(r,c,p) +
    //     bounds-check chain that doesn't always inline in Debug.
    //  3. Specialise on uPortElems == 1 (DenseAmimoIntel: 1 element per
    //     UE port). The inner loop collapses to a single multiply, the
    //     %uElemsPerPort branch is statically dead, and we lift the
    //     conj-uW load out of the cluster loop too.
    const size_t numRows = channelMatrix->m_channel.GetNumRows();
    std::vector<std::complex<double>> conjU(uPortElems);
    for (size_t k = 0; k < uPortElems; ++k)
    {
        conjU[k] = std::conj(uW[k]);
    }
    std::vector<std::complex<double>> sWcopy(sPortElems);
    for (size_t k = 0; k < sPortElems; ++k)
    {
        sWcopy[k] = sW[k];
    }

    // Split complex weights into (re, im) primitive pairs so the inner
    // loops use plain double arithmetic instead of std::complex
    // operator*/+ chains. In Debug builds the std::complex calls don't
    // inline and each MAC pays a ~10x overhead in vtable + temp-object
    // setup. The PSD reduction loop already did this; CalcLongTerm gets
    // the same treatment. Math is identical -- pure refactor.
    std::vector<double> conjURe(uPortElems);
    std::vector<double> conjUIm(uPortElems);
    for (size_t k = 0; k < uPortElems; ++k)
    {
        conjURe[k] = conjU[k].real();
        conjUIm[k] = conjU[k].imag();
    }
    std::vector<double> sWcopyRe(sPortElems);
    std::vector<double> sWcopyIm(sPortElems);
    for (size_t k = 0; k < sPortElems; ++k)
    {
        sWcopyRe[k] = sWcopy[k].real();
        sWcopyIm[k] = sWcopy[k].imag();
    }

    const bool uTrivial = (uPortElems == 1);
    const bool uNeedHop = (uElemsPerPort > 0);
    const bool sNeedHop = (sElemsPerPort > 0);
    auto* longTermD = reinterpret_cast<double*>(longTerm->GetPagePtr(0));
    for (uint16_t sPortIdx = 0; sPortIdx < sPorts; ++sPortIdx)
    {
        const auto startS = sAnt->ArrayIndexFromPortIndex(sPortIdx, 0);
        for (uint16_t uPortIdx = 0; uPortIdx < uPorts; ++uPortIdx)
        {
            const auto startU = uAnt->ArrayIndexFromPortIndex(uPortIdx, 0);
            for (size_t cIndex = 0; cIndex < numClusters; ++cIndex)
            {
                // pageD: real/imag pairs for the cluster slab, indexed
                // as (rxElem + numRows*txElem) * 2 (re), +1 (im).
                const auto* pageD =
                    reinterpret_cast<const double*>(channelMatrix->m_channel.GetPagePtr(cIndex));
                double txSumRe = 0.0;
                double txSumIm = 0.0;
                auto sIndex = startS;
                if (uTrivial)
                {
                    // One UE element per port: conjU[0] is the only rx
                    // contribution. Inline (a+bi)*(c+di) and
                    // (e+fi)*(re+im) by hand.
                    const double cu0Re = conjURe[0];
                    const double cu0Im = conjUIm[0];
                    for (size_t tIndex = 0; tIndex < sPortElems; ++tIndex, ++sIndex)
                    {
                        const size_t cell = (startU + numRows * sIndex) * 2;
                        const double pR = pageD[cell];
                        const double pI = pageD[cell + 1];
                        // rxSum = conjU0 * pagePtr[cell]
                        const double rxR = cu0Re * pR - cu0Im * pI;
                        const double rxI = cu0Re * pI + cu0Im * pR;
                        // txSum += sWcopy[tIndex] * rxSum
                        const double sR = sWcopyRe[tIndex];
                        const double sI = sWcopyIm[tIndex];
                        txSumRe += sR * rxR - sI * rxI;
                        txSumIm += sR * rxI + sI * rxR;
                        if (sNeedHop && tIndex % sElemsPerPort == sElemsPerPort - 1)
                        {
                            sIndex += sIncVal;
                        }
                    }
                }
                else
                {
                    for (size_t tIndex = 0; tIndex < sPortElems; ++tIndex, ++sIndex)
                    {
                        double rxSumRe = 0.0;
                        double rxSumIm = 0.0;
                        auto uIndex = startU;
                        for (size_t rIndex = 0; rIndex < uPortElems; ++rIndex, ++uIndex)
                        {
                            const size_t cell = (uIndex + numRows * sIndex) * 2;
                            const double pR = pageD[cell];
                            const double pI = pageD[cell + 1];
                            const double cuR = conjURe[rIndex];
                            const double cuI = conjUIm[rIndex];
                            rxSumRe += cuR * pR - cuI * pI;
                            rxSumIm += cuR * pI + cuI * pR;
                            if (uNeedHop && rIndex % uElemsPerPort == uElemsPerPort - 1)
                            {
                                uIndex += uIncVal;
                            }
                        }
                        const double sR = sWcopyRe[tIndex];
                        const double sI = sWcopyIm[tIndex];
                        txSumRe += sR * rxSumRe - sI * rxSumIm;
                        txSumIm += sR * rxSumIm + sI * rxSumRe;
                        if (sNeedHop && tIndex % sElemsPerPort == sElemsPerPort - 1)
                        {
                            sIndex += sIncVal;
                        }
                    }
                }
                // longTerm column-major layout: (uPortIdx, sPortIdx,
                // cIndex) at uPortIdx + uPorts*sPortIdx + uPorts*sPorts*cIndex.
                const size_t outIdx =
                    (uPortIdx + size_t(uPorts) * sPortIdx + size_t(uPorts) * sPorts * cIndex) * 2;
                longTermD[outIdx] = txSumRe;
                longTermD[outIdx + 1] = txSumIm;
            }
        }
    }
    return longTerm;
}

std::complex<double>
ThreeGppSpectrumPropagationLossModel::CalculateLongTermComponent(
    Ptr<const MatrixBasedChannelModel::ChannelMatrix> params,
    Ptr<const PhasedArrayModel> sAnt,
    Ptr<const PhasedArrayModel> uAnt,
    const uint16_t sPortIdx,
    const uint16_t uPortIdx,
    const uint16_t cIndex) const
{
    NS_LOG_FUNCTION(this);
    const PhasedArrayModel::ComplexVector& sW = sAnt->GetBeamformingVectorRef();
    const PhasedArrayModel::ComplexVector& uW = uAnt->GetBeamformingVectorRef();
    const auto sPortElems = sAnt->GetNumElemsPerPort();
    const auto uPortElems = uAnt->GetNumElemsPerPort();
    const auto startS = sAnt->ArrayIndexFromPortIndex(sPortIdx, 0);
    const auto startU = uAnt->ArrayIndexFromPortIndex(uPortIdx, 0);
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
            rxSum += std::conj(uW[uIndex - startU]) * params->m_channel(uIndex, sIndex, cIndex);
            const auto testV = rIndex % uElemsPerPort;
            if (const auto ptInc = uElemsPerPort - 1; testV == ptInc)
            {
                const auto incVal = uAnt->GetNumColumns() - uElemsPerPort;
                uIndex += incVal; // Increment by a factor to reach next column in a port
            }
        }

        txSum += sW[sIndex - startS] * rxSum;
        const auto testV = tIndex % sElemsPerPort;
        if (const auto ptInc = sElemsPerPort - 1; testV == ptInc)
        {
            const size_t incVal = sAnt->GetNumColumns() - sElemsPerPort;
            sIndex += incVal; // Increment by a factor to reach next column in a port
        }
    }
    return txSum;
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
    SLS_PHASE_SCOPE("PRX::CalcBeamformingGain");
    NS_LOG_FUNCTION(this);
    Ptr<SpectrumSignalParameters> rxParams;
    {
        SLS_PHASE_SCOPE("PRX::CalcBeamformingGain::ParamsCopy");
        rxParams = params->Copy();
    }
    const size_t numCluster = channelMatrix->m_channel.GetNumPages();
    // compute the doppler term
    // NOTE the update of Doppler is simplified by only taking the center angle of
    // each cluster in to consideration.
    const double slotTime = Simulator::Now().GetSeconds();
    const double factor = 2 * M_PI * slotTime * GetFrequency() / 3e8;
    PhasedArrayModel::ComplexVector doppler(numCluster);

    // Make sure that all the structures that are passed to this function
    // are of the correct dimensions before using the operator [].
    // Per-vector NS_ASSERT_MSG so when one fires we know which dim
    // is short rather than guessing from a bare assert line.
    NS_ASSERT_MSG(numCluster <= channelParams->m_alpha.size(),
                  "PRX::CBG: m_alpha.size()=" << channelParams->m_alpha.size()
                                              << " < numCluster=" << numCluster);
    NS_ASSERT_MSG(numCluster <= channelParams->m_D.size(),
                  "PRX::CBG: m_D.size()=" << channelParams->m_D.size()
                                          << " < numCluster=" << numCluster);
    NS_ASSERT_MSG(
        numCluster <= channelParams->m_angle[MatrixBasedChannelModel::ZOA_INDEX].size(),
        "PRX::CBG: m_angle[ZOA].size()="
            << channelParams->m_angle[MatrixBasedChannelModel::ZOA_INDEX].size()
            << " < numCluster=" << numCluster);
    NS_ASSERT_MSG(
        numCluster <= channelParams->m_angle[MatrixBasedChannelModel::ZOD_INDEX].size(),
        "PRX::CBG: m_angle[ZOD].size()="
            << channelParams->m_angle[MatrixBasedChannelModel::ZOD_INDEX].size()
            << " < numCluster=" << numCluster);
    NS_ASSERT_MSG(
        numCluster <= channelParams->m_angle[MatrixBasedChannelModel::AOA_INDEX].size(),
        "PRX::CBG: m_angle[AOA].size()="
            << channelParams->m_angle[MatrixBasedChannelModel::AOA_INDEX].size()
            << " < numCluster=" << numCluster);
    NS_ASSERT_MSG(
        numCluster <= channelParams->m_angle[MatrixBasedChannelModel::AOD_INDEX].size(),
        "PRX::CBG: m_angle[AOD].size()="
            << channelParams->m_angle[MatrixBasedChannelModel::AOD_INDEX].size()
            << " < numCluster=" << numCluster);
    NS_ASSERT_MSG(numCluster <= longTerm->GetNumPages(),
                  "PRX::CBG: longTerm->GetNumPages()=" << longTerm->GetNumPages()
                                                      << " < numCluster=" << numCluster);

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
    NS_ASSERT_MSG(numCluster <= zoa.size(),
                  "PRX::CBG: m_cachedAngleSincos[ZOA].size()=" << zoa.size()
                                                              << " < numCluster=" << numCluster
                                                              << " (isSameDir=" << isSameDir << ")");
    NS_ASSERT_MSG(numCluster <= zod.size(),
                  "PRX::CBG: m_cachedAngleSincos[ZOD].size()=" << zod.size()
                                                              << " < numCluster=" << numCluster
                                                              << " (isSameDir=" << isSameDir << ")");
    NS_ASSERT_MSG(numCluster <= aoa.size(),
                  "PRX::CBG: m_cachedAngleSincos[AOA].size()=" << aoa.size()
                                                              << " < numCluster=" << numCluster
                                                              << " (isSameDir=" << isSameDir << ")");
    NS_ASSERT_MSG(numCluster <= aod.size(),
                  "PRX::CBG: m_cachedAngleSincos[AOD].size()=" << aod.size()
                                                              << " < numCluster=" << numCluster
                                                              << " (isSameDir=" << isSameDir << ")");

    {
    SLS_PHASE_SCOPE("PRX::CBG::DopplerLoop");
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
        // std::polar folds cos+sin into one sincos on modern libstdc++.
        doppler[cIndex] = std::polar(1.0, tempDoppler);
    }
    } // PRX::CBG::DopplerLoop

    NS_ASSERT(numCluster <= doppler.GetSize());

    // set the channel matrix
    rxParams->spectrumChannelMatrix = GenSpectrumChannelMatrix(rxParams->psd,
                                                               longTerm,
                                                               channelMatrix,
                                                               channelParams,
                                                               doppler,
                                                               numTxPorts,
                                                               numRxPorts,
                                                               isReverse,
                                                               m_channelModel);

    NS_ASSERT_MSG(rxParams->psd->GetValuesN() == rxParams->spectrumChannelMatrix->GetNumPages(),
                  "RX PSD and the spectrum channel matrix should have the same number of RBs ");

    NS_ASSERT_MSG(!params->precodingMatrix || (params->precodingMatrix &&
                                               params->precodingMatrix->GetNumPages() ==
                                                   rxParams->spectrumChannelMatrix->GetNumPages()),
                  "Unexpected mismatch in the number of RBs and channel matrix and precoding "
                  "matrix. MultiModelSpectrumChannel conversion is not yet supported.");

    // Calculate RX PSD from the spectrum channel matrix H and the
    // precoding matrix P as: PSD[rb] = sum_rx |sum_tx (P[tx,0,rb] *
    //                                                    H[rx,tx,rb])|^2
    // Fast path: when no explicit precoding is provided we use the
    // default isotropic 1/sqrt(N) column. That collapses the H*P
    // matmul into a per-(rx,rb) sum, and the trace-of-(HP)^H*(HP)
    // reduces to sum_{rx,rb} |that sum|^2 / N. Skipping the
    // intermediate `hP` allocation + matmul + |z|^2 pass saves
    // ~25KB of allocation and the per-rb x per-(rx,tx) double walk
    // through the H matrix.
    auto specMat = rxParams->spectrumChannelMatrix;
    const size_t hRxPorts = specMat->GetNumRows();
    const size_t hTxPorts = specMat->GetNumCols();
    const uint32_t psdN = rxParams->psd->GetValuesN();
    SLS_PHASE_SCOPE("PRX::CBG::PsdReduction");
    if (!rxParams->precodingMatrix)
    {
        // Default isotropic precoding: P[tx, 0, rb] = 1/sqrt(numTx).
        // PSD[rb] = (1/N) * sum_rx |sum_tx H[rx, tx, rb]|^2
        //
        // Access specMat via raw pointers (per-page base + rx-stride 1
        // / tx-stride hRxPorts) and split each complex<double> into
        // its (re, im) pair so the hot inner loop avoids the
        // operator()() bounds-checks and std::complex calls that the
        // Debug build was paying per cell.
        const double invN = 1.0 / static_cast<double>(hTxPorts);
        for (uint32_t rb = 0; rb < psdN; ++rb)
        {
            const auto* pageD = reinterpret_cast<const double*>(specMat->GetPagePtr(rb));
            double acc = 0.0;
            for (size_t rx = 0; rx < hRxPorts; ++rx)
            {
                double sumRe = 0.0;
                double sumIm = 0.0;
                // Index of (rx, tx, rb) inside this page: rx + hRxPorts*tx.
                // We walk tx with stride hRxPorts (real index hRxPorts*2 in doubles).
                for (size_t tx = 0; tx < hTxPorts; ++tx)
                {
                    const size_t off = (rx + hRxPorts * tx) * 2;
                    sumRe += pageD[off];
                    sumIm += pageD[off + 1];
                }
                acc += sumRe * sumRe + sumIm * sumIm;
            }
            (*rxParams->psd)[rb] = acc * invN;
        }
        // The spectrumChannelMatrix on rxParams stays as the per-cluster
        // matrix the doppler+delay pipeline produced; callers that ask
        // for hP directly will trip on `rxParams->precodingMatrix
        // == nullptr` and reconstruct.
    }
    else
    {
        // General path: explicit precoding matrix provided.
        Ptr<const ComplexMatrixArray> p = rxParams->precodingMatrix;
        MatrixBasedChannelModel::Complex3DVector hP = *specMat * *p;
        for (uint32_t rb = 0; rb < psdN; ++rb)
        {
            double acc = 0.0;
            for (size_t rx = 0; rx < hP.GetNumRows(); ++rx)
            {
                for (size_t tx = 0; tx < hP.GetNumCols(); ++tx)
                {
                    // std::norm(z) == |z|^2 (real-only result, avoids
                    // the conj*z complex multiply).
                    acc += std::norm(hP(rx, tx, rb));
                }
            }
            (*rxParams->psd)[rb] = acc;
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
    const bool isReverse,
    Ptr<MatrixBasedChannelModel> channelModel)
{
    SLS_PHASE_SCOPE("PRX::GenSpectrumChannelMatrix");
    const size_t numCluster = channelMatrix->m_channel.GetNumPages();
    const auto numRb = inPsd->GetValuesN();
    NS_ASSERT_MSG(numCluster <= channelParams->m_delay.size(),
                  "Channel params delays size is smaller than number of clusters");

    // Avoid the full-copy of `*longTerm` on the (common) non-reverse
    // path. `directionalLongTerm` is read-only below; in the reverse
    // case we still need the materialised Transpose, but otherwise
    // we keep a pointer-aliased view of the cached matrix.
    MatrixBasedChannelModel::Complex3DVector reversedLongTerm;
    if (isReverse)
    {
        reversedLongTerm = longTerm->Transpose();
    }
    const MatrixBasedChannelModel::Complex3DVector& directionalLongTerm =
        isReverse ? reversedLongTerm : *longTerm;

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

    // ── Optimization: pack inputs into c-contiguous flat arrays so the
    // hottest dimension of the tensor contraction is unit-stride. The
    // original layout walked the cluster index through Complex3DVector
    // (rx-fast, tx-mid, page-slow) and Complex2DArray (rb-fast,
    // cluster-slow). On a realistic 32x32x17x273 contraction the
    // cluster-stride was 16 KB on longTerm and 4.4 KB on delaySincos
    // — each c step trashed L1.
    //
    //   longTermT[c, rxtx] = longTerm[rx, tx, c]   (c-fast, rxtx-slow)
    //   delayT[c, rb]      = delaySincos[rb, c] * doppler[c]  (c-fast, rb-slow)
    //
    // After the pack, the kernel becomes a rank-17 outer-product
    // accumulation: for each cluster c, scale longTermT[c, :] by
    // delayT[c, rb] and add to chanSpct[:, :, rb]. The inner step is a
    // contiguous read of longTermT (rxtx) AND a contiguous
    // write of chanSpct (rxtx) — best possible cache behaviour, with
    // dpoint-product / FMAs SIMD-friendly.
    const size_t rxtx = static_cast<size_t>(numRxPorts) * numTxPorts;
    std::vector<std::complex<double>> longTermT(numCluster * rxtx);
    {
        // directionalLongTerm has shape (numRxPorts, numTxPorts, numCluster)
        // and the underlying flat index is rx + numRxPorts*tx + rxtx*c.
        // We want longTermT[c, rxtx] = directionalLongTerm flat index above
        // but in the form longTermT[c * rxtx + (rx + numRxPorts*tx)].
        const auto& src = directionalLongTerm.GetValues();
        for (size_t c = 0; c < numCluster; ++c)
        {
            // Source slice: src[c * rxtx ... c * rxtx + rxtx)
            // Dest slice:   longTermT[c * rxtx ... +rxtx)
            const size_t srcBase = c * rxtx;
            const size_t dstBase = c * rxtx;
            std::memcpy(&longTermT[dstBase],
                        &src[srcBase],
                        rxtx * sizeof(std::complex<double>));
        }
    }
    std::vector<std::complex<double>> delayT(numCluster * numRb);
    {
        // Source: cachedDelaySincos has shape (numRb, numCluster), flat
        // index = rb + numRb*c. We want delayT[c * numRb + rb].
        const auto& srcDel = channelParams->m_cachedDelaySincos.GetValues();
        for (size_t c = 0; c < numCluster; ++c)
        {
            const std::complex<double> dopplerC = doppler[c];
            const size_t srcBase = c * numRb;
            const size_t dstBase = c * numRb;
            for (size_t rb = 0; rb < numRb; ++rb)
            {
                delayT[dstBase + rb] = srcDel[srcBase + rb] * dopplerC;
            }
        }
    }

    // Precompute sqrt(*vit[rb]) for the entire PSD. Zeros stay zero, so
    // subbands with zero TX power are skipped after the contraction
    // (their column in chanSpct is just zeroed by the initial Create
    // call -- we never write to it).
    std::vector<double> sqrtVit(numRb, 0.0);
    {
        auto vit = inPsd->ValuesBegin();
        for (size_t rb = 0; rb < numRb; ++rb, ++vit)
        {
            const double v = *vit;
            if (v > 0.0)
                sqrtVit[rb] = std::sqrt(v);
        }
    }

    // Give the channel model a chance to run the per-cluster
    // outer-product on the GPU. The mezanine WebGPU back-end
    // overrides this and dispatches gen_spec_chan_kernel against
    // the longTerm matrix that's already resident on its GPU buffer.
    // On nullptr we fall through to the CPU contraction below.
    if (channelModel)
    {
        if (auto gpuChanSpct = channelModel->TryGenSpectrumChannelMatrix(
                channelMatrix, channelParams, longTerm, inPsd,
                delayT, sqrtVit, numRb, numRxPorts, numTxPorts, isReverse))
        {
            return gpuChanSpct;
        }
    }

    // Outer-product contraction. c-outermost so the chanSpct[:, :, rb]
    // tail (rxtx*16 B per rb) stays in cache while we walk all rbs for
    // a single cluster, and the longTermT slice for c (rxtx*16 B)
    // stays in L1 throughout the inner rb sweep. Inner (rxtx) loop is
    // unit-stride on both longTermT and chanSpct.
    //
    // Drops std::complex<double> in the hot inner loop because in
    // Debug mode the compiler keeps `operator*` / `operator+=` as
    // separate calls (no inlining at -O0). At 4.75M iterations per
    // call that was billions of stack frames per benchmark run.
    // Treating each complex<double> as a pair of raw doubles lets the
    // hot loop be plain scalar adds + muls that even the Debug
    // compiler keeps inline -- and a release build will autovectorise
    // the same shape over AVX2 FMA registers.
    auto* longTermTd = reinterpret_cast<const double*>(longTermT.data());
    auto* delayTd = reinterpret_cast<const double*>(delayT.data());
    for (size_t c = 0; c < numCluster; ++c)
    {
        const double* aRow = longTermTd + c * rxtx * 2;          // [re, im] pairs
        const double* bRow = delayTd + c * numRb * 2;
        for (size_t rb = 0; rb < numRb; ++rb)
        {
            const double bRe = bRow[rb * 2];
            const double bIm = bRow[rb * 2 + 1];
            auto* outRow = reinterpret_cast<double*>(chanSpct->GetPagePtr(rb));
            for (size_t i = 0; i < rxtx; ++i)
            {
                const double aRe = aRow[i * 2];
                const double aIm = aRow[i * 2 + 1];
                // (aRe + j aIm) * (bRe + j bIm) =
                //   (aRe*bRe - aIm*bIm) + j (aRe*bIm + aIm*bRe)
                outRow[i * 2]     += aRe * bRe - aIm * bIm;
                outRow[i * 2 + 1] += aRe * bIm + aIm * bRe;
            }
        }
    }
    // Scale each RB slab by sqrt(*vit[rb]). RBs with zero PSD stay at
    // zero (Complex3DVector::Create zero-inits via valarray default).
    for (size_t rb = 0; rb < numRb; ++rb)
    {
        const double s = sqrtVit[rb];
        if (s == 0.0)
            continue;
        auto* outRow = reinterpret_cast<double*>(chanSpct->GetPagePtr(rb));
        for (size_t i = 0; i < rxtx * 2; ++i)
        {
            outRow[i] *= s;
        }
    }
    return chanSpct;
}

Ptr<const MatrixBasedChannelModel::Complex3DVector>
ThreeGppSpectrumPropagationLossModel::GetLongTerm(
    Ptr<const MatrixBasedChannelModel::ChannelMatrix> channelMatrix,
    Ptr<const PhasedArrayModel> aPhasedArrayModel,
    Ptr<const PhasedArrayModel> bPhasedArrayModel) const
{
    SLS_PHASE_SCOPE("PRX::GetLongTerm");
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
        // We compare the captured snapshot to the current m_generatedTime
        // because the mezanine reuses ChannelMatrix Ptrs across periodic
        // refreshes -- the Ptr identity stays constant but the field
        // advances in place. Reading via cached.m_channel-> would always
        // return the current value (same object) and never detect the
        // refresh.
        update = m_longTermMap[longTermId]->m_capturedGeneratedTime !=
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
        // Give the channel model a chance to provide a pre-computed
        // longTerm (e.g. ThreeGppChannelModelWgpuMezanine running
        // gen_long_term_kernel during UpdateChannel). On a match we
        // skip the CPU CalcLongTerm entirely; on a miss the back-end
        // returns nullptr and we fall through.
        Ptr<const MatrixBasedChannelModel::Complex3DVector> gpuLongTerm;
        if (m_channelModel)
        {
            gpuLongTerm =
                m_channelModel->GetCachedLongTerm(channelMatrix, sAntenna, uAntenna, sW, uW);
        }
        if (gpuLongTerm)
        {
            longTerm = gpuLongTerm;
        }
        else
        {
            // compute the long term component
            longTerm = CalcLongTerm(channelMatrix, sAntenna, uAntenna);
        }
        Ptr<LongTerm> longTermItem = Create<LongTerm>();
        longTermItem->m_longTerm = longTerm;
        longTermItem->m_channel = channelMatrix;
        longTermItem->m_capturedGeneratedTime = channelMatrix->m_generatedTime;
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
    SLS_PHASE_SCOPE("PRX::DoCalcRxPowerSpectralDensity");
    NS_LOG_FUNCTION(this << spectrumSignalParams << a << b << aPhasedArrayModel
                         << bPhasedArrayModel);
    NS_ASSERT_MSG(m_channelModel != nullptr, "Channel model is not set");

    const uint32_t aId = a->GetObject<Node>()->GetId(); // id of the node a
    const uint32_t bId = b->GetObject<Node>()->GetId(); // id of the node b
    NS_ASSERT_MSG(aPhasedArrayModel, "Antenna not found for node " << aId);
    NS_LOG_DEBUG("a node " << aId << " antenna " << aPhasedArrayModel);
    NS_ASSERT_MSG(bPhasedArrayModel, "Antenna not found for node " << bId);
    NS_LOG_DEBUG("b node " << bId << " antenna " << bPhasedArrayModel);

    // If the channel model supports it (currently only ThreeGppChannelModel
    // with `UseGpu=true`), refresh every dirty link in one batch before
    // the per-link `GetChannel` calls start. Default base-class impl is a
    // no-op, so existing back-ends (Friis, two-ray, ...) keep working
    // unchanged.
    {
        SLS_PHASE_SCOPE("PRX::EnsureBatchFresh");
        m_channelModel->EnsureBatchFresh();
    }

    Ptr<const MatrixBasedChannelModel::ChannelMatrix> channelMatrix =
        m_channelModel->GetChannel(a, b, aPhasedArrayModel, bPhasedArrayModel);
    Ptr<const MatrixBasedChannelModel::ChannelParams> channelParams;
    {
        SLS_PHASE_SCOPE("PRX::GetParams");
        channelParams = m_channelModel->GetParams(a, b);
    }
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
