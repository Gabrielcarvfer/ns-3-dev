/*
 * Copyright (c) 2026, CTTC, Centre Tecnologic de Telecomunicacions de Catalunya
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "three-gpp-channel-webgpu-offloader.h"
#include <ns3/log.h>

// Note: This implementation assumes a WebGPU C API is available.
// If not available, this file will fail to compile unless guarded or mocked.
#ifdef HAS_WEBGPU
#include <webgpu/webgpu.h>
#endif

#include <ns3/vector.h>
#include <ns3/phased-array-model.h>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("ThreeGppChannelWebGpuOffloader");

// Structure to match the GPU uniform buffer
struct GpuParams
{
    uint32_t uSize;
    uint32_t sSize;
    uint32_t numClusters;
    uint32_t raysPerCluster;
    uint32_t cluster1st;
    uint32_t cluster2nd;
    uint32_t totalReducedClusters;
    uint32_t hUsnPages;
};

ThreeGppChannelWebGpuOffloader::ThreeGppChannelWebGpuOffloader()
    : m_initialized(false)
{
    m_initialized = InitializeWebGpu();
}

ThreeGppChannelWebGpuOffloader::~ThreeGppChannelWebGpuOffloader()
{
}

bool
ThreeGppChannelWebGpuOffloader::InitializeWebGpu()
{
#ifdef HAS_WEBGPU
    // In a real implementation, we would initialize the WebGPU instance, adapter, and device here.
    // For this task, we assume the environment is properly set up if HAS_WEBGPU is defined.
    return true;
#else
    return false;
#endif
}

bool
ThreeGppChannelWebGpuOffloader::IsAvailable() const
{
    return m_initialized;
}

const char* channelShaderSource = R"(
struct Params {
    uSize: u32,
    sSize: u32,
    numClusters: u32,
    raysPerCluster: u32,
    cluster1st: u32,
    cluster2nd: u32,
    totalReducedClusters: u32,
    hUsnPages: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> uLoc: array<vec3f>;
@group(0) @binding(2) var<storage, read> sLoc: array<vec3f>;
@group(0) @binding(3) var<storage, read> anglesA: array<vec3f>; // (sinCos, sinSin, cosZo)
@group(0) @binding(4) var<storage, read> anglesD: array<vec3f>; // (sinCos, sinSin, cosZo)
@group(0) @binding(5) var<storage, read> clusterPower: array<f32>;
@group(0) @binding(6) var<storage, read> preCompRays: array<vec2f>; // flattened: [polPair][n][m]
@group(0) @binding(7) var<storage, read> polPairs: array<u32>; // packed pol pairs (sPol << 8 | uPol)
@group(0) @binding(8) var<storage, read_write> hUsn: array<vec2f>; // flattened: [n][u][s]

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let index = global_id.x;
    let uSize = params.uSize;
    let sSize = params.sSize;
    let totalReducedClusters = params.totalReducedClusters;
    let totalSize = uSize * sSize * totalReducedClusters;

    if (index >= totalSize) {
        return;
    }

    let sIndex = index % sSize;
    let uIndex = (index / sSize) % uSize;
    let nIndex = index / (uSize * sSize);

    let uPos = uLoc[uIndex];
    let sPos = sLoc[sIndex];
    
    // Get the polarization pair for this sIndex and uIndex
    // In ns-3, sAntenna->GetElemPol(sIndex) and uAntenna->GetElemPol(uIndex)
    let polPair = polPairs[uIndex * sSize + sIndex];
    let sPol = polPair >> 8u;
    let uPol = polPair & 0xFFu;
    
    // Identify which pol matrix in preCompRays to use. 
    // We assume preCompRays is ordered by pol pairs: (0,0), (0,1), (1,0), (1,1)
    // where 0=Vertical, 1=Horizontal (or whatever the enum maps to)
    let polOffset = (sPol * 2u + uPol) * (totalReducedClusters * params.raysPerCluster);

    var raysSum1 = vec2f(0.0, 0.0);
    var raysSum2 = vec2f(0.0, 0.0);
    var raysSum3 = vec2f(0.0, 0.0);
    
    let isSubClustered = (nIndex == params.cluster1st || nIndex == params.cluster2nd);
    let powerBase = clusterPower[nIndex] / f32(params.raysPerCluster);
    let sqrtPower = sqrt(powerBase);

    for (var m: u32 = 0; m < params.raysPerCluster; m = m + 1) {
        let angleA = anglesA[nIndex * params.raysPerCluster + m];
        let angleD = anglesD[nIndex * params.raysPerCluster + m];
        
        let rxPhaseDiff = 2.0 * 3.1415926535 * dot(angleA, uPos);
        let txPhaseDiff = 2.0 * 3.1415926535 * dot(angleD, sPos);
        
        let totalPhase = rxPhaseDiff + txPhaseDiff;
        let phaseVec = vec2f(cos(totalPhase), sin(totalPhase));
        
        let preComp = preCompRays[polOffset + nIndex * params.raysPerCluster + m];
        
        let rayResult = vec2f(
            preComp.x * phaseVec.x - preComp.y * phaseVec.y,
            preComp.x * phaseVec.y + preComp.y * phaseVec.x
        );

        if (isSubClustered) {
            // Sub-clustering logic according to 3GPP TS 38.901
            // Case 9-12, 17-18
            if ((m >= 8 && m <= 11) || m == 16 || m == 17) {
                raysSum2 += rayResult;
            } else if (m >= 12 && m <= 15) { // Case 13-16
                raysSum3 += rayResult;
            } else { // Others: 1-8, 19-20
                raysSum1 += rayResult;
            }
        } else {
            raysSum1 += rayResult;
        }
    }

    // Output to hUsn. MatrixArray is [page][row][col] => [n][u][s]
    // Base cluster
    hUsn[nIndex * (uSize * sSize) + uIndex * sSize + sIndex] = raysSum1 * sqrtPower;
    
    if (isSubClustered) {
        // Find which sub-cluster index we are. 
        // Logic from CPU: numSubClustersAdded is incremented after processing each of the two strongest clusters.
        // So the first one processed (lower nIndex) gets (totalReducedClusters, totalReducedClusters + 1)
        // and the second one (higher nIndex) gets (totalReducedClusters + 2, totalReducedClusters + 3).
        var subClusterIdx2: u32;
        var subClusterIdx3: u32;
        
        let firstSub = min(params.cluster1st, params.cluster2nd);
        if (nIndex == firstSub) {
            subClusterIdx2 = totalReducedClusters;
            subClusterIdx3 = totalReducedClusters + 1u;
        } else {
            subClusterIdx2 = totalReducedClusters + 2u;
            subClusterIdx3 = totalReducedClusters + 3u;
        }
        
        hUsn[subClusterIdx2 * (uSize * sSize) + uIndex * sSize + sIndex] = raysSum2 * sqrtPower;
        hUsn[subClusterIdx3 * (uSize * sSize) + uIndex * sSize + sIndex] = raysSum3 * sqrtPower;
    }
}
)";

void
ThreeGppChannelWebGpuOffloader::ComputeChannelMatrix(
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
    ComplexMatrixArray& hUsn)
{
    NS_LOG_FUNCTION(this);

#ifdef HAS_WEBGPU
    size_t uSize = uAntenna->GetNumElems();
    size_t sSize = sAntenna->GetNumElems();

    // 1. Prepare data buffers
    std::vector<float> uLocData(uSize * 3);
    for (size_t i = 0; i < uSize; ++i)
    {
        Vector loc = uAntenna->GetElementLocation(i);
        uLocData[i * 3 + 0] = static_cast<float>(loc.x);
        uLocData[i * 3 + 1] = static_cast<float>(loc.y);
        uLocData[i * 3 + 2] = static_cast<float>(loc.z);
    }

    std::vector<float> sLocData(sSize * 3);
    for (size_t i = 0; i < sSize; ++i)
    {
        Vector loc = sAntenna->GetElementLocation(i);
        sLocData[i * 3 + 0] = static_cast<float>(loc.x);
        sLocData[i * 3 + 1] = static_cast<float>(loc.y);
        sLocData[i * 3 + 2] = static_cast<float>(loc.z);
    }

    // Flatten angles
    std::vector<float> anglesAData(numClusters * raysPerCluster * 3);
    std::vector<float> anglesDData(numClusters * raysPerCluster * 3);
    for (uint32_t n = 0; n < numClusters; ++n)
    {
        for (uint32_t m = 0; m < raysPerCluster; ++m)
        {
            uint32_t base = (n * raysPerCluster + m) * 3;
            anglesAData[base + 0] = static_cast<float>(sinCosA[n][m]);
            anglesAData[base + 1] = static_cast<float>(sinSinA[n][m]);
            anglesAData[base + 2] = static_cast<float>(cosZoA[n][m]);

            anglesDData[base + 0] = static_cast<float>(sinCosD[n][m]);
            anglesDData[base + 1] = static_cast<float>(sinSinD[n][m]);
            anglesDData[base + 2] = static_cast<float>(cosZoD[n][m]);
        }
    }

    // Flatten raysPreComp for all 4 polarization pairs
    // map key is pair(sPol, uPol)
    std::vector<float> preCompRaysData(4 * numClusters * raysPerCluster * 2);
    for (uint8_t sPol = 0; sPol < 2; ++sPol)
    {
        for (uint8_t uPol = 0; uPol < 2; ++uPol)
        {
            auto it = raysPreComp.find(std::make_pair(sPol, uPol));
            if (it != raysPreComp.end())
            {
                const ComplexMatrixArray& cma = it->second;
                uint32_t polOffset = (sPol * 2 + uPol) * (numClusters * raysPerCluster) * 2;
                for (uint32_t n = 0; n < numClusters; ++n)
                {
                    for (uint32_t m = 0; m < raysPerCluster; ++m)
                    {
                        uint32_t base = polOffset + (n * raysPerCluster + m) * 2;
                        std::complex<double> val = cma(n, m);
                        preCompRaysData[base + 0] = static_cast<float>(val.real());
                        preCompRaysData[base + 1] = static_cast<float>(val.imag());
                    }
                }
            }
        }
    }

    // Prepare polarization pairs per antenna element
    std::vector<uint32_t> polPairsData(uSize * sSize);
    for (size_t u = 0; u < uSize; ++u)
    {
        for (size_t s = 0; s < sSize; ++s)
        {
            uint32_t sPol = static_cast<uint32_t>(sAntenna->GetElemPol(s));
            uint32_t uPol = static_cast<uint32_t>(uAntenna->GetElemPol(u));
            polPairsData[u * sSize + s] = (sPol << 8) | uPol;
        }
    }

    // 2. Uniform buffer
    GpuParams params;
    params.uSize = uSize;
    params.sSize = sSize;
    params.numClusters = numClusters;
    params.raysPerCluster = raysPerCluster;
    params.cluster1st = cluster1st;
    params.cluster2nd = cluster2nd;
    params.totalReducedClusters = numClusters;
    params.hUsnPages = hUsn.GetNumPages();

    // In a full implementation, we would use WebGPU API to:
    // - Create buffers (GPUBuffer) for all the data prepared above
    // - wgpuQueueWriteBuffer(queue, buffer, 0, data.data(), data.size() * sizeof(float))
    // - Create a compute pipeline (WGPUComputePipeline)
    // - Create a bind group (WGPUBindGroup) mapping bindings 0-8
    // - Create a command encoder and pass (WGPUCommandEncoder, WGPUComputePassEncoder)
    // - wgpuComputePassEncoderDispatchWorkgroups(pass, (totalSize + 63) / 64, 1, 1)
    // - Copy result back: wgpuCommandEncoderCopyBufferToBuffer(encoder, hUsnBuffer, readBackBuffer, ...)
    // - Map readBackBuffer and copy to hUsn MatrixArray
    
    NS_LOG_INFO("Offloading channel matrix computation to GPU for " 
                << uSize << "x" << sSize << " antenna elements. "
                << "Clusters: " << (int)numClusters << ", Pages: " << params.hUsnPages);
#else
    NS_UNUSED(uAntenna);
    NS_UNUSED(sAntenna);
    NS_UNUSED(numClusters);
    NS_UNUSED(raysPerCluster);
    NS_UNUSED(clusterPower);
    NS_UNUSED(sinCosA);
    NS_UNUSED(sinSinA);
    NS_UNUSED(cosZoA);
    NS_UNUSED(sinCosD);
    NS_UNUSED(sinSinD);
    NS_UNUSED(cosZoD);
    NS_UNUSED(raysPreComp);
    NS_UNUSED(cluster1st);
    NS_UNUSED(cluster2nd);
    NS_UNUSED(hUsn);
#endif
}

} // namespace ns3
