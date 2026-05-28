//
// Updated draft with WGPU integration fixes:
// - correct ActiveLink lspReadIdx = cid * nUt + uid
// - correct CIR/CFR flat-buffer offsets based on global BS/UE antenna counts
// - populated runtimeLinks for cache writeback
// - deterministic antenna config ordering: antCfgs[0] = BS, antCfgs[1] = UE
// - guards for the current SlsChanWgpu assumption of one global BS nAnt and one global UE nAnt
//

#include "three-gpp-channel-model-wgpu-mezanine.h"

#include "sls-chan-wgpu.h"
#include "sls-phase-timer.h"

#include "ns3/abort.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/mobility-model.h"
#include "ns3/node-list.h"
#include "ns3/node.h"
#include "ns3/phased-array-model.h"
#include "ns3/simulator.h"
#include "ns3/uinteger.h"
#include "ns3/uniform-planar-array.h"
#include "ns3/vector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

NS_LOG_COMPONENT_DEFINE("ThreeGppChannelModelWgpuMezanine");

namespace ns3
{

NS_OBJECT_ENSURE_REGISTERED(ThreeGppChannelModelWgpuMezanine);

ThreeGppChannelModelWgpuMezanine::ThreeGppChannelModelWgpuMezanine()
{
    m_wgpuChannel = std::make_unique<SlsChanWgpu>();
}

ThreeGppChannelModelWgpuMezanine::~ThreeGppChannelModelWgpuMezanine()
{
    m_wgpuChannel = nullptr;
}

TypeId
ThreeGppChannelModelWgpuMezanine::GetTypeId()
{
    static TypeId tid = TypeId("ns3::ThreeGppChannelModelWgpuMezanine")
                            .SetGroupName("Spectrum")
                            .SetParent<ThreeGppChannelModel>()
                            .AddConstructor<ThreeGppChannelModelWgpuMezanine>();
    return tid;
}

namespace
{

struct RuntimeLinkCtx
{
    uint64_t paramsKey{};
    uint64_t matrixKey{};

    uint32_t sNodeId{};
    uint32_t uNodeId{};
    uint32_t sAntId{};
    uint32_t uAntId{};

    uint32_t cid{};
    uint32_t uid{};
    uint32_t lspReadIdx{};

    Ptr<MobilityModel> sMob;
    Ptr<MobilityModel> uMob;
    Ptr<PhasedArrayModel> sAnt;
    Ptr<PhasedArrayModel> uAnt;
    Ptr<ChannelCondition> condition;
};

struct NodeInfo
{
    uint32_t nodeId = 0;
    Ptr<MobilityModel> mob;
    Vector pos;
    Vector vel;
};

struct PanelInfo
{
    uint32_t antennaId = 0;
    Ptr<const UniformPlanarArray> antenna;
    bool isBsSide = false;
    uint32_t panelIdx = 0;
    uint32_t nAnt = 0;
    std::array<uint32_t, 5> antSize{1, 1, 1, 1, 1};
    std::array<float, 4> antSpacing{0.f, 0.f, 0.5f, 0.5f};
    std::array<float, 2> polarAnglesDeg{45.f, -45.f};
    std::array<float, 3> panelOrientation{0.f, 0.f, 0.f};
    uint32_t antModel = 0;
    std::vector<float> thetaDeg;
    std::vector<float> phiDeg;
};

struct SiteRec
{
    uint32_t nodeId = 0;
    uint32_t panelIdx = 0;
    uint32_t cellIdx = 0;
};

struct UtRec
{
    uint32_t nodeId = 0;
    uint32_t panelIdx = 0;
    uint32_t utIdx = 0;
};

inline uint32_t
FlatClusterRayIndex(uint32_t linkIdx, uint32_t c, uint32_t r)
{
    return linkIdx * MAX_CLUSTERS * MAX_RAYS + c * MAX_RAYS + r;
}

inline bool
SameDims(const MatrixBasedChannelModel::Double3DVector& v, uint32_t nCluster, uint32_t nRay)
{
    if (v.size() != nCluster)
    {
        return false;
    }
    for (uint32_t c = 0; c < nCluster; ++c)
    {
        if (v[c].size() != nRay)
        {
            return false;
        }
        for (uint32_t r = 0; r < nRay; ++r)
        {
            if (v[c][r].size() != 4u)
            {
                return false;
            }
        }
    }
    return true;
}

} // namespace

void
ThreeGppChannelModelWgpuMezanine::UpdateChannel()
{
    SLS_PHASE_SCOPE("Mez::UpdateChannel");
    NS_LOG_FUNCTION(this);

    // EnsureBatchFresh drives this synchronously at the start of every
    // tick now (see the override above), so the historical 10 ms
    // self-rescheduling loop is no longer needed.
    if (m_channelMatrixMap.empty())
    {
        NS_LOG_DEBUG("No cached channel matrices yet; nothing to upload to WGPU.");
        return;
    }

    auto* gpu = m_wgpuChannel.get();
    NS_ABORT_MSG_IF(gpu == nullptr, "SlsChanWgpu not constructed");

    using MatrixEntry = Ptr<ChannelMatrix>;
    using ParamsEntry = Ptr<ThreeGppChannelParams>;

    std::unordered_map<uint32_t, NodeInfo> nodeInfoById;
    std::unordered_map<uint32_t, PanelInfo> panelInfoByAntId;
    std::unordered_map<uint32_t, SiteRec> siteByNodeId;
    std::unordered_map<uint32_t, UtRec> utByNodeId;

    std::vector<CellParam> cells;
    std::vector<UtParam> uts;
    std::vector<CellParamSS> cellsSS;
    std::vector<UtParamSS> utsSS;
    std::vector<AntPanelConfigGPU> antCfgs;
    std::vector<float> antThetaFlat;
    std::vector<float> antPhiFlat;
    std::vector<ActiveLink> activeLinks;
    std::vector<RuntimeLinkCtx> runtimeLinks;

    antCfgs.reserve(8);
    runtimeLinks.reserve(m_channelMatrixMap.size());
    activeLinks.reserve(m_channelMatrixMap.size());

    double minX = std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();

    bool haveBsReference = false;
    bool haveUtReference = false;
    uint32_t bsReferencePanelIdx = 0;
    uint32_t utReferencePanelIdx = 1;
    uint32_t globalBsAnt = 0;
    uint32_t globalUeAnt = 0;

    auto updateBounds = [&](const Vector& p) {
        minX = std::min(minX, p.x);
        minY = std::min(minY, p.y);
        maxX = std::max(maxX, p.x);
        maxY = std::max(maxY, p.y);
    };

    auto getVelocity = [](const Ptr<MobilityModel>& mob) -> Vector {
        return mob ? mob->GetVelocity() : Vector{0.0, 0.0, 0.0};
    };

    auto resolveMobilityFromNodeId = [&](uint32_t nodeId) -> Ptr<MobilityModel> {
        Ptr<Node> node = NodeList::GetNode(nodeId);
        NS_ABORT_MSG_IF(node == nullptr, "Node not found for nodeId=" << nodeId);
        Ptr<MobilityModel> mob = node->GetObject<MobilityModel>();
        NS_ABORT_MSG_IF(mob == nullptr, "MobilityModel not found for nodeId=" << nodeId);
        return mob;
    };

    auto resolveAntennaFromAntennaId = [&](uint32_t antennaId) -> Ptr<const UniformPlanarArray> {
        auto it = m_antennaIdToObjectMap.find(antennaId);
        NS_ABORT_MSG_IF(it == m_antennaIdToObjectMap.end(),
                        "Antenna object not registered for antennaId=" << antennaId);
        Ptr<const PhasedArrayModel> antP = it->second;
        Ptr<const UniformPlanarArray> ant = DynamicCast<const UniformPlanarArray>(antP);
        NS_ABORT_MSG_IF(ant == nullptr, "UniformPlanarArray not found for antennaId=" << antennaId);
        return ant;
    };

    auto buildNodeInfo = [&](uint32_t nodeId) -> const NodeInfo& {
        auto it = nodeInfoById.find(nodeId);
        if (it != nodeInfoById.end())
        {
            return it->second;
        }

        Ptr<MobilityModel> mob = resolveMobilityFromNodeId(nodeId);
        NodeInfo info;
        info.nodeId = nodeId;
        info.mob = mob;
        info.pos = mob->GetPosition();
        info.vel = getVelocity(mob);
        updateBounds(info.pos);

        auto [insIt, inserted] = nodeInfoById.emplace(nodeId, info);
        (void)inserted;
        return insIt->second;
    };

    auto appendPanelConfig = [&](PanelInfo& p) {
        AntPanelConfigGPU cfg{};
        cfg.nAnt = p.nAnt;
        cfg.antModel = p.antModel;
        for (size_t i = 0; i < 5; ++i)
        {
            cfg.antSize[i] = p.antSize[i];
        }
        for (size_t i = 0; i < 4; ++i)
        {
            cfg.antSpacing[i] = p.antSpacing[i];
        }
        cfg.antPolarAngles[0] = p.polarAnglesDeg[0];
        cfg.antPolarAngles[1] = p.polarAnglesDeg[1];
        cfg.thetaOffset = static_cast<uint32_t>(antThetaFlat.size());
        cfg.phiOffset = static_cast<uint32_t>(antPhiFlat.size());
        cfg._pad0 = 0;
        // _pad1 was removed when AntPanelConfigGPU shrank from 68 B to
        // the 64 B WGSL stride. _pad0 alone fills the trailing slot.

        p.panelIdx = static_cast<uint32_t>(antCfgs.size());
        antThetaFlat.insert(antThetaFlat.end(), p.thetaDeg.begin(), p.thetaDeg.end());
        antPhiFlat.insert(antPhiFlat.end(), p.phiDeg.begin(), p.phiDeg.end());
        antCfgs.push_back(cfg);
    };

    auto buildPanelInfo = [&](uint32_t antennaId, bool isBsSide) -> const PanelInfo& {
        auto it = panelInfoByAntId.find(antennaId);
        if (it != panelInfoByAntId.end())
        {
            return it->second;
        }

        Ptr<const UniformPlanarArray> ant = resolveAntennaFromAntennaId(antennaId);

        PanelInfo p;
        p.antennaId = antennaId;
        p.antenna = ant;
        p.isBsSide = isBsSide;
        p.nAnt = ant->GetNumElems();
        // antSize layout per AntPanelConfig comment: [Mg, Ng, M, N, P]
        //   Mg, Ng = number of panels (= 1 for a single panel)
        //   M, N   = rows, columns within a panel
        //   P      = polarisations (1 or 2)
        // The previous version flattened the whole array into a 1xN
        // column with P=1, which made mat_elem_pos / mat_pol_idx put
        // every element at row=0 along a single line and assign all
        // of them to polarisation 0 -- coherent gain on the 8x8 BS
        // collapsed because the row*d_v dimension vanished and the
        // two physical polarisations were not separated.
        const uint32_t M = static_cast<uint32_t>(ant->GetNumRows());
        const uint32_t N = static_cast<uint32_t>(ant->GetNumColumns());
        const uint32_t P = static_cast<uint32_t>(ant->GetNumPols());
        p.antSize = {1, 1, M, N, P};
        p.antSpacing = {0.f, 0.f, 0.5f, 0.5f};
        // Polarisation slant angles. CPU UPA stores ONE PolSlantAngle
        // attribute in radians; the dual-polarised second pol is taken
        // to be perpendicular (slant + pi/2). The mezanine previously
        // hardcoded {45, -45} degrees for BS, which mismatched the
        // CPU side for any antenna whose PolSlantAngle wasn't 45 deg
        // and broke beam coherency on dual-pol BS arrays.
        const double polSlantRad = ant->GetPolSlant();
        const double polSlantDeg = polSlantRad * 180.0 / M_PI;
        p.polarAnglesDeg = ant->IsDualPol()
                               ? std::array<float, 2>{
                                     static_cast<float>(polSlantDeg),
                                     static_cast<float>(polSlantDeg + 90.0)}
                               : std::array<float, 2>{
                                     static_cast<float>(polSlantDeg), 0.f};
        p.panelOrientation = {static_cast<float>(ant->GetAlpha()),
                              static_cast<float>(ant->GetBeta()),
                              0.f};
        p.antModel = 0;
        p.thetaDeg.assign(181, 0.f);
        p.phiDeg.assign(360, 0.f);

        if (isBsSide)
        {
            if (!haveBsReference)
            {
                appendPanelConfig(p);
                bsReferencePanelIdx = p.panelIdx;
                globalBsAnt = p.nAnt;
                haveBsReference = true;
            }
            else
            {
                NS_ABORT_MSG_IF(p.nAnt != globalBsAnt,
                                "Current SlsChanWgpu path assumes one global BS antenna count; got "
                                    << p.nAnt << " expected " << globalBsAnt);
                p.panelIdx = bsReferencePanelIdx;
            }
        }
        else
        {
            if (!haveUtReference)
            {
                if (!haveBsReference)
                {
                    NS_ABORT_MSG("BS reference panel must be created before UE reference panel");
                }
                appendPanelConfig(p);
                utReferencePanelIdx = p.panelIdx;
                globalUeAnt = p.nAnt;
                haveUtReference = true;
            }
            else
            {
                NS_ABORT_MSG_IF(p.nAnt != globalUeAnt,
                                "Current SlsChanWgpu path assumes one global UE antenna count; got "
                                    << p.nAnt << " expected " << globalUeAnt);
                p.panelIdx = utReferencePanelIdx;
            }
        }

        auto [insIt, inserted] = panelInfoByAntId.emplace(antennaId, p);
        (void)inserted;
        return insIt->second;
    };

    auto getOrCreateSite = [&](uint32_t nodeId, uint32_t antId) -> const SiteRec& {
        auto it = siteByNodeId.find(nodeId);
        if (it != siteByNodeId.end())
        {
            return it->second;
        }

        const NodeInfo& ni = buildNodeInfo(nodeId);
        const PanelInfo& pi = buildPanelInfo(antId, true);

        CellParam cell{};
        // CellParam.loc is now float[3] (was Vec3f) to match WGSL's
        // vec3<f32> size-12-align-16 layout exactly.
        cell.loc[0] = static_cast<float>(ni.pos.x);
        cell.loc[1] = static_cast<float>(ni.pos.y);
        cell.loc[2] = static_cast<float>(ni.pos.z);

        CellParamSS cellSs{};
        cellSs.antPanelIdx = pi.panelIdx;
        cellSs.antPanelOrientation[0] = pi.panelOrientation[0];
        cellSs.antPanelOrientation[1] = pi.panelOrientation[1];
        cellSs.antPanelOrientation[2] = pi.panelOrientation[2];
        cellSs._pad0 = 0;

        SiteRec rec;
        rec.nodeId = nodeId;
        rec.panelIdx = pi.panelIdx;
        rec.cellIdx = static_cast<uint32_t>(cells.size());

        cells.push_back(cell);
        cellsSS.push_back(cellSs);

        auto [insIt, inserted] = siteByNodeId.emplace(nodeId, rec);
        (void)inserted;
        return insIt->second;
    };

    auto getOrCreateUt = [&](uint32_t nodeId,
                             uint32_t antId,
                             const ParamsEntry& params,
                             bool rxIsSecondInParams) -> const UtRec& {
        auto it = utByNodeId.find(nodeId);
        if (it != utByNodeId.end())
        {
            return it->second;
        }

        const NodeInfo& ni = buildNodeInfo(nodeId);
        const PanelInfo& pi = buildPanelInfo(antId, false);

        const bool isOutdoor = params->m_o2iCondition == ChannelCondition::O2iConditionValue::O2O;
        const float o2iLoss = 0.f;
        Vector vel = rxIsSecondInParams ? params->m_rxSpeed : params->m_txSpeed;

        UtParam ut{};
        ut.loc = Vec3f{static_cast<float>(ni.pos.x),
                       static_cast<float>(ni.pos.y),
                       static_cast<float>(ni.pos.z),
                       0.f};
        ut.d_2d_in = 0.f;
        ut.outdoor_ind = isOutdoor ? 1u : 0u;
        ut.o2i_penetration_loss = o2iLoss;
        ut._p = 0.f;

        vel = rxIsSecondInParams ? params->m_rxSpeed : params->m_txSpeed;

        UtParamSS utSs{};
        utSs.antPanelIdx = pi.panelIdx;
        utSs.outdoor_ind = ut.outdoor_ind;
        utSs.antPanelOrientation[0] = pi.panelOrientation[0];
        utSs.antPanelOrientation[1] = pi.panelOrientation[1];
        utSs.antPanelOrientation[2] = pi.panelOrientation[2];
        utSs.velocity[0] = static_cast<float>(vel.x);
        utSs.velocity[1] = static_cast<float>(vel.y);
        utSs.velocity[2] = static_cast<float>(vel.z);
        utSs._pad0 = 0;

        UtRec rec;
        rec.nodeId = nodeId;
        rec.panelIdx = pi.panelIdx;
        rec.utIdx = static_cast<uint32_t>(uts.size());

        uts.push_back(ut);
        utsSS.push_back(utSs);

        auto [insIt, inserted] = utByNodeId.emplace(nodeId, rec);
        (void)inserted;
        return insIt->second;
    };

    for (const auto& kv : m_channelMatrixMap)
    {
        const uint64_t matrixKey = kv.first;
        const MatrixEntry& ch = kv.second;
        NS_ABORT_MSG_IF(ch == nullptr, "Null ChannelMatrix in m_channelMatrixMap");

        const uint32_t txNodeId = ch->m_nodeIds.first;
        const uint32_t rxNodeId = ch->m_nodeIds.second;
        const uint32_t txAntId = ch->m_antennaPair.first;
        const uint32_t rxAntId = ch->m_antennaPair.second;

        const uint64_t paramsKey = GetKey(txNodeId, rxNodeId);
        auto pit = m_channelParamsMap.find(paramsKey);
        NS_ABORT_MSG_IF(pit == m_channelParamsMap.end(),
                        "Missing channel params for matrix node pair (" << txNodeId << ","
                                                                        << rxNodeId << ")");
        const ParamsEntry& params = DynamicCast<ThreeGppChannelParams>(pit->second);
        NS_ABORT_MSG_IF(params == nullptr, "Cached params are not ThreeGppChannelParams");

        const bool paramsSameDirection =
            params->m_nodeIds.first == txNodeId && params->m_nodeIds.second == rxNodeId;
        const bool paramsReverseDirection =
            params->m_nodeIds.first == rxNodeId && params->m_nodeIds.second == txNodeId;
        NS_ABORT_MSG_IF(!paramsSameDirection && !paramsReverseDirection,
                        "Matrix direction and params direction disagree");

        const SiteRec& site = getOrCreateSite(txNodeId, txAntId);
        const UtRec& ut = getOrCreateUt(rxNodeId, rxAntId, params, paramsSameDirection);

        const NodeInfo& sNodeInfo = buildNodeInfo(txNodeId);
        const NodeInfo& uNodeInfo = buildNodeInfo(rxNodeId);
        const PanelInfo& bsPanel = buildPanelInfo(txAntId, true);
        const PanelInfo& uePanel = buildPanelInfo(rxAntId, false);

        RuntimeLinkCtx ctx{};
        ctx.paramsKey = paramsKey;
        ctx.matrixKey = matrixKey;
        ctx.sNodeId = txNodeId;
        ctx.uNodeId = rxNodeId;
        ctx.sAntId = txAntId;
        ctx.uAntId = rxAntId;
        ctx.cid = site.cellIdx;
        ctx.uid = ut.utIdx;
        ctx.sMob = sNodeInfo.mob;
        ctx.uMob = uNodeInfo.mob;
        ctx.sAnt =
            ConstCast<PhasedArrayModel>(DynamicCast<const PhasedArrayModel>(bsPanel.antenna));
        ctx.uAnt =
            ConstCast<PhasedArrayModel>(DynamicCast<const PhasedArrayModel>(uePanel.antenna));
        ctx.condition = m_channelConditionModel->GetChannelCondition(ctx.sMob, ctx.uMob);
        runtimeLinks.push_back(ctx);
    }

    NS_ABORT_MSG_IF(!haveBsReference || !haveUtReference,
                    "SlsChanWgpu requires antCfgs[0]=BS and antCfgs[1]=UE reference configs");
    NS_ABORT_MSG_IF(antCfgs.size() < 2, "Need at least two antenna configs: [0]=BS, [1]=UE");
    NS_ABORT_MSG_IF(bsReferencePanelIdx != 0, "Expected BS reference panel at antCfgs[0]");
    NS_ABORT_MSG_IF(utReferencePanelIdx != 1, "Expected UE reference panel at antCfgs[1]");
    NS_ABORT_MSG_IF(globalBsAnt == 0 || globalUeAnt == 0,
                    "Global BS/UE antenna counts must be non-zero");

    NS_ABORT_MSG_IF(cells.empty(), "No BS-side cells inferred from cached matrices");
    NS_ABORT_MSG_IF(uts.empty(), "No UT-side terminals inferred from cached matrices");

    const uint32_t nSite = static_cast<uint32_t>(cells.size());
    const uint32_t nUt = static_cast<uint32_t>(uts.size());
    const uint32_t nSnapshots = 14;
    const uint32_t nPrbg = 53;

    uint32_t linkIdx = 0;
    for (auto& ctx : runtimeLinks)
    {
        ctx.lspReadIdx = ctx.cid * nUt + ctx.uid;

        ActiveLink al{};
        al.cid = ctx.cid;
        al.uid = ctx.uid;
        al.linkIdx = linkIdx;
        al.lspReadIdx = ctx.lspReadIdx;

        const uint32_t cirCoeElemsPerLink = nSnapshots * globalUeAnt * globalBsAnt * 24u;
        const uint32_t cirNormDelayElemsPerLink = 24u;
        const uint32_t cirNtapsElemsPerLink = 1u;
        const uint32_t freqChanElemsPerLink = nSnapshots * globalUeAnt * globalBsAnt * nPrbg;

        al.cirCoeOffset = linkIdx * cirCoeElemsPerLink;
        al.cirNormDelayOffset = linkIdx * cirNormDelayElemsPerLink;
        al.cirNtapsOffset = linkIdx * cirNtapsElemsPerLink;
        al.freqChanPrbgOffset = linkIdx * freqChanElemsPerLink;

        activeLinks.push_back(al);
        ++linkIdx;
    }

    const uint32_t nActiveLinks = static_cast<uint32_t>(activeLinks.size());
    NS_ABORT_MSG_IF(nActiveLinks == 0, "No active links inferred from cached matrices");

    const float margin = 50.f;
    const float maxXf = std::isfinite(maxX) ? static_cast<float>(maxX + margin) : 1000.f;
    const float minXf = std::isfinite(minX) ? static_cast<float>(minX - margin) : -1000.f;
    const float maxYf = std::isfinite(maxY) ? static_cast<float>(maxY + margin) : 1000.f;
    const float minYf = std::isfinite(minY) ? static_cast<float>(minY - margin) : -1000.f;

    // uploadCellParams default nSectorPerSite=3 (for the Phase-1
    // calibration deployment). The ns-3 batch path treats every cell
    // as its own site, so we pass 1 explicitly. Without this nSite_
    // would be set to cells.size()/3, dispatching the LSP kernel over
    // a wrongly-sized grid and writing zeros into the LinkParams buffer.
    gpu->uploadCellParams(cells, /*nSectorPerSite=*/1);
    gpu->uploadUtParams(uts);

    // Force the GPU LOS state to agree with the CPU's ChannelCondition
    // verdict. Without this the GPU samples its own LOS probability via
    // cal_los_prob() and writes whatever it gets into LinkParams.losInd
    // -- which then makes calClusterRay read LSPs from the wrong slot
    // (NLOS instead of LOS) and the threshold filter at the end of the
    // kernel zeros out the cluster set. Mirror the base class's
    // gpuScenario + per-link agreement logic so we only force when the
    // condition is unambiguous.
    uint32_t gpuScenario = 0;
    if (m_scenario == "UMi-StreetCanyon" || m_scenario == "UMi")
        gpuScenario = 1;
    else if (m_scenario == "RMa")
        gpuScenario = 2;
    int losAgreed = 0;
    int nlosAgreed = 0;
    for (const auto& ctx : runtimeLinks)
    {
        if (ctx.condition->GetLosCondition() == ChannelCondition::LOS)
            ++losAgreed;
        else if (ctx.condition->GetLosCondition() == ChannelCondition::NLOS)
            ++nlosAgreed;
    }
    float forceLosOutdoor = -1.0f;
    if (losAgreed == static_cast<int>(runtimeLinks.size()))
        forceLosOutdoor = 1.0f;
    else if (nlosAgreed == static_cast<int>(runtimeLinks.size()))
        forceLosOutdoor = 0.0f;
    gpu->setSystemLevelConfig(gpuScenario,
                              /*enablePropagationDelay=*/1,
                              /*o2iBldg=*/0,
                              /*o2iCar=*/0,
                              /*forceLosIndoor=*/forceLosOutdoor,
                              forceLosOutdoor);
    gpu->setCenterFrequencyHz(static_cast<float>(m_frequency));

    const float corrLos[8] = {37.f, 12.f, 30.f, 18.f, 15.f, 15.f, 15.f, 20.f};
    const float corrNlos[7] = {50.f, 40.f, 50.f, 50.f, 50.f, 50.f, 25.f};
    const float corrO2i[7] = {10.f, 10.f, 11.f, 17.f, 25.f, 25.f, 20.f};

    gpu->generateCRN(maxXf, minXf, maxYf, minYf, corrLos, corrNlos, corrO2i);

    // The four bool args used to be tracked locally as updatePL etc.;
    // since the mezanine path is a full LSP regen on every UpdateChannel
    // tick, we just request all updates.
    gpu->calLinkParam(nSite,
                      nUt,
                      1,
                      maxXf,
                      minXf,
                      maxYf,
                      minYf,
                      /*updatePL=*/true,
                      /*updateAllLSPs=*/true,
                      /*updateLos=*/true,
                      /*updateOptionalPL=*/false,
                      gpu->nX(),
                      gpu->nY());

    gpu->uploadSmallScaleConfig(15000.0f,
                                4096,
                                106,
                                nPrbg,
                                nSnapshots,
                                0,
                                0,
                                0,
                                0,
                                static_cast<float>(299792458.0 / m_frequency));

    gpu->uploadAntPanelConfigs(antCfgs, antThetaFlat, antPhiFlat);

    SsCmnParams ss{};
    ss.lambda_0 = static_cast<float>(299792458.0 / m_frequency);
    ss.lgfc = static_cast<float>(std::log10(m_frequency / 1e9));

    // m_scenario is std::string now (was an enum). Compare against the
    // 3GPP scenario string keys that ThreeGppChannelModel accepts.
    const bool isUmI = (m_scenario == "UMi-StreetCanyon");
    const bool isRma = (m_scenario == "RMa");
    const float lgfc_m = std::log10(std::max(m_frequency, 6e9) / 1e9); // UUmMa style
    const float lg1fc_m = std::log10(1.0f + m_frequency / 1e9);        // UMi-style

    // Table 7.5-6: DS (spread) mu/sigma [0]=NLOS, [1]=LOS, [2]=O2I
    if (!isRma)
    {
        const float fUuMa = lgfc_m;
        ss.mu_lgDS[0] = -6.28f - 0.204f * fUuMa;
        ss.mu_lgDS[1] = -6.955f - 0.0963f * fUuMa;
        ss.mu_lgDS[2] = -6.62f;
        ss.sigma_lgDS[0] = 0.39f;
        ss.sigma_lgDS[1] = 0.66f;
        ss.sigma_lgDS[2] = 0.32f;
    }
    if (isUmI)
    {
        ss.mu_lgDS[0] = -6.83f - 0.24f * lg1fc_m;
        ss.mu_lgDS[1] = -7.14f - 0.24f * lg1fc_m;
        ss.mu_lgDS[2] = -6.62f;
        ss.sigma_lgDS[0] = 0.28f + 0.16f * lg1fc_m;
        ss.sigma_lgDS[1] = 0.38f;
        ss.sigma_lgDS[2] = 0.32f;
    }
    if (isRma)
    {
        ss.mu_lgDS[0] = -7.43f;
        ss.mu_lgDS[1] = -7.49f;
        ss.mu_lgDS[2] = -7.47f;
        ss.sigma_lgDS[0] = 0.48f;
        ss.sigma_lgDS[1] = 0.55f;
        ss.sigma_lgDS[2] = 0.24f;
    }

    // Table 7.5-6: ASD mu/sigma
    if (!isRma)
    {
        ss.mu_lgASD[0] = 1.5f - 0.1144f * lgfc_m;
        ss.mu_lgASD[1] = 1.06f + 0.1114f * lgfc_m;
        ss.mu_lgASD[2] = 1.25f;
        ss.sigma_lgASD[0] = 0.28f;
        ss.sigma_lgASD[1] = 0.28f;
        ss.sigma_lgASD[2] = 0.42f;
    }
    if (isUmI)
    {
        ss.mu_lgASD[0] = 1.53f - 0.23f * lg1fc_m;
        ss.mu_lgASD[1] = 1.21f - 0.05f * lg1fc_m;
        ss.mu_lgASD[2] = 1.25f;
        ss.sigma_lgASD[0] = 0.33f + 0.11f * lg1fc_m;
        ss.sigma_lgASD[1] = 0.41f;
        ss.sigma_lgASD[2] = 0.42f;
    }
    if (isRma)
    {
        ss.mu_lgASD[0] = 0.95f;
        ss.mu_lgASD[1] = 0.9f;
        ss.mu_lgASD[2] = 0.67f;
        ss.sigma_lgASD[0] = 0.45f;
        ss.sigma_lgASD[1] = 0.38f;
        ss.sigma_lgASD[2] = 0.18f;
    }

    // Table 7.5-6: ASA mu/sigma
    if (!isRma)
    {
        ss.mu_lgASA[0] = 2.08f - 0.27f * lgfc_m;
        ss.mu_lgASA[1] = 1.81f;
        ss.mu_lgASA[2] = 1.76f;
        ss.sigma_lgASA[0] = 0.11f;
        ss.sigma_lgASA[1] = 0.20f;
        ss.sigma_lgASA[2] = 0.16f;
    }
    if (isUmI)
    {
        ss.mu_lgASA[0] = 1.81f - 0.08f * lg1fc_m;
        ss.mu_lgASA[1] = 1.73f - 0.08f * lg1fc_m;
        ss.mu_lgASA[2] = 1.76f;
        ss.sigma_lgASA[0] = 0.3f + 0.05f * lg1fc_m;
        ss.sigma_lgASA[1] = 0.28f + 0.014f * lg1fc_m;
        ss.sigma_lgASA[2] = 0.16f;
    }
    if (isRma)
    {
        ss.mu_lgASA[0] = 1.52f;
        ss.mu_lgASA[1] = 1.52f;
        ss.mu_lgASA[2] = 1.66f;
        ss.sigma_lgASA[0] = 0.13f;
        ss.sigma_lgASA[1] = 0.24f;
        ss.sigma_lgASA[2] = 0.21f;
    }

    // Table 7.5-6: ZSA mu/sigma
    if (!isRma)
    {
        ss.mu_lgZSA[0] = 1.512f - 0.3236f * lgfc_m;
        ss.mu_lgZSA[1] = 0.95f;
        ss.mu_lgZSA[2] = 1.01f;
        ss.sigma_lgZSA[0] = 0.16f;
        ss.sigma_lgZSA[1] = 0.16f;
        ss.sigma_lgZSA[2] = 0.43f;
    }
    if (isUmI)
    {
        ss.mu_lgZSA[0] = 0.92f - 0.04f * lg1fc_m;
        ss.mu_lgZSA[1] = 0.73f - 0.1f * lg1fc_m;
        ss.mu_lgZSA[2] = 1.01f;
        ss.sigma_lgZSA[0] = 0.41f - 0.07f * lg1fc_m;
        ss.sigma_lgZSA[1] = 0.34f - 0.04f * lg1fc_m;
        ss.sigma_lgZSA[2] = 0.43f;
    }
    if (isRma)
    {
        ss.mu_lgZSA[0] = 0.58f;
        ss.mu_lgZSA[1] = 0.47f;
        ss.mu_lgZSA[2] = 0.93f;
        ss.sigma_lgZSA[0] = 0.37f;
        ss.sigma_lgZSA[1] = 0.4f;
        ss.sigma_lgZSA[2] = 0.22f;
    }

    // K-factor (LOS only)
    if (!isRma)
    {
        ss.mu_K[0] = 0.0f;
        ss.mu_K[1] = 9.0f;
        ss.mu_K[2] = 0.0f;
        ss.sigma_K[0] = 0.0f;
        ss.sigma_K[1] = 3.5f;
        ss.sigma_K[2] = 0.0f;
    }
    if (isRma)
    {
        ss.mu_K[0] = 0.0f;
        ss.mu_K[1] = 7.0f;
        ss.mu_K[2] = 0.0f;
        ss.sigma_K[0] = 0.0f;
        ss.sigma_K[1] = 4.0f;
        ss.sigma_K[2] = 0.0f;
    }
    // nCluster (NLOS / LOS / O2I)
    ss.nCluster[0] = isUmI ? 19 : (isRma ? 10 : 20);
    ss.nCluster[1] = isRma ? 11 : 12;
    ss.nCluster[2] = isRma ? 3 : 12;
    ss.nRayPerCluster[0] = 20;
    ss.nRayPerCluster[1] = 20;
    ss.nRayPerCluster[2] = 20;

    // r_tao
    if (isUmI)
    {
        ss.r_tao[0] = 2.1f;
        ss.r_tao[1] = 3.0f;
        ss.r_tao[2] = 2.2f;
    }
    else if (isRma)
    {
        ss.r_tao[0] = 1.7f;
        ss.r_tao[1] = 3.8f;
        ss.r_tao[2] = 1.7f;
    }
    else
    {
        ss.r_tao[0] = 2.3f;
        ss.r_tao[1] = 2.5f;
        ss.r_tao[2] = 2.2f;
    }

    // C_DS (per-cluster distance spread in meters)
    if (isUmI)
    {
        ss.C_DS[0] = 11.0f;
        ss.C_DS[1] = 5.0f;
        ss.C_DS[2] = 11.0f;
    }
    else if (isRma)
    {
        ss.C_DS[0] = 0.0f;
        ss.C_DS[1] = 0.0f;
        ss.C_DS[2] = 0.0f;
    }
    else
    {
        ss.C_DS[0] = std::max(0.25f, 6.5622f - 3.4084f * lgfc_m);
        ss.C_DS[1] = std::max(0.25f, 6.5622f - 3.4084f * lgfc_m);
        ss.C_DS[2] = 11.0f;
    }

    // C_ASD, C_ASA, C_ZSA
    if (isUmI)
    {
        ss.C_ASD[0] = 10.0f;
        ss.C_ASD[1] = 3.0f;
        ss.C_ASD[2] = 5.0f;
        ss.C_ASA[0] = 22.0f;
        ss.C_ASA[1] = 17.0f;
        ss.C_ASA[2] = 8.0f;
    }
    else if (isRma)
    {
        ss.C_ASD[0] = 2.0f;
        ss.C_ASD[1] = 2.0f;
        ss.C_ASD[2] = 2.0f;
        ss.C_ASA[0] = 3.0f;
        ss.C_ASA[1] = 3.0f;
        ss.C_ASA[2] = 3.0f;
    }
    else
    {
        ss.C_ASD[0] = 2.0f;
        ss.C_ASD[1] = 5.0f;
        ss.C_ASD[2] = 5.0f;
        ss.C_ASA[0] = 15.0f;
        ss.C_ASA[1] = 11.0f;
        ss.C_ASA[2] = 8.0f;
    }
    ss.C_ZSA[0] = 7.0f;
    ss.C_ZSA[1] = 7.0f;
    ss.C_ZSA[2] = 3.0f;
    if (isRma)
    {
        ss.C_ZSA[0] = 3.0f;
        ss.C_ZSA[1] = 3.0f;
        ss.C_ZSA[2] = 3.0f;
    }

    // xi
    if (isRma)
    {
        ss.xi[0] = 4.0f;
        ss.xi[1] = 4.0f;
        ss.xi[2] = 6.0f;
    }
    else
    {
        ss.xi[0] = 3.0f;
        ss.xi[1] = 3.0f;
        ss.xi[2] = 4.0f;
    }

    // mu_XPR, sigma_XPR (same for UMa/UMi, different for RMa)
    if (!isRma)
    {
        ss.mu_XPR[0] = 7.0f;
        ss.mu_XPR[1] = 8.0f;
        ss.mu_XPR[2] = 9.0f;
        ss.sigma_XPR[0] = 3.0f;
        ss.sigma_XPR[1] = 4.0f;
        ss.sigma_XPR[2] = 5.0f;
    }
    if (isRma)
    {
        ss.mu_XPR[0] = 7.0f;
        ss.mu_XPR[1] = 12.0f;
        ss.mu_XPR[2] = 9.0f;
        ss.sigma_XPR[0] = 3.0f;
        ss.sigma_XPR[1] = 4.0f;
        ss.sigma_XPR[2] = 5.0f;
    }

    // delta_tau means/sigmas (log10) per 3GPP TR 38.901 Table 7.6.9-1
    // Indexed by scenario: [UMi=0, UMa=1, RMa=2]
    // Only NLOS values are meaningful (LOS delta_tau = 0 per Eq. 7.6-44)
    ss.mu_lgDT[0] = -7.5f;
    ss.sigma_lgDT[0] = 0.5f; // UMi
    ss.mu_lgDT[1] = -7.4f;
    ss.sigma_lgDT[1] = 0.2f; // UMa
    ss.mu_lgDT[2] = -8.33f;
    ss.sigma_lgDT[2] = 0.26f; // RMa

    ss.C_phi_LOS = 1.0f;
    ss.C_phi_NLOS = 1.0f;
    ss.C_phi_O2I = 1.0f;
    ss.C_theta_LOS = 1.0f;
    ss.C_theta_NLOS = 1.0f;
    ss.C_theta_O2I = 1.0f;

    // nSubCluster + subcluster ray indices (from CUDA reference)
    ss.nSubCluster = 3;
    ss.nUeAnt = globalUeAnt;
    ss.nBsAnt = globalBsAnt;
    // 3GPP TR 38.901 Table 7.5-5 subcluster ray indices (per-cluster)
    // Cluster 0: 10 subclusters {0..7, 18, 19}, Cluster 1: 6 subclusters {8..11, 16, 17},
    // Cluster 2: 4 subclusters {12..15}
    static constexpr unsigned short sub0[10] = {0, 1, 2, 3, 4, 5, 6, 7, 18, 19};
    static constexpr unsigned short sub1[6] = {8, 9, 10, 11, 16, 17};
    static constexpr unsigned short sub2[4] = {12, 13, 14, 15};
    std::copy(std::begin(sub0), std::end(sub0), ss.raysInSubCluster0);
    std::copy(std::begin(sub1), std::end(sub1), ss.raysInSubCluster1);
    std::copy(std::begin(sub2), std::end(sub2), ss.raysInSubCluster2);
    ss.raysInSubClusterSizes[0] = 10;
    ss.raysInSubClusterSizes[1] = 6;
    ss.raysInSubClusterSizes[2] = 4;

    static constexpr float rayOffsets[20] = {0.0447f,  -0.0447f, 0.1413f,  -0.1413f, 0.2492f,
                                             -0.2492f, 0.3715f,  -0.3715f, 0.5129f,  -0.5129f,
                                             0.6797f,  -0.6797f, 0.8844f,  -0.8844f, 1.1481f,
                                             -1.1481f, 1.5195f,  -1.5195f, 2.1551f,  -2.1551f};
    std::copy(std::begin(rayOffsets), std::end(rayOffsets), ss.RayOffsetAngles);

    // The mezanine populates per-condition fields in [NLOS, LOS, O2I]
    // order (see comment at line ~593 "[0]=NLOS, [1]=LOS, [2]=O2I").
    // The WGSL kernel however indexes via
    //   lsp_idx = select(select(1u, 0u, lk.losInd != 0u), 2u, isO2i)
    // which maps LOS=0, NLOS=1, O2I=2 -- i.e. slot 0 and slot 1 are
    // swapped relative to the mezanine's intent. Without this fix, LOS
    // links read NLOS LSP parameters and vice versa; for the
    // AlwaysLosChannelConditionModel bench that meant the kernel ran
    // with totally wrong DS/K-factor priors and produced nCluster=0
    // for every link, tripping a downstream vector[0] assertion in
    // ThreeGppChannelModel.
    // Swap slots [0]<->[1] for every per-condition array so the WGSL
    // sees the right value for each lsp_idx.
    auto swapLN = [](auto& arr) { std::swap(arr[0], arr[1]); };
    swapLN(ss.mu_lgDS);
    swapLN(ss.sigma_lgDS);
    swapLN(ss.mu_lgASD);
    swapLN(ss.sigma_lgASD);
    swapLN(ss.mu_lgASA);
    swapLN(ss.sigma_lgASA);
    swapLN(ss.mu_lgZSA);
    swapLN(ss.sigma_lgZSA);
    swapLN(ss.mu_K);
    swapLN(ss.sigma_K);
    swapLN(ss.mu_XPR);
    swapLN(ss.sigma_XPR);
    swapLN(ss.mu_lgDT);
    swapLN(ss.sigma_lgDT);
    swapLN(ss.nCluster);
    swapLN(ss.nRayPerCluster);
    swapLN(ss.r_tao);
    swapLN(ss.C_DS);
    swapLN(ss.C_ASD);
    swapLN(ss.C_ASA);
    swapLN(ss.C_ZSA);
    swapLN(ss.xi);

    gpu->uploadCmnLinkParamsSmallScale(ss);
    gpu->uploadCellParamsSS(cellsSS);
    gpu->uploadUtParamsSS(utsSS);

    {
        SLS_PHASE_SCOPE("Mez::SmallScaleKernels");
        gpu->calClusterRay(nSite, nUt);
        gpu->generateCIR(activeLinks, nActiveLinks, nSnapshots, 0.0f);

        // After cluster delays + ray angles + xpr + random phases are on
        // the GPU, dispatch the channel-matrix kernel and read back the
        // per-link Complex3DVector blocks. Pass kMatMaxPages as the
        // upper bound -- the kernel writes zeros for pages past the
        // per-link numOverallCluster.
        gpu->genChannelMatrix(activeLinks,
                              nActiveLinks,
                              /*uSize=*/globalUeAnt,
                              /*sSize=*/globalBsAnt,
                              /*numOverallCluster=*/SlsChanWgpu::kMatMaxPages,
                              /*numReducedCluster=*/0u,
                              /*nRays=*/20u,
                              /*cluster1st=*/0u,
                              /*cluster2nd=*/0u);
    }
    std::vector<std::complex<float>> matFlat;
    {
        SLS_PHASE_SCOPE("Mez::ReadMatrix");
        matFlat = gpu->readChannelMatrix(nActiveLinks, globalUeAnt, globalBsAnt);
    }
    const size_t perLinkMatLen =
        size_t(globalUeAnt) * globalBsAnt * SlsChanWgpu::kMatMaxPages;

    NS_LOG_DEBUG("UpdateChannel uploaded " << nSite << " cells, " << nUt << " UEs, "
                                           << antCfgs.size() << " antenna panels, " << nActiveLinks
                                           << " active links");

    std::vector<LinkParams> linkParams;
    std::vector<ClusterParamsGpu> clusterParams;
    std::vector<float> xprFlat;
    std::vector<float> phiNmAoaFlat;
    std::vector<float> phiNmAodFlat;
    std::vector<float> thetaNmZoaFlat;
    std::vector<float> thetaNmZodFlat;
    {
        SLS_PHASE_SCOPE("Mez::ReadParams");
        linkParams = gpu->readLinkParams(nSite, nUt);
        clusterParams = gpu->readClusterParams(nSite, nUt);
        xprFlat = gpu->readXpr();
        phiNmAoaFlat = gpu->readPhiNmAoA();
        phiNmAodFlat = gpu->readPhiNmAoD();
        thetaNmZoaFlat = gpu->readThetaNmZOA();
        thetaNmZodFlat = gpu->readThetaNmZOD();
    }
    SLS_PHASE_SCOPE("Mez::Populate");

    for (const auto& ctx : runtimeLinks)
    {
        const LinkParams& lk = linkParams.at(ctx.lspReadIdx);
        const ClusterParamsGpu& cp = clusterParams.at(ctx.lspReadIdx);

        NS_ABORT_MSG_IF(cp.nCluster > MAX_CLUSTERS, "GPU cluster count out of range.");
        NS_ABORT_MSG_IF(cp.nRayPerCluster > MAX_RAYS, "GPU ray count out of range.");

        Ptr<ThreeGppChannelParams> prev;
        if (auto it = m_channelParamsMap.find(ctx.paramsKey); it != m_channelParamsMap.end())
        {
            prev = it->second;
        }

        Ptr<ThreeGppChannelParams> params = Create<ThreeGppChannelParams>();
        params->m_generatedTime = Simulator::Now();

        Ptr<MobilityModel> aMobOrdered = ctx.sMob;
        Ptr<MobilityModel> bMobOrdered = ctx.uMob;
        if (ctx.sNodeId > ctx.uNodeId)
        {
            std::swap(aMobOrdered, bMobOrdered);
        }

        params->m_nodeIds = std::make_pair(aMobOrdered->GetObject<Node>()->GetId(),
                                           bMobOrdered->GetObject<Node>()->GetId());

        params->m_losCondition = lk.losInd ? ChannelCondition::LOS : ChannelCondition::NLOS;
        params->m_o2iCondition = ctx.condition->GetO2iCondition();

        UpdateLinkGeometry(aMobOrdered,
                           bMobOrdered,
                           &params->m_dis2D,
                           &params->m_dis3D,
                           &params->m_endpointDisplacement2D,
                           &params->m_relativeDisplacement2D,
                           &params->m_lastPositionFirst,
                           &params->m_lastPositionSecond,
                           &params->m_lastRelativePosition2D);

        params->m_txSpeed = aMobOrdered->GetVelocity();
        params->m_rxSpeed = bMobOrdered->GetVelocity();

        params->m_DS = lk.DS;
        params->m_K_factor = lk.K;
        params->m_reducedClusterNumber = static_cast<uint8_t>(cp.nCluster);

        params->m_delay.assign(cp.delays, cp.delays + cp.nCluster);
        params->m_clusterPower.assign(cp.powers, cp.powers + cp.nCluster);
        params->m_cluster1st = static_cast<uint8_t>(cp.strongest2clustersIdx[0]);
        params->m_cluster2nd = static_cast<uint8_t>(cp.strongest2clustersIdx[1]);

        params->m_angle.resize(4);
        params->m_angle[AOA_INDEX].assign(cp.phinAoA, cp.phinAoA + cp.nCluster);
        params->m_angle[AOD_INDEX].assign(cp.phinAoD, cp.phinAoD + cp.nCluster);
        params->m_angle[ZOA_INDEX].assign(cp.thetanZOA, cp.thetanZOA + cp.nCluster);
        params->m_angle[ZOD_INDEX].assign(cp.thetanZOD, cp.thetanZOD + cp.nCluster);

        params->m_rayAoaRadian.assign(cp.nCluster, DoubleVector(cp.nRayPerCluster, 0.0));
        params->m_rayAodRadian.assign(cp.nCluster, DoubleVector(cp.nRayPerCluster, 0.0));
        params->m_rayZoaRadian.assign(cp.nCluster, DoubleVector(cp.nRayPerCluster, 0.0));
        params->m_rayZodRadian.assign(cp.nCluster, DoubleVector(cp.nRayPerCluster, 0.0));
        params->m_crossPolarizationPowerRatios.assign(cp.nCluster,
                                                      DoubleVector(cp.nRayPerCluster, 0.0));

        // The cluster-ray kernel emits angles in degrees (wrap_azimuth
        // returns [-180, 180]; wrap_zenith returns [0, 180]). ns-3's
        // m_rayA*Radian / m_rayZ*Radian fields are documented in
        // radians; downstream ns3::Angles construction asserts
        // inclination in [0, pi] rad and will otherwise terminate
        // with "m_inclination=NN.NN not valid". Convert at the
        // boundary. (Cluster angles in m_angle stay in degrees per
        // ns-3 convention -- GenerateRayAngles converts them right
        // before consumption.)
        constexpr double kDeg2Rad = M_PI / 180.0;
        for (uint32_t c = 0; c < cp.nCluster; ++c)
        {
            for (uint32_t r = 0; r < cp.nRayPerCluster; ++r)
            {
                const uint32_t flat = FlatClusterRayIndex(ctx.lspReadIdx, c, r);
                params->m_rayAoaRadian[c][r] = phiNmAoaFlat.at(flat) * kDeg2Rad;
                params->m_rayAodRadian[c][r] = phiNmAodFlat.at(flat) * kDeg2Rad;
                params->m_rayZoaRadian[c][r] = thetaNmZoaFlat.at(flat) * kDeg2Rad;
                params->m_rayZodRadian[c][r] = thetaNmZodFlat.at(flat) * kDeg2Rad;
                params->m_crossPolarizationPowerRatios[c][r] = xprFlat.at(flat);
            }
        }

        if (prev && SameDims(prev->m_clusterPhase, cp.nCluster, cp.nRayPerCluster))
        {
            params->m_clusterPhase = prev->m_clusterPhase;
        }
        else
        {
            params->m_clusterPhase.assign(cp.nCluster,
                                          Double2DVector(cp.nRayPerCluster, DoubleVector(4u, 0.0)));
        }

        params->m_clusterShadowing.assign(cp.nCluster, 0.0);
        params->m_attenuation_dB.assign(cp.nCluster, 0.0);
        params->m_nonSelfBlocking.clear();
        params->m_norRvAngles.clear();
        // V2V/V2X Doppler scaling terms. The bench doesn't model V2V,
        // and the CPU's GenerateDopplerTerms initialises both to 0
        // for cluster 0 then random ±1 / ±m_vScatt for the rest. For
        // a stationary or low-vScatt config zero is the correct
        // initialisation; otherwise downstream CalcBeamformingGain
        // hits an assertion when m_alpha.size() < numCluster.
        params->m_alpha.assign(cp.nCluster, 0.0);
        params->m_D.assign(cp.nCluster, 0.0);

        if (prev && prev->m_clusterXnNlosSign.size() == cp.nCluster)
        {
            params->m_clusterXnNlosSign = prev->m_clusterXnNlosSign;
        }
        else
        {
            params->m_clusterXnNlosSign.assign(cp.nCluster, 1);
        }

        params->m_delayConsistency = params->m_delay;
        for (auto& d : params->m_delayConsistency)
        {
            d += params->m_dis3D / 3e8;
        }

        // FindStrongestClusters appends 2 or 4 subcluster entries to
        // every per-cluster vector (delay, angle, alpha, D,
        // clusterPower) so that downstream GetNewChannel finds them
        // sized to numOverallCluster = reducedClusterNumber + 2|4
        // (depending on cluster1st == cluster2nd). The
        // CalcBeamformingGain assertion at line 313 of
        // three-gpp-spectrum-propagation-loss-model.cc expects
        // m_alpha / m_D / m_angle[*] to all match the channel
        // matrix's page count, which is the expanded size. The base
        // class also overwrites cluster1st / cluster2nd here -- the
        // GPU's strongest2 are advisory; ns-3 wants the CPU-side
        // verdict.
        Ptr<ParamsTable> dummyTable = Create<ParamsTable>();
        dummyTable->m_cDS = 0.0; // subcluster delay spread; 0 == no offset
        FindStrongestClusters(params,
                              dummyTable,
                              &params->m_cluster1st,
                              &params->m_cluster2nd,
                              &params->m_delay,
                              &params->m_angle,
                              &params->m_alpha,
                              &params->m_D,
                              &params->m_clusterPower);

        params->m_cachedAngleSincos.clear();
        PrecomputeAnglesSinCos(params, &params->m_cachedAngleSincos);

        m_channelParamsMap.insert_or_assign(ctx.paramsKey, params);

        // Build the per-link ChannelMatrix straight from the GPU's
        // matrix kernel output. Layout per link (per kernel):
        //   element (u, s, n) at offset n*uSize*sSize + s*uSize + u
        //   within a uSize*sSize*kMatMaxPages f32-complex slab.
        // ns3::MatrixArray<T> stores column-major:
        //   (r=u, c=s, p=n) at u + uSize*s + uSize*sSize*n
        // Both layouts are contiguous in the same order, so the
        // entire per-link block (numOverallCluster pages worth)
        // is one flat f32->f64 conversion pass -- a single tight
        // loop the compiler can vectorise (CVTPS2PD on x86 turns
        // 2 floats into 2 doubles per instruction; AVX widens to
        // 4). This collapses the prior nested page/cell loop and
        // its repeated GetPagePtr() vtable hops.
        const uint8_t numOverallCluster =
            params->m_cluster1st != params->m_cluster2nd
                ? static_cast<uint8_t>(params->m_reducedClusterNumber + 4)
                : static_cast<uint8_t>(params->m_reducedClusterNumber + 2);
        Ptr<ChannelMatrix> matrix = Create<ChannelMatrix>();
        matrix->m_generatedTime = Simulator::Now();
        matrix->m_nodeIds = std::make_pair(
            ctx.sMob->GetObject<Node>()->GetId(),
            ctx.uMob->GetObject<Node>()->GetId());
        matrix->m_antennaPair = std::make_pair(ctx.sAntId, ctx.uAntId);
        matrix->m_channel = MatrixBasedChannelModel::Complex3DVector(
            globalUeAnt, globalBsAnt, numOverallCluster);
        const size_t perPage = size_t(globalUeAnt) * globalBsAnt;
        const size_t linkBase = size_t(ctx.lspReadIdx) * perLinkMatLen;
        const size_t linkCells = perPage * size_t(numOverallCluster);
        const std::complex<float>* __restrict__ src = matFlat.data() + linkBase;
        std::complex<double>* __restrict__ dst = matrix->m_channel.GetPagePtr(0);
        for (size_t k = 0; k < linkCells; ++k)
        {
            // std::complex<T> is required to be layout-compatible
            // with T[2]; reinterpret to a real-pair so the compiler
            // sees a plain f32 -> f64 widening pair and emits
            // CVTPS2PD on the contiguous stream.
            const auto* sr = reinterpret_cast<const float*>(src + k);
            auto* dr = reinterpret_cast<double*>(dst + k);
            dr[0] = static_cast<double>(sr[0]);
            dr[1] = static_cast<double>(sr[1]);
        }
        m_channelMatrixMap.insert_or_assign(ctx.matrixKey, matrix);
    }

    NS_LOG_DEBUG("Updated GPU-backed channel params + matrix caches.");
}

void
ThreeGppChannelModelWgpuMezanine::EnsureBatchFresh()
{
    NS_LOG_FUNCTION(this);
    // Dedup ourselves: EnsureBatchFresh is called once per per-link
    // CalcRxPowerSpectralDensity invocation, but we only want to do the
    // GPU batch work once per simulator-time tick. Without this, the
    // 2nd per-link call within the same tick would re-enter
    // UpdateChannel with m_channelMatrixMap containing just the 1
    // entry the previous link's GetChannel had populated -- the GPU
    // pipeline would then run with nSite=1/nUt=1, producing garbage
    // LSPs and cp.nCluster=0 downstream.
    const Time now = Simulator::Now();
    if (m_lastMezBatchTime.has_value() && *m_lastMezBatchTime == now)
    {
        return;
    }
    m_lastMezBatchTime = now;

    // 1. LSP draw (base class GPU kernel; fills m_gpuLspCache).
    ThreeGppChannelModel::EnsureBatchFresh();
    // 2. Small-scale (calClusterRay + generateCIR) on GPU, then write
    //    cluster delays / ray angles / XPR / cluster powers into
    //    m_channelParamsMap so the per-link GetNewChannel skips the
    //    CPU small-scale draw.
    //
    //    UpdateChannel needs m_channelMatrixMap to have at least one
    //    entry per link to know what to refresh. That map gets
    //    populated by GetChannel only AFTER its first call. So the
    //    very first EnsureBatchFresh at the very first tick has
    //    nothing to act on -- tick 1 falls through to CPU. From
    //    tick 2 onward the GPU path takes over.
    if (!m_channelMatrixMap.empty())
    {
        UpdateChannel();
    }
}

Ptr<MatrixBasedChannelModel::ChannelMatrix>
ThreeGppChannelModelWgpuMezanine::GetNewChannel(Ptr<const ThreeGppChannelParams> channelParams,
                                                Ptr<const ParamsTable> table3gpp,
                                                Ptr<const MobilityModel> sMob,
                                                Ptr<const MobilityModel> uMob,
                                                Ptr<const PhasedArrayModel> sAntenna,
                                                Ptr<const PhasedArrayModel> uAntenna) const
{
    m_antennaIdToObjectMap[sAntenna->GetId()] = sAntenna;
    m_antennaIdToObjectMap[uAntenna->GetId()] = uAntenna;

    // GPU-fast path: UpdateChannel populated m_channelMatrixMap[key]
    // with the GPU-built matrix this tick. If it's there and recent
    // (params haven't moved past it), short-circuit. The cache lookup
    // collapses the CPU GetNewChannel cost (~5 ms/eval = 84% of the
    // per-eval budget at DenseAmimoIntel scale) to a hash hit.
    const uint64_t matrixKey =
        MatrixBasedChannelModel::GetKey(sAntenna->GetId(), uAntenna->GetId());
    if (auto it = m_channelMatrixMap.find(matrixKey); it != m_channelMatrixMap.end())
    {
        const Time matrixT = it->second->m_generatedTime;
        const Time paramsT = channelParams ? channelParams->m_generatedTime : Time(0);
        if (matrixT >= paramsT)
        {
            return DynamicCast<ChannelMatrix>(
                ConstCast<MatrixBasedChannelModel::ChannelMatrix>(it->second));
        }
    }

    // Fallback: CPU build (first tick before UpdateChannel has run,
    // or any link the GPU pipeline didn't see).
    return ThreeGppChannelModel::GetNewChannel(channelParams,
                                               table3gpp,
                                               sMob,
                                               uMob,
                                               sAntenna,
                                               uAntenna);
}

} // namespace ns3
