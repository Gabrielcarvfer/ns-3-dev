//
// Updated draft with WGPU integration fixes:
// - correct ActiveLink lspReadIdx = cid * nUt + uid
// - correct CIR/CFR flat-buffer offsets based on global BS/UE antenna counts
// - populated runtimeLinks for cache writeback
// - deterministic antenna config ordering: antCfgs[0] = BS, antCfgs[1] = UE
// - guards for the current SlsChanWgpu assumption of one global BS nAnt and one global UE nAnt
//

#include "sls-chan-wgpu.h"
#include "three-gpp-channel-model-wgpu-mezanine.h"
#include "ns3/simulator.h"
#include "ns3/mobility-model.h"
#include "ns3/node-list.h"
#include "ns3/phased-array-model.h"
#include "ns3/log.h"
#include "ns3/abort.h"
#include "ns3/double.h"
#include "ns3/vector.h"
#include "ns3/uinteger.h"
#include "ns3/node.h"
#include "ns3/uniform-planar-array.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

NS_LOG_COMPONENT_DEFINE("ThreeGppChannelModelWgpuMezanine");

namespace ns3 {

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
        static TypeId tid =
                TypeId("ns3::ThreeGppChannelModelWgpuMezanine")
                        .SetGroupName("Spectrum")
                        .SetParent<ThreeGppChannelModel>()
                        .AddConstructor<ThreeGppChannelModelWgpuMezanine>();
        return tid;
    }

    namespace {

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
        NS_LOG_FUNCTION(this);

        Simulator::Schedule(MilliSeconds(10), [this]() { this->UpdateChannel(); });

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
            cfg._pad1 = 0;

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
            p.antSize = {1, 1, 1, static_cast<uint32_t>(p.nAnt), 1};
            p.antSpacing = {0.f, 0.f, 0.5f, 0.5f};
            p.polarAnglesDeg = isBsSide ? std::array<float, 2>{45.f, -45.f}
                                        : std::array<float, 2>{0.f, 0.f};
            p.panelOrientation = {
                    static_cast<float>(ant->GetAlpha()),
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
            cell.loc = Vec3f{static_cast<float>(ni.pos.x),
                             static_cast<float>(ni.pos.y),
                             static_cast<float>(ni.pos.z),
                             0.f};

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
        ut.loc = Vec3f{
                static_cast<float>(ni.pos.x),
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
                            "Missing channel params for matrix node pair (" << txNodeId << "," << rxNodeId
                                                                            << ")");
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
            ctx.sAnt = ConstCast<PhasedArrayModel>(DynamicCast<const PhasedArrayModel>(bsPanel.antenna));
            ctx.uAnt = ConstCast<PhasedArrayModel>(DynamicCast<const PhasedArrayModel>(uePanel.antenna));
            ctx.condition = m_channelConditionModel->GetChannelCondition(ctx.sMob, ctx.uMob);
            runtimeLinks.push_back(ctx);
        }

        NS_ABORT_MSG_IF(!haveBsReference || !haveUtReference,
                        "SlsChanWgpu requires antCfgs[0]=BS and antCfgs[1]=UE reference configs");
        NS_ABORT_MSG_IF(antCfgs.size() < 2,
                        "Need at least two antenna configs: [0]=BS, [1]=UE");
        NS_ABORT_MSG_IF(bsReferencePanelIdx != 0,
                        "Expected BS reference panel at antCfgs[0]");
        NS_ABORT_MSG_IF(utReferencePanelIdx != 1,
                        "Expected UE reference panel at antCfgs[1]");
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

        gpu->uploadCellParams(cells);
        gpu->uploadUtParams(uts);

        const float corrLos[7] = {37.f, 12.f, 30.f, 18.f, 15.f, 15.f, 15.f};
        const float corrNlos[6] = {50.f, 40.f, 50.f, 50.f, 50.f, 50.f};
        const float corrO2i[6] = {10.f, 10.f, 11.f, 17.f, 25.f, 25.f};

        gpu->generateCRN(maxXf, minXf, maxYf, minYf, corrLos, corrNlos, corrO2i);

        gpu->calLinkParam(nSite,
                          nUt,
                          1,
                          maxXf,
                          minXf,
                          maxYf,
                          minYf,
                          true,
                          true,
                          true,
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

    // Keep these aligned with your scenario table when you expose it here.
    ss.mu_lgDS[0] = -7.03f; ss.sigma_lgDS[0] = 0.66f;
    ss.mu_lgDS[1] = -6.44f; ss.sigma_lgDS[1] = 0.39f;
    ss.mu_lgDS[2] = -6.62f; ss.sigma_lgDS[2] = 0.32f;

    ss.mu_lgASD[0] = 1.15f; ss.sigma_lgASD[0] = 0.28f;
    ss.mu_lgASD[1] = 1.41f; ss.sigma_lgASD[1] = 0.28f;
    ss.mu_lgASD[2] = 1.25f; ss.sigma_lgASD[2] = 0.42f;

    ss.mu_lgASA[0] = 1.81f; ss.sigma_lgASA[0] = 0.20f;
    ss.mu_lgASA[1] = 1.87f; ss.sigma_lgASA[1] = 0.20f;
    ss.mu_lgASA[2] = 1.76f; ss.sigma_lgASA[2] = 0.16f;

    ss.mu_lgZSA[0] = 0.95f; ss.sigma_lgZSA[0] = 0.16f;
    ss.mu_lgZSA[1] = 1.26f; ss.sigma_lgZSA[1] = 0.35f;
    ss.mu_lgZSA[2] = 1.01f; ss.sigma_lgZSA[2] = 0.43f;

    ss.mu_K[0] = 9.0f;  ss.sigma_K[0] = 3.5f;
    ss.mu_K[1] = 0.0f;  ss.sigma_K[1] = 0.0f;
    ss.mu_K[2] = 0.0f;  ss.sigma_K[2] = 0.0f;

    ss.r_tao[0] = 2.5f; ss.r_tao[1] = 2.3f; ss.r_tao[2] = 2.2f;
    ss.mu_XPR[0] = 8.0f; ss.mu_XPR[1] = 7.0f; ss.mu_XPR[2] = 9.0f;
    ss.sigma_XPR[0] = 4.0f; ss.sigma_XPR[1] = 3.0f; ss.sigma_XPR[2] = 3.0f;

    ss.nCluster[0] = 12; ss.nCluster[1] = 20; ss.nCluster[2] = 12;
    ss.nRayPerCluster[0] = 20; ss.nRayPerCluster[1] = 20; ss.nRayPerCluster[2] = 20;

    ss.C_DS[0] = 5.0f; ss.C_DS[1] = 11.0f; ss.C_DS[2] = 11.0f;
    ss.C_ASD[0] = 5.0f; ss.C_ASD[1] = 2.0f; ss.C_ASD[2] = 3.0f;
    ss.C_ASA[0] = 11.0f; ss.C_ASA[1] = 15.0f; ss.C_ASA[2] = 8.0f;
    ss.C_ZSA[0] = 7.0f; ss.C_ZSA[1] = 7.0f; ss.C_ZSA[2] = 3.0f;
    ss.xi[0] = 3.0f; ss.xi[1] = 3.0f; ss.xi[2] = 4.0f;

    ss.C_phi_LOS = 1.0f;
    ss.C_phi_NLOS = 1.0f;
    ss.C_phi_O2I = 1.0f;
    ss.C_theta_LOS = 1.0f;
    ss.C_theta_NLOS = 1.0f;
    ss.C_theta_O2I = 1.0f;

        ss.nSubCluster = 3;
        ss.nUeAnt = globalUeAnt;
        ss.nBsAnt = globalBsAnt;

        static constexpr uint32_t sub0[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        static constexpr uint32_t sub1[10] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        static constexpr uint32_t sub2[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        std::copy(std::begin(sub0), std::end(sub0), ss.raysInSubCluster0);
        std::copy(std::begin(sub1), std::end(sub1), ss.raysInSubCluster1);
        std::copy(std::begin(sub2), std::end(sub2), ss.raysInSubCluster2);
        ss.raysInSubClusterSizes[0] = 10;
        ss.raysInSubClusterSizes[1] = 10;
        ss.raysInSubClusterSizes[2] = 10;

        static constexpr float rayOffsets[20] = {
                0.0447f, -0.0447f, 0.1413f, -0.1413f, 0.2492f, -0.2492f, 0.3715f, -0.3715f,
                0.5129f, -0.5129f, 0.6797f, -0.6797f, 0.8844f, -0.8844f, 1.1481f, -1.1481f,
                1.5195f, -1.5195f, 2.1551f, -2.1551f};
        std::copy(std::begin(rayOffsets), std::end(rayOffsets), ss.RayOffsetAngles);

        gpu->uploadCmnLinkParamsSmallScale(ss);
        gpu->uploadCellParamsSS(cellsSS);
        gpu->uploadUtParamsSS(utsSS);

        gpu->calClusterRay(nSite, nUt);
        gpu->generateCIR(activeLinks, nActiveLinks, nSnapshots, 0.0f);

        NS_LOG_DEBUG("UpdateChannel uploaded "
                             << nSite << " cells, "
                             << nUt << " UEs, "
                             << antCfgs.size() << " antenna panels, "
                             << nActiveLinks << " active links");

        const std::vector<LinkParams> linkParams = gpu->readLinkParams(nSite, nUt);
        const std::vector<ClusterParamsGpu> clusterParams = gpu->readClusterParams(nSite, nUt);
        const std::vector<float> xprFlat = gpu->readXpr();
        const std::vector<float> phiNmAoaFlat = gpu->readPhiNmAoA();
        const std::vector<float> phiNmAodFlat = gpu->readPhiNmAoD();
        const std::vector<float> thetaNmZoaFlat = gpu->readThetaNmZOA();
        const std::vector<float> thetaNmZodFlat = gpu->readThetaNmZOD();

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

            params->m_nodeIds =
                    std::make_pair(aMobOrdered->GetObject<Node>()->GetId(),
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

            for (uint32_t c = 0; c < cp.nCluster; ++c)
            {
                for (uint32_t r = 0; r < cp.nRayPerCluster; ++r)
                {
                    const uint32_t flat = FlatClusterRayIndex(ctx.lspReadIdx, c, r);
                    params->m_rayAoaRadian[c][r] = phiNmAoaFlat.at(flat);
                    params->m_rayAodRadian[c][r] = phiNmAodFlat.at(flat);
                    params->m_rayZoaRadian[c][r] = thetaNmZoaFlat.at(flat);
                    params->m_rayZodRadian[c][r] = thetaNmZodFlat.at(flat);
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

            params->m_cachedAngleSincos.clear();
            PrecomputeAnglesSinCos(params, &params->m_cachedAngleSincos);

            m_channelParamsMap.insert_or_assign(ctx.paramsKey, params);
        }

        for (const auto& ctx : runtimeLinks)
        {
            auto pit = m_channelParamsMap.find(ctx.paramsKey);
            NS_ABORT_MSG_IF(pit == m_channelParamsMap.end(), "Params cache writeback failed.");

            Ptr<ThreeGppChannelParams> params = pit->second;

            Ptr<MobilityModel> aMobOrdered = ctx.sMob;
            Ptr<MobilityModel> bMobOrdered = ctx.uMob;
            if (ctx.sNodeId > ctx.uNodeId)
            {
                std::swap(aMobOrdered, bMobOrdered);
            }

            Ptr<const ParamsTable> table3gpp = GetThreeGppTable(aMobOrdered, bMobOrdered, ctx.condition);
            Ptr<ChannelMatrix> matrix =
                    GetNewChannel(params, table3gpp, ctx.sMob, ctx.uMob, ctx.sAnt, ctx.uAnt);

            m_channelMatrixMap.insert_or_assign(ctx.matrixKey, matrix);
        }

        NS_LOG_DEBUG("Updated GPU-backed channel params cache and refreshed channel matrices.");
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
        return ThreeGppChannelModel::GetNewChannel(
                channelParams, table3gpp, sMob, uMob, sAntenna, uAntenna);
    }

} // namespace ns3
