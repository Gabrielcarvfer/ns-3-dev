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

#include <unordered_set>

#include "ns3/spectrum-value.h"

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
#include <cstring>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#if SLS_PROFILE_INSTRUMENT && defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRA_LEAN
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
// windows.h pollutes GetObject, DeleteObject, etc. via macros — undo them
// so ns3::Object::GetObject<T>() still compiles below.
#ifdef GetObject
#undef GetObject
#endif
#ifdef DeleteObject
#undef DeleteObject
#endif
namespace {
static void
mezPrintMem(const char* label)
{
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                             sizeof(pmc)))
    {
        std::fprintf(stderr,
                     "  [MEM %s] WS=%.2fGB  Private=%.2fGB\n",
                     label,
                     pmc.WorkingSetSize / 1e9,
                     pmc.PrivateUsage / 1e9);
        std::fflush(stderr);
    }
}
} // namespace
#else
namespace { static void mezPrintMem(const char*) {} }
#endif

NS_LOG_COMPONENT_DEFINE("ThreeGppChannelModelWgpuMezanine");

namespace ns3
{

NS_OBJECT_ENSURE_REGISTERED(ThreeGppChannelModelWgpuMezanine);

// Always use (1,1,numRb) fakeChanSpct shape regardless of antenna count.
// The GPU reducedPow is beamformed power; invNumTx=1.0 + PsdReduction invN=1 gives
// Power = reducedPow correctly. Using the full (numRx,numTx,numRb) shape would
// expand each PRX copy by numRx*numTx*numRb*16B, causing GB-scale memory growth
// in calibration scenarios with many interference evaluations (e.g. DenseA 32x1 arrays).
static constexpr size_t kMaxMimoFakeEntries = 0;

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
            .AddConstructor<ThreeGppChannelModelWgpuMezanine>()
            // Phase D: periodic refresh schedule for the GPU channel
            // caches. Each UC tick now reuses ChannelMatrix /
            // ChannelParams / longTerm Ptrs in place (PRX captures
            // m_generatedTime by value so identity-stable Ptrs still
            // invalidate correctly), so the per-tick allocation cost
            // is bounded by genuinely-resized links rather than the
            // full link set.
            //
            // Default 100 ms aligns with 3GPP TR 38.901's typical
            // small-scale update granularity AND keeps per-tick UC
            // cost amortised across a slot window: at NR-cali Phase-1
            // scale a full refresh takes ~40 s wall when the link
            // set has grown to ~43k entries (every 10 ms would mean
            // 4000 s wall per second of simulated time). PRX-driven
            // EnsureBatchFresh still triggers UC on every new
            // Simulator::Now() that NR fires events at, so genuine
            // reconfigurations (antenna reshuffle, BWP change) are
            // caught immediately rather than waiting up to 100 ms.
            // The periodic loop just guarantees a refresh happens at
            // least this often.
            // Set 0 to disable the periodic loop entirely.
            .AddAttribute("MezRefreshPeriod",
                          "Period at which mezanine refreshes the GPU "
                          "channel caches independent of PRX events.",
                          TimeValue(MilliSeconds(100)),
                          MakeTimeAccessor(&ThreeGppChannelModelWgpuMezanine::m_refreshPeriod),
                          MakeTimeChecker())
            .AddAttribute("MezUcPeriod",
                          "Run UpdateChannel only once every N EnsureBatchFresh ticks. "
                          "Default 1 = every tick (no reuse). Higher values amortize "
                          "GPU UC cost at the cost of a stale channel for N-1 ticks.",
                          UintegerValue(1),
                          MakeUintegerAccessor(&ThreeGppChannelModelWgpuMezanine::m_ucPeriod),
                          MakeUintegerChecker<uint32_t>(1))
            .AddAttribute("MezBatchSlots",
                          "Number of future NR slots to pre-compute per PeriodicRefresh "
                          "tick. Default 1 = only the current slot (current behaviour). "
                          "M>1 dispatches genSpecBatch M times with Doppler phase "
                          "extrapolated M-1 slots ahead and caches M reducedPow arrays "
                          "per link so that PRX evals for the next M slots return the "
                          "pre-computed result without any GPU dispatch.",
                          UintegerValue(1),
                          MakeUintegerAccessor(&ThreeGppChannelModelWgpuMezanine::m_batchM),
                          MakeUintegerChecker<uint32_t>(1, 64))
            .AddAttribute("MezSlotDuration",
                          "Duration of one NR slot for MezBatchSlots lookahead. "
                          "Default 0.5 ms matches NR numerology mu=1 (subcarrier "
                          "spacing 30 kHz). Set to 1 ms for mu=0.",
                          TimeValue(MicroSeconds(500)),
                          MakeTimeAccessor(&ThreeGppChannelModelWgpuMezanine::m_slotDuration),
                          MakeTimeChecker());
    return tid;
}

void
ThreeGppChannelModelWgpuMezanine::PeriodicRefresh()
{
    // Drive a channel-state update on our own clock. We stamp
    // m_lastMezBatchTime before calling UpdateChannel so that any
    // EnsureBatchFresh call at the same simulator time (e.g. the
    // RunAllLinks lambda firing at the same tick as this PR event)
    // recognises the work is already done and skips its own UC call.
    // Without this stamp, both PR and EBF would run UC at the same sim
    // time, doubling GPU work.
    const Time now = Simulator::Now();
    if (!m_channelMatrixMap.empty() &&
        (!m_lastMezBatchTime.has_value() || *m_lastMezBatchTime != now))
    {
        m_lastMezBatchTime = now;
        UpdateChannel();
        m_lastUCSimTime = now;
    }
    if (!m_refreshPeriod.IsZero())
    {
        Simulator::Schedule(m_refreshPeriod,
                            &ThreeGppChannelModelWgpuMezanine::PeriodicRefresh,
                            this);
    }
    else
    {
        m_periodicRefreshScheduled = false;
    }
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
#if SLS_PROFILE_INSTRUMENT
    // Dump phase stats every N UpdateChannel calls so kill-9 paths
    // (Windows TerminateProcess) still leave a snapshot. The file is
    // rewritten each call.
    {
        static thread_local int s_uc_calls = 0;
        if ((++s_uc_calls % 5) == 0)
        {
            ::sls::detail::dumpPhaseStats();
        }
    }
    {
        static thread_local int s_uc_tick = 0;
        std::fprintf(stderr,
                     "[Mez::UpdateChannel start tick=%d] m_channelMatrixMap.size()=%zu m_channelParamsMap.size()=%zu\n",
                     ++s_uc_tick,
                     m_channelMatrixMap.size(),
                     m_channelParamsMap.size());
        std::fflush(stderr);
    }
#endif

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
    std::vector<RuntimeLinkCtx> runtimeLinks;

    antCfgs.reserve(8);
    runtimeLinks.reserve(m_channelMatrixMap.size());

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

        // Bucket-by-dominant-geometry: the GPU pipeline dispatches the
        // matrix / longTerm / spec kernels with a single (uSize, sSize)
        // per call. NR mixes antenna geometries across bands (a 32-
        // element NR panel + a 1-element LENA calibration antenna on
        // the same gNB), so we can't just abort on the first mismatch.
        // Instead: defer the decision -- record every panel's nAnt
        // here, then after the link sweep keep only the bucket whose
        // (sNAnt, uNAnt) pair has the most links. Links in other
        // buckets are dropped from runtimeLinks and PRX falls through
        // to CPU for them. A future refactor can loop the pipeline
        // per-bucket; this single-bucket short-cut is enough to
        // unblock the calibration runs.
        if (isBsSide)
        {
            if (!haveBsReference)
            {
                appendPanelConfig(p);
                bsReferencePanelIdx = p.panelIdx;
                globalBsAnt = p.nAnt;
                haveBsReference = true;
            }
            else if (p.nAnt == globalBsAnt)
            {
                p.panelIdx = bsReferencePanelIdx;
            }
            else
            {
                // Different geometry from the reference -- give it
                // its own panel slot so we still know what to drop
                // later. The dominant-bucket filter below decides
                // whether this link survives.
                appendPanelConfig(p);
            }
        }
        else
        {
            if (!haveUtReference)
            {
                NS_ABORT_MSG_IF(!haveBsReference,
                                "BS reference panel must be created before UE reference panel");
                appendPanelConfig(p);
                utReferencePanelIdx = p.panelIdx;
                globalUeAnt = p.nAnt;
                haveUtReference = true;
            }
            else if (p.nAnt == globalUeAnt)
            {
                p.panelIdx = utReferencePanelIdx;
            }
            else
            {
                appendPanelConfig(p);
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

    // Group runtimeLinks by (sNAnt, uNAnt). Every bucket runs the full
    // matrix/LT/spec pipeline independently; CRN/LSP/cluster kernels run
    // once for the whole grid and are shared across all buckets.
    if (runtimeLinks.empty())
    {
        return;
    }
    std::map<std::pair<uint32_t, uint32_t>, std::vector<size_t>> bucketMap;
    {
        static thread_local int s_bucket_log_ctr = 0;
        const bool logThis = ((++s_bucket_log_ctr % 100) == 1); // first tick + every 100th
        for (size_t i = 0; i < runtimeLinks.size(); ++i)
        {
            const auto& ctx = runtimeLinks[i];
            if (!ctx.sAnt || !ctx.uAnt)
                continue;
            const auto sN = static_cast<uint32_t>(ctx.sAnt->GetNumElems());
            const auto uN = static_cast<uint32_t>(ctx.uAnt->GetNumElems());
            bucketMap[{sN, uN}].push_back(i);
        }
        if (logThis)
        {
            std::fprintf(stderr, "[Mez bucket histogram tick=%d] %zu buckets, %zu links: ",
                         s_bucket_log_ctr, bucketMap.size(), runtimeLinks.size());
            for (const auto& [key, idxs] : bucketMap)
            {
                std::fprintf(stderr, "(s=%u,u=%u)x%zu ", key.first, key.second, idxs.size());
            }
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
        }
    }
    if (bucketMap.empty())
    {
        return;
    }

    NS_ABORT_MSG_IF(globalBsAnt == 0 || globalUeAnt == 0,
                    "Global BS/UE antenna counts must be non-zero");
    NS_ABORT_MSG_IF(cells.empty(), "No BS-side cells inferred from cached matrices");
    NS_ABORT_MSG_IF(uts.empty(), "No UT-side terminals inferred from cached matrices");

    const uint32_t nSite = static_cast<uint32_t>(cells.size());
    const uint32_t nUt = static_cast<uint32_t>(uts.size());
    const uint32_t nSnapshots = 14;
    const uint32_t nPrbg = 53;

    // Assign lspReadIdx for ALL runtimeLinks (used by cluster kernel indexing).
    for (auto& ctx : runtimeLinks)
    {
        ctx.lspReadIdx = ctx.cid * nUt + ctx.uid;
    }

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

    mezPrintMem("pre-CRN");
    {
        SLS_PHASE_SCOPE("Mez::GenerateCRN");
        gpu->generateCRN(maxXf, minXf, maxYf, minYf, corrLos, corrNlos, corrO2i);
    }
    mezPrintMem("post-CRN");
    {
        SLS_PHASE_SCOPE("Mez::CalLinkParam");
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
    }

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

    static const bool ltEnabled = []() {
        const char* e = std::getenv("MEZ_GPU_LT");
        // Default ON. Set MEZ_GPU_LT=0 to disable.
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    // Run calClusterRay once for the full grid (antenna-independent).
    {
        SLS_PHASE_SCOPE("Mez::SmallScaleKernels");
        {
            SLS_PHASE_SCOPE("Mez::CalClusterRay");
            gpu->calClusterRay(nSite, nUt);
        }
    }
    mezPrintMem("post-SSKernels");

    NS_LOG_DEBUG("UpdateChannel uploaded " << nSite << " cells, " << nUt << " UEs, "
                                           << antCfgs.size() << " antenna panels, "
                                           << runtimeLinks.size() << " total links across "
                                           << bucketMap.size() << " buckets");

    // Read antenna-independent params once (full [nSite x nUt] grid).
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
        // readAllClusterData issues ONE GPU submit+waitIdle for both
        // clusterParamsBuf_ and clusterOutputsBuf_, replacing the previous
        // 6 separate submits (readClusterParams + 5 x sliceClusterOutput).
        auto acd = gpu->readAllClusterData(nSite, nUt);
        clusterParams = std::move(acd.clusterParams);
        if (!acd.packedOutputs.empty())
        {
            using C = SlsChanWgpu;
            const uint64_t nLinks = acd.packedOutputs.size() / C::kPackedLinkStride;
            auto sliceFrom = [&](uint32_t off) {
                std::vector<float> out(nLinks * C::kMaxCr);
                for (uint64_t l = 0; l < nLinks; ++l)
                    std::memcpy(out.data() + l * C::kMaxCr,
                                acd.packedOutputs.data() + l * C::kPackedLinkStride + off,
                                C::kMaxCr * sizeof(float));
                return out;
            };
            xprFlat        = sliceFrom(C::kPackedOffXpr);
            phiNmAoaFlat   = sliceFrom(C::kPackedOffAoA);
            phiNmAodFlat   = sliceFrom(C::kPackedOffAoD);
            thetaNmZoaFlat = sliceFrom(C::kPackedOffZoA);
            thetaNmZodFlat = sliceFrom(C::kPackedOffZoD);
        }
    }
    mezPrintMem("post-ReadParams");

    // Device-level constants for chunk sizing (shared across all buckets).
    const uint64_t maxBuf = gpu->getMaxStorageBufferBindingSize();
    const uint64_t kWgPerDimCap = 65535ull;
    const uint64_t effectiveBufCap = maxBuf == 0 ? (1ull << 31) : maxBuf;
    const uint32_t sbNumRb = static_cast<uint32_t>(m_batchRbFreqs.size());
    const uint64_t rbFreqsHash = HashFloatVector(m_batchRbFreqs);

    // Static thread-local accumulators sized to the largest bucket seen
    // so far; .resize just touches already-allocated pages on repeat calls.
    static thread_local std::vector<ActiveLink> bktChunkLinks;
    static thread_local std::vector<std::complex<float>> matFlat;
    static thread_local std::vector<std::complex<float>> longTermFlat;
    static thread_local std::vector<float> chunkDelays;
    static thread_local std::vector<std::complex<float>> chunkDoppler;

    // ── Per-bucket pipeline ──────────────────────────────────────────────
    // Each unique (sNAnt, uNAnt) pair gets its own matrix/LT/spec dispatch.
    for (const auto& [antKey, bucketLinkIdxs] : bucketMap)
    {
        const uint32_t bSSize = antKey.first;   // BS elements
        const uint32_t bUSize = antKey.second;  // UE elements
        const uint32_t nBucketLinks = static_cast<uint32_t>(bucketLinkIdxs.size());

        // ── Build activeLinks for this bucket ────────────────────────────
        // bucket-local sequential index (0..nBucketLinks-1) maps to
        // matFlat/longTermFlat; lspReadIdx maps to LSP/cluster grid.
        std::vector<ActiveLink> bucketActiveLinks(nBucketLinks);
        for (uint32_t bi = 0; bi < nBucketLinks; ++bi)
        {
            const auto& ctx = runtimeLinks[bucketLinkIdxs[bi]];
            ActiveLink al{};
            al.cid = ctx.cid;
            al.uid = ctx.uid;
            al.linkIdx = bi;
            al.lspReadIdx = ctx.lspReadIdx;
            bucketActiveLinks[bi] = al;
        }

        // ── LongTerm setup for this bucket ───────────────────────────────
        // Use the first link in the bucket as reference antenna.
        uint32_t ltSPorts = 0;
        uint32_t ltUPorts = 0;
        uint32_t ltSPortElems = 0;
        uint32_t ltUPortElems = 0;
        uint32_t ltSElemsPerPort = 0;
        uint32_t ltUElemsPerPort = 0;
        uint32_t ltSIncVal = 0;
        uint32_t ltUIncVal = 0;
        std::vector<uint32_t> ltStartS;
        std::vector<uint32_t> ltStartU;
        std::vector<std::complex<float>> ltSWFlat;
        std::vector<std::complex<float>> ltUWFlat;
        bool ltCanDispatch = false;
        if (ltEnabled)
        {
            const auto& refCtx = runtimeLinks[bucketLinkIdxs[0]];
            if (refCtx.sAnt && refCtx.uAnt)
            {
                ltSPorts = static_cast<uint32_t>(refCtx.sAnt->GetNumPorts());
                ltUPorts = static_cast<uint32_t>(refCtx.uAnt->GetNumPorts());
                ltSPortElems = static_cast<uint32_t>(refCtx.sAnt->GetNumElemsPerPort());
                ltUPortElems = static_cast<uint32_t>(refCtx.uAnt->GetNumElemsPerPort());
                ltSElemsPerPort = static_cast<uint32_t>(refCtx.sAnt->GetHElemsPerPort());
                ltUElemsPerPort = static_cast<uint32_t>(refCtx.uAnt->GetHElemsPerPort());
                ltSIncVal = (ltSElemsPerPort > 0)
                                ? static_cast<uint32_t>(refCtx.sAnt->GetNumColumns()) - ltSElemsPerPort
                                : 0u;
                ltUIncVal = (ltUElemsPerPort > 0)
                                ? static_cast<uint32_t>(refCtx.uAnt->GetNumColumns()) - ltUElemsPerPort
                                : 0u;
                ltStartS.resize(ltSPorts);
                for (uint32_t p = 0; p < ltSPorts; ++p)
                {
                    ltStartS[p] = static_cast<uint32_t>(refCtx.sAnt->ArrayIndexFromPortIndex(p, 0));
                }
                ltStartU.resize(ltUPorts);
                for (uint32_t p = 0; p < ltUPorts; ++p)
                {
                    ltStartU[p] = static_cast<uint32_t>(refCtx.uAnt->ArrayIndexFromPortIndex(p, 0));
                }
                ltSWFlat.resize(size_t(nBucketLinks) * ltSPortElems);
                ltUWFlat.resize(size_t(nBucketLinks) * ltUPortElems);
                bool assembleOk = true;
                for (uint32_t bi = 0; bi < nBucketLinks; ++bi)
                {
                    const auto& ctx = runtimeLinks[bucketLinkIdxs[bi]];
                    if (!ctx.sAnt || !ctx.uAnt)
                    {
                        assembleOk = false;
                        break;
                    }
                    const auto& sBV = ctx.sAnt->GetBeamformingVectorRef();
                    const auto& uBV = ctx.uAnt->GetBeamformingVectorRef();
                    const size_t sBase = bi * ltSPortElems;
                    const size_t uBase = bi * ltUPortElems;
                    NS_ASSERT_MSG(sBV.GetSize() >= ltSPortElems,
                                  "Mez::genLongTerm: sBV[" << ctx.sAntId << "].size()="
                                  << sBV.GetSize() << " < ltSPortElems=" << ltSPortElems);
                    NS_ASSERT_MSG(uBV.GetSize() >= ltUPortElems,
                                  "Mez::genLongTerm: uBV[" << ctx.uAntId << "].size()="
                                  << uBV.GetSize() << " < ltUPortElems=" << ltUPortElems);
                    for (uint32_t k = 0; k < ltSPortElems; ++k)
                    {
                        const auto v = sBV[k];
                        ltSWFlat[sBase + k] = std::complex<float>(
                            static_cast<float>(v.real()), static_cast<float>(v.imag()));
                    }
                    for (uint32_t k = 0; k < ltUPortElems; ++k)
                    {
                        const auto v = uBV[k];
                        ltUWFlat[uBase + k] = std::complex<float>(
                            static_cast<float>(v.real()), static_cast<float>(v.imag()));
                    }
                }
                ltCanDispatch = assembleOk && ltSPortElems > 0 && ltUPortElems > 0 &&
                                ltSPorts > 0 && ltUPorts > 0;
            }
        }

        // ── Chunk sizes for this bucket ──────────────────────────────────
        const uint64_t cmBytesPerLink =
            uint64_t(bUSize) * bSSize * SlsChanWgpu::kMatMaxPages * 2u * sizeof(float);
        const uint64_t ltBytesPerLink =
            ltCanDispatch
                ? uint64_t(ltSPorts) * ltUPorts * SlsChanWgpu::kMatMaxPages * 2u * sizeof(float)
                : 0u;
        const uint64_t maxBytesPerLinkMain = std::max({cmBytesPerLink, ltBytesPerLink, uint64_t(1)});
        const uint64_t ltWgPerLink =
            ltCanDispatch
                ? (uint64_t(ltSPorts) * ltUPorts * SlsChanWgpu::kMatMaxPages + 63u) / 64u
                : 0u;
        const uint64_t maxWgPerLinkMain = std::max({uint64_t(1), ltWgPerLink});
        const uint64_t chunkByBufMain =
            std::max<uint64_t>(1ull,
                               (effectiveBufCap * 9ull) / 10ull /
                                   std::max<uint64_t>(maxBytesPerLinkMain, 1ull));
        const uint64_t chunkByWgMain =
            std::max<uint64_t>(1ull, kWgPerDimCap / std::max<uint64_t>(maxWgPerLinkMain, 1ull));
        uint32_t chunkSize = static_cast<uint32_t>(std::clamp<uint64_t>(
            std::min<uint64_t>({uint64_t(nBucketLinks), chunkByBufMain, chunkByWgMain}),
            1ull, uint64_t(nBucketLinks)));
        chunkSize = std::min<uint32_t>(chunkSize, 256u);

        // gen_spec_pow_kernel output is rb_pow_out[nLinks * nRb]: one float per (link, rb).
        const uint64_t sbBytesPerLink =
            (ltCanDispatch && sbNumRb > 0)
                ? uint64_t(sbNumRb) * sizeof(float)
                : 0u;
        // 1D dispatch: ceil(nLinks * nRb / 64) workgroups in x.
        const uint64_t sbWgPerLink =
            (ltCanDispatch && sbNumRb > 0) ? std::max<uint64_t>(1ull, (sbNumRb + 63u) / 64u) : 0ull;
        // Cap staging readback size at 768 KB per chunk to keep MapAsync fast
        // (D3D12/Dawn readback latency grows super-linearly above ~1 MB:
        // observed 2MB → 20ms vs 768KB → 1ms, 20x cost for 3x data).
        static constexpr uint64_t kMaxStagingReadbackBytes = 786432ull; // 768 KB
        const uint64_t chunkByReadback =
            std::max<uint64_t>(1ull,
                               (sbBytesPerLink > 0) ? kMaxStagingReadbackBytes / sbBytesPerLink
                                                     : uint64_t(nBucketLinks));
        const uint64_t chunkByBufSpec =
            std::max<uint64_t>(1ull,
                               (effectiveBufCap * 9ull) / 10ull /
                                   std::max<uint64_t>(sbBytesPerLink, 1ull));
        const uint64_t chunkByWgSpec =
            std::max<uint64_t>(1ull, kWgPerDimCap / std::max<uint64_t>(sbWgPerLink, 1ull));
        const uint32_t chunkSizeSpec = static_cast<uint32_t>(std::clamp<uint64_t>(
            std::min<uint64_t>({uint64_t(nBucketLinks), chunkByBufSpec, chunkByWgSpec, chunkByReadback}),
            1ull, uint64_t(nBucketLinks)));
        NS_LOG_INFO("Mez::UpdateChannel bucket (s=" << bSSize << ",u=" << bUSize
                    << ") nLinks=" << nBucketLinks
                    << " chunkMain=" << chunkSize << " chunkSpec=" << chunkSizeSpec
                    << " ltSPorts=" << ltSPorts << " ltUPorts=" << ltUPorts
                    << " sbNumRb=" << sbNumRb);
        {
            static thread_local bool s_printedChunk = false;
            if (!s_printedChunk)
            {
                s_printedChunk = true;
                std::fprintf(stderr,
                             "[Mez] bucket s=%u u=%u nLinks=%u "
                             "chunkMain=%u (byBuf=%llu byWg=%llu) "
                             "chunkSpec=%u (byBuf=%llu byWg=%llu) "
                             "ltSPorts=%u ltUPorts=%u ltWgPerLink=%llu sbNumRb=%u\n",
                             bSSize,
                             bUSize,
                             nBucketLinks,
                             chunkSize,
                             (unsigned long long)chunkByBufMain,
                             (unsigned long long)chunkByWgMain,
                             chunkSizeSpec,
                             (unsigned long long)chunkByBufSpec,
                             (unsigned long long)chunkByWgSpec,
                             ltSPorts,
                             ltUPorts,
                             (unsigned long long)ltWgPerLink,
                             sbNumRb);
            }
        }

        // ── Accumulators ─────────────────────────────────────────────────
        const size_t perLinkMatLen = size_t(bUSize) * bSSize * SlsChanWgpu::kMatMaxPages;
        const size_t perLinkLT =
            ltCanDispatch ? size_t(ltSPorts) * ltUPorts * SlsChanWgpu::kMatMaxPages : 0;
        matFlat.resize(size_t(nBucketLinks) * perLinkMatLen);
        longTermFlat.clear();
        if (ltCanDispatch)
        {
            longTermFlat.resize(size_t(nBucketLinks) * perLinkLT);
        }

        // ── Chunk loop: matrix + longTerm ────────────────────────────────
        {
            SLS_PHASE_SCOPE("Mez::MatrixKernels");
            for (uint32_t chunkStart = 0; chunkStart < nBucketLinks; chunkStart += chunkSize)
            {
                const uint32_t chunkLen = std::min(chunkSize, nBucketLinks - chunkStart);
                {
                    SLS_PHASE_SCOPE("Mez::ChunkPrep");
                    bktChunkLinks.resize(chunkLen);
                    for (uint32_t i = 0; i < chunkLen; ++i)
                    {
                        ActiveLink al = bucketActiveLinks[chunkStart + i];
                        al.linkIdx = i;
                        bktChunkLinks[i] = al;
                    }
                }
                {
                    SLS_PHASE_SCOPE("Mez::UploadActiveLinks");
                    gpu->uploadActiveLinkBuf(bktChunkLinks);
                }
                if (ltCanDispatch)
                {
                    // Opt E: fuse genChannelMatrix + readChannelMatrix +
                    // genLongTerm + readLongTerm into a single encoder
                    // submission (4 WaitIdle per chunk → 1).
                    const size_t chunkSWLen = size_t(chunkLen) * ltSPortElems;
                    const size_t chunkUWLen = size_t(chunkLen) * ltUPortElems;
                    SLS_PHASE_SCOPE("Mez::MatrixAndLTFused");
                    gpu->genChannelMatrixAndLongTermFused(
                        chunkLen,
                        /*uSize=*/bUSize,
                        /*sSize=*/bSSize,
                        /*numOverallCluster=*/SlsChanWgpu::kMatMaxPages,
                        /*nRays=*/20u,
                        ltSPorts,
                        ltUPorts,
                        ltSPortElems,
                        ltUPortElems,
                        ltSElemsPerPort,
                        ltUElemsPerPort,
                        ltSIncVal,
                        ltUIncVal,
                        ltSWFlat.data() + size_t(chunkStart) * ltSPortElems,
                        chunkSWLen,
                        ltUWFlat.data() + size_t(chunkStart) * ltUPortElems,
                        chunkUWLen,
                        ltStartS,
                        ltStartU,
                        matFlat.data() + size_t(chunkStart) * perLinkMatLen,
                        longTermFlat.data() + size_t(chunkStart) * perLinkLT);
                    if (chunkStart == 0)
                    {
                        mezPrintMem("c0-post-genMat");
                        mezPrintMem("c0-post-readMat");
                        mezPrintMem("c0-post-genLT");
                        mezPrintMem("c0-post-readLT");
                    }
                }
                else
                {
                    // No beamforming: only genChannelMatrix + readChannelMatrix.
                    {
                        SLS_PHASE_SCOPE("Mez::GenChannelMatrix");
                        gpu->genChannelMatrix(bktChunkLinks,
                                              chunkLen,
                                              /*uSize=*/bUSize,
                                              /*sSize=*/bSSize,
                                              /*numOverallCluster=*/SlsChanWgpu::kMatMaxPages,
                                              /*numReducedCluster=*/0u,
                                              /*nRays=*/20u,
                                              /*cluster1st=*/0u,
                                              /*cluster2nd=*/0u);
                    }
                    if (chunkStart == 0) mezPrintMem("c0-post-genMat");
                    {
                        SLS_PHASE_SCOPE("Mez::ReadChannelMatrix");
                        gpu->readChannelMatrixInto(
                            chunkLen,
                            bUSize,
                            bSSize,
                            matFlat.data() + size_t(chunkStart) * perLinkMatLen);
                    }
                    if (chunkStart == 0) mezPrintMem("c0-post-readMat");
                }
            }
        }
        mezPrintMem("post-MatrixLT");

        // ── Populate: write GPU results into ns-3 channel caches ────────
        {
            SLS_PHASE_SCOPE("Mez::Populate");
            for (uint32_t bi = 0; bi < nBucketLinks; ++bi)
            {
                const auto& ctx = runtimeLinks[bucketLinkIdxs[bi]];
                const LinkParams& lk = linkParams.at(ctx.lspReadIdx);
                const ClusterParamsGpu& cp = clusterParams.at(ctx.lspReadIdx);

                if (cp.nCluster == 0)
                    continue;

                NS_ABORT_MSG_IF(cp.nCluster > MAX_CLUSTERS, "GPU cluster count out of range.");
                NS_ABORT_MSG_IF(cp.nRayPerCluster > MAX_RAYS, "GPU ray count out of range.");

                Ptr<ThreeGppChannelParams> prev;
                if (auto it = m_channelParamsMap.find(ctx.paramsKey); it != m_channelParamsMap.end())
                    prev = it->second;

                Ptr<ThreeGppChannelParams>& paramsRef = m_channelParamsMap[ctx.paramsKey];
                if (!paramsRef)
                    paramsRef = Create<ThreeGppChannelParams>();

                Ptr<ThreeGppChannelParams> params = paramsRef;
                params->m_generatedTime = Simulator::Now();

                Ptr<MobilityModel> aMobOrdered = ctx.sMob;
                Ptr<MobilityModel> bMobOrdered = ctx.uMob;
                if (ctx.sNodeId > ctx.uNodeId)
                    std::swap(aMobOrdered, bMobOrdered);

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
                // lk.DS is in nanoseconds; m_DS expects seconds.
                params->m_DS = lk.DS * 1e-9;
                // lk.K is linear; m_K_factor expects dB.
                params->m_K_factor = (lk.K > 0.0f) ? 10.0 * std::log10(static_cast<double>(lk.K)) : 0.0;
                params->m_reducedClusterNumber = static_cast<uint8_t>(cp.nCluster);
                // cp.delays are in nanoseconds (lk.DS stored as log10(ns));
                // m_delay must be in seconds for PRX and gen_spec_pow_kernel.
                params->m_delay.resize(cp.nCluster);
                for (uint32_t c = 0; c < cp.nCluster; ++c)
                    params->m_delay[c] = static_cast<double>(cp.delays[c]) * 1e-9;
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
                    params->m_clusterPhase = prev->m_clusterPhase;
                else
                    params->m_clusterPhase.assign(
                        cp.nCluster, Double2DVector(cp.nRayPerCluster, DoubleVector(4u, 0.0)));

                params->m_clusterShadowing.assign(cp.nCluster, 0.0);
                params->m_attenuation_dB.assign(cp.nCluster, 0.0);
                params->m_nonSelfBlocking.clear();
                params->m_norRvAngles.clear();
                params->m_alpha.assign(cp.nCluster, 0.0);
                params->m_D.assign(cp.nCluster, 0.0);

                if (prev && prev->m_clusterXnNlosSign.size() == cp.nCluster)
                    params->m_clusterXnNlosSign = prev->m_clusterXnNlosSign;
                else
                    params->m_clusterXnNlosSign.assign(cp.nCluster, 1);

                params->m_delayConsistency = params->m_delay;
                for (auto& d : params->m_delayConsistency)
                    d += params->m_dis3D / 3e8;

                Ptr<ParamsTable> dummyTable = Create<ParamsTable>();
                dummyTable->m_cDS = 0.0;
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

                // Build ChannelMatrix from GPU matrix output.
                const uint8_t numOverallCluster =
                    params->m_cluster1st != params->m_cluster2nd
                        ? static_cast<uint8_t>(params->m_reducedClusterNumber + 4)
                        : static_cast<uint8_t>(params->m_reducedClusterNumber + 2);

                Ptr<ChannelMatrix>& matrixRef = m_channelMatrixMap[ctx.matrixKey];
                if (!matrixRef)
                {
                    matrixRef = Create<ChannelMatrix>();
                    matrixRef->m_channel = MatrixBasedChannelModel::Complex3DVector(
                        bUSize, bSSize, numOverallCluster);
                }
                else if (matrixRef->m_channel.GetNumRows() != bUSize ||
                         matrixRef->m_channel.GetNumCols() != bSSize ||
                         matrixRef->m_channel.GetNumPages() != numOverallCluster)
                {
                    matrixRef->m_channel = MatrixBasedChannelModel::Complex3DVector(
                        bUSize, bSSize, numOverallCluster);
                }
                ChannelMatrix* matrix = PeekPointer(matrixRef);
                matrix->m_generatedTime = Simulator::Now();
                matrix->m_nodeIds = std::make_pair(ctx.sMob->GetObject<Node>()->GetId(),
                                                   ctx.uMob->GetObject<Node>()->GetId());
                matrix->m_antennaPair = std::make_pair(ctx.sAntId, ctx.uAntId);
                const size_t perPage = size_t(bUSize) * bSSize;
                const size_t linkBase = bi * perLinkMatLen;
                const size_t linkCells = perPage * size_t(numOverallCluster);
                NS_ASSERT_MSG(linkBase + linkCells <= matFlat.size(),
                              "Mez::Populate: matFlat OOB: linkBase=" << linkBase
                              << " linkCells=" << linkCells
                              << " matFlat.size()=" << matFlat.size()
                              << " (lspReadIdx=" << ctx.lspReadIdx
                              << " perLinkMatLen=" << perLinkMatLen << ")");
                const std::complex<float>* __restrict__ src = matFlat.data() + linkBase;
                std::complex<double>* __restrict__ dst = matrix->m_channel.GetPagePtr(0);
                for (size_t k = 0; k < linkCells; ++k)
                {
                    const auto* sr = reinterpret_cast<const float*>(src + k);
                    auto* dr = reinterpret_cast<double*>(dst + k);
                    dr[0] = static_cast<double>(sr[0]);
                    dr[1] = static_cast<double>(sr[1]);
                }

                // Build per-link LongTerm from GPU output.
                if (!longTermFlat.empty() && ctx.sAnt && ctx.uAnt)
                {
                    const size_t ltPerPage = size_t(ltUPorts) * ltSPorts;
                    const size_t ltPerLink = ltPerPage * SlsChanWgpu::kMatMaxPages;
                    const size_t ltLinkBase = bi * ltPerLink;
                    const size_t ltCells = ltPerPage * size_t(numOverallCluster);
                    NS_ASSERT_MSG(ltLinkBase + ltCells <= longTermFlat.size(),
                                  "Mez::Populate: longTermFlat OOB: ltLinkBase=" << ltLinkBase
                                  << " ltCells=" << ltCells
                                  << " longTermFlat.size()=" << longTermFlat.size()
                                  << " (lspReadIdx=" << ctx.lspReadIdx
                                  << " ltPerLink=" << ltPerLink << ")");
                    GpuLongTermEntry& entryRef = m_gpuLongTermMap[ctx.matrixKey];
                    Ptr<Complex3DVector> longTerm;
                    if (entryRef.longTerm &&
                        entryRef.longTerm->GetNumRows() == ltUPorts &&
                        entryRef.longTerm->GetNumCols() == ltSPorts &&
                        entryRef.longTerm->GetNumPages() == numOverallCluster)
                    {
                        longTerm = ConstCast<Complex3DVector>(entryRef.longTerm);
                    }
                    else
                    {
                        longTerm = Create<Complex3DVector>(
                            static_cast<uint16_t>(ltUPorts),
                            static_cast<uint16_t>(ltSPorts),
                            numOverallCluster);
                    }
                    const std::complex<float>* ltSrc = longTermFlat.data() + ltLinkBase;
                    std::complex<double>* ltDst = longTerm->GetPagePtr(0);
                    for (size_t k = 0; k < ltCells; ++k)
                    {
                        const auto* sr = reinterpret_cast<const float*>(ltSrc + k);
                        auto* dr = reinterpret_cast<double*>(ltDst + k);
                        dr[0] = static_cast<double>(sr[0]);
                        dr[1] = static_cast<double>(sr[1]);
                    }
                    const auto& sBV = ctx.sAnt->GetBeamformingVectorRef();
                    const auto& uBV = ctx.uAnt->GetBeamformingVectorRef();
                    entryRef.longTerm = longTerm;
                    entryRef.sWHash = HashComplexVector(sBV);
                    entryRef.uWHash = HashComplexVector(uBV);
                    entryRef.sWSize = sBV.GetSize();
                    entryRef.uWSize = uBV.GetSize();
                    entryRef.generatedTime = matrix->m_generatedTime;
                    entryRef.gpuLinkIdx = static_cast<uint32_t>(ctx.lspReadIdx);
                    entryRef.ltUPorts = ltUPorts;
                    entryRef.ltSPorts = ltSPorts;
                }
            }
        }

        mezPrintMem("post-Populate");
        // ── GenSpec batch for this bucket ────────────────────────────────
        if (!m_batchRbFreqs.empty() && ltCanDispatch)
        {
            SLS_PHASE_SCOPE("Mez::GenSpecBatch");
            const uint32_t numClusters = SlsChanWgpu::kMatMaxPages;
            const uint32_t numRb = sbNumRb;
            const uint32_t numRxPorts = ltUPorts;
            const uint32_t numTxPorts = ltSPorts;

            // M>1 slot lookahead: pre-compute reducedPow for M future slots.
            // Layout: slot s, link bi -> m_reducedPowFlat[s*nBucketLinks*numRb + bi*numRb].
            const uint32_t batchM = m_batchM;
            const double slotDurationSec = m_slotDuration.GetSeconds();
            const double slotTime = Simulator::Now().GetSeconds();

            m_reducedPowNumRb = numRb;
            // Grow the flat buffer to hold M slots; never shrink (CPU-miss
            // entries occupy indices beyond nBucketLinks*numRb and must survive).
            const size_t gpuBatchSize = size_t(batchM) * nBucketLinks * numRb;
            if (m_reducedPowFlat.size() < gpuBatchSize)
                m_reducedPowFlat.resize(gpuBatchSize);

            for (uint32_t chunkStart = 0; chunkStart < nBucketLinks; chunkStart += chunkSizeSpec)
            {
                const uint32_t chunkLen = std::min(chunkSizeSpec, nBucketLinks - chunkStart);

                {
                    SLS_PHASE_SCOPE("Mez::SB::UploadLT");
                    // Long-term vectors are the same for all M slots; upload once.
                    gpu->uploadLongTermBatch(longTermFlat.data() +
                                                 size_t(chunkStart) * perLinkLT,
                                             chunkLen,
                                             ltSPorts,
                                             ltUPorts);
                }

                // ── Per-link delays and spatial Doppler projections ───────
                // Delays (cluster-specific, time-independent) are computed
                // once per chunk.  The spatial Doppler projection
                //   proj[link][c] = dot(angles, speeds) + 2*alpha*D
                // is also time-independent; the actual Doppler factor at
                // slot s is exp(j * (slotTime + s*dt) * 2π*fc/c * proj[c]).
                {
                    SLS_PHASE_SCOPE("Mez::SB::DelayProj");
                    chunkDelays.assign(size_t(chunkLen) * numClusters, 0.0f);
                    // Spatial projection reused across M slots.
                    // proj[li * numClusters + c] stores the unitless projection
                    // so that tempDoppler_s[c] = dopplerScale_s * proj[c].
                    std::vector<double> chunkSpatialProj(size_t(chunkLen) * numClusters, 0.0);
                    for (uint32_t li = 0; li < chunkLen; ++li)
                    {
                        const size_t i = size_t(chunkStart) + li;
                        const auto& ctx = runtimeLinks[bucketLinkIdxs[i]];
                        auto pit = m_channelParamsMap.find(ctx.paramsKey);
                        if (pit == m_channelParamsMap.end())
                            continue;
                        const auto p = DynamicCast<const ThreeGppChannelParams>(pit->second);
                        if (!p)
                            continue;
                        const size_t nc = std::min(size_t(numClusters), p->m_delay.size());
                        const size_t baseIdx = size_t(li) * numClusters;
                        for (size_t c = 0; c < nc; ++c)
                            chunkDelays[baseIdx + c] = static_cast<float>(p->m_delay[c]);
                        if (p->m_cachedAngleSincos.size() <= ZOD_INDEX)
                            continue;
                        const Vector sSpeed =
                            ctx.sMob ? ctx.sMob->GetVelocity() : Vector(0, 0, 0);
                        const Vector uSpeed =
                            ctx.uMob ? ctx.uMob->GetVelocity() : Vector(0, 0, 0);
                        const size_t ncDop = std::min(
                            {nc, p->m_alpha.size(), p->m_D.size(),
                             p->m_cachedAngleSincos[ZOA_INDEX].size(),
                             p->m_cachedAngleSincos[ZOD_INDEX].size(),
                             p->m_cachedAngleSincos[AOA_INDEX].size(),
                             p->m_cachedAngleSincos[AOD_INDEX].size()});
                        const auto& zoa = p->m_cachedAngleSincos[ZOA_INDEX];
                        const auto& zod = p->m_cachedAngleSincos[ZOD_INDEX];
                        const auto& aoa = p->m_cachedAngleSincos[AOA_INDEX];
                        const auto& aod = p->m_cachedAngleSincos[AOD_INDEX];
                        for (size_t c = 0; c < ncDop; ++c)
                        {
                            chunkSpatialProj[baseIdx + c] =
                                zoa[c].first * aoa[c].second * uSpeed.x +
                                zoa[c].first * aoa[c].first  * uSpeed.y +
                                zoa[c].second                * uSpeed.z +
                                zod[c].first * aod[c].second * sSpeed.x +
                                zod[c].first * aod[c].first  * sSpeed.y +
                                zod[c].second                * sSpeed.z +
                                2.0 * p->m_alpha[c] * p->m_D[c];
                        }
                    }

                    // ── Per-slot Doppler → dispatch → readback ────────────
                    for (uint32_t s = 0; s < batchM; ++s)
                    {
                        const double dopplerScale_s =
                            2.0 * M_PI * (slotTime + s * slotDurationSec) * m_frequency / 3e8;

                        {
                            SLS_PHASE_SCOPE("Mez::SB::DopplerPrep");
                            chunkDoppler.assign(size_t(chunkLen) * numClusters,
                                                std::complex<float>(1.0f, 0.0f));
                            for (uint32_t li = 0; li < chunkLen; ++li)
                            {
                                const size_t baseIdx = size_t(li) * numClusters;
                                for (size_t c = 0; c < numClusters; ++c)
                                {
                                    const double proj = chunkSpatialProj[baseIdx + c];
                                    if (proj == 0.0)
                                        continue; // unset entry (nc < numClusters)
                                    const double phase = dopplerScale_s * proj;
                                    chunkDoppler[baseIdx + c] = std::complex<float>(
                                        static_cast<float>(std::cos(phase)),
                                        static_cast<float>(std::sin(phase)));
                                }
                            }
                        }

                        {
                            SLS_PHASE_SCOPE("Mez::SB::Dispatch");
                            gpu->genSpecBatch(chunkLen,
                                              numClusters,
                                              numRb,
                                              numRxPorts,
                                              numTxPorts,
                                              ltUPorts,
                                              ltSPorts,
                                              chunkDelays,
                                              chunkDoppler,
                                              m_batchRbFreqs);
                        }
                        {
                            SLS_PHASE_SCOPE("Mez::SB::Readback");
                            gpu->readReducedPowInto(
                                chunkLen,
                                numRb,
                                m_reducedPowFlat.data() +
                                    size_t(s) * nBucketLinks * numRb +
                                    size_t(chunkStart) * numRb);
                        }
                    } // end slot loop
                } // end DelayProj + slot loops scope
            } // end chunk loop

            // Record each link's metadata in the flat buffer index.
            const Time generatedTime = Simulator::Now();
            const uint32_t perSlotStride = nBucketLinks * numRb;
            for (uint32_t bi = 0; bi < nBucketLinks; ++bi)
            {
                const auto& ctx = runtimeLinks[bucketLinkIdxs[bi]];
                GpuChanSpctEntry& eRef = m_gpuChanSpctMap[ctx.matrixKey];
                eRef.reducedPowBaseIdx = bi * numRb; // slot-0 base
                eRef.numRxPorts = numRxPorts;
                eRef.numTxPorts = numTxPorts;
                eRef.numRb = numRb;
                eRef.generatedTime = generatedTime;
                eRef.rbFreqsHash = rbFreqsHash;
                eRef.batchM = batchM;
                eRef.batchStartTimeSec = slotTime;
                eRef.slotDurationSec = slotDurationSec;
                eRef.perSlotStride = perSlotStride;
            }
        }
        mezPrintMem("post-GenSpec");
    } // end per-bucket loop

    mezPrintMem("post-UpdateChannel");
    NS_LOG_DEBUG("Updated GPU-backed channel params + matrix caches.");
}

void
ThreeGppChannelModelWgpuMezanine::CaptureRbFreqs(Ptr<const SpectrumValue> psd)
{
    if (!psd || !m_batchRbFreqs.empty())
        return;
    const size_t numRb =
        static_cast<size_t>(std::distance(psd->ConstBandsBegin(), psd->ConstBandsEnd()));
    std::vector<float> freqs;
    freqs.reserve(numRb);
    for (auto it = psd->ConstBandsBegin(); it != psd->ConstBandsEnd(); ++it)
        freqs.push_back(static_cast<float>(it->fc));
    if (freqs.size() == numRb)
    {
        m_batchRbFreqs = std::move(freqs);
        m_batchRbFreqsHash = HashFloatVector(m_batchRbFreqs);
    }
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
        // UpdateChannel rate gate — memory safety.
        //
        // EnsureBatchFresh can be called thousands of times per simtime-second
        // (once per unique NS-3 slot event in NR). Without a gate, each call
        // allocates D3D12/WARP staging buffers that accumulate to ~50 GB/s.
        //
        // The gate: fire UpdateChannel at most once per effectivePeriod of
        // simulated time. When m_updatePeriod=0 (calibration default: "always
        // fresh") we fall back to 100 ms — matching the typical 3GPP coherence
        // time used by PeriodicRefresh. When m_updatePeriod>0 we respect it
        // directly.
        //
        // Note: PeriodicRefresh (Phase D) is the PRIMARY driver of updates once
        // it arms itself. This path is a secondary guard that handles the window
        // between simulation start and PeriodicRefresh arming, and any scenario
        // where PeriodicRefresh is disabled (m_refreshPeriod=0).
        //
        // m_lastUCSimTime sentinel -1 s means "never fired"; first call always
        // passes regardless of effectivePeriod.
        // Once PeriodicRefresh is armed it owns all UpdateChannel calls.
        // EnsureBatchFresh only fires UpdateChannel during the startup window
        // before the periodic loop is running (m_periodicRefreshScheduled=false)
        // and when the periodic loop is disabled (m_refreshPeriod=0).
        //
        // The rate gate prevents memory explosion in the startup window: without
        // it, EnsureBatchFresh fires on every unique NS-3 tick (~6000/s), each
        // call creating D3D12/WARP staging buffers that accumulate ~50 GB/s.
        //
        // effectivePeriod: use m_updatePeriod when >0, else 100 ms (matches the
        // default m_refreshPeriod so the handoff to PeriodicRefresh is seamless).
        const Time effectivePeriod =
            m_updatePeriod.IsZero() ? MilliSeconds(100) : m_updatePeriod;
        const bool timeToUpdate =
            (m_lastUCSimTime < Seconds(0.0)) ||
            ((now - m_lastUCSimTime) >= effectivePeriod);
        // Opt H compatibility: m_ucPeriod still gates within the time window.
        const bool periodicOk = (m_ucPeriod <= 1) || ((m_ucTickCount % m_ucPeriod) == 0);
        if (timeToUpdate && periodicOk && !m_periodicRefreshScheduled)
        {
            UpdateChannel();
            m_lastUCSimTime = now;
        }
        ++m_ucTickCount;
    }
    // Phase D: kick off the periodic refresh loop the first time
    // EnsureBatchFresh fires. At sim start the channel maps are empty
    // so we have nothing to refresh; once the first PRX call populates
    // them, mez owns the channel-update cadence from then on.
    if (!m_periodicRefreshScheduled && !m_refreshPeriod.IsZero() &&
        !m_channelMatrixMap.empty())
    {
        m_periodicRefreshScheduled = true;
        Simulator::Schedule(m_refreshPeriod,
                            &ThreeGppChannelModelWgpuMezanine::PeriodicRefresh,
                            this);
    }
}

Ptr<const MatrixBasedChannelModel::ChannelMatrix>
ThreeGppChannelModelWgpuMezanine::GetChannel(Ptr<const MobilityModel> aMob,
                                             Ptr<const MobilityModel> bMob,
                                             Ptr<const PhasedArrayModel> aAntenna,
                                             Ptr<const PhasedArrayModel> bAntenna)
{
    NS_LOG_FUNCTION(this);
    // Short-circuit when the mezanine has already populated a matching
    // (params, matrix) pair for this link in the current tick. This
    // bypasses the base class's GenerateChannelParameters call --
    // critical with UpdatePeriod=0 (NR calibration default), where
    // the base would otherwise regenerate channel params on every
    // eval and break the cached matrix's cluster-count alignment.
    const uint64_t paramsKey =
        MatrixBasedChannelModel::GetKey(aMob->GetObject<Node>()->GetId(),
                                        bMob->GetObject<Node>()->GetId());
    const uint64_t matrixKey =
        MatrixBasedChannelModel::GetKey(aAntenna->GetId(), bAntenna->GetId());
    m_linkEndpoints[paramsKey] = {aMob, bMob, aAntenna, bAntenna};

    auto matIt = m_channelMatrixMap.find(matrixKey);
    auto parIt = m_channelParamsMap.find(paramsKey);
    static thread_local uint64_t miss_no_mat_entry = 0;
    static thread_local uint64_t miss_no_par_entry = 0;
    static thread_local uint64_t miss_unaligned = 0;
    static thread_local uint64_t miss_time = 0;
    static thread_local uint64_t miss_antenna = 0;
    static thread_local uint64_t hit_total = 0;
    {
        static thread_local uint64_t mapSizeProbe = 0;
        if ((++mapSizeProbe & 0xFFFF) == 0)
        {
            std::fprintf(stderr,
                         "[Mez::GetChannel probe] m_channelMatrixMap.size()=%zu hits=%llu mat_miss=%llu\n",
                         m_channelMatrixMap.size(),
                         (unsigned long long)hit_total,
                         (unsigned long long)miss_no_mat_entry);
            std::fflush(stderr);
        }
    }
    if (matIt != m_channelMatrixMap.end() && parIt != m_channelParamsMap.end())
    {
        const Ptr<const ChannelMatrix> cachedMatrix = matIt->second;
        const Ptr<const ChannelParams> cachedParams = parIt->second;
        const size_t pages = cachedMatrix->m_channel.GetNumPages();
        const size_t alphaSize = cachedParams->m_alpha.size();
        const size_t dSize = cachedParams->m_D.size();
        const auto& cas = cachedParams->m_cachedAngleSincos;
        const bool casOk = cas.size() > MatrixBasedChannelModel::ZOD_INDEX &&
                           cas[MatrixBasedChannelModel::ZOA_INDEX].size() == pages &&
                           cas[MatrixBasedChannelModel::ZOD_INDEX].size() == pages &&
                           cas[MatrixBasedChannelModel::AOA_INDEX].size() == pages &&
                           cas[MatrixBasedChannelModel::AOD_INDEX].size() == pages;
        const bool aligned = pages > 0 && pages == alphaSize && pages == dSize && casOk;
        const bool sameTime = cachedMatrix->m_generatedTime == cachedParams->m_generatedTime;
        const bool antennaOk =
            !AntennaSetupChanged(aAntenna, bAntenna, cachedMatrix);
        if (aligned && sameTime && antennaOk)
        {
            ++hit_total;
            // Print at first hit and every 200 thereafter to catch steady-state quickly.
            if (hit_total == 1 || (hit_total % 200) == 0)
            {
                const uint64_t total_calls = hit_total + miss_no_mat_entry + miss_no_par_entry +
                                             miss_unaligned + miss_time + miss_antenna;
                std::fprintf(stderr,
                             "[Mez::GetChannel] hit=%llu miss(mat=%llu par=%llu unaligned=%llu"
                             " time=%llu ant=%llu) hit_rate=%.1f%%\n",
                             (unsigned long long)hit_total,
                             (unsigned long long)miss_no_mat_entry,
                             (unsigned long long)miss_no_par_entry,
                             (unsigned long long)miss_unaligned,
                             (unsigned long long)miss_time,
                             (unsigned long long)miss_antenna,
                             100.0 * hit_total / std::max<uint64_t>(total_calls, 1));
                std::fflush(stderr);
            }
            return cachedMatrix;
        }
        if (!aligned) ++miss_unaligned;
        else if (!sameTime) ++miss_time;
        else if (!antennaOk) ++miss_antenna;
    }
    else
    {
        if (matIt == m_channelMatrixMap.end())
        {
            ++miss_no_mat_entry;
            static thread_local std::unordered_set<uint64_t> seenMissKeys;
            static thread_local int printed = 0;
            const bool isNew = seenMissKeys.insert(matrixKey).second;
            if (isNew && printed < 20)
            {
                const uint32_t aId = static_cast<uint32_t>(aAntenna->GetId());
                const uint32_t bId = static_cast<uint32_t>(bAntenna->GetId());
                const uint32_t aNode = aMob->GetObject<Node>()->GetId();
                const uint32_t bNode = bMob->GetObject<Node>()->GetId();
                std::fprintf(stderr,
                             "[Mez::GetChannel mat_miss #%d] matrixKey=0x%llx aAntId=%u bAntId=%u aNode=%u bNode=%u (unique miss keys so far: %zu)\n",
                             ++printed,
                             (unsigned long long)matrixKey,
                             aId, bId, aNode, bNode,
                             seenMissKeys.size());
                std::fflush(stderr);
            }
            else if ((seenMissKeys.size() & 0xFF) == 0 && isNew)
            {
                std::fprintf(stderr,
                             "[Mez::GetChannel mat_miss] unique miss keys so far: %zu (total misses: %llu)\n",
                             seenMissKeys.size(),
                             (unsigned long long)miss_no_mat_entry);
                std::fflush(stderr);
            }
        }
        else ++miss_no_par_entry;
    }
    // Print on every 200th miss so short runs still get stats.
    {
        const uint64_t total_miss = miss_no_mat_entry + miss_no_par_entry +
                                    miss_unaligned + miss_time + miss_antenna;
        if (total_miss == 1 || (total_miss % 200) == 0)
        {
            const uint64_t total_calls = hit_total + total_miss;
            std::fprintf(stderr,
                         "[Mez::GetChannel miss] hit=%llu miss(mat=%llu par=%llu"
                         " unaligned=%llu time=%llu ant=%llu) miss_rate=%.1f%%\n",
                         (unsigned long long)hit_total,
                         (unsigned long long)miss_no_mat_entry,
                         (unsigned long long)miss_no_par_entry,
                         (unsigned long long)miss_unaligned,
                         (unsigned long long)miss_time,
                         (unsigned long long)miss_antenna,
                         100.0 * total_miss / std::max<uint64_t>(total_calls, 1));
            std::fflush(stderr);
        }
    }

    // Cache miss / shape mismatch / antenna change -- fall through to
    // the base class which will GenerateChannelParameters and
    // GetNewChannel as needed. Our GetNewChannel override below also
    // short-circuits to the cached matrix when the params it's handed
    // happen to be the ones we wrote.
    return ThreeGppChannelModel::GetChannel(aMob, bMob, aAntenna, bAntenna);
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
    static thread_local uint64_t gnc_gpu_hit      = 0;
    static thread_local uint64_t gnc_stale_erase  = 0;
    static thread_local uint64_t gnc_cpu_fallback = 0;

    if (auto it = m_channelMatrixMap.find(matrixKey); it != m_channelMatrixMap.end())
    {
        const Time matrixT = it->second->m_generatedTime;
        const Time paramsT = channelParams ? channelParams->m_generatedTime : Time(0);
        const size_t cachedPages = it->second->m_channel.GetNumPages();
        const size_t paramsAlpha = channelParams ? channelParams->m_alpha.size() : 0;
        const size_t cachedRows = it->second->m_channel.GetNumRows();
        const size_t cachedCols = it->second->m_channel.GetNumCols();
        // Verify the cached matrix dimensions match this link's antenna
        // geometry. The GPU pipeline runs per-bucket so every (sElems,
        // uElems) pair gets its own dispatch; a stale cache entry from a
        // previous tick with a different shape must be rejected before
        // PRX's CalcLongTerm OOBs on row/col indices that overflow the
        // matrix. Matrix is stored (uRows, sCols, clusters).
        const size_t sElems = sAntenna->GetNumElems();
        const size_t uElems = uAntenna->GetNumElems();
        const bool dimsMatchNormal = (uElems == cachedRows && sElems == cachedCols);
        const bool dimsMatchSwapped = (uElems == cachedCols && sElems == cachedRows);
        const bool dimsOk = dimsMatchNormal || dimsMatchSwapped;
        // Strict cache validity: matrix must be at least as fresh as
        // params AND the cached pages must match the current params'
        // cluster count (otherwise CalcBeamformingGain asserts
        // numCluster <= m_alpha.size()). When CPU
        // GenerateChannelParameters runs alongside the GPU pipeline
        // it can shrink alpha/D/angle without touching the matrix
        // cache; falling through here lets the base class regenerate
        // an aligned pair.
        if (matrixT >= paramsT && cachedPages == paramsAlpha && cachedPages > 0 && dimsOk)
        {
            ++gnc_gpu_hit;
            const uint64_t total = gnc_gpu_hit + gnc_cpu_fallback;
            // Print on first GPU hit (shows cold-start is over) and every 2000 thereafter.
            if (gnc_gpu_hit == 1 || (gnc_gpu_hit % 2000) == 0)
            {
                std::fprintf(stderr,
                             "[Mez::GetNewChannel] gpu_hit=%llu cpu_fallback=%llu stale_erase=%llu"
                             " hit_rate=%.1f%%\n",
                             (unsigned long long)gnc_gpu_hit,
                             (unsigned long long)gnc_cpu_fallback,
                             (unsigned long long)gnc_stale_erase,
                             100.0 * gnc_gpu_hit / total);
                std::fflush(stderr);
            }
            return DynamicCast<ChannelMatrix>(
                ConstCast<MatrixBasedChannelModel::ChannelMatrix>(it->second));
        }
        // Stale or shape-mismatched -- erase by key so the base class
        // doesn't re-encounter a misleading hit on its own
        // m_channelMatrixMap lookup later in this tick. The base
        // class declares m_channelMatrixMap non-mutable but the
        // virtual is const-qualified; const_cast is the only way
        // to invalidate without an intrusive base-class change.
        ++gnc_stale_erase;
        const_cast<ThreeGppChannelModelWgpuMezanine*>(this)
            ->m_channelMatrixMap.erase(matrixKey);
        m_gpuLongTermMap.erase(matrixKey);
        m_gpuChanSpctMap.erase(matrixKey);
    }

    // Fallback: CPU build (first tick before UpdateChannel has run,
    // or any link the GPU pipeline didn't see).
    ++gnc_cpu_fallback;
    {
        const uint64_t total = gnc_gpu_hit + gnc_cpu_fallback;
        // Print on first CPU fallback and every 200 thereafter.
        if (gnc_cpu_fallback == 1 || (gnc_cpu_fallback % 200) == 0)
        {
            const uint32_t sId = sMob->GetObject<Node>() ? sMob->GetObject<Node>()->GetId() : 0;
            const uint32_t uId = uMob->GetObject<Node>() ? uMob->GetObject<Node>()->GetId() : 0;
            std::fprintf(stderr,
                         "[Mez::GetNewChannel CPU fallback #%llu] sNode=%u uNode=%u"
                         " sAnt=%zu uAnt=%zu  (total: gpu=%llu cpu=%llu stale=%llu"
                         " miss_rate=%.1f%%)\n",
                         (unsigned long long)gnc_cpu_fallback,
                         sId, uId,
                         sAntenna->GetNumElems(), uAntenna->GetNumElems(),
                         (unsigned long long)gnc_gpu_hit,
                         (unsigned long long)gnc_cpu_fallback,
                         (unsigned long long)gnc_stale_erase,
                         100.0 * gnc_cpu_fallback / std::max<uint64_t>(total, 1));
            std::fflush(stderr);
        }
    }
    return ThreeGppChannelModel::GetNewChannel(channelParams,
                                               table3gpp,
                                               sMob,
                                               uMob,
                                               sAntenna,
                                               uAntenna);
}

Ptr<const MatrixBasedChannelModel::Complex3DVector>
ThreeGppChannelModelWgpuMezanine::GetCachedLongTerm(
    Ptr<const ChannelMatrix> channelMatrix,
    Ptr<const PhasedArrayModel> sAnt,
    Ptr<const PhasedArrayModel> uAnt,
    const PhasedArrayModel::ComplexVector& sW,
    const PhasedArrayModel::ComplexVector& uW) const
{
    const uint64_t key = MatrixBasedChannelModel::GetKey(sAnt->GetId(), uAnt->GetId());
    auto it = m_gpuLongTermMap.find(key);
    if (it == m_gpuLongTermMap.end())
    {
        return nullptr;
    }
    const GpuLongTermEntry& entry = it->second;
    // Validity: the cached longTerm was computed from THIS
    // channelMatrix (matching generatedTime) AND from beam weights
    // whose 64-bit hash + size matches the caller's now. A
    // collision on two distinct beam vectors yielding a false hit
    // is ~1 in 1.8e19 -- fine for any realistic sim.
    if (!channelMatrix || entry.generatedTime != channelMatrix->m_generatedTime)
    {
        return nullptr;
    }
    // PRX::CalcBeamformingGain iterates the doppler loop up to
    // channelMatrix->m_channel.GetNumPages() and asserts that the
    // longTerm exposes at least that many pages. When the base class
    // falls through and CPU-rebuilds the matrix with a different page
    // count from the one our mezanine cached (cluster-count drift), the
    // mismatch crashes the assert. Reject the cached longTerm in that
    // case so PRX falls back to CalcLongTerm.
    if (!entry.longTerm ||
        entry.longTerm->GetNumPages() < channelMatrix->m_channel.GetNumPages())
    {
        return nullptr;
    }
    if (entry.sWSize != sW.GetSize() || entry.uWSize != uW.GetSize())
    {
        return nullptr;
    }
    // Validate port-count dimensions against the CURRENT antenna model.
    // The cached longTerm has shape (entry.ltUPorts, entry.ltSPorts,
    // pages). If the antenna's port count has changed since the cache
    // was filled -- e.g. a link that was dominant in the previous tick
    // (cached with 1x4 ports) is now queried with different antennas
    // having (4,4) ports -- the cached longTerm shape no longer
    // matches what PRX::GenSpectrumChannelMatrix expects, and its
    // c * (numRxPorts * numTxPorts) flat-index walk OOBs the
    // underlying valarray.
    const auto curSPorts = static_cast<uint16_t>(sAnt->GetNumPorts());
    const auto curUPorts = static_cast<uint16_t>(uAnt->GetNumPorts());
    if (entry.ltSPorts != curSPorts || entry.ltUPorts != curUPorts)
    {
        return nullptr;
    }
    if (entry.sWHash != HashComplexVector(sW) || entry.uWHash != HashComplexVector(uW))
    {
        return nullptr;
    }
    return entry.longTerm;
}

Ptr<MatrixBasedChannelModel::Complex3DVector>
ThreeGppChannelModelWgpuMezanine::TryGenSpectrumChannelMatrix(
    Ptr<const ChannelMatrix> channelMatrix,
    Ptr<const ChannelParams> channelParams,
    Ptr<const Complex3DVector> longTerm,
    Ptr<const SpectrumValue> inPsd,
    const std::vector<std::complex<double>>& delayT,
    const std::vector<double>& sqrtVit,
    uint32_t numRb,
    uint8_t numRxPorts,
    uint8_t numTxPorts,
    bool isReverse) const
{
    // Batched per-tick path: if UpdateChannel has already run
    // gen_spec_batch_kernel for this link with the current rb_freqs,
    // the chanSpct_unscaled is cached in m_gpuChanSpctMap and the
    // per-eval cost collapses to a hash lookup + per-PRB
    // sqrt(PSD[rb]) scale. Otherwise we either capture rb_freqs from
    // inPsd for the NEXT tick to pre-compute, or fall back to CPU /
    // the per-eval GPU dispatch below.
    if (!isReverse && channelMatrix)
    {
        const uint64_t key = MatrixBasedChannelModel::GetKey(
            channelMatrix->m_antennaPair.first, channelMatrix->m_antennaPair.second);
        auto it = m_gpuChanSpctMap.find(key);
        if (it != m_gpuChanSpctMap.end())
        {
            GpuChanSpctEntry& e = it->second;
            // m_batchRbFreqsHash is captured atomically with
            // m_batchRbFreqs (see below) and the array is write-once,
            // so comparing the stored hash avoids re-running FNV over
            // 273 floats on every PRX eval.
            if (e.generatedTime == channelMatrix->m_generatedTime &&
                e.numRxPorts == numRxPorts && e.numTxPorts == numTxPorts &&
                e.numRb == numRb && e.rbFreqsHash == m_batchRbFreqsHash)
            {
                SLS_PHASE_SCOPE("Mez::TryGenSpecHit");
                // GPU reduction kernel computed:
                //   reducedPow[link,rb] = sum_rx |sum_tx H_unscaled[rx,tx,rb]|^2
                //
                // For small MIMO (numRx*numTx <= kMaxMimoFakeEntries) return a
                // (numRx,numTx,numRb) matrix with amplitude only at H[0,0,rb] so
                // that all signals in NrInterference have consistent dimensions.
                // For large MIMO (e.g. 32x32 bench arrays) stay with (1,1,numRb)
                // since PRX::PsdReduction handles any shape and NrInterference is
                // not used in the bench path.
                //
                // PsdReduction result (isotropic path, precodingMatrix=null):
                //   (numRx,numTx) case: invN=1/numTx, |H[0,0]|^2=reducedPow*inPsd
                //     -> psd[rb] = (1/numTx) * reducedPow[rb] * inPsd[rb]
                //   (1,1) case: invN=1, |H[0,0]|^2=reducedPow*inPsd/numTx
                //     -> psd[rb] = (1/numTx) * reducedPow[rb] * inPsd[rb]
                const bool useFullDims =
                    (size_t(numRxPorts) * numTxPorts <= kMaxMimoFakeEntries);
                const uint8_t fakeRx = useFullDims ? numRxPorts : uint8_t(1);
                const uint8_t fakeTx = useFullDims ? numTxPorts : uint8_t(1);
                if (!e.fakeChanSpct || e.fakeChanSpct->GetNumPages() != numRb ||
                    static_cast<uint8_t>(e.fakeChanSpct->GetNumRows()) != fakeRx ||
                    static_cast<uint8_t>(e.fakeChanSpct->GetNumCols()) != fakeTx)
                    e.fakeChanSpct = Create<Complex3DVector>(fakeRx, fakeTx,
                                                             static_cast<uint16_t>(numRb));
                Ptr<Complex3DVector> chanSpct = e.fakeChanSpct;

                // M>1 slot lookup: find the pre-computed slot whose time
                // best matches Simulator::Now(). Slot 0 is batchStartTimeSec;
                // each subsequent slot covers one slotDurationSec window.
                uint32_t slotIdx = 0;
                if (e.batchM > 1 && e.slotDurationSec > 0.0 &&
                    e.batchStartTimeSec >= 0.0)
                {
                    const double elapsed =
                        Simulator::Now().GetSeconds() - e.batchStartTimeSec;
                    if (elapsed >= 0.0)
                    {
                        const auto raw =
                            static_cast<uint32_t>(elapsed / e.slotDurationSec);
                        slotIdx = raw < e.batchM ? raw : e.batchM - 1u;
                    }
                }
                const float* reducedPow =
                    m_reducedPowFlat.data() + e.reducedPowBaseIdx +
                    size_t(slotIdx) * e.perSlotStride;
                // reducedPow = |sum_tx H_longTerm[rx,tx,rb]|^2 = beamformed power (BF baked in).
                // Target: Power = reducedPow * inPsd (no extra 1/numTx needed).
                //   (numRx,numTx) shape: PsdReduction invN=1/numTx, so |H[0,0]|^2 = numTx*reducedPow*inPsd
                //   (1,1) shape:         PsdReduction invN=1,       so |H[0,0]|^2 = reducedPow*inPsd
                const double invNumTx = useFullDims ? static_cast<double>(numTxPorts) : 1.0;
                for (size_t rb = 0; rb < numRb; ++rb)
                {
                    const double sv = sqrtVit[rb];
                    const double psd =
                        static_cast<double>(reducedPow[rb]) * sv * sv * invNumTx;
                    auto* pageD = reinterpret_cast<double*>(chanSpct->GetPagePtr(rb));
                    pageD[0] = psd > 0.0 ? std::sqrt(psd) : 0.0;
                    pageD[1] = 0.0;
                }
                return chanSpct;
            }
        }
    }

    // First-ever eval (or rb_freqs invalidated): capture rb_freqs from
    // inPsd's subband layout. Next tick's UpdateChannel will see the
    // captured array and dispatch gen_spec_batch_kernel; this eval
    // still falls through to CPU GenSpec.
    if (m_batchRbFreqs.empty() && inPsd)
    {
        const auto bandsEnd = inPsd->ConstBandsEnd();
        std::vector<float> freqs;
        freqs.reserve(numRb);
        for (auto it = inPsd->ConstBandsBegin(); it != bandsEnd && freqs.size() < numRb; ++it)
        {
            freqs.push_back(static_cast<float>(it->fc));
        }
        if (freqs.size() == numRb)
        {
            m_batchRbFreqs = std::move(freqs);
            m_batchRbFreqsHash = HashFloatVector(m_batchRbFreqs);
        }
    }

    // Phase D-3b: on-demand miss caching. When the batched GenSpec
    // path didn't pre-compute this link (link wasn't dominant in the
    // last UpdateChannel tick, OR the very first eval just captured
    // rb_freqs and no UpdateChannel has run yet, OR shape drift
    // invalidated the cached entry), compute chanSpct_unscaled on CPU
    // once and stash it. Subsequent PRX evals on the same
    // (matrixKey, generatedTime, rb_freqs, shape) hit the cache above
    // and skip the contraction entirely.
    //
    // Without this, ~67% of PRX evals at NR-cali densities fall
    // through to PRX::GenSpec's CPU contraction (~540 us each in
    // Debug) -- the first call pays that, every later call is the
    // existing O(rxtx * numRb) scaling-only path.
    //
    // Only the !isReverse case is cached. The cache hit branch above
    // is gated on !isReverse and uses (numRxPorts, numTxPorts, numRb)
    // shape directly; PRX's isReverse path materialises a transposed
    // longTerm before the contraction, so caching with our shape would
    // be wrong for the reverse orientation. Reverse evals just fall
    // through to CPU below.
    if (!isReverse && channelMatrix && channelParams && longTerm &&
        !m_batchRbFreqs.empty() &&
        longTerm->GetNumPages() == channelMatrix->m_channel.GetNumPages() &&
        longTerm->GetNumRows() == numRxPorts &&
        longTerm->GetNumCols() == numTxPorts &&
        delayT.size() == size_t(longTerm->GetNumPages()) * numRb &&
        sqrtVit.size() == numRb)
    {
        SLS_PHASE_SCOPE("Mez::TryGenSpecCpuMiss");
        const size_t numCluster = channelMatrix->m_channel.GetNumPages();
        const size_t rxtx = size_t(numRxPorts) * numTxPorts;
        const uint64_t key = MatrixBasedChannelModel::GetKey(
            channelMatrix->m_antennaPair.first,
            channelMatrix->m_antennaPair.second);

        // Assign a flat-buffer slot to this entry on first sight.
        // On subsequent CPU misses (shape change, stale) the slot is reused.
        GpuChanSpctEntry& eRef = m_gpuChanSpctMap[key];
        const bool isNewEntry = (eRef.numRb == 0 && eRef.numRxPorts == 0);
        if (isNewEntry)
        {
            const uint32_t linkSlot = static_cast<uint32_t>(m_gpuChanSpctMap.size() - 1);
            eRef.reducedPowBaseIdx = linkSlot * static_cast<uint32_t>(numRb);
            const size_t newSize = size_t(eRef.reducedPowBaseIdx) + numRb;
            if (m_reducedPowFlat.size() < newSize)
                m_reducedPowFlat.resize(newSize, 0.0f);
        }

        // Compute chanSpct_unscaled contraction into a temporary buffer, then
        // reduce to per-rb isotropic power stored in m_reducedPowFlat.
        // Column-major layout: H[rx,tx] at index tx * numRxPorts + rx.
        std::vector<float> tmpFlat(rxtx * numRb * 2, 0.0f);
        float* outRaw = tmpFlat.data();
        const auto& ltVals = longTerm->GetValues();
        const auto* ltRaw = reinterpret_cast<const double*>(&ltVals[0]);
        const auto* dRaw = reinterpret_cast<const double*>(delayT.data());
        for (size_t c = 0; c < numCluster; ++c)
        {
            const double* aRow = ltRaw + c * rxtx * 2;
            const double* bRow = dRaw + c * numRb * 2;
            for (size_t rb = 0; rb < numRb; ++rb)
            {
                const double bRe = bRow[rb * 2];
                const double bIm = bRow[rb * 2 + 1];
                float* outRow = outRaw + rb * rxtx * 2;
                for (size_t i = 0; i < rxtx; ++i)
                {
                    const double aRe = aRow[i * 2];
                    const double aIm = aRow[i * 2 + 1];
                    outRow[i * 2] += static_cast<float>(aRe * bRe - aIm * bIm);
                    outRow[i * 2 + 1] += static_cast<float>(aRe * bIm + aIm * bRe);
                }
            }
        }

        // Reduce: reducedPow[rb] = sum_rx |sum_tx H[rx,tx,rb]|^2
        float* reducedPow = m_reducedPowFlat.data() + eRef.reducedPowBaseIdx;
        m_reducedPowNumRb = static_cast<uint32_t>(numRb);
        for (size_t rb = 0; rb < numRb; ++rb)
        {
            const float* rbSlice = outRaw + rb * rxtx * 2;
            float pow_acc = 0.0f;
            for (size_t rx = 0; rx < numRxPorts; ++rx)
            {
                float sum_re = 0.0f;
                float sum_im = 0.0f;
                for (size_t tx = 0; tx < numTxPorts; ++tx)
                {
                    const size_t idx = tx * numRxPorts + rx;
                    sum_re += rbSlice[idx * 2];
                    sum_im += rbSlice[idx * 2 + 1];
                }
                pow_acc += sum_re * sum_re + sum_im * sum_im;
            }
            reducedPow[rb] = pow_acc;
        }

        eRef.numRxPorts = numRxPorts;
        eRef.numTxPorts = numTxPorts;
        eRef.numRb = numRb;
        eRef.generatedTime = channelMatrix->m_generatedTime;
        eRef.rbFreqsHash = m_batchRbFreqsHash;

        // Build fakeChanSpct from reduced power.
        // Use actual antenna dims for small MIMO so NrInterference covariance ops work.
        const bool useFullDims2 =
            (size_t(numRxPorts) * numTxPorts <= kMaxMimoFakeEntries);
        const uint8_t fakeRx2 = useFullDims2 ? numRxPorts : uint8_t(1);
        const uint8_t fakeTx2 = useFullDims2 ? numTxPorts : uint8_t(1);
        if (!eRef.fakeChanSpct || eRef.fakeChanSpct->GetNumPages() != numRb ||
            static_cast<uint8_t>(eRef.fakeChanSpct->GetNumRows()) != fakeRx2 ||
            static_cast<uint8_t>(eRef.fakeChanSpct->GetNumCols()) != fakeTx2)
            eRef.fakeChanSpct =
                Create<Complex3DVector>(fakeRx2, fakeTx2, static_cast<uint16_t>(numRb));
        // reducedPow = beamformed power; target Power = reducedPow * inPsd.
        //   (numRx,numTx) shape: invN=1/numTx, so |H[0,0]|^2 must be numTx*reducedPow*inPsd
        //   (1,1) shape:         invN=1,       so |H[0,0]|^2 = reducedPow*inPsd
        const double invNumTx2 = useFullDims2 ? static_cast<double>(numTxPorts) : 1.0;
        for (size_t rb = 0; rb < numRb; ++rb)
        {
            const double sv = sqrtVit[rb];
            const double psd = static_cast<double>(reducedPow[rb]) * sv * sv * invNumTx2;
            auto* pageD = reinterpret_cast<double*>(eRef.fakeChanSpct->GetPagePtr(rb));
            pageD[0] = psd > 0.0 ? std::sqrt(psd) : 0.0;
            pageD[1] = 0.0;
        }
        return eRef.fakeChanSpct;
    }

    // Disabled by default. Per-eval GPU dispatch overhead (~600 us
    // observed via Dawn on D3D12 for the small chanSpct kernel) is
    // ~3x the CPU contraction cost (~190 us in Debug) at
    // DenseAmimoIntel sizes, so enabling this path makes the bench
    // noticeably slower (16 s -> 27 s in the A/B test). The kernel
    // and host plumbing are kept as scaffolding for a future batched
    // approach: do the per-cluster outer-product for ALL active
    // links in one dispatch during UpdateChannel, cache per-link
    // chanSpct, and have the eval-time hook just do a hash-keyed
    // cache lookup. That requires capturing PSD + Simulator::Now()
    // upstream of UpdateChannel which the current API doesn't expose
    // -- left as follow-up work. Set MEZ_GPU_SPEC=1 to force-enable
    // the per-eval path for measurement / future profiling.
    static const bool gpuSpecEnabled = []() {
        const char* e = std::getenv("MEZ_GPU_SPEC");
        return (e && e[0] == '1' && e[1] == '\0');
    }();
    if (!gpuSpecEnabled)
    {
        return nullptr;
    }
    if (!channelMatrix || !channelParams)
    {
        return nullptr;
    }
    // channelMatrix->m_antennaPair = (sAntId, uAntId) -- the GetKey
    // pair is symmetric (min/max) so the lookup works regardless of
    // PRX's isReverse orientation.
    const uint64_t key = MatrixBasedChannelModel::GetKey(
        channelMatrix->m_antennaPair.first, channelMatrix->m_antennaPair.second);
    auto it = m_gpuLongTermMap.find(key);
    if (it == m_gpuLongTermMap.end())
    {
        return nullptr;
    }
    const GpuLongTermEntry& entry = it->second;
    if (entry.generatedTime != channelMatrix->m_generatedTime)
    {
        return nullptr;
    }
    auto* gpu = m_wgpuChannel.get();
    if (!gpu)
    {
        return nullptr;
    }
    const size_t numCluster = channelMatrix->m_channel.GetNumPages();
    if (delayT.size() != numCluster * numRb || sqrtVit.size() != numRb)
    {
        return nullptr;
    }
    // Pack delayT + sqrtVit into f32 buffers for the kernel. delayT is
    // 24*24=576 complex entries (~9 KB f64 -> ~4.6 KB f32) for the
    // DenseAmimoIntel config -- small enough that the f32 quantisation
    // contributes orders of magnitude less error than the f32 longTerm
    // already does.
    std::vector<std::complex<float>> delayTF(delayT.size());
    for (size_t i = 0; i < delayT.size(); ++i)
    {
        delayTF[i] = std::complex<float>(static_cast<float>(delayT[i].real()),
                                         static_cast<float>(delayT[i].imag()));
    }
    std::vector<float> sqrtVitF(sqrtVit.size());
    for (size_t i = 0; i < sqrtVit.size(); ++i)
    {
        sqrtVitF[i] = static_cast<float>(sqrtVit[i]);
    }
    const std::vector<std::complex<float>> outFlat =
        gpu->genSpecChan(entry.gpuLinkIdx,
                         static_cast<uint32_t>(numCluster),
                         numRb,
                         numRxPorts,
                         numTxPorts,
                         entry.ltUPorts,
                         entry.ltSPorts,
                         isReverse,
                         delayTF,
                         sqrtVitF);
    if (outFlat.empty())
    {
        return nullptr;
    }
    Ptr<Complex3DVector> chanSpct = Create<Complex3DVector>(
        static_cast<uint16_t>(numRxPorts),
        static_cast<uint16_t>(numTxPorts),
        static_cast<uint16_t>(numRb));
    // chanSpct column-major: (rx, tx, rb) at rx + numRxPorts*tx + per_page*rb,
    // which matches the kernel's output layout. Single contiguous f32 -> f64
    // conversion.
    auto* dst = reinterpret_cast<double*>(chanSpct->GetPagePtr(0));
    const size_t cells = size_t(numRxPorts) * numTxPorts * numRb;
    for (size_t k = 0; k < cells; ++k)
    {
        const auto* sr = reinterpret_cast<const float*>(&outFlat[k]);
        dst[k * 2] = static_cast<double>(sr[0]);
        dst[k * 2 + 1] = static_cast<double>(sr[1]);
    }
    return chanSpct;
}

uint64_t
ThreeGppChannelModelWgpuMezanine::HashFloatVector(const std::vector<float>& v)
{
    uint64_t h = 1469598103934665603ull;
    constexpr uint64_t kPrime = 1099511628211ull;
    for (float f : v)
    {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        h ^= bits;
        h *= kPrime;
    }
    return h;
}

uint64_t
ThreeGppChannelModelWgpuMezanine::HashComplexVector(const PhasedArrayModel::ComplexVector& v)
{
    // FNV-1a 64-bit over the raw double pairs (re, im) of each
    // complex element. The PhasedArrayModel::ComplexVector is
    // backed by a contiguous valarray<complex<double>>, so we walk
    // GetValues() as a uint64_t* stream.
    uint64_t h = 1469598103934665603ull; // offset basis
    constexpr uint64_t kPrime = 1099511628211ull;
    const size_t n = v.GetSize();
    const auto& values = v.GetValues();
    const auto* raw = reinterpret_cast<const uint64_t*>(&values[0]);
    const size_t words = n * 2; // 2 doubles per complex<double>
    for (size_t i = 0; i < words; ++i)
    {
        h ^= raw[i];
        h *= kPrime;
    }
    return h;
}

} // namespace ns3
