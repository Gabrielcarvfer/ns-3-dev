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
#include "ns3/net-device.h"
#include "ns3/node-list.h"
#include "ns3/node.h"
#include "ns3/phased-array-model.h"
#include "ns3/simulator.h"
#include "ns3/three-gpp-antenna-model.h"
#include "ns3/uinteger.h"
#include "ns3/uniform-planar-array.h"
#include "ns3/vector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
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
    (void)label;
}
} // namespace
#else
namespace { static void mezPrintMem(const char*) {} }
#endif

NS_LOG_COMPONENT_DEFINE("ThreeGppChannelModelWgpuMezanine");

namespace
{
/// Mix the antenna-pair key with the (sW, uW) beam-pair hashes so the
/// spectrum cache is keyed per (link, beam pair). boost::hash_combine-style.
inline uint64_t
MixBeamKey(uint64_t key, uint64_t sWHash, uint64_t uWHash)
{
    uint64_t h = key * 0x9E3779B97F4A7C15ull;
    h ^= sWHash + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    h ^= uWHash + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    return h;
}
} // namespace

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
        // Panel bearing (alpha) in degrees: the kernels rotate the azimuth
        // into the panel's local frame before the directional pattern
        // lookup. Without it all three sectors of a site evaluated the
        // pattern in the global frame.
        cfg.bearingDeg = p.panelOrientation[0] * 180.0f / static_cast<float>(M_PI);

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
        // Actual element spacings in wavelengths. These were hardcoded to
        // 0.5/0.5, silently misplacing every element of arrays with
        // non-default spacing (e.g. the DenseA gNB uses d_v = 0.8) and
        // distorting the spatial correlation / steering structure.
        p.antSpacing = {0.f,
                        0.f,
                        static_cast<float>(ant->GetAntennaHorizontalSpacing()),
                        static_cast<float>(ant->GetAntennaVerticalSpacing())};
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
        // Element field-pattern tables (per-degree dB attenuation; the
        // kernel computes A_db = theta[ti] + phi[pi] + (antModel==1 ? GMAX : 0)).
        // These were previously left at ZERO with antModel=0 — i.e. every
        // element forced isotropic — while the CPU model applied the 3GPP
        // directive element (8 dBi boresight, 30 dB floors). Serving
        // in-sector links lost ~8 dB of element gain and off-boresight
        // interference gained relatively, collapsing the serving-vs-
        // interference separation (measured 4.0 dB vs the CPU's 20.6 dB)
        // and producing the 16-19 dB SINR deficit.
        const bool is3gppElem =
            DynamicCast<const ThreeGppAntennaModel>(ant->GetAntennaElement()) != nullptr;
        p.antModel = is3gppElem ? 1u : 0u;
        p.thetaDeg.assign(181, 0.f);
        p.phiDeg.assign(360, 0.f);
        if (is3gppElem)
        {
            // TR 38.901 Table 7.3-1: A_v(theta) = -min(12((theta-90)/65)^2, 30),
            // A_h(phi) = -min(12(phi/65)^2, 30); GMAX (8 dBi) added in-kernel.
            for (int t = 0; t <= 180; ++t)
            {
                p.thetaDeg[t] = static_cast<float>(
                    -std::min(12.0 * std::pow((t - 90.0) / 65.0, 2.0), 30.0));
            }
            for (int f = 0; f < 360; ++f)
            {
                const double sf = (f > 180) ? f - 360.0 : f;
                p.phiDeg[f] = static_cast<float>(
                    -std::min(12.0 * std::pow(sf / 65.0, 2.0), 30.0));
            }
        }

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
                // Same geometry as the reference, but panels may differ in
                // BEARING (three sectors per site) — and the bearing drives
                // the directional-pattern rotation. Reuse a config only when
                // both geometry and bearing match; otherwise append a new
                // panel slot (same nAnt keeps it in the same bucket).
                const float myBearing =
                    p.panelOrientation[0] * 180.0f / static_cast<float>(M_PI);
                int found = -1;
                for (size_t ci = 0; ci < antCfgs.size(); ++ci)
                {
                    if (antCfgs[ci].nAnt == p.nAnt &&
                        std::abs(antCfgs[ci].bearingDeg - myBearing) < 0.01f)
                    {
                        found = static_cast<int>(ci);
                        break;
                    }
                }
                if (found >= 0)
                {
                    p.panelIdx = static_cast<uint32_t>(found);
                }
                else
                {
                    appendPanelConfig(p);
                }
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

    // Base-station classifier, cached per node id. A node is BS-side when
    // any of its NetDevices' type names mention Gnb/Enb (NR/LTE); works
    // regardless of UE heights (indoor UEs sit at per-floor z).
    std::unordered_map<uint32_t, bool> bsByNodeId;
    auto nodeIsBs = [&bsByNodeId](uint32_t nodeId) {
        auto it = bsByNodeId.find(nodeId);
        if (it != bsByNodeId.end())
            return it->second;
        bool isBs = false;
        Ptr<Node> node = NodeList::GetNode(nodeId);
        if (node)
        {
            for (uint32_t d = 0; d < node->GetNDevices() && !isBs; ++d)
            {
                const std::string tn = node->GetDevice(d)->GetInstanceTypeId().GetName();
                isBs = tn.find("Gnb") != std::string::npos ||
                       tn.find("Enb") != std::string::npos;
            }
        }
        bsByNodeId.emplace(nodeId, isBs);
        return isBs;
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

        const NodeInfo& sNodeInfo = buildNodeInfo(txNodeId);
        const NodeInfo& uNodeInfo = buildNodeInfo(rxNodeId);

        // The GPU (site, ut) topology models gNB->UE links only. UE-UE and
        // gNB-gNB matrices (e.g. from cross-link or attachment evals) must
        // NOT enter the grid: a UE registered as a "site" corrupts the CRN
        // dimensions and the cid*nUt+uid LSP indexing for every real link —
        // observed as cp.nCluster==0 for ALL links (the pipeline then
        // silently dropped to the CPU fallback paths from refresh 2 on).
        // Classify by NetDevice type name rather than height: indoor UEs
        // sit at per-floor heights and defeat a z comparison. Excluded
        // links keep working through the PRX CPU fallbacks.
        if (!nodeIsBs(txNodeId) || nodeIsBs(rxNodeId))
        {
            continue;
        }

        const SiteRec& site = getOrCreateSite(txNodeId, txAntId);
        const UtRec& ut = getOrCreateUt(rxNodeId, rxAntId, params, paramsSameDirection);

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

    // Evict stale m_gpuChanSpctMap entries for antenna pairs that are no
    // longer in runtimeLinks. The primary source of stale entries is the
    // maxRSRP initial-attachment phase, which uses temporary
    // Copy<UniformPlanarArray> objects (1x1 ports) with different antenna
    // IDs from the user-configured antennas used in real traffic. Without
    // eviction those entries accumulate indefinitely and waste memory.
    // This sweep is O(N) in the map size and runs at most once per
    // UpdateChannel call — cheap compared to the GPU dispatch that follows.
    if (!m_gpuChanSpctMap.empty())
    {
        std::unordered_set<uint64_t> activeKeys;
        activeKeys.reserve(runtimeLinks.size());
        for (const auto& ctx : runtimeLinks)
            activeKeys.insert(ctx.matrixKey);
        // The map is keyed by MixBeamKey(matrixKey, beams); evict by the
        // plain matrixKey stored in each entry.
        for (auto it = m_gpuChanSpctMap.begin(); it != m_gpuChanSpctMap.end();)
            it = activeKeys.count(it->second.matrixKey) ? std::next(it)
                                                        : m_gpuChanSpctMap.erase(it);
    }

    // Group runtimeLinks by (sNAnt, uNAnt). Every bucket runs the full
    // matrix/LT/spec pipeline independently; CRN/LSP/cluster kernels run
    // once for the whole grid and are shared across all buckets.
    if (runtimeLinks.empty())
    {
        return;
    }
    std::map<std::pair<uint32_t, uint32_t>, std::vector<size_t>> bucketMap;
    {
        for (size_t i = 0; i < runtimeLinks.size(); ++i)
        {
            const auto& ctx = runtimeLinks[i];
            if (!ctx.sAnt || !ctx.uAnt)
                continue;
            const auto sN = static_cast<uint32_t>(ctx.sAnt->GetNumElems());
            const auto uN = static_cast<uint32_t>(ctx.uAnt->GetNumElems());
            bucketMap[{sN, uN}].push_back(i);
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

    // ── Static-channel fast paths ─────────────────────────────────────────
    // For m_updatePeriod=0 (static channel) the cluster parameters (delays,
    // powers, cluster/ray angles) are drawn ONCE and never change.  Re-running
    // the full CRN→LSP→clusterRay→ReadParams pipeline (including an 8+ MB
    // GPU→CPU readback) on every periodic refresh tick is pure waste.
    //
    // Path A — no beams changed: nothing in the GPU caches is stale. Just
    //   extend the generatedTime stamps on all existing cache entries so
    //   TryGenSpectrumChannelMatrix and GetCachedLongTerm keep returning hits.
    //   Zero GPU work. Return immediately.
    //
    // Path B — beams changed: skip CRN/LSP/clusterRay/ReadParams/Populate.
    //   Re-run only genChannelMatrixAndLongTermFused (which reads the still-
    //   valid GPU cluster-params buffers from the last full run) and
    //   genSpecBatch to produce fresh reducedPow with the new LT.
    //   Saves the ~8.5 MB readback and the sequential CPU Populate loop on
    //   every UC call where only beam weights have changed.
    const bool clusterParamsFresh =
        m_gpuClusterParamsFresh && m_updatePeriod.IsZero() &&
        runtimeLinks.size() == m_lastRuntimeLinksCount;

    bool anyBeamChanged = false;
    if (clusterParamsFresh)
    {
        for (const auto& rtCtx : runtimeLinks)
        {
            const auto ltIt = m_gpuLongTermMap.find(rtCtx.matrixKey);
            if (ltIt == m_gpuLongTermMap.end())
            {
                anyBeamChanged = true;
                break;
            }
            if (rtCtx.sAnt && rtCtx.uAnt)
            {
                const auto& sBV = rtCtx.sAnt->GetBeamformingVectorRef();
                const auto& uBV = rtCtx.uAnt->GetBeamformingVectorRef();
                if (HashComplexVector(sBV) != ltIt->second.sWHash ||
                    HashComplexVector(uBV) != ltIt->second.uWHash ||
                    sBV.GetSize() != ltIt->second.sWSize ||
                    uBV.GetSize() != ltIt->second.uWSize)
                {
                    anyBeamChanged = true;
                    break;
                }
            }
        }

        if (!anyBeamChanged)
        {
            // Path A: extend all cache lifetimes, zero GPU work.
            SLS_PHASE_SCOPE("Mez::FastPath_A");
            const Time t = Simulator::Now();
            for (auto& [k, e] : m_gpuChanSpctMap)
                e.generatedTime = t;
            for (auto& [k, e] : m_gpuLongTermMap)
                e.generatedTime = t;
            for (auto& [k, m] : m_channelMatrixMap)
                if (m)
                    m->m_generatedTime = t;
            return;
        }
        // Path B: fall through — skipLspCluster=true guards the expensive blocks below.
    }
    const bool skipLspCluster = clusterParamsFresh && anyBeamChanged;

    const float margin = 50.f;
    const float maxXf = std::isfinite(maxX) ? static_cast<float>(maxX + margin) : 1000.f;
    const float minXf = std::isfinite(minX) ? static_cast<float>(minX - margin) : -1000.f;
    const float maxYf = std::isfinite(maxY) ? static_cast<float>(maxY + margin) : 1000.f;
    const float minYf = std::isfinite(minY) ? static_cast<float>(minY - margin) : -1000.f;

    // ltEnabled is stable across UpdateChannel calls (read once from env).
    static const bool ltEnabled = []() {
        const char* e = std::getenv("MEZ_GPU_LT");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    // Declared here (outside if(!skipLspCluster)) so the bucket loop's
    // Populate else-branch (Path C) can access them; empty in Path B.
    std::vector<LinkParams> linkParams;
    std::vector<ClusterParamsGpu> clusterParams;
    std::vector<float> xprFlat;
    std::vector<float> phiNmAoaFlat;
    std::vector<float> phiNmAodFlat;
    std::vector<float> thetaNmZoaFlat;
    std::vector<float> thetaNmZodFlat;

    // Path B / Path C only: upload and dispatch the LSP + cluster pipeline.
    // Path A never reaches here (it returned early).
    // Path B (skipLspCluster=true) skips the expensive CRN/LSP/clusterRay
    // kernels + 8 MB ReadParams readback; the GPU cluster-params buffers from
    // the previous full run are still valid for static channels.
    if (!skipLspCluster)
    {
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
    else if (m_scenario == "InH-OfficeMixed")
        gpuScenario = 3;
    else if (m_scenario == "InH-OfficeOpen")
        gpuScenario = 4;
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

    // ── Scenario-correct LSP-draw parameters ──────────────────────────
    // cal_link_param draws each LSP as cv*sigma + mu. Fill mu/sigma and
    // the sqrt cross-correlation matrices from the CPU-side
    // GetThreeGppTable (which evaluates TR 38.901 Table 7.5-6 for the
    // active scenario, frequency and geometry) instead of relying on
    // SlsChanWgpu's lazy defaults (UMa@3.5GHz marginals + IDENTITY
    // cross-correlation). Slot order in CmnLinkParamsGPU: [0]=LOS,
    // [1]=NLOS, [2]=O2I. Terrestrial Table 7.5-6 entries depend only on
    // scenario + frequency, so one representative link suffices.
    Ptr<const ParamsTable> tLos;
    Ptr<const ParamsTable> tNlos;
    Ptr<const ParamsTable> tO2i;
    if (!runtimeLinks.empty())
    {
        const auto& repCtx = runtimeLinks.front();
        auto condLos = CreateObject<ChannelCondition>();
        condLos->SetLosCondition(ChannelCondition::LOS);
        auto condNlos = CreateObject<ChannelCondition>();
        condNlos->SetLosCondition(ChannelCondition::NLOS);

        tLos = GetThreeGppTable(repCtx.sMob, repCtx.uMob, condLos);
        tNlos = GetThreeGppTable(repCtx.sMob, repCtx.uMob, condNlos);
        // The O2I column only exists for UMa/UMi/RMa (InH asserts on o2i;
        // V2V/NTN have no O2I table). Reuse NLOS for the others -- the
        // O2I slot is only read for links the condition model marked O2I,
        // which cannot happen in those scenarios.
        tO2i = tNlos;
        if (m_scenario == "UMa" || m_scenario == "UMi-StreetCanyon" || m_scenario == "RMa")
        {
            auto condO2i = CreateObject<ChannelCondition>();
            condO2i->SetLosCondition(ChannelCondition::NLOS);
            condO2i->SetO2iCondition(ChannelCondition::O2I);
            tO2i = GetThreeGppTable(repCtx.sMob, repCtx.uMob, condO2i);
        }

        CmnLinkParamsGPU clp{};
        clp.lgfc = static_cast<float>(std::log10(m_frequency / 1e9));
        const auto fillSlot = [&clp](uint32_t s, const Ptr<const ParamsTable>& t) {
            clp.mu_lgDS[s] = static_cast<float>(t->m_uLgDS);
            clp.sigma_lgDS[s] = static_cast<float>(t->m_sigLgDS);
            clp.mu_lgASD[s] = static_cast<float>(t->m_uLgASD);
            clp.sigma_lgASD[s] = static_cast<float>(t->m_sigLgASD);
            clp.mu_lgASA[s] = static_cast<float>(t->m_uLgASA);
            clp.sigma_lgASA[s] = static_cast<float>(t->m_sigLgASA);
            clp.mu_lgZSA[s] = static_cast<float>(t->m_uLgZSA);
            clp.sigma_lgZSA[s] = static_cast<float>(t->m_sigLgZSA);
            clp.mu_K[s] = static_cast<float>(t->m_uK);
            clp.sigma_K[s] = static_cast<float>(t->m_sigK);
        };
        fillSlot(0, tLos);
        fillSlot(1, tNlos);
        fillSlot(2, tO2i);
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                clp.sqrtCorrMatLos[i * 7 + j] = static_cast<float>(tLos->m_sqrtC[i][j]);
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                clp.sqrtCorrMatNlos[i * 6 + j] = static_cast<float>(tNlos->m_sqrtC[i][j]);
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                clp.sqrtCorrMatO2i[i * 6 + j] = static_cast<float>(tO2i->m_sqrtC[i][j]);
        gpu->uploadCmnLinkParams(clp);
    }

    // TR 38.901 Table 7.5-6 correlation distances (m) for the CRN grids,
    // per scenario. Order: LOS [SF,K,DS,ASD,ASA,ZSD,ZSA,DT],
    // NLOS/O2I [SF,DS,ASD,ASA,ZSD,ZSA,DT]. Unlisted scenarios (V2V, NTN)
    // currently fall back to UMa distances.
    float corrLos[8] = {37.f, 12.f, 30.f, 18.f, 15.f, 15.f, 15.f, 20.f};
    float corrNlos[7] = {50.f, 40.f, 50.f, 50.f, 50.f, 50.f, 25.f};
    float corrO2i[7] = {10.f, 10.f, 11.f, 17.f, 25.f, 25.f, 20.f};
    if (m_scenario == "UMi-StreetCanyon" || m_scenario == "UMi")
    {
        const float l[8] = {10.f, 15.f, 7.f, 8.f, 8.f, 12.f, 12.f, 12.f};
        const float n[7] = {13.f, 10.f, 10.f, 9.f, 10.f, 10.f, 15.f};
        const float o[7] = {7.f, 10.f, 11.f, 17.f, 25.f, 25.f, 15.f};
        std::copy(std::begin(l), std::end(l), corrLos);
        std::copy(std::begin(n), std::end(n), corrNlos);
        std::copy(std::begin(o), std::end(o), corrO2i);
    }
    else if (m_scenario == "RMa")
    {
        const float l[8] = {37.f, 40.f, 50.f, 25.f, 35.f, 15.f, 15.f, 40.f};
        const float n[7] = {120.f, 36.f, 30.f, 40.f, 50.f, 50.f, 40.f};
        std::copy(std::begin(l), std::end(l), corrLos);
        std::copy(std::begin(n), std::end(n), corrNlos);
        std::copy(std::begin(n), std::end(n), corrO2i);
    }
    else if (m_scenario == "InH-OfficeMixed" || m_scenario == "InH-OfficeOpen")
    {
        const float l[8] = {10.f, 4.f, 8.f, 7.f, 5.f, 4.f, 4.f, 10.f};
        const float n[7] = {6.f, 5.f, 3.f, 3.f, 4.f, 4.f, 10.f};
        std::copy(std::begin(l), std::end(l), corrLos);
        std::copy(std::begin(n), std::end(n), corrNlos);
        std::copy(std::begin(n), std::end(n), corrO2i);
    }

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

    // ── Table-driven small-scale parameters ───────────────────────────
    // Fill every per-condition Table 7.5-6 field from the CPU-side
    // ParamsTable fetched above (tLos/tNlos/tO2i), replacing the previous
    // hand-coded UMa/UMi/RMa branches. This makes every scenario the CPU
    // model supports (incl. InH-Office*, V2V, NTN) flow through to the
    // GPU small-scale pipeline without per-scenario code here.
    // Slot order here is the mezanine's pre-swap convention
    // [0]=NLOS, [1]=LOS, [2]=O2I; the swapLN() below converts to the
    // kernel's [0]=LOS, [1]=NLOS, [2]=O2I.
    NS_ABORT_MSG_UNLESS(tLos && tNlos && tO2i,
                        "UpdateChannel ran with no runtime links; cannot derive "
                        "scenario tables");
    const auto fillSs = [&ss](uint32_t s, const Ptr<const ParamsTable>& t) {
        ss.mu_lgDS[s] = static_cast<float>(t->m_uLgDS);
        ss.sigma_lgDS[s] = static_cast<float>(t->m_sigLgDS);
        ss.mu_lgASD[s] = static_cast<float>(t->m_uLgASD);
        ss.sigma_lgASD[s] = static_cast<float>(t->m_sigLgASD);
        ss.mu_lgASA[s] = static_cast<float>(t->m_uLgASA);
        ss.sigma_lgASA[s] = static_cast<float>(t->m_sigLgASA);
        ss.mu_lgZSA[s] = static_cast<float>(t->m_uLgZSA);
        ss.sigma_lgZSA[s] = static_cast<float>(t->m_sigLgZSA);
        ss.mu_K[s] = static_cast<float>(t->m_uK);
        ss.sigma_K[s] = static_cast<float>(t->m_sigK);
        ss.nCluster[s] = t->m_numOfCluster;
        ss.nRayPerCluster[s] = t->m_raysPerCluster;
        ss.r_tao[s] = static_cast<float>(t->m_rTau);
        // ParamsTable stores cDS in seconds; the kernel expects nanoseconds.
        ss.C_DS[s] = static_cast<float>(t->m_cDS * 1e9);
        ss.C_ASD[s] = static_cast<float>(t->m_cASD);
        ss.C_ASA[s] = static_cast<float>(t->m_cASA);
        ss.C_ZSA[s] = static_cast<float>(t->m_cZSA);
        ss.xi[s] = static_cast<float>(t->m_perClusterShadowingStd);
        ss.mu_XPR[s] = static_cast<float>(t->m_uXpr);
        ss.sigma_XPR[s] = static_cast<float>(t->m_sigXpr);
    };
    fillSs(0, tNlos);
    fillSs(1, tLos);
    fillSs(2, tO2i);

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

    // ── Spatial consistency (TR 38.901 7.6.3.2 procedure A) ─────────────
    // Links that moved less than the consistency threshold since their
    // params were generated keep their cluster/ray realization: the
    // cluster kernel skips them (preserving packed per-ray XPR / coupling
    // / phases that exist only on-GPU), and after the kernel the host
    // drifts their cluster-level delays/powers/angles via the base-class
    // procedure-A update and shifts the per-ray angles by the per-cluster
    // deltas. Opt out with MEZ_SC=0.
    static const bool scEnabled = []() {
        const char* e = std::getenv("MEZ_SC");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    std::vector<uint32_t> scSkipMask(size_t(nSite) * nUt, 0u);
    std::vector<const RuntimeLinkCtx*> scDriftLinks;
    if (scEnabled)
    {
        std::unordered_set<uint64_t> seenParams;
        for (const auto& ctx : runtimeLinks)
        {
            if (!ctx.condition || !ctx.sMob || !ctx.uMob)
                continue;
            auto pit = m_channelParamsMap.find(ctx.paramsKey);
            if (pit == m_channelParamsMap.end())
                continue;
            auto prev = DynamicCast<ThreeGppChannelParams>(pit->second);
            if (!prev || prev->m_reducedClusterNumber == 0)
                continue;
            // The preserved GPU slot must be the same one that produced
            // these params; a topology change invalidates the alignment.
            auto idxIt = m_prevLspReadIdx.find(ctx.paramsKey);
            if (idxIt == m_prevLspReadIdx.end() || idxIt->second != ctx.lspReadIdx)
                continue;
            if (prev->m_losCondition != ctx.condition->GetLosCondition() ||
                prev->m_o2iCondition != ctx.condition->GetO2iCondition())
                continue;
            // Endpoint displacement below the procedure-A threshold (the
            // canonical ordering matches Populate: min node id first).
            Ptr<MobilityModel> aMob = ctx.sMob;
            Ptr<MobilityModel> bMob = ctx.uMob;
            if (ctx.sNodeId > ctx.uNodeId)
                std::swap(aMob, bMob);
            const Vector pa = aMob->GetPosition();
            const Vector pb = bMob->GetPosition();
            const double da = std::hypot(pa.x - prev->m_lastPositionFirst.x,
                                         pa.y - prev->m_lastPositionFirst.y);
            const double db = std::hypot(pb.x - prev->m_lastPositionSecond.x,
                                         pb.y - prev->m_lastPositionSecond.y);
            if (std::max(da, db) > 1.0)
                continue;
            scSkipMask[ctx.lspReadIdx] = 1u;
            if (seenParams.insert(ctx.paramsKey).second)
                scDriftLinks.push_back(&ctx);
        }
    }
    gpu->uploadClusterSkipMask(scSkipMask);

    // Run calClusterRay once for the full grid (antenna-independent).
    {
        SLS_PHASE_SCOPE("Mez::SmallScaleKernels");
        {
            SLS_PHASE_SCOPE("Mez::CalClusterRay");
            gpu->calClusterRay(nSite, nUt);
        }
    }
    mezPrintMem("post-SSKernels");

    // Apply the host-side procedure-A drift for preserved links, then
    // push the drifted cluster data back to the GPU (the subsequent
    // ReadParams sees the drifted values, so Populate adopts them).
    if (!scDriftLinks.empty())
    {
        SLS_PHASE_SCOPE("Mez::ScDrift");
        std::vector<DriftEntryGpu> driftEntries;
        driftEntries.reserve(scDriftLinks.size());
        auto wrapDelta = [](double d) {
            while (d > 180.0)
                d -= 360.0;
            while (d < -180.0)
                d += 360.0;
            return static_cast<float>(d);
        };
        for (const auto* pctx : scDriftLinks)
        {
            auto params =
                DynamicCast<ThreeGppChannelParams>(m_channelParamsMap[pctx->paramsKey]);
            if (!params)
                continue;
            const Double2DVector oldAngles = params->m_angle;
            Ptr<MobilityModel> aMob = pctx->sMob;
            Ptr<MobilityModel> bMob = pctx->uMob;
            if (pctx->sNodeId > pctx->uNodeId)
                std::swap(aMob, bMob);
            UpdateChannelParameters(params, pctx->condition, aMob, bMob);

            const uint32_t nc =
                std::min<uint32_t>(params->m_reducedClusterNumber, MAX_CLUSTERS);
            DriftEntryGpu de{};
            de.linkIdx = pctx->lspReadIdx;
            ClusterParamsGpu cp{};
            cp.nCluster = nc;
            cp.nRayPerCluster = 20u; // all supported scenarios use 20 rays
            cp.strongest2clustersIdx[0] = params->m_cluster1st;
            cp.strongest2clustersIdx[1] = params->m_cluster2nd;
            for (uint32_t c = 0; c < nc; ++c)
            {
                de.dAOA[c] = wrapDelta(params->m_angle[AOA_INDEX][c] - oldAngles[AOA_INDEX][c]);
                de.dAOD[c] = wrapDelta(params->m_angle[AOD_INDEX][c] - oldAngles[AOD_INDEX][c]);
                de.dZOA[c] = wrapDelta(params->m_angle[ZOA_INDEX][c] - oldAngles[ZOA_INDEX][c]);
                de.dZOD[c] = wrapDelta(params->m_angle[ZOD_INDEX][c] - oldAngles[ZOD_INDEX][c]);
                cp.delays[c] = static_cast<float>(params->m_delay[c] * 1e9); // s -> ns
                cp.powers[c] = static_cast<float>(params->m_clusterPower[c]);
                cp.phinAoA[c] = static_cast<float>(params->m_angle[AOA_INDEX][c]);
                cp.phinAoD[c] = static_cast<float>(params->m_angle[AOD_INDEX][c]);
                cp.thetanZOA[c] = static_cast<float>(params->m_angle[ZOA_INDEX][c]);
                cp.thetanZOD[c] = static_cast<float>(params->m_angle[ZOD_INDEX][c]);
            }
            gpu->writeClusterParams(pctx->lspReadIdx, cp);
            driftEntries.push_back(de);
        }
        gpu->driftPackedAngles(driftEntries);
    }

    NS_LOG_DEBUG("UpdateChannel uploaded " << nSite << " cells, " << nUt << " UEs, "
                                           << antCfgs.size() << " antenna panels, "
                                           << runtimeLinks.size() << " total links across "
                                           << bucketMap.size() << " buckets");

    // Read antenna-independent params once (full [nSite x nUt] grid).
    // (Declarations are outside the if block so Path B can skip this
    //  section while the Populate else-branch can still reference them.)
    //
    // Compact read: only transfer clusterParamsBuf_ (282 KB for ring-3),
    // not clusterOutputsBuf_ (8.2 MB packed per-ray angles/XPR).
    // The per-ray arrays (m_rayAoaRadian etc.) are not needed because:
    //   - DelayProj spatial projections use cluster-level angles already
    //     present in ClusterParamsGpu (phinAoA, thetanZOA, etc.)
    //   - CPU fallback (0.35% of evals) uses GPU-cached longTerm + m_delay
    //     which does not require per-ray angles.
    {
        SLS_PHASE_SCOPE("Mez::ReadParams");
        linkParams = gpu->readLinkParams(nSite, nUt);
        clusterParams = gpu->readClusterParams(nSite, nUt);
        // xprFlat, phiNmAoaFlat etc. remain empty — per-ray data not needed.
    }
    mezPrintMem("post-ReadParams");

    // Mark cluster params as stable after a successful full run.
    // On subsequent calls for static channels (m_updatePeriod=0) the
    // expensive CRN/LSP/cluster pipeline is skipped (Path B / Path A).
    if (!skipLspCluster && m_updatePeriod.IsZero())
    {
        m_gpuClusterParamsFresh = true;
        m_lastRuntimeLinksCount = static_cast<uint32_t>(runtimeLinks.size());
    }
    } // end if (!skipLspCluster)

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
    static thread_local std::vector<std::complex<float>> chunkBatchH;
    static thread_local std::vector<float> chunkBatchPow;
    // Flat delay + spatial-projection cache built in Populate, consumed in
    // DelayProj. Indexed [bi * numClusters + c] within the current bucket.
    // Avoids pointer-chasing through m_channelParamsMap in the hot GenSpec path.
    static thread_local std::vector<float>  bucketDelaysFlat;    // seconds
    static thread_local std::vector<double> bucketSpatialProj;   // unitless

    // ── Per-bucket pipeline ──────────────────────────────────────────────
    // Each unique (sNAnt, uNAnt) pair gets its own matrix/LT/spec dispatch.
    // Reset the per-refresh region allocators: buckets claim disjoint
    // regions of m_specHFlat / m_specPowFlat as they are processed.
    m_specHBucketBase = 0;
    m_powBucketBase = 0;
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
                    // Hop-relative gather can reach
                    // ((portElems/elemsPerPort - 1) * numColumns + elemsPerPort - 1).
                    const uint32_t sMaxRel =
                        (ltSElemsPerPort > 0 && ltSPortElems > 0)
                            ? ((ltSPortElems - 1) / ltSElemsPerPort) *
                                      static_cast<uint32_t>(ctx.sAnt->GetNumColumns()) +
                                  ((ltSPortElems - 1) % ltSElemsPerPort)
                            : (ltSPortElems > 0 ? ltSPortElems - 1 : 0);
                    const uint32_t uMaxRel =
                        (ltUElemsPerPort > 0 && ltUPortElems > 0)
                            ? ((ltUPortElems - 1) / ltUElemsPerPort) *
                                      static_cast<uint32_t>(ctx.uAnt->GetNumColumns()) +
                                  ((ltUPortElems - 1) % ltUElemsPerPort)
                            : (ltUPortElems > 0 ? ltUPortElems - 1 : 0);
                    NS_ASSERT_MSG(sBV.GetSize() > sMaxRel,
                                  "Mez::genLongTerm: sBV[" << ctx.sAntId << "].size()="
                                  << sBV.GetSize() << " <= max rel offset " << sMaxRel);
                    NS_ASSERT_MSG(uBV.GetSize() > uMaxRel,
                                  "Mez::genLongTerm: uBV[" << ctx.uAntId << "].size()="
                                  << uBV.GetSize() << " <= max rel offset " << uMaxRel);
                    // PRX::CalculateLongTermComponent indexes the beam
                    // vector at HOP-RELATIVE offsets within the port
                    // (sW[sIndex - startS], where sIndex walks elemsPerPort
                    // contiguous columns then jumps to the next row):
                    //   rel(k) = (k / elemsPerPort) * numColumns + (k % elemsPerPort).
                    // The kernel uses a contiguous t_index, so the weights
                    // must be pre-gathered at those offsets here. Packing
                    // sBV[0..portElems) verbatim applied the WRONG phase
                    // ramp to rows past the first — every per-link longTerm
                    // decohered, which surfaced as ~15-20 dB per-link gain
                    // over-dispersion and a 16-19 dB SINR deficit.
                    const uint32_t sCols =
                        static_cast<uint32_t>(ctx.sAnt->GetNumColumns());
                    const uint32_t uCols =
                        static_cast<uint32_t>(ctx.uAnt->GetNumColumns());
                    for (uint32_t k = 0; k < ltSPortElems; ++k)
                    {
                        const uint32_t rel =
                            (ltSElemsPerPort > 0)
                                ? (k / ltSElemsPerPort) * sCols + (k % ltSElemsPerPort)
                                : k;
                        const auto v = sBV[rel];
                        ltSWFlat[sBase + k] = std::complex<float>(
                            static_cast<float>(v.real()), static_cast<float>(v.imag()));
                    }
                    for (uint32_t k = 0; k < ltUPortElems; ++k)
                    {
                        const uint32_t rel =
                            (ltUElemsPerPort > 0)
                                ? (k / ltUElemsPerPort) * uCols + (k % ltUElemsPerPort)
                                : k;
                        const auto v = uBV[rel];
                        ltUWFlat[uBase + k] = std::complex<float>(
                            static_cast<float>(v.real()), static_cast<float>(v.imag()));
                    }
                }
                ltCanDispatch = assembleOk && ltSPortElems > 0 && ltUPortElems > 0 &&
                                ltSPorts > 0 && ltUPorts > 0;
                static const bool diagLt = []() {
                    const char* e = std::getenv("MEZ_DIAG_H");
                    return e && e[0] == '1';
                }();
                if (diagLt)
                {
                    std::fprintf(stderr,
                                 "[MEZ_DIAG_H] ltGate bucket(s=%u,u=%u) nLinks=%u "
                                 "assembleOk=%d sPorts=%u uPorts=%u sPortElems=%u "
                                 "uPortElems=%u -> ltCanDispatch=%d\n",
                                 bSSize,
                                 bUSize,
                                 nBucketLinks,
                                 assembleOk ? 1 : 0,
                                 ltSPorts,
                                 ltUPorts,
                                 ltSPortElems,
                                 ltUPortElems,
                                 ltCanDispatch ? 1 : 0);
                }
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
        // No hard cap: chunkByBufMain and chunkByWgMain already prevent buffer
        // overflows. A 256-link cap was too conservative for small topologies
        // (e.g. ring-3 570 links → 3 GPU dispatches instead of 1, harming
        // GPU occupancy). Use 4096 as a generous safety ceiling only.
        chunkSize = std::min<uint32_t>(chunkSize, 4096u);

        // gen_spec_pow_kernel output is the complex per-port matrix
        // H[rx,tx,rb]: rxtx complex floats per (link, rb).
        const uint64_t sbBytesPerLink =
            (ltCanDispatch && sbNumRb > 0)
                ? uint64_t(sbNumRb) * ltUPorts * ltSPorts * sizeof(std::complex<float>)
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
        // Path B (skipLspCluster=true): cluster/link params are unchanged;
        // only update channelMatrix timestamps + rebuild LT entries from the
        // new longTermFlat (beam weights may have changed).  Skip the heavy
        // m_channelParamsMap rebuild (vector allocs, FindStrongestClusters,
        // PrecomputeAnglesSinCos, ray angle copies).
        {
            static const bool diagPop = []() {
                const char* e = std::getenv("MEZ_DIAG_H");
                return e && e[0] == '1';
            }();
            if (diagPop)
            {
                uint32_t nonZero = 0;
                int firstNzIdx = -1;
                for (uint32_t bi = 0; bi < nBucketLinks; ++bi)
                {
                    const auto& c = runtimeLinks[bucketLinkIdxs[bi]];
                    if (c.lspReadIdx < clusterParams.size() &&
                        clusterParams[c.lspReadIdx].nCluster > 0)
                    {
                        ++nonZero;
                        if (firstNzIdx < 0)
                            firstNzIdx = static_cast<int>(bi);
                    }
                }
                std::fprintf(stderr,
                             "[MEZ_DIAG_H] populate bucket(s=%u,u=%u): nLinks=%u "
                             "nonZeroClusters=%u firstNz=%d clusterParams=%zu\n",
                             bSSize,
                             bUSize,
                             nBucketLinks,
                             nonZero,
                             firstNzIdx,
                             clusterParams.size());
            }
        }
        if (skipLspCluster)
        {
            SLS_PHASE_SCOPE("Mez::Populate_LightB");
            const Time now_t = Simulator::Now();
            for (uint32_t bi = 0; bi < nBucketLinks; ++bi)
            {
                const auto& ctx = runtimeLinks[bucketLinkIdxs[bi]];
                // Update channelMatrix timestamp so TryGenSpecHit passes.
                auto matIt = m_channelMatrixMap.find(ctx.matrixKey);
                if (matIt != m_channelMatrixMap.end() && matIt->second)
                    matIt->second->m_generatedTime = now_t;
                // Rebuild LT entry from new GPU longTermFlat output.
                if (!longTermFlat.empty() && ctx.sAnt && ctx.uAnt)
                {
                    auto& existMat = matIt->second;
                    if (!existMat)
                        continue;
                    const size_t numOC = existMat->m_channel.GetNumPages();
                    const size_t ltPerPage = size_t(ltUPorts) * ltSPorts;
                    const size_t ltPerLink = ltPerPage * SlsChanWgpu::kMatMaxPages;
                    const size_t ltLinkBase = static_cast<size_t>(bi) * ltPerLink;
                    const size_t ltCells = ltPerPage * numOC;
                    if (ltLinkBase + ltCells > longTermFlat.size())
                        continue;
                    GpuLongTermEntry& entryRef = m_gpuLongTermMap[ctx.matrixKey];
                    Ptr<Complex3DVector> longTerm;
                    if (entryRef.longTerm &&
                        entryRef.longTerm->GetNumRows() == ltUPorts &&
                        entryRef.longTerm->GetNumCols() == ltSPorts &&
                        entryRef.longTerm->GetNumPages() == static_cast<uint16_t>(numOC))
                        longTerm = ConstCast<Complex3DVector>(entryRef.longTerm);
                    else
                        longTerm = Create<Complex3DVector>(
                            static_cast<uint16_t>(ltUPorts),
                            static_cast<uint16_t>(ltSPorts),
                            static_cast<uint16_t>(numOC));
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
                    entryRef.generatedTime = now_t;
                    entryRef.gpuLinkIdx = static_cast<uint32_t>(ctx.lspReadIdx);
                    entryRef.ltUPorts = ltUPorts;
                    entryRef.ltSPorts = ltSPorts;
                }
            }
        }
        else
        {
        SLS_PHASE_SCOPE("Mez::Populate");
        constexpr uint32_t numClusters_pop = SlsChanWgpu::kMatMaxPages;
        bucketDelaysFlat.assign(size_t(nBucketLinks) * numClusters_pop, 0.0f);
        bucketSpatialProj.assign(size_t(nBucketLinks) * numClusters_pop, 0.0);
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

                // Per-ray angle data is only populated when packedOutputs were
                // transferred. Normal path skips to save 8.2 MB GPU→CPU read.
                constexpr double kDeg2Rad = M_PI / 180.0;
                if (!phiNmAoaFlat.empty())
                {
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

                // Fill flat delay + spatial-projection arrays for fast DelayProj.
                {
                    const size_t bBase = size_t(bi) * numClusters_pop;
                    const size_t nc = std::min(params->m_delay.size(),
                                               size_t(numClusters_pop));
                    for (size_t c = 0; c < nc; ++c)
                        bucketDelaysFlat[bBase + c] = static_cast<float>(params->m_delay[c]);
                    const auto& cas = params->m_cachedAngleSincos;
                    if (cas.size() > ZOD_INDEX)
                    {
                        const size_t ncD = std::min(
                            {nc,
                             cas[ZOA_INDEX].size(), cas[ZOD_INDEX].size(),
                             cas[AOA_INDEX].size(), cas[AOD_INDEX].size()});
                        const Vector sSpd =
                            ctx.sMob ? ctx.sMob->GetVelocity() : Vector(0, 0, 0);
                        const Vector uSpd =
                            ctx.uMob ? ctx.uMob->GetVelocity() : Vector(0, 0, 0);
                        for (size_t c = 0; c < ncD; ++c)
                        {
                            bucketSpatialProj[bBase + c] =
                                cas[ZOA_INDEX][c].first  * cas[AOA_INDEX][c].second * uSpd.x +
                                cas[ZOA_INDEX][c].first  * cas[AOA_INDEX][c].first  * uSpd.y +
                                cas[ZOA_INDEX][c].second                             * uSpd.z +
                                cas[ZOD_INDEX][c].first  * cas[AOD_INDEX][c].second * sSpd.x +
                                cas[ZOD_INDEX][c].first  * cas[AOD_INDEX][c].first  * sSpd.y +
                                cas[ZOD_INDEX][c].second                             * sSpd.z;
                        }
                    }
                }

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
                {
                    static const bool diagLt3 = []() {
                        const char* e = std::getenv("MEZ_DIAG_H");
                        return e && e[0] == '1';
                    }();
                    if (diagLt3 && bi == 0)
                    {
                        std::fprintf(stderr,
                                     "[MEZ_DIAG_H] Populate lt-write probe bucket(s=%u,u=%u): "
                                     "ltFlatEmpty=%d sAnt=%d uAnt=%d\n",
                                     bSSize,
                                     bUSize,
                                     longTermFlat.empty() ? 1 : 0,
                                     ctx.sAnt != nullptr,
                                     ctx.uAnt != nullptr);
                    }
                }
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

            // M>1 slot lookahead: pre-compute the per-port H for M future
            // slots. Layout: slot s, link bi ->
            // m_specHFlat[s*nBucketLinks*numRb*rxtx + bi*numRb*rxtx].
            const uint32_t batchM = m_batchM;
            const double slotDurationSec = m_slotDuration.GetSeconds();
            const double slotTime = Simulator::Now().GetSeconds();

            m_reducedPowNumRb = numRb;
            // Grow the flat H buffer to hold M slots of complex per-port
            // matrices. CPU-miss (D-3b) entries live in m_specHCpuFlat,
            // so this array is exclusively batch-owned.
            //
            // Each BUCKET gets its own region starting at m_specHBucketBase
            // (a per-refresh running offset). The previous layout started
            // every bucket at offset 0, so with more than one antenna
            // bucket the later bucket silently clobbered the earlier
            // bucket's slabs while both buckets' entries carried fresh
            // generatedTime stamps -- evals on earlier-bucket links then
            // HIT and consumed another bucket's channel.
            const size_t rxtxSb = size_t(numRxPorts) * numTxPorts;
            const size_t bucketBase = m_specHBucketBase;
            const size_t bucketRegion = size_t(batchM) * nBucketLinks * numRb * rxtxSb;
            m_specHBucketBase += bucketRegion;
            if (m_specHFlat.size() < bucketBase + bucketRegion)
                m_specHFlat.resize(bucketBase + bucketRegion);
            // Scalar power arrays mirror the H layout without the rxtx factor.
            const size_t powBucketBase = m_powBucketBase;
            const size_t powBucketRegion = size_t(batchM) * nBucketLinks * numRb;
            m_powBucketBase += powBucketRegion;
            if (m_specPowFlat.size() < powBucketBase + powBucketRegion)
            {
                m_specPowFlat.resize(powBucketBase + powBucketRegion);
                m_specPowRevFlat.resize(powBucketBase + powBucketRegion);
            }

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
                    // Fast path: use pre-built flat arrays to avoid m_channelParamsMap
                    // pointer-chasing (~39 ms per bucket saved).
                    if (!bucketDelaysFlat.empty() &&
                        size_t(chunkStart + chunkLen) * numClusters <= bucketDelaysFlat.size())
                    {
                        const size_t cBytes = numClusters * sizeof(float);
                        const size_t spBytes = numClusters * sizeof(double);
                        for (uint32_t li = 0; li < chunkLen; ++li)
                        {
                            const size_t i       = size_t(chunkStart) + li;
                            const size_t bBase   = i * numClusters;
                            const size_t baseIdx = size_t(li) * numClusters;
                            std::memcpy(chunkDelays.data() + baseIdx,
                                        bucketDelaysFlat.data() + bBase, cBytes);
                            std::memcpy(chunkSpatialProj.data() + baseIdx,
                                        bucketSpatialProj.data() + bBase, spBytes);
                        }
                    }
                    else
                    {
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
                    } // end fallback

                    // ── Per-slot Doppler → dispatch (all M) → wait → readback ──
                    // Submit all M slot dispatches without stalling between them.
                    // Each slot writes to a distinct region of reduceBatchOutBuf_
                    // via out_offset.  A single waitForSpecBatch() + batch readback
                    // replaces the previous M×(waitIdle+readback) pattern.
                    chunkBatchH.resize(size_t(batchM) * chunkLen * numRb * rxtxSb);
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
                                              m_batchRbFreqs,
                                              s,
                                              batchM);
                        }
                    } // end slot dispatch loop
                    {
                        SLS_PHASE_SCOPE("Mez::SB::WaitIdle");
                        gpu->waitForSpecBatch();
                    }
                    {
                        SLS_PHASE_SCOPE("Mez::SB::Readback");
                        chunkBatchPow.resize(2 * size_t(batchM) * chunkLen * numRb);
                        gpu->readSpecHAndPowBatchInto(chunkLen,
                                                      numRb,
                                                      static_cast<uint32_t>(rxtxSb),
                                                      batchM,
                                                      chunkBatchH.data(),
                                                      chunkBatchPow.data());
                    }
                    // MEZ_DIAG_H: kernel-I/O consistency — recompute H for
                    // chunk link 0, rb 0 on the CPU from the exact buffers
                    // the kernel consumed and compare with the readback.
                    static const bool diagHsb = []() {
                        const char* e = std::getenv("MEZ_DIAG_H");
                        return e && e[0] == '1';
                    }();
                    if (diagHsb && chunkStart == 0 && !m_batchRbFreqs.empty())
                    {
                        const double fc0 = m_batchRbFreqs[0];
                        std::fprintf(stderr, "[MEZ_DIAG_H] specAB rb0 link0: ");
                        for (uint32_t sp = 0; sp < std::min<uint32_t>(numTxPorts, 4); ++sp)
                        {
                            for (uint32_t up = 0; up < std::min<uint32_t>(numRxPorts, 1); ++up)
                            {
                                std::complex<double> acc{0.0, 0.0};
                                for (uint32_t c = 0; c < numClusters; ++c)
                                {
                                    const double th =
                                        -2.0 * M_PI * fc0 * double(chunkDelays[c]);
                                    const std::complex<double> ds{std::cos(th), std::sin(th)};
                                    const std::complex<double> dop{
                                        double(chunkDoppler[c].real()),
                                        double(chunkDoppler[c].imag())};
                                    const auto lt = longTermFlat[size_t(c) * numRxPorts *
                                                                     numTxPorts +
                                                                 sp * numRxPorts + up];
                                    acc += std::complex<double>(lt.real(), lt.imag()) * ds *
                                           dop;
                                }
                                const auto g = chunkBatchH[size_t(0) * rxtxSb +
                                                           sp * numRxPorts + up];
                                std::fprintf(stderr,
                                             "sp%u gpu(%.3f,%.3f) ref(%.3f,%.3f) | ",
                                             sp,
                                             g.real(),
                                             g.imag(),
                                             acc.real(),
                                             acc.imag());
                            }
                        }
                        std::fprintf(stderr, "\n");
                    }

                    // Scatter chunkBatchH[s][li] → bucket region of m_specHFlat,
                    // and the scalar power halves into m_specPowFlat / RevFlat.
                    const size_t perLinkH = size_t(numRb) * rxtxSb;
                    const size_t powRevHalf = size_t(batchM) * chunkLen * numRb;
                    for (uint32_t s = 0; s < batchM; ++s)
                    {
                        std::memcpy(
                            m_specHFlat.data() + bucketBase +
                                size_t(s) * nBucketLinks * perLinkH +
                                size_t(chunkStart) * perLinkH,
                            chunkBatchH.data() + size_t(s) * chunkLen * perLinkH,
                            size_t(chunkLen) * perLinkH * sizeof(std::complex<float>));
                        std::memcpy(
                            m_specPowFlat.data() + powBucketBase +
                                size_t(s) * nBucketLinks * numRb +
                                size_t(chunkStart) * numRb,
                            chunkBatchPow.data() + size_t(s) * chunkLen * numRb,
                            size_t(chunkLen) * numRb * sizeof(float));
                        std::memcpy(
                            m_specPowRevFlat.data() + powBucketBase +
                                size_t(s) * nBucketLinks * numRb +
                                size_t(chunkStart) * numRb,
                            chunkBatchPow.data() + powRevHalf +
                                size_t(s) * chunkLen * numRb,
                            size_t(chunkLen) * numRb * sizeof(float));
                    }
                } // end DelayProj + slot loops scope
            } // end chunk loop

            // Record each link's metadata in the flat buffer index.
            const Time generatedTime = Simulator::Now();
            const uint32_t perSlotStride =
                nBucketLinks * numRb * static_cast<uint32_t>(rxtxSb);
            for (uint32_t bi = 0; bi < nBucketLinks; ++bi)
            {
                const auto& ctx = runtimeLinks[bucketLinkIdxs[bi]];
                // Beam-aware key: the batch computed H with the refresh-time
                // beam pair of this link's antennas; only evals carrying the
                // same beams may consume it. Evals with other beams miss and
                // are computed + cached per beam pair by the D-3b path.
                const uint64_t entryKey =
                    (ctx.sAnt && ctx.uAnt)
                        ? MixBeamKey(ctx.matrixKey,
                                     ctx.sAnt->GetBeamformingVectorHash(),
                                     ctx.uAnt->GetBeamformingVectorHash())
                        : MixBeamKey(ctx.matrixKey, 0, 0);
                GpuChanSpctEntry& eRef = m_gpuChanSpctMap[entryKey];
                eRef.matrixKey = ctx.matrixKey;
                eRef.specHBaseIdx = static_cast<uint32_t>(
                    bucketBase + size_t(bi) * numRb * rxtxSb); // slot-0 base in bucket region
                eRef.powBaseIdx = static_cast<uint32_t>(
                    powBucketBase + size_t(bi) * numRb);
                eRef.powPerSlotStride = nBucketLinks * numRb;
                eRef.numRxPorts = numRxPorts;
                eRef.numTxPorts = numTxPorts;
                eRef.numRb = numRb;
                eRef.generatedTime = generatedTime;
                eRef.rbFreqsHash = rbFreqsHash;
                eRef.batchM = batchM;
                eRef.batchStartTimeSec = slotTime;
                eRef.slotDurationSec = slotDurationSec;
                eRef.perSlotStride = perSlotStride;
                eRef.cpuSection = false; // batch-owned from this refresh on
            }
        }
        mezPrintMem("post-GenSpec");
        {
            static const bool diagLt2 = []() {
                const char* e = std::getenv("MEZ_DIAG_H");
                return e && e[0] == '1';
            }();
            if (diagLt2)
            {
                std::fprintf(stderr,
                             "[MEZ_DIAG_H] postBucket(s=%u,u=%u): ltMap=%zu chanSpctMap=%zu\n",
                             bSSize,
                             bUSize,
                             m_gpuLongTermMap.size(),
                             m_gpuChanSpctMap.size());
            }
        }
    } // end per-bucket loop

    // Spatial consistency bookkeeping: the next refresh may only drift a
    // link if its params came from the SAME grid slot this refresh.
    m_prevLspReadIdx.clear();
    for (const auto& ctx : runtimeLinks)
    {
        m_prevLspReadIdx[ctx.paramsKey] = ctx.lspReadIdx;
    }

    mezPrintMem("post-UpdateChannel");

    // ── MEZ_DIAG_H=1: per-stage power audit ───────────────────────────
    // Stage 1: per-link total power of the generated channel matrix,
    //   E_mat = sum_c sum_{u,s} |H[u,s,c]|^2 / (numU * numS).
    //   With TR 38.901 normalized cluster powers (sum_c P_c = 1) this
    //   should sit in a tight band around 0 dB for every link. A wide
    //   spread here localizes the fading over-dispersion to the
    //   cluster-ray / matrix kernels.
    // Stage 2: same metric for the GPU longTerm,
    //   E_lt = sum_c sum_{uP,sP} |lt[uP,sP,c]|^2 / (uP * sP).
    static const bool diagH = []() {
        const char* e = std::getenv("MEZ_DIAG_H");
        return e && e[0] == '1';
    }();
    if (diagH)
    {
        std::vector<double> matDb;
        for (const auto& [key, mat] : m_channelMatrixMap)
        {
            const auto& ch = mat->m_channel;
            const size_t n = ch.GetNumRows() * ch.GetNumCols() * ch.GetNumPages();
            if (n == 0)
                continue;
            double acc = 0.0;
            const auto& vals = ch.GetValues();
            for (size_t i = 0; i < n; ++i)
                acc += std::norm(vals[i]);
            const double e = acc / double(ch.GetNumRows() * ch.GetNumCols());
            if (e > 0)
                matDb.push_back(10.0 * std::log10(e));
        }
        std::vector<double> ltDb;
        for (const auto& [key, lte] : m_gpuLongTermMap)
        {
            if (!lte.longTerm)
                continue;
            const auto& lt = *lte.longTerm;
            const size_t n = lt.GetNumRows() * lt.GetNumCols() * lt.GetNumPages();
            if (n == 0)
                continue;
            double acc = 0.0;
            const auto& vals = lt.GetValues();
            for (size_t i = 0; i < n; ++i)
                acc += std::norm(vals[i]);
            const double e = acc / double(lt.GetNumRows() * lt.GetNumCols());
            if (e > 0)
                ltDb.push_back(10.0 * std::log10(e));
        }
        auto dump = [](const char* tag, std::vector<double>& v) {
            if (v.empty())
                return;
            std::sort(v.begin(), v.end());
            const double mean =
                std::accumulate(v.begin(), v.end(), 0.0) / double(v.size());
            double var = 0.0;
            for (double x : v)
                var += (x - mean) * (x - mean);
            var /= double(v.size());
            std::fprintf(stderr,
                         "[MEZ_DIAG_H] %s n=%zu mean=%.2f dB std=%.2f dB "
                         "p5=%.2f p50=%.2f p95=%.2f\n",
                         tag,
                         v.size(),
                         mean,
                         std::sqrt(var),
                         v[v.size() / 20],
                         v[v.size() / 2],
                         v[v.size() * 19 / 20]);
        };
        dump("matrixPower", matDb);
        dump("longTermPower", ltDb);

        // Effective azimuth pattern: per-link mean element power vs
        // off-boresight angle. With the 3GPP element this should track
        // 8 - min(12*(rel/65)^2, 30) dB plus fading noise; a steeper or
        // offset curve localizes a pattern/rotation bug without needing
        // the CPU reference run.
        std::fprintf(stderr, "[MEZ_DIAG_H] patternCurve (relDeg,powDb):");
        int pcN = 0;
        for (const auto& ctx : runtimeLinks)
        {
            if (pcN >= 24)
                break;
            auto mit2 = m_channelMatrixMap.find(ctx.matrixKey);
            if (mit2 == m_channelMatrixMap.end() || !mit2->second || !ctx.sAnt || !ctx.sMob ||
                !ctx.uMob)
                continue;
            const auto& ch = mit2->second->m_channel;
            const size_t n = ch.GetNumRows() * ch.GetNumCols() * ch.GetNumPages();
            if (n == 0)
                continue;
            double acc = 0.0;
            const auto& vals = ch.GetValues();
            for (size_t i = 0; i < n; ++i)
                acc += std::norm(vals[i]);
            const double powDb =
                10.0 * std::log10(acc / double(ch.GetNumRows() * ch.GetNumCols()));
            auto upa = DynamicCast<UniformPlanarArray>(ctx.sAnt);
            if (!upa)
                continue;
            const Vector sp = ctx.sMob->GetPosition();
            const Vector up = ctx.uMob->GetPosition();
            const double az = std::atan2(up.y - sp.y, up.x - sp.x) * 180.0 / M_PI;
            const double bear = upa->GetAlpha() * 180.0 / M_PI;
            const double rel = std::fabs(std::fmod(az - bear + 540.0, 360.0) - 180.0);
            std::fprintf(stderr, " (%.0f,%.1f)", rel, powDb);
            ++pcN;
        }
        std::fprintf(stderr, "\n");

        // Stage 2b: element-wise GPU-vs-CPU longTerm comparison for the
        // first few runtime links. Replicates PRX::CalculateLongTermComponent
        // (sub-array port partition) on the ns-3 channel matrix with the
        // antennas' current beam weights and compares against the GPU lt.
        std::fprintf(stderr,
                     "[MEZ_DIAG_H] ltAB precheck: runtimeLinks=%zu ltMap=%zu matMap=%zu\n",
                     runtimeLinks.size(),
                     m_gpuLongTermMap.size(),
                     m_channelMatrixMap.size());
        int audited = 0;
        for (const auto& ctx : runtimeLinks)
        {
            if (audited >= 4)
                break;
            auto lit = m_gpuLongTermMap.find(ctx.matrixKey);
            auto mit = m_channelMatrixMap.find(ctx.matrixKey);
            if (lit == m_gpuLongTermMap.end() || mit == m_channelMatrixMap.end() ||
                !lit->second.longTerm || !ctx.sAnt || !ctx.uAnt)
            {
                if (audited == 0)
                {
                    std::fprintf(stderr,
                                 "[MEZ_DIAG_H] ltAB skip link(cid=%u,uid=%u): ltFound=%d "
                                 "matFound=%d ltPtr=%d sAnt=%d uAnt=%d\n",
                                 ctx.cid,
                                 ctx.uid,
                                 lit != m_gpuLongTermMap.end(),
                                 mit != m_channelMatrixMap.end(),
                                 lit != m_gpuLongTermMap.end() && lit->second.longTerm,
                                 ctx.sAnt != nullptr,
                                 ctx.uAnt != nullptr);
                }
                continue;
            }
            const auto& gpuLt = *lit->second.longTerm;
            const auto& ch = mit->second->m_channel;
            const auto& sW = ctx.sAnt->GetBeamformingVectorRef();
            const auto& uW = ctx.uAnt->GetBeamformingVectorRef();
            const size_t sPorts = ctx.sAnt->GetNumPorts();
            const size_t uPorts = ctx.uAnt->GetNumPorts();
            const size_t sPortElems = ctx.sAnt->GetNumElemsPerPort();
            const size_t uPortElems = ctx.uAnt->GetNumElemsPerPort();
            const size_t sElemsPerPort = ctx.sAnt->GetHElemsPerPort();
            const size_t uElemsPerPort = ctx.uAnt->GetHElemsPerPort();
            const size_t nPages = std::min<size_t>(ch.GetNumPages(), gpuLt.GetNumPages());
            double gpuPow = 0.0;
            double cpuPow = 0.0;
            double errPow = 0.0;
            for (size_t c = 0; c < nPages; ++c)
            {
                for (size_t sp = 0; sp < sPorts; ++sp)
                {
                    for (size_t up = 0; up < uPorts; ++up)
                    {
                        const auto startS = ctx.sAnt->ArrayIndexFromPortIndex(sp, 0);
                        const auto startU = ctx.uAnt->ArrayIndexFromPortIndex(up, 0);
                        std::complex<double> txSum{0.0, 0.0};
                        size_t sIndex = startS;
                        for (size_t t = 0; t < sPortElems; ++t)
                        {
                            std::complex<double> rxSum{0.0, 0.0};
                            size_t uIndex = startU;
                            for (size_t r = 0; r < uPortElems; ++r)
                            {
                                rxSum += std::conj(uW[uIndex - startU]) *
                                         ch(uIndex, sIndex, c);
                                ++uIndex;
                                if (uElemsPerPort > 0 &&
                                    (r % uElemsPerPort) == uElemsPerPort - 1)
                                    uIndex += ctx.uAnt->GetNumColumns() - uElemsPerPort;
                            }
                            txSum += sW[sIndex - startS] * rxSum;
                            ++sIndex;
                            if (sElemsPerPort > 0 &&
                                (t % sElemsPerPort) == sElemsPerPort - 1)
                                sIndex += ctx.sAnt->GetNumColumns() - sElemsPerPort;
                        }
                        const auto g = gpuLt(up, sp, c);
                        gpuPow += std::norm(g);
                        cpuPow += std::norm(txSum);
                        errPow += std::norm(std::complex<double>(g.real(), g.imag()) - txSum);
                    }
                }
            }
            // Stage 3: cached spec-H band power for this link vs the
            // power implied by its own longTerm. The coherent port sum
            // sum_s lt[c,0,s] with random dt phases gives a band-average
            //   E[sum_rb |sum_s H|^2 / numRb] = sum_c |sum_s lt[c,s]|^2.
            // A per-link mismatch here (while lt matches the CPU) means
            // the eval entry is wired to ANOTHER link's H slab.
            double hBandPow = -1.0;
            double ltCohPow = 0.0;
            const uint64_t auditKey = MixBeamKey(ctx.matrixKey,
                                                 ctx.sAnt->GetBeamformingVectorHash(),
                                                 ctx.uAnt->GetBeamformingVectorHash());
            auto eIt = m_gpuChanSpctMap.find(auditKey);
            if (eIt != m_gpuChanSpctMap.end() && !eIt->second.cpuSection &&
                eIt->second.numRb > 0)
            {
                const auto& e = eIt->second;
                const size_t rxtxE = size_t(e.numRxPorts) * e.numTxPorts;
                const std::complex<float>* hb =
                    m_specHFlat.data() + e.specHBaseIdx; // slot 0
                double acc = 0.0;
                for (size_t rb = 0; rb < e.numRb; ++rb)
                {
                    for (size_t up = 0; up < e.numRxPorts; ++up)
                    {
                        std::complex<double> rowSum{0.0, 0.0};
                        for (size_t sp = 0; sp < e.numTxPorts; ++sp)
                        {
                            const auto h = hb[rb * rxtxE + sp * e.numRxPorts + up];
                            rowSum += std::complex<double>(h.real(), h.imag());
                        }
                        acc += std::norm(rowSum);
                    }
                }
                hBandPow = acc / double(e.numRb);
                for (size_t c = 0; c < gpuLt.GetNumPages(); ++c)
                {
                    for (size_t up = 0; up < uPorts; ++up)
                    {
                        std::complex<double> rowSum{0.0, 0.0};
                        for (size_t sp = 0; sp < sPorts; ++sp)
                        {
                            const auto g = gpuLt(up, sp, c);
                            rowSum += std::complex<double>(g.real(), g.imag());
                        }
                        ltCohPow += std::norm(rowSum);
                    }
                }
            }
            std::fprintf(stderr,
                         "[MEZ_DIAG_H] ltAB link(cid=%u,uid=%u) pages=%zu "
                         "powGpu=%.3f powCpu=%.3f relErr=%.4f ratioDb=%.2f "
                         "| hBand=%.3f ltCoh=%.3f h/lt=%.2f dB\n",
                         ctx.cid,
                         ctx.uid,
                         nPages,
                         gpuPow,
                         cpuPow,
                         cpuPow > 0 ? std::sqrt(errPow / cpuPow) : -1.0,
                         (gpuPow > 0 && cpuPow > 0)
                             ? 10.0 * std::log10(gpuPow / cpuPow)
                             : 0.0,
                         hBandPow,
                         ltCohPow,
                         (hBandPow > 0 && ltCohPow > 0)
                             ? 10.0 * std::log10(hBandPow / ltCohPow)
                             : -99.0);
            if (audited == 0)
            {
                // Spatial-structure check: per-cluster power profile and the
                // per-element phase progression of cluster 0. A LOS-dominated
                // link must show (a) cluster 0 carrying most of the power
                // (K-factor) and (b) a smooth phase ramp across elements
                // (steering vector) — that ramp is what beamforming gain
                // aligns with. Correct total power with a scrambled ramp
                // means no BF gain: matches RSRP low / SINR low symptoms.
                std::fprintf(stderr, "[MEZ_DIAG_H] clusterPow: ");
                for (size_t c = 0; c < std::min<size_t>(ch.GetNumPages(), 8); ++c)
                {
                    double pc = 0.0;
                    for (size_t u = 0; u < ch.GetNumRows(); ++u)
                        for (size_t s = 0; s < ch.GetNumCols(); ++s)
                            pc += std::norm(ch(u, s, c));
                    std::fprintf(stderr, "c%zu=%.3f ", c, pc);
                }
                std::fprintf(stderr, "\n[MEZ_DIAG_H] c0 elem phase/mag: ");
                for (size_t s = 0; s < std::min<size_t>(ch.GetNumCols(), 8); ++s)
                {
                    const auto h = ch(0, s, 0);
                    std::fprintf(stderr,
                                 "s%zu(%.2f rad,%.3f) ",
                                 s,
                                 std::arg(h),
                                 std::abs(h));
                }
                std::fprintf(stderr, "\n");
                // Per-cluster element dump for the first link, port (0,0):
                // shape of the divergence (permuted pages? scaled? unrelated?)
                std::fprintf(stderr, "[MEZ_DIAG_H] ltAB detail (up=0,sp=0): ");
                for (size_t c = 0; c < std::min<size_t>(nPages, 6); ++c)
                {
                    const auto g = gpuLt(0, 0, c);
                    // recompute cpu lt for (0,0,c)
                    const auto startS0 = ctx.sAnt->ArrayIndexFromPortIndex(0, 0);
                    const auto startU0 = ctx.uAnt->ArrayIndexFromPortIndex(0, 0);
                    std::complex<double> txSum{0.0, 0.0};
                    size_t sIndex = startS0;
                    for (size_t t = 0; t < sPortElems; ++t)
                    {
                        std::complex<double> rxSum{0.0, 0.0};
                        size_t uIndex = startU0;
                        for (size_t r = 0; r < uPortElems; ++r)
                        {
                            rxSum += std::conj(uW[uIndex - startU0]) * ch(uIndex, sIndex, c);
                            ++uIndex;
                            if (uElemsPerPort > 0 && (r % uElemsPerPort) == uElemsPerPort - 1)
                                uIndex += ctx.uAnt->GetNumColumns() - uElemsPerPort;
                        }
                        txSum += sW[sIndex - startS0] * rxSum;
                        ++sIndex;
                        if (sElemsPerPort > 0 && (t % sElemsPerPort) == sElemsPerPort - 1)
                            sIndex += ctx.sAnt->GetNumColumns() - sElemsPerPort;
                    }
                    std::fprintf(stderr,
                                 "c%zu gpu(%.3f,%.3f) cpu(%.3f,%.3f) | ",
                                 c,
                                 g.real(),
                                 g.imag(),
                                 txSum.real(),
                                 txSum.imag());
                }
                std::fprintf(stderr,
                             "\n[MEZ_DIAG_H] geom: sPorts=%zu uPorts=%zu sPortElems=%zu "
                             "uPortElems=%zu sElemsPerPort=%zu sCols=%u sW.size=%zu "
                             "sW[0]=(%.4f,%.4f) sW[1]=(%.4f,%.4f) matPages=%zu ltPages=%zu\n",
                             sPorts,
                             uPorts,
                             sPortElems,
                             uPortElems,
                             sElemsPerPort,
                             static_cast<unsigned>(ctx.sAnt->GetNumColumns()),
                             static_cast<size_t>(sW.GetSize()),
                             sW[0].real(),
                             sW[0].imag(),
                             sW.GetSize() > 1 ? sW[1].real() : 0.0,
                             sW.GetSize() > 1 ? sW[1].imag() : 0.0,
                             static_cast<size_t>(ch.GetNumPages()),
                             static_cast<size_t>(gpuLt.GetNumPages()));
            }
            ++audited;
        }
    }

    NS_LOG_DEBUG("Updated GPU-backed channel params + matrix caches.");
}

void
ThreeGppChannelModelWgpuMezanine::SetBypassGpuBatch(bool bypass)
{
    m_bypassGpuBatch = bypass;
}

std::vector<double>
ThreeGppChannelModelWgpuMezanine::ComputeUncachedBatch(
    const std::vector<UncachedBatchLink>& links,
    Ptr<const SpectrumValue> txPsd)
{
    if (links.empty())
        return {};

    // Ensure RB frequencies are captured (needed by genSpecBatch).
    CaptureRbFreqs(txPsd);
    if (m_batchRbFreqs.empty())
        return {};

    const uint32_t numRb = static_cast<uint32_t>(m_batchRbFreqs.size());
    const uint32_t nLinks = static_cast<uint32_t>(links.size());

    // Step 1: Ensure channel matrices and params are available via base-class
    // CPU path. GetChannel populates m_channelMatrixMap + m_channelParamsMap
    // on first call; subsequent calls return the cached entry.
    for (const auto& lnk : links)
        ThreeGppChannelModel::GetChannel(lnk.sMob, lnk.uMob, lnk.sAnt, lnk.uAnt);

    // All links must share the same (sPorts, uPorts) configuration for a
    // single-bucket genSpecBatch dispatch. Use the first link as reference.
    const Ptr<const PhasedArrayModel>& refS = links[0].sAnt;
    const Ptr<const PhasedArrayModel>& refU = links[0].uAnt;
    if (!refS || !refU)
        return {};

    const uint32_t ltSPorts = static_cast<uint32_t>(refS->GetNumPorts());
    const uint32_t ltUPorts = static_cast<uint32_t>(refU->GetNumPorts());
    const uint32_t ltSPortElems = static_cast<uint32_t>(refS->GetNumElemsPerPort());
    const uint32_t ltUPortElems = static_cast<uint32_t>(refU->GetNumElemsPerPort());
    const uint32_t ltSElemsPerPort = static_cast<uint32_t>(refS->GetHElemsPerPort());
    const uint32_t ltUElemsPerPort = static_cast<uint32_t>(refU->GetHElemsPerPort());
    const uint32_t ltSIncVal = (ltSElemsPerPort > 0)
                                   ? static_cast<uint32_t>(refS->GetNumColumns()) - ltSElemsPerPort
                                   : 0u;
    const uint32_t ltUIncVal = (ltUElemsPerPort > 0)
                                   ? static_cast<uint32_t>(refU->GetNumColumns()) - ltUElemsPerPort
                                   : 0u;

    // Step 2: Compute longTerms on CPU for all links and collect per-link delays.
    // Mirrors PRX::CalcLongTerm exactly (same index arithmetic, same port walk).
    // Uses float32 throughout for speed since channel matrices are ~12-20 clusters.
    const uint32_t numClusters = SlsChanWgpu::kMatMaxPages;
    const size_t perLinkLT = static_cast<size_t>(ltUPorts) * ltSPorts * numClusters;

    static thread_local std::vector<std::complex<float>> ltFlat;
    static thread_local std::vector<float> delaysFlat;
    static thread_local std::vector<float> sWRe_buf, sWIm_buf, cuWRe_buf, cuWIm_buf;
    ltFlat.assign(nLinks * perLinkLT, {0.0f, 0.0f});
    delaysFlat.assign(nLinks * numClusters, 0.0f);

    for (uint32_t li = 0; li < nLinks; ++li)
    {
        const auto& lnk = links[li];
        const uint64_t matKey = GetKey(lnk.sAnt->GetId(), lnk.uAnt->GetId());
        auto matIt = m_channelMatrixMap.find(matKey);
        if (matIt == m_channelMatrixMap.end())
            continue;
        const Ptr<const ChannelMatrix>& matrix = matIt->second;

        // Collect delays from cached params.
        const uint64_t paramsKey =
            GetKey(lnk.sMob->GetObject<Node>()->GetId(),
                   lnk.uMob->GetObject<Node>()->GetId());
        if (auto pit = m_channelParamsMap.find(paramsKey); pit != m_channelParamsMap.end())
        {
            const auto& p = pit->second;
            const size_t nc = std::min(static_cast<size_t>(numClusters), p->m_delay.size());
            float* dBase = delaysFlat.data() + static_cast<size_t>(li) * numClusters;
            for (size_t c = 0; c < nc; ++c)
                dBase[c] = static_cast<float>(p->m_delay[c]);
        }

        // Cache per-port element weights as float (re,im) pairs.
        const auto& sW = lnk.sAnt->GetBeamformingVectorRef();
        const auto& uW = lnk.uAnt->GetBeamformingVectorRef();
        sWRe_buf.resize(ltSPortElems);
        sWIm_buf.resize(ltSPortElems);
        cuWRe_buf.resize(ltUPortElems);
        cuWIm_buf.resize(ltUPortElems);
        for (uint32_t k = 0; k < ltSPortElems; ++k)
        {
            sWRe_buf[k] = static_cast<float>(sW[k].real());
            sWIm_buf[k] = static_cast<float>(sW[k].imag());
        }
        for (uint32_t k = 0; k < ltUPortElems; ++k)
        {
            cuWRe_buf[k] = static_cast<float>(std::conj(uW[k]).real());
            cuWIm_buf[k] = static_cast<float>(std::conj(uW[k]).imag());
        }

        const size_t numRows = matrix->m_channel.GetNumRows();
        const size_t nC =
            std::min(static_cast<size_t>(numClusters), matrix->m_channel.GetNumPages());
        std::complex<float>* ltBase = ltFlat.data() + static_cast<size_t>(li) * perLinkLT;
        const bool uTrivial = (ltUPortElems == 1);
        const bool sNeedHop = (ltSElemsPerPort > 0 && ltSIncVal > 0);
        const bool uNeedHop = (ltUElemsPerPort > 0 && ltUIncVal > 0);

        for (uint32_t spIdx = 0; spIdx < ltSPorts; ++spIdx)
        {
            const uint32_t startS =
                static_cast<uint32_t>(lnk.sAnt->ArrayIndexFromPortIndex(spIdx, 0));
            for (uint32_t upIdx = 0; upIdx < ltUPorts; ++upIdx)
            {
                const uint32_t startU =
                    static_cast<uint32_t>(lnk.uAnt->ArrayIndexFromPortIndex(upIdx, 0));
                for (size_t c = 0; c < nC; ++c)
                {
                    const auto* pageD =
                        reinterpret_cast<const double*>(matrix->m_channel.GetPagePtr(c));
                    float txRe = 0.0f, txIm = 0.0f;
                    uint32_t sIdx = startS;
                    if (uTrivial)
                    {
                        const float cu0Re = cuWRe_buf[0];
                        const float cu0Im = cuWIm_buf[0];
                        for (uint32_t tI = 0; tI < ltSPortElems; ++tI, ++sIdx)
                        {
                            const size_t cell = (startU + numRows * sIdx) * 2;
                            const float pR = static_cast<float>(pageD[cell]);
                            const float pI = static_cast<float>(pageD[cell + 1]);
                            const float rxR = cu0Re * pR - cu0Im * pI;
                            const float rxI = cu0Re * pI + cu0Im * pR;
                            txRe += sWRe_buf[tI] * rxR - sWIm_buf[tI] * rxI;
                            txIm += sWRe_buf[tI] * rxI + sWIm_buf[tI] * rxR;
                            if (sNeedHop && tI % ltSElemsPerPort == ltSElemsPerPort - 1)
                                sIdx += ltSIncVal;
                        }
                    }
                    else
                    {
                        for (uint32_t tI = 0; tI < ltSPortElems; ++tI, ++sIdx)
                        {
                            float rxRe = 0.0f, rxIm = 0.0f;
                            uint32_t uIdx = startU;
                            for (uint32_t rI = 0; rI < ltUPortElems; ++rI, ++uIdx)
                            {
                                const size_t cell = (uIdx + numRows * sIdx) * 2;
                                const float pR = static_cast<float>(pageD[cell]);
                                const float pI = static_cast<float>(pageD[cell + 1]);
                                rxRe += cuWRe_buf[rI] * pR - cuWIm_buf[rI] * pI;
                                rxIm += cuWRe_buf[rI] * pI + cuWIm_buf[rI] * pR;
                                if (uNeedHop && rI % ltUElemsPerPort == ltUElemsPerPort - 1)
                                    uIdx += ltUIncVal;
                            }
                            txRe += sWRe_buf[tI] * rxRe - sWIm_buf[tI] * rxIm;
                            txIm += sWRe_buf[tI] * rxIm + sWIm_buf[tI] * rxRe;
                            if (sNeedHop && tI % ltSElemsPerPort == ltSElemsPerPort - 1)
                                sIdx += ltSIncVal;
                        }
                    }
                    // Column-major layout: (uPort, sPort, c) at
                    //   uPort + ltUPorts*sPort + ltUPorts*ltSPorts*c
                    ltBase[upIdx + ltUPorts * spIdx + ltUPorts * ltSPorts * c] = {txRe, txIm};
                }
            }
        }
    }

    // Step 3: Upload longTerms to GPU and run genSpecBatch in chunks.
    // Each chunk is bounded by kMaxStagingReadbackBytes to keep MapAsync fast.
    SlsChanWgpu* gpu = m_wgpuChannel.get();
    static constexpr uint64_t kMaxReadbackBytes = 786432ull; // 768 KB
    const uint64_t sbBytesPerLink = static_cast<uint64_t>(numRb) * sizeof(float);
    const uint32_t chunkSz = static_cast<uint32_t>(
        std::max(1ull,
                 std::min<uint64_t>(static_cast<uint64_t>(nLinks),
                                    kMaxReadbackBytes / sbBytesPerLink)));

    static thread_local std::vector<float> reducedPowFlat;
    static thread_local std::vector<float> chunkDelays;
    static thread_local std::vector<std::complex<float>> chunkDoppler;
    reducedPowFlat.resize(static_cast<size_t>(nLinks) * numRb, 0.0f);

    for (uint32_t cs = 0; cs < nLinks; cs += chunkSz)
    {
        const uint32_t cl = std::min(chunkSz, nLinks - cs);

        gpu->uploadLongTermBatch(ltFlat.data() + static_cast<size_t>(cs) * perLinkLT,
                                 cl, ltSPorts, ltUPorts);

        chunkDelays.assign(static_cast<size_t>(cl) * numClusters, 0.0f);
        std::memcpy(chunkDelays.data(),
                    delaysFlat.data() + static_cast<size_t>(cs) * numClusters,
                    static_cast<size_t>(cl) * numClusters * sizeof(float));
        // No Doppler shift assumed (initial attachment or static channel).
        chunkDoppler.assign(static_cast<size_t>(cl) * numClusters, {1.0f, 0.0f});

        gpu->genSpecBatch(cl, numClusters, numRb,
                          ltUPorts, ltSPorts, ltUPorts, ltSPorts,
                          chunkDelays, chunkDoppler, m_batchRbFreqs,
                          /*outSlotOffset=*/0, /*totalBatchSlots=*/1);
        gpu->waitForSpecBatch();
        // The kernel now outputs the per-port complex H; reduce it to the
        // forward beamformed power sum_rx |sum_tx H|^2 on the CPU.
        const uint32_t rxtxUb = ltUPorts * ltSPorts;
        static thread_local std::vector<std::complex<float>> hScratch;
        hScratch.resize(static_cast<size_t>(cl) * numRb * rxtxUb);
        gpu->readSpecHBatchInto(cl, numRb, rxtxUb, /*nSlots=*/1, hScratch.data());
        for (uint32_t li = 0; li < cl; ++li)
        {
            for (uint32_t rb = 0; rb < numRb; ++rb)
            {
                const std::complex<float>* src =
                    hScratch.data() + (static_cast<size_t>(li) * numRb + rb) * rxtxUb;
                float acc = 0.0f;
                for (uint32_t rx = 0; rx < ltUPorts; ++rx)
                {
                    std::complex<float> rowSum{0.0f, 0.0f};
                    for (uint32_t tx = 0; tx < ltSPorts; ++tx)
                    {
                        rowSum += src[tx * ltUPorts + rx];
                    }
                    acc += std::norm(rowSum);
                }
                reducedPowFlat[(static_cast<size_t>(cs) + li) * numRb + rb] = acc;
            }
        }
    }

    // Step 4: Sum reducedPow across RBs → isotropic received power per link.
    std::vector<double> result(nLinks, 0.0);
    for (uint32_t li = 0; li < nLinks; ++li)
    {
        const float* rp = reducedPowFlat.data() + static_cast<size_t>(li) * numRb;
        double s = 0.0;
        for (uint32_t rb = 0; rb < numRb; ++rb)
            s += static_cast<double>(rp[rb]);
        result[li] = s;
    }
    return result;
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
    // Bypass GPU batch dispatch during initial-attachment / REM-map phases.
    // Base-class LSP draw above still runs so GetChannel stays functional.
    if (m_bypassGpuBatch)
        return;
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
        }
        else ++miss_no_par_entry;
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
        // Spec entries are keyed per (link, beam pair); erase every beam
        // variant of this link. Rare path (stale-matrix invalidation).
        for (auto it = m_gpuChanSpctMap.begin(); it != m_gpuChanSpctMap.end();)
            it = (it->second.matrixKey == matrixKey) ? m_gpuChanSpctMap.erase(it)
                                                     : std::next(it);
    }

    // Fallback: CPU build (first tick before UpdateChannel has run,
    // or any link the GPU pipeline didn't see).
    ++gnc_cpu_fallback;
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
    bool isReverse,
    uint64_t sWBeamHash,
    uint64_t uWBeamHash,
    bool scalarPsdOk) const
{
    // Initial-attachment / REM-map bypass: skip all GPU cache work and fall
    // through to the CPU path so temporary antenna objects never pollute the
    // batch cache with stale entries for dimensions that won't match later.
    if (m_bypassGpuBatch)
        return nullptr;

    // Batched per-tick path: if UpdateChannel has already run
    // gen_spec_batch_kernel for this link with the current rb_freqs,
    // the chanSpct_unscaled is cached in m_gpuChanSpctMap and the
    // per-eval cost collapses to a hash lookup + per-PRB
    // sqrt(PSD[rb]) scale. Otherwise we either capture rb_freqs from
    // inPsd for the NEXT tick to pre-compute, or fall back to CPU /
    // the per-eval GPU dispatch below.
    if (channelMatrix)
    {
        // Beam-aware key: entries are valid only for the exact (sW, uW)
        // beam pair they were computed with. gNB beams change per
        // scheduled UE within an update period; serving a refresh-time
        // beam to all evals silently destroyed the beamforming gain
        // structure (serving links lost gain, interference gained).
        const uint64_t key = MixBeamKey(
            MatrixBasedChannelModel::GetKey(channelMatrix->m_antennaPair.first,
                                            channelMatrix->m_antennaPair.second),
            sWBeamHash,
            uWBeamHash);
        auto it = m_gpuChanSpctMap.find(key);
        if (it != m_gpuChanSpctMap.end())
        {
            GpuChanSpctEntry& e = it->second;
            // m_batchRbFreqsHash is captured atomically with
            // m_batchRbFreqs (see below) and the array is write-once,
            // so comparing the stored hash avoids re-running FNV over
            // 273 floats on every PRX eval.
            //
            // Orientation: entries are stored in DL orientation
            // (numRxPorts = UE ports, numTxPorts = gNB ports). A forward
            // eval must match the stored ports directly and reads the
            // forward reduction; a reverse (UL) eval arrives with the
            // ports swapped and reads the reverse reduction
            // sum_tx |sum_rx H|^2 computed by the same kernel pass.
            const bool fwdOk = !isReverse && e.numRxPorts == numRxPorts &&
                               e.numTxPorts == numTxPorts;
            const bool revOk = isReverse && e.numRxPorts == numTxPorts &&
                               e.numTxPorts == numRxPorts;
            if (e.generatedTime == channelMatrix->m_generatedTime &&
                (fwdOk || revOk) &&
                e.numRb == numRb && e.rbFreqsHash == m_batchRbFreqsHash)
            {
                SLS_PHASE_SCOPE("Mez::TryGenSpecHit");
                // GPU reduction kernel computed:
                //   reducedPow[link,rb]    = sum_rx |sum_tx H_unscaled[rx,tx,rb]|^2
                //   reducedPowRev[link,rb] = sum_tx |sum_rx H_unscaled[rx,tx,rb]|^2
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
                // The batch kernel cached the complex per-port matrix
                // H[rx,tx,rb] in canonical DL orientation with the
                // Complex3DVector page layout. Return the caller-oriented
                // matrix scaled by sqrt(inPsd[rb]) -- the same quantity
                // PRX's CPU GenSpec produces -- so both PsdReduction and
                // NrInterference's MIMO covariance see the real phased
                // per-port channel (a scalar power cache skewed SINR by
                // 16-19 dB while RSRP still matched).
                Ptr<Complex3DVector>& fakeSlot = revOk ? e.fakeChanSpctRev : e.fakeChanSpct;
                if (!fakeSlot || fakeSlot->GetNumPages() != numRb ||
                    fakeSlot->GetNumRows() != numRxPorts ||
                    fakeSlot->GetNumCols() != numTxPorts)
                    fakeSlot = Create<Complex3DVector>(numRxPorts, numTxPorts,
                                                       static_cast<uint16_t>(numRb));
                Ptr<Complex3DVector> chanSpct = fakeSlot;

                // M>1 slot lookup: find the pre-computed slot whose time
                // best matches Simulator::Now(). Slot 0 is batchStartTimeSec;
                // each subsequent slot covers one slotDurationSec window.
                // CPU-miss (D-3b) entries always have batchM == 1.
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
                // Scalar fast path: with one RX port and no precoding the
                // interference covariance is a scalar, so the power-only
                // (1,1) matrix is exact and skips the per-port copy.
                // Delivered power matches the full-H path exactly:
                //   psd[rb] = pow[rb] * inPsd[rb] / numTxCaller.
                if (scalarPsdOk && numRxPorts == 1)
                {
                    const float* powBase;
                    if (e.cpuSection)
                    {
                        powBase = (revOk ? m_specPowCpuRevFlat : m_specPowCpuFlat).data() +
                                  e.powBaseIdx;
                    }
                    else
                    {
                        powBase = (revOk ? m_specPowRevFlat : m_specPowFlat).data() +
                                  e.powBaseIdx + size_t(slotIdx) * e.powPerSlotStride;
                    }
                    Ptr<Complex3DVector>& scalarSlot =
                        revOk ? e.fakeScalarRev : e.fakeScalar;
                    if (!scalarSlot || scalarSlot->GetNumPages() != numRb)
                        scalarSlot = Create<Complex3DVector>(
                            1, 1, static_cast<uint16_t>(numRb));
                    const double invTx = 1.0 / static_cast<double>(numTxPorts);
                    auto psdItS = inPsd->ConstValuesBegin();
                    auto* page0 =
                        reinterpret_cast<double*>(scalarSlot->GetPagePtr(0));
                    for (size_t rb = 0; rb < numRb; ++rb, ++psdItS)
                    {
                        const double psd =
                            static_cast<double>(powBase[rb]) * (*psdItS) * invTx;
                        page0[rb * 2] = psd > 0.0 ? std::sqrt(psd) : 0.0;
                        page0[rb * 2 + 1] = 0.0;
                    }
                    return scalarSlot;
                }

                const std::complex<float>* hBase;
                if (e.cpuSection)
                {
                    hBase = m_specHCpuFlat.data() + e.specHBaseIdx;
                }
                else
                {
                    hBase = m_specHFlat.data() + e.specHBaseIdx +
                            size_t(slotIdx) * e.perSlotStride;
                }
                // Deliver exactly H * sqrt(inPsd[rb]) -- the same quantity
                // the CPU GenSpec contraction returns -- and let
                // PsdReduction apply its own 1/numTx normalization, just as
                // it does for CPU-produced matrices. No empirical scale
                // factors: the old scalar-power path delivered
                // pow*inPsd WITHOUT the 1/numTx for D-3b-cached far links
                // (numTx x too hot), which skewed interference and SINR.
                const size_t rxtx = size_t(e.numRxPorts) * e.numTxPorts;
                auto psdIt = inPsd->ConstValuesBegin();
                if (!revOk)
                {
                    // Canonical layout matches the output page layout:
                    // contiguous scale-and-convert per rb page.
                    for (size_t rb = 0; rb < numRb; ++rb, ++psdIt)
                    {
                        const double v = *psdIt;
                        const double s = (v > 0.0 ? std::sqrt(v) : 0.0);
                        const std::complex<float>* src = hBase + rb * rxtx;
                        auto* pageD = reinterpret_cast<double*>(chanSpct->GetPagePtr(rb));
                        for (size_t i = 0; i < rxtx; ++i)
                        {
                            pageD[i * 2] = static_cast<double>(src[i].real()) * s;
                            pageD[i * 2 + 1] = static_cast<double>(src[i].imag()) * s;
                        }
                    }
                }
                else
                {
                    // UL: transpose the canonical matrix. Cached index is
                    // u + uPorts*s (rx-fast canonical); output index is
                    // s + sPorts*u (rx-fast in the caller's orientation,
                    // whose rows are the gNB s-ports).
                    const size_t uPorts = e.numRxPorts; // canonical rows
                    const size_t sPorts = e.numTxPorts; // canonical cols
                    for (size_t rb = 0; rb < numRb; ++rb, ++psdIt)
                    {
                        const double v = *psdIt;
                        const double s = (v > 0.0 ? std::sqrt(v) : 0.0);
                        const std::complex<float>* src = hBase + rb * rxtx;
                        auto* pageD = reinterpret_cast<double*>(chanSpct->GetPagePtr(rb));
                        for (size_t u = 0; u < uPorts; ++u)
                        {
                            for (size_t sp = 0; sp < sPorts; ++sp)
                            {
                                const std::complex<float> h = src[u + uPorts * sp];
                                const size_t oi = sp + sPorts * u;
                                pageD[oi * 2] = static_cast<double>(h.real()) * s;
                                pageD[oi * 2 + 1] = static_cast<double>(h.imag()) * s;
                            }
                        }
                    }
                }
                return chanSpct;
            }
            else if (!longTerm)
            {
                // Miss-reason accounting, first failing condition only.
                // Gated on !longTerm so only CBG's first probe per eval
                // counts (the two GenSpec-internal retries pass a real
                // longTerm and would triple-count).
                if (e.generatedTime != channelMatrix->m_generatedTime)
                {
                    SLS_PHASE_SCOPE("Mez::Miss::StaleTime");
                }
                else if (!fwdOk && !revOk)
                {
                    SLS_PHASE_SCOPE("Mez::Miss::Ports");
                }
                else if (e.numRb != numRb)
                {
                    SLS_PHASE_SCOPE("Mez::Miss::NumRb");
                }
                else
                {
                    SLS_PHASE_SCOPE("Mez::Miss::RbFreqs");
                }
                if (isReverse)
                {
                    SLS_PHASE_SCOPE("Mez::Miss::IsReverse");
                }
            }
        }
        else if (!longTerm)
        {
            SLS_PHASE_SCOPE("Mez::Miss::NoEntry");
            if (isReverse)
            {
                SLS_PHASE_SCOPE("Mez::Miss::IsReverse");
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
    // Both orientations are cached. `longTerm` always arrives in the
    // canonical (DL) orientation — PRX only materialises the transpose
    // later, inside its own CPU contraction — so the contraction below
    // is canonical and both reductions (fwd + rev) are derived from it.
    // The entry stores canonical ports; the returned fake matrix is
    // shaped in the caller's orientation.
    if (channelMatrix && channelParams && longTerm &&
        !m_batchRbFreqs.empty() &&
        longTerm->GetNumPages() == channelMatrix->m_channel.GetNumPages() &&
        (isReverse ? (longTerm->GetNumRows() == numTxPorts &&
                      longTerm->GetNumCols() == numRxPorts)
                   : (longTerm->GetNumRows() == numRxPorts &&
                      longTerm->GetNumCols() == numTxPorts)) &&
        delayT.size() == size_t(longTerm->GetNumPages()) * numRb &&
        sqrtVit.size() == numRb)
    {
        SLS_PHASE_SCOPE("Mez::TryGenSpecCpuMiss");
        const size_t numCluster = channelMatrix->m_channel.GetNumPages();
        // Canonical (DL) port dims, independent of the caller's orientation.
        const uint32_t canRx = static_cast<uint32_t>(longTerm->GetNumRows());
        const uint32_t canTx = static_cast<uint32_t>(longTerm->GetNumCols());
        const size_t rxtx = size_t(canRx) * canTx;
        // Beam-aware key: the caller's longTerm reflects the CURRENT beam
        // pair (PRX::GetLongTerm recomputes on beam change), so the entry
        // is cached per (link, beam pair) and only matching-beam evals
        // will hit it.
        const uint64_t plainKey =
            MatrixBasedChannelModel::GetKey(channelMatrix->m_antennaPair.first,
                                            channelMatrix->m_antennaPair.second);
        const uint64_t key = MixBeamKey(plainKey, sWBeamHash, uWBeamHash);

        // Assign a CPU-section region on first sight, or when the entry was
        // previously batch-owned (a stale batch baseIdx must NOT be written
        // through — it aliases another link's batch region). Regions vary
        // in size (numRb * rxtx differs per antenna bucket), so allocation
        // is a running offset. Once assigned it is reused across recomputes.
        GpuChanSpctEntry& eRef = m_gpuChanSpctMap[key];
        const bool isNewEntry = (eRef.numRb == 0 && eRef.numRxPorts == 0);
        const size_t entryElems = rxtx * numRb;
        if (isNewEntry || !eRef.cpuSection ||
            size_t(eRef.numRxPorts) * eRef.numTxPorts * eRef.numRb != entryElems)
        {
            eRef.cpuSection = true;
            eRef.specHBaseIdx = static_cast<uint32_t>(m_cpuMissNextOffset);
            m_cpuMissNextOffset += entryElems;
            if (m_specHCpuFlat.size() < m_cpuMissNextOffset)
            {
                m_specHCpuFlat.resize(m_cpuMissNextOffset);
            }
            eRef.powBaseIdx = static_cast<uint32_t>(m_cpuMissPowOffset);
            m_cpuMissPowOffset += numRb;
            if (m_specPowCpuFlat.size() < m_cpuMissPowOffset)
            {
                m_specPowCpuFlat.resize(m_cpuMissPowOffset);
                m_specPowCpuRevFlat.resize(m_cpuMissPowOffset);
            }
        }

        // Contraction H[rx,tx,rb] = sum_c longTerm[c,rx,tx] * delayT[c,rb],
        // written directly into the CPU-section cache in the canonical
        // Complex3DVector page layout (rb pages, rx-fast). This is the
        // exact quantity PRX's CPU GenSpec computes (pre sqrt(psd) scale),
        // so cached evals reproduce the CPU path bit-for-bit modulo f32.
        std::complex<float>* hDst = m_specHCpuFlat.data() + eRef.specHBaseIdx;
        std::fill(hDst, hDst + entryElems, std::complex<float>(0.0f, 0.0f));
        auto* outRaw = reinterpret_cast<float*>(hDst);
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
        m_reducedPowNumRb = static_cast<uint32_t>(numRb);

        // Scalar power rows for the 1-rx-port fast path:
        //   pow[rb]    = sum_canRx |sum_canTx H|^2
        //   powRev[rb] = sum_canTx |sum_canRx H|^2
        {
            float* powF = m_specPowCpuFlat.data() + eRef.powBaseIdx;
            float* powR = m_specPowCpuRevFlat.data() + eRef.powBaseIdx;
            for (size_t rb = 0; rb < numRb; ++rb)
            {
                const std::complex<float>* slice = hDst + rb * rxtx;
                float fwdAcc = 0.0f;
                for (size_t rx = 0; rx < canRx; ++rx)
                {
                    std::complex<float> rs{0.0f, 0.0f};
                    for (size_t tx = 0; tx < canTx; ++tx)
                        rs += slice[tx * canRx + rx];
                    fwdAcc += std::norm(rs);
                }
                powF[rb] = fwdAcc;
                float revAcc = 0.0f;
                for (size_t tx = 0; tx < canTx; ++tx)
                {
                    std::complex<float> cs{0.0f, 0.0f};
                    for (size_t rx = 0; rx < canRx; ++rx)
                        cs += slice[tx * canRx + rx];
                    revAcc += std::norm(cs);
                }
                powR[rb] = revAcc;
            }
        }

        // Store ports canonically (DL orientation) — the hit path's
        // fwdOk/revOk checks compare against this.
        eRef.matrixKey = plainKey;
        eRef.numRxPorts = canRx;
        eRef.numTxPorts = canTx;
        eRef.numRb = numRb;
        eRef.generatedTime = channelMatrix->m_generatedTime;
        eRef.rbFreqsHash = m_batchRbFreqsHash;
        // CPU-miss entries hold exactly one slot; clear any leftover
        // batch metadata so the hit path never indexes slots 1..M-1.
        eRef.batchM = 1;
        eRef.perSlotStride = 0;
        eRef.batchStartTimeSec = -1.0;

        // Scalar fast path for this eval too (1 rx port, no precoding).
        if (scalarPsdOk && numRxPorts == 1)
        {
            const float* powBase =
                (isReverse ? m_specPowCpuRevFlat : m_specPowCpuFlat).data() +
                eRef.powBaseIdx;
            Ptr<Complex3DVector>& scalarSlot =
                isReverse ? eRef.fakeScalarRev : eRef.fakeScalar;
            if (!scalarSlot || scalarSlot->GetNumPages() != numRb)
                scalarSlot = Create<Complex3DVector>(1, 1, static_cast<uint16_t>(numRb));
            const double invTx = 1.0 / static_cast<double>(numTxPorts);
            auto psdItSc = inPsd->ConstValuesBegin();
            auto* page0 = reinterpret_cast<double*>(scalarSlot->GetPagePtr(0));
            for (size_t rb = 0; rb < numRb; ++rb, ++psdItSc)
            {
                const double psd = static_cast<double>(powBase[rb]) * (*psdItSc) * invTx;
                page0[rb * 2] = psd > 0.0 ? std::sqrt(psd) : 0.0;
                page0[rb * 2 + 1] = 0.0;
            }
            return scalarSlot;
        }

        // Build the output matrix in the caller's orientation, scaled by
        // sqrt(inPsd[rb]) — same as the hit path above.
        Ptr<Complex3DVector>& fakeSlot2 = isReverse ? eRef.fakeChanSpctRev : eRef.fakeChanSpct;
        if (!fakeSlot2 || fakeSlot2->GetNumPages() != numRb ||
            fakeSlot2->GetNumRows() != numRxPorts ||
            fakeSlot2->GetNumCols() != numTxPorts)
            fakeSlot2 =
                Create<Complex3DVector>(numRxPorts, numTxPorts, static_cast<uint16_t>(numRb));
        auto psdIt2 = inPsd->ConstValuesBegin();
        if (!isReverse)
        {
            for (size_t rb = 0; rb < numRb; ++rb, ++psdIt2)
            {
                const double v = *psdIt2;
                const double s = v > 0.0 ? std::sqrt(v) : 0.0;
                const std::complex<float>* src = hDst + rb * rxtx;
                auto* pageD = reinterpret_cast<double*>(fakeSlot2->GetPagePtr(rb));
                for (size_t i = 0; i < rxtx; ++i)
                {
                    pageD[i * 2] = static_cast<double>(src[i].real()) * s;
                    pageD[i * 2 + 1] = static_cast<double>(src[i].imag()) * s;
                }
            }
        }
        else
        {
            for (size_t rb = 0; rb < numRb; ++rb, ++psdIt2)
            {
                const double v = *psdIt2;
                const double s = v > 0.0 ? std::sqrt(v) : 0.0;
                const std::complex<float>* src = hDst + rb * rxtx;
                auto* pageD = reinterpret_cast<double*>(fakeSlot2->GetPagePtr(rb));
                for (size_t u = 0; u < canRx; ++u)
                {
                    for (size_t sp = 0; sp < canTx; ++sp)
                    {
                        const std::complex<float> h = src[u + canRx * sp];
                        const size_t oi = sp + canTx * u;
                        pageD[oi * 2] = static_cast<double>(h.real()) * s;
                        pageD[oi * 2 + 1] = static_cast<double>(h.imag()) * s;
                    }
                }
            }
        }
        return fakeSlot2;
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
