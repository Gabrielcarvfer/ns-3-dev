/*
 * Smoke test for ThreeGppChannelModel with UseGpu=true.
 *
 * Build via spectrum/CMakeLists.txt. Run with `gpu_batch_smoke.exe`. It
 * creates a tiny 3-cell / 2-UE deployment, drives the channel model
 * through one EnsureBatchFresh -> GetChannel cycle on both the CPU and
 * GPU paths, and prints LSPs (DS, K, ASA, ASD, ZSA, ZSD) for visual
 * comparison. This is intentionally a smoke test, not a parity test:
 * the GPU CRN draws and the CPU correlated-Gaussian draws are
 * statistically different processes, so the LSPs WILL diverge per
 * realization. We just check the GPU path runs without crashing and
 * produces finite, in-range values.
 *
 * The test is independent of `test-runner` (which currently doesn't
 * link on Windows-MinGW due to a wifi DLL export-ordinal overflow). It
 * links directly against the spectrum DLL + its deps.
 */

#include "ns3/channel-condition-model.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/constant-velocity-mobility-model.h"
#include "ns3/double.h"
#include "ns3/boolean.h"
#include "ns3/isotropic-antenna-model.h"
#include "ns3/node-container.h"
#include "ns3/pointer.h"
#include "ns3/random-variable-stream.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/three-gpp-channel-model.h"
#include "ns3/uinteger.h"
#include "ns3/uniform-planar-array.h"

#include <cmath>
#include <cstdio>
#include <iostream>

using namespace ns3;

namespace
{

struct LinkPair
{
    Ptr<MobilityModel> a;
    Ptr<MobilityModel> b;
    Ptr<PhasedArrayModel> aAnt;
    Ptr<PhasedArrayModel> bAnt;
    const char* label;
};

int g_failures = 0;

void
DriveOnce(Ptr<ThreeGppChannelModel> model, const LinkPair& lp)
{
    auto mat = model->GetChannel(lp.a, lp.b, lp.aAnt, lp.bAnt);
    auto params = model->GetParams(lp.a, lp.b);
    auto p3 =
        DynamicCast<const ThreeGppChannelModel::ThreeGppChannelParams>(params);
    if (!p3)
    {
        std::fprintf(stderr, "  [%s] GetParams returned non-ThreeGpp params\n", lp.label);
        ++g_failures;
        return;
    }
    // Sanity checks: DS must be finite and in (1ns, 10us). K is in dB,
    // typically in (-10, 30); for NLOS it's 0/unused but should still
    // be finite.
    const double ds = p3->m_DS;
    const double k = p3->m_K_factor;
    // 1e-10 s (0.1 ns) accommodates very-close LOS UEs where the
    // delay spread can legitimately collapse below 1 ns. 1e-5 s (10 us)
    // covers the worst-case NLOS upper end.
    const bool dsOk = std::isfinite(ds) && ds > 1e-10 && ds < 1e-5;
    const bool kOk = std::isfinite(k) && k > -20.0 && k < 40.0;
    if (!dsOk || !kOk)
    {
        std::fprintf(stderr,
                     "  [%s] INVALID LSPs: DS=%.3e K=%.3f\n",
                     lp.label,
                     ds,
                     k);
        ++g_failures;
    }
    // Per-link prints get noisy with 700 links; quiet unless the link
    // tripped the sanity check.
    (void)mat;
    if (!dsOk || !kOk)
    {
        std::printf("  [%s] H=(%llux%llux%llu) DS=%.3e K=%.3f\n",
                    lp.label,
                    static_cast<unsigned long long>(mat->m_channel.GetNumRows()),
                    static_cast<unsigned long long>(mat->m_channel.GetNumCols()),
                    static_cast<unsigned long long>(mat->m_channel.GetNumPages()),
                    ds,
                    k);
    }
}

void
RunScenario(const char* label, bool useGpu, bool nlos = false)
{
    std::printf("---- %s (UseGpu=%s, NLOS=%s) ----\n",
                label,
                useGpu ? "true" : "false",
                nlos ? "true" : "false");

    Ptr<ChannelConditionModel> ccm =
        nlos ? Ptr<ChannelConditionModel>(CreateObject<NeverLosChannelConditionModel>())
             : Ptr<ChannelConditionModel>(CreateObject<AlwaysLosChannelConditionModel>());

    Ptr<ThreeGppChannelModel> model = CreateObject<ThreeGppChannelModel>();
    model->SetAttribute("Frequency", DoubleValue(6.0e9));
    model->SetAttribute("Scenario", StringValue("UMa"));
    model->SetAttribute("ChannelConditionModel", PointerValue(ccm));
    model->SetAttribute("UpdatePeriod", TimeValue(MilliSeconds(0)));
    model->SetAttribute("UseGpu", BooleanValue(useGpu));
    model->AssignStreams(1);

    // Deployment scaled up enough to feed the NVIDIA
    // analysis_channel_stats.py with statistically-meaningful samples
    // (SIR/SINR computations collapse on tiny scenarios). Layout:
    // 7 cells in a hex pattern (ISD=150m), 100 UEs scattered in a
    // 600m x 600m box centred on the cells.
    constexpr uint32_t kNumCells = 7;
    constexpr uint32_t kNumUes = 100;
    NodeContainer nodes;
    nodes.Create(kNumCells + kNumUes);

    // Cells (nodes 0..kNumCells-1) on a hex centred at origin, UEs
    // (kNumCells..) scattered. Lower node IDs become "cells" in the
    // RunGpuLspBatch partition heuristic.
    auto mkMob = [](double x, double y, double z) {
        auto m = CreateObject<ConstantPositionMobilityModel>();
        m->SetPosition(Vector(x, y, z));
        return m;
    };
    auto mkAnt = [](uint32_t n) {
        return CreateObjectWithAttributes<UniformPlanarArray>(
            "NumColumns", UintegerValue(n),
            "NumRows", UintegerValue(n),
            "AntennaElement", PointerValue(CreateObject<IsotropicAntennaModel>()),
            "NumVerticalPorts", UintegerValue(1),
            "NumHorizontalPorts", UintegerValue(1));
    };

    auto mkVelMob = [](double x, double y, double z, double vx, double vy) {
        auto m = CreateObject<ConstantVelocityMobilityModel>();
        m->SetPosition(Vector(x, y, z));
        m->SetVelocity(Vector(vx, vy, 0.0));
        return m;
    };

    std::vector<Ptr<MobilityModel>> mobs;
    // 7-cell hex: centre + 6 ring cells at ISD ~150m.
    const double isd = 150.0;
    mobs.push_back(mkMob(0.0, 0.0, 25.0));
    for (uint32_t k = 0; k < 6; ++k)
    {
        const double th = k * (M_PI / 3.0);
        mobs.push_back(mkMob(isd * std::cos(th), isd * std::sin(th), 25.0));
    }
    // UEs uniform in a 600x600m box, deterministic positions (RNG-seeded).
    // Give them small velocities so the spatial-consistency update marks
    // links dirty between tick 1 and tick 2 (which is what exercises
    // RunGpuLspBatch on the GPU).
    auto rng = CreateObject<UniformRandomVariable>();
    rng->SetStream(2);
    for (uint32_t u = 0; u < kNumUes; ++u)
    {
        const double x = rng->GetValue(-200.0, 350.0);
        const double y = rng->GetValue(-200.0, 350.0);
        const double vx = rng->GetValue(-3.0, 3.0);
        const double vy = rng->GetValue(-3.0, 3.0);
        mobs.push_back(mkVelMob(x, y, 1.5, vx, vy));
    }

    std::vector<Ptr<PhasedArrayModel>> ants;
    for (uint32_t i = 0; i < kNumCells; ++i)
        ants.push_back(mkAnt(2));
    for (uint32_t i = 0; i < kNumUes; ++i)
        ants.push_back(mkAnt(1));

    for (uint32_t i = 0; i < kNumCells + kNumUes; ++i)
        nodes.Get(i)->AggregateObject(mobs[i]);

    std::vector<LinkPair> links;
    for (uint32_t c = 0; c < kNumCells; ++c)
    {
        for (uint32_t u = kNumCells; u < kNumCells + kNumUes; ++u)
        {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "c%u-u%u", c, u - kNumCells);
            links.push_back({mobs[c], mobs[u], ants[c], ants[u], _strdup(buf)});
        }
    }

    // ── Tick 1 — populate endpoint registry (CPU path runs LSPs because
    //             m_linkEndpoints was empty when EnsureBatchFresh ran).
    model->EnsureBatchFresh();
    for (auto& lp : links)
        DriveOnce(model, lp);

    // ── Tick 2 — advance the clock past the consistency step, then call
    //             EnsureBatchFresh again. With UseGpu=true this should
    //             run the GPU pipeline for every link in the registry
    //             and populate m_gpuLspCache. Then DriveOnce's
    //             GetChannel will consume the cache (via
    //             GenerateOrFetchLSPs) instead of drawing on the CPU.
    //
    // We bump UpdatePeriod so every link reports as dirty on the next
    // EnsureBatchFresh.
    model->SetAttribute("UpdatePeriod", TimeValue(MilliSeconds(50)));
    Simulator::Schedule(MilliSeconds(100), [&]() {
        std::printf("  [tick2] calling EnsureBatchFresh\n");
        model->EnsureBatchFresh();
        for (auto& lp : links)
            DriveOnce(model, lp);
    });

    Simulator::Stop(MilliSeconds(200));
    Simulator::Run();

    // Dump the GPU batch state for the GPU-enabled scenarios. The
    // resulting file can be fed to analysis_channel_stats.py /
    // minimal_analyzer.py to confirm coupling-loss / SIR / SINR still
    // sit inside the 3GPP Phase-1 envelope when driven via ns-3.
    if (useGpu)
    {
        const std::string h5name = std::string("gpu_batch_smoke_") + (nlos ? "nlos" : "los") +
                                   ".h5";
        std::remove(h5name.c_str());
        model->DumpGpuChannelsToHdf5(h5name,
                                     /*isd=*/150.0,
                                     /*bsHeight=*/25.0,
                                     /*bandwidthHz=*/20e6);
        std::printf("  HDF5 dump written: %s\n", h5name.c_str());
    }

    Simulator::Destroy();
}

} // namespace

int
main()
{
    RngSeedManager::SetSeed(42);
    RngSeedManager::SetRun(1);

    try
    {
        RunScenario("CPU baseline", false);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "CPU scenario crashed: %s\n", e.what());
        return 1;
    }

    try
    {
        RunScenario("GPU batch path (LOS)", true);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "GPU LOS scenario crashed: %s\n", e.what());
        return 2;
    }

    try
    {
        RunScenario("GPU batch path (NLOS)", true, /*nlos=*/true);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "GPU NLOS scenario crashed: %s\n", e.what());
        return 4;
    }

    if (g_failures > 0)
    {
        std::fprintf(stderr, "gpu_batch_smoke FAILED (%d link(s) had invalid LSPs)\n", g_failures);
        return 3;
    }
    std::printf("gpu_batch_smoke OK\n");
    return 0;
}
