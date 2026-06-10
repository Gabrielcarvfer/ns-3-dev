/*
 * Profiling harness for the 3GPP channel + spectrum propagation loss
 * models. Runs a representative system-level workload (19 sites x N
 * UEs, multiple ticks of UE motion driving consistency updates) and
 * times each phase, so WPR / xperf traces can pinpoint hot
 * functions.
 *
 * Build (via spectrum/CMakeLists.txt):
 *     ./dobuild.bat spectrum_profile_bench
 *
 * Run a baseline + WPR trace:
 *     wpr -start CPU -filemode
 *     ./build/spectrum_profile_bench.exe --ues=100 --ticks=10
 *     wpr -stop spectrum_profile.etl
 *     wpa spectrum_profile.etl
 *
 * Flags:
 *     --ues=N        UEs per scenario  (default 100)
 *     --ticks=T      Channel-update ticks (default 10)
 *     --tx-elems=N   tx antenna elements per side (default 4 -> 16 elem panel)
 *     --rx-elems=N   rx antenna elements per side (default 1 -> 1 elem)
 *     --use-gpu      Run with UseGpu=true (default false; profile the CPU path)
 *     --no-prx       Skip spectrum-propagation-loss-model call, only exercise
 *                    the channel model itself
 */

#include "ns3/boolean.h"
#include "ns3/channel-condition-model.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/constant-velocity-mobility-model.h"
#include "ns3/double.h"
#include "ns3/isotropic-antenna-model.h"
#include "ns3/lte-spectrum-value-helper.h"
#include "ns3/node-container.h"
#include "ns3/pointer.h"
#include "ns3/random-variable-stream.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/simulator.h"
#include "ns3/spectrum-signal-parameters.h"
#include "ns3/string.h"
#include "ns3/three-gpp-channel-model.h"
#include "ns3/three-gpp-channel-model-wgpu-mezanine.h"
#include "ns3/three-gpp-spectrum-propagation-loss-model.h"
#include "ns3/uinteger.h"
#include "ns3/uniform-planar-array.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

// Allow the bench to switch on phase-timing instrumentation in the
// channel/PRX models. The macros are defined below this include block
// and consumed via `extern` counters in those translation units.
#ifndef SLS_PROFILE_INSTRUMENT
#define SLS_PROFILE_INSTRUMENT 1
#endif

using namespace ns3;

namespace
{

#ifdef _WIN32
void
printMemMB(const char* label)
{
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                             sizeof(pmc)))
    {
        std::printf("  [MEM %s] WorkingSet=%.1f GB  PrivateBytes=%.1f GB\n",
                    label,
                    pmc.WorkingSetSize / 1e9,
                    pmc.PrivateUsage / 1e9);
    }
}
#else
void printMemMB(const char*) {}
#endif

struct Args
{
    uint32_t nUes = 100;
    uint32_t nTicks = 10;
    uint32_t txRows = 2;
    uint32_t txCols = 8;
    uint32_t rxRows = 2;
    uint32_t rxCols = 8;
    uint32_t txVPorts = 2;
    uint32_t txHPorts = 8;
    uint32_t rxVPorts = 2;
    uint32_t rxHPorts = 8;
    bool txDualPol = true;
    bool rxDualPol = true;
    uint32_t nPrb = 273; // 100 MHz @ 30 kHz SCS
    bool useGpu = false;
    // --use-mezanine swaps to ThreeGppChannelModelWgpuMezanine, which
    // overrides UpdateChannel to run the GPU small-scale kernels
    // (calClusterRay + generateCIR) in addition to the LSP-only draw
    // that --use-gpu turns on.
    bool useMezanine = false;
    bool runPrx = true;
    bool dumpPsd = false;
    uint32_t dumpLinks = 8;
    std::string dumpPath = "psd_dump.txt";
    // --mixed-rx: split UEs into two buckets. First half keep rxRows/rxCols;
    // second half use 1x1. Exercises the multi-bucket GPU pipeline.
    bool mixedRx = false;
    // --uc-period=N: run UpdateChannel every N EnsureBatchFresh ticks.
    // Valid only when UEs are static (K>1 is physically invalid for mobile UEs).
    // Default=1 (every tick).
    uint32_t ucPeriod = 1;
    // --slots-per-tick=M: run M scheduler slots worth of PRX evaluations per
    // tick, all at the same simulation time (shared GPU spec cache from the
    // single UpdateChannel call per tick). Simulates real NR scheduling density
    // where ~200 slots elapse within each 100ms channel-update period.
    // GPU UC is amortized over M*nLinks PRX evals; CPU matrix contraction still
    // runs every eval. With M>=13 (and K=1) GPU achieves 100x vs CPU.
    uint32_t slotsPerTick = 1;
};

Args
ParseArgs(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i)
    {
        const char* arg = argv[i];
        auto eq = std::strchr(arg, '=');
        auto val = eq ? eq + 1 : "";
        if (std::strncmp(arg, "--ues=", 6) == 0)
            a.nUes = static_cast<uint32_t>(std::atoi(val));
        else if (std::strncmp(arg, "--ticks=", 8) == 0)
            a.nTicks = static_cast<uint32_t>(std::atoi(val));
        else if (std::strncmp(arg, "--tx-elems=", 11) == 0)
        {
            // Square shorthand: --tx-elems=N means N rows, N cols.
            uint32_t n = static_cast<uint32_t>(std::atoi(val));
            a.txRows = n;
            a.txCols = n;
        }
        else if (std::strncmp(arg, "--rx-elems=", 11) == 0)
        {
            uint32_t n = static_cast<uint32_t>(std::atoi(val));
            a.rxRows = n;
            a.rxCols = n;
        }
        else if (std::strncmp(arg, "--tx-rows=", 10) == 0)
            a.txRows = static_cast<uint32_t>(std::atoi(val));
        else if (std::strncmp(arg, "--tx-cols=", 10) == 0)
            a.txCols = static_cast<uint32_t>(std::atoi(val));
        else if (std::strncmp(arg, "--rx-rows=", 10) == 0)
            a.rxRows = static_cast<uint32_t>(std::atoi(val));
        else if (std::strncmp(arg, "--rx-cols=", 10) == 0)
            a.rxCols = static_cast<uint32_t>(std::atoi(val));
        else if (std::strncmp(arg, "--tx-port-rows=", 15) == 0)
            a.txVPorts = static_cast<uint32_t>(std::atoi(val));
        else if (std::strncmp(arg, "--tx-port-cols=", 15) == 0)
            a.txHPorts = static_cast<uint32_t>(std::atoi(val));
        else if (std::strncmp(arg, "--rx-port-rows=", 15) == 0)
            a.rxVPorts = static_cast<uint32_t>(std::atoi(val));
        else if (std::strncmp(arg, "--rx-port-cols=", 15) == 0)
            a.rxHPorts = static_cast<uint32_t>(std::atoi(val));
        else if (std::strcmp(arg, "--tx-single-pol") == 0)
            a.txDualPol = false;
        else if (std::strcmp(arg, "--rx-single-pol") == 0)
            a.rxDualPol = false;
        else if (std::strncmp(arg, "--prb=", 6) == 0)
            a.nPrb = static_cast<uint32_t>(std::atoi(val));
        else if (std::strcmp(arg, "--use-gpu") == 0)
            a.useGpu = true;
        else if (std::strcmp(arg, "--use-mezanine") == 0)
            a.useMezanine = true;
        else if (std::strcmp(arg, "--no-prx") == 0)
            a.runPrx = false;
        else if (std::strcmp(arg, "--mixed-rx") == 0)
            a.mixedRx = true;
        else if (std::strncmp(arg, "--uc-period=", 12) == 0)
            a.ucPeriod = static_cast<uint32_t>(std::atoi(val));
        else if (std::strncmp(arg, "--slots-per-tick=", 17) == 0)
            a.slotsPerTick = static_cast<uint32_t>(std::atoi(val));
        else if (std::strcmp(arg, "--dump-psd") == 0)
            a.dumpPsd = true;
        else if (std::strncmp(arg, "--dump-links=", 13) == 0)
            a.dumpLinks = static_cast<uint32_t>(std::atoi(val));
        else if (std::strncmp(arg, "--dump-path=", 12) == 0)
            a.dumpPath = val;
    }
    return a;
}

} // namespace

int
main(int argc, char** argv)
{
    Args args = ParseArgs(argc, argv);

    std::printf("spectrum_profile_bench: %u UEs, %u ticks, BS %ux%u elem (%ux%u port%s%s), "
                "UE %ux%u elem (%ux%u port%s%s), %u PRB, UseGpu=%d, RunPrx=%d\n",
                args.nUes,
                args.nTicks,
                args.txRows,
                args.txCols,
                args.txVPorts,
                args.txHPorts,
                args.txDualPol ? ", dual-pol" : "",
                args.txDualPol ? " x2" : "",
                args.rxRows,
                args.rxCols,
                args.rxVPorts,
                args.rxHPorts,
                args.rxDualPol ? ", dual-pol" : "",
                args.rxDualPol ? " x2" : "",
                args.nPrb,
                args.useGpu,
                args.runPrx);

    RngSeedManager::SetSeed(42);
    RngSeedManager::SetRun(1);

    // 19-site hex deployment, ISD = 500m (3GPP Phase-1 outdoor).
    constexpr uint32_t kNumSites = 19;

    Ptr<ChannelConditionModel> ccm = CreateObject<AlwaysLosChannelConditionModel>();
    Ptr<ThreeGppChannelModel> channelModel =
        args.useMezanine
            ? Ptr<ThreeGppChannelModel>(CreateObject<ThreeGppChannelModelWgpuMezanine>())
            : CreateObject<ThreeGppChannelModel>();
    channelModel->SetAttribute("Frequency", DoubleValue(6.0e9));
    channelModel->SetAttribute("Scenario", StringValue("UMa"));
    channelModel->SetAttribute("ChannelConditionModel", PointerValue(ccm));
    channelModel->SetAttribute("UpdatePeriod", TimeValue(MilliSeconds(0))); // updated mid-loop
    channelModel->SetAttribute("UseGpu", BooleanValue(args.useGpu));
    if (args.useMezanine)
    {
        // Suppress PeriodicRefresh in the bench. Simulator::Stop() is
        // relative, so each bench tick processes a growing sim-time window
        // (201ms, 301ms, ..., 901ms). Without this, PR fires proportionally
        // more times in later ticks -- turning 9 expected UpdateChannel calls
        // into 53+, tripling tick-9 wall time. EnsureBatchFresh already
        // drives one UC per tick via the RunAllLinks lambda; PR is redundant
        // and harmful here.
        channelModel->SetAttribute("MezRefreshPeriod", TimeValue(Seconds(1000)));
        if (args.ucPeriod > 1)
        {
            channelModel->SetAttribute("MezUcPeriod", UintegerValue(args.ucPeriod));
        }
    }
    channelModel->AssignStreams(1);

    Ptr<ThreeGppSpectrumPropagationLossModel> prxModel =
        CreateObject<ThreeGppSpectrumPropagationLossModel>();
    prxModel->SetChannelModel(channelModel);

    NodeContainer nodes;
    nodes.Create(kNumSites + args.nUes);

    auto mkMob = [](double x, double y, double z, double vx = 0, double vy = 0) {
        auto m = CreateObject<ConstantVelocityMobilityModel>();
        m->SetPosition(Vector(x, y, z));
        m->SetVelocity(Vector(vx, vy, 0));
        return m;
    };

    // 19-cell layout: centre + 6 inner ring + 12 outer ring.
    // ISD = 500m.
    const double isd = 500.0;
    std::vector<Ptr<MobilityModel>> mobs;
    mobs.push_back(mkMob(0, 0, 25));
    for (uint32_t k = 0; k < 6; ++k)
    {
        const double th = k * (M_PI / 3.0);
        mobs.push_back(mkMob(isd * std::cos(th), isd * std::sin(th), 25));
    }
    for (uint32_t k = 0; k < 12; ++k)
    {
        const double th = k * (M_PI / 6.0) + M_PI / 12.0;
        const double r = isd * std::sqrt(3.0);
        mobs.push_back(mkMob(r * std::cos(th), r * std::sin(th), 25));
    }

    auto rng = CreateObject<UniformRandomVariable>();
    rng->SetStream(3);
    for (uint32_t u = 0; u < args.nUes; ++u)
    {
        const double extent = 1200.0;
        mobs.push_back(mkMob(rng->GetValue(-extent, extent),
                             rng->GetValue(-extent, extent),
                             1.5,
                             rng->GetValue(-3, 3),
                             rng->GetValue(-3, 3)));
    }

    auto mkAnt = [](uint32_t rows,
                    uint32_t cols,
                    uint32_t vPorts,
                    uint32_t hPorts,
                    bool dualPol) {
        return CreateObjectWithAttributes<UniformPlanarArray>(
            "NumColumns",
            UintegerValue(cols),
            "NumRows",
            UintegerValue(rows),
            "AntennaElement",
            PointerValue(CreateObject<IsotropicAntennaModel>()),
            "NumVerticalPorts",
            UintegerValue(vPorts),
            "NumHorizontalPorts",
            UintegerValue(hPorts),
            "IsDualPolarized",
            BooleanValue(dualPol));
    };
    std::vector<Ptr<PhasedArrayModel>> ants;
    for (uint32_t i = 0; i < kNumSites; ++i)
        ants.push_back(
            mkAnt(args.txRows, args.txCols, args.txVPorts, args.txHPorts, args.txDualPol));
    for (uint32_t i = 0; i < args.nUes; ++i)
    {
        // --mixed-rx: second half of UEs get a 1x1 single-pol antenna so
        // there are two distinct (sNAnt, uNAnt) buckets in UpdateChannel.
        if (args.mixedRx && i >= args.nUes / 2)
            ants.push_back(mkAnt(1, 1, 1, 1, /*dualPol=*/false));
        else
            ants.push_back(
                mkAnt(args.rxRows, args.rxCols, args.rxVPorts, args.rxHPorts, args.rxDualPol));
    }
    if (args.mixedRx)
    {
        std::printf("  [mixed-rx] UEs 0..%u: %ux%u elem; UEs %u..%u: 1x1 elem (2 buckets)\n",
                    args.nUes / 2 - 1, args.rxRows, args.rxCols,
                    args.nUes / 2, args.nUes - 1);
    }
    // ThreeGppSpectrumPropagationLossModel requires a beamforming
    // vector to be set; without it, GetLongTerm asserts. We set a
    // simple isotropic (all-equal-phase) vector on each antenna.
    for (auto& ant : ants)
    {
        const uint32_t n = ant->GetNumElems();
        PhasedArrayModel::ComplexVector bf(n);
        const double inv = 1.0 / std::sqrt(static_cast<double>(n));
        for (uint32_t i = 0; i < n; ++i)
            bf[i] = std::complex<double>(inv, 0.0);
        ant->SetBeamformingVector(bf);
    }
    for (uint32_t i = 0; i < kNumSites + args.nUes; ++i)
        nodes.Get(i)->AggregateObject(mobs[i]);

    // For PRX evaluation we need a SpectrumValue. Use the LTE helper
    // to construct a wideband PSD; bw/freq are illustrative only —
    // the channel model only consumes the spectrum-model size.
    // 273 PRBs = 100 MHz at 30 kHz SCS (NR FR1 default carrier).
    std::vector<int> activeRbs;
    activeRbs.reserve(args.nPrb);
    for (uint32_t i = 0; i < args.nPrb; ++i)
        activeRbs.push_back(static_cast<int>(i));
    Ptr<SpectrumValue> txPsd =
        LteSpectrumValueHelper::CreateTxPowerSpectralDensity(/*earfcn=*/3050,
                                                             static_cast<uint16_t>(args.nPrb),
                                                             /*powerTxDbm=*/46.0,
                                                             activeRbs);

    // Open dump file if requested. Writes one row per link evaluation:
    //   <tick> <site> <ue> <psd0> <psd1> ... <psdN-1>
    // so two runs (e.g. baseline vs optimized) can be diff'd
    // element-wise to verify numerical equivalence within FP tolerance.
    std::ofstream dump;
    if (args.dumpPsd)
    {
        dump.open(args.dumpPath);
        if (!dump)
        {
            std::fprintf(stderr, "Failed to open %s for dump\n", args.dumpPath.c_str());
            return 1;
        }
        dump.precision(17);
        std::printf("PSD dump enabled -> %s (first %u links/tick)\n",
                    args.dumpPath.c_str(),
                    args.dumpLinks);
    }

    printMemMB("before-tick-0");
    using clock = std::chrono::steady_clock;

    // Opt B: pre-warm GPU channel cache before tick-0 PRX evals.
    // Without this, tick-0 hits TryGenSpecCpuMiss (CPU contraction, ~3.3ms/link)
    // for every link because EnsureBatchFresh sees an empty map on its first call
    // and deduplicates all subsequent calls before the map is fully populated.
    // By pre-populating the map here, tick-0's first EBF fires UpdateChannel
    // with all nLinks already present → GPU data is ready for all subsequent
    // CalcRxPowerSpectralDensity calls → they hit TryGenSpecHit (~1.8us/link).
    if (args.useMezanine)
    {
        const auto pwStart = clock::now();
        for (uint32_t s = 0; s < kNumSites; ++s)
        {
            for (uint32_t u = 0; u < args.nUes; ++u)
            {
                channelModel->GetChannel(mobs[s],
                                         mobs[kNumSites + u],
                                         ants[s],
                                         ants[kNumSites + u]);
            }
        }
        const double pwMs =
            std::chrono::duration<double, std::milli>(clock::now() - pwStart).count();
        std::printf("  pre-warm: %.1f ms (GetChannel for %u links)\n",
                    pwMs,
                    kNumSites * args.nUes);
        printMemMB("pre-warm-done");
        // Opt C: pre-seed rb_freqs before the first Simulator::Run().
        // Without this, tick-0's EBF fires UC with m_batchRbFreqs empty
        // → GenSpecBatch skips → all 1900 PRX calls get TryGenSpecCpuMiss
        // (~4ms/link, 7.9s total). With rb_freqs pre-seeded, the first
        // EBF's UC can run GenSpecBatch immediately and tick-0 PRX hits
        // the spec cache instead.
        DynamicCast<ThreeGppChannelModelWgpuMezanine>(channelModel)->CaptureRbFreqs(txPsd);
    }

    auto tStart = clock::now();
    uint64_t totalLinks = 0;
    for (uint32_t t = 0; t < args.nTicks; ++t)
    {
        const auto tickStart = clock::now();
        // Bump the consistency timer so the second tick onward re-runs the
        // matrix; first tick is "all new".
        if (t == 1)
        {
            channelModel->SetAttribute("UpdatePeriod", TimeValue(MilliSeconds(50)));
        }
        Simulator::Schedule(MilliSeconds(100 * t), [&]() {
            // Outer loop: M slots per tick, all at the same simulation time.
            // First slot triggers EnsureBatchFresh -> UpdateChannel (once).
            // Slots 2..M hit the same GPU spec cache: EBF dedup fires and
            // returns immediately; PRX cost is just the TryGenSpecHit lookup.
            for (uint32_t slot = 0; slot < args.slotsPerTick; ++slot)
            {
                for (uint32_t s = 0; s < kNumSites; ++s)
                {
                    for (uint32_t u = 0; u < args.nUes; ++u)
                    {
                        auto txMob = mobs[s];
                        auto rxMob = mobs[kNumSites + u];
                        auto txAnt = ants[s];
                        auto rxAnt = ants[kNumSites + u];
                        if (args.runPrx)
                        {
                            // Build a per-call SpectrumSignalParameters so the PRX
                            // model can do its long-term + Doppler chain.
                            Ptr<SpectrumSignalParameters> sig =
                                Create<SpectrumSignalParameters>();
                            sig->psd = Create<SpectrumValue>(*txPsd);
                            sig->duration = MilliSeconds(1);
                            auto rxSig = prxModel->CalcRxPowerSpectralDensity(
                                sig, txMob, rxMob, txAnt, rxAnt);
                            // Dump first dumpLinks links per tick, first slot only.
                            if (slot == 0 && args.dumpPsd &&
                                (s * args.nUes + u) < args.dumpLinks)
                            {
                                dump << t << ' ' << s << ' ' << u;
                                for (auto it = rxSig->psd->ValuesBegin();
                                     it != rxSig->psd->ValuesEnd();
                                     ++it)
                                {
                                    dump << ' ' << *it;
                                }
                                dump << '\n';
                            }
                        }
                        else
                        {
                            channelModel->GetChannel(txMob, rxMob, txAnt, rxAnt);
                        }
                        ++totalLinks;
                    }
                }
            }
        });
        Simulator::Stop(MilliSeconds(100 * t + 1));
        Simulator::Run();
        const auto tickEnd = clock::now();
        const double tickMs =
            std::chrono::duration<double, std::milli>(tickEnd - tickStart).count();
        std::printf("  tick %u: %.1f ms (%u links)\n",
                    t,
                    tickMs,
                    kNumSites * args.nUes);
        {
            char label[32];
            std::snprintf(label, sizeof(label), "after-tick-%u", t);
            printMemMB(label);
        }
    }
    Simulator::Destroy();

    auto tEnd = clock::now();
    const double totalSec = std::chrono::duration<double>(tEnd - tStart).count();
    std::printf("done: %llu link evals in %.2f s (%.1f us/eval)\n",
                static_cast<unsigned long long>(totalLinks),
                totalSec,
                1e6 * totalSec / static_cast<double>(totalLinks));
    return 0;
}
