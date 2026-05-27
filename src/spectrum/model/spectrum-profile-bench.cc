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
#include "ns3/three-gpp-spectrum-propagation-loss-model.h"
#include "ns3/uinteger.h"
#include "ns3/uniform-planar-array.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Allow the bench to switch on phase-timing instrumentation in the
// channel/PRX models. The macros are defined below this include block
// and consumed via `extern` counters in those translation units.
#ifndef SLS_PROFILE_INSTRUMENT
#define SLS_PROFILE_INSTRUMENT 1
#endif

using namespace ns3;

namespace
{

struct Args
{
    uint32_t nUes = 100;
    uint32_t nTicks = 10;
    uint32_t txElems = 4;
    uint32_t rxElems = 1;
    bool useGpu = false;
    bool runPrx = true;
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
            a.txElems = static_cast<uint32_t>(std::atoi(val));
        else if (std::strncmp(arg, "--rx-elems=", 11) == 0)
            a.rxElems = static_cast<uint32_t>(std::atoi(val));
        else if (std::strcmp(arg, "--use-gpu") == 0)
            a.useGpu = true;
        else if (std::strcmp(arg, "--no-prx") == 0)
            a.runPrx = false;
    }
    return a;
}

} // namespace

int
main(int argc, char** argv)
{
    Args args = ParseArgs(argc, argv);

    std::printf("spectrum_profile_bench: %u UEs, %u ticks, %ux%u BS antenna, "
                "%ux%u UE antenna, UseGpu=%d, RunPrx=%d\n",
                args.nUes,
                args.nTicks,
                args.txElems,
                args.txElems,
                args.rxElems,
                args.rxElems,
                args.useGpu,
                args.runPrx);

    RngSeedManager::SetSeed(42);
    RngSeedManager::SetRun(1);

    // 19-site hex deployment, ISD = 500m (3GPP Phase-1 outdoor).
    constexpr uint32_t kNumSites = 19;

    Ptr<ChannelConditionModel> ccm = CreateObject<AlwaysLosChannelConditionModel>();
    Ptr<ThreeGppChannelModel> channelModel = CreateObject<ThreeGppChannelModel>();
    channelModel->SetAttribute("Frequency", DoubleValue(6.0e9));
    channelModel->SetAttribute("Scenario", StringValue("UMa"));
    channelModel->SetAttribute("ChannelConditionModel", PointerValue(ccm));
    channelModel->SetAttribute("UpdatePeriod", TimeValue(MilliSeconds(0))); // updated mid-loop
    channelModel->SetAttribute("UseGpu", BooleanValue(args.useGpu));
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

    auto mkAnt = [](uint32_t n) {
        return CreateObjectWithAttributes<UniformPlanarArray>(
            "NumColumns",
            UintegerValue(n),
            "NumRows",
            UintegerValue(n),
            "AntennaElement",
            PointerValue(CreateObject<IsotropicAntennaModel>()),
            "NumVerticalPorts",
            UintegerValue(1),
            "NumHorizontalPorts",
            UintegerValue(1));
    };
    std::vector<Ptr<PhasedArrayModel>> ants;
    for (uint32_t i = 0; i < kNumSites; ++i)
        ants.push_back(mkAnt(args.txElems));
    for (uint32_t i = 0; i < args.nUes; ++i)
        ants.push_back(mkAnt(args.rxElems));
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
    // to construct a wideband (100 RB) PSD; bw/freq are illustrative
    // only — the channel model only consumes the spectrum-model size.
    std::vector<int> activeRbs;
    activeRbs.reserve(100);
    for (int i = 0; i < 100; ++i)
        activeRbs.push_back(i);
    Ptr<SpectrumValue> txPsd =
        LteSpectrumValueHelper::CreateTxPowerSpectralDensity(/*earfcn=*/3050,
                                                             /*bandwidthRb=*/100,
                                                             /*powerTxDbm=*/46.0,
                                                             activeRbs);

    using clock = std::chrono::steady_clock;
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
                        prxModel->CalcRxPowerSpectralDensity(sig, txMob, rxMob, txAnt, rxAnt);
                    }
                    else
                    {
                        channelModel->GetChannel(txMob, rxMob, txAnt, rxAnt);
                    }
                    ++totalLinks;
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
