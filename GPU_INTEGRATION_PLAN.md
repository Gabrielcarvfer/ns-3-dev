# GPU integration plan — ThreeGppChannelModel ←→ SlsChanWgpu

Detailed plan to make the WebGPU SLS-channel kernel (now Phase-1
calibrated against the 3GPP UMa-6GHz reference) the per-link computation
back-end for ns-3's `ThreeGppChannelModel` and
`ThreeGppSpectrumPropagationLossModel`, with minimal disruption to the
CPU-side callers.

## Status — 2026-05-27 (Phase 1 minimal-viable shipped)

Done — `UseGpu` boolean attribute on `ThreeGppChannelModel`, default
false. Setting it to true turns on a narrow LSP-only GPU back-end:

1. `MatrixBasedChannelModel::EnsureBatchFresh()` is a virtual no-op on
   the base; `ThreeGppSpectrumPropagationLossModel::DoCalcRxPowerSpectralDensity`
   calls it once before the per-link `GetChannel` loop. The CPU path
   inherits the no-op and is bit-for-bit unchanged.
2. `ThreeGppChannelModel` overrides `EnsureBatchFresh` to walk a
   per-link endpoint registry (populated on every `GetChannel`), call
   `CollectDirtyLinks` using the same predicates the CPU path uses,
   and run `SlsChanWgpu` once over the union of dirty endpoints.
3. The GPU's `LinkParams` output (DS, ASD, ASA, ZSD, ZSA, K, pathloss)
   is converted to ns-3 units (ns->s, linear K->dB) and pushed into
   the channel-params cache via two paths:
     - new links: stash in `m_gpuLspCache`, consumed by
       `GenerateOrFetchLSPs` inside `GenerateChannelParameters`;
     - links being update-cycled: mutate `m_DS` / `m_K_factor` in
       place on the existing params (`UpdateChannelParameters` does
       not re-draw LSPs).
4. Cluster / ray / phase generation still runs on the CPU, consuming
   the GPU-supplied DS and K. That keeps the cluster-coupling /
   blockage / Doppler code untouched and limits the integration
   blast radius.
5. `gpu_batch_smoke.exe` (built when `NS3_ENABLE_3GPP_GPU=ON`) runs a
   3-cell / 2-UE deployment through both back-ends and asserts the
   GPU-applied LSPs land in 3GPP-sane ranges.

What's NOT done yet — the original Phase 1 in §6 envisioned full
cluster/ray/phase readback. Instead, only the large-scale step is
GPU-driven; everything else still draws on the CPU. Phases 2-3 of the
plan are unchanged and remain future work.

Known limitations:
  - The GPU's `cal_link_param_kernel` decides LOS state per link from
    TR 38.901 probability formulas. To keep ns-3 and the GPU in
    agreement, `RunGpuLspBatch` checks each dirty link's
    `ChannelCondition`: if every dirty link is LOS we force GPU LOS,
    if every link is NLOS we force GPU NLOS, otherwise we fall back to
    the formula and accept some K-factor noise on the disagreeing
    links. Per-link force is a future WGSL change.
  - The scenario string (`m_scenario`) is mapped to the GPU's
    `SCENARIO_UM[A|I]` / `SCENARIO_RMA` enum, but the lazy-initialised
    `CmnLinkParamsGPU` inside `SlsChanWgpu::calLinkParam` is still
    hard-coded to UMa correlation distances and LSP mu/sigma. UMi /
    RMa runs will use UMa LSP statistics until that buffer also gets
    a scenario switch.
  - Antenna patterns are stubbed to the WGSL defaults (isotropic with
    102-deg theta tilt) rather than read from the `PhasedArrayModel`.
    Pathloss / coupling-loss readouts are therefore not yet sourced
    from ns-3's actual array geometry.
  - The cell/UT partition uses "lower node-id = cell, higher = UT".
    Works for ns-3 cellular scenarios where BSs are created first; not
    a universal rule.
  - The CMN-link params (CRN correlation distances, mu/sigma LSPs)
    inside `SlsChanWgpu::calLinkParam`'s lazy buffer init are
    hard-coded to UMa. UMi / RMa runs will fall through to UMa
    statistics until those tables get an explicit scenario switch.

## 1. Context

The existing CPU path lives in
`src/spectrum/model/three-gpp-channel-model.{h,cc}` and
`src/spectrum/model/three-gpp-spectrum-propagation-loss-model.{h,cc}`.
Today it does everything sequentially, one link at a time. For each
`GetChannel(txMob, rxMob, txAntenna, rxAntenna)` call:

1. Looks up `m_channelParamsMap[GetKey(txNodeId, rxNodeId)]`.
2. If absent / stale → `GenerateChannelParameters` regenerates LSPs,
   cluster delays/powers, cluster + ray angles, XPR, phases, Doppler
   terms; stores them as a `Ptr<ThreeGppChannelParams>` in the map.
3. Builds a `ChannelMatrix` from the params and the antenna pair via
   `GetNewChannel`, caches in `m_channelMatrixMap[GetKey(antA,antB)]`.
4. Returns the matrix.

The WebGPU port in `sls-chan-wgpu.{h,cc}` + `sls-chan.wgsl` now produces
the same per-link state in batch (`cal_link_param_kernel`,
`cal_cluster_ray_kernel`, `generate_cir_kernel`). The CIR layout the
WGSL kernel produces is `[nActiveLinks][nSnapshots][nBsAnt][nUeAnt][24]`
complex; the small-scale ray buffers (`xpr`, `phi_n_m_AoA/AoD`,
`theta_n_m_ZOA/ZOD`) are flat `[nActiveLinks * MAX_CR]` arrays. We need
to translate between this layout and ns-3's nested
`ThreeGppChannelParams` representation.

**Goal:** the next CPU call into `ThreeGppChannelModel::GetChannel(...)`
finds a cache entry that was filled by the GPU pipeline, with bit-for-bit
the same data shape the CPU code already consumes. No CPU code path
under `GetNewChannel` / `DoCalcRxPowerSpectralDensity` changes.

**Non-goals (this plan):**
- Replacing `GetNewChannel`'s per-element MIMO matrix construction —
  the CPU side stays in charge of folding antenna patterns / beamforming
  into the channel matrix.
- Replacing `ChannelConditionModel` — LOS/NLOS/O2I decisions remain on
  CPU.
- Mobility / displacement-based regeneration decisions
  (`NewChannelParamsNeeded`, `ChannelUpdateNeeded`) — those keep their
  current logic and merely drive a "dirty link" set into the GPU batch.

## 2. Architecture

### Current (CPU-only, per call)

```
SpectrumChannel ─→ ThreeGppSpectrumPropagationLossModel::DoCalcRxPowerSpectralDensity
                            │
                            ├─→ ThreeGppChannelModel::GetChannel(txMob, rxMob, txAnt, rxAnt)
                            │           │
                            │           ├─ NewChannelParamsNeeded ? GenerateChannelParameters
                            │           ├─ ChannelUpdateNeeded ?    UpdateChannelParameters
                            │           ├─ NewChannelMatrixNeeded ? GetNewChannel
                            │           └─ return cached ChannelMatrix
                            │
                            ├─→ ThreeGppChannelModel::GetParams(txMob, rxMob)
                            ├─→ GetLongTerm(...)               ── beamforming, cached
                            ├─→ CalcBeamformingGain(...)       ── Doppler, fresh per-call
                            ├─→ GenSpectrumChannelMatrix(...)  ── per-RB H synthesis
                            └─ return Rx PSD
```

Each `GetChannel` is a single-link CPU compute. The bulk of the work is
`GenerateChannelParameters` (LSP sampling, cluster delays/powers,
cluster + ray angles, XPR/phase) — this is what the GPU kernel can
parallelise across all dirty links.

### Proposed (GPU batch insertion)

```
SpectrumChannel ─→ ThreeGppSpectrumPropagationLossModel::DoCalcRxPowerSpectralDensity
                            │
                            ├─→ ThreeGppChannelModel::EnsureBatchFresh()      ← NEW
                            │       1. Sweep m_channelParamsMap, collect dirty links
                            │       2. Build GPU input snapshots from antennas + mobility + condition
                            │       3. Run SlsChanWgpu pipeline (CRN, cal_link_param, cal_cluster_ray)
                            │       4. Read GPU outputs into per-link C++ ThreeGppChannelParams
                            │       5. Mark links clean, stamp m_generatedTime
                            │
                            ├─→ ThreeGppChannelModel::GetChannel(txMob, rxMob, txAnt, rxAnt)
                            │       — finds a fresh ThreeGppChannelParams in the map; no per-link
                            │         CPU regeneration. Builds (or fetches) the antenna-pair
                            │         ChannelMatrix as before.
                            │
                            └─ unchanged from here down
```

The new entry point `EnsureBatchFresh()` is the only piece of CPU code
that knows about the GPU. Everything else reads a `ThreeGppChannelParams`
shape that's bit-identical to what `GenerateChannelParameters` would
have produced.

## 3. Data mapping (CPU ↔ GPU)

Critical files to read while doing the mapping:
- `src/spectrum/model/three-gpp-channel-model.h:142-219` —
  `ThreeGppChannelParams` fields.
- `src/spectrum/model/sls-chan-wgpu.h` — `CellParam`, `UtParam`,
  `CellParamSS`, `UtParamSS`, `AntPanelConfigGPU`, `SsCmnParams`,
  `LinkParams`, `ClusterParamsGpu`.

| CPU `ThreeGppChannelParams` field | dim | GPU source (after run) | conversion notes |
|---|---|---|---|
| `m_DS` (s) | scalar | `LinkParams.DS` (ns) | divide by 1e9 |
| `m_K_factor` (dB) | scalar | `LinkParams.K` (linear) | `10·log10(K)` (clamp 1e-10) |
| `m_delay[c]` (s) | nCluster | `ClusterParamsGpu.delays[c]` (ns) | divide by 1e9; sorted, normalised to 0 |
| `m_clusterPower[c]` (linear) | nCluster | `ClusterParamsGpu.powers[c]` | as-is, already normalised so Σ = 1 |
| `m_clusterShadowing[c]` (dB) | nCluster | not currently produced by kernel | TODO: optionally add to `cal_cluster_ray_kernel`'s output; for Phase 1 the existing kernel already folds the ξ shadowing into the cluster power normalisation, so we can leave `m_clusterShadowing` as zeros and the CPU `GetNewChannel` won't notice (it uses cluster powers directly). |
| `m_angle[AOA_IDX][c]` (rad) | nCluster | `ClusterParamsGpu.phinAoA[c]` (deg) | deg → rad |
| `m_angle[ZOA_IDX][c]` (rad) | nCluster | `ClusterParamsGpu.thetanZOA[c]` (deg) | deg → rad |
| `m_angle[AOD_IDX][c]` (rad) | nCluster | `ClusterParamsGpu.phinAoD[c]` (deg) | deg → rad |
| `m_angle[ZOD_IDX][c]` (rad) | nCluster | `ClusterParamsGpu.thetanZOD[c]` (deg) | deg → rad |
| `m_rayAoaRadian[c][m]` (rad) | nCl × nRay | `phi_n_m_AoA[c*nRay+m]` (deg) | deg → rad |
| `m_rayZoaRadian[c][m]` | nCl × nRay | `theta_n_m_ZOA[…]` (deg) | deg → rad |
| `m_rayAodRadian[c][m]` | nCl × nRay | `phi_n_m_AoD[…]` (deg) | deg → rad |
| `m_rayZodRadian[c][m]` | nCl × nRay | `theta_n_m_ZOD[…]` (deg) | deg → rad |
| `m_crossPolarizationPowerRatios[c][m]` (linear) | nCl × nRay | `xpr[c*nRay+m]` (linear) | as-is |
| `m_clusterPhase[c][m][pol]` (rad ∈ [-π,π]) | nCl × nRay × 4 | `random_phases` flat buffer (deg ∈ [0, 360)) | deg → rad, shift to [-π, π] |
| `m_attenuation_dB` | nCluster | n/a — no blockage in current kernel | leave as zeros; CPU code applies it additively, so 0 is the right default. |
| `m_alpha`, `m_D` | nCluster | n/a — Doppler scattering terms | compute on CPU from velocity & angles AFTER GPU readback (same code as today); cheap. |
| `m_cluster1st / m_cluster2nd` | u8 | `ClusterParamsGpu.strongest2clustersIdx[0..1]` | direct copy |
| `m_reducedClusterNumber` | u8 | `ClusterParamsGpu.nCluster` (after weak-removal applied in kernel) | direct copy |
| `m_losCondition` | enum | from `ChannelCondition` (CPU-side) | extracted before GPU run; matches what the kernel already used for LSP indexing |
| `m_dis2D`, `m_dis3D` | double | `LinkParams.d2d`, `LinkParams.d3d` | direct |
| `m_txSpeed`, `m_rxSpeed` | Vector | from MobilityModel (CPU-side) | captured at batch-build, not GPU output |
| `m_lastPositionFirst/Second` | Vector | from MobilityModel | captured at batch-build |
| `m_generatedTime` | Time | `Simulator::Now()` | stamped after readback |

The reverse direction — building GPU inputs from the cache — is
straightforward because every input has an obvious source:

| GPU input | Built from |
|---|---|
| `CellParam[]` (one per cell = site×sector) | iterate `NetDeviceContainer` of BS nodes; one entry per sector. Pull position from `MobilityModel::GetPosition`, antenna orientation (`[theta_tilt, phi_tilt, zeta]`) from the `UniformPlanarArray` (`m_alpha`, `m_beta`, `m_polSlant`). |
| `UtParam[]` (one per UE) | iterate the *home* node side of each dirty link; deduplicate (one entry per unique UE node). Position from `MobilityModel::GetPosition`. `outdoor_ind`, `d_2d_in`, `o2i_penetration_loss` from the `ChannelCondition` cache. |
| `CellParamSS[]` / `UtParamSS[]` | same as above, with the small-scale layout (panel idx + orientation + velocity for UE). Velocity comes from `MobilityModel::GetVelocity`. |
| `AntPanelConfigGPU[]` | iterate distinct `UniformPlanarArray` shapes seen in the dirty set. Map `m_numRows / m_numColumns / m_isDualPolarized` → `antSize`; `m_disH/m_disV` → `antSpacing`; `m_polSlant` → `antPolarAngles`. For Phase-1 calibration we leave the element table flat-zero and set `antModel=0`; for production we'll add the 3GPP element pattern as a precomputed table. |
| `SsCmnParams` | populated once per scenario from `GetThreeGppTable(condition)` — exactly the same source the CPU `GenerateLSPs` uses, just snapshotted into the GPU's flat struct. |
| `LinkParamUniforms.scenario / fc / nSite / nUT / nSectorPerSite / nX / nY` | from `m_scenario`, `m_frequency`, and the GPU CRN's `nX_ / nY_`. |
| `ActiveLink[]` | one per dirty `(site, ue)` pair. `cid` = the cell-id of the UE's best-aligned sector at that site (today's per-cell logic) or, when the CPU side already has an assignment policy (e.g. RRC-assigned cell), use that. |

## 4. Batching strategy

### When to batch

Three reasonable triggers, in order of integration cost:

1. **Lazy, on first `GetChannel` after Δt** — when the spectrum loss
   model is about to need a channel, call `EnsureBatchFresh()` at the
   *top* of `DoCalcRxPowerSpectralDensity`. The batch covers every
   link in `m_channelParamsMap` whose `(condition, last update time,
   endpoint displacement)` mark it dirty by the same rules
   `NewChannelParamsNeeded` / `ChannelUpdateNeeded` already encode.
   Multiple TX→RX calls within the same simulation time will share
   the batch (a `m_lastBatchTime` guard avoids re-running the GPU
   pipeline within the same `Simulator::Now()`).

2. **Scheduled, on a slot / TTI boundary** — install an
   `EventImpl` that fires every `m_updatePeriod` (or every slot) and
   rebuilds all params it knows about. Simpler timing semantics,
   but requires hooking into the scheduler from the channel model.

3. **External, driven by NR / LTE phy** — let the phy explicitly
   call `ThreeGppChannelModel::RecomputeAll()` at known points
   (e.g. start of subframe). Best for HPC use cases where the user
   knows their batch boundary; not the default.

**Default this plan picks: (1) lazy on first call.** It needs the
smallest API surface and degrades to the existing behaviour when only
one link is dirty.

### What gets batched

In one GPU run:
- one CRN generation pass (no per-link cost; depends only on the
  deployment bounding box + LSP correlation distances),
- one `cal_link_param_kernel` dispatch covering every dirty link,
- one `cal_cluster_ray_kernel` dispatch covering every dirty link,
- optionally `generate_cir_kernel` if we also want the kernel to do
  the H-matrix synthesis (we **do not** in v1 — CPU `GetNewChannel`
  still owns that, since it knows about beamforming).

For v1, the GPU pipeline stops at `cal_cluster_ray_kernel` and the
CPU keeps `GetNewChannel` exactly as-is. That is the minimum-invasive
slice.

### Sizing the GPU run

The kernel buffers we've already validated for Phase-1 (19 sites, 570
UEs, 10 830 links, 32 MB CRN, ~38 MB cluster/ray arrays) cover the
typical NR cell-deployment scale comfortably. For larger deployments
(hundreds of cells, thousands of UEs) we already have a preflight
guard in `SlsChanWgpu::generateCRN` that aborts cleanly if any single
buffer would exceed the device limit — surface that as an exception
to the channel model so the simulation falls back to CPU rather than
crash.

## 5. API surface — new / changed

### `ThreeGppChannelModel`

```cpp
// new public method
class ThreeGppChannelModel : public MatrixBasedChannelModel
{
public:
    // Force-refresh all dirty links in one GPU batch. Safe to call
    // every Simulator::Now() tick; internally short-circuits if
    // nothing is dirty and the batch already ran this time step.
    void EnsureBatchFresh();

    // For tests + manual control:
    void SetGpuEnabled(bool on);   // attribute hook ("UseGpu")
    bool GetGpuEnabled() const;

private:
    Ptr<SlsChanWgpu> m_gpu;        // owned lazily on first dirty batch
    Time             m_lastBatchTime;
    std::vector<uint64_t> CollectDirtyLinks();
    void BatchToGpu(const std::vector<uint64_t>& dirty);
    void GpuOutputsToCache(const std::vector<uint64_t>& dirty);
};
```

`m_gpu` is `Ptr<SlsChanWgpu>` so the GPU device is owned by the channel
model — one device per `ThreeGppChannelModel` instance, which matches
the existing convention of one model per scenario in ns-3 scripts.

### `ThreeGppSpectrumPropagationLossModel`

No header change. One line added at the top of
`DoCalcRxPowerSpectralDensity` (cc:519):

```cpp
m_channelModel->EnsureBatchFresh();   // NEW — no-op if everything is fresh
```

That's it for the spectrum loss model. The long-term beamforming
cache, the per-RB H synthesis, the Doppler loop — all untouched.

### Attribute / ns-3 plumbing

Add an `Attribute` on `ThreeGppChannelModel`:

- `"UseGpu"` (BooleanValue, default `true`) — turn the batch path off
  to compare against the legacy CPU path.

No new dependencies on the cmake side: `SlsChanWgpu` is already a
member of the spectrum module and `wgpu_native.dll` is already copied
next to the binaries.

## 6. Detailed migration path

### Phase 0 — preparatory refactor (no behaviour change)

- Extract the body of `GenerateChannelParameters` (three-gpp-channel-model.cc:3687–3840)
  into a free function `FillChannelParamsCpu(ThreeGppChannelParams&, …)`.
  This is the function the GPU path **replaces**; keeping it callable
  on its own lets us run mixed CPU/GPU under the same test harness
  and makes `UseGpu=false` work.
- Move the LSP table lookup into a `GetSsCmnParamsForScenario(scenario,
  condition) → SsCmnParams` helper so both paths read the same source
  of truth.
- Add a static_assert helper that mirrors the WGSL std430 layout for
  the small-scale structs — same idea as the `static_assert(sizeof(…))`
  guards already in `sls-chan-wgpu.h`.

Time budget: ≤ 1 day. Pure rearrangement; existing tests must pass
without behaviour change.

### Phase 1 — minimal viable GPU batch

- Add `EnsureBatchFresh()` skeleton that no-ops (returns immediately)
  when `UseGpu` is false.
- Implement `CollectDirtyLinks()` using the existing
  `NewChannelParamsNeeded` / `ChannelUpdateNeeded` predicates over
  `m_channelParamsMap`. Output is a vector of `(txNodeId, rxNodeId,
  txAntennaId, rxAntennaId, condition)` snapshots.
- Implement the snapshot extraction: walk the dirty links, build the
  flat `vector<CellParam>`, `vector<UtParam>`, `vector<CellParamSS>`,
  `vector<UtParamSS>`, `vector<AntPanelConfigGPU>`, `vector<ActiveLink>`
  using the mapping in §3.
- Call into `SlsChanWgpu` (`uploadCellParams` → `uploadUtParams` →
  `generateCRN` → `calLinkParam` → `uploadSmallScaleConfig` →
  `uploadAntPanelConfigs` → `uploadCmnLinkParamsSmallScale` →
  `uploadCellParamsSS` → `uploadUtParamsSS` → `calClusterRay`).
- Implement `GpuOutputsToCache()`: read back `LinkParams[]`,
  `ClusterParamsGpu[]`, `xpr[]`, `phi_nm_*[]`, `theta_nm_*[]`, random
  phases; convert each entry per §3's table; write into the existing
  `m_channelParamsMap` slot for that link.
- Stamp `m_generatedTime`, `m_lastPositionFirst/Second`,
  `m_endpointDisplacement2D` exactly the way the CPU path does.

Verification:
- New unit test
  `src/spectrum/test/three-gpp-channel-model-gpu-test.cc` that builds
  a tiny 2-site / 4-UE scenario with `UseGpu=false` first, snapshots
  every field of every cached `ThreeGppChannelParams`, then runs the
  same scenario with `UseGpu=true` and asserts every field matches
  within the appropriate tolerance (exact equality for integer
  fields; ±1e-3 absolute / ±1e-3 relative for floats).
- Run the existing `test.py -s spectrum` regression set with
  `UseGpu=true` as a CI gate.
- Visual sanity: run `examples/spectrum/three-gpp-channel-example.cc`
  with both back-ends and diff the CSV outputs.

Time budget: 4-6 days. Most of this is the marshalling code and the
exact-shape verification.

### Phase 2 — efficiency

Once the values match:

- Reuse the GPU `cir_buf_cluster` / xpr / random-phases / ray-angle
  buffers across batches when the dirty set is the same shape — avoid
  reallocating wgpu buffers every batch.
- Hold onto `SlsChanWgpu::cellParamsBuf_` and rebuild only when
  topology actually changes (rare for a steady simulation).
- For very large deployments, allow chunking the link set into
  sub-batches that fit in `maxStorageBufferBindingSize` (already a
  preflight check on the kernel side — surface it cleanly).
- Optional: when `m_updatePeriod` is set, run the GPU batch from a
  scheduled event so `DoCalcRxPowerSpectralDensity` never has to wait
  on a `waitIdle()`.

### Phase 3 — extend the GPU pipeline further

After v1 is shipped:

- Move `GetNewChannel`'s per-element MIMO matrix construction into
  the GPU as well (the WGSL `generate_cir_kernel` is mostly there
  already; the missing piece is the antenna-pattern fold and the
  long-term beamforming hook). This is the bigger payoff because
  `GetNewChannel` is the most expensive per-link CPU function.
- Migrate `ThreeGppSpectrumPropagationLossModel::GenSpectrumChannelMatrix`
  to a GPU pass too, so per-RB H synthesis happens on the GPU.

These are deliberately out of scope for v1.

## 7. Edge cases / risks

| risk | mitigation |
|---|---|
| **`UseGpu=true` but no GPU adapter** (CI, headless CI) | `SlsChanWgpu` constructor already calls `std::abort` on adapter failure. Wrap construction in a `try/catch` (or pre-check via `wgpuInstanceRequestAdapter`) inside `ThreeGppChannelModel`, log a warning, and silently fall back to the CPU path. |
| **Antenna config mismatch** (mixed UE / BS panel sizes in one batch) | Each link references an `AntPanelConfigGPU` by index. Build a per-batch dedup table of unique panels; if the count exceeds the WGSL `nPanels` budget, fall back to CPU for that batch (rare). |
| **Mobility update inside a batch** | The CPU code already pulls `GetPosition` / `GetVelocity` at batch-build time and uses *those* values to drive the GPU run. No risk of stale state as long as the batch executes synchronously inside `Simulator::Now()` — which it does in v1. |
| **Wrap-around / hex coordinates** | The current GPU harness assumes a Cartesian deployment. ns-3's MobilityModel returns Cartesian positions too — direct copy. |
| **Layout drift between host / WGSL** | Already guarded by `static_assert` in `sls-chan-wgpu.h` for every struct. Add a runtime self-check in `EnsureBatchFresh` that logs sizeof of every struct on first batch when `SLS_DEBUG=1`. |
| **`m_channelMatrixMap` invalidation** | The cache key for the matrix map is the antenna-pair ID, not the link ID — so when the GPU rewrites a link's `ThreeGppChannelParams`, the matrix cache is invalidated automatically by the existing `NewChannelMatrixNeeded` time-stamp check. No new code needed. |
| **K-factor unit semantics** | Already audited and fixed: kernel stores K in **linear** units, CPU `m_K_factor` expects **dB**. Conversion `10·log10(max(K, 1e-10))` in `GpuOutputsToCache`. |
| **Cluster delays unit** | Kernel stores `delays[c]` in **ns** (units already corrected upstream); CPU `m_delay[c]` expects **seconds**. Convert at readback. |
| **Random-phase wrap** | Kernel produces phases in degrees `[0, 360)`. CPU `m_clusterPhase[c][m][p]` expects radians in `[-π, π]`. `(deg − 180) · π / 180` (or equivalent normalisation). |
| **`m_alpha` / `m_D` Doppler scattering terms** | Currently in the cluster_ray kernel as side-effects of cluster angles; the CPU code recomputes them in `GenerateDopplerTerms` from cluster angles + velocity. Keep that CPU function and run it *after* GPU readback — cheap. |
| **Deterministic seeds** | RNG seeding in `SlsChanWgpu::generateCRN` and the cluster-ray kernel is currently derived from a stable per-link hash. To get the same draws as the CPU path on a per-test basis, we will not bit-match — but the *statistical distributions* will match (already validated against the OEM envelope). The test in Phase 1 uses tolerance, not exact equality, for random fields (angles, phases, XPR). LSPs (DS, K, ASD, ...) will likewise differ realisation-by-realisation between back-ends; the test asserts on the marginal distribution over a large UE set, not on a single pair. |
| **Threading** | ns-3 is single-threaded by default; `SlsChanWgpu`'s wgpu queue operations are blocking on `waitIdle`. The batch runs on the simulation thread; no new threading model needed. |

## 8. Verification

### Bit-shape parity tests

`src/spectrum/test/three-gpp-channel-model-gpu-test.cc` — new test
suite that:

- Spins up two `ThreeGppChannelModel` instances on the same scenario
  with `UseGpu=false` and `UseGpu=true`.
- For a fixed deterministic seed, drives the same mobility trace
  through both.
- After every `Simulator::Now()` step, fetches every cached
  `ThreeGppChannelParams` from both models and checks that every
  field exists, has the right shape, and is in the right physical
  range. Random-value fields (angles, phases) are not equality-checked
  per realisation — they are checked against marginal-distribution
  ranges. LSP fields are equality-checked across realisations only at
  the level of summary statistics (mean / std / quantiles), not
  per-realisation.

### Statistical regression

Reuse `minimal_analyzer.py` against an HDF5 dump from a
`ThreeGppChannelModel + UseGpu=true` run, with the exact same UMa-6GHz
deployment the SLS harness uses today (570 UEs, 19 sites, ISD = 500 m,
per-site Voronoi placement, 80 % indoor). All three CDFs (coupling
loss, wideband SIR, geometry SINR) must land inside the 3GPP OEM
envelope, matching the calibration we already have on the standalone
`sls_chan_validation.exe`.

This is the ultimate sanity check: it means the GPU is producing the
same per-link state for the integrated channel model as it does for the
calibration harness.

### Functional regression

- `./test.py -s spectrum` with both `UseGpu=true` and `UseGpu=false`.
- `./test.py -s nr` (if NR module is enabled) with `UseGpu=true`.

### Performance probe

Add a Phase-1-scale (570 UEs) timing harness that calls
`GetChannel` for every UE × every BS, with `UseGpu={true,false}`, and
prints the wall-clock time of the regenerate phase. The expected
speed-up is 5-20× depending on adapter (D3D12 / Vulkan / Metal).

## 9. Critical files to touch

- `src/spectrum/model/three-gpp-channel-model.h` — add
  `EnsureBatchFresh`, `m_gpu`, `m_useGpu`, `m_lastBatchTime`,
  attribute hooks. Extract `FillChannelParamsCpu` declaration.
- `src/spectrum/model/three-gpp-channel-model.cc` — add the GPU
  batch implementation; rewire `NewChannelParamsNeeded` so it just
  marks the link dirty rather than triggering a per-link regeneration
  when `UseGpu=true`. Extract `FillChannelParamsCpu` body from
  `GenerateChannelParameters`.
- `src/spectrum/model/three-gpp-spectrum-propagation-loss-model.cc` —
  one new `EnsureBatchFresh()` call at the top of
  `DoCalcRxPowerSpectralDensity`.
- `src/spectrum/model/sls-chan-wgpu.{h,cc}` — no behavioural change;
  optionally add a `SetSeed(uint64_t)` so the channel model can
  override the deterministic seed for reproducibility.
- `src/spectrum/CMakeLists.txt` — already includes
  `sls-chan-wgpu.cc` for the validation binary; expand the regular
  spectrum library target to include it as a non-test source so the
  channel model can link against `SlsChanWgpu`.
- `src/spectrum/test/three-gpp-channel-model-gpu-test.cc` — new file.

## 10. Time / sequencing summary

| step | what | who-touches | rough effort |
|---|---|---|---|
| **0** | Refactor `GenerateChannelParameters` → `FillChannelParamsCpu`; introduce `GetSsCmnParamsForScenario` | three-gpp-channel-model.{h,cc} | 1 day |
| **1.a** | Wire `SlsChanWgpu` into the spectrum library target; build-only smoke test | CMakeLists.txt, channel-model.cc | 0.5 day |
| **1.b** | `CollectDirtyLinks` + GPU input snapshot extraction (cells / UEs / antennas / active-link list) | three-gpp-channel-model.cc | 1 day |
| **1.c** | Single-batch GPU run + `GpuOutputsToCache` per-link readback + conversions | three-gpp-channel-model.cc | 1.5 days |
| **1.d** | Hook `EnsureBatchFresh` at top of `DoCalcRxPowerSpectralDensity` | three-gpp-spectrum-propagation-loss-model.cc | 0.25 day |
| **1.e** | Test suite (parity + statistical regression) | three-gpp-channel-model-gpu-test.cc | 1.5 days |
| **1.f** | Run the calibration analyzer end-to-end to confirm Phase-1 match | — | 0.25 day |
| **2** | Reuse GPU buffers across batches; topology cache | sls-chan-wgpu.cc, channel-model.cc | 1 day |
| **3** | Push `GetNewChannel` per-element MIMO into the GPU | three-gpp-channel-model.cc, sls-chan.wgsl | 3-5 days |

Total for v1 (steps 0-1): **~6 working days**. v2 (step 2): +1 day.
v3 (step 3): another week.

## 11. What can go wrong (and what we do about it)

The hardest piece is **field-by-field parity** in Phase 1's
verification. The CPU and GPU code paths apply the same physical model
but in slightly different orders (e.g. CPU normalises cluster powers
*after* the weak-cluster removal; the GPU does it *before*). The first
parity test will almost certainly flag deltas. The plan is to track
each delta to a specific line, decide whether the CPU or GPU is closer
to TR 38.901, and reconcile. The existing struct-stride and K-factor
bugs we found in the past week prove the verification harness pays for
itself many times over.

The other genuine risk is **GPU device availability**. On a Windows
build agent without a D3D12-capable GPU, the model has to silently fall
back to CPU. The current `SlsChanWgpu` constructor aborts when no
adapter is available — that needs to become recoverable before this
integration is safe to enable by default.

## 12. Out of scope explicitly

- Multi-GPU dispatch
- CUDA / HIP back-ends (the wgpu abstraction is the chosen portability
  layer — adding another back-end is a separate exercise)
- Replacing the `ChannelConditionModel` LOS/NLOS decision logic
- Per-RB H synthesis on the GPU (Phase 3 above)
- Beamforming codebook computation on the GPU
- Spatial consistency between links (already handled by the CRN grid;
  no change required at the integration boundary)
