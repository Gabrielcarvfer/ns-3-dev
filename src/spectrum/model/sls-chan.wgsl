// This is a mostly AI translated version of the original Nvidia source for GPU compute,
// but ported to WGPU for cross-platform support
// https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/main/testBenches/chanModels/src/sls_chan_src/sls_chan_large_scale_GPU.cu

// ===========================================================================
// sls_chan.wgsl — WGSL port of sls_chan.cu (wgpu 0.20+)
// ===========================================================================
// Struct field order/padding MUST match your sls_chan.cuh exactly.
// All uint8_t fields are widened to u32 for alignment.
// ===========================================================================

// ── Constants ──────────────────────────────────────────────────────────────
const PI:           f32 = 3.14159265358979323846;
const SCENARIO_UMA: u32 = 0u;
const SCENARIO_UMI: u32 = 1u;
const SCENARIO_RMA: u32 = 2u;

const SF_IDX:  u32 = 0u;
const K_IDX:   u32 = 1u;
const DS_IDX:  u32 = 2u;
const ASD_IDX: u32 = 3u;
const ASA_IDX: u32 = 4u;
const ZSD_IDX: u32 = 5u;
const ZSA_IDX: u32 = 6u;
const D_T_IDX: u32 = 7u;   // delta_tau CRN column

const LOS_MATRIX_SIZE:  u32 = 7u;
const NLOS_MATRIX_SIZE: u32 = 6u;
const O2I_MATRIX_SIZE:  u32 = 6u;

// Max filter length = 2·floor(3·120)+1 = 721
const MAX_FILTER_LEN: u32 = 721u;

// ── Structs ────────────────────────────────────────────────────────────────
// Pad every struct to 16-byte multiples so std430 layout matches host-side.

struct Vec3f { x: f32, y: f32, z: f32, _p: f32 }

struct CellParam {
    cid: u32,
    siteId: u32,
    loc: vec3<f32>,
    antPanelIdx: u32,
    antPanelOrientation: vec3<f32>,
    monostaticInd: u32,
    _pad1: u32,
    _pad2: u32,
    secondAntPanelIdx: u32,
    secondAntPanelOrientation: vec3<f32>,
}

struct UtParam {
    loc:                  Vec3f,
    d_2d_in:              f32,
    outdoor_ind:          u32,   // 0 = indoor, 1 = outdoor
    o2i_penetration_loss: f32,
    _p:                   f32,
    // Add remaining UtParam fields here
}

struct SystemLevelConfig {
    scenario:                     u32,
    enable_propagation_delay:     u32,
    o2i_building_penetr_loss_ind: u32,
    o2i_car_penetr_loss_ind:      u32,
    force_los_prob_indoor:        f32,  // force_los_prob[0]
    force_los_prob_outdoor:       f32,  // force_los_prob[1]
    _p:                           vec2<f32>,
}

struct SimConfig {
    center_freq_hz: f32,
    _p0: f32, _p1: f32, _p2: f32,
}

struct CmnLinkParams {
    sqrtCorrMatLos:  array<f32, 49>,  // 7x7 lower-triangular
    sqrtCorrMatNlos: array<f32, 36>,  // 6x6
    sqrtCorrMatO2i:  array<f32, 36>,  // 6x6
    mu_K:            array<f32, 4>,  // [NLOS, LOS, O2I] + pad
    sigma_K:         array<f32, 4>,
    mu_lgDS:         array<f32, 4>,  // [NLOS, LOS, O2I] + pad
    sigma_lgDS:      array<f32, 4>,
    mu_lgASD:        array<f32, 4>,  // [NLOS, LOS, O2I] + pad
    sigma_lgASD:     array<f32, 4>,
    mu_lgASA:        array<f32, 4>,  // [NLOS, LOS, O2I] + pad
    sigma_lgASA:     array<f32, 4>,
    mu_lgZSA:        array<f32, 4>,  // [NLOS, LOS, O2I] + pad
    sigma_lgZSA:     array<f32, 4>,
    mu_lgDT:         array<f32, 4>,  // delta_tau means: [NLOS, LOS, O2I] + pad
    sigma_lgDT:      array<f32, 4>,  // delta_tau sigmas
    lgfc:            f32,
    _pad:            array<u32, 2>,  // pad to 16-byte boundary
}

struct LinkParams {
    d2d: f32, d2d_in: f32, d2d_out: f32, d3d: f32,
    d3d_in: f32, d3d_out: f32,
    phi_LOS_AOD: f32, phi_LOS_AOA: f32,
    theta_LOS_ZOD: f32, theta_LOS_ZOA: f32,
    losInd: u32, pathloss: f32,
    SF: f32, K: f32, DS: f32, ASD: f32, ASA: f32, ZSD: f32, ZSA: f32,
    mu_lgZSD: f32, sigma_lgZSD: f32, mu_offset_ZOD: f32,
    delta_tau: f32,  // Excess delay per 3GPP TR 38.901 Table 7.6.9-1
}

// Uniforms for calLinkParamKernel
struct LinkParamUniforms {
    maxX: f32, minX: f32, maxY: f32, minY: f32,
    nSite: u32, nUT: u32, nSectorPerSite: u32,
    updatePLAndPenetrationLoss: u32,  // bool as u32
    updateAllLSPs: u32,
    updateLosState: u32,
    updateOptionalPl: u32,    // optional PL indicator
    nX: i32, nY: i32,   // pre-computed final CRN grid dimensions
}

// Uniforms for generateSingleCRNKernel
struct CRNGenUniforms {
    maxX: f32, minX: f32, maxY: f32, minY: f32,
    corrDist:        f32,
    maxRngStates:    u32,
    outputGridOffset: u32,
    _pad:            u32,
    nX:   u32,   // grid width  in pixels
    nY:   u32,   // grid height in pixels
    step: f32,   // metres per pixel
    _pad2: u32,
    boundX: f32,  // = maxX - minX (for padded grid calc)
    boundY: f32,  // = maxY - minY (for padded grid calc)
    rowOffset:    u32,  // Y-axis offset for chunking (in pixels)
    chunkY:       u32,  // number of rows in this chunk (0 = full grid)
};

// Uniforms for normalizeCRNGridsKernel
struct NormUniforms {
    totalElements: u32,
    gridOffset:    u32,   // element offset within the flat buffer
    _pad0: u32, _pad1: u32,
}

// RNG state — xoshiro128** (replaces curandState)
struct RngState { s0: u32, s1: u32, s2: u32, s3: u32 }

// ── RNG helpers ────────────────────────────────────────────────────────────
fn rotl32(x: u32, k: u32) -> u32 { return (x << k) | (x >> (32u - k)); }

fn rng_next(s: ptr<function, RngState>) -> u32 {
    let result = rotl32((*s).s1 * 5u, 7u) * 9u;
    let t = (*s).s1 << 9u;
    (*s).s2 ^= (*s).s0;
    (*s).s3 ^= (*s).s1;
    (*s).s1 ^= (*s).s2;
    (*s).s0 ^= (*s).s3;
    (*s).s2 ^= t;
    (*s).s3 = rotl32((*s).s3, 11u);
    return result;
}

fn rand_uniform(s: ptr<function, RngState>) -> f32 {
    return f32(rng_next(s)) * (1.0 / 4294967296.0);
}

// Box-Muller normal (replaces curand_normal)
fn rand_normal(s: ptr<function, RngState>) -> f32 {
    let u1 = max(rand_uniform(s), 1.175494e-38);
    let u2 = rand_uniform(s);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

// ── Device-equivalent math helpers ────────────────────────────────────────
fn log10(x: f32) -> f32 { return log(x) / log(10.0); }

fn cal_los_prob(scenario: u32, d_2d_out: f32, h_ut: f32,
               force_indoor: f32, force_outdoor: f32,
               outdoor_ind: u32) -> f32 {
    let forced = select(force_indoor, force_outdoor, outdoor_ind == 1u);
    if forced >= 0.0 && forced <= 1.0 { return forced; }
    switch scenario {
        case SCENARIO_UMA: {
            if d_2d_out <= 18.0 { return 1.0; }
            let c = select(0.0, pow((h_ut - 13.0) / 10.0, 1.5), h_ut > 13.0);
            return ((18.0 / d_2d_out) + exp(-d_2d_out / 63.0) * (1.0 - 18.0 / d_2d_out)) *
                   (1.0 + c * 1.25 * pow(d_2d_out / 100.0, 3.0) * exp(-d_2d_out / 150.0));
        }
        case SCENARIO_UMI: {
            if d_2d_out <= 18.0 { return 1.0; }
            return (18.0 / d_2d_out) + exp(-d_2d_out / 36.0) * (1.0 - 18.0 / d_2d_out);
        }
        case SCENARIO_RMA: {
            if d_2d_out <= 10.0 { return 1.0; }
            return exp(-(d_2d_out - 10.0) / 1000.0);
        }
        default: { return 0.0; }
    }
}

fn uma_los_pl(d2d: f32, d3d: f32, h_bs: f32, h_ut: f32, fc: f32,
              s: ptr<function, RngState>) -> f32 {
    let dv = max(d2d, 10.0);
    let g  = select(0.0, 1.25 * pow(dv / 100.0, 3.0) * exp(-dv / 150.0), dv > 18.0);
    let c  = select(0.0, pow((h_ut - 13.0) / 10.0, 1.5) * g, h_ut >= 13.0);
    var h_e: f32;
    if rand_uniform(s) <= 1.0 / (1.0 + c) {
        h_e = 1.0;
    } else {
        let n = i32((h_ut - 1.5 - 12.0) / 3.0) + 1;
        h_e = 12.0 + f32(clamp(i32(rand_uniform(s) * f32(n)), 0, n - 1)) * 3.0;
    }
    let d_bp = 4.0 * (h_bs - h_e) * (h_ut - h_e) * fc * (10.0 / 3.0);
    let pl1  = 28.0 + 22.0 * log10(d3d) + 20.0 * log10(fc);
    let pl2  = 28.0 + 40.0 * log10(d3d) + 20.0 * log10(fc)
             - 9.0  * log10(d_bp * d_bp + (h_bs - h_ut) * (h_bs - h_ut));
    return select(pl2, pl1, dv <= d_bp);
}

fn umi_los_pl(d2d: f32, d3d: f32, h_bs: f32, h_ut: f32, fc: f32) -> f32 {
    let d_bp = 4.0 * h_bs * h_ut * fc * (10.0 / 3.0);
    let pl1  = 32.4 + 21.0 * log10(d3d) + 20.0 * log10(fc);
    let pl2  = 32.4 + 40.0 * log10(d3d) + 20.0 * log10(fc)
             - 9.5  * log10(d_bp * d_bp + (h_bs - h_ut) * (h_bs - h_ut));
    return select(pl2, pl1, d2d <= d_bp);
}

fn rma_los_pl(d2d: f32, d3d: f32, h_bs: f32, h_ut: f32, fc: f32) -> f32 {
    let d_bp = 2.0 * PI * h_bs * h_ut * fc * (10.0 / 3.0);
    let h    = 5.0;
    let pl1  = 20.0 * log10(40.0 * PI * d3d * fc / 3.0)
             + min(0.03  * pow(h, 1.72), 10.0)    * log10(d3d)
             - min(0.044 * pow(h, 1.72), 14.77)
             + 0.002 * log10(h) * d3d;
    return select(pl1 + 40.0 * log10(d3d / d_bp), pl1, d2d <= d_bp);
}

fn cal_pl(cell_loc: Vec3f, ut_loc: Vec3f,
          scenario: u32, fc: f32, is_los: bool, optional_pl: bool,
          s: ptr<function, RngState>) -> f32 {
    let dx = cell_loc.x - ut_loc.x;
    let dy = cell_loc.y - ut_loc.y;
    let dz = cell_loc.z - ut_loc.z;
    let d3d = sqrt(dx*dx + dy*dy + dz*dz);
    let d2d = sqrt(dx*dx + dy*dy);
    let h_bs = cell_loc.z;  let h_ut = ut_loc.z;
    if is_los {
        switch scenario {
            case SCENARIO_UMA: { return uma_los_pl(d2d, d3d, h_bs, h_ut, fc, s); }
            case SCENARIO_UMI: { return umi_los_pl(d2d, d3d, h_bs, h_ut, fc); }
            case SCENARIO_RMA: { return rma_los_pl(d2d, d3d, h_bs, h_ut, fc); }
            default:           { return 0.0; }
        }
    } else if optional_pl {
        switch scenario {
            case SCENARIO_UMA: { return 32.4 + 20.0*log10(fc) + 30.0*log10(d3d); }
            case SCENARIO_UMI: { return 32.4 + 20.0*log10(fc) + 31.9*log10(d3d); }
            case SCENARIO_RMA: { return rma_los_pl(d2d, d3d, h_bs, h_ut, fc); }
            default:           { return 0.0; }
        }
    } else {
        switch scenario {
            case SCENARIO_UMA: {
                let pl1 = uma_los_pl(d2d, d3d, h_bs, h_ut, fc, s);
                let nlos = 13.54 + 39.08*log10(d3d) + 20.0*log10(fc)
                         - 0.6*(h_ut - 1.5);
                return max(pl1, nlos);
            }
            case SCENARIO_UMI: {
                let pl1 = umi_los_pl(d2d, d3d, h_bs, h_ut, fc);
                return max(pl1, 32.4 + 20.0 * log10(fc) + 31.9 * log10(d3d));
            }
            case SCENARIO_RMA: {
                let pl1 = rma_los_pl(d2d, d3d, h_bs, h_ut, fc);
                let W = 20.0; let h = 5.0;
                let nlos = 161.04 - 7.1*log10(W) + 7.5*log10(h)
                         - (24.37 - 3.7*pow(h/h_bs, 2.0)) * log10(h_bs)
                         + (43.42 - 3.1*log10(h_bs)) * (log10(d3d) - 3.0)
                         + 20.0*log10(fc)
                         - (3.2*pow(log10(11.75*h_ut), 2.0) - 4.97);
                return max(pl1, nlos);
            }
            default: { return 0.0; }
        }
    }
}

fn cal_sf_std(scenario: u32, is_los: bool, is_indoor: bool,
              fc: f32, d2d: f32, h_bs: f32, h_ut: f32, optionalPl: u32) -> f32 {
    if is_los {
        switch scenario {
            case SCENARIO_UMA: {
                // TR 36.777 Table B-3: UMa-AV LOS
                if (h_ut > 22.5) { return 4.64 * exp(-0.0066 * h_ut); }
                return select(7.0, 4.0, fc < 6e9f);
            }
            case SCENARIO_UMI: {
                // TR 36.777 Table B-3: UMi-AV LOS
                if (h_ut > 22.5) { return max(5.0 * exp(-0.01 * h_ut), 2.0); }
                return select(7.0, 4.0, fc < 6e9f);
            }
            case SCENARIO_RMA: {
                // TR 36.777 Table B-3: RMa-AV LOS
                if is_indoor { return 8.0; }
                let d_bp = 2.0 * PI * h_bs * h_ut * fc / 3.0e8;
                return select(6.0, 4.0, d2d <= d_bp);
            }
            default: { return 4.0; }
        }
    } else {
        switch scenario {
            case SCENARIO_UMA: {
                // TR 36.777 Table B-3: UMa-AV NLOS
                if (h_ut > 22.5) { return 6.0; }
                return select(7.0, select(7.8, 6.0, optionalPl != 0u), fc < 6e9f);
            }
            case SCENARIO_UMI: {
                // TR 36.777 Table B-3: UMi-AV NLOS
                if (h_ut > 22.5) { return 8.0; }
                return select(7.0, select(8.2, 7.82, optionalPl != 0u), fc < 6e9f);
            }
            default:           { return 6.0; }
        }
    }
}

// ── fill_crn_kernel: group 0 ──────────────────────────────────────────────
@group(0) @binding(0) var<uniform>             fill_uni:    CRNGenUniforms;
@group(0) @binding(1) var<storage, read_write> fill_temp:   array<f32>;
@group(0) @binding(2) var<storage, read_write> fill_rng:    array<RngState>;

var<workgroup> wg_filter: array<f32, MAX_FILTER_LEN>;

@compute @workgroup_size(256, 1, 1)
fn fill_crn_kernel(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let uni      = fill_uni;
    // step = metres per grid cell. The filter pad and grid extent are both in
    // *pixels*, so divide the metre-valued corrDist / bound by step.
    let step     = max(uni.step, 1.0);
    // Correlation distance and pad expressed in pixels.
    let corr_px  = uni.corrDist / step;
    let D        = select(3.0 * corr_px, 0.0, corr_px == 0.0);
    let iD       = u32(D);
    // padded grid = round(bound/step + 1 + 2*D) matching the new CPU sizing.
    let boundF   = uni.boundX;  // = maxX - minX (metres)
    let boundFY  = uni.boundY;  // = maxY - minY (metres)
    // Use u32(x + 0.5) for rounding positive floats to nearest integer.
    let padded_n = select(u32(0u), u32(boundF / step + 1.0 + 2.0 * D + 0.5), iD > 0u);
    let padded_m = select(u32(0u), u32(boundFY / step + 1.0 + 2.0 * D + 0.5), iD > 0u);
    let padded_nx = padded_n;
    let padded_ny = padded_m;
    let total_pad = padded_nx * padded_ny;

    // Use chunkY to compute total elements for this chunk
    let chunk_ny = select(padded_ny, uni.chunkY, uni.chunkY > 0u);
    let chunk_total = padded_nx * chunk_ny;

    let tid = gid.x + uni.rowOffset;
    let total_threads = 128u * 256u;
    let pad_per_thr   = (chunk_total + total_threads - 1u) / total_threads;
    let rng_id        = tid % uni.maxRngStates;
    var rng           = fill_rng[rng_id];

    for (var e = 0u; e < pad_per_thr; e++) {
        let idx = tid * pad_per_thr + e;
        if idx < chunk_total { fill_temp[idx] = rand_normal(&rng); }
    }
    fill_rng[rng_id] = rng;
}

// ── convolve_crn_kernel: group 1 ──────────────────────────────────────────
@group(0) @binding(3) var<storage, read_write> conv_output: array<f32>;

// ── Entry point 2: convolve fill_temp → conv_output ────────────────────────
@compute @workgroup_size(256, 1, 1)
fn convolve_crn_kernel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {
    let uni      = fill_uni;
    // Pixel-domain correlation length: corrDist (m) / step (m/px). All
    // sizing below works in pixels.
    let step     = max(uni.step, 1.0);
    let corr_px  = select(uni.corrDist / step, 0.0, uni.corrDist == 0.0);
    let D        = 3.0 * corr_px;
    let iD       = u32(D);
    let L        = select(2u * iD + 1u, 1u, corr_px == 0.0);
    // final grid = round(bound/step + 1 + 2*D) — valid pixels per channel.
    let final_nx = select(u32(0u), u32(uni.boundX / step + 1.0 + 2.0 * D), iD > 0u);
    let final_ny = select(u32(0u), u32(uni.boundY / step + 1.0 + 2.0 * D), iD > 0u);
    let padded_nx = final_nx + L - 1u;
    let padded_ny = final_ny + L - 1u;
    // Each per-(site, LSP) slot in conv_output is sized uni.nX × uni.nY (the
    // maximum corrDist's final grid). The reader (lsp_at_loc_*) indexes
    // cells at stride uni.nX, so we write at the same stride and only emit
    // the channel's `final_nx × final_ny` valid sub-rectangle. Cells past
    // (final_nx-1, final_ny-1) stay zero — they're outside the UE deployment
    // disk anyway and would never be sampled.
    let stride_nx = uni.nX;
    let chunk_ny  = select(final_ny, uni.chunkY, uni.chunkY > 0u);
    let valid_total = final_nx * chunk_ny;

    let total_threads = 128u * 256u;
    let elems_per_thr = (valid_total + total_threads - 1u) / total_threads;
    let tid = gid.x + gid.y * 256u + uni.rowOffset;

    if lid.x == 0u {
        if corr_px == 0.0 {
            wg_filter[0] = 1.0;
        } else {
            var wsum = 0.0;
            for (var k = 0u; k < L; k++) {
                wg_filter[k] = exp(-abs(f32(k) - f32(iD)) / corr_px);
                wsum += wg_filter[k];
            }
            for (var k = 0u; k < L; k++) {
                wg_filter[k] /= wsum;
            }
        }
    }
    workgroupBarrier();

    for (var e = 0u; e < elems_per_thr; e++) {
        let lin = tid * elems_per_thr + e;
        if lin >= valid_total { break; }
        // (ci, cj) now ranges only over the valid output rectangle.
        let ci = lin / final_ny;   // 0 .. final_nx-1
        let cj = lin % final_ny;   // 0 .. final_ny-1
        var s  = 0.0;
        if corr_px == 0.0 {
            s = fill_temp[ci * padded_ny + cj];
        } else {
            for (var di = 0u; di < L; di++) {
                for (var dj = 0u; dj < L; dj++) {
                    let pi = ci + di; let pj = cj + dj;
                    if pi < padded_nx && pj < padded_ny {
                        s += wg_filter[di] * wg_filter[dj] *
                             fill_temp[pi * padded_ny + pj];
                    }
                }
            }
        }
        // Write at the slot's stride (uni.nX), not the channel's packed
        // stride — this is what `lsp_at_loc_*` indexes with.
        conv_output[uni.outputGridOffset + ci * stride_nx + cj] = s;
    }
}

// ── Bindings — normalize_crn_kernel  (4, 5) ────────────────────────────
@group(0) @binding(4) var<uniform>             norm_uni:    NormUniforms;
@group(0) @binding(5) var<storage, read_write> norm_buffer: array<f32>;

var<workgroup> wg_sum:  array<f32, 256>;
var<workgroup> wg_sum2: array<f32, 256>;

// normalizeCRNGridsKernel
// Dispatch: grid = (1, 1, 1),  workgroup = (256, 1, 1)
// One dispatch per grid; set gridOffset in the uniform.
// Note: normalize uses a single workgroup that loops internally, so no 2D change needed.
@compute @workgroup_size(256, 1, 1)
fn normalize_crn_kernel(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid   = lid.x;
    let total = norm_uni.totalElements;
    let base  = norm_uni.gridOffset;

    // Pass 1: accumulate sum and sum-of-squares
    var local_sum  = 0.0;
    var local_sum2 = 0.0;
    for (var i = tid; i < total; i += 256u) {
        let v = norm_buffer[base + i];
        local_sum  += v;
        local_sum2 += v * v;
    }
    wg_sum[tid]  = local_sum;
    wg_sum2[tid] = local_sum2;
    workgroupBarrier();

    var s = 128u;
    while (s > 0u) {
        if tid < s {
            wg_sum[tid]  += wg_sum[tid + s];
            wg_sum2[tid] += wg_sum2[tid + s];
        }
        workgroupBarrier();
        s >>= 1u;
    }

    if tid == 0u {
        let mean     = wg_sum[0] / f32(total);
        let variance = wg_sum2[0] / f32(total) - mean * mean;
        wg_sum[0] = mean;
        let safeVariance = max(variance, 1e-10);
        wg_sum[1] = 1.0 / sqrt(safeVariance);
    }
    workgroupBarrier();

    let mean    = wg_sum[0];
    let inv_std = wg_sum[1];
    for (var i = tid; i < total; i += 256u) {
        norm_buffer[base + i] = (norm_buffer[base + i] - mean) * inv_std;
    }
}

// ── Bindings — calLinkParamKernel ──────────────────────────────────────────
@group(0) @binding(6)  var<uniform>            link_uni:         LinkParamUniforms;
@group(0) @binding(7)  var<storage, read>       cell_params:      array<CellParam>;
@group(0) @binding(8)  var<storage, read>       ut_params:        array<UtParam>;
@group(0) @binding(9)  var<storage, read>       sys_config:       array<SystemLevelConfig>;
@group(0) @binding(10)  var<storage, read>       sim_config:       array<SimConfig>;
@group(0) @binding(11)  var<storage, read>       cmn_link:         array<CmnLinkParams>;
@group(0) @binding(12)  var<storage, read_write> link_params:      array<LinkParams>;
@group(0) @binding(13)  var<storage, read_write> rng_states_lp:    array<RngState>;
// Combined CRN buffer: concat(LOS data | NLOS data | O2I data) along the
// element axis. `crn_offsets` is similarly concat(losOffs | nlosOffs |
// o2iOffs) — losOffs has nSite*8 entries pointing into the LOS region,
// nlosOffs has nSite*7 pointing into the NLOS region, etc. The host
// pre-bakes the per-section base offset into the values so a single
// indexed lookup `crn_data[crn_offsets[idx] + xy]` finds the right
// sample without per-call section arithmetic.
//
// Why: the previous layout used 6 separate storage bindings
// (3 data + 3 offsets), which together with the other LSP inputs put
// cal_link_param_kernel at 13 storage buffers per stage — over the
// WebGPU spec default of 10. Folding to 2 bindings (1 data + 1
// offsets) brings the kernel to 9 SSBOs/stage, which is the limit
// Dawn enforces by default.
@group(0) @binding(14)  var<storage, read>       crn_data:         array<f32>;
@group(0) @binding(15)  var<storage, read>       crn_offsets:      array<u32>; // nSite*(8+7+7)

// Bilinear interpolation into the combined CRN buffer. `grid_idx` is
// the index WITHIN that section's offsets table (0..nSite*8 for LOS,
// 0..nSite*7 for NLOS/O2I); the section base added below maps it to
// the correct slot in the concatenated `crn_offsets` array.
fn lsp_at_loc_los(grid_idx: u32, x: f32, y: f32, n_x: i32, n_y: i32) -> f32 {
    let uni = link_uni;
    // LOS section starts at index 0 in crn_offsets.
    let base = crn_offsets[grid_idx];
    let nx = n_x; let ny = n_y;
    let norm_x = clamp((x - uni.minX) / (uni.maxX - uni.minX), 0.0, 1.0);
    let norm_y = clamp((y - uni.minY) / (uni.maxY - uni.minY), 0.0, 1.0);
    let gx = norm_x * f32(nx - 1);  let gy = norm_y * f32(ny - 1);
    let x0 = i32(floor(gx));        let y0 = i32(floor(gy));
    let x1 = min(x0 + 1, nx - 1);  let y1 = min(y0 + 1, ny - 1);
    let dx = gx - f32(x0);          let dy = gy - f32(y0);
    let v00 = crn_data[base + u32(y0*nx + x0)];
    let v10 = crn_data[base + u32(y0*nx + x1)];
    let v01 = crn_data[base + u32(y1*nx + x0)];
    let v11 = crn_data[base + u32(y1*nx + x1)];
    return mix(mix(v00, v10, dx), mix(v01, v11, dx), dy);
}

fn lsp_at_loc_nlos(grid_idx: u32, x: f32, y: f32, n_x: i32, n_y: i32) -> f32 {
    let uni = link_uni;
    // NLOS section in crn_offsets starts at nSite*8.
    let base = crn_offsets[grid_idx + uni.nSite * 8u];
    let nx = n_x; let ny = n_y;
    let norm_x = clamp((x - uni.minX) / (uni.maxX - uni.minX), 0.0, 1.0);
    let norm_y = clamp((y - uni.minY) / (uni.maxY - uni.minY), 0.0, 1.0);
    let gx = norm_x * f32(nx - 1); let gy = norm_y * f32(ny - 1);
    let x0 = i32(floor(gx)); let y0 = i32(floor(gy));
    let x1 = min(x0+1, nx-1); let y1 = min(y0+1, ny-1);
    let dx = gx - f32(x0); let dy = gy - f32(y0);
    let v00 = crn_data[base + u32(y0*nx+x0)]; let v10 = crn_data[base + u32(y0*nx+x1)];
    let v01 = crn_data[base + u32(y1*nx+x0)]; let v11 = crn_data[base + u32(y1*nx+x1)];
    return mix(mix(v00, v10, dx), mix(v01, v11, dx), dy);
}

fn lsp_at_loc_o2i(grid_idx: u32, x: f32, y: f32, n_x: i32, n_y: i32) -> f32 {
    let uni = link_uni;
    // O2I section in crn_offsets starts at nSite*(8+7) = nSite*15.
    let base = crn_offsets[grid_idx + uni.nSite * 15u];
    let nx = n_x; let ny = n_y;
    let norm_x = clamp((x - uni.minX) / (uni.maxX - uni.minX), 0.0, 1.0);
    let norm_y = clamp((y - uni.minY) / (uni.maxY - uni.minY), 0.0, 1.0);
    let gx = norm_x * f32(nx - 1); let gy = norm_y * f32(ny - 1);
    let x0 = i32(floor(gx)); let y0 = i32(floor(gy));
    let x1 = min(x0+1, nx-1); let y1 = min(y0+1, ny-1);
    let dx = gx - f32(x0); let dy = gy - f32(y0);
    let v00 = crn_data[base + u32(y0*nx+x0)]; let v10 = crn_data[base + u32(y0*nx+x1)];
    let v01 = crn_data[base + u32(y1*nx+x0)]; let v11 = crn_data[base + u32(y1*nx+x1)];
    return mix(mix(v00, v10, dx), mix(v01, v11, dx), dy);
}

// ── calLinkParamKernel ─────────────────────────────────────────────────────
// Dispatch: grid = (nSite, ceil(nUT/256), 1), workgroup = (256, 1, 1)
@compute @workgroup_size(256, 1, 1)
fn cal_link_param_kernel(
    @builtin(workgroup_id)       wg_id:  vec3<u32>,
    @builtin(local_invocation_id) li_id: vec3<u32>,
    @builtin(num_workgroups)     n_wg:   vec3<u32>,
) {
    let uni       = link_uni;
    let site_idx  = wg_id.x;
    let ue_idx    = wg_id.y * 256u + li_id.x;
    if ue_idx >= uni.nUT { return; }

    let link_idx  = site_idx * uni.nUT + ue_idx;
    let first_sec = site_idx * u32(uni.nSectorPerSite);   // sector-0 of this site
    let cell      = cell_params[first_sec];
    let ut        = ut_params[ue_idx];

    // ── Distances ────────────────────────────────────────────────────────
    let dx = cell.loc.x - ut.loc.x;  let dy = cell.loc.y - ut.loc.y;
    let d2d     = sqrt(dx*dx + dy*dy);
    let d2d_in  = ut.d_2d_in;
    let d2d_out = d2d - d2d_in;
    let vert    = cell.loc.z - ut.loc.z;
    let d3d     = sqrt(d2d*d2d + vert*vert);
    let d3d_in  = select(0.0, d3d * d2d_in / d2d, d2d > 0.0);
    let d3d_out = d3d - d3d_in;

    link_params[link_idx].d2d     = d2d;
    link_params[link_idx].d2d_in  = d2d_in;
    link_params[link_idx].d2d_out = d2d_out;
    link_params[link_idx].d3d     = d3d;
    link_params[link_idx].d3d_in  = d3d_in;
    link_params[link_idx].d3d_out = d3d_out;

    // ── LOS angles ───────────────────────────────────────────────────────
    let phi_aod = atan2(ut.loc.y - cell.loc.y, ut.loc.x - cell.loc.x) * (180.0 / PI);
    var phi_aoa = phi_aod + 180.0;
    if phi_aoa > 180.0 { phi_aoa -= 360.0; }
    let h_diff    = cell.loc.z - ut.loc.z;
    let theta_zod = (PI - acos(h_diff / d3d)) * (180.0 / PI);

    link_params[link_idx].phi_LOS_AOD   = phi_aod;
    link_params[link_idx].phi_LOS_AOA   = phi_aoa;
    link_params[link_idx].theta_LOS_ZOD = theta_zod;
    link_params[link_idx].theta_LOS_ZOA = 180.0 - theta_zod;

    // ── RNG state ────────────────────────────────────────────────────────
    //let global_tid = wg_id.x * n_wg.y * 256u + wg_id.y * 256u + li_id.x;
    //var rng = rng_states_lp[global_tid];
    var rng = rng_states_lp[link_idx];

    let sc = sys_config[0];
    let fc = sim_config[0].center_freq_hz;

    // ── LOS indicator ────────────────────────────────────────────────────
    if uni.updateLosState != 0u {
        let p = cal_los_prob(sc.scenario, d2d_out, ut.loc.z,
                             sc.force_los_prob_indoor, sc.force_los_prob_outdoor,
                             ut.outdoor_ind);
        link_params[link_idx].losInd = select(0u, 1u, rand_uniform(&rng) <= p);
    }

    let is_los    = link_params[link_idx].losInd != 0u;
    let is_indoor = ut.outdoor_ind == 0u;
    let lsp_idx   = select(select(1u, 0u, is_los), 2u, is_indoor); // LOS=0,NLOS=1,O2I=2

    // ── Path loss ────────────────────────────────────────────────────────
    if uni.updatePLAndPenetrationLoss != 0u || uni.updateAllLSPs != 0u {
        var pl = cal_pl(Vec3f(cell.loc.x, cell.loc.y, cell.loc.z, 0.0),
                        Vec3f(ut.loc.x, ut.loc.y, ut.loc.z, 0.0), sc.scenario,
                        fc / 1e9, is_los, false, &rng);
        if ut.outdoor_ind == 0u {
            pl += ut.o2i_penetration_loss;
        }
        link_params[link_idx].pathloss = pl;
    }

    // ── LSPs ─────────────────────────────────────────────────────────────
    if uni.updateAllLSPs != 0u || uni.updatePLAndPenetrationLoss != 0u {
        let cl = cmn_link[0];
        let nx = uni.nX;  let ny = uni.nY;
        let ux = ut.loc.x;  let uy = ut.loc.y;
        // The dispatch is (nSite, ceil(nUT/256), 1) — wg_id.x is already a
        // site index, not a cell index. Dividing by nSectorPerSite was a bug
        // left over from a per-cell dispatch: it (a) made sites 0/1/2 all
        // read CRN-site-0, sites 3/4/5 share CRN-site-1, etc., and (b) had
        // the upper-third of sites reading uninitialised slots, producing
        // extreme cv[K_IDX] values that underflow K to 0 for ~5 % of links.
        let true_site = site_idx;
        let s7 = true_site * 7u;
        let s8 = true_site * 8u;
        let s6 = true_site * 6u;

        var ucv: array<f32, 8>;  // uncorrelated vars [SF,K,DS,ASD,ASA,ZSD,ZSA,DT]
        if is_indoor {
            // O2I grid: nSite * 7 cols (stride-7 contiguous) — cols 0-5 are SF/DS/ASD/ASA/ZSD/ZSA.
            // The O2I LSP set in TR 38.901 covers SF/DS/ASD/ASA/ZSD/ZSA (no K), so we draw those
            // from the O2I grid. The K-factor on indoor-LOS links comes from the OUTDOOR LOS path
            // that penetrates the building -- draw cv[K_IDX] from the LOS grid when is_los, and
            // leave it 0 otherwise. The Cholesky pass below uses O2I_MATRIX_SIZE = 7 and doesn't
            // touch K_IDX, so the raw LOS draw flows through unchanged.
            ucv[SF_IDX]  = lsp_at_loc_o2i(s7+0u, ux, uy, nx, ny);
            ucv[K_IDX]   = select(0.0, lsp_at_loc_los(s8+1u, ux, uy, nx, ny), is_los);
            ucv[DS_IDX]  = lsp_at_loc_o2i(s7+1u, ux, uy, nx, ny);
            ucv[ASD_IDX] = lsp_at_loc_o2i(s7+2u, ux, uy, nx, ny);
            ucv[ASA_IDX] = lsp_at_loc_o2i(s7+3u, ux, uy, nx, ny);
            ucv[ZSD_IDX] = lsp_at_loc_o2i(s7+4u, ux, uy, nx, ny);
            ucv[ZSA_IDX] = lsp_at_loc_o2i(s7+5u, ux, uy, nx, ny);
            ucv[D_T_IDX] = lsp_at_loc_o2i(s7+6u, ux, uy, nx, ny);   // delta_tau CRN (O2I col 6)
        } else if is_los {
            ucv[SF_IDX]  = lsp_at_loc_los(s8+0u, ux, uy, nx, ny);
            ucv[K_IDX]   = lsp_at_loc_los(s8+1u, ux, uy, nx, ny);
            ucv[DS_IDX]  = lsp_at_loc_los(s8+2u, ux, uy, nx, ny);
            ucv[ASD_IDX] = lsp_at_loc_los(s8+3u, ux, uy, nx, ny);
            ucv[ASA_IDX] = lsp_at_loc_los(s8+4u, ux, uy, nx, ny);
            ucv[ZSD_IDX] = lsp_at_loc_los(s8+5u, ux, uy, nx, ny);
            ucv[ZSA_IDX] = lsp_at_loc_los(s8+6u, ux, uy, nx, ny);
            // delta_tau CRN (LOS col 7) — contiguous stride 8 grid
            ucv[D_T_IDX] = lsp_at_loc_los(s8+7u, ux, uy, nx, ny);
        } else {
            // NLOS grid: nSite * 7 cols (stride-7 contiguous) — cols 0-5 are SF/DS/ASD/ASA/ZSD/ZSA
            ucv[SF_IDX]  = lsp_at_loc_nlos(s7+0u, ux, uy, nx, ny);
            ucv[K_IDX]   = 0.0;
            ucv[DS_IDX]  = lsp_at_loc_nlos(s7+1u, ux, uy, nx, ny);
            ucv[ASD_IDX] = lsp_at_loc_nlos(s7+2u, ux, uy, nx, ny);
            ucv[ASA_IDX] = lsp_at_loc_nlos(s7+3u, ux, uy, nx, ny);
            ucv[ZSD_IDX] = lsp_at_loc_nlos(s7+4u, ux, uy, nx, ny);
            ucv[ZSA_IDX] = lsp_at_loc_nlos(s7+5u, ux, uy, nx, ny);
            ucv[D_T_IDX] = lsp_at_loc_nlos(s7+6u, ux, uy, nx, ny);   // delta_tau CRN (NLOS col 6)
        }

        // Cholesky multiply: corr_px = sqrtCorrMat · uncorr  (lower-triangular)
        var cv: array<f32, 7>;
        if is_indoor {
            for (var i = 0u; i < O2I_MATRIX_SIZE; i++) {
                for (var j = 0u; j <= i; j++) {
                    let si = select(i, i+1u, i >= K_IDX);
                    let sj = select(j, j+1u, j >= K_IDX);
                    cv[si] += cl.sqrtCorrMatO2i[i*O2I_MATRIX_SIZE + j] * ucv[sj];
                }
            }
            // K-factor isn't part of the O2I correlation matrix (only
            // SF/DS/ASD/ASA/ZSD/ZSA), so the loop above skips K_IDX and
            // leaves cv[K_IDX] = 0 even when ucv[K_IDX] holds the LOS
            // draw we filled in for is_los. Pass it through so the
            // K-factor lookup downstream sees a randomised value
            // instead of the constant pow(10, mu_K/10) every link gets.
            cv[K_IDX] = ucv[K_IDX];
        } else if is_los {
            for (var i = 0u; i < LOS_MATRIX_SIZE; i++) {
                for (var j = 0u; j <= i; j++) {
                    cv[i] += cl.sqrtCorrMatLos[i*LOS_MATRIX_SIZE + j] * ucv[j];
                }
            }
        } else {
            for (var i = 0u; i < NLOS_MATRIX_SIZE; i++) {
                for (var j = 0u; j <= i; j++) {
                    let si = select(i, i+1u, i >= K_IDX);
                    let sj = select(j, j+1u, j >= K_IDX);
                    cv[si] += cl.sqrtCorrMatNlos[i*NLOS_MATRIX_SIZE + j] * ucv[sj];
                }
            }
            cv[K_IDX] = 0.0;
        }

        // Heights needed for SF std and ZSD calculations
        let h_ut = ut.loc.z;
        let h_bs = cell.loc.z;

        // Shadow fading
        link_params[link_idx].SF = cv[SF_IDX] *
            cal_sf_std(sc.scenario, is_los, is_indoor, fc, d2d, h_bs, h_ut, link_uni.updateOptionalPl);

        if uni.updateAllLSPs != 0u {
            // K-factor (Rician). Drawn whenever the link is LOS, using
            // the LOS LSP slot (mu_K[0] / sigma_K[0]). Earlier code
            // gated on `lsp_idx == 0u`, but lsp_idx is 2 (O2I) for ANY
            // indoor UE -- so all 291 indoor-LOS links in Phase-1 UMa
            // were forced to K=0, which made the NVIDIA analyser
            // produce empty SINR tables (K=0 -> no Rician term ->
            // CIR-derived RX signal NaN). Outdoor LOS links (lsp_idx==0)
            // were unaffected, masking this on simple PL/CL plots.
            // mu_K / sigma_K are stored in dB; convert once to linear
            // here for downstream cluster/CIR kernels. NLOS gets K=0.
            if is_los {
                let K_lin = pow(10.0,
                    (cl.mu_K[0] + cv[K_IDX] * cl.sigma_K[0]) / 10.0);
                link_params[link_idx].K = K_lin;
            } else {
                link_params[link_idx].K = 0.0;
            }

            // DS (convert s → ns via +9)
            link_params[link_idx].DS = pow(10.0,
                cv[DS_IDX] * cl.sigma_lgDS[lsp_idx] + cl.mu_lgDS[lsp_idx] + 9.0);

            // ASD / ASA
            link_params[link_idx].ASD = min(
                pow(10.0, cv[ASD_IDX]*cl.sigma_lgASD[lsp_idx] + cl.mu_lgASD[lsp_idx]), 104.0);
            link_params[link_idx].ASA = min(
                pow(10.0, cv[ASA_IDX]*cl.sigma_lgASA[lsp_idx] + cl.mu_lgASA[lsp_idx]), 104.0);

            // ZSD — scenario-specific mu/sigma
            let h_ut = ut.loc.z;
            let h_bs = cell.loc.z;
            var mu_zsd: f32;  var sigma_zsd: f32;  var mu_off: f32;
            switch sc.scenario {
                case SCENARIO_UMA: {
                    if is_los {
                        mu_zsd    = max(-0.5, -2.1*(d2d/1000.0) - 0.01*(h_ut-1.5) + 0.75);
                        sigma_zsd = 0.4;  mu_off = 0.0;
                    } else {
                        mu_zsd    = max(-0.5, -2.1*(d2d/1000.0) - 0.01*(h_ut-1.5) + 0.9);
                        sigma_zsd = 0.49;
                        mu_off    = pow(10.0, (0.208*cl.lgfc - 0.782) * log10(max(25.0, d2d))
                                                  - 0.13*cl.lgfc + 2.03)
                                           - 0.07*(h_ut - 1.5);
                    }
                }
                case SCENARIO_UMI: {
                    if is_los {
                        mu_zsd    = max(-0.21, -14.8*(d2d/1000.0) - 0.01*abs(h_ut-h_bs) + 0.83);
                        sigma_zsd = 0.35;  mu_off = 0.0;
                    } else {
                        mu_zsd    = max(-0.5, -3.1*(d2d/1000.0) + 0.01*max(h_ut-h_bs,0.0) + 0.2);
                        sigma_zsd = 0.35;
                        mu_off    = -pow(10.0, -1.5*log10(max(10.0,d2d)) + 3.3);
                    }
                }
                case SCENARIO_RMA: {
                    if is_los {
                        mu_zsd    = max(-1.0, -0.17*(d2d/1000.0) - 0.01*(h_ut-1.5) + 0.22);
                        sigma_zsd = 0.34;  mu_off = 0.0;
                    } else {
                        mu_zsd    = max(-1.0, -0.19*(d2d/1000.0) - 0.01*(h_ut-1.5) + 0.28);
                        sigma_zsd = 0.30;
                        mu_off    = atan((35.0-3.5)/d2d) - atan((35.0-1.5)/d2d);
                    }
                }
                default: { mu_zsd = 0.0; sigma_zsd = 0.5; mu_off = 0.0; }
            }
            link_params[link_idx].mu_lgZSD     = mu_zsd;
            link_params[link_idx].sigma_lgZSD  = sigma_zsd;
            link_params[link_idx].mu_offset_ZOD = mu_off;
            link_params[link_idx].ZSD = min(pow(10.0, cv[ZSD_IDX]*sigma_zsd + mu_zsd), 52.0);

            // ZSA
            link_params[link_idx].ZSA = min(
                pow(10.0, cv[ZSA_IDX]*cl.sigma_lgZSA[lsp_idx] + cl.mu_lgZSA[lsp_idx]), 52.0);

            // Excess delay delta_tau per 3GPP TR 38.901 Table 7.6.9-1
            // LOS: delta_tau = 0 (Eq. 7.6-44). NLOS: lognormal from table values.
            // mu/sigma arrays are indexed by scenario: [UMi=0, UMa=1, RMa=2].
    // delta_tau access: mu_lgDT / sigma_lgDT are in SsCmnParams (scenario-level), not in cluster params
            if uni.updateAllLSPs != 0u && sys_config[0].enable_propagation_delay != 0u {
                if !is_los {
                    // NLOS: lg(Delta Tau) = mu_lgDT[i] + sigma_lgDT[i] * r_DT
                    // Per CUDA reference: r_DT = curand_normal(&localState) i.e. N(0,1)
                    let mu = cl.mu_lgDT[lsp_idx];
                    let sigma = cl.sigma_lgDT[lsp_idx];
                    let r_DT = rand_normal(&rng);  // i.i.d. standard normal
                    let lg_delta_tau = mu + sigma * r_DT;
                    link_params[link_idx].delta_tau = pow(10.0, lg_delta_tau);
                } else {
                    // LOS: delta_tau = 0 (no excess delay per 3GPP Eq. 7.6-44)
                    link_params[link_idx].delta_tau = 0.0;
                }
            }
            // When disabled (sys_config[0].enable_propagation_delay == 0), delta_tau stays 0.
        }
    }

    // Write RNG state back
    //rng_states_lp[global_tid] = rng;
    rng_states_lp[link_idx] = rng;
}


// =============================================================================
// sls_small_scale.wgsl
// Port of NVIDIA sls_chan_small_scale_GPU-10.cu  (Apache-2.0)
// Kernels: cal_cluster_ray_kernel · generate_cir_kernel · generate_cfr_kernel_mode1
// Constants: MAX_CLUSTERS=20  MAX_RAYS=20  NMAXTAPS=24  MAX_UE_ANT_ELEMENTS=8
// =============================================================================

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------
const MAX_CLUSTERS          : u32 = 20u;
const MAX_RAYS              : u32 = 20u;
const MAX_CR                : u32 = 400u;   // MAX_CLUSTERS * MAX_RAYS
const MAX_CR4               : u32 = 1600u;  // MAX_CLUSTERS * MAX_RAYS * 4
const NMAXTAPS              : u32 = 24u;
const MAX_UE_ANT            : u32 = 8u;
const ANT_THETA_SIZE        : u32 = 181u;
const ANT_PHI_SIZE          : u32 = 360u;
const N_SUB_CLUSTER         : u32 = 3u;
const GMAX                  : f32 = 8.0;
const TWO_PI                : f32 = 6.28318530717958647692;
const DEG2RAD               : f32 = 0.01745329251994329577;

// ---------------------------------------------------------------------------
// Structs (must match host C++ layout)
// ---------------------------------------------------------------------------

struct SmallScaleSimConfig {
    sc_spacing_hz              : f32,
    fft_size                   : u32,
    n_prb                      : u32,
    n_prbg                     : u32,
    n_snapshot_per_slot        : u32,
    enable_propagation_delay   : u32,
    disable_small_scale_fading : u32,
    disable_pl_shadowing       : u32,
    optional_cfr_dim           : u32,
    lambda0                    : f32,
    _pad0                      : f32,
    _pad1                      : f32,
}

struct SmallScaleSysConfig {
    enable_propagation_delay : u32,
    disable_small_scale_fading : u32,
    disable_pl_shadowing : u32,
    _pad0 : u32,
}

// Antenna panel config – pattern tables are in separate bindings (arrays too large for inline struct)
struct AntPanelConfig {
    nAnt          : u32,
    antModel      : u32,
    antSize       : array<u32, 5>,   // antSize[0..4]: [nPanels, unused, M, N, P]
    antSpacing    : array<f32, 4>,   // antSpacing[0..3]: [unused, unused, d_h, d_v]
    antPolarAngles: array<f32, 2>,   // antPolarAngles per polarization (max 2)
    // offsets into the flat antTheta / antPhi texture arrays
    thetaOffset   : u32,
    phiOffset     : u32,
    _pad0         : u32,
}

// SspCellParam: tail-padded to 32 B to match the C++ host struct's 16-byte
// alignment. The natural WGSL std430 size would be 20 B, but the host's
// `alignas(16)` rounds it up to 32 B; without the matching padding here,
// the reader sees every (32/20)-th cell shifted into the prior cell's
// orientation field, producing a garbage `antPanelIdx` whose `nAnt` lookup
// returns zero and silently skips the BS antenna loop for those links.
struct SspCellParam {
    antPanelIdx         : u32,
    antPanelOrientation : array<f32, 3>,  // [theta_tilt, phi_tilt, zeta_offset]
    _pad0               : u32,
    _pad1               : u32,
    _pad2               : u32,
    _pad3               : u32,
}

// Same story: WGSL std430 layout is 36 B (4+4+12+12+4), but the host pads
// to 48 B via `alignas(16)`. Match it explicitly here.
struct SspUtParam {
    antPanelIdx         : u32,
    outdoor_ind         : u32,            // 1 = outdoor, 0 = indoor (O2I)
    antPanelOrientation : array<f32, 3>,
    velocity            : array<f32, 3>,
    _pad0               : u32,
    _pad1               : u32,
    _pad2               : u32,
    _pad3               : u32,
}

// Common link parameters (scenario-level, 3 entries: NLOS=0, LOS=1, O2I=2)
struct SsCmnParams {
    mu_lgDS        : array<f32, 3>,
    sigma_lgDS     : array<f32, 3>,
    mu_lgASD       : array<f32, 3>,
    sigma_lgASD    : array<f32, 3>,
    mu_lgASA       : array<f32, 3>,
    sigma_lgASA    : array<f32, 3>,
    mu_lgZSA       : array<f32, 3>,
    sigma_lgZSA    : array<f32, 3>,
    mu_K           : array<f32, 3>,
    sigma_K        : array<f32, 3>,
    r_tao          : array<f32, 3>,
    mu_XPR         : array<f32, 3>,
    sigma_XPR      : array<f32, 3>,
    nCluster       : array<u32, 3>,
    nRayPerCluster : array<u32, 3>,
    C_DS           : array<f32, 3>,
    C_ASD          : array<f32, 3>,
    C_ASA          : array<f32, 3>,
    C_ZSA          : array<f32, 3>,
    xi             : array<f32, 3>,
    C_phi_LOS      : f32,
    C_phi_NLOS     : f32,
    C_phi_O2I      : f32,
    C_theta_LOS    : f32,
    C_theta_NLOS   : f32,
    C_theta_O2I    : f32,
    lgfc           : f32,
    lambda_0       : f32,
    mu_lgDT        : array<f32, 3>,  // delta_tau means: NLOS=0, LOS=1, O2I=2
    sigma_lgDT     : array<f32, 3>,  // delta_tau sigmas
    // 3GPP Table 7.5-5 subcluster ray indices (flat, padded to 10 each)
    raysInSubCluster0    : array<u32, 10>,
    raysInSubCluster1    : array<u32, 10>,
    raysInSubCluster2    : array<u32, 10>,
    raysInSubClusterSizes: array<u32, 3>,
    nSubCluster          : u32,
    nUeAnt               : u32,
    nBsAnt               : u32,
    RayOffsetAngles      : array<f32, 20>,
}

// Cluster parameters – one per link
struct ClusterParams {
    nCluster       : u32,
    nRayPerCluster : u32,
    delays         : array<f32, 20>,
    powers         : array<f32, 20>,
    strongest2     : array<u32, 2>,
    phi_n_AoA      : array<f32, 20>,
    phi_n_AoD      : array<f32, 20>,
    theta_n_ZOA    : array<f32, 20>,
    theta_n_ZOD    : array<f32, 20>,
}

// Active link descriptor (replaces pointer-based activeLink<Tcomplex>)
struct ActiveLink {
    cid         : u32,
    uid         : u32,
    linkIdx     : u32,
    lspReadIdx  : u32,
    // Flat buffer offsets (elements, not bytes) into the respective storage arrays
    cirCoeOffset      : u32,   // into buf_cirCoe
    cirNormDelayOffset: u32,   // into buf_cirNormDelay (NMAXTAPS u32 per link)
    cirNtapsOffset    : u32,   // into buf_cirNtaps (1 u32 per link)
    freqChanPrbgOffset: u32,   // into buf_freqChanPrbg
}

// Dispatch-time push constants (WGSL uniform alternative)
struct DispatchUniforms {
    nSite        : u32,
    nUT          : u32,
    nActiveLinks : u32,
    refTime      : f32,
    cfr_norm     : f32,
    // u32 offsets to the cirNormDelay / cirNtaps regions of the
    // packed CIR output buffer. The CIR kernel uses them to compute
    // absolute write positions; CFR sees the regions as bound
    // sub-ranges and never reads these fields.
    cirNormDelayRegionBase : u32,
    cirNtapsRegionBase     : u32,
    _pad0        : u32,
}

// ── group 1: cal_cluster_ray_kernel ──────────────────────────────────────
@group(1) @binding(0) var<storage, read>       cray_buf_link    : array<LinkParams>;
@group(1) @binding(1) var<storage, read>       cray_buf_ut      : array<SspUtParam>;
@group(1) @binding(2) var<storage, read>       cray_buf_cmn     : SsCmnParams;
@group(1) @binding(3) var<storage, read_write> cray_buf_cluster : array<ClusterParams>;
@group(1) @binding(4) var<storage, read_write> cray_buf_rng     : array<RngState>;
@group(1) @binding(5) var<uniform>             cray_disp        : DispatchUniforms;
// Combined cluster-ray output buffer. Packs 6 logically-separate
// f32 arrays (xpr, randomPhases, phi_nm_AoA/AoD, theta_nm_ZOA/ZOD)
// into a single binding so the cal_cluster_ray_kernel fits Dawn-D3D12's
// hard limit of 10 storage buffers per shader stage (was 11). Per-link
// stride and per-array sub-offsets are defined as constants below.
@group(1) @binding(6) var<storage, read_write> cray_packed_out:       array<f32>;
// PACKED_OFF_* are f32 offsets within each link's 3600-entry slab.
// MAX_CR  = MAX_CLUSTERS * MAX_RAYS = 20 * 20 = 400 (xpr / angles)
// MAX_CR4 = 4 * MAX_CR             = 1600         (randomPhases)
// Sum = 400 + 1600 + 4 * 400 = 3600 f32 / link.
const PACKED_LINK_STRIDE : u32 = 3600u;
const PACKED_OFF_XPR     : u32 = 0u;
const PACKED_OFF_RNDP    : u32 = 400u;
const PACKED_OFF_AOA     : u32 = 2000u;
const PACKED_OFF_AOD     : u32 = 2400u;
const PACKED_OFF_ZOA     : u32 = 2800u;
const PACKED_OFF_ZOD     : u32 = 3200u;

// ── group 2: generate_cir_kernel ─────────────────────────────────────────
// Small uniform-layout struct holding ONLY the SsCmnParams fields the
// CIR kernel actually reads. Moving cir_buf_cmn from a full storage
// SsCmnParams binding (binding 2) to this tiny uniform frees one
// storage slot toward Dawn's 10-per-stage limit. All array fields
// are sized as vec4<T> chunks because uniform-buffer arrays need
// 16-byte element strides; CirCmn fields use the .x/.y/.z lanes
// for the 3-element NLOS/LOS/O2I sets and the first 10 of the 12
// available lanes for the 10-element raysInSubCluster arrays.
struct CirCmn {
    lambda_0:                 f32,
    nSubCluster:              u32,
    _pad0:                    u32,
    _pad1:                    u32,
    C_DS:                     vec4<f32>,                // .xyz used (NLOS/LOS/O2I)
    raysInSubClusterSizes:    vec4<u32>,                // .xyz used (sub0/sub1/sub2 sizes)
    raysInSubCluster0:        array<vec4<u32>, 3>,      // 10 used of 12 slots
    raysInSubCluster1:        array<vec4<u32>, 3>,
    raysInSubCluster2:        array<vec4<u32>, 3>,
}

@group(2) @binding(0)  var<uniform>             cir_uni_sim          : SmallScaleSimConfig;
@group(2) @binding(1)  var<uniform>             cir_uni_sys          : SmallScaleSysConfig;
@group(2) @binding(2)  var<uniform>             cir_buf_cmn          : CirCmn;
@group(2) @binding(3)  var<storage, read>       cir_buf_cell         : array<SspCellParam>;
@group(2) @binding(4)  var<storage, read>       cir_buf_ut           : array<SspUtParam>;
@group(2) @binding(5)  var<storage, read>       cir_buf_link         : array<LinkParams>;
@group(2) @binding(6)  var<storage, read_write> cir_buf_cluster      : array<ClusterParams>;
@group(2) @binding(7)  var<storage, read>       cir_buf_antCfg       : array<AntPanelConfig>;
@group(2) @binding(8)  var<storage, read>       cir_buf_antTheta     : array<f32>;
@group(2) @binding(9)  var<storage, read>       cir_buf_antPhi       : array<f32>;
@group(2) @binding(10) var<storage, read>       cir_buf_activeLink   : array<ActiveLink>;
// Packed output buffer: per-link [cirCoe (nSnap * nUeAnt * nBsAnt * 24
// vec2f = 2 u32 each) | cirNormDelay (24 u32) | cirNtaps (1 u32)],
// flat array<u32>. cirCoe writes go through bitcast<u32>(f32) so the
// layout matches what readCirCoe/readCirNormDelay/readCirNtaps slice
// back on the host side.
@group(2) @binding(11) var<storage, read_write> cir_buf_packed_out   : array<u32>;
@group(2) @binding(14) var<uniform>             cir_disp             : DispatchUniforms;
// Read-only view of the cluster-ray packed output buffer. Same layout
// as cray_packed_out (see PACKED_LINK_STRIDE + PACKED_OFF_* constants).
@group(2) @binding(15) var<storage, read>      cir_packed_in        : array<f32>;
// cir_dbg removed to fit Dawn's 10 SSBO/stage limit -- was a
// diagnostic-only read_write storage buffer.

// ── group 3: generate_cfr_kernel_mode1 ───────────────────────────────────
@group(3) @binding(0) var<uniform>             cfr_uni_sim          : SmallScaleSimConfig;
@group(3) @binding(1) var<storage, read>       cfr_buf_cell         : array<SspCellParam>;
@group(3) @binding(2) var<storage, read>       cfr_buf_ut           : array<SspUtParam>;
@group(3) @binding(3) var<storage, read>       cfr_buf_antCfg       : array<AntPanelConfig>;
@group(3) @binding(4) var<storage, read>       cfr_buf_activeLink   : array<ActiveLink>;
@group(3) @binding(5) var<storage, read>       cfr_buf_cirCoe       : array<vec2f>;
@group(3) @binding(6) var<storage, read>       cfr_buf_cirNormDelay : array<u32>;
@group(3) @binding(7) var<storage, read>       cfr_buf_cirNtaps     : array<u32>;
@group(3) @binding(8) var<storage, read_write> cfr_buf_freqChanPrbg : array<vec2f>;
@group(3) @binding(9) var<uniform>             cfr_disp             : DispatchUniforms;

// ---------------------------------------------------------------------------
// RNG helpers  (xoshiro128+)
// ---------------------------------------------------------------------------


fn rng_uniform(s: ptr<function, RngState>) -> f32 {
    // [0, 1)
    return f32(rng_next(s) >> 8u) * (1.0 / 16777216.0);
}

// Box-Muller normal sample (returns one normal, discards second)
fn rng_normal(s: ptr<function, RngState>) -> f32 {
    let u1 = max(rng_uniform(s), 1e-10);
    let u2 = rng_uniform(s);
    return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
}

// ---------------------------------------------------------------------------
// Angle wrapping
// ---------------------------------------------------------------------------
fn wrap_azimuth(phi: f32) -> f32 {
    var w = (phi + 180.0) % 360.0;
    if w < 0.0 { w += 360.0; }
    return w - 180.0;
}

fn wrap_zenith(theta: f32) -> f32 {
    var w = theta % 360.0;
    if w < 0.0 { w += 360.0; }
    if w > 180.0 { w = 360.0 - w; }
    return w;
}

// ---------------------------------------------------------------------------
// Complex arithmetic helpers
// ---------------------------------------------------------------------------
fn cmul(a: vec2f, b: vec2f) -> vec2f {
    return vec2f(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

fn cadd(a: vec2f, b: vec2f) -> vec2f {
    return a + b;
}

fn cexp_j(phase: f32) -> vec2f {
    return vec2f(cos(phase), sin(phase));
}

// ---------------------------------------------------------------------------
// Antenna field pattern
// ---------------------------------------------------------------------------
fn calc_field_components(cfg: AntPanelConfig, theta: f32, phi: f32, zeta: f32) -> vec2f {
    // theta_idx in [0,180], phi_idx in [0,359]
    var ti = i32(round(theta));
    var pi_ = i32(round(phi));
    if ti < 0 || ti >= 360 {
        ti = ti % 360;
        if ti < 0 { ti += 360; }
    }
    if ti > 180 { ti = 360 - ti; }
    if pi_ < 0 || pi_ > 359 {
        pi_ = pi_ % 360;
        if pi_ < 0 { pi_ += 360; }
    }
    let A_db = cir_buf_antTheta[cfg.thetaOffset + u32(ti)]
             + cir_buf_antPhi  [cfg.phiOffset   + u32(pi_)]
             + select(0.0, GMAX, cfg.antModel == 1u);
    let A_sqrt = pow(10.0, A_db / 20.0);
    let z_rad  = zeta * DEG2RAD;
    return vec2f(A_sqrt * cos(z_rad), A_sqrt * sin(z_rad));   // {F_theta, F_phi}
}

// ---------------------------------------------------------------------------
// NLOS ray coefficient  (3GPP 38.901 Eq 7.5-22)
// ---------------------------------------------------------------------------
fn calc_ray_coeff(
    utCfg   : AntPanelConfig, ueAntIdx: u32,
    tZOA: f32, pAOA: f32, zetaUt: f32,
    bsCfg   : AntPanelConfig, bsAntIdx: u32,
    tZOD: f32, pAOD: f32, zetaBs: f32,
    xpr: f32, rph_off: u32,   // offset into cluster's randomPhases flat array
    cluster_link_idx: u32,
    t: f32, vel: vec2f, lambda0: f32
) -> vec2f {
    // --- antenna element positions ---
    // Rx position  (utCfg)
    let Mu  = utCfg.antSize[2];
    let Nu  = utCfg.antSize[3];
    let Pu  = utCfg.antSize[4];
    let p_rx = ueAntIdx % Pu;
    let d_rx = vec2f(
        f32((ueAntIdx / (Nu * Pu)) % Mu) * utCfg.antSpacing[2],
        f32((ueAntIdx / Pu) % Nu)        * utCfg.antSpacing[3]
    );
    // Tx position  (bsCfg)
    let Mb  = bsCfg.antSize[2];
    let Nb  = bsCfg.antSize[3];
    let Pb  = bsCfg.antSize[4];
    let p_tx = bsAntIdx % Pb;
    let d_tx = vec2f(
        f32((bsAntIdx / (Nb * Pb)) % Mb) * bsCfg.antSpacing[2],
        f32((bsAntIdx / Pb) % Nb)        * bsCfg.antSpacing[3]
    );

    // --- field patterns ---
    let Frx = calc_field_components(utCfg, tZOA, pAOA, utCfg.antPolarAngles[p_rx] + zetaUt);
    let Ftx = calc_field_components(bsCfg, tZOD, pAOD, bsCfg.antPolarAngles[p_tx] + zetaBs);

    // angles -> radians
    let tZOA_r = tZOA * DEG2RAD;
    let pAOA_r = pAOA * DEG2RAD;
    let tZOD_r = tZOD * DEG2RAD;
    let pAOD_r = pAOD * DEG2RAD;

    // --- spherical unit vectors ---
    let rRx = vec3f(sin(tZOA_r)*cos(pAOA_r), sin(tZOA_r)*sin(pAOA_r), cos(tZOA_r));
    let rTx = vec3f(sin(tZOD_r)*cos(pAOD_r), sin(tZOD_r)*sin(pAOD_r), cos(tZOD_r));

    // --- array steering phases ---
    let dot_rx = rRx.x*d_rx.x + rRx.y*d_rx.y;      // d_bar_rx.z = 0
    let dot_tx = rTx.x*d_tx.x + rTx.y*d_tx.y;
    let term4 = cexp_j(TWO_PI * dot_rx);
    let term5 = cexp_j(TWO_PI * dot_tx);

    // --- Doppler ---
    let v_speed = length(vel);
    let v_dir   = select(vec2f(0.0), vel / v_speed, v_speed > 0.0);
    let doppler = TWO_PI * (rRx.x*v_dir.x + rRx.y*v_dir.y) * v_speed / lambda0 * t;
    let term6   = cexp_j(doppler);

    // --- polarisation matrix (NLOS) ---
    let sqrt_kappa = sqrt(max(xpr, 1e-10));
    // rph_off is already a fully-resolved offset into cir_packed_in
    // (caller computed it as link_base + PACKED_OFF_RNDP + ...).
    let ph0 = cir_packed_in[rph_off    ] * DEG2RAD;
    let ph1 = cir_packed_in[rph_off + 1u] * DEG2RAD;
    let ph2 = cir_packed_in[rph_off + 2u] * DEG2RAD;
    let ph3 = cir_packed_in[rph_off + 3u] * DEG2RAD;

    // term2[i][j]
    let t200 = cexp_j(ph0);
    let t201 = vec2f(cos(ph1)/sqrt_kappa, sin(ph1)/sqrt_kappa);
    let t210 = vec2f(cos(ph2)/sqrt_kappa, sin(ph2)/sqrt_kappa);
    let t211 = cexp_j(ph3);

    // term1 = {Frx.x+0j, Frx.y+0j}  (vec2f reinterpreted as complex)
    // term3 = {Ftx.x+0j, Ftx.y+0j}
    let t1_0 = vec2f(Frx.x, 0.0);
    let t1_1 = vec2f(Frx.y, 0.0);
    let t3_0 = vec2f(Ftx.x, 0.0);
    let t3_1 = vec2f(Ftx.y, 0.0);

    // result = sum_{i,j} term1[i] * term2[i][j] * term3[j] * term4 * term5 * term6
    var result = vec2f(0.0);
    // (i=0, j=0)
    result = cadd(result, cmul(cmul(cmul(cmul(cmul(t1_0, t200), t3_0), term4), term5), term6));
    // (i=0, j=1)
    result = cadd(result, cmul(cmul(cmul(cmul(cmul(t1_0, t201), t3_1), term4), term5), term6));
    // (i=1, j=0)
    result = cadd(result, cmul(cmul(cmul(cmul(cmul(t1_1, t210), t3_0), term4), term5), term6));
    // (i=1, j=1)
    result = cadd(result, cmul(cmul(cmul(cmul(cmul(t1_1, t211), t3_1), term4), term5), term6));
    return result;
}

// ---------------------------------------------------------------------------
// LOS ray coefficient  (3GPP 38.901 Eq 7.5-22, LOS term)
// ---------------------------------------------------------------------------
fn calc_los_coeff(
    utCfg   : AntPanelConfig, ueAntIdx: u32,
    tZOA: f32, pAOA: f32, zetaUt: f32,
    bsCfg   : AntPanelConfig, bsAntIdx: u32,
    tZOD: f32, pAOD: f32, zetaBs: f32,
    t: f32, vel: vec2f, lambda0: f32, d3d: f32
) -> vec2f {
    let Mu  = utCfg.antSize[2]; let Nu = utCfg.antSize[3]; let Pu = utCfg.antSize[4];
    let p_rx = ueAntIdx % Pu;
    let d_rx = vec2f(
        f32((ueAntIdx / (Nu * Pu)) % Mu) * utCfg.antSpacing[2],
        f32((ueAntIdx / Pu) % Nu)        * utCfg.antSpacing[3]
    );
    let Mb  = bsCfg.antSize[2]; let Nb = bsCfg.antSize[3]; let Pb = bsCfg.antSize[4];
    let p_tx = bsAntIdx % Pb;
    let d_tx = vec2f(
        f32((bsAntIdx / (Nb * Pb)) % Mb) * bsCfg.antSpacing[2],
        f32((bsAntIdx / Pb) % Nb)        * bsCfg.antSpacing[3]
    );

    let Frx = calc_field_components(utCfg, tZOA, pAOA, utCfg.antPolarAngles[p_rx] + zetaUt);
    let Ftx = calc_field_components(bsCfg, tZOD, pAOD, bsCfg.antPolarAngles[p_tx] + zetaBs);

    let tZOA_r = tZOA * DEG2RAD;
    let pAOA_r = pAOA * DEG2RAD;
    let tZOD_r = tZOD * DEG2RAD;
    let pAOD_r = pAOD * DEG2RAD;

    let rRx = vec3f(sin(tZOA_r)*cos(pAOA_r), sin(tZOA_r)*sin(pAOA_r), cos(tZOA_r));
    let rTx = vec3f(sin(tZOD_r)*cos(pAOD_r), sin(tZOD_r)*sin(pAOD_r), cos(tZOD_r));

    let dot_rx = rRx.x*d_rx.x + rRx.y*d_rx.y;
    let dot_tx = rTx.x*d_tx.x + rTx.y*d_tx.y;
    let term4  = cexp_j(TWO_PI * dot_rx);
    let term5  = cexp_j(TWO_PI * dot_tx);

    let v_speed = length(vel);
    let v_dir   = select(vec2f(0.0), vel / v_speed, v_speed > 0.0);
    let doppler = TWO_PI * (rRx.x*v_dir.x + rRx.y*v_dir.y) * v_speed / lambda0 * t;
    let term6   = cexp_j(doppler);

    // LOS path phase (term7)
    let term7 = cexp_j(-TWO_PI * d3d / lambda0);

    // LOS polarisation matrix [1 0; 0 -1]
    let t1_0 = vec2f(Frx.x, 0.0);
    let t1_1 = vec2f(Frx.y, 0.0);
    let t3_0 = vec2f(Ftx.x, 0.0);
    let t3_1 = vec2f(Ftx.y, 0.0);

    var result = vec2f(0.0);
    // (i=0,j=0): t2=1+0j
    result = cadd(result, cmul(cmul(cmul(cmul(cmul(t1_0, vec2f(1.0,0.0)), t3_0), term7), term4), cmul(term5, term6)));
    // (i=1,j=1): t2=-1+0j
    result = cadd(result, cmul(cmul(cmul(cmul(cmul(t1_1, vec2f(-1.0,0.0)), t3_1), term7), term4), cmul(term5, term6)));
    return result;
}

// ---------------------------------------------------------------------------
// =====================   KERNEL 1: cal_cluster_ray_kernel   ===================
// Dispatch: grid(nSite, ceil(nUT/256)), block(256,1,1)
// ---------------------------------------------------------------------------

@compute @workgroup_size(256, 1, 1)
fn cal_cluster_ray_kernel(
    @builtin(workgroup_id)       wg : vec3<u32>,
    @builtin(local_invocation_id) li: vec3<u32>
) {
    let site_idx = wg.x;
    let ue_idx   = wg.y * 256u + li.x;
    if ue_idx >= cray_disp.nUT { return; }

    let link_idx = site_idx * cray_disp.nUT + ue_idx;
    let lk       = cray_buf_link[link_idx];
    let ut       = cray_buf_ut[ue_idx];

    let is_o2i   = u32(ut.outdoor_ind == 0u);
    let lsp_idx  = select(select(1u, 0u, lk.losInd != 0u), 2u, is_o2i != 0u);

    // per-thread RNG state
    var rng = cray_buf_rng[link_idx];

    // --------------- 1. Cluster delays & powers ---------------
    let r_tau     = cray_buf_cmn.r_tao[lsp_idx];
    let xi_       = cray_buf_cmn.xi[lsp_idx];
    var n_cluster = cray_buf_cmn.nCluster[lsp_idx];
    let n_ray     = cray_buf_cmn.nRayPerCluster[lsp_idx];

    var delays : array<f32, 20>;
    var powers : array<f32, 20>;

    // Generate raw delays
    for (var n = 0u; n < n_cluster; n++) {
        if lk.DS > 0.0 {
            let u = max(rng_uniform(&rng), 1e-10);
            delays[n] = -r_tau * lk.DS * log(u);
        } else {
            delays[n] = 0.0;
        }
    }

    // Normalise delays (subtract min)
    var min_d = delays[0];
    for (var n = 1u; n < n_cluster; n++) { if delays[n] < min_d { min_d = delays[n]; } }
    for (var n = 0u; n < n_cluster; n++) { delays[n] -= min_d; }

    // Sort delays (bubble sort, max 20 elements)
    for (var i = 0u; i < n_cluster - 1u; i++) {
        for (var j = 0u; j < n_cluster - 1u - i; j++) {
            if delays[j] > delays[j+1u] {
                let tmp = delays[j]; delays[j] = delays[j+1u]; delays[j+1u] = tmp;
            }
        }
    }

    // Unscaled powers: exp(-delay*(r_tau-1)/(r_tau*DS)) * 10^(-xi*N(0,1)/10)
    var p_sum = 0.0;
    for (var n = 0u; n < n_cluster; n++) {
        let xi_rand = rng_normal(&rng);
        powers[n] = exp(-delays[n] * (r_tau - 1.0) / (r_tau * max(lk.DS, 1e-30)))
                  * pow(10.0, -xi_ * xi_rand / 10.0);
        p_sum += powers[n];
    }
    // Normalise
    for (var n = 0u; n < n_cluster; n++) { powers[n] /= p_sum; }

    // Delays are already in nanoseconds here: lk.DS is stored as 10^(mu+9)
    // (3GPP TR 38.901 gives mu_lgDS in log10(seconds); the +9 in
    // cal_link_param_kernel pre-converts to log10(nanoseconds)). The previous
    // `delays[n] *= 1e9` second-to-ns conversion was a leftover from CUDA where
    // lk.DS is in seconds, and it bloated tap_idx by 1e9 (and made delays in
    // the cluster_buf nonsensically large for downstream consumers).

    // Filter clusters below threshold (1e-3 relative to max)
    var max_p = powers[0];
    for (var n = 1u; n < n_cluster; n++) { if powers[n] > max_p { max_p = powers[n]; } }
    let threshold = max_p * 1e-3;

    var valid = 0u;
    for (var i = 0u; i < n_cluster; i++) {
        if powers[i] >= threshold {
            if valid != i {
                delays[valid] = delays[i];
                powers[valid] = powers[i];
            }
            valid++;
        } else if lk.losInd != 0u && ut.outdoor_ind != 0u && i == 0u {
            // LOS outdoor: keep cluster 0 with power 0
            powers[0] = 0.0;
            valid = 1u;
        }
    }
    n_cluster = valid;

    // Find two strongest
    var s0 = 0u; var s1 = 1u;
    if n_cluster >= 2u {
        if powers[1] > powers[0] { s0 = 1u; s1 = 0u; }
        for (var n = 2u; n < n_cluster; n++) {
            if powers[n] > powers[s0] { s1 = s0; s0 = n; }
            else if powers[n] > powers[s1] { s1 = n; }
        }
    } else { s0 = 0u; s1 = 0u; }

    // Apply K-factor (LOS) — lk.K is already in linear units from cal_link_param
    if lk.losInd != 0u {
        let K_lin = lk.K;
        let sc    = 1.0 / (1.0 + K_lin);
        for (var n = 1u; n < n_cluster; n++) { powers[n] *= sc; }
        powers[0] = K_lin / (1.0 + K_lin) + powers[0] * sc;
    }

    // --------------- 2. Cluster angles (genClusterAngleGPU) ---------------
    var phi_n_AoA   : array<f32, 20>;
    var phi_n_AoD   : array<f32, 20>;
    var theta_n_ZOA : array<f32, 20>;
    var theta_n_ZOD : array<f32, 20>;

    let C_ASA = cray_buf_cmn.C_ASA[lsp_idx];
    let C_ASD = cray_buf_cmn.C_ASD[lsp_idx];
    let C_ZSA = cray_buf_cmn.C_ZSA[lsp_idx];

    // C_ZSD is computed per-link from mu_lgZSD: (3/8) * 10^(mu_lgZSD) — not a cluster array
    let C_ZSD = (3.0 / 8.0) * pow(10.0, lk.mu_lgZSD);

    // Select C_phi, C_theta based on scenario
    let C_phi   = select(select(cray_buf_cmn.C_phi_NLOS,   cray_buf_cmn.C_phi_LOS,   lk.losInd != 0u), cray_buf_cmn.C_phi_O2I,   is_o2i != 0u);
    let C_theta = select(select(cray_buf_cmn.C_theta_NLOS, cray_buf_cmn.C_theta_LOS, lk.losInd != 0u), cray_buf_cmn.C_theta_O2I, is_o2i != 0u);

    // Scale C_phi for LOS K-factor (3GPP 38.901 Eq 7.5-10) — lk.K is linear
    var C_phi_scaled   = C_phi;
    var C_theta_scaled = C_theta;
    if lk.losInd != 0u && is_o2i == 0u {
        let K_dB = 10.0 * log10(max(lk.K, 1e-10));
        let Kfac  = 1.1035 - 0.028 * K_dB - 0.002 * K_dB * K_dB + 0.0001 * K_dB * K_dB * K_dB;
        C_phi_scaled   = C_phi   * Kfac;
        C_theta_scaled = C_theta * Kfac;
    }

    // Eq 7.5-9 / 7.5-14: phi_n = 2*(ASA/1.4)*invCDF * sign * eps + phi_LOS
    // Use lookup of the 3GPP per-cluster sign and coupling from RayOffsetAngles
    var max_abs_power = 0.0;
    for (var n = 0u; n < n_cluster; n++) { max_abs_power = max(max_abs_power, abs(powers[n])); }

    for (var n = 0u; n < n_cluster; n++) {
        // AoA
        let arg_asa    = -log(powers[n] / max(max_abs_power, 1e-30)) / (2.0 * C_phi_scaled / 1.4);
        let sign_n_asa = select(-1.0, 1.0, rng_uniform(&rng) >= 0.5);
        let eps_n_asa  = rng_normal(&rng) * lk.ASA / 7.0;
        phi_n_AoA[n]   = wrap_azimuth(sign_n_asa * arg_asa + eps_n_asa + lk.phi_LOS_AOA);
        if lk.losInd != 0u {
            phi_n_AoA[n] = wrap_azimuth(phi_n_AoA[n] - phi_n_AoA[0] + lk.phi_LOS_AOA);
        }

        // AoD
        let arg_asd    = -log(powers[n] / max(max_abs_power, 1e-30)) / (2.0 * C_ASD / 1.4);
        let sign_n_asd = select(-1.0, 1.0, rng_uniform(&rng) >= 0.5);
        let eps_n_asd  = rng_normal(&rng) * lk.ASD / 7.0;
        phi_n_AoD[n]   = wrap_azimuth(sign_n_asd * arg_asd + eps_n_asd + lk.phi_LOS_AOD);
        if lk.losInd != 0u {
            phi_n_AoD[n] = wrap_azimuth(phi_n_AoD[n] - phi_n_AoD[0] + lk.phi_LOS_AOD);
        }

        // ZoA
        let arg_zsa    = -log(powers[n] / max(max_abs_power, 1e-30)) / (2.0 * C_theta_scaled / 1.4);
        let sign_n_zsa = select(-1.0, 1.0, rng_uniform(&rng) >= 0.5);
        let eps_n_zsa  = rng_normal(&rng) * lk.ZSA / 7.0;
        theta_n_ZOA[n] = wrap_zenith(sign_n_zsa * arg_zsa + eps_n_zsa + 90.0);
        if lk.losInd != 0u {
            theta_n_ZOA[n] = wrap_zenith(theta_n_ZOA[n] - theta_n_ZOA[0] + lk.theta_LOS_ZOA);
        }

        // ZoD
        let arg_zsd    = -log(powers[n] / max(max_abs_power, 1e-30)) / (2.0 * C_ZSD / 1.4);
        let sign_n_zsd = select(-1.0, 1.0, rng_uniform(&rng) >= 0.5);
        let eps_n_zsd  = rng_normal(&rng) * lk.ZSD / 7.0;
        theta_n_ZOD[n] = wrap_zenith(sign_n_zsd * arg_zsd + eps_n_zsd
                                     + lk.theta_LOS_ZOD + lk.mu_offset_ZOD);
        if lk.losInd != 0u {
            theta_n_ZOD[n] = wrap_zenith(theta_n_ZOD[n] - theta_n_ZOD[0] + lk.theta_LOS_ZOD);
        }
    }

    // --------------- 3. Ray angles (genRayAngleGPU) ---------------
    // phi_nm_* / theta_nm_* written directly to flat storage — no private staging
    let alpha_m = cray_buf_cmn.RayOffsetAngles;

    for (var c = 0u; c < n_cluster; c++) {
        for (var r = 0u; r < n_ray; r++) {
            let idx = c * MAX_RAYS + r;
            let s_asa = select(-1.0, 1.0, rng_uniform(&rng) >= 0.5);
            let s_asd = select(-1.0, 1.0, rng_uniform(&rng) >= 0.5);
            let s_zsa = select(-1.0, 1.0, rng_uniform(&rng) >= 0.5);
            let s_zsd = select(-1.0, 1.0, rng_uniform(&rng) >= 0.5);

            let link_base = link_idx * PACKED_LINK_STRIDE;
            cray_packed_out[link_base + PACKED_OFF_AOA + idx] =
                wrap_azimuth(phi_n_AoA[c]   + C_ASA * alpha_m[r] * s_asa);
            cray_packed_out[link_base + PACKED_OFF_AOD + idx] =
                wrap_azimuth(phi_n_AoD[c]   + C_ASD * alpha_m[r] * s_asd);
            cray_packed_out[link_base + PACKED_OFF_ZOA + idx] =
                wrap_zenith (theta_n_ZOA[c] + C_ZSA * alpha_m[r] * s_zsa);
            // ZOD: 3/8 * 10^mu_lgZSD offset (3GPP Eq 7.5-20)
            let zsd_offset = (3.0 / 8.0) * pow(10.0, lk.mu_lgZSD) * alpha_m[r] * s_zsd;
            cray_packed_out[link_base + PACKED_OFF_ZOD + idx] =
                wrap_zenith(theta_n_ZOD[c] + zsd_offset);
        }
    }

    // --------------- 4. XPR and random phases ---------------
    // xpr / randomPhases written directly to flat storage — no private staging
    for (var c = 0u; c < n_cluster; c++) {
        for (var r = 0u; r < n_ray; r++) {
            let idx = c * n_ray + r;
            let link_base = link_idx * PACKED_LINK_STRIDE;
            cray_packed_out[link_base + PACKED_OFF_XPR + idx] =
                pow(10.0, (cray_buf_cmn.mu_XPR[lsp_idx]
                           + cray_buf_cmn.sigma_XPR[lsp_idx] * rng_normal(&rng)) / 10.0);
            for (var p = 0u; p < 4u; p++) {
                cray_packed_out[link_base + PACKED_OFF_RNDP + idx * 4u + p] =
                    (rng_uniform(&rng) - 0.5) * 360.0;
            }
        }
    }

    // --------------- 5. Write output ---------------
    cray_buf_cluster[link_idx].nCluster        = n_cluster;
    cray_buf_cluster[link_idx].nRayPerCluster  = n_ray;
    cray_buf_cluster[link_idx].strongest2[0]   = s0;
    cray_buf_cluster[link_idx].strongest2[1]   = s1;

    for (var n = 0u; n < n_cluster; n++) {
        cray_buf_cluster[link_idx].delays[n]      = delays[n];
        cray_buf_cluster[link_idx].powers[n]      = powers[n];
        cray_buf_cluster[link_idx].phi_n_AoA[n]   = phi_n_AoA[n];
        cray_buf_cluster[link_idx].phi_n_AoD[n]   = phi_n_AoD[n];
        cray_buf_cluster[link_idx].theta_n_ZOA[n] = theta_n_ZOA[n];
        cray_buf_cluster[link_idx].theta_n_ZOD[n] = theta_n_ZOD[n];
    }
    // xpr / randomPhases / phi_nm_* / theta_nm_* already written element-by-element above

    // Persist RNG state
    cray_buf_rng[link_idx] = rng;
}

// ---------------------------------------------------------------------------
// =====================   KERNEL 2: generate_cir_kernel   ====================
// Dispatch: grid(nActiveLinks, nSnapshots), block(NMAXTAPS,1,1)
// Each workgroup processes one (link, snapshot) pair.
// The outer BS-antenna loop is serial inside the workgroup (like CUDA).
// ---------------------------------------------------------------------------

var<workgroup> wg_Hlink  : array<vec2f, 192>;   // NMAXTAPS(24) * MAX_UE_ANT(8)
var<workgroup> wg_tapIdx : array<u32,   24>;
var<workgroup> wg_tapCnt : atomic<u32>;

// Per-cluster parallelization scaffolding: each of the 24 workgroup
// threads handles one cluster of the link. Strongest clusters expand
// into N_SUB_CLUSTER(=3) sub-taps; regular clusters produce a single
// tap. wg_clusterTapOffset[c] = starting slot in wg_Hlink/wg_tapIdx
// for cluster c; wg_clusterTapCount[c] = number of taps (1 or 3).
// wg_tapCountTotal = sum over active clusters (capped at NMAXTAPS).
var<workgroup> wg_clusterTapOffset : array<u32, 24>;
var<workgroup> wg_clusterTapCount  : array<u32, 24>;
var<workgroup> wg_tapCountTotal    : u32;

// Packed-buffer helpers for cir_buf_packed_out. The buffer is a flat
// array<u32>; cirCoe entries (vec2f = 2 floats) are stored as 2 u32
// via bitcast. cirNormDelay and cirNtaps entries are u32 directly.
fn pack_write_vec2f(idx_u32: u32, v: vec2f) {
    cir_buf_packed_out[idx_u32]      = bitcast<u32>(v.x);
    cir_buf_packed_out[idx_u32 + 1u] = bitcast<u32>(v.y);
}
fn pack_read_vec2f(idx_u32: u32) -> vec2f {
    return vec2f(bitcast<f32>(cir_buf_packed_out[idx_u32]),
                 bitcast<f32>(cir_buf_packed_out[idx_u32 + 1u]));
}

@compute @workgroup_size(24, 1, 1)      // NMAXTAPS threads
fn generate_cir_kernel(
    @builtin(workgroup_id)       wg : vec3<u32>,
    @builtin(local_invocation_id) li: vec3<u32>
) {
    let active_link_idx = wg.x;
    let snapshot_idx    = wg.y;

    if active_link_idx >= cir_disp.nActiveLinks { return; }

    let al  = cir_buf_activeLink[active_link_idx];
    let cid = al.cid; let uid = al.uid;
    let lk  = cir_buf_link[al.lspReadIdx];

    let ut       = cir_buf_ut[uid];
    let cell     = cir_buf_cell[cid];
    let is_o2i   = u32(ut.outdoor_ind == 0u);
    let lsp_idx  = select(select(1u, 0u, lk.losInd != 0u), 2u, is_o2i != 0u);

    let utCfg   = cir_buf_antCfg[ut.antPanelIdx];
    let bsCfg   = cir_buf_antCfg[cell.antPanelIdx];
    let nUtAnt  = utCfg.nAnt;
    let nCellAnt = bsCfg.nAnt;

    let vel = vec2f(ut.velocity[0], ut.velocity[1]);

    // Snapshot time
    let time_offset  = 1e-3 / (cir_uni_sim.sc_spacing_hz * 15e3 * f32(cir_uni_sim.n_snapshot_per_slot));
    var snap_time    = cir_disp.refTime + f32(snapshot_idx) * time_offset;
    if cir_uni_sys.enable_propagation_delay == 1u {
        snap_time += lk.d3d / 3.0e8;
    }

    // CIR snapshot offset in the packed output buffer. al.cirCoeOffset
    // is kept in vec2f-units (= linkIdx * nSnap*nUe*nBs*NMAXTAPS) so
    // CFR can use it unchanged via its sub-range view of the packed
    // buffer. The packed buffer's u32 layout starts with the entire
    // cirCoe region at offset 0, so the kernel multiplies the
    // vec2f-units offset by 2 to get u32-units.
    let snap_off_vec2f = al.cirCoeOffset
                       + snapshot_idx * nUtAnt * nCellAnt * NMAXTAPS;
    let snap_off_u32 = snap_off_vec2f * 2u;

    // ---- disabled fading path ----
    if cir_uni_sys.disable_small_scale_fading == 1u {
        if li.x == 0u {
            let path_gain = -(lk.pathloss - lk.SF);
            let ps        = pow(10.0, path_gain / 20.0);
            let tZOD = wrap_zenith (lk.theta_LOS_ZOD - cell.antPanelOrientation[0]);
            let pAOD = wrap_azimuth(lk.phi_LOS_AOD   - cell.antPanelOrientation[1]);
            let tZOA = wrap_zenith (lk.theta_LOS_ZOA - ut.antPanelOrientation[0]);
            let pAOA = wrap_azimuth(lk.phi_LOS_AOA   - ut.antPanelOrientation[1]);
            let zetaBs = cell.antPanelOrientation[2];
            let zetaUt = ut.antPanelOrientation[2];
            for (var ua = 0u; ua < nUtAnt; ua++) {
                for (var ba = 0u; ba < nCellAnt; ba++) {
                    let coeff = calc_los_coeff(
                        utCfg, ua, tZOA, pAOA, zetaUt,
                        bsCfg, ba, tZOD, pAOD, zetaBs,
                        snap_time, vel, cir_buf_cmn.lambda_0, lk.d3d
                    );
                    let dst_u32 = snap_off_u32 + ((ua * nCellAnt + ba) * NMAXTAPS) * 2u;
                    pack_write_vec2f(dst_u32, coeff * ps);
                }
            }
            if snapshot_idx == 0u {
                cir_buf_packed_out[cir_disp.cirNormDelayRegionBase + al.cirNormDelayOffset] = 0u;
                cir_buf_packed_out[cir_disp.cirNtapsRegionBase     + al.cirNtapsOffset]     = 1u;
            }
        }
        return;
    }

    let cp     = cir_buf_cluster[al.lspReadIdx];
    let n_cl   = cp.nCluster;
    let n_ray  = cp.nRayPerCluster;

    // K-factor (Rician). lk.K is stored in *linear* units by cal_link_param.
    var KR       = 0.0;
    var los_power = 0.0;
    if lk.losInd != 0u && is_o2i == 0u {
        KR        = lk.K;
        los_power = KR / (KR + 1.0);
    }

    // (Diagnostic cir_dbg block removed in the SSBO-pack refactor --
    // generate_cir_kernel needed to lose 4 storage buffers to fit
    // Dawn's 10/stage limit; cir_dbg was diagnostic-only.)

    let zetaBs = cell.antPanelOrientation[2];
    let zetaUt = ut.antPanelOrientation[2];

    // -----------------------------------------------------------------
    // Per-cluster parallelization scaffolding.
    //
    // The original CUDA port wrapped the cluster loop in an `if tid == 0`
    // gate so all of the ray accumulation ran on lane 0 of a 24-thread
    // workgroup -- the other 23 threads were idle, which is why even a
    // few hundred workgroups per dispatch ran into the D3D12 watchdog.
    //
    // Each of the 24 threads now handles one cluster (li.x = c). Tap
    // indices and per-cluster tap counts are link-level state (don't
    // depend on the BS antenna), so we compute them once on thread 0
    // and broadcast via workgroup memory, then re-use them across the
    // BA loop. Sub-cluster tap_count growth (strongest clusters expand
    // into N_SUB_CLUSTER taps) is folded into a prefix-sum so each
    // thread knows where to write in wg_Hlink/wg_tapIdx without
    // collisions or atomics.
    // -----------------------------------------------------------------

    if li.x == 0u {
        var acc = 0u;
        for (var ci = 0u; ci < n_cl; ci++) {
            let is_strong = (ci == cp.strongest2[0] || ci == cp.strongest2[1]);
            let nTaps = select(1u, N_SUB_CLUSTER, is_strong);
            wg_clusterTapOffset[ci] = acc;
            wg_clusterTapCount[ci]  = nTaps;
            acc += nTaps;
            if acc > NMAXTAPS { acc = NMAXTAPS; break; }
        }
        wg_tapCountTotal = acc;
    }
    workgroupBarrier();

    let my_c            = li.x;
    let my_is_cluster   = my_c < n_cl;
    let my_is_strongest = my_is_cluster && (my_c == cp.strongest2[0] || my_c == cp.strongest2[1]);
    let my_tap_off      = select(0u, wg_clusterTapOffset[my_c], my_is_cluster);
    let my_tap_cnt      = select(0u, wg_clusterTapCount[my_c],  my_is_cluster);
    var my_cl_power     = select(0.0, cp.powers[my_c], my_is_cluster);
    if my_is_cluster && lk.losInd != 0u && is_o2i == 0u && my_c == 0u {
        my_cl_power = max(my_cl_power - los_power, 0.0);
    }
    let my_link_packed_base = al.lspReadIdx * PACKED_LINK_STRIDE;
    let my_cl_delay_base    = select(0.0, cp.delays[my_c], my_is_cluster);
    let C_DS                = cir_buf_cmn.C_DS[lsp_idx];

    // Each thread writes its tap indices once (link-level, shared across
    // BS antennas).
    if my_is_cluster && my_tap_off < NMAXTAPS {
        for (var sc = 0u; sc < my_tap_cnt && (my_tap_off + sc) < NMAXTAPS; sc++) {
            var cl_delay = my_cl_delay_base;
            if my_is_strongest {
                if sc == 1u { cl_delay += 1.28 * C_DS; }
                if sc == 2u { cl_delay += 2.56 * C_DS; }
            }
            var tap_idx = 0u;
            if cir_uni_sys.enable_propagation_delay == 1u {
                tap_idx = u32(round((cl_delay * 1e-9 + lk.d3d / 3.0e8 + lk.delta_tau)
                                    * cir_uni_sim.sc_spacing_hz * f32(cir_uni_sim.fft_size)));
            } else {
                tap_idx = u32(round(cl_delay * 1e-9
                                    * cir_uni_sim.sc_spacing_hz * f32(cir_uni_sim.fft_size)));
            }
            wg_tapIdx[my_tap_off + sc] = tap_idx;
        }
    }
    workgroupBarrier();

    let n_ray_f = f32(n_ray);

    // ---- Outer BS antenna loop (serial) ----
    for (var ba = 0u; ba < nCellAnt; ba++) {

        // Reset workgroup H_link in parallel across threads.
        for (var i = li.x; i < NMAXTAPS * MAX_UE_ANT; i += 24u) { wg_Hlink[i] = vec2f(0.0); }
        workgroupBarrier();

        // ---- parallel cluster work: each thread handles one cluster ----
        if my_is_cluster && my_tap_off < NMAXTAPS {
            if my_is_strongest {
                for (var sc = 0u; sc < N_SUB_CLUSTER && (my_tap_off + sc) < NMAXTAPS; sc++) {
                    var sc_power = 0.0;
                    if      sc == 0u { sc_power = sqrt(10.0 / 20.0); }
                    else if sc == 1u { sc_power = sqrt( 6.0 / 20.0); }
                    else             { sc_power = sqrt( 4.0 / 20.0); }

                    let sc_size = cir_buf_cmn.raysInSubClusterSizes[sc];
                    let power   = sqrt(my_cl_power / n_ray_f);
                    let my_tap  = my_tap_off + sc;

                    for (var ri = 0u; ri < sc_size; ri++) {
                        let ri_vec  = ri / 4u;
                        let ri_lane = ri % 4u;
                        var ray_local_idx = 0u;
                        if sc == 0u { ray_local_idx = my_c * n_ray + cir_buf_cmn.raysInSubCluster0[ri_vec][ri_lane]; }
                        else if sc == 1u { ray_local_idx = my_c * n_ray + cir_buf_cmn.raysInSubCluster1[ri_vec][ri_lane]; }
                        else             { ray_local_idx = my_c * n_ray + cir_buf_cmn.raysInSubCluster2[ri_vec][ri_lane]; }

                        let tZOA = cir_packed_in[my_link_packed_base + PACKED_OFF_ZOA + ray_local_idx]
                                   - ut.antPanelOrientation[0];
                        let pAOA = cir_packed_in[my_link_packed_base + PACKED_OFF_AOA + ray_local_idx]
                                   - ut.antPanelOrientation[1];
                        let tZOD = cir_packed_in[my_link_packed_base + PACKED_OFF_ZOD + ray_local_idx]
                                   - cell.antPanelOrientation[0];
                        let pAOD = cir_packed_in[my_link_packed_base + PACKED_OFF_AOD + ray_local_idx]
                                   - cell.antPanelOrientation[1];
                        let xpr  = cir_packed_in[my_link_packed_base + PACKED_OFF_XPR + ray_local_idx];
                        let rph_off = my_link_packed_base + PACKED_OFF_RNDP + ray_local_idx * 4u;

                        for (var ua = 0u; ua < nUtAnt; ua++) {
                            let rc = calc_ray_coeff(
                                utCfg, ua, tZOA, pAOA, zetaUt,
                                bsCfg, ba, tZOD, pAOD, zetaBs,
                                xpr, rph_off, al.lspReadIdx,
                                snap_time, vel, cir_buf_cmn.lambda_0
                            );
                            let hi = ua * NMAXTAPS + my_tap;
                            wg_Hlink[hi] = cadd(wg_Hlink[hi], rc * power * sc_power);
                        }
                    }
                }
            } else {
                let power = sqrt(my_cl_power / n_ray_f);
                let my_tap = my_tap_off;
                for (var r = 0u; r < n_ray; r++) {
                    let ray_local_idx = my_c * n_ray + r;
                    let tZOA = cir_packed_in[my_link_packed_base + PACKED_OFF_ZOA + ray_local_idx]
                               - ut.antPanelOrientation[0];
                    let pAOA = cir_packed_in[my_link_packed_base + PACKED_OFF_AOA + ray_local_idx]
                               - ut.antPanelOrientation[1];
                    let tZOD = cir_packed_in[my_link_packed_base + PACKED_OFF_ZOD + ray_local_idx]
                               - cell.antPanelOrientation[0];
                    let pAOD = cir_packed_in[my_link_packed_base + PACKED_OFF_AOD + ray_local_idx]
                               - cell.antPanelOrientation[1];
                    let xpr  = cir_packed_in[my_link_packed_base + PACKED_OFF_XPR + ray_local_idx];
                    let rph_off = my_link_packed_base + PACKED_OFF_RNDP + ray_local_idx * 4u;

                    for (var ua = 0u; ua < nUtAnt; ua++) {
                        let rc = calc_ray_coeff(
                            utCfg, ua, tZOA, pAOA, zetaUt,
                            bsCfg, ba, tZOD, pAOD, zetaBs,
                            xpr, rph_off, al.lspReadIdx,
                            snap_time, vel, cir_buf_cmn.lambda_0
                        );
                        let hi = ua * NMAXTAPS + my_tap;
                        wg_Hlink[hi] = cadd(wg_Hlink[hi], rc * power);
                    }
                }
            }
        }

        workgroupBarrier();

        // LOS component (added at tap 0). Single-threaded -- the LOS
        // contribution is small (one calc_los_coeff per UA) and we'd
        // need a barrier+broadcast to parallelize cleanly. Thread 0 is
        // already idle while other threads finished their cluster work
        // identically in the barrier above.
        if lk.losInd != 0u && is_o2i == 0u && li.x == 0u {
            let los_scale = sqrt(KR / (KR + 1.0));
            let tZOA = wrap_zenith (lk.theta_LOS_ZOA - ut.antPanelOrientation[0]);
            let pAOA = wrap_azimuth(lk.phi_LOS_AOA   - ut.antPanelOrientation[1]);
            let tZOD = wrap_zenith (lk.theta_LOS_ZOD - cell.antPanelOrientation[0]);
            let pAOD = wrap_azimuth(lk.phi_LOS_AOD   - cell.antPanelOrientation[1]);

            for (var ua = 0u; ua < nUtAnt; ua++) {
                let H_los = calc_los_coeff(
                    utCfg, ua, tZOA, pAOA, zetaUt,
                    bsCfg, ba, tZOD, pAOD, zetaBs,
                    snap_time, vel, cir_buf_cmn.lambda_0, lk.d3d
                );
                let hi = ua * NMAXTAPS; // tap 0
                wg_Hlink[hi] = cadd(H_los * los_scale, wg_Hlink[hi]);
            }
        }

        workgroupBarrier();

        // Path loss scaling: parallel across (ua, tap) slots; each
        // thread owns a strided slice of wg_Hlink (no overlap, no
        // race).
        if cir_uni_sys.disable_pl_shadowing == 0u {
            let path_gain = -(lk.pathloss - lk.SF);
            let ps        = pow(10.0, path_gain / 20.0);
            for (var idx = li.x; idx < NMAXTAPS * MAX_UE_ANT; idx += 24u) {
                wg_Hlink[idx] = wg_Hlink[idx] * ps;
            }
        }

        workgroupBarrier();

        let buf_len_u32 = arrayLength(&cir_buf_packed_out);

        // Write H_link -> cirCoe for this BS antenna (parallel
        // across taps). Each thread writes one tap; taps beyond
        // wg_tapCountTotal are skipped.
        let total_taps = wg_tapCountTotal;
        if li.x < total_taps {
            let ti = li.x;
            for (var ua = 0u; ua < nUtAnt; ua++) {
                let dst_u32 = snap_off_u32
                            + ((ua * nCellAnt + ba) * NMAXTAPS + ti) * 2u;
                if dst_u32 + 1u < buf_len_u32 {
                    let cur = pack_read_vec2f(dst_u32);
                    pack_write_vec2f(dst_u32, cadd(cur, wg_Hlink[ua * NMAXTAPS + ti]));
                }
            }
        }
    } // end BS antenna loop

    workgroupBarrier();

    // Sparse tap indexing (only thread 0, only snapshot 0). The
    // per-cluster pre-pass above already wrote tap_count_total values
    // into wg_tapIdx (no UINT_MAX sentinels needed). Sort+dedup in
    // place, then write to packed_out cirNormDelay / cirNtaps regions.
    if li.x == 0u && snapshot_idx == 0u {
        let tc = wg_tapCountTotal;

        // Bubble sort wg_tapIdx[0..tc)
        if tc > 1u {
            for (var i = 0u; i < tc - 1u; i++) {
                for (var j = 0u; j < tc - 1u - i; j++) {
                    if wg_tapIdx[j] > wg_tapIdx[j+1u] {
                        let tmp = wg_tapIdx[j]; wg_tapIdx[j] = wg_tapIdx[j+1u]; wg_tapIdx[j+1u] = tmp;
                    }
                }
            }
        }

        // Deduplicate (into packed_out, cirNormDelay region)
        var n_unique = 0u;
        if tc > 0u {
            cir_buf_packed_out[cir_disp.cirNormDelayRegionBase + al.cirNormDelayOffset] = wg_tapIdx[0];
            n_unique = 1u;
            for (var i = 1u; i < tc; i++) {
                if wg_tapIdx[i] != wg_tapIdx[n_unique - 1u] {
                    cir_buf_packed_out[cir_disp.cirNormDelayRegionBase + al.cirNormDelayOffset + n_unique] = wg_tapIdx[i];
                    n_unique++;
                }
            }
        }
        cir_buf_packed_out[cir_disp.cirNtapsRegionBase + al.cirNtapsOffset] = n_unique;
    }
}

// ---------------------------------------------------------------------------
// ================   KERNEL 3: generate_cfr_kernel_mode1   ===================
// Mode 1: CIR -> CFR per PRBG (one thread per PRBG, serial over ant pairs)
// Dispatch: grid(nActiveLinks, nSnapshots), block(N_Prbg, 1, 1)
// ---------------------------------------------------------------------------

var<workgroup> wg_cirDelayUs2Pi : array<f32, 24>;   // NMAXTAPS

@compute @workgroup_size(64, 1, 1)   // N_Prbg must be <= 64; adjust if needed
fn generate_cfr_kernel_mode1(
    @builtin(workgroup_id)       wg : vec3<u32>,
    @builtin(local_invocation_id) li: vec3<u32>
) {
    let active_link_idx = wg.x;
    let batch_idx       = wg.y;
    if active_link_idx >= cfr_disp.nActiveLinks { return; }

    let al  = cfr_buf_activeLink[active_link_idx];
    let cid = al.cid; let uid = al.uid;

    let ut   = cfr_buf_ut[uid];
    let cell = cfr_buf_cell[cid];
    let utCfg  = cfr_buf_antCfg[ut.antPanelIdx];
    let bsCfg  = cfr_buf_antCfg[cell.antPanelIdx];
    let nUtAnt  = utCfg.nAnt;
    let nCellAnt = bsCfg.nAnt;

    let N_Prbg     = cfr_uni_sim.n_prbg;
    let N_sc_Prbg  = u32(ceil(f32(cfr_uni_sim.n_prb * 12u) / f32(N_Prbg)));
    let N_sc_over2 = cfr_uni_sim.fft_size >> 1u;

    let prbg_idx = li.x;

    // Read CIR tap count and build normalised delay (2π * delay_us) in shared memory
    let cir_ntaps = min(cfr_buf_cirNtaps[al.cirNtapsOffset], NMAXTAPS);
    if (cir_ntaps == 0u || cir_ntaps > 1000u) { return; }  // sanity cap

    if li.x == 0u {
        for (var i = 0u; i < cir_ntaps; i++) {
            let delay_us = f32(cfr_buf_cirNormDelay[al.cirNormDelayOffset + i]) * 1e6
                         / (cfr_uni_sim.sc_spacing_hz * f32(cfr_uni_sim.fft_size));
            wg_cirDelayUs2Pi[i] = TWO_PI * delay_us;
        }
    }
    workgroupBarrier();

    if prbg_idx >= N_Prbg { return; }

    let optional_dim = (cfr_uni_sim.optional_cfr_dim == 1u);

    // Iterate over all antenna pairs
    for (var ua = 0u; ua < nUtAnt; ua++) {
        for (var ba = 0u; ba < nCellAnt; ba++) {

            // CIR source offset for this (batch, ua, ba)
            let cir_base = al.cirCoeOffset
                         + batch_idx * nUtAnt * nCellAnt * NMAXTAPS
                         + (ua * nCellAnt + ba) * NMAXTAPS;

            // CFR destination offset
            var cfr_dst = 0u;
            if optional_dim {
                cfr_dst = al.freqChanPrbgOffset
                        + ((batch_idx * N_Prbg + prbg_idx) * nUtAnt + ua) * nCellAnt + ba;
            } else {
                cfr_dst = al.freqChanPrbgOffset
                        + ((batch_idx * nUtAnt + ua) * nCellAnt + ba) * N_Prbg + prbg_idx;
            }

            // Frequency of first SC in this PRBG (kHz)
            let local_sc = prbg_idx * N_sc_Prbg;
            let freq_khz = f32(i32(local_sc) - i32(N_sc_over2))
                         * cfr_uni_sim.sc_spacing_hz * 1e-3;

            // calCfrbyCir
            var cfr = vec2f(0.0);
            for (var tap = 0u; tap < cir_ntaps; tap++) {
                let delay = wg_cirDelayUs2Pi[tap];
                let phase = -freq_khz * delay * 1e-3;
                if abs(phase) <= 1e6 {
                    let coeff = cfr_buf_cirCoe[cir_base + tap];
                    cfr = cadd(cfr, cmul(coeff, cexp_j(phase)));
                }
            }
            cfr = cfr * cfr_disp.cfr_norm;
            cfr_buf_freqChanPrbg[cfr_dst] = cfr;
        }
    }
}

// ---------------------------------------------------------------------------
// gen_channel_matrix_kernel
//
// Produces the per-link Complex3DVector(uElem, sElem, numOverallCluster) that
// ThreeGppChannelModel::GetNewChannel returns on CPU. The CPU
// GenerateChannelCoefficients (3GPP TR 38.901 Eq 7.5-22 & 7.5-28) does:
//
//   hUsn(u, s, n) = sqrt(P_n / M) * sum over rays m of:
//       polarisation-weighted field response F_uPolU(rayAoa, rayZoa) * conj-Hermitian-style
//                                        x F_sPolS(rayAod, rayZod)
//       * array-steering rxPhase(u, ray)
//       * array-steering txPhase(s, ray)
//
// where for each ray the polarisation-weighted response uses Phi_thetaTheta /
// Phi_thetaPhi / Phi_phiTheta / Phi_phiPhi (4 random phases + xpr factor).
//
// For the two strongest clusters (cluster1st, cluster2nd) the rays split into
// three subclusters that become three output pages. For LOS links cluster 0
// gets an additional Rician term scaled by sqrt(K/(K+1)).
//
// SCAFFOLD STATUS (this commit): kernel is in place with the correct dispatch
// shape and output layout, but the per-cell math is a placeholder. The next
// commit will replace `compute_cell` with the full ray sum, antenna field
// pattern evaluation, and subcluster / LOS handling. Once the math is in,
// the mezanine override of GetNewChannel will return the cached matrix and
// the CPU GetNewChannel cost (~5 ms/eval, the 84% bottleneck) collapses.
// ---------------------------------------------------------------------------

struct ChanMatDispatch {
    n_active_links     : u32,
    n_overall_cluster  : u32,
    u_size             : u32,
    s_size             : u32,
    n_rays             : u32,
    _pad0              : u32,
    _pad1              : u32,
    _pad2              : u32,
    lambda0            : f32, // c / frequency
    _pad3              : f32,
    _pad4              : f32,
    _pad5              : f32,
}

// Matrix-kernel bindings live in group(0) at numbers 30+ to avoid
// clashing with the LSP kernels' bindings (0..15).
@group(0) @binding(30) var<uniform>             mat_disp        : ChanMatDispatch;
@group(0) @binding(31) var<storage, read>       mat_buf_link    : array<LinkParams>;
@group(0) @binding(32) var<storage, read>       mat_buf_cluster : array<ClusterParams>;
@group(0) @binding(33) var<storage, read>       mat_packed_in   : array<f32>;
@group(0) @binding(34) var<storage, read>       mat_buf_active  : array<ActiveLink>;
@group(0) @binding(35) var<storage, read>       mat_buf_cell    : array<SspCellParam>;
@group(0) @binding(36) var<storage, read>       mat_buf_ut      : array<SspUtParam>;
@group(0) @binding(37) var<storage, read>       mat_buf_antCfg  : array<AntPanelConfig>;
@group(0) @binding(38) var<storage, read>       mat_buf_antTheta: array<f32>;
@group(0) @binding(39) var<storage, read>       mat_buf_antPhi  : array<f32>;
@group(0) @binding(40) var<storage, read_write> mat_buf_out     : array<vec2f>;

// Chunk (e): field-pattern eval. Reads mat_buf_antTheta/Phi.
fn mat_field_components(cfg: AntPanelConfig, theta: f32, phi: f32, zeta: f32) -> vec2f {
    var ti = i32(round(theta));
    var pi_ = i32(round(phi));
    if ti < 0 || ti >= 360 {
        ti = ti % 360;
        if ti < 0 { ti += 360; }
    }
    if ti > 180 { ti = 360 - ti; }
    if pi_ < 0 || pi_ > 359 {
        pi_ = pi_ % 360;
        if pi_ < 0 { pi_ += 360; }
    }
    let A_db = mat_buf_antTheta[cfg.thetaOffset + u32(ti)]
             + mat_buf_antPhi  [cfg.phiOffset   + u32(pi_)]
             + select(0.0, GMAX, cfg.antModel == 1u);
    let A_sqrt = pow(10.0, A_db / 20.0);
    let z_rad  = zeta * DEG2RAD;
    return vec2f(A_sqrt * cos(z_rad), A_sqrt * sin(z_rad));
}

// CPU UniformPlanarArray uses a HALF-BASED dual-pol layout: the
// first half of element indices (0..M*N-1) carry polarisation 0,
// the second half (M*N..2*M*N-1) carry polarisation 1. Within each
// half, elements walk columns then rows: elem k -> row=k/N, col=k%N.
// CPU::GetElementLocation returns a 3D position: x=0, y=col*d_h,
// z=row*d_v (for default panel orientation). The downstream array-
// steering phase = 2*pi*(sin*cos*x + sin*sin*y + cos*z), so the
// row direction picks up cos(ZOA) and the col direction picks up
// sin(ZOA)*sin(AOA). Returning a 3D vec3f preserves that -- an
// earlier draft used a vec2f and silently dropped the z component,
// which made the row*d_v contribution use the wrong projection
// angle and collapsed beam coherency on any vertically-extended
// array (8x8 BS at DenseAmimoIntel scale).
fn mat_elem_pos(cfg: AntPanelConfig, elem_idx: u32) -> vec3f {
    let M = cfg.antSize[2];
    let N = cfg.antSize[3];
    let MN = M * N;
    let local = select(elem_idx, elem_idx - MN, elem_idx >= MN);
    let row = local / N;
    let col = local % N;
    return vec3f(
        0.0,
        f32(col) * cfg.antSpacing[2], // y = d_h * col
        f32(row) * cfg.antSpacing[3]  // z = d_v * row
    );
}

fn mat_pol_idx(cfg: AntPanelConfig, elem_idx: u32) -> u32 {
    let M = cfg.antSize[2];
    let N = cfg.antSize[3];
    return select(0u, 1u, elem_idx >= M * N);
}

// 3GPP TR 38.901 Table 7.5-5 subcluster ray groupings (0-indexed).
// CPU GenerateChannelCoefficients switch-case at three-gpp-channel-model.cc:
//   case 9,10,11,12,17,18  -> subcluster 2
//   case 13,14,15,16       -> subcluster 3
//   default (0..8, 19)     -> subcluster 1
// flt: 0=all, 1=sub1, 2=sub2, 3=sub3.
// NB: parameter is `flt` not `filter` -- WGSL reserves `filter` for
// texture sampling and even an unused parameter named `filter` taints
// the surrounding module, breaking unrelated pipelines silently.
fn mat_ray_passes_filter(m: u32, flt: u32) -> bool {
    if flt == 0u { return true; }
    if flt == 2u {
        let a = m >= 9u && m <= 12u;
        let b = m == 17u || m == 18u;
        return a || b;
    }
    if flt == 3u {
        return m >= 13u && m <= 16u;
    }
    if m >= 9u && m <= 18u {
        return false;
    }
    return true;
}

// Max #overall-cluster pages we'll ever write. The CPU mezanine sizes the
// readback to use the per-link actual count from
// channelParams->m_reducedClusterNumber so unused pages just stay zeroed.
const MAT_MAX_PAGES : u32 = 24u;

@compute @workgroup_size(64, 1, 1)
fn gen_channel_matrix_kernel(
    @builtin(workgroup_id)       wg : vec3<u32>,
    @builtin(local_invocation_id) li : vec3<u32>
) {
    let link_idx = wg.x;
    let page_idx = wg.y;
    let us_idx   = wg.z * 64u + li.x;

    if link_idx >= mat_disp.n_active_links { return; }
    if page_idx >= mat_disp.n_overall_cluster { return; }
    if us_idx   >= mat_disp.u_size * mat_disp.s_size { return; }

    let u_idx = us_idx / mat_disp.s_size;
    let s_idx = us_idx % mat_disp.s_size;

    let al = mat_buf_active[link_idx];
    let lk = mat_buf_link[al.lspReadIdx];
    let cp = mat_buf_cluster[al.lspReadIdx];

    // ── Per-link page bounds: skip work for output pages past the link's
    // actual numOverallCluster (n_red + 2|4 depending on whether the two
    // strongest cluster indices differ).
    let n_red = cp.nCluster;
    let n_overall = select(n_red + 2u, n_red + 4u,
                            cp.strongest2[0] != cp.strongest2[1]);
    if page_idx >= n_overall {
        // Still need to write the unused slot so the CPU sees a
        // deterministic value (column-major (u,s,n) within per-link slab).
        let per_page = mat_disp.u_size * mat_disp.s_size;
        let per_link = per_page * MAT_MAX_PAGES;
        let out_idx = link_idx * per_link + page_idx * per_page
                    + s_idx * mat_disp.u_size + u_idx;
        mat_buf_out[out_idx] = vec2f(0.0, 0.0);
        return;
    }

    let cell = mat_buf_cell[al.cid];
    let ut   = mat_buf_ut[al.uid];
    let bsCfg = mat_buf_antCfg[cell.antPanelIdx];
    let utCfg = mat_buf_antCfg[ut.antPanelIdx];

    // Map output page -> (input_cluster, ray_filter).
    let c1 = cp.strongest2[0];
    let c2 = cp.strongest2[1];
    let minStr = min(c1, c2);
    let maxStr = max(c1, c2);
    var src_cluster: u32 = 0u;
    var ray_flt: u32 = 0u;
    if page_idx < n_red {
        src_cluster = page_idx;
        ray_flt = select(0u, 1u, page_idx == c1 || page_idx == c2);
    } else if page_idx == n_red {
        src_cluster = minStr; ray_flt = 2u;
    } else if page_idx == n_red + 1u {
        src_cluster = minStr; ray_flt = 3u;
    } else if page_idx == n_red + 2u {
        src_cluster = maxStr; ray_flt = 2u;
    } else {
        src_cluster = maxStr; ray_flt = 3u;
    }

    let d_rx = mat_elem_pos(utCfg, u_idx);
    let d_tx = mat_elem_pos(bsCfg, s_idx);
    let polU = mat_pol_idx(utCfg, u_idx);
    let polS = mat_pol_idx(bsCfg, s_idx);
    let zetaUt = 0.0;
    let zetaBs = 0.0;

    let n_rays = mat_disp.n_rays;
    let cluster_scale = sqrt(cp.powers[src_cluster] / f32(n_rays));

    let link_base = al.lspReadIdx * PACKED_LINK_STRIDE;
    let ray_local_base = src_cluster * MAX_RAYS;

    var rays_re = 0.0;
    var rays_im = 0.0;
    for (var m = 0u; m < n_rays; m++) {
        if !mat_ray_passes_filter(m, ray_flt) { continue; }
        let ray_local = ray_local_base + m;
        let pAOA = mat_packed_in[link_base + PACKED_OFF_AOA + ray_local];
        let pAOD = mat_packed_in[link_base + PACKED_OFF_AOD + ray_local];
        let tZOA = mat_packed_in[link_base + PACKED_OFF_ZOA + ray_local];
        let tZOD = mat_packed_in[link_base + PACKED_OFF_ZOD + ray_local];
        let xpr  = mat_packed_in[link_base + PACKED_OFF_XPR + ray_local];
        let rph_off = link_base + PACKED_OFF_RNDP + ray_local * 4u;

        let Frx = mat_field_components(utCfg, tZOA, pAOA, utCfg.antPolarAngles[polU] + zetaUt);
        let Ftx = mat_field_components(bsCfg, tZOD, pAOD, bsCfg.antPolarAngles[polS] + zetaBs);

        let tZOA_r = tZOA * DEG2RAD;
        let pAOA_r = pAOA * DEG2RAD;
        let tZOD_r = tZOD * DEG2RAD;
        let pAOD_r = pAOD * DEG2RAD;
        let rRx = vec3f(sin(tZOA_r)*cos(pAOA_r), sin(tZOA_r)*sin(pAOA_r), cos(tZOA_r));
        let rTx = vec3f(sin(tZOD_r)*cos(pAOD_r), sin(tZOD_r)*sin(pAOD_r), cos(tZOD_r));
        let dot_rx = dot(rRx, d_rx);
        let dot_tx = dot(rTx, d_tx);
        let term4 = cexp_j(TWO_PI * dot_rx);
        let term5 = cexp_j(TWO_PI * dot_tx);

        let sqrt_kappa = sqrt(max(xpr, 1e-10));
        let ph0 = mat_packed_in[rph_off    ] * DEG2RAD;
        let ph1 = mat_packed_in[rph_off + 1u] * DEG2RAD;
        let ph2 = mat_packed_in[rph_off + 2u] * DEG2RAD;
        let ph3 = mat_packed_in[rph_off + 3u] * DEG2RAD;
        let t200 = cexp_j(ph0);
        let t201 = vec2f(cos(ph1)/sqrt_kappa, sin(ph1)/sqrt_kappa);
        let t210 = vec2f(cos(ph2)/sqrt_kappa, sin(ph2)/sqrt_kappa);
        let t211 = cexp_j(ph3);

        // mat_field_components returns vec2f(F_theta, F_phi) (see
        // its comment). 3GPP TR 38.901 Eq 7.5-22 pairs ph0..ph3
        // with (Theta*Theta, Theta*Phi, Phi*Theta, Phi*Phi):
        //   ray = ph0 * F_rxTheta * F_txTheta
        //       + ph1 * F_rxTheta * F_txPhi    [sqrt(1/k) in ph1]
        //       + ph2 * F_rxPhi   * F_txTheta  [sqrt(1/k) in ph2]
        //       + ph3 * F_rxPhi   * F_txPhi
        // Since Frx.x = F_theta and Frx.y = F_phi, t1_0 is F_theta
        // and t1_1 is F_phi -- pairings (i, j) -> (theta, phi)
        // basis come out as in calc_ray_coeff.
        let t1_0 = vec2f(Frx.x, 0.0);
        let t1_1 = vec2f(Frx.y, 0.0);
        let t3_0 = vec2f(Ftx.x, 0.0);
        let t3_1 = vec2f(Ftx.y, 0.0);

        var ray = vec2f(0.0);
        ray = cadd(ray, cmul(cmul(cmul(cmul(t1_0, t200), t3_0), term4), term5));
        ray = cadd(ray, cmul(cmul(cmul(cmul(t1_0, t201), t3_1), term4), term5));
        ray = cadd(ray, cmul(cmul(cmul(cmul(t1_1, t210), t3_0), term4), term5));
        ray = cadd(ray, cmul(cmul(cmul(cmul(t1_1, t211), t3_1), term4), term5));

        rays_re += ray.x;
        rays_im += ray.y;
    }
    var cell_val = vec2f(rays_re * cluster_scale, rays_im * cluster_scale);

    // ── LOS Rician term (3GPP TR 38.901 Eq 7.5-29/30) ───────────────
    // For LOS links every page is scaled by atten1 = sqrt(1/(K+1));
    // page 0 additionally gets sqrt(K/(K+1)) * LOS_ray. lk.K is in
    // linear units (cal_link_param_kernel converted from dB before
    // writing LinkParams).
    if lk.losInd != 0u {
        let k_lin = max(lk.K, 1e-10);
        let atten1 = sqrt(1.0 / (k_lin + 1.0));
        let attenK = sqrt(k_lin / (k_lin + 1.0));
        cell_val = cell_val * atten1;
        if page_idx == 0u && mat_disp.lambda0 > 0.0 {
            let tZOA = lk.theta_LOS_ZOA;
            let pAOA = lk.phi_LOS_AOA;
            let tZOD = lk.theta_LOS_ZOD;
            let pAOD = lk.phi_LOS_AOD;
            let Frx = mat_field_components(utCfg, tZOA, pAOA, utCfg.antPolarAngles[polU] + zetaUt);
            let Ftx = mat_field_components(bsCfg, tZOD, pAOD, bsCfg.antPolarAngles[polS] + zetaBs);
            let tZOA_r = tZOA * DEG2RAD;
            let pAOA_r = pAOA * DEG2RAD;
            let tZOD_r = tZOD * DEG2RAD;
            let pAOD_r = pAOD * DEG2RAD;
            let rRx = vec3f(sin(tZOA_r)*cos(pAOA_r), sin(tZOA_r)*sin(pAOA_r), cos(tZOA_r));
            let rTx = vec3f(sin(tZOD_r)*cos(pAOD_r), sin(tZOD_r)*sin(pAOD_r), cos(tZOD_r));
            let dot_rx = dot(rRx, d_rx);
            let dot_tx = dot(rTx, d_tx);
            let term4 = cexp_j(TWO_PI * dot_rx);
            let term5 = cexp_j(TWO_PI * dot_tx);
            // [1 0; 0 -1] polarisation matrix (3GPP Eq 7.5-29) collapses
            // to F_uTheta * F_sTheta - F_uPhi * F_sPhi.
            let pol_combo = vec2f(Frx.x * Ftx.x - Frx.y * Ftx.y, 0.0);
            let term7 = cexp_j(-TWO_PI * lk.d3d / mat_disp.lambda0);
            let los_ray = cmul(cmul(cmul(pol_combo, term7), term4), term5);
            cell_val = cell_val + attenK * los_ray;
        }
    }

    // Column-major (u, s, n) inside per-link block.
    let per_page = mat_disp.u_size * mat_disp.s_size;
    let per_link = per_page * MAT_MAX_PAGES;
    let out_idx  = link_idx * per_link
                 + page_idx * per_page
                 + s_idx    * mat_disp.u_size
                 + u_idx;
    mat_buf_out[out_idx] = cell_val;
}

// ---------------------------------------------------------------------------
// gen_long_term_kernel
//
// Replaces the CPU CalcLongTerm matrix-multiply that PRX::GetLongTerm runs
// every per-link DoCalcRxPowerSpectralDensity call. For each (link, sPort,
// uPort, cluster), reduce the channel matrix along the per-port element
// block by the (sW, conj(uW)) beam weights:
//
//   longTerm[u, s, c] = sum_t sW[t] * sum_r conj(uW[r]) * H[startU+r, startS+t, c]
//
// where (startU, startS) come from ArrayIndexFromPortIndex(uPort, 0) and
// the inner indices walk with the sub-array partition stride (sIncVal /
// uIncVal hop at every elemsPerPort-th step, mirroring the CPU loop).
//
// Bindings live at 50+ to stay clear of the LSP kernels (0..15) and the
// matrix kernel (30..40).
// ---------------------------------------------------------------------------

struct LongTermDispatch {
    n_active_links     : u32,
    u_size             : u32,  // numRows of channel matrix (= total UE elements)
    s_size             : u32,  // numCols of channel matrix (= total BS elements)
    n_pages            : u32,  // = MAT_MAX_PAGES
    s_ports            : u32,
    u_ports            : u32,
    s_port_elems       : u32,  // sAnt->GetNumElemsPerPort()
    u_port_elems       : u32,  // uAnt->GetNumElemsPerPort()
    s_elems_per_port   : u32,  // sAnt->GetHElemsPerPort() (0 -> no hop)
    u_elems_per_port   : u32,  // uAnt->GetHElemsPerPort()
    s_inc_val          : u32,  // sAnt->GetNumColumns() - sElemsPerPort
    u_inc_val          : u32,  // uAnt->GetNumColumns() - uElemsPerPort
}

@group(0) @binding(50) var<uniform>             lt_disp     : LongTermDispatch;
// lt_chan_in is the same backing buffer as mat_buf_out — the channel
// matrix that gen_channel_matrix_kernel just produced.
@group(0) @binding(51) var<storage, read>       lt_chan_in  : array<vec2f>;
@group(0) @binding(52) var<storage, read>       lt_s_w      : array<vec2f>;  // per-link [s_port_elems]
@group(0) @binding(53) var<storage, read>       lt_u_w      : array<vec2f>;  // per-link [u_port_elems]
@group(0) @binding(54) var<storage, read>       lt_start_s  : array<u32>;    // [s_ports]
@group(0) @binding(55) var<storage, read>       lt_start_u  : array<u32>;    // [u_ports]
@group(0) @binding(56) var<storage, read_write> lt_out      : array<vec2f>;  // per-link [u_ports*s_ports*n_pages]

@compute @workgroup_size(64)
fn gen_long_term_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let per_link_lt = lt_disp.u_ports * lt_disp.s_ports * lt_disp.n_pages;
    let total = lt_disp.n_active_links * per_link_lt;
    if (tid >= total) { return; }

    let link_idx = tid / per_link_lt;
    let in_link  = tid % per_link_lt;
    let ports_per_page = lt_disp.u_ports * lt_disp.s_ports;
    let c_index = in_link / ports_per_page;
    let pp      = in_link % ports_per_page;
    // Output column-major (u, s, c) -> u + uPorts*s + uPorts*sPorts*c
    let u_port = pp % lt_disp.u_ports;
    let s_port = pp / lt_disp.u_ports;

    let per_link_mat = lt_disp.u_size * lt_disp.s_size * lt_disp.n_pages;
    let page_base = link_idx * per_link_mat + c_index * lt_disp.u_size * lt_disp.s_size;
    let s_w_base = link_idx * lt_disp.s_port_elems;
    let u_w_base = link_idx * lt_disp.u_port_elems;
    let start_s = lt_start_s[s_port];
    let start_u = lt_start_u[u_port];

    let s_need_hop = lt_disp.s_elems_per_port > 0u;
    let u_need_hop = lt_disp.u_elems_per_port > 0u;

    var tx_sum = vec2f(0.0, 0.0);
    var s_index = start_s;
    for (var t_index: u32 = 0u; t_index < lt_disp.s_port_elems; t_index = t_index + 1u) {
        var rx_sum = vec2f(0.0, 0.0);
        var u_index = start_u;
        for (var r_index: u32 = 0u; r_index < lt_disp.u_port_elems; r_index = r_index + 1u) {
            let cell = page_base + u_index + lt_disp.u_size * s_index;
            let h = lt_chan_in[cell];
            let uw = lt_u_w[u_w_base + r_index];
            // conj(uw) = (uw.x, -uw.y); cmul((a,-b), (c,d)) = (ac + bd, ad - bc)
            rx_sum = rx_sum + vec2f(uw.x * h.x + uw.y * h.y,
                                    uw.x * h.y - uw.y * h.x);
            u_index = u_index + 1u;
            if (u_need_hop && (r_index % lt_disp.u_elems_per_port == lt_disp.u_elems_per_port - 1u)) {
                u_index = u_index + lt_disp.u_inc_val;
            }
        }
        let sw = lt_s_w[s_w_base + t_index];
        // sw * rx_sum
        tx_sum = tx_sum + vec2f(sw.x * rx_sum.x - sw.y * rx_sum.y,
                                sw.x * rx_sum.y + sw.y * rx_sum.x);
        s_index = s_index + 1u;
        if (s_need_hop && (t_index % lt_disp.s_elems_per_port == lt_disp.s_elems_per_port - 1u)) {
            s_index = s_index + lt_disp.s_inc_val;
        }
    }

    let out_idx = link_idx * per_link_lt
                + c_index * ports_per_page
                + s_port * lt_disp.u_ports
                + u_port;
    lt_out[out_idx] = tx_sum;
}

// ---------------------------------------------------------------------------
// gen_spec_chan_kernel
//
// Replaces PRX::GenSpectrumChannelMatrix's per-cluster outer-product
// contraction. For each output cell (rx, tx, rb), reduce the longTerm
// (already on GPU, output of gen_long_term_kernel) by delayT[c, rb]
// (doppler * delaySincos, packed on CPU because both factors vary per
// eval), then scale by sqrt(inPsd[rb]).
//
//   chanSpct[rx, tx, rb] = sqrt(vit[rb]) * sum_c longTerm[rx, tx, c]
//                                              * delayT[c, rb]
//
// longTerm in mezanine cache has shape (u_ports = UE_ports, s_ports =
// BS_ports, n_pages). For !isReverse the mapping is (u=rx, s=tx); for
// isReverse it's (u=tx, s=rx). The transpose is folded into the index
// computation.
//
// Bindings live at 60+ to stay clear of the LSP kernels (0..15), the
// matrix kernel (30..40), and the long-term kernel (50..56).
// ---------------------------------------------------------------------------

struct SpecChanDispatch {
    n_clusters       : u32,
    n_rb             : u32,
    n_rx_ports       : u32,
    n_tx_ports       : u32,
    is_reverse       : u32,  // 0 or 1
    link_idx         : u32,  // which per-link slab in spec_long_term
    lt_u_ports       : u32,  // longTerm u-dim (UE ports in mezanine convention)
    lt_s_ports       : u32,  // longTerm s-dim (BS ports)
    lt_n_pages       : u32,  // longTerm per-link page count (kMatMaxPages)
    _pad0            : u32,
    _pad1            : u32,
    _pad2            : u32,
}

@group(0) @binding(60) var<uniform>             spec_disp      : SpecChanDispatch;
@group(0) @binding(61) var<storage, read>       spec_long_term : array<vec2f>;
@group(0) @binding(62) var<storage, read>       spec_delay_t   : array<vec2f>;  // [n_clusters * n_rb]
@group(0) @binding(63) var<storage, read>       spec_sqrt_vit  : array<f32>;     // [n_rb]
@group(0) @binding(64) var<storage, read_write> spec_chan_out  : array<vec2f>;   // [n_rx_ports * n_tx_ports * n_rb]

@compute @workgroup_size(64)
fn gen_spec_chan_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let per_page = spec_disp.n_rx_ports * spec_disp.n_tx_ports;
    let total = per_page * spec_disp.n_rb;
    if (tid >= total) { return; }

    // chanSpct column-major: (rx, tx, rb) at rx + n_rx_ports*tx + per_page*rb.
    let rb = tid / per_page;
    let pp = tid % per_page;
    let rx = pp % spec_disp.n_rx_ports;
    let tx = pp / spec_disp.n_rx_ports;

    let lt_per_page = spec_disp.lt_u_ports * spec_disp.lt_s_ports;
    let per_link_lt = lt_per_page * spec_disp.lt_n_pages;
    let lt_base = spec_disp.link_idx * per_link_lt;

    // For !isReverse: longTerm shape (rxPorts, txPorts, c) -> u=rx, s=tx.
    // For isReverse: longTerm was generated in opposite direction; PRX::GenSpec
    // transposes it so (u, s) maps to (tx, rx) at the read side.
    var u_idx: u32;
    var s_idx: u32;
    if (spec_disp.is_reverse == 1u) {
        u_idx = tx;
        s_idx = rx;
    } else {
        u_idx = rx;
        s_idx = tx;
    }

    var acc = vec2f(0.0, 0.0);
    for (var c: u32 = 0u; c < spec_disp.n_clusters; c = c + 1u) {
        let lt_idx = lt_base + c * lt_per_page + s_idx * spec_disp.lt_u_ports + u_idx;
        let lt = spec_long_term[lt_idx];
        let dt = spec_delay_t[c * spec_disp.n_rb + rb];
        // chanSpct += longTerm[u,s,c] * delayT[c, rb]
        acc = acc + vec2f(lt.x * dt.x - lt.y * dt.y,
                          lt.x * dt.y + lt.y * dt.x);
    }
    let s = spec_sqrt_vit[rb];
    acc = vec2f(acc.x * s, acc.y * s);
    let out_idx = rb * per_page + tx * spec_disp.n_rx_ports + rx;
    spec_chan_out[out_idx] = acc;
}

// ---------------------------------------------------------------------------
// gen_spec_batch_kernel
//
// Per-tick batched version of gen_spec_chan_kernel. Processes ALL active
// links in a single dispatch and writes the *unscaled* per-link chanSpct
// (i.e. without the sqrt(PSD[rb]) factor). The per-eval hook then just
// does a hash-keyed lookup + the cheap per-PRB sqrt scale on CPU --
// amortising the GPU dispatch + readback over hundreds of evals.
//
// Inputs in addition to gen_spec_chan_kernel:
//   - sb_delays   [n_links * n_clusters] f32 cluster delays
//   - sb_doppler  [n_links * n_clusters] vec2f doppler factor (CPU-computed
//                 from snapshotted angles + speeds + Simulator::Now())
//   - sb_rb_freqs [n_rb] f32 subband centre frequencies (captured from
//                 inPsd on the first eval, then re-uploaded each tick)
// Outputs:
//   - sb_chan_out [n_links * (n_rx*n_tx*n_rb)] vec2f
//                 chanSpct_unscaled (rx, tx, rb) column-major per link
//
// delaySincos is computed inside the kernel as exp(-j*2*pi*fc*tau)
// rather than uploaded from CPU. f32 precision over the typical 3GPP
// delay range (~ns to ~us) at mid-band frequencies stays well under
// the few-percent margin that PSD aggregate equivalence requires.
//
// Bindings @group(0) at 70..75 to stay clear of the per-eval GenSpec
// kernel (60..64) and the long-term kernel (50..56).
// ---------------------------------------------------------------------------

struct SpecBatchDispatch {
    n_links            : u32,
    n_clusters         : u32,
    n_rb               : u32,
    n_rx_ports         : u32,
    n_tx_ports         : u32,
    lt_u_ports         : u32,
    lt_s_ports         : u32,
    lt_n_pages         : u32,
}

@group(0) @binding(70) var<uniform>             sb_disp      : SpecBatchDispatch;
@group(0) @binding(71) var<storage, read>       sb_long_term : array<vec2f>;
@group(0) @binding(72) var<storage, read>       sb_doppler   : array<vec2f>;
@group(0) @binding(73) var<storage, read>       sb_delays    : array<f32>;
@group(0) @binding(74) var<storage, read>       sb_rb_freqs  : array<f32>;
@group(0) @binding(75) var<storage, read_write> sb_chan_out  : array<vec2f>;

const SB_NEG_TWO_PI: f32 = -6.28318530717958647692;

@compute @workgroup_size(64)
fn gen_spec_batch_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let per_page = sb_disp.n_rx_ports * sb_disp.n_tx_ports;
    let per_link_out = per_page * sb_disp.n_rb;
    let total = sb_disp.n_links * per_link_out;
    if (tid >= total) { return; }

    let link_idx = tid / per_link_out;
    let in_link  = tid % per_link_out;
    let rb = in_link / per_page;
    let pp = in_link % per_page;
    let rx = pp % sb_disp.n_rx_ports;
    let tx = pp / sb_disp.n_rx_ports;

    let lt_per_page = sb_disp.lt_u_ports * sb_disp.lt_s_ports;
    let per_link_lt = lt_per_page * sb_disp.lt_n_pages;
    let lt_base = link_idx * per_link_lt;

    let dop_base = link_idx * sb_disp.n_clusters;
    let del_base = link_idx * sb_disp.n_clusters;

    let fc = sb_rb_freqs[rb];

    // Non-reverse direction only -- the mezanine path always generates
    // channelMatrix in the BS->UE direction so for downlink evals
    // (a=BS, b=UE) isReverse=false and (u, s) maps to (rx, tx). Uplink
    // (isReverse=true) currently isn't covered by this batched path
    // and the caller falls back to CPU GenSpec.
    var acc = vec2f(0.0, 0.0);
    for (var c: u32 = 0u; c < sb_disp.n_clusters; c = c + 1u) {
        // delaySincos[c, rb] = exp(-j * 2*pi * fc * delay[c])
        let theta = SB_NEG_TWO_PI * fc * sb_delays[del_base + c];
        let ds = vec2f(cos(theta), sin(theta));
        let dop = sb_doppler[dop_base + c];
        // delayT = doppler * delaySincos
        let dt = vec2f(dop.x * ds.x - dop.y * ds.y,
                       dop.x * ds.y + dop.y * ds.x);
        // longTerm[u=rx, s=tx, c]
        let lt_idx = lt_base + c * lt_per_page + tx * sb_disp.lt_u_ports + rx;
        let lt = sb_long_term[lt_idx];
        acc = acc + vec2f(lt.x * dt.x - lt.y * dt.y,
                          lt.x * dt.y + lt.y * dt.x);
    }
    // NOTE: NOT scaled by sqrt(PSD[rb]) -- that happens per eval on CPU.
    let out_idx = link_idx * per_link_out + rb * per_page
                + tx * sb_disp.n_rx_ports + rx;
    sb_chan_out[out_idx] = acc;
}
