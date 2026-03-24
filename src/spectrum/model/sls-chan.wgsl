// This is an AI translated version of the original Nvidia source for GPU compute,
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

const LOS_MATRIX_SIZE:  u32 = 7u;
const NLOS_MATRIX_SIZE: u32 = 6u;
const O2I_MATRIX_SIZE:  u32 = 6u;

// Max filter length = 2·floor(3·120)+1 = 721
const MAX_FILTER_LEN: u32 = 721u;

// ── Structs ────────────────────────────────────────────────────────────────
// Pad every struct to 16-byte multiples so std430 layout matches host-side.

struct Vec3f { x: f32, y: f32, z: f32, _p: f32 }

struct CellParam {
    loc: Vec3f,
    // Add remaining CellParam fields from sls_chan.cuh here
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
    o2i_building_penetr_loss_ind: u32,
    o2i_car_penetr_loss_ind:      u32,
    _p:                           u32,
    force_los_prob_indoor:        f32,  // force_los_prob[0]
    force_los_prob_outdoor:       f32,  // force_los_prob[1]
    _p2:                          vec2<f32>,
}

struct SimConfig {
    center_freq_hz: f32,
    _p0: f32, _p1: f32, _p2: f32,
}

struct CmnLinkParams {
    sqrtCorrMatLos:  array<f32, 49>,  // 7×7 lower-triangular
    sqrtCorrMatNlos: array<f32, 36>,  // 6×6
    sqrtCorrMatO2i:  array<f32, 36>,  // 6×6
    mu_K:        array<f32, 4>,  sigma_K:     array<f32, 4>,
    mu_lgDS:     array<f32, 4>,  sigma_lgDS:  array<f32, 4>,
    mu_lgASD:    array<f32, 4>,  sigma_lgASD: array<f32, 4>,
    mu_lgASA:    array<f32, 4>,  sigma_lgASA: array<f32, 4>,
    mu_lgZSA:    array<f32, 4>,  sigma_lgZSA: array<f32, 4>,
    lgfc: f32,
    _pad: array<f32, 3>,
}

struct LinkParams {
    d2d: f32, d2d_in: f32, d2d_out: f32, d3d: f32,
    d3d_in: f32, d3d_out: f32,
    phi_LOS_AOD: f32, phi_LOS_AOA: f32,
    theta_LOS_ZOD: f32, theta_LOS_ZOA: f32,
    losInd: u32, pathloss: f32,
    SF: f32, K: f32, DS: f32, ASD: f32, ASA: f32, ZSD: f32, ZSA: f32,
    mu_lgZSD: f32, sigma_lgZSD: f32, mu_offset_ZOD: f32,
    _pad: f32,
}

// Uniforms for calLinkParamKernel
struct LinkParamUniforms {
    maxX: f32, minX: f32, maxY: f32, minY: f32,
    nSite: u32, nUT: u32, nSectorPerSite: u32,
    updatePLAndPenetrationLoss: u32,  // bool as u32
    updateAllLSPs: u32,
    updateLosState: u32,
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
}

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
        h_e  = 12.0 + f32(i32(rand_uniform(s) * f32(n))) * 3.0;
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
                let lpl = uma_los_pl(d2d, d3d, h_bs, h_ut, fc, s);
                return max(lpl, 13.54 + 39.08*log10(d3d) + 20.0*log10(fc) - 0.6*(h_ut-1.5));
            }
            case SCENARIO_UMI: {
                let lpl = umi_los_pl(d2d, d3d, h_bs, h_ut, fc);
                return max(lpl, 35.3*log10(d3d) + 22.4 + 21.3*log10(fc) - 0.3*(h_ut-1.5));
            }
            case SCENARIO_RMA: {
                let lpl = rma_los_pl(d2d, d3d, h_bs, h_ut, fc);
                let W = 20.0; let h = 5.0;
                let nlos = 161.04 - 7.1*log10(W) + 7.5*log10(h)
                         - (24.37 - 3.7*pow(h/h_bs, 2.0)) * log10(h_bs)
                         + (43.42 - 3.1*log10(h_bs)) * (log10(d3d) - 3.0)
                         + 20.0*log10(fc)
                         - (3.2*pow(log10(11.75*h_ut), 2.0) - 4.97);
                return max(lpl, nlos);
            }
            default: { return 0.0; }
        }
    }
}

fn cal_sf_std(scenario: u32, is_los: bool, is_indoor: bool,
              fc: f32, d2d: f32) -> f32 {
    if is_los {
        switch scenario {
            case SCENARIO_UMA, SCENARIO_UMI: {
                return select(4.0, 7.0, is_indoor); // fc < 6e9
            }
            case SCENARIO_RMA: {
                if is_indoor { return 8.0; }
                let d_bp = 2.0 * PI * 35.0 * 1.5 * fc / 3.0e8;
                return select(6.0, 4.0, d2d <= d_bp);
            }
            default: { return 4.0; }
        }
    } else {
        switch scenario {
            case SCENARIO_UMA: { return select(6.0, 7.0, is_indoor); } // fc < 6e9
            case SCENARIO_UMI: { return select(7.82, 7.0, is_indoor); }
            case SCENARIO_RMA: { return 8.0; }
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
    let corr_px  = select(uni.corrDist / uni.step, 0.0, uni.corrDist == 0.0);
    let D        = 3.0 * corr_px;
    let iD       = u32(D);
    let L        = select(2u * iD + 1u, 1u, corr_px == 0.0);
    let padded_nx = uni.nX + L - 1u;
    let padded_ny = uni.nY + L - 1u;
    let total_pad = padded_nx * padded_ny;

    let total_threads = 128u * 256u;
    let pad_per_thr   = (total_pad + total_threads - 1u) / total_threads;
    let tid           = gid.x;
    let rng_id        = tid % uni.maxRngStates;
    var rng           = fill_rng[rng_id];

    for (var e = 0u; e < pad_per_thr; e++) {
        let idx = tid * pad_per_thr + e;
        if idx < total_pad { fill_temp[idx] = rand_normal(&rng); }
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
    let corr_px  = select(uni.corrDist / uni.step, 0.0, uni.corrDist == 0.0);
    let D        = 3.0 * corr_px;
    let iD       = u32(D);
    let L        = select(2u * iD + 1u, 1u, corr_px == 0.0);
    let final_nx  = uni.nX;
    let final_ny  = uni.nY;
    let padded_nx = final_nx + L - 1u;
    let padded_ny = final_ny + L - 1u;
    let total_out = final_nx * final_ny;

    let total_threads = 128u * 256u;
    let elems_per_thr = (total_out + total_threads - 1u) / total_threads;
    let tid           = gid.x;

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
        if lin >= total_out { break; }
        let ci = lin / final_ny;
        let cj = lin % final_ny;
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
        conv_output[uni.outputGridOffset + lin] = s;
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
    loop {
        if s == 0u { break; }
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
        wg_sum[1] = select(1.0, 1.0 / sqrt(variance), variance > 1e-10);
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
// Flat CRN buffers: all sites × LSPs concatenated; offset arrays give each grid's start
@group(0) @binding(14)  var<storage, read>       crn_los:          array<f32>;
@group(0) @binding(15)  var<storage, read>       crn_nlos:         array<f32>;
@group(0) @binding(16) var<storage, read>       crn_o2i:          array<f32>;
@group(0) @binding(17) var<storage, read>       crn_los_offsets:  array<u32>; // nSite*7
@group(0) @binding(18) var<storage, read>       crn_nlos_offsets: array<u32>; // nSite*6
@group(0) @binding(19) var<storage, read>       crn_o2i_offsets:  array<u32>; // nSite*6

// Bilinear interpolation into a flat CRN buffer (LOS variant)
fn lsp_at_loc_los(grid_idx: u32, x: f32, y: f32, n_x: i32, n_y: i32) -> f32 {
    let uni = link_uni;
    let base = crn_los_offsets[grid_idx];
    let nx = n_x; let ny = n_y;
    let norm_x = clamp((x - uni.minX) / (uni.maxX - uni.minX), 0.0, 1.0);
    let norm_y = clamp((y - uni.minY) / (uni.maxY - uni.minY), 0.0, 1.0);
    let gx = norm_x * f32(nx - 1);  let gy = norm_y * f32(ny - 1);
    let x0 = i32(floor(gx));        let y0 = i32(floor(gy));
    let x1 = min(x0 + 1, nx - 1);  let y1 = min(y0 + 1, ny - 1);
    let dx = gx - f32(x0);          let dy = gy - f32(y0);
    let v00 = crn_los[base + u32(y0*nx + x0)];
    let v10 = crn_los[base + u32(y0*nx + x1)];
    let v01 = crn_los[base + u32(y1*nx + x0)];
    let v11 = crn_los[base + u32(y1*nx + x1)];
    return mix(mix(v00, v10, dx), mix(v01, v11, dx), dy);
}

fn lsp_at_loc_nlos(grid_idx: u32, x: f32, y: f32, n_x: i32, n_y: i32) -> f32 {
    let uni = link_uni;
    let base = crn_nlos_offsets[grid_idx];
    let nx = n_x; let ny = n_y;
    let norm_x = clamp((x - uni.minX) / (uni.maxX - uni.minX), 0.0, 1.0);
    let norm_y = clamp((y - uni.minY) / (uni.maxY - uni.minY), 0.0, 1.0);
    let gx = norm_x * f32(nx - 1); let gy = norm_y * f32(ny - 1);
    let x0 = i32(floor(gx)); let y0 = i32(floor(gy));
    let x1 = min(x0+1, nx-1); let y1 = min(y0+1, ny-1);
    let dx = gx - f32(x0); let dy = gy - f32(y0);
    let v00 = crn_nlos[base + u32(y0*nx+x0)]; let v10 = crn_nlos[base + u32(y0*nx+x1)];
    let v01 = crn_nlos[base + u32(y1*nx+x0)]; let v11 = crn_nlos[base + u32(y1*nx+x1)];
    return mix(mix(v00, v10, dx), mix(v01, v11, dx), dy);
}

fn lsp_at_loc_o2i(grid_idx: u32, x: f32, y: f32, n_x: i32, n_y: i32) -> f32 {
    let uni = link_uni;
    let base = crn_o2i_offsets[grid_idx];
    let nx = n_x; let ny = n_y;
    let norm_x = clamp((x - uni.minX) / (uni.maxX - uni.minX), 0.0, 1.0);
    let norm_y = clamp((y - uni.minY) / (uni.maxY - uni.minY), 0.0, 1.0);
    let gx = norm_x * f32(nx - 1); let gy = norm_y * f32(ny - 1);
    let x0 = i32(floor(gx)); let y0 = i32(floor(gy));
    let x1 = min(x0+1, nx-1); let y1 = min(y0+1, ny-1);
    let dx = gx - f32(x0); let dy = gy - f32(y0);
    let v00 = crn_o2i[base + u32(y0*nx+x0)]; let v10 = crn_o2i[base + u32(y0*nx+x1)];
    let v01 = crn_o2i[base + u32(y1*nx+x0)]; let v11 = crn_o2i[base + u32(y1*nx+x1)];
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
        var pl = cal_pl(cell.loc, ut.loc, sc.scenario,
                        fc / 1e9, is_los, false, &rng);
        pl += ut.o2i_penetration_loss;
        link_params[link_idx].pathloss = pl;
    }

    // ── LSPs ─────────────────────────────────────────────────────────────
    if uni.updateAllLSPs != 0u || uni.updatePLAndPenetrationLoss != 0u {
        let cl = cmn_link[0];
        let nx = uni.nX;  let ny = uni.nY;
        let ux = ut.loc.x;  let uy = ut.loc.y;
        let true_site = site_idx / uni.nSectorPerSite;
        let s7 = true_site * 7u;
        let s6 = true_site * 6u;

        var ucv: array<f32, 7>;   // uncorrelated vars  [SF,K,DS,ASD,ASA,ZSD,ZSA]
        if is_indoor {
            ucv[SF_IDX]  = lsp_at_loc_o2i(s6+0u, ux, uy, nx, ny);
            ucv[K_IDX]   = 0.0;
            ucv[DS_IDX]  = lsp_at_loc_o2i(s6+1u, ux, uy, nx, ny);
            ucv[ASD_IDX] = lsp_at_loc_o2i(s6+2u, ux, uy, nx, ny);
            ucv[ASA_IDX] = lsp_at_loc_o2i(s6+3u, ux, uy, nx, ny);
            ucv[ZSD_IDX] = lsp_at_loc_o2i(s6+4u, ux, uy, nx, ny);
            ucv[ZSA_IDX] = lsp_at_loc_o2i(s6+5u, ux, uy, nx, ny);
        } else if is_los {
            ucv[SF_IDX]  = lsp_at_loc_los(s7+0u, ux, uy, nx, ny);
            ucv[K_IDX]   = lsp_at_loc_los(s7+1u, ux, uy, nx, ny);
            ucv[DS_IDX]  = lsp_at_loc_los(s7+2u, ux, uy, nx, ny);
            ucv[ASD_IDX] = lsp_at_loc_los(s7+3u, ux, uy, nx, ny);
            ucv[ASA_IDX] = lsp_at_loc_los(s7+4u, ux, uy, nx, ny);
            ucv[ZSD_IDX] = lsp_at_loc_los(s7+5u, ux, uy, nx, ny);
            ucv[ZSA_IDX] = lsp_at_loc_los(s7+6u, ux, uy, nx, ny);
        } else {
            ucv[SF_IDX]  = lsp_at_loc_nlos(s6+0u, ux, uy, nx, ny);
            ucv[K_IDX]   = 0.0;
            ucv[DS_IDX]  = lsp_at_loc_nlos(s6+1u, ux, uy, nx, ny);
            ucv[ASD_IDX] = lsp_at_loc_nlos(s6+2u, ux, uy, nx, ny);
            ucv[ASA_IDX] = lsp_at_loc_nlos(s6+3u, ux, uy, nx, ny);
            ucv[ZSD_IDX] = lsp_at_loc_nlos(s6+4u, ux, uy, nx, ny);
            ucv[ZSA_IDX] = lsp_at_loc_nlos(s6+5u, ux, uy, nx, ny);
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

        // Shadow fading
        link_params[link_idx].SF = cv[SF_IDX] *
            cal_sf_std(sc.scenario, is_los, is_indoor, fc, d2d);

        if uni.updateAllLSPs != 0u {
            // K-factor
            link_params[link_idx].K = select(0.0,
                cv[K_IDX] * cl.sigma_K[lsp_idx] + cl.mu_K[lsp_idx],
                lsp_idx == 0u);

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
                        mu_off    = 7.66*cl.lgfc - 5.96
                                  - pow(10.0, (0.208*cl.lgfc - 0.782)
                                       * log10(max(25.0, d2d))
                                  + (2.03 - 0.13*cl.lgfc) - 0.07*(h_ut-1.5));
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
        }
    }

    // Write RNG state back
    //rng_states_lp[global_tid] = rng;
    rng_states_lp[link_idx] = rng;
}