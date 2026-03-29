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

struct SspCellParam {
    antPanelIdx         : u32,
    antPanelOrientation : array<f32, 3>,  // [theta_tilt, phi_tilt, zeta_offset]
    _pad0               : u32,
}

struct SspUtParam {
    antPanelIdx         : u32,
    outdoor_ind         : u32,            // 1 = outdoor, 0 = indoor (O2I)
    antPanelOrientation : array<f32, 3>,
    velocity            : array<f32, 3>,
    _pad0               : u32,
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
    _pad0        : u32,
    _pad1        : u32,
    _pad2        : u32,
}

// ── group 1: cal_cluster_ray_kernel ──────────────────────────────────────
@group(1) @binding(0) var<storage, read>       cray_buf_link    : array<LinkParams>;
@group(1) @binding(1) var<storage, read>       cray_buf_ut      : array<SspUtParam>;
@group(1) @binding(2) var<storage, read>       cray_buf_cmn     : SsCmnParams;
@group(1) @binding(3) var<storage, read_write> cray_buf_cluster : array<ClusterParams>;
@group(1) @binding(4) var<storage, read_write> cray_buf_rng     : array<RngState>;
@group(1) @binding(5) var<uniform>             cray_disp        : DispatchUniforms;
@group(1) @binding(6) var<storage, read_write> cray_buf_xpr:          array<f32>;
@group(1) @binding(7) var<storage, read_write> cray_buf_randomPhases: array<f32>;
@group(1) @binding(8) var<storage, read_write> cray_buf_phi_nm_AoA:     array<f32>;
@group(1) @binding(9) var<storage, read_write> cray_buf_phi_nm_AoD:     array<f32>;
@group(1) @binding(10) var<storage, read_write> cray_buf_theta_nm_ZOA:  array<f32>;
@group(1) @binding(11) var<storage, read_write> cray_buf_theta_nm_ZOD:  array<f32>;

// ── group 2: generate_cir_kernel ─────────────────────────────────────────
@group(2) @binding(0)  var<uniform>             cir_uni_sim          : SmallScaleSimConfig;
@group(2) @binding(1)  var<uniform>             cir_uni_sys          : SmallScaleSysConfig;
@group(2) @binding(2)  var<storage, read>       cir_buf_cmn          : SsCmnParams;
@group(2) @binding(3)  var<storage, read>       cir_buf_cell         : array<SspCellParam>;
@group(2) @binding(4)  var<storage, read>       cir_buf_ut           : array<SspUtParam>;
@group(2) @binding(5)  var<storage, read>       cir_buf_link         : array<LinkParams>;
@group(2) @binding(6)  var<storage, read_write> cir_buf_cluster      : array<ClusterParams>;
@group(2) @binding(7)  var<storage, read>       cir_buf_antCfg       : array<AntPanelConfig>;
@group(2) @binding(8)  var<storage, read>       cir_buf_antTheta     : array<f32>;
@group(2) @binding(9)  var<storage, read>       cir_buf_antPhi       : array<f32>;
@group(2) @binding(10) var<storage, read>       cir_buf_activeLink   : array<ActiveLink>;
@group(2) @binding(11) var<storage, read_write> cir_buf_cirCoe       : array<vec2f>;
@group(2) @binding(12) var<storage, read_write> cir_buf_cirNormDelay : array<u32>;
@group(2) @binding(13) var<storage, read_write> cir_buf_cirNtaps     : array<u32>;
@group(2) @binding(14) var<uniform>             cir_disp             : DispatchUniforms;
@group(2) @binding(15) var<storage, read> cir_buf_xpr          : array<f32>;
@group(2) @binding(16) var<storage, read> cir_buf_randomPhases  : array<f32>;
@group(2) @binding(17) var<storage, read> cir_buf_phi_nm_AoA    : array<f32>;
@group(2) @binding(18) var<storage, read> cir_buf_phi_nm_AoD    : array<f32>;
@group(2) @binding(19) var<storage, read> cir_buf_theta_nm_ZOA  : array<f32>;
@group(2) @binding(20) var<storage, read> cir_buf_theta_nm_ZOD  : array<f32>;

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
    let M_rx = cir_buf_cluster[cluster_link_idx].nCluster; // dummy – using antSize remove
    // Use antSize from cfg
    let N_rx = bsCfg.antSize[3]; // reuse variable names – actually computing for Rx below
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
    // randomPhases are in degrees – convert to radians inline
    let cp = cir_buf_cluster[cluster_link_idx]; // re-read needed for random phases remove
    // Note: rph_off is the flat index into cp.randomPhases
    let ph0 = cir_buf_randomPhases[rph_off    ] * DEG2RAD;
    let ph1 = cir_buf_randomPhases[rph_off + 1u] * DEG2RAD;
    let ph2 = cir_buf_randomPhases[rph_off + 2u] * DEG2RAD;
    let ph3 = cir_buf_randomPhases[rph_off + 3u] * DEG2RAD;

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
    let lsp_idx  = select(select(0u, 1u, lk.losInd != 0u), 2u, is_o2i != 0u);

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

    // Convert delays to nanoseconds
    for (var n = 0u; n < n_cluster; n++) { delays[n] *= 1e9; }

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

    // Apply K-factor (LOS)
    if lk.losInd != 0u {
        let K_lin = pow(10.0, lk.K / 10.0);
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

    // Select C_phi, C_theta based on scenario
    let C_phi   = select(select(cray_buf_cmn.C_phi_NLOS,   cray_buf_cmn.C_phi_LOS,   lk.losInd != 0u), cray_buf_cmn.C_phi_O2I,   is_o2i != 0u);
    let C_theta = select(select(cray_buf_cmn.C_theta_NLOS, cray_buf_cmn.C_theta_LOS, lk.losInd != 0u), cray_buf_cmn.C_theta_O2I, is_o2i != 0u);

    // Scale C_phi for LOS K-factor (3GPP 38.901 Eq 7.5-10)
    var C_phi_scaled   = C_phi;
    var C_theta_scaled = C_theta;
    if lk.losInd != 0u && is_o2i == 0u {
        let K_lin = pow(10.0, lk.K / 10.0);
        let Kfac  = 1.1035 - 0.028 * lk.K - 0.002 * lk.K * lk.K + 0.0001 * lk.K * lk.K * lk.K;
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
        let arg_zsd    = -log(powers[n] / max(max_abs_power, 1e-30)) / (2.0 * C_ZSA / 1.4);
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

            let _b = link_idx * MAX_CR + idx;
            cray_buf_phi_nm_AoA  [_b] = wrap_azimuth(phi_n_AoA[c]   + C_ASA * alpha_m[r] * s_asa);
            cray_buf_phi_nm_AoD  [_b] = wrap_azimuth(phi_n_AoD[c]   + C_ASD * alpha_m[r] * s_asd);
            cray_buf_theta_nm_ZOA[_b] = wrap_zenith (theta_n_ZOA[c] + C_ZSA * alpha_m[r] * s_zsa);
            // ZOD: 3/8 * 10^mu_lgZSD offset (3GPP Eq 7.5-20)
            let zsd_offset = (3.0 / 8.0) * pow(10.0, lk.mu_lgZSD) * alpha_m[r] * s_zsd;
            cray_buf_theta_nm_ZOD[_b] = wrap_zenith(theta_n_ZOD[c] + zsd_offset);
        }
    }

    // --------------- 4. XPR and random phases ---------------
    // xpr / randomPhases written directly to flat storage — no private staging
    for (var c = 0u; c < n_cluster; c++) {
        for (var r = 0u; r < n_ray; r++) {
            let idx = c * n_ray + r;
            cray_buf_xpr[link_idx * MAX_CR + idx] =
                pow(10.0, (cray_buf_cmn.mu_XPR[lsp_idx]
                           + cray_buf_cmn.sigma_XPR[lsp_idx] * rng_normal(&rng)) / 10.0);
            for (var p = 0u; p < 4u; p++) {
                cray_buf_randomPhases[link_idx * MAX_CR4 + idx * 4u + p] =
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
    let lsp_idx  = select(select(0u, 1u, lk.losInd != 0u), 2u, is_o2i != 0u);

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

    // CIR snapshot offset in output buffer
    let snap_off = al.cirCoeOffset
                 + snapshot_idx * nUtAnt * nCellAnt * NMAXTAPS;

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
                    let dst = snap_off + (ua * nCellAnt + ba) * NMAXTAPS;
                    cir_buf_cirCoe[dst] = coeff * ps;
                }
            }
            if snapshot_idx == 0u {
                cir_buf_cirNormDelay[al.cirNormDelayOffset] = 0u;
                cir_buf_cirNtaps[al.cirNtapsOffset]         = 1u;
            }
        }
        return;
    }

    let cp     = cir_buf_cluster[al.lspReadIdx];
    let n_cl   = cp.nCluster;
    let n_ray  = cp.nRayPerCluster;

    // K-factor
    var KR       = 0.0;
    var los_power = 0.0;
    if lk.losInd != 0u && is_o2i == 0u {
        KR        = pow(10.0, lk.K / 10.0);
        los_power = KR / (KR + 1.0);
    }

    let zetaBs = cell.antPanelOrientation[2];
    let zetaUt = ut.antPanelOrientation[2];

    // ---- Outer BS antenna loop (serial) ----
    for (var ba = 0u; ba < nCellAnt; ba++) {

        // Reset workgroup H_link and tap count
        if li.x == 0u { atomicStore(&wg_tapCnt, 0u); }
        for (var i = li.x; i < NMAXTAPS * MAX_UE_ANT; i += 24u) { wg_Hlink[i] = vec2f(0.0); }
        workgroupBarrier();

        var tap_count = 0u;

        // ---- cluster loop ----
        for (var c = 0u; c < n_cl; c++) {
            let is_strongest = (c == cp.strongest2[0] || c == cp.strongest2[1]);
            var cl_power = cp.powers[c];
            if lk.losInd != 0u && is_o2i == 0u && c == 0u {
                cl_power = max(cl_power - los_power, 0.0);
            }

            if is_strongest {
                // Sub-cluster processing (3 sub-clusters per 3GPP Table 7.5-5)
                for (var sc = 0u; sc < cir_buf_cmn.nSubCluster; sc++) {
                    var sc_power = 0.0;
                    if      sc == 0u { sc_power = sqrt(10.0 / 20.0); }
                    else if sc == 1u { sc_power = sqrt( 6.0 / 20.0); }
                    else             { sc_power = sqrt( 4.0 / 20.0); }

                    var cl_delay = cp.delays[c];
                    let C_DS     = cir_buf_cmn.C_DS[lsp_idx];
                    if sc == 1u { cl_delay += 1.28 * C_DS; }
                    if sc == 2u { cl_delay += 2.56 * C_DS; }

                    // tap index
                    var tap_idx = 0u;
                    if cir_uni_sys.enable_propagation_delay == 1u {
                        tap_idx = u32(round((cl_delay * 1e-9 + lk.d3d / 3.0e8)
                                           * cir_uni_sim.sc_spacing_hz * f32(cir_uni_sim.fft_size)));
                    } else {
                        tap_idx = u32(round(cl_delay * 1e-9
                                           * cir_uni_sim.sc_spacing_hz * f32(cir_uni_sim.fft_size)));
                    }
                    if li.x == 0u { wg_tapIdx[tap_count] = tap_idx; }

                    // ray loop – only thread 0 executes (matches CUDA tid==0 path for tapIdx)
                    let sc_size = cir_buf_cmn.raysInSubClusterSizes[sc];
                    let power   = sqrt(cl_power / f32(n_ray));

                    if li.x == 0u {
                        for (var ri = 0u; ri < sc_size; ri++) {
                            var ray_global_idx = 0u;
                            if sc == 0u { ray_global_idx = c * n_ray + cir_buf_cmn.raysInSubCluster0[ri]; }
                            else if sc == 1u { ray_global_idx = c * n_ray + cir_buf_cmn.raysInSubCluster1[ri]; }
                            else             { ray_global_idx = c * n_ray + cir_buf_cmn.raysInSubCluster2[ri]; }

                            let tZOA = cir_buf_theta_nm_ZOA[ray_global_idx] - ut.antPanelOrientation[0];
                            let pAOA = cir_buf_phi_nm_AoA[ray_global_idx]   - ut.antPanelOrientation[1];
                            let tZOD = cir_buf_theta_nm_ZOD[ray_global_idx] - cell.antPanelOrientation[0];
                            let pAOD = cir_buf_phi_nm_AoD[ray_global_idx]   - cell.antPanelOrientation[1];
                            let xpr  = cir_buf_xpr[ray_global_idx];
                            let rph_off = ray_global_idx * 4u;

                            for (var ua = 0u; ua < nUtAnt; ua++) {
                                let rc = calc_ray_coeff(
                                    utCfg, ua, tZOA, pAOA, zetaUt,
                                    bsCfg, ba, tZOD, pAOD, zetaBs,
                                    xpr, rph_off, al.lspReadIdx,
                                    snap_time, vel, cir_buf_cmn.lambda_0
                                );
                                let hi = ua * NMAXTAPS + tap_count;
                                wg_Hlink[hi] = cadd(wg_Hlink[hi], rc * power * sc_power);
                            }
                        }
                    }
                    tap_count++;
                }
            } else {
                // Regular cluster
                var cl_delay = cp.delays[c];
                var tap_idx = 0u;
                if cir_uni_sys.enable_propagation_delay == 1u {
                    tap_idx = u32(round((cl_delay * 1e-9 + lk.d3d / 3.0e8)
                                       * cir_uni_sim.sc_spacing_hz * f32(cir_uni_sim.fft_size)));
                } else {
                    tap_idx = u32(round(cl_delay * 1e-9
                                       * cir_uni_sim.sc_spacing_hz * f32(cir_uni_sim.fft_size)));
                }
                if li.x == 0u { wg_tapIdx[tap_count] = tap_idx; }

                let power = sqrt(cl_power / f32(n_ray));
                if li.x == 0u {
                    for (var r = 0u; r < n_ray; r++) {
                        let ray_global_idx = c * n_ray + r;
                        let tZOA = cir_buf_theta_nm_ZOA[ray_global_idx] - ut.antPanelOrientation[0];
                        let pAOA = cir_buf_phi_nm_AoA[ray_global_idx]   - ut.antPanelOrientation[1];
                        let tZOD = cir_buf_theta_nm_ZOD[ray_global_idx] - cell.antPanelOrientation[0];
                        let pAOD = cir_buf_phi_nm_AoD[ray_global_idx]   - cell.antPanelOrientation[1];
                        let xpr  = cir_buf_xpr[ray_global_idx];
                        let rph_off = ray_global_idx * 4u;

                        for (var ua = 0u; ua < nUtAnt; ua++) {
                            let rc = calc_ray_coeff(
                                utCfg, ua, tZOA, pAOA, zetaUt,
                                bsCfg, ba, tZOD, pAOD, zetaBs,
                                xpr, rph_off, al.lspReadIdx,
                                snap_time, vel, cir_buf_cmn.lambda_0
                            );
                            let hi = ua * NMAXTAPS + tap_count;
                            wg_Hlink[hi] = cadd(wg_Hlink[hi], rc * power);
                        }
                    }
                }
                tap_count++;
            }
        } // end cluster loop

        workgroupBarrier();

        // LOS component (added at tap 0)
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

        // Path loss scaling
        if cir_uni_sys.disable_pl_shadowing == 0u && li.x == 0u {
            let path_gain = -(lk.pathloss - lk.SF);
            let ps        = pow(10.0, path_gain / 20.0);
            for (var ua = 0u; ua < nUtAnt; ua++) {
                for (var t2 = 0u; t2 < NMAXTAPS; t2++) {
                    wg_Hlink[ua * NMAXTAPS + t2] = wg_Hlink[ua * NMAXTAPS + t2] * ps;
                }
            }
        }

        workgroupBarrier();

        let buf_len = arrayLength(&cir_buf_cirCoe);

        // Write H_link -> cirCoe for this BS antenna
        if li.x == 0u {
            for (var ti = 0u; ti < tap_count; ti++) {
                for (var ua = 0u; ua < nUtAnt; ua++) {
                    let dst = snap_off + (ua * nCellAnt + ba) * NMAXTAPS + ti;
                    if dst < buf_len {
                    cir_buf_cirCoe[dst] = cadd(cir_buf_cirCoe[dst], wg_Hlink[ua * NMAXTAPS + ti]);
                    }
                }
            }
        }
    } // end BS antenna loop

    workgroupBarrier();

    // Sparse tap indexing (only thread 0, only snapshot 0)
    if li.x == 0u && snapshot_idx == 0u {
        let tap_count = atomicLoad(&wg_tapCnt); // reuse stored value via last wg_tapIdx valid count
        // Determine actual tap_count from cluster processing above.
        // Because we wrote sequentially into wg_tapIdx, we count unique sorted entries.

        // Use the last tap_count written above – track it in wg_tapCnt
        // (wg_tapCnt is set atomically at the end of cluster loop, see below – but
        //  we keep it simple: sort wg_tapIdx[0..tap_count-1] and deduplicate)

        // We re-derive tap_count: count non-UINT_MAX entries
        var tc = 0u;
        for (var i = 0u; i < NMAXTAPS; i++) {
            if wg_tapIdx[i] != 0xFFFFFFFFu { tc++; } else { break; }
        }

        // Bubble sort wg_tapIdx[0..tc)
        for (var i = 0u; i < tc - 1u; i++) {
            for (var j = 0u; j < tc - 1u - i; j++) {
                if wg_tapIdx[j] > wg_tapIdx[j+1u] {
                    let tmp = wg_tapIdx[j]; wg_tapIdx[j] = wg_tapIdx[j+1u]; wg_tapIdx[j+1u] = tmp;
                }
            }
        }

        // Deduplicate
        var n_unique = 0u;
        if tc > 0u {
            cir_buf_cirNormDelay[al.cirNormDelayOffset] = wg_tapIdx[0];
            n_unique = 1u;
            for (var i = 1u; i < tc; i++) {
                if wg_tapIdx[i] != wg_tapIdx[n_unique - 1u] {
                    cir_buf_cirNormDelay[al.cirNormDelayOffset + n_unique] = wg_tapIdx[i];
                    n_unique++;
                }
            }
        }
        cir_buf_cirNtaps[al.cirNtapsOffset] = n_unique;
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
