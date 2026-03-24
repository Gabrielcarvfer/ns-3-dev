#!/usr/bin/env python3
"""
sls-chan-validate.py

Validates:
- Large-scale UMa 3.5 GHz outputs from link_params.csv
- Small-scale power outputs from small_scale_params.csv
- Optional detailed per-link small-scale outputs from small_scale_detail.csv
- Optional per-ray outputs from ray_params.csv

The script auto-skips checks whose input CSV/columns are not present.
"""

import argparse
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

FC_GHZ = 3.5
H_BS = 25.0
H_UT = 1.5
H_E = 1.0

MAX_CLUSTERS = 20
MAX_RAYS = 20
MAX_TAPS = 24

REF = {
    "ds_ns": {
        "LOS": {"mu": 1.97, "sigma": 0.66},
        "NLOS": {"mu": 2.56, "sigma": 0.39},
    },
    "asd_deg": {
        "LOS": {"mu": 1.15, "sigma": 0.28},
        "NLOS": {"mu": 1.41, "sigma": 0.28},
    },
    "asa_deg": {
        "LOS": {"mu": 1.81, "sigma": 0.20},
        "NLOS": {"mu": 1.87, "sigma": 0.20},
    },
    "zsa_deg": {
        "LOS": {"mu": 0.95, "sigma": 0.16},
        "NLOS": {"mu": 1.26, "sigma": 0.35},
    },
    "sf_sigma": {"LOS": 4.0, "NLOS": 6.0},
}


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def d_bp():
    return 4.0 * (H_BS - H_E) * (H_UT - H_E) * FC_GHZ * 1e9 / 3e8


def uma_los_pl(d2d, d3d):
    dbp = d_bp()
    pl1 = 28.0 + 22.0 * np.log10(d3d) + 20.0 * np.log10(FC_GHZ)
    pl2 = (
            28.0
            + 40.0 * np.log10(d3d)
            + 20.0 * np.log10(FC_GHZ)
            - 9.0 * np.log10(dbp**2 + (H_BS - H_UT) ** 2)
    )
    return np.where(d2d <= dbp, pl1, pl2)


def uma_nlos_pl(d3d):
    return 32.4 + 20.0 * np.log10(FC_GHZ) + 30.0 * np.log10(d3d)


def p_los_uma(d):
    p = (18.0 / d) + np.exp(-d / 63.0) * (1.0 - 18.0 / d)
    return np.where(d <= 18.0, 1.0, p)


def pick_col(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None


def add_distance_bins(df, col="d2d_m"):
    if col not in df.columns:
        return df
    out = df.copy()
    positive = out[col] > 0
    out = out[positive].copy()
    out["d_bin"] = pd.cut(out[col], bins=np.logspace(1, 3.3, 20), include_lowest=True)
    return out


def load_csv_optional(path):
    if not path or not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_link_csv(path):
    df = pd.read_csv(path)
    df = df[df["d3d_m"] > 0].copy()
    return add_distance_bins(df, "d2d_m")


def load_ss_csv(path):
    df = load_csv_optional(path)
    if df is None:
        return None
    if "d2d_m" in df.columns:
        df = add_distance_bins(df, "d2d_m")
    return df


def state_label_from_is_los(v):
    return "LOS" if int(v) == 1 else "NLOS"


def summarize_lsp(df):
    rows = []
    for los_val, los_label in [(1, "LOS"), (0, "NLOS")]:
        sub = df[df["is_los"] == los_val]
        for field, lbl in [
            ("ds_ns", "DS"),
            ("asd_deg", "ASD"),
            ("asa_deg", "ASA"),
            ("zsa_deg", "ZSA"),
        ]:
            vals = sub[sub[field] > 0][field]
            if vals.empty:
                continue
            lv = np.log10(vals)
            ref = REF[field].get(los_label, {})
            rows.append(
                {
                    "State": los_label,
                    "Metric": lbl,
                    "N": len(vals),
                    "Sim mu(log10)": round(float(lv.mean()), 3),
                    "Ref mu(log10)": round(float(ref.get("mu", np.nan)), 3),
                    "Sim sigma(log10)": round(float(lv.std(ddof=1)), 3),
                    "Ref sigma(log10)": round(float(ref.get("sigma", np.nan)), 3),
                }
            )
    return pd.DataFrame(rows)


def summarize_small_scale_power(df):
    rows = []
    if df is None:
        return pd.DataFrame(rows)

    cir_col = pick_col(df, "cir_power_db")
    cfr_col = pick_col(df, "cfr_power_db")
    if cir_col is None and cfr_col is None:
        return pd.DataFrame(rows)

    for los_val, label in [(1, "LOS"), (0, "NLOS")]:
        sub = df[df["is_los"] == los_val] if "is_los" in df.columns else df
        row = {"State": label, "N": len(sub)}
        if cir_col:
            s = sub[cir_col].replace([-np.inf, np.inf], np.nan).dropna()
            row["CIR mean dB"] = round(float(s.mean()), 3) if len(s) else np.nan
            row["CIR std dB"] = round(float(s.std(ddof=1)), 3) if len(s) > 1 else np.nan
        if cfr_col:
            s = sub[cfr_col].replace([-np.inf, np.inf], np.nan).dropna()
            row["CFR mean dB"] = round(float(s.mean()), 3) if len(s) else np.nan
            row["CFR std dB"] = round(float(s.std(ddof=1)), 3) if len(s) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_pl_vs_distance(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    d2d = np.logspace(1, 3.3, 400)
    d3d = np.sqrt(d2d**2 + (H_BS - H_UT) ** 2)

    panels = [
        (axes[0], 1, "LOS", "steelblue", lambda x2d, x3d: uma_los_pl(x2d, x3d)),
        (axes[1], 0, "NLOS", "tomato", lambda x2d, x3d: uma_nlos_pl(x3d)),
    ]

    for ax, los_val, label, color, ref_fn in panels:
        sub = df[df["is_los"] == los_val]
        ax.scatter(
            sub["d2d_m"],
            sub["pl_sim_db"],
            s=4,
            alpha=0.25,
            color=color,
            label="Simulated",
        )
        ax.plot(d2d, ref_fn(d2d, d3d), "k-", lw=2.5, label="3GPP UMa ref")
        ax.set_xscale("log")
        ax.set_xlabel("2D distance (m)")
        ax.set_ylabel("Path loss (dB)")
        ax.set_title(f"UMa {label} path loss @ {FC_GHZ} GHz")
        ax.grid(True, which="both", ls="--", alpha=0.35)
        ax.legend(markerscale=4)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pl_vs_distance.png"), dpi=150)
    plt.close(fig)


def plot_sf_distribution(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, los_val, label in [(axes[0], 1, "LOS"), (axes[1], 0, "NLOS")]:
        sub = df[df["is_los"] == los_val]
        sf = sub["sf_db"].replace([-np.inf, np.inf], np.nan).dropna()
        if sf.empty:
            continue
        sig = REF["sf_sigma"][label]
        bins = np.linspace(-30, 30, 70)
        ax.hist(
            sf,
            bins=bins,
            density=True,
            alpha=0.65,
            color="steelblue",
            label=f"Sim μ={sf.mean():.2f} σ={sf.std(ddof=1):.2f} dB",
        )
        x = np.linspace(-30, 30, 400)
        ax.plot(x, norm.pdf(x, 0, sig), "r-", lw=2.5, label=f"3GPP σ={sig:.0f} dB")
        ax.set_xlabel("Shadow fading (dB)")
        ax.set_title(f"Shadow fading — {label}")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "sf_distribution.png"), dpi=150)
    plt.close(fig)


def plot_los_probability(df, outdir):
    grp = df.groupby("d_bin", observed=True)
    d_mid = grp["d2d_m"].median()
    p_sim = grp["is_los"].mean()
    d_ref = np.logspace(1, 3.3, 400)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(d_mid.values, p_sim.values, "o-", color="steelblue", ms=5, label="Simulated")
    ax.plot(d_ref, p_los_uma(d_ref), "r--", lw=2.5, label="3GPP UMa")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("2D distance (m)")
    ax.set_ylabel("LOS probability")
    ax.set_title("LOS probability vs distance — UMa")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "los_probability.png"), dpi=150)
    plt.close(fig)


def plot_lsp_cdfs(df, outdir):
    lsps = [
        ("ds_ns", "DS (ns)"),
        ("asd_deg", "ASD (deg)"),
        ("asa_deg", "ASA (deg)"),
        ("zsa_deg", "ZSA (deg)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for col, (field, label) in enumerate(lsps):
        for row, (los_val, los_label) in enumerate([(1, "LOS"), (0, "NLOS")]):
            ax = axes[row][col]
            sub = df[(df["is_los"] == los_val) & (df[field] > 0)][field]
            ref = REF[field].get(los_label)
            if sub.empty or ref is None:
                ax.set_visible(False)
                continue
            lv = np.log10(sub.values)
            ax.plot(np.sort(lv), np.linspace(0, 1, len(lv)), lw=1.5, color="steelblue", label="Sim")
            x_ref = np.linspace(ref["mu"] - 4 * ref["sigma"], ref["mu"] + 4 * ref["sigma"], 300)
            ax.plot(
                x_ref,
                norm.cdf(x_ref, ref["mu"], ref["sigma"]),
                "r--",
                lw=2,
                label=f"3GPP μ={ref['mu']:.2f}, σ={ref['sigma']:.2f}",
            )
            ax.set_xlabel(f"log10({label})")
            ax.set_ylabel("CDF")
            ax.set_title(f"{los_label} — {label}")
            ax.grid(True, ls="--", alpha=0.35)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "lsp_cdfs.png"), dpi=150)
    plt.close(fig)


def plot_small_scale_power(ss_df, outdir):
    if ss_df is None:
        return

    cir_col = pick_col(ss_df, "cir_power_db")
    cfr_col = pick_col(ss_df, "cfr_power_db")
    if cir_col is None and cfr_col is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    if cir_col:
        for los_val, label, color in [(1, "LOS", "steelblue"), (0, "NLOS", "tomato")]:
            sub = ss_df[ss_df["is_los"] == los_val] if "is_los" in ss_df.columns else ss_df
            vals = sub[cir_col].replace([-np.inf, np.inf], np.nan).dropna()
            if len(vals):
                axes[0].hist(vals, bins=50, alpha=0.55, density=True, label=label, color=color)
        axes[0].set_title("CIR power")
        axes[0].set_xlabel("dB")
        axes[0].grid(True, ls="--", alpha=0.35)
        axes[0].legend()

    if cfr_col:
        for los_val, label, color in [(1, "LOS", "steelblue"), (0, "NLOS", "tomato")]:
            sub = ss_df[ss_df["is_los"] == los_val] if "is_los" in ss_df.columns else ss_df
            vals = sub[cfr_col].replace([-np.inf, np.inf], np.nan).dropna()
            if len(vals):
                axes[1].hist(vals, bins=50, alpha=0.55, density=True, label=label, color=color)
        axes[1].set_title("CFR power")
        axes[1].set_xlabel("dB")
        axes[1].grid(True, ls="--", alpha=0.35)
        axes[1].legend()

    if cir_col and cfr_col:
        vals = ss_df[[cir_col, cfr_col]].replace([-np.inf, np.inf], np.nan).dropna()
        axes[2].scatter(vals[cir_col], vals[cfr_col], s=6, alpha=0.25, color="purple")
        axes[2].set_xlabel("CIR power (dB)")
        axes[2].set_ylabel("CFR power (dB)")
        axes[2].set_title("CIR vs CFR")
        axes[2].grid(True, ls="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "small_scale_power.png"), dpi=150)
    plt.close(fig)


def validate_detail_bounds(detail_df):
    rows = []
    if detail_df is None:
        return pd.DataFrame(rows)

    ncl_col = pick_col(detail_df, "n_cluster", "nCluster")
    nray_col = pick_col(detail_df, "n_ray_per_cluster", "nRayPerCluster")
    ntaps_col = pick_col(detail_df, "n_taps", "nTaps", "cir_ntaps")

    if ncl_col:
        s = detail_df[ncl_col].dropna()
        rows.append(
            {
                "Check": "n_cluster <= MAX_CLUSTERS",
                "Pass ratio": round(float((s <= MAX_CLUSTERS).mean()), 4),
                "Min": int(s.min()) if len(s) else np.nan,
                "Max": int(s.max()) if len(s) else np.nan,
            }
        )
    if nray_col:
        s = detail_df[nray_col].dropna()
        rows.append(
            {
                "Check": "n_ray_per_cluster <= MAX_RAYS",
                "Pass ratio": round(float((s <= MAX_RAYS).mean()), 4),
                "Min": int(s.min()) if len(s) else np.nan,
                "Max": int(s.max()) if len(s) else np.nan,
            }
        )
    if ntaps_col:
        s = detail_df[ntaps_col].dropna()
        rows.append(
            {
                "Check": "n_taps <= MAX_TAPS",
                "Pass ratio": round(float((s <= MAX_TAPS).mean()), 4),
                "Min": int(s.min()) if len(s) else np.nan,
                "Max": int(s.max()) if len(s) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def validate_detail_targets(detail_df, args):
    rows = []
    if detail_df is None:
        return pd.DataFrame(rows)

    ncl_col = pick_col(detail_df, "n_cluster", "nCluster")
    nray_col = pick_col(detail_df, "n_ray_per_cluster", "nRayPerCluster")
    ntaps_col = pick_col(detail_df, "n_taps", "nTaps", "cir_ntaps")

    if ncl_col and args.target_clusters_los is not None and "is_los" in detail_df.columns:
        for los_val, label, tgt in [(1, "LOS", args.target_clusters_los), (0, "NLOS", args.target_clusters_nlos)]:
            if tgt is None:
                continue
            sub = detail_df[detail_df["is_los"] == los_val]
            if len(sub):
                rows.append(
                    {
                        "Check": f"{label} n_cluster == target",
                        "Target": tgt,
                        "Pass ratio": round(float((sub[ncl_col] == tgt).mean()), 4),
                        "Mean": round(float(sub[ncl_col].mean()), 3),
                    }
                )

    if nray_col and args.target_rays_per_cluster is not None:
        rows.append(
            {
                "Check": "n_ray_per_cluster == target",
                "Target": args.target_rays_per_cluster,
                "Pass ratio": round(float((detail_df[nray_col] == args.target_rays_per_cluster).mean()), 4),
                "Mean": round(float(detail_df[nray_col].mean()), 3),
            }
        )

    if ntaps_col and args.target_ntaps is not None:
        rows.append(
            {
                "Check": "n_taps == target",
                "Target": args.target_ntaps,
                "Pass ratio": round(float((detail_df[ntaps_col] == args.target_ntaps).mean()), 4),
                "Mean": round(float(detail_df[ntaps_col].mean()), 3),
            }
        )

    return pd.DataFrame(rows)


def summarize_detail_lsp(detail_df):
    rows = []
    if detail_df is None:
        return pd.DataFrame(rows)

    for field, label in [
        ("ds_ns", "DS"),
        ("asd_deg", "ASD"),
        ("asa_deg", "ASA"),
        ("zsa_deg", "ZSA"),
    ]:
        if field not in detail_df.columns or "is_los" not in detail_df.columns:
            continue
        for los_val, los_label in [(1, "LOS"), (0, "NLOS")]:
            vals = detail_df[(detail_df["is_los"] == los_val) & (detail_df[field] > 0)][field]
            if vals.empty:
                continue
            lv = np.log10(vals)
            ref = REF[field].get(los_label, {})
            rows.append(
                {
                    "State": los_label,
                    "Metric": label,
                    "N": len(vals),
                    "Sim mu(log10)": round(float(lv.mean()), 3),
                    "Ref mu(log10)": round(float(ref.get("mu", np.nan)), 3),
                    "Sim sigma(log10)": round(float(lv.std(ddof=1)), 3),
                    "Ref sigma(log10)": round(float(ref.get("sigma", np.nan)), 3),
                }
            )
    return pd.DataFrame(rows)


def plot_detail_histograms(detail_df, outdir):
    if detail_df is None:
        return

    ncl_col = pick_col(detail_df, "n_cluster", "nCluster")
    nray_col = pick_col(detail_df, "n_ray_per_cluster", "nRayPerCluster")
    ntaps_col = pick_col(detail_df, "n_taps", "nTaps", "cir_ntaps")

    cols = [c for c in [ncl_col, nray_col, ntaps_col] if c]
    if not cols:
        return

    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]

    for ax, c in zip(axes, cols):
        vals = detail_df[c].dropna()
        ax.hist(vals, bins=min(40, max(10, int(vals.nunique()) + 1)), alpha=0.75, color="steelblue")
        ax.set_title(c)
        ax.grid(True, ls="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "small_scale_detail_hist.png"), dpi=150)
    plt.close(fig)


def summarize_ray_csv(ray_df, args):
    rows = []
    if ray_df is None:
        return pd.DataFrame(rows)

    cl_col = pick_col(ray_df, "cluster_idx", "cluster", "cluster_id")
    ray_col = pick_col(ray_df, "ray_idx", "ray", "ray_id")
    xpr_col = pick_col(ray_df, "xpr_db", "xpr")
    delay_col = pick_col(ray_df, "delay_ns", "ray_delay_ns")

    if cl_col:
        s = ray_df[cl_col].dropna()
        rows.append(
            {
                "Check": "cluster_idx < MAX_CLUSTERS",
                "Pass ratio": round(float(((s >= 0) & (s < MAX_CLUSTERS)).mean()), 4),
                "Min": int(s.min()) if len(s) else np.nan,
                "Max": int(s.max()) if len(s) else np.nan,
            }
        )

    if ray_col:
        s = ray_df[ray_col].dropna()
        rows.append(
            {
                "Check": "ray_idx < MAX_RAYS",
                "Pass ratio": round(float(((s >= 0) & (s < MAX_RAYS)).mean()), 4),
                "Min": int(s.min()) if len(s) else np.nan,
                "Max": int(s.max()) if len(s) else np.nan,
            }
        )

    key_cols = [c for c in [pick_col(ray_df, "site"), pick_col(ray_df, "ut"), cl_col] if c]
    if len(key_cols) == 3 and ray_col and args.target_rays_per_cluster is not None:
        counts = ray_df.groupby(key_cols, observed=True)[ray_col].nunique()
        rows.append(
            {
                "Check": "unique rays per cluster == target",
                "Target": args.target_rays_per_cluster,
                "Pass ratio": round(float((counts == args.target_rays_per_cluster).mean()), 4),
                "Mean": round(float(counts.mean()), 3),
                "Min": int(counts.min()) if len(counts) else np.nan,
                "Max": int(counts.max()) if len(counts) else np.nan,
            }
        )

    if xpr_col:
        s = ray_df[xpr_col].replace([-np.inf, np.inf], np.nan).dropna()
        if len(s):
            rows.append(
                {
                    "Check": "xpr summary",
                    "Mean dB": round(float(s.mean()), 3),
                    "Std dB": round(float(s.std(ddof=1)), 3) if len(s) > 1 else np.nan,
                    "Min dB": round(float(s.min()), 3),
                    "Max dB": round(float(s.max()), 3),
                }
            )

    if delay_col:
        s = ray_df[delay_col].replace([-np.inf, np.inf], np.nan).dropna()
        if len(s):
            rows.append(
                {
                    "Check": "ray delay summary",
                    "Mean ns": round(float(s.mean()), 3),
                    "Std ns": round(float(s.std(ddof=1)), 3) if len(s) > 1 else np.nan,
                    "Min ns": round(float(s.min()), 3),
                    "Max ns": round(float(s.max()), 3),
                }
            )

    return pd.DataFrame(rows)


def plot_ray_csv(ray_df, outdir):
    if ray_df is None:
        return

    cl_col = pick_col(ray_df, "cluster_idx", "cluster", "cluster_id")
    ray_col = pick_col(ray_df, "ray_idx", "ray", "ray_id")
    xpr_col = pick_col(ray_df, "xpr_db", "xpr")
    delay_col = pick_col(ray_df, "delay_ns", "ray_delay_ns")

    cols = [c for c in [cl_col, ray_col, xpr_col, delay_col] if c]
    if not cols:
        return

    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]

    for ax, c in zip(axes, cols):
        vals = ray_df[c].replace([-np.inf, np.inf], np.nan).dropna()
        ax.hist(vals, bins=50, alpha=0.75, color="tomato")
        ax.set_title(c)
        ax.grid(True, ls="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "ray_params_hist.png"), dpi=150)
    plt.close(fig)


def print_df(title, df):
    print(f"\n── {title} " + "─" * max(8, 72 - len(title)))
    if df is None or len(df) == 0:
        print("(no data)")
    else:
        print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link-csv", default="link_params.csv")
    ap.add_argument("--ss-csv", default="small_scale_params.csv")
    ap.add_argument("--detail-csv", default="small_scale_detail.csv")
    ap.add_argument("--ray-csv", default="ray_params.csv")
    ap.add_argument("--out", default="validation_plots")

    ap.add_argument("--target-clusters-los", type=int, default=None)
    ap.add_argument("--target-clusters-nlos", type=int, default=None)
    ap.add_argument("--target-rays-per-cluster", type=int, default=None)
    ap.add_argument("--target-ntaps", type=int, default=None)

    args = ap.parse_args()
    ensure_outdir(args.out)

    print(f"Loading large-scale CSV: {args.link_csv}")
    link_df = load_link_csv(args.link_csv)

    ss_df = load_ss_csv(args.ss_csv)
    detail_df = load_csv_optional(args.detail_csv)
    ray_df = load_csv_optional(args.ray_csv)

    print(f"Total links: {len(link_df)}")
    if ss_df is not None:
        print(f"Small-scale summary rows: {len(ss_df)}")
    else:
        print("Small-scale power CSV not found; skipping power validation.")

    if detail_df is not None:
        print(f"Detailed small-scale rows: {len(detail_df)}")
    else:
        print("Detailed small-scale CSV not found; skipping tap/cluster validation.")

    if ray_df is not None:
        print(f"Ray rows: {len(ray_df)}")
    else:
        print("Ray CSV not found; skipping per-ray validation.")

    plot_pl_vs_distance(link_df, args.out)
    plot_sf_distribution(link_df, args.out)
    plot_los_probability(link_df, args.out)
    plot_lsp_cdfs(link_df, args.out)
    plot_small_scale_power(ss_df, args.out)
    plot_detail_histograms(detail_df, args.out)
    plot_ray_csv(ray_df, args.out)

    print_df("Large-scale LSP statistics vs 3GPP UMa", summarize_lsp(link_df))
    print_df("Small-scale CIR/CFR summary", summarize_small_scale_power(ss_df))
    print_df("Detail bounds checks", validate_detail_bounds(detail_df))
    print_df("Detail target checks", validate_detail_targets(detail_df, args))
    print_df("Detail LSP checks vs 3GPP UMa", summarize_detail_lsp(detail_df))
    print_df("Ray checks", summarize_ray_csv(ray_df, args))

    print(f"\nDone. Plots written to: {args.out}/")


if __name__ == "__main__":
    main()
