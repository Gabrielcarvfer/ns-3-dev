#!/usr/bin/env python3
"""
validate_3gpp.py — Compare link_params.csv against 3GPP TR 38.901 UMa targets.
Usage: python validate_3gpp.py [--csv link_params.csv] [--out validation_plots/]
Requires: numpy pandas matplotlib scipy
"""
import argparse, math, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

REF = {
    "ds_ns":   {"LOS": {"mu": 1.97, "sigma": 0.66}, "NLOS": {"mu": 2.56, "sigma": 0.39}},
    "asd_deg": {"LOS": {"mu": 1.15, "sigma": 0.28}, "NLOS": {"mu": 1.41, "sigma": 0.28}},
    "asa_deg": {"LOS": {"mu": 1.81, "sigma": 0.20}, "NLOS": {"mu": 1.87, "sigma": 0.20}},
    "zsa_deg": {"LOS": {"mu": 0.95, "sigma": 0.16}, "NLOS": {"mu": 1.26, "sigma": 0.35}},
    "sf_sigma": {"LOS": 4.0, "NLOS": 6.0},
}

FC_GHZ, H_BS, H_UT, H_E = 3.5, 25.0, 1.5, 1.0

def d_bp():
    return 4.0 * (H_BS - H_E) * (H_UT - H_E) * FC_GHZ * 1e9 / 3e8

def uma_los_pl(d2d, d3d):
    dbp = d_bp()
    pl1 = 28.0 + 22.0*np.log10(d3d) + 20.0*np.log10(FC_GHZ)
    pl2 = 28.0 + 40.0*np.log10(d3d) + 20.0*np.log10(FC_GHZ) \
          - 9.0*np.log10(dbp**2 + (H_BS-H_UT)**2)
    return np.where(d2d <= dbp, pl1, pl2)

def uma_nlos_pl(d3d):
    return 32.4 + 20.0*np.log10(FC_GHZ) + 30.0*np.log10(d3d)

def p_los_uma(d):
    p = (18.0/d) + np.exp(-d/63.0)*(1.0 - 18.0/d)
    return np.where(d <= 18.0, 1.0, p)

def load(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["d3d_m"] > 0].copy()
    df["d_bin"] = pd.cut(df["d2d_m"], bins=np.logspace(1, 3.3, 20), include_lowest=True)
    return df

def plot_pl_vs_distance(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    d = np.logspace(1, 3.3, 400)
    for ax, los_val, label, color, ref_fn in [
        (axes[0], 1, "LOS",  "steelblue", lambda d: uma_los_pl(d, d)),
        (axes[1], 0, "NLOS", "tomato",    uma_nlos_pl),
    ]:
        sub = df[df["is_los"] == los_val]
        ax.scatter(sub["d2d_m"], sub["pl_sim_db"], s=2, alpha=0.25, color=color, label="Simulated")
        ax.plot(d, ref_fn(d), "k-", lw=2.5, label="3GPP TR 38.901")
        ax.set_xscale("log"); ax.set_xlabel("2D distance (m)"); ax.set_ylabel("Path loss (dB)")
        ax.set_title(f"UMa {label}  fc={FC_GHZ} GHz")
        ax.legend(markerscale=5); ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{outdir}/pl_vs_distance.png", dpi=150); plt.close(fig)

def plot_sf_distribution(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, los_val, label in [(axes[0], 1, "LOS"), (axes[1], 0, "NLOS")]:
        sub = df[df["is_los"] == los_val]
        sf = sub["sf_db"]
        sig   = REF["sf_sigma"][label]
        bins  = np.linspace(-30, 30, 70)
        ax.hist(sf, bins=bins, density=True, alpha=0.65, color="steelblue",
                label=f"Sim  μ={sf.mean():.2f} σ={sf.std():.2f} dB")
        x = np.linspace(-30, 30, 400)
        ax.plot(x, norm.pdf(x, 0, sig), "r-", lw=2.5, label=f"3GPP σ={sig:.0f} dB")
        ax.set_xlabel("PL (dB)"); ax.set_title(f"Shadow fading — {label}")
        ax.legend(); ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{outdir}/sf_distribution.png", dpi=150); plt.close(fig)

def plot_los_probability(df, outdir):
    grp   = df.groupby("d_bin", observed=True)
    d_mid = grp["d2d_m"].median()
    p_sim = grp["is_los"].mean()
    d_ref = np.logspace(1, 3.3, 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(d_mid.values, p_sim.values, "o-", color="steelblue", ms=5, label="Simulated")
    ax.plot(d_ref, p_los_uma(d_ref), "r--", lw=2.5, label="3GPP TR 38.901")
    ax.set_xscale("log"); ax.set_ylim(0, 1.05)
    ax.set_xlabel("2D distance (m)"); ax.set_ylabel("LOS probability")
    ax.set_title("LOS probability vs distance — UMa")
    ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{outdir}/los_probability.png", dpi=150); plt.close(fig)

def plot_lsp_cdfs(df, outdir):
    lsps = [("ds_ns","DS (ns)"),("asd_deg","ASD (deg)"),("asa_deg","ASA (deg)"),("zsa_deg","ZSA (deg)")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for col, (field, label) in enumerate(lsps):
        for row, (los_val, los_label) in enumerate([(1, "LOS"), (0, "NLOS")]):
            ax  = axes[row][col]
            sub = df[(df["is_los"] == los_val) & (df[field] > 0)][field]
            ref = REF[field].get(los_label)
            if sub.empty or ref is None: ax.set_visible(False); continue
            lv = np.log10(sub.values)
            ax.plot(np.sort(lv), np.linspace(0,1,len(lv)), lw=1.5, color="steelblue", label="Simulated")
            x_ref = np.linspace(ref["mu"]-4*ref["sigma"], ref["mu"]+4*ref["sigma"], 300)
            ax.plot(x_ref, norm.cdf(x_ref, ref["mu"], ref["sigma"]),
                    "r--", lw=2, label=f'3GPP μ={ref["mu"]:.2f}')
            ax.set_xlabel(f'log₁₀({label})'); ax.set_ylabel("CDF")
            ax.set_title(f"{los_label} — {label}")
            ax.legend(fontsize=7); ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{outdir}/lsp_cdfs.png", dpi=150); plt.close(fig)

def print_summary(df):
    rows = []
    for los_val, los_label in [(1,"LOS"),(0,"NLOS")]:
        sub = df[df["is_los"] == los_val]
        for field, lbl in [("ds_ns","DS"),("asd_deg","ASD"),("asa_deg","ASA"),("zsa_deg","ZSA")]:
            vals = sub[sub[field] > 0][field]
            if vals.empty: continue
            lv  = np.log10(vals)
            ref = REF[field].get(los_label, {})
            rows.append({"State": los_label, "LSP": lbl, "N": len(vals),
                         "Sim μ":  f"{lv.mean():.3f}", "3GPP μ": f"{ref.get('mu',float('nan')):.3f}",
                         "Sim σ":  f"{lv.std():.3f}",  "3GPP σ": f"{ref.get('sigma',float('nan')):.3f}"})
    print("\n── LSP statistics vs 3GPP TR 38.901 ─────────────────────────")
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="link_params.csv")
    ap.add_argument("--out", default="validation_plots")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"Loading {args.csv} …")
    df = load(args.csv)

    print(f"Total links: {len(df)}")
    print(f"LOS outdoor: {len(df[(df['is_los']==1) & (df['is_outdoor']==1)])}")
    print(f"NLOS outdoor SF==0: {(df[(df['is_los']==0) & (df['is_outdoor']==1)]['sf_db'] == 0.0).sum()}")
    print(f"NLOS outdoor SF!=0: {(df[(df['is_los']==0) & (df['is_outdoor']==1)]['sf_db'] != 0.0).sum()}")

    # Are the SF==0 links clustered by cell or UT?
    nlos_out = df[(df['is_los']==0) & (df['is_outdoor']==1)]
    zero_sf  = nlos_out[nlos_out['sf_db'] == 0.0]
    print(f"Unique cells with SF==0 NLOS: {zero_sf['cell_id'].nunique()}")
    print(f"Unique UTs  with SF==0 NLOS: {zero_sf['ut_id'].nunique()}")
    print(f"Total sites: {df['cell_id'].nunique()}, Total UTs: {df['ut_id'].nunique()}")
    plot_pl_vs_distance(df, args.out)
    plot_sf_distribution(df, args.out)
    plot_los_probability(df, args.out)
    plot_lsp_cdfs(df, args.out)
    print_summary(df)
    print(f"\nDone. Plots in {args.out}/")
