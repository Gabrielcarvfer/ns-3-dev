#!/usr/bin/env python3
#
# Copyright (c) 2026, Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
"""Generate the reference values of three-gpp-ntn-channel-calibration-test-suite.cc.

The values are read from the tables of 3GPP TR 38.811 V15.4.0, not from ns-3:

- Table 6.6.1-1, LOS probability per elevation;
- Tables 6.6.2-1 to 6.6.2-3, shadow fading and clutter loss per elevation and band;
- Table 6.6.6.2.1-1, tropospheric scintillation loss per elevation;
- Tables 6.7.2-1a to 6.7.2-8b, fast fading parameters per scenario, LOS state,
  band and elevation: the means and standard deviations of the LSPs and their
  cross-correlations.

and, when 3GPP TR 38.821 V16.2.0 is also given, from its Tables 6.1.1.2-1 and
6.1.1.2-2: the 5%, 50% and 95% percentiles of the coupling loss, geometry SIR
and geometry SINR of the system level calibration, averaged over the companies.

The TR is distributed as a .doc file, which must first be saved as .docx (for
example with Word or LibreOffice):

    https://www.3gpp.org/ftp/Specs/archive/38_series/38.811/38811-f40.zip
    https://www.3gpp.org/ftp/Specs/archive/38_series/38.821/38821-g20.zip

Some symbols of the parameter names use a symbol font, which the text of the
.docx does not carry; the rows of Tables 6.7.2-1a to 6.7.2-8b are therefore read
by position, in the row order of TR 38.901 Table 7.5-6.

Usage:
    three-gpp-ntn-channel-calibration-reference.py 38811-f40.docx 38821-g20.docx > three-gpp-ntn-channel-calibration-reference.h
"""

import argparse
import re
import xml.etree.ElementTree as ET
import zipfile

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
ELEVATIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
SCENARIOS = {
    "Dense Urban": "DenseUrban",
    "Urban": "Urban",
    "Suburban": "Suburban",
    "Rural": "Rural",
}
# LSP order of the reference arrays and of the ThreeGppChannelModel square root matrices
ORDER = ["SF", "K", "DS", "ASD", "ASA", "ZSD", "ZSA"]
# cross-correlation rows of Tables 6.7.2-1a to 6.7.2-8b, as in TR 38.901 Table 7.5-6
XCORR_ROWS = [
    ("ASD", "DS"), ("ASA", "DS"), ("ASA", "SF"), ("ASD", "SF"), ("DS", "SF"),
    ("ASD", "ASA"), ("ASD", "K"), ("ASA", "K"), ("DS", "K"), ("SF", "K"),
    ("ZSD", "SF"), ("ZSA", "SF"), ("ZSD", "K"), ("ZSA", "K"), ("ZSD", "DS"),
    ("ZSA", "DS"), ("ZSD", "ASD"), ("ZSA", "ASD"), ("ZSD", "ASA"), ("ZSA", "ASA"),
    ("ZSD", "ZSA"),
]  # fmt: skip


def text(el):
    return re.sub(r"\s+", " ", "".join(n.text or "" for n in el.iter(W + "t"))).strip()


def tables(path):
    """Yield (caption, rows) for every table, with the paragraph that precedes it."""
    root = ET.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
    caption = ""
    for el in root.find(W + "body"):
        if el.tag == W + "p":
            t = text(el)
            if t:
                caption = t
        elif el.tag == W + "tbl":
            yield caption, [[text(tc) for tc in tr.findall(W + "tc")] for tr in el.iter(W + "tr")]


def number(s):
    s = s.replace("−", "-").replace("–", "-").strip()
    return None if s in ("N/A", "") else float(s)


def fast_fading(path):
    out = {}
    for caption, rows in tables(path):
        m = re.match(
            r"Table 6\.7\.2-\d[ab]: Channel model parameters for (Dense Urban|Urban|Suburban|Rural)"
            r" Scenario \((LOS|NLOS)\) (?:in|at) (S|Ka) band",
            caption,
        )
        if not m:
            continue
        values = []
        for r in rows[2:]:
            if len(r) < 10 or "Table 6.6.2" in " ".join(r):
                continue
            try:
                values.append([number(x) for x in r[-9:]])
            except ValueError:
                continue
        names = []
        for p in ("DS", "ASD", "ASA", "ZSA", "ZSD"):
            names += ["mu" + p, "sigma" + p]
        has_k = any(r[0].startswith("K-factor") for r in rows)
        if has_k:
            names += ["muK", "sigmaK"]
        names += ["x_%s_%s" % pair for pair in XCORR_ROWS]
        names += ["rTau", "muXPR", "sigmaXPR", "N", "M", "cDS", "cASD", "cASA", "cZSA", "zeta"]
        names += ["d_" + c for c in ["DS", "ASD", "ASA", "SF", "K", "ZSA", "ZSD"]]
        assert len(names) == len(values), (caption, len(names), len(values))
        per = {e: {n: values[i][k] for i, n in enumerate(names)} for k, e in enumerate(ELEVATIONS)}
        out[(SCENARIOS[m.group(1)], m.group(2) == "LOS", m.group(3))] = (
            per,
            has_k and m.group(2) == "LOS",
        )
    assert len(out) == 16
    return out


def los_probability(path):
    for caption, rows in tables(path):
        if caption.startswith("Table 6.6.1-1"):
            res = []
            for r in rows[1:]:
                e = int(r[0].rstrip("°"))
                du, u, sr = (float(x.rstrip("%")) / 100 for x in r[1:4])
                res += [
                    ("DenseUrban", e, du),
                    ("Urban", e, u),
                    ("Suburban", e, sr),
                    ("Rural", e, sr),
                ]
            return res


def sf_cl(path):
    scenarios = {"6.6.2-1": ["DenseUrban"], "6.6.2-2": ["Urban"], "6.6.2-3": ["Suburban", "Rural"]}
    res = []
    for caption, rows in tables(path):
        m = re.match(r"Table (6\.6\.2-[123])", caption)
        if not m:
            continue
        for r in rows[3:]:
            e = int(r[0].rstrip("°"))
            v = [float(x) for x in r[1:7]]
            for s in scenarios[m.group(1)]:
                res.append((s, "S", e, v[0], v[1], v[2]))
                res.append((s, "Ka", e, v[3], v[4], v[5]))
    return res


def troposphere(path):
    for caption, rows in tables(path):
        if caption.startswith("Table 6.6.6.2.1-1"):
            return [(int(r[0]), float(r[1].split()[0])) for r in rows[1:]]


def sls_calibration(path):
    """Rows (case, DL, CL, SIR, SINR percentiles) of TR 38.821 Tables 6.1.1.2-1 and 6.1.1.2-2."""
    res = []
    for caption, rows in tables(path):
        m = re.match(r"Table 6\.1\.1\.2-([12])", caption)
        if not m:
            continue
        for r in rows[2:]:
            if not r[0].startswith("SC"):
                continue
            v = [float(x) for x in r[1:10]]
            res.append((int(r[0][2:]), m.group(1) == "1", v[0:3], v[3:6], v[6:9]))
    return res


def fmt(v):
    return repr(round(v, 6))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("docx", help="TR 38.811 V15.4.0 saved as .docx")
    parser.add_argument("tr38821", help="TR 38.821 V16.2.0 saved as .docx")
    args = parser.parse_args()
    path = args.docx

    print("""/*
 * Copyright (c) 2026, Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

// Generated by three-gpp-ntn-channel-calibration-reference.py; do not edit.

#ifndef THREE_GPP_NTN_CHANNEL_CALIBRATION_REFERENCE_H
#define THREE_GPP_NTN_CHANNEL_CALIBRATION_REFERENCE_H

#include <array>

namespace ns3
{
namespace ntncalibration
{

// clang-format off
/// LSPs of one table and elevation of TR 38.811 Tables 6.7.2-1a to 6.7.2-8b, in the
/// [SF, K, DS, ASD, ASA, ZSD, ZSA] order (log10 for the spreads, dB for SF and K)
struct LspReference
{
    const char* scenario;                         ///< DenseUrban, Urban, Suburban or Rural
    bool los;                                     ///< LOS table
    const char* band;                             ///< S or Ka
    int elevation;                                ///< reference elevation in degrees
    bool hasK;                                    ///< whether the K-factor is defined
    std::array<double, 7> mu;                     ///< means (0 for SF)
    std::array<double, 7> sigma;                  ///< standard deviations (0 for SF)
    std::array<std::array<double, 7>, 7> corr;    ///< cross-correlation matrix
};

/// LOS probability of TR 38.811 Table 6.6.1-1
struct LosReference
{
    const char* scenario; ///< DenseUrban, Urban, Suburban or Rural
    int elevation;        ///< reference elevation in degrees
    double pLos;          ///< LOS probability
};

/// Shadow fading and clutter loss of TR 38.811 Tables 6.6.2-1 to 6.6.2-3
struct SfClReference
{
    const char* scenario; ///< DenseUrban, Urban, Suburban or Rural
    const char* band;     ///< S or Ka
    int elevation;        ///< reference elevation in degrees
    double sfLos;         ///< LOS shadow fading standard deviation in dB
    double sfNlos;        ///< NLOS shadow fading standard deviation in dB
    double clNlos;        ///< NLOS clutter loss in dB
};

/// Tropospheric scintillation loss of TR 38.811 Table 6.6.6.2.1-1
struct TroposphereReference
{
    int elevation; ///< reference elevation in degrees
    double loss;   ///< loss in dB
};

/// System level calibration result of TR 38.821 Tables 6.1.1.2-1 (DL) and 6.1.1.2-2 (UL):
/// 5%, 50% and 95% percentiles, averaged over the companies, in dB
struct SlsReference
{
    int sc;                     ///< calibration study case of TR 38.821 Table 6.1.1.1-9
    bool dl;                    ///< DL (Table 6.1.1.2-1) or UL (Table 6.1.1.2-2)
    std::array<double, 3> cl;   ///< coupling loss
    std::array<double, 3> sir;  ///< geometry SIR
    std::array<double, 3> sinr; ///< geometry SINR
};
""")
    ff = fast_fading(path)
    print("/// TR 38.811 Tables 6.7.2-1a to 6.7.2-8b")
    print("inline constexpr std::array<LspReference, %d> kLsp{{" % (len(ff) * len(ELEVATIONS)))
    for (scen, los, band), (per, has_k) in sorted(ff.items()):
        for e in ELEVATIONS:
            p = per[e]
            mu = [0.0] + [p["muK"] if has_k else 0.0] + [p["mu" + x] for x in ORDER[2:]]
            sig = [0.0] + [p["sigmaK"] if has_k else 0.0] + [p["sigma" + x] for x in ORDER[2:]]
            corr = [[1.0 if i == j else 0.0 for j in range(7)] for i in range(7)]
            for a, b in XCORR_ROWS:
                v = p["x_%s_%s" % (a, b)]
                if v is not None:
                    i, j = ORDER.index(a), ORDER.index(b)
                    corr[i][j] = corr[j][i] = v
            rows = ", ".join("{" + ", ".join(fmt(x) for x in r) + "}" for r in corr)
            print(
                '    {"%s", %s, "%s", %d, %s, {%s}, {%s}, {{%s}}},'
                % (
                    scen,
                    "true" if los else "false",
                    band,
                    e,
                    "true" if has_k else "false",
                    ", ".join(fmt(x) for x in mu),
                    ", ".join(fmt(x) for x in sig),
                    rows,
                )
            )
    print("}};\n")
    lp = los_probability(path)
    print("/// TR 38.811 Table 6.6.1-1")
    print("inline constexpr std::array<LosReference, %d> kLos{{" % len(lp))
    for s, e, p in lp:
        print('    {"%s", %d, %s},' % (s, e, fmt(p)))
    print("}};\n")
    sc = sf_cl(path)
    print("/// TR 38.811 Tables 6.6.2-1 to 6.6.2-3")
    print("inline constexpr std::array<SfClReference, %d> kSfCl{{" % len(sc))
    for s, b, e, a, n, c in sc:
        print('    {"%s", "%s", %d, %s, %s, %s},' % (s, b, e, fmt(a), fmt(n), fmt(c)))
    print("}};\n")
    tr = troposphere(path)
    print("/// TR 38.811 Table 6.6.6.2.1-1")
    print("inline constexpr std::array<TroposphereReference, %d> kTroposphere{{" % len(tr))
    for e, v in tr:
        print("    {%d, %s}," % (e, fmt(v)))
    print("}};\n")
    sls = sls_calibration(args.tr38821)
    print("/// TR 38.821 Tables 6.1.1.2-1 and 6.1.1.2-2")
    print("inline constexpr std::array<SlsReference, %d> kSls{{" % len(sls))
    for sc, dl, cl, sir, sinr in sls:
        print(
            "    {%d, %s, {%s}, {%s}, {%s}},"
            % (
                sc,
                "true" if dl else "false",
                ", ".join(fmt(x) for x in cl),
                ", ".join(fmt(x) for x in sir),
                ", ".join(fmt(x) for x in sinr),
            )
        )
    print("}};")
    print("""// clang-format on

} // namespace ntncalibration
} // namespace ns3

#endif // THREE_GPP_NTN_CHANNEL_CALIBRATION_REFERENCE_H""")


if __name__ == "__main__":
    main()
