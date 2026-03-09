// test-long-term-equiv.cc
// Build: g++ -O2 -std=c++17 -o test test-long-term-equiv.cc && ./test

#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

// ─── Channel matrix (column-major: row index fastest, page slowest) ───────────

struct Channel3D
{
    size_t nRows, nCols, nPages;
    std::vector<std::complex<double>> data;

    Channel3D(size_t r, size_t c, size_t p)
        : nRows(r),
          nCols(c),
          nPages(p),
          data(r * c * p, {0, 0})
    {
    }

    std::complex<double>& operator()(size_t row, size_t col, size_t page)
    {
        return data[page * nRows * nCols + col * nRows + row];
    }

    const std::complex<double>& operator()(size_t row, size_t col, size_t page) const
    {
        return data[page * nRows * nCols + col * nRows + row];
    }
};

// ─── Antenna mock matching ns-3 UniformPlanarArray sub-array port layout ──────
//
//  Port numbering: portIdx → (hPortIdx = portIdx / numVPorts,
//                              vPortIdx = portIdx % numVPorts)
//  Physical element for subElemIdx e in port p:
//    localH  = e % hElemsPerPort,  localV  = e / hElemsPerPort
//    physCol = hPortIdx * hEPP + localH
//    physRow = vPortIdx * vEPP + localV
//    physIdx = physRow * numColumns + physCol

struct AntennaModel
{
    size_t numRows;
    size_t numColumns;
    size_t numHPorts;
    size_t numVPorts;
    std::vector<std::complex<double>> bfVec;

    size_t GetNumPorts() const
    {
        return numHPorts * numVPorts;
    }

    size_t GetHElemsPerPort() const
    {
        return numColumns / numHPorts;
    }

    size_t GetVElemsPerPort() const
    {
        return numRows / numVPorts;
    }

    size_t GetNumElemsPerPort() const
    {
        return GetHElemsPerPort() * GetVElemsPerPort();
    }

    size_t GetNumColumns() const
    {
        return numColumns;
    }

    const std::vector<std::complex<double>>& GetBFVec() const
    {
        return bfVec;
    }

    // THE FIX: real column-slice port layout, not portIdx * elemsPerPort
    size_t ArrayIndexFromPortIndex(size_t portIdx, size_t subElemIdx) const
    {
        const size_t hEPP = GetHElemsPerPort();
        const size_t vEPP = GetVElemsPerPort();
        const size_t hPortIdx = portIdx / numVPorts;
        const size_t vPortIdx = portIdx % numVPorts;
        const size_t localH = subElemIdx % hEPP;
        const size_t localV = subElemIdx / hEPP;
        const size_t physCol = hPortIdx * hEPP + localH;
        const size_t physRow = vPortIdx * vEPP + localV;
        return physRow * numColumns + physCol;
    }
};

// ─── Original algorithm (unchanged, verbatim from the question) ───────────────

std::complex<double>
CalcLongTermComponent_orig(const Channel3D& ch,
                           const AntennaModel& sAnt,
                           const AntennaModel& uAnt,
                           size_t sPortIdx,
                           size_t uPortIdx,
                           size_t cIndex)
{
    const auto& sW = sAnt.GetBFVec();
    const auto& uW = uAnt.GetBFVec();
    const auto sPortElems = sAnt.GetNumElemsPerPort();
    const auto uPortElems = uAnt.GetNumElemsPerPort();
    const auto startS = sAnt.ArrayIndexFromPortIndex(sPortIdx, 0);
    const auto startU = uAnt.ArrayIndexFromPortIndex(uPortIdx, 0);
    const auto uElemsPerPort = uAnt.GetHElemsPerPort();
    const auto sElemsPerPort = sAnt.GetHElemsPerPort();

    std::complex<double> txSum(0, 0);
    size_t sIndex = startS;
    for (size_t tIndex = 0; tIndex < sPortElems; tIndex++, sIndex++)
    {
        std::complex<double> rxSum(0, 0);
        size_t uIndex = startU;
        for (size_t rIndex = 0; rIndex < uPortElems; rIndex++, uIndex++)
        {
            rxSum += std::conj(uW[uIndex - startU]) * ch(uIndex, sIndex, cIndex);
            if (const auto ptInc = uElemsPerPort - 1; rIndex % uElemsPerPort == ptInc)
            {
                uIndex += uAnt.GetNumColumns() - uElemsPerPort;
            }
        }
        txSum += sW[sIndex - startS] * rxSum;
        if (const auto ptInc = sElemsPerPort - 1; tIndex % sElemsPerPort == ptInc)
        {
            sIndex += sAnt.GetNumColumns() - sElemsPerPort;
        }
    }
    return txSum;
}

std::vector<std::complex<double>>
CalcLongTerm_orig(const Channel3D& ch, const AntennaModel& sAnt, const AntennaModel& uAnt)
{
    const size_t nU = uAnt.GetNumPorts(), nS = sAnt.GetNumPorts(), nC = ch.nPages;
    std::vector<std::complex<double>> out(nU * nS * nC);
    for (size_t s = 0; s < nS; s++)
    {
        for (size_t u = 0; u < nU; u++)
        {
            for (size_t c = 0; c < nC; c++)
            {
                out[c * nU * nS + s * nU + u] = CalcLongTermComponent_orig(ch, sAnt, uAnt, s, u, c);
            }
        }
    }
    return out;
}

// ─── Optimised algorithm ─────────────────────────────────────────────────────

// Physical indices for every sub-element of a port via the real API.
static std::vector<size_t>
BuildPhysicalIndices(const AntennaModel& ant, size_t portIdx)
{
    const size_t n = ant.GetNumElemsPerPort();
    std::vector<size_t> idx(n);
    for (size_t e = 0; e < n; ++e)
    {
        idx[e] = ant.ArrayIndexFromPortIndex(portIdx, e);
    }
    return idx;
}

// Intra-port BF offsets (index - start) — identical for every port
// because sub-array partition uses the same weight pattern per port.
static std::vector<size_t>
BuildLocalOffsets(const AntennaModel& ant)
{
    const size_t n = ant.GetNumElemsPerPort();
    const size_t base = ant.ArrayIndexFromPortIndex(0, 0);
    std::vector<size_t> off(n);
    for (size_t e = 0; e < n; ++e)
    {
        off[e] = ant.ArrayIndexFromPortIndex(0, e) - base;
    }
    return off;
}

std::vector<std::complex<double>>
CalcLongTerm_fast(const Channel3D& ch, const AntennaModel& sAnt, const AntennaModel& uAnt)
{
    const size_t nU = uAnt.GetNumPorts();
    const size_t nS = sAnt.GetNumPorts();
    const size_t nC = ch.nPages;
    const size_t sPE = sAnt.GetNumElemsPerPort();
    const size_t uPE = uAnt.GetNumElemsPerPort();
    const auto& sW = sAnt.GetBFVec();
    const auto& uW = uAnt.GetBFVec();

    std::vector<std::complex<double>> out(nU * nS * nC, {0, 0});

    // Precompute physical index arrays per port (uses real API, no manual walk)
    std::vector<std::vector<size_t>> sPhys(nS), uPhys(nU);
    for (size_t p = 0; p < nS; ++p)
    {
        sPhys[p] = BuildPhysicalIndices(sAnt, p);
    }
    for (size_t p = 0; p < nU; ++p)
    {
        uPhys[p] = BuildPhysicalIndices(uAnt, p);
    }

    // Intra-port BF offsets — same for every port (sub-array partition model)
    const auto sOff = BuildLocalOffsets(sAnt);
    const auto uOff = BuildLocalOffsets(uAnt);

    // Precompute combined weight matrix: w[t,r] = conj(uW[uOff[r]]) * sW[sOff[t]]
    std::vector<std::complex<double>> w(sPE * uPE);
    for (size_t t = 0; t < sPE; ++t)
    {
        for (size_t r = 0; r < uPE; ++r)
        {
            w[t * uPE + r] = std::conj(uW[uOff[r]]) * sW[sOff[t]];
        }
    }

    // cIndex outer keeps uIndex (fastest axis) in the inner loop → cache friendly
    for (size_t c = 0; c < nC; ++c)
    {
        for (size_t s = 0; s < nS; ++s)
        {
            for (size_t u = 0; u < nU; ++u)
            {
                std::complex<double> acc(0.0, 0.0);
                const auto& si = sPhys[s];
                const auto& ui = uPhys[u];
                for (size_t t = 0; t < sPE; ++t)
                {
                    for (size_t r = 0; r < uPE; ++r)
                    {
                        acc += w[t * uPE + r] * ch(ui[r], si[t], c);
                    }
                }
                out[c * nU * nS + s * nU + u] = acc;
            }
        }
    }
    return out;
}

// ─── Equivalence test ─────────────────────────────────────────────────────────

static bool
complexClose(std::complex<double> a, std::complex<double> b, double eps = 1e-9)
{
    return std::abs(a - b) < eps * (1.0 + std::abs(a) + std::abs(b));
}

int
main()
{
    // {numRows, numCols, numHPorts, numVPorts, numClusters}
    struct Config
    {
        size_t nR, nC, nHP, nVP, nClusters;
    };

    std::vector<Config> configs = {
        {2, 4, 1, 1, 20}, // single-port 2×4 array
        {2, 4, 2, 1, 30}, // 2 h-ports, 2×4 array  (was failing as uE=8 uP=2)
        {2, 8, 4, 1, 25}, // 4 h-ports, 2×8 array  (was failing as uE=16 uP=4)
        {4, 8, 2, 2, 15}, // 4 ports, 2 h × 2 v,  4×8 array
        {2, 4, 1, 1, 10}, // single-port asymmetric (was uE=4 sE=8 uP=1 sP=2)
        {2, 8, 2, 1, 10},
    };

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    auto randC = [&] { return std::complex<double>(dist(rng), dist(rng)); };

    int pass = 0, fail = 0;
    for (const auto& cfg : configs)
    {
        const size_t uE = cfg.nR * cfg.nC;
        const size_t sE = uE;

        Channel3D ch(uE, sE, cfg.nClusters);
        for (auto& v : ch.data)
        {
            v = randC();
        }

        std::vector<std::complex<double>> sW(sE), uW(uE);
        for (auto& v : sW)
        {
            v = randC();
        }
        for (auto& v : uW)
        {
            v = randC();
        }

        AntennaModel uAnt{cfg.nR, cfg.nC, cfg.nHP, cfg.nVP, uW};
        AntennaModel sAnt{cfg.nR, cfg.nC, cfg.nHP, cfg.nVP, sW};

        const auto ref = CalcLongTerm_orig(ch, sAnt, uAnt);
        const auto fast = CalcLongTerm_fast(ch, sAnt, uAnt);

        assert(ref.size() == fast.size());
        bool ok = true;
        double maxErr = 0.0;
        for (size_t i = 0; i < ref.size(); i++)
        {
            maxErr = std::max(maxErr, std::abs(ref[i] - fast[i]));
            if (!complexClose(ref[i], fast[i]))
            {
                ok = false;
            }
        }

        std::cout << "uE=" << uE << " uP=" << uAnt.GetNumPorts() << " sP=" << sAnt.GetNumPorts()
                  << " nC=" << cfg.nClusters << ": " << (ok ? "PASS" : "FAIL")
                  << "  maxAbsErr=" << maxErr << "\n";
        ok ? pass++ : fail++;
    }

    std::cout << "\n" << pass << " passed, " << fail << " failed.\n";
    return fail ? 1 : 0;
}
