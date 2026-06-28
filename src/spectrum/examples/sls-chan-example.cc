// This is an AI translated version of the original Nvidia source for GPU compute,
// but ported to WGPU for cross-platform support
// https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/main/testBenches/chanModels/src/sls_chan_src/sls_chan_large_scale_GPU.cu
//
// SlsChanWgpu now owns its WebGPU instance/device internally (see
// SlsChanWgpu::SlsChanWgpu in sls-chan-wgpu.cc), so this harness only
// builds the cell/UT topology, drives the large-scale pipeline and
// reads the LSP results back.

#include "sls-chan-wgpu.h"

#include <iostream>

int
main()
{
    SlsChanWgpu sls;

    // ── Build a minimal cell + UT layout for testing ──────────────────────
    const uint32_t nSite = 7;
    const uint32_t nUT = 210;
    const uint32_t nSector = 3;

    std::vector<CellParam> cells(nSite * nSector);
    for (uint32_t s = 0; s < nSite; ++s)
    {
        for (uint32_t k = 0; k < nSector; ++k)
        {
            auto& c = cells[s * nSector + k];
            // CellParam.loc is a vec3<f32> (length 3), not 4.
            c.loc[0] = static_cast<float>(s * 200); // 200 m ISD
            c.loc[1] = 0.f;
            c.loc[2] = 25.f; // 25 m height
        }
    }

    std::vector<UtParam> uts(nUT);
    for (uint32_t u = 0; u < nUT; ++u)
    {
        auto& ut = uts[u];
        ut.loc = {static_cast<float>(static_cast<int>(u) * 14 - 700),
                  static_cast<float>(static_cast<int>(u) * 7),
                  1.5f,
                  0.f};
        ut.d_2d_in = 0.f;
        ut.outdoor_ind = 1u; // outdoor
        ut.o2i_penetration_loss = 0.f;
    }

    sls.uploadCellParams(cells, nSector);
    sls.uploadUtParams(uts);

    // Generate CRN grids. The on-device LSP layout is now 8/7/7 channels
    // (LOS adds an explicit ZSD slot, NLOS/O2I each carry a final
    // additional slot — these mirror the canonical UMa correlation
    // distances baked into ThreeGppChannelModelWgpuMezanine).
    float corrLos[8] = {37.f, 12.f, 30.f, 18.f, 15.f, 15.f, 15.f, 20.f};
    float corrNlos[7] = {50.f, 40.f, 50.f, 50.f, 50.f, 50.f, 25.f};
    float corrO2i[7] = {10.f, 10.f, 11.f, 17.f, 25.f, 25.f, 20.f};
    sls.generateCRN(700.f, -700.f, 700.f, -700.f, corrLos, corrNlos, corrO2i);

    // Run main link parameter kernel
    sls.calLinkParam(nSite,
                     nUT,
                     nSector,
                     700.f,
                     -700.f,
                     700.f,
                     -700.f,
                     /*updatePL=*/true,
                     /*updateAllLSPs=*/true,
                     /*updateLos=*/true,
                     /*updateOptionalPL=*/false,
                     /*nX=*/141,
                     /*nY=*/141);

    auto results = sls.readLinkParams(nSite, nUT);

    for (uint32_t nLink = 0; nLink < results.size(); nLink++)
    {
        std::cout << "Link[" << nLink << "]: PL=" << results[nLink].pathloss
                  << " dB, DS=" << results[nLink].DS << " ns, LOS=" << results[nLink].losInd
                  << ", d2d=" << results[nLink].d2d << ", d3d=" << results[nLink].d3d << "\n";
    }
    return 0;
}
