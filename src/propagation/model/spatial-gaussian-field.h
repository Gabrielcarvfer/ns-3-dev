/*
 * Copyright (c) 2026, Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#ifndef SPATIAL_GAUSSIAN_FIELD_H
#define SPATIAL_GAUSSIAN_FIELD_H

#include "ns3/vector.h"

#include <array>
#include <cstdint>

namespace ns3
{

/**
 * @ingroup propagation
 * @brief Stateless, position-keyed, spatially-correlated Gaussian random field.
 *
 * The value at a point is a pure function of its coordinates, the global RNG
 * seed and run, a class-specific salt and a per-field key, so every instance
 * and every call observes the same realization, even when the models sampling
 * it are regenerated (e.g. by a REM generator). This suits the drop-based
 * spatial consistency of 3GPP TR 38.901 Sec. 7.6.3.1, and the absence of
 * mutable state makes the field thread-safe.
 *
 * The field is white noise on a fixed grid (one hashed N(0,1) value per cell)
 * filtered with a separable exponential kernel, L2-normalized so that every
 * position has an exactly N(0,1) marginal, with the decay length scaled so
 * that the autocorrelation equals 1/e at the correlation distance as the
 * exp(-d/dcor) of TR 38.901.
 *
 * @note The autocorrelation is not exactly exponential: it is smooth at the
 * origin, within about 0.04 of exp(-d/dcor) up to three correlation distances
 * and exactly zero beyond. A spec-exact sum-of-sinusoids field (the QuaDRiGa
 * approach) would carry a ripple of about 0.05 at every lag and cost more per
 * sample, which was judged not worth the complexity.
 */
class SpatialGaussianField
{
  public:
    /// Per-cell N(0,1) generator of the white-noise background.
    enum class CellGenerator
    {
        BoxMuller, ///< Box-Muller transform of two 53-bit uniforms
        /// Standardized sum of the four 16-bit chunks of the hashed state (Irwin-Hall),
        /// cheaper than Box-Muller; the windowed sum of Sample() restores a Gaussian marginal
        IrwinHall
    };

    /// Number of grid cells per dimension of a filter window.
    static constexpr std::size_t WINDOW_CELLS = 14;

    /**
     * @brief Filter window of one sampling position (grid origin, separable weights and
     *        their L2 normalization), computed once with ComputeWindow() and reusable
     *        with SampleWindow() for every field sampled at that position.
     */
    struct Window
    {
        int64_t ix{0};                         ///< grid x-coordinate of the first window cell
        int64_t iy{0};                         ///< grid y-coordinate of the first window cell
        std::array<double, WINDOW_CELLS> wx{}; ///< separable filter weights along x
        std::array<double, WINDOW_CELLS> wy{}; ///< separable filter weights along y
        double invL2Norm{0};                   ///< reciprocal L2 norm of the 2D weights
    };

    /**
     * @brief Construct a field.
     *
     * @param salt Class-specific salt mixed into every cell, so that fields
     *             keyed on the same field key by different classes are
     *             statistically independent.
     * @param generator The per-cell N(0,1) generator.
     */
    explicit SpatialGaussianField(uint64_t salt,
                                  CellGenerator generator = CellGenerator::BoxMuller);

    /**
     * @brief SplitMix64 mixing round.
     * @param x The state to mix.
     * @return The mixed state.
     */
    static uint64_t SplitMix64(uint64_t x);

    /**
     * @brief Hash prefix of one field, mixing the global RNG seed and run, the salt
     *        and the field key once for all cells (see CellFromPrefix()).
     *
     * @param fieldKey Per-field key selecting one independent realization
     *                 (e.g. packed site id and slot).
     * @return The mixed hash state of the field.
     */
    uint64_t Prefix(uint64_t fieldKey) const;

    /**
     * @brief Deterministic N(0,1) value of one cell of the field.
     *
     * @param prefix Hash prefix of the field, see Prefix().
     * @param ix Integer grid x-coordinate of the cell.
     * @param iy Integer grid y-coordinate of the cell.
     * @return A standard-normal value deterministic in (prefix, ix, iy).
     */
    double CellFromPrefix(uint64_t prefix, int64_t ix, int64_t iy) const;

    /**
     * @brief Deterministic N(0,1) value of one cell of the field.
     *
     * @param fieldKey Per-field key, see Prefix().
     * @param ix Integer grid x-coordinate of the cell.
     * @param iy Integer grid y-coordinate of the cell.
     * @return A standard-normal value deterministic in (seed, run, salt,
     *         fieldKey, ix, iy).
     */
    double Cell(uint64_t fieldKey, int64_t ix, int64_t iy) const;

    /**
     * @brief Compute the filter window of a sampling position.
     *
     * @param position The sampling position (only x and y are used).
     * @param corrDist The correlation distance, must be positive.
     * @return The filter window.
     */
    static Window ComputeWindow(const Vector& position, double corrDist);

    /**
     * @brief Sample the field through a precomputed filter window.
     *
     * @param prefix Hash prefix of the field, see Prefix().
     * @param window The filter window, see ComputeWindow().
     * @return A standard-normal sample.
     */
    double SampleWindow(uint64_t prefix, const Window& window) const;

    /**
     * @brief Sample the spatially-correlated field at a position; a non-positive
     *        correlation distance degrades to an uncorrelated, position-repeatable draw.
     *
     * @param fieldKey Per-field key, see Prefix().
     * @param position The sampling position (only x and y are used).
     * @param corrDist The correlation distance at which the autocorrelation
     *                 equals 1/e.
     * @return A standard-normal sample.
     */
    double Sample(uint64_t fieldKey, const Vector& position, double corrDist) const;

  private:
    uint64_t m_salt;           ///< class-specific salt
    CellGenerator m_generator; ///< per-cell N(0,1) generator
};

} // namespace ns3

#endif // SPATIAL_GAUSSIAN_FIELD_H
