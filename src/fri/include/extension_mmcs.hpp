#pragma once

/**
 * @file extension_mmcs.hpp
 * @brief ExtensionMmcs: wraps MerkleTreeMmcs<Val,...> for extension-field matrices.
 *
 * Each EF element is decomposed into EF::DEGREE Val elements before committing,
 * so the inner commitment stores a Val matrix of width = ef_width * EF::DEGREE.
 */

#include "p3_util/util.hpp"

#include <vector>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace p3_merkle {

/**
 * ExtensionMmcs wraps an inner MMCS for Val and handles EF matrices by
 * decomposing each EF element into DEGREE Val elements.
 *
 * Template parameters:
 *   Val       - base field element type
 *   EF        - extension field type (must have DEGREE, coeffs[])
 *   InnerMmcs - MerkleTreeMmcs<Val, Val, H, C, N, CHUNK> (or compatible)
 */
template <typename Val, typename EF, typename InnerMmcs>
class ExtensionMmcs {
public:
    using Commitment   = typename InnerMmcs::Commitment;
    using OpeningProof = typename InnerMmcs::OpeningProof;

    struct ProverData {
        typename InnerMmcs::ProverData inner;
        size_t ef_width;  // number of EF elements per row (before decomposition)
    };

    InnerMmcs inner_mmcs;

    ExtensionMmcs() = default;
    explicit ExtensionMmcs(InnerMmcs mmcs) : inner_mmcs(std::move(mmcs)) {}

    // -----------------------------------------------------------------------
    // commit_matrix: commit a flat EF vector as a matrix of ef_width EF elements
    // -----------------------------------------------------------------------
    std::pair<Commitment, ProverData> commit_matrix(
        const std::vector<EF>& data,
        size_t ef_width) const
    {
        if (ef_width == 0 || data.size() % ef_width != 0) {
            throw std::invalid_argument("ExtensionMmcs::commit_matrix: bad dimensions");
        }
        size_t height = data.size() / ef_width;

        // Decompose: each EF element becomes DEGREE Val elements
        constexpr size_t DEGREE = EF::DEGREE;
        size_t val_width = ef_width * DEGREE;
        std::vector<Val> val_data;
        val_data.reserve(height * val_width);

        for (const EF& elem : data) {
            for (size_t d = 0; d < DEGREE; ++d) {
                val_data.push_back(elem.coeffs[d]);
            }
        }

        auto [commit, inner_pd] = inner_mmcs.commit_matrix(val_data, val_width);

        ProverData pd;
        pd.inner    = std::move(inner_pd);
        pd.ef_width = ef_width;

        return {commit, pd};
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    size_t log_height(const ProverData& d) const {
        return inner_mmcs.log_height(d.inner);
    }

    // Returns log2(ef_width) — the logical width of the EF matrix.
    // The FRI protocol uses this to determine the folding arity per round.
    // ef_width = inner_width / DEGREE, so log(ef_width) = log(inner_width) - log(DEGREE).
    size_t log_width(const ProverData& d) const {
        constexpr size_t DEGREE = EF::DEGREE;
        size_t inner_log_w = inner_mmcs.log_width(d.inner);
        size_t log_degree  = p3_util::log2_strict_usize(DEGREE);
        return inner_log_w - log_degree;
    }

    std::vector<EF> get_row(const ProverData& d, size_t row) const {
        constexpr size_t DEGREE = EF::DEGREE;
        auto val_row = inner_mmcs.get_row_as_vals(d.inner, row);
        size_t ef_width = d.ef_width;
        std::vector<EF> result(ef_width);
        for (size_t j = 0; j < ef_width; ++j) {
            EF elem{};
            for (size_t k = 0; k < DEGREE; ++k) {
                elem.coeffs[k] = val_row[j * DEGREE + k];
            }
            result[j] = elem;
        }
        return result;
    }

    void open_row(const ProverData& d, size_t row_index, OpeningProof& proof) const {
        inner_mmcs.open_row(d.inner, row_index, proof);
    }

    bool verify_row(const Commitment& commit,
                    size_t row_index,
                    const std::vector<EF>& row_vals,
                    const OpeningProof& proof) const
    {
        constexpr size_t DEGREE = EF::DEGREE;
        std::vector<Val> val_row;
        val_row.reserve(row_vals.size() * DEGREE);
        for (const EF& elem : row_vals) {
            for (size_t d = 0; d < DEGREE; ++d) {
                val_row.push_back(elem.coeffs[d]);
            }
        }
        return inner_mmcs.verify_row(commit, row_index, val_row, proof);
    }

    // -----------------------------------------------------------------------
    // For use as FriMmcs: observe_commitment is a no-op (challenger handles it)
    // -----------------------------------------------------------------------
    void observe_commitment(const Commitment& /*c*/) const {}

    bool verify_query(size_t, size_t,
                      const std::vector<Commitment>&,
                      const OpeningProof&,
                      const EF&) const
    {
        return true;
    }
};

} // namespace p3_merkle
