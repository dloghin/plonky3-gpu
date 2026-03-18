#pragma once

/**
 * @file extension_mmcs.hpp
 * @brief ExtensionMmcs: adapter that lifts a base-field MMCS to extension fields.
 *
 * Mirrors plonky3/commit/src/adapters/extension_mmcs.rs.
 *
 * ExtensionMmcs<F, EF, InnerMmcs> wraps an InnerMmcs (which commits to
 * base-field matrices of type RowMajorMatrix<F>) and presents a new MMCS
 * interface that operates on extension-field matrices RowMajorMatrix<EF>.
 *
 * Key operations:
 *   commit   -- decomposes each EF matrix column-wise into EF::DEGREE base-field
 *               matrices and commits them via the inner MMCS.
 *   open_batch -- opens the inner base-field matrices and reconstitutes the
 *               EF values by interleaving the D base-field coefficients.
 *   verify_batch -- verifies the inner base-field opening and checks
 *               consistency with the claimed EF opened values.
 *
 * Template requirements:
 *   F        -- base prime field (e.g. BabyBear)
 *   EF       -- extension field with:
 *                 EF::DEGREE          (size_t, extension degree D)
 *                 EF(std::array<F,D>) constructor
 *                 EF::operator[](size_t) -> F  (coefficient access)
 *   InnerMmcs -- satisfies the Mmcs concept for base field F; requires:
 *                 InnerMmcs::Commitment
 *                 InnerMmcs::ProverData
 *                 InnerMmcs::Proof
 *                 commit(vector<RowMajorMatrix<F>>) -> pair<Commitment,ProverData>
 *                 open_batch(size_t, ProverData) -> BatchOpening<F,Proof>
 *                 verify_batch(Commitment, vector<Dimensions>, size_t,
 *                              BatchOpening<F,Proof>) -> bool
 *                 get_matrices(ProverData) -> vector<const RowMajorMatrix<F>*>
 */

#include "mmcs.hpp"
#include "dense_matrix.hpp"
#include "domain.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p3_commit {

template<typename F, typename EF, typename InnerMmcs>
class ExtensionMmcs : public MmcsBase<ExtensionMmcs<F, EF, InnerMmcs>, EF> {
public:
    // -----------------------------------------------------------------------
    // Public type aliases (required by the Mmcs concept)
    // -----------------------------------------------------------------------
    using Commitment = typename InnerMmcs::Commitment;
    using ProverData = typename InnerMmcs::ProverData;
    using Proof      = typename InnerMmcs::Proof;

    // Extension degree (number of base-field coefficients per EF element)
    static constexpr size_t D = EF::DEGREE;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------
    explicit ExtensionMmcs(InnerMmcs inner) : inner_(std::move(inner)) {}

    // Allow read-only access to the wrapped inner MMCS.
    const InnerMmcs& inner() const { return inner_; }
    InnerMmcs& inner() { return inner_; }

    // -----------------------------------------------------------------------
    // decompose
    //
    // Splits one extension-field matrix M[h x w] into D base-field matrices,
    // each of size h x w.  Matrix d contains coefficient d of each EF element:
    //   result[d][r][c] = M[r][c].coeffs[d]
    // -----------------------------------------------------------------------
    static std::vector<p3_matrix::RowMajorMatrix<F>>
    decompose(const p3_matrix::RowMajorMatrix<EF>& mat) {
        size_t h = mat.height();
        size_t w = mat.width();

        std::vector<p3_matrix::RowMajorMatrix<F>> result;
        result.reserve(D);
        for (size_t d = 0; d < D; ++d) {
            result.emplace_back(h, w);
        }

        for (size_t r = 0; r < h; ++r) {
            for (size_t c = 0; c < w; ++c) {
                const EF& elem = mat.get_unchecked(r, c);
                for (size_t d = 0; d < D; ++d) {
                    result[d].set_unchecked(r, c, elem[d]);
                }
            }
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // reconstitute_row
    //
    // Given D vectors of base-field values (one per coefficient plane), each
    // of length w, reconstitutes a vector of w extension-field elements.
    //
    // base_rows[d][c] -> the d-th coefficient of the c-th column element.
    // -----------------------------------------------------------------------
    static std::vector<EF>
    reconstitute_row(const std::vector<std::vector<F>>& base_rows) {
        if (base_rows.size() != D) {
            throw std::invalid_argument(
                "reconstitute_row: expected D base-field rows");
        }
        size_t w = base_rows[0].size();
        std::vector<EF> result;
        result.reserve(w);

        for (size_t c = 0; c < w; ++c) {
            std::array<F, D> coeffs;
            for (size_t d = 0; d < D; ++d) {
                coeffs[d] = base_rows[d][c];
            }
            result.emplace_back(coeffs);
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // commit
    //
    // Accepts a batch of extension-field matrices, decomposes each into D
    // base-field matrices, and commits the full flattened list via inner_.
    //
    // For n input EF matrices the inner MMCS receives n*D base-field matrices
    // in the order: [M0_coeff0, M0_coeff1, ..., M0_coeff_{D-1},
    //                M1_coeff0, ..., M1_coeff_{D-1}, ...,
    //                M_{n-1}_coeff0, ..., M_{n-1}_coeff_{D-1}]
    // -----------------------------------------------------------------------
    std::pair<Commitment, ProverData>
    commit(std::vector<p3_matrix::RowMajorMatrix<EF>> matrices) {
        std::vector<p3_matrix::RowMajorMatrix<F>> base_matrices;
        base_matrices.reserve(matrices.size() * D);

        for (const auto& m : matrices) {
            auto decomposed = decompose(m);
            for (auto& bm : decomposed) {
                base_matrices.push_back(std::move(bm));
            }
        }
        return inner_.commit(std::move(base_matrices));
    }

    // -----------------------------------------------------------------------
    // open_batch
    //
    // Opens the inner MMCS at the given row index and reconstitutes the
    // extension-field values from the D consecutive base-field rows.
    //
    // The inner prover data contains n*D base-field matrices.  We group them
    // in blocks of D, reconstruct one EF row per block, and return a
    // BatchOpening<EF, Proof>.
    // -----------------------------------------------------------------------
    BatchOpening<EF, Proof>
    open_batch(size_t index, const ProverData& data) {
        // Open the inner base-field matrices.
        BatchOpening<F, Proof> inner_opening = inner_.open_batch(index, data);

        size_t n_base = inner_opening.opened_values.size();
        if (n_base % D != 0) {
            throw std::runtime_error(
                "ExtensionMmcs::open_batch: inner matrix count not divisible by D");
        }

        size_t n_ef = n_base / D;
        std::vector<std::vector<EF>> ef_values;
        ef_values.reserve(n_ef);

        for (size_t i = 0; i < n_ef; ++i) {
            // Gather the D base-field rows for EF matrix i.
            std::vector<std::vector<F>> base_rows(D);
            for (size_t d = 0; d < D; ++d) {
                base_rows[d] = inner_opening.opened_values[i * D + d];
            }
            ef_values.push_back(reconstitute_row(base_rows));
        }

        return BatchOpening<EF, Proof>(
            std::move(ef_values),
            std::move(inner_opening.opening_proof)
        );
    }

    // -----------------------------------------------------------------------
    // verify_batch
    //
    // Verifies an extension-field BatchOpening against a commitment.
    //
    // Procedure:
    //   1. Decompose the claimed EF opened_values back into D base-field rows
    //      per EF matrix.
    //   2. Expand the EF Dimensions list into n*D base-field Dimensions
    //      (same height, same width for each coefficient plane).
    //   3. Delegate to inner_.verify_batch with the base-field data.
    //
    // Returns true iff the inner verification succeeds.
    // -----------------------------------------------------------------------
    bool verify_batch(
        const Commitment&                commit,
        const std::vector<Dimensions>&   dims,
        size_t                           index,
        BatchOpening<EF, Proof>          opening
    ) {
        size_t n_ef = dims.size();

        // Expand Dimensions: each EF matrix contributes D base-field matrices
        // with the same width and height.
        std::vector<Dimensions> base_dims;
        base_dims.reserve(n_ef * D);
        for (const auto& d : dims) {
            for (size_t k = 0; k < D; ++k) {
                base_dims.push_back(d);
            }
        }

        // Decompose the EF opened values back to base-field rows.
        std::vector<std::vector<F>> base_values;
        base_values.reserve(n_ef * D);
        for (const auto& ef_row : opening.opened_values) {
            // ef_row[c] is the EF element at column c.
            // Produce D base-field rows, one per coefficient.
            std::vector<std::vector<F>> coeff_rows(D);
            for (size_t d = 0; d < D; ++d) {
                coeff_rows[d].reserve(ef_row.size());
            }
            for (const EF& elem : ef_row) {
                for (size_t d = 0; d < D; ++d) {
                    coeff_rows[d].push_back(elem[d]);
                }
            }
            for (size_t d = 0; d < D; ++d) {
                base_values.push_back(std::move(coeff_rows[d]));
            }
        }

        BatchOpening<F, Proof> base_opening(
            std::move(base_values),
            std::move(opening.opening_proof)
        );

        return inner_.verify_batch(commit, base_dims, index, base_opening);
    }

    // -----------------------------------------------------------------------
    // get_matrices
    //
    // Returns pointers to the inner base-field matrices (not the EF matrices,
    // since those are not stored — only the decomposed base-field form is).
    // -----------------------------------------------------------------------
    std::vector<const p3_matrix::RowMajorMatrix<F>*>
    get_matrices(const ProverData& data) const {
        return inner_.get_matrices(data);
    }

private:
    InnerMmcs inner_;
};

} // namespace p3_commit
