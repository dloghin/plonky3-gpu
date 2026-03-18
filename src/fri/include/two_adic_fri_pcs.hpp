#pragma once

#include "fri_params.hpp"
#include "fri_proof.hpp"
#include "fri_folding.hpp"
#include "fri_prover.hpp"
#include "fri_verifier.hpp"
#include "interpolation.hpp"
#include "dense_matrix.hpp"
#include "util.hpp"
#include "p3_util/util.hpp"

#include <vector>
#include <utility>
#include <map>
#include <functional>
#include <stdexcept>

namespace p3_fri {

// ---------------------------------------------------------------------------
// TwoAdicMultiplicativeCoset<Val>
// Represents the coset  shift * <two_adic_generator(log_n)>  of size 2^log_n.
// ---------------------------------------------------------------------------
template <typename Val>
struct TwoAdicMultiplicativeCoset {
    size_t log_n;
    Val    shift;

    size_t size() const { return size_t(1) << log_n; }
};

// ---------------------------------------------------------------------------
// TwoAdicFriPcs
//
// Template parameters:
//   Val       - base prime field (e.g. BabyBear)
//   Challenge - extension field (e.g. BabyBear4)
//   Dft       - DFT type with coset_lde_batch(mat, added_bits, shift)
//   InputMmcs - MMCS for committing LDE matrices; must expose:
//                 commit(vector<RowMajorMatrix<Val>>) -> pair<Commitment,ProverData>
//                 matrix_width(ProverData, mat_idx) -> size_t
//                 get_value(ProverData, mat_idx, row, col) -> Val
//                 open(query_index, vector<ProverData>, OpeningProof&)
//                 verify_query(query_index, log_max_height,
//                              vector<Commitment>, OpeningProof, Challenge) -> bool
//   FriMmcs   - MMCS used internally by the FRI protocol
// ---------------------------------------------------------------------------
template <
    typename Val,
    typename Challenge,
    typename Dft,
    typename InputMmcs,
    typename FriMmcs
>
class TwoAdicFriPcs {
public:
    using Coset      = TwoAdicMultiplicativeCoset<Val>;
    using Matrix     = p3_matrix::RowMajorMatrix<Val>;
    using Commitment = typename InputMmcs::Commitment;
    using ProverData = typename InputMmcs::ProverData;
    using InputProof = typename InputMmcs::OpeningProof;
    using Witness    = uint64_t;
    using Proof      = FriProof<Challenge, FriMmcs, Witness, InputProof>;
    using OpenedValues = std::vector<std::vector<std::vector<Challenge>>>;
    // OpenedValues indexed as [batch][point_idx][col]

    // OpenData: one entry per polynomial batch being opened.
    // mat_idx identifies which matrix inside the ProverData (from commit()).
    struct OpenData {
        ProverData             prover_data;
        size_t                 mat_idx;  // index of the matrix within prover_data
        Coset                  domain;
        std::vector<Challenge> points;
    };

    // CommitmentsWithPoints: used in verify().
    struct CommitmentsWithPoints {
        Commitment                              commitment;
        size_t                                  mat_idx;
        Coset                                   domain;
        std::vector<Challenge>                  points;
        std::vector<std::vector<Challenge>>     opened_values; // [point][col]
    };

    TwoAdicFriPcs(Dft dft, InputMmcs mmcs, FriParameters<FriMmcs> fri)
        : dft_(std::move(dft))
        , mmcs_(std::move(mmcs))
        , fri_(std::move(fri))
    {}

    // -----------------------------------------------------------------------
    // natural_domain_for_degree
    // Returns the coset with shift = Val::generator() and size = degree.
    // -----------------------------------------------------------------------
    Coset natural_domain_for_degree(size_t degree) const {
        return Coset{ p3_util::log2_strict_usize(degree), Val::generator() };
    }

    // -----------------------------------------------------------------------
    // commit
    // For each (domain, matrix) pair: compute LDE, bit-reverse rows.
    // Commit all LDE matrices together via mmcs_.
    // Returns (commitment, prover_data).
    // -----------------------------------------------------------------------
    std::pair<Commitment, ProverData> commit(
        std::vector<std::pair<Coset, Matrix>> evaluations)
    {
        std::vector<Matrix> lde_mats;
        lde_mats.reserve(evaluations.size());
        for (auto& [domain, mat] : evaluations) {
            Matrix lde = dft_.coset_lde_batch(std::move(mat), fri_.log_blowup, domain.shift);
            p3_matrix::reverse_matrix_index_bits(lde);
            lde_mats.push_back(std::move(lde));
        }
        return mmcs_.commit(std::move(lde_mats));
    }

    // -----------------------------------------------------------------------
    // open
    //
    // 1. Interpolate f(z_j) for each (batch b, point z_j, column col).
    // 2. Observe opened values; sample alpha.
    // 3. Build quotient vectors grouped by LDE height (in bit-reversed order).
    // 4. Call prove_fri.
    // -----------------------------------------------------------------------
    template <typename Challenger>
    std::pair<OpenedValues, Proof> open(
        const std::vector<OpenData>& open_data,
        Challenger& challenger)
    {
        size_t nb = open_data.size();
        OpenedValues opened_values(nb);

        // --- Step 1: interpolate opened values ---
        for (size_t b = 0; b < nb; ++b) {
            auto& od      = open_data[b];
            size_t log_lde = od.domain.log_n + fri_.log_blowup;
            size_t n_lde   = size_t(1) << log_lde;
            size_t nw      = mmcs_.matrix_width(od.prover_data, od.mat_idx);
            size_t np      = od.points.size();

            opened_values[b].assign(np, std::vector<Challenge>(nw, Challenge::zero_val()));

            Val omega_lde = Val::two_adic_generator(log_lde);
            // Subgroup H of size n_lde
            std::vector<Val> subgroup(n_lde);
            {
                Val c = Val::one_val();
                for (size_t i = 0; i < n_lde; ++i) { subgroup[i] = c; c = c * omega_lde; }
            }
            auto diff_invs = p3_interpolation::compute_diff_invs(subgroup, od.domain.shift);

            for (size_t col = 0; col < nw; ++col) {
                // Collect LDE evals in natural order (matrix is bit-reversed)
                std::vector<Challenge> lde_evals(n_lde);
                for (size_t i = 0; i < n_lde; ++i) {
                    size_t row_br = p3_util::reverse_bits_len(i, log_lde);
                    Val v = mmcs_.get_value(od.prover_data, od.mat_idx, row_br, col);
                    lde_evals[i] = embed_base<Val, Challenge>(v);
                }
                for (size_t j = 0; j < np; ++j) {
                    opened_values[b][j][col] =
                        p3_interpolation::interpolate_coset_with_precomputation<Val, Challenge>(
                            lde_evals, od.domain.shift, od.points[j], subgroup, diff_invs);
                }
            }
        }

        // --- Step 2: observe opened values, sample alpha ---
        for (auto& bov : opened_values)
            for (auto& pov : bov)
                for (auto& v : pov)
                    challenger.observe_challenge(v);
        Challenge alpha = challenger.sample_challenge();

        // --- Step 3: build quotient vectors ---
        // Key: q[i] = sum_{b,j,col} alpha^k * (fz - f(x_{br(i)})) / (z - x_{br(i)})
        // where row i (bit-reversed order) of the committed matrix holds f(x_{br(i)}).
        std::map<size_t, std::vector<Challenge>, std::greater<size_t>> h2q;
        size_t alpha_idx = 0;

        for (size_t b = 0; b < nb; ++b) {
            auto& od      = open_data[b];
            size_t log_lde = od.domain.log_n + fri_.log_blowup;
            size_t n_lde   = size_t(1) << log_lde;

            if (!h2q.count(log_lde))
                h2q[log_lde] = std::vector<Challenge>(n_lde, Challenge::zero_val());
            auto& q = h2q[log_lde];

            size_t nw = mmcs_.matrix_width(od.prover_data, od.mat_idx);
            Val omega = Val::two_adic_generator(log_lde);

            // Precompute x values in bit-reversed order:
            //   x_br[i] = shift * omega^{bit_rev(i, log_lde)}
            std::vector<Val> x_br(n_lde);
            for (size_t i = 0; i < n_lde; ++i) {
                size_t k = p3_util::reverse_bits_len(i, log_lde);
                x_br[i] = od.domain.shift * omega.exp_u64(static_cast<uint64_t>(k));
            }

            for (size_t j = 0; j < od.points.size(); ++j) {
                Challenge z = od.points[j];

                // Precompute (z - x_br[i])^{-1} for all i using batch inversion.
                std::vector<Challenge> z_minus_x_diffs(n_lde);
                for (size_t i = 0; i < n_lde; ++i) {
                    z_minus_x_diffs[i] = z - embed_base<Val, Challenge>(x_br[i]);
                }
                auto z_minus_x_inv = p3_interpolation::batch_multiplicative_inverse(z_minus_x_diffs);

                for (size_t col = 0; col < nw; ++col) {
                    Challenge fz = opened_values[b][j][col];
                    Challenge ak = alpha.exp_u64(static_cast<uint64_t>(alpha_idx++));

                    for (size_t i = 0; i < n_lde; ++i) {
                        // row i of bit-reversed matrix = f(x_{br(i)})
                        Val fx_val = mmcs_.get_value(od.prover_data, od.mat_idx, i, col);
                        Challenge fx = embed_base<Val, Challenge>(fx_val);
                        q[i] = q[i] + ak * (fz - fx) * z_minus_x_inv[i];
                    }
                }
            }
        }

        // --- Step 4: prepare FRI inputs (sorted descending by size via map) ---
        std::vector<std::vector<Challenge>> fri_inputs;
        fri_inputs.reserve(h2q.size());
        for (auto& [lh, qv] : h2q)
            fri_inputs.push_back(std::move(qv));

        // --- Step 5: prove FRI ---
        // Pass all prover datas so the InputMmcs can open them at query time.
        std::vector<ProverData> all_pd;
        all_pd.reserve(nb);
        for (auto& od : open_data)
            all_pd.push_back(od.prover_data);

        auto proof = prove_fri<Val, Challenge, FriMmcs, Challenger, Witness, InputMmcs, InputProof>(
            fri_, fri_inputs, challenger, all_pd, mmcs_);

        return { std::move(opened_values), std::move(proof) };
    }

    // -----------------------------------------------------------------------
    // verify
    //
    // Calls verify_fri with an eval_at_query callback that:
    //   1. Reads opened LDE row values from the InputProof.
    //   2. Reconstructs the combined quotient value using opened_values + alpha.
    // -----------------------------------------------------------------------
    template <typename Challenger>
    bool verify(
        const std::vector<CommitmentsWithPoints>& data,
        const Proof& proof,
        Challenger& challenger)
    {
        size_t nb = data.size();

        // Replay: observe opened values, sample alpha
        for (auto& d : data)
            for (auto& pov : d.opened_values)
                for (auto& v : pov)
                    challenger.observe_challenge(v);
        Challenge alpha = challenger.sample_challenge();

        // Collect input commitments
        std::vector<Commitment> input_commits;
        input_commits.reserve(nb);
        for (auto& d : data)
            input_commits.push_back(d.commitment);

        size_t log_blowup = fri_.log_blowup;

        // Build eval_at_query callback
        auto eval_fn = [&](size_t query_index, size_t log_max_height,
                           const InputProof& ip) -> Challenge
        {
            Challenge result = Challenge::zero_val();
            size_t alpha_idx = 0;

            for (size_t b = 0; b < nb; ++b) {
                auto& d        = data[b];
                size_t log_lde = d.domain.log_n + log_blowup;

                // Scale query index down to this domain's height
                size_t height_diff = log_max_height - log_lde;
                size_t local_idx   = query_index >> height_diff;

                // x = shift * omega^{bit_rev(local_idx, log_lde)}
                Val omega = Val::two_adic_generator(log_lde);
                size_t k  = p3_util::reverse_bits_len(local_idx, log_lde);
                Val x_val = d.domain.shift * omega.exp_u64(static_cast<uint64_t>(k));
                Challenge x = embed_base<Val, Challenge>(x_val);

                size_t nw = d.opened_values.empty() ? 0 : d.opened_values[0].size();

                for (size_t j = 0; j < d.points.size(); ++j) {
                    Challenge z         = d.points[j];
                    Challenge denom_inv = (z - x).inv();

                    for (size_t col = 0; col < nw; ++col) {
                        Challenge fz = d.opened_values[j][col];
                        // f(x) comes from the input proof (opened LDE row values)
                        Val fx_val = mmcs_.get_proof_value(ip, b, local_idx, col);
                        Challenge fx = embed_base<Val, Challenge>(fx_val);
                        Challenge ak = alpha.exp_u64(static_cast<uint64_t>(alpha_idx++));
                        result = result + ak * (fz - fx) * denom_inv;
                    }
                }
            }
            return result;
        };

        return verify_fri<Val, Challenge, FriMmcs, Challenger, Witness, InputMmcs, InputProof>(
            fri_, input_commits, proof, challenger, mmcs_, eval_fn);
    }

private:
    Dft                    dft_;
    InputMmcs              mmcs_;
    FriParameters<FriMmcs> fri_;
};

} // namespace p3_fri
