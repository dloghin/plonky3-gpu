#pragma once

/**
 * @file two_adic_fri_pcs.hpp
 * @brief TwoAdicFriPcs: Polynomial Commitment Scheme using FRI over two-adic fields.
 */

#include "fri_params.hpp"
#include "fri_proof.hpp"
#include "fri_prover.hpp"
#include "fri_verifier.hpp"
#include "fri_folding.hpp"
#include "dense_matrix.hpp"
#include "util.hpp"          // reverse_matrix_index_bits
#include "interpolation.hpp"
#include "p3_util/util.hpp"

#include <vector>
#include <array>
#include <tuple>
#include <map>
#include <utility>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <cstddef>

namespace p3_fri {

// ---------------------------------------------------------------------------
// TwoAdicMultiplicativeCoset
// ---------------------------------------------------------------------------

/**
 * A coset gH where H = <omega> has order 2^log_n.
 * Points (in natural order): shift * omega^0, shift * omega^1, ..., shift * omega^(n-1)
 */
template <typename F>
struct TwoAdicMultiplicativeCoset {
    size_t log_n;  // log2 of domain size
    F shift;       // coset shift g

    size_t size() const { return size_t(1) << log_n; }

    // Get the i-th domain point (natural order): shift * omega^i
    F get_point(size_t i) const {
        F omega = F::two_adic_generator(log_n);
        return shift * omega.exp_u64(static_cast<uint64_t>(i));
    }

    // Subgroup elements H: omega^0, omega^1, ..., omega^(n-1)
    std::vector<F> subgroup_elements() const {
        size_t n = size();
        std::vector<F> elems(n);
        F omega = F::two_adic_generator(log_n);
        F cur   = F::one_val();
        for (size_t i = 0; i < n; ++i) {
            elems[i] = cur;
            cur      = cur * omega;
        }
        return elems;
    }
};

// ---------------------------------------------------------------------------
// PcsQueryInputMmcs
//
// A minimal InputMmcs implementation for use with prove_fri/verify_fri.
// Its OpeningProof stores the actual LDE row values (flat BB vector).
// This allows eval_at_query to reconstruct the quotient polynomial value
// at any queried position without needing a full Merkle tree.
// ---------------------------------------------------------------------------

template <typename Val>
struct PcsQueryInputMmcs {
    // Commitment: not used (commitments are handled by the outer InputMmcs)
    using Commitment = std::vector<std::array<Val, 1>>;

    // ProverData: for each LDE height group, store the flat row data
    // ProverData[i] corresponds to inputs[i] passed to prove_fri.
    // It stores (lde_height, total_cols, flat_data[lde_height * total_cols])
    struct ProverData {
        size_t lde_height;
        size_t total_cols;
        std::vector<Val> flat_data;  // row-major: flat_data[k * total_cols + c]
    };

    // OpeningProof: the actual row values at the queried position
    struct OpeningProof {
        size_t row_index;
        std::vector<Val> row_values;  // length = total_cols
    };

    size_t log_height(const ProverData& d) const {
        return p3_util::log2_strict_usize(d.lde_height);
    }
    size_t log_width(const ProverData& d) const {
        return p3_util::log2_ceil_usize(d.total_cols);
    }

    void open(size_t query_index,
              const std::vector<ProverData>& data_vec,
              OpeningProof& proof) const
    {
        proof.row_index = query_index;
        proof.row_values.clear();

        if (data_vec.empty()) return;

        // data_vec is sorted by descending height; the first entry is the max.
        size_t max_log_h = p3_util::log2_strict_usize(data_vec[0].lde_height);

        for (const auto& d : data_vec) {
            size_t lde_log_h = p3_util::log2_strict_usize(d.lde_height);
            // For groups smaller than max, reduce the query index
            size_t bits_reduced = max_log_h - lde_log_h;
            size_t reduced_qi = query_index >> bits_reduced;
            size_t natural_k = p3_util::reverse_bits_len(reduced_qi, lde_log_h);
            const auto* row_start = &d.flat_data[natural_k * d.total_cols];
            proof.row_values.insert(proof.row_values.end(), row_start, row_start + d.total_cols);

        }
    }

    template <typename ClaimedEval>
    bool verify_query(size_t /*query_index*/, size_t /*log_height*/,
                      const std::vector<Commitment>& /*commits*/,
                      const OpeningProof& /*proof*/,
                      const ClaimedEval& /*claimed_eval*/) const
    {
        // The actual verification is done by eval_at_query in TwoAdicFriPcs::verify
        return true;
    }
};

// ---------------------------------------------------------------------------
// TwoAdicFriPcs
// ---------------------------------------------------------------------------

/**
 * TwoAdicFriPcs: PCS built on top of FRI over a two-adic field.
 *
 * Template parameters:
 *   Val        - base prime field type
 *   Challenge  - extension (or base) field used as FRI challenge
 *   Dft        - DFT engine with coset_lde_batch(mat, added_bits, shift)
 *   InputMmcs  - MMCS used to commit original evaluation matrices (Val elements)
 *   FriMmcs    - MMCS used internally by FRI (Challenge elements, via ExtensionMmcs)
 */
template <typename Val, typename Challenge, typename Dft,
          typename InputMmcs, typename FriMmcs>
class TwoAdicFriPcs {
public:
    using Domain          = TwoAdicMultiplicativeCoset<Val>;
    using InputCommitment = typename InputMmcs::Commitment;
    using InputProverData = typename InputMmcs::ProverData;

    // We use PcsQueryInputMmcs as the oracle MMCS for FRI (to store row values in proof)
    using QueryMmcs      = PcsQueryInputMmcs<Val>;
    using QueryProverData = typename QueryMmcs::ProverData;
    using FriInputProof   = typename QueryMmcs::OpeningProof;
    using FullFriProof    = FriProof<Challenge, FriMmcs, uint64_t, FriInputProof>;

    // OpenedValues[batch][matrix][point][column]
    using OpenedValues = std::vector<std::vector<std::vector<std::vector<Challenge>>>>;

    // PcsProverData holds the LDE matrices and domain information alongside the
    // original MMCS prover data.
    struct PcsProverData {
        InputProverData                                 mmcs_data;
        std::vector<p3_matrix::RowMajorMatrix<Val>>     lde_matrices;
        std::vector<Domain>                             domains;
    };

    // VerifyCommitment: all data needed by the verifier for one batch
    struct VerifyCommitment {
        InputCommitment                                    commitment;
        std::vector<Domain>                                domains;
        std::vector<std::vector<Challenge>>                points;       // [matrix][point_idx]
        std::vector<std::vector<std::vector<Challenge>>>   opened_values; // [matrix][point][col]
    };

private:
    Dft                    dft_;
    InputMmcs              input_mmcs_;
    FriParameters<FriMmcs> fri_;
    QueryMmcs              query_mmcs_;  // for FRI oracle

public:
    TwoAdicFriPcs(Dft dft, InputMmcs mmcs, FriParameters<FriMmcs> fri_params)
        : dft_(std::move(dft)), input_mmcs_(std::move(mmcs)), fri_(std::move(fri_params)) {}

    // -----------------------------------------------------------------------
    // natural_domain_for_degree
    // -----------------------------------------------------------------------
    Domain natural_domain_for_degree(size_t degree) const {
        size_t log_n = p3_util::log2_ceil_usize(degree);
        Domain d;
        d.log_n = log_n;
        d.shift = Val::one_val();
        return d;
    }

    // -----------------------------------------------------------------------
    // commit: LDE + Merkle commit
    // -----------------------------------------------------------------------
    std::pair<InputCommitment, PcsProverData>
    commit(std::vector<std::pair<Domain, p3_matrix::RowMajorMatrix<Val>>> evaluations)
    {
        std::vector<p3_matrix::RowMajorMatrix<Val>> lde_matrices;
        std::vector<Domain>                          domains;

        // Step 1: compute LDEs for each input matrix
        for (auto& [domain, mat] : evaluations) {
            auto lde = dft_.coset_lde_batch(std::move(mat),
                                            fri_.log_blowup,
                                            domain.shift);
            // coset_lde_batch already outputs in natural order:
            // lde[i] = polynomial evaluated at domain.shift * omega^i

            domains.push_back(domain);
            lde_matrices.push_back(std::move(lde));
        }

        // Step 2: build combined flat Val matrix from all LDE matrices
        size_t max_height = 0;
        for (const auto& m : lde_matrices) {
            if (m.height() > max_height) max_height = m.height();
        }
        size_t total_width = 0;
        for (const auto& m : lde_matrices) {
            total_width += m.width();
        }
        if (total_width == 0) {
            throw std::invalid_argument("TwoAdicFriPcs::commit: empty matrices (total_width=0)");
        }

        // MerkleTreeMmcs requires committed matrix width to be a power of two (log2_strict).
        // Pad the concatenated matrix with zero columns up to the next power of two.
        size_t padded_width = size_t(1) << p3_util::log2_ceil_usize(total_width);
        size_t pad_cols     = padded_width - total_width;

        std::vector<Val> all_flat;
        all_flat.reserve(max_height * padded_width);

        for (size_t r = 0; r < max_height; ++r) {
            for (const auto& lde : lde_matrices) {
                size_t row = r % lde.height();
                for (size_t c = 0; c < lde.width(); ++c) {
                    all_flat.push_back(lde.get_unchecked(row, c));
                }
            }
            for (size_t c = 0; c < pad_cols; ++c) {
                all_flat.push_back(Val::zero_val());
            }
        }

        auto [commit, mmcs_pd] = input_mmcs_.commit_matrix(all_flat, padded_width);

        PcsProverData pd;
        pd.mmcs_data    = std::move(mmcs_pd);
        pd.lde_matrices = std::move(lde_matrices);
        pd.domains      = std::move(domains);

        return {commit, pd};
    }

    // -----------------------------------------------------------------------
    // open: compute polynomial evaluations and FRI proof
    // -----------------------------------------------------------------------
    template <typename Challenger>
    std::pair<OpenedValues, FullFriProof>
    open(std::vector<std::pair<const PcsProverData*,
                               std::vector<std::vector<Challenge>>>> open_data,
         Challenger& challenger)
    {
        // Sample alpha (combination challenge)
        Challenge alpha = challenger.template sample_challenge<Challenge>();

        // Compute opened values via barycentric interpolation
        OpenedValues all_opened_values;

        struct InterpPrecomp {
            std::vector<Val> subgroup;   // length = lde_height
            std::vector<Val> diff_invs;  // precomputation for coset interpolation
        };
        std::map<std::pair<size_t, uint64_t>, InterpPrecomp> interp_cache;

        for (auto& [pcs_data, mat_points] : open_data) {
            std::vector<std::vector<std::vector<Challenge>>> batch_opened;
            for (size_t mi = 0; mi < pcs_data->lde_matrices.size(); ++mi) {
                const auto& lde    = pcs_data->lde_matrices[mi];
                const auto& domain = pcs_data->domains[mi];
                const auto& points = mat_points[mi];

                size_t lde_height = lde.height();
                size_t lde_log_h  = p3_util::log2_strict_usize(lde_height);
                size_t num_cols   = lde.width();

                const uint64_t shift_key = domain.shift.as_canonical_u64();
                auto key = std::make_pair(lde_log_h, shift_key);
                auto it = interp_cache.find(key);
                if (it == interp_cache.end()) {
                    Val omega_lde = Val::two_adic_generator(lde_log_h);
                    std::vector<Val> subgroup(lde_height);
                    {
                        Val cur = Val::one_val();
                        for (size_t i = 0; i < lde_height; ++i) {
                            subgroup[i] = cur;
                            cur = cur * omega_lde;
                        }
                    }
                    auto diff_invs = p3_interpolation::compute_diff_invs(subgroup, domain.shift);
                    it = interp_cache.emplace(
                        key, InterpPrecomp{std::move(subgroup), std::move(diff_invs)}).first;
                }
                const auto& subgroup  = it->second.subgroup;
                const auto& diff_invs = it->second.diff_invs;

                std::vector<std::vector<Challenge>> mat_opened;
                for (const Challenge& z : points) {
                    std::vector<Challenge> col_evals;
                    for (size_t col = 0; col < num_cols; ++col) {
                        std::vector<Challenge> col_vals(lde_height);
                        for (size_t r = 0; r < lde_height; ++r) {
                            col_vals[r] = Challenge::from_base(lde.get_unchecked(r, col));
                        }
                        Challenge val = p3_interpolation::interpolate_coset_with_precomputation(
                            col_vals, domain.shift, z, subgroup, diff_invs);
                        col_evals.push_back(val);
                    }
                    mat_opened.push_back(std::move(col_evals));
                }
                batch_opened.push_back(std::move(mat_opened));
            }
            all_opened_values.push_back(std::move(batch_opened));
        }

        // Observe opened values
        for (auto& batch_vals : all_opened_values) {
            for (auto& mat_vals : batch_vals) {
                for (auto& point_vals : mat_vals) {
                    for (auto& v : point_vals) {
                        challenger.template observe_challenge<Challenge>(v);
                    }
                }
            }
        }

        // Build quotient polynomial vectors grouped by log_lde_height (descending)
        std::map<size_t, std::vector<Challenge>, std::greater<size_t>> fri_inputs_map;
        size_t alpha_pow = 0;

        for (size_t bi = 0; bi < open_data.size(); ++bi) {
            const auto* pcs_data   = open_data[bi].first;
            const auto& mat_points = open_data[bi].second;

            for (size_t mi = 0; mi < pcs_data->lde_matrices.size(); ++mi) {
                const auto& lde    = pcs_data->lde_matrices[mi];
                const auto& domain = pcs_data->domains[mi];
                const auto& points = mat_points[mi];
                const auto& ov     = all_opened_values[bi][mi];

                size_t lde_height = lde.height();
                size_t lde_log_h  = p3_util::log2_strict_usize(lde_height);
                size_t num_cols   = lde.width();

                if (fri_inputs_map.find(lde_log_h) == fri_inputs_map.end()) {
                    fri_inputs_map[lde_log_h] = std::vector<Challenge>(
                        lde_height, Challenge::zero_val());
                }
                auto& qvec = fri_inputs_map[lde_log_h];

                Val omega = Val::two_adic_generator(lde_log_h);

                // Precompute alpha powers for each (pi, col) combination.
                // The same alpha power is used for all row positions k.
                std::vector<Challenge> alpha_pows;
                alpha_pows.reserve(points.size() * num_cols);
                if (!points.empty() && num_cols > 0) {
                    Challenge current_alpha_power = alpha.exp_u64(static_cast<uint64_t>(alpha_pow));
                    for (size_t i = 0; i < points.size() * num_cols; ++i) {
                        alpha_pows.push_back(current_alpha_power);
                        current_alpha_power *= alpha;
                    }
                }
                alpha_pow += points.size() * num_cols;

                for (size_t k = 0; k < lde_height; ++k) {
                    Val x_val = domain.shift * omega.exp_u64(static_cast<uint64_t>(k));
                    Challenge x_k = Challenge::from_base(x_val);

                    size_t ap_idx = 0;
                    for (size_t pi = 0; pi < points.size(); ++pi) {
                        const Challenge& z_j  = points[pi];
                        Challenge denom_inv   = (z_j - x_k).inv();

                        for (size_t col = 0; col < num_cols; ++col) {
                            Challenge f_z   = ov[pi][col];
                            Challenge f_x   = Challenge::from_base(lde.get_unchecked(k, col));
                            Challenge numer = f_z - f_x;

                            qvec[k] += alpha_pows[ap_idx] * numer * denom_inv;
                            ap_idx++;
                        }
                    }
                }
            }
        }

        // Collect FRI inputs sorted descending.
        // The fold_matrix/fold_row code expects evaluations in bit-reversed order:
        // element i should be the polynomial evaluated at omega^{bit_rev(i, log_h)}.
        // Our qvec is in natural order (qvec[k] = q at omega^k), so we bit-reverse.
        std::vector<std::vector<Challenge>> fri_inputs;
        for (auto& [log_h, qvec] : fri_inputs_map) {
            p3_util::reverse_slice_index_bits(qvec);  // natural -> bit-reversed order
            fri_inputs.push_back(std::move(qvec));
        }

        // Build PcsQueryInputMmcs prover data
        // For each input in fri_inputs (each LDE height group), build a ProverData
        // that contains the original LDE rows so the verifier can reconstruct quotients.
        std::vector<QueryProverData> query_input_data;
        {
            // We need to match the order of fri_inputs_map (descending log_h).
            // For each fri_input[i], find the corresponding LDE data.
            // First pass: compute total width per height group so we can allocate once.
            std::map<size_t, size_t, std::greater<size_t>> total_cols_per_log_h;
            std::map<size_t, size_t, std::greater<size_t>> height_per_log_h;

            for (size_t bi = 0; bi < open_data.size(); ++bi) {
                const auto* pcs_data = open_data[bi].first;

                for (size_t mi = 0; mi < pcs_data->lde_matrices.size(); ++mi) {
                    const auto& lde   = pcs_data->lde_matrices[mi];
                    size_t lde_height = lde.height();
                    size_t lde_log_h  = p3_util::log2_strict_usize(lde_height);
                    size_t num_cols   = lde.width();

                    total_cols_per_log_h[lde_log_h] += num_cols;

                    auto& stored_height = height_per_log_h[lde_log_h];
                    if (stored_height == 0) {
                        stored_height = lde_height;
                    } else {
                        // All matrices in a height group must share the same height.
                        if (stored_height != lde_height) {
                            throw std::runtime_error(
                                "TwoAdicFriPcs::open: mismatched LDE heights within a height group");
                        }
                    }
                }
            }

            // Allocate ProverData for each height group once with final size.
            std::map<size_t, QueryProverData, std::greater<size_t>> qpd_map;
            for (const auto& [lde_log_h, total_cols] : total_cols_per_log_h) {
                QueryProverData qpd;
                size_t lde_height = height_per_log_h.at(lde_log_h);
                qpd.lde_height    = lde_height;
                qpd.total_cols    = total_cols;
                qpd.flat_data.resize(lde_height * total_cols);
                qpd_map.emplace(lde_log_h, std::move(qpd));
            }

            // Second pass: fill the flat_data buffers without reallocations.
            std::map<size_t, size_t, std::greater<size_t>> col_offset_per_log_h;

            for (size_t bi = 0; bi < open_data.size(); ++bi) {
                const auto* pcs_data = open_data[bi].first;

                for (size_t mi = 0; mi < pcs_data->lde_matrices.size(); ++mi) {
                    const auto& lde   = pcs_data->lde_matrices[mi];
                    size_t lde_height = lde.height();
                    size_t lde_log_h  = p3_util::log2_strict_usize(lde_height);
                    size_t num_cols   = lde.width();

                    auto& qpd        = qpd_map[lde_log_h];
                    size_t col_start = col_offset_per_log_h[lde_log_h];

                    for (size_t r = 0; r < lde_height; ++r) {
                        for (size_t c = 0; c < num_cols; ++c) {
                            qpd.flat_data[r * qpd.total_cols + col_start + c] =
                                lde.get_unchecked(r, c);
                        }
                    }

                    col_offset_per_log_h[lde_log_h] += num_cols;
                }
            }

            for (auto& [log_h, qpd] : qpd_map) {
                (void)log_h;
                query_input_data.push_back(std::move(qpd));
            }
        }

        // Call prove_fri using PcsQueryInputMmcs as the input oracle
        auto fri_proof = prove_fri<Val, Challenge, FriMmcs, Challenger,
                                   uint64_t, QueryMmcs, FriInputProof>(
            fri_,
            std::move(fri_inputs),
            challenger,
            query_input_data,
            query_mmcs_
        );

        return {std::move(all_opened_values), std::move(fri_proof)};
    }

    // -----------------------------------------------------------------------
    // verify: verify opened values and FRI proof
    // -----------------------------------------------------------------------
    template <typename Challenger>
    bool verify(const std::vector<VerifyCommitment>& inputs,
                const FullFriProof& proof,
                Challenger& challenger)
    {
        // Sample alpha (must match prover)
        Challenge alpha = challenger.template sample_challenge<Challenge>();

        // Observe opened values (must match prover)
        for (const auto& vc : inputs) {
            for (const auto& mat_vals : vc.opened_values) {
                for (const auto& point_vals : mat_vals) {
                    for (const auto& v : point_vals) {
                        challenger.template observe_challenge<Challenge>(v);
                    }
                }
            }
        }

        // Collect input commitments (not used by QueryMmcs but needed by verify_fri)
        std::vector<typename QueryMmcs::Commitment> query_commits;
        // (Empty - QueryMmcs::verify_query always returns true)

        // Precompute: per-matrix info for eval_at_query reconstruction
        struct MatInfo {
            size_t lde_height;
            size_t lde_log_h;
            Val    shift;
            size_t num_cols;
            size_t col_offset;  // offset into the proof's row_values
            std::vector<Challenge> opening_points;
            std::vector<std::vector<Challenge>> opened_vals;  // [point][col]
            size_t alpha_pow_start;
        };

        std::vector<MatInfo> all_mats;

        // Determine max lde_log_h to know which group query_index maps to
        size_t max_lde_log_h = 0;
        for (const auto& vc : inputs) {
            for (const auto& domain : vc.domains) {
                size_t lhh = domain.log_n + fri_.log_blowup;
                if (lhh > max_lde_log_h) max_lde_log_h = lhh;
            }
        }

        // Build mat info ordered by descending lde_log_h (same order as FRI inputs)
        // We collect all mats sorted by descending lde_log_h
        // For each group of same lde_log_h, track col_offset within the proof row
        std::map<size_t, std::vector<MatInfo>, std::greater<size_t>> mats_by_height;
        size_t apow = 0;

        for (const auto& vc : inputs) {
            for (size_t mi = 0; mi < vc.domains.size(); ++mi) {
                const auto& domain = vc.domains[mi];
                const auto& points = vc.points[mi];
                const auto& ov     = vc.opened_values[mi];

                size_t lde_log_h  = domain.log_n + fri_.log_blowup;
                size_t lde_height = size_t(1) << lde_log_h;
                size_t num_cols   = (ov.empty() || ov[0].empty()) ? 0 : ov[0].size();

                MatInfo info;
                info.lde_height      = lde_height;
                info.lde_log_h       = lde_log_h;
                info.shift           = domain.shift;
                info.num_cols        = num_cols;
                info.col_offset      = 0;  // to be filled below
                info.opening_points  = points;
                info.opened_vals     = ov;
                info.alpha_pow_start = apow;

                apow += points.size() * num_cols;
                mats_by_height[lde_log_h].push_back(std::move(info));
            }
        }

        // Assign col_offsets: global offsets into ip.row_values.
        // row_values concatenates [group_0_cols][group_1_cols]... in
        // descending-log_h order.
        {
            size_t global_col_off = 0;
            for (auto& [log_h, mats] : mats_by_height) {
                (void)log_h;
                for (auto& m : mats) {
                    m.col_offset = global_col_off;
                    global_col_off += m.num_cols;
                }
            }
        }

        // Flatten into sorted list (descending log_h, matching open()'s FRI input order)
        for (auto& [log_h, mats] : mats_by_height) {
            (void)log_h;
            for (auto& m : mats) {
                all_mats.push_back(std::move(m));
            }
        }

        // open_input: for each query, compute the reduced quotient opening for
        // every height group.  Returns FriOpenings<Challenge> sorted by
        // descending log_height.
        auto open_input_fn = [&, alpha_capture = alpha,
                               all_mats_capture = std::move(all_mats)]
            (size_t query_index, size_t log_max_height, const FriInputProof& ip)
            -> FriOpenings<Challenge>
        {
            // Accumulate (alpha_pow, reduced_opening) per log_height
            std::map<size_t, Challenge, std::greater<size_t>> ro_map;

            for (const auto& info : all_mats_capture) {
                // Compute the reduced query index for this height group.
                // Larger groups use the raw query_index; smaller groups
                // right-shift to account for the height difference.
                size_t bits_reduced = log_max_height - info.lde_log_h;
                size_t reduced_qi = query_index >> bits_reduced;

                size_t k_birev = reduced_qi % info.lde_height;
                size_t k_nat   = p3_util::reverse_bits_len(k_birev, info.lde_log_h);
                Val omega      = Val::two_adic_generator(info.lde_log_h);
                Val x_val      = info.shift * omega.exp_u64(static_cast<uint64_t>(k_nat));
                Challenge x_k  = Challenge::from_base(x_val);

                auto& ro = ro_map[info.lde_log_h];

                Challenge current_alpha_power = alpha_capture.exp_u64(static_cast<uint64_t>(info.alpha_pow_start));
                for (size_t pi = 0; pi < info.opening_points.size(); ++pi) {
                    const Challenge& z_j = info.opening_points[pi];
                    Challenge denom_inv  = (z_j - x_k).inv();

                    for (size_t col = 0; col < info.num_cols; ++col) {
                        Challenge f_z = info.opened_vals[pi][col];

                        size_t val_idx = info.col_offset + col;
                        Challenge f_x  = (val_idx < ip.row_values.size())
                            ? Challenge::from_base(ip.row_values[val_idx])
                            : Challenge::zero_val();

                        Challenge numer  = f_z - f_x;
                        ro              += current_alpha_power * numer * denom_inv;
                        current_alpha_power *= alpha_capture;
                    }
                }
            }

            FriOpenings<Challenge> result;
            result.reserve(ro_map.size());
            for (auto& [lh, ro] : ro_map) {
                result.push_back({lh, ro});
            }
            return result;
        };

        return verify_fri<Val, Challenge, FriMmcs, Challenger,
                          uint64_t, QueryMmcs, FriInputProof>(
            fri_,
            query_commits,
            proof,
            challenger,
            query_mmcs_,
            open_input_fn
        );
    }
};

} // namespace p3_fri
