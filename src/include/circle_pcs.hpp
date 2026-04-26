#pragma once

/**
 * @file circle_pcs.hpp
 * @brief Minimal Circle-domain polynomial commitment scheme facade.
 *
 * The existing repository PCS tests use lightweight mock commitments.  This
 * Circle PCS follows that style: commitments are deterministic hashes of the
 * committed evaluations, openings are computed from Circle interpolation, and
 * the opening proof carries the committed data so verification can recompute
 * the same values.
 */

#include "circle_fft.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p3_circle {

template <typename Val, typename Dft = void, typename InputMmcs = void, typename FriMmcs = void>
class CirclePcs {
public:
    using Domain = CircleDomain<Val>;
    using Point = CirclePoint<Val>;
    using Matrix = p3_matrix::RowMajorMatrix<Val>;

    struct Commitment {
        std::uint64_t digest = 0;
        bool operator==(const Commitment& rhs) const { return digest == rhs.digest; }
        bool operator!=(const Commitment& rhs) const { return !(*this == rhs); }
    };

    struct PcsProverData {
        std::vector<Domain> domains;
        std::vector<Matrix> evaluations;
        std::vector<Matrix> coefficients;
    };

    struct OpeningProof {
        PcsProverData data;
    };

    using OpenedValues = std::vector<std::vector<std::vector<std::vector<Val>>>>;

    struct VerifyCommitment {
        Commitment commitment;
        std::vector<Domain> domains;
        std::vector<std::vector<Point>> points;
        std::vector<std::vector<std::vector<Val>>> opened_values;
    };

    CirclePcs() = default;

    template <typename... Args>
    explicit CirclePcs(Args&&...) {}

    Domain natural_domain_for_degree(std::size_t degree) const {
        std::size_t log_n = p3_util::log2_ceil_usize(degree);
        if (log_n == 0) {
            log_n = 1;
        }
        return Domain::standard(log_n);
    }

    std::pair<Commitment, PcsProverData>
    commit(std::vector<std::pair<Domain, Matrix>> evaluations) const {
        PcsProverData pd;
        pd.domains.reserve(evaluations.size());
        pd.evaluations.reserve(evaluations.size());
        pd.coefficients.reserve(evaluations.size());

        CircleFft<Val> fft;
        for (auto& item : evaluations) {
            const auto& domain = item.first;
            auto& evals = item.second;
            if (evals.height() != domain.size()) {
                throw std::invalid_argument("CirclePcs::commit: evaluation height does not match domain");
            }
            pd.coefficients.push_back(fft.icfft(evals, domain));
            pd.domains.push_back(domain);
            pd.evaluations.push_back(std::move(evals));
        }

        return {commitment_for(pd), std::move(pd)};
    }

    template <typename Challenger>
    std::pair<OpenedValues, OpeningProof>
    open(std::vector<std::pair<const PcsProverData*, std::vector<std::vector<Point>>>> open_data,
         Challenger& challenger) const {
        (void)challenger;
        OpenedValues all_opened;
        all_opened.reserve(open_data.size());

        for (const auto& batch : open_data) {
            const PcsProverData* pd = batch.first;
            const auto& mat_points = batch.second;
            if (pd == nullptr) {
                throw std::invalid_argument("CirclePcs::open: null prover data");
            }
            if (mat_points.size() != pd->coefficients.size()) {
                throw std::invalid_argument("CirclePcs::open: point matrix count mismatch");
            }

            std::vector<std::vector<std::vector<Val>>> batch_opened;
            batch_opened.reserve(pd->coefficients.size());
            for (std::size_t mi = 0; mi < pd->coefficients.size(); ++mi) {
                std::vector<std::vector<Val>> mat_opened;
                for (const auto& point : mat_points[mi]) {
                    mat_opened.push_back(evaluate_coefficients(pd->coefficients[mi], point, pd->domains[mi].log_n));
                }
                batch_opened.push_back(std::move(mat_opened));
            }
            all_opened.push_back(std::move(batch_opened));
        }

        OpeningProof proof;
        if (!open_data.empty() && open_data.front().first != nullptr) {
            proof.data = *open_data.front().first;
        }
        return {std::move(all_opened), std::move(proof)};
    }

    template <typename Challenger>
    bool verify(const std::vector<VerifyCommitment>& inputs,
                const OpeningProof& proof,
                Challenger& challenger) const {
        (void)challenger;
        if (inputs.empty()) {
            return false;
        }
        const Commitment recomputed = commitment_for(proof.data);
        for (const auto& input : inputs) {
            if (input.commitment != recomputed) {
                return false;
            }
            if (input.domains.size() != proof.data.domains.size() ||
                input.points.size() != proof.data.coefficients.size() ||
                input.opened_values.size() != proof.data.coefficients.size()) {
                return false;
            }
            for (std::size_t mi = 0; mi < proof.data.coefficients.size(); ++mi) {
                if (input.domains[mi].log_n != proof.data.domains[mi].log_n ||
                    input.domains[mi].shift != proof.data.domains[mi].shift ||
                    input.opened_values[mi].size() != input.points[mi].size()) {
                    return false;
                }
                for (std::size_t pi = 0; pi < input.points[mi].size(); ++pi) {
                    const auto expected = evaluate_coefficients(
                        proof.data.coefficients[mi], input.points[mi][pi], input.domains[mi].log_n);
                    if (expected != input.opened_values[mi][pi]) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

private:
    static std::vector<Val> evaluate_coefficients(const Matrix& coeffs,
                                                  const Point& point,
                                                  std::size_t log_n) {
        const auto basis = circle_basis(point, log_n);
        std::vector<Val> out(coeffs.width(), Val::zero_val());
        for (std::size_t c = 0; c < coeffs.width(); ++c) {
            Val acc = Val::zero_val();
            for (std::size_t r = 0; r < coeffs.height(); ++r) {
                acc += coeffs.get_unchecked(r, c) * basis[r];
            }
            out[c] = acc;
        }
        return out;
    }

    static Commitment commitment_for(const PcsProverData& pd) {
        std::uint64_t acc = 0xcbf29ce484222325ULL;
        mix(acc, static_cast<std::uint64_t>(pd.domains.size()));
        for (std::size_t i = 0; i < pd.domains.size(); ++i) {
            mix(acc, static_cast<std::uint64_t>(pd.domains[i].log_n));
            mix(acc, pd.domains[i].shift.x.as_canonical_u64());
            mix(acc, pd.domains[i].shift.y.as_canonical_u64());
            mix(acc, static_cast<std::uint64_t>(pd.evaluations[i].height()));
            mix(acc, static_cast<std::uint64_t>(pd.evaluations[i].width()));
            for (const auto& v : pd.evaluations[i].values) {
                mix(acc, v.as_canonical_u64());
            }
        }
        return Commitment{acc};
    }

    static void mix(std::uint64_t& acc, std::uint64_t value) {
        acc ^= value + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
    }
};

} // namespace p3_circle
