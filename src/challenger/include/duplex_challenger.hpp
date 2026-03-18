#pragma once

/**
 * @file duplex_challenger.hpp
 * @brief DuplexChallenger: duplex-sponge Fiat-Shamir transcript.
 *
 * Mirrors plonky3/challenger/src/duplex_challenger.rs and
 * plonky3/challenger/src/grinding_challenger.rs.
 *
 * Template: DuplexChallenger<F, Perm, WIDTH, RATE>
 *   F     – base field element type (must provide as_canonical_u64(), FIELD_BITS)
 *   Perm  – permutation satisfying void permute_mut(std::array<F, WIDTH>&)
 *   WIDTH – sponge state width
 *   RATE  – number of field elements absorbed per permutation call
 *
 * FRI test configuration:
 *   DuplexChallenger<BabyBear, Poseidon2BabyBear<16>, 16, 8>
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cassert>

namespace p3_challenger {

/**
 * @brief Duplex sponge challenger (Fiat-Shamir transcript).
 *
 * State machine:
 *   observe(v) : push v onto input_buffer; if len == RATE, call duplexing().
 *   sample()   : if output_buffer empty, call duplexing(); pop & return back.
 *
 * duplexing():
 *   1. Copy input_buffer -> sponge_state[0..len]
 *   2. Clear input_buffer
 *   3. Apply permutation to sponge_state
 *   4. Set output_buffer = sponge_state[0..RATE]
 */
template <typename F, typename Perm, size_t WIDTH, size_t RATE>
class DuplexChallenger {
    static_assert(RATE <= WIDTH, "RATE must be <= WIDTH");

    Perm permutation_;
    std::array<F, WIDTH> sponge_state_;  // initially all zeros
    std::vector<F> input_buffer_;        // pending observations (max RATE)
    std::vector<F> output_buffer_;       // available samples

    void absorb_without_invalidation(F value) {
        input_buffer_.push_back(value);
        if (input_buffer_.size() == RATE) {
            duplexing();
        }
    }

    void observe_witness(uint64_t witness) {
        output_buffer_.clear();
        absorb_without_invalidation(F(static_cast<uint32_t>(witness & 0xFFFFFFFFu)));
        absorb_without_invalidation(F(static_cast<uint32_t>(witness >> 32)));
    }

    void duplexing() {
        // 1. Copy input_buffer into sponge_state[0..len]
        std::copy(input_buffer_.begin(), input_buffer_.end(), sponge_state_.begin());
        // 2. Clear input_buffer
        input_buffer_.clear();
        // 3. Apply permutation
        permutation_.permute_mut(sponge_state_);
        // 4. Set output_buffer = sponge_state[0..RATE]
        output_buffer_.assign(sponge_state_.begin(), sponge_state_.begin() + RATE);
    }

public:
    explicit DuplexChallenger(Perm perm)
        : permutation_(std::move(perm)), sponge_state_{}, input_buffer_(), output_buffer_() {
        input_buffer_.reserve(RATE);
    }

    // Copy constructor (needed for grinding clone)
    DuplexChallenger(const DuplexChallenger&) = default;
    DuplexChallenger& operator=(const DuplexChallenger&) = default;

    /**
     * @brief Observe (absorb) a single field element.
     *
     * Pushes value onto input_buffer. If input_buffer reaches RATE, calls duplexing().
     */
    void observe(F value) {
        // Observing clears the output_buffer (invalidates pending samples)
        output_buffer_.clear();
        absorb_without_invalidation(value);
    }

    /**
     * @brief Observe a slice of field elements.
     */
    void observe_slice(const std::vector<F>& values) {
        if (values.empty()) {
            return;
        }
        // Bulk absorb: clear once, then absorb each value.
        output_buffer_.clear();
        for (const F& v : values) {
            absorb_without_invalidation(v);
        }
    }

    /**
     * @brief Sample a single field element.
     *
     * If output_buffer is empty, calls duplexing() first.
     * Returns and removes the last element of output_buffer.
     */
    F sample() {
        // If we have buffered inputs, we must perform a duplexing so that the
        // challenge will reflect them. Or if we've run out of outputs, we must
        // perform a duplexing to get more.
        if (!input_buffer_.empty() || output_buffer_.empty()) {
            duplexing();
        }
        F val = output_buffer_.back();
        output_buffer_.pop_back();
        return val;
    }

    /**
     * @brief Sample `bits` random bits, returned as size_t.
     *
     * Samples enough field elements to cover `bits` bits.
     * F must provide: uint64_t as_canonical_u64() const  and  static constexpr size_t FIELD_BITS.
     */
    size_t sample_bits(size_t bits) {
        assert(bits <= sizeof(size_t) * 8);
        size_t result = 0;
        size_t bits_remaining = bits;
        while (bits_remaining > 0) {
            F val = sample();
            size_t chunk = bits_remaining < F::FIELD_BITS ? bits_remaining : F::FIELD_BITS;
            uint64_t mask = (chunk == 64) ? ~uint64_t(0) : ((uint64_t(1) << chunk) - 1u);
            result |= static_cast<size_t>(val.as_canonical_u64() & mask) << (bits - bits_remaining);
            bits_remaining -= chunk;
        }
        return result;
    }

    // ------------------------------------------------------------------
    // FieldChallenger: extension-field methods
    // ------------------------------------------------------------------

    /**
     * @brief Sample an extension field element by sampling D base field elements.
     *
     * EF must provide: static constexpr size_t DEGREE  and  std::array<F,D> coeffs.
     */
    template <typename EF>
    EF sample_algebra_element() {
        EF result{};
        for (size_t i = 0; i < EF::DEGREE; ++i) {
            result.coeffs[i] = sample();
        }
        return result;
    }

    /**
     * @brief Observe an extension field element by observing its D base coefficients.
     */
    template <typename EF>
    void observe_algebra_element(const EF& value) {
        output_buffer_.clear();
        for (size_t i = 0; i < EF::DEGREE; ++i) {
            absorb_without_invalidation(value.coeffs[i]);
        }
    }

    /**
     * @brief Observe a slice of extension field elements.
     */
    template <typename EF>
    void observe_algebra_slice(const std::vector<EF>& values) {
        if (values.empty()) {
            return;
        }
        output_buffer_.clear();
        for (const EF& v : values) {
            for (size_t i = 0; i < EF::DEGREE; ++i) {
                absorb_without_invalidation(v.coeffs[i]);
            }
        }
    }

    // ------------------------------------------------------------------
    // CanObserve for MerkleCap
    // ------------------------------------------------------------------

    /**
     * @brief Observe a MerkleCap (vector of hash digests).
     *
     * Each digest is an std::array<F, N>; we observe every element of each hash.
     *
     * MerkleCap<Hash> is std::vector<Hash> where Hash = std::array<F, N>.
     */
    template <typename Hash>
    void observe_merkle_cap(const std::vector<Hash>& cap) {
        if (cap.empty()) {
            return;
        }
        output_buffer_.clear();
        for (const Hash& h : cap) {
            for (const F& elem : h) {
                absorb_without_invalidation(elem);
            }
        }
    }

    // ------------------------------------------------------------------
    // GrindingChallenger: proof-of-work
    // ------------------------------------------------------------------

    /**
     * @brief Brute-force search for a witness such that hashing
     *        (state || witness) yields a sampled element whose canonical
     *        representation has `bits` trailing zero bits.
     *
     * Algorithm:
     *   1. Clone current state.
     *   2. For witness = 0, 1, 2, ...:
     *        - Clone saved state.
     *        - Observe witness.
     *        - Sample an element.
     *        - If low `bits` bits are zero → observe witness in real state, return.
     */
    uint64_t grind(size_t bits) {
        DuplexChallenger saved = *this;
        for (uint64_t witness = 0; ; ++witness) {
            DuplexChallenger attempt = saved;
            attempt.observe_witness(witness);
            F elem = attempt.sample();
            uint64_t val = elem.as_canonical_u64();
            uint64_t mask = (bits == 64) ? ~uint64_t(0) : ((uint64_t(1) << bits) - 1u);
            if ((val & mask) == 0u) {
                // Accept: record witness in real challenger and consume
                // the verification sample (so state matches check_witness)
                observe_witness(witness);
                sample();  // consume one output element, matching check_witness behavior
                return witness;
            }
        }
    }

    /**
     * @brief Verify a proof-of-work witness.
     *
     * Observes the witness then samples an element; returns true iff
     * the element's low `bits` bits are all zero.
     */
    bool check_witness(size_t bits, uint64_t witness) {
        observe_witness(witness);
        F elem = sample();
        uint64_t val = elem.as_canonical_u64();
        uint64_t mask = (bits == 64) ? ~uint64_t(0) : ((uint64_t(1) << bits) - 1u);
        return (val & mask) == 0u;
    }
};

} // namespace p3_challenger
