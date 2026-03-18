#pragma once
#include <cstddef>
#include <algorithm>

namespace p3_fri {

// Configuration for a FRI protocol instantiation.
// FriMmcs is the Merkle-based multi-commitment scheme type.
template <typename FriMmcs>
struct FriParameters {
    size_t log_blowup;             // log2 of the blowup factor
    size_t log_final_poly_len;     // log2 of the final polynomial length
    size_t max_log_arity;          // max log2 folding arity per round
    size_t num_queries;            // number of FRI queries
    size_t commit_proof_of_work_bits;  // PoW bits for commit phase
    size_t query_proof_of_work_bits;   // PoW bits for query phase
    FriMmcs mmcs;                  // the MMCS instance

    size_t blowup() const { return size_t(1) << log_blowup; }
    size_t final_poly_len() const { return size_t(1) << log_final_poly_len; }
    // The log2 height at which we stop folding:
    // final domain size = blowup * final_poly_len => log = log_blowup + log_final_poly_len
    size_t log_final_height() const { return log_blowup + log_final_poly_len; }
};

// Compute the log2 folding arity for a given FRI round.
// Ensures we commit at every input-height level.
//
// log_current_height: current log2 height before folding
// has_next_input: whether there is a next input vector to be mixed in
// next_input_log_height: log2 height of the next input vector (valid when has_next_input)
// log_final_height: log2 height at which we stop folding (= log_final_poly_len + log_blowup)
// max_log_arity: maximum log2 arity allowed per round
//
// Returns the log2 arity to use in this round.
inline size_t compute_log_arity_for_round(
    size_t log_current_height,
    bool has_next_input,
    size_t next_input_log_height,
    size_t log_final_height,
    size_t max_log_arity
) {
    // Must fold down to at least log_final_height.
    // If there's a next input to mix in, we must stop just before its level.
    size_t target;
    if (has_next_input) {
        // Must fold until height matches next_input_log_height
        target = next_input_log_height;
    } else {
        target = log_final_height;
    }

    // log_arity is how much we fold in this step
    if (log_current_height <= target) return 0;
    size_t available = log_current_height - target;
    return std::min(available, max_log_arity);
}

} // namespace p3_fri
