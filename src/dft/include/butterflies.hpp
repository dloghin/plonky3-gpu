#pragma once

#include <cstddef>

namespace p3_dft {

/**
 * @brief Apply a twiddle-free DIT butterfly to two rows in-place.
 *
 * (a[c], b[c]) -> (a[c] + b[c], a[c] - b[c]) for each column c.
 *
 * @tparam F Field element type
 * @param a  Pointer to first row
 * @param b  Pointer to second row
 * @param width  Number of columns
 */
template<typename F>
inline void twiddle_free_butterfly(F* a, F* b, size_t width) {
    for (size_t c = 0; c < width; ++c) {
        F sum = a[c] + b[c];
        F diff = a[c] - b[c];
        a[c] = sum;
        b[c] = diff;
    }
}

/**
 * @brief Apply a DIT butterfly with a twiddle factor -- safe version.
 *
 * Uses a temporary to avoid aliasing issues.
 */
template<typename F>
inline void dit_butterfly_safe(F* a, F* b, const F& twiddle, size_t width) {
    for (size_t c = 0; c < width; ++c) {
        F ai = a[c];
        F bi = twiddle * b[c];
        a[c] = ai + bi;
        b[c] = ai - bi;
    }
}

} // namespace p3_dft
