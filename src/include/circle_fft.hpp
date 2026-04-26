#pragma once

/**
 * @file circle_fft.hpp
 * @brief Circle-domain evaluation and interpolation.
 *
 * This is a correctness-first port of the Circle basis used by Plonky3.  The
 * Rust implementation uses twinned butterflies; here we expose the same
 * high-level transforms and use dense basis evaluation/interpolation, which is
 * suitable for tests and small CPU examples.
 */

#include "circle_domain.hpp"
#include "dense_matrix.hpp"
#include "p3_util/util.hpp"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p3_circle {

template <typename F>
std::vector<F> circle_basis(CirclePoint<F> p, std::size_t log_n) {
    const std::size_t n = std::size_t{1} << log_n;
    std::vector<F> basis;
    basis.reserve(n);
    basis.push_back(F::one_val());
    if (n == 1) {
        return basis;
    }
    basis.push_back(p.y);
    F x = p.x;
    for (std::size_t layer = 0; layer < log_n - 1; ++layer) {
        const std::size_t cur = basis.size();
        for (std::size_t i = 0; i < cur; ++i) {
            basis.push_back(basis[i] * x);
        }
        x = x.square().double_val() - F::one_val();
    }
    return basis;
}

template <typename F>
class CircleFft {
public:
    using Matrix = p3_matrix::RowMajorMatrix<F>;
    using Domain = CircleDomain<F>;

    CircleFft() = default;
    explicit CircleFft(Domain domain) : domain_(std::move(domain)) {}

    Matrix cfft(const Matrix& coeffs) const {
        return evaluate(require_domain(), coeffs);
    }

    Matrix cfft(const Matrix& coeffs, const Domain& domain) const {
        return evaluate(domain, coeffs);
    }

    Matrix icfft(const Matrix& evals) const {
        return interpolate(require_domain(), evals);
    }

    Matrix icfft(const Matrix& evals, const Domain& domain) const {
        return interpolate(domain, evals);
    }

    Matrix coset_cfft(const Matrix& coeffs, const CirclePoint<F>& shift) const {
        const std::size_t log_n = p3_util::log2_strict_usize(coeffs.height());
        return evaluate(Domain(log_n, shift), coeffs);
    }

    static Matrix evaluate(const Domain& domain, const Matrix& coeffs) {
        const std::size_t n = domain.size();
        if (coeffs.height() > n) {
            throw std::invalid_argument("CircleFft::evaluate: coefficient height exceeds domain size");
        }
        const std::size_t width = coeffs.width();
        std::vector<F> out(n * width, F::zero_val());
        const auto pts = domain.points();
        for (std::size_t r = 0; r < n; ++r) {
            const auto basis = circle_basis(pts[r], domain.log_n);
            for (std::size_t c = 0; c < width; ++c) {
                F acc = F::zero_val();
                for (std::size_t i = 0; i < coeffs.height(); ++i) {
                    acc += coeffs.get_unchecked(i, c) * basis[i];
                }
                out[r * width + c] = acc;
            }
        }
        return Matrix(std::move(out), width);
    }

    static Matrix interpolate(const Domain& domain, const Matrix& evals) {
        const std::size_t n = domain.size();
        if (evals.height() != n) {
            throw std::invalid_argument("CircleFft::interpolate: eval height must equal domain size");
        }
        const std::size_t width = evals.width();
        auto pts = domain.points();

        std::vector<F> a(n * n);
        for (std::size_t r = 0; r < n; ++r) {
            const auto basis = circle_basis(pts[r], domain.log_n);
            for (std::size_t c = 0; c < n; ++c) {
                a[r * n + c] = basis[c];
            }
        }

        std::vector<F> rhs = evals.values;

        // Solve the Vandermonde-like system A * coeffs = evals for all columns
        // at once.  The Circle basis is linearly independent on a twin-coset.
        for (std::size_t pivot = 0; pivot < n; ++pivot) {
            std::size_t pivot_row = pivot;
            while (pivot_row < n && a[pivot_row * n + pivot] == F::zero_val()) {
                ++pivot_row;
            }
            if (pivot_row == n) {
                throw std::runtime_error("CircleFft::interpolate: singular circle basis matrix");
            }
            if (pivot_row != pivot) {
                for (std::size_t c = 0; c < n; ++c) {
                    std::swap(a[pivot * n + c], a[pivot_row * n + c]);
                }
                for (std::size_t c = 0; c < width; ++c) {
                    std::swap(rhs[pivot * width + c], rhs[pivot_row * width + c]);
                }
            }

            const F inv_pivot = a[pivot * n + pivot].inv();
            for (std::size_t c = pivot; c < n; ++c) {
                a[pivot * n + c] *= inv_pivot;
            }
            for (std::size_t c = 0; c < width; ++c) {
                rhs[pivot * width + c] *= inv_pivot;
            }

            for (std::size_t r = 0; r < n; ++r) {
                if (r == pivot) {
                    continue;
                }
                const F factor = a[r * n + pivot];
                if (factor == F::zero_val()) {
                    continue;
                }
                for (std::size_t c = pivot; c < n; ++c) {
                    a[r * n + c] -= factor * a[pivot * n + c];
                }
                for (std::size_t c = 0; c < width; ++c) {
                    rhs[r * width + c] -= factor * rhs[pivot * width + c];
                }
            }
        }

        return Matrix(std::move(rhs), width);
    }

    Matrix extrapolate(const Domain& source_domain,
                       const Matrix& source_evals,
                       const Domain& target_domain) const {
        if (target_domain.log_n < source_domain.log_n) {
            throw std::invalid_argument("CircleFft::extrapolate: target domain is smaller");
        }
        return evaluate(target_domain, interpolate(source_domain, source_evals));
    }

private:
    const Domain& require_domain() const {
        if (!domain_.has_value()) {
            throw std::invalid_argument("CircleFft method requires a domain");
        }
        return *domain_;
    }

    std::optional<Domain> domain_;
};

} // namespace p3_circle
