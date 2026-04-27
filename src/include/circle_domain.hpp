#pragma once

/**
 * @file circle_domain.hpp
 * @brief Circle group points and twin-coset domains.
 *
 * Mirrors the core algebra from `plonky3/circle/src/point.rs` and
 * `plonky3/circle/src/domain.rs`.
 */

#include "mersenne31.hpp"
#include "p3_util/util.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p3_circle {

inline std::size_t cfft_permute_index(std::size_t index, std::size_t log_n) {
    const std::size_t shifted = index >> 1u;
    const std::size_t lsb = index & 1u;
    const std::size_t perm_input = (lsb == 0)
        ? shifted
        : ((std::size_t{1} << log_n) - shifted - 1);
    return p3_util::reverse_bits_len(perm_input, log_n);
}

template <typename F>
struct CirclePoint;

template <typename F>
struct CircleGeneratorTraits;

template <>
struct CircleGeneratorTraits<p3_field::Mersenne31> {
    static constexpr std::size_t CIRCLE_TWO_ADICITY = 31;

    static CirclePoint<p3_field::Mersenne31> generator(std::size_t bits);
};

template <typename F>
struct CirclePoint {
    F x;
    F y;

    CirclePoint() : x(F::one_val()), y(F::zero_val()) {}
    CirclePoint(const F& x_, const F& y_) : x(x_), y(y_) {}

    static CirclePoint zero() {
        return CirclePoint(F::one_val(), F::zero_val());
    }

    static CirclePoint generator(std::size_t log_order) {
        return CircleGeneratorTraits<F>::generator(log_order);
    }

    static CirclePoint from_projective_line(const F& t) {
        const F t2 = t.square();
        const F denom = F::one_val() + t2;
        const F inv_denom = denom.inv();
        return CirclePoint((F::one_val() - t2) * inv_denom, t.double_val() * inv_denom);
    }

    std::optional<F> to_projective_line() const {
        const F denom = x + F::one_val();
        if (denom == F::zero_val()) {
            return std::nullopt;
        }
        return y * denom.inv();
    }

    CirclePoint operator-() const {
        return CirclePoint(x, F::zero_val() - y);
    }

    CirclePoint operator+(const CirclePoint& rhs) const {
        return CirclePoint(x * rhs.x - y * rhs.y, x * rhs.y + y * rhs.x);
    }

    CirclePoint operator-(const CirclePoint& rhs) const {
        return CirclePoint(x * rhs.x + y * rhs.y, y * rhs.x - x * rhs.y);
    }

    CirclePoint& operator+=(const CirclePoint& rhs) {
        *this = *this + rhs;
        return *this;
    }

    bool operator==(const CirclePoint& rhs) const {
        return x == rhs.x && y == rhs.y;
    }

    bool operator!=(const CirclePoint& rhs) const {
        return !(*this == rhs);
    }

    CirclePoint double_point() const {
        return CirclePoint(x.square().double_val() - F::one_val(), x.double_val() * y);
    }

    CirclePoint repeated_double(std::size_t n) const {
        CirclePoint out = *this;
        for (std::size_t i = 0; i < n; ++i) {
            out = out.double_point();
        }
        return out;
    }

    CirclePoint scalar_mul(std::size_t scalar) const {
        CirclePoint result = CirclePoint::zero();
        CirclePoint base = *this;
        while (scalar != 0) {
            if ((scalar & 1u) != 0) {
                result += base;
            }
            scalar >>= 1u;
            if (scalar != 0) {
                base = base.double_point();
            }
        }
        return result;
    }

    F v_n(std::size_t log_n) const {
        if (log_n == 0) {
            throw std::invalid_argument("CirclePoint::v_n requires log_n >= 1");
        }
        CirclePoint p = *this;
        for (std::size_t i = 0; i < log_n - 1; ++i) {
            p.x = p.x.square().double_val() - F::one_val();
        }
        return p.x;
    }

    F v_n_prod(std::size_t log_n) const {
        if (log_n <= 1) {
            return F::one_val();
        }
        CirclePoint p = *this;
        F out = p.x;
        for (std::size_t i = 0; i < log_n - 2; ++i) {
            p.x = p.x.square().double_val() - F::one_val();
            out *= p.x;
        }
        return out;
    }

    F v_tilde_p(const CirclePoint& at) const {
        auto t = (at - *this).to_projective_line();
        if (!t.has_value()) {
            throw std::invalid_argument("CirclePoint::v_tilde_p hit projective infinity");
        }
        return *t;
    }

    F s_p_at_p(std::size_t log_n) const {
        if (log_n == 0) {
            throw std::invalid_argument("CirclePoint::s_p_at_p requires log_n >= 1");
        }
        F pow2 = F::one_val();
        const F two = F::two_val();
        for (std::size_t i = 0; i < 2 * log_n - 1; ++i) {
            pow2 *= two;
        }
        return F::zero_val() - (v_n_prod(log_n) * pow2 * y);
    }
};

inline CirclePoint<p3_field::Mersenne31>
CircleGeneratorTraits<p3_field::Mersenne31>::generator(std::size_t bits) {
    using F = p3_field::Mersenne31;
    if (bits > CIRCLE_TWO_ADICITY) {
        throw std::invalid_argument("Mersenne31 circle generator bits exceed circle two-adicity");
    }

    CirclePoint<F> base(F(uint32_t{311014874u}), F(uint32_t{1584694829u}));
    for (std::size_t i = 0; i < CIRCLE_TWO_ADICITY - bits; ++i) {
        base = base.double_point();
    }
    return base;
}

template <typename F>
inline CirclePoint<F> operator*(const CirclePoint<F>& p, std::size_t scalar) {
    return p.scalar_mul(scalar);
}

template <typename F>
inline CirclePoint<F> operator*(std::size_t scalar, const CirclePoint<F>& p) {
    return p.scalar_mul(scalar);
}

template <typename F>
struct CircleLagrangeSelectors {
    F is_first_row;
    F is_last_row;
    F is_transition;
    F inv_vanishing;
};

template <typename F>
class CircleDomain {
public:
    std::size_t log_n;
    CirclePoint<F> shift;

    CircleDomain() : log_n(1), shift(CirclePoint<F>::generator(2)) {}
    CircleDomain(std::size_t log_n_, CirclePoint<F> shift_)
        : log_n(log_n_), shift(std::move(shift_)) {
        if (log_n == 0) {
            throw std::invalid_argument("CircleDomain requires log_n >= 1");
        }
    }

    static CircleDomain standard(std::size_t log_n) {
        return CircleDomain(log_n, CirclePoint<F>::generator(log_n + 1));
    }

    std::size_t size() const {
        return std::size_t{1} << log_n;
    }

    bool is_standard() const {
        return shift == CirclePoint<F>::generator(log_n + 1);
    }

    CirclePoint<F> generator() const {
        return subgroup_generator();
    }

    CirclePoint<F> subgroup_generator() const {
        return CirclePoint<F>::generator(log_n - 1);
    }

    CirclePoint<F> at(std::size_t index) const {
        const std::size_t idx = index >> 1u;
        if ((index & 1u) == 0) {
            return shift + subgroup_generator() * idx;
        }
        return -shift + subgroup_generator() * (idx + 1);
    }

    std::vector<CirclePoint<F>> points() const {
        std::vector<CirclePoint<F>> out;
        out.reserve(size());
        for (std::size_t i = 0; i < size(); ++i) {
            out.push_back(at(i));
        }
        return out;
    }

    F first_point() const {
        auto t = shift.to_projective_line();
        if (!t.has_value()) {
            throw std::invalid_argument("CircleDomain first point is at projective infinity");
        }
        return *t;
    }

    std::optional<F> next_point(const F& projective_point) const {
        if (!is_standard()) {
            return std::nullopt;
        }
        return (CirclePoint<F>::from_projective_line(projective_point) +
                CirclePoint<F>::generator(log_n)).to_projective_line();
    }

    CircleDomain create_disjoint_domain(std::size_t min_size) const {
        if (!is_standard()) {
            throw std::invalid_argument("create_disjoint_domain requires a standard domain");
        }
        std::size_t next_log_n = p3_util::log2_ceil_usize(min_size);
        if (next_log_n == log_n) {
            ++next_log_n;
        }
        if (next_log_n == 0) {
            next_log_n = 1;
        }
        return standard(next_log_n);
    }

    std::vector<CircleDomain> split_domains(std::size_t num_chunks) const {
        if (!is_standard()) {
            throw std::invalid_argument("split_domains requires a standard domain");
        }
        const std::size_t log_chunks = p3_util::log2_strict_usize(num_chunks);
        if (log_chunks > log_n) {
            throw std::invalid_argument("split_domains: too many chunks");
        }
        std::vector<CircleDomain> out;
        out.reserve(num_chunks);
        const std::size_t step = std::size_t{1} << (log_n - log_chunks);
        for (std::size_t i = 0; i < num_chunks; ++i) {
            out.emplace_back(log_n - log_chunks, at(i * step));
        }
        return out;
    }

    F vanishing_poly(const CirclePoint<F>& at_point) const {
        return at_point.v_n(log_n) - shift.v_n(log_n);
    }

    F s_p(const CirclePoint<F>& p, const CirclePoint<F>& at_point) const {
        return vanishing_poly(at_point) * p.v_tilde_p(at_point).inv();
    }

    F s_p_normalized(const CirclePoint<F>& p, const CirclePoint<F>& at_point) const {
        return vanishing_poly(at_point) * (p.v_tilde_p(at_point) * p.s_p_at_p(log_n)).inv();
    }

    CircleLagrangeSelectors<F> selectors_at_point(const CirclePoint<F>& point) const {
        const auto neg_shift = -shift;
        const F z = vanishing_poly(point);
        return CircleLagrangeSelectors<F>{
            s_p(shift, point),
            s_p(neg_shift, point),
            F::one_val() - s_p_normalized(neg_shift, point),
            z.inv(),
        };
    }

    std::vector<F> y_twiddles() const {
        const std::size_t half = size() >> 1u;
        std::vector<F> ys;
        ys.reserve(half);
        const auto g = subgroup_generator();
        auto p = shift;
        for (std::size_t i = 0; i < half; ++i) {
            ys.push_back(p.y);
            p += g;
        }
        p3_util::reverse_slice_index_bits(ys);
        return ys;
    }

    F nth_y_twiddle(std::size_t index) const {
        return at(cfft_permute_index(index << 1u, log_n)).y;
    }

    std::vector<F> x_twiddles(std::size_t layer) const {
        if (layer + 2 > log_n) {
            return {};
        }
        const auto generator = subgroup_generator() * (std::size_t{1} << layer);
        auto p = shift * (std::size_t{1} << layer);
        std::vector<F> xs;
        xs.reserve(std::size_t{1} << (log_n - layer - 2));
        for (std::size_t i = 0; i < (std::size_t{1} << (log_n - layer - 2)); ++i) {
            xs.push_back(p.x);
            p += generator;
        }
        p3_util::reverse_slice_index_bits(xs);
        return xs;
    }

    F nth_x_twiddle(std::size_t index) const {
        return (shift + subgroup_generator() * index).x;
    }
};

inline std::size_t forward_backward_index(std::size_t i, std::size_t len) {
    i %= 2 * len;
    return i < len ? i : 2 * len - 1 - i;
}

} // namespace p3_circle
