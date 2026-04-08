#pragma once

#include "mersenne31.hpp"
#include "radix2_dit.hpp"
#include "dense_matrix.hpp"
#include "p3_util/util.hpp"

#include <vector>
#include <cstddef>
#include <stdexcept>

namespace p3_dft {

class Mersenne31Complex {
public:
    static constexpr size_t TWO_ADICITY = 32;

    Mersenne31Complex()
        : real_(p3_field::Mersenne31::zero_val()), imag_(p3_field::Mersenne31::zero_val()) {}
    Mersenne31Complex(const p3_field::Mersenne31& real, const p3_field::Mersenne31& imag)
        : real_(real), imag_(imag) {}

    static Mersenne31Complex zero_val() {
        return Mersenne31Complex();
    }
    static Mersenne31Complex one_val() {
        return Mersenne31Complex(p3_field::Mersenne31::one_val(), p3_field::Mersenne31::zero_val());
    }
    static Mersenne31Complex new_real(const p3_field::Mersenne31& x) {
        return Mersenne31Complex(x, p3_field::Mersenne31::zero_val());
    }
    static Mersenne31Complex new_complex(const p3_field::Mersenne31& real, const p3_field::Mersenne31& imag) {
        return Mersenne31Complex(real, imag);
    }

    const p3_field::Mersenne31& real() const { return real_; }
    const p3_field::Mersenne31& imag() const { return imag_; }

    Mersenne31Complex add(const Mersenne31Complex& other) const {
        return Mersenne31Complex(real_ + other.real_, imag_ + other.imag_);
    }
    Mersenne31Complex sub(const Mersenne31Complex& other) const {
        return Mersenne31Complex(real_ - other.real_, imag_ - other.imag_);
    }
    Mersenne31Complex mul(const Mersenne31Complex& other) const {
        // (a + ib)(c + id) = (ac - bd) + i(ad + bc), with i^2 = -1.
        const auto ac = real_ * other.real_;
        const auto bd = imag_ * other.imag_;
        const auto ad = real_ * other.imag_;
        const auto bc = imag_ * other.real_;
        return Mersenne31Complex(ac - bd, ad + bc);
    }
    bool equals(const Mersenne31Complex& other) const {
        return real_ == other.real_ && imag_ == other.imag_;
    }

    Mersenne31Complex conjugate() const {
        return Mersenne31Complex(real_, p3_field::Mersenne31::zero_val() - imag_);
    }
    Mersenne31Complex halve() const {
        static const auto two_inv = p3_field::Mersenne31(1073741824u);
        return Mersenne31Complex(real_ * two_inv, imag_ * two_inv);
    }
    Mersenne31Complex inv() const {
        const auto norm = real_ * real_ + imag_ * imag_;
        const auto norm_inv = norm.inv();
        return Mersenne31Complex(real_ * norm_inv, (p3_field::Mersenne31::zero_val() - imag_) * norm_inv);
    }

    Mersenne31Complex square() const {
        const auto real_sq = (real_ - imag_) * (real_ + imag_);
        const auto imag_sq = (real_ * imag_).double_val();
        return Mersenne31Complex(real_sq, imag_sq);
    }
    Mersenne31Complex exp_u64(uint64_t power) const {
        if (power == 0) return one_val();
        Mersenne31Complex result = one_val();
        Mersenne31Complex base = *this;
        while (power > 0) {
            if (power & 1u) result = result * base;
            if (power > 1) base = base.square();
            power >>= 1;
        }
        return result;
    }
    Mersenne31Complex exp_power_of_2(size_t n) const {
        Mersenne31Complex out = *this;
        for (size_t i = 0; i < n; ++i) out = out.square();
        return out;
    }

    static Mersenne31Complex two_adic_generator(size_t bits) {
        if (bits > TWO_ADICITY) {
            throw std::invalid_argument("bits exceeds TWO_ADICITY (31) for Mersenne31Complex");
        }
        // From Plonky3's mersenne_31.rs EXT_TWO_ADIC_GENERATORS[32]:
        // a primitive 2^32-th root of unity in Complex<Mersenne31>.
        const Mersenne31Complex base(
            p3_field::Mersenne31(static_cast<uint32_t>(1166849849u)),
            p3_field::Mersenne31(static_cast<uint32_t>(1117296306u)));
        return base.exp_power_of_2(TWO_ADICITY - bits);
    }

    friend Mersenne31Complex operator+(const Mersenne31Complex& a, const Mersenne31Complex& b) {
        return a.add(b);
    }
    friend Mersenne31Complex operator-(const Mersenne31Complex& a, const Mersenne31Complex& b) {
        return a.sub(b);
    }
    friend Mersenne31Complex operator*(const Mersenne31Complex& a, const Mersenne31Complex& b) {
        return a.mul(b);
    }

private:
    p3_field::Mersenne31 real_;
    p3_field::Mersenne31 imag_;
};

class Mersenne31ComplexRadix2Dit
    : public TwoAdicSubgroupDft<Mersenne31Complex, Mersenne31ComplexRadix2Dit> {
public:
    p3_matrix::RowMajorMatrix<Mersenne31Complex> dft_batch(
        p3_matrix::RowMajorMatrix<Mersenne31Complex> mat)
    {
        return impl_.dft_batch(std::move(mat));
    }

    p3_matrix::RowMajorMatrix<Mersenne31Complex> idft_batch(
        p3_matrix::RowMajorMatrix<Mersenne31Complex> mat)
    {
        return impl_.idft_batch(std::move(mat));
    }

private:
    Radix2Dit<Mersenne31Complex> impl_;
};

class Mersenne31Dft {
public:
    template<typename Dft = Mersenne31ComplexRadix2Dit>
    static p3_matrix::RowMajorMatrix<Mersenne31Complex> dft_batch(
        const p3_matrix::RowMajorMatrix<p3_field::Mersenne31>& mat)
    {
        Dft dft;
        auto packed = dft_preprocess(mat);
        auto dft_out = dft.dft_batch(std::move(packed));
        return dft_postprocess(dft_out);
    }

    template<typename Dft = Mersenne31ComplexRadix2Dit>
    static p3_matrix::RowMajorMatrix<p3_field::Mersenne31> idft_batch(
        const p3_matrix::RowMajorMatrix<Mersenne31Complex>& mat)
    {
        Dft dft;
        auto pre = idft_preprocess(mat);
        auto idft_out = dft.idft_batch(std::move(pre));
        return idft_postprocess(idft_out);
    }

private:
    static p3_matrix::RowMajorMatrix<Mersenne31Complex> dft_preprocess(
        const p3_matrix::RowMajorMatrix<p3_field::Mersenne31>& input)
    {
        const size_t h = input.height();
        const size_t w = input.width();
        if ((h & 1u) != 0u) {
            throw std::invalid_argument("Mersenne31Dft input height must be even");
        }
        std::vector<Mersenne31Complex> out;
        out.reserve((h / 2) * w);
        for (size_t r = 0; r < h; r += 2) {
            for (size_t c = 0; c < w; ++c) {
                out.emplace_back(input.get_unchecked(r, c), input.get_unchecked(r + 1, c));
            }
        }
        return p3_matrix::RowMajorMatrix<Mersenne31Complex>(std::move(out), w);
    }

    static p3_matrix::RowMajorMatrix<Mersenne31Complex> dft_postprocess(
        const p3_matrix::RowMajorMatrix<Mersenne31Complex>& input)
    {
        const size_t h = input.height();
        const size_t w = input.width();
        const size_t log_h = p3_util::log2_strict_usize(h);
        const auto omega = Mersenne31Complex::two_adic_generator(log_h + 1);
        auto omega_j = omega;

        std::vector<Mersenne31Complex> out;
        out.reserve((h + 1) * w);

        for (size_t c = 0; c < w; ++c) {
            const auto x = input.get_unchecked(0, c);
            out.push_back(Mersenne31Complex::new_real(x.real() + x.imag()));
        }

        for (size_t j = 1; j < h; ++j) {
            for (size_t c = 0; c < w; ++c) {
                const auto x = input.get_unchecked(j, c);
                const auto y = input.get_unchecked(h - j, c);
                const auto even = x + y.conjugate();
                const auto odd = Mersenne31Complex::new_complex(
                    x.imag() + y.imag(),
                    y.real() - x.real());
                out.push_back((even + odd * omega_j).halve());
            }
            omega_j = omega_j * omega;
        }

        for (size_t c = 0; c < w; ++c) {
            const auto x = input.get_unchecked(0, c);
            out.push_back(Mersenne31Complex::new_real(x.real() - x.imag()));
        }

        return p3_matrix::RowMajorMatrix<Mersenne31Complex>(std::move(out), w);
    }

    static p3_matrix::RowMajorMatrix<Mersenne31Complex> idft_preprocess(
        const p3_matrix::RowMajorMatrix<Mersenne31Complex>& input)
    {
        const size_t h = input.height() - 1;
        const size_t w = input.width();
        const size_t log_h = p3_util::log2_strict_usize(h);
        const auto omega_inv = Mersenne31Complex::two_adic_generator(log_h + 1).inv();
        auto omega_j = Mersenne31Complex::one_val();

        std::vector<Mersenne31Complex> out;
        out.reserve(h * w);
        for (size_t j = 0; j < h; ++j) {
            for (size_t c = 0; c < w; ++c) {
                const auto x = input.get_unchecked(j, c);
                const auto y = input.get_unchecked(h - j, c);
                const auto even = x + y.conjugate();
                const auto odd = Mersenne31Complex::new_complex(
                    x.imag() + y.imag(),
                    y.real() - x.real());
                out.push_back((even - odd * omega_j).halve());
            }
            omega_j = omega_j * omega_inv;
        }
        return p3_matrix::RowMajorMatrix<Mersenne31Complex>(std::move(out), w);
    }

    static p3_matrix::RowMajorMatrix<p3_field::Mersenne31> idft_postprocess(
        const p3_matrix::RowMajorMatrix<Mersenne31Complex>& input)
    {
        const size_t h = input.height();
        const size_t w = input.width();
        std::vector<p3_field::Mersenne31> out(h * 2 * w);
        for (size_t r = 0; r < h; ++r) {
            for (size_t c = 0; c < w; ++c) {
                const auto& v = input.get_unchecked(r, c);
                out[(r * 2) * w + c] = v.real();
                out[(r * 2 + 1) * w + c] = v.imag();
            }
        }
        return p3_matrix::RowMajorMatrix<p3_field::Mersenne31>(std::move(out), w);
    }
};

} // namespace p3_dft
