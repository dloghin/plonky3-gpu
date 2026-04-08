#pragma once

#include "traits.hpp"
#include "radix2_dit.hpp"
#include "dense_matrix.hpp"

#include <thread>
#include <vector>
#include <algorithm>
#include <utility>

namespace p3_dft {

template<typename F>
class Radix2DitParallel : public TwoAdicSubgroupDft<F, Radix2DitParallel<F>> {
public:
    explicit Radix2DitParallel(size_t thread_count = 0)
        : thread_count_(thread_count == 0 ? default_thread_count() : thread_count) {}

    p3_matrix::RowMajorMatrix<F> dft_batch(p3_matrix::RowMajorMatrix<F> mat) {
        return run_column_parallel(std::move(mat), [](p3_matrix::RowMajorMatrix<F> col_mat) {
            Radix2Dit<F> dit;
            return dit.dft_batch(std::move(col_mat));
        });
    }

    p3_matrix::RowMajorMatrix<F> idft_batch(p3_matrix::RowMajorMatrix<F> mat) {
        return run_column_parallel(std::move(mat), [](p3_matrix::RowMajorMatrix<F> col_mat) {
            Radix2Dit<F> dit;
            return dit.idft_batch(std::move(col_mat));
        });
    }

    p3_matrix::RowMajorMatrix<F> coset_dft_batch(
        p3_matrix::RowMajorMatrix<F> mat, const F& shift)
    {
        return run_column_parallel(std::move(mat), [shift](p3_matrix::RowMajorMatrix<F> col_mat) {
            Radix2Dit<F> dit;
            return dit.coset_dft_batch(std::move(col_mat), shift);
        });
    }

    p3_matrix::RowMajorMatrix<F> coset_idft_batch(
        p3_matrix::RowMajorMatrix<F> mat, const F& shift)
    {
        return run_column_parallel(std::move(mat), [shift](p3_matrix::RowMajorMatrix<F> col_mat) {
            Radix2Dit<F> dit;
            return dit.coset_idft_batch(std::move(col_mat), shift);
        });
    }

private:
    size_t thread_count_;

    static size_t default_thread_count() {
        const unsigned hc = std::thread::hardware_concurrency();
        return hc == 0 ? 1u : static_cast<size_t>(hc);
    }

    template<typename TransformFn>
    p3_matrix::RowMajorMatrix<F> run_column_parallel(
        p3_matrix::RowMajorMatrix<F> mat, TransformFn&& transform)
    {
        const size_t h = mat.height();
        const size_t w = mat.width();
        if (h <= 1 || w <= 1 || thread_count_ <= 1) {
            return transform(std::move(mat));
        }

        p3_matrix::RowMajorMatrix<F> out(h, w);
        const size_t workers = std::min(thread_count_, w);
        const size_t chunk = (w + workers - 1) / workers;

        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (size_t worker = 0; worker < workers; ++worker) {
            const size_t c_begin = worker * chunk;
            const size_t c_end = std::min(w, c_begin + chunk);
            if (c_begin >= c_end) break;

            threads.emplace_back([&, c_begin, c_end]() {
                for (size_t c = c_begin; c < c_end; ++c) {
                    std::vector<F> col(h);
                    for (size_t r = 0; r < h; ++r) {
                        col[r] = mat.get_unchecked(r, c);
                    }
                    auto transformed_col = transform(p3_matrix::RowMajorMatrix<F>(std::move(col), 1));
                    for (size_t r = 0; r < h; ++r) {
                        out.set_unchecked(r, c, transformed_col.get_unchecked(r, 0));
                    }
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }
        return out;
    }
};

} // namespace p3_dft
