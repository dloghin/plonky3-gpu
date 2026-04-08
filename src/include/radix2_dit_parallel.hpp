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
        return run_column_parallel(std::move(mat),
            [](Radix2Dit<F>& dit, p3_matrix::RowMajorMatrix<F> m) {
                return dit.dft_batch(std::move(m));
            });
    }

    p3_matrix::RowMajorMatrix<F> idft_batch(p3_matrix::RowMajorMatrix<F> mat) {
        return run_column_parallel(std::move(mat),
            [](Radix2Dit<F>& dit, p3_matrix::RowMajorMatrix<F> m) {
                return dit.idft_batch(std::move(m));
            });
    }

    p3_matrix::RowMajorMatrix<F> coset_dft_batch(
        p3_matrix::RowMajorMatrix<F> mat, const F& shift)
    {
        return run_column_parallel(std::move(mat),
            [shift](Radix2Dit<F>& dit, p3_matrix::RowMajorMatrix<F> m) {
                return dit.coset_dft_batch(std::move(m), shift);
            });
    }

    p3_matrix::RowMajorMatrix<F> coset_idft_batch(
        p3_matrix::RowMajorMatrix<F> mat, const F& shift)
    {
        return run_column_parallel(std::move(mat),
            [shift](Radix2Dit<F>& dit, p3_matrix::RowMajorMatrix<F> m) {
                return dit.coset_idft_batch(std::move(m), shift);
            });
    }

private:
    size_t thread_count_;

    static size_t default_thread_count() {
        const unsigned hc = std::thread::hardware_concurrency();
        return hc == 0 ? 1u : static_cast<size_t>(hc);
    }

    /**
     * @param op  Called as op(dit, matrix). Receives one Radix2Dit per sequential
     *            call or per worker thread so twiddle caches are reused across
     *            all columns in that scope (not recreated per column).
     */
    template<typename OpFn>
    p3_matrix::RowMajorMatrix<F> run_column_parallel(
        p3_matrix::RowMajorMatrix<F> mat, OpFn&& op)
    {
        const size_t h = mat.height();
        const size_t w = mat.width();
        if (h <= 1 || w <= 1 || thread_count_ <= 1) {
            Radix2Dit<F> dit;
            return op(dit, std::move(mat));
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
                Radix2Dit<F> dit;
                const size_t local_w = c_end - c_begin;

                // Process this worker's columns as one submatrix to avoid
                // per-column vector/matrix allocations in the hot loop.
                std::vector<F> block_vals(h * local_w);
                for (size_t r = 0; r < h; ++r) {
                    for (size_t c = 0; c < local_w; ++c) {
                        block_vals[r * local_w + c] = mat.get_unchecked(r, c_begin + c);
                    }
                }

                auto transformed_block =
                    op(dit, p3_matrix::RowMajorMatrix<F>(std::move(block_vals), local_w));

                for (size_t r = 0; r < h; ++r) {
                    for (size_t c = 0; c < local_w; ++c) {
                        out.set_unchecked(r, c_begin + c, transformed_block.get_unchecked(r, c));
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
