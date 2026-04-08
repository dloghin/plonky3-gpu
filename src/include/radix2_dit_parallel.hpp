#pragma once

#include "traits.hpp"
#include "radix2_dit.hpp"
#include "dense_matrix.hpp"

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace p3_dft {

template<typename F>
class Radix2DitParallel : public TwoAdicSubgroupDft<F, Radix2DitParallel<F>> {
public:
    explicit Radix2DitParallel(size_t thread_count = 0)
        : thread_count_(thread_count == 0 ? default_thread_count() : thread_count)
    {
        start_pool();
    }

    ~Radix2DitParallel() { stop_pool(); }

    Radix2DitParallel(const Radix2DitParallel&) = delete;
    Radix2DitParallel& operator=(const Radix2DitParallel&) = delete;
    Radix2DitParallel(Radix2DitParallel&&) = delete;
    Radix2DitParallel& operator=(Radix2DitParallel&&) = delete;

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
    size_t pool_workers_ = 0;
    std::vector<std::thread> pool_;
    std::mutex pool_mu_;
    std::condition_variable pool_cv_;
    std::condition_variable done_cv_;
    bool stopping_ = false;
    size_t work_epoch_ = 0;
    size_t done_count_ = 0;
    size_t remote_tasks_ = 0;
    std::function<void(size_t, Radix2Dit<F>&)> work_fn_;

    static size_t default_thread_count() {
        const unsigned hc = std::thread::hardware_concurrency();
        return hc == 0 ? 1u : static_cast<size_t>(hc);
    }

    void start_pool() {
        if (thread_count_ <= 1) return;
        pool_workers_ = thread_count_ - 1; // caller thread participates as worker 0
        pool_.reserve(pool_workers_);
        for (size_t worker_id = 0; worker_id < pool_workers_; ++worker_id) {
            pool_.emplace_back([this, worker_id]() { worker_loop(worker_id); });
        }
    }

    void stop_pool() {
        {
            std::lock_guard<std::mutex> lock(pool_mu_);
            stopping_ = true;
            ++work_epoch_;
        }
        pool_cv_.notify_all();
        for (auto& t : pool_) {
            if (t.joinable()) t.join();
        }
        pool_.clear();
        pool_workers_ = 0;
    }

    void worker_loop(size_t worker_id) {
        Radix2Dit<F> dit; // Reused across calls to preserve twiddle memoization.
        size_t seen_epoch = 0;
        for (;;) {
            std::function<void(size_t, Radix2Dit<F>&)> fn;
            size_t remote_tasks = 0;
            {
                std::unique_lock<std::mutex> lock(pool_mu_);
                pool_cv_.wait(lock, [&]() { return stopping_ || work_epoch_ != seen_epoch; });
                if (stopping_) return;
                seen_epoch = work_epoch_;
                fn = work_fn_;
                remote_tasks = remote_tasks_;
            }

            if (fn && worker_id < remote_tasks) {
                fn(worker_id + 1, dit);
            }

            {
                std::lock_guard<std::mutex> lock(pool_mu_);
                ++done_count_;
                if (done_count_ == pool_workers_) done_cv_.notify_one();
            }
        }
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
        const size_t tasks = std::min(thread_count_, w);
        const size_t chunk = (w + tasks - 1) / tasks;

        auto run_task = [&](size_t task_id, Radix2Dit<F>& dit) {
            const size_t c_begin = task_id * chunk;
            const size_t c_end = std::min(w, c_begin + chunk);
            if (c_begin >= c_end) return;

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
        };

        {
            std::lock_guard<std::mutex> lock(pool_mu_);
            done_count_ = 0;
            remote_tasks_ = tasks > 0 ? tasks - 1 : 0;
            work_fn_ = run_task;
            ++work_epoch_;
        }
        pool_cv_.notify_all();

        // Caller thread participates as task 0.
        Radix2Dit<F> caller_dit;
        run_task(0, caller_dit);

        {
            std::unique_lock<std::mutex> lock(pool_mu_);
            done_cv_.wait(lock, [&]() { return done_count_ == pool_workers_; });
        }
        return out;
    }
};

} // namespace p3_dft
