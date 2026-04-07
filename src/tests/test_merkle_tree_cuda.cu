#include <gtest/gtest.h>

#include "baby_bear.hpp"
#include "dense_matrix.hpp"
#include "merkle_tree.hpp"
#include "merkle_tree_cuda.hpp"
#include "merkle_tree_mmcs.hpp"
#include "padding_free_sponge.hpp"
#include "poseidon2_cuda.hpp"
#include "truncated_permutation.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

using p3_field::BabyBear;
using p3_matrix::RowMajorMatrix;
using p3_merkle_tree::build_merkle_tree_cuda;
using p3_merkle_tree::get_cap;
using p3_merkle_tree::MerkleCap;
using p3_merkle_tree::MerkleTree;
using p3_merkle_tree::MerkleTreeMmcs;
using p3_symmetric::PaddingFreeSponge;
using p3_symmetric::TruncatedPermutation;
using poseidon2::Poseidon2Cuda;

namespace {

constexpr size_t WIDTH = 16;
constexpr size_t RATE = 8;
constexpr size_t DIGEST = 8;
constexpr size_t ROUNDS_F = 8;
constexpr size_t ROUNDS_P = 14;
constexpr uint64_t D_EXP = 5;

using F = BabyBear;
using PoseidonCuda = Poseidon2Cuda<F, WIDTH, ROUNDS_F, ROUNDS_P, D_EXP>;

struct PoseidonWrapper {
    PoseidonCuda poseidon;
    void permute_mut(std::array<F, WIDTH>& state) const { poseidon.permute(state.data()); }
};

using CpuSponge = PaddingFreeSponge<PoseidonWrapper, F, WIDTH, RATE, DIGEST>;
using CpuTrunc = TruncatedPermutation<PoseidonWrapper, F, 2, DIGEST, WIDTH>;
using CpuMmcs = MerkleTreeMmcs<F, F, CpuSponge, CpuTrunc, DIGEST>;

PoseidonCuda make_poseidon() {
    PoseidonCuda p{};
    for (size_t r = 0; r < ROUNDS_F / 2; ++r) {
        for (size_t i = 0; i < WIDTH; ++i) {
            p.initial_external_constants[r][i] = F(static_cast<uint32_t>((13u * r + 7u * i + 3u) % F::PRIME));
            p.terminal_external_constants[r][i] = F(static_cast<uint32_t>((17u * r + 5u * i + 11u) % F::PRIME));
        }
    }
    for (size_t r = 0; r < ROUNDS_P; ++r) {
        p.internal_constants[r] = F(static_cast<uint32_t>((19u * r + 23u) % F::PRIME));
    }
    return p;
}

RowMajorMatrix<F> make_matrix(size_t rows, size_t cols, uint32_t start) {
    std::vector<F> vals(rows * cols);
    for (size_t i = 0; i < vals.size(); ++i) vals[i] = F(start + static_cast<uint32_t>(i));
    return RowMajorMatrix<F>(std::move(vals), cols);
}

std::vector<RowMajorMatrix<F>> clone_mats(const std::vector<RowMajorMatrix<F>>& mats) {
    std::vector<RowMajorMatrix<F>> out;
    out.reserve(mats.size());
    for (const auto& m : mats) out.emplace_back(m.values, m.width());
    return out;
}

} // namespace

TEST(MerkleTreeCuda, CapMatchesCpuSingleMatrix) {
    const auto poseidon = make_poseidon();
    const std::vector<RowMajorMatrix<F>> mats = {make_matrix(8, 3, 1)};

    PoseidonWrapper wrapper{poseidon};
    CpuMmcs mmcs(CpuSponge{wrapper}, CpuTrunc{wrapper}, 0);
    auto [cpu_cap, cpu_tree] = mmcs.commit(clone_mats(mats));
    (void)cpu_tree;

    MerkleTree<F, F, DIGEST> gpu_tree =
        build_merkle_tree_cuda<F, WIDTH, RATE, DIGEST, ROUNDS_F, ROUNDS_P, D_EXP>(
            clone_mats(mats), poseidon, 0);
    const MerkleCap<F, DIGEST> gpu_cap = get_cap(gpu_tree);

    EXPECT_EQ(gpu_cap, cpu_cap);
}

TEST(MerkleTreeCuda, CapMatchesCpuMixedHeight) {
    const auto poseidon = make_poseidon();
    const std::vector<RowMajorMatrix<F>> mats = {
        make_matrix(8, 2, 1),
        make_matrix(4, 3, 100),
    };

    PoseidonWrapper wrapper{poseidon};
    CpuMmcs mmcs(CpuSponge{wrapper}, CpuTrunc{wrapper}, 0);
    auto [cpu_cap, cpu_tree] = mmcs.commit(clone_mats(mats));
    (void)cpu_tree;

    MerkleTree<F, F, DIGEST> gpu_tree =
        build_merkle_tree_cuda<F, WIDTH, RATE, DIGEST, ROUNDS_F, ROUNDS_P, D_EXP>(
            clone_mats(mats), poseidon, 0);
    const MerkleCap<F, DIGEST> gpu_cap = get_cap(gpu_tree);

    EXPECT_EQ(gpu_cap, cpu_cap);
}

TEST(MerkleTreeCuda, CapHeightProducesExpectedSize) {
    const auto poseidon = make_poseidon();
    const std::vector<RowMajorMatrix<F>> mats = {make_matrix(8, 2, 50)};

    MerkleTree<F, F, DIGEST> gpu_tree =
        build_merkle_tree_cuda<F, WIDTH, RATE, DIGEST, ROUNDS_F, ROUNDS_P, D_EXP>(
            clone_mats(mats), poseidon, 2);
    const MerkleCap<F, DIGEST> gpu_cap = get_cap(gpu_tree);

    EXPECT_EQ(gpu_cap.cap.size(), 4u);
}

