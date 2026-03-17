/**
 * @file test_commit.cpp
 * @brief Google Test suite for commit abstractions.
 *
 * Tests cover:
 *   1. Dimensions -- construction and equality
 *   2. TwoAdicMultiplicativeCoset -- size, elements, natural domain
 *   3. BatchOpening -- construction and field access
 *   4. OpenedValues -- type alias usage
 *   5. ExtensionMmcs -- decompose, reconstitute, commit/open round-trip
 *                       using a mock InnerMmcs
 */

#include <gtest/gtest.h>

#include "domain.hpp"
#include "mmcs.hpp"
#include "pcs.hpp"
#include "extension_mmcs.hpp"

#include "baby_bear.hpp"
#include "extension_field.hpp"
#include "dense_matrix.hpp"

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

using namespace p3_commit;
using namespace p3_field;
using namespace p3_matrix;

// ===========================================================================
// Concrete field types
// ===========================================================================

using BB  = BabyBear;
using BB4 = BabyBear4;  // BinomialExtensionField<BabyBear,4,11>

// ===========================================================================
// Mock InnerMmcs for testing ExtensionMmcs
//
// The mock simply stores committed matrices and returns their rows on open.
// Verification always succeeds (for structural testing).
// ===========================================================================

struct MockProof {};  // empty proof

struct MockProverData {
    std::vector<RowMajorMatrix<BB>> matrices;
};

struct MockMmcs : public MmcsBase<MockMmcs, BB> {
    using Commitment = size_t;   // commit = number of matrices (placeholder)
    using ProverData = MockProverData;
    using Proof      = MockProof;

    std::pair<Commitment, ProverData>
    commit(std::vector<RowMajorMatrix<BB>> matrices) {
        size_t c = matrices.size();
        MockProverData pd;
        pd.matrices = std::move(matrices);
        return {c, std::move(pd)};
    }

    BatchOpening<BB, Proof>
    open_batch(size_t index, const MockProverData& data) {
        std::vector<std::vector<BB>> opened;
        opened.reserve(data.matrices.size());
        for (const auto& m : data.matrices) {
            if (index < m.height()) {
                opened.push_back(m.row(index));
            } else {
                opened.push_back({});
            }
        }
        return BatchOpening<BB, Proof>(std::move(opened), MockProof{});
    }

    bool verify_batch(
        const Commitment& /*commit*/,
        const std::vector<Dimensions>& /*dims*/,
        size_t /*index*/,
        const BatchOpening<BB, Proof>& /*opening*/
    ) {
        return true;
    }

    std::vector<const RowMajorMatrix<BB>*>
    get_matrices(const MockProverData& data) const {
        std::vector<const RowMajorMatrix<BB>*> ptrs;
        ptrs.reserve(data.matrices.size());
        for (const auto& m : data.matrices) {
            ptrs.push_back(&m);
        }
        return ptrs;
    }
};

// ===========================================================================
// 1. Dimensions tests
// ===========================================================================

TEST(Dimensions, DefaultConstructor) {
    Dimensions d;
    EXPECT_EQ(d.width, 0u);
    EXPECT_EQ(d.height, 0u);
}

TEST(Dimensions, ValueConstructor) {
    Dimensions d(4, 8);
    EXPECT_EQ(d.width, 4u);
    EXPECT_EQ(d.height, 8u);
}

TEST(Dimensions, Equality) {
    EXPECT_EQ(Dimensions(3, 5), Dimensions(3, 5));
    EXPECT_NE(Dimensions(3, 5), Dimensions(3, 6));
    EXPECT_NE(Dimensions(3, 5), Dimensions(4, 5));
}

// ===========================================================================
// 2. TwoAdicMultiplicativeCoset tests
// ===========================================================================

TEST(TwoAdicMultiplicativeCoset, Size) {
    for (size_t log_n = 0; log_n <= 5; ++log_n) {
        TwoAdicMultiplicativeCoset<BB> coset(log_n, BB::one_val());
        EXPECT_EQ(coset.size(), size_t(1) << log_n)
            << "log_n = " << log_n;
    }
}

TEST(TwoAdicMultiplicativeCoset, FirstPoint) {
    BB shift(BB::PRIME - 1u);  // some non-trivial shift
    TwoAdicMultiplicativeCoset<BB> coset(3, shift);
    EXPECT_EQ(coset.first_point(), shift);
}

TEST(TwoAdicMultiplicativeCoset, ElementsCount) {
    for (size_t log_n = 0; log_n <= 4; ++log_n) {
        TwoAdicMultiplicativeCoset<BB> coset(log_n, BB::one_val());
        auto elems = coset.elements();
        EXPECT_EQ(elems.size(), size_t(1) << log_n)
            << "log_n = " << log_n;
    }
}

TEST(TwoAdicMultiplicativeCoset, FirstElementIsShift) {
    BB shift(7u);
    TwoAdicMultiplicativeCoset<BB> coset(3, shift);
    auto elems = coset.elements();
    ASSERT_FALSE(elems.empty());
    EXPECT_EQ(elems[0], shift);
}

TEST(TwoAdicMultiplicativeCoset, ElementsFormCyclicGroup) {
    // For shift = 1, elements are 1, g, g^2, ..., g^(n-1).
    // Consecutive elements satisfy: elems[i+1] = elems[i] * g.
    constexpr size_t log_n = 3;
    TwoAdicMultiplicativeCoset<BB> coset(log_n, BB::one_val());
    auto elems = coset.elements();
    BB g = BB::two_adic_generator(log_n);

    for (size_t i = 0; i + 1 < elems.size(); ++i) {
        EXPECT_EQ(elems[i + 1], elems[i] * g)
            << "element sequence broken at i = " << i;
    }
}

TEST(TwoAdicMultiplicativeCoset, NaturalDomainForDegree) {
    // degree 0 -> log_n = 0, size 1
    {
        auto d = TwoAdicMultiplicativeCoset<BB>::natural_domain_for_degree(0);
        EXPECT_EQ(d.size(), 1u);
        EXPECT_EQ(d.shift, BB::one_val());
    }
    // degree 1 -> log_n = 0, size 1  (ceil(log2(1)) = 0)
    {
        auto d = TwoAdicMultiplicativeCoset<BB>::natural_domain_for_degree(1);
        EXPECT_EQ(d.size(), 1u);
    }
    // degree 2 -> log_n = 1, size 2
    {
        auto d = TwoAdicMultiplicativeCoset<BB>::natural_domain_for_degree(2);
        EXPECT_EQ(d.size(), 2u);
    }
    // degree 5 -> log_n = 3 (ceil(log2(5)) = 3), size 8
    {
        auto d = TwoAdicMultiplicativeCoset<BB>::natural_domain_for_degree(5);
        EXPECT_EQ(d.size(), 8u);
    }
    // degree 8 -> log_n = 3 (ceil(log2(8)) = 3), size 8
    {
        auto d = TwoAdicMultiplicativeCoset<BB>::natural_domain_for_degree(8);
        EXPECT_EQ(d.size(), 8u);
    }
}

TEST(TwoAdicMultiplicativeCoset, Equality) {
    TwoAdicMultiplicativeCoset<BB> a(3, BB::one_val());
    TwoAdicMultiplicativeCoset<BB> b(3, BB::one_val());
    TwoAdicMultiplicativeCoset<BB> c(4, BB::one_val());

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

// ===========================================================================
// 3. BatchOpening tests
// ===========================================================================

TEST(BatchOpening, DefaultConstructor) {
    BatchOpening<BB, MockProof> bo;
    EXPECT_TRUE(bo.opened_values.empty());
}

TEST(BatchOpening, ValueConstructor) {
    std::vector<std::vector<BB>> vals = {
        {BB(1u), BB(2u)},
        {BB(3u), BB(4u)}
    };
    MockProof proof;
    BatchOpening<BB, MockProof> bo(vals, proof);

    ASSERT_EQ(bo.opened_values.size(), 2u);
    EXPECT_EQ(bo.opened_values[0][0], BB(1u));
    EXPECT_EQ(bo.opened_values[1][1], BB(4u));
}

// ===========================================================================
// 4. OpenedValues type alias
// ===========================================================================

TEST(OpenedValues, TypeAlias) {
    // Verify the type is usable as a 3D vector.
    OpenedValues<BB4> ov;
    ov.push_back({{BB4::one_val(), BB4::zero_val()}});
    ASSERT_EQ(ov.size(), 1u);
    ASSERT_EQ(ov[0].size(), 1u);
    EXPECT_EQ(ov[0][0][0], BB4::one_val());
}

// ===========================================================================
// 5. MmcsBase helpers via MockMmcs
// ===========================================================================

TEST(MmcsBase, CommitMatrix) {
    MockMmcs mmcs;
    RowMajorMatrix<BB> mat(2, 3, BB(5u));
    auto [commit, data] = mmcs.commit_matrix(std::move(mat));
    EXPECT_EQ(commit, 1u);  // 1 matrix committed
    EXPECT_EQ(data.matrices.size(), 1u);
}

TEST(MmcsBase, GetMaxHeight) {
    MockMmcs mmcs;
    std::vector<RowMajorMatrix<BB>> mats;
    mats.emplace_back(4, 2, BB(1u));
    mats.emplace_back(8, 3, BB(2u));
    mats.emplace_back(2, 1, BB(3u));
    auto [commit, data] = mmcs.commit(std::move(mats));

    size_t max_h = mmcs.get_max_height(data);
    EXPECT_EQ(max_h, 8u);
}

// ===========================================================================
// 6. ExtensionMmcs tests
// ===========================================================================

// Helper: build a BB4 matrix with known values.
static RowMajorMatrix<BB4> make_bb4_matrix(
    size_t rows, size_t cols,
    // value[r][c] = BB4([r*cols+c, 0, 0, 0])  (embed base field)
    bool embed_only = true
) {
    std::vector<BB4> vals;
    vals.reserve(rows * cols);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            uint32_t v = static_cast<uint32_t>(r * cols + c + 1);
            if (embed_only) {
                // Each EF element is just [v, 0, 0, 0]
                vals.emplace_back(std::array<BB, 4>{BB(v), BB(), BB(), BB()});
            } else {
                // Each EF element uses all 4 coefficients
                vals.emplace_back(std::array<BB, 4>{
                    BB(v), BB(v + 1u), BB(v + 2u), BB(v + 3u)});
            }
        }
    }
    return RowMajorMatrix<BB4>(std::move(vals), cols);
}

TEST(ExtensionMmcs, DecomposeShape) {
    auto mat = make_bb4_matrix(3, 5);
    auto decomposed = ExtensionMmcs<BB, BB4, MockMmcs>::decompose(mat);

    ASSERT_EQ(decomposed.size(), 4u);  // D = 4
    for (const auto& bm : decomposed) {
        EXPECT_EQ(bm.height(), 3u);
        EXPECT_EQ(bm.width(), 5u);
    }
}

TEST(ExtensionMmcs, DecomposeValues) {
    // EF element at (r,c) = [v, v+1, v+2, v+3]  where v = r*cols+c+1
    auto mat = make_bb4_matrix(2, 3, /*embed_only=*/false);
    auto decomposed = ExtensionMmcs<BB, BB4, MockMmcs>::decompose(mat);

    ASSERT_EQ(decomposed.size(), 4u);

    for (size_t r = 0; r < 2; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            uint32_t v = static_cast<uint32_t>(r * 3 + c + 1);
            for (size_t d = 0; d < 4; ++d) {
                BB expected(v + static_cast<uint32_t>(d));
                EXPECT_EQ(decomposed[d].get_unchecked(r, c), expected)
                    << "r=" << r << " c=" << c << " d=" << d;
            }
        }
    }
}

TEST(ExtensionMmcs, ReconstituteRow) {
    // Build D=4 base-field rows, each with 3 elements.
    std::vector<std::vector<BB>> base_rows(4);
    for (size_t d = 0; d < 4; ++d) {
        for (size_t c = 0; c < 3; ++c) {
            base_rows[d].push_back(BB(static_cast<uint32_t>(d * 10 + c + 1)));
        }
    }

    auto ef_row = ExtensionMmcs<BB, BB4, MockMmcs>::reconstitute_row(base_rows);
    ASSERT_EQ(ef_row.size(), 3u);

    for (size_t c = 0; c < 3; ++c) {
        for (size_t d = 0; d < 4; ++d) {
            BB expected(static_cast<uint32_t>(d * 10 + c + 1));
            EXPECT_EQ(ef_row[c][d], expected)
                << "c=" << c << " d=" << d;
        }
    }
}

TEST(ExtensionMmcs, CommitProducesCorrectInnerMatrixCount) {
    MockMmcs inner;
    ExtensionMmcs<BB, BB4, MockMmcs> ext_mmcs(inner);

    // Commit two EF matrices (2 x 3 and 4 x 2).
    std::vector<RowMajorMatrix<BB4>> matrices;
    matrices.push_back(make_bb4_matrix(2, 3));
    matrices.push_back(make_bb4_matrix(4, 2));

    auto [commit, prover_data] = ext_mmcs.commit(std::move(matrices));

    // 2 EF matrices * D=4 = 8 inner base-field matrices.
    EXPECT_EQ(commit, 8u);
    EXPECT_EQ(prover_data.matrices.size(), 8u);
}

TEST(ExtensionMmcs, CommitOpenRoundTrip) {
    // Build a single 4x3 BB4 matrix with fully populated coefficients.
    auto ef_mat = make_bb4_matrix(4, 3, /*embed_only=*/false);

    MockMmcs inner;
    ExtensionMmcs<BB, BB4, MockMmcs> ext_mmcs(inner);

    std::vector<RowMajorMatrix<BB4>> matrices;
    matrices.push_back(ef_mat);

    auto [commit, prover_data] = ext_mmcs.commit(std::move(matrices));

    // Open at row 2.
    size_t query_row = 2;
    auto opening = ext_mmcs.open_batch(query_row, prover_data);

    // We committed 1 EF matrix, so we get back 1 EF row.
    ASSERT_EQ(opening.opened_values.size(), 1u);

    const auto& ef_row = opening.opened_values[0];
    ASSERT_EQ(ef_row.size(), 3u);  // width = 3

    // Verify each element matches the original matrix.
    for (size_t c = 0; c < 3; ++c) {
        EXPECT_EQ(ef_row[c], ef_mat.get_unchecked(query_row, c))
            << "column " << c;
    }
}

TEST(ExtensionMmcs, VerifyBatchDelegatesToInner) {
    auto ef_mat = make_bb4_matrix(4, 3, /*embed_only=*/false);

    MockMmcs inner;
    ExtensionMmcs<BB, BB4, MockMmcs> ext_mmcs(inner);

    std::vector<RowMajorMatrix<BB4>> matrices;
    matrices.push_back(ef_mat);

    auto [commit, prover_data] = ext_mmcs.commit(std::move(matrices));

    size_t query_row = 1;
    auto opening = ext_mmcs.open_batch(query_row, prover_data);

    // Build the dimensions list for the single EF matrix.
    std::vector<Dimensions> dims = {Dimensions(3, 4)};  // width=3, height=4

    // MockMmcs::verify_batch always returns true.
    bool ok = ext_mmcs.verify_batch(commit, dims, query_row, opening);
    EXPECT_TRUE(ok);
}

TEST(ExtensionMmcs, EmbeddedBaseFieldRoundTrip) {
    // Use only base-field elements embedded in the extension (coeff[0] = v,
    // others = 0).  This exercises the common FRI path.
    auto ef_mat = make_bb4_matrix(8, 4, /*embed_only=*/true);

    MockMmcs inner;
    ExtensionMmcs<BB, BB4, MockMmcs> ext_mmcs(inner);

    std::vector<RowMajorMatrix<BB4>> matrices;
    matrices.push_back(ef_mat);

    auto [commit, prover_data] = ext_mmcs.commit(std::move(matrices));

    for (size_t row = 0; row < 8; ++row) {
        auto opening = ext_mmcs.open_batch(row, prover_data);
        ASSERT_EQ(opening.opened_values.size(), 1u);

        const auto& ef_row = opening.opened_values[0];
        ASSERT_EQ(ef_row.size(), 4u);

        for (size_t c = 0; c < 4; ++c) {
            EXPECT_EQ(ef_row[c], ef_mat.get_unchecked(row, c))
                << "row=" << row << " col=" << c;
        }
    }
}
