#pragma once

#include "poseidon.hpp"
#include <array>

namespace p3_poseidon {

/**
 * @brief Generic circulant MDS matrix implementation
 *
 * A circulant matrix is defined by its first row; each subsequent row
 * is a cyclic shift of the previous row. This implementation computes
 * matrix-vector products naively in O(WIDTH^2) time.
 *
 * For production use, consider FFT-based implementations for larger widths.
 *
 * @tparam F Field type
 * @tparam WIDTH Matrix dimension
 */
template<typename F, size_t WIDTH>
class CirculantMdsMatrix : public MdsPermutation<F, WIDTH> {
private:
    // First row of the circulant matrix
    std::array<F, WIDTH> first_row;

public:
    /**
     * @brief Construct a circulant MDS matrix from the first row
     * @param row First row of the circulant matrix
     */
    explicit CirculantMdsMatrix(const std::array<F, WIDTH>& row)
        : first_row(row) {}

    /**
     * @brief Construct from integer values (converted to field elements)
     * @param row_values Integer values for the first row
     */
    explicit CirculantMdsMatrix(const std::array<uint64_t, WIDTH>& row_values) {
        for (size_t i = 0; i < WIDTH; ++i) {
            first_row[i] = F(row_values[i]);
        }
    }

    std::array<F, WIDTH> permute(const std::array<F, WIDTH>& input) const override {
        std::array<F, WIDTH> result;

        // For each output element
        for (size_t i = 0; i < WIDTH; ++i) {
            result[i] = F::zero();
            // Compute dot product of i-th row with input
            for (size_t j = 0; j < WIDTH; ++j) {
                // i-th row is a cyclic shift of first_row by i positions
                size_t idx = (j + WIDTH - i) % WIDTH;
                result[i] += first_row[idx] * input[j];
            }
        }

        return result;
    }

    void permute_mut(std::array<F, WIDTH>& state) const override {
        state = permute(state);
    }
};

/**
 * @brief Helper function to convert first row to first column
 *
 * For a circulant matrix, the first column can be computed by
 * reversing the first row (except the first element).
 */
template<size_t N>
constexpr std::array<int64_t, N> first_row_to_first_col(const std::array<int64_t, N>& row) {
    std::array<int64_t, N> col;
    col[0] = row[0];
    for (size_t i = 1; i < N; ++i) {
        col[i] = row[N - i];
    }
    return col;
}

/**
 * @brief BabyBear MDS matrix (WIDTH = 8)
 *
 * First row: [7, 1, 3, 8, 8, 3, 4, 9]
 * This provides MDS properties for the BabyBear field.
 */
template<typename BabyBear>
class MdsMatrixBabyBear8 : public CirculantMdsMatrix<BabyBear, 8> {
public:
    MdsMatrixBabyBear8()
        : CirculantMdsMatrix<BabyBear, 8>(
            std::array<uint64_t, 8>{7, 1, 3, 8, 8, 3, 4, 9}
        ) {}
};

/**
 * @brief BabyBear MDS matrix (WIDTH = 12)
 *
 * First row: [1, 1, 2, 1, 8, 9, 10, 7, 5, 9, 4, 10]
 */
template<typename BabyBear>
class MdsMatrixBabyBear12 : public CirculantMdsMatrix<BabyBear, 12> {
public:
    MdsMatrixBabyBear12()
        : CirculantMdsMatrix<BabyBear, 12>(
            std::array<uint64_t, 12>{1, 1, 2, 1, 8, 9, 10, 7, 5, 9, 4, 10}
        ) {}
};

/**
 * @brief BabyBear MDS matrix (WIDTH = 16)
 *
 * First row: [1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3]
 */
template<typename BabyBear>
class MdsMatrixBabyBear16 : public CirculantMdsMatrix<BabyBear, 16> {
public:
    MdsMatrixBabyBear16()
        : CirculantMdsMatrix<BabyBear, 16>(
            std::array<uint64_t, 16>{1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3}
        ) {}
};

/**
 * @brief BabyBear MDS matrix (WIDTH = 24)
 */
template<typename BabyBear>
class MdsMatrixBabyBear24 : public CirculantMdsMatrix<BabyBear, 24> {
public:
    MdsMatrixBabyBear24()
        : CirculantMdsMatrix<BabyBear, 24>(
            std::array<uint64_t, 24>{
                0x2D0AAAAB, 0x64850517, 0x17F5551D, 0x04ECBEB5,
                0x6D91A8D5, 0x60703026, 0x18D6F3CA, 0x729601A7,
                0x77CDA9E2, 0x3C0F5038, 0x26D52A61, 0x0360405D,
                0x68FC71C8, 0x2495A71D, 0x5D57AFC2, 0x1689DD98,
                0x3C2C3DBE, 0x0C23DC41, 0x0524C7F2, 0x6BE4DF69,
                0x0A6E572C, 0x5C7790FA, 0x17E118F6, 0x0878A07F
            }
        ) {}
};

/**
 * @brief Goldilocks MDS matrix (WIDTH = 8)
 *
 * First row: [7, 1, 3, 8, 8, 3, 4, 9]
 */
template<typename Goldilocks>
class MdsMatrixGoldilocks8 : public CirculantMdsMatrix<Goldilocks, 8> {
public:
    MdsMatrixGoldilocks8()
        : CirculantMdsMatrix<Goldilocks, 8>(
            std::array<uint64_t, 8>{7, 1, 3, 8, 8, 3, 4, 9}
        ) {}
};

/**
 * @brief Goldilocks MDS matrix (WIDTH = 12)
 *
 * First row: [1, 1, 2, 1, 8, 9, 10, 7, 5, 9, 4, 10]
 */
template<typename Goldilocks>
class MdsMatrixGoldilocks12 : public CirculantMdsMatrix<Goldilocks, 12> {
public:
    MdsMatrixGoldilocks12()
        : CirculantMdsMatrix<Goldilocks, 12>(
            std::array<uint64_t, 12>{1, 1, 2, 1, 8, 9, 10, 7, 5, 9, 4, 10}
        ) {}
};

/**
 * @brief Goldilocks MDS matrix (WIDTH = 16)
 *
 * First row: [1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3]
 */
template<typename Goldilocks>
class MdsMatrixGoldilocks16 : public CirculantMdsMatrix<Goldilocks, 16> {
public:
    MdsMatrixGoldilocks16()
        : CirculantMdsMatrix<Goldilocks, 16>(
            std::array<uint64_t, 16>{1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3}
        ) {}
};

/**
 * @brief Mersenne31 MDS matrix (WIDTH = 16)
 *
 * Same first row as BabyBear16 (these matrices are MDS for both fields)
 */
template<typename Mersenne31>
class MdsMatrixMersenne3116 : public CirculantMdsMatrix<Mersenne31, 16> {
public:
    MdsMatrixMersenne3116()
        : CirculantMdsMatrix<Mersenne31, 16>(
            std::array<uint64_t, 16>{1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3}
        ) {}
};

/**
 * @brief Mersenne31 MDS matrix (WIDTH = 32)
 */
template<typename Mersenne31>
class MdsMatrixMersenne3132 : public CirculantMdsMatrix<Mersenne31, 32> {
public:
    MdsMatrixMersenne3132()
        : CirculantMdsMatrix<Mersenne31, 32>(
            std::array<uint64_t, 32>{
                0x0BC00000, 0x2BED8F81, 0x337E0652, 0x4C4535D1,
                0x4AF2DC32, 0x2DB4050F, 0x676A7CE3, 0x3A06B68E,
                0x5E95C1B1, 0x2C5F54A0, 0x2332F13D, 0x58E757F1,
                0x3AA6DCCE, 0x607EE630, 0x4ED57FF0, 0x6E08555B,
                0x4C155556, 0x587FD0CE, 0x462F1551, 0x032A43CC,
                0x5E2E43EA, 0x71609B02, 0x0ED97E45, 0x562CA7E9,
                0x2CB70B1D, 0x4E941E23, 0x174A61C1, 0x117A9426,
                0x73562137, 0x54596086, 0x487C560B, 0x68A4ACAB
            }
        ) {}
};

} // namespace p3_poseidon

