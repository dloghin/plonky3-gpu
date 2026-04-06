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
                0x2d0aaaab, 0x64850517, 0x17f5551d, 0x04ecbeb5,
                0x6d91a8d5, 0x60703026, 0x18d6f3ca, 0x729601a7,
                0x77cda9e2, 0x3c0f5038, 0x26d52a61, 0x0360405d,
                0x68fc71c8, 0x2495a71d, 0x5d57afc2, 0x1689dd98,
                0x3c2c3dbe, 0x0c23dc41, 0x0524c7f2, 0x6be4df69,
                0x0a6e572c, 0x5c7790fa, 0x17e118f6, 0x0878a07f
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
                0x0bc00000, 0x2bed8f81, 0x337e0652, 0x4c4535d1,
                0x4af2dc32, 0x2db4050f, 0x676a7ce3, 0x3a06b68e,
                0x5e95c1b1, 0x2c5f54a0, 0x2332f13d, 0x58e757f1,
                0x3aa6dcce, 0x607ee630, 0x4ed57ff0, 0x6e08555b,
                0x4c155556, 0x587fd0ce, 0x462f1551, 0x032a43cc,
                0x5e2e43ea, 0x71609b02, 0x0ed97e45, 0x562ca7e9,
                0x2cb70b1d, 0x4e941e23, 0x174a61c1, 0x117a9426,
                0x73562137, 0x54596086, 0x487c560b, 0x68a4acab
            }
        ) {}
};

/**
 * @brief KoalaBear MDS circulant matrices (p3-koala-bear mds.rs; same first rows as BabyBear for 8,12,16)
 */
template<typename KoalaBear>
class MdsMatrixKoalaBear8 : public CirculantMdsMatrix<KoalaBear, 8> {
public:
    MdsMatrixKoalaBear8()
        : CirculantMdsMatrix<KoalaBear, 8>(
            std::array<uint64_t, 8>{7, 1, 3, 8, 8, 3, 4, 9}
        ) {}
};

template<typename KoalaBear>
class MdsMatrixKoalaBear12 : public CirculantMdsMatrix<KoalaBear, 12> {
public:
    MdsMatrixKoalaBear12()
        : CirculantMdsMatrix<KoalaBear, 12>(
            std::array<uint64_t, 12>{1, 1, 2, 1, 8, 9, 10, 7, 5, 9, 4, 10}
        ) {}
};

template<typename KoalaBear>
class MdsMatrixKoalaBear16 : public CirculantMdsMatrix<KoalaBear, 16> {
public:
    MdsMatrixKoalaBear16()
        : CirculantMdsMatrix<KoalaBear, 16>(
            std::array<uint64_t, 16>{1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3}
        ) {}
};

template<typename KoalaBear>
class MdsMatrixKoalaBear24 : public CirculantMdsMatrix<KoalaBear, 24> {
public:
    MdsMatrixKoalaBear24()
        : CirculantMdsMatrix<KoalaBear, 24>(
            std::array<uint64_t, 24>{
                0x2d0aaaab, 0x64850517, 0x17f5551d, 0x04ecbeb5,
                0x6d91a8d5, 0x60703026, 0x18d6f3ca, 0x729601a7,
                0x77cda9e2, 0x3c0f5038, 0x26d52a61, 0x0360405d,
                0x68fc71c8, 0x2495a71d, 0x5d57afc2, 0x1689dd98,
                0x3c2c3dbe, 0x0c23dc41, 0x0524c7f2, 0x6be4df69,
                0x0a6e572c, 0x5c7790fa, 0x17e118f6, 0x0878a07f
            }
        ) {}
};

template<typename KoalaBear>
class MdsMatrixKoalaBear32 : public CirculantMdsMatrix<KoalaBear, 32> {
public:
    MdsMatrixKoalaBear32()
        : CirculantMdsMatrix<KoalaBear, 32>(
            std::array<uint64_t, 32>{
                0x0bc00000, 0x2bed8f81, 0x337e0652, 0x4c4535d1,
                0x4af2dc32, 0x2db4050f, 0x676a7ce3, 0x3a06b68e,
                0x5e95c1b1, 0x2c5f54a0, 0x2332f13d, 0x58e757f1,
                0x3aa6dcce, 0x607ee630, 0x4ed57ff0, 0x6e08555b,
                0x4c155556, 0x587fd0ce, 0x462f1551, 0x032a43cc,
                0x5e2e43ea, 0x71609b02, 0x0ed97e45, 0x562ca7e9,
                0x2cb70b1d, 0x4e941e23, 0x174a61c1, 0x117a9426,
                0x73562137, 0x54596086, 0x487c560b, 0x68a4acab
            }
        ) {}
};

} // namespace p3_poseidon

