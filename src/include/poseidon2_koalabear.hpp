#pragma once

/**
 * Poseidon2 for KoalaBear (S-box degree 3), matching p3-koala-bear / p3-monty-31.
 */

#include "koala_bear.hpp"
#include "poseidon2.hpp"
#include "external.hpp"
#include "internal.hpp"
#include <array>
#include <memory>
#include <vector>

namespace poseidon2 {

using p3_field::KoalaBear;

constexpr uint64_t KOALABEAR_S_BOX_DEGREE = 3;
constexpr size_t KOALABEAR_POSEIDON2_HALF_FULL_ROUNDS = 4;
constexpr size_t KOALABEAR_POSEIDON2_PARTIAL_ROUNDS_16 = 20;
constexpr size_t KOALABEAR_POSEIDON2_PARTIAL_ROUNDS_24 = 23;

inline void koala_internal_layer_mat_mul_16(std::array<KoalaBear, 16>& state, KoalaBear sum) {
    state[1] += sum;
    state[2] = state[2].double_val() + sum;
    state[3] = state[3].halve() + sum;
    state[4] = sum + state[4].double_val() + state[4];
    state[5] = sum + state[5].double_val().double_val();
    state[6] = sum - state[6].halve();
    state[7] = sum - (state[7].double_val() + state[7]);
    state[8] = sum - state[8].double_val().double_val();
    state[9] = state[9].div_2exp_u64(8);
    state[9] += sum;
    state[10] = state[10].div_2exp_u64(3);
    state[10] += sum;
    state[11] = state[11].div_2exp_u64(24);
    state[11] += sum;
    state[12] = state[12].div_2exp_u64(8);
    state[12] = sum - state[12];
    state[13] = state[13].div_2exp_u64(3);
    state[13] = sum - state[13];
    state[14] = state[14].div_2exp_u64(4);
    state[14] = sum - state[14];
    state[15] = state[15].div_2exp_u64(24);
    state[15] = sum - state[15];
}

inline void koala_internal_layer_mat_mul_24(std::array<KoalaBear, 24>& state, KoalaBear sum) {
    state[1] += sum;
    state[2] = state[2].double_val() + sum;
    state[3] = state[3].halve() + sum;
    state[4] = sum + state[4].double_val() + state[4];
    state[5] = sum + state[5].double_val().double_val();
    state[6] = sum - state[6].halve();
    state[7] = sum - (state[7].double_val() + state[7]);
    state[8] = sum - state[8].double_val().double_val();
    state[9] = state[9].div_2exp_u64(8);
    state[9] += sum;
    state[10] = state[10].div_2exp_u64(2);
    state[10] += sum;
    state[11] = state[11].div_2exp_u64(3);
    state[11] += sum;
    state[12] = state[12].div_2exp_u64(4);
    state[12] += sum;
    state[13] = state[13].div_2exp_u64(5);
    state[13] += sum;
    state[14] = state[14].div_2exp_u64(6);
    state[14] += sum;
    state[15] = state[15].div_2exp_u64(24);
    state[15] += sum;
    state[16] = state[16].div_2exp_u64(8);
    state[16] = sum - state[16];
    state[17] = state[17].div_2exp_u64(3);
    state[17] = sum - state[17];
    state[18] = state[18].div_2exp_u64(4);
    state[18] = sum - state[18];
    state[19] = state[19].div_2exp_u64(5);
    state[19] = sum - state[19];
    state[20] = state[20].div_2exp_u64(6);
    state[20] = sum - state[20];
    state[21] = state[21].div_2exp_u64(7);
    state[21] = sum - state[21];
    state[22] = state[22].div_2exp_u64(9);
    state[22] = sum - state[22];
    state[23] = state[23].div_2exp_u64(24);
    state[23] = sum - state[23];
}

template<size_t WIDTH>
class KoalaBearPoseidon2Internal : public InternalLayer<KoalaBear, WIDTH, KOALABEAR_S_BOX_DEGREE> {
    static_assert(WIDTH == 16 || WIDTH == 24, "KoalaBear Poseidon2 internal layer: WIDTH must be 16 or 24");

    std::vector<KoalaBear> constants_;

public:
    explicit KoalaBearPoseidon2Internal(std::vector<KoalaBear> constants)
        : constants_(std::move(constants)) {}

    void permute_state(std::array<KoalaBear, WIDTH>& state) override {
        for (const auto& rc : constants_) {
            state[0] += rc;
            state[0] = state[0].template injective_exp_n<KOALABEAR_S_BOX_DEGREE>();

            KoalaBear part_sum = KoalaBear::zero_val();
            for (size_t i = 1; i < WIDTH; ++i) {
                part_sum += state[i];
            }
            KoalaBear full_sum = part_sum + state[0];
            state[0] = part_sum - state[0];

            if constexpr (WIDTH == 16) {
                koala_internal_layer_mat_mul_16(state, full_sum);
            } else {
                koala_internal_layer_mat_mul_24(state, full_sum);
            }
        }
    }
};

// --- Round constants (canonical u32, from p3-koala-bear poseidon2.rs) ---

inline std::vector<std::array<KoalaBear, 16>> koalabear_poseidon2_rc_16_external_initial() {
    static constexpr uint32_t D[4][16] = {
        {0x7ee56a48u, 0x11367045u, 0x12e41941u, 0x7ebbc12bu, 0x1970b7d5u, 0x662b60e8u, 0x3e4990c6u,
         0x679f91f5u, 0x350813bbu, 0x00874ad4u, 0x28a0081au, 0x18fa5872u, 0x5f25b071u, 0x5e5d5998u,
         0x5e6fd3e7u, 0x5b2e2660u},
        {0x6f1837bfu, 0x3fe6182bu, 0x1edd7ac5u, 0x57470d00u, 0x43d486d5u, 0x1982c70fu, 0x0ea53af9u,
         0x61d6165bu, 0x51639c00u, 0x2dec352cu, 0x2950e531u, 0x2d2cb947u, 0x08256cefu, 0x1a0109f6u,
         0x1f51faf3u, 0x5cef1c62u},
        {0x3d65e50eu, 0x33d91626u, 0x133d5a1eu, 0x0ff49b0du, 0x38900cd1u, 0x2c22cc3fu, 0x28852bb2u,
         0x06c65a02u, 0x7b2cf7bcu, 0x68016e1au, 0x15e16bc0u, 0x5248149au, 0x6dd212a0u, 0x18d6830au,
         0x5001be82u, 0x64dac34eu},
        {0x5902b287u, 0x426583a0u, 0x0c921632u, 0x3fe028a5u, 0x245f8e49u, 0x43bb297eu, 0x7873dbd9u,
         0x3cc987dfu, 0x286bb4ceu, 0x640a8dcdu, 0x512a8e36u, 0x03a4cf55u, 0x481837a2u, 0x03d6da84u,
         0x73726ac7u, 0x760e7fdfu},
    };
    std::vector<std::array<KoalaBear, 16>> v;
    v.reserve(4);
    for (const auto& row : D) {
        std::array<KoalaBear, 16> a{};
        for (size_t i = 0; i < 16; ++i) {
            a[i] = KoalaBear(row[i]);
        }
        v.push_back(a);
    }
    return v;
}

inline std::vector<std::array<KoalaBear, 16>> koalabear_poseidon2_rc_16_external_final() {
    static constexpr uint32_t D[4][16] = {
        {0x43e7dc24u, 0x259a5d61u, 0x27e85a3bu, 0x1b9133fau, 0x343e5628u, 0x485cd4c2u, 0x16e269f5u,
         0x165b60c6u, 0x25f683d9u, 0x124f81f9u, 0x174331f9u, 0x77344dc5u, 0x5a821dbau, 0x5fc4177fu,
         0x54153bf5u, 0x5e3f1194u},
        {0x3bdbf191u, 0x088c84a3u, 0x68256c9bu, 0x3c90bbc6u, 0x6846166au, 0x03f4238du, 0x463335fbu,
         0x5e3d3551u, 0x6e59ae6fu, 0x32d06cc0u, 0x596293f3u, 0x6c87edb2u, 0x08fc60b5u, 0x34bcca80u,
         0x24f007f3u, 0x62731c6fu},
        {0x1e1db6c6u, 0x0ca409bbu, 0x585c1e78u, 0x56e94edcu, 0x16d22734u, 0x18e11467u, 0x7b2c3730u,
         0x770075e4u, 0x35d1b18cu, 0x22be3db5u, 0x4fb1fbb7u, 0x477cb3edu, 0x7d5311c6u, 0x5b62ae7du,
         0x559c5fa8u, 0x77f15048u},
        {0x3211570bu, 0x490fef6au, 0x77ec311fu, 0x2247171bu, 0x4e0ac711u, 0x2edf69c9u, 0x3b5a8850u,
         0x65809421u, 0x5619b4aau, 0x362019a7u, 0x6bf9d4edu, 0x5b413dffu, 0x617e181eu, 0x5e7ab57bu,
         0x33ad7833u, 0x3466c7cau},
    };
    std::vector<std::array<KoalaBear, 16>> v;
    v.reserve(4);
    for (const auto& row : D) {
        std::array<KoalaBear, 16> a{};
        for (size_t i = 0; i < 16; ++i) {
            a[i] = KoalaBear(row[i]);
        }
        v.push_back(a);
    }
    return v;
}

inline std::vector<KoalaBear> koalabear_poseidon2_rc_16_internal() {
    static constexpr uint32_t D[20] = {
        0x54dfeb5du, 0x7d40afd6u, 0x722cb316u, 0x106a4573u, 0x45a7ccdbu, 0x44061375u, 0x154077a5u,
        0x45744faau, 0x4eb5e5eeu, 0x3794e83fu, 0x47c7093cu, 0x5694903cu, 0x69cb6299u, 0x373df84cu,
        0x46a0df58u, 0x46b8758au, 0x3241ebcbu, 0x0b09d233u, 0x1af42357u, 0x1e66cec2u,
    };
    std::vector<KoalaBear> v;
    v.reserve(20);
    for (uint32_t x : D) {
        v.emplace_back(x);
    }
    return v;
}

inline std::vector<std::array<KoalaBear, 24>> koalabear_poseidon2_rc_24_external_initial() {
    static constexpr uint32_t D[4][24] = {
        {0x1d0939dcu, 0x6d050f8du, 0x628058adu, 0x2681385du, 0x3e3c62beu, 0x032cfad8u, 0x5a91ba3cu,
         0x015a56e6u, 0x696b889cu, 0x0dbcd780u, 0x5881b5c9u, 0x2a076f2eu, 0x55393055u, 0x6513a085u,
         0x547ac78fu, 0x4281c5b8u, 0x3e7a3f6cu, 0x34562c19u, 0x2c04e679u, 0x0ed78234u, 0x5f7a1aa9u,
         0x0177640eu, 0x0ea4f8d1u, 0x15be7692u},
        {0x6eafdd62u, 0x71a572c6u, 0x72416f0au, 0x31ce1ad3u, 0x2136a0cfu, 0x1507c0ebu, 0x1eb6e07au,
         0x3a0ccf7bu, 0x38e4bf31u, 0x44128286u, 0x6b05e976u, 0x244a9b92u, 0x6e4b32a8u, 0x78ee2496u,
         0x4761115bu, 0x3d3a7077u, 0x75d3c670u, 0x396a2475u, 0x26dd00b4u, 0x7df50f59u, 0x0cb922dfu,
         0x0568b190u, 0x5bd3fcd6u, 0x1351f58eu},
        {0x52191b5fu, 0x119171b8u, 0x1e8bb727u, 0x27d21f26u, 0x36146613u, 0x1ee817a2u, 0x71abe84eu,
         0x44b88070u, 0x5dc04410u, 0x2aeaa2f6u, 0x2b7bb311u, 0x6906884du, 0x0522e053u, 0x0c45a214u,
         0x1b016998u, 0x479b1052u, 0x3acc89beu, 0x0776021au, 0x7a34a1f5u, 0x70f87911u, 0x2caf9d9eu,
         0x026aff1bu, 0x2c42468eu, 0x67726b45u},
        {0x09b6f53cu, 0x73d76589u, 0x5793eeb0u, 0x29e720f3u, 0x75fc8bdfu, 0x4c2fae0eu, 0x20b41db3u,
         0x7e491510u, 0x2cadef18u, 0x57fc24d6u, 0x4d1ade4au, 0x36bf8e3cu, 0x3511b63cu, 0x64d8476fu,
         0x732ba706u, 0x46634978u, 0x0521c17cu, 0x5ee69212u, 0x3559cba9u, 0x2b33df89u, 0x653538d6u,
         0x5fde8344u, 0x4091605du, 0x2933bddeu},
    };
    std::vector<std::array<KoalaBear, 24>> v;
    v.reserve(4);
    for (const auto& row : D) {
        std::array<KoalaBear, 24> a{};
        for (size_t i = 0; i < 24; ++i) {
            a[i] = KoalaBear(row[i]);
        }
        v.push_back(a);
    }
    return v;
}

inline std::vector<std::array<KoalaBear, 24>> koalabear_poseidon2_rc_24_external_final() {
    static constexpr uint32_t D[4][24] = {
        {0x7d232359u, 0x389d82f9u, 0x259b2e6cu, 0x45a94defu, 0x0d497380u, 0x5b049135u, 0x3c268399u,
         0x78feb2f9u, 0x300a3eecu, 0x505165bbu, 0x20300973u, 0x2327c081u, 0x1a45a2f4u, 0x5b32ea2eu,
         0x2d5d1a70u, 0x053e613eu, 0x5433e39fu, 0x495529f0u, 0x1eaa1aa9u, 0x578f572au, 0x698ede71u,
         0x5a0f9dbau, 0x398a2e96u, 0x0c7b2925u},
        {0x2e6b9564u, 0x026b00deu, 0x7644c1e9u, 0x5c23d0bdu, 0x3470b5efu, 0x6013cf3au, 0x48747288u,
         0x13b7a543u, 0x3eaebd44u, 0x0004e60cu, 0x1e8363a2u, 0x2343259au, 0x69da0c2au, 0x06e3e4c4u,
         0x1095018eu, 0x0deea348u, 0x1f4c5513u, 0x4f9a3a98u, 0x3179112bu, 0x524abb1fu, 0x21615ba2u,
         0x23ab4065u, 0x1202a1d1u, 0x21d25b83u},
        {0x6ed17c2fu, 0x391e6b09u, 0x5e4ed894u, 0x6a2f58f2u, 0x5d980d70u, 0x3fa48c5eu, 0x1f6366f7u,
         0x63540f5fu, 0x6a8235edu, 0x14c12a78u, 0x6edde1c9u, 0x58ce1c22u, 0x718588bbu, 0x334313adu,
         0x7478dbc7u, 0x647ad52fu, 0x39e82049u, 0x6fee146au, 0x082c2f24u, 0x1f093015u, 0x30173c18u,
         0x53f70c0du, 0x6028ab0cu, 0x2f47a1eeu},
        {0x26a6780eu, 0x3540bc83u, 0x1812b49fu, 0x5149c827u, 0x631dd925u, 0x001f2deau, 0x7dc05194u,
         0x3789672eu, 0x7cabf72eu, 0x242dbe2fu, 0x0b07a51du, 0x38653650u, 0x50785c4eu, 0x60e8a7e0u,
         0x07464338u, 0x3482d6e1u, 0x08a69f1eu, 0x3f2aff24u, 0x5814c30du, 0x13fecab2u, 0x61cb291au,
         0x68c8226fu, 0x5c757eeau, 0x289b4e1eu},
    };
    std::vector<std::array<KoalaBear, 24>> v;
    v.reserve(4);
    for (const auto& row : D) {
        std::array<KoalaBear, 24> a{};
        for (size_t i = 0; i < 24; ++i) {
            a[i] = KoalaBear(row[i]);
        }
        v.push_back(a);
    }
    return v;
}

inline std::vector<KoalaBear> koalabear_poseidon2_rc_24_internal() {
    static constexpr uint32_t D[23] = {
        0x1395d4cau, 0x5dbac049u, 0x51fc2727u, 0x13407399u, 0x39ac6953u, 0x45e8726cu, 0x75a7311cu,
        0x599f82c9u, 0x702cf13bu, 0x026b8955u, 0x44e09bbcu, 0x2211207fu, 0x5128b4e3u, 0x591c41afu,
        0x674f5c68u, 0x3981d0d3u, 0x2d82f898u, 0x707cd267u, 0x3b4cca45u, 0x2ad0dc3cu, 0x0cb79b37u,
        0x23f2f4e8u, 0x3de4e739u,
    };
    std::vector<KoalaBear> v;
    v.reserve(23);
    for (uint32_t x : D) {
        v.emplace_back(x);
    }
    return v;
}

inline std::shared_ptr<Poseidon2<KoalaBear, KoalaBear, 16, KOALABEAR_S_BOX_DEGREE>>
default_koalabear_poseidon2_16() {
    ExternalLayerConstants<KoalaBear, 16> ext(
        koalabear_poseidon2_rc_16_external_initial(),
        koalabear_poseidon2_rc_16_external_final()
    );
    auto ext_layer = std::make_shared<GenericPoseidon2External<KoalaBear, KoalaBear, 16, KOALABEAR_S_BOX_DEGREE>>(
        std::move(ext)
    );
    auto int_layer = std::make_shared<KoalaBearPoseidon2Internal<16>>(
        koalabear_poseidon2_rc_16_internal()
    );
    return std::make_shared<Poseidon2<KoalaBear, KoalaBear, 16, KOALABEAR_S_BOX_DEGREE>>(
        ext_layer, int_layer
    );
}

inline std::shared_ptr<Poseidon2<KoalaBear, KoalaBear, 24, KOALABEAR_S_BOX_DEGREE>>
default_koalabear_poseidon2_24() {
    ExternalLayerConstants<KoalaBear, 24> ext(
        koalabear_poseidon2_rc_24_external_initial(),
        koalabear_poseidon2_rc_24_external_final()
    );
    auto ext_layer = std::make_shared<GenericPoseidon2External<KoalaBear, KoalaBear, 24, KOALABEAR_S_BOX_DEGREE>>(
        std::move(ext)
    );
    auto int_layer = std::make_shared<KoalaBearPoseidon2Internal<24>>(
        koalabear_poseidon2_rc_24_internal()
    );
    return std::make_shared<Poseidon2<KoalaBear, KoalaBear, 24, KOALABEAR_S_BOX_DEGREE>>(
        ext_layer, int_layer
    );
}

} // namespace poseidon2
