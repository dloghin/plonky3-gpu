#pragma once

#include "cuda_compat.hpp"
#include "mersenne31.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace monolith {

using p3_field::Mersenne31;

template <size_t WIDTH>
class MonolithMdsMatrixMersenne31 {
public:
    static_assert(WIDTH == 16, "MonolithMdsMatrixMersenne31 currently supports WIDTH=16 only");

    P3_HOST_DEVICE void permute_mut(std::array<Mersenne31, WIDTH>& state) const {
        constexpr std::array<uint32_t, 16> kCirculant = {
            61402u, 17845u, 26798u, 59689u, 12021u, 40901u, 41351u, 27521u,
            56951u, 12034u, 53865u, 43244u, 7454u, 33823u, 28750u, 1108u,
        };
        std::array<Mersenne31, WIDTH> out{};
        for (size_t i = 0; i < WIDTH; ++i) {
            uint64_t acc = 0;
            for (size_t j = 0; j < WIDTH; ++j) {
                acc += static_cast<uint64_t>(kCirculant[(j + WIDTH - i) % WIDTH]) * state[j].value();
            }
            out[i] = Mersenne31(acc);
        }
        state = out;
    }

    P3_HOST_DEVICE std::array<Mersenne31, WIDTH> permute(
        const std::array<Mersenne31, WIDTH>& input
    ) const {
        auto out = input;
        permute_mut(out);
        return out;
    }

private:
};

template <size_t WIDTH = 16, size_t NUM_FULL_ROUNDS = 5>
class MonolithMersenne31 {
public:
    static_assert(WIDTH == 16, "MonolithMersenne31 currently supports WIDTH=16 only");
    static_assert(NUM_FULL_ROUNDS == 5, "MonolithMersenne31 currently supports NUM_FULL_ROUNDS=5 only");

    static constexpr size_t NUM_BARS = 8;
    using State = std::array<Mersenne31, WIDTH>;
    using RoundConstants = std::array<std::array<Mersenne31, WIDTH>, NUM_FULL_ROUNDS>;
    using Mds = MonolithMdsMatrixMersenne31<WIDTH>;

    MonolithMersenne31() = default;

    P3_HOST_DEVICE void permute_mut(State& state) const {
        concrete(state);
        for (size_t r = 0; r < NUM_FULL_ROUNDS; ++r) {
            bars(state);
            bricks(state);
            concrete(state);
            add_round_constants(state, r);
        }
        bars(state);
        bricks(state);
        concrete(state);
    }

    P3_HOST_DEVICE State permute(const State& input) const {
        State out = input;
        permute_mut(out);
        return out;
    }

    P3_HOST static const RoundConstants& round_constants() {
        static const RoundConstants constants = []() {
            RoundConstants out{};
            const auto round_constants_u32_vals = round_constants_u32();
            for (size_t r = 0; r < NUM_FULL_ROUNDS; ++r) {
                for (size_t i = 0; i < WIDTH; ++i) {
                    out[r][i] = Mersenne31(round_constants_u32_vals[r][i]);
                }
            }
            return out;
        }();
        return constants;
    }

private:
    P3_HOST_DEVICE static uint8_t s_box(uint8_t y) {
        const uint8_t y_rot_1 = static_cast<uint8_t>((y << 1) | (y >> 7));
        const uint8_t y_rot_2 = static_cast<uint8_t>((y << 2) | (y >> 6));
        const uint8_t y_rot_3 = static_cast<uint8_t>((y << 3) | (y >> 5));
        const uint8_t tmp = static_cast<uint8_t>(y ^ (static_cast<uint8_t>(~y_rot_1) & y_rot_2 & y_rot_3));
        return static_cast<uint8_t>((tmp << 1) | (tmp >> 7));
    }

    P3_HOST_DEVICE static uint8_t final_s_box(uint8_t y) {
        y &= 0x7fu;
        const uint8_t y_rot_1 = static_cast<uint8_t>(((y >> 6) | (y << 1)) & 0x7fu);
        const uint8_t y_rot_2 = static_cast<uint8_t>(((y >> 5) | (y << 2)) & 0x7fu);
        const uint8_t tmp = static_cast<uint8_t>((y ^ (static_cast<uint8_t>(~y_rot_1) & y_rot_2)) & 0x7fu);
        return static_cast<uint8_t>(((tmp >> 6) | (tmp << 1)) & 0x7fu);
    }

    P3_HOST_DEVICE static Mersenne31 bar(const Mersenne31& element) {
        const uint32_t x = element.value();
        const uint32_t b0 = static_cast<uint32_t>(s_box(static_cast<uint8_t>(x & 0xffu)));
        const uint32_t b1 = static_cast<uint32_t>(s_box(static_cast<uint8_t>((x >> 8) & 0xffu)));
        const uint32_t b2 = static_cast<uint32_t>(s_box(static_cast<uint8_t>((x >> 16) & 0xffu)));
        const uint32_t b3 = static_cast<uint32_t>(final_s_box(static_cast<uint8_t>((x >> 24) & 0x7fu)));
        return Mersenne31((b3 << 24) | (b2 << 16) | (b1 << 8) | b0);
    }

    P3_HOST_DEVICE void concrete(State& state) const { mds_.permute_mut(state); }

    P3_HOST_DEVICE static void bars(State& state) {
        for (size_t i = 0; i < NUM_BARS; ++i) {
            state[i] = bar(state[i]);
        }
    }

    P3_HOST_DEVICE static void bricks(State& state) {
        Mersenne31 prev = state[0];
        for (size_t i = 1; i < WIDTH; ++i) {
            Mersenne31 current = state[i];
            state[i] += prev.square();
            prev = current;
        }
    }

    P3_HOST_DEVICE static void add_round_constants(State& state, size_t round_idx) {
        const auto round_constants_u32_vals = round_constants_u32();
        for (size_t i = 0; i < WIDTH; ++i) {
            state[i] += Mersenne31(round_constants_u32_vals[round_idx][i]);
        }
    }

    P3_HOST_DEVICE static constexpr std::array<std::array<uint32_t, WIDTH>, NUM_FULL_ROUNDS>
    round_constants_u32() {
        return {{
            {{1033436816u, 348863691u, 2081103763u, 994924237u, 64925253u, 677331122u, 1735246508u, 26616398u, 1538025930u, 1710098735u, 995978747u, 1336376181u, 2051827886u, 447361871u, 1829769948u, 718914942u}},
            {{474392908u, 549190350u, 140657697u, 642927328u, 325988066u, 2087527882u, 1429283917u, 537644603u, 2072852575u, 707584548u, 482862777u, 829305883u, 1016581262u, 148132697u, 397768408u, 50011713u}},
            {{897025585u, 597857797u, 389941735u, 1101342757u, 1318622762u, 1954712215u, 1789281623u, 529033351u, 913202249u, 1707514131u, 616819674u, 197082924u, 1180366701u, 241453365u, 1700285697u, 1755996717u}},
            {{1917698553u, 1252360787u, 1273610561u, 212500927u, 1268578595u, 1403584286u, 612974258u, 1024938353u, 1546879084u, 1752198737u, 757476618u, 916242693u, 1739315286u, 1012279900u, 1254788910u, 1865871347u}},
            {{534908981u, 1994856941u, 1598293579u, 510970053u, 1868253334u, 1194878847u, 360986778u, 1303396410u, 337495830u, 1233499389u, 1058246115u, 1413610001u, 799568848u, 48161847u, 1339121921u, 1110912837u}},
        }};
    }

    Mds mds_{};
};

} // namespace monolith
