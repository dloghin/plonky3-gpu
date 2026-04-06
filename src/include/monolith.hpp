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
    static constexpr std::array<uint32_t, 16> kCirculant = {
        61402u, 17845u, 26798u, 59689u, 12021u, 40901u, 41351u, 27521u,
        56951u, 12034u, 53865u, 43244u, 7454u, 33823u, 28750u, 1108u,
    };
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
            add_round_constants(state, round_constants()[r]);
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

    P3_HOST_DEVICE static const RoundConstants& round_constants() {
        static const RoundConstants constants = {{
            {{Mersenne31(1033436816u), Mersenne31(348863691u), Mersenne31(2081103763u), Mersenne31(994924237u), Mersenne31(64925253u), Mersenne31(677331122u), Mersenne31(1735246508u), Mersenne31(26616398u), Mersenne31(1538025930u), Mersenne31(1710098735u), Mersenne31(995978747u), Mersenne31(1336376181u), Mersenne31(2051827886u), Mersenne31(447361871u), Mersenne31(1829769948u), Mersenne31(718914942u)}},
            {{Mersenne31(474392908u), Mersenne31(549190350u), Mersenne31(140657697u), Mersenne31(642927328u), Mersenne31(325988066u), Mersenne31(2087527882u), Mersenne31(1429283917u), Mersenne31(537644603u), Mersenne31(2072852575u), Mersenne31(707584548u), Mersenne31(482862777u), Mersenne31(829305883u), Mersenne31(1016581262u), Mersenne31(148132697u), Mersenne31(397768408u), Mersenne31(50011713u)}},
            {{Mersenne31(897025585u), Mersenne31(597857797u), Mersenne31(389941735u), Mersenne31(1101342757u), Mersenne31(1318622762u), Mersenne31(1954712215u), Mersenne31(1789281623u), Mersenne31(529033351u), Mersenne31(913202249u), Mersenne31(1707514131u), Mersenne31(616819674u), Mersenne31(197082924u), Mersenne31(1180366701u), Mersenne31(241453365u), Mersenne31(1700285697u), Mersenne31(1755996717u)}},
            {{Mersenne31(1917698553u), Mersenne31(1252360787u), Mersenne31(1273610561u), Mersenne31(212500927u), Mersenne31(1268578595u), Mersenne31(1403584286u), Mersenne31(612974258u), Mersenne31(1024938353u), Mersenne31(1546879084u), Mersenne31(1752198737u), Mersenne31(757476618u), Mersenne31(916242693u), Mersenne31(1739315286u), Mersenne31(1012279900u), Mersenne31(1254788910u), Mersenne31(1865871347u)}},
            {{Mersenne31(534908981u), Mersenne31(1994856941u), Mersenne31(1598293579u), Mersenne31(510970053u), Mersenne31(1868253334u), Mersenne31(1194878847u), Mersenne31(360986778u), Mersenne31(1303396410u), Mersenne31(337495830u), Mersenne31(1233499389u), Mersenne31(1058246115u), Mersenne31(1413610001u), Mersenne31(799568848u), Mersenne31(48161847u), Mersenne31(1339121921u), Mersenne31(1110912837u)}},
        }};
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
        y &= 0x7Fu;
        const uint8_t y_rot_1 = static_cast<uint8_t>(((y >> 6) | (y << 1)) & 0x7Fu);
        const uint8_t y_rot_2 = static_cast<uint8_t>(((y >> 5) | (y << 2)) & 0x7Fu);
        const uint8_t tmp = static_cast<uint8_t>((y ^ (static_cast<uint8_t>(~y_rot_1) & y_rot_2)) & 0x7Fu);
        return static_cast<uint8_t>(((tmp >> 6) | (tmp << 1)) & 0x7Fu);
    }

    P3_HOST_DEVICE static Mersenne31 bar(const Mersenne31& element) {
        const uint32_t x = element.value();
        const uint32_t b0 = static_cast<uint32_t>(s_box(static_cast<uint8_t>(x & 0xFFu)));
        const uint32_t b1 = static_cast<uint32_t>(s_box(static_cast<uint8_t>((x >> 8) & 0xFFu)));
        const uint32_t b2 = static_cast<uint32_t>(s_box(static_cast<uint8_t>((x >> 16) & 0xFFu)));
        const uint32_t b3 = static_cast<uint32_t>(final_s_box(static_cast<uint8_t>((x >> 24) & 0x7Fu)));
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

    P3_HOST_DEVICE static void add_round_constants(
        State& state,
        const std::array<Mersenne31, WIDTH>& round_constants
    ) {
        for (size_t i = 0; i < WIDTH; ++i) {
            state[i] += round_constants[i];
        }
    }

    Mds mds_{};
};

} // namespace monolith
