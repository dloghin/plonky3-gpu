/**
 * @brief KoalaBear field, MDS, and Poseidon2 tests (vectors from p3-koala-bear).
 */

#include "koala_bear.hpp"
#include "mds_matrix.hpp"
#include "poseidon2_koalabear.hpp"
#include "round_numbers.hpp"

#include <gtest/gtest.h>

using p3_field::KoalaBear;
using p3_poseidon::MdsMatrixKoalaBear8;
using p3_poseidon::MdsMatrixKoalaBear12;
using p3_poseidon::MdsMatrixKoalaBear16;

namespace {

TEST(KoalaBearField, PrimeAndConstants) {
    EXPECT_EQ(KoalaBear::PRIME, 0x7f000001u);
    EXPECT_EQ(KoalaBear::TWO_ADICITY, 24u);
}

TEST(KoalaBearField, MontgomeryRoundTrip) {
    for (uint32_t c : {0u, 1u, 2u, 100u, 0x34167c58u, KoalaBear::PRIME - 1u}) {
        KoalaBear x(c);
        EXPECT_EQ(x.as_canonical_u64(), static_cast<uint64_t>(c));
    }
}

TEST(KoalaBearField, RustLibRsVectors) {
    KoalaBear f(100u);
    EXPECT_EQ(f.as_canonical_u64(), 100u);

    KoalaBear m1(0x34167c58u);
    KoalaBear m2(0x61f3207bu);
    KoalaBear expected_prod(0x54b46b81u);
    EXPECT_EQ(m1 * m2, expected_prod);

    EXPECT_EQ(m1.template injective_exp_n<3>().template injective_exp_root_n<3>(), m1);
    EXPECT_EQ(m2.template injective_exp_n<3>().template injective_exp_root_n<3>(), m2);
    EXPECT_EQ(KoalaBear::two_val().template injective_exp_n<3>().template injective_exp_root_n<3>(),
              KoalaBear::two_val());
}

TEST(KoalaBearField, TwoAdicGenerators) {
    KoalaBear base(0x6ac49f88u);
    for (size_t bits = 0; bits <= KoalaBear::TWO_ADICITY; ++bits) {
        KoalaBear g = KoalaBear::two_adic_generator(bits);
        KoalaBear acc = base;
        for (size_t i = 0; i < KoalaBear::TWO_ADICITY - bits; ++i) {
            acc = acc.square();
        }
        EXPECT_EQ(g, acc) << "bits=" << bits;
    }
}

TEST(KoalaBearField, TwoAdicOrder) {
    KoalaBear one = KoalaBear::one_val();
    for (size_t k = 1; k <= KoalaBear::TWO_ADICITY; ++k) {
        KoalaBear g = KoalaBear::two_adic_generator(k);
        KoalaBear p = g;
        for (size_t i = 0; i < k - 1; ++i) {
            p = p.square();
        }
        EXPECT_NE(p, one);
        p = p.square();
        EXPECT_EQ(p, one);
    }
}

TEST(KoalaBearField, Inverse) {
    KoalaBear a(12345u);
    EXPECT_EQ(a * a.inv(), KoalaBear::one_val());
}

TEST(KoalaBearPoseidon2, RoundNumbers128) {
    auto r16 = poseidon2::poseidon2_round_numbers_128(16, 3, KoalaBear::PRIME);
    EXPECT_EQ(r16.first, 8u);
    EXPECT_EQ(r16.second, 20u);

    auto r24 = poseidon2::poseidon2_round_numbers_128(24, 3, KoalaBear::PRIME);
    EXPECT_EQ(r24.first, 8u);
    EXPECT_EQ(r24.second, 23u);
}

TEST(KoalaBearMds, Width8) {
    MdsMatrixKoalaBear8<KoalaBear> mds;
    std::array<KoalaBear, 8> input = {
        KoalaBear(391474477u),  KoalaBear(1174409341u), KoalaBear(666967492u),
        KoalaBear(1852498830u), KoalaBear(1801235316u), KoalaBear(820595865u),
        KoalaBear(585587525u),  KoalaBear(1348326858u),
    };
    auto out = mds.permute(input);
    std::array<KoalaBear, 8> expected = {
        KoalaBear(947631349u),  KoalaBear(1348484024u), KoalaBear(1002291099u),
        KoalaBear(1962469348u), KoalaBear(831049401u),   KoalaBear(1648283812u),
        KoalaBear(1017255940u), KoalaBear(589556689u),
    };
    EXPECT_EQ(out, expected);
}

TEST(KoalaBearMds, Width12) {
    MdsMatrixKoalaBear12<KoalaBear> mds;
    std::array<KoalaBear, 12> input = {
        KoalaBear(918423259u),  KoalaBear(673549090u),  KoalaBear(364157140u),
        KoalaBear(9832898u),    KoalaBear(493922569u),  KoalaBear(1171855651u),
        KoalaBear(246075034u),  KoalaBear(1542167926u), KoalaBear(1787615541u),
        KoalaBear(1696819900u), KoalaBear(1884530130u), KoalaBear(422386768u),
    };
    auto out = mds.permute(input);
    std::array<KoalaBear, 12> expected = {
        KoalaBear(3672342u),    KoalaBear(689021900u),  KoalaBear(1455700352u),
        KoalaBear(1687414333u), KoalaBear(1231524540u), KoalaBear(1572686242u),
        KoalaBear(42253424u),   KoalaBear(696666080u),  KoalaBear(950244312u),
        KoalaBear(678673484u),  KoalaBear(530048499u),  KoalaBear(135761510u),
    };
    EXPECT_EQ(out, expected);
}

TEST(KoalaBearMds, Width16) {
    MdsMatrixKoalaBear16<KoalaBear> mds;
    std::array<KoalaBear, 16> input = {
        KoalaBear(1983708094u), KoalaBear(1477844074u), KoalaBear(1638775686u),
        KoalaBear(98517138u),   KoalaBear(70746308u),   KoalaBear(968700066u),
        KoalaBear(275567720u),  KoalaBear(1359144511u), KoalaBear(960499489u),
        KoalaBear(1215199187u), KoalaBear(474302783u),  KoalaBear(79320256u),
        KoalaBear(1923147803u), KoalaBear(1197733438u), KoalaBear(1638511323u),
        KoalaBear(303948902u),
    };
    auto out = mds.permute(input);
    std::array<KoalaBear, 16> expected = {
        KoalaBear(54729128u),   KoalaBear(2128589920u), KoalaBear(81963306u),
        KoalaBear(842781423u),  KoalaBear(59798772u),   KoalaBear(1955488131u),
        KoalaBear(274677035u),  KoalaBear(372631613u),  KoalaBear(1610234661u),
        KoalaBear(608093248u),  KoalaBear(1204230235u), KoalaBear(1081779929u),
        KoalaBear(873712545u),  KoalaBear(436245025u),  KoalaBear(339463618u),
        KoalaBear(255045423u),
    };
    EXPECT_EQ(out, expected);
}

TEST(KoalaBearPoseidon2, DefaultWidth16) {
    auto perm = poseidon2::default_koalabear_poseidon2_16();
    std::array<KoalaBear, 16> state = {
        KoalaBear(894848333u),  KoalaBear(1437655012u), KoalaBear(1200606629u),
        KoalaBear(1690012884u), KoalaBear(71131202u),   KoalaBear(1749206695u),
        KoalaBear(1717947831u), KoalaBear(120589055u),  KoalaBear(19776022u),
        KoalaBear(42382981u),   KoalaBear(1831865506u), KoalaBear(724844064u),
        KoalaBear(171220207u),  KoalaBear(1299207443u), KoalaBear(227047920u),
        KoalaBear(1783754913u),
    };
    perm->permute_mut(state);
    std::array<KoalaBear, 16> expected = {
        KoalaBear(1934285469u), KoalaBear(604889435u),  KoalaBear(133449501u),
        KoalaBear(1026180808u), KoalaBear(1830659359u), KoalaBear(176667110u),
        KoalaBear(1391183747u), KoalaBear(351743874u),  KoalaBear(1238264085u),
        KoalaBear(1292768839u), KoalaBear(2023573270u), KoalaBear(1201586780u),
        KoalaBear(1360691759u), KoalaBear(1230682461u), KoalaBear(748270449u),
        KoalaBear(651545025u),
    };
    EXPECT_EQ(state, expected);
}

TEST(KoalaBearPoseidon2, DefaultWidth24) {
    auto perm = poseidon2::default_koalabear_poseidon2_24();
    std::array<KoalaBear, 24> state = {
        KoalaBear(886409618u),  KoalaBear(1327899896u), KoalaBear(1902407911u),
        KoalaBear(591953491u),  KoalaBear(648428576u),  KoalaBear(1844789031u),
        KoalaBear(1198336108u), KoalaBear(355597330u),  KoalaBear(1799586834u),
        KoalaBear(59617783u),   KoalaBear(790334801u),  KoalaBear(1968791836u),
        KoalaBear(559272107u),  KoalaBear(31054313u),   KoalaBear(1042221543u),
        KoalaBear(474748436u),  KoalaBear(135686258u),  KoalaBear(263665994u),
        KoalaBear(1962340735u), KoalaBear(1741539604u), KoalaBear(2026927696u),
        KoalaBear(449439011u),  KoalaBear(1131357108u), KoalaBear(50869465u),
    };
    perm->permute_mut(state);
    std::array<KoalaBear, 24> expected = {
        KoalaBear(382801106u),  KoalaBear(82839311u),   KoalaBear(1503190615u),
        KoalaBear(1987418517u), KoalaBear(854076995u),  KoalaBear(1862291425u),
        KoalaBear(262755189u),  KoalaBear(1050814217u), KoalaBear(722724562u),
        KoalaBear(741265943u),  KoalaBear(1026879332u), KoalaBear(754316749u),
        KoalaBear(1966025564u), KoalaBear(1518878196u), KoalaBear(502200188u),
        KoalaBear(1368172258u), KoalaBear(845459257u),  KoalaBear(1711434837u),
        KoalaBear(724453836u),  KoalaBear(171032289u),  KoalaBear(655223446u),
        KoalaBear(1098636135u), KoalaBear(407832555u),  KoalaBear(1707498914u),
    };
    EXPECT_EQ(state, expected);
}

} // namespace
