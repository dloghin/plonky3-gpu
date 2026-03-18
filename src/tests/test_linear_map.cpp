#include <gtest/gtest.h>
#include "p3_util/linear_map.hpp"

using namespace p3_util;

TEST(LinearMapTest, InsertAndGet) {
    LinearMap<int, std::string> map;
    map.insert(1, "one");
    map.insert(2, "two");

    auto* v1 = map.get(1);
    auto* v2 = map.get(2);
    auto* v3 = map.get(3);

    ASSERT_NE(v1, nullptr);
    ASSERT_NE(v2, nullptr);
    EXPECT_EQ(nullptr, v3);

    EXPECT_EQ(*v1, "one");
    EXPECT_EQ(*v2, "two");
}

TEST(LinearMapTest, InsertOverwrites) {
    LinearMap<int, int> map;
    map.insert(42, 100);
    map.insert(42, 200);

    EXPECT_EQ(map.size(), 1u);
    auto* v = map.get(42);
    ASSERT_NE(v, nullptr);
    EXPECT_EQ(*v, 200);
}

TEST(LinearMapTest, GetMut) {
    LinearMap<std::string, int> map;
    map.insert("x", 5);

    auto* mut_v = map.get_mut("x");
    ASSERT_NE(mut_v, nullptr);
    *mut_v = 99;

    EXPECT_EQ(*map.get("x"), 99);
}

TEST(LinearMapTest, GetOrInsertWith_Present) {
    LinearMap<int, int> map;
    map.insert(1, 10);

    int calls = 0;
    auto& v = map.get_or_insert_with(1, [&]() { ++calls; return 999; });

    EXPECT_EQ(v, 10);
    EXPECT_EQ(calls, 0);  // factory not called
}

TEST(LinearMapTest, GetOrInsertWith_Absent) {
    LinearMap<int, int> map;

    int calls = 0;
    auto& v = map.get_or_insert_with(7, [&]() { ++calls; return 42; });

    EXPECT_EQ(v, 42);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(map.size(), 1u);
}

TEST(LinearMapTest, Values) {
    LinearMap<int, int> map;
    map.insert(1, 10);
    map.insert(2, 20);
    map.insert(3, 30);

    auto vals = map.values();
    ASSERT_EQ(vals.size(), 3u);

    int sum = 0;
    for (auto* p : vals) sum += *p;
    EXPECT_EQ(sum, 60);
}

TEST(LinearMapTest, Keys) {
    LinearMap<int, int> map;
    map.insert(5, 0);
    map.insert(10, 0);

    auto keys = map.keys();
    ASSERT_EQ(keys.size(), 2u);
    EXPECT_EQ(*keys[0], 5);
    EXPECT_EQ(*keys[1], 10);
}

TEST(LinearMapTest, Iteration) {
    LinearMap<int, int> map;
    map.insert(1, 100);
    map.insert(2, 200);

    int key_sum = 0, val_sum = 0;
    for (const auto& [k, v] : map) {
        key_sum += k;
        val_sum += v;
    }
    EXPECT_EQ(key_sum, 3);
    EXPECT_EQ(val_sum, 300);
}

TEST(LinearMapTest, EmptyMap) {
    LinearMap<int, int> map;
    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0u);
    EXPECT_EQ(map.get(0), nullptr);
}

TEST(LinearMapTest, MultipleTypes) {
    LinearMap<std::string, double> map;
    map.insert("pi", 3.14159);
    map.insert("e", 2.71828);

    auto* pi = map.get("pi");
    ASSERT_NE(pi, nullptr);
    EXPECT_NEAR(*pi, 3.14159, 1e-5);
}
