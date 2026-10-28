// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_tree_migration — the tree side of pressure migration:
// peek_lru_leaf_paths (coldest unpinned leaves as full hash paths) and
// evict_chain (drop a pushed chain up to a branch, a pin or the root).

#include "test_utils.h"
#include "radix_client.h"

#include <cassert>
#include <vector>

using namespace radixshmem;

static std::vector<HashType> hashes_of(std::vector<int64_t> toks) {
    return RadixClient::compute_hashes(toks.data(), (int)toks.size(), 1);
}

static int hit(RadixClient& c, const std::vector<HashType>& h) {
    return c.query(h.data(), (int)h.size(), COMPONENT_MASK_FULL, /*local_only=*/true,
                   /*lock=*/false, /*update_meta=*/false).common_hit;
}

// ==================================================================
static void test_peek_order_paths_and_filters() {
    TEST_BEGIN("peek: coldest first, full root->leaf paths, pins / depth / scan filters");

    ServerLauncher launcher("/test_tm_01", default_cfg());
    RadixClient c("/test_tm_01");

    auto a = hashes_of({1, 2, 3});
    auto b = hashes_of({1, 2, 4});   // splits A: stem [1,2] + leaves [3], [4]
    auto d = hashes_of({9, 8});
    insert_auto(c, a.data(), 3);
    insert_auto(c, b.data(), 3);
    insert_auto(c, d.data(), 2);

    auto p = c.peek_lru_leaf_paths(10, 512);
    assert(p.size() == 3);
    assert(p[0] == a && p[1] == b && p[2] == d);   // whole paths, oldest leaf first

    auto unpin_a = pin_path(c, a.data(), 3);
    p = c.peek_lru_leaf_paths(10, 512);
    assert(p.size() == 2 && p[0] == b && p[1] == d);   // pinned leaf skipped
    unpin_a();

    p = c.peek_lru_leaf_paths(10, 2);
    assert(p.size() == 1 && p[0] == d);                // 3-deep paths skipped
    p = c.peek_lru_leaf_paths(10, 512, /*scan_limit=*/1);
    assert(p.size() == 1 && p[0] == a);                // one leaf visited
    p = c.peek_lru_leaf_paths(1, 512);
    assert(p.size() == 1 && p[0] == a);                // max_leaves

    (void)c.query(a.data(), 3, COMPONENT_MASK_FULL, true);   // touch A
    p = c.peek_lru_leaf_paths(10, 512);
    assert(p.size() == 3 && p[0] == b && p[1] == d && p[2] == a);

    // read-only: peeking touched nothing and pinned nothing
    assert(hit(c, a) == 3 && hit(c, b) == 3 && hit(c, d) == 2);
    expect_healthy(c);
    TEST_PASS();
}

// ==================================================================
static void test_evict_chain_stops_at_branch_then_takes_stem() {
    TEST_BEGIN("evict_chain: stops at a branch; the stem goes with the last child");

    ServerLauncher launcher("/test_tm_02", default_cfg());
    RadixClient c("/test_tm_02");

    auto a = hashes_of({1, 2, 3});
    auto b = hashes_of({1, 2, 4});
    insert_auto(c, a.data(), 3);
    insert_auto(c, b.data(), 3);
    uint32_t free0 = c.mempool_free();

    assert(c.evict_chain(a.data(), 3) == 1);     // leaf [3]; stem keeps child [4]
    assert(hit(c, a) == 2 && hit(c, b) == 3);
    assert(c.evict_chain(b.data(), 3) == 3);     // leaf [4] + childless stem [1,2]
    assert(hit(c, a) == 0 && hit(c, b) == 0);
    assert(c.mempool_free() == free0 + 4);
    assert(c.peek_lru_leaf_paths(10, 512).empty());
    expect_healthy(c);
    TEST_PASS();
}

// ==================================================================
static void test_evict_chain_noops() {
    TEST_BEGIN("evict_chain: pinned / mid-node / interior terminal / unknown path evict nothing");

    ServerLauncher launcher("/test_tm_03", default_cfg());
    RadixClient c("/test_tm_03");

    auto d = hashes_of({5, 6, 7});
    insert_auto(c, d.data(), 3);
    auto unpin = pin_path(c, d.data(), 3);
    assert(c.evict_chain(d.data(), 3) == 0);     // pinned
    assert(hit(c, d) == 3);
    unpin();

    auto e = hashes_of({11, 12, 13, 14});
    insert_auto(c, e.data(), 4);
    assert(c.evict_chain(e.data(), 2) == 0);     // path ends inside the node
    assert(hit(c, e) == 4);

    auto f = hashes_of({21, 22});
    auto g = hashes_of({21, 22, 23});
    insert_auto(c, f.data(), 2);
    insert_auto(c, g.data(), 3);
    assert(c.evict_chain(f.data(), 2) == 0);     // terminal has a child
    assert(hit(c, g) == 3);
    assert(c.evict_chain(g.data(), 3) == 3);     // leaf [23] then [21,22]
    assert(hit(c, f) == 0);

    auto x = hashes_of({99, 98});
    assert(c.evict_chain(x.data(), 2) == 0);     // not in the tree
    assert(c.evict_chain(d.data(), 3) == 3);     // unpinned now
    assert(hit(c, e) == 4);
    expect_healthy(c);
    TEST_PASS();
}

int main() {
    printf("=== test_tree_migration ===\n\n");
    test_peek_order_paths_and_filters();
    test_evict_chain_stops_at_branch_then_takes_stem();
    test_evict_chain_noops();
    return tests_report();
}
