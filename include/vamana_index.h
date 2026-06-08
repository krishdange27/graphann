#pragma once

#include <cstdint>
#include <vector>
#include <mutex>
#include <string>
#include <unordered_map>
#include <atomic>
#include <array>

// Result of a single query search.
struct SearchResult {
    std::vector<uint32_t> ids;   // nearest neighbor IDs (sorted by true distance)
    uint32_t dist_cmps;          // number of distance computations
    double   latency_us;         // search latency in microseconds
    uint32_t hops;               // number of nodes expanded during search
};

// ---------------------------------------------------------------------------
// VamanaIndex
//
// Original algorithm: greedy beam search (greedy_search) + alpha-RNG pruning
// (robust_prune) + parallel incremental build.
//
// Navigation Optimization additions (P2):
//
//   GOAL: make the search smarter over time by learning which edges are
//   genuinely useful for reaching true nearest neighbors.
//
//   EdgeScore { traversals, helpful } — packed into one 8-byte struct so a
//   single map lookup retrieves both counters with one cache miss, not two.
//
//   edge_scores_[shard][key] where shard = (u % NUM_EDGE_SHARDS).
//   NUM_EDGE_SHARDS = 64 independent maps, each with its own mutex.
//   Threads working on different source nodes u almost never share a shard,
//   so contention drops ~64× compared to a single global mutex.
//
//   During greedy_search(), score snapshot acquires only the ONE shard mutex
//   for the current hop's source node — not a global lock.
//
//   After search() returns, edge-score updates touch at most NUM_EDGE_SHARDS
//   different mutexes, acquired one shard at a time (no deadlock possible
//   because we never hold two shard locks simultaneously).
//
//   Cold-start: all scores start at 0.0, so early queries behave identically
//   to the original algorithm. The bonus kicks in gradually as data accumulates.
// ---------------------------------------------------------------------------
class VamanaIndex {
  public:
    VamanaIndex() = default;
    ~VamanaIndex();

    // ---- Build ----
    // entropy_scale: controls strength of Entropy-Based Edge Pruning.
    //   0.0 = pure alpha-RNG (original behavior)
    //   0.3 = recommended default (diversity bonus up to 30% alpha relaxation)
    //   1.0 = strong diversity preference (may reduce recall — use carefully)
    void build(const std::string& data_path, uint32_t R, uint32_t L,
               float alpha, float gamma, float entropy_scale = 0.0f);

    // ---- Search ----
    SearchResult search(const float* query, uint32_t K, uint32_t L) const;

    // ---- Persistence ----
    void save(const std::string& path) const;
    void load(const std::string& index_path, const std::string& data_path);

    // ---- Navigation stats ----
    uint32_t get_npts() const { return npts_; }
    uint32_t get_dim()  const { return dim_;  }

    // Average hops per query across all search() calls since last reset.
    double get_avg_hops() const {
        uint64_t q = total_queries_.load();
        return (q == 0) ? 0.0 : (double)total_hops_.load() / (double)q;
    }

    // Call between experiment runs to clear hop counters.
    void reset_hop_stats() const {
        total_hops_.store(0);
        total_queries_.store(0);
    }

    // Call between experiment runs to clear edge usefulness scores.
    void reset_edge_scores() const {
        for (uint32_t s = 0; s < NUM_EDGE_SHARDS; s++) {
            std::lock_guard<std::mutex> lock(edge_shard_mutex_[s]);
            edge_scores_[s].clear();
        }
    }

    // ---- Adaptive Graph Learning (P3) ----
    //
    // refine_graph() rewrites each node's adjacency list using accumulated
    // edge-usefulness scores.  Called automatically from search() every
    // REFINE_INTERVAL (=1000) queries; may also be called manually.
    //
    // Per node u the algorithm:
    //   1. Reads usefulness(u,v) = helpful/traversals for every outgoing edge.
    //      No division by zero: guarded by traversals > 0 check.
    //   2. Skips nodes where total_traversals < MIN_TRAVERSALS (=10).
    //      Cold nodes have no reliable signal — skipping avoids removing
    //      valid edges based on statistically meaningless zeros.
    //   3. Computes per-node adaptive thresholds via percentile interpolation:
    //        T_remove = 30th-percentile of u's usefulness distribution
    //        T_boost  = 80th-percentile of u's usefulness distribution
    //   4. Classifies edges:
    //        score > T_boost   → duplicated up to MAX_BOOST_COPIES (=2) times.
    //        T_remove ≤ score  → kept once.
    //        score < T_remove  → dropped.
    //   5. Sorts list descending by usefulness; boosted bucket first.
    //   6. Trims to at most R_ neighbours AFTER sorting (degree cap).
    //   7. Rescues single best edge if list would otherwise be empty.
    //   8. Skips write-back if list is unchanged (avoids cache thrashing).
    //   9. Prints one log line before loop and one summary after (observability).
    //
    // Thread safety — locks never held simultaneously:
    //   acquire shard_mutex_[shard_of(u)]  → read scores  → release
    //   acquire locks_[u]                  → write graph_  → release
    //
    // Declared const because it is called from search() const.
    // graph_ and R_ are mutable to allow writes/reads from const context.
    void refine_graph() const;

  private:
    // Candidate = (distance, node_id).  std::set orders by distance first.
    using Candidate = std::pair<float, uint32_t>;

    // ---- Core algorithms ----

    // Greedy beam search from start_node_.
    //
    // Returns:
    //   candidates       — up to L candidates sorted by TRUE distance
    //   dist_cmps        — distance computations performed
    //   hops             — nodes expanded (each expansion = one hop)
    //   traversed_edges  — every (u, v) edge followed, in traversal order
    //
    // The candidate set stores TRUE distances throughout — no bonus is baked
    // into the stored values.  The priority bonus is applied only to the
    // threshold comparison when deciding whether to insert a new candidate,
    // so the distances returned are always correct for recall evaluation.
    std::tuple<
        std::vector<Candidate>,
        uint32_t,
        uint32_t,
        std::vector<std::pair<uint32_t, uint32_t>>
    >
    greedy_search(const float* query, uint32_t L) const;

    // Alpha-RNG pruning with optional entropy-based diversity bonus.
    //
    // entropy_scale > 0 activates Entropy-Based Edge Pruning (P3):
    //   For each candidate c, we compute its diversity relative to already-
    //   selected neighbors S as:
    //
    //     diversity(c, S) = min_{n in S} dist(c, n)
    //
    //   This is normalized by dist(node, c) to make it scale-invariant:
    //
    //     norm_diversity(c) = diversity(c, S) / dist(node, c)
    //
    //   The effective pruning threshold for c is then relaxed:
    //
    //     alpha_eff(c) = alpha * (1 + entropy_scale * norm_diversity(c))
    //
    //   A more diverse candidate gets a higher alpha_eff, making the α-RNG
    //   condition harder to trigger → diverse edges are less likely to be pruned.
    //   entropy_scale = 0.0 reproduces the original α-RNG exactly.
    void robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                      float alpha, uint32_t R, float entropy_scale = 0.0f);

    // ---- Data ----
    float*   data_      = nullptr;
    uint32_t npts_      = 0;
    uint32_t dim_       = 0;
    bool     owns_data_ = false;

    // ---- Build parameters (stored for reference / save) ----
    float entropy_scale_ = 0.0f;   // P1: entropy pruning strength (0 = disabled)

    // Maximum out-degree cap.  Set in build() from the R parameter, and
    // derived from the max observed degree in load().  refine_graph() uses
    // this to enforce the same degree bound after every refinement pass.
    // Declared mutable so refine_graph() const can read/write it.
    mutable uint32_t R_ = 32;      // default matches typical build default

    // ---- Graph ----
    // Declared mutable so that refine_graph() const can rewrite adjacency
    // lists.  refine_graph() is called from search() const (the P3 trigger),
    // and must therefore be const itself.  The mutable qualifier is correct
    // here: graph_ is logically part of the object's observable state only
    // in that its structure affects search quality, not its identity — callers
    // already accept that search() has the side-effect of updating edge scores.
    mutable std::vector<std::vector<uint32_t>> graph_;
    uint32_t start_node_ = 0;

    // ---- Concurrency: per-node locks for parallel build ----
    mutable std::vector<std::mutex> locks_;

    // ---- Navigation optimization: sharded edge scoring ----
    //
    // WHY SHARDING?
    //   A single global mutex serializes ALL concurrent queries on every
    //   score read (greedy_search) and every score write (search).  With
    //   T threads each doing ~hops reads + ~edges writes per query, the
    //   single mutex becomes the bottleneck — queries queue up waiting.
    //
    //   We partition edge-space into NUM_EDGE_SHARDS independent buckets.
    //   Edge (u→v) belongs to shard (u % NUM_EDGE_SHARDS).  Each shard has
    //   its own mutex, so threads working on different source nodes u are
    //   almost never in the same shard → contention drops ~64×.
    //
    // WHY MERGED EdgeScore STRUCT?
    //   The original design stored traversals and helpful in two separate
    //   unordered_maps.  Every edge update required two independent hash
    //   lookups: one per map.  With the merged struct, ONE lookup retrieves
    //   both counters.  The struct is 8 bytes — fits in a single cache line
    //   alongside the key, halving cache misses during dense update loops.

    static constexpr uint32_t NUM_EDGE_SHARDS = 64;

    // Packed traversal + helpful counters for one directed edge.
    struct EdgeScore {
        uint32_t traversals = 0;
        uint32_t helpful    = 0;
    };

    // Encodes directed edge (u→v) as a uint64 key within a shard.
    // (u is NOT encoded in the key because it's already implicit in the shard.)
    static uint32_t shard_of(uint32_t u) { return u % NUM_EDGE_SHARDS; }
    static uint64_t edge_key(uint32_t u, uint32_t v) {
        return ((uint64_t)u << 32) | (uint64_t)v;
    }

    mutable std::array<std::unordered_map<uint64_t, EdgeScore>,
                       NUM_EDGE_SHARDS>   edge_scores_;
    mutable std::array<std::mutex,
                       NUM_EDGE_SHARDS>   edge_shard_mutex_;

    // ---- Navigation optimization: hop tracking ----
    mutable std::atomic<uint64_t> total_hops_{0};
    mutable std::atomic<uint64_t> total_queries_{0};

    // ---- Helpers ----
    const float* get_vector(uint32_t id) const {
        return data_ + (size_t)id * dim_;
    }

    // Returns usefulness score [0.0, 1.0] for edge (u→v).
    // Returns 0.0 if edge has never been traversed (cold start = no bias).
    // CALLER MUST HOLD edge_shard_mutex_[shard_of(u)].
    float edge_usefulness(uint32_t u, uint32_t v) const {
        auto& shard = edge_scores_[shard_of(u)];
        auto it = shard.find(edge_key(u, v));
        if (it == shard.end() || it->second.traversals == 0)
            return 0.0f;
        return (float)it->second.helpful / (float)it->second.traversals;
    }
};
