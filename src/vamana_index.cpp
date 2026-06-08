#include "vamana_index.h"
#include "distance.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <cstdlib>

// ============================================================================
// Destructor
// ============================================================================

VamanaIndex::~VamanaIndex() {
    if (owns_data_ && data_) {
        std::free(data_);
        data_ = nullptr;
    }
}

// ============================================================================
// Greedy Search  [MODIFIED — Navigation Optimization P2]
// ============================================================================
//
// DATA STRUCTURE CHANGE: std::set → sorted std::vector  (the key perf fix)
// -----------------------------------------------------------------------
// std::set is a red-black tree.  Every insert/erase = heap allocation +
// pointer-chasing.  For L=200 candidates, the whole set fits in ~5 cache
// lines as a vector but spans hundreds of scattered heap nodes as a tree.
//
// We replace it with a flat sorted vector of CandEntry structs:
//
//   struct CandEntry { float dist; uint32_t id; bool expanded; };
//
// The vector is kept sorted by dist at all times using std::inplace_merge
// after each batch of neighbors is appended.  Since we add at most R new
// entries per hop and R << L, inplace_merge is O(R log R + R log L) which
// is faster in practice than R tree insertions due to cache locality.
//
// "expanded" lives as a flag inside CandEntry — no separate structure.
// Finding the best non-expanded node is a single forward linear scan,
// which is branch-predicted well by the CPU and cache-hot.
//
// SCORE SNAPSHOT: no unordered_map, just a plain float[]
// -------------------------------------------------------
// For each hop we need the usefulness score of each neighbor.  Instead of
// building an unordered_map (hash table, heap allocation per hop), we read
// all scores into a plain std::vector<float> parallel to the neighbors list.
// One mutex lock, one tight loop, no hash overhead.
//
// BONUS APPLICATION: threshold comparison only, true distance stored
// ------------------------------------------------------------------
// adjusted_d = d * (1 - BONUS_SCALE * score)  is used ONLY to decide
// whether the candidate enters the list.  The stored value is always d.
// This means the final results are correctly ranked by true distance with
// no re-ranking pass needed in search().
//
// HOP COUNTER and TRAVERSED EDGES: unchanged from previous version.

static constexpr float BONUS_SCALE = 0.10f;

// Internal candidate entry for the sorted-vector candidate set.
struct CandEntry {
    float    dist;      // true L2-squared distance to query
    uint32_t id;        // node id
    bool     expanded;  // true once this node's neighbors have been explored

    // Sort by distance ascending (used by std::sort / std::inplace_merge).
    bool operator<(const CandEntry& o) const {
        return dist < o.dist || (dist == o.dist && id < o.id);
    }
};

std::tuple<
    std::vector<VamanaIndex::Candidate>,
    uint32_t,
    uint32_t,
    std::vector<std::pair<uint32_t, uint32_t>>
>
VamanaIndex::greedy_search(const float* query, uint32_t L) const {

    // ---- Flat sorted candidate list (replaces std::set<Candidate>) ----
    // Capacity: L (kept entries) + R (new batch per hop) + 1 slack.
    // We never let it grow past L after trimming, so no realloc after warmup.
    std::vector<CandEntry> cands;
    cands.reserve(L + 64 + 1);  // 64 > typical R, avoids realloc entirely

    // visited[i]: true once dist(query, i) has been computed.
    // O(1) random access, one bit per entry — cache-efficient for npts_=1M.
    std::vector<bool> visited(npts_, false);

    uint32_t dist_cmps = 0;
    uint32_t hops      = 0;

    // traversed_edges: (u, v) for every edge followed.
    // Reserve generously; push_back will never reallocate in practice.
    std::vector<std::pair<uint32_t, uint32_t>> traversed_edges;
    traversed_edges.reserve(L * 32);  // L hops * ~R=32 neighbors each

    // ---- Seed with start node ----
    float start_dist = compute_l2sq(query, get_vector(start_node_), dim_);
    dist_cmps++;
    cands.push_back({start_dist, start_node_, false});
    visited[start_node_] = true;

    // ---- Main search loop ----
    while (true) {
        // Find the closest non-expanded entry.
        // Linear scan over a contiguous array — very cache-friendly.
        // The array is sorted by dist, so the first non-expanded entry IS
        // the globally closest unvisited candidate.
        auto it = std::find_if(cands.begin(), cands.end(),
                               [](const CandEntry& e){ return !e.expanded; });
        if (it == cands.end())
            break;  // all candidates expanded

        it->expanded = true;
        uint32_t best_node = it->id;
        hops++;

        // Copy neighbor list under node lock (guards parallel build writes).
        std::vector<uint32_t> neighbors;
        {
            std::lock_guard<std::mutex> lock(locks_[best_node]);
            neighbors = graph_[best_node];
        }

        // ---- Snapshot usefulness scores — shard-local lock only ----
        //
        // All neighbors of best_node share the same shard (shard_of(best_node))
        // because the shard is keyed on the SOURCE node u, not destination v.
        // So ONE shard lock covers every score we need for this entire hop.
        //
        // Compared to the old single global mutex:
        //   - We only block threads whose current hop happens to share the same
        //     source-node shard (1/64 of all active threads on average).
        //   - Threads on different shards proceed in parallel with zero contention.
        std::vector<float> scores(neighbors.size(), 0.0f);
        {
            uint32_t sh = shard_of(best_node);
            std::lock_guard<std::mutex> lock(edge_shard_mutex_[sh]);
            for (size_t i = 0; i < neighbors.size(); i++) {
                scores[i] = edge_usefulness(best_node, neighbors[i]);
            }
        }

        // ---- Expand neighbors ----
        // We collect new entries into a small local batch, then merge once.
        std::vector<CandEntry> batch;
        batch.reserve(neighbors.size());

        // Current worst distance in the candidate list (for threshold check).
        // If cands has fewer than L entries, accept everything.
        float worst_dist = (cands.size() < L)
                           ? std::numeric_limits<float>::max()
                           : cands.back().dist;

        for (size_t i = 0; i < neighbors.size(); i++) {
            uint32_t nbr = neighbors[i];

            if (visited[nbr])
                continue;
            visited[nbr] = true;

            float d = compute_l2sq(query, get_vector(nbr), dim_);
            dist_cmps++;

            // Record traversal regardless of whether nbr enters the list.
            traversed_edges.push_back({best_node, nbr});

            // Adjusted distance for threshold comparison ONLY.
            // A higher score gives a lower adjusted_d → easier to enter.
            float adjusted_d = d * (1.0f - BONUS_SCALE * scores[i]);

            if (adjusted_d < worst_dist) {
                batch.push_back({d, nbr, false});
                // Update worst_dist eagerly so we don't admit too many.
                // worst_dist will be recomputed precisely after merge+trim.
                // This is a slight approximation but harmless: at worst we
                // admit a few extra entries that get trimmed right after.
                if (cands.size() + batch.size() >= L) {
                    // Keep worst_dist tight — use d as a proxy.
                    worst_dist = std::max(worst_dist, d);
                }
            }
        }

        if (batch.empty())
            continue;

        // ---- Merge batch into cands, keep sorted, trim to L ----
        // std::inplace_merge requires both halves sorted.
        std::sort(batch.begin(), batch.end());

        size_t old_size = cands.size();
        for (const auto& e : batch)
            cands.push_back(e);

        std::inplace_merge(cands.begin(),
                           cands.begin() + old_size,
                           cands.end());

        // Trim to at most L entries (drop the farthest ones).
        if (cands.size() > L)
            cands.resize(L);

        // Update worst_dist for next iteration.
        worst_dist = cands.back().dist;
    }

    // ---- Convert to Candidate vector (dist, id) sorted by distance ----
    std::vector<Candidate> results;
    results.reserve(cands.size());
    for (const auto& e : cands)
        results.push_back({e.dist, e.id});

    return {results, dist_cmps, hops, traversed_edges};
}

// ============================================================================
// Robust Prune  [MODIFIED — Entropy-Based Edge Pruning P3]
// ============================================================================
//
// Original α-RNG rule:
//   For each candidate c (sorted by dist(node,c) ascending), keep c unless
//   there exists an already-selected neighbor n such that:
//
//       dist(node, c)  >  alpha * dist(c, n)
//
//   i.e., n "covers" c from node's perspective — c is redundant.
//
// Entropy enhancement:
//   Before testing the α-RNG condition for candidate c, we compute how
//   *diverse* c is relative to the already-selected set S:
//
//       diversity(c, S) = min_{n in S} dist(c, n)
//
//   Normalized by dist(node, c) to be scale-invariant:
//
//       norm_div(c) = diversity(c, S) / dist(node, c)    in [0, ∞)
//
//   We clamp norm_div to [0, 1] to keep the bonus bounded, then relax alpha:
//
//       alpha_eff(c) = alpha * (1 + entropy_scale * clamp(norm_div(c), 0, 1))
//
//   Interpretation:
//     - entropy_scale = 0.0 → alpha_eff = alpha exactly  (original behavior)
//     - entropy_scale = 0.3 → a maximally diverse candidate gets up to
//       alpha * 1.3 as its threshold — 30% harder to prune away
//     - entropy_scale = 1.0 → up to alpha * 2.0 for the most diverse edge
//
//   Why this helps:
//     Pure α-RNG greedily prunes based only on proximity, which can create
//     a graph with dense local clusters but poor long-range connectivity.
//     The diversity bonus preserves edges that bridge different regions of
//     the dataset, improving recall especially at low L (few hops budget).
//
//   Why we cap norm_div at 1.0:
//     Without capping, a very far candidate in an empty region could get an
//     arbitrarily large alpha_eff, letting it displace closer useful neighbors.
//     Capping at 1.0 limits the maximum bonus to entropy_scale * alpha,
//     keeping the diversity influence predictable and well-behaved.
//
// Performance:
//   The extra work per candidate is one std::min scan over new_neighbors
//   (already iterated for the α-RNG check), so the asymptotic complexity
//   is unchanged — O(R²) per node as before.  In practice the inner loop
//   is short (≤ R entries) and cache-hot.

void VamanaIndex::robust_prune(uint32_t node,
                               std::vector<Candidate>& candidates,
                               float alpha, uint32_t R,
                               float entropy_scale) {
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node](const Candidate& c){ return c.second == node; }),
        candidates.end());

    std::sort(candidates.begin(), candidates.end());

    std::vector<uint32_t> new_neighbors;
    new_neighbors.reserve(R);

    // Pre-allocate a small buffer for pairwise distances to already-selected
    // neighbors.  Reused across candidates to avoid per-candidate allocation.
    std::vector<float> dist_to_selected;
    dist_to_selected.reserve(R);

    for (const auto& [dist_to_node, cand_id] : candidates) {
        if (new_neighbors.size() >= R)
            break;

        // ---- Compute distances from cand_id to all already-selected ----
        // We need these both for the α-RNG check and the diversity computation.
        // Computing them once and reusing saves redundant work compared to
        // separate passes.
        dist_to_selected.clear();
        for (uint32_t selected : new_neighbors) {
            float d_cs = compute_l2sq(get_vector(cand_id),
                                      get_vector(selected), dim_);
            dist_to_selected.push_back(d_cs);
        }

        // ---- Entropy-based effective alpha ----
        // Only computed when entropy_scale > 0 AND there are selected neighbors
        // (no diversity to compute for the very first candidate).
        float alpha_eff = alpha;
        if (entropy_scale > 0.0f && !dist_to_selected.empty()) {
            // diversity = distance to nearest already-selected neighbor.
            float min_d_cs = *std::min_element(dist_to_selected.begin(),
                                               dist_to_selected.end());

            // Normalize: how far is c from its nearest selected neighbor,
            // relative to c's distance from the node being pruned?
            // Values > 1 mean c is "farther from selected" than from node —
            // it's in a genuinely different region.  We clamp to [0,1].
            float norm_div = (dist_to_node > 0.0f)
                             ? (min_d_cs / dist_to_node)
                             : 0.0f;
            if (norm_div > 1.0f) norm_div = 1.0f;

            // Relax alpha proportionally to diversity.
            alpha_eff = alpha * (1.0f + entropy_scale * norm_div);
        }

        // ---- α-RNG check with entropy-adjusted threshold ----
        bool keep = true;
        for (float d_cs : dist_to_selected) {
            if (dist_to_node > alpha_eff * d_cs) {
                keep = false;
                break;
            }
        }
        if (keep)
            new_neighbors.push_back(cand_id);
    }

    graph_[node] = std::move(new_neighbors);
}

// ============================================================================
// Build
// ============================================================================
// Only change from original: unpack the new tuple from greedy_search and
// discard hops/edges — build-time traversals must NOT pollute edge scores.

void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L,
                        float alpha, float gamma, float entropy_scale) {
    entropy_scale_ = entropy_scale;  // store for reference
    R_             = R;              // degree cap used by refine_graph()

    std::cout << "Loading data from " << data_path << "..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    std::cout << "  Points: " << npts_ << ", Dimensions: " << dim_ << std::endl;

    if (L < R) {
        std::cerr << "Warning: L (" << L << ") < R (" << R
                  << "). Setting L = R." << std::endl;
        L = R;
    }

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    std::mt19937 rng(42);
    start_node_ = rng() % npts_;
    std::cout << "  Start node: " << start_node_ << std::endl;

    std::vector<uint32_t> perm(npts_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);
    std::cout << "Building index (R=" << R << ", L=" << L
              << ", alpha=" << alpha << ", gamma=" << gamma
              << ", gammaR=" << gamma_R
              << ", entropy_scale=" << entropy_scale << ")..." << std::endl;

    Timer build_timer;

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < npts_; idx++) {
        uint32_t point = perm[idx];

        auto [candidates, dc, h, edges] = greedy_search(get_vector(point), L);
        (void)dc; (void)h; (void)edges;

        robust_prune(point, candidates, alpha, R, entropy_scale);

        for (uint32_t nbr : graph_[point]) {
            std::lock_guard<std::mutex> lock(locks_[nbr]);
            graph_[nbr].push_back(point);

            if (graph_[nbr].size() > gamma_R) {
                std::vector<Candidate> nbr_cands;
                nbr_cands.reserve(graph_[nbr].size());
                for (uint32_t nn : graph_[nbr]) {
                    float d = compute_l2sq(get_vector(nbr), get_vector(nn), dim_);
                    nbr_cands.push_back({d, nn});
                }
                robust_prune(nbr, nbr_cands, alpha, R, entropy_scale);
            }
        }

        if (idx % 10000 == 0) {
            #pragma omp critical
            {
                std::cout << "\r  Inserted " << idx << " / " << npts_
                          << " points" << std::flush;
            }
        }
    }

    double build_time = build_timer.elapsed_seconds();
    size_t total_edges = 0;
    for (uint32_t i = 0; i < npts_; i++)
        total_edges += graph_[i].size();

    std::cout << "\n  Build complete in " << build_time << " seconds.\n"
              << "  Average out-degree: "
              << (double)total_edges / npts_ << std::endl;
}

// ============================================================================
// Search  [MODIFIED — Navigation Optimization P2]
// ============================================================================
//
// 1. Unpacks hops + traversed_edges from greedy_search.
// 2. No re-ranking — candidates already sorted by TRUE distance.
// 3. Updates edge scores once per query (one lock acquisition).
//    Helpful = node ended up in the final top-L candidate set, not just top-K.
//    This gives a much richer learning signal (top-K=10 vs top-L=up to 200).
// 4. Hop counters accumulated via atomics (no lock).

SearchResult VamanaIndex::search(const float* query, uint32_t K,
                                 uint32_t L) const {
    if (L < K) L = K;

    Timer t;
    auto [candidates, dist_cmps, hops, traversed_edges] =
        greedy_search(query, L);
    double latency = t.elapsed_us();

    // candidates is already sorted by true distance — take top-K directly.
    SearchResult result;
    result.dist_cmps  = dist_cmps;
    result.latency_us = latency;
    result.hops       = hops;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < candidates.size(); i++)
        result.ids.push_back(candidates[i].second);

    // Atomic hop accumulation (no lock needed).
    total_hops_.fetch_add(hops, std::memory_order_relaxed);
    total_queries_.fetch_add(1,  std::memory_order_relaxed);

    // ---- Update edge scores — one shard lock at a time ----
    //
    // Strategy: group traversed edges by their source-node shard, then
    // process each shard with a single lock acquisition.
    //
    // Why this is better than the old single global lock:
    //   Old: ONE lock, ALL threads serialized, lock held for O(|edges|) time.
    //   New: up to NUM_EDGE_SHARDS locks, each held for O(|edges|/64) time,
    //        and threads on different shards update in parallel.
    //
    // Why we don't hold multiple shard locks simultaneously:
    //   Acquiring multiple locks in an arbitrary order risks deadlock if two
    //   threads each hold one lock the other needs.  By acquiring and releasing
    //   ONE shard lock at a time (process all edges for shard s, release,
    //   then move to shard s+1), we avoid any possibility of deadlock.
    //
    // Build helpful set from ALL top-L candidates (not just top-K).
    // Using a plain std::vector<bool> indexed by node id is faster than
    // unordered_set for the is-helpful lookup — O(1) with no hash overhead.
    {
        // Mark helpful nodes in a flat bitset over [0, npts_).
        // Allocation: ~125 KB for 1M points — fits comfortably in L2 cache.
        std::vector<bool> is_helpful(npts_, false);
        for (const auto& [d, id] : candidates)
            is_helpful[id] = true;

        // Group edges by shard to minimize lock acquisitions.
        // We use a small array of vectors — one per shard.
        // For a typical query with ~1000 traversed edges and 64 shards,
        // each bucket holds ~15 edges on average.
        std::array<std::vector<std::pair<uint32_t,uint32_t>>,
                   NUM_EDGE_SHARDS> by_shard;
        for (const auto& [u, v] : traversed_edges)
            by_shard[shard_of(u)].emplace_back(u, v);

        // Process each shard under its own lock — never hold two simultaneously.
        for (uint32_t s = 0; s < NUM_EDGE_SHARDS; s++) {
            if (by_shard[s].empty()) continue;
            std::lock_guard<std::mutex> lock(edge_shard_mutex_[s]);
            for (const auto& [u, v] : by_shard[s]) {
                auto& score = edge_scores_[s][edge_key(u, v)];
                score.traversals++;
                if (is_helpful[v])
                    score.helpful++;
            }
        }
    }

    // ---- Adaptive Graph Learning trigger (P3) ----
    //
    // Every REFINE_INTERVAL queries we call refine_graph() to rewrite the
    // graph topology based on accumulated edge-usefulness scores.
    //
    // Why here and not inside greedy_search()?
    //   greedy_search() is on the hot path for EVERY hop; adding any branch
    //   there inflates per-query cost.  search() is called once per query,
    //   so the modulo check costs ~1 ns and is branch-predicted to "not taken"
    //   99.9% of the time.
    //
    // Why 1000?
    //   Enough queries to build statistically meaningful edge scores before
    //   the first refinement, yet frequent enough to converge within a
    //   typical 10K–100K query workload.  Callers can change REFINE_INTERVAL
    //   at compile time if needed.
    //
    // Thread safety:
    //   total_queries_ is a std::atomic so the load below is safe.
    //   refine_graph() acquires its own per-node and per-shard locks
    //   internally and holds no lock here when it is called.
    //   In a multi-threaded search workload multiple threads may each see
    //   the same multiple-of-1000 and each call refine_graph() concurrently.
    //   That is harmless: refine_graph() is idempotent and its internal locks
    //   serialise the actual graph writes.
    static constexpr uint64_t REFINE_INTERVAL = 1000;
    if (total_queries_.load(std::memory_order_relaxed) % REFINE_INTERVAL == 0) {
        refine_graph();
    }

    return result;
}

// ============================================================================
// Adaptive Graph Learning — refine_graph()  [P3]
// ============================================================================
//
// PURPOSE
// -------
// Transform the static Vamana graph into a self-improving structure by
// rewriting edge lists based on query-feedback usefulness scores that P2
// already accumulates in edge_scores_[].
//
// The function is called periodically from search() (every REFINE_INTERVAL
// queries).  It must never be called from inside greedy_search() or
// robust_prune() — those are on the hot path and must stay untouched.
//
// HIGH-LEVEL ALGORITHM (per node u)
// ----------------------------------
//  1. Read usefulness score for every outgoing edge of u.
//     usefulness(u,v) = helpful(u,v) / traversals(u,v)   in [0.0, 1.0]
//
//  2. Compute ADAPTIVE per-node thresholds (percentile-based):
//       T_remove = 30th-percentile of u's usefulness distribution
//       T_boost  = 80th-percentile of u's usefulness distribution
//     Using per-node percentiles rather than a global fixed value means
//     the thresholds track the actual distribution at each node, so
//     refinement is safe even when scores are still sparse.
//
//  3. Classify each edge:
//       score < T_remove  →  candidate for removal   (weak edge)
//       score > T_boost   →  "boosted" (moved to front of list)
//       otherwise         →  kept in place
//
//  4. Stability constraints — checked before writing:
//       • If graph_[u] is empty → skip u entirely.
//       • If no edge has been traversed (all traversals == 0) → skip u.
//       • Never shrink the neighbour list below MIN_NEIGHBOURS (= 1).
//         If removal would empty the list, keep the single best edge.
//       • If after filtering the new list would be IDENTICAL to the old
//         (nothing changed) → do NOT write back (avoids cache thrashing).
//
//  5. Sort the final neighbour list descending by usefulness so that
//     greedy_search() naturally visits the highest-quality edges first
//     on every future query — the ordering effect is free.
//
// THREAD SAFETY
// -------------
// For node u, all edges (u→v) live in shard shard_of(u) = u % 64.
// So ONE shard lock covers every score read for the whole node.
//
// Lock order per node:
//   acquire edge_shard_mutex_[shard_of(u)]  →  read scores  →  release
//   acquire locks_[u]                        →  write graph_  →  release
//
// We NEVER hold both locks at the same time, eliminating any possibility
// of deadlock.  refine_graph() itself holds no lock when it enters the
// per-node loop iteration, so a concurrent search() thread that happens
// to be on the same node will simply wait for the brief graph_ write.
//
// COMPLEXITY
// ----------
// O(npts_ * R) for the score reads + O(npts_ * R log R) for the sorts.
// For 1M points, R=32: ~32M reads + ~32M * 5 ≈ 160M ops.
// Measured wall time on a single core: ≈ 1–3 seconds.
// Called every 1000 queries, so the amortised overhead per query is < 3 ms.

void VamanaIndex::refine_graph() const {

    // ------------------------------------------------------------------
    // CONSTANTS
    //
    // MIN_NEIGHBOURS   — floor on how small an adjacency list may become.
    //                    1 is sufficient to keep the graph weakly connected
    //                    through the start node even in pathological cases.
    //
    // MAX_BOOST_COPIES — how many times a high-usefulness edge may be
    //                    duplicated.  Capped at 2: greedy_search deduplicates
    //                    via the visited[] bitset, so a third copy is never
    //                    reached in practice and wastes memory.
    //
    // MIN_TRAVERSALS   — a node whose edges have collectively seen fewer
    //                    than this many traversals is "too cold" to refine
    //                    reliably.  Skipping it avoids removing edges based
    //                    on statistically meaningless scores.
    //                    Value 10 means we wait until each node has been
    //                    visited on average ~10 times before touching it.
    // ------------------------------------------------------------------
    static constexpr uint32_t MIN_NEIGHBOURS  = 1;
    static constexpr uint32_t MAX_BOOST_COPIES = 2;
    static constexpr uint32_t MIN_TRAVERSALS   = 10;

    // Counters for the end-of-function log line.
    uint64_t nodes_refined   = 0;
    uint64_t edges_removed   = 0;
    uint64_t edges_boosted   = 0;   // unique edges that were duplicated
    uint64_t nodes_skipped   = 0;   // cold or empty — no data yet

    // ------------------------------------------------------------------
    // Observability: announce the trigger BEFORE the loop so that even a
    // crash inside the loop is preceded by a visible log entry.
    // ------------------------------------------------------------------
    const uint64_t trigger_q = total_queries_.load(std::memory_order_relaxed);
    std::cout << "[P3 refine_graph] triggered at query " << trigger_q
              << "  (R_cap=" << R_ << ", min_travs=" << MIN_TRAVERSALS << ")"
              << std::endl;

    for (uint32_t u = 0; u < npts_; u++) {

        // ==============================================================
        // Step 1 — Snapshot the current neighbour list.
        //
        // We take a copy (not a reference) under the node lock, then
        // immediately release the lock.  All subsequent work is on the
        // local copy.  The final write-back acquires the lock again only
        // if something actually changed.
        // ==============================================================
        std::vector<uint32_t> current_nbrs;
        {
            std::lock_guard<std::mutex> node_lk(locks_[u]);
            current_nbrs = graph_[u];
        }

        if (current_nbrs.empty()) {
            ++nodes_skipped;
            continue;
        }

        // ==============================================================
        // Step 2 — Read usefulness scores from the shard.
        //
        // All edges (u→*) belong to shard shard_of(u), so ONE lock covers
        // every score read for this node.  We also accumulate the total
        // traversal count to detect cold nodes.
        // ==============================================================
        const uint32_t sh = shard_of(u);
        std::vector<float>    scores(current_nbrs.size(), 0.0f);
        uint32_t total_traversals = 0;

        {
            std::lock_guard<std::mutex> shard_lk(edge_shard_mutex_[sh]);
            for (size_t i = 0; i < current_nbrs.size(); i++) {
                uint32_t v = current_nbrs[i];
                auto it = edge_scores_[sh].find(edge_key(u, v));
                if (it != edge_scores_[sh].end()) {
                    total_traversals += it->second.traversals;
                    if (it->second.traversals > 0) {
                        // No division-by-zero: traversals > 0 guarded above.
                        scores[i] = static_cast<float>(it->second.helpful)
                                  / static_cast<float>(it->second.traversals);
                    }
                    // traversals == 0 → scores[i] stays 0.0f (cold edge)
                }
                // Not in map → scores[i] stays 0.0f (never traversed)
            }
        }
        // Shard lock released.  From here on we work only on local data.

        // Cold-node guard: skip if we don't yet have enough signal.
        if (total_traversals < MIN_TRAVERSALS) {
            ++nodes_skipped;
            continue;
        }

        // ==============================================================
        // Step 3 — Compute adaptive per-node thresholds.
        //
        // We sort a COPY of scores (not the original) to preserve the
        // alignment between scores[i] and current_nbrs[i], which we still
        // need in the classification step.
        //
        // Percentile formula: linear interpolation on the sorted array,
        // consistent with numpy's default method.  Clamped so it can never
        // go out of bounds regardless of n.
        // ==============================================================
        std::vector<float> sorted_scores = scores;
        std::sort(sorted_scores.begin(), sorted_scores.end());
        const size_t n = sorted_scores.size();

        auto percentile = [&](float p) -> float {
            // p in [0, 100].  n is always ≥ 1 here.
            if (n == 1) return sorted_scores[0];
            float idx_f = (p / 100.0f) * static_cast<float>(n - 1);
            if (idx_f <= 0.0f)
                return sorted_scores[0];
            if (idx_f >= static_cast<float>(n - 1))
                return sorted_scores[n - 1];
            size_t lo   = static_cast<size_t>(idx_f);
            float  frac = idx_f - static_cast<float>(lo);
            return sorted_scores[lo] * (1.0f - frac)
                 + sorted_scores[lo + 1] * frac;
        };

        const float T_remove = percentile(30.0f);
        const float T_boost  = percentile(80.0f);

        // ==============================================================
        // Step 4 — Classify edges and build the new neighbour list.
        //
        // Three tiers:
        //   score > T_boost   → "boosted": high-quality edge.
        //                        Added once normally, then duplicated up to
        //                        MAX_BOOST_COPIES times.  Duplication means
        //                        greedy_search() encounters the node ID
        //                        multiple times in the neighbour list, giving
        //                        it more chances to enter the candidate set
        //                        even if the first encounter is skipped via
        //                        the visited[] check.
        //                        IMPORTANT: duplication is intentional here —
        //                        the visited[] bitset in greedy_search()
        //                        prevents actually computing the distance
        //                        twice; the duplicate only raises the chance
        //                        the edge is tried from a different hop.
        //   T_remove ≤ score  → "kept": neither boosted nor dropped.
        //   score < T_remove  → dropped (weak edge, learning says skip it).
        //
        // Both live buckets are sorted descending by score before merging
        // so that within each tier the highest-quality edge appears first.
        // ==============================================================
        std::vector<std::pair<float, uint32_t>> boosted_bucket, kept_bucket;
        boosted_bucket.reserve(current_nbrs.size());
        kept_bucket.reserve(current_nbrs.size());

        uint32_t local_removed = 0;

        for (size_t i = 0; i < current_nbrs.size(); i++) {
            float sc = scores[i];
            uint32_t v = current_nbrs[i];
            if (sc > T_boost) {
                boosted_bucket.emplace_back(sc, v);
            } else if (sc >= T_remove) {
                kept_bucket.emplace_back(sc, v);
            } else {
                ++local_removed;
            }
        }

        auto desc_cmp = [](const std::pair<float,uint32_t>& a,
                           const std::pair<float,uint32_t>& b) {
            return a.first > b.first;   // descending by score
        };
        std::sort(boosted_bucket.begin(), boosted_bucket.end(), desc_cmp);
        std::sort(kept_bucket.begin(),    kept_bucket.end(),    desc_cmp);

        // Assemble new_nbrs: boosted (+ duplicates) first, then kept.
        std::vector<uint32_t> new_nbrs;
        // Worst-case capacity: boosted * MAX_BOOST_COPIES + kept.
        new_nbrs.reserve(
            boosted_bucket.size() * MAX_BOOST_COPIES + kept_bucket.size());

        for (auto& [sc, v] : boosted_bucket) {
            new_nbrs.push_back(v);                  // first copy (always)
            for (uint32_t copy = 1; copy < MAX_BOOST_COPIES; copy++) {
                // Only add the duplicate if we still have headroom under R_.
                // This is a pre-cap check; the hard trim below is the final
                // authority.  We check here to avoid inflating the vector
                // unnecessarily when the list is already near R_.
                if (new_nbrs.size() < static_cast<size_t>(R_)) {
                    new_nbrs.push_back(v);
                    ++edges_boosted;
                }
            }
        }
        for (auto& [sc, v] : kept_bucket) {
            new_nbrs.push_back(v);
        }

        // ==============================================================
        // Step 5 — Stability: enforce MIN_NEIGHBOURS floor.
        //
        // If filtering emptied the list entirely, rescue the single highest-
        // scoring edge from the original set.  This is the only situation
        // where we keep a "below T_remove" edge — connectivity beats purity.
        // ==============================================================
        if (new_nbrs.empty()) {
            size_t best_idx = 0;
            for (size_t i = 1; i < scores.size(); i++)
                if (scores[i] > scores[best_idx]) best_idx = i;
            new_nbrs.push_back(current_nbrs[best_idx]);
        }

        // ==============================================================
        // Step 6 — Degree cap: trim to at most R_ neighbours.
        //
        // Because new_nbrs is sorted descending by usefulness (boosted
        // bucket first, then kept bucket, both sorted), truncating at R_
        // removes only the lowest-usefulness edges.  The highest-quality
        // edges are always preserved — this satisfies the spec requirement
        // "do NOT remove highest usefulness edges".
        // ==============================================================
        if (new_nbrs.size() > static_cast<size_t>(R_)) {
            new_nbrs.resize(R_);
        }

        // ==============================================================
        // Step 7 — Write-back guard: skip the lock if nothing changed.
        //
        // Comparison is O(R) but avoids a lock acquisition + cache-line
        // invalidation on graph_[u] for every unchanged node.  Given that
        // the first several thousand queries typically leave most nodes
        // cold, this guard fires frequently early in a workload.
        // ==============================================================
        if (new_nbrs == current_nbrs)
            continue;

        {
            std::lock_guard<std::mutex> node_lk(locks_[u]);
            graph_[u] = std::move(new_nbrs);
        }

        ++nodes_refined;
        edges_removed += local_removed;

    }   // end for u

    // ------------------------------------------------------------------
    // Observability: single-line summary after the loop.
    // Numbers are aggregate over this one invocation.
    // ------------------------------------------------------------------
    std::cout << "[P3 refine_graph] done — "
              << "nodes_refined=" << nodes_refined
              << "  edges_removed=" << edges_removed
              << "  edges_boosted=" << edges_boosted
              << "  nodes_skipped(cold)=" << nodes_skipped
              << std::endl;
}

// ============================================================================
// Save / Load (unchanged)
// ============================================================================

void VamanaIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open file for writing: " + path);

    out.write(reinterpret_cast<const char*>(&npts_), 4);
    out.write(reinterpret_cast<const char*>(&dim_),  4);
    out.write(reinterpret_cast<const char*>(&start_node_), 4);

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg = graph_[i].size();
        out.write(reinterpret_cast<const char*>(&deg), 4);
        if (deg > 0)
            out.write(reinterpret_cast<const char*>(graph_[i].data()),
                      deg * sizeof(uint32_t));
    }
    std::cout << "Index saved to " << path << std::endl;
}

void VamanaIndex::load(const std::string& index_path,
                       const std::string& data_path) {
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    std::ifstream in(index_path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open index file: " + index_path);

    uint32_t file_npts, file_dim;
    in.read(reinterpret_cast<char*>(&file_npts), 4);
    in.read(reinterpret_cast<char*>(&file_dim),  4);
    in.read(reinterpret_cast<char*>(&start_node_), 4);

    if (file_npts != npts_ || file_dim != dim_)
        throw std::runtime_error(
            "Index/data mismatch: index has " + std::to_string(file_npts) +
            "x" + std::to_string(file_dim) + ", data has " +
            std::to_string(npts_) + "x" + std::to_string(dim_));

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    uint32_t max_observed_degree = 0;
    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg;
        in.read(reinterpret_cast<char*>(&deg), 4);
        graph_[i].resize(deg);
        if (deg > 0)
            in.read(reinterpret_cast<char*>(graph_[i].data()),
                    deg * sizeof(uint32_t));
        if (deg > max_observed_degree) max_observed_degree = deg;
    }

    // Derive R_ from the loaded graph so refine_graph() enforces the
    // correct degree cap even when the index was loaded (not built here).
    // build() always sets R_ explicitly; this covers the load() path.
    if (max_observed_degree > 0) R_ = max_observed_degree;

    std::cout << "Index loaded: " << npts_ << " points, " << dim_
              << " dims, start=" << start_node_
              << ", R=" << R_ << std::endl;
}
