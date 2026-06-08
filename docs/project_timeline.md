# Project Timeline

## Phase 1: Project Proposal

Objective:
Study graph-based Approximate Nearest Neighbor search using the Vamana graph index and identify opportunities for improvement.

Deliverables:
- Literature review
- Dataset selection
- Enhancement planning

---

## Phase 2: Baseline Analysis

Studied the original GraphANN implementation.

Key Components:
- Greedy beam search
- Alpha-RNG robust pruning
- Vamana graph construction

---

## Phase 3: Navigation Optimization

Introduced edge usefulness scores.

Goal:
Guide graph traversal toward more informative neighbors.

Expected Benefit:
Reduce unnecessary exploration during search.

---

## Phase 4: Entropy-Based Edge Pruning

Modified robust pruning to incorporate neighborhood diversity.

Goal:
Improve graph connectivity and structural quality.

Expected Benefit:
Maintain recall while reducing redundant edges.

---

## Phase 5: Adaptive Graph Learning

Introduced graph refinement based on query behavior.

Goal:
Allow the graph structure to improve dynamically.

Expected Benefit:
Long-term improvement of graph quality.

---

## Phase 6: Experimental Validation

Evaluated:
- Original GraphANN
- Enhanced GraphANN

Dataset:
- SIFT1M

Metrics:
- Recall@10
- Distance Computations
- Average Latency
- P99 Latency