# Adaptive DiskANN

![C++](https://img.shields.io/badge/C%2B%2B-17-blue?logo=c%2B%2B)
![ANN Search](https://img.shields.io/badge/Domain-Approximate%20Nearest%20Neighbor-purple)
![Research](https://img.shields.io/badge/Type-Research-success)
![Published](https://img.shields.io/badge/Publication-IJERT%202026-orange)
![SIFT1M](https://img.shields.io/badge/Benchmark-SIFT1M-red)

### Navigation-Aware Search and Graph Optimization for Approximate Nearest Neighbor Retrieval

**SIFT1M Benchmark • 1M Vectors • 10K Queries • 128 Dimensions**

---

## Authors

* Abhijeet Kumar
* Krish Dange
* Sai Jagadeesh

Department of Artificial Intelligence and Data Science
Indian Institute of Technology Madras

---

## Publication

### Making DiskANN Adaptive: Navigation-Aware Search, Diversity Pruning, and Feedback-Driven Graph Refinement for Graph-Based ANN

**Published in:** International Journal of Engineering Research and Technology (IJERT)
**Volume:** 15
**Issue:** 06
**Publication Date:** June 2026

This repository contains the implementation, benchmark results, and reproducibility artifacts associated with the published research paper.

**Paper:** `paper/paper.pdf`
**Publication Certificate:** `paper/certificate.pdf`

---

## Research Highlights

* Achieves a **17.6% reduction in average distance computations** while maintaining **Recall@10 above 0.99**.
* Evaluated on the **SIFT1M benchmark** containing **1 million vectors** and **10,000 query vectors**.
* Introduces navigation-aware graph traversal and feedback-driven graph optimization for graph-based ANN retrieval.
* Combines traversal prioritization, diversity-aware pruning, and adaptive graph refinement within a Vamana/DiskANN framework.
* Provides reproducible implementations, benchmark results, project reports, and publication artifacts.

---

## Why Adaptive DiskANN?

Modern vector databases, recommendation systems, semantic search engines, and Retrieval-Augmented Generation (RAG) pipelines depend on efficient Approximate Nearest Neighbor (ANN) search over high-dimensional embeddings.

Graph-based ANN methods such as **Vamana** and **DiskANN** achieve excellent recall-latency tradeoffs by navigating sparse proximity graphs. However, graph traversal frequently explores redundant paths that increase search effort without improving retrieval quality.

Adaptive DiskANN investigates graph-level optimization strategies that improve search efficiency while preserving high recall performance.

The project focuses on:

* **Navigation-Aware Search** for traversal prioritization.
* **Edge Usefulness Scoring** for identifying valuable graph connections.
* **Diversity-Aware Pruning** for reducing redundant graph structure.
* **Adaptive Graph Refinement** using query-feedback information.

Together, these techniques reduce search effort while maintaining the retrieval quality expected from large-scale ANN systems.

---

## Key Results

Adaptive DiskANN was evaluated on the **SIFT1M benchmark** containing **1,000,000 base vectors**, **10,000 query vectors**, and **128-dimensional embeddings**.

### Baseline vs Adaptive DiskANN

| Metric                                      | Baseline GraphANN | Adaptive DiskANN |
| ------------------------------------------- | ----------------: | ---------------: |
| Recall@10                                   |            0.9960 |           0.9943 |
| Avg Distance Computations                   |            4044.6 |           3332.8 |
| Relative Reduction in Distance Computations |                 – |        **17.6%** |

### Headline Result

> Adaptive DiskANN achieved a **17.6% reduction in average distance computations** while preserving **Recall@10 above 0.99** on the SIFT1M benchmark.

### Key Takeaways

* Maintained **Recall@10 above 0.99** while reducing search effort by **17.6%**.
* Reduced the number of distance evaluations required during graph traversal.
* Improved search efficiency without significant recall degradation.
* Demonstrated effectiveness on a large-scale benchmark containing one million vectors.

---

## Contributions

### Team Contributions

* Designed and evaluated Adaptive DiskANN enhancements for graph-based ANN retrieval.
* Developed a reproducible experimental framework for large-scale ANN benchmarking.
* Conducted comparative analysis between baseline and adaptive search strategies.
* Evaluated performance using recall, latency, and search-effort metrics.
* Authored and published the associated research paper.

### My Contributions

* Implemented Navigation-Aware Search for graph traversal optimization.
* Developed Edge Usefulness Scoring mechanisms for traversal prioritization.
* Contributed to benchmarking and large-scale evaluation on the SIFT1M dataset.
* Participated in repository organization, documentation, and reproducibility workflows.

My contributions focused primarily on search adaptation, traversal optimization, and experimental validation within the Adaptive DiskANN framework.

---

## System Architecture

Adaptive DiskANN extends a DiskANN-style graph search pipeline with navigation-aware traversal, edge usefulness estimation, diversity-aware pruning, and feedback-driven graph refinement.

```text
                Query
                  │
                  ▼
     Navigation-Aware Search
                  │
                  ▼
        Candidate Expansion
                  │
                  ▼
      Edge Usefulness Scoring
                  │
                  ▼
     Adaptive Graph Refinement
                  │
                  ▼
            Top-K Results
```

The system uses information gathered during query execution to identify valuable graph connections, enabling future searches to prioritize more effective traversal paths while preserving retrieval quality.

---

## Core Innovations

### Navigation-Aware Search

Introduces traversal prioritization mechanisms that guide graph exploration toward historically useful search paths, reducing unnecessary node expansions and distance evaluations.

### Edge Usefulness Scoring

Maintains edge-level usefulness information derived from search behavior, allowing the traversal process to distinguish between highly informative and less informative graph connections.

### Diversity-Aware Pruning

Preserves graph navigability while reducing redundant neighborhood structure, helping maintain search quality with lower traversal overhead.

### Adaptive Graph Refinement

Uses query feedback collected during search to improve graph connectivity over time, reinforcing useful connections and improving future search efficiency.

---

## Experimental Methodology

### Dataset

**SIFT1M**

* 1,000,000 base vectors
* 10,000 query vectors
* 128 dimensions

### Evaluation Metrics

* Recall@10
* Average Distance Computations
* Average Search Latency
* P99 Latency

### Evaluation Pipeline

```text
SIFT1M Dataset
      │
      ▼
Baseline GraphANN
      │
      ▼
Adaptive DiskANN
      │
      ├── Recall@10
      ├── Distance Computations
      ├── Average Latency
      └── P99 Latency
```

---

## Detailed Benchmark Results

Complete benchmark data and experimental configurations are available in:

* `results/baseline_results.md`
* `results/adaptive_results.md`
* `results/comparison.md`
* `results/experiment_setup.md`

---

## Repository Structure

```text
graphann/
│
├── src/                      # Core Adaptive DiskANN implementation
├── include/                  # Header files
├── scripts/                  # Dataset conversion and benchmarking utilities
│
├── docs/
│   └── project_timeline.md
│
├── paper/
│   ├── paper.pdf
│   ├── certificate.pdf
│   └── README.md
│
├── reports/
│   ├── proposal.pdf
│   ├── milestone1.pdf
│   ├── enhancement_report.pdf
│   ├── final_presentation.pdf
│   └── diskann_original_paper.pdf
│
├── results/
│   ├── baseline_results.md
│   ├── adaptive_results.md
│   ├── comparison.md
│   └── experiment_setup.md
│
├── CITATION.bib
├── CMakeLists.txt
├── README.md
└── .gitignore
```

---

## Quick Start

### Build

```bash
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

This produces:

* `build_index`
* `search_index`

---

## Build Instructions

### Build Index

```bash
./build_index \
  --data <dataset.fbin> \
  --output <index.bin> \
  --R 32 \
  --L 75 \
  --alpha 1.2 \
  --gamma 1.5
```

### Run Search

```bash
./search_index \
  --index <index.bin> \
  --data <dataset.fbin> \
  --queries <queries.fbin> \
  --gt <groundtruth.ibin> \
  --K 10 \
  --L 10,20,30,50,75,100,150,200
```

---

## Research Artifacts

### Publication

* Published research paper
* Publication certificate

### Experimental Results

* Baseline GraphANN evaluation
* Adaptive DiskANN evaluation
* Comparative analysis
* Experimental setup and configuration

### Development Reports

* Project proposal
* Milestone report
* Enhancement report
* Final presentation

### Reference Material

* Original DiskANN paper used as the baseline reference

---

## Citation

If you use this repository or build upon this work, please cite:

```bibtex
@article{adaptive_diskann_2026,
  title   = {Making DiskANN Adaptive: Navigation-Aware Search, Diversity Pruning, and Feedback-Driven Graph Refinement for Graph-Based ANN},
  author  = {Abhijeet Kumar and Krish Dange and Sai Jagadeesh},
  journal = {International Journal of Engineering Research and Technology},
  volume  = {15},
  number  = {06},
  year    = {2026}
}
```

See `CITATION.bib` for complete citation metadata.
