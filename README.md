# GraphANN Enhancements

> Algorithms for Data Science Course Project
>
> Enhancement of GraphANN through:
>
> - Navigation Optimization
> - Entropy-Based Edge Pruning
> - Adaptive Graph Learning

---

## Overview

This project extends the GraphANN Approximate Nearest Neighbor (ANN) search framework based on the Vamana graph index.

The goal of this project is to improve graph traversal efficiency while maintaining high search quality on large-scale datasets.

The original implementation was provided as part of the course project. Our work introduces three enhancements aimed at improving graph quality and search efficiency:

1. Navigation Optimization
2. Entropy-Based Edge Pruning
3. Adaptive Graph Learning

All experiments were evaluated on the SIFT1M benchmark dataset.

---

## Project Evolution

### Stage 1: Original GraphANN

The baseline implementation uses:

- Vamana graph construction
- Greedy beam search
- Alpha-RNG robust pruning

---

### Stage 2: Navigation Optimization

Introduced edge usefulness scores to guide graph traversal toward more informative neighbors.

Benefits:

- Better graph navigation
- Reduced unnecessary exploration

---

### Stage 3: Entropy-Based Edge Pruning

Modified the pruning process by incorporating neighborhood diversity.

Benefits:

- Improved graph structure
- Better balance between local and long-range connectivity

---

### Stage 4: Adaptive Graph Learning

Introduced online graph refinement based on query feedback.

Benefits:

- Dynamic graph improvement
- Adaptive restructuring of graph connectivity

---

# Repository Structure

```text
graphann/
│
├── src/
│   ├── build_index.cpp
│   ├── search_index.cpp
│   ├── distance.cpp
│   ├── io_utils.cpp
│   └── vamana_index.cpp
│
├── include/
│   ├── distance.h
│   ├── io_utils.h
│   ├── timer.h
│   └── vamana_index.h
│
├── scripts/
│   ├── convert_vecs.py
│   └── run_sift1m.sh
│
├── docs/
│   ├── project_timeline.md
│   ├── enhancement_design.md
│   ├── future_work.md
│   └── comparison_with_submission2.md
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
│   ├── enhanced_results.md
│   └── comparison.md
│
├── CMakeLists.txt
└── README.md
```

---

# Dataset

Experiments were performed using the SIFT1M benchmark dataset.

Dataset Statistics:

| Property | Value |
|-----------|---------|
| Base Vectors | 1,000,000 |
| Query Vectors | 10,000 |
| Dimensions | 128 |

---

# Experimental Setup

## Index Construction Parameters

| Parameter | Value |
|------------|--------|
| R | 32 |
| Build L | 75 |
| Alpha | 1.2 |
| Gamma | 1.5 |

## Search Parameters

| Parameter | Value |
|------------|--------|
| K | 10 |
| Search L | 10,20,30,50,75,100,150,200 |

---

# Results Summary

## Baseline GraphANN

| L | Recall@10 |
|---|---:|
| 10 | 0.7768 |
| 20 | 0.8909 |
| 50 | 0.9665 |
| 100 | 0.9891 |
| 200 | 0.9960 |

---

## Enhanced GraphANN

| L | Recall@10 |
|---|---:|
| 10 | 0.7734 |
| 20 | 0.8867 |
| 50 | 0.9643 |
| 100 | 0.9857 |
| 200 | 0.9943 |

---

## Key Observation

The enhanced implementation reduces distance computations while maintaining comparable search quality.

Example:

| Metric | Baseline | Enhanced |
|----------|----------:|----------:|
| Recall@10 (L=100) | 0.9891 | 0.9857 |
| Avg Dist Cmps (L=100) | 2434.9 | 2101.9 |

This corresponds to approximately **13.7% fewer distance computations**.

Complete experimental results are available in the `results/` directory.

---

# Building the Project

## Clone Repository

```bash
git clone https://github.com/krishdange27/graphann.git
cd graphann
```

## Build

```bash
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

---

# Dataset Conversion

Convert SIFT `.fvecs` and `.ivecs` files into binary format:

```bash
python3 scripts/convert_vecs.py sift_base.fvecs sift_base.fbin
python3 scripts/convert_vecs.py sift_query.fvecs sift_query.fbin
python3 scripts/convert_vecs.py sift_groundtruth.ivecs sift_gt.ibin
```

---

# Building the Index

```bash
./build_index \
  --data sift_base.fbin \
  --output sift_index.bin \
  --R 32 \
  --L 75 \
  --alpha 1.2 \
  --gamma 1.5
```

---

# Running Search

```bash
./search_index \
  --index sift_index.bin \
  --data sift_base.fbin \
  --queries sift_query.fbin \
  --gt sift_gt.ibin \
  --K 10 \
  --L 10,20,30,50,75,100,150,200
```

---

# Documentation

Project reports and development history are available in:

```text
reports/
```

including:

- Project Proposal
- Milestone 1 Report
- Enhancement Report
- Final Presentation
- Original Research Paper

---

# Reproducibility

All results reported in this repository were reproduced using:

- Same dataset
- Same build parameters
- Same search parameters

Both baseline and enhanced implementations were evaluated to validate the effectiveness of the proposed enhancements.

---

# Future Work

Potential directions for future research include:

- Evaluation on DEEP1M
- Evaluation on GIST1M
- Dynamic insertion and deletion support
- Learning-based graph optimization
- Distributed graph construction
- Hardware-aware ANN search optimization

---

# Contributors

Algorithms for Data Science Project Team

- Krish Dange
- Team Members

---

# Acknowledgements

This project is based on the GraphANN/Vamana framework provided for the Algorithms for Data Science course.

The original ideas behind graph-based ANN search are inspired by:

- DiskANN
- Vamana Graph Index
- Approximate Nearest Neighbor Search literature