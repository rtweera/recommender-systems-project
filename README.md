# Recommender Systems Project

Book recommendation system project demonstrating multiple recommendation techniques and a realistic hybrid architecture.

## Overview

This repository contains:

- A **realistic recommender implementation** with clear cold-start vs warm-user logic
- A **comprehensive/demo implementation** for concept exploration
- Notebooks, docs, and submission assets related to the project

Core techniques used:

- Content-based filtering (TF-IDF over item metadata)
- Collaborative filtering (user similarity)
- Network-based recommendation (PageRank / personalized PageRank)
- Personalized score boosts based on user profile
- Hybrid score combination

## Repository Structure

- `/src/realistic_recommender.py` – production-style architecture with feature caching and strategy routing
- `/src/comprehensive_recommender.py` – educational/demo-style end-to-end recommender code
- `/data/` – dataset files and dataset link note
- `/notebooks/` – exploration and notebook-based versions
- `/docs/` – demo and comparison notes
- `/to_submit/` – packaged submission artifacts

## Requirements

- Python `>=3.11, <3.14`
- Dependencies are defined in `/pyproject.toml`:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `networkx`

## Setup

If you use Poetry:

```bash
poetry install
```

If you use pip (minimum packages):

```bash
pip install numpy pandas scikit-learn "networkx[default]"
```

## Run

Run the realistic recommender:

```bash
python src/realistic_recommender.py
```

The script demonstrates:

- Cold-start recommendation flow
- Warm-user hybrid recommendation flow
- Explanation of where each algorithm is used in the architecture

## Notes

- The realistic implementation in `src/realistic_recommender.py` uses an in-script sample dataset to demonstrate the architecture and recommendation flow.
- Additional datasets and project artifacts are available under `/data` and `/to_submit`.
