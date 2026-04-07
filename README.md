# ML Arena 5 - Neural Network

This folder is a dedicated repo-style package for Problem 5: Neural Network.

This repository is part of the ML Arena family, a set of 5 dedicated repositories where each repository focuses on one core ML problem.

## ML Arena Family Repositories

- ML-Arena-1 - Linear Regression 
- ML-Arena-2 - Logistic Regression
- ML-Arena-3 - Decision Tree
- ML-Arena-4 - K-Means Clustering 
- **ML-Arena-5 - Neural Network (Current Repo)**

## Dataset

* File: `dataset.csv`
* Source: UCI Wine Quality
* Rows: 6497
* Columns: 13
* Targets:

  * `quality`
  * `quality_binary`

This is a tabular dataset, allowing you to focus on core neural network concepts without image processing.

## Repository Layout

```
Neural-Networks/
|- README.md
|- CONTRIBUTING.md
|- dataset.csv
|- exploration/exploration.ipynb
|- library/training.ipynb
|- scratch/training.ipynb
`- .github/PULL_REQUEST_TEMPLATE.md
```

## Task Tracks

- Exploration: Data understanding, distributions, correlation, and insights.
- Library: Baseline model with standard ML libraries.
- Scratch: First-principles implementation using NumPy.
- Optimization: Improvements on top of library or scratch baseline.

## Quick Start

1. Fork this folder as its own repo (Neural-Networks).
2. Clone your fork and create a branch.
3. Pick one issue.
4. Work in only one notebook per issue based on track:
   - `exploration/exploration.ipynb`
   - `library/training.ipynb`
   - `scratch/training.ipynb`
5. Run all cells top to bottom.
6. Open a pull request with `Closes #<issue-number>`.

See `CONTRIBUTING.md` for detailed rules.