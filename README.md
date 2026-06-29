# Task 9 — Hyperparameter Tuning

## Overview
Systematically tuned hyperparameters of an SVM classifier to achieve validated
performance gain over the default baseline without overfitting.

## Results
| Metric | Score |
|---|---|
| Baseline Test F1 | 0.9630 |
| Tuned Test F1 | 0.9725 |
| Gain | +0.0095 |
| Best CV F1 | 0.9774 |

## Best Configuration
| Parameter | Value |
|---|---|
| Model | SVC (Support Vector Classifier) |
| Kernel | linear |
| C | 0.01 |
| Gamma | 0.01 |
| Degree | 2 |

## Approach
- Dataset: Breast Cancer (sklearn) with noise columns added for realism
- Search Strategy: RandomizedSearchCV (50 iterations)
- Cross Validation: 5-Fold CV on training set only
- Metric: F1 Score
- Test set kept completely unseen until final evaluation

## Project Structure
task9_hyperparameter_tuning/
├── src/tune.py               # Main tuning script
├── results/best_config.json  # Best params and scores
└── requirements.txt          # Dependencies

## Tools Used
- scikit-learn (RandomizedSearchCV, SVC)
- NumPy, Pandas
- Python 3.x