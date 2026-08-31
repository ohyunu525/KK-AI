# ModelExperiment11 experiment report

Selected common setting: **dropout020**. lowest eligible mean validation structure; improvement threshold passed.

Primary validation structure: **0.599398 -> 0.579470** (+3.32% improvement).
This is a selection-set estimate; it is not an unbiased estimate of generalization.

Hyperparameters: {"id": "dropout020", "learning_rate": 0.001, "structure_dropout": 0.2, "weight_decay": 0.0001}

Completed training trials: 28. Search candidates: 8. Seeds: [41, 42, 43]. Fractions: [1.0].
Train / validation / historical test: {'test': 1000, 'train': 8000, 'validation': 1000}. Split seed: 42.
The unchanged full and sign-only architectures share one selected setting, fixed loss weights, sensor prefixes, initialization/shuffle seeds, batch size, and maximum epoch budget.

| Candidate | Validation structure | Seed SD | Improvement | Eligible |
| --- | ---: | ---: | ---: | --- |
| dropout020 | 0.579470 | 0.002059 | +3.32% | True |
| random_02 | 0.579655 | 0.008931 | +3.29% | True |
| baseline | 0.599398 | 0.004792 | +0.00% | True |

Primary checkpoint (best_structure.pt); positive improvement is better:

| Split | Model | Fraction | Metric | Baseline | Selected | Improvement |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| fresh_test | g05_full_reconstruction | 1 | absolute_sign_accuracy | 88.81% | 87.95% | -0.86 pp |
| fresh_test | g05_full_reconstruction | 1 | global_sign_accuracy | 87.13% | 86.37% | -0.77 pp |
| fresh_test | g05_full_reconstruction | 1 | loss_structure | 0.593162 | 0.560248 | +5.55% |
| fresh_test | g05_full_reconstruction | 1 | mean_position_3d_error | 0.753956 | 0.714383 | +5.25% |
| fresh_test | g05_full_reconstruction | 1 | mean_position_mae | 0.370326 | 0.351396 | +5.11% |
| fresh_test | g05_full_reconstruction | 1 | relative_sign_accuracy | 95.92% | 96.09% | +0.17 pp |
| fresh_test | g05_sign_only | 1 | absolute_sign_accuracy | 88.75% | 88.69% | -0.07 pp |
| fresh_test | g05_sign_only | 1 | global_sign_accuracy | 87.37% | 86.97% | -0.40 pp |
| fresh_test | g05_sign_only | 1 | loss_structure | 0.589209 | 0.576140 | +2.22% |
| fresh_test | g05_sign_only | 1 | mean_position_3d_error | 0.741436 | 0.725827 | +2.11% |
| fresh_test | g05_sign_only | 1 | mean_position_mae | 0.364382 | 0.356067 | +2.28% |
| fresh_test | g05_sign_only | 1 | relative_sign_accuracy | 95.92% | 95.91% | -0.01 pp |
| historical_test | g05_full_reconstruction | 1 | absolute_sign_accuracy | 88.27% | 87.83% | -0.43 pp |
| historical_test | g05_full_reconstruction | 1 | global_sign_accuracy | 86.80% | 86.37% | -0.43 pp |
| historical_test | g05_full_reconstruction | 1 | loss_structure | 0.599775 | 0.578731 | +3.51% |
| historical_test | g05_full_reconstruction | 1 | mean_position_3d_error | 0.756561 | 0.721064 | +4.69% |
| historical_test | g05_full_reconstruction | 1 | mean_position_mae | 0.370637 | 0.353957 | +4.50% |
| historical_test | g05_full_reconstruction | 1 | relative_sign_accuracy | 95.92% | 95.71% | -0.21 pp |
| historical_test | g05_sign_only | 1 | absolute_sign_accuracy | 88.01% | 88.51% | +0.49 pp |
| historical_test | g05_sign_only | 1 | global_sign_accuracy | 86.67% | 86.67% | +0.00 pp |
| historical_test | g05_sign_only | 1 | loss_structure | 0.594501 | 0.585041 | +1.59% |
| historical_test | g05_sign_only | 1 | mean_position_3d_error | 0.749003 | 0.730890 | +2.42% |
| historical_test | g05_sign_only | 1 | mean_position_mae | 0.367211 | 0.358444 | +2.39% |
| historical_test | g05_sign_only | 1 | relative_sign_accuracy | 96.01% | 96.00% | -0.01 pp |
| validation | g05_full_reconstruction | 1 | absolute_sign_accuracy | 87.39% | 86.91% | -0.47 pp |
| validation | g05_full_reconstruction | 1 | global_sign_accuracy | 85.47% | 85.37% | -0.10 pp |
| validation | g05_full_reconstruction | 1 | loss_structure | 0.602085 | 0.574612 | +4.56% |
| validation | g05_full_reconstruction | 1 | mean_position_3d_error | 0.760328 | 0.726951 | +4.39% |
| validation | g05_full_reconstruction | 1 | mean_position_mae | 0.372623 | 0.357009 | +4.19% |
| validation | g05_full_reconstruction | 1 | relative_sign_accuracy | 96.03% | 96.40% | +0.37 pp |
| validation | g05_sign_only | 1 | absolute_sign_accuracy | 87.59% | 87.43% | -0.16 pp |
| validation | g05_sign_only | 1 | global_sign_accuracy | 85.73% | 85.67% | -0.07 pp |
| validation | g05_sign_only | 1 | loss_structure | 0.596712 | 0.584328 | +2.08% |
| validation | g05_sign_only | 1 | mean_position_3d_error | 0.750233 | 0.733955 | +2.17% |
| validation | g05_sign_only | 1 | mean_position_mae | 0.368727 | 0.360138 | +2.33% |
| validation | g05_sign_only | 1 | relative_sign_accuracy | 96.15% | 96.11% | -0.04 pp |

Full paired values, sample SD across seeds (ddof=1), losses and both checkpoint selections are in paired_comparisons.csv / comparison_summary.csv. routing_comparisons.csv retains the original full versus sign-only comparison under identical settings.

Test status: completed after the immutable selection lock.
Historical test was examined in earlier work. The fresh holdout uses the preregistered new simulation seed and the original sensor coordinates; it is generated only after selection.
No train+validation refit or test-based seed/checkpoint selection is performed.
The same validation split is used for epoch and hyperparameter selection. Three training seeds measure initialization/order variation, not three independent data splits. Search is bounded, not globally optimal. No evidence is claimed for unsearched sensor fractions, other sensor layouts, noisy measurements, or real data.
Position and sign metrics use the unchanged joint 120-permutation assignment. Their values are not position-only matching diagnostics. Lower composite loss does not imply every component improves.

Reproduction requires the saved source/data hashes, study configuration, runtime, and device. Exact interrupted-resume equality is scoped to that environment. [PyTorch reproducibility](https://docs.pytorch.org/docs/2.7/notes/randomness.html). Seeded log-uniform random candidates supplement evidence-based anchors; the general motivation for bounded random search is [Bergstra & Bengio (2012)](https://www.jmlr.org/papers/v13/bergstra12a.html).
