# Seed 42: fixed dropout020 fraction experiment

Verified 12/12 independent runs, 906 epochs and 36 checkpoints.

Settings: AdamW, lr=0.001, weight decay=0.0001, structure dropout=0.2, batch=128, maximum epochs=150, dual-objective patience=20, min_delta=0, min_epochs=0.
Fixed 80/10/10 split (seed 42), train-only normalization, unchanged loss weights and 120-permutation matching.

Primary table uses best_structure.pt selected only by validation. best_total.pt remains a separate saved evaluation; no checkpoint or fraction was chosen using test scores. Test metrics use the historical 1,000-sample test split, not a new holdout.

| Fraction | G05 points | Model | Epochs | Best structure epoch | Validation structure | Test 3D error | Test magnitude MAE |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0 | sign-only | 75 | 55 | 0.579643 | 0.722286 | 0.158173 |
| 0 | 0 | full | 75 | 55 | 0.579643 | 0.722286 | 0.158173 |
| 0.1 | 3 | sign-only | 83 | 55 | 0.579643 | 0.722286 | 0.158173 |
| 0.1 | 3 | full | 64 | 44 | 0.581422 | 0.728135 | 0.154857 |
| 0.25 | 8 | sign-only | 83 | 55 | 0.579643 | 0.722286 | 0.158173 |
| 0.25 | 8 | full | 64 | 44 | 0.574682 | 0.732711 | 0.155263 |
| 0.5 | 16 | sign-only | 80 | 55 | 0.579643 | 0.722286 | 0.158173 |
| 0.5 | 16 | full | 77 | 43 | 0.579491 | 0.734606 | 0.155313 |
| 0.75 | 24 | sign-only | 80 | 55 | 0.579643 | 0.722286 | 0.158173 |
| 0.75 | 24 | full | 76 | 49 | 0.574576 | 0.726735 | 0.155683 |
| 1 | 32 | sign-only | 80 | 55 | 0.579643 | 0.722286 | 0.158173 |
| 1 | 32 | full | 69 | 49 | 0.580052 | 0.721829 | 0.154723 |

Only this seed is included. One seed cannot establish variance or statistical significance. Identical hardware/software is required for strict reproducibility; cross-device bitwise equality is not guaranteed.

Existing-artifact preservation: 521 files unchanged.

[All recorded metrics](runs.csv) | [Summary](summary.csv) | [Paired model comparisons](pairwise_summary.csv) | [Protocol](protocol.json) | [Audit](seed_audit.json) | [Checkpoint hashes](checkpoint_hashes.json)

From the project root, fresh execution on another computer:

```powershell
.\Modelexperiment11\run_fixed_seed.ps1 -Seed 42
```

Use that computer's assigned seed. The launcher rejects existing output directories unless -Resume is explicit. It checks frozen source/data hashes and never reads another seed's results or weights.
