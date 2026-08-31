"""Independent JSON/CSV audit; no training, inference, or selection mutations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_seal(record):
    payload = {k: v for k, v in record.items() if k != "fingerprint"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert record["fingerprint"] == hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def close(a, b, tolerance=1e-10):
    assert math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance), (a, b)


def audit(root: Path):
    study, selected = read(root / "study.json"), read(root / "selection.json")
    check_seal(study)
    check_seal(selected)
    assert selected["study_fingerprint"] == study["fingerprint"]
    assert selected["test_used_for_selection"] is False
    spec = study["spec"]
    split = study["split"]
    combined = split["train"] + split["validation"] + split["test"]
    assert len(combined) == len(set(combined))
    assert sorted(combined) == list(range(len(combined)))
    indices = np.random.default_rng(42).permutation(len(combined))
    assert split["train"] == indices[:int(len(indices) * 0.8)].tolist()
    assert split["validation"] == indices[int(len(indices) * 0.8):int(len(indices) * 0.9)].tolist()
    assert split["test"] == indices[int(len(indices) * 0.9):].tolist()
    assert sha(Path(study["data"]["path"])) == study["data"]["sha256"]
    for name, expected in study["source_sha256"].items():
        assert sha(root / "sources" / name) == expected
        assert sha(Path(__file__).resolve().parents[1] / "Codes" / name) == expected
    for relative, expected in selected["artifact_sha256"].items():
        assert sha(root / relative) == expected, relative
    promotion = read(root / "promotion.json")
    check_seal(promotion)
    assert sha(root / "promotion.json") == selected["promotion_sha256"]
    records, total_epochs = {}, 0
    for path in sorted((root / "runs").glob("*/result.json")):
        record = read(path)
        check_seal(record)
        config = record["configuration"]
        assert config["tuning"]["study_fingerprint"] == study["fingerprint"]
        assert record["status"] == "validation_complete" and record["test_evaluated"] is False
        assert "test_metrics" not in json.dumps(record)
        key = (config["tuning"]["candidate_id"], config["model"]["name"],
               config["observation"]["g05_fraction"], config["training"]["seed"])
        assert key not in records
        records[key] = record
        assert config["training"]["max_epochs"] == spec["max_epochs"]
        assert config["training"]["batch_size"] == spec["batch_size"]
        assert config["training"]["loss_weights"] == dict.fromkeys(("position", "magnitude", "relative_sign", "global_sign"), 1.0)
        assert config["normalization"] == study["normalization"]
        candidate = next(c for c in study["candidates"] if c["id"] == key[0])
        for parameter in ("learning_rate", "weight_decay"):
            assert config["training"][parameter] == candidate[parameter]
        assert config["training"]["regularization"]["structure_dropout"] == candidate["structure_dropout"]
        history = read(path.with_name("history.json"))
        assert sha(path.with_name("history.json")) == record["history_sha256"]
        assert [r["epoch"] for r in history] == list(range(1, len(history) + 1))
        assert len(history) == record["training_result"]["epochs_completed"]
        total_epochs += len(history)
        bests, last_improved = {"structure": math.inf, "total": math.inf}, 0
        stopped_at = None
        for row in history:
            for phase in ("train", "validation"):
                loss = row[phase]
                assert all(v is None or math.isfinite(v) for v in loss.values())
                close(loss["structure"], loss["position"] + loss["magnitude"] + loss["relative_sign"], 2e-6)
                close(loss["total"], loss["structure"] + (loss["global_sign"] or 0), 2e-6)
            if any(row["validation"][k] < bests[k] for k in bests):
                last_improved = row["epoch"]
            bests = {k: min(v, row["validation"][k]) for k, v in bests.items()}
            if spec["early_stopping_patience"] and row["epoch"] - last_improved >= spec["early_stopping_patience"]:
                stopped_at = row["epoch"]
                assert stopped_at == len(history), "Training continued after stopping"
        assert len(history) == spec["max_epochs"] or stopped_at is not None
        for objective in ("structure", "total"):
            expected = min(history, key=lambda r: r["validation"][objective])
            evaluation = record["evaluations"][objective]
            assert evaluation["selected_epoch"] == expected["epoch"]
            assert evaluation["validation_losses"] == expected["validation"]
            assert evaluation["selected_validation_loss"] == expected["validation"][objective]
            assert sha(root / evaluation["checkpoint_path"]) == evaluation["checkpoint_sha256"]
    expected_keys = set(itertools.product([c["id"] for c in study["candidates"]],
                                         spec["models"], spec["fractions"], spec["screen_seeds"]))
    expected_keys |= set(itertools.product(promotion["promoted"], spec["models"], spec["fractions"],
                                          spec["confirmation_seeds"]))
    assert set(records) == expected_keys

    def score(candidate, seeds):
        return statistics.mean(records[(candidate, model, fraction, seed)]["evaluations"]["structure"]["selected_validation_loss"]
                               for model, fraction, seed in itertools.product(spec["models"], spec["fractions"], seeds))

    screen_order = sorted((c["id"] for c in study["candidates"]),
                          key=lambda c: (score(c, spec["screen_seeds"]), c != "baseline", c))
    assert promotion["promoted"] == ["baseline"] + [c for c in screen_order if c != "baseline"][:spec["top_k"]]
    all_seeds = spec["screen_seeds"] + spec["confirmation_seeds"]
    assert selected["seeds"] == all_seeds
    scores = {c: score(c, all_seeds) for c in promotion["promoted"]}
    eligible = []
    for row in selected["ranking"]:
        candidate = row["candidate_id"]
        close(row["score"], scores[candidate])
        per_seed = [score(candidate, [s]) for s in all_seeds]
        close(row["seed_std"], statistics.stdev(per_seed))
        condition_regressions = []
        for model, fraction in itertools.product(spec["models"], spec["fractions"]):
            baseline = statistics.mean(records[("baseline", model, fraction, s)]["evaluations"]["structure"]["selected_validation_loss"]
                                       for s in all_seeds)
            actual = statistics.mean(records[(candidate, model, fraction, s)]["evaluations"]["structure"]["selected_validation_loss"]
                                     for s in all_seeds)
            close(row["by_condition"][f"{model}/g{fraction:g}"], actual)
            condition_regressions.append(100 * (actual - baseline) / baseline)
        is_eligible = max(condition_regressions) <= spec["max_model_regression_pct"] + 1e-12
        assert row["eligible"] == is_eligible
        if is_eligible:
            eligible.append(candidate)
    winner = min(eligible, key=lambda c: (scores[c], c != "baseline", c))
    if 100 * (scores["baseline"] - scores[winner]) / scores["baseline"] < spec["min_improvement_pct"]:
        winner = "baseline"
    assert winner == selected["selected_candidate_id"]
    expected_runs = set(itertools.product(set(("baseline", winner)), spec["models"], spec["fractions"], all_seeds))
    actual_runs = {(r["configuration"]["tuning"]["candidate_id"], r["configuration"]["model"]["name"],
                    r["configuration"]["observation"]["g05_fraction"], r["configuration"]["training"]["seed"])
                   for r in selected["evaluation_runs"]}
    assert actual_runs == expected_runs
    assert len(actual_runs) == len(selected["evaluation_runs"])
    final_count = 0
    final_path = root / "final" / "result.json"
    if final_path.exists():
        final = read(final_path)
        check_seal(final)
        assert final["selection_fingerprint"] == selected["fingerprint"]
        marker = read(root / "final_evaluation_started.json")
        assert marker["selection_fingerprint"] == selected["fingerprint"]
        assert datetime.fromisoformat(marker["started_at"]) >= datetime.fromisoformat(selected["locked_at"])
        final_keys = set()
        for record in final["records"]:
            check_seal(record)
            assert record["selection_fingerprint"] == selected["fingerprint"]
            assert datetime.fromisoformat(record["evaluated_at"]) >= datetime.fromisoformat(marker["started_at"])
            key = (record["candidate_id"], record["model"], record["fraction"], record["seed"])
            assert key in expected_runs
            expected_ckpt = records[key]["evaluations"][record["checkpoint_selection"]]
            assert record["checkpoint_sha256"] == expected_ckpt["checkpoint_sha256"]
            assert record["selected_epoch"] == expected_ckpt["selected_epoch"]
            assert record["sample_count"] == (len(split["test"]) if record["split"] == "historical_test"
                                               else spec["fresh_test"]["samples"])
            for value in [*record["metrics"].values(), *record["losses"].values()]:
                assert value is None or math.isfinite(value)
            final_keys.add((*key, record["split"], record["checkpoint_selection"]))
        assert len(final_keys) == len(final["records"]) == len(expected_runs) * 4
        for path, expected in final["artifact_sha256"].items():
            assert sha(root / path) == expected
        with (root / "paired_comparisons.csv").open(encoding="utf-8-sig", newline="") as file:
            comparisons = list(csv.DictReader(file))
        for row in comparisons:
            if row["baseline"] and row["selected"]:
                before, after = float(row["baseline"]), float(row["selected"])
                delta = after - before if row["metric"].endswith("accuracy") else before - after
                close(float(row["improvement"]), delta)
                if before:
                    close(float(row["improvement_pct"]), 100 * delta / abs(before))
        final_count = len(final_keys)
    return {"passed": True, "study_fingerprint": study["fingerprint"], "trial_count": len(records),
            "total_training_epochs": total_epochs, "selected_candidate": winner,
            "validation_structure_scores": scores, "final_evaluation_records": final_count,
            "checks": ["disjoint fixed split", "same hyperparameters/seed budget/loss definition",
                       "all scheduled trials complete", "first validation minima and loss decomposition",
                       "early stopping replay", "data/source artifact identity", "baseline-inclusive promotion",
                       "independent ranking and guardrail calculation", "frozen checkpoints",
                       "test timestamp gate and full paired evaluation matrix", "comparison arithmetic"],
            "limitations": "Checks saved artifacts and arithmetic; exact training replay is tested separately."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parent / "studies" / "main")
    args = parser.parse_args()
    result = audit(args.study_dir)
    path = args.study_dir / ("independent_final_audit.json" if result["final_evaluation_records"] else "independent_selection_audit.json")
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
