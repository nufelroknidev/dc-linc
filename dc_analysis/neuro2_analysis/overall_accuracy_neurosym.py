#!/usr/bin/env python3
"""
Compute accuracy for FOLIO neuro-symbolic runs with deterministic majority vote,
focusing only on conclusions that contain operators.

Rules:
- Only evaluate examples from operator_indices_flat.json (indices with operator != NONE)
- Ignore tokens equal to "Error" when voting.
- Majority vote over remaining labels.
- Deterministic tie-break: pick the first label encountered among the tied labels.
- If a row contains only "Error", default to "Uncertain".

Examples
--------
python overall_accuracy_neurosym.py \
  --gens /app/linc/outputs/Mistral-7B-v0.1_folio-neurosymbolic-1shot_generations_prc.json \
  --refs /app/linc/outputs/Mistral-7B-v0.1_folio-neurosymbolic-1shot_references.json \
  --save-csv /app/linc/dc_analysis/neuro2_analysis/neurosym_operator_predictions.csv \
  --save-json /app/linc/dc_analysis/neuro2_analysis/neurosym_operator_summary.json
"""

from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional
import csv
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

DEFAULT_BASE = Path("/app/linc/outputs")
DEFAULT_GENS = DEFAULT_BASE / "Mistral-7B-v0.1_folio-neurosymbolic-1shot_generations_prc.json"
DEFAULT_REFS = DEFAULT_BASE / "Mistral-7B-v0.1_folio-neurosymbolic-1shot_references.json"
DEFAULT_INDICES = Path("/app/linc/dc_analysis/analysis2/operator_indices_flat.json")
DEFAULT_CSV = Path("/app/linc/dc_analysis/neuro2_analysis/neurosym_operator_predictions.csv")
DEFAULT_JSON = Path("/app/linc/dc_analysis/neuro2_analysis/neurosym_operator_summary.json")
DEFAULT_FIG = Path("/app/linc/dc_analysis/neuro2_analysis/neurosym_operator_confusion_matrix.png")


def majority_vote_ignore_error(row: Iterable[str]) -> str:
    votes: List[str] = [x for x in row if x != "Error"]
    if not votes:
        return "Uncertain"

    counts = Counter()
    seen_order: List[str] = []
    seen = set()
    for x in votes:
        counts[x] += 1
        if x not in seen:
            seen_order.append(x)
            seen.add(x)

    maxc = max(counts.values())
    for x in seen_order:
        if counts[x] == maxc:
            return x
    return votes[0]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(path: Path, indices: List[int], references: List[str], predictions: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "reference", "prediction", "correct"])  # header
        for idx, r, p in zip(indices, references, predictions):
            w.writerow([idx, r, p, str(p == r)])


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def plot_confusion(references: List[str],
                  predictions: List[str],
                  save_path: Path | None = None,
                  labels: Optional[List[str]] = None,
                  normalize: Optional[str] = None):
    """Plot a confusion matrix with a **fixed, explicit label order** and optional normalisation.

    - If `labels` is None, we default to ["True", "False", "Uncertain"],
      restricted to those that actually appear, then append any unexpected labels.
    - `normalize` may be one of {None, "true", "pred", "all"} per scikit‑learn.
    - Annotations show counts, and if normalized, also percentages.
    """
    present = sorted(set(references) | set(predictions))
    if labels is None:
        preferred = ["True", "False", "Uncertain"]
        labels = [l for l in preferred if l in present] + [l for l in present if l not in set(preferred)]

    # Compute counts and (optionally) normalized matrix
    cm_counts = confusion_matrix(references, predictions, labels=labels)
    norm = None if (normalize is None or str(normalize).lower() == "none") else normalize
    if norm:
        cm = confusion_matrix(references, predictions, labels=labels, normalize=norm)
    else:
        cm = cm_counts.astype(float)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm if norm else cm_counts, cmap="Blues", interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Proportion" if norm else "Count", rotation=90, va="bottom")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells (counts and, if normalized, percentages)
    for i in range(len(labels)):
        for j in range(len(labels)):
            cell_txt = f"{cm_counts[i, j]}"
            if norm:
                cell_txt += f"\n({cm[i, j]*100:.1f}%)"
            
            # Choose text color based on cell darkness
            text_color = "white" if (cm[i, j] > 0.5 and norm) or (cm_counts[i, j] > cm_counts.max()/2 and not norm) else "black"
            ax.text(j, i, cell_txt, ha="center", va="center", color=text_color)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    ax.set_title("Confusion Matrix (LINC Operator-only Examples)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved confusion matrix figure -> {save_path}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=Path, default=DEFAULT_GENS, help="Path to generations JSON")
    ap.add_argument("--refs", type=Path, default=DEFAULT_REFS, help="Path to references JSON")
    ap.add_argument("--indices", type=Path, default=DEFAULT_INDICES, help="Path to operator indices JSON")
    ap.add_argument("--save-csv", type=Path, default=DEFAULT_CSV, help="Optional path to write per-example CSV")
    ap.add_argument("--save-json", type=Path, default=DEFAULT_JSON, help="Optional path to write summary JSON")
    ap.add_argument("--save-fig", type=Path, default=DEFAULT_FIG, help="Optional path to save confusion matrix figure")
    ap.add_argument("--normalize", choices=["none", "true", "pred", "all"], default="none",
                help="Normalization for confusion matrix: row-wise ('true'), column-wise ('pred'), overall ('all'), or 'none'.")
    args = ap.parse_args()

    # Load data
    generations = load_json(args.gens)
    references = load_json(args.refs)
    operator_indices = load_json(args.indices)

    print(f"Loaded {len(generations)} generations from {args.gens}")
    print(f"Loaded {len(references)} references from {args.refs}")
    print(f"Loaded {len(operator_indices)} operator indices from {args.indices}")
    
    if len(generations) != len(references):
        raise ValueError(f"Length mismatch: {len(generations)} generations vs {len(references)} references.")

    # Filter data to only include examples with operators
    filtered_gens = []
    filtered_refs = []
    valid_indices = []
    
    for idx in operator_indices:
        if 0 <= idx < len(generations):
            filtered_gens.append(generations[idx])
            filtered_refs.append(references[idx])
            valid_indices.append(idx)
    
    print(f"Filtered to {len(filtered_gens)} examples with operators (excluding NONE)")

    # Calculate predictions and accuracy
    predictions = [majority_vote_ignore_error(row) for row in filtered_gens]

    correct = sum(int(p == r) for p, r in zip(predictions, filtered_refs))
    n = len(filtered_refs)
    accuracy = correct / n if n else 0.0

    # Classification report (precision/recall/F1)
    report = classification_report(filtered_refs, predictions, output_dict=True, zero_division=0)

    # Print concise summary
    print("\nOperator-only Accuracy Analysis")
    print("------------------------------")
    print(f"Count:    {n}")
    print(f"Correct:  {correct}")
    print(f"Accuracy (pass@1 majority): {accuracy:.3f}")
    print("\nPer-class Precision/Recall/F1:")
    for label, stats in report.items():
        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue
        print(f"{label:10s} P={stats['precision']:.3f} R={stats['recall']:.3f} F1={stats['f1-score']:.3f}")

    # Save results
    if args.save_csv:
        save_csv(args.save_csv, valid_indices, filtered_refs, predictions)
        print(f"Saved per-example results -> {args.save_csv}")
    
    if args.save_json:
        summary = {
            "count": n, 
            "correct": correct, 
            "accuracy": accuracy, 
            "report": report,
            "operator_indices_count": len(operator_indices),
            "valid_indices_count": len(valid_indices),
            "valid_indices": valid_indices
        }
        save_json(args.save_json, summary)
        print(f"Saved summary -> {args.save_json}")
    
    if args.save_fig:
        # Stable label order for display: True, False, Uncertain (then any others)
        present = sorted(set(filtered_refs) | set(predictions))
        preferred = ["True", "False", "Uncertain"]
        label_order = [l for l in preferred if l in present] + [l for l in present if l not in set(preferred)]
        norm = None if args.normalize == "none" else args.normalize
        print(f"Confusion matrix label order: {label_order}")
        print(f"Confusion matrix normalization: {args.normalize}")
        plot_confusion(references=filtered_refs, predictions=predictions, save_path=args.save_fig, 
                      labels=label_order, normalize=norm)


if __name__ == "__main__":
    main()