#!/usr/bin/env python3
"""
Comparative analysis of regular LINC (neurosymbolic) and D&C LINC (dcneurosymbolic) 
performance on examples containing logical operators.

This script analyzes when:
- Both models succeed
- One model succeeds while the other fails
- Both models fail

It also provides operator-specific analysis to gain insights into which 
operators are handled better by each approach.
"""

import json
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Set, Optional, Union, Any

# Default file paths
DEFAULT_BASE = Path("/app/linc/outputs")
DEFAULT_NEURO_GENS = DEFAULT_BASE / "Mistral-7B-v0.1_folio-neurosymbolic-1shot_generations_prc.json"
DEFAULT_DC_GENS = DEFAULT_BASE / "Mistral-7B-v0.1_folio-dcneurosymbolic-1shot_generations_prc.json"
DEFAULT_REFS = DEFAULT_BASE / "Mistral-7B-v0.1_folio-neurosymbolic-1shot_references.json" # Both use same references
DEFAULT_OUTPUT_DIR = Path("/app/linc/dc_analysis/analysis3")
DEFAULT_RESULTS_FILE = DEFAULT_OUTPUT_DIR / "comparative_results.json"
DEFAULT_CSV_FILE = DEFAULT_OUTPUT_DIR / "comparative_results.csv"
DEFAULT_FIG_DIR = DEFAULT_OUTPUT_DIR / "figures"

# Operator indices from the provided dictionary
OPERATOR_INDICES = {
    "OR": [0, 60, 82, 94, 124, 128, 165],
    "XOR": [3, 10, 58, 129, 130],
    "NOR": [4, 74, 90, 112, 177],
    "IMPLIES": [9, 31, 38, 40, 48, 63, 78, 84, 95, 97, 103, 104, 116, 137, 163, 169, 173, 180],
    "NOT": [11, 12, 15, 30, 34, 36, 41, 45, 53, 71, 75, 93, 138, 140, 148, 150, 151, 156, 162, 166, 171, 181],
    "AND": [13, 18, 25, 49, 50, 61, 62, 64, 86, 101, 118, 127, 167, 174, 179]
}

# Combine all indices into a flat list for overall analysis
ALL_OPERATOR_INDICES = []
for indices in OPERATOR_INDICES.values():
    ALL_OPERATOR_INDICES.extend(indices)
ALL_OPERATOR_INDICES.sort()


def majority_vote_ignore_error(row):
    """
    Compute majority vote for a row of predictions, ignoring "Error" values.
    """
    if isinstance(row, list):
        votes = [x for x in row if x != "Error"]
        if not votes:
            return "Uncertain"
        
        counts = Counter(votes)
        most_common = counts.most_common()
        if most_common:
            max_count = most_common[0][1]
            for vote, count in most_common:
                if count == max_count:
                    return vote
            return most_common[0][0]
        return "Uncertain"
    
    # Handle DC-LINC case where each row is a single-element list
    return row[0] if isinstance(row, list) else row


def load_json(path):
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    """Save data as JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved results to: {path}")


def compute_comparative_results(neuro_gens, dc_gens, references, indices):
    """
    Compute the success/failure patterns for both models on the specified indices.
    
    Returns:
    - Dictionary with comparative statistics
    - DataFrame with per-example results
    """
    # Preprocess to use only specified indices
    filtered_refs = [references[i] for i in indices if i < len(references)]
    filtered_neuro = [neuro_gens[i] for i in indices if i < len(neuro_gens)]
    filtered_dc = [dc_gens[i] for i in indices if i < len(dc_gens)]
    valid_indices = [i for i in indices if i < len(references)]
    
    # Compute majority votes for neurosymbolic
    neuro_preds = [majority_vote_ignore_error(row) for row in filtered_neuro]
    
    # For DC neurosymbolic, each item is already a single prediction
    dc_preds = [majority_vote_ignore_error(row) for row in filtered_dc]
    
    # Check which predictions match references
    neuro_correct = [pred == ref for pred, ref in zip(neuro_preds, filtered_refs)]
    dc_correct = [pred == ref for pred, ref in zip(dc_preds, filtered_refs)]
    
    # Count success/failure patterns
    both_success = sum(1 for n, dc in zip(neuro_correct, dc_correct) if n and dc)
    both_fail = sum(1 for n, dc in zip(neuro_correct, dc_correct) if not n and not dc)
    neuro_only = sum(1 for n, dc in zip(neuro_correct, dc_correct) if n and not dc)
    dc_only = sum(1 for n, dc in zip(neuro_correct, dc_correct) if not n and dc)
    
    total = len(valid_indices)
    
    # Build comparison DataFrame
    data = {
        "index": valid_indices,
        "reference": filtered_refs,
        "neuro_pred": neuro_preds,
        "dc_pred": dc_preds,
        "neuro_correct": neuro_correct,
        "dc_correct": dc_correct,
        "result_category": ["both_success" if n and dc else 
                           "neuro_only" if n and not dc else
                           "dc_only" if not n and dc else
                           "both_fail" for n, dc in zip(neuro_correct, dc_correct)]
    }
    df = pd.DataFrame(data)
    
    # Compute statistics
    stats = {
        "total_examples": total,
        "both_success": {
            "count": both_success,
            "percent": round((both_success / total) * 100, 2) if total > 0 else 0
        },
        "both_fail": {
            "count": both_fail,
            "percent": round((both_fail / total) * 100, 2) if total > 0 else 0
        },
        "neuro_only": {
            "count": neuro_only,
            "percent": round((neuro_only / total) * 100, 2) if total > 0 else 0
        },
        "dc_only": {
            "count": dc_only,
            "percent": round((dc_only / total) * 100, 2) if total > 0 else 0
        },
        "neuro_overall": {
            "count": sum(neuro_correct),
            "percent": round((sum(neuro_correct) / total) * 100, 2) if total > 0 else 0
        },
        "dc_overall": {
            "count": sum(dc_correct),
            "percent": round((sum(dc_correct) / total) * 100, 2) if total > 0 else 0
        }
    }
    
    return stats, df


def analyze_by_operator(results_df):
    """
    Analyze results broken down by operator type.
    """
    operator_stats = {}
    operator_dfs = {}
    
    for op_name, indices in OPERATOR_INDICES.items():
        # Filter the DataFrame for this operator
        op_df = results_df[results_df["index"].isin(indices)].copy()
        
        if len(op_df) == 0:
            continue
            
        # Compute statistics for this operator
        both_success = sum((op_df["neuro_correct"] & op_df["dc_correct"]))
        both_fail = sum((~op_df["neuro_correct"] & ~op_df["dc_correct"]))
        neuro_only = sum((op_df["neuro_correct"] & ~op_df["dc_correct"]))
        dc_only = sum((~op_df["neuro_correct"] & op_df["dc_correct"]))
        total = len(op_df)
        
        op_stats = {
            "total_examples": total,
            "both_success": {
                "count": both_success,
                "percent": round((both_success / total) * 100, 2) if total > 0 else 0
            },
            "both_fail": {
                "count": both_fail,
                "percent": round((both_fail / total) * 100, 2) if total > 0 else 0
            },
            "neuro_only": {
                "count": neuro_only,
                "percent": round((neuro_only / total) * 100, 2) if total > 0 else 0
            },
            "dc_only": {
                "count": dc_only,
                "percent": round((dc_only / total) * 100, 2) if total > 0 else 0
            },
            "neuro_overall": {
                "count": sum(op_df["neuro_correct"]),
                "percent": round((sum(op_df["neuro_correct"]) / total) * 100, 2) if total > 0 else 0
            },
            "dc_overall": {
                "count": sum(op_df["dc_correct"]),
                "percent": round((sum(op_df["dc_correct"]) / total) * 100, 2) if total > 0 else 0
            }
        }
        
        operator_stats[op_name] = op_stats
        operator_dfs[op_name] = op_df
    
    return operator_stats, operator_dfs


def plot_comparative_results(stats, output_dir=DEFAULT_FIG_DIR):
    """Generate comparative bar charts showing the performance differences."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Overall comparison (both models)
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["Both Success", "Neuro Only", "DC Only", "Both Fail"]
    counts = [
        stats["both_success"]["count"],
        stats["neuro_only"]["count"],
        stats["dc_only"]["count"],
        stats["both_fail"]["count"]
    ]
    percentages = [
        stats["both_success"]["percent"],
        stats["neuro_only"]["percent"],
        stats["dc_only"]["percent"],
        stats["both_fail"]["percent"]
    ]
    
    colors = ['green', 'blue', 'orange', 'red']
    bars = ax.bar(categories, counts, color=colors)
    
    # Add counts and percentages as labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + 0.5,
            f"{counts[i]} ({percentages[i]}%)",
            ha='center', va='bottom'
        )
    
    ax.set_title("Comparison of LINC and DC-LINC on Operator Examples")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_comparison.png"))
    
    # Plot 2: Individual model accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    model_names = ["LINC (Neurosymbolic)", "DC-LINC"]
    accuracy = [stats["neuro_overall"]["percent"], stats["dc_overall"]["percent"]]
    
    bars = ax.bar(model_names, accuracy, color=['blue', 'orange'])
    
    # Add percentages as labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = stats["neuro_overall"]["count"] if i == 0 else stats["dc_overall"]["count"]
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + 0.5,
            f"{count}/{stats['total_examples']} ({accuracy[i]}%)",
            ha='center', va='bottom'
        )
    
    ax.set_title("Overall Accuracy: LINC vs DC-LINC")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)  # Set y-axis to percentage scale
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
    
    return [
        os.path.join(output_dir, "overall_comparison.png"),
        os.path.join(output_dir, "accuracy_comparison.png")
    ]


def plot_operator_comparison(operator_stats, output_dir=DEFAULT_FIG_DIR):
    """Generate operator-specific comparison charts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Accuracy by operator type
    fig, ax = plt.subplots(figsize=(12, 8))
    
    operators = list(operator_stats.keys())
    neuro_acc = [operator_stats[op]["neuro_overall"]["percent"] for op in operators]
    dc_acc = [operator_stats[op]["dc_overall"]["percent"] for op in operators]
    
    # Get sample sizes for annotation
    sample_sizes = [operator_stats[op]["total_examples"] for op in operators]
    
    x = np.arange(len(operators))  # the label locations
    width = 0.35  # the width of the bars
    
    rects1 = ax.bar(x - width/2, neuro_acc, width, label='LINC (Neurosymbolic)')
    rects2 = ax.bar(x + width/2, dc_acc, width, label='DC-LINC')
    
    # Add labels, title and axis ticks
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Operator Type')
    ax.set_xticks(x)
    ax.set_xticklabels(operators)
    ax.legend()
    ax.set_ylim(0, 100)
    
    # Add sample size and accuracy annotations
    for i, (r1, r2) in enumerate(zip(rects1, rects2)):
        height1 = r1.get_height()
        height2 = r2.get_height()
        count1 = operator_stats[operators[i]]["neuro_overall"]["count"]
        count2 = operator_stats[operators[i]]["dc_overall"]["count"]
        
        # Add LINC annotation
        ax.annotate(f'{height1}%\n({count1}/{sample_sizes[i]})',
                    xy=(r1.get_x() + r1.get_width() / 2, height1),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
        
        # Add DC-LINC annotation
        ax.annotate(f'{height2}%\n({count2}/{sample_sizes[i]})',
                    xy=(r2.get_x() + r2.get_width() / 2, height2),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "operator_accuracy.png"))
    
    # Plot 2: Comparative results by operator
    fig, axs = plt.subplots(len(operators), 1, figsize=(10, 3*len(operators)))
    
    categories = ["Both Success", "Neuro Only", "DC Only", "Both Fail"]
    colors = ['green', 'blue', 'orange', 'red']
    
    for i, op in enumerate(operators):
        ax = axs[i] if len(operators) > 1 else axs
        
        counts = [
            operator_stats[op]["both_success"]["count"],
            operator_stats[op]["neuro_only"]["count"],
            operator_stats[op]["dc_only"]["count"],
            operator_stats[op]["both_fail"]["count"]
        ]
        
        percentages = [
            operator_stats[op]["both_success"]["percent"],
            operator_stats[op]["neuro_only"]["percent"],
            operator_stats[op]["dc_only"]["percent"],
            operator_stats[op]["both_fail"]["percent"]
        ]
        
        bars = ax.bar(categories, counts, color=colors)
        
        # Add counts and percentages as labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:  # Only add label if there's a visible bar
                ax.text(
                    bar.get_x() + bar.get_width() / 2, height + 0.1,
                    f"{counts[j]} ({percentages[j]}%)",
                    ha='center', va='bottom', fontsize=8
                )
        
        ax.set_title(f"Operator: {op} (n={operator_stats[op]['total_examples']})")
        ax.set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "operator_comparison_detail.png"))
    
    return [
        os.path.join(output_dir, "operator_accuracy.png"),
        os.path.join(output_dir, "operator_comparison_detail.png")
    ]


def main():
    parser = argparse.ArgumentParser(description="Compare LINC and DC-LINC performance on operator examples.")
    parser.add_argument("--neuro", type=Path, default=DEFAULT_NEURO_GENS, help="Path to neurosymbolic generations JSON")
    parser.add_argument("--dc", type=Path, default=DEFAULT_DC_GENS, help="Path to DC neurosymbolic generations JSON")
    parser.add_argument("--refs", type=Path, default=DEFAULT_REFS, help="Path to references JSON")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_FILE, help="Output CSV file path")
    parser.add_argument("--json", type=Path, default=DEFAULT_RESULTS_FILE, help="Output JSON file path")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading neurosymbolic generations from: {args.neuro}")
    neuro_gens = load_json(args.neuro)
    
    print(f"Loading DC neurosymbolic generations from: {args.dc}")
    dc_gens = load_json(args.dc)
    
    print(f"Loading references from: {args.refs}")
    references = load_json(args.refs)
    
    # Compute overall comparative results
    print("\nComputing overall comparative results...")
    overall_stats, overall_df = compute_comparative_results(
        neuro_gens, dc_gens, references, ALL_OPERATOR_INDICES
    )
    
    # Analyze by operator type
    print("\nAnalyzing results by operator type...")
    operator_stats, operator_dfs = analyze_by_operator(overall_df)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    overall_plots = plot_comparative_results(overall_stats, os.path.join(args.output_dir, "figures"))
    operator_plots = plot_operator_comparison(operator_stats, os.path.join(args.output_dir, "figures"))
    
    # Save results
    print("\nSaving results...")
    combined_results = {
        "overall": overall_stats,
        "by_operator": operator_stats
    }
    
    save_json(combined_results, args.json)
    overall_df.to_csv(args.csv, index=False)
    
    # Save operator-specific CSVs
    for op, df in operator_dfs.items():
        op_csv = os.path.join(args.output_dir, f"{op}_results.csv")
        df.to_csv(op_csv, index=False)
        print(f"Saved {op} results to: {op_csv}")
    
    # Print summary
    print("\n=== COMPARATIVE ANALYSIS SUMMARY ===")
    print(f"Total examples analyzed: {overall_stats['total_examples']}")
    print(f"Both models correct: {overall_stats['both_success']['count']} ({overall_stats['both_success']['percent']}%)")
    print(f"Only LINC correct: {overall_stats['neuro_only']['count']} ({overall_stats['neuro_only']['percent']}%)")
    print(f"Only DC-LINC correct: {overall_stats['dc_only']['count']} ({overall_stats['dc_only']['percent']}%)")
    print(f"Both models incorrect: {overall_stats['both_fail']['count']} ({overall_stats['both_fail']['percent']}%)")
    print("\nOverall accuracy:")
    print(f"LINC (Neurosymbolic): {overall_stats['neuro_overall']['percent']}%")
    print(f"DC-LINC: {overall_stats['dc_overall']['percent']}%")
    
    print("\n=== OPERATOR-SPECIFIC ACCURACY ===")
    for op, stats in operator_stats.items():
        print(f"{op} (n={stats['total_examples']}): LINC = {stats['neuro_overall']['percent']}%, DC-LINC = {stats['dc_overall']['percent']}%")
    
    print("\nAnalysis complete! Results saved to:")
    print(f"  - JSON: {args.json}")
    print(f"  - CSV: {args.csv}")
    print(f"  - Figures: {os.path.join(args.output_dir, 'figures')}")


if __name__ == "__main__":
    main()