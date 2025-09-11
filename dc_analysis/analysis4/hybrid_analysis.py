#!/usr/bin/env python3
"""
Hybrid Model Analysis for LINC and DC-LINC.

This script implements and evaluates a hybrid model that uses:
- Neurosymbolic (LINC) predictions when they are True or False
- DC-LINC predictions when LINC returns "Uncertain"

The analysis is focused on specific logical operators (OR, XOR, NOR, IMPLIES, NOT, AND)
and compares the performance of this hybrid approach with the individual models.
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
DEFAULT_REFS = DEFAULT_BASE / "Mistral-7B-v0.1_folio-neurosymbolic-1shot_references.json"  # Both use same references
DEFAULT_OUTPUT_DIR = Path("/app/linc/dc_analysis/analysis4")
DEFAULT_RESULTS_FILE = DEFAULT_OUTPUT_DIR / "hybrid_results.json"
DEFAULT_CSV_FILE = DEFAULT_OUTPUT_DIR / "hybrid_results.csv"
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


def hybrid_prediction(neuro_pred, dc_pred):
    """
    Implement the hybrid model strategy:
    - Use neurosymbolic prediction if it's True or False
    - Fall back to DC-LINC prediction when neurosymbolic is Uncertain
    """
    if neuro_pred in [True, False, "True", "False"]:
        return neuro_pred
    else:
        return dc_pred


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


def format_table(rows, header=True):
    """Simple table formatter that returns a string with a formatted table."""
    if not rows:
        return ""
    
    # Calculate the maximum width for each column
    col_widths = [0] * len(rows[0])
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Format the table
    result = []
    
    # Add header
    header_row = "| " + " | ".join([str(cell).ljust(col_widths[i]) for i, cell in enumerate(rows[0])]) + " |"
    result.append(header_row)
    
    # Add separator
    separator = "|-" + "-|-".join(["-" * col_widths[i] for i in range(len(col_widths))]) + "-|"
    result.append(separator)
    
    # Add data rows
    start_idx = 1 if header else 0
    for row in rows[start_idx:]:
        formatted_row = "| " + " | ".join([str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)]) + " |"
        result.append(formatted_row)
    
    return "\n".join(result)


def compute_hybrid_results(neuro_gens, dc_gens, references, indices):
    """
    Compute the results for individual models and the hybrid model.
    
    Returns:
    - Dictionary with statistics
    - DataFrame with per-example results
    """
    # Preprocess to use only specified indices
    filtered_refs = [references[i] for i in indices if i < len(references)]
    filtered_neuro = [neuro_gens[i] for i in indices if i < len(neuro_gens)]
    filtered_dc = [dc_gens[i] for i in indices if i < len(dc_gens)]
    valid_indices = [i for i in indices if i < len(references)]
    
    # Compute majority votes for neurosymbolic
    neuro_preds = [majority_vote_ignore_error(row) for row in filtered_neuro]
    
    # For DC neurosymbolic, each item might need majority voting
    dc_preds = [majority_vote_ignore_error(row) for row in filtered_dc]
    
    # Compute hybrid predictions
    hybrid_preds = [hybrid_prediction(n, d) for n, d in zip(neuro_preds, dc_preds)]
    
    # Check which predictions match references
    neuro_correct = [pred == ref for pred, ref in zip(neuro_preds, filtered_refs)]
    dc_correct = [pred == ref for pred, ref in zip(dc_preds, filtered_refs)]
    hybrid_correct = [pred == ref for pred, ref in zip(hybrid_preds, filtered_refs)]
    
    # Count neuro predictions by type
    neuro_true_false = sum(1 for p in neuro_preds if p in [True, False, "True", "False"])
    neuro_uncertain = sum(1 for p in neuro_preds if p not in [True, False, "True", "False"])
    
    # Count improvement scenarios
    neuro_wrong_hybrid_right = sum(1 for n, h in zip(neuro_correct, hybrid_correct) if not n and h)
    dc_wrong_hybrid_right = sum(1 for d, h in zip(dc_correct, hybrid_correct) if not d and h)
    both_wrong_hybrid_right = sum(1 for n, d, h in zip(neuro_correct, dc_correct, hybrid_correct) if not n and not d and h)
    
    total = len(valid_indices)
    
    # Build comparison DataFrame
    data = {
        "index": valid_indices,
        "reference": filtered_refs,
        "neuro_pred": neuro_preds,
        "dc_pred": dc_preds,
        "hybrid_pred": hybrid_preds,
        "neuro_correct": neuro_correct,
        "dc_correct": dc_correct,
        "hybrid_correct": hybrid_correct
    }
    df = pd.DataFrame(data)
    
    # Add result category
    df["result_category"] = "other"
    # Both correct
    mask_both = (df["neuro_correct"] & df["dc_correct"])
    df.loc[mask_both, "result_category"] = "both_correct"
    # Hybrid improves on both
    mask_hybrid_improves = (~df["neuro_correct"] & ~df["dc_correct"] & df["hybrid_correct"])
    df.loc[mask_hybrid_improves, "result_category"] = "hybrid_improves_both"
    # Hybrid improves on neuro
    mask_hybrid_improves_neuro = (~df["neuro_correct"] & df["hybrid_correct"])
    df.loc[mask_hybrid_improves_neuro, "result_category"] = "hybrid_improves_neuro"
    # Hybrid improves on dc
    mask_hybrid_improves_dc = (~df["dc_correct"] & df["hybrid_correct"])
    df.loc[mask_hybrid_improves_dc, "result_category"] = "hybrid_improves_dc"
    # All wrong
    mask_all_wrong = (~df["neuro_correct"] & ~df["dc_correct"] & ~df["hybrid_correct"])
    df.loc[mask_all_wrong, "result_category"] = "all_wrong"
    # Hybrid worse than neuro
    mask_hybrid_worse_neuro = (df["neuro_correct"] & ~df["hybrid_correct"])
    df.loc[mask_hybrid_worse_neuro, "result_category"] = "hybrid_worse_than_neuro"
    # Hybrid worse than dc
    mask_hybrid_worse_dc = (df["dc_correct"] & ~df["hybrid_correct"])
    df.loc[mask_hybrid_worse_dc, "result_category"] = "hybrid_worse_than_dc"
    
    # Compute statistics
    stats = {
        "total_examples": total,
        "neuro_stats": {
            "correct": sum(neuro_correct),
            "percent": round((sum(neuro_correct) / total) * 100, 2) if total > 0 else 0,
            "true_false_count": neuro_true_false,
            "true_false_percent": round((neuro_true_false / total) * 100, 2) if total > 0 else 0,
            "uncertain_count": neuro_uncertain,
            "uncertain_percent": round((neuro_uncertain / total) * 100, 2) if total > 0 else 0
        },
        "dc_stats": {
            "correct": sum(dc_correct),
            "percent": round((sum(dc_correct) / total) * 100, 2) if total > 0 else 0
        },
        "hybrid_stats": {
            "correct": sum(hybrid_correct),
            "percent": round((sum(hybrid_correct) / total) * 100, 2) if total > 0 else 0,
            "neuro_wrong_hybrid_right": neuro_wrong_hybrid_right,
            "dc_wrong_hybrid_right": dc_wrong_hybrid_right,
            "both_wrong_hybrid_right": both_wrong_hybrid_right
        },
        "improvement": {
            "over_neuro": round(((sum(hybrid_correct) - sum(neuro_correct)) / total) * 100, 2) if total > 0 else 0,
            "over_dc": round(((sum(hybrid_correct) - sum(dc_correct)) / total) * 100, 2) if total > 0 else 0
        },
        "category_counts": df["result_category"].value_counts().to_dict()
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
        total = len(op_df)
        neuro_correct = sum(op_df["neuro_correct"])
        dc_correct = sum(op_df["dc_correct"])
        hybrid_correct = sum(op_df["hybrid_correct"])
        
        # Count neuro predictions by type
        neuro_true_false = sum(1 for p in op_df["neuro_pred"] if p in [True, False, "True", "False"])
        neuro_uncertain = sum(1 for p in op_df["neuro_pred"] if p not in [True, False, "True", "False"])
        
        # Count improvement scenarios
        neuro_wrong_hybrid_right = sum(1 for n, h in zip(op_df["neuro_correct"], op_df["hybrid_correct"]) if not n and h)
        dc_wrong_hybrid_right = sum(1 for d, h in zip(op_df["dc_correct"], op_df["hybrid_correct"]) if not d and h)
        both_wrong_hybrid_right = sum(1 for n, d, h in 
                                zip(op_df["neuro_correct"], op_df["dc_correct"], op_df["hybrid_correct"]) 
                                if not n and not d and h)
        
        op_stats = {
            "total_examples": total,
            "neuro_stats": {
                "correct": neuro_correct,
                "percent": round((neuro_correct / total) * 100, 2) if total > 0 else 0,
                "true_false_count": neuro_true_false,
                "true_false_percent": round((neuro_true_false / total) * 100, 2) if total > 0 else 0,
                "uncertain_count": neuro_uncertain,
                "uncertain_percent": round((neuro_uncertain / total) * 100, 2) if total > 0 else 0
            },
            "dc_stats": {
                "correct": dc_correct,
                "percent": round((dc_correct / total) * 100, 2) if total > 0 else 0
            },
            "hybrid_stats": {
                "correct": hybrid_correct,
                "percent": round((hybrid_correct / total) * 100, 2) if total > 0 else 0,
                "neuro_wrong_hybrid_right": neuro_wrong_hybrid_right,
                "dc_wrong_hybrid_right": dc_wrong_hybrid_right,
                "both_wrong_hybrid_right": both_wrong_hybrid_right
            },
            "improvement": {
                "over_neuro": round(((hybrid_correct - neuro_correct) / total) * 100, 2) if total > 0 else 0,
                "over_dc": round(((hybrid_correct - dc_correct) / total) * 100, 2) if total > 0 else 0
            },
            "category_counts": op_df["result_category"].value_counts().to_dict()
        }
        
        operator_stats[op_name] = op_stats
        operator_dfs[op_name] = op_df
    
    return operator_stats, operator_dfs


def plot_hybrid_results(stats, output_dir=DEFAULT_FIG_DIR):
    """Generate comparative bar charts showing the performance differences."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Accuracy comparison across all three approaches
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ["LINC", "DC-LINC", "Hybrid"]
    accuracies = [
        stats["neuro_stats"]["percent"],
        stats["dc_stats"]["percent"],
        stats["hybrid_stats"]["percent"]
    ]
    
    colors = ['blue', 'orange', 'green']
    bars = ax.bar(models, accuracies, color=colors)
    
    # Add counts and percentages as labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if i == 0:
            count = stats["neuro_stats"]["correct"]
        elif i == 1:
            count = stats["dc_stats"]["correct"]
        else:
            count = stats["hybrid_stats"]["correct"]
            
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + 0.5,
            f"{count}/{stats['total_examples']} ({accuracies[i]}%)",
            ha='center', va='bottom'
        )
    
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
    
    # Plot 2: LINC prediction distribution (True/False vs Uncertain)
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ["True/False", "Uncertain"]
    counts = [
        stats["neuro_stats"]["true_false_count"],
        stats["neuro_stats"]["uncertain_count"]
    ]
    percentages = [
        stats["neuro_stats"]["true_false_percent"],
        stats["neuro_stats"]["uncertain_percent"]
    ]
    
    bars = ax.bar(categories, counts, color=['blue', 'gray'])
    
    # Add counts and percentages as labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + 0.5,
            f"{counts[i]} ({percentages[i]}%)",
            ha='center', va='bottom'
        )
    
    ax.set_title("LINC Prediction Distribution")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "neuro_prediction_distribution.png"))
    
    # Plot 3: Improvement of hybrid model over individual models
    fig, ax = plt.subplots(figsize=(8, 6))
    improvements = [
        stats["improvement"]["over_neuro"],
        stats["improvement"]["over_dc"]
    ]
    
    colors = ['lightblue', 'lightsalmon']
    if improvements[0] < 0:
        colors[0] = 'lightcoral'
    if improvements[1] < 0:
        colors[1] = 'lightcoral'
        
    bars = ax.bar(["Over LINC", "Over DC-LINC"], improvements, color=colors)
    
    # Add percentages as labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + 0.1 if height > 0 else height - 0.5,
            f"{improvements[i]}%",
            ha='center', va='bottom' if height > 0 else 'top'
        )
    
    ax.set_title("Hybrid Model Improvement")
    ax.set_ylabel("Improvement (%)")
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hybrid_improvement.png"))
    
    return [
        os.path.join(output_dir, "accuracy_comparison.png"),
        os.path.join(output_dir, "neuro_prediction_distribution.png"),
        os.path.join(output_dir, "hybrid_improvement.png")
    ]


def plot_operator_comparison(operator_stats, output_dir=DEFAULT_FIG_DIR):
    """Generate operator-specific comparison charts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Accuracy by operator type for all three models
    fig, ax = plt.subplots(figsize=(14, 8))
    
    operators = list(operator_stats.keys())
    neuro_acc = [operator_stats[op]["neuro_stats"]["percent"] for op in operators]
    dc_acc = [operator_stats[op]["dc_stats"]["percent"] for op in operators]
    hybrid_acc = [operator_stats[op]["hybrid_stats"]["percent"] for op in operators]
    
    x = np.arange(len(operators))  # the label locations
    width = 0.25  # the width of the bars
    
    rects1 = ax.bar(x - width, neuro_acc, width, label='LINC')
    rects2 = ax.bar(x, dc_acc, width, label='DC-LINC')
    rects3 = ax.bar(x + width, hybrid_acc, width, label='Hybrid')
    
    # Add labels, title and axis ticks
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Operator Type')
    ax.set_xticks(x)
    ax.set_xticklabels(operators)
    ax.legend()
    ax.set_ylim(0, 100)
    
    # Add sample size annotation
    for i, op in enumerate(operators):
        ax.annotate(f'n={operator_stats[op]["total_examples"]}',
                    xy=(i, 5),
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "operator_accuracy.png"))
    
    # Plot 2: Hybrid improvement over individual models by operator
    fig, ax = plt.subplots(figsize=(14, 8))
    
    neuro_imp = [operator_stats[op]["improvement"]["over_neuro"] for op in operators]
    dc_imp = [operator_stats[op]["improvement"]["over_dc"] for op in operators]
    
    x = np.arange(len(operators))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, neuro_imp, width, label='Improvement over LINC')
    rects2 = ax.bar(x + width/2, dc_imp, width, label='Improvement over DC-LINC')
    
    # Add labels and title
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Hybrid Model Improvement by Operator Type')
    ax.set_xticks(x)
    ax.set_xticklabels(operators)
    ax.legend()
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Color bars based on whether improvement is positive or negative
    for rect in rects1:
        if rect.get_height() < 0:
            rect.set_color('lightcoral')
        else:
            rect.set_color('lightblue')
            
    for rect in rects2:
        if rect.get_height() < 0:
            rect.set_color('indianred')
        else:
            rect.set_color('lightskyblue')
    
    # Add value labels
    for i, rect in enumerate(rects1):
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -10),  # adjust vertical position based on bar direction
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8)
                    
    for i, rect in enumerate(rects2):
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -10),
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "operator_improvement.png"))
    
    # Plot 3: LINC uncertainty rates by operator
    fig, ax = plt.subplots(figsize=(12, 6))
    
    uncertain_rates = [operator_stats[op]["neuro_stats"]["uncertain_percent"] for op in operators]
    
    bars = ax.bar(operators, uncertain_rates, color='lightgray')
    
    # Add labels and title
    ax.set_ylabel('Uncertainty Rate (%)')
    ax.set_title('LINC Uncertainty Rate by Operator Type')
    ax.set_ylim(0, 100)
    
    # Add value labels
    for i, rect in enumerate(bars):
        height = rect.get_height()
        count = operator_stats[operators[i]]["neuro_stats"]["uncertain_count"]
        total = operator_stats[operators[i]]["total_examples"]
        ax.annotate(f'{height}% ({count}/{total})',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "operator_uncertainty.png"))
    
    return [
        os.path.join(output_dir, "operator_accuracy.png"),
        os.path.join(output_dir, "operator_improvement.png"),
        os.path.join(output_dir, "operator_uncertainty.png")
    ]


def generate_summary_report(results, operator_stats, output_file):
    """Generate a detailed summary report from the results."""
    # Start building the report
    lines = []
    lines.append("=" * 80)
    lines.append("HYBRID MODEL ANALYSIS: LINC + DC-LINC")
    lines.append("=" * 80)
    lines.append("")
    lines.append("This report analyzes a hybrid model that uses:")
    lines.append("- LINC (Neurosymbolic) predictions when they are True/False")
    lines.append("- DC-LINC predictions when LINC returns Uncertain")
    lines.append("")
    
    # Overall statistics
    lines.append("OVERALL RESULTS")
    lines.append("-" * 80)
    lines.append(f"Total examples analyzed: {results['total_examples']}")
    lines.append("")
    
    # Create a table for accuracy comparison
    accuracy_table = [
        ["Model", "Correct", "Accuracy"],
        ["LINC", f"{results['neuro_stats']['correct']}/{results['total_examples']}", f"{results['neuro_stats']['percent']}%"],
        ["DC-LINC", f"{results['dc_stats']['correct']}/{results['total_examples']}", f"{results['dc_stats']['percent']}%"],
        ["Hybrid", f"{results['hybrid_stats']['correct']}/{results['total_examples']}", f"{results['hybrid_stats']['percent']}%"],
    ]
    lines.append(format_table(accuracy_table))
    lines.append("")
    
    # Hybrid model improvement
    lines.append("Hybrid model improvement:")
    lines.append(f"- Over LINC: {results['improvement']['over_neuro']}%")
    lines.append(f"- Over DC-LINC: {results['improvement']['over_dc']}%")
    lines.append("")
    
    # LINC prediction distribution
    lines.append("LINC prediction distribution:")
    lines.append(f"- True/False predictions: {results['neuro_stats']['true_false_count']} ({results['neuro_stats']['true_false_percent']}%)")
    lines.append(f"- Uncertain predictions: {results['neuro_stats']['uncertain_count']} ({results['neuro_stats']['uncertain_percent']}%)")
    lines.append("")
    
    # Operator-specific analysis
    lines.append("OPERATOR-SPECIFIC RESULTS")
    lines.append("-" * 80)
    
    # Create a summary table of accuracies by operator
    op_table = [["Operator", "Sample Size", "LINC", "DC-LINC", "Hybrid", "Improvement"]]
    
    for op_name, stats in operator_stats.items():
        improvement = f"+{stats['improvement']['over_neuro']}%" if stats['improvement']['over_neuro'] >= 0 else f"{stats['improvement']['over_neuro']}%"
        op_table.append([
            op_name,
            stats["total_examples"],
            f"{stats['neuro_stats']['percent']}%",
            f"{stats['dc_stats']['percent']}%",
            f"{stats['hybrid_stats']['percent']}%",
            improvement
        ])
    
    lines.append(format_table(op_table))
    lines.append("")
    
    # Add detailed breakdown by operator
    lines.append("DETAILED BREAKDOWN BY OPERATOR")
    lines.append("-" * 80)
    
    for op_name, stats in operator_stats.items():
        lines.append(f"\n## {op_name} (n={stats['total_examples']})")
        
        # Accuracy comparison
        lines.append(f"- LINC accuracy: {stats['neuro_stats']['percent']}% ({stats['neuro_stats']['correct']}/{stats['total_examples']})")
        lines.append(f"- DC-LINC accuracy: {stats['dc_stats']['percent']}% ({stats['dc_stats']['correct']}/{stats['total_examples']})")
        lines.append(f"- Hybrid accuracy: {stats['hybrid_stats']['percent']}% ({stats['hybrid_stats']['correct']}/{stats['total_examples']})")
        
        # LINC prediction distribution
        lines.append(f"- LINC True/False rate: {stats['neuro_stats']['true_false_percent']}% ({stats['neuro_stats']['true_false_count']}/{stats['total_examples']})")
        lines.append(f"- LINC Uncertain rate: {stats['neuro_stats']['uncertain_percent']}% ({stats['neuro_stats']['uncertain_count']}/{stats['total_examples']})")
        
        # Hybrid improvement
        neuro_imp = stats['improvement']['over_neuro']
        dc_imp = stats['improvement']['over_dc']
        
        lines.append(f"- Hybrid improvement over LINC: {'+' if neuro_imp >= 0 else ''}{neuro_imp}%")
        lines.append(f"- Hybrid improvement over DC-LINC: {'+' if dc_imp >= 0 else ''}{dc_imp}%")
        
        # Recovery cases
        lines.append(f"- Cases where hybrid recovered from LINC errors: {stats['hybrid_stats']['neuro_wrong_hybrid_right']}")
        lines.append(f"- Cases where hybrid recovered from DC-LINC errors: {stats['hybrid_stats']['dc_wrong_hybrid_right']}")
        lines.append("")
    
    # Key observations
    lines.append("\nKEY OBSERVATIONS")
    lines.append("-" * 80)
    
    # Overall performance
    if results["improvement"]["over_neuro"] > 0 and results["improvement"]["over_dc"] > 0:
        lines.append("- The hybrid model outperforms both individual models.")
    elif results["improvement"]["over_neuro"] > 0:
        lines.append(f"- The hybrid model outperforms LINC by {results['improvement']['over_neuro']}% but " +
                     f"{'underperforms compared to' if results['improvement']['over_dc'] < 0 else 'matches'} DC-LINC.")
    elif results["improvement"]["over_dc"] > 0:
        lines.append(f"- The hybrid model outperforms DC-LINC by {results['improvement']['over_dc']}% but " +
                     f"{'underperforms compared to' if results['improvement']['over_neuro'] < 0 else 'matches'} LINC.")
    else:
        lines.append("- The hybrid model does not improve over either individual model.")
    
    # LINC uncertainty
    lines.append(f"- LINC produces uncertain predictions in {results['neuro_stats']['uncertain_percent']}% of cases, " +
                 f"which the hybrid model replaces with DC-LINC predictions.")
    
    # Best operators for hybrid approach
    op_improvements = [(op, stats["improvement"]["over_neuro"]) for op, stats in operator_stats.items()]
    op_improvements.sort(key=lambda x: x[1], reverse=True)
    
    if op_improvements[0][1] > 0:
        best_ops = [op for op, imp in op_improvements if imp > 0]
        if best_ops:
            best_ops_str = ", ".join(best_ops)
            lines.append(f"- The hybrid approach is most beneficial for the following operators: {best_ops_str}.")
    
    # Worst operators for hybrid approach
    op_improvements.sort(key=lambda x: x[1])
    
    if op_improvements[0][1] < 0:
        worst_ops = [op for op, imp in op_improvements if imp < 0]
        if worst_ops:
            worst_ops_str = ", ".join(worst_ops)
            lines.append(f"- The hybrid approach performs worse than LINC alone for: {worst_ops_str}.")
    
    # Write the report to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Summary report saved to: {output_file}")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Hybrid model analysis for LINC and DC-LINC.")
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
    
    # Compute hybrid model results
    print("\nComputing hybrid model results...")
    overall_stats, overall_df = compute_hybrid_results(
        neuro_gens, dc_gens, references, ALL_OPERATOR_INDICES
    )
    
    # Analyze by operator type
    print("\nAnalyzing results by operator type...")
    operator_stats, operator_dfs = analyze_by_operator(overall_df)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    overall_plots = plot_hybrid_results(overall_stats, os.path.join(args.output_dir, "figures"))
    operator_plots = plot_operator_comparison(operator_stats, os.path.join(args.output_dir, "figures"))
    
    # Generate summary report
    print("\nGenerating summary report...")
    report_file = os.path.join(args.output_dir, "summary_report.txt")
    report_lines = generate_summary_report(overall_stats, operator_stats, report_file)
    
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
    print("\n" + "\n".join(report_lines))
    
    print("\nAnalysis complete! Results saved to:")
    print(f"  - JSON: {args.json}")
    print(f"  - CSV: {args.csv}")
    print(f"  - Summary Report: {report_file}")
    print(f"  - Figures: {os.path.join(args.output_dir, 'figures')}")


if __name__ == "__main__":
    main()