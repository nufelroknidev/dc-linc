#!/usr/bin/env python3
"""
Script to generate a summary report of the comparative analysis between
LINC and DC-LINC on logical operator examples.
"""

import json
import os
import argparse
from pathlib import Path

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

# Default paths
DEFAULT_RESULTS = Path("/app/linc/dc_analysis/analysis3/comparative_results.json")
DEFAULT_OUTPUT = Path("/app/linc/dc_analysis/analysis3/summary_report.txt")

def load_json(path):
    """Load JSON data from a file."""
    with open(path, "r") as f:
        return json.load(f)

def generate_summary(results, output_file):
    """Generate a detailed summary report from the results."""
    overall = results["overall"]
    by_operator = results["by_operator"]
    
    # Start building the report
    lines = []
    lines.append("=" * 80)
    lines.append("COMPARATIVE ANALYSIS: LINC vs DC-LINC ON LOGICAL OPERATORS")
    lines.append("=" * 80)
    lines.append("")
    
    # Overall statistics
    lines.append("OVERALL RESULTS")
    lines.append("-" * 80)
    lines.append(f"Total examples analyzed: {overall['total_examples']}")
    lines.append("")
    
    # Create a table for the overall success categories
    overall_table = [
        ["Category", "Count", "Percentage"],
        ["Both models correct", overall["both_success"]["count"], f"{overall['both_success']['percent']}%"],
        ["Only LINC correct", overall["neuro_only"]["count"], f"{overall['neuro_only']['percent']}%"],
        ["Only DC-LINC correct", overall["dc_only"]["count"], f"{overall['dc_only']['percent']}%"],
        ["Both models incorrect", overall["both_fail"]["count"], f"{overall['both_fail']['percent']}%"],
    ]
    lines.append(format_table(overall_table))
    lines.append("")
    
    # Overall accuracy
    accuracy_table = [
        ["Model", "Correct", "Accuracy"],
        ["LINC (Neurosymbolic)", f"{overall['neuro_overall']['count']}/{overall['total_examples']}", f"{overall['neuro_overall']['percent']}%"],
        ["DC-LINC", f"{overall['dc_overall']['count']}/{overall['total_examples']}", f"{overall['dc_overall']['percent']}%"],
    ]
    lines.append(format_table(accuracy_table))
    lines.append("")
    lines.append("")
    
    # Operator-specific analysis
    lines.append("OPERATOR-SPECIFIC RESULTS")
    lines.append("-" * 80)
    
    # Create a summary table of accuracies by operator
    op_accuracy_table = [
        ["Operator", "Sample Size", "LINC Accuracy", "DC-LINC Accuracy", "Diff (DC-LINC - LINC)"]
    ]
    
    for op, stats in by_operator.items():
        diff = stats["dc_overall"]["percent"] - stats["neuro_overall"]["percent"]
        diff_str = f"+{diff:.2f}%" if diff > 0 else f"{diff:.2f}%"
        op_accuracy_table.append([
            op,
            stats["total_examples"],
            f"{stats['neuro_overall']['percent']}%",
            f"{stats['dc_overall']['percent']}%",
            diff_str
        ])
    
    lines.append(format_table(op_accuracy_table))
    lines.append("")
    
    # Add detailed breakdown by operator
    lines.append("DETAILED BREAKDOWN BY OPERATOR")
    lines.append("-" * 80)
    
    for op, stats in by_operator.items():
        lines.append(f"\n## {op} (n={stats['total_examples']})")
        
        # Success categories for this operator
        op_table = [
            ["Category", "Count", "Percentage"],
            ["Both correct", stats["both_success"]["count"], f"{stats['both_success']['percent']}%"],
            ["Only LINC correct", stats["neuro_only"]["count"], f"{stats['neuro_only']['percent']}%"],
            ["Only DC-LINC correct", stats["dc_only"]["count"], f"{stats['dc_only']['percent']}%"],
            ["Both incorrect", stats["both_fail"]["count"], f"{stats['both_fail']['percent']}%"],
        ]
        lines.append(format_table(op_table))
        lines.append("")
    
    # Key observations
    lines.append("\nKEY OBSERVATIONS")
    lines.append("-" * 80)
    
    # Overall performance comparison
    if overall["neuro_overall"]["percent"] > overall["dc_overall"]["percent"]:
        lines.append(f"- Overall, LINC slightly outperforms DC-LINC by " +
                     f"{overall['neuro_overall']['percent'] - overall['dc_overall']['percent']:.2f}% " +
                     f"({overall['neuro_overall']['percent']}% vs {overall['dc_overall']['percent']}%).")
    elif overall["dc_overall"]["percent"] > overall["neuro_overall"]["percent"]:
        lines.append(f"- Overall, DC-LINC slightly outperforms LINC by " +
                     f"{overall['dc_overall']['percent'] - overall['neuro_overall']['percent']:.2f}% " +
                     f"({overall['dc_overall']['percent']}% vs {overall['neuro_overall']['percent']}%).")
    else:
        lines.append("- Overall, LINC and DC-LINC perform similarly in accuracy.")
    
    # Operator strengths
    lines.append("- Operator-specific performance:")
    
    for op, stats in by_operator.items():
        diff = stats["dc_overall"]["percent"] - stats["neuro_overall"]["percent"]
        
        if abs(diff) < 5:
            lines.append(f"  * {op}: Both models perform similarly.")
        elif diff > 0:
            lines.append(f"  * {op}: DC-LINC performs better by {diff:.2f}% " +
                         f"({stats['dc_overall']['percent']}% vs {stats['neuro_overall']['percent']}%).")
        else:
            lines.append(f"  * {op}: LINC performs better by {-diff:.2f}% " +
                         f"({stats['neuro_overall']['percent']}% vs {stats['dc_overall']['percent']}%).")
    
    # Overall success pattern
    both_success = overall["both_success"]["percent"]
    both_fail = overall["both_fail"]["percent"]
    complement = overall["neuro_only"]["percent"] + overall["dc_only"]["percent"]
    
    lines.append(f"- The models agree on {both_success + both_fail:.2f}% of examples " +
                 f"(both correct: {both_success:.2f}%, both incorrect: {both_fail:.2f}%).")
    lines.append(f"- The models complement each other on {complement:.2f}% of examples, " +
                 f"suggesting potential benefits from an ensemble approach.")
    
    # Highest differences
    op_diffs = [(op, stats["dc_overall"]["percent"] - stats["neuro_overall"]["percent"]) 
                for op, stats in by_operator.items()]
    op_diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    if op_diffs:
        max_diff_op, max_diff = op_diffs[0]
        if abs(max_diff) > 10:
            if max_diff > 0:
                lines.append(f"- The largest performance gap is on {max_diff_op} operator, " +
                             f"where DC-LINC outperforms LINC by {max_diff:.2f}%.")
            else:
                lines.append(f"- The largest performance gap is on {max_diff_op} operator, " +
                             f"where LINC outperforms DC-LINC by {-max_diff:.2f}%.")
    
    # Write the report to file
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Summary report saved to: {output_file}")
    
    # Also print the report to console
    print("\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Generate a summary report for the comparative analysis.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Path to the results JSON file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to save the summary report")
    args = parser.parse_args()
    
    # Load results
    results = load_json(args.results)
    
    # Generate summary
    generate_summary(results, args.output)


if __name__ == "__main__":
    main()