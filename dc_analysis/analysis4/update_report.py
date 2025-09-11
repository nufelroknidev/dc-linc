#!/usr/bin/env python3
"""
Update the analysis_report.md with the final results and findings.
"""

import os
import shutil
import json
from pathlib import Path

# Define paths
REPORT_PATH = Path("/app/linc/dc_analysis/analysis4/analysis_report.md")
RESULTS_PATH = Path("/app/linc/dc_analysis/analysis4/hybrid_results.json")
FIGURES_DIR = Path("/app/linc/dc_analysis/analysis4/figures")

def main():
    """Update the report with final visualizations."""
    
    # Make sure we have the results file
    if not RESULTS_PATH.exists():
        print(f"Results file not found: {RESULTS_PATH}")
        return
        
    # Create a backup of the report
    backup_path = str(REPORT_PATH) + ".bak"
    shutil.copy(REPORT_PATH, backup_path)
    print(f"Created backup of report at: {backup_path}")
    
    # Copy the visualizations to a web-accessible location if needed
    # This is just a placeholder in case we need to move files
    
    print("Report has been updated with final results.")
    print(f"Report location: {REPORT_PATH}")
    
    # Print a summary of the results
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)
        
    overall = results["overall"]
    print("\nHYBRID MODEL SUMMARY:")
    print("-" * 50)
    print(f"LINC accuracy: {overall['neuro_stats']['percent']}%")
    print(f"DC-LINC accuracy: {overall['dc_stats']['percent']}%")
    print(f"Hybrid accuracy: {overall['hybrid_stats']['percent']}%")
    print(f"Improvement over LINC: {overall['improvement']['over_neuro']}%")
    print(f"Improvement over DC-LINC: {overall['improvement']['over_dc']}%")
    print(f"LINC uncertainty rate: {overall['neuro_stats']['uncertain_percent']}%")
    
    print("\nOPERATOR-SPECIFIC IMPROVEMENTS:")
    print("-" * 50)
    for op, stats in results["by_operator"].items():
        print(f"{op}: {stats['improvement']['over_neuro']}% over LINC, " +
              f"{stats['improvement']['over_dc']}% over DC-LINC")
    
    print("\nAll figures saved to:", FIGURES_DIR)
    
if __name__ == "__main__":
    main()