# Hybrid Model Analysis: LINC + DC-LINC

This directory contains scripts and results for analyzing a hybrid approach that combines LINC (neurosymbolic) and DC-LINC models using a simple strategy:

- Use LINC predictions when they are definitive (True or False)
- Fall back to DC-LINC predictions when LINC returns "Uncertain"

The intuition behind this approach is that when the neurosymbolic model cannot prove a result, we let the DC model try.

## Key Files

- `hybrid_analysis.py` - Main script for hybrid model analysis
- `hybrid_results.json` - JSON file with aggregated results 
- `hybrid_results.csv` - CSV file with per-example results
- `figures/` - Directory containing generated visualizations
- `summary_report.txt` - Detailed textual report of findings

## Analysis Focus

The analysis focuses on examples containing logical operators (OR, XOR, NOR, IMPLIES, NOT, AND), analyzing:

1. Overall accuracy of the hybrid approach compared to individual models
2. Performance breakdown by operator type
3. Analysis of when the hybrid approach helps or hurts
4. LINC uncertainty rates and how they affect hybrid performance

## Running the Analysis

To run the analysis with default settings:

```
python3 hybrid_analysis.py
```

To specify custom input and output paths:

```
python3 hybrid_analysis.py \
  --neuro PATH_TO_NEURO_GENERATIONS \
  --dc PATH_TO_DC_GENERATIONS \
  --refs PATH_TO_REFERENCES \
  --output-dir OUTPUT_DIRECTORY
```

## Output

The script generates:

1. Overall performance statistics for all three approaches
2. Per-operator breakdown of performance
3. Visualizations comparing the models
4. CSV files with detailed per-example results
5. A comprehensive summary report