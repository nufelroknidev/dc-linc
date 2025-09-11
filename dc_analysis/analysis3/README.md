# Comparative Operator Analysis

This directory contains scripts and results for comparative analysis between regular LINC (neurosymbolic) and D&C LINC (dcneurosymbolic) models on logical operator examples.

## Key Files

- `compare_operators.py` - Main script for comparative analysis
- `comparative_results.json` - JSON file with aggregated results 
- `comparative_results.csv` - CSV file with per-example results
- `figures/` - Directory containing generated visualizations

## Analysis Focus

The analysis compares performance on examples containing logical operators (OR, XOR, NOR, IMPLIES, NOT, AND), categorizing examples into:

- Both models correct (both success)
- Only LINC correct (neuro only)
- Only DC-LINC correct (dc only)
- Both models incorrect (both fail)

## Running the Analysis

To run the comparison with default settings:

```
python3 compare_operators.py
```

To specify custom input and output paths:

```
python3 compare_operators.py \
  --neuro PATH_TO_NEURO_GENERATIONS \
  --dc PATH_TO_DC_GENERATIONS \
  --refs PATH_TO_REFERENCES \
  --output-dir OUTPUT_DIRECTORY
```

## Output

The script generates:

1. Overall comparative statistics
2. Per-operator breakdown of performance
3. Visualizations showing the comparison
4. CSV files with detailed results

## Operator Indices

The analysis uses predefined operator indices from the analysis2 directory to filter examples.