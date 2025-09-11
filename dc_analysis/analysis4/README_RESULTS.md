# LINC/DC-LINC Hybrid Model Analysis

This project analyzes the performance of a hybrid model that combines LINC (neurosymbolic) and DC-LINC (divide-and-conquer neurosymbolic) approaches using a simple fallback strategy.

## Hybrid Strategy

The hybrid model uses:
- LINC predictions when they are definitive (True or False)
- DC-LINC predictions when LINC returns "Uncertain"

## Key Findings

1. **The hybrid model significantly outperforms both individual models**:
   - Hybrid accuracy: 55.56% (40/72)
   - LINC accuracy: 41.67% (30/72)
   - DC-LINC accuracy: 40.28% (29/72)

2. **Improvement margins**:
   - 13.89% improvement over LINC
   - 15.28% improvement over DC-LINC

3. **LINC uncertainty is common**:
   - LINC gives uncertain predictions in 59.72% of cases
   - This high uncertainty rate provides many opportunities for the hybrid model to leverage DC-LINC

4. **Operator-specific performance**:
   - Greatest improvements on NOR (+40.0%), XOR (+20.0%), and IMPLIES (+16.67%)
   - Modest improvements on OR (+14.29%) and NOT (+13.64%)
   - No change on AND (0.0%)

5. **Complementary strengths**:
   - LINC performs better on AND operators (53.33% vs 26.67%)
   - DC-LINC performs better on NOR operators (60.0% vs 20.0%) and XOR operators (40.0% vs 20.0%)
   - The hybrid model successfully preserves the strengths of each

## Structure

```
dc_analysis/
├── analysis4/                        # Hybrid model analysis
│   ├── hybrid_analysis.py            # Main analysis script
│   ├── update_report.py              # Script to update report with final results
│   ├── hybrid_results.json           # Full results in JSON format
│   ├── hybrid_results.csv            # Per-example results in CSV format
│   ├── summary_report.txt            # Detailed text report
│   ├── analysis_report.md            # Markdown report with visualizations
│   ├── README.md                     # Documentation
│   ├── figures/                      # Generated visualizations
│   └── *_results.csv                 # Operator-specific result files
```

## Running the Analysis

```bash
# Run the full analysis
cd /app/linc
python3 dc_analysis/analysis4/hybrid_analysis.py

# Update the report with latest results
python3 dc_analysis/analysis4/update_report.py
```

## Conclusion

The hybrid model demonstrates that a simple fallback strategy can effectively combine the strengths of both approaches. When the neurosymbolic model fails to prove a result (indicated by uncertainty), the divide-and-conquer approach often succeeds by breaking the problem down differently. This approach could be further refined by using different combinations based on the specific logical operators present in a given problem.