# DC-LINC: Operator-Aware Divide-and-Conquer for Neuro-Symbolic Logical Reasoning

**MSc Computer Science dissertation — University of Bath, 2025.**  
Extends the [LINC](https://arxiv.org/abs/2310.15164) neuro-symbolic pipeline with operator-aware conclusion decomposition, achieving **55.6% accuracy on the FOLIO operator subset** — a **+13.9 pp gain** over the LINC baseline via a zero-parameter fallback ensemble.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-forthcoming-b31b1b.svg)](#citation)

---

## Key results

| System | FOLIO (all 182) | FOLIO (operator subset, n=72) |
|--------|:-:|:-:|
| LINC baseline (re-implemented) | 58.2% | 41.7% |
| DC-LINC (this work) | 56.0% | 40.3% |
| **Merged fallback** (this work) | — | **55.6% (+13.9 pp)** |
| Oracle upper bound | — | 59.7% |

The fallback is purely rule-based: keep LINC when it returns a decisive answer; use DC-LINC when LINC returns Uncertain. DC-LINC and LINC are complementary on **37.5%** of operator-bearing queries — the structural foundation of the gain.

<details>
<summary>Per-operator breakdown</summary>

| Operator | n | LINC | DC-LINC | Δ (pp) |
|----------|:-:|:----:|:-------:|:------:|
| NOR | 5 | 20.0% | 60.0% | **+40.0** |
| XOR | 5 | 20.0% | 40.0% | **+20.0** |
| OR  | 7 | 14.3% | 28.6% | +14.3 |
| NOT | 22 | 50.0% | 50.0% | 0.0 |
| IMPLIES | 18 | 44.4% | 38.9% | −5.6 |
| AND | 15 | 53.3% | 26.7% | **−26.7** |

NOR/XOR gains are large but sample-small (n=5, all True ground-truth labels) — interpret with caution.  
AND regression is the key failure mode: one Uncertain child drives the full conjunction to Uncertain, overriding LINC's decisive monolithic proofs.

</details>

---

## How it works

LINC translates natural-language premises + conclusion to FOL, then calls the Prover9 theorem prover. When the conclusion carries a top-level logical connective, DC-LINC inserts one decomposition layer before the prover:

```
NL conclusion ──► Detect operator (rule-based, 7 types + NONE)
                        │
              ┌─────────┴──────────┐
        operator found         NONE (atomic / quantified)
              │                    │
       Split into children      standard LINC path
       (LLM, 10-trial consensus)
              │
       Prove each child independently
       (LLM + Prover9, n=10, majority vote)
              │
       Merge via three-valued truth tables
       {True, False, Uncertain} → final label
```

**Operator detection is deterministic** — regex + token cues, no LLM involvement.  
**Child text extraction** uses the LLM with operator-specific few-shot prompts, stabilised by a 10-trial majority vote.  
**Quantifier safeguards** prevent splits that would cross binder scopes (e.g. ∀x[P(x) ∧ Q(x)] is treated as atomic).

---

## Quick start

### Option 1 — Docker (recommended, includes Prover9)

```bash
docker build -t dc-linc .
docker run --gpus all dc-linc \
  python runner.py \
    --model mistralai/Mistral-7B-v0.1 \
    --tasks folio-dcneurosymbolic-1shot \
    --n_samples 10
```

### Option 2 — Conda

```bash
conda create -n linc python=3.10
conda activate linc
pip install -r requirements.txt
```

Prover9/Mace4 must be compiled from source (LADR distribution) and placed on `$PATH`.  
See the Dockerfile for the exact build steps used in this study.

```bash
python runner.py \
  --model mistralai/Mistral-7B-v0.1 \
  --tasks folio-dcneurosymbolic-1shot \
  --n_samples 10
```

### Run the LINC baseline

```bash
python runner.py \
  --model mistralai/Mistral-7B-v0.1 \
  --tasks folio-neurosymbolic-1shot \
  --n_samples 10
```

### Reproduce the hybrid / per-operator analysis

```bash
cd dc_analysis/analysis4
python hybrid_analysis.py
```

Outputs: per-operator accuracy tables, complementarity breakdown, confusion matrices, and summary report.

**Hardware:** Evaluated on NVIDIA RTX 6000 Ada (62 GB VRAM, HexCloud HPC). Full FOLIO run ~12 hours. Mistral-7B-v0.1 is loaded in fp32 — a GPU with ≥24 GB VRAM is required.

---

## Repository layout

```
.
├── eval/                             # Baseline LINC re-implementation (Python 3.10)
├── dc_eval/
│   ├── base.py                       # Task class — adds dcneurosymbolic mode
│   ├── utils.py                      # Main D&C evaluation loop (complete_code_custom)
│   ├── dc_helpers/
│   │   ├── three_valued_logic.py     # Merge rules for 7 operators
│   │   ├── splitting.py              # Adapter → run_pipeline_sentence
│   │   └── parsing.py                # FOL extraction + Prover9 invocation
│   └── dc_llm_splitter/
│       └── dnc_llm_splitter/
│           ├── models/detectors.py   # Rule-based operator detection (11 detectors)
│           ├── run_pipeline_enhanced.py
│           └── steps/split_*.py      # Per-operator LLM splitters (AND/OR/XOR/NOR/NOT/IMPLIES/IFF)
├── dc_analysis/
│   ├── analysis4/                    # Definitive results — tables and figures used in the paper
│   ├── dcneuro1_analysis/            # DC-LINC full-dataset accuracy scripts
│   └── neuro1_analysis/              # LINC full-dataset accuracy scripts
├── analysis/                         # Post-hoc LINC baseline analysis
├── outputs/                          # Raw generations, processed results, references (JSON)
├── runner.py                         # Entry point
├── requirements.txt
└── CITATION.cff
```

---

## Tech stack

| Component | Detail |
|-----------|--------|
| Language | Python 3.10 |
| LLM | Mistral-7B-v0.1 (fp32, temp=0.8, top_p=0.95, top_k=0) |
| Theorem prover | Prover9 (refutation, 10 s timeout) |
| Deep learning | PyTorch 2.6, HuggingFace Transformers 4.54.1, Accelerate 1.9.0 |
| Dataset | FOLIO (Han et al., 2024) — 182 items post-filter, 72 operator-bearing |
| Reproducibility | Docker multi-stage build, pinned deps (`requirements.txt`), fixed seeds, full intermediate output logging |

---

## Citation

```bibtex
@article{rokni2025dclinc,
  title   = {{DC-LINC}: Operator-Aware Divide-and-Conquer for Neuro-Symbolic Logical Reasoning},
  author  = {Rokni, Nufel},
  year    = {2025},
  note    = {arXiv preprint — DOI forthcoming}
}
```

---

*MSc Computer Science, University of Bath, 2024–2025. Supervisor: Dr Mingzhi Dong.*
