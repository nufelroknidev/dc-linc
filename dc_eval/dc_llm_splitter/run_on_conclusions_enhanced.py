from __future__ import annotations
import argparse, json
from pathlib import Path
from dnc_llm_splitter.io_utils import read_conclusions
from dnc_llm_splitter.run_pipeline_enhanced import make_llm, run_pipeline_sentence
from dnc_llm_splitter.config import load_settings

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", default="data/conclusions.txt")
    ap.add_argument("--out", default="results_enhanced.jsonl")
    ap.add_argument("--strategy", choices=["auto","llm","rule"], default="rule",
                    help="operator detection: auto (LLM→rule fallback), llm (LLM only), rule (deterministic only)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    cfg_path = (here / args.config).resolve()
    in_path = (here / args.input).resolve()
    out_path = (here / args.out).resolve()

    cfg = load_settings(str(cfg_path))
    if args.strategy:
        cfg.strategy = args.strategy
    llm = make_llm(cfg)

    sents = read_conclusions(str(in_path))

    strategy = (args.strategy or getattr(cfg, "strategy", None) or "auto")
    with open(out_path, "w", encoding="utf-8") as f:
        for _, s in enumerate(sents, start=1):
            print(f"sentence: {s}")
            for _ in range(5):
                r = run_pipeline_sentence(llm, sentence=s, strategy=strategy)
                op = r.get('operator') or r.get('detector', {}).get('operator')
                split_block = r.get('split', {}) or {}
                # unify legacy shapes
                if 'children' not in split_block:
                    # map possible legacy keys
                    for k in ('conjuncts','disjuncts','clauses','options'):
                        if k in split_block and isinstance(split_block.get(k), list):
                            split_block['children'] = split_block[k]
                            break
                children = split_block.get('children', [])
                
                r=f"operator={op}, children={children}"
                print(f"result: {r}\n")
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print()

if __name__ == "__main__":
    main()
