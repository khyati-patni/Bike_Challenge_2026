"""
run_pipeline.py — BiKE v2 master runner
─────────────────────────────────────────
MODEL ARCHITECTURE:
  Cosine baselines   — SciBERT cosine, PubMedBERT cosine  (no training)
  Structure-only KG  — Node2Vec, Metapath2Vec, EPHEN       (no BERT features)
  BERT-init KG       — R-GCN, RotatE, TransE, ComplEx, GraphSAGE
                       (initialised from SciBERT, configurable via GRAPH_INIT_ENCODER)
  Ensemble           — fuses BERT-init KG models + best cosine

EVALUATION: NatUKE 4-stage (20/40/60/80% train splits)
METRICS:    Hits@1, @5, @20, @50 and MRR

Usage:
  python run_pipeline.py                    # full run
  python run_pipeline.py --skip-encode      # embeddings already computed
  python run_pipeline.py --models cosine_scibert cosine_pubmedbert node2vec
"""
import argparse, json, pathlib, subprocess, sys, time

ROOT    = pathlib.Path(__file__).parent
SCRIPTS = ROOT / "scripts"
RES     = ROOT / "results"


def run(script, label, extra_args=None):
    print(f"\n{'━'*68}\n  {label}\n{'━'*68}")
    cmd = [sys.executable, str(SCRIPTS / script)] + (extra_args or [])
    t0  = time.time()
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"\n[ERROR] {script} failed"); sys.exit(res.returncode)
    print(f"  ✓ {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["cosine_scibert", "cosine_pubmedbert",
                             "node2vec", "metapath2vec", "ephen",
                             "rgcn", "rotate", "transe", "complex", "graphsage",
                             "ensemble"])
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--skip-encode", action="store_true")
    ap.add_argument("--skip-graph",  action="store_true")
    ap.add_argument("--no-stats",    action="store_true")
    args = ap.parse_args()

    if not args.skip_encode:
        run("stage1_encode_embeddings.py",
            "Stage 1 · encode DOIs + labels with SciBERT and PubMedBERT")
    if not args.skip_graph:
        run("stage2_build_graph.py", "Stage 2 · build multi-relational KG")

    run("stage5_natuke_stages.py",
        "Stage 5 · NatUKE 4-stage evaluation (all models)",
        extra_args=["--seed", str(args.seed), "--models"] + args.models)

    if not args.no_stats:
        run("stage6_significance_test.py",
            "Stage 6 · significance tests vs NatUKE baselines")

    print(f"\n{'━'*68}\n  PIPELINE COMPLETE\n{'━'*68}")
    summary_path = RES / "stage_summary.json"
    if summary_path.exists():
        print(f"  Results: {summary_path}")
        summary = json.loads(summary_path.read_text())
        sys.path.insert(0, str(SCRIPTS))
        from config import TASKS, STAGE_NAMES
        print(f"\n  Primary-k scores (3rd stage):")
        for model, task_dict in summary.items():
            scores = []
            for task in TASKS:
                if task not in task_dict: continue
                v = task_dict[task].get("3rd", {}).get("primary_k_val", 0.0)
                scores.append(f"{TASKS[task]['short']}={v:.2f}")
            print(f"    {model:<20}  {' '.join(scores)}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
