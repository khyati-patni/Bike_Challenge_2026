# BiKE v2 — Biomedical Information Knowledge Extraction, v2 by Team Graphminds
Link for full repo : https://drive.google.com/drive/folders/1YQ1FWoT9Qpf0Ml7-5Snv6kz_QI2DqmTz?usp=drive_link

---

## What's New in This Version

- **4 additional graph embedding methods**: Node2Vec, Metapath2Vec, EPHEN, GraphSAGE
  (on top of the existing R-GCN, RotatE, TransE)
- **Full Hits@K reporting**: all k values (1, 5, 10, 20, 50) reported per model,
  matching the NatUKE benchmark table format
- Ensemble now fuses **all 8 graph methods + cosine** for maximum coverage

## Architecture

```
doi_texts.json
     │
     ▼ Stage 1
  SciBERT ──── doi_embeddings.npy      ─────┐
  PubMedBERT ─ doi_embeddings.npy           │
               label_embeddings.npy         │
                                            ▼ Stage 2
                                    graph.json (KG)
                                            │
       ┌──────────┬──────────┬─────────────┼────────────┬─────────────┬──────────┐
       ▼          ▼          ▼             ▼            ▼             ▼          ▼
   Stage 3a   Stage 3b   Stage 3c    Stage 3d     Stage 3e      Stage 3f   Stage 3g
    R-GCN     RotatE     TransE      Node2Vec   Metapath2Vec    EPHEN    GraphSAGE
       │          │          │             │            │             │          │
       └──────────┴──────────┴─────────────┴────────────┴─────────────┴──────────┘
                                           │
                                           ▼ Stage 4
                                 Evaluate: ALL Hits@K, MRR
                                (+ frequency calibration
                                 + ensemble scoring)
                                           │
                        ┌──────────────────┘
                        ▼ Stage 5
                 10-fold CV loop
              (all 9 models, both encoders)
              → mean ± std per metric
                        │
                        ▼ Stage 6
              Wilcoxon signed-rank test (internal)
              Sign test vs NatUKE reference
              → significance_report.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

unzip scripts.zip

# Full run: both encoders, 10 folds, all 9 models
python run_pipeline.py

# Quick test: SciBERT only, 5 folds, subset of models
python run_pipeline.py --encoder scibert --folds 5 --models rotate ensemble

# Run only graph embedding models (no BERT encoders needed)
python run_pipeline.py --skip-encode --models node2vec metapath2vec ephen graphsage ensemble

# If embeddings already exist (skip CrossRef fetch + encoding):
python run_pipeline.py --skip-encode

# Resume from CV (graph already built):
python run_pipeline.py --skip-encode --skip-graph

# Stats only (CV results must already exist):
python run_pipeline.py --eval-only
```

---

## Script Map

| Script | Role |
|---|---|
| `stage1_encode_embeddings.py` | SciBERT + PubMedBERT → `.npy` embedding files |
| `stage2_build_graph.py` | Multi-relational KG → `graph.json` |
| `stage3a_rgcn.py` | R-GCN training (importable + standalone) |
| `stage3b_rotate.py` | RotatE training (importable + standalone) |
| `stage3c_transe.py` | TransE training (importable + standalone) |
| `stage3d_node2vec.py` | **Node2Vec** biased random walk embeddings |
| `stage3e_metapath2vec.py` | **Metapath2Vec** heterogeneous graph walks |
| `stage3f_ephen.py` | **EPHEN** spectral heat-kernel + text fusion |
| `stage3g_graphsage.py` | **GraphSAGE** inductive neighbourhood aggregation |
| `stage4_evaluate.py` | Scoring functions, calibration, **all** Hits@K / MRR 

---


## Outputs

```
results/
  scibert/
    cv_raw.json          ← per-fold scores for every model × task
    cv_summary.json      ← mean ± std over all folds
    fold_1/ … fold_10/   ← per-fold saved embeddings
      rgcn_node_emb.npy, rgcn_rel_weights.npy
      rotate_ent_emb.npy, rotate_rel_emb.npy
      transe_ent_emb.npy, transe_rel_emb.npy
      node2vec_node_emb.npy, node2vec_rel_bias.npy
      metapath2vec_node_emb.npy, metapath2vec_rel_bias.npy
      ephen_node_emb.npy, ephen_rel_bias.npy
      graphsage_node_emb.npy, graphsage_rel_weights.npy
  pubmedbert/
    cv_raw.json
    cv_summary.json
    fold_1/ … fold_10/
  combined_cv_summary.json
  significance_report.txt
  significance_report.json
```

## NatUKE Reference Numbers (Table II, EPHEN / Metapath2Vec, 3rd fold)

| Task | Metric | NatUKE Best | Method |
|---|---|---|---|
| bioActivity | Hits@5 | 0.60 | EPHEN |
| collectionType | Hits@1 | 0.75 | EPHEN |
| collectionSite | Hits@20 | 0.55 | EPHEN |
| collectionSpecie | Hits@50 | 0.44 | Metapath2Vec |
| name | Hits@50 | 0.20 | Metapath2Vec |



---

## Requirements

```
torch>=2.0
transformers>=4.35
numpy
pandas
scipy
scikit-learn
gensim>=4.3          # Node2Vec + Metapath2Vec Word2Vec training
```
---

## Citation

If you use BiKE v2, please also cite the original NatUKE benchmark:

```
@inproceedings{doCarmo2022natuke,
  title={NatUKE: A Benchmark for Natural Product Knowledge Extraction from Academic Literature},
  author={do Carmo, Paulo Viviurka and Marx, Edgard and Marcacini, Ricardo and
          Valli, Marilia and Silva e Silva, Joao Victor and Pilon, Alan},
  booktitle={ICSC},
  year={2022}
}
```

---

## License

Apache 2.0 — same as NatUKE.
