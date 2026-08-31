# Decentralized Multi-Agent System for Cancer Drug Sensitivity Prediction

A multi-agent debate framework that predicts cancer cell line drug sensitivity by combining evidence from four specialized AI agents — genomics, transcriptomics, pharmacology, and pathway analysis — resolved through a structured axiom-based debate protocol.

This code accompanies the manuscript:
> **A Decentralized Multi-Agent Debate Framework for Interpretable Cancer Drug Sensitivity Prediction**  
> Shrinithi Natarajan, Bhaskarjyoti Das — Johns Hopkins University

---

## What This Does

Given a cancer cell line and a drug, the system:
1. Dispatches four agents concurrently to query curated biological databases (CCLE/DepMap, GDSC2, CTRPv2, CIViC, KEGG, RPPA)
2. Each agent produces a sensitivity verdict (SENSITIVE / RESISTANT) with supporting evidence
3. A structured debate resolves conflicts using 14 axiom rules (genomic biomarkers, pathway context, pharmacological signal)
4. Outputs a final verdict with a full reasoning trace

The full 100-case gold-standard benchmark achieves AUROC 0.917, Cohen's κ = 0.740.

---

## Prerequisites

### Python

Requires **Python ≥ 3.11**. Install dependencies with [uv](https://github.com/astral-sh/uv) (recommended) or pip:

```bash
# with uv (faster)
uv sync

# or with pip
pip install -e .

# dev dependencies (for running tests)
pip install -e ".[dev]"
```

### API Keys

Copy `.env.example` to `.env` and fill in the keys you need:

```bash
cp .env.example .env
```

| Variable | Where to get it | Required for |
|---|---|---|
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com) | Default model (Gemini) |
| `GROQ_API_KEY` | [Groq Console](https://console.groq.com) | Groq-hosted models |
| `OPENROUTER_API_KEY` | [OpenRouter](https://openrouter.ai) | OpenRouter models |
| `ONCOKB_TOKEN` | [OncoKB](https://oncokb.org/apiAccess) | Enhanced genomic annotations (optional) |

The system supports Gemini (via Google AI Studio or Vertex AI), Groq, OpenRouter, NVIDIA NIM, and DeepSeek. You only need the key for the provider you use.

**Vertex AI** (used in paper experiments): set `VERTEX_PROJECT` to your GCP project ID and ensure `gcloud auth application-default login` has been run.

### Database Paths (optional)

Defaults point to `src/data/processed/`. Override in `.env` if you store DBs elsewhere:

```
GENOMICS_DB=src/data/processed/genomics.db
TRANSCRIPTOMICS_DB=src/data/processed/transcriptomics.db
PHARMACOLOGY_DB=src/data/processed/pharmacology.db
PATHWAYS_DB=src/data/processed/pathways.db
```

---

## Building the Databases

The processed `.db` files are not included in the repo. Build them from raw public data sources:

### 1. Download raw data

| Source | Download from | Place at |
|---|---|---|
| CCLE model info | [DepMap portal](https://depmap.org/portal/download/all/) → `model.csv` | `src/data/raw/ccle/model.csv` |
| CCLE mutations | DepMap → `OmicsSomaticMutations.csv` | `src/data/raw/ccle/OmicsSomaticMutations.csv` |
| CCLE copy number | DepMap → `OmicsCNGene.csv` | `src/data/raw/ccle/OmicsCNGene.csv` |
| CCLE expression | DepMap → `OmicsExpressionProteinCodingGenesTPMLogp1.csv` | `src/data/raw/ccle/OmicsExpressionProteinCodingGenesTPMLogp1.csv` |
| CCLE RPPA | DepMap → `CCLE_RPPA_20181003.csv` | `src/data/raw/ccle/CCLE_RPPA_20181003.csv` |
| GDSC2 IC50 | [GDSC](https://www.cancerrxgene.org/downloads/bulk_download) → `GDSC2_fitted_dose_response.csv` | `src/data/raw/gdsc2/ic50.csv` |
| CTRPv2 | [DepMap portal](https://depmap.org/portal/download/all/) → search "CTRP" | `src/data/raw/ctrp/` |
| HGNC gene names | [HGNC](https://www.genenames.org/download/archive/) → `hgnc_complete_set.txt` | `src/data/raw/hgnc/hgnc_complete_set.txt` |

### 2. Run ETL scripts

```bash
# Build genomics, transcriptomics, pharmacology DBs (~20-40 min)
PYTHONPATH=. python src/data/etl_full_rebuild.py

# Build CIViC clinical evidence DB (auto-downloads nightly TSV)
PYTHONPATH=. python src/data/etl_civic.py

# Build KEGG pathway DB (~20-40 min, auto-downloads KGML files)
PYTHONPATH=. python src/data/etl_kegg.py

# Build CTRPv2 pharmacology DB
PYTHONPATH=. python src/data/etl_ctrp.py
```

---

## Running the System

### Validate on the gold-standard benchmark (100 cases)

```bash
PYTHONPATH=. python experiments/run_mini_validation.py
```

Results are written to `experiments/results/traces_gold_standard.jsonl`. Metrics print to stdout.

### Run the ablation study (14 conditions)

```bash
PYTHONPATH=. python experiments/run_mini_ablation.py
```

### Run the random forest baseline

```bash
PYTHONPATH=. python experiments/run_rf_baseline.py
```

### Evaluate faithfulness

```bash
PYTHONPATH=. python experiments/eval_faithfulness.py
```

### Reproduce paper figures

```bash
PYTHONPATH=. python experiments/generate_figures.py
# Output: experiments/results/paper_figures.html
```

---

## Repository Structure

```
src/
  agents/          # Four specialized agents (genomics, transcriptomics,
  |                #   pharmacology, pathway) + monolithic LLM baseline
  mcp_servers/     # MCP tool servers backing each agent
  protocols/       # DebateEngine and AxiomResolver (14 resolution rules)
  llm/             # LLM client with provider routing, caching, rate limiting
  data/            # ETL scripts and database loaders
  schemas/         # Pydantic schemas (EvidencePack, AxiomRules, etc.)
  orchestrator.py  # Top-level case runner
  run_logger.py    # Per-run structured logging

experiments/
  run_mini_validation.py   # Gold-standard 100-case evaluation
  run_mini_ablation.py     # 14-condition ablation
  run_rf_baseline.py       # Random forest baseline
  eval_faithfulness.py     # LLM-as-judge faithfulness scorer
  generate_figures.py      # Paper figures
  results/                 # Pre-computed results (traces, metrics, figures)

data/
  cases/
    cases_gold_standard.yaml  # 100-case literature-validated benchmark

tests/                     # pytest test suite
trials/                    # Early-stage experiments (3 trial designs)
```

---

## Running Tests

```bash
pytest
```

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this code, please cite:

```bibtex
@article{natarajan2026dmas,
  title   = {A Decentralized Multi-Agent Debate Framework for Interpretable
             Cancer Drug Sensitivity Prediction},
  author  = {Natarajan, Shrinithi and Das, Bhaskarjyoti},
  journal = {PLOS Computational Biology},
  year    = {2026},
  note    = {Under review}
}
```

Code archived at: `[INSERT ZENODO DOI]`
