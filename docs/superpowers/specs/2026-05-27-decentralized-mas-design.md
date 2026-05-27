# Decentralized Multi-Agent Consensus Framework for Cancer Drug Resistance Prediction
## Design Specification — 2026-05-27

---

## 1. Problem Statement

Monolithic LLMs fail at cancer drug resistance (CDR) prediction because integrating genomics, transcriptomics, pharmacology, and pathway data in a single prompt causes context overwhelm, hallucination, and silent agreement between reasoning steps. This framework replaces that with four specialized agents — each owning one biological modality — that produce structured evidence, debate conflicts using a formalized axiom hierarchy, and reach consensus through an auditable protocol.

---

## 2. Architecture Overview

Five layers, each with a single clear responsibility:

```
Layer 1: Data Substrate        → SQLite databases (case-scoped)
Layer 2: A2T Protocol          → 4 MCP servers (typed tool schemas)
Layer 3: Specialized Agents    → 4 agents (one per biological modality)
Layer 4: A2A Consensus         → Debate engine + Hierarchy of Truth
Layer 5: Output & Audit        → Prediction + confidence + debate trace
```

Data flows strictly downward: substrate → MCP → agents → debate → output. No layer reaches across or skips another.

---

## 3. Data Layer

### 3.1 Case-First Approach

Evaluation cases are selected before any data is downloaded. A `cases.yaml` config file defines the 30–50 (cell_line, drug) pairs used throughout the project. ETL pipelines filter raw datasets to only those cell lines and drugs, producing lean SQLite databases. The MCP server code is fully general — it runs SQL queries and has no knowledge of which cases are in the database. Adding new cases requires only updating `cases.yaml` and re-running ETL.

### 3.2 Datasets

| Database | Source | Notes |
|---|---|---|
| `genomics.db` | GDSC2 (mutations, CNV), CCLE (CNV) | Filter to selected cell lines |
| `transcriptomics.db` | CCLE RNAseq (TPM → z-scores) | Filter to selected cell lines; includes methylation signals as silencing indicator |
| `pharmacology.db` | GDSC2 (IC50), ChEMBL REST API (drug targets, MOA) | ChEMBL replaces DrugBank — fully open, no license required |
| `pathways.db` | KEGG REST API (KGML → gene-interaction graphs) | Pre-computed bypass routes |

**Validation set**: CTRPv2 for held-out evaluation. Check cell line overlap with GDSC2 in Week 1 — if overlap is insufficient, fall back to 5-fold stratified cross-validation on GDSC2.

### 3.3 ETL Structure

```
data/
├── raw/                  # Original downloads (never modified)
│   ├── gdsc2/
│   ├── ccle/
│   └── ctpv2/
├── processed/            # SQLite databases (ETL output)
│   ├── genomics.db
│   ├── transcriptomics.db
│   ├── pharmacology.db
│   └── pathways.db
└── cases.yaml            # Single source of truth for evaluation scope
```

---

## 4. A2T Protocol — MCP Servers

Each MCP server wraps one SQLite database and exposes typed, schema-validated tools via FastMCP. Structured JSON returns are what makes the system deterministic rather than linguistic — agents cite specific data fields, not paraphrased text.

### 4.1 Genomics MCP Server

| Tool | Parameters | Returns |
|---|---|---|
| `get_mutations` | `cell_line, gene?` | `[{gene, mutation, type, cosmic_id}]` |
| `get_cnv` | `cell_line, gene?` | `[{gene, cnv_value, status}]` |
| `check_mutation_impact` | `gene, mutation` | `{is_driver, is_actionable, drugs_targeting}` |

### 4.2 Transcriptomics MCP Server

| Tool | Parameters | Returns |
|---|---|---|
| `get_expression` | `cell_line, gene?` | `[{gene, z_score, tpm, percentile}]` |
| `get_pathway_expression` | `cell_line, pathway_id` | `{pathway_genes: [{gene, z_score}], mean_z}` |
| `check_silencing` | `cell_line, gene` | `{is_silenced, z_score, threshold_used}` |

Note: methylation is folded into this server as a silencing signal. A dedicated methylation/epigenomics agent is noted as future work.

### 4.3 Pharmacology MCP Server

| Tool | Parameters | Returns |
|---|---|---|
| `get_ic50` | `cell_line, drug` | `{ic50, ln_ic50, z_score, auc}` |
| `get_drug_info` | `drug` | `{target_genes, moa, pathway, drug_class}` |
| `get_sensitivity_profile` | `drug, mutation?` | `{median_ic50, resistant_fraction, n_cell_lines}` |

### 4.4 Pathway MCP Server

| Tool | Parameters | Returns |
|---|---|---|
| `get_pathway_genes` | `pathway_id` | `[{gene, role, position}]` |
| `check_bypass` | `pathway_id, blocked_gene` | `{bypass_exists, bypass_genes}` |
| `get_upstream_regulators` | `gene` | `[{regulator, relationship, pathway}]` |

---

## 5. Specialized Agents

### 5.1 Agent Decomposition

Each agent maps to one level of biological organization:

| Agent | Modality | Biological Level |
|---|---|---|
| Genomics Agent | DNA alterations (mutations, CNV) | DNA |
| Transcriptomics Agent | RNA expression + silencing signals | RNA |
| Pharmacology Agent | Drug response history + MOA | Drug interaction |
| Pathway Agent | Pathway topology, bypass, regulators | Network |

This mirrors the Central Dogma and creates natural conflict points (e.g., mutation present but gene silenced) that the debate protocol must resolve.

### 5.2 Agent Internal Structure

Each agent follows the same pipeline:
1. Receive (cell_line, drug) input
2. Call MCP tools to retrieve structured data
3. Reason over data using modality-specific system prompt + axioms
4. Emit a structured `EvidencePack` JSON

### 5.3 EvidencePack Schema

```json
{
  "agent_id": "genomics_agent",
  "cell_line": "A375",
  "drug": "Vemurafenib",
  "verdict": "SENSITIVE",
  "confidence": 0.85,
  "evidence_tier": "T1_STRUCTURAL",
  "key_findings": [
    {
      "biomarker": "BRAF_V600E",
      "value": "mutant",
      "interpretation": "Oncogenic driver present; drug target structurally active",
      "data_source": "GDSC_mutations",
      "axiom_invoked": "CENTRAL_DOGMA_DNA_PRIMACY"
    }
  ],
  "caveats": [],
  "conflict_flags": []
}
```

### 5.4 LLM Abstraction Layer

A thin backend abstraction decouples agent logic from any specific LLM API. Swapping models requires one config change:

```python
class LLMBackend:
    def complete(self, messages, tools=None) -> str: ...

class GeminiBackend(LLMBackend): ...   # Google AI Studio — covers both Gemini Flash and Gemma 3 (same API, different model_id)
class GroqBackend(LLMBackend): ...     # Qwen 2.5 72B via Groq
```

Target models:
- **Gemini 1.5 Flash** (Google AI Studio, `GeminiBackend`) — primary development model; best free-tier tool-use reliability
- **Qwen 2.5 72B** (Groq, `GroqBackend`) — strong structured output; very fast inference
- **Gemma 3 27B** (Google AI Studio, `GeminiBackend` with different model_id) — third model for LLM-agnosticism claim

### 5.5 Response Caching

Every LLM call is cached to disk keyed by `(model, hash(system_prompt), hash(input))`. After the first run, free-tier rate limits are irrelevant and results are fully reproducible — which is a thesis requirement.

---

## 6. A2A Consensus Protocol

### 6.1 Debate State Machine

```
IndependentAnalysis → EvidenceExchange → ConflictDetection
                                               ↓ no conflicts
                                           Consensus → FinalPrediction
                                               ↓ conflicts
                                        StructuredDebate → AxiomResolution → RevisedPositions → ConflictDetection
                                               ↓ max rounds (3) reached
                                        ForcedResolution → Consensus
```

### 6.2 Hierarchy of Truth — 5 Axiom Tiers

| Tier | Rule | Example |
|---|---|---|
| **T1 Structural Primacy** | DNA-level deletion or loss-of-function overrides all downstream signals | Homozygous EGFR deletion → RESISTANT regardless of expression or IC50 |
| **T2 Transcriptional Gate** | Gene silenced (z-score < −2.0) → protein functionally absent regardless of DNA status | BRAF V600E present but z-score −2.3 → flag conflict |
| **T3 Pathway Bypass** | Active bypass route → resistance likely even if primary target is present | ERK bypass via MEK2 active → escalate resistance risk |
| **T4 Pharmacological Prior** | Historical IC50 is population-level baseline; overridden by T1–T3 molecular evidence | IC50 shifts prediction ±2 tiers when molecular evidence is present |
| **T5 Statistical Consensus** | Fallback when all molecular evidence is absent or ambiguous | Use median IC50-based prediction |

Agents can only change their verdict if presented with a higher-tier axiom challenge. They cannot simply agree without mechanistic justification.

### 6.3 Anti-Sycophancy Mechanisms

1. **Blind initial analysis** — agents produce EvidencePacks independently before seeing any other agent's output
2. **Devil's advocate injection** — one agent per round is assigned to challenge the emerging consensus using its own data
3. **Confidence decay** — verdict flip costs 15% confidence per flip, discouraging weathervaning
4. **Axiom-locked positions** — verdict changes require a higher-tier axiom challenge, not just agreement
5. **Dissent logging** — overridden minority positions are preserved in the final trace

---

## 7. Evaluation Framework

### 7.1 Predictive Accuracy

Binary classification: SENSITIVE vs. RESISTANT per (cell_line, drug) pair.

| Metric | Target |
|---|---|
| AUROC | > 0.75 |
| AUPRC | > 0.60 |
| Spearman ρ | > 0.50 |
| Cohen's κ | > 0.40 |

Splits: GDSC2 for development, CTRPv2 for held-out validation (or 5-fold CV on GDSC2 if overlap is insufficient).

### 7.2 Reasoning Quality

| Metric | How |
|---|---|
| Axiom invocation rate | Count per tier across all cases |
| Conflict resolution accuracy | Pre-debate vs. post-debate prediction accuracy |
| Debate convergence | Average rounds to consensus (target: 1.5–2.5) |
| Hallucination rate | Agent claims not present in MCP returns (target: < 5%) |

### 7.3 Ablation Studies (4 core)

| Ablation | What Is Removed |
|---|---|
| No Debate | Agents vote independently; majority wins |
| No Axioms | Free debate without Hierarchy of Truth |
| Monolithic LLM | Single model receives all data in one prompt |
| No MCP | Agents receive data as natural language instead of structured JSON |

### 7.4 Multi-Model Comparison

Run the full evaluation with all three models (Gemini Flash, Qwen 2.5, Gemma 3). This validates the LLM-agnosticism claim and is a publishable finding on its own.

### 7.5 Case Studies (3–5)

| Case | Cell Line | Drug | Known Mechanism |
|---|---|---|---|
| BRAF-mutant melanoma | A375 | Vemurafenib | BRAF V600E → sensitive |
| EGFR bypass | HCC827 | Gefitinib | MET amplification bypass → resistant |
| Expression silencing | MCF7 | Tamoxifen | ESR1 expression variation |
| Multi-drug resistance | K562 | Imatinib | BCR-ABL + MDR1 expression |

---

## 8. Technology Stack

| Component | Technology |
|---|---|
| LLM APIs | Google AI Studio (Gemini Flash, Gemma 3), Groq (Qwen 2.5 72B) |
| MCP Servers | Python + FastMCP |
| Data Storage | SQLite (one DB per modality) |
| Agent Orchestration | Python asyncio + custom orchestrator |
| Schema Validation | Pydantic |
| Evaluation | scikit-learn + custom metrics |
| Visualization | Matplotlib + Plotly |

No LangChain, AutoGen, or other orchestration frameworks. The debate protocol is the thesis contribution — it must be visible, not hidden behind a framework.

---

## 9. Project Structure

```
decentralized-mas/
├── cases.yaml                  # Evaluation scope — single source of truth
├── pyproject.toml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/                   # ETL pipelines
│   ├── mcp_servers/            # Layer 2: Genomics, Transcriptomics, Pharmacology, Pathway
│   ├── agents/                 # Layer 3: BaseAgent + 4 specialized agents
│   ├── llm/                    # LLM abstraction layer + response cache
│   ├── protocols/              # Layer 4: DebateEngine, AxiomResolver, ConflictDetector, Consensus
│   ├── schemas/                # Pydantic models: EvidencePack, DebateMessage, AxiomRules
│   └── evaluation/             # Metrics, ablation runner, visualizations
├── experiments/
│   ├── configs/
│   └── results/
├── tests/
└── notebooks/
```

---

## 10. Revised 9-Week Timeline

| Week | Phase | Key Deliverables |
|---|---|---|
| 1 | Case selection + Data | `cases.yaml`. Download GDSC2 + CCLE. ETL → `genomics.db`, `transcriptomics.db` |
| 2 | Data + MCP start | ChEMBL → `pharmacology.db`. KEGG → `pathways.db`. Validation tests. Genomics + Pharmacology MCP servers |
| 3 | MCP Servers | All 4 MCP servers complete. Integration tests. Validate on 5 known cases |
| 4 | Agent Design | `BaseAgent` + LLM abstraction + response cache. All 4 agent system prompts. EvidencePack parsing. Test on 10 cases |
| 5–6 | A2A Protocol | Debate state machine, ConflictDetector, AxiomResolver, anti-sycophancy, ConsensusAggregator. End-to-end test on 20 cases |
| 7 | Evaluation | Full 30–50 cases with Gemini Flash. 4 ablations. 3–5 case study deep-dives |
| 8 | Multi-model + Analysis | Re-run with Qwen 2.5 and Gemma 3. All metrics computed. Figures generated |
| 9 | Writing | Results chapter, discussion, limitations |

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| CTRPv2 / GDSC2 overlap too small | Check in Week 1; fall back to 5-fold CV on GDSC2 |
| Week 4 prompt iteration overruns | Budget 3 days of buffer before moving to protocol |
| Free-tier rate limits during evaluation | Response caching eliminates this after first run |
| Agents produce malformed EvidencePack JSON | Pydantic validation + retry with format correction prompt |
| Debate doesn't converge | Max 3 rounds; forced resolution via weighted confidence |

---

## 12. Development Workflow

- **Before any code addition**: invoke `simplify` skill to keep the codebase clean
- **Before any git push**: invoke `code-review` skill (or `review` for PRs)
- **All implementation work**: use superpowers skills (TDD, writing-plans, executing-plans)
