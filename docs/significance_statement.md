# Innovation Experiment — Significance Statement

---

## NIH Instruction #1: The Problem and Why It Must Be Solved Now

Drug resistance is responsible for more than 90% of cancer deaths in patients with metastatic disease (Housman *et al.*, 2014). It is not a side effect of cancer treatment — it is the primary reason treatment fails. Targeted therapies, which were supposed to change this story, have largely not. In BRAF-mutant melanoma, the majority of patients who initially respond to BRAF/MEK inhibitor combinations develop resistance within two years (Proietti *et al.*, 2020). In EGFR-driven lung cancer, acquired resistance emerges within a median of 9–13 months even in patients with the canonical activating mutation (Rotow & Bivona, 2017). Globally, the WHO estimates that cancer claims 10 million lives per year, and the fraction attributable to treatment failure from resistance is growing as targeted therapy use expands. This is not a problem on the horizon — it is happening now, at scale, and it is killing patients who responded to first-line therapy.

The reason we cannot reliably predict resistance before it happens is not a shortage of data. Public repositories — GDSC2 (Iorio *et al.*, 2016), CCLE (Barretina *et al.*, 2012), CTRPv2 — contain pharmacological sensitivity profiles matched with genomic, transcriptomic, and pathway annotations for thousands of cell line–drug pairs. The data exists. What is missing is a computational framework that reasons over that data the way a trained oncologist does: by recognising that evidence from different biological layers has a natural causal ordering, and that when those layers disagree, the hierarchy — not a statistical average — should determine the answer. A DNA-level deletion of a drug target cannot be overridden by an expression signal. A transcriptionally silenced gene cannot confer drug sensitivity regardless of what its mutation status says. A pathway bypass renders the primary target irrelevant even if it is functional. No current computational tool encodes this logic explicitly, and the cost of that gap is measured in lost lives.

---

## NIH Instruction #2: Scientific Premise — Preliminary Data and the Gap

**We hypothesise that a decentralised multi-agent system, in which four specialised agents reason over modality-specific databases and resolve conflicts through an explicit five-tier biological Hierarchy of Truth, will predict drug resistance more accurately and more interpretably than monolithic LLM baselines and naive multi-agent majority-vote systems.** This hypothesis is grounded in three converging lines of evidence.

*First*, the molecular determinants of resistance are well-characterised at the level of biological causality. The central dogma provides a natural tier ordering: structural DNA alterations are upstream of transcriptional state, which is upstream of protein activity, which determines pathway flux, which is the proximate driver of drug response. This ordering is not controversial — it is textbook biology. What is novel is encoding it as executable logic rather than leaving it implicit in model weights.

*Second*, our preliminary implementation demonstrates that this encoding is technically feasible. We have built and tested four specialised agents (Genomics, Transcriptomics, Pharmacology, Pathway), each connected to a curated SQLite database via the Model Context Protocol (MCP). Each agent produces a structured `EvidencePack` with a verdict, confidence, evidence tier, and key findings. A `ConflictDetector` identifies disagreements between agents; an `AxiomResolver` enforces the tier ordering to determine which agent's verdict prevails; a `DebateEngine` orchestrates up to three rounds of structured argumentation before forcing resolution. All components have been implemented and validated against 40 GDSC2-verified cell line–drug pairs, with 83 unit and integration tests passing. The five-tier axiom system is defined as follows:

- **T1 — Structural Primacy**: DNA-level deletion or loss-of-function overrides all downstream signals
- **T2 — Transcriptional Gate**: Transcriptional silencing (mRNA z-score < −2.0) renders a gene functionally absent regardless of mutation status
- **T3 — Pathway Bypass**: An active bypass route confers resistance even if the primary drug target is intact
- **T4 — Pharmacological Prior**: Historical IC50 data is a population-level baseline, overridable by T1–T3 evidence
- **T5 — Statistical Consensus**: When molecular evidence is absent, population statistics determine the prediction

Successful completion of this proposal will produce the first computational drug resistance framework in which every prediction is accompanied by a structured reasoning trace that names the axiom tier that resolved the case, the agents that were overridden, and the dissenting positions that were preserved. This is directly analogous to the minority opinions recorded in human tumour board decisions — a form of clinical accountability that no existing AI tool provides.

*Third*, this proposal addresses specific weaknesses in prior work. The most relevant existing approaches fall into two categories. Classical ML methods (Yang *et al.*, 2021; Liu *et al.*, 2020) achieve competitive AUROC values on held-out benchmarks but are black boxes: there is no mechanism to inspect which data layer drove a particular prediction, or to understand why the model is confident when evidence is contradictory. Their predictions cannot be audited by a clinician. LLM-based approaches (e.g., CellHit, Parca *et al.*, 2025; EvoMDT, Chen *et al.*, 2024) introduce natural-language reasoning but suffer from two compounding failure modes. *Context overwhelm* occurs when a single model is given mutations, expression values, IC50 data, and pathway annotations simultaneously: the model's reasoning degrades as it attempts to weight incommensurable signals without a principled framework. *Sycophancy* (Turpin *et al.*, 2023) occurs when a multi-agent system resolves conflicts through social consensus rather than biological logic, causing a correct minority signal to be suppressed rather than preserved. Neither failure mode is addressed by scaling data or model parameters. Both are addressed by the architecture proposed here. This proposal will determine the smallest axiom tier granularity that preserves predictive accuracy — the question of whether all five tiers are necessary, or whether a coarser hierarchy suffices, is addressed in the ablation studies.

---

## NIH Instruction #3: How This Project Will Improve Scientific Knowledge and Clinical Practice

The project will be completed in three integrated aims. **Aim 1** establishes predictive accuracy: the full framework will be run against the GDSC2 development set and evaluated on AUROC, AUPRC, Cohen's κ, and Spearman ρ between predicted confidence and measured z-score magnitude. **Aim 2** establishes reasoning quality: hallucination rate (fraction of cited biomarkers absent from the underlying database) and biological faithfulness score (fraction of axiom invocations consistent with established molecular biology) will be computed for every run, introducing evaluation dimensions that are absent from all prior work in this area. **Aim 3** runs six ablation studies — removing the axiom hierarchy, collapsing to a single agent, replacing structured MCP tool returns with free-text summaries, randomising tier order, removing the debate loop, and using majority vote instead of axiom resolution — to demonstrate that each architectural choice contributes independently to performance.

The direct scientific benefit is a reproducible, model-agnostic framework for multi-modal biological reasoning that the field can adopt, extend, and critique. The direct technical benefit is an open-source MCP-based agent-to-tool protocol that can be adapted to other multi-modal biomedical tasks beyond drug resistance. The direct clinical benefit is a new class of AI output: not a confidence score, but a structured argument with explicitly logged uncertainties that a clinician can interrogate in the language of molecular oncology.

---

## NIH Instruction #4: What Will Change If the Aims Are Achieved

If the proposed aims are achieved, the primary change will be a reframing of multi-modal data fusion in computational oncology from a *statistical problem* to a *structured argumentation problem*. The field currently measures progress in AUROC points and optimises for data scale. This work proposes that reasoning quality metrics — faithfulness, hallucination rate, dissent preservation — are at least equally important for clinical systems, and it provides the first framework for measuring them. That reframing, if it holds, would change what the field builds toward.

At the level of methods, the axiom-governed debate architecture is generalisable beyond drug resistance. Any domain where evidence from multiple modalities must be reconciled against a known causal hierarchy — rare disease diagnosis, treatment response prediction, biomarker discovery — could adopt the same protocol layer. The longer-term potential is a general framework for biologically faithful AI reasoning that reduces hallucination not by instruction-tuning but by grounding inference in explicit domain axioms.

*As a result, our knowledge of how multi-modal molecular evidence should be integrated during clinical inference will be advanced by a transparent, auditable, and biologically faithful reasoning architecture — one that makes the reasoning process a first-class scientific object rather than a black-box intermediate, and that brings computational oncology one step closer to the kind of AI-assisted decision support that clinicians can actually trust.*

---

## Reflection

The most useful lesson from studying the Wrenshall sock example was structural: the hypothesis and preliminary data should appear early and together, so the reviewer understands what you are claiming and why you believe it before you describe what you intend to do. My first draft buried both under NIH instruction headers, turning the document into an outline rather than a story. Revising to lead with the problem, immediately restate the hypothesis, and then support it with the three converging lines of evidence made the document read as an argument rather than a checklist. Dr. Locke's framing of *strengths* before *weaknesses* also clarified the structure: the three evidence pillars (biological prior, preliminary implementation, gap in prior work) establish credibility before the weaknesses of prior approaches are named. The ablation studies section grew directly out of Wrenshall's note about naming the specific weakness that each aim addresses — each ablation is precisely that: a named gap being formally tested.

---

## Feedback

*[To be completed after peer or supervisor review. Record: who gave the feedback, how it was solicited, and what specific changes were made in response.]*
