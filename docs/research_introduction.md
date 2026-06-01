# Research Introduction

## About Me

Hey everyone!

I am Shrinithi Natarajan and I am currently employed as a software development engineer at Apple, Inc India. By day I code, but by heart I've always had a burning passion for biology (think of it as my favourite side mission that turned into the main storyline). My foundation in computer science and absolute love for biology brought me to this stream where I hope to learn, collaborate, and work with experts in this field. I have a few publications of my own where I explore using Machine Learning and Graph Theory in the field of disease analysis and prediction algorithms.

At one point I was a research intern at the Life Sciences department in Tata Consultancy Services in India. This is where I was introduced to bovine cancer and, using data analytics (to the best of my abilities then), I was tasked with finding the underlying cause of inflammation in cattle udder that ultimately led to a type of bovine cancer called mastitis. Since then I became fascinated in the field of cancer. I would love to learn more about how this can be modelled mathematically and how we can derive answers from such models.

The best moment of my life was when I had the opportunity to intern at the Indian Institute of Science in Bangalore, India, where I built an RNA analysis pipeline for an ongoing research work. In return, I got to do some absolutely wonderful experiments in their laboratory where I learnt from the best of the best!

---

## The Broad Problem: Cancer Drug Resistance

Cancer remains one of the leading causes of death worldwide, and while targeted therapies have transformed oncology over the past two decades, their long-term effectiveness is consistently undermined by drug resistance. Whether resistance is intrinsic — present before treatment begins — or acquired through selective pressure during therapy, it converts initially responsive tumours into lethal, treatment-refractory disease. Understanding *why* a given cancer cell resists a given drug is not merely an academic question; it is the central bottleneck between drug discovery and durable patient benefit.

## A Specific Problem: Multi-Modal Evidence, Single-Model Reasoning

The molecular determinants of drug resistance span multiple biological layers simultaneously: somatic mutations alter the drug target itself (DNA), transcriptional silencing can functionally abolish a protein despite its gene being intact (RNA), signalling pathway rewiring creates bypass routes that circumvent the blocked target (network topology), and historical pharmacological data captures population-level sensitivity priors that any mechanistic model must be reconciled against. Current computational approaches — whether classical machine learning on genomic features or large language models given all available data in a single prompt — treat this as a flat retrieval problem. They either collapse the multi-modal signal into a feature vector, losing the interpretable chain of biological causality, or overwhelm a single reasoning agent with contradictory evidence from incommensurable data types, producing confident but poorly grounded predictions.

## The Critical Research Gap

No existing framework formalises *how* evidence from different biological modalities should be weighted against each other when they conflict — what might be called a Hierarchy of Truth for oncology reasoning. The consequence is that current AI-assisted resistance prediction is vulnerable to two failure modes that mirror known problems in human clinical decision-making: **context overwhelm** (too much data degrades reasoning quality) and **sycophancy** (a model converges on a confident answer by suppressing minority signals rather than by genuinely resolving them).

**A concrete example.** Consider the melanoma cell line A375 treated with Vemurafenib, a BRAF inhibitor. A genomics model sees the canonical BRAF V600E driver mutation and confidently predicts sensitivity — correctly, in isolation. Now imagine a second scenario: the same BRAF V600E mutation is present, but the BRAF gene's mRNA expression has a z-score of −2.5, meaning the gene is effectively transcriptionally silenced. A flat model that concatenates all features may still predict sensitivity because the mutation signal dominates numerically. A clinician, however, would immediately flag the contradiction: *if the gene is not being expressed, the mutation cannot confer drug sensitivity, regardless of what the DNA says.* A third layer of complexity arises if the MAPK pathway — which BRAF sits in — has an active bypass route through MEK2; even if BRAF is functional, the drug's downstream target is already being circumvented, pushing the prediction back toward resistance. No current model has an explicit mechanism to reason through these three layers in order and explain which piece of evidence overrode the others.

The Park, Leahey & Funk (2023) *Nature* paper on the observed decline in disruptive science is directly relevant here. Their central finding — that contemporary research increasingly *consolidates* existing knowledge rather than *destabilising* it — maps precisely onto this failure mode. Most computational drug resistance tools are consolidatory: they build larger models on larger datasets, achieving incremental AUROC improvements by better capturing the existing literature. The CD index (disruptiveness metric) of such work is low by design; the models cite heavily from prior art and deepen existing frameworks rather than proposing a new organising principle for how biological evidence should be reasoned about.

## Hypothesis and Statement of Purpose

This project proposes that drug resistance prediction can be made both more accurate and more interpretable by decomposing the reasoning task across specialised agents, each responsible for a single biological modality, and by coordinating their predictions through a structured, axiom-governed debate protocol rather than by aggregating them statistically. The central hypothesis is: **a decentralised multi-agent system that resolves inter-modality conflicts using an explicit Hierarchy of Truth will outperform both monolithic LLM baselines and naive majority-vote multi-agent systems, while producing mechanistically grounded reasoning traces that are auditable by a human clinician.**

This is a disruptive rather than consolidatory research contribution in the sense Park et al. describe. Rather than citing the prior literature to build a better single model, it proposes a new *architecture* for how biological knowledge should be organised during inference — one that makes the reasoning process a first-class scientific object rather than a black-box intermediate.

## Variables and Measurement

The primary prediction task is binary classification of (cell line, drug) pairs as SENSITIVE or RESISTANT, evaluated against GDSC2 ground-truth labels. The key variables are:

- **Predictive accuracy**: AUROC, AUPRC, Cohen's κ, and Spearman ρ between predicted confidence and z-score magnitude
- **Reasoning quality**: hallucination rate (fraction of cited biomarkers not present in the structured data), biological faithfulness score (fraction of axiom invocations consistent with the Hierarchy of Truth), and debate convergence rate
- **Anti-sycophancy**: frequency with which a minority agent's correct signal is preserved in the final trace despite being overridden by majority vote — quantifying whether the system *logs* correct dissent rather than silencing it
- **Disruptiveness of the architecture**: measured via ablation — removing the axiom hierarchy, collapsing to a single agent, or replacing structured MCP tool returns with free-text data summaries, and observing performance degradation

The four modalities assessed are genomics (somatic mutations, copy number variation), transcriptomics (mRNA expression z-scores), pharmacology (historical IC50 and drug target data), and pathway topology (KEGG signalling maps and bypass routes), drawn from GDSC2 and CCLE.

## How Conclusions Connect Back to the Central Problem

If the hypothesis holds, the results will demonstrate that the biological knowledge already present in public databases is *sufficient* for high-quality resistance prediction when the reasoning architecture respects the causal hierarchy of biology — DNA before RNA before protein before network before population statistics. This would shift the field's attention from *data scale* (the consolidatory direction) toward *reasoning architecture* (the disruptive direction), providing a reproducible framework for building clinical decision-support tools that explain themselves in the language of molecular oncology rather than in gradient magnitudes. The debate traces produced by each run are themselves a scientific contribution: a structured, citable record of which evidence tier resolved each case, directly analogous to the minority opinions preserved in real tumour board decisions.

---

## Reflection: Disruptive Science and This Project

Park, Leahey & Funk's finding that papers increasingly build on a narrower prior knowledge base — measured by citations that *consolidate* rather than *destabilise* existing work — resonates uncomfortably with the trajectory of computational oncology. The dominant research pattern is: take a known architecture (transformer, GNN, ensemble), train it on a larger version of an existing benchmark (GDSC, CCLE, TCGA), and report a 1–3% AUROC improvement. The CD index of such work is structurally low: it depends heavily on prior art and does not open new lines of investigation.

This project attempts a different move. Rather than improving prediction by scaling data or model parameters, it introduces a new *protocol layer* — the axiom-governed debate — as the unit of contribution. If the protocol is correct, it should be applicable to any biological reasoning task where evidence from multiple modalities must be reconciled against a causal hierarchy. That generalisability is what makes it potentially disruptive in the Park et al. sense: it could cause researchers to *stop* treating multi-modal fusion as a statistical problem and *start* treating it as a structured argumentation problem. Whether it achieves that depends on the ablation results, but the architecture is deliberately designed to make that claim testable.
