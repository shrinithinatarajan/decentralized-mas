# Generate debate trace card figures matching the two-panel layout style.
# Usage: PYTHONPATH=. python experiments/generate_trace_cards.py
from __future__ import annotations
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path

OUT_DIR = Path('images')
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({'font.family': 'sans-serif', 'figure.dpi': 150})

# Agent accent colors (dark pill fills)
AGENT_COLOR = {
    'genomics':       '#1B3F8B',
    'transcriptomics':'#6B2E9E',
    'pathway':        '#1A6B45',
    'pharmacology':   '#8B1A1A',
}
AGENT_LABEL = {
    'genomics':       'T1 GENOMICS',
    'transcriptomics':'T2 TRANSCRIPTOMICS',
    'pathway':        'T3 PATHWAY',
    'pharmacology':   'T4 PHARMACOLOGY',
}
VERDICT_COLOR = {
    'SENSITIVE': '#27AE60',
    'RESISTANT': '#C0392B',
    'UNCERTAIN': '#7F8C8D',
}
VERDICT_BG = {
    'SENSITIVE': '#EAFAF1',
    'RESISTANT': '#FDEDEC',
    'UNCERTAIN': '#F2F3F4',
}


def _wrap(text: str, width: int) -> str:
    return '\n'.join(textwrap.wrap(text, width))


def draw_card(
    filename: str,
    case_title: str,
    true_label: str,
    correct: bool,
    agents: list[dict],  # {key: genomics/transcriptomics/pathway/pharmacology,
                         #  verdict, gcs, inline_data, reasoning}
    outcome_method: str,    # e.g. 'R1 CONSENSUS: SENSITIVE'
    outcome_verdict: str,   # SENSITIVE / RESISTANT
    outcome_gcs: float,
    outcome_lines: list[str],  # bullet lines below header
    insight: str,
) -> None:
    FW, FH = 13.0, 6.2
    fig = plt.figure(figsize=(FW, FH), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ---- section labels ----
    ax.text(0.35, FH - 0.22, 'ROUND 1 — INDEPENDENT',
            fontsize=8, color='#AAAAAA', fontweight='bold',
            ha='left', va='top', transform=ax.transData)
    ax.text(8.1, FH - 0.22, 'OUTCOME',
            fontsize=8, color='#AAAAAA', fontweight='bold',
            ha='left', va='top', transform=ax.transData)

    # ---- agent rows (left panel) ----
    LEFT_X0 = 0.25
    LEFT_W  = 7.5
    ROW_H   = 1.26
    ROW_GAP = 0.12
    TOP_Y   = FH - 0.55

    for i, ag in enumerate(agents):
        akey  = ag['key']
        ac    = AGENT_COLOR[akey]
        vc    = VERDICT_COLOR[ag['verdict']]
        vbg   = VERDICT_BG[ag['verdict']]
        ry    = TOP_Y - i * (ROW_H + ROW_GAP)

        # card background
        card = FancyBboxPatch((LEFT_X0, ry - ROW_H), LEFT_W, ROW_H,
                              boxstyle='round,pad=0.05', linewidth=0,
                              facecolor='#F7F8FA', zorder=1)
        ax.add_patch(card)
        # left accent bar
        bar = Rectangle((LEFT_X0, ry - ROW_H), 0.14, ROW_H,
                         linewidth=0, facecolor=ac, zorder=2)
        ax.add_patch(bar)

        # agent pill
        pill_x = LEFT_X0 + 0.26
        pill_y = ry - 0.35
        pill_w = 2.05
        pill_h = 0.34
        apill = FancyBboxPatch((pill_x, pill_y), pill_w, pill_h,
                               boxstyle='round,pad=0.06', linewidth=0,
                               facecolor=ac, zorder=3)
        ax.add_patch(apill)
        ax.text(pill_x + pill_w / 2, pill_y + pill_h / 2,
                AGENT_LABEL[akey],
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='white', zorder=4)

        # verdict pill (outlined)
        vp_x = pill_x + pill_w + 0.16
        vp_y = pill_y
        vp_w = 1.15
        vp_h = pill_h
        vpill = FancyBboxPatch((vp_x, vp_y), vp_w, vp_h,
                               boxstyle='round,pad=0.06', linewidth=1.6,
                               edgecolor=vc, facecolor=vbg, zorder=3)
        ax.add_patch(vpill)
        ax.text(vp_x + vp_w / 2, vp_y + vp_h / 2,
                ag['verdict'],
                ha='center', va='center', fontsize=8, fontweight='bold',
                color=vc, zorder=4)

        # GCS
        gcs_x = vp_x + vp_w + 0.22
        ax.text(gcs_x, pill_y + pill_h / 2,
                f"GCS {ag['gcs']:.3f}",
                ha='left', va='center', fontsize=9.5, fontweight='bold',
                color='#222222', zorder=4)

        # inline data (right of GCS, lighter)
        ax.text(gcs_x + 1.05, pill_y + pill_h / 2,
                ag['inline_data'],
                ha='left', va='center', fontsize=7.5,
                color='#AAAAAA', zorder=4)

        # reasoning text
        wrapped = _wrap(ag['reasoning'], 85)
        ax.text(pill_x, ry - 0.82,
                wrapped,
                ha='left', va='top', fontsize=7.8,
                color='#555555', style='italic',
                linespacing=1.4, zorder=4)

    # ---- outcome panel (right) ----
    OC = VERDICT_COLOR[outcome_verdict]
    OBG = '#EEF8F2' if outcome_verdict == 'SENSITIVE' else '#FDF1F0'
    PX0 = 8.1
    PW  = FW - PX0 - 0.25
    PY0 = 0.25
    PH  = FH - 0.55

    opanel = FancyBboxPatch((PX0, PY0), PW, PH,
                            boxstyle='round,pad=0.05', linewidth=2,
                            edgecolor=OC, facecolor=OBG, zorder=1)
    ax.add_patch(opanel)

    # outcome header
    ax.text(PX0 + 0.18, PY0 + PH - 0.18,
            outcome_method,
            ha='left', va='top', fontsize=10, fontweight='bold',
            color=OC, zorder=2)

    # detail lines
    line_y = PY0 + PH - 0.52
    for line in outcome_lines:
        ax.text(PX0 + 0.18, line_y,
                line, ha='left', va='top',
                fontsize=7.8, color='#444444', zorder=2,
                linespacing=1.3)
        line_y -= 0.30

    # large GCS
    ax.text(PX0 + PW / 2, PY0 + PH - 2.05,
            f"{outcome_gcs:.3f}",
            ha='center', va='center', fontsize=32, fontweight='bold',
            color=OC, zorder=2)
    ax.text(PX0 + PW / 2, PY0 + PH - 2.50,
            f'· Ground truth: {true_label} {"✓" if correct else "✗"}',
            ha='center', va='center', fontsize=8,
            color='#888888', zorder=2)

    # key insight
    ax.text(PX0 + 0.18, PY0 + 1.82,
            'Key insight:',
            ha='left', va='top', fontsize=8.5, fontweight='bold',
            color=OC, zorder=2)
    wrapped_ins = _wrap(insight, 36)
    ax.text(PX0 + 0.18, PY0 + 1.54,
            wrapped_ins,
            ha='left', va='top', fontsize=7.5,
            color='#555555', style='italic',
            linespacing=1.35, zorder=2)

    fig.savefig(OUT_DIR / filename, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close(fig)
    print(f'  Saved {OUT_DIR / filename}')


# ---------------------------------------------------------------------------
# Trace definitions
# ---------------------------------------------------------------------------
def main() -> None:
    # --- Trace 1: BT-474 + Lapatinib (unanimous R1 consensus) ---
    draw_card(
        filename='trace1.png',
        case_title='BT-474  +  Lapatinib',
        true_label='SENSITIVE',
        correct=True,
        agents=[
            dict(key='genomics', verdict='SENSITIVE', gcs=0.895,
                 inline_data='ERBB2 CNV=14.07 · RPPA=3.47',
                 reasoning=('ERBB2 (HER2) amplification confirmed by CNV data (log2=14.07) and RPPA '
                             'protein expression (3.47). CIViC documents HER2 amplification as a '
                             'validated sensitivity marker for Lapatinib in breast cancer. '
                             'Voting SENSITIVE on T1_STRUCTURAL axiom.')),
            dict(key='transcriptomics', verdict='SENSITIVE', gcs=0.960,
                 inline_data='ERBB2 z=3.097 · TPM=1419',
                 reasoning=('ERBB2 TPM=1419 — constitutively high expression of the primary drug target. '
                             'Type A (overexpression-driven) protocol: z>2 overrides neutral z-score. '
                             'Voting SENSITIVE.')),
            dict(key='pathway', verdict='SENSITIVE', gcs=0.645,
                 inline_data='ErbB active · no bypass',
                 reasoning=('EGFR/ERBB2 present in active ErbB signalling and PI3K-Akt pathways. '
                             'Bypass check: no activating bypass routes for ERBB2 targets in KEGG data. '
                             'Active pathway + no escape mechanism → Voting SENSITIVE.')),
            dict(key='pharmacology', verdict='SENSITIVE', gcs=0.853,
                 inline_data='ln_ic50=0.108 · z=−1.967',
                 reasoning=('z-score = −1.967 (z < −0.5 = SENSITIVE). AUC=0.636 confirms strong absolute '
                             'dose-response. No contradictions in pharmacology data. Voting SENSITIVE.')),
        ],
        outcome_method='R1 CONSENSUS: SENSITIVE ✓',
        outcome_verdict='SENSITIVE',
        outcome_gcs=0.960,
        outcome_lines=[
            'All 4 agents agree in Round 1 · no debate required',
            'Winning agent: T2 Transcriptomics (highest GCS = 0.960)',
        ],
        insight=('Concordant evidence across all four modalities — '
                 'genomic amplification, transcriptional overexpression, '
                 'pharmacological sensitivity, and active pathway — '
                 'produces the highest-confidence correct prediction in the dataset. '
                 'No resolver required.'),
    )

    # --- Trace 2: NCI-H1573 + Erlotinib (minority T3 wins via AxiomResolver) ---
    draw_card(
        filename='trace2.png',
        case_title='NCI-H1573  +  Erlotinib',
        true_label='RESISTANT',
        correct=True,
        agents=[
            dict(key='genomics', verdict='SENSITIVE', gcs=0.700,
                 inline_data='EGFR WT · RPPA=2.52',
                 reasoning=('EGFR is wild-type with no structural alterations. RPPA shows high protein '
                             'expression (2.52). In the absence of a driver mutation, high protein abundance '
                             'is used as primary evidence. Voting SENSITIVE with lower confidence.')),
            dict(key='transcriptomics', verdict='UNCERTAIN', gcs=0.500,
                 inline_data='EGFR 235 TPM · z=1.84',
                 reasoning=('Erlotinib is a mutation-driven (Type B) drug. Sensitivity is determined by '
                             'activating mutations (exon 19 del / L858R), not by overexpression. '
                             'Without mutation status, predicting sensitivity is speculative. Voting UNCERTAIN.')),
            dict(key='pathway', verdict='RESISTANT', gcs=0.920,
                 inline_data='ERBB2/IGF1R/MET bypass detected',
                 reasoning=('EGFR in active MAPK, PI3K-Akt, ErbB, and EGFR-TKI resistance pathways. '
                             'Activating bypass genes ERBB2, IGF1R, and MET identified in bypass registry '
                             'with pathway activity scores ≥0.25. Escape routes confirmed → Voting RESISTANT.')),
            dict(key='pharmacology', verdict='SENSITIVE', gcs=0.818,
                 inline_data='ln_ic50=1.257 · z=−1.130',
                 reasoning=('IC50 z-score = −1.130 (z < −0.5 = SENSITIVE). However, AUC=0.903 indicates '
                             'poor absolute dose-response — a conflict between relative and absolute metrics. '
                             'Confidence capped due to discrepancy. Voting SENSITIVE.')),
        ],
        outcome_method='AXIOM RESOLVER: RESISTANT ✓',
        outcome_verdict='RESISTANT',
        outcome_gcs=0.694,
        outcome_lines=[
            'T1+T4 vote SENSITIVE; T3 votes RESISTANT; T2 abstains',
            'AxiomResolver: T3_PATHWAY tier > T4_PHARMACOLOGICAL tier',
            'Winning agent: T3 Pathway (GCS = 0.920)',
        ],
        insight=('Bypass pathway genes (ERBB2, IGF1R, MET) provide a mechanistic '
                 'escape route that neither IC50 data nor protein expression can '
                 'detect. The axiom hierarchy correctly elevates the pathway '
                 'signal over pharmacological and genomic priors.'),
    )

    # --- Trace 3: MDA-MB-231 + Olaparib (R2 consensus, T4 drives) ---
    draw_card(
        filename='trace3.png',
        case_title='MDA-MB-231  +  Olaparib',
        true_label='RESISTANT',
        correct=True,
        agents=[
            dict(key='genomics', verdict='UNCERTAIN', gcs=0.500,
                 inline_data='PARP1/2 hemizygous del · Chronos>−0.5',
                 reasoning=('PARP1/2 hemizygous deletions detected; non-essential by DepMap Chronos. '
                             'No evidence of homologous recombination deficiency. '
                             'Olaparib sensitivity requires HRD context. Voting UNCERTAIN.')),
            dict(key='transcriptomics', verdict='UNCERTAIN', gcs=0.500,
                 inline_data='PARP1 78 TPM',
                 reasoning=('Olaparib is a Type C (mechanism-agnostic) drug: efficacy depends on '
                             'synthetic lethality in BRCA1/2-mutant context, not PARP expression. '
                             'Target expression is not a reliable predictor. Voting UNCERTAIN.')),
            dict(key='pathway', verdict='SENSITIVE', gcs=0.645,
                 inline_data='BER/Apoptosis active · no bypass in R1',
                 reasoning=('PARP1/2 in active Base Excision Repair and Apoptosis pathways. No bypass '
                             'mechanism detected in R1. Pathway activity supports sensitivity signal. '
                             'Voting SENSITIVE — confidence weakens in Round 2 after peer critique.')),
            dict(key='pharmacology', verdict='RESISTANT', gcs=0.387,
                 inline_data='AUC=0.967 · z=+0.125',
                 reasoning=('AUC=0.967 (very high = poor absolute dose-response). z-score borderline '
                             'but AUC dominates: the cell line does not respond absolutely to PARP '
                             'inhibition. Voting RESISTANT.')),
        ],
        outcome_method='R2 CONSENSUS: RESISTANT ✓',
        outcome_verdict='RESISTANT',
        outcome_gcs=0.390,
        outcome_lines=[
            'T1 + T2 abstain; T3 loses confidence in Round 2',
            'Peer critique: no HRD evidence undermines T3 SENSITIVE',
            'Winning agent: T4 Pharmacology (IC50 evidence)',
        ],
        insight=('When genomic and transcriptomic agents abstain and pathway '
                 'evidence lacks HRD confirmation, the pharmacological prior '
                 '(high AUC = poor response) is sufficient to drive consensus '
                 'in Round 2. Debate resolves the initial pathway signal.'),
    )

    print('Done.')


if __name__ == '__main__':
    main()
