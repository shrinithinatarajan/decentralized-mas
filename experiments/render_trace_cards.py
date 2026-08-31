# Render trace card PNGs from standalone HTML using Chrome headless.
# Usage: PYTHONPATH=. python experiments/render_trace_cards.py
from __future__ import annotations
import subprocess, textwrap
from pathlib import Path

OUT_DIR = Path('images')
OUT_DIR.mkdir(exist_ok=True)
TMP_DIR = Path('/private/tmp/trace_html')
TMP_DIR.mkdir(exist_ok=True)

CHROME = ('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; font-size: 13px;
       background: #fff; padding: 20px 24px 20px 24px; }
.trace-grid { display: grid; grid-template-columns: 1fr 420px; gap: 28px; align-items: start; }
.trace-col-head { font-size: 10px; font-weight: 800; letter-spacing: 0.12em;
                  text-transform: uppercase; color: #888; margin-bottom: 12px; }
.acard { border-radius: 8px; padding: 11px 14px; margin-bottom: 8px;
         border: 1.5px solid #e0e0e0; background: #fff; }
.acard.t1 { border-left: 4px solid #2563EB; }
.acard.t2 { border-left: 4px solid #7C3AED; }
.acard.t3 { border-left: 4px solid #065f46; }
.acard.t4 { border-left: 4px solid #991b1b; }
.acard.winner { box-shadow: 0 0 0 2px #16A34A66; }
.acard-top { display: flex; align-items: center; gap: 7px; margin-bottom: 6px; }
.apill { font-size: 10.5px; font-weight: 800; padding: 3px 10px;
         border-radius: 4px; color: #fff; letter-spacing: 0.04em;
         text-transform: uppercase; white-space: nowrap; }
.apill.t1 { background: #1e3a8a; }
.apill.t2 { background: #5b21b6; }
.apill.t3 { background: #065f46; }
.apill.t4 { background: #991b1b; }
.vpill { font-size: 11px; font-weight: 700; padding: 2px 10px;
         border-radius: 12px; border: 1.5px solid; background: transparent;
         white-space: nowrap; }
.vpill.S { color: #16A34A; border-color: #16A34A; }
.vpill.R { color: #DC2626; border-color: #DC2626; }
.vpill.U { color: #6B7280; border-color: #6B7280; }
.acard-gcs { margin-left: auto; font-size: 13px; font-weight: 700;
             color: #111; white-space: nowrap; }
.acard-data { font-size: 10.5px; color: #888; margin-left: 4px;
              text-align: right; white-space: nowrap; }
.acard-reasoning { font-size: 12px; color: #444; line-height: 1.5; font-style: italic; }
.fbox { border-radius: 8px; padding: 14px 16px; margin-top: 4px; }
.fbox.correct { background: #f0fdf4; border: 1.5px solid #16A34A; }
.fbox.wrong   { background: #fef2f2; border: 1.5px solid #DC2626; }
.fbox-title { font-size: 14px; font-weight: 800; margin-bottom: 6px; }
.fbox-title.correct { color: #15803d; }
.fbox-title.wrong   { color: #b91c1c; }
.fbox-line { font-size: 12px; color: #333; line-height: 1.6; }
.fbox-conf { font-size: 36px; font-weight: 800; color: #111;
             margin: 8px 0 4px 0; }
.fbox-conf.correct { color: #15803d; }
.fbox-conf.wrong   { color: #b91c1c; }
.fbox-insight { margin-top: 10px; padding-top: 8px;
               border-top: 1px solid #d1fae5; }
.fbox-insight.wrong { border-top-color: #fecaca; }
.fbox-insight-head { font-size: 11.5px; font-weight: 800;
                     color: #166534; margin-bottom: 3px; }
.fbox-insight-head.wrong { color: #991b1b; }
.fbox-insight-text { font-size: 12px; color: #333;
                     line-height: 1.5; font-style: italic; }
"""


def html_trace(title: str, agents_html: str, outcome_html: str) -> str:
    return f"""<!doctype html><html><head><meta charset=utf-8>
<style>{CSS}</style></head><body>
<div class=\"trace-grid\">
  <div>
    <div class=\"trace-col-head\">Round 1 &mdash; Independent</div>
    {agents_html}
  </div>
  <div>
    <div class=\"trace-col-head\">Outcome</div>
    {outcome_html}
  </div>
</div>
</body></html>"""


# ---------------------------------------------------------------------------
# Trace 1 — BT-474 + Lapatinib (unanimous R1 consensus)
# ---------------------------------------------------------------------------
T1_AGENTS = """
<div class="acard t1">
  <div class="acard-top">
    <span class="apill t1">T1 Genomics</span>
    <span class="vpill S">SENSITIVE</span>
    <span class="acard-gcs">GCS 0.895</span>
    <span class="acard-data">ERBB2 CNV=14.07 &middot; RPPA=3.47</span>
  </div>
  <div class="acard-reasoning">ERBB2 amplification confirmed by CNV and RPPA protein data. CIViC documents HER2 amplification as a validated sensitivity marker for Lapatinib in breast cancer. Voting <strong>SENSITIVE</strong> on T1_STRUCTURAL axiom.</div>
</div>
<div class="acard t2">
  <div class="acard-top">
    <span class="apill t2">T2 Transcriptomics</span>
    <span class="vpill S">SENSITIVE</span>
    <span class="acard-gcs">GCS 0.960</span>
    <span class="acard-data">ERBB2 z=3.097 &middot; TPM=1419</span>
  </div>
  <div class="acard-reasoning">ERBB2 TPM=1419 &mdash; constitutively high expression of the primary drug target. Type A (overexpression-driven) protocol: &gt;50 TPM with z&gt;2 overrides neutral z-score. Voting <strong>SENSITIVE</strong>.</div>
</div>
<div class="acard t3">
  <div class="acard-top">
    <span class="apill t3">T3 Pathway</span>
    <span class="vpill S">SENSITIVE</span>
    <span class="acard-gcs">GCS 0.645</span>
    <span class="acard-data">ErbB active &middot; no bypass</span>
  </div>
  <div class="acard-reasoning">EGFR/ERBB2 present in active ErbB signalling and PI3K-Akt pathways. Bypass check: no activating bypass routes for ERBB2 targets in KEGG data. Active pathway + no escape mechanism &rarr; Voting <strong>SENSITIVE</strong>.</div>
</div>
<div class="acard t4">
  <div class="acard-top">
    <span class="apill t4">T4 Pharmacology</span>
    <span class="vpill S">SENSITIVE</span>
    <span class="acard-gcs">GCS 0.853</span>
    <span class="acard-data">ln_ic50=0.108 &middot; z=&minus;1.967</span>
  </div>
  <div class="acard-reasoning">z-score = &minus;1.967 (z &lt; &minus;0.5 = SENSITIVE). AUC=0.636 confirms strong absolute dose-response. No contradictions in pharmacology data. Voting <strong>SENSITIVE</strong>.</div>
</div>
"""

T1_OUTCOME = """
<div class="fbox correct">
  <div class="fbox-title correct">R1 CONSENSUS: SENSITIVE &#10003;</div>
  <div class="fbox-line">All 4 agents agree in Round 1 &middot; no debate required</div>
  <div class="fbox-line">Winning agent: <strong>T2 Transcriptomics</strong> (highest GCS = 0.960)</div>
  <div class="fbox-conf correct">0.960</div>
  <div class="fbox-line">&middot; Ground truth: SENSITIVE &#10003;</div>
  <div class="fbox-insight">
    <div class="fbox-insight-head">Key insight:</div>
    <div class="fbox-insight-text">Concordant evidence across all four modalities &mdash; genomic amplification, transcriptional overexpression, pharmacological sensitivity, and active pathway &mdash; produces the highest-confidence correct prediction in the dataset. No resolver required.</div>
  </div>
</div>
"""

# ---------------------------------------------------------------------------
# Trace 2 — NCI-H1573 + Erlotinib (minority T3 wins via AxiomResolver)
# ---------------------------------------------------------------------------
T2_AGENTS = """
<div class="acard t1">
  <div class="acard-top">
    <span class="apill t1">T1 Genomics</span>
    <span class="vpill S">SENSITIVE</span>
    <span class="acard-gcs">GCS 0.700</span>
    <span class="acard-data">EGFR WT &middot; RPPA=2.52</span>
  </div>
  <div class="acard-reasoning">EGFR is wild-type with no structural alterations. RPPA shows high protein expression (2.52). In the absence of a driver mutation, high protein abundance is used as primary evidence. Voting <strong>SENSITIVE</strong> with lower confidence.</div>
</div>
<div class="acard t2">
  <div class="acard-top">
    <span class="apill t2">T2 Transcriptomics</span>
    <span class="vpill U">UNCERTAIN</span>
    <span class="acard-gcs">GCS 0.500</span>
    <span class="acard-data">EGFR 235 TPM &middot; z=1.84</span>
  </div>
  <div class="acard-reasoning">Erlotinib is a mutation-driven (Type B) drug. Sensitivity is determined by activating mutations (exon 19 del / L858R), not by overexpression. Without confirmed mutation status, predicting sensitivity is speculative. Voting <strong>UNCERTAIN</strong>.</div>
</div>
<div class="acard t3 winner">
  <div class="acard-top">
    <span class="apill t3">T3 Pathway</span>
    <span class="vpill R">RESISTANT</span>
    <span class="acard-gcs">GCS 0.920</span>
    <span class="acard-data">ERBB2/IGF1R/MET bypass detected</span>
  </div>
  <div class="acard-reasoning">EGFR in active MAPK, PI3K-Akt, ErbB, and EGFR-TKI resistance pathways (activity &ge;0.25). Bypass genes ERBB2, IGF1R, and MET confirmed in bypass registry as established escape routes for EGFR TKIs. Voting <strong>RESISTANT</strong>.</div>
</div>
<div class="acard t4">
  <div class="acard-top">
    <span class="apill t4">T4 Pharmacology</span>
    <span class="vpill S">SENSITIVE</span>
    <span class="acard-gcs">GCS 0.818</span>
    <span class="acard-data">ln_ic50=1.257 &middot; z=&minus;1.130</span>
  </div>
  <div class="acard-reasoning">IC50 z-score = &minus;1.130 (z &lt; &minus;0.5 = SENSITIVE). However AUC=0.903 reveals poor absolute dose-response &mdash; a conflict between relative and absolute metrics. Confidence capped. Voting <strong>SENSITIVE</strong>.</div>
</div>
"""

T2_OUTCOME = """
<div class="fbox correct">
  <div class="fbox-title correct">AXIOM RESOLVER: RESISTANT &#10003;</div>
  <div class="fbox-line">T1+T4 vote SENSITIVE &middot; T3 votes RESISTANT &middot; T2 abstains</div>
  <div class="fbox-line">AxiomResolver: T3_PATHWAY tier &gt; T4_PHARMACOLOGICAL tier</div>
  <div class="fbox-line">Winning agent: <strong>T3 Pathway</strong> (GCS = 0.920)</div>
  <div class="fbox-conf correct">0.920</div>
  <div class="fbox-line">&middot; Ground truth: RESISTANT &#10003;</div>
  <div class="fbox-insight">
    <div class="fbox-insight-head">Key insight:</div>
    <div class="fbox-insight-text">Bypass pathway genes (ERBB2, IGF1R, MET) expose a resistance mechanism that neither IC50 data nor protein expression can detect. The axiom hierarchy correctly elevates the pathway signal over the pharmacological and genomic priors.</div>
  </div>
</div>
"""

# ---------------------------------------------------------------------------
# Trace 3 — MDA-MB-231 + Olaparib (R2 consensus, T4 drives)
# ---------------------------------------------------------------------------
T3_AGENTS = """
<div class="acard t1">
  <div class="acard-top">
    <span class="apill t1">T1 Genomics</span>
    <span class="vpill U">UNCERTAIN</span>
    <span class="acard-gcs">GCS 0.500</span>
  </div>
  <div class="acard-reasoning">PARP1/PARP2 hemizygous deletion only &mdash; not homozygous. No evidence of HRD status from genomic data alone. Olaparib sensitivity requires BRCA1/2-mutant context. Voting <strong>UNCERTAIN</strong>.</div>
</div>
<div class="acard t2">
  <div class="acard-top">
    <span class="apill t2">T2 Transcriptomics</span>
    <span class="vpill U">UNCERTAIN</span>
    <span class="acard-gcs">GCS 0.500</span>
    <span class="acard-data">PARP1 78 TPM</span>
  </div>
  <div class="acard-reasoning">Olaparib is a Type C (synthetic lethality) drug. T2 gate abstains &mdash; PARP expression level is not the mechanistic driver; efficacy depends on HRD context. Voting <strong>UNCERTAIN</strong>.</div>
</div>
<div class="acard t3">
  <div class="acard-top">
    <span class="apill t3">T3 Pathway</span>
    <span class="vpill S">SENSITIVE</span>
    <span class="acard-gcs">GCS 0.645</span>
    <span class="acard-data">BER active &middot; no bypass in R1</span>
  </div>
  <div class="acard-reasoning">PARP1/PARP2 in active Base Excision Repair pathway (score &gt;0.25). No bypass routes detected in R1. Votes <strong>SENSITIVE</strong> &mdash; confidence weakens in Round 2 after peer critique flags absent HRD evidence.</div>
</div>
<div class="acard t4 winner">
  <div class="acard-top">
    <span class="apill t4">T4 Pharmacology</span>
    <span class="vpill R">RESISTANT</span>
    <span class="acard-gcs">GCS 0.387</span>
    <span class="acard-data">AUC=0.967 &middot; z=+0.125</span>
  </div>
  <div class="acard-reasoning">AUC=0.967 (very high = poor absolute dose-response). z-score borderline, but high AUC dominates: the cell line does not respond to PARP inhibition in absolute terms. Voting <strong>RESISTANT</strong>.</div>
</div>
"""

T3_OUTCOME = """
<div class="fbox correct">
  <div class="fbox-title correct">R2 CONSENSUS: RESISTANT &#10003;</div>
  <div class="fbox-line">T1 + T2 abstain &middot; T3 loses confidence after peer critique in R2</div>
  <div class="fbox-line">Peer critique: absent HRD evidence undermines T3 SENSITIVE signal</div>
  <div class="fbox-line">Winning agent: <strong>T4 Pharmacology</strong> (IC50 evidence)</div>
  <div class="fbox-conf correct">0.390</div>
  <div class="fbox-line">&middot; Ground truth: RESISTANT &#10003;</div>
  <div class="fbox-insight">
    <div class="fbox-insight-head">Key insight:</div>
    <div class="fbox-insight-text">When genomic and transcriptomic agents abstain and pathway evidence lacks HRD confirmation, the pharmacological prior (high AUC = poor absolute response) is sufficient to drive Round 2 consensus. Debate resolves the initial ambiguous pathway signal.</div>
  </div>
</div>
"""


def render(name: str, agents_html: str, outcome_html: str) -> None:
    html = html_trace(name, agents_html, outcome_html)
    html_path = TMP_DIR / f'{name}.html'
    html_path.write_text(html, encoding='utf-8')
    out_path = (OUT_DIR / f'{name}.png').resolve()
    cmd = [
        CHROME,
        '--headless=new',
        '--disable-gpu',
        '--no-sandbox',
        f'--screenshot={out_path}',
        '--window-size=1200,720',
        '--hide-scrollbars',
        f'file://{html_path.resolve()}',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  ERROR: {result.stderr[:300]}')
    else:
        print(f'  Saved {out_path}')


if __name__ == '__main__':
    render('trace1', T1_AGENTS, T1_OUTCOME)
    render('trace2', T2_AGENTS, T2_OUTCOME)
    render('trace3', T3_AGENTS, T3_OUTCOME)
    print('Done.')
