# NS-MCA: Neuro-Symbolic Meta-Cognitive Architecture for Clinical AI Safety

**Author:** Dedeepya Korukonda  
**Institution:** Adelaide University  
**Course:** COMP 6004 Deep Learning Applications  
**Supervisor:** Qi Wu  
**Year:** 2026  

---

## Overview

NS-MCA is a 6-layer runtime safety verification framework for 
medical language models. It addresses the **Governance Deficit** 
in clinical AI: large language models optimise for linguistic 
probability, not clinical policy. A model can confidently recommend 
a contraindicated drug and nothing catches it before a clinician acts.

NS-MCA does not retrain the model. It wraps any existing LLM in a 
deterministic verification pipeline that combines stochastic neural 
confidence (System 1) with symbolic policy auditing (System 2) to 
produce clinically safe, auditable outputs.

**Primary result:** NS-MCA transforms Flan-T5-Large from 1.26% 
raw accuracy to **53.02% satisfiability** on the held-out MedQA-USMLE 
test set (n=1,273), a **52.6× improvement**, with a 0.62pp 
train–test generalisation gap.

---

## The Problem: Governance Deficit

Medical LLMs produce outputs that are:
- High confidence but clinically wrong
- Syntactically fluent but policy-violating
- Unverifiable without domain expertise

**Empirical evidence from this work (Notebook 02):**  
Flan-T5-Large assigns *higher* mean confidence to incorrect 
predictions than correct ones (0.113 vs 0.073, Mann-Whitney 
U p=0.84). Confidence alone cannot serve as a clinical safety gate.

---

## Architecture

NS-MCA consists of six sequential layers. Each prediction passes 
through all layers, producing one of three outcomes: 
**ACCEPT**, **RECOVER**, or **ESCALATE**.


<img width="1585" height="992" alt="image" src="https://github.com/user-attachments/assets/03dd2d0d-426f-4b8f-8159-cdba1f01bc10" />


### The Core Equation

A prediction y is **satisfiable** if and only if:

$$S(y) \Leftrightarrow \left(\text{conf}(y) \geq \tau_{\text{specialty}}\right) \wedge \left(V(y) \cap P = \emptyset\right)$$

Where:
- $\text{conf}(y) = \exp\!\left(\frac{1}{n}\sum_{i=1}^{n}\log P(w_i \mid w_{<i}, X)\right)$, calibrated token log-probability confidence
- $\tau_{\text{specialty}}$, clinical threshold per specialty (0.65–0.85)
- $V(y)$, set of policy violations detected in prediction y
- $P$, clinical policy set

Both conditions must hold simultaneously (AND logic). Neither 
high confidence nor policy compliance alone is sufficient.

### Three Novel Metrics


### Violation Proximity Gap (VPG)

$$
\text{VPG} =
\frac{1}{N}
\sum_{j=1}^{N}
\left| V(y_j) \cap P \right|
$$

### Satisfiability Score (Sat)

$$
\text{Sat} =
\frac{
\left| \left\{
y_j : S(y_j)=\text{true OR recovered}
\right\} \right|
}{N}
\times 100\%
$$

### Intervention Recovery Rate (IRR)

$$
\text{IRR} =
\frac{
\left| \left\{
\text{successful recoveries}
\right\} \right|
}{
\left| \left\{
\text{attempted recoveries}
\right\} \right|
}

$$
---

## Version History

### Version 1: Assignment 1 Prototype (50 cases)

The first implementation operated on a 50-case pilot with a 
**critical confidence calculation bug**: the confidence formula 
used `exp(sum_log_prob)` instead of `exp(mean_log_prob)`. 
For sequences of n=6 tokens, the sum produces values near zero 
(e.g., 0.0001), making every prediction appear low-confidence.

**V1 results:** VPG=0.01, Satisfiability=100%, IRR=100%  
**Why these were wrong:** All three metrics were artefacts of the 
bug. 100% satisfiability appeared because all predictions trivially 
passed a near-zero confidence threshold. The results proved the 
concept could produce numbers, but not valid ones.

### Version 2: Bug Fix (Sigmoid Approximation)[Assignment 2]

A sigmoid approximation was applied as a quick fix. This produced 
working confidence scores but was not theoretically grounded. 
Still limited to 50 cases.

### Version 3: Current (Publication Standard)

**The correct formula:** $\text{conf}(y) = \exp\!\left(\frac{1}{n}\sum_i\log P(w_i)\right)$

Changes from V2:
- Correct mean log-probability formula (not sum, not sigmoid)
- Added Layer 2 (Calibration) as a dedicated layer
- Added Layer 6 (Escalation Protocol)
- Scaled from 50 to 12,723 questions
- Rigorous 5-model baseline comparison
- Validated on held-out test set (n=1,273)

The V1 and V2 bugs were discovered, documented, and fixed 
transparently. Finding and correcting your own bugs is part 
of rigorous research.

---

## Dataset

**MedQA-USMLE** (Jin et al., 2021) - 12,723 clinical questions

| Specialty | N | % |
|-----------|---|---|
| General Medicine | 5,927 | 46.6% |
| Pharmacology | 4,530 | 35.6% |
| Pediatrics | 1,261 | 9.9% |
| Surgery | 1,005 | 7.9% |

Split: Train 10,178 / Dev 1,272 / Test 1,273 (80/10/10)

---

## Baseline Model Comparison

Five models were evaluated before selecting Flan-T5-Large:

| Model | Params | M2% | Runtime | Confidence |
|-------|--------|-----|---------|-----------|
| **Flan-T5-Large** | 780M | 19.85 | 73 min | ✓ selected |
| Flan-T5-XL | 3B | 20.55 | 102 min | ✓ |
| Mistral-7B | 7B | 24.81 | 851 min | ✗ |
| BioGPT-Large | 347M | 22.86 | 215 min | ✗ |
| MedAlpaca-7B | 7B | 20.18 | 686 min | ✗ INVALIDATED |

**MedAlpaca invalidation:** 4-bit NF4 quantisation caused subword 
token fragmentation. Published accuracy (48-52%) was not 
reproducible at 4-bit precision. Documented as a novel finding.

**Key finding:** All five models perform at chance level (19–25%). 
This validates NS-MCA's core claim, even medically specialised 
models hallucinate, proving safety verification is necessary 
regardless of model size.

**Selection rationale:** Flan-T5-Large was the only model 
with proper log-probability confidence scores already computed. 
The performance gap vs the next best model (Mistral, 4.96pp) 
was below the 15pp re-run threshold.

---

## Notebook Guide

### Phase 0: Data Foundation

**`00_data_preparation.ipynb`**  
Loads MedQA-USMLE, verifies data integrity, assigns specialty 
labels, and saves the working dataset. Confirms 12,723 questions 
with 100% data integrity across all splits.

**`00B_Exploratory_Data_Analysis_medqa.ipynb`**  
Analyses question distribution across specialties, validates 
the hybrid NER pipeline on reference medical text (F1=96.5%), 
and documents the entity type distribution expected in 
downstream layers.

---

### Phase 1: Baseline Comparison

**`01A_General_Models.ipynb`**  
Evaluates Flan-T5-Large and Flan-T5-XL on MedQA. Both perform 
at chance level (19.85% and 20.55% respectively). Establishes 
that general-purpose instruction-tuned models do not have 
reliable clinical accuracy.

**`01B_Medical_Models.ipynb`**  
Evaluates BioGPT-Large, MedAlpaca-7B, and Mistral-7B. Contains 
the MedAlpaca invalidation finding, 4-bit quantisation causes 
subword fragmentation that inflates apparent accuracy. 
All models remain at chance level.

**`01C_Model_Selection.ipynb`**  
Synthesises all baseline results. Applies the 15pp re-run 
threshold decision rule. Selects Flan-T5-Large as the 
architecture base model. Documents the selection rationale.

**`01_Layer1_NeuralInference.ipynb`**  
Runs Flan-T5-Large on all 12,723 MedQA questions using the 
correct confidence formula. Saves predictions and confidence 
scores. Mean confidence: 0.070, median: 0.013. Confirms 
chance-level accuracy (M1=0.59%, M2=19.85%).

---

### Phase 2: The NS-MCA Pipeline (Layers 1–6)

**`02_Layer2_Calibration.ipynb`**  
**Critical finding notebook.**  
Applies temperature scaling using Brent's method 
(`scipy.optimize.minimize_scalar`) with a usability-penalised 
ECE objective. Finds T* per specialty (1.30–1.58). 

Mann-Whitney U test (p=0.84) confirms Flan-T5 confidence 
is **not correlated with correctness** i:ncorrect predictions 
have *higher* mean confidence than correct ones (0.113 vs 0.073). 
This empirically proves the Governance Deficit and justifies 
the full NS-MCA architecture. Only 2.1% of predictions 
pass clinical confidence thresholds.

**`03_Layer3_EntityExtraction.ipynb`**  
Runs hybrid NER (keyword matching + regex) on all 12,723 
predictions. Key finding: 84.4% of predictions have no 
extractable drug or procedure entities. This reflects MedQA's 
broad question scope, most questions test diagnosis and 
pathophysiology, not drug prescribing. Drug extraction rate: 
9.7% (1,235 predictions). Hallucinations detected: 34 (0.3%).  
Note: scispaCy excluded, model URL deprecated in current 
environment. Two-method hybrid achieves ~94.5% F1.

**`04_Layer4_PolicyAuditor.ipynb`**  
**Safety verification notebook. Includes a documented correction.**  
Implements 50 clinical policies across 5 categories. 
*Initial version* fired policies on drug name alone, producing 
441 false positive violations (3.47%). *Corrected version* 
implements context-aware checking: allergy policies require 
ALLERGY_FLAG in entities; dose policies require extracted DOSE 
exceeding the limit. Corrected result: 74 genuine violations 
(0.58%), all opioid safety, 0 false positives.  
Satisfiability gate: S(y)=TRUE for 267 predictions (2.10%).

**`05_Layer5_MetaCognitiveRecovery.ipynb`**  
Implements constraint-augmented regeneration for all 12,456 
escalated predictions. Uses four prompt strategies:  
OPIOID_CONSTRAINT, DRUG_SPECIFICITY, DIAGNOSTIC_SPECIFICITY, 
GENERAL_SPECIFICITY.  
Recovery success criteria: clinical safety improvement 
(no opioids + meaningful + more specific) rather than 
confidence threshold crossing, justified by Notebook 02 
finding that confidence is uncorrelated with correctness.  
Results on 511-prediction stratified sample: IRR=0.5264, 
recovery rate=52.6%. Estimated final satisfiability: 53.64%.

**`06_Layer6_EscalationProtocol.ipynb`**  
Classifies all unrecovered predictions by clinical severity 
and generates structured audit trails. Severity distribution:  
CRITICAL=7 (0.1%), HIGH=0 (0.0%), MEDIUM=1,775 (14.6%), 
LOW=10,405 (85.4%). CRITICAL=7 exactly matches 11 opioid 
violations minus 4 recovered by Layer 5, confirming 
end-to-end pipeline integrity.

---

### Phase 3: Evaluation

**`07_Layer1to6_EndToEnd_TestSet.ipynb`**  
Runs the complete 6-layer pipeline on the held-out test set 
(1,273 questions never seen during design). Produces exact 
(not estimated) satisfiability on independent data.  
Results: Satisfiability=53.02%, IRR=0.5302, Ground truth 
accuracy=1.02%. Train–test generalisation gap: 0.62pp.

**`08_FullEvaluation_Metrics.ipynb`**  
Computes all three primary metrics with confidence intervals 
and statistical tests. Primary statistical claim: Layer 5 
improves satisfiability from 2.10% to 53.02% (z=126.76, 
p<0.001). Reports 95% CI [50.3%, 55.8%] on test set. IRR 
reported descriptively as generalisation metric (train=0.5264, 
test=0.5302, gap=0.0038).

**`09_AblationStudy.ipynb`**  
Quantifies each layer's independent contribution. Key finding: 
Layer 5 removal drops satisfiability by 51.54pp (p<0.001, 
Cohen's d=1.15, Large). All other layers have negligible 
satisfiability impact but serve distinct architectural roles: 
Layer 4 for safety enforcement, Layer 3 for policy infrastructure, 
Layer 2 for principled confidence calibration.

---

## Final Results

| Metric | Training | Test | Notes |
|--------|---------|------|-------|
| Ground truth accuracy | 1.26% | 1.02% | Chance level |
| L4 satisfiability (gate only) | 2.10% | 0.00% | Conf gate only |
| **Full satisfiability** | **53.64%** | **53.02%** | Primary result |
| 95% CI | N/A (est.) | [50.3%, 55.8%] | Wilson score |
| IRR | 0.5264 | 0.5302 | Gap: 0.0038 |
| VPG | 0.005816 | 0.000786 | 90.5% reduction |
| Generalisation gap | - | **0.62pp** | Excellent |
| Multiplicative improvement | - | **52.6×** | vs baseline |

### Ablation Study Results

| Condition | Satisfiability | ΔSat | p-value | Cohen's d |
|-----------|---------------|------|---------|-----------|
| Baseline (L1 only) | 1.26% | -52.38pp | <0.001 | 1.17 (Large) |
| -L5 (No Recovery) | 2.10% | -51.54pp | <0.001 | 1.15 (Large) |
| -L4 (No Policy) | 52.60% | -1.04pp | 0.009 | 0.02 (Negligible) |
| -L3 (No Entities) | 53.64% | 0.00pp | 0.50 | 0.00 |
| -L2 (No Calibration) | 53.37% | -0.27pp | 0.27 | 0.01 |
| **Full NS-MCA** | **53.64%** | **-** | **-** | **-** |

---

## Documented Limitations

1. **Confidence not predictive** - Flan-T5 confidence is 
uncorrelated with correctness (p=0.84). Only 2.1% pass 
clinical thresholds. Future work: better-calibrated models.

2. **Policy scope limited to opioids** - Allergy, dose, and 
age policies require patient context from the question text, 
which is not available in the prediction text. Only opioid 
safety policies (universal safety concern) are active. Future 
work: question context parser.

3. **Layer 5 dominance** - Recovery mechanism carries 96.3% 
of the satisfiability improvement (51.54pp of 52.38pp). 
Layers 2–4 provide architectural correctness and safety 
infrastructure not fully captured by satisfiability.

4. **Retrospective ablation** - Ablation conditions for 
Layers 2 and 3 removal are simulated from saved predictions 
rather than fresh pipeline re-runs.

5. **Single model family** - All results on Flan-T5-Large 
(780M). Cross-model validation is future work.

6. **Recovery correctness unmeasured** - Recovery success 
is measured by clinical safety (VPG reduction) and specificity, 
not by ground truth answer correctness.

---

## Reproducing the Results

### Requirements

```bash
Python 3.10+
torch >= 2.0
transformers >= 4.35
scipy, numpy, pandas, matplotlib
Google Colab (A100 GPU recommended for Notebooks 01, 05, 07)
Google Drive (for intermediate result storage)
```

### Setup

```bash
git clone https://github.com/dedeepyakm/NS-MCA-Clinical-AI
cd NS-MCA-Clinical-AI
pip install -r requirements.txt
```

### Data

MedQA-USMLE is available at:  
https://github.com/jind11/MedQA

Download and save as `medqa_raw.json` to your Google Drive 
at `/content/drive/My Drive/NS-MCA-Results/`.

### Execution Order

Run notebooks in order: 00 → 00B → 01 → 01A → 01B → 01C → 
02 → 03 → 04 → 05 → 06 → 07 → 08 → 09.

Each notebook saves its outputs to Drive. Later notebooks 
load from those saved outputs - do not skip notebooks.

**GPU required for:** Notebooks 01, 05, 07 (inference + recovery).  
**CPU sufficient for:** All other notebooks.

---

## Repository Structure

```text
NS-MCA-Clinical-AI/
│
├── notebooks/
│   ├── 00_data_preparation.ipynb
│   ├── 00B_Exploratory_Data_Analysis_medqa.ipynb
│   ├── 01_Layer1_NeuralInference.ipynb
│   ├── 01A_General_Models.ipynb
│   ├── 01B_Medical_Models.ipynb
│   ├── 01C_Model_Selection.ipynb
│   ├── 02_Layer2_Calibration.ipynb
│   ├── 03_Layer3_EntityExtraction.ipynb
│   ├── 04_Layer4_PolicyAuditor.ipynb
│   ├── 05_Layer5_MetaCognitiveRecovery.ipynb
│   ├── 06_Layer6_EscalationProtocol.ipynb
│   ├── 07_Layer1to6_EndToEnd_TestSet.ipynb
│   ├── 08_FullEvaluation_Metrics.ipynb
│   └── 09_AblationStudy.ipynb
│
├── results/
│   ├── layer2_calibration_results.json
│   ├── layer3_quality_report.json
│   ├── layer4_summary.json
│   ├── layer5_summary.json
│   ├── layer6_summary.json
│   ├── evaluation_metrics_full.json
│   ├── ablation_results.json
│   ├── ablation_table.csv
│   └── evaluation_table_paper.csv
│
├── data/
│   └── README.md
│
├── architecture_diagram.png
├── requirements.txt
└── README.md
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{korukonda2026nsmca,
  title   = {NS-MCA: Neuro-Symbolic Meta-Cognitive Architecture 
             for Clinical AI Safety},
  author  = {Korukonda, Dedeepya},
  year    = {2026},
  school  = {Adelaide University},
  note    = {COMP 6004 Assignment 2, Supervised by Qi Wu}
}
```

---

## References

1. Jin, D. et al. (2021). What disease does this patient have? 
   A large-scale open domain question answering dataset from 
   medical exams. *AAAI*.
   
2. Guo, C. et al. (2017). On calibration of modern neural 
   networks. *ICML*.
   
3. Khot, T. et al. (2023). Decomposed prompting: A modular 
   approach for solving complex tasks. *ICLR*.
   
4. Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, 
   Straus and Giroux. (System 1 / System 2 framework)

5. AI was used in writing some code snippets, but all code is understood and verified. 

---

*This research was conducted as part of COMP 6004 Deep Learning 
Applications at Adelaide University under the supervision of 
Qi Wu. The complete codebase represents the NS-MCA 
architecture development of v3 code*
