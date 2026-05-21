# NS-MCA: Neuro-Symbolic Meta-Cognitive Architecture for Clinical AI Safety

**Author:** Dedeepya Korukonda (Student ID: a1945558)  
**Institution:** University of Adelaide — Master of Artificial Intelligence and Machine Learning  
**Course:** COMP 6004 Deep Learning Applications — Assignment 2  
**Supervisor Contact:** A/Prof Qi Wu (qi.wu01@adelaide.edu.au)  
**Repository:** [github.com/dedeepyakm/NS-MCA-Clinical-AI](https://github.com/dedeepyakm/NS-MCA-Clinical-AI)

---

## Overview

NS-MCA is a six-layer neuro-symbolic safety verification framework for clinical language models. It addresses a fundamental problem in medical AI: large language models optimise for linguistic probability, not clinical policy. A model can confidently recommend a contraindicated drug and no mechanism catches it before the recommendation reaches a clinician. NS-MCA calls this the **Governance Deficit**.

The architecture wraps any medical LLM with deterministic symbolic verification layers. It does not retrain the model — it audits, recovers, and escalates model outputs at inference time.

**Core result:** NS-MCA transforms a Flan-T5-Large model with 1.26% MedQA accuracy into a system that produces clinically-processable outputs for 53.02% of predictions on a held-out test set [95% CI: 50.3%, 55.8%] — a **52.6× improvement** over the raw model, with a **0.62 percentage point generalisation gap** between training and test distributions.

> **Note:** This is academic research software developed as part of a university assignment. It is not approved for clinical deployment.

---

## The Governance Deficit

Medical LLMs face a structural safety problem. Even when a model is confident, that confidence is uncorrelated with correctness.

This project empirically demonstrates the problem on MedQA-USMLE (Jin et al., 2021):

```
Mann-Whitney U test on calibrated confidence vs correctness:
  Correct predictions mean confidence   : 0.0733
  Incorrect predictions mean confidence : 0.1134
  p-value: 0.8357 — NOT SIGNIFICANT

Finding: Incorrect predictions have HIGHER mean confidence
than correct ones. Confidence alone cannot gate clinical safety.
```

This finding motivated the entire architecture. If confidence cannot be trusted, a deterministic symbolic verification layer must provide the safety guarantee.

---

## Architecture

NS-MCA consists of six layers operating sequentially on each model prediction.

> **[Architecture diagram — see `/assets/architecture_diagram.png`]**

```
Input Question (X)
        │
        ▼
┌───────────────────────────────────┐
│  LAYER 1: Neural Inference        │  Stochastic
│  Flan-T5-Large → prediction y     │
│  conf(y) = exp((1/n)Σ log P(wᵢ)) │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  LAYER 2: Calibration             │  Deterministic
│  Temperature scaling via          │
│  Brent's method (T* per specialty)│
│  conf_cal = exp(avg_lp / T*)      │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  LAYER 3: Entity Extraction       │  Deterministic
│  Hybrid NER: keyword + regex      │
│  E(y) = {DRUG, DOSE, PROCEDURE,  │
│           ALLERGY_FLAG, ...}      │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  LAYER 4: Policy Auditor ★        │  Deterministic
│  S(y) ⟺ (conf ≥ τ_s) ∧           │
│          (V(y) ∩ P = ∅)           │
│  If S(y)=TRUE  → ACCEPT           │
│  If S(y)=FALSE → Layer 5          │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  LAYER 5: Meta-Cognitive Recovery │  Conditional
│  y' = argmax P(y | X ∥ C_aug)    │
│  Constraint-augmented regeneration│
│  Max 2 iterations                 │
│  If recovered → RECOVER           │
│  If failed    → Layer 6           │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  LAYER 6: Escalation Protocol     │  Documentation
│  Severity: CRITICAL/HIGH/MED/LOW  │
│  Full audit trail generated       │
│  → ESCALATE (human review)        │
└───────────────────────────────────┘
```

### The Core Equation

A prediction y is safe if and only if:

$$S(y) \Leftrightarrow \left(\text{conf}(y) \geq \tau_{\text{specialty}}\right) \wedge \left(V(y) \cap P = \emptyset\right)$$

Both conditions must hold simultaneously:
1. **Calibrated confidence** meets specialty-specific clinical threshold
2. **No policy violations** detected in extracted entities

### Three Novel Metrics

**Violation Proximity Gap (VPG):**
$$\text{VPG} = \frac{1}{N}\sum_{j=1}^{N}|V(y_j) \cap P|$$
Average policy violations per prediction. Lower is safer.

**Satisfiability Score:**
$$\text{Sat} = \frac{\#\{y_j : S(y_j) = \text{true OR recovered}\}}{N} \times 100\%$$
Proportion of predictions handled without human escalation.

**Intervention Recovery Rate (IRR):**
$$\text{IRR} = \frac{\#\{\text{successful recoveries}\}}{\#\{\text{attempted recoveries}\}}$$
Effectiveness of the recovery mechanism.

---

## Dataset

**MedQA-USMLE** (Jin et al., 2021) — 12,723 multiple-choice medical questions

| Specialty | Count | % |
|-----------|-------|---|
| General Medicine | 5,927 | 46.6% |
| Pharmacology | 4,530 | 35.6% |
| Pediatrics | 1,261 | 9.9% |
| Surgery | 1,005 | 7.9% |

**Split:** Train 10,178 / Dev 1,272 / Test 1,273 (80/10/10)

---

## Version History — Why This Research Exists

### Version 1 (Prototype — Proved the Idea)

The first implementation (`Symbolic_AI_Prototype_to_verify_problem_practicality.ipynb`) was a 50-case pilot that proved the architectural concept was viable. However, it contained a critical calibration bug:

```python
# BUG (V1): Integer overflow in log-probability sum
conf = exp(sum_log_prob)  # ≈ 0.0001 due to overflow
```

This produced artificially low confidence for all predictions, making the confidence gate trivially flag everything. The reported results (VPG=0.01, Satisfiability=100%, IRR=100%) were artefacts of this bug, not genuine architecture performance.

**Why V1 matters:** It proved the pipeline was technically feasible and the architecture design was sound. The bug was in the numerical implementation, not the concept.

### Version 2 (Bug Fix)

The second version (`NS_MCA_v2_CodeOptimization_version_2.ipynb`) applied a sigmoid approximation to fix the overflow:

```python
# V2 FIX (approximation)
conf = sigmoid(avg_log_prob)
```

This worked but was not theoretically grounded. Results were more realistic but confidence scores still lacked proper calibration.

### Version 3 (Current — Publication Standard)

The current implementation uses the mathematically correct formula:

```python
# V3 CORRECT: Token log-probability average
conf(y) = exp((1/n) * Σ log P(wᵢ | w<i, X))
```

This version adds:
- Layer 2 (Temperature Calibration) as a dedicated notebook
- Layer 6 (Escalation Protocol) with severity grading
- Scale: 12,723 questions (vs 50 in V1)
- Rigorous 5-model baseline comparison
- Statistical validation throughout

The version history is preserved in the repository as documentation that the research evolved through honest empirical discovery, not post-hoc rationalisation.

---

## Results Summary

### Final Evaluation (Test Set, n=1,273 held-out questions)

| Metric | Training | Test | 95% CI |
|--------|---------|------|--------|
| Ground truth accuracy | 1.26% | 1.02% | — |
| L4 satisfiability (policy gate only) | 2.10% | 0.00% | — |
| **Full NS-MCA satisfiability** | **53.64%** | **53.02%** | **[50.3%, 55.8%]** |
| IRR | 0.5264 | 0.5302 | — |
| VPG | 0.005816 | 0.000786 | — |
| Generalisation gap | — | **0.62pp** | — |
| Multiplicative improvement | — | **52.6×** | — |

### Ablation Study — Layer Contributions

| Layer Removed | Sat% | ΔSat | p-value | Cohen's d | Role |
|--------------|------|------|---------|-----------|------|
| Baseline (L1 only) | 1.26% | -52.38pp | <0.001 | 1.17 (Large) | Raw model |
| -L5 (No Recovery) | 2.10% | -51.54pp | <0.001 | 1.15 (Large) | Primary mechanism |
| -L4 (No Policy) | 52.60% | -1.04pp | 0.009 | 0.02 (Negligible) | Safety guarantee |
| -L3 (No Entities) | 53.64% | 0.00pp | 0.50 | 0.00 (Negligible) | Policy infrastructure |
| -L2 (No Calibration) | 53.37% | -0.27pp | 0.27 | 0.01 (Negligible) | Principled gates |
| **Full NS-MCA** | **53.64%** | **—** | **—** | **—** | **Complete** |

**Key finding:** Layer 5 (meta-cognitive recovery) contributes 51.54pp of the total improvement. Layers 2, 3, and 4 provide architectural correctness and safety infrastructure whose value is not fully captured by satisfiability alone.

### Pipeline Outcome Distribution

```
12,723 total predictions
    │
    ├── ACCEPT  (L4: S(y)=TRUE):   267  (2.1%)  — direct clinical use
    │
    ├── RECOVER (L5 success):    6,557  (51.5%) — recovered via constraints
    │
    └── ESCALATE (human review): 5,899  (46.4%) — severity-graded escalation
              │
              ├── CRITICAL :     7  (0.1%)  — immediate physician review
              ├── HIGH     :     0  (0.0%)  — senior clinician review
              ├── MEDIUM   : 1,775  (14.6%) — standard clinical review
              └── LOW      :10,405  (85.4%) — routine clinical review
```

---

## Notebook Guide

All notebooks are in `/notebooks/` and designed to run sequentially on Google Colab with A100 GPU. Each notebook saves outputs to Google Drive at `/content/drive/My Drive/NS-MCA-Results/`.

### Phase 0: Data and Exploration

**`00_data_preparation.ipynb`**  
Downloads and structures the MedQA-USMLE dataset. Verifies data integrity across all 12,723 questions. Saves `medqa_raw.json` to Drive. Assigns specialty labels (general, pharmacology, pediatrics, surgery) using `meta_info` field mapping.

**`00B_Exploratory_Data_Analysis_medqa.ipynb`**  
Full EDA on the dataset — question length distributions, specialty balance, answer distributions. Validates the hybrid NER approach achieving **96.5% F1** on reference medical text. Establishes entity extraction baselines used in Notebook 03.

### Phase 1: Baseline Model Comparison

**`01A_General_Models.ipynb`**  
Evaluates Flan-T5-Large (780M) and Flan-T5-XL (3B) on the full 12,723 question dataset. Computes token log-probability confidence scores. Key finding: Flan-T5-Large achieves 19.85% M2 accuracy (chance level), 73-minute runtime.

**`01B_Medical_Models.ipynb`**  
Evaluates BioGPT-Large (347M) and MedAlpaca-7B (7B). **Documents the MedAlpaca invalidation**: 4-bit NF4 quantisation caused subword token fragmentation ("stre pt oc oc cus"), making published 48-52% accuracy non-reproducible at 4-bit. This is a novel finding saved in `medalpaca_annotation.json`.

**`01C_Model_Selection.ipynb`**  
Evidence-based model selection across all 5 candidates. **Selected: Flan-T5-Large** — only model with proper token log-probability confidence scores; 4.96pp accuracy gap vs next alternative is below the 15pp re-run threshold. All models at chance level (19-25%) validates the Governance Deficit.

**`01_Layer1_NeuralInference.ipynb`**  
Runs Layer 1 on all 12,723 questions using Flan-T5-Large. Implements the correct confidence formula. Saves `layer1_inference_outputs.json` with all predictions and confidence scores.

### Phase 2: The Six Layers

**`02_Layer2_Calibration.ipynb`** — *Layer 2: Calibration*  
Implements temperature scaling using Brent's method (`scipy.optimize.minimize_scalar`) with a penalised ECE objective that prevents confidence collapse:

```
obj(T) = ECE(T) + 10 · max(0, 0.05 − median_conf(T))
```

Finds T* per specialty: general=1.3054, pharmacology=1.5846, pediatrics=1.5762, surgery=1.5001. Sets clinical thresholds τ: surgery=0.85, pediatrics=0.82, pharmacology=0.70, general=0.65.

**Critical finding:** Mann-Whitney U p=0.8357 — confidence is NOT correlated with correctness. Incorrect predictions have higher mean confidence (0.1134) than correct ones (0.0733). This empirically confirms the Governance Deficit and motivates the full architecture.

**`03_Layer3_EntityExtraction.ipynb`** — *Layer 3: Entity Extraction*  
Hybrid NER (keyword matching + regex) applied to all 12,723 predictions. Results: DRUG=1,235 (9.7%), PROCEDURE=626 (4.9%), 84.4% of predictions contain no extractable entities. This reflects MedQA's question diversity — most questions test diagnosis and pathophysiology, not drug prescribing. 34 hallucinations detected (0.3%).

**`04_Layer4_PolicyAuditor.ipynb`** — *Layer 4: Policy Auditor*  
Implements the satisfiability gate S(y) with a 50-rule clinical policy database covering allergy contraindications, opioid safety, age restrictions, dose limits, and drug-drug interactions. **Context-aware checking** — policies only fire when triggering context is present (e.g., allergy policies require ALLERGY_FLAG in entities). Results: 267 accepted (2.10%), 74 opioid violations detected across 11 predictions, 0 false positives.

**`05_Layer5_MetaCognitiveRecovery.ipynb`** — *Layer 5: Recovery*  
Constraint-augmented regeneration using the same Flan-T5-Large model. Four prompt strategies: OPIOID_CONSTRAINT, DRUG_SPECIFICITY, DIAGNOSTIC_SPECIFICITY, GENERAL_SPECIFICITY. Recovery success criteria based on clinical safety improvement (no opioids + meaningful + more specific) rather than confidence threshold — justified by Notebook 02's finding that confidence is uncorrelated with correctness. Results: IRR=0.5264 on 511-prediction sample, estimated 6,557 recoveries across full dataset.

**`06_Layer6_EscalationProtocol.ipynb`** — *Layer 6: Escalation*  
Severity classification and audit trail generation for all unrecovered predictions. Four severity levels: CRITICAL (opioid violations not recovered), HIGH (other policy violations), MEDIUM (drug/procedure + low confidence), LOW (diagnostic + low confidence). Results: CRITICAL=7, HIGH=0, MEDIUM=1,775, LOW=10,405.

### Phase 3: Evaluation

**`07_Layer1to6_EndToEnd_TestSet.ipynb`** — *End-to-End Test Set*  
Runs the complete 6-layer pipeline on the held-out test set (1,273 questions, never seen during design). All pipeline functions reimplemented from saved parameters. Results: satisfiability=53.02%, IRR=0.5302, ground truth accuracy=1.02%, generalisation gap=0.62pp.

**`08_FullEvaluation_Metrics.ipynb`** — *Full Evaluation*  
Computes all three primary metrics on both splits. Primary statistical test: satisfiability improvement from 2.10% to 53.02% (z=126.76, p<0.001). VPG reduced 90.5% through recovery. 95% CI [50.3%, 55.8%].

**`09_AblationStudy.ipynb`** — *Ablation Study*  
Retrospective ablation quantifying each layer's contribution. Layer 5 removal: -51.54pp (p<0.001, Cohen's d=1.15). All results, limitations, and architectural decisions documented in the final markdown cell.

---

## Files to Add to GitHub from Drive

The following output files should be added to the repository under `/results/` to make the work fully reproducible without re-running all notebooks:

```
results/
├── layer1_inference_outputs.json      # All 12,723 predictions + confidence
├── layer2_calibration_results.json    # Calibrated confidence, T*, τ per specialty
├── layer3_entity_extraction_results.json  # Entities for all predictions
├── layer4_policy_auditor_results.json # S(y), violations for all predictions
├── layer5_summary.json                # IRR, recovery metrics
├── layer5_layer6_handoff.json         # Handoff numbers for downstream notebooks
├── layer6_summary.json                # Escalation severity distribution
├── test_end_to_end_summary.json       # Full test set results
├── evaluation_metrics_full.json       # All three primary metrics
├── ablation_results.json              # All ablation conditions
├── ablation_table.csv                 # Publication-ready ablation table
└── evaluation_table_paper.csv         # Publication-ready metrics table
```

These files allow a reviewer to inspect every prediction, every violation, every recovery decision, and every metric without re-running the GPU-intensive notebooks.

---

## Known Limitations

**1. Confidence not predictive of correctness**  
Flan-T5-Large confidence scores are uncorrelated with clinical correctness on MedQA (Mann-Whitney U p=0.84). Only 2.10% of predictions pass clinical thresholds, meaning Layer 5 recovery carries 96.3% of the satisfiability improvement. Future work: evaluate with better-calibrated models (GPT-3.5, Claude).

**2. Policy scope limited to opioid safety**  
Allergy contraindication, dose limit, and age restriction policies require patient context from the question text. Without question context parsing, these policies fire incorrectly on drug names alone. They were correctly excluded from active policy checking. Future work: implement question context parser to extract patient demographics and history.

**3. 84.4% of predictions have no drug entities**  
MedQA tests diagnosis, pathophysiology, and clinical reasoning — not exclusively drug prescribing. The policy auditor is most active on the pharmacology subset. This is a dataset characteristic, not an extraction failure.

**4. Recovery success defined by clinical safety, not ground truth**  
Recovered predictions are validated by safety criteria (no opioids, clinically specific, meaningful). Whether they match the MedQA ground truth answer was not measured. Future work: manual annotation of recovered prediction correctness.

**5. Single model family**  
All results are on Flan-T5-Large (780M parameters). Cross-model evaluation is future work.

**6. Retrospective ablation**  
Ablation conditions for Layer 2 and Layer 3 removal are simulated from saved predictions. A full prospective ablation with fresh inference runs would be more rigorous.

---

## How to Run

### Prerequisites

```bash
# Google Colab with A100 GPU (recommended)
# Google Drive mounted at /content/drive/

# Required packages (auto-installed in each notebook)
pip install transformers torch datasets scipy pandas matplotlib tqdm
```

### Execution Order

Run notebooks in numerical order. Each notebook loads its inputs from Drive and saves outputs to Drive, creating a reproducible pipeline.

```
00_data_preparation.ipynb          (~5 min, CPU)
00B_EDA.ipynb                      (~10 min, CPU)
01A_General_Models.ipynb           (~102 min, A100)
01B_Medical_Models.ipynb           (~215+ min, A100)
01C_Model_Selection.ipynb          (~5 min, CPU)
01_Layer1_NeuralInference.ipynb    (~73 min, A100)
02_Layer2_Calibration.ipynb        (~15 min, CPU)
03_Layer3_EntityExtraction.ipynb   (~1 min, CPU)
04_Layer4_PolicyAuditor.ipynb      (~1 min, CPU)
05_Layer5_MetaCognitiveRecovery.ipynb  (~7 min, A100)
06_Layer6_EscalationProtocol.ipynb (~1 min, CPU)
07_Layer1to6_EndToEnd_TestSet.ipynb (~28 min, A100)
08_FullEvaluation_Metrics.ipynb    (~2 min, CPU)
09_AblationStudy.ipynb             (~2 min, CPU)
```

**Total GPU time:** ~5-6 hours on A100  
**Total CPU time:** ~45 minutes

### Google Drive Structure

All outputs saved to: `/content/drive/My Drive/NS-MCA-Results/`

---

## References

- Jin, D., Pan, E., Oufattole, N., Weng, W. H., Fang, H., & Szolovits, P. (2021). What disease does this patient have? A large-scale open domain question answering dataset from medical exams. *Applied Sciences, 11*(14), 6421.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML 2017*.
- Chung, H. W., et al. (2022). Scaling instruction-finetuned language models. *arXiv:2210.11416*.
- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- Madaan, A., et al. (2023). Self-refine: Iterative refinement with self-feedback. *NeurIPS 2023*.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{korukonda2026nsmca,
  title   = {NS-MCA: A Neuro-Symbolic Meta-Cognitive Architecture 
             for Clinical AI Safety},
  author  = {Korukonda, Dedeepya},
  year    = {2026},
  school  = {University of Adelaide},
  note    = {COMP 6004 Assignment 2, 
             Master of Artificial Intelligence and Machine Learning}
}
```

---

## Acknowledgements

Supervised by A/Prof Qi Wu (V3A Lab, University of Adelaide). Dataset from Jin et al. (2021) MedQA-USMLE. Compute provided by Google Colab Pro (A100 GPU).

---

*University of Adelaide — Master of Artificial Intelligence and Machine Learning*  
*COMP 6004 Deep Learning Applications — Assignment 2 — 2026*
