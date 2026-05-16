# NS-MCA: Neuro-Symbolic Meta-Cognitive Architecture for Clinical AI Safety

## Complete 6-Layer Architecture Specification

This document provides the formal mathematical and architectural specification
of the NS-MCA system.

## Data Preparation and Validation Pipeline

### Notebook 00: Data Preparation
Extract MedQA-USMLE dataset from ZIP file, load raw data, and prepare for analysis.
- Input: `data_clean.zip` (125.6 MB, 310 files)
- Process: Extract US USMLE questions, validate schema, stratify by specialty
- Output: `medqa_raw.json` (13.4 MB, 12,723 cases) saved to Google Drive

### Notebook 00B: Exploratory Data Analysis (EDA)
Comprehensive validation of dataset quality, integrity, and model compatibility before Layer 1 implementation.

**Data Quality Validation**
- Schema compliance: 100% (all fields present and correctly typed)
- Missing values: 0% (complete dataset)
- Data leakage: 0% (no questions appear in multiple splits)
- Duplicate questions: 4 (0.03%, legitimate clinical variants - kept as-is)
- **Overall Integrity Score: 90%** ✓

**Dataset Characteristics**
- Total cases: 12,723 clinical multiple-choice questions
- Split distribution: Train 80% (10,178) | Dev 10% (1,272) | Test 10% (1,273)
- Specialty distribution: General 40.4% | Pharmacology 37.1% | Surgery 12.5% | Pediatrics 10%
- Question complexity: Mean 726 chars (range 66-3,577), 96.4% with clinical depth (>300 chars)

**NLP & Tokenization Analysis**
- Entity extraction (Hybrid NER): 92.3% F1 score across 100-question sample
- Flan-T5 tokenization: Mean 177.8 tokens per question (100% fit within 512 limit, zero truncation)
- Medical vocabulary coverage: 43% single-token (acceptable - Flan-T5 handles subword tokenization naturally)
- Semantic preservation: >98% of content retained

**Policy Annotation Planning**
- Coverage: 27% of dataset (3,434 cases) to be manually annotated
- Stratification: Maintains split (80/10/10) and specialty ratios
- Constraint types: 4 categories (Contraindication, Dose Limit, Required Procedure, Mandatory Check)
- Quality assurance: 2+ annotators per case, ≥85% inter-annotator agreement target
- Timeline: 2-3 weeks with distributed team effort

**Output Files**
- `00B_01_split_distribution.png` - Train/dev/test split visualization
- `00B_02_question_analysis.png` - Question length distribution
- `00B_03_specialty_analysis.png` - Specialty analysis with adaptive threshold planning

**Readiness Assessment**
| Component | Score | Status |
|-----------|-------|--------|
| Data Integrity | 90% | ✓ Acceptable |
| Entity Extraction | 92.3% F1 | ✓ Good |
| Tokenization Fit | 100% | ✓ Perfect |
| Layer 2 (NER) Readiness | 80% | ✓ Ready with monitoring |
| Annotation Planning | 100% | ✓ Ready to commence |

**Validation Complete** - Ready for Notebook 01: Layer 1 (Neural Inference)


### Layer 1: Neural Inference + Calibration
- Model: Flan-T5-Large (12 encoder, 12 decoder layers, 1024 hidden)
- Input: Medical question + specialty
- Output: Generated response + calibrated confidence
- Equation: conf(y) = (1/n) Σ log P(w_i | w_{<i}, X; T*)

### Layer 2: Semantic Bridge (Hybrid NER)
- Methods: Keyword matching (92.1% F1) + Regex (93.8% F1) + ScispaCy (90% F1)
- Entity types: DRUG, DOSE, FREQUENCY, ROUTE, DURATION, INDICATION, ALLERGY
- Output: Structured entities with confidence scores

### Layer 3: Symbolic Mapping
- Input: Entities from Layer 2
- Process: Deterministic entity→predicate conversion
- Output: Formal logic predicates ready for verification

### Layer 4: Deterministic Policy Auditor
- Equation: G(y,s) = 𝟙[conf(y) ≥ τ_s] × ∏_{p∈P} ¬Violation(y,p)
- Adaptive thresholds: τ_surgery=0.85, τ_pharma=0.75, τ_general=0.65, τ_pediatrics=0.70
- Output: Safety decision (TRUE/FALSE) + violations detected

### Layer 5: Meta-Cognitive Governor
- Max iterations: 2 (guaranteed termination)
- Process: Constraint augmentation → regenerate → re-audit
- Output: Recovered response OR escalation flag

### Layer 6: Escalation Protocol
- Decision tree: CRITICAL (no output) vs HIGH (caution) vs MEDIUM (proceed)
- Output: Final recommendation + audit trail JSON

## Reproducibility
- All random seeds fixed (numpy, torch, random)
- All intermediate outputs saved as JSON
- All configuration in config/ directory
- All results logged with timestamps
- Full audit trails for every case
