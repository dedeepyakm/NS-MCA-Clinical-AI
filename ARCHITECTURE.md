# NS-MCA: Neuro-Symbolic Meta-Cognitive Architecture for Clinical AI Safety

## Complete 6-Layer Architecture Specification

This document provides the formal mathematical and architectural specification
of the NS-MCA system.

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
