# MedQA Dataset Documentation

## Source
- **Original**: Jin et al., 2021 - MedQA Dataset
- **GitHub**: https://github.com/jind11/MedQA
- **Paper**: "What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams"

## Dataset Used
- **Region**: US USMLE (United States Medical Licensing Examination)
- **Format**: JSONL (train, dev, test splits)
- **Total Cases**: 1,273 (approximately)

## File Structure
- train.jsonl: Training set (~1,000 cases)
- dev.jsonl: Validation set (~100 cases)
- test.jsonl: Test set (~100 cases)

## Fields per Case
- question: Medical question text
- options: Multiple choice options
- answer: Correct answer
- answer_idx: Index of correct answer
- specialty: Medical specialty (e.g., surgery, pharmacology, general)
- split: train/dev/test

## Citation
If using this dataset, cite:
Jin, Di, et al. "What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams." 
arXiv preprint arXiv:2009.13081 (2020).
