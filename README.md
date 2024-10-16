# ASA Physical Status Classification Using NLP Models

This repository contains code and resources for the paper **"Comparison of NLP Machine Learning Models with Human Physicians for ASA Physical Status Classification"**, published in *npj Digital Medicine* (2024): https://doi.org/10.1038/s41746-024-01259-6. The study introduces natural language processing (NLP) models to classify ASA (American Society of Anesthesiologists) Physical Status (ASA-PS) using pre-anesthesia evaluation summaries and compares the performance with human physicians.

## Overview

ASA-PS classification is crucial for assessing patients' comorbidities prior to surgery, but it has traditionally suffered from inter-rater variability among healthcare professionals. This project aims to mitigate this inconsistency by automating ASA-PS classification using cutting-edge NLP models.

The study involved training and comparing three NLP models:
- **ClinicalBigBird**
- **BioClinicalBERT**
- **GPT-4**

ClinicalBigBird emerged as the best-performing model, surpassing both human anesthesiologists and other models in terms of specificity, precision, and F1-score.

## Features

- **NLP-based ASA-PS classification:** Uses pre-anesthesia free-text summaries to assign ASA-PS classes.
- **Model comparison:** Performance is evaluated against anesthesiology residents and board-certified anesthesiologists.
- **Shapley value-based interpretability:** The repository includes code for generating Shapley plots to visualize the contribution of each word in the text towards ASA-PS classification.

## Data

The dataset contains 717,389 surgical cases from a tertiary academic hospital, covering surgeries between October 2004 and May 2023. Data was split into training, tuning, and test sets, with tuning and test datasets labeled by a consensus of board-certified anesthesiologists.

## Models

1. **ClinicalBigBird:** Specialized for processing long clinical texts (up to 4096 tokens) and fine-tuned for ASA-PS classification. Achieved the best performance with AUROC scores above 0.91.
2. **BioClinicalBERT:** Biomedical-domain-specific BERT model, but limited to 512 tokens.
3. **GPT-4:** Demonstrated high accuracy in medical exams, but less optimized for ASA-PS classification.

## Citation

If you use this code or data, please cite the original paper:

```bibtex
@article{Yoon2024ASAPSClassification,
  title={Comparison of NLP Machine Learning Models with Human Physicians for ASA Physical Status Classification},
  author={Soo Bin Yoon, Jipyeong Lee, Hyung-Chul Lee, Chul-Woo Jung, Hyeonhoon Lee},
  journal={npj Digital Medicine},
  year={2024},
  doi={10.1038/s41746-024-01259-6}
}
