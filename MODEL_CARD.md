# Model Card — PneumoniaMNIST Classifier

This model card documents a binary classification model built as an engineering reference implementation. It serves strictly as a **demonstration of an ML training and evaluation pipeline, not a clinical diagnostic tool.**

---

## Model Details

**Task:** Binary image classification (0: Normal, 1: Pneumonia).
**Input Data:** 28×28 single-channel (grayscale) chest X-ray images from the PneumoniaMNIST sub-dataset of MedMNIST v2.
**Architecture:** A lightweight convolutional neural network (~105k parameters) consisting of two Conv2D/ReLU/MaxPool2D blocks followed by a two-layer fully connected classifier head with Dropout (p=0.25).
**Output Layer:** Two raw logits. Probabilities are derived via Softmax (P(pneumonia)). A hard classification label is assigned using a default decision threshold of τ = 0.5.
**Training Protocol:** Optimized using Adam. A weighted Cross-Entropy loss function was utilized to mitigate class imbalance. The optimal model checkpoint was selected based on maximizing validation Area Under the ROC Curve (AUC).

---

## Intended Use & Boundaries

**Intended Use:** Serves exclusively as a technical baseline to demonstrate end-to-end pipeline reproducibility, configuration capture, robust checkpointing/resumption, and evaluation hygiene on a public benchmark.
**Prohibited Use:** This model is **NOT** intended, validated, or certified for clinical diagnostics, screening, triage, or any form of automated decision support involving human patients. It has not undergone regulatory auditing (e.g., FDA 510(k) clearance) or clinical validation.

---

## Dataset Characteristics & Limitations

The PneumoniaMNIST dataset is heavily processed from its pediatric chest X-ray source, introducing several constraints that alter its behavior relative to real-world medical data:

**Extreme Downsampling (28×28):** Spatial reduction discards fine-grained diagnostic features (e.g., subtle focal opacities, air bronchograms) that a radiologist relies upon. The model is forced to learn coarse spatial heuristics rather than true clinical pathology.
**Cohort Specificity:** The source data is entirely pediatric and sourced from highly localized clinical sites. The model possesses no structural basis for generalizing to adult physiology, varying scanner modalities, or differing institutional capture protocols.
**Prevalence and Distribution Shift:** The training and validation splits exhibit a heavy historical bias (~74% pneumonia prevalence), whereas the held-out test split drops to ~62.5%. This controlled distribution shift explains why validation metrics overstate final test performance.

---

## Empirical Performance & Error Profiling

### Quantitative Evaluation (Held-out Test Split, n=624)
**Test AUC:** 0.932 *(Compared to a Validation AUC of ~0.993; the delta confirms the predicted impact of the train/test prevalence shift)*.
**Decision Threshold:** τ = 0.5

### Confusion Matrix
| | Predicted Normal | Predicted Pneumonia |
| :--- | :--- | :--- |
| **True Normal** | 162 | 72 |
| **True Pneumonia** | 9 | 381 |

### Performance Metrics
*   **Pneumonia Sensitivity (Recall):** 0.977
*   **Normal Recall (Specificity):** 0.692
*   **False Positive Rate (FPR) on Healthy Subjects:** 30.8%

### Error Path Analysis & Risk Assessment
The model displays a significant **asymmetric bias toward predicting the positive class (Pneumonia).**

**The Screening Perspective:** Erring toward false positives is traditionally preferred in automated screening pipelines, as missing a sick patient (9 false negatives) carries a higher clinical cost than over-flagging a healthy one.
**The Operational Reality (Alarm Fatigue):** A **30.8% false-positive rate** is prohibitively high for practical clinical environments. Deploying this threshold at scale would induce severe alarm fatigue among clinicians and trigger costly, unnecessary secondary diagnostic procedures. The decision threshold (τ) must be explicitly tuned via a formal cost-benefit utility matrix before any downstream application.

---

## Fairness, Auditing & Subgroup Analysis

**Demographic Blind Spots:** The PneumoniaMNIST dataset does not include stratified metadata (e.g., age, sex, ethnicity, or specific clinical site markers). Consequently, **a granular subgroup fairness or algorithmic bias audit is impossible to perform.**
**Deployment Risk:** The inability to audit performance across subgroups represents a critical barrier to deployment. In a production healthcare system, unmonitored performance drops across specific demographics can lead to severe disparities in care quality.
**Class Disparity:** The sole measurable axis of variance is the performance delta between classes, showing a clear optimization bias toward Sensitivity (0.977) over Specificity (0.692).

---

## Vulnerabilities & Misuse Risks

*   **Automation Bias:** High-scoring summary statistics (like a 0.99 validation AUC) risk tricking downstream operators into treating uncalibrated model probabilities as definitive diagnostic truths.
*   **Out-of-Distribution (OOD) Fragility:** When exposed to adult X-rays, alternative imaging positions (AP vs. PA), or artifacts like medical tubing/jewelry, the model's behavior is completely undefined and prone to silent failures.

---

## Operational Readiness Checklist

To transition a pipeline of this nature from an engineering artifact toward a viable clinical trial or deployment, the following rigorous engineering and scientific phases must be executed:

**High-Fidelity Inputs:** Retrain using full-resolution DICOM imagery to preserve subtle spatial structures.
**Subgroup Auditing:** Procure metadata to perform disaggregated evaluations across age, sex, and institutional cohorts to ensure algorithmic equity.
**Probability Calibration:** Apply Platt Scaling or Temperature Scaling to map raw output scores to authentic, reliable empirical probabilities.
**Contextual Threshold Optimization:** Derive the operational threshold (τ) by evaluating an explicit clinical utility function that balances the exact institutional costs of a false negative versus a false positive.
**Prospective Clinical Validation:** Run the model in a non-interventional shadowing mode on live, unseen hospital streams to evaluate true operational distribution drift.