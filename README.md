# ❤️ Cardiovascular Disease Risk Prediction Using Machine Learning

An end-to-end Machine Learning project for predicting cardiovascular disease risk using behavioural and clinical health data.

The project compares multiple Machine Learning and Deep Learning models across two healthcare datasets and evaluates their performance both within the same dataset and across different datasets. The best-performing model was then saved and integrated into a deployed web application for prediction.

---

## 🚀 Live Application

🔗 **Live Demo:**  
(https://heart-risk-screen.emergent.host/?utm_source=share)

The deployed application allows users to interact with the trained prediction system through a web interface.

---

## 📌 Project Objective

The main objective of this project is to develop a reliable cardiovascular disease risk prediction system using machine learning.

The project focuses on:

- Comparing multiple ML models
- Evaluating models on different healthcare datasets
- Studying cross-dataset generalization
- Selecting the best-performing model
- Saving the trained model and preprocessing pipeline
- Deploying the final prediction system as a web application

---

## 📊 Datasets

Two healthcare datasets were used in this project:

### 1. Framingham Dataset

The Framingham dataset contains demographic, behavioural and clinical health features used for cardiovascular risk prediction.

### 2. BRFSS Dataset

The Behavioral Risk Factor Surveillance System (BRFSS) dataset contains large-scale behavioural and health-related information.

Using two different datasets allows the project to evaluate how well models generalize between different data distributions.

---

# 🔬 Experimental Setup

The project was evaluated using two types of experiments.

## Same-Dataset Evaluation

Models were trained and tested using data from the same dataset.

- Framingham → Framingham
- BRFSS → BRFSS

## Cross-Dataset Evaluation

Models were trained on one dataset and evaluated on the other.

Four evaluation scenarios were considered:

1. Framingham → BRFSS
2. BRFSS → Framingham
3. Framingham → Framingham
4. BRFSS → BRFSS

This helps evaluate the generalization capability of the models across different healthcare datasets.

---

# 🤖 Models Compared

Five different Machine Learning / Deep Learning approaches were investigated:

- XGBoost
- CatBoost
- FT-Transformer
- TabTransformer
- AutoGluon

The models were compared using appropriate classification metrics.

---

# ⚙️ Data Preprocessing

The preprocessing pipeline included several steps to prepare the healthcare data for modelling.

Major preprocessing steps included:

- Handling missing values
- Replacing invalid / coded missing values
- Feature engineering
- Feature selection
- Data transformation
- Feature scaling
- Class imbalance handling
- Dataset-specific preprocessing

The preprocessing pipeline was saved separately so that the same transformations can be applied during inference.

Saved preprocessing file:

`preprocessing_pipeline.pkl`

---

# 🧠 Model Selection

After evaluating the different models across the experimental scenarios, the best-performing model was selected for deployment.

The final trained FT-Transformer model was saved along with its configuration and required model files.

The saved model directory contains:

```text
ft_transformer_model/
├── callbacks.sav
├── config.yml
├── custom_params.sav
├── datamodule.sav
└── model.ckpt
