# Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn using the IBM Telco Customer Churn dataset.

The project focuses on identifying customers who are likely to leave a telecom service. Since the dataset is imbalanced, the pipeline uses **SMOTE** and evaluates models using **precision, recall, F1-score, and ROC-AUC** rather than relying only on accuracy.

---

## 📌 Project Overview

Customer churn is an important business problem because retaining an existing customer is often more valuable than acquiring a replacement.

The goal of this project is to build a machine learning pipeline that can identify customers who are likely to churn based on demographic information, account details, subscribed services, contract information, and billing data.

Three classification models were compared:

- Logistic Regression
- Random Forest
- XGBoost

---

## 📊 Dataset

The project uses the **IBM Telco Customer Churn dataset**.

Dataset size:

- **7,043 customers**
- **21 original columns**
- **5,174 non-churn customers**
- **1,869 churn customers**

### Target Variable

`Churn`

- `No` → `0`
- `Yes` → `1`

The training data contains approximately:

- **73.46% non-churn customers**
- **26.54% churn customers**

This shows that the target variable is imbalanced.

---

## 🔍 Features

The dataset contains customer information such as:

- Gender
- Senior citizen status
- Partner and dependent status
- Tenure
- Phone service
- Internet service
- Online security
- Online backup
- Device protection
- Tech support
- Streaming services
- Contract type
- Payment method
- Monthly charges
- Total charges

The `customerID` column was removed because it is an identifier and does not provide useful predictive information.

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

### 1. TotalCharges Conversion

`TotalCharges` was converted to a numeric datatype using:

```python
pd.to_numeric(errors="coerce")
```

Missing values resulting from conversion were handled using the median.

### 2. Customer ID Removal

The `customerID` column was removed before training.

### 3. Target Encoding

The target variable was converted from:

```text
Yes → 1
No  → 0
```

### 4. Numerical Features

Numerical features were standardized using:

**StandardScaler**

### 5. Categorical Features

Categorical variables were transformed using:

**OneHotEncoder**

with:

```python
handle_unknown="ignore"
```

### 6. Train-Test Split

The dataset was split into:

- **80% training data**
- **20% testing data**

A stratified split was used to preserve the churn distribution.

---

## ⚖️ Handling Class Imbalance

The dataset is imbalanced, with approximately 73% of customers belonging to the non-churn class.

A model that simply predicts "No Churn" for every customer could therefore achieve relatively high accuracy while being practically useless.

To address this problem, the project uses:

**SMOTE — Synthetic Minority Over-sampling Technique**

SMOTE is included inside the training pipeline so that oversampling is applied during model training.

This allows the models to learn the minority churn class more effectively.

---

## 🤖 Models

Three machine learning models were evaluated:

### Logistic Regression

A linear classification model that provides a strong baseline for binary classification.

### Random Forest

An ensemble of decision trees designed to capture nonlinear relationships while reducing overfitting compared with individual decision trees.

### XGBoost

A gradient boosting algorithm that builds trees sequentially to improve prediction performance.

---

## 📈 Model Results

| Model | Churn Precision | Churn Recall | Churn F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.503 | **0.794** | **0.616** | **0.8400** |
| Random Forest | 0.571 | 0.559 | 0.565 | 0.8162 |
| XGBoost | **0.586** | 0.594 | 0.590 | 0.8181 |

---

## 🏆 Model Evaluation

For this problem, accuracy alone is not sufficient because the dataset is imbalanced.

The most important metrics considered were:

- **Precision** — Of customers predicted to churn, how many actually churned?
- **Recall** — Of all customers who actually churned, how many were identified?
- **F1-score** — Balance between precision and recall
- **ROC-AUC** — Ability of the model to distinguish churners from non-churners across classification thresholds

### Logistic Regression

Logistic Regression achieved:

- Churn Precision: **0.503**
- Churn Recall: **0.794**
- Churn F1-score: **0.616**
- ROC-AUC: **0.8400**

It achieved the **highest churn recall, highest churn F1-score, and highest ROC-AUC** among the three tested models.

This makes Logistic Regression particularly useful when the business priority is identifying as many potential churners as possible.

### XGBoost

XGBoost achieved the highest churn precision:

**0.586**

However, its churn recall of **0.594** was considerably lower than Logistic Regression's **0.794**.

Therefore, the "best" model depends on the business objective. If missing a potential churner is considered more costly than contacting a customer who would not actually churn, Logistic Regression is the stronger choice in these experiments.

---

## 📊 Confusion Matrix

The notebook generates a confusion matrix for the XGBoost pipeline.

![Confusion Matrix](confusion_matrix.png)

A confusion matrix helps visualize:

- True Positives
- True Negatives
- False Positives
- False Negatives

For churn prediction, **false negatives are especially important**, because they represent customers who actually churn but were not identified by the model.

---

## 📈 ROC Curve

The notebook also generates an ROC curve for XGBoost.

![ROC Curve](roc_curve.png)

The ROC curve visualizes the trade-off between the true positive rate and false positive rate at different classification thresholds.

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Imbalanced-learn
- SMOTE
- XGBoost
- Matplotlib
- Google Colab

---

## 🔄 Machine Learning Pipeline

```text
IBM Telco Churn Dataset
        ↓
Data Cleaning
        ↓
Feature / Target Separation
        ↓
Train-Test Split
        ↓
ColumnTransformer
        ↓
 ┌───────────────────────┐
 │ Numerical Features    │ → StandardScaler
 │ Categorical Features  │ → OneHotEncoder
 └───────────────────────┘
        ↓
SMOTE
        ↓
Model Training
        ↓
 ┌──────────────────────┐
 │ Logistic Regression  │
 │ Random Forest        │
 │ XGBoost              │
 └──────────────────────┘
        ↓
Model Evaluation
        ↓
Precision / Recall / F1 / ROC-AUC
```

---

## 💡 Key Takeaways

- Accuracy alone can be misleading for imbalanced classification problems.
- SMOTE can help models learn from an underrepresented target class.
- Recall is particularly important when failing to identify a churner is costly.
- Logistic Regression produced the highest churn recall (**79.4%**) and ROC-AUC (**0.8400**) in this experiment.
- XGBoost produced the highest churn precision (**58.6%**).
- Model selection should depend on the business cost of false positives versus false negatives.

---

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── customer_churn_prediction.ipynb
├── churn.csv
├── confusion_matrix.png
├── roc_curve.png
└── README.md
```

---

## 🚀 How to Run

Install the required libraries:

```bash
pip install pandas scikit-learn imbalanced-learn xgboost matplotlib
```

Open and run:

```text
customer_churn_prediction.ipynb
```

Run the notebook cells in order to reproduce preprocessing, model training, evaluation, and visualizations.

---

## 🎯 Business Perspective

In a real customer-retention system, failing to identify a customer who is about to churn may be more expensive than incorrectly flagging a customer as a churn risk.

For that reason, this project gives particular attention to **recall for the churn class** instead of optimizing only for overall accuracy.

The Logistic Regression model identified approximately **79.4% of actual churners** in the test data, making it the strongest model among the tested approaches when churn recall is the primary objective.
