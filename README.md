# 💳 Credit Risk Prediction System

An end-to-end machine learning application that analyses a customer's demographic and financial profile to predict the likelihood of **credit card default**. Built entirely in Python using a Random Forest classifier and a Streamlit frontend.

---

## 📁 Project Structure

```
credit risk prediction/
│
├── train.csv               # Training dataset (45,528 customer records)
├── test.csv                # Test dataset (unlabelled, for batch prediction)
│
├── train_model.py          # Model training script — run this first
├── streamlit_app.py        # Full frontend + backend application
├── requirements.txt        # Python dependencies
│
├── model.pkl               # Trained Random Forest model (auto-generated)
├── label_encoders.pkl      # Fitted label encoders (auto-generated)
└── metrics.json            # Model evaluation metrics (auto-generated)
```

---

## ⚙️ Setup & Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

This will:
- Load and preprocess `train.csv`
- Train a Random Forest classifier
- Evaluate on a 20% validation split
- Save `model.pkl`, `label_encoders.pkl`, and `metrics.json`

### 3. Launch the app

```bash
streamlit run streamlit_app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Training records | 45,528 |
| Validation split | 80 / 20 |
| Class weighting | Balanced (handles class imbalance) |
| Accuracy | ~96% |
| ROC-AUC | ~0.995 |

---

## 📊 Dataset Features

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Customer age in years |
| `gender` | Categorical | F / M |
| `owns_car` | Categorical | Y / N |
| `owns_house` | Categorical | Y / N |
| `no_of_children` | Numeric | Number of children |
| `net_yearly_income` | Numeric | Annual income (₹) |
| `no_of_days_employed` | Numeric | Total days employed |
| `occupation_type` | Categorical | Job category (19 types) |
| `total_family_members` | Numeric | Total dependents including self |
| `migrant_worker` | Binary | 1 = migrant worker |
| `yearly_debt_payments` | Numeric | Annual debt repayment amount (₹) |
| `credit_limit` | Numeric | Approved credit limit (₹) |
| `credit_limit_used(%)` | Numeric | Percentage of credit limit utilised |
| `credit_score` | Numeric | Credit bureau score (300–900) |
| `prev_defaults` | Numeric | Number of previous defaults |
| `default_in_last_6months` | Binary | 1 = defaulted in last 6 months |
| `credit_card_default` | Binary | **Target** — 1 = default, 0 = no default |

---

## 🖥️ Application Pages

### 🏠 Home
Overview of the system with key metrics — training records, features, model accuracy and ROC-AUC score.

### 🔍 Single Prediction
Enter a customer's details through an interactive form and get an instant prediction with:
- Default / No Default verdict
- Probability score (%)
- Risk level (Low / Medium / High)
- Visual probability gauge

### 📂 Batch Prediction
Upload a customer dataset CSV or use the included test file to:
- Predict default risk for all customers at once
- View colour-coded risk levels (Low / Medium / High)
- See aggregate default rate statistics
- Download predictions as a CSV file

### 📊 Model Performance
Detailed evaluation metrics including:
- Accuracy and ROC-AUC score
- Full classification report table
- Confusion matrix
- Top feature importances bar chart

### 📈 Data Explorer
Explore the training dataset with:
- Default rate distribution
- Gender breakdown
- Credit score distribution by default status
- Occupation type counts
- Full summary statistics table

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Frontend UI |
| `scikit-learn` | Machine learning model |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `joblib` | Model serialisation |
