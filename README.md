
# 🛍️ ShopSense AI
### AI E-Commerce Customer Intelligence & Recommendation System

ShopSense AI is a data science project that analyzes e-commerce data to understand customer behavior, segment customers, predict repeat purchases, and recommend relevant products.

## 🚀 Features

- **Business Overview:** View customer, order, and product statistics.
- **Customer Segmentation:** Explore customer groups using RFM analysis and K-Means clustering.
- **Purchase Prediction:** Estimate the likelihood of repeat purchases using a trained machine learning model.
- **Product Recommendations:** Recommend popular products that a customer has not purchased.
- **Data Export:** Download customer segment and recommendation results as CSV files.
- **Interactive Dashboard:** Explore project results using Streamlit.

## 🧰 Technology Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook
- Git and GitHub

## 📂 Project Structure

```text
AI E-Commerce Customer Intelligence/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning_eda.ipynb
│   ├── 03_customer_intelligence.ipynb
│   ├── 04_customer_segmentation.ipynb
│   ├── 05_predictive_modeling.ipynb
│   └── 06_recommendation_engine.ipynb
├── reports/
├── src/
├── tests/
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "AI E-Commerce Customer Intelligence"
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
python -m streamlit run app/app.py
```

Open the local URL displayed in the terminal.

## 🤖 Machine Learning Workflow

1. Explore the dataset using Pandas.
2. Clean and prepare transaction data.
3. Perform exploratory data analysis (EDA).
4. Calculate customer-level RFM features.
5. Segment customers using K-Means clustering.
6. Train and evaluate a repeat-purchase prediction model.
7. Generate product recommendations.
8. Display results in a Streamlit dashboard.

## 📊 Model Evaluation

The current model evaluation recorded during development:

| Metric | Result |
|---|---:|
| Accuracy | 77% |
| ROC-AUC | 0.86 |
| Class 1 Precision | 0.50 |
| Class 1 Recall | 0.83 |

These are development results, not a guarantee of future performance. The target definition, test methodology, and feature consistency should be verified before final submission.

## 🖥️ Dashboard Pages

1. Overview
2. Customer Segmentation
3. Purchase Prediction
4. Product Recommendations

## 🔒 Data and Model Files

The application uses processed CSV files and locally saved model artifacts.

Do not upload private customer data, credentials, or unnecessarily large raw datasets to a public repository. Add appropriate exclusions to `.gitignore`.

## 🎯 Project Objective

The objective is to demonstrate how data analysis and machine learning can support customer understanding, repeat-purchase analysis, and product discovery in e-commerce.

## 👨‍💻 Author

**Sujeet Goswami**

MCA Data Science Project
