# Customer Churn Prediction using Machine Learning

## Project Overview

Customer churn is one of the most critical challenges faced by businesses in the telecommunications industry. Retaining existing customers is often more cost-effective than acquiring new ones. This project aims to predict whether a customer is likely to discontinue a service using machine learning techniques.

The project utilizes the Telco Customer Churn dataset and implements multiple classification algorithms to identify customers at risk of churning. The results help businesses take proactive measures to improve customer retention and reduce revenue loss.

---

## Objectives

* Analyze customer behavior and service usage patterns.
* Identify factors influencing customer churn.
* Build and compare multiple machine learning models.
* Evaluate model performance using standard classification metrics.
* Generate actionable business insights for customer retention.

---

## Dataset

**Dataset:** Telco Customer Churn Dataset

The dataset contains customer demographic information, account details, subscribed services, billing information, and churn status.

### Features Include:

* Gender
* Senior Citizen Status
* Partner and Dependents
* Tenure
* Phone Service
* Internet Service
* Online Security
* Tech Support
* Contract Type
* Payment Method
* Monthly Charges
* Total Charges
* Churn Status (Target Variable)

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost

---

## Project Workflow

### 1. Data Preprocessing

* Loaded customer churn dataset.
* Converted data types where necessary.
* Handled missing values.
* Removed irrelevant columns.
* Encoded categorical variables.

### 2. Exploratory Data Analysis (EDA)

* Customer churn distribution analysis.
* Contract type vs churn analysis.
* Data visualization using charts and graphs.

### 3. Feature Engineering

* One-hot encoding of categorical variables.
* Feature scaling using StandardScaler.

### 4. Model Building

The following machine learning algorithms were implemented:

#### Logistic Regression

* Baseline classification model.
* Scaled numerical features before training.

#### Random Forest Classifier

* Ensemble learning technique.
* Used feature importance analysis.

#### XGBoost Classifier

* Gradient boosting-based algorithm.
* High-performance predictive model.

---

## Model Evaluation

The models were evaluated using:

* Accuracy Score
* Recall Score
* Classification Report
* Confusion Matrix
* ROC-AUC Score
* ROC Curve

---

## Visualizations

The project includes:

* Customer Churn Distribution
* Contract Type vs Churn Analysis
* Confusion Matrix
* ROC Curve
* Top Feature Importance Visualization

---

## Key Business Insights

* Customers with month-to-month contracts are more likely to churn.
* Customers with longer tenure tend to remain loyal.
* Contract type significantly impacts churn behavior.
* Certain service combinations influence customer retention.
* Early identification of high-risk customers can improve retention strategies.

---

## Project Structure

```text
customer-churn-prediction/
│
├── churn_project.py
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
```

Navigate to the project folder:

```bash
cd customer-churn-prediction
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## Running the Project

Execute the Python script:

```bash
python churn_project.py
```

---

## Future Improvements

* Hyperparameter tuning for better model performance.
* Deployment using Flask or Streamlit.
* Real-time churn prediction dashboard.
* Automated customer retention recommendation system.

---

## Author

**Shah Newaj Ahshan Mazumder**

BCA Student | Data Analytics Intern

---

## License

This project is developed for educational and internship purposes.
