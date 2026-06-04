import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df = df.drop("customerID", axis=1)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")
plt.show()

plt.figure(figsize=(8,4))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Contract Type vs Churn")
plt.xticks(rotation=15)
plt.show()

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=5000)
lr.fit(X_train_scaled, y_train)
pred = lr.predict(X_test_scaled)
prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
roc_lr = roc_auc_score(y_test, prob_lr)

print("\n===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print("ROC-AUC Score:", roc_lr)

ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot()
plt.title("Logistic Regression Confusion Matrix")
plt.show()

fpr, tpr, _ = roc_curve(y_test, prob_lr)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_lr:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
prob_rf = rf.predict_proba(X_test)[:, 1]
roc_rf = roc_auc_score(y_test, prob_rf)

print("\n===== Random Forest =====")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
print("ROC-AUC Score:", roc_rf)

xgb = XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
prob_xgb = xgb.predict_proba(X_test)[:, 1]
roc_xgb = roc_auc_score(y_test, prob_xgb)

print("\n===== XGBoost =====")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))
print("ROC-AUC Score:", roc_xgb)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features")
print(importance.head(10))

plt.figure(figsize=(10,5))
sns.barplot(x="Importance", y="Feature", data=importance.head(10))
plt.title("Top 10 Important Features")
plt.show()

print("\n===== Model Comparison =====")
models = {
    "Logistic Regression": (pred, prob_lr),
    "Random Forest":       (rf_pred, prob_rf),
    "XGBoost":             (xgb_pred, prob_xgb),
}
for name, (p, prob) in models.items():
    print(f"{name} — Accuracy: {accuracy_score(y_test, p):.4f} | "
          f"Recall: {recall_score(y_test, p):.4f} | "
          f"ROC-AUC: {roc_auc_score(y_test, prob):.4f}")
