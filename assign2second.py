import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
data = pd.read_csv(__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '../raw data collected for project/cybersecurity_intrusion_data.csv')))
print("Dataset Shape:", data.shape)

# Handle missing values and encode categorical variables
data['encryption_used'] = data['encryption_used'].fillna('Unknown')
data = pd.get_dummies(data, columns=['encryption_used', 'browser_type', 'protocol_type'])
data = data.drop('session_id', axis=1)

# Split data
X = data.drop('attack_detected', axis=1)
y = data['attack_detected']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models with optimized parameters
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=37
)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=37,
    eval_metric='logloss',
    use_label_encoder=False
)

# Create ensemble model
ensemble = VotingClassifier(
    estimators=[
        ('random_forest', rf_model),
        ('xgboost', xgb_model)
    ],
    voting='hard', 
    n_jobs= 5
)

# Train and evaluate ensemble
ensemble.fit(X_train_scaled, y_train)
y_pred_ensemble = ensemble.predict(X_test_scaled)

# Evaluate individual models for comparison
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Print performance metrics
def print_metrics(y_true, y_pred, model_name):
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

print_metrics(y_test, y_pred_ensemble, "Ensemble Model")
print_metrics(y_test, y_pred_rf, "Random Forest")
print_metrics(y_test, y_pred_xgb, "XGBoost")

# Plot confusion matrix for ensemble
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_ensemble), 
            annot=True, fmt='d', cmap='Blues')
plt.title('Ensemble Model Confusion Matrix')
plt.show()

# Feature importance analysis
rf_importance = rf_model.feature_importances_
xgb_importance = xgb_model.feature_importances_

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'RF_importance': rf_importance,
    'XGB_importance': xgb_importance
})

feature_importance['Average_importance'] = (feature_importance['RF_importance'] + feature_importance['XGB_importance']) / 2

print("Top 10 Features by Average Importance:")
print(feature_importance.sort_values('Average_importance', ascending=False).head(10))
