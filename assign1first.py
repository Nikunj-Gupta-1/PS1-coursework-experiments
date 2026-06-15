import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

data = pd.read_csv(__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '../raw data collected for project/cybersecurity_intrusion_data.csv')))
data2 = data
print("Dataset Shape:", data.shape)
print("\nDataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())

# Check for missing values and basic statistics
print("\nMissing values:")
print(data.isnull().sum())
print("\nBasic statistics:")
print(data.describe())

# Drop session_id
data = data.drop('session_id',  axis=1)
data['encryption_used'] = data['encryption_used'].fillna('Unknown')

# Identify categorical columns
categorical_cols = [ 'encryption_used', 'browser_type','protocol_type',]

# One-hot encode categorical columns
data = pd.get_dummies(data, columns=categorical_cols)

# Separate features and target
X = data.drop('attack_detected', axis=1)
y = data['attack_detected']

print(X.head())
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=37, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=37
)
rf_model.fit(X_train_scaled, y_train)
# Predict
y_pred = rf_model.predict(X_test_scaled)



# xgb_model = xgb.XGBClassifier(
#     n_estimators=200,
#     max_depth=10,
#     learning_rate=0.1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     use_label_encoder=False,
#     eval_metric='logloss'
# )
# xgb_model.fit(X_train_scaled, y_train)
# y_pred = xgb_model.predict(X_test_scaled)
# print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)



param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train_scaled, y_train)
print("Best params:", grid.best_params_)
print("Best score:", grid.best_score_)
