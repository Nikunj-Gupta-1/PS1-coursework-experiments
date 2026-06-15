import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    data = pd.read_csv(file_path)
    print("\nDataset loaded successfully.")
    print("Shape:", data.shape)
    print("Columns:", list(data.columns))
    return data

def preprocess_data(data, target_col, feature_cols, scaling=True):
    df = data.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(df[col].mean())
    df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, drop_first=True)
    X = df[feature_cols]
    y = df[target_col]
    if scaling:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    return X_scaled, y, df

def hyperparameter_tuning(X, y):
    # RF
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, n_jobs=-1)
    rf_grid.fit(X, y)
    rf_model = rf_grid.best_estimator_
    print("Best RF params:", rf_grid.best_params_)

    # XGB
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'learning_rate': [0.05, 0.1]
    }
    xgb_grid = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'), xgb_params, cv=3, n_jobs=-1)
    xgb_grid.fit(X, y)
    xgb_model = xgb_grid.best_estimator_
    print("Best XGB params:", xgb_grid.best_params_)

    # Logistic Regression
    lr_params = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }
    lr_grid = GridSearchCV(LogisticRegression(max_iter=1000), lr_params, cv=3, n_jobs=-1)
    lr_grid.fit(X, y)
    lr_model = lr_grid.best_estimator_
    print("Best LR params:", lr_grid.best_params_)

    return rf_model, xgb_model, lr_model

def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"\n{name} Performance:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        plot_confusion_matrix(y_test, y_pred, title=f'{name} Confusion Matrix')

def plot_confusion_matrix(y_true, y_pred, title):
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.show()

def feature_importance(models, feature_names):
    rf_importance = models['Random Forest'].feature_importances_
    xgb_importance = models['XGBoost'].feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'RF_importance': rf_importance,
        'XGB_importance': xgb_importance
    })
    importance_df['Average'] = (importance_df['RF_importance'] + importance_df['XGB_importance']) / 2
    top_features = importance_df.sort_values('Average', ascending=False).head(10)
    print("\nTop 10 Features by Average Importance:")
    print(top_features[['Feature', 'Average']])

def main():
    while True:
        print("\n--- Prediction Model Menu ---")
        print("1. Load Dataset")
        print("2. Select Target Column")
        print("3. Select Feature Columns")
        print("4. Train and Evaluate Models with Hyperparameter Tuning")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            file_path = input("Enter dataset file path (default: cybersecurity_intrusion_data.csv): ")
            if file_path.strip() == '':
                file_path = __import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '../raw data collected for project/cybersecurity_intrusion_data.csv'))
            data = load_data(file_path)

            # If healthcare dataset, set sample target and features
            if 'healthcare' in file_path.lower():
                target_col = 'disease_present'
                feature_cols = ['age', 'blood_pressure', 'cholesterol_level', 'blood_sugar', 'gender', 'smoker']
                print(f"Sample target column: {target_col}")
                print(f"Sample feature columns: {feature_cols}")

        elif choice == '2':
            target_col = input("Enter target column: ")
            print(f"Target column set to: {target_col}")

        elif choice == '3':
            feature_cols_input = input("Enter feature columns separated by commas: ")
            feature_cols = [col.strip() for col in feature_cols_input.split(',')]
            print(f"Feature columns set to: {feature_cols}")

        elif choice == '4':
            scaling_input = input("Scale features? (y/n, default: y): ")
            scaling = True if scaling_input.strip().lower() in ['y', ''] else False

            X_scaled, y, processed_data = preprocess_data(data, target_col, feature_cols, scaling=scaling)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

            rf_model, xgb_model, lr_model = hyperparameter_tuning(X_train, y_train)

            ensemble = VotingClassifier(
                estimators=[('RF', rf_model), ('XGB', xgb_model), ('LR', lr_model)],
                voting='hard',
                n_jobs=5
            )
            ensemble.fit(X_train, y_train)

            models = {
                'Random Forest': rf_model,
                'XGBoost': xgb_model,
                'Logistic Regression': lr_model,
                'Ensemble': ensemble
            }

            evaluate_models(models, X_test, y_test)
            feature_importance(models, feature_cols)

        elif choice == '5':
            print("Exiting program. Goodbye!")
            break

        else:
            print("Invalid choice. Please select from 1 to 5.")

if __name__ == '__main__':
    main()
