import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    df = pd.read_csv(file_path)
    print("\nData loaded successfully.")
    print(f"Dataset shape: {df.shape}")
    return df

def display_columns_with_numbers(df):
    print("\nAvailable columns:")
    for idx, col in enumerate(df.columns):
        print(f"{idx}: {col}")

def preprocess_data(df, target_col, feature_cols, scaling=True):
    # Fill missing categorical values
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('Unknown')

    # Store which features are categorical
    categorical_cols = [col for col in feature_cols if df[col].dtype == 'object']

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, drop_first=True)

    # After encoding, find the updated feature columns
    new_feature_cols = []
    for col in feature_cols:
        if col in categorical_cols:
            # Include all dummy variables created from this categorical column
            new_cols = [c for c in df.columns if c.startswith(col + '_')]
            new_feature_cols.extend(new_cols)
        else:
            new_feature_cols.append(col)

    X = df[new_feature_cols]
    y = df[target_col]

    if scaling:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, df
    else:
        return X.values, y, df


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Models and parameter grids for tuning
    rf_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }
    xgb_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'learning_rate': [0.05, 0.1]
    }
    lr_grid = {
        'C': [0.1, 1.0, 10],
        'solver': ['lbfgs']
    }

    rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_grid, cv=3, n_jobs=-1)
    xgb = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'), xgb_grid, cv=3, n_jobs=-1)
    lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), lr_grid, cv=3, n_jobs=-1)

    # Fit models
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    # Ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf.best_estimator_),
            ('xgb', xgb.best_estimator_),
            ('lr', lr.best_estimator_)
        ],
        voting='hard'
    )
    ensemble.fit(X_train, y_train)

    # Predictions
    y_pred_ensemble = ensemble.predict(X_test)
    y_pred_rf = rf.best_estimator_.predict(X_test)
    y_pred_xgb = xgb.best_estimator_.predict(X_test)
    y_pred_lr = lr.best_estimator_.predict(X_test)

    # Evaluation
    print_metrics(y_test, y_pred_ensemble, "Ensemble Model")
    print_metrics(y_test, y_pred_rf, "Random Forest")
    print_metrics(y_test, y_pred_xgb, "XGBoost")
    print_metrics(y_test, y_pred_lr, "Logistic Regression")

    # Confusion matrix for ensemble
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_ensemble), annot=True, fmt='d', cmap='Blues')
    plt.title('Ensemble Model Confusion Matrix')
    plt.show()

def print_metrics(y_true, y_pred, model_name):
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

def main():
    data = None
    target_col = None
    feature_cols = None

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
            display_columns_with_numbers(data)

        elif choice == '2':
            if data is not None:
                display_columns_with_numbers(data)
                col_num = int(input("Enter the column number for the target variable: "))
                target_col = data.columns[col_num]
                print(f"Target column set to: {target_col}")
            else:
                print("Please load the dataset first.")

        elif choice == '3':
            if data is not None:
                display_columns_with_numbers(data)
                col_nums_input = input("Enter feature column numbers separated by commas: ")
                col_nums = [int(num.strip()) for num in col_nums_input.split(',')]
                feature_cols = [data.columns[num] for num in col_nums]
                print(f"Feature columns set to: {feature_cols}")
            else:
                print("Please load the dataset first.")

        elif choice == '4':
            if data is not None and target_col is not None and feature_cols is not None:
                scaling_input = input("Scale features? (y/n, default: y): ")
                scaling = True if scaling_input.strip().lower() in ['y', ''] else False

                X_scaled, y, processed_data = preprocess_data(data, target_col, feature_cols, scaling=scaling)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

                train_and_evaluate_models(X_train, X_test, y_train, y_test)
            else:
                print("Please ensure dataset, target, and feature columns are selected first.")

        elif choice == '5':
            print("Exiting.")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
