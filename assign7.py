import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip() 
    print("\nData loaded successfully.")
    print(f"Dataset shape: {df.shape}")
    return df

def display_columns_with_numbers(df):
    print("\nAvailable columns:")
    for idx, col in enumerate(df.columns):
        print(f"{idx}: {col}")

def convert_target_to_binary(df, target_col):
    # If target column is already numeric and binary, do nothing
    if df[target_col].dtype in ['int64', 'float64'] and df[target_col].nunique() == 2:
        return df[target_col]
    
    # If target column is string, map unique values to 0 and 1
    elif df[target_col].dtype == 'object' and df[target_col].nunique() == 2:
        unique_vals = df[target_col].unique()
        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        return df[target_col].map(mapping)
    
    # If target is numeric but not binary (more than two classes), raise an error
    elif df[target_col].dtype in ['int64', 'float64']:
        raise ValueError("Target column is numeric but not binary.")
    
    # If target is string but has more than 2 classes, raise an error
    else:
        raise ValueError("Target column has more than two unique values; cannot convert to binary.")


def preprocess_data(df, target_col, feature_cols, scaling=True):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('Unknown')

    numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in feature_cols if df[col].dtype == 'object']
    
    # One-hot encode categorical features
    df_categorical = pd.get_dummies(df[categorical_cols], drop_first=True)
    df_features = pd.concat([df[numeric_cols], df_categorical], axis=1)
    
    
    X = df_features.values
    y = convert_target_to_binary(df, target_col)
    
    if scaling:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, df
    else:
        return X, y, df



def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=2,
        random_state=42
    )

    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    xgb = XGBClassifier(
        n_estimators=1000,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight = scale_pos_weight
    )
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('xgb', xgb)
        ],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    y_pred_xgb = xgb.predict(X_test)
    y_pred_ensemble = ensemble.predict(X_test)

    
    print_metrics(y_test, y_pred_rf, "Random Forest")
    print_metrics(y_test, y_pred_xgb, "XGBoost")
    print_metrics(y_test, y_pred_ensemble, "Ensemble Model")

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_ensemble), annot=True, fmt='d', cmap='Blues')
    plt.title('Ensemble Model Confusion Matrix')
    plt.show()
    cv_scores = cross_val_score(ensemble, np.vstack([X_train, X_test]), np.hstack([y_train, y_test]), cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

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
        print("4. Train and Evaluate Models")
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
