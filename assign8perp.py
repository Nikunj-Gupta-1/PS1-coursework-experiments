import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load and validate dataset"""
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        print(f"\nData loaded successfully.")
        print(f"Dataset shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def display_columns_with_numbers(df):
    """Display columns with their indices"""
    print("\nAvailable columns:")
    for idx, col in enumerate(df.columns):
        print(f"{idx}: {col}")

def convert_target_to_binary(df, target_col):
    """Convert target column to binary format with error handling"""
    try:
        if df[target_col].dtype in ['int64', 'float64'] and df[target_col].nunique() == 2:
            return df[target_col]
        elif df[target_col].dtype == 'object' and df[target_col].nunique() == 2:
            unique_vals = df[target_col].unique()
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            return df[target_col].map(mapping)
        elif df[target_col].dtype in ['int64', 'float64']:
            raise ValueError("Target column is numeric but not binary.")
        else:
            raise ValueError("Target column has more than two unique values; cannot convert to binary.")
    except Exception as e:
        print(f"Error converting target column: {str(e)}")
        return None

def advanced_feature_engineering(df, feature_cols):
    """Advanced feature engineering for cybersecurity data"""
    df_engineered = df[feature_cols].copy()
    
    # Create interaction features for cybersecurity-specific combinations
    if 'login_attempts' in feature_cols and 'failed_logins' in feature_cols:
        df_engineered['login_failure_rate'] = df_engineered['failed_logins'] / (df_engineered['login_attempts'] + 1)
    
    if 'session_duration' in feature_cols and 'network_packet_size' in feature_cols:
        df_engineered['data_transfer_rate'] = df_engineered['network_packet_size'] / (df_engineered['session_duration'] + 1)
    
    if 'ip_reputation_score' in feature_cols and 'failed_logins' in feature_cols:
        df_engineered['risk_score'] = df_engineered['failed_logins'] * (1 - df_engineered['ip_reputation_score'])
    
    # Log transform for skewed numerical features
    numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_engineered[col].min() >= 0:  # Only for non-negative values
            df_engineered[f'{col}_log'] = np.log1p(df_engineered[col])
    
    return df_engineered

def preprocess_data(df, target_col, feature_cols, scaling_method='robust', feature_selection=True):
    """Enhanced preprocessing with feature engineering and selection"""
    try:
        # Handle missing values
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('Unknown')
        
        # Advanced feature engineering
        df_features = advanced_feature_engineering(df, feature_cols)
        
        # Handle categorical variables
        categorical_cols = [col for col in df_features.columns if df_features[col].dtype == 'object']
        if categorical_cols:
            df_categorical = pd.get_dummies(df_features[categorical_cols], drop_first=True)
            numeric_cols = [col for col in df_features.columns if col not in categorical_cols]
            df_features = pd.concat([df_features[numeric_cols], df_categorical], axis=1)
        
        # Handle infinite values
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(df_features.median())
        
        X = df_features.values
        y = convert_target_to_binary(df, target_col)
        
        if y is None:
            return None, None, None
        
        # Feature scaling
        if scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        X_scaled = scaler.fit_transform(X)
        
        # Feature selection
        if feature_selection and X_scaled.shape[1] > 10:
            selector = SelectKBest(score_func=f_classif, k=min(20, X_scaled.shape[1]))
            X_scaled = selector.fit_transform(X_scaled, y)
            print(f"Feature selection: {X_scaled.shape[1]} features selected")
        
        return X_scaled, y, df_features.columns.tolist()
    
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None, None, None

def get_optimized_models():
    """Get optimized models with hyperparameter tuning"""
    
    # Random Forest with optimized parameters
    rf = RandomForestClassifier(
        n_estimators=1500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    # XGBoost with optimized parameters based on research
    xgb = XGBClassifier(
        n_estimators=1500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    # LightGBM for additional ensemble diversity
    lgb = LGBMClassifier(
        n_estimators=1500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # Extra Trees for ensemble diversity
    et = ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    return rf, xgb, lgb, et

def hyperparameter_tuning(X_train, y_train, model_type='xgboost'):
    """Perform hyperparameter tuning for specified model"""
    
    if model_type == 'xgboost':
        param_dist = {
            'n_estimators': [1000, 1500, 2000],
            'max_depth': [6, 8, 10, 12],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }
        
        model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
        
    elif model_type == 'random_forest':
        param_dist = {
            'n_estimators': [1000, 1500, 2000],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Perform randomized search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    print(f"Best parameters for {model_type}: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def train_and_evaluate_models(X_train, X_test, y_train, y_test, balance_data=True, tune_hyperparams=False):
    """Enhanced model training with class balancing and hyperparameter tuning"""
    
    try:
        # Handle class imbalance
        if balance_data:
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"Original training set: {X_train.shape[0]} samples")
            print(f"Balanced training set: {X_train_balanced.shape[0]} samples")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Get models
        rf, xgb, lgb, et = get_optimized_models()
        
        # Hyperparameter tuning if requested
        if tune_hyperparams:
            print("Performing hyperparameter tuning...")
            xgb = hyperparameter_tuning(X_train_balanced, y_train_balanced, 'xgboost')
        
        # Calculate scale_pos_weight for XGBoost
        scale_pos_weight = len(y_train_balanced[y_train_balanced == 0]) / len(y_train_balanced[y_train_balanced == 1])
        xgb.set_params(scale_pos_weight=scale_pos_weight)
        lgb.set_params(scale_pos_weight=scale_pos_weight)
        
        # Train individual models
        print("Training models...")
        rf.fit(X_train_balanced, y_train_balanced)
        xgb.fit(X_train_balanced, y_train_balanced)
        lgb.fit(X_train_balanced, y_train_balanced)
        et.fit(X_train_balanced, y_train_balanced)
        
        # Create enhanced ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('xgb', xgb),
                ('lgb', lgb),
                ('et', et)
            ],
            voting='soft'
        )
        ensemble.fit(X_train_balanced, y_train_balanced)
        
        # Make predictions
        models = {
            'Random Forest': rf,
            'XGBoost': xgb,
            'LightGBM': lgb,
            'Extra Trees': et,
            'Enhanced Ensemble': ensemble
        }
        
        results = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'accuracy': accuracy,
                'auc': auc_score,
                'predictions': y_pred
            }
            
            print_enhanced_metrics(y_test, y_pred, y_pred_proba, name)
        
        # Plot confusion matrix for best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        plot_confusion_matrix(y_test, results[best_model_name]['predictions'], best_model_name)
        
        # Cross-validation with stratification
        print("\nPerforming stratified cross-validation...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(ensemble, X_train_balanced, y_train_balanced, cv=skf, scoring='accuracy')
        print(f"Stratified CV Accuracy Scores: {cv_scores}")
        print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return ensemble, results
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        return None, None

def print_enhanced_metrics(y_true, y_pred, y_pred_proba, model_name):
    """Print comprehensive model metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    """Enhanced main function with error handling"""
    data = None
    target_col = None
    feature_cols = None
    
    while True:
        print("\n--- Enhanced Cybersecurity Threat Detection ---")
        print("1. Load Dataset")
        print("2. Select Target Column")
        print("3. Select Feature Columns")
        print("4. Train and Evaluate Models (Basic)")
        print("5. Train with Advanced Features (Recommended)")
        print("6. Train with Hyperparameter Tuning (Best Performance)")
        print("7. Exit")
        
        try:
            choice = input("Enter your choice (1-7): ")
            
            if choice == '1':
                file_path = input("Enter dataset file path (default: cybersecurity_intrusion_data.csv): ")
                if file_path.strip() == '':
                    file_path = __import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '../raw data collected for project/cybersecurity_intrusion_data.csv'))
                data = load_data(file_path)
                if data is not None:
                    display_columns_with_numbers(data)
                    
            elif choice == '2':
                if data is not None:
                    display_columns_with_numbers(data)
                    col_num = int(input("Enter the column number for the target variable: "))
                    if 0 <= col_num < len(data.columns):
                        target_col = data.columns[col_num]
                        print(f"Target column set to: {target_col}")
                    else:
                        print("Invalid column number.")
                else:
                    print("Please load the dataset first.")
                    
            elif choice == '3':
                if data is not None:
                    display_columns_with_numbers(data)
                    col_nums_input = input("Enter feature column numbers separated by commas: ")
                    try:
                        col_nums = [int(num.strip()) for num in col_nums_input.split(',')]
                        if all(0 <= num < len(data.columns) for num in col_nums):
                            feature_cols = [data.columns[num] for num in col_nums]
                            print(f"Feature columns set to: {feature_cols}")
                        else:
                            print("One or more column numbers are invalid.")
                    except ValueError:
                        print("Invalid input format. Please enter numbers separated by commas.")
                else:
                    print("Please load the dataset first.")
                    
            elif choice in ['4', '5', '6']:
                if data is not None and target_col is not None and feature_cols is not None:
                    
                    # Configuration based on choice
                    if choice == '4':
                        balance_data = False
                        tune_hyperparams = False
                        scaling_method = 'standard'
                        feature_selection = False
                    elif choice == '5':
                        balance_data = True
                        tune_hyperparams = False
                        scaling_method = 'robust'
                        feature_selection = True
                    else:  # choice == '6'
                        balance_data = True
                        tune_hyperparams = True
                        scaling_method = 'robust'
                        feature_selection = True
                    
                    print(f"\nProcessing with advanced features: {choice != '4'}")
                    print(f"Class balancing: {balance_data}")
                    print(f"Hyperparameter tuning: {tune_hyperparams}")
                    
                    # Preprocess data
                    X_scaled, y, feature_names = preprocess_data(
                        data, target_col, feature_cols, 
                        scaling_method=scaling_method, 
                        feature_selection=feature_selection
                    )
                    
                    if X_scaled is not None and y is not None:
                        # Split data with stratification
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y, test_size=0.2, random_state=42, stratify=y
                        )
                        
                        print(f"Training set size: {X_train.shape}")
                        print(f"Test set size: {X_test.shape}")
                        print(f"Class distribution - Train: {np.bincount(y_train)}")
                        print(f"Class distribution - Test: {np.bincount(y_test)}")
                        
                        # Train models
                        ensemble, results = train_and_evaluate_models(
                            X_train, X_test, y_train, y_test,
                            balance_data=balance_data,
                            tune_hyperparams=tune_hyperparams
                        )
                        
                        if results:
                            print("\n" + "="*50)
                            print("FINAL RESULTS SUMMARY")
                            print("="*50)
                            for model_name, metrics in results.items():
                                print(f"{model_name}: Accuracy = {metrics['accuracy']:.4f}, AUC = {metrics['auc']:.4f}")
                    else:
                        print("Error in data preprocessing.")
                else:
                    print("Please ensure dataset, target, and feature columns are selected first.")
                    
            elif choice == '7':
                print("Exiting.")
                break
            else:
                print("Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\nOperation interrupted by user.")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()
