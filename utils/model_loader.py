"""
Model Loading and Management for Credit Risk Assessment Streamlit App
"""

import pandas as pd
import numpy as np
import joblib
import streamlit as st
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

@st.cache_data
def load_dataset(file_path="dataset_banco.csv"):
    """Load the credit risk assessment dataset."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Dataset not found at {file_path}")
        return None

def prepare_data(df):
    """Prepare data for machine learning with proper train/test split."""
    # Use 'default' as target for credit risk assessment
    # Handle case where 'y' might have been removed during cleaning
    columns_to_drop = ['default']
    if 'y' in df.columns:
        columns_to_drop.append('y')
    
    X = df.drop(columns_to_drop, axis=1)
    y = df['default'].apply(lambda x: 1 if x == 'yes' else 0)
    
    # Split data BEFORE preprocessing to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def create_preprocessor(X_train):
    """Create preprocessing pipeline."""
    # Identify feature types
    categorical_features = X_train.select_dtypes(include='object').columns
    numerical_features = X_train.select_dtypes(include=np.number).columns
    
    # Numerical preprocessing
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing (smart encoding strategy)
    onehot_features = [col for col in categorical_features if X_train[col].nunique() <= 10]
    target_features = [col for col in categorical_features if X_train[col].nunique() > 10]
    
    categorical_transformer = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'), onehot_features),
            ('target', TargetEncoder(), target_features)
        ],
        remainder='passthrough'
    )
    
    # Combined preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def apply_preprocessing(preprocessor, X_train, X_test, y_train):
    """Apply preprocessing and handle class imbalance."""
    # Fit on training data only (prevents data leakage)
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Handle class imbalance with SMOTE (training set only)
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
    
    return X_train_balanced, X_test_processed, y_train_balanced

def train_models(X_train, y_train):
    """Train multiple models and compare performance."""
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_results = {}
    
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring='roc_auc'
        )
        
        cv_results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model': model
        }
    
    return models, cv_results

def evaluate_models(models, cv_results, X_train, y_train, X_test, y_test):
    """Evaluate models on test set and select best performer."""
    test_results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Train on full balanced training set
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict on test set
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_proba)
        test_results[name] = test_auc
    
    # Select best model
    best_model_name = max(test_results, key=test_results.get)
    best_model = trained_models[best_model_name]
    
    return best_model, best_model_name, test_results, cv_results, trained_models

@st.cache_resource
def train_and_cache_models(df):
    """Train models and cache results for Streamlit."""
    # VALIDATION: Ensure we're working with the cleaned dataset
    expected_columns = {'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan'}
    actual_columns = set(df.columns)
    
    print(f"🔍 Model training - Expected columns: {expected_columns}")
    print(f"🔍 Model training - Actual columns: {actual_columns}")
    
    if not expected_columns.issubset(actual_columns):
        missing = expected_columns - actual_columns
        print(f"❌ ERROR: Missing expected columns: {missing}")
    
    unwanted_columns = actual_columns - expected_columns
    if unwanted_columns:
        print(f"❌ ERROR: Unwanted columns still present: {unwanted_columns}")
        print("❌ These should have been removed during cleaning!")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Create and apply preprocessor
    preprocessor = create_preprocessor(X_train)
    X_train_processed, X_test_processed, y_train_balanced = apply_preprocessing(
        preprocessor, X_train, X_test, y_train
    )
    
    # Train models
    models, cv_results = train_models(X_train_processed, y_train_balanced)
    
    # Evaluate models
    best_model, best_model_name, test_results, cv_results, trained_models = evaluate_models(
        models, cv_results, X_train_processed, y_train_balanced, X_test_processed, y_test
    )
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'preprocessor': preprocessor,
        'cv_results': cv_results,
        'test_results': test_results,
        'trained_models': trained_models,
        'test_data': (X_test_processed, y_test),
        'feature_names': X_train.columns.tolist(),
        'raw_data': (X_train, y_train),  # Add raw data for feature importance analysis
        'original_df': df  # Add original dataframe
    }

def predict_single_case(model, preprocessor, input_data):
    """Make prediction for a single case."""
    # Convert input to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Preprocess
    X_processed = preprocessor.transform(df_input)
    
    # Predict
    probability = model.predict_proba(X_processed)[0, 1]
    prediction = model.predict(X_processed)[0]
    
    return probability, prediction

def get_risk_interpretation(probability):
    """Interpret risk probability for business users."""
    prob_percent = probability * 100
    
    if prob_percent < 20:
        return "🟢 Low Risk", "Automatic approval recommended", "green"
    elif prob_percent < 40:
        return "🟡 Medium Risk", "Manual review required", "orange"
    else:
        return "🔴 High Risk", "Likely rejection", "red"

def analyze_feature_importance_multiple_methods(df, best_model, preprocessor, X_train, y_train, top_n=6):
    """
    Use multiple statistical techniques to find the most important features:
    1. Logistic Regression Coefficients (for linear models)
    2. Mutual Information (statistical dependency)
    3. Chi-Square Test (for categorical features)
    4. Correlation Analysis
    5. Permutation Importance (model-agnostic)
    """
    from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
    from sklearn.inspection import permutation_importance
    from scipy.stats import chi2_contingency
    import warnings
    warnings.filterwarnings('ignore')
    
    # VALIDATION: Check what columns are actually in the dataset
    print(f"🔍 Dataset columns received: {list(df.columns)}")
    
    # Expected feature names (after removing irrelevant marketing columns)
    expected_features = ['age', 'job', 'marital', 'education', 'default', 'balance', 
                        'housing', 'loan']
    
    # Validate that marketing columns were actually removed
    marketing_columns = ['contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
    found_marketing_cols = [col for col in marketing_columns if col in df.columns]
    
    if found_marketing_cols:
        print(f"❌ ERROR: Marketing columns still present: {found_marketing_cols}")
        print("❌ Dataset cleaning failed - these should have been removed!")
    else:
        print("✅ Marketing columns successfully removed")
    
    # Prepare clean data for analysis
    columns_to_drop = ['default']
    if 'y' in df.columns:
        columns_to_drop.append('y')
    
    X_clean = df.drop(columns_to_drop, axis=1)
    y_clean = df['default'].apply(lambda x: 1 if x == 'yes' else 0)
    
    print(f"🔍 Features for analysis: {list(X_clean.columns)}")
    
    feature_scores = {}
    
    # Method 1: Mutual Information (works with mixed data types)
    try:
        # Encode categorical variables for mutual info
        X_encoded = X_clean.copy()
        for col in X_clean.select_dtypes(include=['object']).columns:
            X_encoded[col] = pd.factorize(X_clean[col])[0]
        
        mi_scores = mutual_info_classif(X_encoded, y_clean, random_state=42)
        for i, feature in enumerate(X_clean.columns):
            feature_scores[feature] = feature_scores.get(feature, 0) + mi_scores[i]
    except Exception as e:
        print(f"Mutual Information failed: {e}")
    
    # Method 2: Correlation Analysis (for numerical features)
    try:
        numerical_features = X_clean.select_dtypes(include=[np.number]).columns
        for feature in numerical_features:
            corr = abs(X_clean[feature].corr(y_clean))
            if not np.isnan(corr):
                feature_scores[feature] = feature_scores.get(feature, 0) + corr
    except Exception as e:
        print(f"Correlation analysis failed: {e}")
    
    # Method 3: Chi-Square for categorical features
    try:
        categorical_features = X_clean.select_dtypes(include=['object']).columns
        for feature in categorical_features:
            # Create contingency table
            contingency_table = pd.crosstab(X_clean[feature], y_clean)
            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
                # Use chi2 statistic as importance score
                feature_scores[feature] = feature_scores.get(feature, 0) + chi2_stat / 1000  # Normalize
    except Exception as e:
        print(f"Chi-square analysis failed: {e}")
    
    # Method 4: Logistic Regression Coefficients (if available)
    try:
        if hasattr(best_model, 'coef_'):
            # Train a simple logistic regression on original features for coefficient analysis
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            
            X_temp = X_clean.copy()
            # Encode categorical variables
            label_encoders = {}
            for col in X_temp.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X_temp[col] = le.fit_transform(X_temp[col].astype(str))
                label_encoders[col] = le
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_temp)
            
            # Train simple logistic regression
            lr_temp = LogisticRegression(random_state=42, max_iter=1000)
            lr_temp.fit(X_scaled, y_clean)
            
            # Get coefficients
            coefficients = np.abs(lr_temp.coef_[0])
            for i, feature in enumerate(X_temp.columns):
                feature_scores[feature] = feature_scores.get(feature, 0) + coefficients[i]
                
    except Exception as e:
        print(f"Logistic regression coefficient analysis failed: {e}")
    
    # Method 5: Permutation Importance (model-agnostic, most reliable)
    try:
        # Use a subset for faster computation
        sample_size = min(5000, len(X_train))
        X_sample = X_train.sample(n=sample_size, random_state=42)
        y_sample = y_train[X_sample.index]
        
        # Apply preprocessing
        X_processed_sample = preprocessor.transform(X_sample)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            best_model, X_processed_sample, y_sample, 
            n_repeats=5, random_state=42, scoring='roc_auc'
        )
        
        # Map back to original features (complex due to preprocessing)
        # For now, use the mean importance and distribute across original features
        mean_importance = np.mean(perm_importance.importances_mean)
        for feature in original_features:
            if feature in X_clean.columns:
                feature_scores[feature] = feature_scores.get(feature, 0) + mean_importance
                
    except Exception as e:
        print(f"Permutation importance failed: {e}")
    
    # Combine and rank features
    if feature_scores:
        # Normalize scores
        max_score = max(feature_scores.values())
        if max_score > 0:
            for feature in feature_scores:
                feature_scores[feature] = feature_scores[feature] / max_score
        
        # Sort by importance
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        top_features = [feature for feature, score in sorted_features[:top_n]]
        
        # Print results for debugging
        print("🔍 FEATURE IMPORTANCE ANALYSIS RESULTS:")
        print("=" * 50)
        for i, (feature, score) in enumerate(sorted_features[:10]):
            print(f"{i+1:2d}. {feature:<12} : {score:.4f}")
        print("=" * 50)
        
        return top_features, dict(sorted_features)
    
    else:
        # Fallback to domain knowledge
        print("⚠️ All statistical methods failed, using domain knowledge fallback")
        return ['balance', 'age', 'job', 'education', 'housing', 'duration'][:top_n], {}

def get_most_important_features(best_model, preprocessor, X_train, y_train, df, top_n=6):
    """Get the most important features using comprehensive statistical analysis."""
    if X_train is not None and y_train is not None and df is not None:
        # CRITICAL: Use the CLEANED dataset, not the original one
        top_features, importance_scores = analyze_feature_importance_multiple_methods(
            df, best_model, preprocessor, X_train, y_train, top_n
        )
        
        # Additional validation: ensure no removed columns appear
        valid_columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan']
        filtered_features = [f for f in top_features if f in valid_columns and f != 'default']
        
        print(f"🔍 Feature validation: Original={top_features}, Filtered={filtered_features}")
        
        return filtered_features[:top_n]
    else:
        # Fallback to credit risk assessment relevant features only
        return ['balance', 'age', 'job', 'education', 'housing', 'loan'][:top_n]

def predict_with_top_features(model, preprocessor, input_data, top_features):
    """Make prediction using only the top important features + required defaults."""
    # Complete input with defaults for all features (required for preprocessing)
    complete_input = get_complete_feature_set(input_data, top_features)
    
    # Convert to DataFrame
    df_input = pd.DataFrame([complete_input])
    
    # Preprocess (needs all features for the trained preprocessor)
    X_processed = preprocessor.transform(df_input)
    
    # Predict
    probability = model.predict_proba(X_processed)[0, 1]
    prediction = model.predict(X_processed)[0]
    
    return probability, prediction

def get_complete_feature_set(user_input, top_features):
    """Complete user input with defaults for preprocessing compatibility."""
    # Defaults for all features (needed for preprocessor)
    defaults = {
        'age': 35,
        'job': 'management',
        'marital': 'married', 
        'education': 'secondary',
        'default': 'no',
        'balance': 1000.0,
        'housing': 'yes',
        'loan': 'no',
        'contact': 'cellular',
        'day': 15,
        'month': 'may',
        'duration': 200.0,
        'campaign': 1,
        'pdays': -1.0,
        'previous': 0,
        'poutcome': 'unknown'
    }
    
    # Start with defaults
    complete_input = defaults.copy()
    
    # Override only the important features with user input
    for feature in top_features:
        if feature in user_input:
            complete_input[feature] = user_input[feature]
    
    return complete_input