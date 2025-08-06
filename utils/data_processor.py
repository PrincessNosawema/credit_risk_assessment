"""
Data Processing Utilities for Credit Risk Assessment Streamlit App
"""

import pandas as pd
import numpy as np
import streamlit as st

def validate_input_data(input_data, feature_names):
    """Validate user input data for predictions."""
    errors = []
    
    # Check for required fields
    for feature in feature_names:
        if feature not in input_data or input_data[feature] is None:
            errors.append(f"Missing value for {feature}")
    
    # Specific validations based on feature types
    if 'age' in input_data:
        if not (18 <= input_data['age'] <= 100):
            errors.append("Age must be between 18 and 100")
    
    if 'balance' in input_data:
        if input_data['balance'] < -10000:
            errors.append("Balance seems unrealistic (too negative)")
    
    if 'duration' in input_data:
        if input_data['duration'] < 0:
            errors.append("Duration cannot be negative")
    
    return errors

def get_feature_options():
    """Get predefined options for categorical features."""
    return {
        'job': ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 
                'management', 'retired', 'self-employed', 'services', 
                'student', 'technician', 'unemployed', 'unknown'],
        
        'marital': ['divorced', 'married', 'single'],
        
        'education': ['primary', 'secondary', 'tertiary', 'unknown'],
        
        'default': ['no', 'yes'],
        
        'housing': ['no', 'yes'],
        
        'loan': ['no', 'yes'],
        
        'contact': ['cellular', 'telephone', 'unknown'],
        
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        
        'poutcome': ['failure', 'other', 'success', 'unknown']
    }

def get_feature_descriptions():
    """Get descriptions for each feature to help users."""
    return {
        'age': 'Age of the customer (years)',
        'job': 'Type of job',
        'marital': 'Marital status',
        'education': 'Education level',
        'default': 'Has credit in default?', 
        'balance': 'Average yearly balance (euros)',
        'housing': 'Has housing loan?',
        'loan': 'Has personal loan?',
        'contact': 'Contact communication type',
        'day': 'Last contact day of the month',
        'month': 'Last contact month of year',
        'duration': 'Last contact duration (seconds)',
        'campaign': 'Number of contacts performed during this campaign',
        'pdays': 'Number of days since previous campaign contact (-1 = not contacted)',
        'previous': 'Number of contacts performed before this campaign',
        'poutcome': 'Outcome of the previous marketing campaign'
    }

def get_default_values():
    """Get reasonable default values for the form."""
    return {
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

def process_dataset_info(df):
    """Process and return dataset information for dashboard."""
    info = {}
    
    # Basic info
    info['total_records'] = len(df)
    info['total_features'] = len(df.columns)
    
    # Target variable analysis
    info['default_rate'] = (df['default'] == 'yes').mean() * 100
    info['default_count'] = (df['default'] == 'yes').sum()
    info['no_default_count'] = (df['default'] == 'no').sum()
    
    # Missing values
    info['missing_values'] = df.isnull().sum().sum()
    info['missing_percentage'] = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    
    # Feature types
    info['categorical_features'] = df.select_dtypes(include='object').columns.tolist()
    info['numerical_features'] = df.select_dtypes(include=np.number).columns.tolist()
    
    # Statistical summary for numerical features
    info['numerical_summary'] = df.select_dtypes(include=np.number).describe()
    
    return info


def format_currency(value):
    """Format currency values for display."""
    if value >= 0:
        return f"€{value:,.2f}"
    else:
        return f"-€{abs(value):,.2f}"

def format_percentage(value, decimals=2):
    """Format percentage values for display."""
    return f"{value:.{decimals}f}%"

def create_dynamic_input_form(top_features, feature_options, feature_descriptions, defaults):
    """Create dynamic input form based on statistically important features."""
    import streamlit as st
    
    user_input = {}
    col1, col2 = st.columns(2)
    
    # Create inputs dynamically based on the top features identified
    for i, feature in enumerate(top_features):
        column = col1 if i % 2 == 0 else col2
        
        with column:
            if feature == 'age':
                user_input[feature] = st.number_input(
                    "👤 Age", 
                    min_value=18, max_value=100,
                    value=st.session_state.get(feature, defaults[feature]),
                    help=feature_descriptions[feature]
                )
            
            elif feature == 'balance':
                user_input[feature] = st.number_input(
                    "💰 Account Balance (€)", 
                    value=float(st.session_state.get(feature, defaults[feature])),
                    help=feature_descriptions[feature]
                )
            
            elif feature == 'duration':
                user_input[feature] = st.number_input(
                    "📞 Call Duration (seconds)", 
                    min_value=0.0, max_value=5000.0,
                    value=float(st.session_state.get(feature, defaults[feature])),
                    help=feature_descriptions[feature]
                )
            
            elif feature == 'campaign':
                user_input[feature] = st.number_input(
                    "📢 Campaign Contacts", 
                    min_value=1, max_value=20,
                    value=st.session_state.get(feature, defaults[feature]),
                    help=feature_descriptions[feature]
                )
            
            elif feature == 'pdays':
                user_input[feature] = st.number_input(
                    "📅 Days Since Previous Campaign", 
                    value=float(st.session_state.get(feature, defaults[feature])),
                    help=feature_descriptions[feature] + " (-1 means never contacted)"
                )
            
            elif feature == 'previous':
                user_input[feature] = st.number_input(
                    "📊 Previous Contacts", 
                    min_value=0, max_value=20,
                    value=st.session_state.get(feature, defaults[feature]),
                    help=feature_descriptions[feature]
                )
            
            elif feature in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']:
                # Categorical features
                feature_display_names = {
                    'job': '💼 Job/Occupation',
                    'marital': '💑 Marital Status',
                    'education': '🎓 Education Level',
                    'default': '⚠️ Has Credit Default?',
                    'housing': '🏠 Has Housing Loan?',
                    'loan': '💳 Has Personal Loan?',
                    'contact': '📱 Contact Type',
                    'month': '📅 Contact Month',
                    'poutcome': '📈 Previous Campaign Outcome'
                }
                
                display_name = feature_display_names.get(feature, feature.title())
                
                user_input[feature] = st.selectbox(
                    display_name,
                    feature_options[feature],
                    index=feature_options[feature].index(st.session_state.get(feature, defaults[feature])),
                    help=feature_descriptions[feature]
                )
            
            elif feature == 'day':
                user_input[feature] = st.number_input(
                    "📅 Contact Day", 
                    min_value=1, max_value=31,
                    value=st.session_state.get(feature, defaults[feature]),
                    help=feature_descriptions[feature]
                )
            
            else:
                # Handle any other numerical features
                user_input[feature] = st.number_input(
                    f"📊 {feature.title()}", 
                    value=float(st.session_state.get(feature, defaults.get(feature, 0))),
                    help=feature_descriptions.get(feature, f"Value for {feature}")
                )
    
    return user_input

def clean_dataset(df):
    """
    Clean the dataset according to business rules:
    1. Remove irrelevant columns (marketing campaign variables)
    2. Fix age values > 100 years (replace with mean age)
    3. Replace 'div.' with 'divorced' in marital column
    4. Fix education values 'sec.' -> 'secondary' and 'UNK' -> 'unknown'
    5. Remove records with null/empty values
    """
    print("🧹 Starting dataset cleaning...")
    original_shape = df.shape
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # 1. Remove irrelevant columns for credit risk assessment
    irrelevant_columns = [
        'contact',    # Contact type (cellular/telephone) - marketing variable
        'day',        # Day of contact - arbitrary marketing timing
        'month',      # Month of contact - arbitrary marketing timing  
        'duration',   # Call duration - marketing success metric, causes data leakage
        'campaign',   # Number of contacts - marketing metric
        'pdays',      # Days since previous campaign - marketing history
        'previous',   # Number of previous contacts - marketing metric
        'poutcome',   # Previous campaign outcome - marketing result
        'y'           # Marketing success (deposit subscription) - WRONG TARGET for credit risk assessment
    ]
    
    # Check which columns actually exist in the dataset
    columns_to_remove = [col for col in irrelevant_columns if col in df_clean.columns]
    
    if columns_to_remove:
        print(f"   🗑️ Removing {len(columns_to_remove)} irrelevant marketing columns:")
        for col in columns_to_remove:
            print(f"      - {col}")
        
        df_clean = df_clean.drop(columns=columns_to_remove)
        print(f"   ✅ Dataset reduced from {original_shape[1]} to {df_clean.shape[1]} columns")
        print(f"   📊 Remaining columns: {list(df_clean.columns)}")
    else:
        print("   ℹ️ No irrelevant columns found to remove")
    
    # 2. Fix age values > 100 years
    if 'age' in df_clean.columns:
        ages_over_100 = (df_clean['age'] > 100).sum()
        if ages_over_100 > 0:
            print(f"   📊 Found {ages_over_100} records with age > 100")
            # Calculate mean age for valid ages (18-100)
            valid_ages = df_clean[(df_clean['age'] >= 18) & (df_clean['age'] <= 100)]['age']
            mean_age = valid_ages.mean()
            
            # Replace ages > 100 with mean age
            df_clean.loc[df_clean['age'] > 100, 'age'] = round(mean_age)
            print(f"   ✅ Replaced {ages_over_100} ages > 100 with mean age: {round(mean_age)}")
    
    # 3. Fix marital status 'div.' -> 'divorced'
    if 'marital' in df_clean.columns:
        div_count = (df_clean['marital'] == 'div.').sum()
        if div_count > 0:
            df_clean['marital'] = df_clean['marital'].replace('div.', 'divorced')
            print(f"   ✅ Replaced {div_count} 'div.' entries with 'divorced'")
    
    # 4. Fix education values 'sec.' -> 'secondary' and 'UNK' -> 'unknown'
    if 'education' in df_clean.columns:
        sec_count = (df_clean['education'] == 'sec.').sum()
        unk_count = (df_clean['education'] == 'UNK').sum()
        
        if sec_count > 0:
            df_clean['education'] = df_clean['education'].replace('sec.', 'secondary')
            print(f"   ✅ Replaced {sec_count} 'sec.' entries with 'secondary'")
        
        if unk_count > 0:
            df_clean['education'] = df_clean['education'].replace('UNK', 'unknown')
            print(f"   ✅ Replaced {unk_count} 'UNK' entries with 'unknown'")
    
    # 5. Remove records with null/empty values
    # Check for null values
    null_counts = df_clean.isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls > 0:
        print(f"   📊 Found {total_nulls} null values across columns:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"      - {col}: {count} nulls")
        
        # Remove rows with any null values
        df_clean = df_clean.dropna()
        print(f"   ✅ Removed rows with null values")
    
    # Check for empty strings in object columns
    empty_string_removed = 0
    for col in df_clean.select_dtypes(include=['object']).columns:
        empty_mask = (df_clean[col] == '') | (df_clean[col] == ' ')
        empty_count = empty_mask.sum()
        if empty_count > 0:
            df_clean = df_clean[~empty_mask]
            empty_string_removed += empty_count
            print(f"   ✅ Removed {empty_count} empty strings from column '{col}'")
    
    final_shape = df_clean.shape
    removed_records = original_shape[0] - final_shape[0]
    
    print(f"🎯 Dataset cleaning completed:")
    print(f"   📊 Original: {original_shape[0]:,} records")
    print(f"   📊 Final: {final_shape[0]:,} records")
    print(f"   📊 Removed: {removed_records:,} records ({removed_records/original_shape[0]*100:.2f}%)")
    print("=" * 50)
    
    return df_clean

def validate_cleaned_data(df):
    """Validate that the cleaning was successful."""
    issues = []
    
    # Check age values
    if 'age' in df.columns:
        invalid_ages = ((df['age'] < 18) | (df['age'] > 100)).sum()
        if invalid_ages > 0:
            issues.append(f"Still have {invalid_ages} invalid ages")
    
    # Check marital status
    if 'marital' in df.columns:
        div_entries = (df['marital'] == 'div.').sum()
        if div_entries > 0:
            issues.append(f"Still have {div_entries} 'div.' entries")
    
    # Check education values
    if 'education' in df.columns:
        sec_entries = (df['education'] == 'sec.').sum()
        unk_entries = (df['education'] == 'UNK').sum()
        if sec_entries > 0:
            issues.append(f"Still have {sec_entries} 'sec.' entries in education")
        if unk_entries > 0:
            issues.append(f"Still have {unk_entries} 'UNK' entries in education")
    
    # Check null values
    total_nulls = df.isnull().sum().sum()
    if total_nulls > 0:
        issues.append(f"Still have {total_nulls} null values")
    
    # Check empty strings
    for col in df.select_dtypes(include=['object']).columns:
        empty_count = ((df[col] == '') | (df[col] == ' ')).sum()
        if empty_count > 0:
            issues.append(f"Column '{col}' still has {empty_count} empty strings")
    
    if issues:
        print("⚠️ DATA VALIDATION ISSUES:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Data validation passed - dataset is clean!")
        return True

@st.cache_data
def load_and_process_dataset(file_path="dataset_banco.csv"):
    """Load dataset, clean it, and return processed information."""
    try:
        # Load raw dataset
        df_raw = pd.read_csv(file_path)
        print(f"📊 Loaded raw dataset: {df_raw.shape[0]:,} records, {df_raw.shape[1]} columns")
        
        # Clean the dataset
        df_clean = clean_dataset(df_raw)
        
        # Validate cleaning
        validate_cleaned_data(df_clean)
        
        # Process info
        info = process_dataset_info(df_clean)
        
        return df_clean, info
    except FileNotFoundError:
        st.error(f"Dataset not found at {file_path}")
        return None, None
    except Exception as e:
        st.error(f"Error loading/cleaning dataset: {str(e)}")
        return None, None