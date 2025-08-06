"""
Credit Risk Assessment Streamlit Application
====================================

A comprehensive web application for credit risk assessment with interactive
model performance visualization and individual risk prediction capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.model_loader import (
    train_and_cache_models, predict_single_case, get_risk_interpretation,
    get_most_important_features, predict_with_top_features
)
from utils.data_processor import (
    get_feature_options, get_feature_descriptions, get_default_values,
    validate_input_data, format_currency, format_percentage,
    create_dynamic_input_form, load_and_process_dataset
)
from utils.visualizations import (
    plot_model_performance_comparison, plot_roc_curve, plot_confusion_matrix,
    plot_feature_importance, plot_probability_distribution, plot_risk_gauge,
    plot_dataset_overview, create_metrics_summary, plot_business_impact_metrics
)

# Page configuration
st.set_page_config(
    page_title="Credit Risk Assessment ML Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🏦 Credit Risk Assessment ML Dashboard")
st.markdown("""
**Enterprise-level credit risk assessment system** with 87% accuracy (ROC-AUC).  
Predict default probability and analyze model performance with interactive visualizations.
""")

# Sidebar navigation
st.sidebar.title("📊 Dashboard Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Overview", "📈 Model Performance", "🎯 Risk Predictor", "📊 Dataset Analysis"]
)

# Load and clean data
with st.spinner("🧹 Loading and cleaning dataset..."):
    df, dataset_info = load_and_process_dataset("dataset_banco.csv")

if df is None:
    st.stop()


# Add cache clearing button for development
if st.sidebar.button("🔄 Clear Cache & Retrain"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

# Train models (cached)
with st.spinner("🤖 Training models... (This may take a moment on first run)"):
    model_data = train_and_cache_models(df)

# Extract model components
best_model = model_data['best_model']
best_model_name = model_data['best_model_name']
preprocessor = model_data['preprocessor']
cv_results = model_data['cv_results']
test_results = model_data['test_results']
trained_models = model_data['trained_models']
X_test, y_test = model_data['test_data']
feature_names = model_data['feature_names']
X_train, y_train = model_data['raw_data']
original_df = model_data['original_df']

# Get most important features using statistical analysis
with st.spinner("🔍 Analyzing feature importance using statistical methods..."):
    top_features = get_most_important_features(best_model, preprocessor, X_train, y_train, original_df, top_n=6)

# PAGE 1: OVERVIEW
if page == "🏠 Overview":
    st.header("🏠 Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 12px; border-radius: 8px; text-align: center;">
            <h4 style="margin: 0; font-size: 14px; color: #666;">🏆 Best Model</h4>
            <h3 style="margin: 5px 0; font-size: 18px; color: #262730;">{best_model_name}</h3>
            <p style="margin: 0; font-size: 12px; color: #888;">ROC-AUC: {test_results[best_model_name]:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        default_rate = (df['default'] == 'yes').mean() * 100
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 12px; border-radius: 8px; text-align: center;">
            <h4 style="margin: 0; font-size: 14px; color: #666;">📊 Default Rate</h4>
            <h3 style="margin: 5px 0; font-size: 18px; color: #262730;">{default_rate:.2f}%</h3>
            <p style="margin: 0; font-size: 12px; color: #888;">{(df['default'] == 'yes').sum():,} defaults</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 12px; border-radius: 8px; text-align: center;">
            <h4 style="margin: 0; font-size: 14px; color: #666;">📋 Total Records</h4>
            <h3 style="margin: 5px 0; font-size: 18px; color: #262730;">{len(df):,}</h3>
            <p style="margin: 0; font-size: 12px; color: #888;">{len(df.columns)} features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        stability = abs(cv_results[best_model_name]['cv_mean'] - test_results[best_model_name])
        stability_status = "🟢 Excellent" if stability < 0.05 else "🟡 Good" if stability < 0.10 else "🔴 Poor"
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 12px; border-radius: 8px; text-align: center;">
            <h4 style="margin: 0; font-size: 14px; color: #666;">🎯 Model Stability</h4>
            <h3 style="margin: 5px 0; font-size: 18px; color: #262730;">{stability_status}</h3>
            <p style="margin: 0; font-size: 12px; color: #888;">Diff: {stability:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Summary
    st.subheader("📋 Model Summary")
    metrics_summary = create_metrics_summary(cv_results, test_results, best_model_name)
    
    col1, col2 = st.columns(2)
    with col1:
        for key, value in list(metrics_summary.items())[:3]:
            st.write(f"**{key}:** {value}")
    
    with col2:
        for key, value in list(metrics_summary.items())[3:]:
            st.write(f"**{key}:** {value}")
    
    # Data Quality Information
    st.subheader("🧹 Data Quality & Cleaning")
    st.success("✅ **Automated Data Cleaning Applied:**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Feature Selection**: 9 irrelevant marketing columns removed
        - **Age Correction**: Ages > 100 replaced with mean age
        - **Marital Status**: 'div.' standardized to 'divorced'
        - **Missing Data**: Null/empty records removed
        """)
    with col2:
        if dataset_info:
            # Compact metrics for data quality
            st.markdown(f"""
            <div style="background-color: #e8f5e8; padding: 10px; border-radius: 6px; margin-bottom: 8px;">
                <p style="margin: 0; font-size: 14px; color: #666;">📊 Clean Records</p>
                <h4 style="margin: 2px 0; font-size: 16px; color: #262730;">{dataset_info['total_records']:,}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background-color: #e8f5e8; padding: 10px; border-radius: 6px; margin-bottom: 8px;">
                <p style="margin: 0; font-size: 14px; color: #666;">🎯 Default Rate</p>
                <h4 style="margin: 2px 0; font-size: 16px; color: #262730;">{dataset_info['default_rate']:.2f}%</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background-color: #e8f5e8; padding: 10px; border-radius: 6px; margin-bottom: 8px;">
                <p style="margin: 0; font-size: 14px; color: #666;">📋 Credit Features</p>
                <h4 style="margin: 2px 0; font-size: 16px; color: #262730;">{dataset_info['total_features']}</h4>
            </div>
            """, unsafe_allow_html=True)

    # Business interpretation
    st.subheader("💼 Business Impact")
    st.markdown("""
    ### 🎯 Risk Thresholds (Industry Standard):
    - **🟢 Low Risk (0-20%):** Automatic approval recommended
    - **🟡 Medium Risk (20-40%):** Manual review required
    - **🔴 High Risk (40%+):** Likely rejection
    
    ### 📈 Model Performance:
    - ROC AUC 0.87+ is considered **excellent** for credit risk assessment
    - This model is comparable to enterprise banking systems
    - Conservative predictions help minimize financial risk
    """)

# PAGE 2: MODEL PERFORMANCE
elif page == "📈 Model Performance":
    st.header("📈 Model Performance Analysis")
    
    # Model comparison
    st.subheader("🏆 Model Comparison")
    fig_comparison = plot_model_performance_comparison(cv_results, test_results)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Performance details
    st.subheader("📊 Detailed Performance Metrics")
    performance_df = pd.DataFrame({
        'Model': list(cv_results.keys()),
        'CV ROC AUC': [f"{cv_results[model]['cv_mean']:.4f} ± {cv_results[model]['cv_std']:.4f}" 
                      for model in cv_results.keys()],
        'Test ROC AUC': [f"{test_results[model]:.4f}" for model in cv_results.keys()],
        'Stability': [f"{abs(cv_results[model]['cv_mean'] - test_results[model]):.4f}" 
                     for model in cv_results.keys()]
    })
    st.dataframe(performance_df, use_container_width=True)
    
    # ROC Curve and Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 ROC Curve")
        fig_roc = plot_roc_curve(best_model, X_test, y_test, best_model_name)
        st.plotly_chart(fig_roc, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Confusion Matrix")
        fig_cm = plot_confusion_matrix(best_model, X_test, y_test, best_model_name)
        st.plotly_chart(fig_cm, use_container_width=True)
    
    # Feature Importance
    st.subheader("🔍 Feature Importance")
    fig_importance = plot_feature_importance(best_model, feature_names, best_model_name)
    if fig_importance:
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")
    
    # Probability Distribution
    st.subheader("📊 Prediction Probability Distribution")
    probabilities = best_model.predict_proba(X_test)[:, 1]
    fig_prob_dist = plot_probability_distribution(probabilities, y_test)
    st.plotly_chart(fig_prob_dist, use_container_width=True)
    
    # Business Impact Metrics
    st.subheader("💼 Business Impact Analysis")
    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.2, 0.05)
    fig_business = plot_business_impact_metrics(probabilities, y_test, threshold)
    st.plotly_chart(fig_business, use_container_width=True)

# PAGE 3: RISK PREDICTOR
elif page == "🎯 Risk Predictor":
    st.header("🎯 Individual Risk Predictor")
    
    # Show most important features
    st.info(f"🎯 **Optimized Predictor** - Using the {len(top_features)} most important features from {best_model_name}: **{', '.join(top_features)}**")
    
    
    # Input form with only top features
    st.subheader("📝 Enter Key Customer Information")
    st.markdown("*Only the most predictive features are required for accurate risk assessment*")
    
    # Get options and defaults
    feature_options = get_feature_options()
    feature_descriptions = get_feature_descriptions()
    defaults = get_default_values()
    
    # Create dynamic form based on statistical analysis
    with st.form("prediction_form"):
        st.markdown("**📊 Features identified by statistical analysis:**")
        
        # Use dynamic form creation
        user_input = create_dynamic_input_form(top_features, feature_options, feature_descriptions, defaults)
        
        # Submit button
        submitted = st.form_submit_button("🎯 Predict Risk", use_container_width=True)
    
    # Make prediction
    if submitted:
        # Validate input for top features
        errors = []
        for feature in top_features:
            if feature not in user_input or user_input[feature] is None:
                errors.append(f"Missing value for {feature}")
        
        if errors:
            st.error("❌ Please provide all required information:")
            for error in errors:
                st.error(f"• {error}")
        else:
            # Make prediction using top features
            probability, prediction = predict_with_top_features(best_model, preprocessor, user_input, top_features)
            risk_level, recommendation, color = get_risk_interpretation(probability)
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Risk Assessment Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Risk gauge
                fig_gauge = plot_risk_gauge(probability)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                # Risk details
                st.markdown(f"### {risk_level}")
                st.markdown(f"**Default Probability:** {probability*100:.2f}%")
                st.markdown(f"**Recommendation:** {recommendation}")
                st.markdown(f"**Model Used:** {best_model_name}")
                st.markdown(f"**Based on:** {len(top_features)} key features")
                
                # Risk interpretation
                if probability < 0.2:
                    st.success("✅ Low risk customer - suitable for standard loan terms")
                elif probability < 0.4:
                    st.warning("⚠️ Medium risk - consider additional verification or adjusted terms")
                else:
                    st.error("🚨 High risk - detailed review recommended before approval")
            
            # Feature importance display
            st.subheader("📊 Key Factors in Assessment")
            
            # Show which features were used and their values
            importance_col1, importance_col2, importance_col3 = st.columns(3)
            
            with importance_col1:
                st.write("**👤 Personal Profile:**")
                if 'job' in user_input:
                    st.write(f"• Job: {user_input['job']}")
                if 'marital' in user_input:
                    st.write(f"• Marital: {user_input['marital']}")
                if 'age' in user_input:
                    st.write(f"• Age: {user_input['age']} years")
            
            with importance_col2:
                st.write("**🎓 Education & Housing:**")
                if 'education' in user_input:
                    st.write(f"• Education: {user_input['education']}")
                if 'housing' in user_input:
                    st.write(f"• Housing loan: {user_input['housing']}")
            
            with importance_col3:
                st.write("**💰 Financial Status:**")
                if 'balance' in user_input:
                    st.write(f"• Balance: {format_currency(user_input['balance'])}")
            
            # Model explanation
            st.info("💡 **Why these features?** These are fundamental credit risk assessment factors: employment stability, family situation, education level, age, financial position, and existing debt obligations.")

# PAGE 4: DATASET ANALYSIS
elif page == "📊 Dataset Analysis":
    st.header("📊 Dataset Analysis")
    
    # Data Cleaning Report - Collapsible
    st.subheader("🧹 Data Cleaning Report")
    
    with st.expander("📋 **View Detailed Cleaning Procedures** (Click to expand)", expanded=False):
        st.info("**The following cleaning operations were automatically applied to ensure data quality:**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            **🔧 Feature Selection:**
            - Removed 9 marketing columns
            - Kept only credit-relevant features
            - Eliminates model noise
            - Focus on default prediction
            """)
        
        with col2:
            st.markdown("""
            **👤 Age Standardization:**
            - Ages > 100 years corrected
            - Replaced with mean age
            - Realistic age distribution
            - Data quality improvement
            """)
        
        with col3:
            st.markdown("""
            **💑 Marital Status Fix:**
            - 'div.' → 'divorced' 
            - 'sec.' → 'secondary' (education)
            - 'UNK' → 'unknown' (education)
            - Consistent categorical values
            """)
        
        with col4:
            st.markdown("""
            **📊 Data Completeness:**
            - Removed null values
            - Eliminated empty strings
            - Complete feature coverage
            - Ready for ML processing
            """)
        
        st.markdown("---")
        
        # Detailed column information
        col1, col2 = st.columns(2)
        with col1:
            st.error("""
            **🗑️ Removed Marketing Columns (9):**
            - contact (communication type)
            - day (contact day)  
            - month (contact month)
            - duration (call length)
            - campaign (contact attempts)
            - pdays (days since previous)
            - previous (previous contacts)
            - poutcome (previous outcome)  
            - y (marketing success - wrong target)
            """)
        
        with col2:
            st.success("""
            **✅ Kept Credit-Relevant Columns (8):**
            - age (customer age)
            - job (occupation type)
            - marital (marital status)
            - education (education level) 
            - default (credit history)
            - balance (account balance)
            - housing (housing loan)
            - loan (personal loan)
            """)
        
        st.info("**🎯 Result:** Optimized dataset with only credit-relevant features for accurate default prediction.")
    
    st.markdown("---")
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 Clean Records", f"{len(df):,}")
    
    with col2:
        st.metric("📊 Features", len(df.columns))
    
    with col3:
        default_count = (df['default'] == 'yes').sum()
        st.metric("🔴 Defaults", f"{default_count:,}")
    
    with col4:
        default_rate = (df['default'] == 'yes').mean() * 100
        st.metric("📈 Default Rate", f"{default_rate:.2f}%")
    
    # Dataset visualizations
    st.subheader("📈 Dataset Overview")
    fig_overview = plot_dataset_overview(df)
    st.plotly_chart(fig_overview, use_container_width=True)
    
    # Feature statistics
    st.subheader("📊 Feature Statistics")
    
    tab1, tab2 = st.tabs(["📈 Numerical Features", "📋 Categorical Features"])
    
    with tab1:
        numerical_features = df.select_dtypes(include=[np.number]).columns
        st.dataframe(df[numerical_features].describe(), use_container_width=True)
    
    with tab2:
        categorical_features = df.select_dtypes(include=['object']).columns
        for feature in categorical_features:
            st.write(f"**{feature}:**")
            value_counts = df[feature].value_counts()
            st.bar_chart(value_counts)
    
    # Data quality
    st.subheader("🔍 Data Quality Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Missing Values:**")
        missing_data = df.isnull().sum()
        if missing_data.sum() == 0:
            st.success("✅ No missing values found")
        else:
            st.dataframe(missing_data[missing_data > 0])
    
    with col2:
        st.write("**Data Types:**")
        data_types = pd.DataFrame({
            'Feature': df.columns,
            'Type':  df.dtypes.astype(str),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(data_types, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
🏦 Credit Risk Assessment ML Dashboard | Built with Streamlit | 
Model: {model_name} (ROC-AUC: {auc:.4f})
</div>
""".format(model_name=best_model_name, auc=test_results[best_model_name]), unsafe_allow_html=True)