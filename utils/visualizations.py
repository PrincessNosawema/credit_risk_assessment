"""
Visualization Utilities for Credit Risk Assessment Streamlit App
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

def plot_model_performance_comparison(cv_results, test_results):
    """Create interactive comparison of model performance."""
    models = list(cv_results.keys())
    cv_scores = [cv_results[model]['cv_mean'] for model in models]
    test_scores = [test_results[model] for model in models]
    cv_stds = [cv_results[model]['cv_std'] for model in models]
    
    fig = go.Figure()
    
    # CV Scores
    fig.add_trace(go.Bar(
        name='Cross-Validation ROC AUC',
        x=models,
        y=cv_scores,
        error_y=dict(type='data', array=cv_stds),
        marker_color='lightblue',
        text=[f'{score:.4f}' for score in cv_scores],
        textposition='auto'
    ))
    
    # Test Scores
    fig.add_trace(go.Bar(
        name='Test ROC AUC',
        x=models,
        y=test_scores,
        marker_color='darkblue',
        text=[f'{score:.4f}' for score in test_scores],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='ROC AUC Score',
        barmode='group',
        height=500
    )
    
    return fig

def plot_roc_curve(model, X_test, y_test, model_name):
    """Create interactive ROC curve."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    fig = go.Figure()
    
    # ROC Curve
    fig.add_trace(go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {auc:.4f})',
        line=dict(color='blue', width=3)
    ))
    
    # Random classifier line
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Random Classifier (AUC = 0.5000)',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600,
        height=500
    )
    
    return fig

def plot_confusion_matrix(model, X_test, y_test, model_name):
    """Create confusion matrix heatmap."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Blues',
        title=f'Confusion Matrix - {model_name}'
    )
    
    fig.update_layout(
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=400,
        height=400
    )
    
    return fig

def plot_feature_importance(model, feature_names, model_name, top_n=10):
    """Plot feature importance if available."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None
    
    # Get top N features
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' 
                   for i in indices]
    top_importances = importances[indices]
    
    fig = go.Figure(go.Bar(
        x=top_importances,
        y=top_features,
        orientation='h',
        marker_color='green'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Feature Importance - {model_name}',
        xaxis_title='Importance',
        yaxis_title='Features',
        height=400
    )
    
    return fig

def plot_probability_distribution(probabilities, actual_labels):
    """Plot distribution of predicted probabilities."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Probability Distribution by Class', 'Overall Distribution')
    )
    
    # Distribution by class
    default_probs = probabilities[actual_labels == 1]
    no_default_probs = probabilities[actual_labels == 0]
    
    fig.add_trace(
        go.Histogram(x=default_probs, name='Default', opacity=0.7, nbinsx=50),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=no_default_probs, name='No Default', opacity=0.7, nbinsx=50),
        row=1, col=1
    )
    
    # Overall distribution
    fig.add_trace(
        go.Histogram(x=probabilities, name='All Predictions', opacity=0.7, nbinsx=50),
        row=2, col=1
    )
    
    fig.update_layout(
        title='Predicted Probability Distributions',
        height=600
    )
    
    return fig

def plot_risk_gauge(probability):
    """Create a gauge chart for risk probability."""
    prob_percent = probability * 100
    
    # Determine color based on risk level
    if prob_percent < 20:
        color = "green"
    elif prob_percent < 40:
        color = "orange"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prob_percent,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Default Risk Probability"},
        delta = {'reference': 20},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 20], 'color': "lightgray"},
                {'range': [20, 40], 'color': "gray"},
                {'range': [40, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def plot_dataset_overview(df):
    """Create overview visualizations of the dataset."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Target Distribution', 'Age Distribution', 
                       'Balance Distribution', 'Job Distribution'),
        specs=[[{"type": "pie"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "bar"}]]
    )
    
    # Target distribution
    target_counts = df['default'].value_counts()
    fig.add_trace(
        go.Pie(labels=target_counts.index, values=target_counts.values, name="Default"),
        row=1, col=1
    )
    
    # Age distribution
    fig.add_trace(
        go.Histogram(x=df['age'], name='Age', nbinsx=30),
        row=1, col=2
    )
    
    # Balance distribution
    fig.add_trace(
        go.Histogram(x=df['balance'], name='Balance', nbinsx=50),
        row=2, col=1
    )
    
    # Job distribution (top 10)
    job_counts = df['job'].value_counts().head(10)
    fig.add_trace(
        go.Bar(x=job_counts.values, y=job_counts.index, orientation='h', name='Job'),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Dataset Overview',
        height=800,
        showlegend=False
    )
    
    return fig

def create_metrics_summary(cv_results, test_results, best_model_name):
    """Create a summary of key metrics."""
    best_cv = cv_results[best_model_name]['cv_mean']
    best_test = test_results[best_model_name]
    stability = abs(best_cv - best_test)
    
    metrics = {
        'Best Model': best_model_name,
        'Cross-Validation ROC AUC': f'{best_cv:.4f}',
        'Test ROC AUC': f'{best_test:.4f}',
        'Model Stability': f'{stability:.4f}',
        'Performance Rating': '🟢 Excellent' if best_test > 0.85 else '🟡 Good' if best_test > 0.75 else '🔴 Needs Improvement'
    }
    
    return metrics

def plot_business_impact_metrics(probabilities, actual_defaults, threshold=0.2):
    """Calculate and visualize business impact metrics."""
    predictions = (probabilities >= threshold).astype(int)
    
    # Calculate business metrics
    tp = np.sum((predictions == 1) & (actual_defaults == 1))  # True Positives
    tn = np.sum((predictions == 0) & (actual_defaults == 0))  # True Negatives
    fp = np.sum((predictions == 1) & (actual_defaults == 0))  # False Positives
    fn = np.sum((predictions == 0) & (actual_defaults == 1))  # False Negatives
    
    # Business metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / len(predictions)
    
    # Create visualization
    metrics = ['Precision', 'Recall', 'Accuracy']
    values = [precision, recall, accuracy]
    
    fig = go.Figure(data=[
        go.Bar(x=metrics, y=values, text=[f'{v:.3f}' for v in values], textposition='auto')
    ])
    
    fig.update_layout(
        title=f'Business Impact Metrics (Threshold: {threshold:.1%})',
        yaxis_title='Score',
        height=400
    )
    
    return fig