import shap
import numpy as np
import logging
from typing import List, Dict, Any
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def explain_with_shap(model_fn, input_vec, background_data, n_samples=50):
    """
    Explain model predictions using SHAP values.
    
    Args:
        model_fn: Function that takes input vector and returns prediction
        input_vec: Input vector to explain
        background_data: Background data for SHAP
        n_samples: Number of background samples to use
        
    Returns:
        Dictionary containing SHAP values and feature importance
    """
    try:
        # Sample background data
        background = shap.sample(background_data, n_samples)
        
        # Create explainer
        explainer = shap.KernelExplainer(model_fn, background)
        
        # Get SHAP values
        shap_values = explainer.shap_values(input_vec)
        
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        return {
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'expected_value': explainer.expected_value
        }
    except Exception as e:
        logger.error(f"Error in explain_with_shap: {str(e)}")
        raise

def analyze_qa_response(response: str, context: str) -> Dict[str, Any]:
    """
    Analyze QA response for key insights.
    
    Args:
        response: Model's answer
        context: Input context
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        # Split response into sentences
        sentences = response.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Extract key terms
        key_terms = set()
        for sentence in sentences:
            words = sentence.split()
            key_terms.update([w.lower() for w in words if len(w) > 3])
        
        # Check context coverage
        context_words = set(context.lower().split())
        coverage = len(key_terms & context_words) / len(key_terms) if key_terms else 0
        
        return {
            'num_sentences': len(sentences),
            'key_terms': list(key_terms),
            'context_coverage': coverage,
            'response_length': len(response)
        }
    except Exception as e:
        logger.error(f"Error in analyze_qa_response: {str(e)}")
        raise

def get_explanation_metrics(ground_truth: str, prediction: str) -> Dict[str, float]:
    """
    Calculate explanation quality metrics.
    
    Args:
        ground_truth: Reference answer
        prediction: Model's answer
        
    Returns:
        Dictionary containing various metrics
    """
    try:
        # Convert to sets of words
        gt_words = set(ground_truth.lower().split())
        pred_words = set(prediction.lower().split())
        
        # Calculate metrics
        precision = len(gt_words & pred_words) / len(pred_words) if pred_words else 0
        recall = len(gt_words & pred_words) / len(gt_words) if gt_words else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'word_overlap': len(gt_words & pred_words)
        }
    except Exception as e:
        logger.error(f"Error in get_explanation_metrics: {str(e)}")
        raise
