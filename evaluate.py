import pandas as pd
import numpy as np
from sklearn.metrics import recall_score
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any
import logging
from qa_pipeline import make_prompt, answer_query
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import json
import nltk
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import wandb
wandb.login(key="643ef2e6e5cdc8482cfe81c6c54752338fe78ef2")

# Download all necessary NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('tokenizers/punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK resources: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced medical terminology for better relevance scoring
ENHANCED_MEDICAL_TERMS = {
    'diabetes': ['diabetes', 'diabetic', 'glucose', 'insulin', 'hyperglycemia', 'hypoglycemia', 'glycemic', 'metabolic'],
    'cardiovascular': ['heart', 'cardiac', 'cardiovascular', 'hypertension', 'blood pressure', 'artery', 'vein', 'circulation'],
    'renal': ['kidney', 'renal', 'nephropathy', 'creatinine', 'glomerular', 'filtration', 'urinary'],
    'ophthalmic': ['eye', 'retinal', 'retinopathy', 'ophthalmic', 'vision', 'visual', 'ocular'],
    'metabolic': ['metabolic', 'metabolism', 'endocrine', 'hormone', 'thyroid', 'pancreatic'],
    'neurological': ['nerve', 'neuropathy', 'neurological', 'sensory', 'motor', 'neural'],
    'complications': ['complication', 'adverse', 'side effect', 'risk factor', 'progression'],
    'diagnosis': ['diagnosis', 'diagnosed', 'condition', 'disorder', 'syndrome', 'pathology'],
    'treatment': ['treatment', 'therapy', 'medication', 'drug', 'prescription', 'intervention'],
    'monitoring': ['monitoring', 'test', 'measurement', 'level', 'value', 'assessment'],
    'clinical': ['clinical', 'medical', 'physician', 'doctor', 'healthcare', 'patient'],
    'symptoms': ['symptom', 'sign', 'manifestation', 'presentation', 'indication']
}

def safe_tokenize(text: str) -> List[str]:
    """Safe tokenization with fallback to simple split."""
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text.lower())
    except Exception as e:
        logger.warning(f"Using fallback tokenization: {e}")
        # Fallback to simple word splitting
        return text.lower().split()

def calculate_bleu_improved(prediction: str, reference: str) -> float:
    """Calculate improved BLEU score with better handling of medical text."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        # Handle empty strings
        if not prediction or not reference:
            return 0.0
            
        # Clean and normalize text
        prediction = re.sub(r'[^\w\s]', ' ', prediction.lower())
        reference = re.sub(r'[^\w\s]', ' ', reference.lower())
        
        pred_tokens = safe_tokenize(prediction)
        ref_tokens = safe_tokenize(reference)
        
        # Handle empty token lists
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # Use smoothing function to handle short sequences
        smoothing = SmoothingFunction().method1
        
        # Calculate BLEU with different n-gram weights for medical text
        weights = [(0.4, 0.3, 0.2, 0.1)]  # Give more weight to unigrams and bigrams
        
        bleu_scores = []
        for weight in weights:
            score = sentence_bleu([ref_tokens], pred_tokens, weights=weight, smoothing_function=smoothing)
            bleu_scores.append(score)
        
        return max(bleu_scores)  # Return the best score
    except Exception as e:
        logger.error(f"Error calculating improved BLEU score: {str(e)}")
        return 0.0

def calculate_recall_improved(prediction: str, reference: str) -> float:
    """Calculate improved recall with better word matching."""
    try:
        # Handle empty strings
        if not prediction or not reference:
            return 0.0
        
        # Clean and normalize text
        prediction = re.sub(r'[^\w\s]', ' ', prediction.lower())
        reference = re.sub(r'[^\w\s]', ' ', reference.lower())
            
        pred_words = set(safe_tokenize(prediction))
        ref_words = set(safe_tokenize(reference))
        
        if not ref_words:
            return 0.0
        
        # Remove common stop words for better medical content matching
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            pred_words = pred_words - stop_words
            ref_words = ref_words - stop_words
        except:
            pass  # Continue without stop word removal if NLTK is not available
        
        # Calculate recall with minimum word length filter
        pred_words = {word for word in pred_words if len(word) > 2}
        ref_words = {word for word in ref_words if len(word) > 2}
        
        if not ref_words:
            return 0.0
            
        return len(pred_words & ref_words) / len(ref_words)
    except Exception as e:
        logger.error(f"Error calculating improved recall: {str(e)}")
        return 0.0

def calculate_similarity_improved(prediction: str, reference: str) -> float:
    """Calculate improved cosine similarity with better text preprocessing."""
    try:
        # Handle empty strings
        if not prediction or not reference:
            return 0.0
        
        # Clean and normalize text
        prediction = re.sub(r'[^\w\s]', ' ', prediction.lower())
        reference = re.sub(r'[^\w\s]', ' ', reference.lower())
        
        # Use TF-IDF with better parameters for medical text
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),  # Include bigrams for medical terms
            min_df=1,
            max_df=0.95,
            stop_words='english'
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([prediction, reference])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            # Fallback to simple vectorization
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([prediction, reference])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
    except Exception as e:
        logger.error(f"Error calculating improved similarity: {str(e)}")
        return 0.0

def calculate_medical_relevance_improved(prediction: str, reference: str) -> float:
    """Calculate improved medical relevance score with enhanced terminology."""
    try:
        # Handle empty strings
        if not prediction or not reference:
            return 0.0
        
        pred_lower = prediction.lower()
        ref_lower = reference.lower()
        
        # Flatten the enhanced medical terms dictionary
        all_medical_terms = []
        for category, terms in ENHANCED_MEDICAL_TERMS.items():
            all_medical_terms.extend(terms)
        
        # Count medical terms in both prediction and reference
        pred_medical_terms = sum(1 for term in all_medical_terms if term in pred_lower)
        ref_medical_terms = sum(1 for term in all_medical_terms if term in ref_lower)
        
        if ref_medical_terms == 0:
            return 0.0
            
        # Calculate overlap with category-based weighting
        total_overlap = 0
        category_overlaps = {}
        
        for category, terms in ENHANCED_MEDICAL_TERMS.items():
            category_overlap = 0
            for term in terms:
                if term in pred_lower and term in ref_lower:
                    category_overlap += 1
            category_overlaps[category] = category_overlap
            total_overlap += category_overlap
        
        # Weight categories by importance
        category_weights = {
            'diagnosis': 1.5,
            'treatment': 1.3,
            'complications': 1.2,
            'clinical': 1.1,
            'monitoring': 1.0
        }
        
        weighted_overlap = 0
        for category, overlap in category_overlaps.items():
            weight = category_weights.get(category, 1.0)
            weighted_overlap += overlap * weight
        
        # Calculate final score
        base_score = total_overlap / ref_medical_terms if ref_medical_terms > 0 else 0
        weighted_score = weighted_overlap / ref_medical_terms if ref_medical_terms > 0 else 0
        
        # Combine base and weighted scores
        final_score = 0.7 * base_score + 0.3 * weighted_score
        
        return min(final_score, 1.0)  # Cap at 1.0
        
    except Exception as e:
        logger.error(f"Error calculating improved medical relevance: {str(e)}")
        return 0.0

def calculate_explanation_quality_improved(prediction: str, reference: str) -> float:
    """Calculate improved overall explanation quality score."""
    try:
        # Calculate all improved metrics
        bleu = calculate_bleu_improved(prediction, reference)
        recall = calculate_recall_improved(prediction, reference)
        similarity = calculate_similarity_improved(prediction, reference)
        medical_relevance = calculate_medical_relevance_improved(prediction, reference)
        
        # Additional quality factors
        length_factor = min(len(prediction.split()) / 50.0, 1.0)  # Reward longer explanations
        structure_factor = 1.0 if prediction.count('.') >= 2 else 0.5  # Reward proper sentence structure
        
        # Weighted combination with improved weights
        quality_score = (
            0.25 * bleu + 
            0.25 * recall + 
            0.20 * similarity + 
            0.20 * medical_relevance +
            0.05 * length_factor +
            0.05 * structure_factor
        )
        
        return quality_score
        
    except Exception as e:
        logger.error(f"Error calculating improved explanation quality: {str(e)}")
        return 0.0

def evaluate_response_improved(response: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
    """Calculate improved evaluation metrics for a single response."""
    try:
        prediction = response.get('explanation', '')
        reference = ground_truth.get('explanation', '')
        
        # Calculate all improved metrics
        bleu_score = calculate_bleu_improved(prediction, reference)
        recall_score = calculate_recall_improved(prediction, reference)
        similarity_score = calculate_similarity_improved(prediction, reference)
        medical_relevance_score = calculate_medical_relevance_improved(prediction, reference)
        explanation_quality_score = calculate_explanation_quality_improved(prediction, reference)
        
        metrics = {
            'bleu_score': bleu_score,
            'recall_score': recall_score,
            'similarity_score': similarity_score,
            'medical_relevance_score': medical_relevance_score,
            'explanation_quality_score': explanation_quality_score,
            'response_time': response.get('response_time', 0)
        }
        
        # Log the comparison for debugging
        logger.info(f"Prediction: {prediction[:100]}...")
        logger.info(f"Reference: {reference[:100]}...")
        logger.info(f"Improved BLEU: {bleu_score:.4f}, Recall: {recall_score:.4f}, Medical Relevance: {medical_relevance_score:.4f}")
        
        return metrics
    except Exception as e:
        logger.error(f"Error in evaluate_response_improved: {str(e)}")
        raise

def plot_metrics(metrics_df: pd.DataFrame):
    """Create interactive plots for evaluation metrics."""
    try:
        # BLEU Score Distribution
        fig_bleu = px.histogram(metrics_df, x='bleu_score', 
                              title='BLEU Score Distribution',
                              nbins=20)
        st.plotly_chart(fig_bleu)
        
        # Recall Score Distribution
        fig_recall = px.histogram(metrics_df, x='recall_score',
                                title='Recall Score Distribution',
                                nbins=20)
        st.plotly_chart(fig_recall)
        
        # Medical Relevance Score Distribution
        fig_medical = px.histogram(metrics_df, x='medical_relevance_score',
                                 title='Medical Relevance Score Distribution',
                                 nbins=20)
        st.plotly_chart(fig_medical)
        
        # Explanation Quality Score Distribution
        fig_quality = px.histogram(metrics_df, x='explanation_quality_score',
                                 title='Explanation Quality Score Distribution',
                                 nbins=20)
        st.plotly_chart(fig_quality)
        
        # Response Time vs Quality Score
        fig_time = px.scatter(metrics_df, x='response_time', y='explanation_quality_score',
                            title='Response Time vs Explanation Quality Score')
        st.plotly_chart(fig_time)
        
        # Correlation Matrix
        corr_matrix = metrics_df[['bleu_score', 'recall_score', 'similarity_score', 
                                'medical_relevance_score', 'explanation_quality_score']].corr()
        fig_corr = px.imshow(corr_matrix,
                           title='Metric Correlation Matrix',
                           labels=dict(color='Correlation'))
        st.plotly_chart(fig_corr)
        
        # Summary statistics
        st.subheader("Metric Summary")
        summary_stats = metrics_df[['bleu_score', 'recall_score', 'similarity_score', 
                                  'medical_relevance_score', 'explanation_quality_score']].describe()
        st.dataframe(summary_stats)
        
    except Exception as e:
        logger.error(f"Error plotting metrics: {str(e)}")
        st.error("Error creating visualizations")

def main():
    st.set_page_config(page_title="QA System Evaluation", layout="wide")
    
    # Initialize Weights & Biases
    wandb.init(project="medical-qa-system", name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    st.title("Medical QA System Evaluation")
    
    # Check if test_data.csv exists
    if not os.path.exists('test_data.csv'):
        st.error("test_data.csv not found! Please ensure the file exists in the current directory.")
        return
    
    try:
        # Load test data directly from test_data.csv
        test_data = pd.read_csv('test_data.csv')
        st.write(f"Number of test cases: {len(test_data)}")
        
        # Validate required columns
        required_columns = ['Patient_ID', 'Code', 'Description']
        missing_columns = [col for col in required_columns if col not in test_data.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return
        
        # Show sample of test data
        st.subheader("Sample Test Data")
        st.dataframe(test_data.head())
        
        if st.button("Run Evaluation"):
            metrics_list = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, test_case in test_data.iterrows():
                status_text.text(f"Processing test case {i+1}/{len(test_data)}: Patient_ID={test_case['Patient_ID']}, Code={test_case['Code']}")
                print(f"Processing test case {i+1}/{len(test_data)}: Patient_ID={test_case['Patient_ID']}, Code={test_case['Code']}")
                
                try:
                    response = requests.post(
                        "http://localhost:8000/explain-diagnosis",
                        json={
                            "patient_id": str(test_case['Patient_ID']),
                            "icd_code": str(test_case['Code']),
                            "window_days": 30
                        },
                        timeout=30  # Add timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if 'explanation' in result and result['explanation']:
                            metrics = evaluate_response_improved(
                                result,
                                {"explanation": str(test_case["Description"])}
                            )
                            metrics_list.append(metrics)
                        else:
                            error_msg = f"API response for case {i} missing or empty 'explanation': {result}"
                            print(error_msg)
                            logger.error(error_msg)
                            # Add empty metrics for failed cases
                            metrics_list.append({
                                'bleu_score': 0.0,
                                'recall_score': 0.0,
                                'similarity_score': 0.0,
                                'medical_relevance_score': 0.0,
                                'explanation_quality_score': 0.0,
                                'response_time': 0
                            })
                    else:
                        error_msg = f"Error in API request for case {i}: {response.text}"
                        print(error_msg)
                        st.error(error_msg)
                        # Add empty metrics for failed cases
                        metrics_list.append({
                            'bleu_score': 0.0,
                            'recall_score': 0.0,
                            'similarity_score': 0.0,
                            'medical_relevance_score': 0.0,
                            'explanation_quality_score': 0.0,
                            'response_time': 0
                        })
                except requests.exceptions.Timeout:
                    error_msg = f"Timeout for test case {i}"
                    print(error_msg)
                    logger.error(error_msg)
                    metrics_list.append({
                        'bleu_score': 0.0,
                        'recall_score': 0.0,
                        'similarity_score': 0.0,
                        'medical_relevance_score': 0.0,
                        'explanation_quality_score': 0.0,
                        'response_time': 0
                    })
                except Exception as e:
                    error_msg = f"Error processing test case {i}: {str(e)}"
                    print(error_msg)
                    logger.error(error_msg)
                    metrics_list.append({
                        'bleu_score': 0.0,
                        'recall_score': 0.0,
                        'similarity_score': 0.0,
                        'medical_relevance_score': 0.0,
                        'explanation_quality_score': 0.0,
                        'response_time': 0
                    })
                
                # Update progress
                progress_bar.progress((i + 1) / len(test_data))
            
            status_text.text("Evaluation complete!")
            
            if metrics_list:
                metrics_df = pd.DataFrame(metrics_list)
                
                # Display summary statistics
                st.subheader("Summary Statistics")
                st.write(metrics_df.describe())
                
                # Plot metrics
                st.subheader("Evaluation Metrics Visualization")
                plot_metrics(metrics_df)
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_filename = f"evaluation_results_{timestamp}.csv"
                metrics_df.to_csv(results_filename, index=False)
                st.success(f"Results saved to {results_filename}")
                
                # Log metrics to Weights & Biases
                wandb.log({
                    "avg_bleu_score": float(metrics_df['bleu_score'].mean()),
                    "avg_recall_score": float(metrics_df['recall_score'].mean()),
                    "avg_similarity_score": float(metrics_df['similarity_score'].mean()),
                    "avg_medical_relevance_score": float(metrics_df['medical_relevance_score'].mean()),
                    "avg_explanation_quality_score": float(metrics_df['explanation_quality_score'].mean()),
                    "avg_response_time": float(metrics_df['response_time'].mean()),
                    "success_rate": float(success_rate),
                })
                # Log the result file as an artifact
                wandb.save(results_filename)
                
                # Display success rate
                successful_cases = len([m for m in metrics_list if m['explanation_quality_score'] > 0.1])
                success_rate = successful_cases / len(metrics_list) * 100
                st.info(f"Success rate: {success_rate:.1f}% ({successful_cases}/{len(metrics_list)} cases)")
                
                # Display average scores
                avg_quality = np.mean([m['explanation_quality_score'] for m in metrics_list])
                avg_medical = np.mean([m['medical_relevance_score'] for m in metrics_list])
                st.info(f"Average Explanation Quality: {avg_quality:.3f}")
                st.info(f"Average Medical Relevance: {avg_medical:.3f}")
                
                # Show detailed results
                st.subheader("Detailed Results")
                detailed_results = []
                for i, (test_case, metrics) in enumerate(zip(test_data.iterrows(), metrics_list)):
                    detailed_results.append({
                        'Test_Case': i+1,
                        'Patient_ID': test_case[1]['Patient_ID'],
                        'ICD_Code': test_case[1]['Code'],
                        'Reference': test_case[1]['Description'],
                        'BLEU_Score': metrics['bleu_score'],
                        'Recall_Score': metrics['recall_score'],
                        'Medical_Relevance': metrics['medical_relevance_score'],
                        'Quality_Score': metrics['explanation_quality_score']
                    })
                
                detailed_df = pd.DataFrame(detailed_results)
                st.dataframe(detailed_df)
                
            else:
                st.error("No valid results to display")
                
    except Exception as e:
        st.error(f"Error loading or processing test data: {str(e)}")
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    # Test NLTK functionality
    def test_nltk_functionality():
        """Test NLTK functionality to ensure everything is working."""
        print("Testing NLTK functionality...")
        
        try:
            # Test tokenization
            test_text = "Patient LIB193263 was diagnosed with diabetes mellitus."
            tokens = safe_tokenize(test_text)
            print(f"✓ Tokenization test passed: {tokens}")
            
            # Test BLEU score calculation
            prediction = "Patient has diabetes mellitus"
            reference = "Patient diagnosed with diabetes mellitus"
            bleu_score = calculate_bleu_improved(prediction, reference)
            print(f"✓ BLEU score test passed: {bleu_score:.4f}")
            
            # Test recall calculation
            recall_score = calculate_recall_improved(prediction, reference)
            print(f"✓ Recall test passed: {recall_score:.4f}")
            
            # Test similarity calculation
            similarity_score = calculate_similarity_improved(prediction, reference)
            print(f"✓ Similarity test passed: {similarity_score:.4f}")
            
            print("🎉 All NLTK functionality tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ NLTK functionality test failed: {e}")
            return False
    
    # Run NLTK test first
    if test_nltk_functionality():
        main()
    else:
        print("NLTK functionality test failed. Please check the installation.")
        sys.exit(1)