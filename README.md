# 🩺 Medical QA System

A comprehensive medical question-answering and explanation system for clinical diagnoses, leveraging machine learning, natural language processing, and graph-based retrieval. The system supports interactive evaluation, advanced metrics, and explainability for medical AI.

---

## ✨ Features

- 🚀 **FastAPI backend** for scalable, GPU-accelerated medical QA APIs
- 🖥️ **Streamlit evaluation interface** for interactive testing and visualization
- 🧬 **ICD code graph-based retrieval** for context-aware explanations
- 🧠 **Node2Vec embeddings** for graph representation
- ⚡ **FAISS and Whoosh indexing** for efficient semantic and lexical search
- 🩻 **SHAP-based explanation generation** for model interpretability and transparency
- 📊 **Comprehensive evaluation metrics**: BLEU, Recall, Cosine Similarity, Response Time, Medical Relevance
- 📈 **Experiment tracking** with Weights & Biases (wandb)
- 🛠️ **Automated and manual dependency installation scripts**
- 🧹 **Extensive data preprocessing and validation**

---

## 🗂️ Project Structure

```
questionanswer/
│
├── app.py                  # FastAPI backend and API endpoints
├── evaluate.py             # Streamlit-based evaluation and metrics dashboard
├── icd_qa_app.py           # Streamlit app for ICD graph-augmented QA
├── qa_pipeline.py          # Core QA pipeline, prompt engineering, and answer generation
├── explain.py              # SHAP-based explanation utilities and analysis
├── generate_shap_heatmap.py # SHAP attention heatmap generation script
├── icd_graph.py            # ICD code graph construction and operations
├── icd_eval_graph.py       # ICD/metric graph generation and visualization
├── embeddings.py           # Node2Vec embeddings for ICD graph
├── indexing.py             # FAISS and Whoosh indexing utilities
├── data_loader.py          # Data loading, validation, and preprocessing
├── install_dependencies.py # Automated dependency installation script
├── requirements.txt        # Python dependencies
├── tsne_embedding_analysis.py # Streamlit t-SNE embedding analysis app
├── quick_tsne_analysis.py      # Standalone t-SNE analysis script
├── run_tsne_example.py         # Example runner for t-SNE analysis
├── README.md               # Project documentation (this file)
│
├── data/                   # Medical datasets (CSV)
│   ├── Patient_info.csv
│   ├── Diagnostics.csv
│   ├── Biochemical_parameters.csv
│   └── Glucose_measurements.csv
│
├── icd_index/              # Index files for ICD code retrieval
│   ├── _MAIN_1.toc
│   ├── MAIN_e49zbj0vu591txbo.seg
│   └── MAIN_WRITELOCK
│
├── images/                 # Visualizations and diagrams
│   ├── icd_graph.png
│   ├── icd_graph_advanced.png
│   ├── icd_graph_from_diagnostics.png
│   ├── metric_graph_from_eval.png
│   └── shap_attention_heatmap.png
│
├── wandb/                  # Experiment tracking logs and outputs
│   └── ... (run folders)
│
├── test_data.csv           # Test cases for evaluation
├── evaluation_results_*.csv# Evaluation results (auto-generated)
├── icd_graph_edges.csv     # Exported ICD graph edges
├── __pycache__/            # Python bytecode cache (auto-generated)
└── ...
```

---

## 🧬 ICD-9 Embedding Semantic Analysis (t-SNE)

This project includes advanced tools for evaluating the semantic quality of learned ICD-9 embeddings using t-distributed stochastic neighbor embedding (t-SNE). The analysis focuses on clinical clusters around common T1D comorbidities like Diabetes (250.00), Hypertension (401.9), and Hyperlipidemia (272.4).

### 📁 Key Files

- `tsne_embedding_analysis.py` - **Interactive Streamlit app** for comprehensive analysis
- `quick_tsne_analysis.py` - **Standalone script** for quick analysis and batch processing
- `run_tsne_example.py` - **Example runner** for t-SNE analysis

### 🚀 Quick Start

#### Option 1: Interactive Analysis (Recommended)

Run the Streamlit app for an interactive, comprehensive analysis:

```bash
streamlit run tsne_embedding_analysis.py
```

This will open a web interface at `http://localhost:8501` with:
- Interactive t-SNE visualizations
- Real-time parameter adjustment
- Cluster analysis and statistics
- Export capabilities
- Detailed clinical insights

#### Option 2: Quick Batch Analysis

Run the standalone script for quick analysis:

```bash
python quick_tsne_analysis.py
```

**Command-line options:**
```bash
python quick_tsne_analysis.py --perplexity 30 --iterations 1000 --output-dir ./results
```

#### Option 3: Example Configurations

Run the example runner to see different parameter settings:

```bash
python run_tsne_example.py
```

### 📊 Output Files

Both scripts generate the following outputs:

- `tsne_visualization.png` - Main t-SNE plot showing clinical clusters
- `cluster_analysis.png` - Cluster analysis with centroids and size distribution
- `tsne_results.csv` - Embedding coordinates and metadata
- `cluster_statistics.csv` - Cluster quality metrics

### 🎯 Clinical Categories Analyzed

| Category         | Key Codes                        | Description                | Color      |
|------------------|----------------------------------|----------------------------|------------|
| **Diabetes**     | 250.00, 250.01, 250.02, 250.03   | Diabetes Mellitus          | 🟠 Orange  |
| **Hypertension** | 401.9, 401.0, 401.1              | Hypertension               | 🟢 Green   |
| **Hyperlipidemia** | 272.4, 272.0, 272.1            | Hyperlipidemia             | 🔴 Red     |
| **Renal**        | 585.9, 585.1, 585.2              | Chronic Kidney Disease     | 🟣 Purple  |
| **Cardiovascular** | 410.00, 410.01, 428.0          | Cardiovascular Disease     | 🟤 Brown   |
| **Ophthalmic**   | 362.01, 362.02, 362.03           | Diabetic Retinopathy       | 🟡 Pink    |
| **Neurological** | 357.2, 357.3, 357.4              | Diabetic Neuropathy        | ⚫ Gray    |

### 🔍 Analysis Features

- **Semantic Quality Assessment**: Clinical fidelity validation, vector retrieval improvement, comorbidity pattern discovery
- **Cluster Analysis**: Density metrics, centroid analysis, inter-cluster distances
- **Key Code Highlighting**: Diabetes (250.00), Hypertension (401.9), Hyperlipidemia (272.4)

### 📈 Expected Results

- **Clear Clinical Clusters**: Distinct groups for diabetes, hypertension, and hyperlipidemia
- **Semantic Grounding**: Improved vector retrieval through latent structure
- **Clinical Fidelity**: Patterns that reflect real-world comorbidity relationships
- **QA Enhancement**: Better evidence context during question answering

### ⚙️ Technical Details

#### Dependencies

- `scikit-learn` for t-SNE and PCA
- `matplotlib` and `seaborn` for static plots
- `plotly` for interactive visualizations (Streamlit app)
- `pandas` and `numpy` for data manipulation
- `node2vec` for graph embeddings

#### Parameters

- **Perplexity**: Controls local vs. global structure balance (default: 30)
- **Iterations**: t-SNE optimization iterations (default: 1000)
- **Embedding Size**: Node2Vec embedding dimensions (default: 128)

### 📝 Example Usage

#### Interactive Analysis Session

```bash
# Start Streamlit app
streamlit run tsne_embedding_analysis.py

# Navigate to http://localhost:8501
# Adjust perplexity and iterations in sidebar
# Explore different clinical categories
# Export results as needed
```

#### Batch Analysis

```bash
# Quick analysis with default parameters
python quick_tsne_analysis.py

# Custom analysis
python quick_tsne_analysis.py \
  --perplexity 25 \
  --iterations 1500 \
  --output-dir ./analysis_results

# Check results
ls -la analysis_results/
```

#### Example Runner

```bash
python run_tsne_example.py
```
This will run several analyses with different parameter settings and summarize the results.

### 🛠️ Troubleshooting

#### Common Issues

1. **Data Loading Errors**
   ```bash
   # Ensure data files exist
   ls data/Diagnostics.csv
   ls test_data.csv
   ```

2. **Memory Issues**
   ```bash
   # Reduce embedding size or perplexity
   python quick_tsne_analysis.py --perplexity 20 --iterations 500
   ```

3. **Missing Dependencies**
   ```bash
   # Install required packages
   pip install scikit-learn matplotlib seaborn plotly
   ```

#### Performance Tips

- **Large Datasets**: Use PCA preprocessing (already implemented)
- **Interactive Mode**: Use Streamlit app for parameter tuning
- **Batch Processing**: Use standalone script for automation

### 🔬 Research Context

This analysis validates the findings described in your research:

> "To evaluate the semantic quality of the learned ICD-9 embeddings, we applied t-distributed stochastic neighbor embedding (t-SNE) on Node2Vec outputs. Clusters formed around common comorbidities of T1D—e.g., '250.00' (Diabetes), '401.9' (Hypertension), and '272.4' (Hyperlipidemia)—validating the embeddings' clinical fidelity. These patterns improved semantic grounding during vector retrieval, as the model leveraged this latent structure to fetch more relevant evidence contexts during QA generation."

### 🎯 Next Steps

1. **Review Clusters**: Examine clinical category separation
2. **Validate Patterns**: Confirm T1D comorbidity relationships
3. **Export Results**: Save visualizations for publications
4. **Iterate**: Adjust parameters for optimal visualization
5. **Integrate**: Use insights to improve QA system performance

---

## 🔍 SHAP-Based Model Interpretability

This project includes comprehensive SHAP (SHapley Additive exPlanations) integration for model interpretability and explanation generation. The SHAP implementation provides insights into how the medical QA system makes decisions and generates explanations.

### 📁 Key Files

- `explain.py` - **Core SHAP utilities** for model explanation and analysis
- `generate_shap_heatmap.py` - **SHAP attention heatmap generation** for text inputs
- `images/shap_attention_heatmap.png` - **Generated SHAP visualization** showing token importance

### 🚀 Quick Start

#### Generate SHAP Attention Heatmap

```bash
python generate_shap_heatmap.py
```

This will generate a SHAP attention heatmap for the example medical query "What are the symptoms of diabetes?" and save it as `shap_attention_heatmap.png`.

#### Use SHAP Explanation Functions

```python
from explain import explain_with_shap, analyze_qa_response, get_explanation_metrics

# Explain model predictions
shap_results = explain_with_shap(model_fn, input_vector, background_data)

# Analyze QA response quality
analysis = analyze_qa_response(response_text, context_text)

# Calculate explanation metrics
metrics = get_explanation_metrics(ground_truth, prediction)
```

### 🔬 SHAP Features

#### 1. Model Explanation (`explain_with_shap`)
- **KernelExplainer**: Uses SHAP's kernel explainer for model-agnostic explanations
- **Feature Importance**: Calculates mean absolute SHAP values for feature ranking
- **Background Sampling**: Efficient background data sampling for computational efficiency
- **Error Handling**: Robust error handling with detailed logging

#### 2. Response Analysis (`analyze_qa_response`)
- **Sentence Analysis**: Breaks down responses into meaningful sentences
- **Key Term Extraction**: Identifies important medical terminology
- **Context Coverage**: Measures how well the response covers the input context
- **Response Quality**: Evaluates response length and structure

#### 3. Explanation Metrics (`get_explanation_metrics`)
- **Precision**: Measures accuracy of predicted terms against ground truth
- **Recall**: Measures coverage of ground truth terms in prediction
- **F1 Score**: Balanced metric combining precision and recall
- **Word Overlap**: Direct count of overlapping terms

#### 4. Attention Heatmap (`plot_shap_attention_heatmap`)
- **Token-Level Analysis**: Shows importance of individual tokens in input text
- **Visual Heatmap**: Color-coded visualization of SHAP values
- **Model Integration**: Works with transformer-based models (T5, BERT, etc.)
- **Export Capability**: Saves high-quality visualizations for analysis

### 📊 SHAP Visualization

The generated SHAP attention heatmap (`images/shap_attention_heatmap.png`) shows:

- **Token Importance**: Color intensity indicates SHAP value magnitude
- **Positive/Negative Impact**: Red (positive) vs Blue (negative) contributions
- **Medical Context**: Highlights medically relevant terms in queries
- **Model Behavior**: Reveals which parts of input influence model decisions

### 🎯 Clinical Applications

#### 1. **Diagnostic Decision Support**
- Understand which patient symptoms most influence diagnosis
- Validate model reasoning against clinical guidelines
- Identify potential biases in model predictions

#### 2. **Medical Education**
- Visualize how medical knowledge is applied in QA responses
- Demonstrate clinical reasoning patterns
- Support medical training and education

#### 3. **Quality Assurance**
- Monitor model performance and consistency
- Detect anomalous or unreliable predictions
- Ensure medical accuracy and relevance

#### 4. **Research Validation**
- Validate model behavior against clinical expertise
- Support research publications with interpretability evidence
- Enable peer review of AI medical systems

### ⚙️ Technical Implementation

#### Dependencies
```python
import shap
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
```

#### Model Integration
```python
# Example with T5 model
model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def model_fn(texts):
    outputs = pipeline(list(texts))
    return np.array([len(o['generated_text']) for o in outputs])
```

#### SHAP Configuration
- **Background Samples**: 50 samples for computational efficiency
- **Kernel Explainer**: Model-agnostic approach for flexibility
- **Visualization**: Coolwarm colormap for intuitive interpretation

### 📈 Expected Benefits

#### 1. **Transparency**
- Clear understanding of model decision-making process
- Identifiable reasoning patterns for medical queries
- Auditable AI system for clinical use

#### 2. **Trust**
- Build confidence in AI medical systems
- Support clinical adoption and acceptance
- Enable informed decision-making by healthcare providers

#### 3. **Quality Improvement**
- Identify areas for model enhancement
- Optimize prompt engineering based on SHAP insights
- Improve medical accuracy and relevance

#### 4. **Compliance**
- Support regulatory requirements for AI transparency
- Enable clinical validation and approval processes
- Provide documentation for medical AI systems

### 🔧 Customization Options

#### 1. **Model-Specific Explainers**
```python
# For transformer models
explainer = shap.Explainer(model_fn, tokenizer)

# For traditional ML models
explainer = shap.KernelExplainer(model_fn, background_data)
```

#### 2. **Custom Metrics**
```python
# Add domain-specific evaluation metrics
def calculate_medical_accuracy(prediction, ground_truth):
    # Custom medical accuracy calculation
    pass
```

#### 3. **Visualization Customization**
```python
# Customize heatmap appearance
plt.figure(figsize=(12, 4))
plt.imshow([values], cmap='RdBu', aspect='auto')
plt.title("Custom SHAP Analysis")
```

### 🛠️ Troubleshooting

#### Common Issues

1. **Memory Issues**
   ```bash
   # Reduce background sample size
   shap_results = explain_with_shap(model_fn, input_vec, background_data, n_samples=25)
   ```

2. **Model Compatibility**
   ```python
   # Ensure model function returns appropriate output format
   def model_fn(texts):
       return np.array([model_output for model_output in model(texts)])
   ```

3. **Visualization Errors**
   ```python
   # Check matplotlib backend
   import matplotlib
   matplotlib.use('Agg')  # For headless environments
   ```

### 🔬 Research Context

This SHAP implementation supports the research goal of creating interpretable medical AI systems:

> "The integration of SHAP-based explanations provides transparency into the medical QA system's decision-making process, enabling clinical validation and trust-building. By visualizing token importance and model reasoning patterns, healthcare providers can understand and validate AI-generated medical explanations."

### 🎯 Integration with QA Pipeline

The SHAP functionality integrates seamlessly with the existing QA pipeline:

1. **Pre-Processing**: SHAP analysis of input queries
2. **Generation**: Real-time explanation quality assessment
3. **Post-Processing**: Response analysis and improvement
4. **Evaluation**: Comprehensive metrics for system validation

This creates a complete feedback loop for continuous improvement of the medical QA system.

### 📚 References

- **t-SNE**: van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE.
- **Node2Vec**: Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning for networks.
- **ICD-9 Codes**: International Classification of Diseases, 9th Revision

---

**Note**: This analysis requires your medical dataset to be properly loaded. Ensure you have access to the required CSV files in the `data/` directory.

---

## 📁 Data

> **Note:** The datasets in the `data/` folder are **not publicly available** due to privacy and licensing restrictions. To access the dataset required for this project, please email Zenodo (https://zenodo.org/) and request access to the relevant medical data. You must obtain permission before working with the data files.

- 👤 **Patient_info.csv**: Patient demographic and static information
- 📝 **Diagnostics.csv**: Patient diagnoses with ICD codes and descriptions
- 🧪 **Biochemical_parameters.csv**: Lab results and biochemical measurements
- 🩸 **Glucose_measurements.csv**: Time-series glucose data (large file)
- 🧾 **test_data.csv**: Test cases for QA evaluation
- 📄 **evaluation_results_*.csv**: Evaluation results from different runs
- 📊 **metrics.csv**: Summary of evaluation metrics
- 🕸️ **icd_graph_edges.csv**: Exported ICD graph edges for analysis

---

## ⚙️ Installation

### 🚦 Automated (Recommended)

```bash
python install_dependencies.py
```

### 🛠️ Manual

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

> **Note:** For some ML packages, you may need to install Microsoft Visual C++ Build Tools on Windows.

---

## 🚀 Usage

### 1️⃣ Start the FastAPI Backend

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 2️⃣ Run the Evaluation Interface

```bash
streamlit run evaluate.py
```

- Access the dashboard at [http://localhost:8501](http://localhost:8501)

### 3️⃣ ICD Graph-Augmented QA (Optional)

```bash
streamlit run icd_qa_app.py
```

### 4️⃣ SHAP Model Interpretability (Optional)

```bash
# Generate SHAP attention heatmap
python generate_shap_heatmap.py

# Use SHAP functions in your code
python -c "
from explain import explain_with_shap, analyze_qa_response
# Your SHAP analysis code here
"
```

---

## 🔌 API Endpoints

### `POST /explain-diagnosis`

Explain a medical diagnosis for a given patient and ICD code.

**Request:**
```json
{
  "patient_id": "string",
  "icd_code": "string",
  "window_days": integer
}
```

**Response:**
```json
{
  "explanation": "string",
  "similar_cases": [
    {
      "patient_id": "string",
      "icd_code": "string",
      "similarity_score": float
    }
  ],
  "icd_definition": "string",
  "response_time": float
}
```

### `GET /health`

Check the health status of the API.

**Response:**
```json
{
  "status": "healthy",
  "version": "string"
}
```

---

## 📈 Evaluation Metrics

- 🟦 **BLEU Score**: Text generation quality
- 🟩 **Recall Score**: Information coverage
- 🟨 **Cosine Similarity**: Semantic alignment
- 🟧 **Medical Relevance**: Domain-specific accuracy
- ⏱️ **Response Time**: System latency

Results are visualized with Plotly and saved to CSV.

---

## 🚀 Performance Improvements

### Overview
This project has undergone comprehensive improvements to enhance the quality, relevance, and reliability of its medical QA outputs. The following summarizes the key changes and their impact:

### Current Performance Issues (Baseline)
| Metric | Current Mean | Target | Issue |
|--------|-------------|--------|-------|
| BLEU Score | 0.031 | >0.1 | Low text generation quality |
| Recall Score | 0.433 | >0.6 | Poor information coverage |
| Similarity Score | 0.171 | >0.3 | Weak semantic alignment |
| Medical Relevance | 0.463 | >0.7 | Insufficient medical terminology |
| Overall Quality | 0.266 | >0.5 | Poor combined performance |

### Key Improvements Implemented

#### 1. Enhanced QA Pipeline
- **Multi-Model Architecture:** T5-base integration, fallback mechanisms, and automatic model selection.
- **Prompt Engineering:** Structured prompts, expanded medical vocabulary, and comprehensive patient context extraction.
- **Response Enhancement:** Medical terminology injection, improved grammar and punctuation (TextBlob), and more detailed explanations.

#### 2. Advanced Evaluation Metrics
- Improved BLEU, Recall, and Similarity calculations (n-gram weighting, stop word removal, bigram TF-IDF, category-based weighting).
- Enhanced medical relevance scoring and category-based weighting.

#### 3. Medical Terminology Expansion
- Expanded and context-aware medical dictionary.
- Automatic terminology injection and standard phrase integration.

#### 4. Text Processing
- NLTK-based tokenization, medical-specific preprocessing, and TextBlob-based sentence structure enhancement.

### Expected Performance Gains
- **BLEU Score:** 0.031 → 0.1+ (3x improvement)
- **Recall Score:** 0.433 → 0.6+ (40% improvement)
- **Similarity Score:** 0.171 → 0.3+ (75% improvement)
- **Medical Relevance:** 0.463 → 0.7+ (50% improvement)
- **Overall Quality:** 0.266 → 0.5+ (90% improvement)

### Monitoring and Validation
- Compare new results with baseline metrics and track improvements.
- Validate medical accuracy, clinical relevance, and response time.
- Continuous improvement via prompt engineering, model fine-tuning, and feedback.

### Additional Recommendations
- Fine-tune models on medical data, expand terminology, add human/clinical evaluation, and optimize system performance.

---

## 📝 Key Scripts and Their Roles

- `app.py`: Main FastAPI backend, data/model initialization, API endpoints
- `evaluate.py`: Streamlit dashboard for running and visualizing QA evaluation
- `icd_qa_app.py`: Streamlit app for graph-augmented QA and ICD code exploration
- `qa_pipeline.py`: Core logic for prompt creation, answer generation, and medical terminology enhancement
- `explain.py`: SHAP-based model explanation utilities and response analysis
- `generate_shap_heatmap.py`: SHAP attention heatmap generation for model interpretability
- `icd_graph.py`: ICD code graph construction from diagnosis data
- `icd_eval_graph.py`: Command-line tool for generating and visualizing ICD/metric graphs
- `embeddings.py`: Node2Vec graph embeddings for ICD codes
- `indexing.py`: FAISS and Whoosh index creation and retrieval
- `data_loader.py`: Data loading, validation, and preprocessing utilities
- `install_dependencies.py`: Automated dependency installation with error handling
- `requirements.txt`: All Python dependencies for the project
- `README.md`: This documentation file
- `test_data.csv`: Test cases for evaluation
- `evaluation_results_*.csv`: Evaluation results from different runs
- `icd_graph_edges.csv`: Exported ICD graph edges for analysis
- `graph_data/`: Precomputed graph data (e.g., icd_graph.json)
- `icd_index/`: Index files for ICD code retrieval
- `images/`: Visualizations and diagrams
- `wandb/`: Experiment tracking logs and outputs
- `__pycache__/`: Python bytecode cache (auto-generated)

---

## 🖼️ Visualizations

- `images/icd_graph.png`: ICD code co-occurrence graph
- `images/icd_graph_advanced.png`: Advanced ICD graph visualization
- `images/icd_graph_from_diagnostics.png`: Graph from diagnosis data
- `images/metric_graph_from_eval.png`: Metric correlation graph

---

## 📊 Experiment Tracking

- `wandb/`: Contains logs, configs, and outputs for experiment runs (Weights & Biases)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Create a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

**For any issues or questions, please open an issue** 