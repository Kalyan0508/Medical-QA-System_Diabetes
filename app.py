from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import logging
import os
from data_loader import load_and_preprocess
from icd_graph import build_icd_graph
from embeddings import train_node2vec
from indexing import build_faiss_index, build_whoosh_index, retrieve
from qa_pipeline import make_prompt, answer_query
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical QA System",
             description="Question answering system for medical diagnoses")

# Initialize global variables
df_patients = None
df_dx = None
df_bio = None
df_glu = None
G = None
emb = None
patient_vecs = None
faiss_idx = None
whoosh_ix = None

data_ready = False
data_error = None

def initialize_data():
    """Initialize all required data and models."""
    global df_patients, df_dx, df_bio, df_glu, G, emb, patient_vecs, faiss_idx, whoosh_ix
    
    try:
        # Check if data directory exists
        data_dir = "data"
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
        # Check for required files
        required_files = [
            "Patient_info.csv",
            "Diagnostics.csv",
            "Biochemical_parameters.csv",
            "Glucose_measurements.csv"
        ]
        
        for file in required_files:
            file_path = os.path.join(data_dir, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        logger.info("Loading and preprocessing data...")
        df_patients, df_dx, df_bio, df_glu = load_and_preprocess(data_dir)
        
        if df_patients is None or df_dx is None or df_bio is None or df_glu is None:
            raise ValueError("Failed to load one or more dataframes")
            
        logger.info("Building ICD graph...")
        G = build_icd_graph(df_patients, df_dx)
        
        if G is None or len(G.nodes()) == 0:
            raise ValueError("Failed to build ICD graph")
            
        logger.info("Training node embeddings...")
        emb = train_node2vec(G)
        
        if emb is None or len(emb) == 0:
            raise ValueError("Failed to train node embeddings")
            
        logger.info("Preparing patient vectors...")
        patient_vecs = [(node, vec) for node, vec in emb.items()]
        
        if not patient_vecs:
            raise ValueError("Failed to prepare patient vectors")
            
        logger.info("Building FAISS index...")
        faiss_idx = build_faiss_index(patient_vecs, dim=128)
        
        if faiss_idx is None:
            raise ValueError("Failed to build FAISS index")
            
        logger.info("Building Whoosh index...")
        whoosh_ix = build_whoosh_index(
            {code: desc for code, desc in zip(df_dx['Code'], df_dx['Description'])}
        )
        
        if whoosh_ix is None:
            raise ValueError("Failed to build Whoosh index")
            
        logger.info("Startup complete!")
        return True
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        return False

def background_initialize():
    global data_ready, data_error
    global df_patients, df_dx, df_bio, df_glu, G, emb, patient_vecs, faiss_idx, whoosh_ix
    try:
        result = initialize_data()
        if result:
            data_ready = True
            data_error = None
        else:
            data_ready = False
            data_error = "Initialization failed"
    except Exception as e:
        data_error = str(e)
        data_ready = False

@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=background_initialize)
    thread.start()

class Query(BaseModel):
    patient_id: str = Field(..., description="Use a real Patient_ID from Patient_info.csv (e.g., LIB193263)")
    icd_code: str = Field(..., description="Use a real ICD code from Diagnostics.csv (e.g., 272.4)")
    window_days: int = Field(30, description="Number of days for the window (default: 30)")

@app.post("/explain-diagnosis")
async def explain(q: Query):
    """Explain a diagnosis for a patient."""
    try:
        if data_error:
            raise HTTPException(status_code=500, detail=f"Initialization error: {data_error}")
        if not data_ready or any(x is None for x in [df_patients, df_dx, df_bio, df_glu, G, emb, patient_vecs, faiss_idx, whoosh_ix]):
            raise HTTPException(status_code=503, detail="Service not initialized properly")
        # Validate patient ID
        if q.patient_id not in df_patients['Patient_ID'].values:
            raise HTTPException(status_code=404, detail="Patient ID not found")
        # Validate ICD code
        if q.icd_code not in df_dx['Code'].values:
            raise HTTPException(status_code=404, detail="ICD code not found")
        # Get similar cases and definition
        def_text, similar_codes = retrieve(q.icd_code, faiss_idx, patient_vecs, whoosh_ix, k=5)
        # Generate explanation
        prompt_dict = make_prompt(q.patient_id, q.icd_code, def_text, similar_codes, 
                                df_dx, df_glu, df_bio, q.window_days)
        logger.info(f"Prompt for patient {q.patient_id}, code {q.icd_code}: {prompt_dict}")
        explanation = answer_query(prompt_dict)
        logger.info(f"Answer for patient {q.patient_id}, code {q.icd_code}: {explanation}")
        # Fallback for empty or generic answers
        if not explanation or explanation.strip() == '' or explanation.startswith('Unable to generate'):
            explanation = ("No sufficient data available to generate a meaningful explanation for this patient and code. "
                           "Please check if the patient has relevant glucose or biochemical data in the selected window.")
        return {
            "explanation": explanation,
            "similar_cases": similar_codes,
            "definition": def_text,
            "status": "success"
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    if data_error:
        return {"status": "error", "message": data_error}
    if not data_ready:
        return {"status": "initializing", "message": "Service is warming up, please wait..."}
    return {"status": "healthy"}

@app.get("/")
def read_root():
    if data_error:
        return {"status": "error", "message": data_error}
    if not data_ready:
        return {"status": "initializing", "message": "Service is warming up, please wait..."}
    return {"message": "Medical QA System is running. Visit /docs for API documentation."}

if __name__ == "__main__":
    import uvicorn
    # Ensure data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")
        logger.warning("Created data directory. Please add required CSV files.")
    
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
