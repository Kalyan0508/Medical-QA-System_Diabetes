import pandas as pd
import numpy as np
from typing import Tuple
import logging
import os
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize QA pipeline with error handling
try:
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        device=0
    )
except Exception as e:
    logger.error(f"Failed to initialize QA pipeline: {str(e)}")
    qa_pipeline = None

def validate_dataframe(df: pd.DataFrame, required_columns: list, name: str) -> pd.DataFrame:
    """Validate dataframe structure and data types."""
    if df is None or df.empty:
        raise ValueError(f"Empty dataframe provided for {name}")
        
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {name}: {missing_cols}")
    
    # Check for duplicate records
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate records in {name}")
        df = df.drop_duplicates()
    
    # Check for null values in critical columns
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        logger.warning(f"Found null values in {name}: {null_counts[null_counts > 0]}")
    
    return df

def load_and_preprocess(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and preprocess all data files"""
    try:
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
        logger.info("Loading data files...")
        
        # Load CSV files with optimized settings for large files
        try:
            # Load smaller files first
            logger.info("Loading Patient_info.csv...")
            df_patients = pd.read_csv(
                os.path.join(data_dir, "Patient_info.csv"),
                low_memory=False
            )
            
            logger.info("Loading Diagnostics.csv...")
            df_dx = pd.read_csv(
                os.path.join(data_dir, "Diagnostics.csv"),
                low_memory=False
            )
            
            logger.info("Loading Biochemical_parameters.csv...")
            df_bio = pd.read_csv(
                os.path.join(data_dir, "Biochemical_parameters.csv"),
                parse_dates=['Reception_date'],
                low_memory=False
            )
            
            # Load glucose measurements in chunks and process immediately
            logger.info("Loading and processing Glucose_measurements.csv...")
            chunk_size = 50000  # Reduced chunk size for better memory management
            df_glu_list = []
            
            # Get total number of rows for progress tracking
            total_rows = sum(1 for _ in open(os.path.join(data_dir, "Glucose_measurements.csv"))) - 1
            processed_rows = 0
            
            for chunk in pd.read_csv(
                os.path.join(data_dir, "Glucose_measurements.csv"),
                chunksize=chunk_size,
                low_memory=False
            ):
                # Process each chunk
                chunk['Measurement_datetime'] = pd.to_datetime(
                    chunk['Measurement_date'].astype(str) + ' ' + chunk['Measurement_time'].astype(str)
                )
                
                # Calculate rolling means for this chunk
                chunk = chunk.groupby('Patient_ID', group_keys=False).apply(
                    lambda x: x.sort_values('Measurement_datetime')
                    .set_index('Measurement_datetime')
                    .assign(
                        glucose_7d_mean=lambda x: x['Measurement'].rolling('7D', min_periods=1).mean(),
                        glucose_30d_mean=lambda x: x['Measurement'].rolling('30D', min_periods=1).mean()
                    )
                    .reset_index()
                )
                
                df_glu_list.append(chunk)
                processed_rows += len(chunk)
                logger.info(f"Processed {processed_rows}/{total_rows} rows ({processed_rows/total_rows*100:.1f}%)")
            
            # Combine processed chunks
            logger.info("Combining processed chunks...")
            df_glu = pd.concat(df_glu_list, ignore_index=True)
            del df_glu_list  # Free memory
            
        except pd.errors.EmptyDataError:
            raise ValueError("One or more data files are empty")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV files: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error loading data files: {str(e)}")
        
        # Log actual column names for debugging
        logger.info("Column names in Diagnostics.csv:")
        logger.info(df_dx.columns.tolist())
        
        # Validate required columns based on actual data structure
        df_patients = validate_dataframe(df_patients, ["Patient_ID", "Sex", "Birth_year", "Initial_measurement_date"], "Patient_info")
        df_dx = validate_dataframe(df_dx, ["Patient_ID", "Code", "Description"], "Diagnostics")
        df_bio = validate_dataframe(df_bio, ["Patient_ID", "Reception_date", "Name", "Value"], "Biochemical_parameters")
        df_glu = validate_dataframe(df_glu, ["Patient_ID", "Measurement_date", "Measurement_time", "Measurement"], "Glucose_measurements")
        
        logger.info("Cleaning data...")
        # Data cleaning with validation
        for df, name in [(df_patients, "patients"), (df_dx, "diagnostics"), 
                        (df_glu, "glucose"), (df_bio, "biochemical")]:
            if df['Patient_ID'].isnull().any():
                logger.warning(f"Found null Patient_IDs in {name} data")
                df = df.dropna(subset=['Patient_ID'])
        
        # Sort data chronologically with validation
        try:
            df_patients = df_patients.sort_values('Initial_measurement_date')
            df_dx = df_dx.sort_values('Patient_ID')
            df_glu = df_glu.sort_values('Measurement_datetime')
            df_bio = df_bio.sort_values('Reception_date')
        except Exception as e:
            logger.error(f"Error sorting data: {str(e)}")
            raise ValueError("Error sorting data chronologically")
        
        # Anomaly detection for biochemical parameters with validation
        normal_ranges = {
            "HbA1c": (4.0, 5.6),
            "Fasting Glucose": (70, 100),
            "Postprandial Glucose": (70, 140)
        }
        
        try:
            df_bio["anomaly"] = df_bio.apply(
                lambda row: (
                    row["Name"] in normal_ranges and
                    (row["Value"] < normal_ranges[row["Name"]][0] or
                     row["Value"] > normal_ranges[row["Name"]][1])
                ),
                axis=1
            )
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            df_bio["anomaly"] = False
        
        # Data quality metrics
        logger.info(f"Total patients: {len(df_patients)}")
        logger.info(f"Total diagnoses: {len(df_dx)}")
        logger.info(f"Total glucose measurements: {len(df_glu)}")
        logger.info(f"Total biochemical tests: {len(df_bio)}")
        
        return df_patients, df_dx, df_bio, df_glu
        
    except Exception as e:
        logger.error(f"Error in data loading and preprocessing: {str(e)}")
        raise

def make_prompt(pid, icd_code, def_text, similar_pids, df_dx, df_glu, df_bio, window_days=30):
    """Create a prompt for the QA system with error handling."""
    try:
        if not all(isinstance(x, (pd.DataFrame, type(None))) for x in [df_dx, df_glu, df_bio]):
            raise TypeError("Invalid dataframe input")
        if not isinstance(window_days, int) or window_days <= 0:
            raise ValueError("window_days must be a positive integer")
        end = pd.Timestamp.now()
        start = end - pd.Timedelta(days=window_days)
        # Filter data with error handling
        try:
            glu_sub = df_glu[(df_glu.Patient_ID==pid)&(df_glu.Measurement_datetime.between(start,end))]
            bio_sub = df_bio[(df_bio.Patient_ID==pid)&(df_bio.Reception_date.between(start,end))]
        except Exception as e:
            logger.error(f"Error filtering data: {str(e)}")
            glu_sub = pd.DataFrame()
            bio_sub = pd.DataFrame()
        # Glucose trend
        if not glu_sub.empty:
            try:
                glucose_trend = f"{glu_sub.glucose_7d_mean.iloc[0]:.1f} → {glu_sub.glucose_7d_mean.iloc[-1]:.1f}"
            except Exception as e:
                logger.error(f"Error calculating glucose trend: {str(e)}")
                glucose_trend = "Glucose data present but trend calculation failed."
        else:
            glucose_trend = "No glucose data available for this patient in the selected window."
        # Biochemical anomalies
        if not bio_sub.empty and 'anomaly' in bio_sub.columns and bio_sub.anomaly.any():
            try:
                bio_anomalies = ', '.join(bio_sub[bio_sub.anomaly].Name.unique().tolist())
                if not bio_anomalies:
                    bio_anomalies = "No biochemical anomalies found."
            except Exception as e:
                logger.error(f"Error getting biochemical anomalies: {str(e)}")
                bio_anomalies = "Biochemical data present but anomaly extraction failed."
        elif not bio_sub.empty:
            bio_anomalies = "No biochemical anomalies found."
        else:
            bio_anomalies = "No biochemical data available for this patient in the selected window."
        # Diagnosis description
        try:
            patient_info = df_dx[df_dx.Patient_ID == pid].iloc[0] if not df_dx[df_dx.Patient_ID == pid].empty else None
            diagnosis_desc = patient_info.Description if patient_info is not None else "No diagnosis description available."
        except Exception as e:
            logger.error(f"Error getting patient diagnosis: {str(e)}")
            diagnosis_desc = "No diagnosis description available."
        context = f"""
Patient {pid} was diagnosed with ICD code {icd_code} based on data from {start.date()} to {end.date()}.
Diagnosis description: {diagnosis_desc}
Code definition: {def_text}
Similar past cases (Patient IDs): {similar_pids if similar_pids else 'None found'}
Glucose trend (7d mean start→end): {glucose_trend}
Biochemical anomalies: {bio_anomalies}
"""
        question = f"Why was Patient {pid} diagnosed with {icd_code}?"
        return {
            "context": context,
            "question": question
        }
    except Exception as e:
        logger.error(f"Error creating prompt: {str(e)}")
        raise

def answer_query(prompt_dict):
    """Generate answer using QA pipeline with error handling."""
    try:
        if qa_pipeline is None:
            raise RuntimeError("QA pipeline not initialized")
            
        if not isinstance(prompt_dict, dict) or "question" not in prompt_dict or "context" not in prompt_dict:
            raise ValueError("Invalid prompt dictionary format")
            
        result = qa_pipeline(
            question=prompt_dict["question"],
            context=prompt_dict["context"]
        )
        return result["answer"]
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return "Unable to generate answer due to an error"
