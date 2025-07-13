import faiss
import numpy as np
from whoosh import index, fields, writing
from whoosh.qparser import QueryParser
import os
from typing import List, Tuple, Dict
import logging
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_faiss_index(patient_vecs: List[Tuple[str, np.ndarray]], dim: int) -> faiss.Index:
    """Build FAISS index for patient vectors."""
    try:
        if not patient_vecs:
            raise ValueError("Empty patient vectors list provided")
            
        if dim <= 0:
            raise ValueError("Dimension must be positive")
            
        # Validate input vectors
        for pid, vec in patient_vecs:
            if not isinstance(pid, str):
                raise TypeError(f"Invalid patient ID type: {type(pid)}")
            if not isinstance(vec, np.ndarray):
                raise TypeError(f"Invalid vector type: {type(vec)}")
            if vec.shape[0] != dim:
                raise ValueError(f"Vector dimension mismatch: expected {dim}, got {vec.shape[0]}")
        
        # Convert string IDs to integers using a mapping
        id_to_int = {pid: i for i, (pid, _) in enumerate(patient_vecs)}
        ids = np.array([id_to_int[pid] for pid, _ in patient_vecs], dtype='int64')
        
        # Stack vectors with validation
        try:
            mat = np.vstack([vec for _, vec in patient_vecs]).astype('float32')
        except Exception as e:
            raise ValueError(f"Error stacking vectors: {str(e)}")
        
        # Create and configure index with error handling
        try:
            idx = faiss.IndexFlatL2(dim)
            idx = faiss.IndexIDMap(idx)
            idx.add_with_ids(mat, ids)
        except Exception as e:
            raise RuntimeError(f"Error creating FAISS index: {str(e)}")
        
        # Store the mapping for later use
        idx.id_to_code = {i: pid for pid, i in id_to_int.items()}
        
        return idx
    except Exception as e:
        logger.error(f"Error building FAISS index: {str(e)}")
        raise

def build_whoosh_index(icd_definitions: Dict[str, str], idx_dir: str = "icd_index") -> index.Index:
    """Build Whoosh index for ICD code definitions."""
    try:
        if not icd_definitions:
            raise ValueError("Empty ICD definitions dictionary provided")
            
        # Validate input data
        for code, desc in icd_definitions.items():
            if not isinstance(code, str):
                raise TypeError(f"Invalid ICD code type: {type(code)}")
            if not isinstance(desc, str):
                raise TypeError(f"Invalid description type: {type(desc)}")
        
        # Create or clean index directory
        if os.path.exists(idx_dir):
            try:
                shutil.rmtree(idx_dir)
            except Exception as e:
                raise RuntimeError(f"Error cleaning index directory: {str(e)}")
        
        try:
            os.mkdir(idx_dir)
        except Exception as e:
            raise RuntimeError(f"Error creating index directory: {str(e)}")
            
        # Create schema and index
        try:
            schema = fields.Schema(
                code=fields.ID(stored=True),
                desc=fields.TEXT(stored=True)
            )
            
            ix = index.create_in(idx_dir, schema)
            writer = ix.writer()
            
            for code, desc in icd_definitions.items():
                writer.add_document(code=code, desc=desc)
                
            writer.commit()
            return ix
        except Exception as e:
            raise RuntimeError(f"Error creating Whoosh index: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error building Whoosh index: {str(e)}")
        raise

def retrieve(icd_code: str, faiss_idx: faiss.Index, patient_vecs: List[Tuple[str, np.ndarray]], 
            whoosh_ix: index.Index, k: int = 5) -> Tuple[str, List[str]]:
    """Retrieve similar cases and definition for an ICD code."""
    try:
        if not isinstance(icd_code, str):
            raise TypeError(f"Invalid ICD code type: {type(icd_code)}")
            
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
            
        if faiss_idx is None:
            raise ValueError("FAISS index not initialized")
            
        if whoosh_ix is None:
            raise ValueError("Whoosh index not initialized")
        
        # Get definition from Whoosh with error handling
        try:
            qp = QueryParser("code", whoosh_ix.schema)
            q = qp.parse(icd_code)
            
            with whoosh_ix.searcher() as s:
                results = s.search(q)
                def_text = results[0]["desc"] if results else ""
        except Exception as e:
            logger.error(f"Error retrieving definition: {str(e)}")
            def_text = ""
        
        # Get vector for the query code with validation
        query_vec = None
        try:
            for pid, vec in patient_vecs:
                if pid == icd_code:
                    query_vec = vec
                    break
                    
            if query_vec is None:
                logger.warning(f"No vector found for ICD code {icd_code}")
                return def_text, []
        except Exception as e:
            logger.error(f"Error finding query vector: {str(e)}")
            return def_text, []
            
        # Search for similar codes with error handling
        try:
            D, I = faiss_idx.search(query_vec.reshape(1,-1), k+1)  # k+1 to exclude self
            
            # Convert back to ICD codes with validation
            similar_codes = []
            for i in I[0]:
                if i != -1 and i in faiss_idx.id_to_code:
                    similar_codes.append(faiss_idx.id_to_code[i])
            
            return def_text, similar_codes
        except Exception as e:
            logger.error(f"Error searching for similar codes: {str(e)}")
            return def_text, []
            
    except Exception as e:
        logger.error(f"Error in retrieve: {str(e)}")
        return "", []
