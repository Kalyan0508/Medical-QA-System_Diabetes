from node2vec import Node2Vec
import numpy as np
import logging
from typing import Dict
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_node2vec(G: nx.Graph, emb_size: int = 128, walk_len: int = 30, num_walks: int = 200) -> Dict[str, np.ndarray]:
    """
    Train Node2Vec embeddings on the graph.
    
    Args:
        G: NetworkX graph
        emb_size: Embedding dimension
        walk_len: Length of random walks
        num_walks: Number of walks per node
        
    Returns:
        Dictionary mapping node IDs to their embeddings
    """
    try:
        # Validate input parameters
        if not isinstance(G, nx.Graph):
            raise TypeError(f"Invalid graph type: {type(G)}")
            
        if not G.nodes():
            raise ValueError("Empty graph provided")
            
        if emb_size <= 0:
            raise ValueError("Embedding size must be positive")
            
        if walk_len <= 0:
            raise ValueError("Walk length must be positive")
            
        if num_walks <= 0:
            raise ValueError("Number of walks must be positive")
        
        logger.info("Initializing Node2Vec...")
        try:
            n2v = Node2Vec(
                G,
                dimensions=emb_size,
                walk_length=walk_len,
                num_walks=num_walks,
                workers=4,
                p=1,  # Return parameter
                q=1   # In-out parameter
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing Node2Vec: {str(e)}")
        
        logger.info("Training embeddings...")
        try:
            model = n2v.fit(
                window=10,
                min_count=1,
                batch_words=4,
                workers=4
            )
        except Exception as e:
            raise RuntimeError(f"Error training embeddings: {str(e)}")
        
        logger.info("Building embedding lookup...")
        try:
            embeddings = {}
            for node in G.nodes():
                try:
                    embeddings[node] = model.wv.get_vector(node)
                except KeyError:
                    logger.warning(f"No embedding found for node {node}")
                    continue
                except Exception as e:
                    logger.error(f"Error getting embedding for node {node}: {str(e)}")
                    continue
                    
            if not embeddings:
                raise ValueError("No valid embeddings were generated")
                
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"Error building embedding lookup: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error in train_node2vec: {str(e)}")
        raise
