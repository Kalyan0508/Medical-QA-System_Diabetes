import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from fastapi.responses import FileResponse
import os
from fastapi import APIRouter
from datetime import timedelta

def build_icd_graph(df_patients, df_dx):
    """Build a graph of ICD codes based on patient diagnoses."""
    G = nx.Graph()
    
    # Add nodes (ICD codes)
    for code in df_dx['Code'].unique():
        G.add_node(code)
    
    # Add edges based on co-occurrence in patients
    for pid, grp in df_dx.groupby("Patient_ID"):
        codes = grp["Code"].tolist()
        # Add edges between all pairs of codes for this patient
        for c1, c2 in combinations(set(codes), 2):
            if G.has_edge(c1, c2):
                G[c1][c2]['weight'] += 1
            else:
                G.add_edge(c1, c2, weight=1)
    
    # Calculate node weights based on frequency
    code_counts = df_dx['Code'].value_counts()
    for node in G.nodes():
        G.nodes[node]['weight'] = code_counts.get(node, 0)
    
    return G

def visualize_graph(G, top_n=20):
    """Visualize the top N most frequent ICD codes and their connections."""
    # Get top N nodes by weight
    node_weights = {node: G.nodes[node]['weight'] for node in G.nodes()}
    top_nodes = sorted(node_weights.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_nodes = [node for node, _ in top_nodes]
    
    # Create subgraph with top nodes
    subgraph = G.subgraph(top_nodes)
    
    # Draw the graph
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(subgraph)
    
    # Draw nodes
    nx.draw_networkx_nodes(subgraph, pos, 
                          node_size=[G.nodes[node]['weight']*10 for node in subgraph.nodes()],
                          node_color='lightblue',
                          alpha=0.7)
    
    # Draw edges
    nx.draw_networkx_edges(subgraph, pos, alpha=0.3)
    
    # Draw labels
    nx.draw_networkx_labels(subgraph, pos, font_size=8)
    
    plt.title(f"Top {top_n} ICD Codes and Their Connections")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return subgraph

def save_graph_to_neo4j(G, uri, user, pwd):
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session() as sess:
        # Clear existing
        sess.run("MATCH (n) DETACH DELETE n")
        # Create nodes
        for n, data in G.nodes(data=True):
            sess.run("CREATE (:ICD {code:$code})", code=n)
        # Create edges
        for u, v, attrs in G.edges(data=True):
            sess.run("""
              MATCH (a:ICD {code:$u}), (b:ICD {code:$v})
              CREATE (a)-[:RELATED {relation:$rel, weight:$w}]->(b)
            """, u=u, v=v, rel=attrs.get("relation"), w=attrs.get("weight",1))

def create_icd_graph(df_dx):
    # Count occurrences of each ICD code
    icd_counts = df_dx['Code'].value_counts()
    plt.figure(figsize=(10, 6))
    icd_counts.plot(kind='bar')
    plt.title('ICD Code Frequency')
    plt.xlabel('ICD Code')
    plt.ylabel('Count')
    plt.tight_layout()
    # Save the plot
    output_path = "icd_graph.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

router = APIRouter()

@router.get("/icd-graph")
def get_icd_graph():
    # Make sure df_dx is loaded and available
    output_path = create_icd_graph(df_dx)
    return FileResponse(output_path, media_type="image/png")
