import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
import numpy as np
import os
import argparse
import sys

# Optional: for interactive HTML export
try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False

def load_icd_descriptions():
    diag_csv = os.path.join('data', 'Diagnostics.csv')
    if not os.path.exists(diag_csv):
        print(f"Diagnostics.csv not found at {diag_csv}")
        return {}
    df = pd.read_csv(diag_csv)
    return dict(zip(df['Code'], df['Description']))

def generate_icd_graph_from_diagnostics(top_n=30, annotate=False, export_html=False, export_json=False, export_csv=False):
    DIAG_CSV = os.path.join('data', 'Diagnostics.csv')
    OUTPUT_PNG = 'icd_graph_from_diagnostics.png'
    OUTPUT_HTML = 'icd_graph_interactive.html'
    OUTPUT_JSON = 'icd_graph.json'
    OUTPUT_CSV = 'icd_graph_edges.csv'
    print(f"Loading {DIAG_CSV} ...")
    df = pd.read_csv(DIAG_CSV)
    patient_codes = df.groupby('Patient_ID')['Code'].apply(list)
    G = nx.Graph()
    for codes in patient_codes:
        unique_codes = set(codes)
        for code in unique_codes:
            if not G.has_node(code):
                G.add_node(code)
        for c1, c2 in combinations(unique_codes, 2):
            if G.has_edge(c1, c2):
                G[c1][c2]['weight'] += 1
            else:
                G.add_edge(c1, c2, weight=1)
    all_codes = [code for codes in patient_codes for code in codes]
    code_counts = Counter(all_codes)
    for node in G.nodes():
        G.nodes[node]['weight'] = code_counts.get(node, 0)
    icd_desc = load_icd_descriptions() if annotate else {}
    def visualize_icd_graph(G, top_n=30, output_png=OUTPUT_PNG):
        node_weights = {node: G.nodes[node]['weight'] for node in G.nodes()}
        top_nodes = sorted(node_weights.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_nodes = [node for node, _ in top_nodes]
        subgraph = G.subgraph(top_nodes)
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(subgraph, seed=42)
        nodes = nx.draw_networkx_nodes(
            subgraph, pos,
            node_size=[subgraph.nodes[node]['weight']*20 for node in subgraph.nodes()],
            node_color=[subgraph.nodes[node]['weight'] for node in subgraph.nodes()],
            cmap=plt.cm.viridis,
            alpha=0.9
        )
        nx.draw_networkx_edges(
            subgraph, pos,
            width=[subgraph[u][v]['weight']/2 for u, v in subgraph.edges()],
            alpha=0.3
        )
        labels = {node: f"{node}\n{icd_desc.get(node, '')}" if annotate else node for node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=10, font_weight='bold')
        plt.title(f'ICD Code Co-occurrence Network (Top {top_n} Codes)')
        plt.colorbar(nodes, label='ICD Code Frequency')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_png, dpi=200)
        plt.close()
        print(f'Graph saved to {output_png}')
        # Summary
        print(f"Top {top_n} ICD codes by frequency:")
        for node in top_nodes:
            print(f"{node}: {G.nodes[node]['weight']}" + (f" - {icd_desc.get(node, '')}" if annotate else ""))
    visualize_icd_graph(G, top_n=top_n, output_png=OUTPUT_PNG)
    # Export interactive HTML
    if export_html and HAS_PYVIS:
        node_weights = {node: G.nodes[node]['weight'] for node in G.nodes()}
        top_nodes = sorted(node_weights.items(), key=lambda x: x[1], reverse=True)[:top_n]
        subgraph = G.subgraph(top_nodes)
        net = Network(notebook=False, width='1000px', height='800px')
        for node in subgraph.nodes():
            net.add_node(node, label=f"{node}\n{icd_desc.get(node, '')}" if annotate else node, title=f'Frequency: {subgraph.nodes[node]["weight"]}')
        for u, v, d in subgraph.edges(data=True):
            net.add_edge(u, v, value=d['weight'])
        net.show(OUTPUT_HTML)
        print(f'Interactive graph saved to {OUTPUT_HTML}')
    elif export_html:
        print("pyvis not installed. Skipping HTML export.")
    # Export as JSON
    if export_json:
        nx.write_gexf(G, OUTPUT_JSON)
        print(f'Graph exported as GEXF (can be opened in Gephi): {OUTPUT_JSON}')
    # Export as CSV
    if export_csv:
        edges = nx.to_pandas_edgelist(G)
        edges.to_csv(OUTPUT_CSV, index=False)
        print(f'Graph edges exported as CSV: {OUTPUT_CSV}')

def generate_metric_graph_from_eval(corr_threshold=0.2):
    EVAL_CSV = 'evaluation_results_20250624_154425.csv'
    OUTPUT_PNG = 'metric_graph_from_eval.png'
    print(f"Loading {EVAL_CSV} ...")
    df = pd.read_csv(EVAL_CSV)
    corr = df.corr()
    G = nx.Graph()
    metrics = corr.columns
    for metric in metrics:
        G.add_node(metric)
    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            if i < j:
                weight = corr.loc[m1, m2]
                if abs(weight) >= corr_threshold:
                    G.add_edge(m1, m2, weight=weight)
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    edges = nx.draw_networkx_edges(
        G, pos,
        width=[abs(G[u][v]['weight'])*5 for u, v in G.edges()],
        edge_color=[G[u][v]['weight'] for u, v in G.edges()],
        edge_cmap=plt.cm.coolwarm,
        alpha=0.7
    )
    nodes = nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    plt.title('Metric Correlation Network')
    plt.colorbar(edges, label='Correlation')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200)
    plt.close()
    print(f'Graph saved to {OUTPUT_PNG}')
    # Summary
    print("Metric correlation summary (edges shown):")
    for u, v, d in G.edges(data=True):
        print(f"{u} <-> {v}: correlation={d['weight']:.2f}")

def main():
    parser = argparse.ArgumentParser(description="ICD/Metric Graph Generator")
    parser.add_argument('--mode', choices=['icd', 'metric'], help='Which graph to generate')
    parser.add_argument('--top_n', type=int, default=30, help='Top N ICD codes to visualize')
    parser.add_argument('--annotate', action='store_true', help='Annotate ICD nodes with descriptions')
    parser.add_argument('--export_html', action='store_true', help='Export ICD graph as interactive HTML (pyvis)')
    parser.add_argument('--export_json', action='store_true', help='Export ICD graph as GEXF (for Gephi)')
    parser.add_argument('--export_csv', action='store_true', help='Export ICD graph edges as CSV')
    parser.add_argument('--corr_threshold', type=float, default=0.2, help='Correlation threshold for metric graph')
    parser.add_argument('--noinput', action='store_true', help='Run non-interactively (for automation)')
    args = parser.parse_args()

    if not args.mode and not args.noinput:
        print("Select which graph to generate:")
        print("1. ICD co-occurrence graph from Diagnostics.csv")
        print("2. Metric correlation graph from evaluation_results_20250624_154425.csv")
        choice = input("Enter 1 or 2: ").strip()
        if choice == '1':
            args.mode = 'icd'
        elif choice == '2':
            args.mode = 'metric'
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)
    if args.mode == 'icd':
        generate_icd_graph_from_diagnostics(
            top_n=args.top_n,
            annotate=args.annotate,
            export_html=args.export_html,
            export_json=args.export_json,
            export_csv=args.export_csv
        )
    elif args.mode == 'metric':
        generate_metric_graph_from_eval(corr_threshold=args.corr_threshold)
    else:
        print("No valid mode selected. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main() 