import streamlit as st
import pandas as pd
import requests
import networkx as nx
from pyvis.network import Network
import os
from icd_graph import build_icd_graph  # Import your real graph builder
from data_loader import load_and_preprocess  # Import your data loader

# --- Load Data ---
@st.cache_data
def load_test_data():
    return pd.read_csv("test_data.csv")

@st.cache_data
def load_eval_results():
    return pd.read_csv("evaluation_results_20250624_154425.csv")

test_data = load_test_data()
eval_results = load_eval_results()

# --- ICD Graph Loading (integrated with your project) ---
@st.cache_resource
def load_icd_graph_real():
    # Use your data loader to get patient and diagnosis data
    data_dir = "data"
    df_patients, df_dx, df_bio, df_glu = load_and_preprocess(data_dir)
    # Build the ICD graph using your real function
    G = build_icd_graph(df_patients, df_dx)
    return G

G = load_icd_graph_real()

# --- Streamlit UI ---
st.title("ICD-9 Graph-Augmented QA System")

# Patient and ICD selection
patient_ids = test_data['Patient_ID'].unique()
selected_pid = st.selectbox("Select Patient ID:", patient_ids)
filtered_codes = test_data[test_data['Patient_ID'] == selected_pid]['Code'].unique()
selected_code = st.selectbox("Select ICD Code:", filtered_codes)

# Show description
desc = test_data[(test_data['Patient_ID'] == selected_pid) & (test_data['Code'] == selected_code)]['Description'].values[0]
st.info(f"**Diagnosis Description:** {desc}")

# Show evaluation results for this case
case_index = test_data[(test_data['Patient_ID'] == selected_pid) & (test_data['Code'] == selected_code)].index
if len(case_index) > 0:
    eval_row = eval_results[eval_results['Test_Case'] == (case_index[0] + 1)]
    if not eval_row.empty:
        st.subheader("Evaluation Results")
        st.write(eval_row.T)
    else:
        st.warning("No evaluation results found for this case.")
else:
    st.warning("Case not found in test data.")

# --- Query Backend for Explanation ---
if st.button("Explain Diagnosis"):
    try:
        response = requests.post(
            "http://localhost:8000/explain-diagnosis",
            json={"patient_id": selected_pid, "icd_code": str(selected_code), "window_days": 30},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            st.success("Explanation:")
            st.markdown(f"**{data.get('explanation', 'No explanation returned.')}**")
            st.markdown("---")
            st.markdown("**Similar Cases:**")
            st.code(str(data.get("similar_cases", [])))
            st.markdown("**ICD Definition:**")
            st.code(data.get("definition", "No definition returned."))
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")

# --- ICD Graph Visualization ---
st.subheader("ICD Graph Visualization")

def draw_icd_graph(G, highlight_code=None):
    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    # Highlight the selected ICD code
    if highlight_code and highlight_code in G.nodes:
        node = net.get_node(str(highlight_code))
        if node:
            node['color'] = 'red'
            node['size'] = 30
    net.repulsion(node_distance=120, central_gravity=0.2)
    net.show("icd_graph.html")
    return "icd_graph.html"

graph_file = draw_icd_graph(G, highlight_code=str(selected_code))
with open(graph_file, "r", encoding="utf-8") as f:
    html_string = f.read()
st.components.v1.html(html_string, height=550, scrolling=True)

st.caption("The selected ICD code is highlighted in red on the graph.") 