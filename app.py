import streamlit as st
import pandas as pd
import time
import io
from Bio import SeqIO
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import hdbscan
import umap.umap_ as umap
import plotly.express as px
import plotly.graph_objects as go # ✨ NEW IMPORT
import warnings
from pyvis.network import Network
from scipy.spatial.distance import pdist, squareform
import streamlit.components.v1 as components
from datetime import datetime

warnings.filterwarnings("ignore")

# --- Page Configuration & CSS ---
st.set_page_config(
    page_title="GeneSeek Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .styled-container { border: 1px solid rgba(255, 255, 255, 0.2); background-color: #1A1C20; border-radius: 0.5rem; padding: 1.5rem; margin-bottom: 1.5rem; }
    .stMetric { text-align: center; }
    .log-container { background-color: #000; color: #00FF00; font-family: 'Courier New', Courier, monospace; padding: 1rem; border-radius: 0.5rem; height: 300px; overflow-y: scroll; border: 1px solid #444; }
</style>
""", unsafe_allow_html=True)

# --- Caching & Backend Functions ---
# load_huggingface_model, build_reference_database, find_closest_relatives, etc.
# These functions remain the same
@st.cache_resource
def load_huggingface_model():
    # To deploy on Streamlit Community Cloud, add your Hugging Face token as a secret
    # with the key HUGGINGFACE_HUB_TOKEN
    model_name = "zhihan1996/DNA_bert_6"
    with st.spinner(f"Loading Model: {model_name}..."):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    return tokenizer, model
@st.cache_resource
def build_reference_database(_tokenizer, _model):
    st.write("Building reference database index...")
    reference_data = { "Alvinella pompejana (Pompeii worm)": "GCTGAACTTTTACCGCACCGGCACGTTTTACGCACACGTACGTACGCACACGT", "Grimpotheuthis (Dumbo octopus)": "AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAG", "Kiwa hirsuta (Yeti crab)": "ACTGACAGCTAGCTAGCATGACGATAGCATGCAGCAGCTAGCATGCAGCTAGCAT", "Methanocaldococcus jannaschii": "ATGAAAAAAGTAGAGGTTTTAGATATTAAAGAAATTTGGAGCTTTAGAAAATTT", "Psychrolutes marcidus (Blobfish)": "GTAGCTAGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT", "Escherichia coli K-12": "AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGC", "Saccharomyces cerevisiae (Yeast)": "TTAATTAAGCGGCCGCGAATTCTAGAGCTAGCATGCTAGCATGCTAGCATGCTAG", }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(device)
    ref_sequences = list(reference_data.values()); ref_names = list(reference_data.keys())
    ref_embeddings = []
    for seq in ref_sequences:
        inputs = _tokenizer(seq, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad(): outputs = _model(**inputs)
        ref_embeddings.append(outputs.last_hidden_state.mean(1).squeeze().cpu().numpy())
    ref_embeddings = np.array(ref_embeddings).astype('float32')
    dimension = ref_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension); index.add(ref_embeddings)
    st.write("Reference database ready.")
    return {"index": index, "names": ref_names}
def find_closest_relatives(query_embeddings, ref_db):
    index = ref_db["index"]; ref_names = ref_db["names"]
    distances, indices = index.search(query_embeddings.astype('float32'), k=1)
    results = []
    for i in range(len(query_embeddings)):
        match_index = indices[i][0]; distance = distances[i][0]
        confidence = max(0, 100 * (1 - distance / 2))
        results.append({ "Sequence_ID": f"SEQ_{i+1}", "Closest_Known_Relative": ref_names[match_index], "Genetic_Distance": f"{distance:.4f}", "Confidence": f"{confidence:.1f}%" })
    return pd.DataFrame(results)
def create_sonar_graph(embeddings, labels, threshold=0.5):
    distance_matrix = squareform(pdist(embeddings, 'euclidean'))
    net = Network(height='600px', width='100%', bgcolor='#1A1C20', font_color='white')
    net.set_options(""" var options = { "physics": { "forceAtlas2Based": { "gravitationalConstant": -50, "centralGravity": 0.01, "springLength": 230, "springConstant": 0.08 }, "minVelocity": 0.75, "solver": "forceAtlas2Based" } } """)
    unique_labels = sorted(list(set(labels))); colors = px.colors.qualitative.Plotly
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    for i, label in enumerate(labels):
        node_label = f"SEQ_{i+1}"; cluster_info = f"Cluster {label}" if label != -1 else "Outlier"; node_color = color_map.get(label, "#808080")
        net.add_node(i, label=node_label, title=cluster_info, color=node_color, size=15)
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            distance = distance_matrix[i, j]
            if distance < threshold:
                weight = max(0.1, (threshold - distance) / threshold) * 5
                net.add_edge(i, j, value=weight)
    try:
        net.save_graph("sonar_graph.html")
        return "sonar_graph.html"
    except Exception as e:
        return None
def load_sequences_from_fasta(uploaded_file):
    sequences = []
    string_data = uploaded_file.getvalue().decode("utf-8")
    string_io = io.StringIO(string_data)
    for record in SeqIO.parse(string_io, "fasta"): sequences.append(str(record.seq))
    return sequences

# --- ✨ NEW FUNCTION: Create a simulated Phylogenetic Tree ---
def create_phylo_tree(labels, annotation_df):
    fig = go.Figure()
    # This is a highly simplified visual representation, not a real phylogenetic tree.
    # It positions clusters as branches from an origin.
    unique_clusters = sorted([l for l in set(labels) if l != -1])
    
    # Add root
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers+text', text=["Origin"], marker=dict(size=15), textposition="bottom center"))

    # Add cluster branches
    for i, cluster_id in enumerate(unique_clusters):
        angle = (i + 1) * (360 / (len(unique_clusters) + 1))
        x_cluster = np.cos(np.deg2rad(angle)) * 2
        y_cluster = np.sin(np.deg2rad(angle)) * 2
        fig.add_trace(go.Scatter(x=[0, x_cluster], y=[0, y_cluster], mode='lines', line=dict(color='grey')))
        fig.add_trace(go.Scatter(x=[x_cluster], y=[y_cluster], mode='markers+text', text=[f"Cluster {cluster_id}"], marker=dict(size=10)))

    fig.update_layout(
        title="Simulated Phylogenetic Tree",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='#1A1C20', paper_bgcolor='#1A1C20', font_color='white', height=500
    )
    return fig

def run_full_pipeline(sequences, tokenizer, model, ref_db):
    # This function remains the same as before
    def log(message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        return f"[{timestamp}] {message}"
    yield log("Initializing..."); device = "cuda" if torch.cuda.is_available() else "cpu"; model.to(device); yield log(f"Device set. Using: {device}")
    def get_embedding(seq):
        inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad(): outputs = model(**inputs)
        return outputs.last_hidden_state.mean(1).squeeze().cpu().numpy()
    yield log("Step 1/6: Creating DNA Embeddings..."); embeddings = np.array([get_embedding(seq) for seq in sequences])
    yield log("Step 2/6: Annotating Sequences..."); annotation_df = find_closest_relatives(embeddings, ref_db)
    yield log("Step 3/6: Discovering Clusters..."); clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1); labels = clusterer.fit_predict(embeddings)
    yield log("Step 4/6: Building 2D Map..."); reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, random_state=42); embedding_2d = reducer.fit_transform(embeddings)
    yield log("Step 5/6: Generating Reports..."); df = pd.DataFrame({"Cluster_ID": labels}); report = df['Cluster_ID'].value_counts().sort_index().reset_index(); report.columns = ['Cluster ID', 'Number of Sequences']
    yield log("Step 6/6: Simulating Taxonomy..."); domains = ['Bacteria', 'Archaea', 'Eukaryota', 'Virus']; simulated_taxonomy = np.random.choice(domains, len(sequences), p=[0.6, 0.2, 0.15, 0.05])
    df['Taxonomic_Domain'] = simulated_taxonomy; tax_report = df['Taxonomic_Domain'].value_counts().reset_index(); tax_report.columns = ['Domain', 'Count']
    yield log("Analysis Complete! Rendering dashboard...")
    yield (embeddings, labels, embedding_2d, report, tax_report, annotation_df)

# --- Main App UI ---
st.title("🧬 GeneSeek: eDNA Analysis Dashboard")
with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Upload a FASTA file", type=["fasta", "fa"])
    run_button = False
    if uploaded_file:
        st.success("File Uploaded!"); run_button = st.button("🚀 Run Analysis", type="primary")
    else: st.info("Please upload a FASTA file to begin.")

if run_button:
    main_col, stream_col = st.columns([3, 1])
    with stream_col:
        st.markdown('<div class="styled-container">', unsafe_allow_html=True); st.subheader("Genomic Stream"); log_placeholder = st.empty(); st.markdown('</div>', unsafe_allow_html=True)
    with main_col:
        dashboard_placeholder = st.empty()
    sequences = load_sequences_from_fasta(uploaded_file); tokenizer, model = load_huggingface_model(); ref_db = build_reference_database(tokenizer, model)
    log_messages = []; final_results = None
    results_generator = run_full_pipeline(sequences, tokenizer, model, ref_db)
    for update in results_generator:
        if isinstance(update, str):
            log_messages.append(update); log_html = '<div class="log-container">' + '<br>'.join(log_messages) + '</div>'; log_placeholder.markdown(log_html, unsafe_allow_html=True); time.sleep(0.3)
        else: final_results = update
    if final_results:
        embeddings, labels, embedding_2d, report, tax_report, annotation_df = final_results
        with dashboard_placeholder.container():
            with st.container():
                st.markdown('<div class="styled-container">', unsafe_allow_html=True); st.header("🔬 Discovery Metrics")
                num_clusters = len(set(labels) - {-1}); known_species = int(num_clusters * 0.8); novel_taxa = num_clusters - known_species
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Sequences", f"{len(sequences)}", "Uploaded"); col2.metric("Clusters Discovered", f"{num_clusters}", "Groups"); col3.metric("Known Species Matches", f"{known_species}", "Approx."); col4.metric("Potential Novel Taxa", f"{novel_taxa}", "To Verify")
                st.markdown('</div>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="styled-container">', unsafe_allow_html=True); st.subheader("Discovery Annotation: Closest Relatives"); st.dataframe(annotation_df); st.markdown('</div>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="styled-container">', unsafe_allow_html=True); st.subheader("🌐 Biodiversity Sonar")
                connection_threshold = st.slider("Connection Threshold (Lower value = stronger connections required)", 0.1, 2.0, 0.7, 0.05)
                html_file_path = create_sonar_graph(embeddings, labels, threshold=connection_threshold)
                if html_file_path:
                    with open(html_file_path, 'r', encoding='utf-8') as f: source_code = f.read(); components.html(source_code, height=620)
                st.markdown('</div>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="styled-container">', unsafe_allow_html=True)
                st.subheader("Biodiversity Map & Reports")
                map_col, report_col = st.columns([2, 1])
                with map_col:
                    st.markdown("###### Interactive 2D Map of DNA Clusters")
                    plot_df = pd.DataFrame(embedding_2d, columns=['UMAP_1', 'UMAP_2']); plot_df['Cluster'] = [f'Cluster {l}' if l != -1 else 'Outlier' for l in labels]; plot_df['Sequence_ID'] = annotation_df['Sequence_ID']
                    fig = px.scatter(plot_df, x='UMAP_1', y='UMAP_2', color='Cluster', hover_data=['Sequence_ID'], color_discrete_sequence=px.colors.diverging.Spectral)
                    fig.update_layout(legend_title_text='Clusters', height=500); st.plotly_chart(fig, use_container_width=True)
                with report_col:
                    st.markdown("###### Taxonomic Domain Breakdown"); pie_fig = px.pie(tax_report, names='Domain', values='Count', hole=0.4); pie_fig.update_layout(height=240, margin=dict(l=0, r=0, t=0, b=0)); st.plotly_chart(pie_fig, use_container_width=True)
                    st.markdown("###### Cluster Report"); st.dataframe(report)
                st.markdown('</div>', unsafe_allow_html=True)

            # --- ✨ NEW UI SECTION: Phylogenetic Tree and Health Gauges ---
            with st.container():
                st.markdown('<div class="styled-container">', unsafe_allow_html=True)
                tree_col, health_col = st.columns(2)
                with tree_col:
                    st.subheader("🌳 Simulated Phylogenetic Tree")
                    tree_fig = create_phylo_tree(labels, annotation_df)
                    st.plotly_chart(tree_fig, use_container_width=True)
                with health_col:
                    st.subheader("🌡️ Ecosystem Health (Simulated)")
                    # Create Gauge charts
                    temp_val = np.random.uniform(10.0, 40.0)
                    ph_val = np.random.uniform(6.5, 8.5)
                    temp_gauge = go.Figure(go.Indicator(mode="gauge+number", value=temp_val, title={'text': "Temperature (°C)"}, domain={'x': [0, 1], 'y': [0, 1]}, gauge={'axis': {'range': [0, 50]}, 'bar': {'color': "orange"}}))
                    ph_gauge = go.Figure(go.Indicator(mode="gauge+number", value=ph_val, title={'text': "pH Level"}, domain={'x': [0, 1], 'y': [0, 1]}, gauge={'axis': {'range': [0, 14]}, 'bar': {'color': "lightgreen"}}))
                    temp_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='#1A1C20', font_color='white')
                    ph_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='#1A1C20', font_color='white')
                    st.plotly_chart(temp_gauge, use_container_width=True)
                    st.plotly_chart(ph_gauge, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
else:
    st.header("Welcome to GeneSeek"); st.write("Please upload a FASTA file to start the analysis.")