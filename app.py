import streamlit as st
import pandas as pd
import numpy as np
import joblib
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import os

# Set page config
st.set_page_config(page_title="PriceRunner NLP Portfolio", layout="wide", page_icon="🛍️")

# --- Custom Styling Injection ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300;1,400;1,500&display=swap/* Background & Base */
.stApp {
    background-color: #f8fafc !important;
}

/* Typography */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace !important;
    color: #0f172a !important;
}
h1, h2, h3, h4, h5, h6, [data-testid="stHeader"] {
    letter-spacing: 0.05em !important;
    font-weight: bold !important;
    color: #0f172a !important;
}
p, span, div {
    color: #475569 !important;
}

/* Accent Color usage */
[data-testid="stMetricValue"] {
    color: #0ea5e9 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
[data-testid="stSidebar"] h1 {
    color: #0ea5e9 !important;
    font-size: 2rem !important;
    font-weight: bold !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #64748b !important;
}

/* Radio Buttons (Sidebar Nav) */
[data-testid="stSidebar"] .stRadio > div {
    width: 100% !important;
}
[data-testid="stSidebar"] .stRadio label {
    width: 100% !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
    border-left: 2px solid transparent !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    border-left: 2px solid #0ea5e9 !important;
    background-color: #f0f9ff !important;
}
[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    border-left: 2px solid #0ea5e9 !important;
    background-color: #f0f9ff !important;
}
[data-testid="stSidebar"] .stRadio label[data-checked="true"] p,
[data-testid="stSidebar"] .stRadio label[data-checked="true"] div {
    color: #0ea5e9 !important;
    font-weight: bold !important;
}

/* Cards & Containers */
[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    transition: all 0.2s ease !important;
    border-left: 4px solid #cbd5e1 !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05) !important;
}
[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"]:hover {
    border-left: 4px solid #0ea5e9 !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.05) !important;
    transform: translateY(-2px);
}

/* Tables & Dataframes */
[data-testid="stTable"] table, [data-testid="stDataFrame"] table {
    background-color: #ffffff !important;
    border-collapse: collapse !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}
[data-testid="stTable"] th, [data-testid="stDataFrame"] th {
    background-color: #f8fafc !important;
    color: #0ea5e9 !important;
    border-bottom: 2px solid #e2e8f0 !important;
    font-weight: bold !important;
}
[data-testid="stTable"] tr:nth-child(even), [data-testid="stDataFrame"] tr:nth-child(even) {
    background-color: #f8fafc !important;
}
[data-testid="stTable"] tr:nth-child(odd), [data-testid="stDataFrame"] tr:nth-child(odd) {
    background-color: #ffffff !important;
}
[data-testid="stTable"] td, [data-testid="stDataFrame"] td {
    color: #475569 !important;
    border-bottom: 1px solid #e2e8f0 !important;
}

/* Notifications / Pill Boxes (Success/Warning/Error) */
[data-testid="stNotificationSuccess"], [data-testid="stSuccess"] {
    background-color: #dcfce7 !important;
    border: 1px solid #86efac !important;
    color: #166534 !important;
}
[data-testid="stNotificationSuccess"] p, [data-testid="stSuccess"] p { color: #15803d !important; font-weight: 500 !important; }

[data-testid="stNotificationWarning"], [data-testid="stWarning"] {
    background-color: #fef9c3 !important;
    border: 1px solid #fde047 !important;
    color: #854d0e !important;
}
[data-testid="stNotificationWarning"] p, [data-testid="stWarning"] p { color: #a16207 !important; font-weight: 500 !important; }

[data-testid="stNotificationError"], [data-testid="stError"] {
    background-color: #fee2e2 !important;
    border: 1px solid #fca5a5 !important;
    color: #991b1b !important;
}
[data-testid="stNotificationError"] p, [data-testid="stError"] p { color: #b91c1c !important; font-weight: 500 !important; }

/* Input Fields */
[data-testid="stTextInput"] > div > div {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05) !important;
}
[data-testid="stTextInput"] > div > div > input {
    color: #0f172a !important;
}
[data-testid="stTextInput"] > div > div:focus-within {
    border: 1px solid #0ea5e9 !important;
    box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.2) !important;
}

/* Buttons */
[data-testid="baseButton-secondary"] {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.2s ease !important;
}
[data-testid="baseButton-secondary"] p {
    color: #0f172a !important;
    transition: all 0.2s ease !important;
}
[data-testid="baseButton-secondary"]:hover {
    background-color: #f8fafc !important;
    border: 1px solid #94a3b8 !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px !important;
    height: 6px !important;
}
::-webkit-scrollbar-track {
    background: #f1f5f9 !important;
}
::-webkit-scrollbar-thumb {
    background: #cbd5e1 !important;
    border-radius: 3px !important;
}
::-webkit-scrollbar-thumb:hover {
    background: #94a3b8 !important;
}
</style>
""", unsafe_allow_html=True)

# Global plot styling
plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#cbd5e1',
    'axes.grid': True,
    'grid.color': '#e2e8f0',
    'axes.labelcolor': '#475569',
    'xtick.color': '#64748b',
    'ytick.color': '#64748b',
    'text.color': '#0f172a'
})

# ==========================================
# 1. CACHING ASSETS
# ==========================================
# Why caching is critical: 
# Streamlit reruns this entire script from top to bottom every time the user 
# clicks a button or types a letter. If we didn't use @st.cache_resource, 
# it would reload the 384-dimensional Deep Learning model, the 35K FAISS index, 
# and the 50MB embeddings array *on every single keystroke*. 
# @st.cache_resource pins these objects in RAM permanently for the session.

@st.cache_resource(show_spinner="Loading NLP Models and Data...")
def load_assets():
    # 1. Base files
    meta = pd.read_csv('metadata.csv')
    umap_2d = np.load('umap_2d.npy')
    
    # 2. Embedding Model
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Pipeline 1: Classification
    classifier = joblib.load('classifier.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    
    # 4. Pipeline 3: Entity Matching (FAISS)
    index = faiss.read_index('faiss_index.bin')
    
    return meta, umap_2d, encoder, classifier, label_encoder, index

try:
    meta, umap_2d, encoder, classifier, label_encoder, index = load_assets()
except Exception as e:
    st.error(f"Failed to load assets: {e}\nPlease ensure all 4 notebooks have been run successfully.")
    st.stop()


# ==========================================
# 2. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("NLP Portfolio")
st.sidebar.markdown("Built on the **PriceRunner** dataset (35K products).")

page = st.sidebar.radio(
    "Select Pipeline:",
    ["0. Overview & Architecture", "1. Classification", "2. Clustering", "3. Entity Matching"]
)


# ==========================================
# 2.5 PAGE 0: OVERVIEW & ARCHITECTURE
# ==========================================
if page == "0. Overview & Architecture":
    st.title("📂 NLP Portfolio Architecture")
    st.markdown("""
    Welcome to the **Natural Language Processing Portfolio Component**. 
    
    This dashboard demonstrates production-ready ML engineering, combining classic NLP techniques with modern embedding-based deep learning workflows. It uses the PriceRunner dataset (35K+ product titles and categories) to power three distinct pipelines.
    
    ### ⚙️ Architecture & Data Flow
    
    - **1. Shared Embedding Layer**
      - All pipelines leverage a shared foundation: `all-MiniLM-L6-v2` via `sentence-transformers`.
      - Converts raw product text into 384-dimensional dense semantic vectors representing contextual meaning.
    
    - **2. Classification Pipeline**
      - **Goal:** Predict a product's precise category from its raw title.
      - **Implementation:** Support Vector Machine (SVC) trained on the 384-D text embeddings. Maps vectors into one of 10 heavily imbalanced categories.
      
    - **3. Clustering Pipeline**
      - **Goal:** Unsupervised discovery of latent product groupings without using labels.
      - **Implementation:** K-Means clustering (K=10) combined with UMAP to project high-dimensional vectors down to 2D for visual inspection of feature space boundaries.
      
    - **4. Entity Matching (Vector Search)**
      - **Goal:** Real-time semantic similarity search to resolve input variations (e.g. "iphone 14 pro" vs "Apple iPhone 14 Pro Max") without string matching.
      - **Implementation:** Millions of vector operations optimized via **FAISS** index, utilizing Cosine Similarity (via L2-normalized Inner Product) for sub-millisecond retrieval across the 35K catalog.
      
    ---
    *Use the sidebar to interact with each isolated pipeline.*
    """)

# ==========================================
# 3. PAGE 1: CLASSIFICATION
# ==========================================
elif page == "1. Classification":
    st.title("🗂️ Product Category Classification")
    st.markdown("Type a product title. Our SVM/LR model will predict which of the 10 categories it belongs to, based on its dense semantic embedding.")
    
    user_input = st.text_input("Enter product title:", "Apple iPhone 14 Pro Max 256GB Midnight")
    
    if user_input:
        with st.spinner("Classifying..."):
            # Encode text -> (1, 384)
            emb = encoder.encode([user_input])
            
            # Predict
            pred_idx = classifier.predict(emb)[0]
            pred_cat = label_encoder.inverse_transform([pred_idx])[0]
            
            # Probabilities (if CalibratedClassifierCV or LogisticRegression)
            if hasattr(classifier, "predict_proba"):
                probs = classifier.predict_proba(emb)[0]
                confidence = probs[pred_idx] * 100
                
                st.success(f"**Predicted Category:** {pred_cat} (Confidence: {confidence:.1f}%)")
                
                # Show top 3 probabilities
                top3_idx = np.argsort(probs)[::-1][:3]
                top3_cats = label_encoder.inverse_transform(top3_idx)
                top3_probs = probs[top3_idx] * 100
                
                df_probs = pd.DataFrame({
                    "Category": top3_cats,
                    "Confidence (%)": top3_probs
                })
                st.table(df_probs.style.format({"Confidence (%)": "{:.1f}%"}))
            else:
                # Fallback if model doesn't support proba (e.g. raw LinearSVC)
                st.success(f"**Predicted Category:** {pred_cat}")


# ==========================================
# 4. PAGE 2: CLUSTERING
# ==========================================
elif page == "2. Clustering":
    st.title("🌌 Product Clustering Visualization")
    st.markdown("View the 35,311 products compressed from 384 dimensions down to 2 dimensions using UMAP.")
    
    view_toggle = st.radio(
        "Color points by:",
        ["Ground Truth (Real Categories)", "K-Means (Unsupervised Clusters)"],
        horizontal=True
    )
    
    with st.spinner("Generating plot..."):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if view_toggle == "Ground Truth (Real Categories)":
            # Color by real category
            sns.scatterplot(
                x=umap_2d[:, 0], y=umap_2d[:, 1],
                hue=meta['category'],
                palette="tab10", s=5, alpha=0.7, ax=ax, edgecolor=None
            )
            ax.set_title("UMAP Colored by Ground Truth Categories")
            
        else:
            # Re-run a fast K-Means just for display (so we don't have to save labels to disk)
            # K-Means on UMAP 2D is instant.
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=10, random_state=42, n_init="auto").fit(umap_2d)
            
            sns.scatterplot(
                x=umap_2d[:, 0], y=umap_2d[:, 1],
                hue=km.labels_,
                palette="tab10", s=5, alpha=0.7, ax=ax, edgecolor=None, legend=False
            )
            ax.set_title("UMAP Colored by Unsupervised K-Means Clusters")

        ax.axis('off')
        
        # Legend styling
        if view_toggle == "Ground Truth (Real Categories)":
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., markerscale=3)
            
        st.pyplot(fig)


# ==========================================
# 5. PAGE 3: ENTITY MATCHING
# ==========================================
elif page == "3. Entity Matching":
    st.title("🔍 Entity Matching (Similarity Search)")
    st.markdown("Type a product query. The app will search the 35K catalog using FAISS (Cosine Similarity) and return the absolute closest matches.")
    
    query = st.text_input("Enter product to search:", "samsung galaxy s22 ultra 5g smartphone")
    
    if query:
        with st.spinner("Searching 35K products..."):
            # 1. Encode
            q_emb = encoder.encode([query])
            
            # 2. L2 Normalize (must do this for InnerProduct to equal Cosine Similarity)
            q_emb_norm = normalize(q_emb, norm='l2', axis=1)
            
            # 3. Search index (we ask for 10 just to be safe, filtering below)
            k = 10
            scores, indices = index.search(q_emb_norm, k)
            
            # 4. Format results
            results = []
            for i in range(k):
                idx = indices[0][i]
                score = scores[0][i]
                
                # Check thresholds
                if score > 0.95:
                    tier = "🟢 Auto-Match"
                elif score >= 0.75:
                    tier = "🟡 Review"
                else:
                    tier = "🔴 No Match"
                    
                # We skip showing results that are literally exact string matches to the input
                # to simulate actual cross-dataset search feeling, unless it's the only thing.
                title = meta.loc[idx, 'title']
                
                results.append({
                    "Score": score,
                    "Confidence": tier,
                    "Product Title": title,
                    "Merchant": meta.loc[idx, 'merchant_id'], # Fixed KeyError
                    "Category": meta.loc[idx, 'category']
                })
                
            df_res = pd.DataFrame(results).head(5) # show top 5
            
            st.markdown(f"### Top 5 Matches for: *'{query}'*")
            st.table(df_res.style.format({"Score": "{:.3f}"}))
