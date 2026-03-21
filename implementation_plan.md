# NLP Portfolio Project — Implementation Plan

## Background

The dataset ([pricerunner_aggregate.csv](file:///C:/Users/Asus/Desktop/2000/pricerunner_aggregate.csv)) has **35,311 rows × 7 columns**: `Product ID`, `Product Title`, `Category ID`, `Category Label`, `Merchant ID`, `Merchant Name`, and (likely) `Price`. Titles are pre-cleaned (case-folded, punctuation removed). The project builds three NLP pipelines on top of a shared `sentence-transformers` embedding layer, delivered as four Jupyter notebooks and one Streamlit app.

---

## Proposed Changes

### Shared Foundation

#### [NEW] [00_embeddings.ipynb](file:///C:/Users/Asus/Desktop/2000/00_embeddings.ipynb)

**Cells in order:**

1. **Install check** – verify required libraries are importable (`sentence_transformers`, `faiss`, `umap`, `hdbscan`).
2. **Load data** – `pd.read_csv()`, print `.shape`, `.dtypes`, `.head()`, `.value_counts()` for category and merchant columns.
3. **Why Sentence Transformers?** – Markdown cell explaining:
   - TF-IDF encodes vocabulary frequencies; two titles with the same meaning but different words get a near-zero similarity. Sentence Transformers map semantics, so *"iphone 14 pro max"* and *"apple smartphone 14 pro"* will still land nearby in embedding space.
   - For ~35K short titles, `all-MiniLM-L6-v2` is fast (CPU batch in minutes), produces 384-dim dense vectors, and is free/local.
4. **Embed titles** – `SentenceTransformer('all-MiniLM-L6-v2').encode(titles, batch_size=256, show_progress_bar=True)` → numpy array `(35311, 384)`.
5. **Save to disk** – `np.save('embeddings.npy', emb)` + `df[['Product ID','Product Title','Category Label','Merchant Name']].to_csv('metadata.csv', index=False)`. Saving once prevents multi-minute re-computation in every pipeline notebook.
6. **Sanity check** – pick 3 random titles; compute cosine similarity against all others (`sklearn.metrics.pairwise.cosine_similarity`); print top 5 neighbours. If the results are semantically coherent (e.g., neighbours of a laptop title are also laptops), embeddings are working.

---

### Pipeline 1 — Classification

#### [NEW] [01_classification.ipynb](file:///C:/Users/Asus/Desktop/2000/01_classification.ipynb)

**Cells in order:**

1. **Load embeddings + metadata** – `np.load`, `pd.read_csv`.
2. **Why stratified split?** – With 10 unequal categories, a random split might undersample a small class in training. Stratification preserves each class's proportion in both halves.
3. **Train/test split** – `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`.
4. **Why Logistic Regression?** – Embeddings are already rich feature vectors; a linear head is sufficient and trains in seconds. A neural classifier would require epochs, GPU, and careful tuning for only marginal gain on 35K samples.
5. **Train LR** – `LogisticRegression(max_iter=1000, C=1.0, random_state=42)`.
6. **Evaluate** – `accuracy_score` + `classification_report` (per-class precision/recall/F1).
7. **Confusion matrix** – `ConfusionMatrixDisplay.from_estimator` with a `seaborn` heatmap. Explain in markdown which pairs are confused and *why* (e.g., "Tablets" ↔ "Computers" share vocabulary like *"screen"*, *"gb"*, *"processor"*).
8. **Optional SVM** – `SVC(kernel='rbf', C=1.0)` trained and evaluated the same way. Compare with LR: SVMs tend to win on small, well-separated embedding spaces; LR is more interpretable and faster.

---

### Pipeline 2 — Clustering

#### [NEW] [02_clustering.ipynb](file:///C:/Users/Asus/Desktop/2000/02_clustering.ipynb)

**Cells in order:**

1. **Load embeddings + metadata**.
2. **Why UMAP?** – PCA is linear and destroys curved manifolds; t-SNE doesn't preserve global structure and is slow (O(n²)). UMAP is fast (~1 min for 35K), preserves both local *and* global topology, and produces embeddings suitable for downstream clustering. Parameters: `n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine'`.
3. **UMAP reduction** – fit on full 384-dim array, save 2D result.
4. **K-Means (K=10)** – `KMeans(n_clusters=10, random_state=42, n_init='auto')`. Explain: K-Means partitions space into K Voronoi cells by iteratively moving centroids; every point gets exactly one cluster. Good when clusters are roughly spherical and similar in size.
5. **HDBSCAN** – `hdbscan.HDBSCAN(min_cluster_size=50, metric='euclidean')` on UMAP 2D coordinates. Explain: HDBSCAN is density-based; it finds clusters of arbitrary shape, leaves low-density points as noise (label = -1), and doesn't require K to be specified in advance. Better for messy real-world data but harder to control.
6. **Dual UMAP plots** – side-by-side scatter: left coloured by K-Means label, right coloured by ground-truth category. Visual comparison reveals how well the unsupervised structure matches the true taxonomy.
7. **Silhouette Score** – `silhouette_score(X_2d, labels)` for both K-Means and HDBSCAN (exclude noise points for HDBSCAN). Higher = more compact, well-separated clusters.
8. **Adjusted Rand Index** – `adjusted_rand_score(true_labels, predicted_labels)`. ARI = 1 means perfect recovery; ARI ≈ 0 means random. This bridges supervised and unsupervised worlds.

---

### Pipeline 3 — Entity Matching

#### [NEW] [03_entity_matching.ipynb](file:///C:/Users/Asus/Desktop/2000/03_entity_matching.ipynb)

**Cells in order:**

1. **Load embeddings + metadata**.
2. **Why FAISS?** – Brute-force cosine similarity on 35K × 384 = 1.35 billion multiplications per query. FAISS builds an approximate index (IVF or flat L2 after L2-normalising vectors) and answers nearest-neighbor queries in milliseconds. We use `faiss.IndexFlatIP` (inner product = cosine similarity after normalisation) for exact results at this scale.
3. **Build FAISS index** – L2-normalise embeddings → `faiss.IndexFlatIP` → `index.add()`. Search with `index.search(embeddings, k=6)` (k=6 because the nearest neighbour is always itself).
4. **Three-tier logic** – after dropping self-pairs:
   - ≥ 0.95 → **Auto-match**: high confidence, no human review needed. Engineering reason: at this threshold false positives are extremely rare; manual review of millions of high-volume pairs is operationally infeasible.
   - 0.75–0.95 → **Flag for review**: probable match but uncertain; a human or a rule-based check can resolve. Engineering reason: binary thresholds lose precision at the boundary, so a review queue is the right middle ground.
   - < 0.75 → **Reject**: dissimilar enough to discard.
5. **Cross-merchant filter** – keep only pairs where `merchant_A ≠ merchant_B`. Same-merchant duplicates are an inventory problem, not entity matching.
6. **Sample display** – show 10 auto-matches and 10 flagged pairs in a `pd.DataFrame` with columns: title A, title B, merchant A, merchant B, cosine score.
7. **Manual precision evaluation** – sample 50 pairs uniformly from the auto-match tier; manually label each (match / not-match); compute `precision = matches / 50`. Explain *why full ground truth is hard*: there is no complete cross-merchant duplicate index in the dataset, so recall cannot be measured without exhaustive human annotation.

---

### Streamlit Dashboard

#### [NEW] [app.py](file:///C:/Users/Asus/Desktop/2000/app.py)

**Structure:**

```
app.py
├── @st.cache_resource: load_assets()
│   ├── embeddings.npy
│   ├── metadata.csv
│   ├── LR model (joblib)
│   ├── SentenceTransformer (for live inference)
│   ├── UMAP 2D coords
│   └── FAISS index
└── Pages (sidebar radio)
    ├── Classification
    ├── Clustering
    └── Entity Matching
```

**Why `@st.cache_resource`?** Streamlit reruns the entire script on every interaction. Without caching, the 384-dim encoder and FAISS index would reload on every keypress, making the app unusably slow. `cache_resource` pins them in memory for the session lifetime.

**Classification page** – `st.text_input` → encode with transformer → `lr.predict_proba` → display predicted category + bar chart of top-3 probabilities.

**Clustering page** – Plotly/Matplotlib scatter of UMAP 2D coords; `st.radio` toggles colour source between K-Means cluster ID and ground-truth category label.

**Entity Matching page** – `st.text_input` → encode → L2-normalise → `index.search(k=6)` → display top 5 cross-dataset neighbours (title, merchant, score) in a table.

---

## Verification Plan

### Automated (run inside each notebook)
| Notebook | Command | Expected result |
|---|---|---|
| `00_embeddings.ipynb` | Run all cells | `embeddings.npy` and `metadata.csv` created; 5 nearest neighbours are semantically sensible |
| `01_classification.ipynb` | Run all cells | Overall accuracy ≥ 85%; no class with F1 = 0 |
| `02_clustering.ipynb` | Run all cells | ARI > 0.3 (meaningful structure recovered); both UMAP plots render |
| `03_entity_matching.ipynb` | Run all cells | Auto-match sample precision ≥ 0.80 |

### Manual (Streamlit)
1. `cd C:\Users\Asus\Desktop\2000`
2. `streamlit run app.py`
3. **Classification**: type *"apple iphone 14 pro 256gb"* → expect `Mobile Phones` with high confidence.
4. **Clustering**: toggle between cluster/ground-truth views → both scatter plots render without error.
5. **Entity Matching**: type *"samsung galaxy s23"* → expect 5 product rows from different merchants.
