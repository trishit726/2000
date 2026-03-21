# 🛍️ PriceRunner NLP Portfolio

A production-ready Natural Language Processing and Machine Learning engineering dashboard. This project demonstrates classic NLP techniques combined with modern, embedding-based deep learning workflows, built on the [PriceRunner](https://www.kaggle.com/datasets) dataset containing over 35,000 product titles and categories.

## ⚙️ Architecture & Data Flow

This repository is structured around four core pipelines, seamlessly integrated into a sleek **Streamlit** dashboard.

### 1. Shared Embedding Layer
*   **Foundation:** Utilizes `all-MiniLM-L6-v2` via `sentence-transformers`.
*   **Process:** Converts raw product text into 384-dimensional dense semantic vectors representing contextual meaning. This shared layer powers all downstream tasks.

### 2. Classification Pipeline
*   **Goal:** Predict a product's precise category from its raw title.
*   **Implementation:** Support Vector Machine (SVC) and Logistic Regression models trained on the 384-D text embeddings. Maps vectors into one of 10 heavily imbalanced categories with calibrated confidence scores.

### 3. Clustering Pipeline
*   **Goal:** Unsupervised discovery of latent product groupings without using labels.
*   **Implementation:** K-Means clustering (K=10) combined with **UMAP** to project high-dimensional vectors down to 2D for visual inspection of feature space boundaries and class separation.

### 4. Entity Matching (Vector Search)
*   **Goal:** Real-time semantic similarity search to resolve input variations (e.g., "iphone 14 pro" vs "Apple iPhone 14 Pro Max") without relying on exact string matching.
*   **Implementation:** Millions of vector operations optimized via a **FAISS** index, utilizing Cosine Similarity (via L2-normalized Inner Product) for sub-millisecond retrieval across the entire 35K catalog.

## 📂 Repository Structure

*   `00_embeddings.ipynb`: Data preprocessing and base vector generation.
*   `01_classification.ipynb`: Model training, evaluation, and serialization for categorical prediction.
*   `02_clustering.ipynb`: Unsupervised grouping and UMAP projections.
*   `03_entity_matching.ipynb`: FAISS indexing and semantic search evaluation.
*   `app.py`: The interactive **Streamlit** frontend demonstrating the models in real-time.

## 🚀 Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/trishit726/2000.git
    cd 2000
    ```
2.  **Install dependencies:**
    Ensure you have Python 3.9+ installed and run:
    ```bash
    pip install pandas numpy scikit-learn sentence-transformers faiss-cpu matplotlib seaborn streamlit joblib
    ```
3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 🎨 UI/UX
The dashboard features a custom-built, premium light theme combining airy slate backgrounds with vibrant sky-blue accents to provide a sleek, data-centric user experience.
