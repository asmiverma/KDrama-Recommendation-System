# K-Drama Recommender System

Portfolio-ready content-based recommendation project for Korean dramas, built with a complete data-science workflow: EDA, feature engineering, model comparison, evaluation, and deployment in Streamlit.

## Project Overview

This project recommends similar K-Dramas based on story and metadata.

Input:

- A selected drama title

Output:

- Top-K similar dramas with similarity scores

Models included:

- Baseline: TF-IDF + cosine similarity
- Improved: Sentence Transformer embeddings + cosine similarity

## Dataset

- Source: MyDramaList
- Size: 1,646 Korean dramas
- Core fields: Title, Year, Score, Synopsis, Genre, Tags, Poster URL

## Data Preprocessing

Pipeline steps:

- Normalize column names and handle missing values
- Remove duplicate titles
- Clean text (lowercase, punctuation cleanup, spacing normalization)
- Combine Synopsis + Genre + Tags into one feature column
- Apply stopword removal through vectorizer configuration
- Experiment with n-grams (unigrams + bigrams)

## Recommendation Method

### 1) Baseline Model

- Vectorization: TF-IDF
- Similarity metric: Cosine similarity
- Strength: Fast, interpretable, lightweight

### 2) Semantic Model

- Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
- Similarity metric: Cosine similarity
- Strength: Better semantic understanding of plot descriptions

### Why compare both?

- TF-IDF is efficient and easy to explain
- Sentence embeddings usually capture meaning better
- Side-by-side evaluation shows practical model trade-offs

## Model Evaluation

Because explicit user feedback is unavailable, the project uses a proxy relevance setup (shared genre/tags) for offline ranking evaluation.

Metrics reported:

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)

This makes model selection more objective and recruiter-friendly.

## Visualization

The notebooks include:

- Genre distribution
- Rating distribution
- Most common tags
- Network and year trends
- Similarity score distribution
- Drama clusters with PCA or t-SNE
- Sample recommendation case studies

## Application Features

The Streamlit app supports:

- Title search
- Genre filtering
- Top-K control
- Model switch (TF-IDF vs Sentence Transformer)
- Poster display
- Similarity score display

## Tech Stack

| Area          | Tools                               |
| ------------- | ----------------------------------- |
| Language      | Python                              |
| Data          | Pandas, NumPy                       |
| Visualization | Matplotlib, Seaborn, Plotly         |
| NLP / ML      | scikit-learn, sentence-transformers |
| App Layer     | Streamlit                           |
| Data Source   | MyDramaList                         |

## Project Structure

```
KDrama-Recommendation-System/
|- app.py
|- recommender.py
|- requirements.txt
|- README.md
|- data/
|  |- kdrama_list_cleaned.csv
|  |- kdrama_list_original.csv
|- models/
|- notebooks/
   |- 01_data_cleaning.ipynb
   |- 02_eda.ipynb
   |- 03_feature_engineering.ipynb
   |- 04_model_experiments.ipynb
```

## How to Run the Project

1. Clone the repository and open it.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the web app:

```bash
streamlit run app.py
```

4. Open the local URL shown in the terminal.

## Future Improvements

- Add approximate nearest neighbor search for faster retrieval
- Add hybrid recommendation with collaborative signals
- Track online metrics from user interactions
- Build API endpoints for production deployment
- Add CI checks for reproducibility and testing
