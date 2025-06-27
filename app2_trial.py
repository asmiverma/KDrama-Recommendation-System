import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="KDrama Recommender - SBERT+", layout="wide")

# ---------------------------------------
# 1. Load Data
# ---------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('kdrama_list_cleaned.csv')
    df = df.fillna('')
    return df

df = load_data()

# ---------------------------------------
# 2. Format Text for SBERT
# ---------------------------------------
def format_weighted_input(row):
    synopsis = row['Synopsis']
    genre = row['Genre']
    tags = row['Tags']
    
    # Weight synopsis 3x to emphasize story
    return f"Synopsis: {synopsis}. {synopsis}. {synopsis}. Genre: {genre}. Tags: {tags}."

df['Formatted'] = df.apply(format_weighted_input, axis=1)

# ---------------------------------------
# 3. Load Better SBERT Model
# ---------------------------------------
@st.cache_resource
def load_model():
    model = SentenceTransformer('all-mpnet-base-v2')
    return model

model = load_model()

# ---------------------------------------
# 4. Precompute Embeddings
# ---------------------------------------
@st.cache_data
def compute_embeddings(texts):
    return model.encode(texts, show_progress_bar=True)

all_embeddings = compute_embeddings(df['Formatted'].tolist())

# ---------------------------------------
# 5. Recommendation Function
# ---------------------------------------
def get_recommendations(input_title, top_n=10):
    input_row = df[df['Title'].str.lower() == input_title.lower()]

    if input_row.empty:
        st.error("Drama title not found in the database. Please check spelling.")
        return None

    idx = input_row.index[0]
    query_embedding = all_embeddings[idx].reshape(1, -1)

    similarities = cosine_similarity(query_embedding, all_embeddings).flatten()
    similarities[idx] = -1  # Exclude itself

    top_indices = similarities.argsort()[-top_n:][::-1]
    results = df.iloc[top_indices][['Title', 'Year', 'Score', 'Genre', 'Synopsis', 'img URL']].copy()
    results['Similarity'] = similarities[top_indices]
    return results

# ---------------------------------------
# 6. Streamlit UI
# ---------------------------------------
st.title("🌸 KDrama Recommender (Enhanced SBERT Model)")
st.write("""
This app uses an advanced SentenceTransformer model (**all-mpnet-base-v2**) to recommend 10 similar K-Dramas
based on synopsis, genre, and tags. The synopsis is weighted more heavily for better plot matching.
""")

# Dropdown of all titles
selected_title = st.selectbox("Choose a K-Drama:", sorted(df['Title'].unique()))

if st.button("Recommend"):
    with st.spinner("Finding recommendations..."):
        results = get_recommendations(selected_title)
        if results is not None:
            st.subheader(f"Recommendations similar to **{selected_title}**")

            for _, row in results.iterrows():
                with st.container():
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.subheader(f"{row['Title']} ({row['Year']})")
                        st.write(f"⭐ **Score:** {row['Score']}")
                        st.write(f"🎭 **Genre:** {row['Genre']}")
                        st.write(f"🧭 **Similarity Score:** {row['Similarity']:.2f}")
                        st.write((row['Synopsis'][:400] + "...") if pd.notnull(row['Synopsis']) else "No synopsis available.")

                    with col2:
                        if pd.notnull(row['img URL']) and row['img URL'].startswith('http'):
                            st.image(row['img URL'].split('?')[0], use_container_width=True, caption=row['Title'])
                        else:
                            st.warning("⚠️ No poster image available.")

                    st.markdown("---")
