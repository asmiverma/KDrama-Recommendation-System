import streamlit as st

from recommender import (
    build_feature_column,
    build_sentence_embeddings,
    build_tfidf_model,
    load_dataset,
    recommend_with_embeddings,
    recommend_with_tfidf,
)


st.set_page_config(page_title="K-Drama Recommender", page_icon="🎬", layout="wide")


@st.cache_data
def load_prepared_data():
    df = load_dataset("data/kdrama_list_cleaned.csv")
    return build_feature_column(df)


@st.cache_resource
def get_tfidf_assets(texts, ngram_min: int, ngram_max: int):
    return build_tfidf_model(texts, ngram_range=(ngram_min, ngram_max), max_features=15000)


@st.cache_resource
def get_embedding_assets(texts):
    return build_sentence_embeddings(texts, model_name="all-MiniLM-L6-v2")


def render_recommendations(results):
    for _, row in results.iterrows():
        with st.container(border=True):
            left, right = st.columns([3, 1])

            with left:
                year = int(row["Year"]) if row["Year"] == row["Year"] else "N/A"
                score = f"{row['Score']:.1f}" if row["Score"] == row["Score"] else "N/A"
                st.subheader(f"{row['Title']} ({year})")
                st.write(f"Score: {score}")
                st.write(f"Genre: {row['Genre'] if row['Genre'] else 'N/A'}")
                st.write(f"Similarity: {row['similarity']:.3f}")
                synopsis = row["Synopsis"] if row["Synopsis"] else "No synopsis available."
                st.write(synopsis[:360] + ("..." if len(synopsis) > 360 else ""))

            with right:
                img_url = row["img_URL"]
                if isinstance(img_url, str) and img_url.startswith("http"):
                    st.image(img_url.split("?")[0], use_container_width=True)
                else:
                    st.info("Poster not available")


def main():
    st.title("K-Drama Recommender System")
    st.caption("Portfolio edition: TF-IDF baseline + semantic embedding comparison")

    df = load_prepared_data()

    st.sidebar.header("Recommendation Settings")
    method = st.sidebar.radio("Model", ["TF-IDF (Baseline)", "Sentence Transformer"], index=0)
    top_k = st.sidebar.slider("Top K", min_value=5, max_value=20, value=10, step=1)

    all_genres = sorted(
        {
            g.strip()
            for item in df["Genre"].dropna().tolist()
            for g in str(item).split(",")
            if g.strip()
        }
    )
    selected_genres = st.sidebar.multiselect("Filter by Genre", all_genres)

    title_query = st.text_input("Search drama title", placeholder="Type part of a title...").strip().lower()

    filtered_df = df.copy()
    if selected_genres:
        pattern = "|".join(selected_genres)
        filtered_df = filtered_df[filtered_df["Genre"].str.contains(pattern, case=False, regex=True)]

    if title_query:
        filtered_df = filtered_df[filtered_df["Title"].str.lower().str.contains(title_query)]

    title_options = sorted(filtered_df["Title"].unique().tolist())

    if not title_options:
        st.warning("No titles found for the current filters. Try clearing search or genre filters.")
        return

    selected_title = st.selectbox("Select a drama", title_options)

    if st.button("Recommend Similar Dramas", type="primary"):
        with st.spinner("Generating recommendations..."):
            if method == "TF-IDF (Baseline)":
                _, tfidf_matrix = get_tfidf_assets(df["combined_text"], ngram_min=1, ngram_max=2)
                results = recommend_with_tfidf(selected_title, df, tfidf_matrix, top_k=top_k)
            else:
                try:
                    _, embeddings = get_embedding_assets(df["combined_text"].tolist())
                    results = recommend_with_embeddings(selected_title, df, embeddings, top_k=top_k)
                except ImportError:
                    st.error(
                        "SentenceTransformer dependencies are missing. Install requirements.txt and retry."
                    )
                    return

        st.success(f"Top {top_k} recommendations for: {selected_title}")
        render_recommendations(results)


if __name__ == "__main__":
    main()
