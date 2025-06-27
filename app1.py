import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ----------------------------------------
# 1️⃣ Load and Clean Data
# ----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('kdrama_list_cleaned.csv')
    # Drop any rows missing critical fields
    df = df.dropna(subset=['Title', 'Synopsis', 'Genre', 'Tags', 'img URL'])
    # Create a combined text field for NLP
    df['Combined'] = df['Synopsis'].fillna('') + ' ' + df['Genre'].fillna('') + ' ' + df['Tags'].fillna('')
    return df

df = load_data()

# ----------------------------------------
# 2️⃣ TF-IDF Vectorization
# ----------------------------------------
@st.cache_resource
def vectorize_text(text_data):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(text_data)
    return vectorizer, vectors

vectorizer, vectors = vectorize_text(df['Combined'])

# ----------------------------------------
# 3️⃣ Recommendation Function
# ----------------------------------------
def get_recommendations(selected_title, df, vectors, top_n=10):
    selected_title = selected_title.lower().strip()

    matches = df[df['Title'].str.lower() == selected_title]
    if matches.empty:
        return None

    idx = matches.index[0]
    cosine_similarities = linear_kernel(vectors[idx:idx+1], vectors).flatten()
    similar_indices = cosine_similarities.argsort()[-top_n-1:-1][::-1]

    recommendations = df.iloc[similar_indices][['Title', 'Year', 'Genre', 'Score', 'Synopsis', 'img URL']]
    recommendations['Similarity'] = cosine_similarities[similar_indices]
    return recommendations

# ----------------------------------------
# 4️⃣ Streamlit UI
# ----------------------------------------
st.set_page_config(page_title="K-Drama Recommender", page_icon="📺", layout="centered")
st.title("📺 K-Drama Recommender App")

st.write("""
Welcome!  
Enter the name of a K-Drama you like, and we'll recommend 10 similar shows based on their **synopsis, genre, and tags**.
""")

# Input box
input_title = st.text_input("🎯 Enter K-Drama Title:")

# Button
if st.button("🔍 Recommend"):
    if not input_title.strip():
        st.warning("⚠️ Please enter a K-Drama title first.")
    else:
        # ✅ 1. Get recommendations
        results = get_recommendations(input_title, df, vectors, top_n=10)

        if results is None:
            st.error("❌ Sorry! Title not found in the database. Please check spelling or try another drama.")
        else:
            st.success(f"✅ Top 10 recommendations similar to **{input_title}**:")

            # ✅ 2. Display recommendations in a loop
            for i, row in results.iterrows():
                with st.container():
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.subheader(f"{row['Title']} ({row['Year']})")
                        st.write(f"⭐ **Rating:** {row['Score']}")
                        st.write(f"🎭 **Genre:** {row['Genre'] if pd.notnull(row['Genre']) else 'N/A'}")
                        st.write(row['Synopsis'][:400] + "...")

                    with col2:
                        if pd.notnull(row['img URL']) and row['img URL'].startswith('http'):
                            try:
                                st.image(row['img URL'], use_container_width=True, caption=row['Title'])
                            except Exception:
                                st.warning("⚠️ Couldn't load poster image.")
                        else:
                            st.warning("⚠️ No poster image available.")

                    st.markdown("---")

