# K-Drama Recommender System
This project is a content-based recommender system for Korean dramas built as an interactive web application using Streamlit. Given the title of a K-Drama, the app suggests similar shows based on their plot summaries, genres, and thematic tags. The system relies on classical Natural Language Processing (NLP) techniques to analyze and compare textual descriptions of dramas, enabling users to discover new shows aligned with their interests.

## Overview
As the number of K-Dramas grows, it becomes increasingly difficult for users to find dramas aligned with their preferences. This project builds a text-based recommendation engine that suggests similar shows by analyzing their plots, genres, and themes.

The app demonstrates a simple yet effective Natural Language Processing (NLP) approach using TF-IDF (Term Frequency–Inverse Document Frequency) to represent textual data and compute similarities.

## Dataset
The dataset consists of information about 1,646 Korean dramas scraped from MyDramaList in April 2023. Each entry includes details such as title, year, score, synopsis, cast, network, rating, genre, tags, and an image URL for the drama's poster. This rich set of attributes provides a strong foundation for content-based recommendation by capturing both narrative and thematic features of each show.

## Data Cleaning
To ensure data consistency and usability, the following cleaning steps were performed:
* Renamed columns for clarity (e.g., Img URL to img_URL)
* Removed duplicate entries
* Handled missing values by filling them with empty strings
* Cleaned image URLs by stripping query parameters (?v=1)
* Combined relevant text fields (Synopsis, Genre, and Tags) for analysis

## NLP Approach: TF-IDF
The core of this recommender system is based on Term Frequency–Inverse Document Frequency (TF-IDF), a well-established NLP technique for converting textual data into numerical form. TF-IDF measures the importance of words in a document relative to their frequency across the entire corpus. By applying TF-IDF to the combined text fields of each drama, the system represents every show as a high-dimensional vector that reflects its narrative and thematic content.

This approach enables meaningful comparisons between dramas. By computing the cosine similarity between these TF-IDF vectors, the system can quantify how similar two dramas are in terms of their plot descriptions, genres, and thematic tags. Unlike simple keyword matching, TF-IDF reduces the weight of very common words and emphasizes more distinctive terms, improving the quality of recommendations.

## Recommendation Logic
The logic behind generating recommendations is as follows:

* User selects a drama title from a dropdown
* Its TF-IDF vector is retrieved from the matrix
* Cosine similarity is computed between the selected vector and all others
* The top 10 most similar dramas (excluding itself) are returned

## Streamlit Web App Features
The app is built using Streamlit, an open-source Python framework for building interactive web apps easily.

Features include:
* Clean interface with title and description
* Dropdown for drama title input
* Recommendations displayed with:
* Title and Year
* Score
* Genre
* Synopsis snippet
* Poster image loaded via URL

![image](https://github.com/user-attachments/assets/069c67db-e0fc-491a-bf3b-1f17d9da82dd)

## Requirements
* Python 3.8 or higher
* pandas
* scikit-learn
* streamlit
* numpy

## Future Work
While the current implementation uses TF-IDF for its simplicity and interpretability, there are many opportunities to extend and improve the system. Future directions include incorporating more advanced NLP models such as Sentence Transformers for semantic similarity, adding genre-based filtering, supporting multilingual recommendations, and integrating user ratings for hybrid recommendation strategies. Enhancing the UI/UX and deploying the app online for broader accessibility are also valuable goals for future development.

## Conclusion
This project demonstrates a practical, interpretable approach to content-based recommendation using classical NLP techniques. By leveraging TF-IDF to analyze narrative and thematic elements of K-Dramas, it provides users with personalized, relevant recommendations in an accessible web application. It serves as a strong foundation for exploring more advanced recommendation strategies in the future.

## Contact
This project is just for learing purposes, if there is any error, please contact me at:
Email: asmiasmiverma@gmail.com
