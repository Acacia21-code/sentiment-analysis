import streamlit as st
import joblib

model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon=":clapper:")

st.title("Movie Review Sentiment Analyzer")
st.write("Type a movie review below and the model will predict whether it's positive or negative.")

review = st.text_area("Enter a movie review:", height=150)

if st.button("Analyze Sentiment"):
    if review.strip() =="":
        st.warning("Please enter a review first.")
    else:
        review_tfidf = vectorizer.transform([review])
        prediction = model.predict(review_tfidf)[0]
        probability = model.predict_proba(review_tfidf)[0]

        if prediction ==1:
            confidence = probability[1] * 100
            st.success(f"Positive review({confidence:.1f}% confidence)")
        else:
            confidence = probability[0] * 100
            st.error(f"Negative review({confidence:.1f}% confidence)")

st.markdown("---")
st.caption("Model: Logistic Regression + TF-IDF, trained on the IMDB Large Movie Review Dataset (87.7% validation accuracy)")
