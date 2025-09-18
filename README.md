# Sentiment Analysis on Movie Reviews

## Project Overview

This project builds a binary sentiment classifier to predict whether a movie review is **positive** or **negative** using the IMDb Large Movie Review Dataset. It demonstrates a complete machine learning pipeline including:

- Automatic dataset download and extraction
- Data loading and preprocessing
- Text vectorization using TF-IDF with unigrams and bigrams
- Training a Logistic Regression classifier
- Model evaluation with accuracy, classification report, and confusion matrix visualization

This project is designed to be easy to run, well-structured, and suitable for showcasing practical machine learning skills on text data.

---

## Dataset

The project uses the [IMDb Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), which contains 50,000 movie reviews labeled as positive or negative. The dataset is automatically downloaded and extracted by the code.

---

## Installation

Make sure you have Python 3 installed. Then install the required Python packages:

```bash
pip install numpy pandas scikit-learn matplotlib requests tqdm