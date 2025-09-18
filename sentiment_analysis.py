import os
import tarfile
import requests
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Download and extract dataset
def download_and_extract_imdb(data_url, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(extract_path, filename)
    if not os.path.exists(filepath):
        print("Downloading dataset...")
        response = requests.get(data_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
    else:
        print("Dataset archive already downloaded.")
    # Extract
    extracted_folder = os.path.join(extract_path, 'aclImdb')
    if not os.path.exists(extracted_folder):
        print("Extracting dataset...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(path=extract_path)
    else:
        print("Dataset already extracted.")
    return extracted_folder

# Step 2: Load data from extracted folders
def load_imdb_data(data_dir):
    texts, labels = [], []
    for label_type in ['pos', 'neg']:
        folder_path = os.path.join(data_dir, label_type)
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                with open(os.path.join(folder_path, filename), encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(1 if label_type == 'pos' else 0)
    return texts, labels

# Step 3: Plot confusion matrix
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def main():
    DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    DATA_PATH = "./data"

    # Download and extract dataset
    imdb_folder = download_and_extract_imdb(DATA_URL, DATA_PATH)

    # Load training data
    train_dir = os.path.join(imdb_folder, 'train')
    print("Loading training data...")
    texts, labels = load_imdb_data(train_dir)
    print(f"Loaded {len(texts)} training samples.")

    # Split into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels)

    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_val_tfidf = vectorizer.transform(x_val)

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(x_train_tfidf, y_train)

    # Predict and evaluate
    y_pred = model.predict(x_val_tfidf)
    print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=['Negative', 'Positive']))

    # Plot confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plot_confusion_matrix(cm, classes=['Negative', 'Positive'])

if __name__ == "__main__":
    main()
