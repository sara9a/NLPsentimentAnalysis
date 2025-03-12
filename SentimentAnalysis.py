import string
import re
import pandas as pd
import nltk
from imblearn.over_sampling import SMOTE
from nltk import WordNetLemmatizer, pos_tag
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from collections import Counter



# Data Cleaning Class
class DataCleaning:
    def __init__(self, filepath="sampled_data.csv"):
        self.df = pd.read_csv(filepath)
        self.df = self.df[['Id', 'Score', 'Text']].head(10000).dropna(subset=['Text'])

    def convert_lowercase(self):
        self.df['Clean Text'] = self.df['Text'].str.lower()

    def remove_punctuation(self):
        self.df['Clean Text'] = self.df['Clean Text'].apply(lambda text: text.translate(str.maketrans('', '', string.punctuation)))

    def remove_stopwords(self):
        default_stopwords = set(stopwords.words('english'))
        sentiment_words = {
            "not", "no", "but", "yet", "cannot", "won't", "shouldn't", "couldn't",
            "don't", "doesn't", "didn't", "hadn't", "hasn't", "haven't",
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
            "she", "her", "hers", "herself", "they", "them", "their", "theirs", "themselves",
            "what", "which", "who", "whom", "this", "that", "these", "those",
            "do", "does", "did", "doing", "can", "will", "just", "should", "now",
            "own", "only", "so", "too", "very", "because", "against", "between",
            "again", "further", "once", "then", "nor", "than", "here", "there",
            "when", "where", "why", "how"
        }
        custom_stopwords = default_stopwords - sentiment_words
        self.df['Clean Text'] = self.df['Clean Text'].apply(lambda text: " ".join([word for word in text.split() if word not in custom_stopwords]))

    def remove_special_characters(self):
        self.df['Clean Text'] = self.df['Clean Text'].apply(lambda text: re.sub(r'[^a-zA-Z0-9\s]', '', text))

    def preprocess(self):

        self.convert_lowercase()
        self.remove_punctuation()
        self.remove_stopwords()
        self.remove_special_characters()
        return self.df

# Stemming Class
class StemmingProcessor:
    def __init__(self, df):
        self.df = df
        self.ps = PorterStemmer()

    def apply_stemming(self):
        self.df['Processed Text'] = self.df['Clean Text'].apply(lambda text: " ".join([self.ps.stem(word) for word in text.split()]))
        return self.df

# Lemmatization Class
class LemmatizationProcessor:
    def __init__(self, df):
        self.df = df
        self.lemmatizer = WordNetLemmatizer()
        self.wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

    def apply_lemmatization(self):
        self.df['Processed Text'] = self.df['Clean Text'].apply(lambda text: " ".join(
            [self.lemmatizer.lemmatize(word, self.wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tag(text.split())]
        ))
        return self.df
class ModelTrainer:
    def __init__(self, df, model_type):
        self.df = df
        self.model_type = model_type

    def train_and_evaluate(self):
        # Features and target
        X = self.df['Processed Text']
        y = self.df['Score']

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        X_vectorized = vectorizer.fit_transform(X)

        # Splitting data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

        # Applying SMOTE for balancing classes
        print("Before SMOTE:", Counter(y_train))
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print("After SMOTE:", Counter(y_train_resampled))

        # Training Naive Bayes or SVC
        if self.model_type == "naive_bayes":
            print("\nTraining Naive Bayes Classifier...")
            model = MultinomialNB(alpha=0.5)
        elif self.model_type == "svc":
            print("\nTraining SVC (Support Vector Classifier)...")
            model = SVC(kernel='linear', C=1, random_state=42)
        else:
            raise ValueError("Invalid model type selected.")

        model.fit(X_train_resampled, y_train_resampled)

        # Predictions
        y_pred = model.predict(X_test)

        # Evaluation
        print("\nModel Evaluation:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))


# Main Execution
if __name__ == "__main__":
    # Preprocess Data
    cleaner = DataCleaning()
    cleaned_df = cleaner.preprocess()

    # User Input for Preprocessing Type
    print("Choose text processing method:")
    print("1. Stemming")
    print("2. Lemmatization")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        print("\nApplying Stemming...")
        processor = StemmingProcessor(cleaned_df)
        final_df = processor.apply_stemming()
    elif choice == "2":
        print("\nApplying Lemmatization...")
        processor = LemmatizationProcessor(cleaned_df)
        final_df = processor.apply_lemmatization()
    else:
        print("\nInvalid choice! Defaulting to Lemmatization.")
        processor = LemmatizationProcessor(cleaned_df)
        final_df = processor.apply_lemmatization()

    # User Input for Model Selection
    print("\nChoose model:")
    print("1. Naive Bayes")
    print("2. SVC (Support Vector Classifier)")
    model_choice = input("Enter your choice (1 or 2): ").strip()

    if model_choice == "1":
        trainer = ModelTrainer(final_df, "naive_bayes")
    elif model_choice == "2":
        trainer = ModelTrainer(final_df, "svc")
    else:
        print("\nInvalid choice! Defaulting to Naive Bayes.")
        trainer = ModelTrainer(final_df, "naive_bayes")

    # Train and Evaluate Model
    trainer.train_and_evaluate()
