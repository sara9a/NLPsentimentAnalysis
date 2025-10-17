# NLPsentimentAnalysis

Short description
- A small Python project for text-based sentiment analysis. The main script (SentimentAnalysis.py) reads raw data, cleans and preprocesses text (stemming or lemmatization), converts text to TF‑IDF vectors, balances classes with SMOTE, and trains/evaluates a classification model (Naive Bayes or SVC).

Features
- Data cleaning: lowercase conversion, punctuation removal, stopword removal (with exceptions for sentiment-significant words), removal of non-alphanumeric characters.
- Text processing: choice of Porter stemming or WordNet lemmatization with POS tagging.
- Feature extraction: TF‑IDF vectorization.
- Class imbalance handling: SMOTE applied to the training split.
- Models: Multinomial Naive Bayes (alpha=0.5) and Support Vector Classifier (linear kernel).
- Simple console interaction to select preprocessing and model.

Repository files
- SentimentAnalysis.py — Main script with classes for DataCleaning, Stemming/Lemmatization and ModelTrainer.
- sampled_data.csv — Expected input file (not included here).

Requirements
- Python 3.8+
- Python packages:
  - pandas
  - scikit-learn
  - nltk
  - imbalanced-learn
  - joblib (optional, for model persistence)
- NLTK resources (download once):
  - punkt, stopwords, wordnet, averaged_perceptron_tagger
  Example:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('wordnet')
  nltk.download('averaged_perceptron_tagger')
  ```

Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sara9a/NLPsentimentAnalysis.git
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   # Linux/macOS
   source venv/bin/activate
   # Windows
   venv\Scripts\activate

   # If a requirements.txt is provided:
   pip install -r requirements.txt

   # Or install manually:
   pip install pandas scikit-learn nltk imbalanced-learn
   ```

Usage
- Run the script:
  ```bash
  python SentimentAnalysis.py
  ```
- Interactive choices:
  - Preprocessing: 1 = Stemming, 2 = Lemmatization
  - Model: 1 = Naive Bayes, 2 = SVC
- The script expects `sampled_data.csv` in the repository root and uses columns `Id`, `Score`, `Text` (first 10,000 rows by default).
- Output: console logs showing class distribution before/after SMOTE, Accuracy and a classification report.

Important notes
- SMOTE and sparse matrices:
  Some versions of imbalanced-learn cannot resample scipy.sparse matrices directly. If you encounter an error, convert the training matrix before SMOTE with `.toarray()` or use RandomOverSampler as an alternative.
- Labels:
  The script expects a `Score` column. Depending on the score scale (e.g., 1–5), it may make sense to convert labels to binary classes (positive/negative) before training.
- Performance:
  TF‑IDF and dense conversions can use a lot of memory. For larger datasets consider batch processing, HashingVectorizer, or incremental training.

Suggested improvements
- Add a CLI (argparse) for file path, sample size, preprocessing and model options.
- Persist the TF‑IDF vectorizer and trained model (joblib.dump/load).
- Use cross-validation and GridSearchCV for hyperparameter tuning.
- Improve negation handling and custom stopword lists.
- Convert the workflow into an sklearn Pipeline for cleaner step chaining.
- Add unit tests and logging for maintainability.

Example quick workflow
1. Ensure NLTK resources are installed.
2. Place `sampled_data.csv` in the repository root.
3. Run `python SentimentAnalysis.py`, select lemmatization and Naive Bayes.
4. Check the console output for class distribution and evaluation metrics.

Contributing & contact
- Project: sara9a/NLPsentimentAnalysis
- Author / Contact: sara9a (GitHub), DewanschJ (GitHub)
- For questions or suggestions open an issue or submit a pull request.
