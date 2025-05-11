import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from pathlib import Path
import os

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

class EnhancedBullyingDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.spanish_stopwords = set(stopwords.words('spanish'))
        self.stemmer = SnowballStemmer('spanish')
    
    def preprocess_text(self, text):
        """Preprocess the text by removing special characters, stopwords, and stemming."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-záéíóúñ\s]', '', text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text, language='spanish')
        
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.spanish_stopwords]
        
        return ' '.join(tokens)
    
    def load_data(self, filepath):
        """Load and preprocess the training data."""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath, sep=";")
        
        # Preprocess text
        print("Preprocessing text data...")
        df['processed_text'] = df['texto'].apply(self.preprocess_text)
        
        return df
    
    def train(self, X, y):
        """Train the model with the given data."""
        print("Training the model...")
        
        # Vectorize the text data
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words=list(self.spanish_stopwords),
            min_df=2,
            max_df=0.9
        )
        
        X_vec = self.vectorizer.fit_transform(X)
        
        # Train the model
        self.model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=0.5,
            random_state=42,
            solver='liblinear'
        )
        
        self.model.fit(X_vec, y)
        
        # Print feature importance
        self._print_feature_importance()
    
    def _print_feature_importance(self, top_n=20):
        """Print the most important features for each class."""
        if not hasattr(self.model, 'coef_'):
            print("Model doesn't have coefficients to show feature importance.")
            return
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        print("\nTop features for bullying detection:")
        
        # For binary classification, we only have one set of coefficients
        if len(self.model.classes_) == 2:
            # Get the coefficients for the positive class (bullying)
            top_indices = np.argsort(self.model.coef_[0])[-top_n:][::-1]
            top_features = [(feature_names[j], self.model.coef_[0][j]) 
                          for j in top_indices]
            print("\nTop features indicating bullying (Class 1):")
            for feat, coef in top_features:
                print(f"  {feat}: {coef:.4f}")
                
            # For the negative class, we can show the most negative coefficients
            bottom_indices = np.argsort(self.model.coef_[0])[:top_n]
            bottom_features = [(feature_names[j], self.model.coef_[0][j]) 
                             for j in bottom_indices]
            print("\nTop features indicating non-bullying (Class 0):")
            for feat, coef in bottom_features:
                print(f"  {feat}: {coef:.4f}")
        else:
            # For multi-class case (if we add more classes later)
            for i, class_label in enumerate(self.model.classes_):
                top_indices = np.argsort(self.model.coef_[i])[-top_n:][::-1]
                top_features = [(feature_names[j], self.model.coef_[i][j]) 
                              for j in top_indices]
                print(f"\nClass {class_label}:")
                for feat, coef in top_features:
                    print(f"  {feat}: {coef:.4f}")
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data."""
        print("\nEvaluating the model...")
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_vec)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Bullying', 'Bullying']))
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        return accuracy
    
    def save_model(self, model_dir):
        """Save the model and vectorizer to disk."""
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'enhanced_bullying_model.joblib')
        vectorizer_path = os.path.join(model_dir, 'enhanced_vectorizer.joblib')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        print(f"\nModel saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")

def main():
    # Define paths
    data_dir = Path("/home/sublime-dev/dev/python/Almabot/modelo_pln/data/raw/")
    model_dir = Path("/home/sublime-dev/dev/python/Almabot/modelo_pln/data/models/")
    
    # Initialize the detector
    detector = EnhancedBullyingDetector()
    
    # Try to load enhanced data, fall back to original if not available
    data_path = data_dir / "enhanced_training_data.csv"
    if not data_path.exists():
        data_path = data_dir / "training_phrases.csv"
    
    # Load and preprocess data
    df = detector.load_data(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df['es_acoso'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['es_acoso']
    )
    
    # Train the model
    detector.train(X_train, y_train)
    
    # Evaluate the model
    detector.evaluate(X_test, y_test)
    
    # Save the model
    detector.save_model(model_dir)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
