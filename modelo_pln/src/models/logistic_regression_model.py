import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
import os
import json

class BullyingDetectionModel:
    def __init__(self):
        # Descargar stopwords si no están disponibles
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Obtener stopwords en español
        spanish_stopwords = set(stopwords.words('spanish'))
        
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            stop_words=list(spanish_stopwords),
            min_df=2,
            max_df=0.95
        )
        self.model = LogisticRegression(
            class_weight='balanced',
            max_iter=2000,
            C=0.1,
            random_state=42
        )
        
    def preprocess_data(self, texts):
        """Preprocesa los textos para el modelo"""
        return self.vectorizer.fit_transform(texts)
    
    def train(self, X_train, y_train):
        """Entrena el modelo"""
        X_vectors = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_vectors, y_train)
        
    def predict(self, text, threshold=0.5):
        """Realiza predicciones sobre un texto con umbral ajustable"""
        X_vector = self.vectorizer.transform([text])
        probability = self.model.predict_proba(X_vector)[0]
        prediction = 1 if probability[1] > threshold else 0
        return prediction, probability
    
    def evaluate(self, X_test, y_test):
        """Evalúa el modelo"""
        X_vectors = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_vectors)
        return classification_report(y_test, y_pred)
    
    def save_model(self, path='/app/data/models/'):
        """Guarda el modelo y el vectorizador"""
        try:
            # Asegurarse de que el directorio existe
            os.makedirs(path, exist_ok=True)
            
            # Guardar el modelo
            model_path = os.path.join(path, 'bullying_detection_model.joblib')
            joblib.dump(self.model, model_path)
            print(f"Modelo guardado en: {model_path}")
            
            # Guardar el vectorizador
            vectorizer_path = os.path.join(path, 'vectorizer.joblib')
            joblib.dump(self.vectorizer, vectorizer_path)
            print(f"Vectorizador guardado en: {vectorizer_path}")
            
            # Guardar metadatos del modelo
            metadata = {
                'model_type': 'LogisticRegression',
                'vectorizer_type': 'TfidfVectorizer',
                'features': self.vectorizer.get_feature_names_out().tolist(),
                'n_features': len(self.vectorizer.get_feature_names_out()),
                'classes': self.model.classes_.tolist(),
                'n_classes': len(self.model.classes_)
            }
            
            metadata_path = os.path.join(path, 'model_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"Metadatos guardados en: {metadata_path}")
            
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")
            raise

    def load_model(self, path='/app/data/models/'):
        """Carga el modelo y el vectorizador"""
        try:
            # Cargar el modelo
            model_path = os.path.join(path, 'bullying_detection_model.joblib')
            self.model = joblib.load(model_path)
            print(f"Modelo cargado desde: {model_path}")
            
            # Cargar el vectorizador
            vectorizer_path = os.path.join(path, 'vectorizer.joblib')
            self.vectorizer = joblib.load(vectorizer_path)
            print(f"Vectorizador cargado desde: {vectorizer_path}")
            
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            raise 

    def analyze_text(self, text):
        """Analiza un texto y proporciona un análisis detallado"""
        prediction, probability = self.predict(text)
        confidence = max(probability)
        
        # Obtener las palabras más importantes
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.model.coef_[0]
        word_importance = list(zip(feature_names, coefficients))
        word_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'prediction': prediction,
            'probability': probability.tolist(),
            'confidence': float(confidence),
            'risk_level': 'alto' if probability[1] > 0.8 else 'medio' if probability[1] > 0.5 else 'bajo',
            'key_terms': [word for word, _ in word_importance[:5]]
        } 