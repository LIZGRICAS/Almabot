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
        
    def predict(self, text, threshold=0.3):
        """
        Realiza predicciones sobre un texto con umbral ajustable.
        
        Args:
            text (str): Texto a analizar
            threshold (float): Umbral de confianza para la predicción (0-1)
            
        Returns:
            tuple: (predicción, probabilidades)
            - predicción (int): 1 si es bullying, 0 si no lo es
            - probabilidades (list): [prob_no_bullying, prob_bullying]
        """
        try:
            text_lower = text.lower()
            
            # Lista de indicadores fuertes de bullying (en minúsculas)
            strong_indicators = [
                # Amenazas
                'amenaza', 'amenazas', 'amenazando', 'amenazado', 'amenazada', 'amenazan', 'amenazas',
                # Acoso
                'acoso', 'acosar', 'acosado', 'acosada', 'acosando', 'acosan', 'acosador', 'acosadora', 'acosadores', 'acosadoras',
                # Violencia física
                'golpear', 'golpeado', 'golpeada', 'golpeando', 'golpearon',
                'pegar', 'pegando', 'pegó', 'pegaron', 'golpes',
                'peleando', 'pelea', 'pelearon', 'pelear',
                # Insultos
                'insultar', 'insultado', 'insultada', 'insultando', 'insultos', 'insultan', 'insulto',
                # Intimidación
                'intimidar', 'intimidado', 'intimidada', 'intimidación', 'intimidan', 'intimidación',
                # Hostigamiento
                'hostigar', 'hostigamiento', 'hostigando', 'hostigado', 'hostigada', 'hostigamiento',
                # Ciberacoso
                'ciberacoso', 'ciberbullying', 'ciberacoso', 'ciberbully', 'ciberacoso',
                # Humillación
                'humillar', 'humillación', 'humillando', 'humillado', 'humillada', 'humillación',
                # Agresión
                'agredir', 'agresión', 'agredido', 'agredida', 'agrediendo', 'agresor', 'agresora', 'agresiones',
                # Maltrato
                'maltratar', 'maltrato', 'maltratado', 'maltratada', 'maltratando', 'maltratan', 'maltratador', 'maltratadora',
                # Abuso
                'abusar', 'abuso', 'abusando', 'abusado', 'abusada', 'abusador', 'abusadora',
                # Matoneo
                'matoneo', 'matoneando', 'matonaje', 'matón', 'matona',
                # Violencia
                'violencia', 'violento', 'violenta', 'violentado', 'violentada',
                # Persecución
                'acorralar', 'acorralando', 'acorralado', 'acorralada',
                'perseguir', 'persiguiendo', 'perseguido', 'perseguida', 'persecución',
                # Otros términos relacionados
                'chantaje', 'chantajear', 'chantajeando', 'chantajeado',
                'amenazante', 'amenazador', 'amenazadora',
                'acosamiento', 'acoso escolar', 'bullying', 'bulling',
                'vejación', 'vejar', 'vejado', 'vejada',
                'ofender', 'ofendido', 'ofendida', 'ofensa',
                'denigrar', 'denigrado', 'denigrada',
                'difamar', 'difamación', 'difamado', 'difamada',
                'calumniar', 'calumnia', 'calumniado', 'calumniada',
                'molestar', 'molestando', 'molestado', 'molestada', 'molestias',
                'burlar', 'burlas', 'burlándose', 'burlado', 'burlada',
                'ofender', 'ofendiendo', 'ofendido', 'ofendida', 'ofensa',
                'discriminar', 'discriminación', 'discriminado', 'discriminada',
                'excluir', 'excluyendo', 'excluido', 'excluida', 'exclusión',
                # Términos comunes mal escritos
                'amenza', 'amenaz', 'amenasas', 'amenasando',
                'asoso', 'acoso', 'acosso', 'acoso',
                'insulto', 'insult', 'insulto', 'insulta',
                'intimidar', 'intimidar', 'intimidar', 'intimidar',
                'bullyng', 'bulling', 'bully', 'bulli',
                'maltrato', 'maltrato', 'maltrato', 'maltrato'
            ]
            
            # Crear una expresión regular para buscar palabras completas
            import re
            pattern = r'\b(' + '|'.join(map(re.escape, strong_indicators)) + r')\b'
            
            # Si hay indicadores fuertes, clasificar como bullying con alta confianza
            if re.search(pattern, text_lower):
                return 1, [0.05, 0.95]  # 95% de confianza para indicadores fuertes
            
            # Si no hay vectorizador o modelo cargado, usar solo las palabras clave
            if not hasattr(self, 'vectorizer') or not hasattr(self, 'model'):
                medium_indicators = [
                    'molestar', 'molestan', 'molestando', 'molestado', 'molestada',
                    'burlar', 'burlan', 'burlándose', 'burlado', 'burlada',
                    'ofender', 'ofendido', 'ofendida', 'ofendiendo',
                    'humillar', 'humillado', 'humillada', 'humillación',
                    'excluir', 'excluido', 'excluida', 'exclusión'
                ]
                if any(indicator in text_lower for indicator in medium_indicators):
                    return 1, [0.2, 0.8]  # 80% de confianza para indicadores medios
                return 0, [0.9, 0.1]  # 90% de confianza en que no es bullying
            
            # Si hay vectorizador y modelo, usarlos para la predicción
            X_vector = self.vectorizer.transform([text])
            
            # Obtener probabilidades
            probability = self.model.predict_proba(X_vector)[0]
            
            # Obtener la predicción basada en el umbral
            prediction = 1 if probability[1] >= threshold else 0
            
            # Si hay indicadores medios, ajustar la confianza
            medium_indicators = [
                'molestar', 'molestan', 'molestando', 'molestado', 'molestada',
                'burlar', 'burlan', 'burlándose', 'burlado', 'burlada',
                'ofender', 'ofendido', 'ofendida', 'ofendiendo',
                'humillar', 'humillado', 'humillada', 'humillación',
                'excluir', 'excluido', 'excluida', 'exclusión',
                'ignorar', 'ignorado', 'ignorada', 'ignorando',
                'hablar mal', 'hablan mal', 'habló mal', 'difamar',
                'chisme', 'chismes', 'chismear', 'rechazar', 'rechazado'
            ]
            
            # Si hay indicadores medios, aumentar la confianza en la predicción
            if any(indicator in text_lower for indicator in medium_indicators):
                if prediction == 1:
                    probability = [0.1, 0.9]  # Aumentar confianza en bullying
                else:
                    probability = [0.8, 0.2]  # Aún así, darle un poco de probabilidad
            
            # Si el mensaje parece pedir ayuda, ser más sensible
            if any(word in text_lower for word in ['ayuda', 'miedo', 'triste', 'solo', 'sola']) and prediction == 0:
                if probability[1] > 0.3:  # Si hay alguna indicación de bullying
                    prediction = 1
                    probability = [0.3, 0.7]  # Ajustar confianza
            
            # Ensure probability is a list before returning
            if hasattr(probability, 'tolist'):
                probability = probability.tolist()
            return prediction, probability
            
        except Exception as e:
            print(f"Error en la predicción: {str(e)}")
            # En caso de error, asumir que no es bullying pero con baja confianza
            return 0, [0.6, 0.4]
    
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