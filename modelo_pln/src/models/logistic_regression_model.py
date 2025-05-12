import numpy as np
import pandas as pd
import os
import json
import re
import emoji
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    cross_validate,
    StratifiedKFold
)
from sklearn.metrics import (
    classification_report, 
    f1_score, 
    precision_score, 
    recall_score,
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.exceptions import NotFittedError
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from collections import Counter

class BullyingDetectionModel:
    def __init__(self):
        # Descargar recursos de NLTK si no están disponibles
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        # Configuración de preprocesamiento
        self.spanish_stopwords = set(stopwords.words('spanish'))
        self.stemmer = SnowballStemmer('spanish')
        
        # Características de texto
        self.text_features = FeatureUnion([
            ('tfidf_word', TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                stop_words=list(self.spanish_stopwords),
                min_df=3,
                max_df=0.9,
                analyzer='word'
            )),
            ('tfidf_char', TfidfVectorizer(
                ngram_range=(3, 5),
                analyzer='char_wb',
                min_df=5,
                max_df=0.9
            ))
        ])
        
        # Modelo final con Stacking
        base_models = [
            ('lr', LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                C=0.8,
                random_state=42,
                solver='liblinear',
                penalty='l1'
            )),
            ('svm', SVC(
                class_weight='balanced',
                probability=True,
                kernel='linear',
                C=0.5,
                random_state=42
            ))
        ]
        
        self.model = StackingClassifier(
            estimators=base_models,
            final_estimator=RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42
            ),
            stack_method='predict_proba',
            n_jobs=-1
        )
        
    def extract_linguistic_features(self, text):
        """Extrae características lingüísticas del texto"""
        features = {}
        
        # Características básicas
        features['length'] = len(text)
        words = word_tokenize(text, language='spanish')
        features['word_count'] = len(words)
        
        # Características léxicas
        if words:
            # Longitud promedio de palabras
            features['avg_word_length'] = sum(len(word) for word in words) / len(words)
            
            # Diversidad léxica (ratio de palabras únicas)
            features['lexical_diversity'] = len(set(words)) / len(words) if words else 0
            
            # Frecuencia de signos de puntuación
            features['exclamation_count'] = text.count('!') + text.count('¡')
            features['question_count'] = text.count('?') + text.count('¿')
            features['period_count'] = text.count('.')
            
            # Uso de mayúsculas
            features['upper_ratio'] = sum(1 for word in words if word.isupper()) / len(words)
            
            # Emociones (palabras clave)
            emotion_words = ['triste', 'miedo', 'asustado', 'solo', 'sola', 'ayuda', 
                           'odio', 'dolor', 'llorar', 'sufrir', 'sufro', 'sufres', 'sufre']
            features['emotion_word_count'] = sum(1 for word in words if word in emotion_words)
            
            # Intensificadores
            intensifiers = ['muy', 'mucho', 'muchísimo', 'demasiado', 'realmente', 'totalmente']
            features['intensifier_count'] = sum(1 for word in words if word in intensifiers)
            
            # Negaciones
            negations = ['no', 'nunca', 'jamás', 'tampoco', 'nadie', 'nada', 'ninguno']
            features['negation_count'] = sum(1 for word in words if word in negations)
            
            # Pronombres en primera persona
            first_person = ['yo', 'me', 'mi', 'mí', 'conmigo', 'mío', 'mía', 'míos', 'mías']
            features['first_person_count'] = sum(1 for word in words if word in first_person)
            
            # Palabras de incertidumbre
            uncertainty = ['quizás', 'tal vez', 'a lo mejor', 'puede que', 'posiblemente']
            features['uncertainty_count'] = sum(1 for word in words if word in uncertainty)
            
        else:
            features.update({
                'avg_word_length': 0,
                'lexical_diversity': 0,
                'exclamation_count': 0,
                'question_count': 0,
                'period_count': 0,
                'upper_ratio': 0,
                'emotion_word_count': 0,
                'intensifier_count': 0,
                'negation_count': 0,
                'first_person_count': 0,
                'uncertainty_count': 0
            })
            
        return features
    
    def preprocess_text(self, text):
        """Preprocesa un texto para el análisis"""
        if not isinstance(text, str):
            return ""
        
        # Convertir emojis a texto
        text = emoji.demojize(text, delimiters=(" ", " "))
        
        # Convertir a minúsculas
        text = text.lower()
        
        # Manejar negaciones (importante para análisis de sentimiento)
        text = re.sub(r'\bno\s+', 'no_', text)
        text = re.sub(r'\bnunca\s+', 'nunca_', text)
        
        # Eliminar caracteres especiales pero mantener signos de puntuación importantes
        text = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s.,!?]', ' ', text)
        
        # Manejar repetición de caracteres (ej: nooooo -> no)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # Tokenizar
        tokens = word_tokenize(text, language='spanish')
        
        # Eliminar stopwords
        tokens = [word for word in tokens if word not in self.spanish_stopwords]
        
        # Aplicar stemming
        tokens = [self.stemmer.stem(word) for word in tokens]
        
        return ' '.join(tokens)
    
    def preprocess_data(self, texts):
        """Preprocesa los textos para el modelo"""
        return self.text_features.fit_transform(texts)
    
    def train(self, X_train, y_train, cv_folds=5):
        """
        Entrena el modelo con validación cruzada
        
        Args:
            X_train: Lista de textos de entrenamiento
            y_train: Etiquetas de entrenamiento
            cv_folds: Número de divisiones para la validación cruzada
            
        Returns:
            dict: Diccionario con métricas de evaluación
        """
        # Extraer características
        print("Extrayendo características...")
        X_vectors = self.text_features.fit_transform(X_train)
        
        # Entrenar el modelo final con todos los datos
        print("Entrenando el modelo final...")
        self.model.fit(X_vectors, y_train)
        
        # Realizar validación cruzada
        print(f"Realizando validación cruzada ({cv_folds} folds)...")
        from sklearn.model_selection import cross_validate
        
        scoring = {
            'accuracy': 'accuracy',
            'f1': 'f1_weighted',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted'
        }
        
        cv_results = cross_validate(
            self.model, X_vectors, y_train,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Calcular métricas promedio
        metrics = {
            'train_accuracy': np.mean(cv_results['train_accuracy']),
            'test_accuracy': np.mean(cv_results['test_accuracy']),
            'train_f1': np.mean(cv_results['train_f1']),
            'test_f1': np.mean(cv_results['test_f1']),
            'train_precision': np.mean(cv_results['train_precision']),
            'test_precision': np.mean(cv_results['test_precision']),
            'train_recall': np.mean(cv_results['train_recall']),
            'test_recall': np.mean(cv_results['test_recall']),
        }
        
        # Imprimir resumen
        print("\nMétricas de entrenamiento (promedio):")
        print(f"  Precisión: {metrics['train_accuracy']:.4f}")
        print(f"  F1-score: {metrics['train_f1']:.4f}")
        print(f"  Precisión: {metrics['train_precision']:.4f}")
        print(f"  Recall: {metrics['train_recall']:.4f}")
        
        print("\nMétricas de validación (promedio):")
        print(f"  Precisión: {metrics['test_accuracy']:.4f}")
        print(f"  F1-score: {metrics['test_f1']:.4f}")
        print(f"  Precisión: {metrics['test_precision']:.4f}")
        print(f"  Recall: {metrics['test_recall']:.4f}")
        
        return metrics
        
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
            
            # Si no hay características de texto o modelo cargado, usar solo las palabras clave
            if not hasattr(self, 'text_features') or not hasattr(self, 'model'):
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
            
            # Si hay características de texto y modelo, usarlos para la predicción
            X_vector = self.text_features.transform([text])
            
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
        X_vectors = self.text_features.transform(X_test)
        y_pred = self.model.predict(X_vectors)
        
        # Calcular métricas adicionales
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        # Obtener el reporte de clasificación
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Agregar métricas adicionales al reporte
        report['accuracy'] = accuracy
        report['weighted_avg'] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': len(y_test)
        }
        
        return report
    
    def save_model(self, path='/app/data/models/'):
        """Guarda el modelo y las características de texto"""
        try:
            # Asegurarse de que el directorio existe
            os.makedirs(path, exist_ok=True)
            
            # Guardar el modelo
            model_path = os.path.join(path, 'bullying_detection_model.joblib')
            joblib.dump(self.model, model_path)
            print(f"Modelo guardado en: {model_path}")
            
            # Guardar las características de texto
            features_path = os.path.join(path, 'text_features.joblib')
            joblib.dump(self.text_features, features_path)
            print(f"Características de texto guardadas en: {features_path}")
            
            # Guardar metadatos del modelo
            metadata = {
                'model_type': 'StackingClassifier',
                'base_models': ['LogisticRegression', 'SVC'],
                'final_estimator': 'RandomForestClassifier',
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
        """Carga el modelo y las características de texto"""
        try:
            # Cargar el modelo
            model_path = os.path.join(path, 'bullying_detection_model.joblib')
            self.model = joblib.load(model_path)
            print(f"Modelo cargado desde: {model_path}")
            
            # Cargar las características de texto
            features_path = os.path.join(path, 'text_features.joblib')
            self.text_features = joblib.load(features_path)
            print(f"Características de texto cargadas desde: {features_path}")
            
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            raise 

    def analyze_text(self, text):
        """Analiza un texto y proporciona un análisis detallado"""
        prediction, probability = self.predict(text)
        confidence = max(probability)
        
        # Obtener las palabras más importantes
        # Para el modelo de stacking, no podemos obtener coeficientes directos
        # En su lugar, usaremos las características de texto para el análisis
        feature_names = []
        
        # Obtener nombres de características de los transformadores
        for name, transformer in self.text_features.transformer_list:
            if hasattr(transformer, 'get_feature_names_out'):
                feature_names.extend([f"{name}__{f}" for f in transformer.get_feature_names_out()])
        
        # Para el modelo de stacking, no hay una forma directa de obtener la importancia de las características
        # Usaremos un enfoque de permutación para estimar la importancia
        from sklearn.inspection import permutation_importance
        
        # Tomar una muestra del texto para el análisis
        sample_text = [text]
        X_sample = self.text_features.transform(sample_text)
        
        # Calcular importancia por permutación
        result = permutation_importance(
            self.model, X_sample, [1],  # Asumimos clase 1 (bullying) para el análisis
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Obtener las características más importantes
        importances = result.importances_mean
        word_importance = list(zip(feature_names, importances))
        word_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'prediction': prediction,
            'probability': probability.tolist(),
            'confidence': float(confidence),
            'risk_level': 'alto' if probability[1] > 0.8 else 'medio' if probability[1] > 0.5 else 'bajo',
            'key_terms': [word for word, _ in word_importance[:5]]
        } 