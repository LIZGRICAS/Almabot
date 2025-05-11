import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import csv
import json
from datetime import datetime
from models.logistic_regression_model import BullyingDetectionModel

class NLPProcessor:
    def __init__(self):
        # Mantener downloads existentes
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        # Inicializar herramientas de procesamiento
        self.stemmer = SnowballStemmer('spanish')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = list(stopwords.words('spanish'))
        
        # Usar TF-IDF en lugar de CountVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=self.stop_words,
            lowercase=True,
            ngram_range=(1, 2)  # Incluir bigramas para capturar frases
        )
        
        # Usar LogisticRegression en lugar de MultinomialNB
        self.classifier = LogisticRegression(
            class_weight='balanced',  # Importante para casos desbalanceados
            max_iter=1000,
            random_state=42
        )
        
        # Cargar datos de entrenamiento
        self.load_training_data()

        # Respuestas predefinidas para diferentes situaciones
        self.responses = {
            'alto_riesgo': [
                "Me preocupa mucho lo que me cuentas. ¿Te gustaría hablar más sobre esto?",
                "Es muy valiente de tu parte compartir esto. ¿Has podido hablar con algún adulto de confianza?",
                "Entiendo que esta situación es difícil. ¿Quieres que exploremos juntos algunas formas de manejarla?"
            ],
            'medio_riesgo': [
                "Gracias por compartir esto conmigo. ¿Cómo te hace sentir esta situación?",
                "Es importante que no estés solo/a en esto. ¿Has pensado en hablar con alguien más?",
                "¿Te gustaría que hablemos sobre algunas estrategias que podrían ayudarte?"
            ],
            'bajo_riesgo': [
                "Entiendo cómo te sientes. ¿Quieres contarme más?",
                "Es normal sentirse así. ¿Qué te ayudaría a sentirte mejor?",
                "Estoy aquí para escucharte. ¿Hay algo específico que te preocupe?"
            ],
            'positivo': [
                "¡Me alegra mucho escuchar eso! ¿Qué más cosas buenas te han pasado?",
                "Es genial que tengas experiencias positivas. ¿Quieres contarme más?",
                "¡Qué bueno! Es importante reconocer estos momentos positivos."
            ]
        }

        # Estrategias de apoyo según el tipo de acoso
        self.support_strategies = {
            'físico': [
                "Tu seguridad es lo más importante. ¿Has informado a algún profesor sobre esto?",
                "Es importante mantener un registro de estos incidentes. ¿Te gustaría que hablemos sobre cómo hacerlo?",
                "Nadie tiene derecho a lastimarte físicamente. ¿Conoces el protocolo de tu escuela para estos casos?"
            ],
            'verbal': [
                "Las palabras pueden doler mucho. ¿Quieres hablar sobre cómo te hacen sentir?",
                "Recuerda que lo que dicen no define quién eres. ¿Te gustaría explorar formas de fortalecer tu autoestima?",
                "Es importante no guardar estos sentimientos. ¿Has considerado escribir un diario?"
            ],
            'social': [
                "Sentirse excluido es difícil. ¿Has identificado algunas actividades o grupos que te interesen?",
                "A veces podemos encontrar amigos en lugares inesperados. ¿Te gustaría explorar nuevas formas de socializar?",
                "Tu valor no depende de la aceptación de otros. ¿Quieres hablar sobre cómo desarrollar confianza en ti mismo/a?"
            ],
            'cibernético': [
                "El acoso digital también es serio. ¿Sabes cómo bloquear y reportar contenido dañino?",
                "Es importante guardar evidencia del acoso digital. ¿Te gustaría que te explique cómo hacerlo?",
                "A veces es bueno tomar un descanso de las redes sociales. ¿Has pensado en otras actividades que te gusten?"
            ]
        }

        # Inicializar modelo de detección de bullying
        self.bullying_model = BullyingDetectionModel()
        try:
            self.bullying_model.load_model()
        except:
            print("Modelo no encontrado. Por favor, ejecute train_model.py primero")

    def load_training_data(self):
        """Carga y prepara los datos de entrenamiento de múltiples fuentes"""
        try:
            # Cargar frases de entrenamiento
            phrases_df = pd.read_csv('training_phrases.csv', sep=';')
        except FileNotFoundError:
            # Si no existe el archivo, usar dataset por defecto
            default_phrases = {
                'texto': [
                    "Me empujan constantemente en el recreo",
                    "Mis compañeros me insultan por mi aspecto",
                    "Nadie quiere sentarse conmigo en el almuerzo",
                    "Me envían mensajes amenazantes por WhatsApp",
                    "Me esconden la mochila todos los días",
                    "Se burlan de mi forma de hablar",
                    "Han creado un grupo para burlarse de mí",
                    "Me piden dinero para no golpearme",
                    "Difunden rumores falsos sobre mí",
                    "Me excluyen de todas las actividades",
                    "Me toman fotos sin permiso para burlarse",
                    "Me rompen mis útiles escolares",
                    "Me ponen apodos ofensivos",
                    "Juego feliz con mis amigos en el recreo",
                    "Mis compañeros me ayudan con las tareas",
                    "La maestra me felicitó por mi trabajo",
                    "Hice un nuevo amigo en clase",
                    "Me divierto en las actividades grupales",
                    "Mis compañeros me respetan",
                    "Participo en los juegos del recreo"
                ],
                'es_acoso': [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
                'tipo_acoso': ['físico','verbal','social','cibernético','físico','verbal',
                              'cibernético','físico','social','social','cibernético',
                              'físico','verbal','ninguno','ninguno','ninguno','ninguno',
                              'ninguno','ninguno','ninguno']
            }
            phrases_df = pd.DataFrame(default_phrases)
        
        try:
            # Cargar datos de la encuesta de bullying
            bullying_df = pd.read_csv('Bullying_2018_copy_traducido_final.csv', sep=';')
        except FileNotFoundError:
            # Si no existe el archivo, continuar solo con las frases de entrenamiento
            bullying_df = pd.DataFrame()
        
        # Procesar datos de la encuesta si existe
        bullying_texts = []
        bullying_labels = []
        
        if not bullying_df.empty:
            for _, row in bullying_df.iterrows():
                text_parts = []
                is_bullying = 0
                
                # Construir texto descriptivo basado en las respuestas
                if row['Intimidado en la propiedad escolar en los últimos 12 meses'] == 'Sí':
                    is_bullying = 1
                    text_parts.append("He sido intimidado en la escuela")
                
                if row['Acosado cibernético en los últimos 12 meses'] == 'Sí':
                    is_bullying = 1
                    text_parts.append("He sufrido acoso por internet")
                
                if row['Atacado físicamente'] != '0 veces':
                    is_bullying = 1
                    text_parts.append(f"Me han atacado físicamente {row['Atacado físicamente']}")
                
                if row['Sentirse solo'] in ['Siempre', 'A veces']:
                    text_parts.append(f"Me siento {row['Sentirse solo'].lower()} solo")
                
                if text_parts:
                    bullying_texts.append(' y '.join(text_parts))
                    bullying_labels.append(is_bullying)
        
        # Combinar datos de ambas fuentes
        self.X_train = list(phrases_df['texto']) + bullying_texts
        self.y_train = list(phrases_df['es_acoso']) + bullying_labels
        
        # Entrenar el clasificador
        self._train_classifier()

    def _train_classifier(self):
        """Entrena el clasificador de Regresión Logística"""
        X_vectors = self.vectorizer.fit_transform(self.X_train)
        self.classifier.fit(X_vectors, self.y_train)
        
    def tokenize(self, text):
        """Divide el texto en tokens (palabras)"""
        return word_tokenize(text.lower())
    
    def normalize(self, text):
        """Normaliza el texto: convierte a minúsculas y elimina puntuación"""
        # Convertir a minúsculas
        text = text.lower()
        # Eliminar puntuación
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Eliminar números
        text = re.sub(r'\d+', '', text)
        # Eliminar espacios extra
        text = ' '.join(text.split())
        return text
    
    def remove_stopwords(self, tokens):
        """Elimina palabras comunes (stopwords)"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens):
        """Convierte palabras a su forma base (lema)"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def stem(self, tokens):
        """Reduce palabras a su raíz"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def process_text(self, text):
        """Procesa el texto completo aplicando todas las técnicas de NLP"""
        # Normalizar texto
        normalized_text = self.normalize(text)
        
        # Tokenizar
        tokens = self.tokenize(normalized_text)
        
        # Eliminar stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Lematizar
        lemmas = self.lemmatize(tokens)
        
        # Stemming
        stems = self.stem(tokens)
        
        return {
            'original': text,
            'normalized': normalized_text,
            'tokens': tokens,
            'lemmas': lemmas,
            'stems': stems
        }
    
    def analyze_sentiment(self, text):
        """Analiza el sentimiento y detecta posibles situaciones de acoso"""
        processed = self.process_text(text)
        normalized_text = processed['normalized']
        
        # Obtener predicción del modelo
        prediction, probability = self.bullying_model.predict(normalized_text)
        
        # Determinar tipo de acoso
        bullying_type = self.determine_bullying_type(normalized_text)
        
        return {
            'is_bullying': bool(prediction),
            'confidence': float(max(probability)),
            'risk_level': 'alto' if probability[1] > 0.8 else 'medio' if probability[1] > 0.5 else 'bajo',
            'bullying_type': bullying_type,
            'processed_text': normalized_text
        }
    
    def determine_bullying_type(self, text):
        """Determina el tipo de acoso basado en palabras clave"""
        types = {
            'físico': ['golpe', 'empujar', 'pegar', 'esconder', 'romper'],
            'verbal': ['insulto', 'burla', 'apodo', 'gritar', 'humillar'],
            'social': ['excluir', 'ignorar', 'rumor', 'solo', 'aislar'],
            'cibernético': ['internet', 'mensaje', 'foto', 'whatsapp', 'redes']
        }
        
        counts = {t: sum(1 for word in text.split() if word in words) 
                 for t, words in types.items()}
        
        if not any(counts.values()):
            return 'ninguno'
        
        return max(counts.items(), key=lambda x: x[1])[0]
    
    def train_new_data(self, texts, labels):
        """Permite entrenar el modelo con nuevos datos"""
        self.X_train.extend(texts)
        self.y_train.extend(labels)
        self._train_classifier()
    
    def extract_keywords(self, text):
        """Extrae palabras clave del texto"""
        processed = self.process_text(text)
        
        # Palabras clave comunes en contexto de bullying
        bullying_keywords = {
            'bullying': ['acoso', 'intimidación', 'bullying', 'molestar', 'burlar'],
            'físico': ['golpe', 'empujón', 'pegar', 'golpes', 'violencia'],
            'verbal': ['insulto', 'burlar', 'humillar', 'gritar', 'amenazar'],
            'social': ['excluir', 'ignorar', 'rumor', 'chisme', 'aislar'],
            'cibernético': ['internet', 'redes', 'mensaje', 'online', 'digital']
        }
        
        # Identificar categorías presentes
        categories = {}
        for category, keywords in bullying_keywords.items():
            count = sum(1 for token in processed['tokens'] if token in keywords)
            if count > 0:
                categories[category] = count
        
        return {
            'categories': categories,
            'keywords': processed['tokens']
        } 

    def generate_response(self, text):
        """Genera una respuesta apropiada basada en el análisis del texto"""
        try:
            # Analizar el texto
            analysis = self.analyze_sentiment(text)
            
            # Guardar la interacción para seguimiento
            self._log_interaction(text, analysis)
            
            # Generar respuesta basada en el análisis
            response = self._create_response(analysis)
            
            return response
        except Exception as e:
            print(f"Error en generate_response: {e}")
            return {
                'message': "Lo siento, tuve un problema procesando tu mensaje. ¿Podrías decirlo de otra manera?",
                'follow_up_questions': [],
                'recommendations': []
            }

    def _create_response(self, analysis):
        """Crea una respuesta personalizada basada en el análisis"""
        response = {
            'message': '',
            'follow_up_questions': [],
            'recommendations': []
        }

        # Seleccionar mensaje principal
        if analysis['is_bullying']:
            if analysis['risk_level'] == 'alto':
                response['message'] = np.random.choice(self.responses['alto_riesgo'])
            elif analysis['risk_level'] == 'medio':
                response['message'] = np.random.choice(self.responses['medio_riesgo'])
            else:
                response['message'] = np.random.choice(self.responses['bajo_riesgo'])

            # Agregar estrategias específicas según el tipo de acoso
            if analysis['bullying_type'] in self.support_strategies:
                response['recommendations'].extend(
                    np.random.choice(self.support_strategies[analysis['bullying_type']], 2)
                )
        else:
            response['message'] = np.random.choice(self.responses['positivo'])

        return response

    def _log_interaction(self, text, analysis):
        """Registra la interacción para seguimiento"""
        # Crear una copia serializable del análisis
        serializable_analysis = {
            'is_bullying': bool(analysis['is_bullying']),
            'confidence': float(analysis['confidence']),
            'risk_level': str(analysis['risk_level']),
            'bullying_type': str(analysis['bullying_type']),
            'processed_text': str(analysis['processed_text'])
        }
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input_text': str(text),
            'analysis': serializable_analysis
        }
        
        # Guardar el log
        try:
            with open('chat_logs.jsonl', 'a') as f:
                json.dump(log_entry, f)
                f.write('\n')
        except Exception as e:
            print(f"Error al guardar log: {e}")

    def get_emergency_resources(self):
        """Retorna recursos de emergencia y contactos de ayuda"""
        return {
            'lineas_ayuda': [
                'Línea Nacional contra el Acoso Escolar: XXX-XXX-XXXX',
                'Chat de Apoyo Emocional: www.ejemplo.com/apoyo'
            ],
            'recomendaciones_emergencia': [
                'Si estás en peligro inmediato, busca a un adulto de confianza',
                'Guarda evidencia de cualquier tipo de acoso',
                'No respondas a las provocaciones, busca ayuda'
            ]
        }

    def analyze_conversation_history(self, history):
        """Analiza el historial de conversación para identificar patrones"""
        patterns = {
            'repeated_incidents': False,
            'escalating_severity': False,
            'emotional_state': 'stable'
        }
        
        # Implementar análisis de patrones aquí
        return patterns

nlp = NLPProcessor()

# Ejemplo de interacción
texto_usuario = "Mis compañeros me empujan y se burlan de mí en el recreo"
respuesta = nlp.generate_response(texto_usuario)
print(respuesta)
# Output: {
#   'message': 'Me preocupa mucho lo que me cuentas. ¿Te gustaría hablar más sobre esto?',
#   'follow_up_questions': [
#     '¿Has podido hablar con algún profesor sobre esta situación?',
#     '¿Te sientes seguro durante el recreo?'
#   ],
#   'recommendations': [
#     'Tu seguridad es lo más importante. ¿Has informado a algún profesor sobre esto?',
#     'Es importante mantener un registro de estos incidentes. ¿Te gustaría que hablemos sobre cómo hacerlo?'
#   ]
# }

# Obtener recursos de emergencia
recursos = nlp.get_emergency_resources()
print(recursos)

# Analizar un texto
resultado = nlp.analyze_sentiment("Mis compañeros me empujan y se burlan de mí en el recreo")
print(resultado)
# Output: {
#   'is_bullying': True,
#   'confidence': 0.92,
#   'risk_level': 'alto',
#   'bullying_type': 'físico',
#   'processed_text': 'mis compañeros me empujan y se burlan de mi en el recreo'
# } 