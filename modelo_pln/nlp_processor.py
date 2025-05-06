import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import re
import string

class NLPProcessor:
    def __init__(self):
        # Descargar recursos necesarios de NLTK
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        # Inicializar herramientas de procesamiento
        self.stemmer = SnowballStemmer('spanish')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('spanish'))
        
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
        """Analiza el sentimiento del texto"""
        processed = self.process_text(text)
        
        # Palabras positivas y negativas en español
        positive_words = {'feliz', 'contento', 'alegre', 'bueno', 'mejor', 'excelente', 'genial'}
        negative_words = {'triste', 'mal', 'tristeza', 'deprimido', 'ansioso', 'miedo', 'preocupado'}
        
        # Contar palabras positivas y negativas
        positive_count = sum(1 for token in processed['tokens'] if token in positive_words)
        negative_count = sum(1 for token in processed['tokens'] if token in negative_words)
        
        # Calcular sentimiento
        total_words = len(processed['tokens'])
        if total_words == 0:
            return 0
        
        sentiment_score = (positive_count - negative_count) / total_words
        
        return {
            'score': sentiment_score,
            'positive_words': positive_count,
            'negative_words': negative_count,
            'total_words': total_words
        }
    
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