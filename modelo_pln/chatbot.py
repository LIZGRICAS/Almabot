from nlp_processor import NLPProcessor
import random
from datetime import datetime

class ChatbotTerapia:
    def __init__(self):
        self.nlp = NLPProcessor()
        self.conversation_history = []
        self.user_context = {
            'name': None,
            'age': None,
            'school': None,
            'neighborhood': None,
            'emotional_state': None,
            'bullying_type': None
        }
        
        # Respuestas empáticas
        self.empathic_responses = {
            'positive': [
                "Me alegra que estés teniendo una experiencia positiva. ¿Te gustaría compartir más sobre eso?",
                "Es bueno escuchar que las cosas van bien. ¿Hay algo específico que quieras discutir?",
                "Me alegra que estés en un buen momento. ¿Cómo puedo ayudarte a mantener esta positividad?"
            ],
            'negative': [
                "Lamento mucho que estés pasando por esto. ¿Te gustaría contarme más sobre lo que te está afectando?",
                "Entiendo que esto debe ser difícil para ti. ¿Qué es lo que más te preocupa en este momento?",
                "Es normal sentirse así. ¿Quieres que exploremos juntos qué está causando estos sentimientos?",
                "Me duele escuchar que estás pasando por un momento difícil. ¿Hay algo específico que te gustaría compartir?",
                "Estoy aquí para escucharte. ¿Qué te ha llevado a sentirte así?"
            ],
            'neutral': [
                "Entiendo. ¿Podrías contarme más sobre lo que te preocupa?",
                "Gracias por compartir esto conmigo. ¿Qué te gustaría hacer al respecto?",
                "Estoy aquí para escucharte. ¿Qué sientes que necesitas en este momento?",
                "¿Te gustaría que exploremos juntos algunas opciones para manejar esta situación?",
                "¿Qué te gustaría que hiciéramos para ayudarte a sentirte mejor?"
            ]
        }
        
        # Consejos específicos basados en patrones
        self.advice_patterns = {
            'loneliness': [
                "Según nuestros datos, tener 3 o más amigos cercanos puede ayudar a prevenir el bullying.",
                "Es importante mantener conexiones sociales. ¿Te gustaría hablar sobre cómo hacer más amigos?",
                "No estás solo/a. Muchos estudiantes han pasado por situaciones similares."
            ],
            'physical': [
                "Si has sido atacado físicamente, es importante reportarlo inmediatamente a las autoridades escolares.",
                "La violencia física nunca es aceptable. ¿Has hablado con alguien sobre esto?",
                "Recuerda que tienes derecho a estar seguro/a en la escuela."
            ],
            'cyberbullying': [
                "El cyberbullying es tan serio como el bullying presencial. ¿Has guardado evidencia de los mensajes?",
                "Es importante no responder a los agresores en línea. ¿Has bloqueado a las personas que te acosan?",
                "¿Has hablado con tus padres o profesores sobre el cyberbullying que estás experimentando?"
            ],
            'school_absence': [
                "Faltar a la escuela no es la solución. ¿Has hablado con un consejero escolar sobre esto?",
                "Es importante mantener tu educación. ¿Te gustaría explorar opciones para sentirte más seguro/a en la escuela?",
                "¿Has considerado cambiar de escuela si la situación es muy grave?"
            ]
        }
    
    def process_message(self, message):
        """Procesa el mensaje del usuario y genera una respuesta apropiada"""
        # Analizar sentimiento
        sentiment = self.nlp.analyze_sentiment(message)
        
        # Extraer palabras clave
        keywords = self.nlp.extract_keywords(message)
        
        # Actualizar contexto del usuario
        self._update_user_context(keywords)
        
        # Generar respuesta
        response = self._generate_response(sentiment, keywords)
        
        # Guardar en historial
        self._save_to_history(message, response)
        
        return response
    
    def _update_user_context(self, keywords):
        """Actualiza el contexto del usuario basado en las palabras clave"""
        categories = keywords['categories']
        
        # Actualizar tipo de bullying
        if categories:
            self.user_context['bullying_type'] = max(categories.items(), key=lambda x: x[1])[0]
        
        # Actualizar estado emocional basado en palabras clave
        emotional_keywords = {
            'triste': 'negative',
            'feliz': 'positive',
            'preocupado': 'negative',
            'ansioso': 'negative',
            'contento': 'positive'
        }
        
        for keyword in keywords['keywords']:
            if keyword in emotional_keywords:
                self.user_context['emotional_state'] = emotional_keywords[keyword]
    
    def _generate_response(self, sentiment, keywords):
        """Genera una respuesta basada en el análisis del mensaje"""
        # Determinar el estado emocional predominante
        if sentiment['score'] > 0.1:
            emotional_state = 'positive'
        elif sentiment['score'] < -0.1:
            emotional_state = 'negative'
        else:
            emotional_state = 'neutral'
        
        # Obtener respuesta empática
        response = random.choice(self.empathic_responses[emotional_state])
        
        # Agregar consejos específicos si se detectan patrones
        if keywords['categories']:
            for category in keywords['categories']:
                if category in self.advice_patterns:
                    response += "\n\n" + random.choice(self.advice_patterns[category])
        
        return response
    
    def _save_to_history(self, message, response):
        """Guarda la interacción en el historial de conversación"""
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'message': message,
            'response': response,
            'context': self.user_context.copy()
        })
    
    def get_conversation_history(self):
        """Retorna el historial de conversación"""
        return self.conversation_history
    
    def get_user_context(self):
        """Retorna el contexto actual del usuario"""
        return self.user_context 