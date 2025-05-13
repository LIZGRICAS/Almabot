from src.nlp.nlp_processor import NLPProcessor
from src.models.logistic_regression_model import BullyingDetectionModel
import json
import os
import random
import re
import datetime

class Chatbot:
    def __init__(self):
        self.nlp = NLPProcessor()
        self.bullying_model = BullyingDetectionModel()
        try:
            self.bullying_model.load_model()
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
        
        self.log_file = 'chat_logs.jsonl'
        self.logs = []
        self._load_logs()
    
    def _load_logs(self):
        """
        Load previous chat logs if they exist
        """
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        self.logs.append(json.loads(line))
                    except:
                        continue
    
    def _save_log(self, user_id, message, response, analysis):
        """
        Save chat log to file
        """
        log_entry = {
            'user_id': user_id,
            'message': message,
            'response': response,
            'analysis': analysis,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.logs.append(log_entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def process_message(self, user_id, message):
        """
        Process a message and determine if it involves bullying
        Returns a response and analysis
        """
        if not message or not message.strip():
            return "Por favor, escribe un mensaje para que pueda ayudarte.", {
                'is_bullying': False,
                'confidence': 0.0,
                'bullying_type': "ninguno",
                'keywords': {'categories': {}, 'keywords': []},
                'sentiment': 'neutral',
                'sentiment_score': 0.0,
                'emotion': {'type': 'neutral', 'intensity': 0.0},
                'risk': {'level': 'none', 'type': 'none'},
                'context': {'hypothetical': False, 'context_words': []}
            }
        
        try:
            # Preprocess the message
            message_lower = message.lower()
            
            # Check for common greetings and neutral messages first
            greetings = ["hola", "buenos días", "buenas tardes", "buenas noches", "saludos", "hey", "qué tal", "cómo estás", "cómo te va", "qué hay"]
            
            # Check for wellness questions or advice seeking that shouldn't be classified as bullying
            wellness_questions = [
                "qué puedo hacer", "cómo puedo", "qué debo hacer", "cómo me puedo", "qué me recomiendas",
                "qué me sugieres", "cómo debo", "necesito ayuda", "necesito consejo", "me puedes ayudar",
                "tienes algún consejo", "tienes alguna recomendación", "qué harías", "qué haría"
            ]
            
            # Check for neutral conversation starters that shouldn't be classified as bullying
            conversation_starters = [
                "quiero conversar", "quiero hablar", "quiero platicar", "podemos hablar", "podemos conversar",
                "me gustaría hablar", "me gustaría conversar", "me gustaría platicar", "quisiera hablar",
                "quisiera conversar", "quisiera platicar", "vamos a hablar", "vamos a conversar", "vamos a platicar",
                "hablemos", "conversemos", "platiquemos", "cuéntame", "dime", "pregunta"
            ]
            
            # If the message is just a greeting or very short neutral message, skip bullying detection
            if message_lower in greetings or (len(message_lower.split()) <= 3 and any(greeting in message_lower for greeting in greetings)):
                print(f"Detected greeting: '{message_lower}'. Skipping bullying detection.")
                # Generate a friendly greeting response
                return self._generate_greeting_response(message_lower), {
                    'is_bullying': False,
                    'confidence': 0.0,
                    'bullying_type': "ninguno",
                    'keywords': {'categories': {}, 'keywords': message_lower.split()},
                    'sentiment': 'positive',
                    'sentiment_score': 0.5,
                    'emotion': {'type': 'joy', 'intensity': 0.3},
                    'risk': {'level': 'none', 'type': 'none'},
                    'context': {'hypothetical': False, 'context_words': []}
                }
            
            # Check if this is a neutral conversation starter
            is_conversation_starter = any(starter in message_lower for starter in conversation_starters) or message_lower in conversation_starters
            
            # If it's a neutral conversation starter, provide a conversational response
            if is_conversation_starter and not any(word in message_lower for word in ["bullying", "acoso", "molestan", "burlan", "pegan", "triste", "mal"]):
                print(f"Detected conversation starter: '{message_lower}'. Providing conversational response.")
                # Generate a conversational response
                return self._generate_conversation_response(message_lower), {
                    'is_bullying': False,
                    'confidence': 0.9,
                    'bullying_type': "ninguno",
                    'keywords': {'categories': {}, 'keywords': message_lower.split()},
                    'sentiment': 'neutral',
                    'sentiment_score': 0.2,
                    'emotion': {'type': 'neutral', 'intensity': 0.3},
                    'risk': {'level': 'low', 'type': 'none'},
                    'context': {'hypothetical': False, 'context_words': []}
                }
            
            # Check for wellness questions or advice seeking (not bullying)
            is_wellness_question = any(question in message_lower for question in wellness_questions)
            
            # If it's a wellness question without specific bullying context, treat it as non-bullying
            if is_wellness_question and not any(word in message_lower for word in ["bullying", "acoso", "molestan", "burlan", "pegan"]):
                print(f"Detected wellness question: '{message_lower}'. Providing supportive response.")
                # Generate a supportive response for wellness questions
                return self._generate_wellness_response(message_lower), {
                    'is_bullying': False,
                    'confidence': 0.8,
                    'bullying_type': "ninguno",
                    'keywords': {'categories': {}, 'keywords': message_lower.split()},
                    'sentiment': 'neutral',
                    'sentiment_score': 0.0,
                    'emotion': {'type': 'neutral', 'intensity': 0.5},
                    'risk': {'level': 'low', 'type': 'none'},
                    'context': {'hypothetical': False, 'context_words': []}
                }
            
            # Check for emotional expressions that might indicate distress
            emotional_expressions = {
                # Tristeza
                "tristeza": ["me siento triste", "estoy triste", "me siento mal", "me siento deprimido", "me siento deprimida", 
                            "me siento desanimado", "me siento desanimada", "me siento abatido", "me siento abatida",
                            "tengo ganas de llorar", "he llorado", "lloro", "me siento sin ánimo", "no tengo ganas",
                            "triste", "me siento un poco triste", "estoy un poco triste", "me siento algo triste"],
                
                # Ansiedad
                "ansiedad": ["me siento ansioso", "me siento ansiosa", "estoy nervioso", "estoy nerviosa", 
                            "me preocupa", "tengo miedo", "me asusta", "me da pánico", "me siento estresado", "me siento estresada"],
                
                # Soledad
                "soledad": ["me siento solo", "me siento sola", "estoy solo", "estoy sola", "nadie me quiere", 
                           "no tengo amigos", "no tengo amigas", "nadie habla conmigo", "me siento aislado", "me siento aislada"],
                
                # Bullying
                "bullying": ["me molestan", "se burlan", "me insultan", "me hacen sentir", "me excluyen",
                            "no me incluyen", "me dejan fuera", "me ignoran", "no me hablan",
                            "me pegan", "me empujan", "me golpean", "me amenazan"]
            }
            
            # Detect emotional expressions in the message
            detected_emotions = {}
            for emotion, indicators in emotional_expressions.items():
                if any(indicator in message_lower for indicator in indicators):
                    detected_emotions[emotion] = True
                    print(f"Detected emotional expression '{emotion}' in: '{message_lower}'")
            
            # If tristeza (sadness) is detected without bullying indicators, provide an empathetic response
            if "tristeza" in detected_emotions and not "bullying" in detected_emotions:
                print(f"Detected sadness without bullying in: '{message_lower}'")
                return self._generate_sadness_response(message_lower), {
                    'is_bullying': False,
                    'confidence': 0.0,
                    'bullying_type': "ninguno",
                    'keywords': {'categories': {}, 'keywords': message_lower.split()},
                    'sentiment': 'negative',
                    'sentiment_score': -0.6,
                    'emotion': {'type': 'sadness', 'intensity': 0.7},
                    'risk': {'level': 'low', 'type': 'emotional_distress'},
                    'context': {'hypothetical': False, 'context_words': []}
                }
                
            # If ansiedad (anxiety) is detected without bullying indicators, provide a calming response
            if "ansiedad" in detected_emotions and not "bullying" in detected_emotions:
                print(f"Detected anxiety without bullying in: '{message_lower}'")
                return self._generate_anxiety_response(message_lower), {
                    'is_bullying': False,
                    'confidence': 0.0,
                    'bullying_type': "ninguno",
                    'keywords': {'categories': {}, 'keywords': message_lower.split()},
                    'sentiment': 'negative',
                    'sentiment_score': -0.5,
                    'emotion': {'type': 'anxiety', 'intensity': 0.7},
                    'risk': {'level': 'low', 'type': 'emotional_distress'},
                    'context': {'hypothetical': False, 'context_words': []}
                }
                
            # If soledad (loneliness) is detected without bullying indicators, provide a supportive response
            if "soledad" in detected_emotions and not "bullying" in detected_emotions:
                print(f"Detected loneliness without bullying in: '{message_lower}'")
                return self._generate_loneliness_response(message_lower), {
                    'is_bullying': False,
                    'confidence': 0.0,
                    'bullying_type': "ninguno",
                    'keywords': {'categories': {}, 'keywords': message_lower.split()},
                    'sentiment': 'negative',
                    'sentiment_score': -0.7,
                    'emotion': {'type': 'loneliness', 'intensity': 0.8},
                    'risk': {'level': 'low', 'type': 'emotional_distress'},
                    'context': {'hypothetical': False, 'context_words': []}
                }
            
            # Check for bullying indicators directly in the message using a more comprehensive approach
            bullying_indicators = {
                # Verbal bullying
                "verbal": ["molestan", "burlan", "insultan", "gritan", "dicen cosas", "hablan mal", 
                          "rumores", "apodos", "burlas", "insultos", "ofenden", "humillan", "amenazan",
                          "se ríen", "se burlan", "me llaman", "me dicen"],
                # Physical bullying
                "físico": ["pegan", "empujan", "golpean", "patean", "lastiman", "agreden", "tiran", 
                          "rompen", "quitan", "esconden", "roban", "dañan", "me pegan", "me empujan",
                          "me golpean", "me patean", "me lastiman"],
                # Social bullying
                "social": ["ignoran", "excluyen", "rechazan", "aíslan", "no me juntan", "no me invitan", 
                          "solo", "sola", "no me hablan", "nadie juega", "dejan fuera", "no me incluyen",
                          "me ignoran", "me excluyen", "me rechazan"],
                # Cyberbullying
                "cibernético": ["mensajes", "internet", "redes", "fotos", "videos", "whatsapp", 
                               "instagram", "facebook", "tiktok", "publican", "comparten", "mandan"]
            }
            
            # School context indicators
            school_context = ["escuela", "colegio", "clase", "recreo", "compañeros", "compañeras", 
                             "maestro", "maestra", "profesor", "profesora", "alumno", "alumna", 
                             "salón", "patio", "baño", "pasillo", "aula", "curso", "grado", "año escolar"]
            
            # Detect bullying type based on indicators in the message
            detected_bullying_type = None
            for btype, indicators in bullying_indicators.items():
                if any(indicator in message_lower for indicator in indicators):
                    detected_bullying_type = btype
                    print(f"Detected {btype} bullying indicators in: '{message_lower}'")
                    break
                    
            # Check for school context
            is_school_context = any(context in message_lower for context in school_context)
            
            # If both bullying indicators and school context are detected
            if detected_bullying_type and is_school_context:
                print(f"Detected school bullying context in: '{message_lower}'")
                prediction = 1
                confidence = 0.85
                bullying_type = detected_bullying_type
            # If only bullying indicators are detected without clear school context
            elif detected_bullying_type:
                print(f"Detected bullying indicators without clear school context in: '{message_lower}'")
                prediction = 1
                confidence = 0.75
                bullying_type = detected_bullying_type
            # If "bullying" emotion was detected but no specific indicators
            elif "bullying" in detected_emotions:
                print(f"Detected bullying emotion without specific indicators in: '{message_lower}'")
                prediction = 1
                confidence = 0.6
                bullying_type = "psicológico"
            
            # Check if the message is about a hypothetical situation rather than a real experience
            context_indicators = {
                "película": 0.7, "serie": 0.7, "cuento": 0.7, "historia": 0.6, "libro": 0.6, 
                "videojuego": 0.8, "juego": 0.7, "ficción": 0.8, "personaje": 0.7,
                "tarea": 0.6, "proyecto": 0.6, "trabajo escolar": 0.7, "investigación": 0.7,
                "pregunta": 0.5, "hipotético": 0.8, "si alguien": 0.6
            }
            
            # Check if the message is about a hypothetical situation rather than a real experience
            context_words = [word for word in context_indicators.keys() if word in message_lower]
            context_confidence_reduction = max([context_indicators[word] for word in context_words]) if context_words else 0
            
            # Get sentiment analysis
            try:
                sentiment_result = self.nlp.analyze_sentiment(message)
                sentiment = sentiment_result.get('sentiment', 'neutral')
                sentiment_score = sentiment_result.get('score', 0.0)
            except Exception as e:
                print(f"Error in sentiment analysis: {e}")
                sentiment = 'neutral'
                sentiment_score = 0.0
            
            # Detect positive sentiment to avoid false positives
            positive_indicators = [
                "gracias", "feliz", "contento", "contenta", "alegre", "agradecido", "agradecida",
                "me gusta", "me encanta", "me hace feliz", "me alegra", "me divierte",
                "estoy bien", "estoy feliz", "estoy contento", "estoy contenta",
                "me siento bien", "me siento feliz", "me siento contento", "me siento contenta",
                "genial", "excelente", "fantástico", "maravilloso", "increíble", "asombroso",
                "divertido", "divertida", "gracioso", "graciosa", "positivo", "positiva"
            ]
            
            # If the message has strong positive sentiment, it's likely not bullying
            is_positive = sentiment == 'positive' and sentiment_score > 0.6
            has_positive_indicators = any(indicator in message_lower for indicator in positive_indicators)
            
            if is_positive or has_positive_indicators:
                print(f"Detected positive sentiment in: '{message_lower}'. Skipping bullying detection.")
                return self._generate_positive_response(message_lower), {
                    'is_bullying': False,
                    'confidence': 0.9,
                    'bullying_type': "ninguno",
                    'keywords': {'categories': {}, 'keywords': message_lower.split()},
                    'sentiment': 'positive',
                    'sentiment_score': sentiment_score,
                    'emotion': {'type': 'joy', 'intensity': 0.7},
                    'risk': {'level': 'none', 'type': 'none'},
                    'context': {'hypothetical': False, 'context_words': []}
                }
            
            # Use the model to predict if the message involves bullying
            try:
                prediction_result = self.bullying_model.predict(message)
                if isinstance(prediction_result, tuple) and len(prediction_result) == 3:
                    prediction, confidence, bullying_type = prediction_result
                else:
                    # Si el modelo devuelve menos valores de los esperados, establecemos valores predeterminados
                    print(f"Unexpected prediction result format: {prediction_result}")
                    prediction = 0
                    confidence = 0.0
                    bullying_type = "ninguno"
            except Exception as e:
                print(f"Error in prediction: {e}")
                prediction = 0
                confidence = 0.0
                bullying_type = "ninguno"
            
            # Adjust confidence based on context
            if context_confidence_reduction > 0:
                confidence = max(0, confidence - context_confidence_reduction)
                if confidence < 0.5:  # If confidence drops below threshold after adjustment
                    prediction = 0
                    bullying_type = "ninguno"
            
            # Determine emotion type and intensity
            emotion_type = "neutral"
            emotion_intensity = 0.5
            
            if prediction == 1:
                # If bullying is detected, set emotion based on bullying type
                if bullying_type == "verbal":
                    emotion_type = "sadness"
                    emotion_intensity = 0.7
                elif bullying_type == "físico":
                    emotion_type = "fear"
                    emotion_intensity = 0.8
                elif bullying_type == "social":
                    emotion_type = "loneliness"
                    emotion_intensity = 0.7
                else:  # psicológico
                    emotion_type = "distress"
                    emotion_intensity = 0.7
            else:
                # If no bullying, set emotion based on sentiment
                if sentiment == "positive":
                    emotion_type = "joy"
                    emotion_intensity = 0.6
                elif sentiment == "negative":
                    emotion_type = "sadness"
                    emotion_intensity = 0.5
            
            # Enhanced risk assessment logic with more detailed risk types
            risk_level = "low"
            risk_type = "none"
            
            # High-risk indicators (words suggesting immediate danger or severe situations)
            high_risk_words = [
                "suicid", "matar", "morir", "muerte", "no quiero vivir", "acabar con todo",
                "golpear", "pegar", "lastimar", "herir", "sangre", "herida", "hospital",
                "arma", "cuchillo", "pistola", "amenaza", "amenazar", "miedo", "terror",
                "abuso", "abusar", "violencia", "violento", "violenta", "agresión", "agresor"
            ]
            
            # Medium-risk indicators (words suggesting ongoing issues)
            medium_risk_words = [
                "todos los días", "siempre", "constantemente", "cada día", "no para", 
                "no puedo más", "cansado", "cansada", "harto", "harta", "desesperado", "desesperada",
                "triste", "deprimido", "deprimida", "solo", "sola", "nadie me ayuda",
                "no tengo amigos", "no tengo amigas", "no le importo a nadie"
            ]
            
            # Check for high-risk indicators
            has_high_risk = any(word in message_lower for word in high_risk_words)
            has_medium_risk = any(word in message_lower for word in medium_risk_words)
            
            # Determine risk based on bullying prediction, risk words, and sentiment
            if prediction == 1:
                if has_high_risk or confidence > 0.85:
                    risk_level = "high"
                    risk_type = "bullying_severe"
                elif has_medium_risk or confidence > 0.7:
                    risk_level = "medium"
                    risk_type = "bullying_ongoing"
                else:
                    risk_level = "low"
                    risk_type = "bullying_potential"
                    
                # Adjust risk level based on bullying type
                if bullying_type == "físico" and risk_level != "high":
                    risk_level = "high"  # Physical bullying is always high risk
                    risk_type = "bullying_physical"
                elif bullying_type == "cibernético" and risk_level == "low":
                    risk_level = "medium"  # Cyberbullying is at least medium risk
                    risk_type = "bullying_cyber"
            elif has_high_risk:
                risk_level = "medium"
                risk_type = "emotional_distress_severe"
            elif has_medium_risk or (sentiment == "negative" and sentiment_score < -0.6):
                risk_level = "low"
                risk_type = "emotional_distress"
            elif sentiment == "negative" and sentiment_score < -0.4:
                risk_level = "low"
                risk_type = "negative_emotion"
            
            # Generate a response based on the prediction
            response = self._generate_response(prediction, confidence, bullying_type)
            
            # Create analysis dictionary
            analysis = {
                'is_bullying': prediction == 1,
                'confidence': confidence,
                'bullying_type': bullying_type,
                'keywords': {'categories': {}, 'keywords': message_lower.split()},
                'sentiment': sentiment,
                'sentiment_score': sentiment_score,
                'emotion': {'type': emotion_type, 'intensity': emotion_intensity},
                'risk': {'level': risk_level, 'type': risk_type},
                'context': {'hypothetical': bool(context_words), 'context_words': context_words}
            }
            
            # Save the interaction to logs
            self._save_log(user_id, message, response, analysis)
            
            return response, analysis
            
        except Exception as e:
            print(f"Error processing message: {e}")
            return "Lo siento, hubo un error al procesar tu mensaje. Por favor, intenta de nuevo.", {
                'is_bullying': False,
                'confidence': 0.0,
                'bullying_type': "ninguno",
                'keywords': {'categories': {}, 'keywords': []},
                'sentiment': 'neutral',
                'sentiment_score': 0.0,
                'emotion': {'type': 'neutral', 'intensity': 0.0},
                'risk': {'level': 'none', 'type': 'none'},
                'context': {'hypothetical': False, 'context_words': []}
            }
    
    def _generate_greeting_response(self, greeting):
        """
        Generate a friendly response to a greeting
        """
        greeting_responses = [
            f"¡Hola! ¿Cómo estás hoy? Estoy aquí para escucharte y ayudarte en lo que necesites.",
            f"¡Hola! Me alegra que estés aquí. ¿En qué puedo ayudarte hoy?",
            f"¡Hola! Soy tu asistente virtual. ¿Hay algo de lo que quieras hablar?",
            f"¡Hola! Estoy aquí para ti. ¿Cómo te sientes hoy?",
            f"¡Hola! Es un gusto saludarte. ¿Hay algo en particular que te gustaría conversar?"
        ]
        return random.choice(greeting_responses)
        
    def _generate_conversation_response(self, message):
        """
        Generate a friendly response for conversation starters
        """
        conversation_responses = [
            "Claro, me encanta conversar. ¿Hay algún tema en particular del que te gustaría hablar? Estoy aquí para escucharte y ayudarte en lo que necesites.",
            
            "Por supuesto, estoy aquí para conversar contigo. Podemos hablar de cómo te sientes, de tus intereses o de cualquier cosa que te gustaría compartir. ¿Qué te gustaría contarme hoy?",
            
            "Me alegra que quieras conversar. Las conversaciones son una excelente manera de conectar y compartir. ¿Hay algo específico que te gustaría compartir o preguntar?",
            
            "Estoy aquí para escucharte y conversar sobre lo que necesites. ¿Cómo estás hoy? ¿Hay algo en particular que te gustaría compartir conmigo?",
            
            "Conversar es una gran manera de expresarnos y conocernos mejor. ¿Qué te gustaría compartir hoy? Estoy aquí para escucharte con atención."
        ]
        return random.choice(conversation_responses)
        
    def _generate_wellness_response(self, message):
        """
        Generate a supportive response for wellness questions
        """
        wellness_responses = [
            "Entiendo que quieras sentirte mejor. Aquí hay algunas sugerencias que podrían ayudarte:\n\n"
            "1. Habla con alguien de confianza sobre cómo te sientes\n"
            "2. Realiza actividades que disfrutes y te relajen\n"
            "3. Practica ejercicios de respiración profunda cuando te sientas ansioso/a\n"
            "4. Establece una rutina diaria que incluya tiempo para ti\n"
            "5. Recuerda que está bien pedir ayuda cuando la necesites\n\n"
            "¿Hay alguna de estas sugerencias que te gustaría explorar más?",
            
            "Para sentirte mejor, puedes intentar lo siguiente:\n\n"
            "1. Dedica tiempo a actividades que te gusten y te hagan sentir bien\n"
            "2. Conectáte con amigos o familiares que te apoyen\n"
            "3. Practica la atención plena o mindfulness para reducir el estrés\n"
            "4. Mantén hábitos saludables: buena alimentación, sueño y ejercicio\n"
            "5. Expresa tus emociones de forma saludable, como escribir un diario\n\n"
            "¿Cuál de estas ideas crees que podría funcionarte mejor?",
            
            "Hay varias estrategias que pueden ayudarte a sentirte mejor:\n\n"
            "1. Identifica qué situaciones o pensamientos te hacen sentir mal\n"
            "2. Busca actividades que te generen alegría y satisfacción\n"
            "3. Habla con un adulto de confianza sobre tus preocupaciones\n"
            "4. Practica técnicas de relajación como respiración profunda\n"
            "5. Recuerda tus fortalezas y logros pasados\n\n"
            "¿Te gustaría que hablemos más sobre alguna de estas estrategias?"
        ]
        return random.choice(wellness_responses)
        
    def _generate_positive_response(self, message):
        """
        Generate a response for positive messages
        """
        positive_responses = [
            "¡Me alegra mucho escuchar eso! Es genial que te sientas así. ¿Hay algo más que te gustaría compartir o en lo que pueda ayudarte?",
            "¡Qué bueno! Me da gusto saber que estás teniendo una experiencia positiva. Estoy aquí para seguir conversando sobre lo que necesites.",
            "¡Eso suena fantástico! Es importante reconocer y disfrutar de los momentos positivos. ¿Hay algo más en lo que pueda apoyarte hoy?",
            "¡Excelente! Me encanta escuchar cosas positivas. Si hay algo más en lo que pueda ayudarte o si quieres seguir compartiendo, estoy aquí para ti.",
            "¡Qué maravilla! Gracias por compartir esa experiencia positiva conmigo. Estoy aquí para seguir conversando sobre lo que tú quieras."
        ]
        return random.choice(positive_responses)
        
    def _generate_sadness_response(self, message):
        """
        Generate an empathetic response for messages expressing sadness
        """
        sadness_responses = [
            "Entiendo que te sientas triste. A veces todos pasamos por momentos difíciles. ¿Te gustaría hablar sobre lo que te está haciendo sentir así? Estoy aquí para escucharte.",
            
            "Lamento que estés pasando por un momento triste. Quiero que sepas que no estás solo/a en esto. ¿Hay algo específico que te esté preocupando o entristeciendo que te gustaría compartir?",
            
            "La tristeza es una emoción natural y es importante permitirnos sentirla. ¿Hay algo que pueda hacer para apoyarte en este momento? A veces simplemente hablar sobre lo que sentimos puede ayudarnos.",
            
            "Me importa cómo te sientes y quiero que sepas que estoy aquí para ti. ¿Hay algo en particular que te haya hecho sentir triste hoy? Compartirlo puede ser un primer paso para sentirte mejor.",
            
            "Gracias por compartir conmigo cómo te sientes. La tristeza es una emoción difícil, pero también nos ayuda a procesar situaciones complicadas. ¿Te gustaría hablar sobre lo que te está pasando?"
        ]
        return random.choice(sadness_responses)
        
    def _generate_anxiety_response(self, message):
        """
        Generate a calming response for messages expressing anxiety
        """
        anxiety_responses = [
            "Entiendo que te sientas ansioso/a. La ansiedad puede ser muy intensa, pero hay formas de manejarla. ¿Te gustaría que exploremos algunas técnicas de respiración que pueden ayudarte a calmarte?",
            
            "La ansiedad es una respuesta natural de nuestro cuerpo, aunque a veces puede ser abrumadora. ¿Hay algo específico que esté causando tu ansiedad? Identificar la causa puede ser el primer paso para manejarla.",
            
            "Lamento que estés experimentando ansiedad. Estoy aquí para apoyarte. Una técnica que puede ayudar es la respiración profunda: inhala contando hasta 4, mantén el aire contando hasta 2, y exhala contando hasta 6. ¿Te gustaría intentarlo?",
            
            "Es comprensible sentirse ansioso/a a veces. ¿Hay algo específico que te preocupe? A veces, hablar sobre nuestras preocupaciones puede ayudarnos a verlas desde una perspectiva diferente.",
            
            "La ansiedad puede ser muy difícil de manejar. Quiero que sepas que no estás solo/a en esto. ¿Has identificado qué situaciones o pensamientos desencadenan tu ansiedad? Esto podría ayudarnos a encontrar estrategias específicas para ti."
        ]
        return random.choice(anxiety_responses)
        
    def _generate_loneliness_response(self, message):
        """
        Generate a supportive response for messages expressing loneliness
        """
        loneliness_responses = [
            "Sentirse solo/a puede ser muy difícil. Quiero que sepas que estoy aquí para ti y que no estás solo/a en este momento. ¿Te gustaría hablar sobre lo que te hace sentir así?",
            
            "La soledad es una experiencia que todos sentimos en algún momento. ¿Hay algo específico que te esté haciendo sentir solo/a? A veces, identificar la causa puede ayudarnos a encontrar formas de conectar con otros.",
            
            "Entiendo lo que es sentirse solo/a y quiero que sepas que me importa cómo te sientes. ¿Has pensado en actividades o grupos donde podrías conocer personas con intereses similares a los tuyos?",
            
            "Lamento que te sientas solo/a. Es una emoción difícil, pero también puede ser una oportunidad para reflexionar sobre qué tipo de conexiones son importantes para ti. ¿Hay personas con las que te gustaría reconectar?",
            
            "La soledad puede ser muy dolorosa. Estoy aquí para escucharte y apoyarte. ¿Te gustaría hablar sobre estrategias para construir conexiones significativas con otras personas?"
        ]
        return random.choice(loneliness_responses)
        
    def _generate_response(self, prediction, confidence, bullying_type):
        # Get the risk level from the analysis
        risk_level = self._get_risk_level(prediction, confidence, bullying_type)
        
        # Emergency resources (only show for high risk situations)
        emergency_resources = (
            "\n\n📞 Recursos de ayuda inmediata:\n"
            "- Línea de ayuda contra el bullying: 123-456-789 (24/7, anónimo y gratuito)\n"
            "- Línea de emergencia: 911\n"
            "- Línea de la vida: 01 800 911 2000 (Atención psicológica)\n"
            "- Chat de ayuda: www.chatayuda.org.mx"
        )
        
        # Support resources (show for medium risk situations)
        support_resources = (
            "\n\n🌟 Recursos de apoyo:\n"
            "- Línea de ayuda contra el bullying: 123-456-789\n"
            "- Sitio web con información: www.noalbullying.org.mx"
        )
        
        # Determine which resources to show based on risk level
        resources = ""
        if risk_level == "high":
            resources = emergency_resources
        elif risk_level == "medium":
            resources = support_resources
        
        # If not bullying, provide a general supportive response
        if prediction == 0 or confidence < 0.5:
            general_responses = [
                "Es un placer poder charlar contigo. Las conversaciones nos ayudan a conectar y a entendernos mejor. ¿Hay algo específico en lo que pueda ayudarte o sobre lo que quieras hablar?",
                "Gracias por compartir conmigo. Estoy aquí para escucharte y apoyarte. ¿Hay algo más que te gustaría contarme o alguna pregunta que tengas?",
                "Aprecio que te comuniques conmigo. Estoy aquí para ayudarte en lo que necesites. ¿Hay algún tema específico del que te gustaría hablar hoy?",
                "Me gusta poder conversar contigo. Tu bienestar es importante. ¿Hay algo en particular en lo que pueda ayudarte o sobre lo que quieras hablar más?",
                "Valoro mucho que compartas tus pensamientos conmigo. Estoy aquí para escucharte y apoyarte. ¿Hay algo más que te gustaría explorar en nuestra conversación?"
            ]
            return random.choice(general_responses)
        
        # For bullying situations, provide targeted responses based on type and risk level
        if bullying_type == "verbal":
            if risk_level == "high":
                verbal_responses = [
                    f"Lo que me cuentas sobre esas palabras hirientes es muy serio. Nadie merece ser tratado así. Es importante que sepas que esto no es tu culpa y que mereces respeto. ¿Has hablado con algún adulto de confianza sobre esta situación?\n\n💬 Pasos importantes ahora:\n- Habla con un adulto de confianza hoy mismo\n- No respondas con más insultos\n- Mantente cerca de amigos que te apoyen\n- Recuerda que estas palabras no definen quién eres{resources}",
                    f"Entiendo que estas palabras te están afectando profundamente, y quiero que sepas que esto es acoso verbal y no es aceptable. Tu bienestar es lo más importante ahora. ¿Hay algún adulto en quien confíes para contarle lo que está pasando?\n\n💬 Acciones inmediatas:\n- Busca ayuda con un adulto de confianza\n- Documenta los incidentes (fecha, hora, qué se dijo)\n- Evita situaciones de confrontación\n- Recuerda que mereces ser tratado/a con respeto{resources}"
                ]
            else:  # medium or low risk
                verbal_responses = [
                    f"Las palabras pueden doler profundamente, y entiendo lo difícil que debe ser escuchar cosas hirientes. Quiero que sepas que lo que dicen de ti no define quién eres realmente. Tú vales mucho más que esas palabras. ¿Te gustaría compartir conmigo lo que te están diciendo? A veces expresarlo puede ayudar a procesarlo.\n\n💬 Cómo manejar el acoso verbal:\n- No respondas con más insultos\n- Practica respuestas asertivas\n- Busca apoyo en amigos o adultos de confianza\n- Recuerda que las palabras ofensivas dicen más de quien las dice que de ti{resources}",
                    f"Entiendo que las palabras pueden causar un dolor real, y lamento que estés pasando por esto. Es importante que sepas que esos comentarios no reflejan quién eres tú. ¿Has podido hablar con alguien de confianza sobre esta situación?\n\n💬 Estrategias frente al acoso verbal:\n- Mantén la calma y no respondas de la misma manera\n- Aléjate de la situación si es posible\n- Documenta los incidentes (fecha, hora, qué se dijo)\n- Habla con un adulto de confianza sobre lo que está pasando{resources}"
                ]
            return random.choice(verbal_responses)
            
        elif bullying_type == "físico":
            # Physical bullying is always treated as high risk
            physical_responses = [
                f"Lo que me cuentas es muy serio y quiero que sepas que no está bien que alguien te lastime físicamente. Tu seguridad es lo más importante. Es fundamental que hables con un adulto de confianza sobre esto lo antes posible. ¿Hay algún adulto con quien te sientas seguro/a para hablar?\n\n🛡️ Ante el acoso físico:\n- Tu seguridad es prioritaria\n- Aléjate de situaciones peligrosas\n- Habla inmediatamente con un adulto de confianza\n- Recuerda que tienes derecho a estar seguro/a{emergency_resources}",
                f"Nadie tiene derecho a lastimarte físicamente y lo que estás viviendo no es tu culpa. Es muy importante que busques ayuda de inmediato con un adulto de confianza como un familiar, profesor o consejero escolar. ¿Hay alguien así en quien puedas confiar?\n\n🛡️ Pasos importantes:\n- Mantente alejado/a de quien te lastima\n- Habla con un adulto de confianza hoy mismo\n- No enfrentes solo/a esta situación\n- Recuerda que mereces respeto y seguridad{emergency_resources}"
            ]
            return random.choice(physical_responses)
            
        elif bullying_type == "social":
            if risk_level == "high":
                social_responses = [
                    f"La exclusión social que estás experimentando parece estar afectándote profundamente, y es comprensible. Esta situación es seria y merece atención. ¿Has podido hablar con algún adulto de confianza sobre cómo te sientes?\n\n🤝 Pasos importantes ahora:\n- Habla con un adulto de confianza como un familiar o consejero escolar\n- Busca grupos o actividades fuera de tu entorno habitual\n- Mantén contacto con las personas que te hacen sentir valorado/a\n- Considera hablar con un profesional sobre estos sentimientos{resources}",
                    f"Entiendo que esta exclusión social te está causando un dolor significativo. Quiero que sepas que no estás solo/a y que esto no es un reflejo de tu valor como persona. ¿Hay algún adulto con quien puedas hablar sobre estos sentimientos?\n\n🤝 Acciones recomendadas:\n- Busca apoyo profesional para manejar estos sentimientos\n- Explora actividades donde puedas conocer personas con intereses similares\n- Habla con un adulto de confianza sobre la situación\n- Practica el autocuidado y la autocompasión diariamente{resources}"
                ]
            else:  # medium or low risk
                social_responses = [
                    f"Ser excluido o ignorado puede ser muy doloroso, y entiendo que te sientas así. Quiero que sepas que no hay nada malo en ti y que mereces amistades que te valoren. ¿Te gustaría hablar sobre cómo te has estado sintiendo con esta situación?\n\n🤝 Ante la exclusión social:\n- Busca grupos o actividades donde puedas conocer nuevas personas\n- Cultiva las amistades positivas que ya tienes\n- Recuerda que la calidad de las amistades es más importante que la cantidad\n- Habla con alguien de confianza sobre cómo te sientes{resources}",
                    f"El rechazo social puede ser muy difícil de manejar, y es normal sentirse triste o confundido/a. Quiero que sepas que tu valor no depende de la aceptación de los demás. ¿Has podido identificar algunas personas o grupos donde te sientes más aceptado/a?\n\n🤝 Consejos para manejar la exclusión:\n- Fortalece tu autoestima recordando tus cualidades y logros\n- Busca actividades donde puedas conocer personas con intereses similares\n- Mantén las amistades que te hacen sentir valorado/a\n- Habla sobre tus sentimientos con alguien de confianza{resources}"
                ]
            return random.choice(social_responses)
            
        elif bullying_type == "cibernético":
            if risk_level == "high":
                cyber_responses = [
                    f"El acoso en línea que me describes es muy serio y puede tener un impacto profundo. Es importante que sepas que esto no es tu culpa y que hay medidas que puedes tomar de inmediato. ¿Has hablado con algún adulto de confianza sobre esto?\n\n💻 Acciones inmediatas para el ciberacoso:\n- Guarda capturas de pantalla como evidencia\n- Bloquea a las personas que te están acosando\n- No respondas a los mensajes de acoso\n- Habla con un adulto de confianza hoy mismo\n- Reporta el contenido a la plataforma{resources}",
                    f"Lo que me cuentas sobre este acoso en línea requiere atención inmediata. El ciberacoso es una forma seria de intimidación y no debes enfrentarlo solo/a. ¿Hay algún adulto con quien puedas hablar sobre esto ahora?\n\n💻 Pasos a seguir inmediatamente:\n- No respondas a los mensajes o publicaciones\n- Guarda toda la evidencia (capturas de pantalla)\n- Bloquea a los acosadores en todas las plataformas\n- Habla con tus padres o un adulto de confianza\n- Reporta el acoso a las plataformas sociales{resources}"
                ]
            else:  # medium or low risk
                cyber_responses = [
                    f"El acoso en línea puede ser muy doloroso y difícil de manejar. Quiero que sepas que no estás solo/a y que hay formas de protegerte en el mundo digital. ¿Has tomado alguna medida hasta ahora?\n\n💻 Estrategias contra el ciberacoso:\n- Guarda evidencia de los mensajes o publicaciones\n- Utiliza las herramientas de privacidad de las redes sociales\n- Bloquea a las personas que te molestan\n- Habla con un adulto de confianza sobre lo que está pasando{resources}",
                    f"Entiendo que estos mensajes o publicaciones te están afectando. El mundo digital debería ser un espacio seguro para todos. ¿Has podido hablar con alguien sobre esta situación?\n\n💻 Consejos para protegerte en línea:\n- Revisa y ajusta tu configuración de privacidad\n- No compartas información personal con desconocidos\n- Guarda evidencia del acoso (capturas de pantalla)\n- Bloquea a quienes te envían mensajes negativos\n- Habla con un adulto de confianza{resources}"
                ]
            return random.choice(cyber_responses)
            
        else:  # psicológico u otros
            if risk_level == "high":
                psychological_responses = [
                    f"Lo que me describes suena como una forma seria de intimidación psicológica que está teniendo un impacto significativo en ti. Es importante que busques apoyo profesional para manejar esta situación. ¿Hay algún adulto de confianza con quien puedas hablar hoy mismo?\n\n🧠 Acciones inmediatas:\n- Habla con un adulto de confianza hoy mismo\n- Considera buscar apoyo de un profesional de salud mental\n- Establece límites claros con quienes te están afectando\n- Practica técnicas de autocuidado y manejo del estrés{resources}",
                    f"Esta situación de intimidación psicológica que describes requiere atención inmediata. No debes enfrentar esto solo/a, y hay personas que pueden ayudarte. ¿Puedes hablar con un adulto de confianza sobre lo que estás experimentando?\n\n🧠 Pasos importantes a seguir:\n- Busca apoyo de un adulto de confianza inmediatamente\n- Considera hablar con un consejero escolar o psicólogo\n- Mantén un registro detallado de los incidentes\n- Prioriza tu bienestar emocional y físico{resources}"
                ]
            else:  # medium or low risk
                psychological_responses = [
                    f"La intimidación psicológica puede ser muy dañina y a veces difícil de explicar a los demás. Quiero que sepas que tus sentimientos son válidos y que no estás solo/a en esto. Vamos a buscar juntos la mejor manera de ayudarte. ¿Te sentirías cómodo/a compartiendo más detalles sobre tu experiencia?\n\n🧠 Ante la intimidación psicológica:\n- No minimices lo que sientes\n- Habla con un adulto de confianza\n- Practica técnicas de relajación\n- Recuerda que mereces respeto{resources}",
                    f"Entiendo que estás pasando por una situación difícil. El acoso psicológico puede ser muy sutil pero igualmente dañino. Es importante que sepas que no estás exagerando y que tus sentimientos son completamente válidos. ¿Has podido identificar patrones en esta situación?\n\n🧠 Estrategias de afrontamiento:\n- Lleva un diario de los incidentes\n- Establece límites claros\n- Busca apoyo profesional si es posible\n- Practica el autocuidado diariamente{resources}"
                ]
            return random.choice(psychological_responses)
            
    def _get_risk_level(self, prediction, confidence, bullying_type):
        """Helper method to determine risk level based on prediction, confidence, and bullying type"""
        if prediction != 1:
            return "low"
            
        if bullying_type == "físico":
            return "high"  # Physical bullying is always high risk
        elif confidence > 0.8:
            return "high"
        elif confidence > 0.6 or bullying_type == "cibernético":
            return "medium"
        else:
            return "low"
    
    def get_user_history(self, user_id):
        """
        Get chat history for a specific user
        """
        user_logs = [log for log in self.logs if log['user_id'] == user_id]
        return user_logs
