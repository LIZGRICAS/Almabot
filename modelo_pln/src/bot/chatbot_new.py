from src.nlp.nlp_processor import NLPProcessor
from src.models.logistic_regression_model import BullyingDetectionModel
import json
import os
import random
import re

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
            'timestamp': self.nlp.get_timestamp()
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
            
            # Check for bullying indicators in emotional expressions
            emotional_bullying_indicators = [
                "me siento triste porque", "me siento mal porque", "estoy triste porque",
                "me molestan", "se burlan", "me insultan", "me hacen sentir", "me excluyen",
                "no me incluyen", "me dejan fuera", "me ignoran", "no me hablan",
                "me siento solo", "me siento sola", "nadie quiere", "no quieren"
            ]
            
            # If the message contains emotional bullying indicators, increase the likelihood of bullying detection
            if any(indicator in message_lower for indicator in emotional_bullying_indicators):
                print(f"Detected emotional bullying indicator in: '{message_lower}'")
                # Check for specific bullying contexts
                if "compañeros" in message_lower or "escuela" in message_lower or "colegio" in message_lower or "clase" in message_lower:
                    # This is likely school bullying
                    prediction = 1
                    confidence = 0.85
                    if "molestan" in message_lower or "burlan" in message_lower:
                        bullying_type = "verbal"
                    elif "pegan" in message_lower or "empujan" in message_lower or "golpean" in message_lower:
                        bullying_type = "físico"
                    elif "ignoran" in message_lower or "excluyen" in message_lower or "solo" in message_lower or "sola" in message_lower:
                        bullying_type = "social"
                    else:
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
            sentiment_result = self.nlp.analyze_sentiment(message)
            sentiment = sentiment_result['sentiment']
            sentiment_score = sentiment_result['score']
            
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
            prediction, confidence, bullying_type = self.bullying_model.predict(message)
            
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
            
            # Determine risk level and type
            risk_level = "low"
            risk_type = "none"
            
            if prediction == 1:
                if confidence > 0.8:
                    risk_level = "high"
                    risk_type = "bullying"
                elif confidence > 0.6:
                    risk_level = "medium"
                    risk_type = "bullying"
                else:
                    risk_level = "low"
                    risk_type = "potential_bullying"
            elif sentiment == "negative" and sentiment_score < -0.5:
                risk_level = "low"
                risk_type = "emotional_distress"
            
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
        
    def _generate_response(self, prediction, confidence, bullying_type):
        # Emergency resources (only show for high risk situations)
        emergency_resources = (
            "\n\n📞 Recursos de ayuda:\n"
            "- Línea de ayuda contra el bullying: 123-456-789 (24/7, anónimo y gratuito)\n"
            "- Línea de emergencia: 911\n"
            "- Línea de la vida: 01 800 911 2000 (Atención psicológica)\n"
            "- Chat de ayuda: www.chatayuda.org.mx"
        )
        
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
        
        # For bullying situations, provide targeted responses based on type
        if bullying_type == "verbal":
            verbal_responses = [
                f"Las palabras pueden doler profundamente, y entiendo lo difícil que debe ser escuchar cosas hirientes. Quiero que sepas que lo que dicen de ti no define quién eres realmente. Tú vales mucho más que esas palabras. ¿Te gustaría compartir conmigo lo que te están diciendo? A veces expresarlo puede ayudar a procesarlo.\n\n💬 Cómo manejar el acoso verbal:\n- No respondas con más insultos\n- Practica respuestas asertivas\n- Busca apoyo en amigos o adultos de confianza\n- Recuerda que las palabras ofensivas dicen más de quien las dice que de ti{emergency_resources}",
                f"Entiendo que las palabras pueden causar un dolor real, y lamento que estés pasando por esto. Es importante que sepas que esos comentarios no reflejan quién eres tú. ¿Has podido hablar con alguien de confianza sobre esta situación?\n\n💬 Estrategias frente al acoso verbal:\n- Mantén la calma y no respondas de la misma manera\n- Aléjate de la situación si es posible\n- Documenta los incidentes (fecha, hora, qué se dijo)\n- Habla con un adulto de confianza sobre lo que está pasando{emergency_resources}"
            ]
            return random.choice(verbal_responses)
            
        elif bullying_type == "físico":
            physical_responses = [
                f"Lo que me cuentas es muy serio y quiero que sepas que no está bien que alguien te lastime físicamente. Tu seguridad es lo más importante. Es fundamental que hables con un adulto de confianza sobre esto lo antes posible. ¿Hay algún adulto con quien te sientas seguro/a para hablar?\n\n🛡️ Ante el acoso físico:\n- Tu seguridad es prioritaria\n- Aléjate de situaciones peligrosas\n- Habla inmediatamente con un adulto de confianza\n- Recuerda que tienes derecho a estar seguro/a{emergency_resources}",
                f"Nadie tiene derecho a lastimarte físicamente y lo que estás viviendo no es tu culpa. Es muy importante que busques ayuda de inmediato con un adulto de confianza como un familiar, profesor o consejero escolar. ¿Hay alguien así en quien puedas confiar?\n\n🛡️ Pasos importantes:\n- Mantente alejado/a de quien te lastima\n- Habla con un adulto de confianza hoy mismo\n- No enfrentes solo/a esta situación\n- Recuerda que mereces respeto y seguridad{emergency_resources}"
            ]
            return random.choice(physical_responses)
            
        elif bullying_type == "social":
            social_responses = [
                f"Ser excluido o ignorado puede ser muy doloroso, y entiendo que te sientas así. Quiero que sepas que no hay nada malo en ti y que mereces amistades que te valoren. ¿Te gustaría hablar sobre cómo te has estado sintiendo con esta situación?\n\n🤝 Ante la exclusión social:\n- Busca grupos o actividades donde puedas conocer nuevas personas\n- Cultiva las amistades positivas que ya tienes\n- Recuerda que la calidad de las amistades es más importante que la cantidad\n- Habla con alguien de confianza sobre cómo te sientes{emergency_resources}",
                f"El rechazo social puede ser muy difícil de manejar, y es normal sentirse triste o confundido/a. Quiero que sepas que tu valor no depende de la aceptación de los demás. ¿Has podido identificar algunas personas o grupos donde te sientes más aceptado/a?\n\n🤝 Consejos para manejar la exclusión:\n- Fortalece tu autoestima recordando tus cualidades y logros\n- Busca actividades donde puedas conocer personas con intereses similares\n- Mantén las amistades que te hacen sentir valorado/a\n- Habla sobre tus sentimientos con alguien de confianza{emergency_resources}"
            ]
            return random.choice(social_responses)
            
        else:  # psicológico u otros
            psychological_responses = [
                f"La intimidación psicológica puede ser muy dañina y a veces difícil de explicar a los demás. Quiero que sepas que tus sentimientos son válidos y que no estás solo/a en esto. Vamos a buscar juntos la mejor manera de ayudarte. ¿Te sentirías cómodo/a compartiendo más detalles sobre tu experiencia?\n\n🧠 Ante la intimidación psicológica:\n- No minimices lo que sientes\n- Habla con un adulto de confianza\n- Practica técnicas de relajación\n- Recuerda que mereces respeto{emergency_resources}",
                f"Entiendo que estás pasando por una situación difícil. El acoso psicológico puede ser muy sutil pero igualmente dañino. Es importante que sepas que no estás exagerando y que tus sentimientos son completamente válidos. ¿Has podido identificar patrones en esta situación?\n\n🧠 Estrategias de afrontamiento:\n- Lleva un diario de los incidentes\n- Establece límites claros\n- Busca apoyo profesional si es posible\n- Practica el autocuidado diariamente{emergency_resources}"
            ]
            return random.choice(psychological_responses)
    
    def get_user_history(self, user_id):
        """
        Get chat history for a specific user
        """
        user_logs = [log for log in self.logs if log['user_id'] == user_id]
        return user_logs
