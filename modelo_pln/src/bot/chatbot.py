from src.nlp.nlp_processor import NLPProcessor
from src.models.logistic_regression_model import BullyingDetectionModel
import json
import os
import random
from datetime import datetime

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
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    self.logs = [json.loads(line) for line in f if line.strip()]
            else:
                self.logs = []
        except Exception as e:
            print(f"Error al cargar logs: {e}")
            self.logs = []
    
    def _save_log(self, user_id, message, response, analysis):
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'message': message,
                'response': response,
                'analysis': analysis
            }
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            self.logs.append(log_entry)
        except Exception as e:
            print(f"Error al guardar log: {e}")
    
    def process_message(self, user_id, message):
        try:
            # Initialize variables
            message_lower = message.lower()
            prediction = 0
            confidence = 0.0
            bullying_type = "ninguno"
            
            # Check for positive sentiment words first - expanded list
            positive_words = ["feliz", "contento", "alegre", "bien", "genial", "excelente", "maravilloso", 
                             "fantástico", "divertido", "agradable", "bueno", "positivo", "encantado",
                             "felicidad", "sonrisa", "sonreír", "reír", "risa", "disfrutar", "disfrutando",
                             "amo", "me gusta", "me encanta", "gracias", "agradecido", "agradecida",
                             "satisfecho", "satisfecha", "orgulloso", "orgullosa", "tranquilo", "tranquila",
                             "relajado", "relajada", "cómodo", "cómoda", "seguro", "segura"]
            
            # Positive phrases that indicate good emotional state
            positive_phrases = ["me siento bien", "estoy feliz", "me va bien", "todo está bien", 
                              "me gusta la escuela", "tengo amigos", "me divierto", "me ayudan", 
                              "me apoyan", "me quieren", "me respetan", "me valoran",
                              "la paso bien", "me tratan bien", "me siento seguro", "me siento segura"]
            
            # Check for positive sentiment in message
            has_positive_word = any(word in message_lower for word in positive_words)
            has_positive_phrase = any(phrase in message_lower for phrase in positive_phrases)
            
            # Improved positive sentiment detection with context analysis
            if (has_positive_word or has_positive_phrase):
                # Check if there are any negative words that could indicate sarcasm or mixed feelings
                negative_words = ["pero", "aunque", "sin embargo", "triste", "mal", "preocupado", "preocupada",
                                "miedo", "ansiedad", "ansioso", "ansiosa", "problema", "difícil"]
                
                # If the message is short and purely positive without negative context, it's likely not bullying
                if not any(word in message_lower for word in negative_words) and len(message_lower.split()) < 10:
                    model_prediction = 0
                    model_confidence = 0.95
                    probability = [0.95, 0.05]
                else:
                    # Run model prediction for mixed sentiment messages
                    model_prediction, probability = self.bullying_model.predict(message)
                    model_confidence = max(probability) if probability else 0.0
            else:
                # Run model prediction for non-positive messages
                model_prediction, probability = self.bullying_model.predict(message)
                model_confidence = max(probability) if probability else 0.0
            
            # Extract keywords for analysis
            keywords = self.nlp.extract_keywords(message)
            
            # Check for context that might indicate a false positive
            context_indicators = {
                "película": 0.7, "serie": 0.7, "cuento": 0.7, "historia": 0.6, "libro": 0.6, 
                "videojuego": 0.8, "juego": 0.7, "ficción": 0.8, "personaje": 0.7,
                "tarea": 0.6, "proyecto": 0.6, "trabajo escolar": 0.7, "investigación": 0.7,
                "pregunta": 0.5, "hipotético": 0.8, "si alguien": 0.6
            }
            
            # Check if the message is about a hypothetical situation rather than a real experience
            context_words = [word for word in context_indicators.keys() if word in message_lower]
            context_confidence_reduction = max([context_indicators[word] for word in context_words]) if context_words else 0
            
            # Enhanced bullying type detection with more comprehensive keyword lists
            # Physical bullying keywords
            physical_keywords = ["empujar", "empujan", "golpear", "golpean", "pegar", "pegan", "patear", "patean",
                               "puñetazo", "cachetada", "pellizcar", "pellizcan", "tirar", "tiran", "jalar", "jalan",
                               "escupir", "escupen", "tropezar", "tropiezan", "zancadilla", "zancadillas",
                               "encerrar", "encierran", "atrapar", "atrapan"]
            
            # Cyberbullying keywords
            cyber_keywords = ["mensaje", "whatsapp", "facebook", "instagram", "amenaza", "tiktok", "snapchat",
                            "twitter", "correo", "email", "foto", "video", "publicar", "publicación", "compartir",
                            "difundir", "difunden", "viral", "stalkear", "stalkean", "hackear", "hackean"]
            
            # Verbal bullying keywords
            verbal_keywords = ["insultar", "insulto", "gritar", "burlar", "burla", "apodo", "apodos", "mote", "motes",
                             "humillar", "humillan", "ridiculizar", "ridiculizan", "ofender", "ofenden", "ofensa",
                             "grosería", "groserías", "maldecir", "maldicen", "burlarse", "se burlan"]
            
            # Social bullying keywords
            social_keywords = ["excluir", "excluyen", "ignorar", "ignoran", "rechazar", "rechazan", "dejar fuera",
                             "dejan fuera", "no invitar", "no invitan", "no incluir", "no incluyen", "aislar", "aíslan",
                             "solo", "sola", "soledad", "nadie me habla", "nadie juega conmigo", "no tienen en cuenta"]
            
            # Psychological bullying keywords
            psychological_keywords = ["amenazar", "amenazan", "intimidar", "intimidan", "manipular", "manipulan",
                                    "chantajear", "chantajean", "controlar", "controlan", "presionar", "presionan",
                                    "miedo", "temor", "ansiedad", "angustia", "pánico", "terror"]
            
            # Apply context confidence reduction to model confidence
            adjusted_confidence = model_confidence * (1 - context_confidence_reduction)
            
            # Check for physical bullying keywords
            if any(word in message_lower for word in physical_keywords):
                bullying_type = "físico"
                prediction = 1 if adjusted_confidence > 0.3 else model_prediction
                confidence = max(0.9, adjusted_confidence) if prediction == 1 else adjusted_confidence
            # Check for cyberbullying keywords
            elif any(word in message_lower for word in cyber_keywords):
                bullying_type = "cibernético"
                prediction = 1 if adjusted_confidence > 0.3 else model_prediction
                confidence = max(0.85, adjusted_confidence) if prediction == 1 else adjusted_confidence
            # Check for verbal bullying keywords
            elif any(word in message_lower for word in verbal_keywords):
                bullying_type = "verbal"
                prediction = 1 if adjusted_confidence > 0.3 else model_prediction
                confidence = max(0.85, adjusted_confidence) if prediction == 1 else adjusted_confidence
            # Check for social bullying keywords
            elif any(word in message_lower for word in social_keywords):
                bullying_type = "social"
                prediction = 1 if adjusted_confidence > 0.3 else model_prediction
                confidence = max(0.85, adjusted_confidence) if prediction == 1 else adjusted_confidence
            # Check for psychological bullying keywords
            elif any(word in message_lower for word in psychological_keywords):
                bullying_type = "psicológico"
                prediction = 1 if adjusted_confidence > 0.3 else model_prediction
                confidence = max(0.85, adjusted_confidence) if prediction == 1 else adjusted_confidence
            # If model detected bullying but we haven't categorized it yet
            elif model_prediction == 1:
                prediction = 1
                confidence = adjusted_confidence
                
                # Determine bullying type based on keywords if not already set
                if "verbal" in message_lower:
                    bullying_type = "verbal"
                elif "social" in message_lower:
                    bullying_type = "social"
                elif "cibernético" in message_lower or "internet" in message_lower:
                    bullying_type = "cibernético"
                elif "físico" in message_lower:
                    bullying_type = "físico"
                else:
                    bullying_type = "psicológico"  # Default type if bullying detected but type unclear
            
            # Generate appropriate response based on detection
            response = self._generate_response(prediction, confidence, bullying_type)
            
            # More detailed emotion analysis based on message content
            emotion_type = 'neutral'
            emotion_intensity = 0.0
            sentiment_score = 0.0
            risk_level = 'low'
            
            # Emotion detection based on keywords
            fear_words = ["miedo", "temor", "asustado", "asustada", "terror", "pánico", "aterrado", "aterrada"]
            sadness_words = ["triste", "tristeza", "deprimido", "deprimida", "depresión", "llorar", "lloro", "llanto"]
            anger_words = ["enojado", "enojada", "enfadado", "enfadada", "rabia", "ira", "furia", "molesto", "molesta"]
            anxiety_words = ["ansioso", "ansiosa", "ansiedad", "nervioso", "nerviosa", "preocupado", "preocupada"]
            shame_words = ["vergüenza", "avergonzado", "avergonzada", "humillado", "humillada", "ridículo", "ridícula"]
            hopeless_words = ["desesperado", "desesperada", "sin esperanza", "no puedo más", "no aguanto", "rendirme"]
            
            # Determine emotion type and intensity
            if any(word in message_lower for word in fear_words):
                emotion_type = 'fear'
                emotion_intensity = 0.8
                sentiment_score = -0.7
            elif any(word in message_lower for word in sadness_words):
                emotion_type = 'sadness'
                emotion_intensity = 0.7
                sentiment_score = -0.6
            elif any(word in message_lower for word in anger_words):
                emotion_type = 'anger'
                emotion_intensity = 0.6
                sentiment_score = -0.5
            elif any(word in message_lower for word in anxiety_words):
                emotion_type = 'anxiety'
                emotion_intensity = 0.7
                sentiment_score = -0.6
            elif any(word in message_lower for word in shame_words):
                emotion_type = 'shame'
                emotion_intensity = 0.8
                sentiment_score = -0.7
            elif any(word in message_lower for word in hopeless_words):
                emotion_type = 'hopelessness'
                emotion_intensity = 0.9
                sentiment_score = -0.8
                risk_level = 'high'
            elif prediction == 1:
                emotion_type = 'distress'
                emotion_intensity = 0.7
                sentiment_score = -0.6
                risk_level = 'medium'
            elif has_positive_word or has_positive_phrase:
                emotion_type = 'happiness'
                emotion_intensity = 0.8
                sentiment_score = 0.7
            
            # Determine risk level based on emotion and prediction
            if prediction == 1:
                if emotion_type in ['fear', 'hopelessness'] and emotion_intensity > 0.7:
                    risk_level = 'high'
                elif emotion_intensity > 0.5:
                    risk_level = 'medium'
                else:
                    risk_level = 'low'
            
            # Create enhanced analysis object with additional fields for API compatibility
            analysis = {
                'is_bullying': bool(prediction),
                'confidence': float(confidence),
                'bullying_type': bullying_type,
                'keywords': keywords,
                'sentiment': 'negative' if sentiment_score < 0 else 'positive' if sentiment_score > 0 else 'neutral',
                'sentiment_score': sentiment_score,
                'emotion': {
                    'type': emotion_type,
                    'intensity': emotion_intensity
                },
                'risk': {
                    'level': risk_level,
                    'type': 'bullying' if prediction == 1 else 'emotional_distress' if emotion_intensity > 0.6 else 'none'
                },
                'context': {
                    'hypothetical': bool(context_words),
                    'context_words': context_words
                }
            }
            
            # Save log and return response
            self._save_log(user_id, message, response, analysis)
            return response, analysis  # Return both response and analysis
        except Exception as e:
            print(f"Error al procesar el mensaje: {e}")
            return "Lo siento, ha ocurrido un error al procesar tu mensaje. Por favor, intenta de nuevo.", {}


    def _generate_response(self, prediction, confidence, bullying_type):
        # Emergency resources (only show for high risk situations)
        emergency_resources = (
            "\n\n📞 Recursos de ayuda:"
            "\n- Línea de ayuda contra el bullying: 123-456-789 (24/7, anónimo y gratuito)"
            "\n- Línea de emergencia: 911"
            "\n- Línea de la vida: 01 800 911 2000 (Atención psicológica)"
            "\n- Chat de ayuda: www.chatayuda.org.mx"
        )
        
        # Check for positive messages (when prediction is 0 with high confidence)
        if prediction == 0 and confidence > 0.8:
            # Expanded positive response options with more personalized and engaging responses
            positive_responses = [
                "¡Me alegra mucho escuchar que estás feliz! Compartir momentos positivos es muy importante. ¿Hay algo específico que te haya hecho sentir así?",
                "Es genial saber que te sientes bien. Esos momentos son los que debemos atesorar. ¿Quieres contarme más sobre tu día?",
                "¡Qué bueno! Es importante reconocer y disfrutar esos momentos positivos. ¿Qué otras cosas buenas te han pasado últimamente?",
                "Me encanta escuchar eso. Compartir las cosas buenas nos ayuda a sentirnos aún mejor. ¿Hay alguien con quien hayas compartido esta felicidad?",
                "Eso suena maravilloso. Es importante celebrar las cosas positivas, por pequeñas que sean. ¿Qué crees que ha contribuido a que te sientas así?",
                "Qué alegría escuchar eso. Cuando nos sentimos bien, es un buen momento para reflexionar sobre las cosas que valoramos. ¿Qué es lo que más aprecias en este momento?"
            ]
            return random.choice(positive_responses)
        
        elif prediction == 1:
            # Enhanced base responses by bullying type with more empathy and personalization
            base_responses = {
                "cibernético": (
                    "Lamento mucho que estés pasando por esto. El ciberacoso es algo serio y quiero que sepas que no estás solo/a en esta situación. "
                    "Lo que estás experimentando es válido y mereces apoyo. Vamos a pensar juntos en cómo enfrentar esta situación. "
                    "¿Te sentirías cómodo/a contándome un poco más sobre lo que está ocurriendo?"
                ),
                "físico": (
                    "Entiendo que estás en una situación muy difícil y quiero que sepas que tu seguridad es lo más importante. "
                    "Nadie tiene derecho a lastimarte físicamente bajo ninguna circunstancia. "
                    "Es realmente importante que busques ayuda de inmediato. ¿Hay algún adulto de confianza con quien te sientas seguro/a hablando sobre esto?"
                ),
                "verbal": (
                    "Las palabras pueden doler profundamente, y entiendo lo difícil que debe ser escuchar cosas hirientes. "
                    "Quiero que sepas que lo que dicen de ti no define quién eres realmente. Tú vales mucho más que esas palabras. "
                    "¿Te gustaría compartir conmigo lo que te están diciendo? A veces expresarlo puede ayudar a procesarlo."
                ),
                "social": (
                    "Sentirse excluido/a o rechazado/a es realmente doloroso, y entiendo que puede hacerte sentir muy solo/a. "
                    "Quiero que sepas que mereces amistades genuinas y personas que te valoren por quien eres. "
                    "¿Te gustaría contarme más sobre lo que está pasando en tu entorno social?"
                ),
                "psicológico": (
                    "La intimidación psicológica puede ser muy dañina y a veces difícil de explicar a los demás. "
                    "Quiero que sepas que tus sentimientos son válidos y que no estás solo/a en esto. "
                    "Vamos a buscar juntos la mejor manera de ayudarte. ¿Te sentirías cómodo/a compartiendo más detalles sobre tu experiencia?"
                ),
                "other": (
                    "Entiendo que estás pasando por un momento difícil, y quiero que sepas que estoy aquí para escucharte sin juzgarte. "
                    "A veces las situaciones complicadas pueden parecer abrumadoras, pero juntos podemos encontrar formas de manejarlas. "
                    "¿Te gustaría contarme más sobre lo que te preocupa?"
                )
            }
            
            # Select base response based on bullying type
            base_response = base_responses.get(bullying_type, base_responses["other"])
            
            # Specific recommendations by bullying type
            recommendations = {
                "cibernético": (
                    "\n\n🔍 Recomendaciones específicas para ciberacoso:"
                    "\n- No respondas a los mensajes de acoso"
                    "\n- Toma capturas de pantalla como evidencia (fecha y hora visibles)"
                    "\n- Reporta el contenido o perfil en la plataforma donde ocurre"
                    "\n- Configura la privacidad de tus redes sociales"
                    "\n- Habla con un adulto de confianza sobre la situación"
                ),
                "físico": (
                    "\n\n🛡️ Recomendaciones para acoso físico:"
                    "\n- Aléjate de situaciones de riesgo"
                    "\n- Busca ayuda de un adulto de confianza inmediatamente"
                    "\n- Documenta cualquier lesión con fotos y fechas"
                    "\n- No te quedes solo/a, busca compañía de amigos o adultos"
                ),
                "verbal": (
                    "\n\n💬 Cómo manejar el acoso verbal:"
                    "\n- No respondas con más insultos"
                    "\n- Practica respuestas asertivas"
                    "\n- Busca apoyo en amigos o adultos de confianza"
                    "\n- Recuerda que las palabras ofensivas dicen más de quien las dice que de ti"
                ),
                "social": (
                    "\n\n🤝 Sobre el acoso social:"
                    "\n- Busca nuevos grupos de interés donde te sientas aceptado/a"
                    "\n- Habla con un consejero escolar sobre la situación"
                    "\n- Recuerda que mereces tener amigos que te valoren"
                    "\n- No cambies quién eres por encajar"
                ),
                "psicológico": (
                    "\n\n🧠 Ante la intimidación psicológica:"
                    "\n- No minimices lo que sientes"
                    "\n- Habla con un adulto de confianza"
                    "\n- Practica técnicas de relajación"
                    "\n- Recuerda que mereces respeto"
                ),
                "other": (
                    "\n\n💡 Algunas sugerencias que podrían ayudarte:"
                    "\n- No guardes lo que sientes, busca a alguien de confianza"
                    "\n- Escribe en un diario para expresar tus emociones"
                    "\n- Recuerda que mereces sentirte seguro/a y respetado/a"
                )
            }
            
            # Get the specific recommendation or fallback to general one
            recommendation = recommendations.get(bullying_type, recommendations["other"])
            
            # Combine base response with specific recommendation and emergency resources
            response = f"{base_response}{recommendation}{emergency_resources}"
        else:
            # Enhanced responses for non-bullying cases that are more conversational and engaging
            responses = [
                "Gracias por compartir conmigo. Me interesa mucho conocer tus experiencias y cómo te sientes. Si en algún momento necesitas ayuda o simplemente quieres conversar sobre algo, estoy aquí para escucharte. ¿Hay algún tema específico del que te gustaría hablar?",
                "Me alegra que me cuentes cómo estás. Poder expresarnos es muy importante para nuestro bienestar. Recuerda que estoy aquí para ti, tanto en los momentos difíciles como en los buenos. ¿Hay algo más que te gustaría compartir conmigo hoy?",
                "Gracias por confiar en mí. Construir confianza es fundamental en cualquier conversación. Si en algún momento tienes alguna preocupación o simplemente quieres charlar, puedes hacerlo con toda libertad. ¿Te gustaría hablar de algo en particular?",
                "Aprecio mucho que te tomes el tiempo para conversar conmigo. Cada intercambio nos ayuda a conocernos mejor. ¿Hay algún tema que te interese o sobre el que tengas curiosidad?",
                "Es un placer poder charlar contigo. Las conversaciones nos ayudan a conectar y a entendernos mejor. ¿Hay algo específico en lo que pueda ayudarte o sobre lo que quieras hablar?"
            ]
            response = random.choice(responses)
        
        return response
    
    def get_user_history(self, user_id):
        try:
            return [log for log in self.logs if log.get('user_id') == user_id]
        except Exception as e:
            print(f"Error al obtener historial: {e}")
            return []

def main():
    chatbot = Chatbot()
    user_id = "user123"
    test_messages = [
        "Me empujan constantemente en el pasillo",
        "Me siento bien en la escuela",
        "Me envían mensajes amenazantes por WhatsApp"
    ]
    
    for message in test_messages:
        print(f"\nUsuario: {message}")
        response = chatbot.process_message(user_id, message)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main() 