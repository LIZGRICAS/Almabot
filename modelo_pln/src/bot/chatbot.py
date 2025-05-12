from nlp.nlp_processor import NLPProcessor
from models.logistic_regression_model import BullyingDetectionModel
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
            
            # Check for positive sentiment words first
            positive_words = ["feliz", "contento", "alegre", "bien", "genial", "excelente", "maravilloso", 
                             "fantástico", "divertido", "agradable", "bueno", "positivo", "encantado"]
            
            # If the message contains positive words and is very short, it's likely not bullying
            if any(word in message_lower for word in positive_words) and len(message_lower.split()) < 5:
                # Skip model prediction for clearly positive messages
                model_prediction = 0
                model_confidence = 0.9
                probability = [0.9, 0.1]
            else:
                # Run model prediction for other messages
                model_prediction, probability = self.bullying_model.predict(message)
                model_confidence = max(probability) if probability else 0.0
            
            # Extract keywords for analysis
            keywords = self.nlp.extract_keywords(message)
            
            # Check for physical bullying keywords
            if any(word in message_lower for word in ["empujar", "empujan", "golpear", "golpean", "pegar", "pegan"]):
                bullying_type = "físico"
                prediction = 1
                confidence = max(0.9, model_confidence)  # Use at least 0.9 confidence for explicit mentions
            # Check for cyberbullying keywords
            elif any(word in message_lower for word in ["mensaje", "whatsapp", "facebook", "instagram", "amenaza"]):
                bullying_type = "cibernético"
                prediction = 1
                confidence = max(0.85, model_confidence)
            # Check for verbal bullying keywords
            elif any(word in message_lower for word in ["insultar", "insulto", "gritar", "burlar", "burla"]):
                bullying_type = "verbal"
                prediction = 1
                confidence = max(0.85, model_confidence)
            # Check for social bullying keywords
            elif any(word in message_lower for word in ["excluir", "excluyen", "ignorar", "ignoran", "rechazar"]):
                bullying_type = "social"
                prediction = 1
                confidence = max(0.85, model_confidence)
            # If model detected bullying but we haven't categorized it yet
            elif model_prediction == 1:
                prediction = 1
                confidence = model_confidence
                
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
            
            # Create analysis object with additional fields for API compatibility
            analysis = {
                'is_bullying': bool(prediction),
                'confidence': float(confidence),
                'bullying_type': bullying_type,
                'keywords': keywords,
                'sentiment': 'negative' if prediction == 1 else 'neutral',
                'sentiment_score': 0.3 if prediction == 1 else 0.0,
                'emotion': {
                    'type': 'fear' if prediction == 1 else 'neutral',
                    'intensity': 0.7 if prediction == 1 else 0.0
                },
                'risk': {
                    'level': 'medium' if prediction == 1 else 'low',
                    'type': 'bullying' if prediction == 1 else 'none'
                }
            }
            
            # Save log and return response
            self._save_log(user_id, message, response, analysis)
            return response, analysis  # Return both response and analysis
        except Exception as e:
            print(f"Error al procesar el mensaje: {e}")
            return "Lo siento, ha ocurrido un error al procesar tu mensaje. Por favor, intenta de nuevo.", {}


    def _generate_response(self, prediction, confidence, bullying_type):
        # Emergency resources (only show once)
        emergency_resources = (
            "\n\n📞 Recursos de ayuda:"
            "\n- Línea de ayuda contra el bullying: 123-456-789 (24/7, anónimo y gratuito)"
            "\n- Línea de emergencia: 911"
            "\n- Línea de la vida: 01 800 911 2000 (Atención psicológica)"
            "\n- Chat de ayuda: www.chatayuda.org.mx"
        )
        
        # Check for positive messages (when prediction is 0 with high confidence)
        if prediction == 0 and confidence > 0.8:
            # Positive response options
            positive_responses = [
                "¡Me alegra mucho escuchar que estás feliz! ¿Hay algo específico que te haya hecho sentir así?",
                "Es genial saber que te sientes bien. ¿Quieres contarme más sobre tu día?",
                "¡Qué bueno! Es importante reconocer y disfrutar esos momentos positivos. ¿Qué más cosas buenas te han pasado?"
            ]
            return random.choice(positive_responses)
        
        elif prediction == 1:
            # Base responses by bullying type
            base_responses = {
                "cibernético": (
                    "Lamento mucho que estés pasando por esto. El ciberacoso es algo serio y no estás solo/a. "
                    "Vamos a ayudarte a enfrentar esta situación. ¿Te gustaría contarme más sobre lo que está pasando?"
                ),
                "físico": (
                    "Entiendo que estás en una situación muy difícil. La seguridad es lo primero. "
                    "Es importante que busques ayuda de inmediato. ¿Hay un adulto de confianza con quien puedas hablar?"
                ),
                "verbal": (
                    "Las palabras pueden doler mucho, pero recuerda que lo que dicen de ti no te define. "
                    "¿Te gustaría hablar sobre lo que te están diciendo?"
                ),
                "social": (
                    "Sentirse excluido/a o rechazado/a es muy doloroso. Mereces rodearte de personas que te valoren. "
                    "¿Quieres contarme más sobre lo que está pasando?"
                ),
                "psicológico": (
                    "La intimidación psicológica puede ser muy dañina. No estás solo/a en esto. "
                    "Vamos a buscar la mejor manera de ayudarte. ¿Te gustaría contarme más?"
                ),
                "other": (
                    "Entiendo que estás pasando por un momento difícil. "
                    "Estoy aquí para escucharte y ayudarte. ¿Quieres contarme más sobre lo que te preocupa?"
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
            # For non-bullying cases
            responses = [
                "Gracias por compartir conmigo. Si en algún momento necesitas ayuda o quieres hablar sobre algo, aquí estoy para ti. ¿Hay algo más en lo que pueda ayudarte?",
                "Me alegra que me cuentes cómo estás. Recuerda que si en algún momento necesitas ayuda o quieres hablar, puedes hacerlo sin problema. ¿Hay algo más en lo que pueda ayudarte?",
                "Gracias por confiar en mí. Si en algún momento necesitas hablar sobre algún problema o inquietud, no dudes en decírmelo. ¿Hay algo más en lo que pueda ayudarte?"
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