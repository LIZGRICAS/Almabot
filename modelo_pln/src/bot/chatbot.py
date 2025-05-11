from nlp.nlp_processor import NLPProcessor
from models.logistic_regression_model import BullyingDetectionModel
import json
import os
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
            prediction, probability = self.bullying_model.predict(message)
            confidence = max(probability)
            keywords = self.nlp.extract_keywords(message)
            
            # Mejorar la detección de bullying
            message_lower = message.lower()
            bullying_type = "other"  # Default to 'other' which is a valid ENUM value
            
            # Check for cyberbullying first
            cyberbullying_indicators = ["ciber", "internet", "whatsapp", "redes", "red social", "mensaje", "amenaza", "amenazas", "online", "internet"]
            if any(indicator in message_lower for indicator in cyberbullying_indicators):
                bullying_type = "cibernético"  # Use the correct type from training data
                prediction = 1  # Force bullying detection
            # Check for physical bullying
            elif any(word in message_lower for word in ["empujar", "empujan", "golpear", "golpean", "pegar", "pegan"]):
                bullying_type = "bullying"
                prediction = 1
            # Check for suicide risk
            elif any(term in message_lower for term in ["suicid", "matar", "morir", "acabar con todo"]):
                bullying_type = "suicide"
                prediction = 1
            # Check for abuse
            elif any(term in message_lower for term in ["abuso", "abusar", "acoso", "acosar"]):
                bullying_type = "abuse"
                prediction = 1
            # Check for verbal bullying
            elif any(term in message_lower for term in ["insultar", "insulto", "burlar", "burlan", "ofender"]):
                bullying_type = "bullying"
                prediction = 1
            # Check for social bullying
            elif any(term in message_lower for term in ["excluir", "excluyen", "ignorar", "ignoran", "hablan mal"]):
                bullying_type = "bullying"
                prediction = 1
            
            response = self._generate_response(prediction, confidence, bullying_type)
            
            analysis = {
                'is_bullying': bool(prediction),
                'confidence': float(confidence),
                'bullying_type': bullying_type,
                'keywords': keywords
            }
            self._save_log(user_id, message, response, analysis)
            
            return response
            
        except Exception as e:
            print(f"Error al procesar el mensaje: {e}")
            return "Lo siento, ha ocurrido un error al procesar tu mensaje. Por favor, intenta de nuevo."
    
    def _generate_response(self, prediction, confidence, bullying_type):
        if prediction == 1:
            # Base response for all bullying types
            base_response = (
                "Entiendo que estás pasando por una situación muy difícil. "
                "Es importante que sepas que no estás solo/a y que hay personas que pueden ayudarte. "
                "¿Te gustaría hablar más sobre lo que está pasando?"
            ) if confidence > 0.8 else (
                "Parece que estás pasando por una situación complicada. "
                "¿Te gustaría contarme más sobre lo que está sucediendo?"
            )
            
            # Emergency resources
            emergency_resources = (
                "\n\nRecursos de emergencia:"
                "\n- Línea de ayuda contra el bullying: 123-456-789"
                "\n- Línea de emergencia: 911"
                "\n- Centro de apoyo estudiantil"
            )
            
            # Specific recommendations by bullying type
            recommendations = {
                "físico": (
                    "\n\nRecomendaciones específicas para acoso físico:"
                    "\n- Busca ayuda inmediatamente de un adulto de confianza"
                    "\n- Documenta cualquier lesión o incidente con fechas y detalles"
                    "\n- No te enfrentes solo/a a la situación"
                    f"{emergency_resources}"
                ),
                "verbal": (
                    "\n\nRecomendaciones específicas para acoso verbal:"
                    "\n- No respondas a los insultos o provocaciones"
                    "\n- Guarda evidencia de los mensajes o comentarios"
                    "\n- Habla con un consejero escolar o profesor de confianza"
                    f"{emergency_resources}"
                ),
                "cibernético": (
                    "\n\nRecomendaciones específicas para ciberacoso:"
                    "\n- No respondas a los mensajes de acoso"
                    "\n- Guarda capturas de pantalla como evidencia"
                    "\n- Reporta el perfil o contenido en la plataforma"
                    "\n- Habla con un adulto de confianza sobre la situación"
                    f"{emergency_resources}"
                ),
                "social": (
                    "\n\nRecomendaciones específicas para acoso social:"
                    "\n- Busca apoyo en otros grupos o actividades"
                    "\n- Habla con tus padres o profesores"
                    "\n- Recuerda que tienes derecho a ser respetado/a"
                    f"{emergency_resources}"
                ),
                "other": (
                    "\n\nRecomendaciones generales:"
                    "\n- No estás solo/a en esto"
                    "\n- Habla con un adulto de confianza"
                    "\n- Recuerda que mereces respeto y apoyo"
                    f"{emergency_resources}"
                )
            }
            
            response = base_response + recommendations.get(bullying_type, "")
            response += "\n\nRecursos de emergencia:\n- Línea de ayuda contra el bullying: 123-456-789\n- Psicólogo escolar disponible en horario de clases\n- Centro de apoyo estudiantil"
        else:
            response = (
                "Me alegro de que me cuentes cómo te sientes. "
                "¿Hay algo más en lo que pueda ayudarte?"
            )
        
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