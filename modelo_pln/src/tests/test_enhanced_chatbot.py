"""
Test script for the enhanced chatbot with Ollama integration.
This script tests the bullying detection and LLM response generation.
"""

import os
import sys
import json
from datetime import datetime

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modelo_pln.src.bot.enhanced_chatbot import EnhancedChatbot

def test_enhanced_chatbot():
    """Test the enhanced chatbot with various scenarios"""
    print("\n=== PRUEBA DEL CHATBOT MEJORADO CON INTEGRACIÓN DE OLLAMA ===\n")
    
    # Initialize the enhanced chatbot
    chatbot = EnhancedChatbot(model_name="mistral:7b-instruct")
    
    # Test cases organized by category
    test_cases = {
        "Bullying Físico": [
            "Me empujan constantemente en el pasillo",
            "Me golpean cuando nadie está mirando",
            "Me quitan mis cosas y las rompen"
        ],
        "Bullying Verbal": [
            "Me insultan por mi apariencia",
            "Se burlan de mi forma de hablar",
            "Me ponen apodos ofensivos"
        ],
        "Bullying Social": [
            "Me excluyen de todas las actividades grupales",
            "Nadie quiere sentarse conmigo en el almuerzo",
            "Difunden rumores falsos sobre mí"
        ],
        "Bullying Cibernético": [
            "Me envían mensajes amenazantes por WhatsApp",
            "Publican fotos mías sin mi permiso para burlarse",
            "Me acosan en redes sociales con comentarios ofensivos"
        ],
        "No Bullying": [
            "Me siento feliz en la escuela",
            "Mis amigos me ayudan con las tareas",
            "La maestra me felicitó por mi trabajo"
        ],
        "Conversación": [
            "Hola, ¿cómo estás?",
            "¿Puedes ayudarme con un problema?",
            "Gracias por tu ayuda"
        ]
    }
    
    # Test user ID
    user_id = "test_user_" + datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Test each category
    for category, messages in test_cases.items():
        print(f"\n{category}:")
        print("-" * 70)
        
        for message in messages:
            print(f"Mensaje del usuario: {message}")
            
            # Process message
            response, analysis = chatbot.process_message(user_id, message)
            
            # Print results
            print(f"¿Bullying? {'SÍ' if analysis['is_bullying'] else 'NO'} (Confianza: {analysis['confidence']:.2f})")
            if analysis['is_bullying']:
                print(f"Tipo: {analysis['bullying_type']}")
            
            print(f"\nRespuesta del chatbot:\n{response}\n")
            print("-" * 70)
    
    # Test conversation flow
    print("\n=== PRUEBA DE FLUJO DE CONVERSACIÓN ===\n")
    conversation_user_id = "conversation_test_user"
    
    conversation = [
        "Hola, me llamo Juan y tengo 10 años",
        "En la escuela hay unos niños que siempre me molestan",
        "Me empujan y me quitan mi almuerzo",
        "No sé qué hacer, tengo miedo de ir a la escuela",
        "¿Debería decirle a mi mamá?",
        "Gracias por tu ayuda"
    ]
    
    for message in conversation:
        print(f"Usuario: {message}")
        response, _ = chatbot.process_message(conversation_user_id, message)
        print(f"AlmaBot: {response}\n")
    
    # Print conversation summary
    print("\n=== RESUMEN DE LA CONVERSACIÓN ===\n")
    summary = chatbot.get_conversation_summary(conversation_user_id)
    print(summary)
    
    print("\nPrueba completada.")

if __name__ == "__main__":
    test_enhanced_chatbot()
