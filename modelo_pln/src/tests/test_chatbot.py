from chatbot import ChatbotTerapia

def test_chatbot():
    # Inicializar el chatbot
    bot = ChatbotTerapia()
    
    # Mensajes de prueba
    test_messages = [
        "Hola, me siento triste porque mis compañeros me molestan en la escuela",
        "Me han estado enviando mensajes ofensivos por internet",
        "No quiero ir a la escuela porque me da miedo",
        "Me siento solo y no tengo amigos",
        "Estoy contento porque hoy nadie me molestó"
    ]
    
    # Procesar cada mensaje
    for message in test_messages:
        print("\nUsuario:", message)
        response = bot.process_message(message)
        print("Chatbot:", response)
        
        # Mostrar análisis
        print("\nAnálisis:")
        print("Contexto del usuario:", bot.get_user_context())
        print("Historial de conversación:", len(bot.get_conversation_history()), "mensajes")
        print("-" * 50)

if __name__ == "__main__":
    test_chatbot() 