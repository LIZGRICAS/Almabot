import os
import json
from datetime import datetime
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from dotenv import load_dotenv
from logistic_regression_model import BullyingDetectionModel
from nlp_processor import NLPProcessor
import random
from classes import UsuarioAnonimo, Mensaje, SesionTerapia

# Cargar variables de entorno
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

# Inicializar modelo y NLP
bullying_model = BullyingDetectionModel()
bullying_model.load_model()
nlp = NLPProcessor()

# Log de interacciones
LOG_FILE = 'chat_logs.jsonl'

EMPATHY_RESPONSES = [
    "Lamento que te sientas así 😔. Recuerda que no estás solo/a.",
    "Siento mucho que estés pasando por esto. ¿Quieres contarme más?",
    "Entiendo que no es fácil sentirse así. Estoy aquí para escucharte.",
    "Gracias por confiar en mí. ¿Te gustaría hablar con alguien de confianza?"
]

MAIN_MENU = [
    ["🗣 Hablar sobre cómo me siento"],
    ["💡 Consejos y recursos"],
    ["🚨 Emergencia"]
]

FEELINGS_MENU = [
    ["😊 Estoy feliz", "😢 Estoy triste"],
    ["😠 Estoy enojado", "😨 Tengo miedo"],
    ["⬅️ Volver"]
]

RESOURCES_MENU = [
    ["¿Qué hacer si sufro bullying?"],
    ["¿Cómo ayudar a un amigo?"],
    ["⬅️ Volver al menú principal"]
]

CHILD_FRIENDLY_MENU = [
    ["😊 Contar cómo me siento"],
    ["💭 Pedir consejos"],
    ["🆘 Necesito ayuda urgente"]
]

EMOJI_RESPONSES = {
    "triste": "😢",
    "feliz": "",
    "enojado": "😠",
    "asustado": "😨",
    "solo": "😔"
}

def save_log(user_id, message, response, analysis):
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'message': message,
        'response': response,
        'analysis': analysis
    }
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja el comando /start"""
    response = "¡Hola! 😊 ¿Cómo te sientes hoy? Puedes contarme lo que quieras."
    
    # Crear botones inline
    keyboard = [
        [
            InlineKeyboardButton("😊 Contar cómo me siento", callback_data="feelings"),
            InlineKeyboardButton("💭 Pedir consejos", callback_data="advice")
        ],
        [
            InlineKeyboardButton("🆘 Necesito ayuda urgente", callback_data="emergency")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        text=response,
        reply_markup=reply_markup
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja el comando /help"""
    help_message = (
        "Estoy aquí para ayudarte. Puedes:\n\n"
        "1. Contarme lo que te preocupa\n"
        "2. Pedirme consejos sobre cómo manejar situaciones difíciles\n"
        "3. Obtener información sobre recursos de ayuda\n\n"
        "Comandos disponibles:\n"
        "/start - Iniciar una nueva conversación\n"
        "/help - Mostrar este mensaje de ayuda\n"
        "/emergencia - Mostrar recursos de emergencia"
    )
    await update.message.reply_text(help_message)

async def emergency(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja el comando /emergencia"""
    emergency_message = (
        "🚨 RECURSOS DE EMERGENCIA 🚨\n\n"
        "Si estás en una situación difícil:\n\n"
        "1. Habla con un adulto de confianza:\n"
        "   - Tus padres o familiares\n"
        "   - Tu profesor o consejero escolar\n"
        "   - Un adulto en quien confíes\n\n"
        "2. Líneas de ayuda:\n"
        "   📞 Línea contra el bullying: 123-456-789\n"
        "   📞 Psicólogo escolar: 123-456-789\n\n"
        "3. Recuerda:\n"
        "   - No estás solo\n"
        "   - Es importante pedir ayuda\n"
        "   - Hay personas que pueden ayudarte\n\n"
        "¿Quieres que te ayude a encontrar a alguien con quien hablar? 🤝"
    )
    
    keyboard = [
        ["📞 Llamar a un adulto de confianza"],
        ["💬 Hablar con el consejero escolar"],
        ["⬅️ Volver al menú principal"]
    ]
    
    await update.message.reply_text(
        emergency_message,
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    )

async def handle_emotional_state(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message.text.lower()
    user_id = str(update.effective_user.id)
    
    # Simple emotion detection
    emotions = {
        "triste": ["triste", "solo", "deprimido", "mal"],
        "feliz": ["feliz", "contento", "alegre", "bien"],
        "enojado": ["enojado", "molesto", "irritado"],
        "asustado": ["asustado", "miedo", "temor", "nervioso"]
    }
    
    detected_emotion = None
    for emotion, keywords in emotions.items():
        if any(keyword in message for keyword in keywords):
            detected_emotion = emotion
            break
    
    if detected_emotion:
        emoji = EMOJI_RESPONSES.get(detected_emotion, "")
        if detected_emotion == "feliz":
            response = "¡Me alegra mucho que te sientas bien! 😃 Recuerda que siempre puedes contarme lo que quieras."
        elif detected_emotion == "triste":
            response = "Siento que estés triste 😢. ¿Quieres contarme qué pasó? Estoy aquí para escucharte."
        else:
            response = f"Entiendo que te sientes {detected_emotion} {emoji}. ¿Quieres contarme más sobre lo que te hace sentir así?"
    else:
        response = "¿Cómo te sientes hoy? Estoy aquí para escucharte 😊"
    
    await update.message.reply_text(
        response,
        reply_markup=ReplyKeyboardMarkup(CHILD_FRIENDLY_MENU, resize_keyboard=True)
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    message = update.message.text.strip().lower()
    
    # Despedida
    if message in ["adiós", "gracias", "bye", "hasta luego"]:
        await update.message.reply_text("¡Hasta pronto! 🌈 Si necesitas hablar, aquí estaré.")
        return
    
    # Ayuda afirmativa
    if message in ["sí", "si", "claro", "por favor", "ayúdame"]:
        if context.user_data.get("last_help") == True and message in ["sí", "si"]:
            await update.message.reply_text("Recuerda que puedes hablar con un adulto de confianza o llamar a la línea de ayuda 📞 123-456-789.")
            return
        context.user_data["last_help"] = message in ["sí", "si"]
        await update.message.reply_text(
            "¡Gracias por confiar en mí! 💛\n¿Quieres que te muestre los números de ayuda o recursos disponibles?",
            reply_markup=ReplyKeyboardMarkup([["📞 Recursos de ayuda"], ["⬅️ Volver al menú principal"]], resize_keyboard=True)
        )
        return
    
    # Emociones guiadas
    if message in ["😊 estoy feliz", "😢 estoy triste", "😠 estoy enojado", "😨 tengo miedo"]:
        emociones = {
            "😊 estoy feliz": "¡Me alegra mucho que te sientas bien! 😃",
            "😢 estoy triste": "Siento que estés triste 😢. ¿Quieres contarme qué pasó?",
            "😠 estoy enojado": "A veces nos enojamos, es normal. ¿Quieres contarme por qué?",
            "😨 tengo miedo": "Si tienes miedo, puedes contarme o hablar con un adulto de confianza. Estoy aquí para escucharte."
        }
        await update.message.reply_text(emociones[message])
        return
    
    # Check for emergency keywords
    emergency_keywords = ["ayuda", "emergencia", "peligro", "miedo", "urgencia"]
    if any(keyword in message for keyword in emergency_keywords):
        await emergency(update, context)
        return
    
    # Process message with existing bullying detection
    prediction, probability = bullying_model.predict(message, threshold=0.3)
    
    # Add emotional support
    if prediction == 1:
        response = (
            "Entiendo que estás pasando por una situación difícil 😔\n"
            "Recuerda que no estás solo y que hay personas que pueden ayudarte.\n"
            "¿Te gustaría que te ayude a encontrar a alguien con quien hablar? 🤝"
        )
    else:
        # Handle emotional state
        await handle_emotional_state(update, context)
        return
    
    # Save log with enhanced analysis
    analysis = {
        'is_bullying': bool(prediction),
        'confidence': float(max(probability)),
        'keywords': nlp.extract_keywords(message),
        'timestamp': datetime.now().isoformat()
    }
    
    save_log(user_id, message, response, analysis)
    
    await update.message.reply_text(
        response,
        reply_markup=ReplyKeyboardMarkup(CHILD_FRIENDLY_MENU, resize_keyboard=True)
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja los callbacks de los botones"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "feelings":
        await query.message.reply_text(
            "¿Cómo te sientes? Elige una opción o cuéntame con tus palabras.",
            reply_markup=ReplyKeyboardMarkup(FEELINGS_MENU, resize_keyboard=True)
        )
    elif query.data == "advice":
        await query.message.reply_text(
            "¿Sobre qué tema necesitas consejos?",
            reply_markup=ReplyKeyboardMarkup(RESOURCES_MENU, resize_keyboard=True)
        )
    elif query.data == "emergency":
        await emergency(update, context)

class ChatBotTerapia:
    def __init__(self):
        self.modelo = BullyingDetectionModel()
        self.modelo.load_model()
        self.nlp = NLPProcessor()
        self.sesiones_activas = {}
        
    def iniciar_sesion(self, user_id):
        usuario = UsuarioAnonimo()
        usuario.id = user_id
        sesion = SesionTerapia(usuario)
        self.sesiones_activas[user_id] = sesion
        return sesion
        
    def procesar_mensaje(self, user_id, texto):
        if user_id not in self.sesiones_activas:
            self.iniciar_sesion(user_id)
            
        sesion = self.sesiones_activas[user_id]
        mensaje = Mensaje(texto, "usuario")
        
        # Análisis de bullying
        prediction, probability = self.modelo.predict(texto, threshold=0.3)
        mensaje.bullying_detectado = bool(prediction)
        
        # Análisis de emociones
        emocion = self.detectar_emocion(texto)
        mensaje.emocion_detectada = emocion
        
        sesion.agregar_mensaje(mensaje)
        return self.generar_respuesta(mensaje)

def main():
    """Función principal del bot"""
    try:
        # Crear la aplicación
        application = Application.builder().token(TELEGRAM_TOKEN).build()
        
        # Agregar manejadores
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("emergencia", emergency))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_handler(CallbackQueryHandler(button_callback))
        
        # Iniciar el bot
        print("Iniciando bot...")
        application.run_polling()
        
    except Exception as e:
        print(f"Error al iniciar el bot: {e}")

if __name__ == '__main__':
    main() 