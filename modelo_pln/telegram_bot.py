import os
import logging
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from chatbot import ChatbotTerapia

# Configurar logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Cargar variables de entorno
load_dotenv()

# Inicializar el chatbot terapéutico
chatbot = ChatbotTerapia()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando de inicio del bot."""
    keyboard = [
        [InlineKeyboardButton("🤝 Hablar sobre bullying", callback_data='talk_bullying')],
        [InlineKeyboardButton("📊 Estadísticas", callback_data='statistics')],
        [InlineKeyboardButton("🆘 Necesito ayuda", callback_data='need_help')],
        [InlineKeyboardButton("ℹ️ Información", callback_data='info')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_message = (
        "¡Hola! Soy AlmaBot, tu asistente virtual. 👋\n\n"
        "Estoy aquí para escucharte y ayudarte con temas relacionados al bullying "
        "y bienestar emocional. ¿Cómo puedo ayudarte hoy?"
    )
    
    if update.message:
        await update.message.reply_text(welcome_message, reply_markup=reply_markup)
    elif update.callback_query:
        await update.callback_query.message.reply_text(welcome_message, reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manejador de callbacks de botones."""
    query = update.callback_query
    await query.answer()
    
    if query.data == 'talk_bullying':
        message = (
            "El bullying es un problema serio que afecta a muchas personas. "
            "¿Quieres hablar sobre:\n\n"
            "1. Tu experiencia personal\n"
            "2. Cómo ayudar a alguien más\n"
            "3. Estrategias de prevención"
        )
        keyboard = [
            [InlineKeyboardButton("Mi experiencia", callback_data='personal_exp')],
            [InlineKeyboardButton("Ayudar a otros", callback_data='help_others')],
            [InlineKeyboardButton("Prevención", callback_data='prevention')],
            [InlineKeyboardButton("« Volver", callback_data='back_main')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(message, reply_markup=reply_markup)
    
    elif query.data == 'statistics':
        message = (
            "📊 Estadísticas sobre bullying:\n\n"
            "• 1 de cada 3 estudiantes reporta ser víctima de bullying\n"
            "• El 70% de los casos ocurren en la escuela\n"
            "• Solo el 30% de las víctimas busca ayuda\n\n"
            "Recuerda: No estás solo/a. Hay ayuda disponible."
        )
        keyboard = [[InlineKeyboardButton("« Volver", callback_data='back_main')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(message, reply_markup=reply_markup)
    
    elif query.data == 'need_help':
        message = (
            "🆘 Si necesitas ayuda inmediata:\n\n"
            "1. Habla con un adulto de confianza (padre, profesor, consejero)\n"
            "2. Contacta líneas de ayuda:\n"
            "   • Línea de ayuda contra el bullying: XXX-XXX-XXXX\n"
            "   • Emergencias: 911\n\n"
            "3. No te quedes callado/a, buscar ayuda es un acto de valentía."
        )
        keyboard = [[InlineKeyboardButton("« Volver", callback_data='back_main')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(message, reply_markup=reply_markup)
    
    elif query.data == 'info':
        message = (
            "ℹ️ Información importante:\n\n"
            "El bullying puede ser:\n"
            "• Físico (golpes, empujones)\n"
            "• Verbal (insultos, burlas)\n"
            "• Social (exclusión, rumores)\n"
            "• Cibernético (acoso en línea)\n\n"
            "Todos merecemos ser tratados con respeto."
        )
        keyboard = [[InlineKeyboardButton("« Volver", callback_data='back_main')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(message, reply_markup=reply_markup)
    
    elif query.data == 'back_main':
        await start(update, context)
    
    elif query.data in ['personal_exp', 'help_others', 'prevention']:
        responses = {
            'personal_exp': (
                "Lamento que estés pasando por esta situación. Es importante que sepas que:\n\n"
                "1. No es tu culpa\n"
                "2. No estás solo/a\n"
                "3. Hay personas que pueden ayudarte\n\n"
                "¿Te gustaría hablar más sobre esto?"
            ),
            'help_others': (
                "Para ayudar a alguien que sufre bullying:\n\n"
                "1. Escucha sin juzgar\n"
                "2. Muestra apoyo\n"
                "3. Anima a buscar ayuda profesional\n"
                "4. Reporta la situación a las autoridades correspondientes"
            ),
            'prevention': (
                "Estrategias de prevención del bullying:\n\n"
                "1. Fomenta el respeto mutuo\n"
                "2. Reporta incidentes\n"
                "3. Crea un ambiente inclusivo\n"
                "4. Aprende sobre el impacto del bullying"
            )
        }
        message = responses[query.data]
        keyboard = [[InlineKeyboardButton("« Volver", callback_data='talk_bullying')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(message, reply_markup=reply_markup)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manejador de mensajes de texto con análisis de sentimiento y consejos específicos."""
    user_message = update.message.text
    
    # Procesar el mensaje con el chatbot
    response = chatbot.process_message(user_message)
    
    # Enviar la respuesta
    await update.message.reply_text(response)

def main():
    """Función principal del bot."""
    # Crear el bot
    application = Application.builder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()

    # Agregar handlers
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Iniciar el bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 