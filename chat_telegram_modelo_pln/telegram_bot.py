import telebot
from telebot.types import KeyboardButton, ReplyKeyboardMarkup
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os
from dotenv import load_dotenv

class TelegramBot:
    def __init__(self, token):
        self.bot = telebot.TeleBot(token)
        self.model = None
        self.vectorizer = None
        self.training_data = None
        self.setup_handlers()
        
    def setup_handlers(self):
        @self.bot.message_handler(commands=['start'])
        def send_welcome(message):
            markup = self.create_main_menu()
            self.bot.reply_to(message, "¡Hola! Soy AlmaBot, tu asistente virtual. ¿Qué te gustaría hacer?", reply_markup=markup)

        @self.bot.message_handler(func=lambda message: True)
        def handle_message(message):
            if message.text == "📊 Entrenar Modelo":
                self.start_training(message)
            elif message.text == "💬 Chat con AlmaBot":
                self.start_chat(message)
            elif message.text == "📊 Ver Estadísticas":
                self.show_statistics(message)
            elif message.text.lower() in ["hola", "menu", "menú", "ayuda"]:
                markup = self.create_main_menu()
                self.bot.reply_to(message, "¡Hola! ¿Qué te gustaría hacer?", reply_markup=markup)
            elif message.text == "🔙 Volver":
                markup = self.create_main_menu()
                self.bot.reply_to(message, "¡Hola! ¿Qué te gustaría hacer?", reply_markup=markup)
            else:
                self.bot.reply_to(message, "No entendí tu mensaje. Por favor, selecciona una opción del menú")

    def create_main_menu(self):
        markup = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
        markup.add(
            KeyboardButton("📊 Entrenar Modelo"),
            KeyboardButton("💬 Chat con AlmaBot"),
            KeyboardButton("📊 Ver Estadísticas")
        )
        return markup

    def start_training(self, message):
        markup = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
        markup.add(
            KeyboardButton("✅ Empezar Entrenamiento"),
            KeyboardButton("🔙 Volver")
        )
        self.bot.reply_to(message, "¿Deseas entrenar el modelo de detección de bullying?", reply_markup=markup)

    def start_chat(self, message):
        markup = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
        markup.add(
            KeyboardButton("🔙 Volver")
        )
        self.bot.reply_to(message, "¡Hola! Estoy aquí para escucharte. ¿Cómo estás hoy?", reply_markup=markup)

    def show_statistics(self, message):
        if self.model is not None:
            stats = "📊 Estadísticas del Modelo:\n"
            stats += f"- Precisión: {self.model.score(self.X_test, self.y_test):.2f}\n"
            stats += f"- Total de datos: {len(self.training_data)}\n"
            self.bot.reply_to(message, stats)
        else:
            self.bot.reply_to(message, "No hay estadísticas disponibles. Primero entrena el modelo.")

    def train_model(self):
        try:
            # Cargar datos de entrenamiento
            self.training_data = pd.read_csv('Bullying_2018_copy_traducido_final.csv')
            
            # Preprocesamiento
            X = self.training_data['text']
            y = self.training_data['label']
            
            # Dividir datos
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Vectorizar texto
            self.vectorizer = TfidfVectorizer()
            X_train_vectorized = self.vectorizer.fit_transform(self.X_train)
            
            # Entrenar modelo
            self.model = LogisticRegression()
            self.model.fit(X_train_vectorized, self.y_train)
            
            # Guardar modelo y vectorizador
            joblib.dump(self.model, 'models/bullying_detection_model.joblib')
            joblib.dump(self.vectorizer, 'models/vectorizer.joblib')
            
            return True
            
        except Exception as e:
            print(f"Error al entrenar el modelo: {str(e)}")
            return False

    def process_message(self, text):
        if self.model is None:
            return "El modelo no está entrenado. Por favor, entrena el modelo primero."
            
        try:
            text_vectorized = self.vectorizer.transform([text])
            prediction = self.model.predict(text_vectorized)[0]
            
            if prediction == 1:
                return "⚠️ He detectado que este mensaje podría ser bullying. ¿Te gustaría hablar más sobre esto?"
            else:
                return "😊 Gracias por compartir. ¿Hay algo más en lo que pueda ayudarte?"
                
        except Exception as e:
            return f"Error al procesar el mensaje: {str(e)}"

def main():
    # Cargar variables de entorno
    load_dotenv()
    
    # Obtener token de Telegram
    token = os.getenv('TELEGRAM_TOKEN')
    if not token:
        print("Error: No se encontró el token de Telegram")
        return
        
    # Crear e iniciar el bot
    bot = TelegramBot(token)
    print("Bot iniciado. Esperando mensajes...")
    bot.bot.polling()

if __name__ == "__main__":
    main()
