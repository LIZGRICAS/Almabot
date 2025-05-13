import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.logistic_regression_model import BullyingDetectionModel

def test_model():
    """Prueba el modelo de detección de bullying con varios ejemplos"""
    # Cargar el modelo desde la ruta local
    model = BullyingDetectionModel()
    try:
        # Intentar cargar desde la ruta local
        local_model_path = os.path.join(os.getcwd(), 'models')
        model.load_model(path=local_model_path)
        print(f"Modelo cargado correctamente desde: {local_model_path}")
    except Exception as e:
        print(f"Error al cargar el modelo desde ruta local: {e}")
        print("Entrenando un nuevo modelo para las pruebas...")
        
        # Si no se puede cargar, entrenar un modelo simple para las pruebas
        training_data = [
            ("Me empujan en la escuela", 1),
            ("Me insultan todos los días", 1),
            ("Nadie quiere jugar conmigo", 1),
            ("Me envían mensajes ofensivos", 1),
            ("Me siento feliz con mis amigos", 0),
            ("La maestra me felicitó hoy", 0),
            ("Jugamos todos juntos en el recreo", 0),
            ("Me gusta ir a la escuela", 0)
        ]
        
        X = [item[0] for item in training_data]
        y = [item[1] for item in training_data]
        
        model.train(X, y)
    
    # Ejemplos de prueba organizados por categoría
    test_cases = {
        "Bullying Físico": [
            "Me empujan constantemente en el pasillo",
            "Me golpean cuando nadie está mirando",
            "Me quitan mis cosas y las rompen",
            "Me dan patadas durante el recreo"
        ],
        "Bullying Verbal": [
            "Me insultan por mi apariencia",
            "Se burlan de mi forma de hablar",
            "Me ponen apodos ofensivos",
            "Dicen cosas hirientes sobre mi familia"
        ],
        "Bullying Social": [
            "Me excluyen de todas las actividades grupales",
            "Nadie quiere sentarse conmigo en el almuerzo",
            "Difunden rumores falsos sobre mí",
            "Han convencido a todos de que no me hablen"
        ],
        "Bullying Cibernético": [
            "Me envían mensajes amenazantes por WhatsApp",
            "Publican fotos mías sin mi permiso para burlarse",
            "Crearon un grupo para hablar mal de mí",
            "Me acosan en redes sociales con comentarios ofensivos"
        ],
        "No Bullying": [
            "Me siento feliz en la escuela",
            "Mis amigos me ayudan con las tareas",
            "La maestra me felicitó por mi trabajo",
            "Juego con mis compañeros durante el recreo"
        ]
    }
    
    print("\nPRUEBAS DEL MODELO DE DETECCIÓN DE BULLYING")
    print("=" * 70)
    
    # Estadísticas de precisión por categoría
    category_stats = {}
    
    # Procesar cada categoría
    for category, texts in test_cases.items():
        print(f"\n{category}:")
        print("-" * 70)
        
        correct = 0
        expected_bullying = category != "No Bullying"
        
        for text in texts:
            prediction, probabilities = model.predict(text)
            is_bullying = prediction == 1
            confidence = probabilities[1] if is_bullying else probabilities[0]
            
            # Verificar si la predicción es correcta
            is_correct = (is_bullying == expected_bullying)
            if is_correct:
                correct += 1
            
            # Determinar el tipo de bullying si es detectado
            bullying_type = ""
            if is_bullying:
                # Analizar el texto para determinar el tipo de bullying
                if any(word in text.lower() for word in ["empuj", "golpe", "patad", "pega", "rompe"]):
                    bullying_type = "físico"
                elif any(word in text.lower() for word in ["insult", "burl", "apodo", "grit", "ofens"]):
                    bullying_type = "verbal"
                elif any(word in text.lower() for word in ["excluy", "rumor", "hablen", "grupo", "social"]):
                    bullying_type = "social"
                elif any(word in text.lower() for word in ["mensaje", "whatsapp", "red", "internet", "foto"]):
                    bullying_type = "cibernético"
            
            # Mostrar resultados
            print(f"Texto: {text}")
            print(f"¿Bullying? {'SÍ' if is_bullying else 'NO'} (Confianza: {confidence:.2f})")
            if bullying_type:
                print(f"Tipo: {bullying_type}")
            print(f"Predicción {'✓ CORRECTA' if is_correct else '✗ INCORRECTA'}")
            print("-" * 70)
        
        # Calcular precisión para esta categoría
        accuracy = correct / len(texts) * 100
        category_stats[category] = accuracy
        print(f"Precisión en {category}: {accuracy:.1f}%")
    
    # Mostrar resumen de precisión
    print("\nRESUMEN DE PRECISIÓN:")
    print("=" * 70)
    total_correct = sum(stats for stats in category_stats.values())
    overall_accuracy = total_correct / len(category_stats) 
    print(f"Precisión general: {overall_accuracy:.1f}%")
    
    # Mostrar precisión por categoría
    for category, accuracy in category_stats.items():
        print(f"{category}: {accuracy:.1f}%")
    
    print("\nPrueba completada.")
    return overall_accuracy

if __name__ == "__main__":
    test_model()