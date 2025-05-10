from logistic_regression_model import BullyingDetectionModel

def test_model():
    # Cargar el modelo
    model = BullyingDetectionModel()
    model.load_model()
    
    # Ejemplos de prueba
    test_texts = [
        "Me siento mal porque mis compañeros me molestan en el recreo",
        "Me empujan constantemente en el pasillo",
        "Me siento feliz en la escuela",
        "Mis amigos me ayudan con las tareas",
        "Me envían mensajes amenazantes por WhatsApp",
        "Me excluyen de todas las actividades grupales"
    ]
    
    print("\nPruebas del modelo:")
    print("-" * 50)
    
    for text in test_texts:
        prediction, probability = model.predict(text)
        result = "BULLYING" if prediction == 1 else "NO BULLYING"
        confidence = max(probability)
        
        print(f"\nTexto: {text}")
        print(f"Predicción: {result}")
        print(f"Confianza: {confidence:.2%}")
        print(f"Probabilidades: {probability}")

if __name__ == "__main__":
    test_model() 