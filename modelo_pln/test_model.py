import sys
import os
import re
sys.path.append('/home/sublime-dev/dev/python/Almabot/modelo_pln/src')

# Import the model class
from models.logistic_regression_model import BullyingDetectionModel

def contains_strong_indicators(text):
    """Helper function to test our strong indicators logic"""
    text_lower = text.lower()
    
    # Lista de indicadores fuertes de bullying (en minúsculas)
    strong_indicators = [
        'amenaza', 'amenazas', 'amenazando', 'amenazado', 'amenazada',
        'acoso', 'acosar', 'acosado', 'acosada', 'acosando',
        'golpear', 'golpeado', 'golpeada', 'golpeando',
        'pegar', 'pegando', 'peleando', 'pelea',
        'insultar', 'insultado', 'insultada', 'insultando', 'insultos',
        'intimidar', 'intimidado', 'intimidada', 'intimidación',
        'hostigar', 'hostigamiento', 'hostigando', 'hostigado', 'hostigada',
        'ciberacoso', 'ciberbullying', 'ciberacoso', 'ciberbully',
        'humillar', 'humillación', 'humillando', 'humillado', 'humillada',
        'agredir', 'agresión', 'agredido', 'agredida', 'agrediendo',
        'maltratar', 'maltrato', 'maltratado', 'maltratada', 'maltratando',
        'acosan', 'amenazan', 'golpearon', 'pelearon', 'insultan', 'intimidan'
    ]
    
    # Create a regex pattern to match whole words
    pattern = r'\b(' + '|'.join(map(re.escape, strong_indicators)) + r')\b'
    return bool(re.search(pattern, text_lower))

def test_model():
    # Create an instance of the model
    print("Creating model instance...")
    model = BullyingDetectionModel()
    
    # Test cases
    test_cases = [
        # Bullying scenarios - Acoso y amenazas
        ("Me envian amenzas por internet", True, "amenazas"),
        ("Me están amenazando por mensajes", True, "amenazando"),
        ("Recibo amenazas en redes sociales", True, "amenazas"),
        
        # Bullying scenarios - Hostigamiento
        ("Me están hostigando en la escuela", True, "hostigando"),
        ("Me acosan en el transporte público", True, "acosan"),
        ("Me persiguen después de clases", True, "persiguen"),
        
        # Bullying scenarios - Ciberacoso
        ("Sufro de ciberacoso", True, "ciberacoso"),
        ("Me hacen bullying en línea", True, "bullying"),
        ("Me están acosando por redes sociales", True, "acosando"),
        
        # Bullying scenarios - Maltrato y abuso
        ("Me maltratan en el colegio", True, "maltratan"),
        ("Me están maltratando en la escuela", True, "maltratando"),
        ("Sufro de abuso en mi casa", True, "abuso"),
        ("Mi compañero me golpeó ayer", True, "golpeó"),
        
        # Bullying scenarios - Intimidación y humillación
        ("Me humillan frente a todos", True, "humillan"),
        ("Me intimidan para que haga sus tareas", True, "intimidan"),
        ("Se burlan de mi apariencia", True, "burlan"),
        
        # Non-bullying scenarios
        ("Me siento mal hoy", False, ""),
        ("Me gusta el helado", False, ""),
        ("Voy al parque con mis amigos", False, ""),
        ("Estoy feliz con mis calificaciones", False, ""),
        ("Tengo mucha tarea que hacer", False, ""),
        ("Mañana es mi cumpleaños", False, "")
    ]
    
    print("\nTesting model with updated strong indicators:")
    print("=" * 70)
    
    total_tests = len(test_cases)
    correct_predictions = 0
    false_positives = 0
    false_negatives = 0
    
    print(f"\n{'='*80}")
    print(f"{'TESTING BULLYING DETECTION MODEL':^80}")
    print(f"{'='*80}")
    
    for i, (text, expected_result, expected_keyword) in enumerate(test_cases, 1):
        # Test our strong indicators function
        has_indicators = contains_strong_indicators(text.lower())
        
        # Test the model's prediction
        try:
            prediction, confidence = model.predict(text)
            prediction_result = prediction == 1
            confidence_level = max(confidence)  # Get the highest confidence score
            
            # Check if prediction matches expected result
            is_correct = (prediction_result == expected_result)
            if is_correct:
                correct_predictions += 1
            else:
                if prediction_result:  # False positive
                    false_positives += 1
                else:  # False negative
                    false_negatives += 1
            
            # Print test result
            print(f"\n{'='*80}")
            print(f"TEST {i}/{total_tests}: {'PASS' if is_correct else 'FAIL'}")
            print(f"{'='*80}")
            print(f"TEXT: {text}")
            print(f"EXPECTED: {'Bullying' if expected_result else 'Not Bullying'}")
            print(f"PREDICTION: {'Bullying' if prediction_result else 'Not Bullying'}")
            print(f"CONFIDENCE: {confidence_level:.2f}")
            print(f"KEYWORD MATCHED: {expected_keyword if expected_keyword else 'None'}")
            
            # Print warning for false positives/negatives
            if not is_correct:
                if prediction_result:
                    print("\n⚠️  FALSE POSITIVE: Incorrectly identified as bullying")
                else:
                    print("\n⚠️  FALSE NEGATIVE: Failed to detect bullying")
            
        except Exception as e:
            print(f"\n❌ ERROR in prediction: {e}")
    
    # Print summary
    accuracy = (correct_predictions / total_tests) * 100
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {total_tests}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    if false_negatives > 0:
        print("\n⚠️  WARNING: Some bullying cases were not detected (false negatives)")
    if false_positives > 0:
        print("⚠️  WARNING: Some non-bullying cases were flagged (false positives)")
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETED")
    
    print("\nTesting completed!")

if __name__ == "__main__":
    test_model()
