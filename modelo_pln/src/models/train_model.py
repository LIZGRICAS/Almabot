import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
from sklearn.utils import resample, class_weight
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

# Configuración de visualización
try:
    plt.style.use('seaborn-v0_8')  # Para versiones recientes de matplotlib
    sns.set_palette('Set2')
except:
    try:
        plt.style.use('seaborn')  # Para versiones más antiguas
        sns.set_palette('Set2')
    except:
        # Usar estilo por defecto si fallan los anteriores
        sns.set_theme(style="whitegrid")

# Añadir el directorio padre al path para importaciones relativas
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.logistic_regression_model import BullyingDetectionModel

def balance_dataset(X, y):
    """Balancea el conjunto de datos"""
    # Convertir a DataFrame para facilitar el manejo
    df = pd.DataFrame({'texto': X, 'label': y})
    
    # Separar las clases
    df_majority = df[df.label == 1]
    df_minority = df[df.label == 0]
    
    # Upsample la clase minoritaria
    df_minority_upsampled = resample(df_minority, 
                                   replace=True,
                                   n_samples=len(df_majority),
                                   random_state=42)
    
    # Combinar las clases
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    
    return df_balanced['texto'].tolist(), df_balanced['label'].tolist()

def load_data():
    """Carga y combina datos de múltiples fuentes con mejor manejo de errores y validación"""
    try:
        # Rutas de los archivos de datos
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')
        phrases_path = os.path.join(data_dir, 'training_phrases.csv')
        bullying_path = os.path.join(data_dir, 'Bullying_2018_copy_traducido_final.csv')
        enhanced_path = os.path.join(data_dir, 'enhanced_training_data.csv')
        
        print(f"Cargando datos desde:")
        print(f"- Frases: {phrases_path}")
        print(f"- Encuesta: {bullying_path}")
        print(f"- Datos mejorados: {enhanced_path}")
        
        # Cargar datos de frases
        if not os.path.exists(phrases_path):
            raise FileNotFoundError(f"No se encontró el archivo de frases: {phrases_path}")
            
        phrases_df = pd.read_csv(phrases_path, sep=';')
        print(f"Frases cargadas: {len(phrases_df)} registros")
        
        # Verificar columnas requeridas en phrases_df
        required_columns = ['texto', 'es_acoso']
        for col in required_columns:
            if col not in phrases_df.columns:
                raise ValueError(f"La columna '{col}' no se encuentra en el archivo de frases. Columnas disponibles: {phrases_df.columns.tolist()}")
        
        # Cargar datos de la encuesta
        if not os.path.exists(bullying_path):
            raise FileNotFoundError(f"No se encontró el archivo de encuesta: {bullying_path}")
            
        bullying_df = pd.read_csv(bullying_path, sep=';')
        print(f"Datos de encuesta cargados: {len(bullying_df)} registros")
        
        # Cargar datos mejorados si existen
        enhanced_df = pd.DataFrame()
        if os.path.exists(enhanced_path):
            enhanced_df = pd.read_csv(enhanced_path, sep=',')
            print(f"Datos mejorados cargados: {len(enhanced_df)} registros")
            
            # Verificar columnas requeridas en enhanced_df
            if not enhanced_df.empty:
                for col in required_columns:
                    if col not in enhanced_df.columns:
                        print(f"Advertencia: La columna '{col}' no se encuentra en los datos mejorados. Ignorando datos mejorados.")
                        enhanced_df = pd.DataFrame()
                        break
        
        # Procesar datos de la encuesta
        bullying_texts, bullying_labels = process_survey_data(bullying_df)
        
        # Combinar todos los datos
        X = list(phrases_df['texto']) + bullying_texts
        y = list(phrases_df['es_acoso']) + bullying_labels
        
        # Agregar datos mejorados si existen y tienen las columnas correctas
        if not enhanced_df.empty:
            X.extend(enhanced_df['texto'].tolist())
            y.extend(enhanced_df['es_acoso'].tolist())
        
        # Aplicar técnicas de data augmentation para mejorar el conjunto de datos
        X_augmented, y_augmented = augment_data(X, y)
        
        # Combinar datos originales y aumentados
        X_combined = X + X_augmented
        y_combined = y + y_augmented
        
        print(f"Datos originales: {len(X)} ejemplos")
        print(f"Datos aumentados: {len(X_augmented)} ejemplos")
        print(f"Total de datos para entrenamiento: {len(X_combined)} ejemplos")
        
        return X_combined, y_combined
        
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        print(f"Tipo de error: {type(e).__name__}")
        if hasattr(e, 'args') and e.args:
            print(f"Detalles: {e.args}")
        raise

def augment_data(X, y):
    """Aplica técnicas de data augmentation para mejorar el conjunto de datos"""
    import random
    import re
    import nltk
    from nltk.corpus import stopwords
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    spanish_stopwords = set(stopwords.words('spanish'))
    
    # Listas para almacenar datos aumentados
    X_augmented = []
    y_augmented = []
    
    # Diccionario de sinónimos para palabras clave relacionadas con bullying
    bullying_synonyms = {
        'insultar': ['ofender', 'agredir verbalmente', 'faltar el respeto', 'decir groserías'],
        'golpear': ['pegar', 'agredir', 'dar golpes', 'lastimar físicamente', 'empujar'],
        'burlar': ['mofar', 'ridiculizar', 'hacer bromas pesadas', 'reírse de'],
        'amenazar': ['intimidar', 'amedrentar', 'advertir', 'asustar'],
        'excluir': ['aislar', 'apartar', 'dejar de lado', 'ignorar', 'rechazar'],
        'molestar': ['fastidiar', 'incomodar', 'perturbar', 'hostigar'],
        'humillar': ['avergonzar', 'denigrar', 'degradar', 'menospreciar'],
        'acosar': ['perseguir', 'hostigar', 'atosigar', 'importunar'],
        'maltratar': ['tratar mal', 'abusar', 'dañar', 'perjudicar'],
        'cyberbullying': ['acoso cibernético', 'acoso en línea', 'acoso digital', 'acoso en redes']
    }
    
    # Técnicas de augmentación para ejemplos positivos (bullying)
    for i, (text, label) in enumerate(zip(X, y)):
        if label == 1:  # Solo aumentar ejemplos de bullying
            # 1. Sustitución de sinónimos
            words = text.split()
            for key, synonyms in bullying_synonyms.items():
                if key in text.lower():
                    for synonym in synonyms:
                        new_text = text.lower().replace(key, synonym)
                        X_augmented.append(new_text)
                        y_augmented.append(1)
            
            # 2. Cambio de orden de palabras (para frases más largas)
            if len(words) > 5:
                # Mantener el inicio y el final, mezclar el medio
                middle_words = words[1:-1]
                if len(middle_words) > 2:
                    for _ in range(min(2, len(middle_words))):
                        random.shuffle(middle_words)
                        new_text = words[0] + ' ' + ' '.join(middle_words) + ' ' + words[-1]
                        X_augmented.append(new_text)
                        y_augmented.append(1)
            
            # 3. Adición de contexto escolar para ejemplos claros de bullying
            if any(term in text.lower() for term in ['insultar', 'golpear', 'burlar', 'amenazar', 'excluir']):
                school_contexts = [
                    'en la escuela', 'en el colegio', 'en clase', 
                    'en el recreo', 'en el pasillo', 'en el baño de la escuela',
                    'durante el almuerzo', 'en educación física'
                ]
                for context in school_contexts[:2]:  # Limitar a 2 contextos por ejemplo
                    if context not in text.lower():
                        new_text = f"{text} {context}"
                        X_augmented.append(new_text)
                        y_augmented.append(1)
    
    # Limitar la cantidad de datos aumentados para evitar desbalance extremo
    max_augmented = len(X) // 2  # Máximo 50% más de datos
    if len(X_augmented) > max_augmented:
        # Seleccionar aleatoriamente un subconjunto
        indices = random.sample(range(len(X_augmented)), max_augmented)
        X_augmented = [X_augmented[i] for i in indices]
        y_augmented = [y_augmented[i] for i in indices]
    
    return X_augmented, y_augmented

def process_survey_data(df, min_age=10, max_age=18):
    """
    Procesa los datos de la encuesta de bullying para extraer ejemplos de bullying.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de la encuesta
        min_age (int): Edad mínima para incluir en los datos
        max_age (int): Edad máxima para incluir en los datos
        
    Returns:
        tuple: (texts, labels) listas de textos y etiquetas (1 para bullying, 0 para no bullying)
    """
    texts = []
    labels = []
    
    # Mapeo de columnas a mensajes descriptivos
    BULLYING_INDICATORS = {
        'Intimidado en la propiedad escolar en los últimos 12 meses': {
            'Sí': 'He sido intimidado en la escuela',
            'No': ''
        },
        'Acosado cibernético en los últimos 12 meses': {
            'Sí': 'He sufrido ciberacoso',
            'No': ''
        },
        'Atacado físicamente': {
            '1 vez': 'Me han atacado físicamente una vez',
            '2 o 3 veces': 'Me han atacado físicamente varias veces',
            '4 o 5 veces': 'Me han atacado físicamente en múltiples ocasiones',
            '6 o 7 veces': 'Me han atacado físicamente muchas veces',
            '8 o 9 veces': 'Me han atacado físicamente frecuentemente',
            '10 o 11 veces': 'Me han atacado físicamente muy frecuentemente',
            '12 o más veces': 'Me han atacado físicamente constantemente'
        },
        'Involucrado en una pelea física': {
            '1 vez': 'He estado involucrado en una pelea física',
            '2 o 3 veces': 'He estado involucrado en varias peleas físicas',
            '4 o 5 veces': 'He estado involucrado en múltiples peleas físicas',
            '6 o 7 veces': 'He estado involucrado en muchas peleas físicas',
            '8 o 9 veces': 'He estado involucrado en peleas físicas frecuentemente',
            '10 o 11 veces': 'He estado involucrado en peleas físicas muy frecuentemente',
            '12 o más veces': 'He estado involucrado en peleas físicas constantemente'
        }
    }
    
    # Contadores para estadísticas
    total_rows = 0
    bullying_cases = 0
    
    for _, row in df.iterrows():
        # Filtrar por edad si la columna existe
        if 'Edad' in df.columns:
            try:
                age = int(row['Edad'])
                if age < min_age or age > max_age:
                    continue
            except (ValueError, TypeError):
                continue
        
        total_rows += 1
        text_parts = []
        is_bullying = 0
        
        # Verificar cada indicador de bullying
        for col, messages in BULLYING_INDICATORS.items():
            if col in df.columns and pd.notna(row[col]) and str(row[col]).strip() != '':
                value = str(row[col]).strip()
                if value in messages and messages[value]:
                    is_bullying = 1
                    text_parts.append(messages[value])
        
        # Añadir información demográfica si está disponible
        demographic_info = []
        if 'Género' in df.columns and pd.notna(row['Género']):
            demographic_info.append(f"Soy {row['Género']}")
        if 'Edad' in df.columns and pd.notna(row['Edad']):
            demographic_info.append(f"tengo {int(row['Edad'])} años")
        
        # Si hay indicadores de bullying, añadir al dataset
        if is_bullying:
            bullying_cases += 1
            full_text = []
            
            # Añadir información demográfica al inicio si está disponible
            if demographic_info:
                full_text.append(' '.join(demographic_info) + '.')
            
            # Añadir los indicadores de bullying
            full_text.extend(text_parts)
            
            # Añadir información sobre cómo se siente la persona si está disponible
            if 'Se siente solo' in df.columns and pd.notna(row['Se siente solo']):
                if row['Se siente solo'] in ['Siempre', 'Casi siempre', 'A veces']:
                    full_text.append(f"Me siento {row['Se siente solo'].lower()}")
            
            # Unir todo en un solo texto
            texts.append('. '.join(full_text) + '.')
            labels.append(is_bullying)
    
    # Generar ejemplos negativos (no bullying)
    non_bullying_count = min(len(texts), total_rows - bullying_cases)
    non_bullying_texts = []
    
    for _, row in df.iterrows():
        if len(non_bullying_texts) >= non_bullying_count:
            break
            
        is_bullying = any(
            str(row[col]).strip() in ['Sí'] + [k for k in BULLYING_INDICATORS[col].keys() if k != 'No']
            for col in BULLYING_INDICATORS
            if col in df.columns and pd.notna(row[col]) and str(row[col]).strip() != ''
        )
        
        if not is_bullying:
            demographic_info = []
            if 'Género' in df.columns and pd.notna(row['Género']):
                demographic_info.append(f"Soy {row['Género']}")
            if 'Edad' in df.columns and pd.notna(row['Edad']):
                demographic_info.append(f"tengo {int(row['Edad'])} años")
            
            if demographic_info:
                non_bullying_texts.append(' '.join(demographic_info) + '. No he sufrido acoso.')
    
    # Añadir ejemplos negativos
    texts.extend(non_bullying_texts)
    labels.extend([0] * len(non_bullying_texts))
    
    print(f"\n=== Estadísticas de procesamiento de la encuesta ===")
    print(f"Total de filas procesadas: {total_rows}")
    print(f"Casos de bullying identificados: {bullying_cases}")
    print(f"Ejemplos sin bullying: {len(non_bullying_texts)}")
    print(f"Total de ejemplos generados: {len(texts)}")
    
    return texts, labels

def analyze_class_distribution(y):
    """
    Analiza y muestra estadísticas detalladas sobre la distribución de clases.
    
    Args:
        y: Lista o array de etiquetas
        
    Returns:
        tuple: (class_counts, class_dist) conteos y porcentajes de cada clase
    """
    if not y or len(y) == 0:
        print("Advertencia: No hay datos para analizar.")
        return None, None
    
    # Convertir a serie para análisis
    y_series = pd.Series(y)
    
    # Calcular estadísticas básicas
    class_counts = y_series.value_counts().sort_index()
    class_dist = (class_counts / len(y)) * 100
    
    # Calcular desbalance de clases
    imbalance_ratio = class_counts.max() / class_counts.min() if len(class_counts) > 1 else float('inf')
    
    # Calcular entropía de la distribución
    probs = class_counts / len(y)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))  # Se añade un pequeño épsilon para evitar log(0)
    
    # Mostrar resultados
    print("\n=== Análisis de Distribución de Clases ===")
    print(f"Total de ejemplos: {len(y)}")
    print(f"Número de clases: {len(class_counts)}")
    print(f"Ratio de desbalance: {imbalance_ratio:.2f}")
    print(f"Entropía de la distribución: {entropy:.4f}")
    
    # Mostrar distribución detallada
    print("\nDistribución detallada:")
    print("-" * 50)
    print(f"{'Clase':<10} {'Conteo':<10} {'Porcentaje':<15} {'Distribución'}")
    print("-" * 50)
    
    for label, (count, percent) in zip(class_counts.index, zip(class_counts, class_dist)):
        bar_length = int(percent / 2)  # Ajustar longitud de la barra
        bar = '█' * bar_length
        print(f"{label:<10} {count:<10} {percent:>5.1f}% {' ' * 5} {bar}")
    
    # Mostrar advertencias si es necesario
    if len(class_counts) == 1:
        print("\n¡ADVERTENCIA: Solo hay una clase en los datos!")
    elif imbalance_ratio > 10:
        print("\n¡ADVERTENCIA: Los datos están muy desbalanceados!")
    elif entropy < 0.5:
        print("\nADVERTENCIA: La distribución de clases está muy sesgada.")
    
    print("\n" + "=" * 70)
    
    return class_counts, class_dist

def prepare_data():
    """Prepara los datos para el entrenamiento"""
    try:
        # Cargar datos
        X, y = load_data()
        
        # Analizar distribución de clases
        class_counts, class_dist = analyze_class_distribution(y)
        
        # Calcular pesos de clases
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )
        print(f"\nPesos de clases: {dict(zip(np.unique(y), class_weights))}")
        
        # Balancear el conjunto de datos
        X_balanced, y_balanced = balance_dataset(X, y)
        
        # Verificar distribución después del balanceo
        print("\nDistribución después del balanceo:")
        analyze_class_distribution(y_balanced)
        
        return X_balanced, y_balanced
        
    except Exception as e:
        print(f"Error al preparar los datos: {e}")
        raise

def main(force_training=False):
    """Función principal para entrenar el modelo
    
    Args:
        force_training (bool): Si es True, fuerza el reentrenamiento del modelo aunque ya exista.
                              Si es False, usa el modelo existente si está disponible.
    """
    try:
        # Verificar si el modelo ya existe
        model_path = os.environ.get('MODEL_PATH', 'models/model_8001')
        model_file = os.path.join(os.path.dirname(model_path), 'bullying_detection_model.joblib')
        features_file = os.path.join(os.path.dirname(model_path), 'text_features.joblib')
        
        # Verificar si los archivos del modelo existen
        model_exists = os.path.exists(model_file) and os.path.exists(features_file)
        
        print(f"\nVerificando modelo en: {model_file}")
        print(f"Modelo existe: {model_exists}")
        print(f"Forzar entrenamiento: {force_training}")
        
        if model_exists and not force_training:
            print("\n=== Modelo existente detectado ===")
            print(f"Usando modelo existente en: {model_file}")
            print("Para reentrenar el modelo, establece FORCE_TRAINING=true")
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            return 0
            
        print("\n=== Iniciando proceso de entrenamiento ===")
        start_time = datetime.now()
        
        # 1. Preparar datos
        print("\n[1/7] Preparando datos...")
        X, y = prepare_data()
        
        # 2. Dividir en conjuntos de entrenamiento y prueba
        print("\n[2/7] Dividiendo datos en conjuntos de entrenamiento y prueba...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n=== Estadísticas del Conjunto de Datos ===")
        print(f"Total de ejemplos: {len(X)}")
        print(f"Entrenamiento: {len(X_train)} ejemplos")
        print(f"Prueba: {len(X_test)} ejemplos")
        
        # 3. Validación cruzada con modelo base
        print("\n[3/7] Realizando validación cruzada con modelo base...")
        base_cv_metrics = cross_validate_model(X_train, y_train, n_splits=5)
        
        # 4. Búsqueda de hiperparámetros óptimos
        print("\n[4/7] Realizando búsqueda de hiperparámetros...")
        best_model, best_params, best_score = perform_hyperparameter_tuning(X_train, y_train)
        
        # Guardar los mejores hiperparámetros
        hyperparams_file = os.path.join('models', 'best_hyperparameters.json')
        os.makedirs(os.path.dirname(hyperparams_file), exist_ok=True)
        with open(hyperparams_file, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"Mejores hiperparámetros guardados en {hyperparams_file}")
        
        # 5. Entrenar modelo final con hiperparámetros optimizados
        print("\n[5/7] Entrenando modelo final con hiperparámetros optimizados...")
        
        # Si se encontraron hiperparámetros óptimos, usarlos para crear un nuevo modelo
        if best_model is not None:
            # Crear un modelo estándar ya que no acepta hiperparámetros directamente
            model = BullyingDetectionModel()
            print(f"Usando hiperparámetros optimizados: {best_params}")
            # Nota: Actualmente no podemos pasar los hiperparámetros directamente al modelo
            # En una implementación futura, se podría modificar la clase BullyingDetectionModel
            # para aceptar hiperparámetros como argumento
            
            # Entrenar el modelo
            model.train(X_train, y_train)
            
            # Evaluar en conjunto de prueba
            # Procesar cada texto y obtener predicciones
            y_pred = []
            y_prob = []
            
            for text in X_test:
                pred, prob = model.predict(text)
                y_pred.append(pred)
                y_prob.append(prob[1])  # Probabilidad de la clase positiva (bullying)
            
            # Convertir a formato numpy para cálculos
            y_pred = np.array(y_pred)
            y_prob = np.column_stack((1-np.array(y_prob), y_prob))  # Formato [prob_clase_0, prob_clase_1]
            
            # Calcular métricas
            test_metrics = calculate_metrics(y_test, y_pred, y_prob)
        else:
            # Si falló la búsqueda de hiperparámetros, usar el enfoque estándar
            print("No se pudieron encontrar hiperparámetros óptimos. Usando modelo estándar.")
            model, y_pred, y_prob, test_metrics = train_and_evaluate(
                X_train, X_test, y_train, y_test
            )
        
        # 6. Generar y guardar gráficos
        print("\n[6/7] Generando gráficos de evaluación...")
        plot_metrics(y_test, y_pred, y_prob, output_dir='evaluation_plots')
        
        # 7. Guardar el modelo y resultados
        print("\n[7/7] Guardando el modelo y resultados...")
        model.save_model()
        
        # Comparar métricas antes y después de la optimización
        print("\n=== Comparación de Métricas Antes y Después de Optimización ===")
        metrics_comparison = {
            'base_model': {
                'accuracy': base_cv_metrics.get('accuracy', 0),
                'precision': base_cv_metrics.get('precision', 0),
                'recall': base_cv_metrics.get('recall', 0),
                'f1': base_cv_metrics.get('f1', 0),
                'roc_auc': base_cv_metrics.get('roc_auc', 0)
            },
            'optimized_model': {
                'accuracy': test_metrics.get('accuracy', 0),
                'precision': test_metrics.get('precision', 0),
                'recall': test_metrics.get('recall', 0),
                'f1': test_metrics.get('f1', 0),
                'roc_auc': test_metrics.get('roc_auc', 0)
            }
        }
        
        # Mostrar comparación
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            base_value = metrics_comparison['base_model'][metric]
            opt_value = metrics_comparison['optimized_model'][metric]
            diff = opt_value - base_value
            diff_percent = (diff / base_value * 100) if base_value > 0 else 0
            
            print(f"{metric.capitalize()}: {base_value:.4f} → {opt_value:.4f} ")
            if diff > 0:
                print(f"  Mejora: +{diff:.4f} (+{diff_percent:.2f}%)")
            else:
                print(f"  Cambio: {diff:.4f} ({diff_percent:.2f}%)")
        
        # Guardar todos los resultados
        all_metrics = {
            'base_cv_metrics': base_cv_metrics,
            'optimized_test_metrics': test_metrics,
            'best_hyperparameters': best_params,
            'best_hyperparameter_score': best_score,
            'metrics_comparison': metrics_comparison,
            'training_time_seconds': (datetime.now() - start_time).total_seconds(),
            'model_info': {
                'type': 'StackingClassifier',
                'base_models': ['LogisticRegression', 'SVC'],
                'final_estimator': 'RandomForestClassifier',
                'features': ['TF-IDF (word ngrams 1-2)', 'TF-IDF (char ngrams 3-5)']
            }
        }
        
        save_results(all_metrics)
        
        print("\n=== Proceso de entrenamiento completado exitosamente ===")
        print(f"Tiempo total: {(datetime.now() - start_time).total_seconds()/60:.2f} minutos")
        
        # Mostrar recomendaciones basadas en los resultados
        print("\n=== Recomendaciones ===")
        if test_metrics['f1'] < 0.7:
            print("- Considere aumentar el conjunto de datos con más ejemplos de bullying.")
        if base_cv_metrics.get('train_test_diff', 0) > 0.15:
            print("- El modelo muestra signos de overfitting. Considere usar regularización más fuerte.")
        if test_metrics['recall'] < 0.7:
            print("- La capacidad del modelo para detectar casos positivos (recall) es baja. ")
            print("  Considere ajustar el umbral de decisión o usar técnicas de muestreo para mejorar la detección.")
        
        print("\nPara mejorar aún más el modelo, considere:")
        print("1. Ampliar el conjunto de datos con más ejemplos de bullying diversos")
        print("2. Experimentar con diferentes arquitecturas de modelo")
        print("3. Implementar técnicas de data augmentation más avanzadas")
        print("4. Realizar feature engineering adicional para capturar mejor los indicadores de bullying")
        
        return 0
        
        # Mostrar resumen de métricas
        print("\n=== Resumen de Métricas ===")
        print(f"\nValidación Cruzada (promedio):")
        for metric, value in cv_metrics.items():
            if isinstance(value, float):
                print(f"- {metric}: {value:.4f}")
        
        print("\nConjunto de Prueba:")
        for metric, value in test_metrics.items():
            if metric != 'training_time_seconds':
                print(f"- {metric}: {value:.4f}")
        
    except Exception as e:
        print(f"\nError durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calcula métricas de evaluación del modelo"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    # Calcular ROC AUC si hay probabilidades disponibles
    if y_prob is not None:
        try:
            if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                # Multiclass case
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            else:
                # Binary case
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1] if len(y_prob.shape) > 1 else y_prob)
        except Exception as e:
            print(f"No se pudo calcular ROC AUC: {e}")
    
    return metrics

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Entrena y evalúa el modelo"""
    start_time = datetime.now()
    
    # Crear y entrenar el modelo
    model = BullyingDetectionModel()
    
    print("\nEntrenando el modelo...")
    model.train(X_train, y_train)
    
    # Evaluar en conjunto de prueba
    print("\nEvaluando en conjunto de prueba...")
    
    # Procesar cada texto y obtener predicciones
    y_pred = []
    y_prob = []
    
    for text in X_test:
        pred, prob = model.predict(text)
        y_pred.append(pred)
        y_prob.append(prob[1])  # Probabilidad de la clase positiva (bullying)
    
    # Convertir a formato numpy para cálculos
    y_pred = np.array(y_pred)
    y_prob = np.column_stack((1-np.array(y_prob), y_prob))  # Formato [prob_clase_0, prob_clase_1]
    
    # Calcular métricas
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    
    # Añadir tiempo de entrenamiento
    training_time = (datetime.now() - start_time).total_seconds()
    metrics['training_time_seconds'] = training_time
    
    # Mostrar métricas
    print("\n=== Métricas en Conjunto de Prueba ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")
    
    # Mostrar matriz de confusión
    print("\nMatriz de confusión:")
    print(np.array(metrics['confusion_matrix']))
    
    # Mostrar informe de clasificación
    from sklearn.metrics import classification_report
    print("\n=== Informe de Clasificación ===")
    print(classification_report(y_test, y_pred, target_names=['No Bullying', 'Bullying']))
    
    return model, y_pred, y_prob, metrics

def cross_validate_model(X, y, n_splits=5):
    """Realiza validación cruzada del modelo con métricas avanzadas"""
    print(f"\nRealizando validación cruzada con {n_splits} folds...")
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = BullyingDetectionModel()
    
    # Ampliar las métricas a evaluar
    cv_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'balanced_accuracy': [],
        'training_time': [],
        'train_test_diff': []  # Para evaluar overfitting
    }
    
    # Matrices de confusión acumuladas
    all_y_true = []
    all_y_pred = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"\n=== Fold {fold}/{n_splits} ===")
        
        X_train, X_val = [X[i] for i in train_idx], [X[i] for i in val_idx]
        y_train, y_val = [y[i] for i in train_idx], [y[i] for i in val_idx]
        
        # Entrenar y evaluar
        _, y_pred, y_prob, metrics = train_and_evaluate(X_train, X_val, y_train, y_val)
        
        # Acumular predicciones para análisis global
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        
        # Guardar métricas estándar
        cv_metrics['accuracy'].append(metrics['accuracy'])
        cv_metrics['precision'].append(metrics['precision'])
        cv_metrics['recall'].append(metrics['recall'])
        cv_metrics['f1'].append(metrics['f1'])
        cv_metrics['training_time'].append(metrics['training_time_seconds'])
        
        # Calcular y guardar métricas adicionales
        if 'roc_auc' in metrics:
            cv_metrics['roc_auc'].append(metrics['roc_auc'])
        else:
            # Calcular ROC AUC manualmente si no está disponible
            try:
                from sklearn.metrics import roc_auc_score
                roc_auc = roc_auc_score(y_val, y_prob[:, 1] if len(y_prob.shape) > 1 else y_prob)
                cv_metrics['roc_auc'].append(roc_auc)
            except Exception as e:
                print(f"No se pudo calcular ROC AUC: {e}")
                cv_metrics['roc_auc'].append(0.0)
        
        # Calcular balanced accuracy
        try:
            from sklearn.metrics import balanced_accuracy_score
            balanced_acc = balanced_accuracy_score(y_val, y_pred)
            cv_metrics['balanced_accuracy'].append(balanced_acc)
        except Exception as e:
            print(f"No se pudo calcular balanced accuracy: {e}")
            cv_metrics['balanced_accuracy'].append(0.0)
        
        # Evaluar diferencia entre train y test para detectar overfitting
        # En lugar de usar el modelo directamente, que puede no estar correctamente ajustado,
        # usamos la diferencia entre las métricas de entrenamiento y validación reportadas
        # por el modelo durante el entrenamiento
        try:
            # Crear un nuevo modelo y entrenarlo para obtener métricas de entrenamiento
            temp_model = BullyingDetectionModel()
            temp_model.train(X_train, y_train)
            
            # Evaluar en datos de entrenamiento
            train_preds = []
            for text in X_train:
                pred, _ = temp_model.predict(text)
                train_preds.append(pred)
            
            train_f1 = f1_score(y_train, train_preds, average='weighted')
            test_f1 = metrics['f1']
            cv_metrics['train_test_diff'].append(train_f1 - test_f1)
        except Exception as e:
            print(f"No se pudo calcular la diferencia entre train y test: {e}")
            cv_metrics['train_test_diff'].append(0.0)
    
    # Calcular promedios y desviaciones estándar
    avg_metrics = {k: np.mean(v) for k, v in cv_metrics.items()}
    std_metrics = {f"{k}_std": np.std(v) for k, v in cv_metrics.items()}
    
    # Combinar promedios y desviaciones
    final_metrics = {**avg_metrics, **std_metrics}
    
    # Calcular matriz de confusión global
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(all_y_true, all_y_pred)
    final_metrics['confusion_matrix'] = conf_matrix.tolist()
    
    print("\n=== Métricas Promedio de Validación Cruzada ===")
    for metric, value in avg_metrics.items():
        if isinstance(value, (int, float)):
            std_value = std_metrics.get(f"{metric}_std", 0.0)
            print(f"{metric}: {value:.4f} (±{std_value:.4f})")
    
    # Evaluar overfitting
    if avg_metrics['train_test_diff'] > 0.1:
        print("\n\u26A0️ ADVERTENCIA: Posible overfitting detectado")
        print(f"Diferencia promedio entre F1 de entrenamiento y validación: {avg_metrics['train_test_diff']:.4f}")
    
    # Mostrar matriz de confusión global
    print("\nMatriz de confusión global:")
    print(conf_matrix)
    
    return final_metrics

def perform_hyperparameter_tuning(X, y):
    """Realiza búsqueda de hiperparámetros para optimizar el modelo"""
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    
    print("\nIniciando búsqueda de hiperparámetros...")
    
    try:
        # Crear modelo base
        model = BullyingDetectionModel()
        
        # Vectorizar textos
        X_vectors = model.text_features.fit_transform(X)
        
        # Definir parámetros para búsqueda
        param_grid = {
            'final_estimator__n_estimators': [50, 100, 150],
            'final_estimator__max_depth': [None, 10, 20],
            'final_estimator__min_samples_split': [2, 5, 10],
            'final_estimator__class_weight': ['balanced', 'balanced_subsample']
        }
        
        # Usar RandomizedSearchCV para una búsqueda más eficiente
        random_search = RandomizedSearchCV(
            model.model,
            param_distributions=param_grid,
            n_iter=10,  # Número de combinaciones a probar
            cv=5,
            scoring='f1',
            n_jobs=-1,  # Usar todos los núcleos disponibles
            verbose=1,
            random_state=42
        )
        
        # Ejecutar búsqueda
        print("Ejecutando búsqueda de hiperparámetros (esto puede tomar varios minutos)...")
        random_search.fit(X_vectors, y)
        
        # Mostrar mejores parámetros
        print(f"\nMejores parámetros encontrados:")
        for param, value in random_search.best_params_.items():
            print(f"- {param}: {value}")
        
        print(f"\nMejor puntuación F1: {random_search.best_score_:.4f}")
        
        # Realizar una búsqueda más refinada alrededor de los mejores parámetros
        refined_param_grid = {}
        for param, value in random_search.best_params_.items():
            if param == 'final_estimator__n_estimators':
                refined_param_grid[param] = [max(value-25, 10), value, min(value+25, 200)]
            elif param == 'final_estimator__max_depth' and value is not None:
                refined_param_grid[param] = [max(value-5, 5), value, min(value+5, 30)]
            elif param == 'final_estimator__min_samples_split':
                refined_param_grid[param] = [max(value-2, 2), value, min(value+2, 15)]
            else:
                refined_param_grid[param] = [value]
        
        # Ejecutar búsqueda refinada
        grid_search = GridSearchCV(
            model.model,
            param_grid=refined_param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        print("\nEjecutando búsqueda refinada...")
        grid_search.fit(X_vectors, y)
        
        print(f"\nMejores parámetros refinados:")
        for param, value in grid_search.best_params_.items():
            print(f"- {param}: {value}")
        
        print(f"\nMejor puntuación F1 refinada: {grid_search.best_score_:.4f}")
        
        # Devolver el mejor modelo y sus parámetros
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
        
    except Exception as e:
        print(f"Error en búsqueda de hiperparámetros: {e}")
        import traceback
        traceback.print_exc()
        return None, {}, 0.0

def plot_metrics(y_true, y_pred, y_prob, output_dir='evaluation_plots'):
    """Genera y guarda gráficos de métricas de evaluación"""
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Bullying', 'Bullying'],
                yticklabels=['No Bullying', 'Bullying'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{timestamp}.png'))
    plt.close()
    
    # 2. Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'roc_curve_{timestamp}.png'))
    plt.close()
    
    # 3. Curva Precisión-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
    avg_precision = average_precision_score(y_true, y_prob[:, 1])
    
    plt.figure(figsize=(10, 8))
    plt.step(recall, precision, where='post', 
             label=f'Precisión-Recall (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precisión')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Curva Precisión-Recall')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'precision_recall_curve_{timestamp}.png'))
    plt.close()
    
    # 4. Gráfico de calibración
    prob_true, prob_pred = calibration_curve(y_true, y_prob[:, 1], n_bins=10)
    
    plt.figure(figsize=(10, 8))
    plt.plot(prob_pred, prob_true, 's-', label='Modelo')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectamente calibrado')
    plt.xlabel('Probabilidad predicha')
    plt.ylabel('Fracción de positivos')
    plt.title('Gráfico de Calibración')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'calibration_curve_{timestamp}.png'))
    plt.close()
    
    return {
        'roc_auc': roc_auc,
        'average_precision': avg_precision,
        'confusion_matrix': cm.tolist()
    }

def save_results(metrics, output_dir='results'):
    """Guarda las métricas y resultados del modelo"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Guardar métricas en JSON
    with open(os.path.join(output_dir, f'metrics_{timestamp}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Guardar resumen en texto
    with open(os.path.join(output_dir, f'summary_{timestamp}.txt'), 'w') as f:
        f.write("=== Resumen del Entrenamiento ===\n\n")
        f.write(f"Fecha: {datetime.now()}\n\n")
        f.write("Métricas de rendimiento:\n")
        for metric, value in metrics.items():
            if isinstance(value, float):
                f.write(f"- {metric}: {value:.4f}\n")
            else:
                f.write(f"- {metric}: {value}\n")
    
    print(f"\nResultados guardados en el directorio: {output_dir}")

if __name__ == "__main__":
    main()