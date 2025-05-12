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
    """Carga y combina datos de múltiples fuentes"""
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
        
        return X, y
        
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        print(f"Tipo de error: {type(e).__name__}")
        if hasattr(e, 'args') and e.args:
            print(f"Detalles: {e.args}")
        raise

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

def main():
    """Función principal para entrenar el modelo"""
    try:
        print("\n=== Iniciando proceso de entrenamiento ===")
        start_time = datetime.now()
        
        # 1. Preparar datos
        print("\n[1/5] Preparando datos...")
        X, y = prepare_data()
        
        # 2. Dividir en conjuntos de entrenamiento y prueba
        print("\n[2/5] Dividiendo datos en conjuntos de entrenamiento y prueba...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n=== Estadísticas del Conjunto de Datos ===")
        print(f"Total de ejemplos: {len(X)}")
        print(f"Entrenamiento: {len(X_train)} ejemplos")
        print(f"Prueba: {len(X_test)} ejemplos")
        
        # 3. Validación cruzada
        print("\n[3/5] Realizando validación cruzada...")
        cv_metrics = cross_validate_model(X_train, y_train, n_splits=5)
        
        # 4. Entrenar modelo final con todos los datos de entrenamiento
        print("\n[4/5] Entrenando modelo final...")
        model, y_pred, y_prob, test_metrics = train_and_evaluate(
            X_train, X_test, y_train, y_test
        )
        
        # 5. Generar y guardar gráficos
        print("\n[5/5] Generando gráficos de evaluación...")
        plot_metrics(y_test, y_pred, y_prob, output_dir='evaluation_plots')
        
        # 6. Guardar el modelo
        print("\nGuardando el modelo...")
        model.save_model()
        
        # 7. Guardar resultados
        print("\nGuardando resultados...")
        all_metrics = {
            'cv_metrics': cv_metrics,
            'test_metrics': test_metrics,
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

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Entrena y evalúa el modelo"""
    # Crear y entrenar el modelo
    model = BullyingDetectionModel()
    
    print("\nEntrenando el modelo...")
    start_time = datetime.now()
    model.train(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")
    
    # Evaluar en el conjunto de prueba
    print("\nEvaluando el modelo en el conjunto de prueba...")
    y_pred = []
    y_prob = []
    
    for text in X_test:
        pred, prob = model.predict(text)
        y_pred.append(pred)
        y_prob.append(prob[1])
    
    y_prob = np.column_stack((1-np.array(y_prob), y_prob))
    
    # Calcular métricas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'training_time_seconds': training_time
    }
    
    # Mostrar informe de clasificación
    print("\n=== Informe de Clasificación ===")
    print(classification_report(y_test, y_pred, target_names=['No Bullying', 'Bullying']))
    
    return model, y_pred, y_prob, metrics

def cross_validate_model(X, y, n_splits=5):
    """Realiza validación cruzada del modelo"""
    print(f"\nRealizando validación cruzada con {n_splits} folds...")
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = BullyingDetectionModel()
    
    cv_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'training_time': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"\n=== Fold {fold}/{n_splits} ===")
        
        X_train, X_val = [X[i] for i in train_idx], [X[i] for i in val_idx]
        y_train, y_val = [y[i] for i in train_idx], [y[i] for i in val_idx]
        
        # Entrenar y evaluar
        _, y_pred, _, metrics = train_and_evaluate(X_train, X_val, y_train, y_val)
        
        # Guardar métricas
        cv_metrics['accuracy'].append(metrics['accuracy'])
        cv_metrics['precision'].append(metrics['precision'])
        cv_metrics['recall'].append(metrics['recall'])
        cv_metrics['f1'].append(metrics['f1'])
        cv_metrics['training_time'].append(metrics['training_time_seconds'])
    
    # Calcular promedios
    avg_metrics = {k: np.mean(v) for k, v in cv_metrics.items()}
    
    print("\n=== Métricas Promedio de Validación Cruzada ===")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return avg_metrics

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