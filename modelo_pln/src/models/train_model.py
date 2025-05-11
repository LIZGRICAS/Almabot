import pandas as pd
import os
import sys

# Añadir el directorio padre al path para importaciones relativas
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.logistic_regression_model import BullyingDetectionModel
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

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

def prepare_data():
    """Prepara los datos de entrenamiento"""
    try:
        # Definir rutas absolutas en el directorio de datos
        phrases_path = '/app/data/raw/training_phrases.csv'
        bullying_path = '/app/data/raw/Bullying_2018_copy_traducido_final.csv'
        
        print(f"Intentando cargar archivos desde:")
        print(f"Phrases: {phrases_path}")
        print(f"Bullying: {bullying_path}")
        
        # Cargar datos de entrenamiento
        phrases_df = pd.read_csv(phrases_path, sep=';')
        print(f"Datos de frases cargados: {len(phrases_df)} registros")
        
        # Cargar datos de la encuesta
        bullying_df = pd.read_csv(bullying_path, sep=';')
        print(f"Datos de encuesta cargados: {len(bullying_df)} registros")
        
        # Procesar datos de la encuesta
        bullying_texts = []
        bullying_labels = []
        
        for _, row in bullying_df.iterrows():
            text_parts = []
            is_bullying = 0
            
            if row['Intimidado en la propiedad escolar en los últimos 12 meses'] == 'Sí':
                is_bullying = 1
                text_parts.append("He sido intimidado en la escuela")
            
            if row['Acosado cibernético en los últimos 12 meses'] == 'Sí':
                is_bullying = 1
                text_parts.append("He sufrido acoso por internet")
            
            if row['Atacado físicamente'] != '0 veces':
                is_bullying = 1
                text_parts.append(f"Me han atacado físicamente {row['Atacado físicamente']}")
            
            if text_parts:
                bullying_texts.append(' y '.join(text_parts))
                bullying_labels.append(is_bullying)
        
        print(f"Textos de bullying procesados: {len(bullying_texts)}")
        
        # Combinar datos
        X = list(phrases_df['texto']) + bullying_texts
        y = list(phrases_df['es_acoso']) + bullying_labels
        
        # Balancear el conjunto de datos
        X, y = balance_dataset(X, y)
        
        print(f"Total de ejemplos de entrenamiento después del balanceo: {len(X)}")
        return X, y
        
    except FileNotFoundError as e:
        print(f"Error: No se pudo encontrar el archivo: {e}")
        raise
    except Exception as e:
        print(f"Error inesperado: {e}")
        raise

def main():
    """Función principal para entrenar el modelo"""
    try:
        print("\n=== Iniciando proceso de entrenamiento ===")
        
        # Crear/verificar directorio de modelos
        models_dir = '/app/data/models'
        print(f"Directorio de modelos creado/verificado: {models_dir}")
        
        # Preparar datos
        X, y = prepare_data()
        
        # Balancear dataset
        X_balanced, y_balanced = balance_dataset(X, y)
        print(f"\nTotal de ejemplos de entrenamiento después del balanceo: {len(X_balanced)}")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42
        )
        print(f"\nConjunto de entrenamiento: {len(X_train)} ejemplos")
        print(f"Conjunto de prueba: {len(X_test)} ejemplos")
        
        # Crear y entrenar modelo
        model = BullyingDetectionModel()
        model.train(X_train, y_train)
        
        # Evaluar modelo
        print("\n=== Evaluación del modelo ===")
        report = model.evaluate(X_test, y_test)
        print(report)
        
        # Validación cruzada
        print("\n=== Validación cruzada ===")
        cv_scores = cross_val_score(
            model.model, model.vectorizer.transform(X_balanced), y_balanced,
            cv=5, scoring='accuracy'
        )
        print(f"Precisión media: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Guardar modelo
        print("\n=== Guardando modelo ===")
        model.save_model(path=models_dir)
        
        # Cargar y verificar modelo
        print("\n=== Verificando modelo guardado ===")
        model.load_model()
        print("Modelo cargado correctamente")
        
        # Generar gráficos de métricas
        print("\n=== Generando gráficos ===")
        y_pred = model.model.predict(model.vectorizer.transform(X_test))
        y_prob = model.model.predict_proba(model.vectorizer.transform(X_test))[:, 1]
        plot_metrics(y_test, y_pred, y_prob)
        
        # Prueba con un ejemplo individual
        texto = "Me siento mal porque mis compañeros me molestan en el recreo"
        prediction, probability = model.predict(texto)
        print(f"\nPrueba con texto individual:")
        print(f"Texto: {texto}")
        print(f"Predicción: {prediction}")
        print(f"Probabilidad: {probability}")
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        raise

def plot_metrics(y_test, y_pred, y_prob):
    """Genera y guarda gráficos de métricas de evaluación"""
    try:
        # Crear directorio para gráficos si no existe
        plots_dir = '/app/data/plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusión')
        plt.colorbar()
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, 'roc_curve.png'))
        plt.close()
        
        print(f"\nGráficos guardados en: {plots_dir}")
        
    except Exception as e:
        print(f"Error al generar gráficos: {e}")

if __name__ == "__main__":
    main() 