import pandas as pd
import random
from pathlib import Path

# Function to generate more training examples
def generate_more_examples():
    # Bullying examples with more variety
    bullying_examples = [
        # Cyberbullying examples
        ("Recibo mensajes amenazantes en mis redes sociales", "cibernético"),
        ("Crearon un perfil falso para burlarse de mí", "cibernético"),
        ("Compartieron mis fotos privadas sin permiso", "cibernético"),
        ("Me dejan comentarios ofensivos en mis publicaciones", "cibernético"),
        ("Me acosan por mensajes de texto constantemente", "cibernético"),
        ("Me graban sin mi consentimiento para burlarse de mí", "cibernético"),
        ("Me envían mensajes ofensivos por WhatsApp", "cibernético"),
        ("Crearon un grupo para hablar mal de mí", "cibernético"),
        ("Me etiquetan en fotos humillantes a propósito", "cibernético"),
        ("Me amenazan con publicar información personal", "cibernético"),
        
        # Physical bullying examples
        ("Me empujan contra el casillero a diario", "físico"),
        ("Me quitan mis cosas y las esconden", "físico"),
        ("Me dan golpes cuando nadie está mirando", "físico"),
        ("Me pellizcan o empujan en los pasillos", "físico"),
        ("Me lanzan objetos en clase para molestarme", "físico"),
        ("Me dan patadas en el recreo", "físico"),
        ("Me empujan en las escaleras", "físico"),
        ("Me quitan el dinero del almuerzo", "físico"),
        ("Me rompen mis útiles escolares a propósito", "físico"),
        ("Me escupen cuando paso por su lado", "físico"),
        
        # Verbal bullying examples
        ("Me insultan por mi apariencia física", "verbal"),
        ("Se burlan de mi forma de vestir", "verbal"),
        ("Me dicen apodos ofensivos", "verbal"),
        ("Se ríen de mí por mis calificaciones", "verbal"),
        ("Me gritan cosas ofensivas en el pasillo", "verbal"),
        ("Se burlan de mi forma de hablar", "verbal"),
        ("Me dicen que no sirvo para nada", "verbal"),
        ("Se ríen de mi familia", "verbal"),
        ("Me dicen que nadie me quiere", "verbal"),
        ("Se burlan de mis gustos musicales", "verbal"),
        
        # Social bullying examples
        ("Nadie me habla en la escuela", "social"),
        ("Me excluyen de los trabajos en equipo", "social"),
        ("Se sientan lejos de mí a propósito", "social"),
        ("Hacen fiestas y no me invitan", "social"),
        ("Se hacen los que no me escuchan cuando hablo", "social"),
        ("Se ríen cuando paso por su lado", "social"),
        ("Me ignoran en el grupo de WhatsApp", "social"),
        ("No me dejan jugar con ellos", "social"),
        ("Hacen chismes falsos sobre mí", "social"),
        ("Me hacen quedar mal frente a los demás", "social"),
        
        # Psychological bullying examples
        ("Me amenazan con hacerme daño si cuento algo", "psicológico"),
        ("Me hacen sentir que no valgo nada", "psicológico"),
        ("Me manipulan para que haga cosas que no quiero", "psicológico"),
        ("Me hacen sentir culpable por todo", "psicológico"),
        ("Me amenazan con lastimar a mi familia", "psicológico"),
        ("Me hacen sentir que merezco el maltrato", "psicológico"),
        ("Me amenazan con difundir secretos míos", "psicológico"),
        ("Me hacen sentir miedo de ir a la escuela", "psicológico"),
        ("Me amenazan con lastimar a mis mascotas", "psicológico"),
        ("Me hacen sentir que no tengo amigos", "psicológico")
    ]
    
    # Non-bullying examples
    non_bullying_examples = [
        "Me llevo bien con mis compañeros de clase",
        "Ayer jugué fútbol con mis amigos en el recreo",
        "La maestra me felicitó por mi participación",
        "Hice un nuevo amigo en el taller de arte",
        "Me gusta trabajar en equipo con mis compañeros",
        "Hoy almorcé con mis amigos en la escuela",
        "Mis compañeros me ayudaron con la tarea de matemáticas",
        "Me siento cómodo en mi salón de clases",
        "Ayer tuve una discusión amistosa con un compañero",
        "Me gusta participar en las actividades escolares",
        "Mis amigos me invitaron a su grupo de estudio",
        "Comparto mis útiles con quien los necesite",
        "Me siento respetado por mis compañeros",
        "Ayer me reí mucho con mis amigos en el recreo",
        "Me gusta ayudar a los nuevos estudiantes"
    ]
    
    # Create DataFrames
    bullying_df = pd.DataFrame([{"texto": text, "es_acoso": 1, "tipo_acoso": tipo} 
                              for text, tipo in bullying_examples])
    
    non_bullying_df = pd.DataFrame([{"texto": text, "es_acoso": 0, "tipo_acoso": "ninguno"} 
                                   for text in non_bullying_examples])
    
    # Combine and shuffle
    combined_df = pd.concat([bullying_df, non_bullying_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return combined_df

def main():
    # Define paths
    data_dir = Path("/home/sublime-dev/dev/python/Almabot/modelo_pln/data/raw/")
    output_path = data_dir / "enhanced_training_data.csv"
    
    # Generate new examples
    print("Generating enhanced training data...")
    new_data = generate_more_examples()
    
    # Load existing data
    try:
        existing_data = pd.read_csv(data_dir / "training_phrases.csv", sep=";")
        # Combine with existing data
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        # Remove duplicates
        combined_data = combined_data.drop_duplicates(subset=['texto'])
        print(f"Combined with existing data. Total unique examples: {len(combined_data)}")
    except FileNotFoundError:
        combined_data = new_data
        print("No existing training data found. Creating new dataset.")
    
    # Save the enhanced dataset
    combined_data.to_csv(output_path, sep=";", index=False, encoding='utf-8')
    print(f"Enhanced training data saved to {output_path}")
    
    # Print some statistics
    print("\nDataset statistics:")
    print(f"Total examples: {len(combined_data)}")
    print(f"Bullying examples: {len(combined_data[combined_data['es_acoso'] == 1])}")
    print(f"Non-bullying examples: {len(combined_data[combined_data['es_acoso'] == 0])}")
    print("\nBullying types distribution:")
    print(combined_data[combined_data['tipo_acoso'] != 'ninguno']['tipo_acoso'].value_counts())

if __name__ == "__main__":
    main()
