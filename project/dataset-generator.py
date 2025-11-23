import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables de .env
load_dotenv()

# Cargar API Key desde variable de entorno
OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")

class SyntheticDatasetGenerator:
    """
    Genera vacantes laborales sintéticas usando OpenAI API.
    Crea datos balanceados para entrenar modelos de detección de sesgos.
    """
    
    def __init__(self, key=None):
        self.client = OpenAI(api_key=key)
        
        self.categorias = [
            "sesgo_genero_masculino",
            "sesgo_genero_femenino",
            "sesgo_edad_joven",
            "sesgo_edad_mayor",
            "sesgo_nacionalidad",
            "neutral_inclusivo"
        ]

    def generar_lote(self, categoria, cantidad=5):
        """
        Solicita a OpenAI un lote de ejemplos para una categoría específica.
        Retorna una lista de diccionarios.
        """
        print(f"🤖 Generando {cantidad} ejemplos para: {categoria}...")

        prompt_system = """
        Eres un experto en Recursos Humanos y Ética.
        Tu tarea es generar ejemplos de fragmentos de ofertas de trabajo en ESPAÑOL 
        para entrenar un modelo de detección de sesgos.
        """

        prompt_user = f"""
        Genera {cantidad} ejemplos de frases cortas de vacantes laborales que cumplan estrictamente con esta categoría: '{categoria}'.
        
        Formato de salida requerido (JSON puro, una lista de objetos):
        [
            {{"texto": "ejemplo 1...", "etiqueta": "{categoria}", "explicacion": "por qué es este sesgo"}},
            {{"texto": "ejemplo 2...", "etiqueta": "{categoria}", "explicacion": "por qué es este sesgo"}}
        ]
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt_user}
                ],
                response_format={"type": "json_object"}
            )

            # Parsear la respuesta JSON
            contenido = response.choices[0].message.content
            datos = json.loads(contenido)
            
            # Aseguramos que sea una lista (la clave suele variar si no especificamos, 
            # pero al pedir lista de objetos en el prompt, GPT suele envolverlo en un dict)
            # Intentamos extraer la lista de valores si viene dentro de una key como "ejemplos"
            if isinstance(datos, dict):
                # Busca cualquier lista dentro del diccionario
                for key, value in datos.items():
                    if isinstance(value, list):
                        return value
                return [] # Si no encuentra lista
            return datos

        except Exception as e:
            print(f"❌ Error generando lote: {e}")
            return []

    def crear_dataset_completo(self, ejemplos_por_categoria=10):
        """
        Orquesta la generación de todas las categorías y guarda el CSV.
        """
        dataset_final = []

        for cat in self.categorias:
            datos = self.generar_lote(cat, ejemplos_por_categoria)
            dataset_final.extend(datos)

        # Convertir a DataFrame
        df = pd.DataFrame(dataset_final)
        
        # Mapear etiquetas textuales a IDs numéricos (útil para PyTorch después)
        label_map = {cat: i for i, cat in enumerate(self.categorias)}
        df['label_id'] = df['etiqueta'].map(label_map)

        # Guardar
        archivo_salida = "dataset_sintetico_diversia.csv"
        df.to_csv(archivo_salida, index=False, encoding='utf-8')
        
        print(f"\n✅ ¡Éxito! Dataset generado con {len(df)} registros.")
        print(f"📁 Guardado en: {archivo_salida}")
        print("\nMuestra de datos:")
        print(df[['etiqueta', 'texto']].head())

# --- Ejecución ---
if __name__ == "__main__":
    # Recuerda poner tu API Key aquí si no está en variables de entorno
    # api_key = "sk-..." 
    
    # Instanciamos (asumiendo que la key está en entorno o pasada como argumento)
    generador = SyntheticDatasetGenerator(OPEN_AI_API_KEY) # Pasa api_key="sk-..." si es necesario
    
    # Generamos 10 ejemplos por cada una de las 6 categorías (Total 60 ejemplos rápidos)
    generador.crear_dataset_completo(ejemplos_por_categoria=10)