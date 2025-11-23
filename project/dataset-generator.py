import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Loading environment variables
load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")

class SyntheticDatasetGenerator:
    """
    Generate training dataset for bias detection in job postings.
    """
    
    def __init__(self, key=None):
        self.client = OpenAI(api_key=key)        
        self.categories = [
            "sesgo_genero_masculino",
            "sesgo_genero_femenino",
            "sesgo_edad_joven",
            "sesgo_edad_mayor",
            "sesgo_nacionalidad",
            "neutral_inclusivo"
        ]

    def generate_bias_examples(self, category, quantity=10):
        """
        Ask OpenAI for a batch of bias examples for a specific category.
        Returns a list of dictionaries.
        """
        print(f"🤖 Generating {quantity} examples for: {category}...")

        prompt_system = """
        Eres un experto en Recursos Humanos y Ética.
        Tu tarea es generar ejemplos de fragmentos de ofertas de trabajo en ESPAÑOL para entrenar un modelo de detección de sesgos.
        """

        prompt_user = f"""
        Genera {quantity} ejemplos de frases cortas de vacantes laborales que cumplan estrictamente con esta categoría: '{category}'.
        
        Devuelve un objeto JSON con una clave "ejemplos" que contenga una lista de objetos:
        {{
            "ejemplos": [
                {{"texto": "ejemplo 1...", "etiqueta": "{category}", "explicacion": "por qué es este sesgo"}},
                {{"texto": "ejemplo 2...", "etiqueta": "{category}", "explicacion": "por qué es este sesgo"}}
            ]
        }}
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

            content = response.choices[0].message.content
            
            # Sanitize and parse JSON
            try:
                if content is None:
                    print("❌ Error: Response content is None.")
                    return []
                data = json.loads(content)
            except json.JSONDecodeError:
                print("❌ Error: Response is not valid JSON.")
                return []
            
            # If the response is a dict with a list inside, extract it
            if isinstance(data, dict):
                if 'texto' in data and 'etiqueta' in data and 'explicacion' in data:
                    print(f"   ✓ Generated 1 example (single object)")
                    return [data]  # Wrap single object in a list
                
                if 'ejemplos' in data and isinstance(data['ejemplos'], list):
                    print(f"   ✓ Generated {len(data['ejemplos'])} examples")
                    return data['ejemplos']
                
                if all(key.isdigit() for key in data.keys()):
                    values = list(data.values())
                    print(f"   ✓ Generated {len(values)} examples (from numeric keys)")
                    return values
                
                for key, value in data.items():
                    if isinstance(value, list):
                        print(f"   ✓ Generated {len(value)} examples (from key '{key}')")
                        return value
                print(f"   ⚠️  Warning: Response is a dict but contains no list. Keys: {list(data.keys())}")
                return []
            if isinstance(data, list):
                print(f"   ✓ Generated {len(data)} examples")
                return data
            print(f"   ⚠️  Warning: Unexpected data type: {type(data)}")
            return []

        except Exception as e:
            print(f"❌ Error during OpenAI API call: {e}")
            return []

    def generate_debiasing_examples(self, category, quantity=10):
        """
        Generate examples for debiasing: biased text -> debiased text pairs.
        Returns a list of dictionaries with 'biased_text' and 'debiased_text'.
        """
        print(f"🤖 Generating {quantity} debiasing examples for category '{category}'...")

        if category == "neutral_inclusivo":
            # For neutral category, we want identity mappings (Input == Output)
            # This teaches the model NOT to change text that is already inclusive.
            prompt_system = """
            Eres un experto en Recursos Humanos y Lenguaje Inclusivo.
            Tu tarea es generar ejemplos de frases de ofertas de trabajo que YA son neutrales e inclusivas.
            Para estos casos, el 'texto_sesgado' y el 'texto_corregido' deben ser IDÉNTICOS.
            El objetivo es enseñar al modelo a preservar textos que ya son correctos.
            """
            prompt_user = f"""
            Genera {quantity} ejemplos de frases de ofertas de trabajo en ESPAÑOL que sean perfectamente neutrales e inclusivas.
            Varía los sectores (Tecnología, Salud, Educación, Servicios, etc.).
            
            Devuelve un objeto JSON con una clave "ejemplos" que contenga una lista de objetos:
            {{
                "ejemplos": [
                    {{
                        "texto_sesgado": "frase neutral...",
                        "texto_corregido": "frase neutral...",
                        "tipo_sesgo": "neutral_inclusivo",
                        "explicacion": "El texto ya es neutral, no requiere cambios."
                    }}
                ]
            }}
            """
        else:
            # Enhanced prompt for biased categories with Style Guide and Few-Shot examples
            prompt_system = """
            Eres un experto en Recursos Humanos, Diversidad, Equidad e Inclusión (DEI).
            Tu tarea es crear un dataset de entrenamiento para corregir sesgos en ofertas de empleo.
            Debes generar pares de frases: {Original con Sesgo} -> {Corregida Inclusiva}.

            GUÍA DE ESTILO PARA LA CORRECCIÓN:
            1. Género: Evita el masculino genérico. Usa desdoblamientos (candidatos/as) con moderación, prefiere sustantivos colectivos o abstractos (e.g., "Jefatura" en vez de "Jefe", "el equipo de ingeniería" en vez de "los ingenieros", "persona seleccionada").
            2. Edad: Elimina requisitos de edad numérica o términos como "joven", "recién graduado" si implican edad. Enfócate en experiencia o conocimientos.
            3. Nacionalidad/Origen: Elimina requisitos de nacionalidad específica. Usa "nivel nativo o bilingüe" para idiomas, no para personas.
            4. Tono: Mantén un tono profesional y acogedor.

            La corrección debe preservar estrictamente el requisito técnico o la responsabilidad del puesto, cambiando solo la forma de expresarlo.
            """

            examples_text = ""
            if "genero" in category:
                examples_text = """
                Ejemplos de referencia:
                - Sesgado: "Buscamos un desarrollador experto. El candidato debe..." -> Corregido: "Buscamos profesional de desarrollo experto. La persona seleccionada debe..."
                - Sesgado: "Se necesita secretaria de buena presencia." -> Corregido: "Se necesita asistente administrativo/a con excelente trato al cliente."
                """
            elif "edad" in category:
                examples_text = """
                Ejemplos de referencia:
                - Sesgado: "Empresa joven busca recién titulados menores de 25 años." -> Corregido: "Empresa dinámica busca personas recién tituladas o con formación reciente."
                - Sesgado: "Se requiere carnet de conducir y ser mayor de 30 años." -> Corregido: "Se requiere carnet de conducir y madurez profesional."
                """

            prompt_user = f"""
            Genera {quantity} pares de ejemplos de frases de ofertas de trabajo en ESPAÑOL para la categoría '{category}'.
            Asegúrate de variar los sectores (Tecnología, Ventas, Salud, Construcción, Administración, etc.).
            
            {examples_text}

            Devuelve un objeto JSON con una clave "ejemplos" que contenga una lista de objetos:
            {{
                "ejemplos": [
                    {{
                        "texto_sesgado": "frase con sesgo...",
                        "texto_corregido": "frase corregida...",
                        "tipo_sesgo": "{category}",
                        "explicacion": "breve explicación del cambio"
                    }}
                ]
            }}
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

            content = response.choices[0].message.content
            
            try:
                if content is None:
                    print("❌ Error: Response content is None.")
                    return []
                data = json.loads(content)
            except json.JSONDecodeError:
                print("❌ Error: Response is not valid JSON.")
                return []
            
            if isinstance(data, dict):
                if 'ejemplos' in data and isinstance(data['ejemplos'], list):
                    print(f"   ✓ Generated {len(data['ejemplos'])} debiasing examples")
                    return data['ejemplos']
                
                for key, value in data.items():
                    if isinstance(value, list):
                        print(f"   ✓ Generated {len(value)} debiasing examples (from key '{key}')")
                        return value
                print(f"   ⚠️  Warning: Response is a dict but contains no list. Keys: {list(data.keys())}")
                return []
            if isinstance(data, list):
                print(f"   ✓ Generated {len(data)} debiasing examples")
                return data
            print(f"   ⚠️  Warning: Unexpected data type: {type(data)}")
            return []

        except Exception as e:
            print(f"❌ Error during OpenAI API call: {e}")
            return []

    def create_bias_dataset(self, examples_per_category=10):
        """
        Orchestrates the generation of all categories and saves the CSV.
        Appends new records to existing dataset if file exists.
        """
        dataset_final = []

        batch_size = 10
        for cat in self.categories:
            remaining = examples_per_category
            while remaining > 0:
                batch = min(batch_size, remaining)
                data = self.generate_bias_examples(cat, batch)
                dataset_final.extend(data)
                remaining -= batch

        df_new = pd.DataFrame(dataset_final)
        
        label_map = {cat: i for i, cat in enumerate(self.categories)}
        df_new['label_id'] = df_new['etiqueta'].map(label_map)

        output_file = "dataset-bias-identification.csv"
        if os.path.exists(output_file):
            print(f"📂 Found existing dataset. Loading...")
            df_existing = pd.read_csv(output_file, encoding='utf-8')
            print(f"   Existing records: {len(df_existing)}")
            
            # Append new data to existing data
            df = pd.concat([df_existing, df_new], ignore_index=True)
            print(f"   New records: {len(df_new)}")
            print(f"   Total records: {len(df)}")
        else:
            print(f"📄 No existing dataset found. Creating new file...")
            df = df_new

        df['label_id'] = df['etiqueta'].map(label_map)
        
        df = df.sort_values(by='label_id', ignore_index=True)

        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n✅ Success! Dataset saved with {len(df)} total records.")
        print(f"📁 Saved to: {output_file}")
        print("\nSample of newly generated data:")
        print(df_new[['etiqueta', 'texto']].head())

    def create_debiasing_dataset(self, examples_per_category=10):
        """
        Generate dataset for debiasing model training.
        Creates pairs of biased -> debiased text.
        """
        dataset_final = []

        # Generate in batches of 10 to avoid overwhelming the API
        batch_size = 10
        for cat in self.categories:
            remaining = examples_per_category
            while remaining > 0:
                batch = min(batch_size, remaining)
                data = self.generate_debiasing_examples(cat, batch)
                dataset_final.extend(data)
                remaining -= batch

        df_new = pd.DataFrame(dataset_final)
        
        output_file = "dataset-debiasing.csv"
        if os.path.exists(output_file):
            print(f"📂 Found existing debiasing dataset. Loading...")
            df_existing = pd.read_csv(output_file, encoding='utf-8')
            print(f"   Existing records: {len(df_existing)}")
            
            df = pd.concat([df_existing, df_new], ignore_index=True)
            print(f"   New records: {len(df_new)}")
            print(f"   Total records: {len(df)}")
        else:
            print(f"📄 No existing debiasing dataset found. Creating new file...")
            df = df_new

        # Add label_id mapping
        label_map = {cat: i for i, cat in enumerate(self.categories)}
        if 'tipo_sesgo' in df.columns:
            df['label_id'] = df['tipo_sesgo'].map(label_map)
            df = df.sort_values(by='label_id', ignore_index=True)

        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n✅ Success! Debiasing dataset saved with {len(df)} total records.")
        print(f"📁 Saved to: {output_file}")
        print("\nSample of newly generated data:")
        if not df_new.empty:
            print(df_new[['texto_sesgado', 'texto_corregido']].head())

if __name__ == "__main__":
    generador = SyntheticDatasetGenerator(OPEN_AI_API_KEY)
    #generador.create_bias_dataset(examples_per_category=5)
    generador.create_debiasing_dataset(examples_per_category=5)
