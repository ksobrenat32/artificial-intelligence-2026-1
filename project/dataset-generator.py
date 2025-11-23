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

    def generate_examples(self, category, quantity=10):
        """
        Ask OpenAI for a batch of examples for a specific category.
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
                print(f"   📝 Raw response content: {content[:1000]}...")  # Print first 1000 chars for debugging
                data = json.loads(content)
            except json.JSONDecodeError:
                print("❌ Error: Response is not valid JSON.")
                return []
            
            # If the response is a dict with a list inside, extract it
            if isinstance(data, dict):
                # Check if it's a single example (has 'texto', 'etiqueta', 'explicacion' keys)
                if 'texto' in data and 'etiqueta' in data and 'explicacion' in data:
                    print(f"   ✓ Generated 1 example (single object)")
                    return [data]  # Wrap single object in a list
                
                # Check for 'ejemplos' key specifically
                if 'ejemplos' in data and isinstance(data['ejemplos'], list):
                    print(f"   ✓ Generated {len(data['ejemplos'])} examples")
                    return data['ejemplos']
                
                # Check if the dict has numeric string keys (like '0', '1', etc.) - extract values
                if all(key.isdigit() for key in data.keys()):
                    values = list(data.values())
                    print(f"   ✓ Generated {len(values)} examples (from numeric keys)")
                    return values
                
                # Otherwise, look for any list inside the dict
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

    def create_complete_dataset(self, examples_per_category=10):
        """
        Orchestrates the generation of all categories and saves the CSV.
        Appends new records to existing dataset if file exists.
        """
        dataset_final = []

        # Generate in batches of 10 to avoid overwhelming the API
        batch_size = 10
        for cat in self.categories:
            remaining = examples_per_category
            while remaining > 0:
                batch = min(batch_size, remaining)
                data = self.generate_examples(cat, batch)
                dataset_final.extend(data)
                remaining -= batch

        # Convert new data to DataFrame
        df_new = pd.DataFrame(dataset_final)
        
        # Map textual labels to numeric IDs (useful for PyTorch later)
        label_map = {cat: i for i, cat in enumerate(self.categories)}
        df_new['label_id'] = df_new['etiqueta'].map(label_map)

        # Check if dataset file already exists
        output_file = "dataset.csv"
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

        # Ensure all records have correct label_id mapping
        df['label_id'] = df['etiqueta'].map(label_map)
        
        # Sort by label_id for better organization
        df = df.sort_values(by='label_id', ignore_index=True)

        # Save to CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n✅ Success! Dataset saved with {len(df)} total records.")
        print(f"📁 Saved to: {output_file}")
        print("\nSample of newly generated data:")
        print(df_new[['etiqueta', 'texto']].head())

if __name__ == "__main__":
    generador = SyntheticDatasetGenerator(OPEN_AI_API_KEY)
    generador.create_complete_dataset(examples_per_category=100)