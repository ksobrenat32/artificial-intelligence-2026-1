import os
import torch
import pandas as pd
import numpy as np
import pickle
import re
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from pathlib import Path

class DiversIA_AI_Analyzer:
    """
    Advanced class for DiversIA using Transformers with embeddings and LLM for suggestions.
    """

    def __init__(self, model_path="model-bias-detection", vector_store_path="vector_store", llm_model="Qwen/Qwen2.5-3B-Instruct"):
        print("Loading AI models...")
        self.model_path = Path(model_path)
        self.vector_store_path = Path(vector_store_path)
        self.llm_model_name = llm_model
        self.clasificador = None
        self.embedding_model = None
        self.llm = None
        self.embeddings = None
        self.examples = None
        self.label_info = {}
        
        # Load the fine-tuned bias-detection model
        if self.model_path.exists():
            try:
                self.clasificador = pipeline(
                    "text-classification",
                    model=str(self.model_path),
                    tokenizer=str(self.model_path),
                    device=0 if torch.cuda.is_available() else -1,
                    top_k=None
                )
                print(f"✓ Bias-detection model loaded from: {self.model_path.resolve()}")
            except Exception as e:
                print(f"Error loading bias-detection model: {e}")
                self.clasificador = None
        else:
            print(f"⚠️ Bias-detection model not found at {self.model_path.resolve()}")
            print("   Run first: python3 training.py")
            self.clasificador = None
        
        # Load vector store for debiasing
        if self.vector_store_path.exists():
            try:
                embeddings_path = self.vector_store_path / "embeddings.npy"
                self.embeddings = np.load(embeddings_path)
                
                # Load examples
                store_path = self.vector_store_path / "store.pkl"
                with open(store_path, 'rb') as f:
                    self.examples = pickle.load(f)
                
                # Load metadata
                metadata_path = self.vector_store_path / "metadata.pkl"
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Load embedding model
                embedding_model_name = metadata.get('embedding_model', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
                self.embedding_model = SentenceTransformer(embedding_model_name)
                
                print(f"✓ Vector store loaded: {len(self.examples)} examples")
                print(f"  Embedding model: {embedding_model_name}")
            except Exception as e:
                print(f"⚠️ Error loading vector store: {e}")
                self.embeddings = None
                self.examples = None
        else:
            print(f"⚠️ Vector store not found at {self.vector_store_path.resolve()}")
            print("   Run: python3 training.py to generate it")
        
        # Load LLM for generating unbiased text
        if self.embeddings is not None and self.examples is not None:
            try:
                print(f"Loading LLM: {self.llm_model_name}...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.llm = pipeline(
                    "text-generation",
                    model=self.llm_model_name,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    max_new_tokens=150,
                )
                print(f"✓ LLM loaded successfully ({device})")
            except Exception as e:
                print(f"⚠️ Error loading LLM: {e}")
                print("   Trying with smaller model...")
                try:
                    self.llm = pipeline(
                        "text-generation",
                        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        device_map="auto" if torch.cuda.is_available() else None,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        max_new_tokens=150,
                    )
                    print("✓ Fallback LLM loaded (TinyLlama)")
                except Exception as e2:
                    print(f"⚠️ Could not load any LLM: {e2}")
                    self.llm = None


    def load_label_info(self, csv_path, label_col="etiqueta", id_col="label_id", explain_col="explicacion"):
        """
        Load label metadata from CSV to enrich predictions.
        """
        df = pd.read_csv(csv_path, encoding="utf-8")
        info = {}
        
        if label_col not in df.columns:
            print(f"Advertencia: columna '{label_col}' no encontrada. Usando etiquetas del modelo.")
            return {}

        for _, row in df.drop_duplicates(subset=[label_col]).iterrows():
            raw = str(row[label_col]).strip()
            key = raw
            
            # Display legible
            if "genero" in key:
                display = "discriminación de género"
            elif "edad" in key:
                display = "discriminación por edad"
            elif "nacionalidad" in key:
                display = "discriminación por nacionalidad"
            elif "neutral" in key or "inclusivo" in key:
                display = "lenguaje inclusivo"
            else:
                display = raw.replace("_", " ").strip()

            explain = str(row[explain_col]).strip() if explain_col in df.columns and not pd.isna(row.get(explain_col)) else ""
            
            info[key] = {"display": display, "explain": explain}

        self.label_info = info
        print(f"Loaded information for {len(info)} labels from {csv_path}")
        return info

    def classify_text(self, text):
        """
        Uses the fine-tuned model to classify the text.
        Returns a dict with 'labels' and 'scores'.
        """
        text = str(text).strip()
        if not text or self.clasificador is None:
            return None

        results = self.clasificador(text)
        
        # Extract the first list (results for the first text)
        if isinstance(results, list) and len(results) > 0:
            results = results[0] if isinstance(results[0], list) else results
        
        # Convert to compatible format
        labels = [r['label'] for r in results]
        scores = [r['score'] for r in results]
        
        return {"labels": labels, "scores": scores}
    
    def find_similar_examples(self, text, top_k=3):
        """
        Find the most similar examples from the vector store.
        """
        if self.embeddings is None or self.examples is None or self.embedding_model is None:
            return []
        
        # Generate embedding for input text
        query_embedding = self.embedding_model.encode([text])[0]
        
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        similar_examples = []
        for idx in top_indices:
            similar_examples.append({
                'biased': self.examples[idx]['texto_sesgado'],
                'corrected': self.examples[idx]['texto_corregido'],
                'similarity': float(similarities[idx]),
                'explanation': self.examples[idx].get('explicacion', '')
            })
        
        return similar_examples

    def generate_unbiased_phrase(self, biased_phrase):
        """
        Generates an unbiased version using LLM with few-shot examples from vector store.
        """
        if not biased_phrase.strip():
            return None
        
        if self.llm is None:
            return None
        
        try:
            # Find similar examples
            similar_examples = self.find_similar_examples(biased_phrase, top_k=3)
            
            if not similar_examples:
                return None
            
            # Build prompt with examples
            prompt = """Eres un experto en lenguaje inclusivo. Tu tarea es reescribir frases sesgadas en lenguaje neutral e inclusivo.
            Ejemplos:
            """
            
            for i, ex in enumerate(similar_examples, 1):
                prompt += f"\nFrase sesgada {i}: {ex['biased']}"
                prompt += f"\nFrase corregida {i}: {ex['corrected']}\n"
            
            prompt += f"\nAhora reescribe esta frase de manera inclusiva:\nFrase sesgada: {biased_phrase}"
            prompt += "\nFrase corregida:"
            
            # Generate with LLM
            result = self.llm(
                prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.llm.tokenizer.eos_token_id if hasattr(self.llm, 'tokenizer') else None
            )
            
            if result and len(result) > 0:
                generated_text = result[0]['generated_text']
                
                # Extract only the answer after the prompt
                if "Frase corregida:" in generated_text:
                    answer = generated_text.split("Frase corregida:")[-1].strip()
                    
                    # Clean up the answer
                    answer = answer.split('\n')[0].strip()
                    
                    # Remove quotes if present
                    if answer.startswith('"') and answer.endswith('"'):
                        answer = answer[1:-1]
                    
                    # Validate it's different and reasonable
                    if (len(answer) > 5 and 
                        answer.lower() != biased_phrase.lower() and
                        len(answer) < len(biased_phrase) * 3):
                        return answer
                    
        except Exception as e:
            print(f"⚠️ Error generating suggestion: {str(e)}")
        
        return None

    def evaluate_job_posting(self, full_text):
        """
        Splits the job posting into sentences and analyzes each semantically.
        Uses newlines and all punctuation marks for better segmentation.
        """
        print(f"\n--- Starting Analysis with AI (Deep Learning) ---")

        # Split by multiple punctuation marks and newlines for better segmentation
        # This regex splits on: periods, exclamation marks, question marks, newlines, semicolons, and colons
        sentence_pattern = r'[.!?;\n:]+'
        sentences = re.split(sentence_pattern, full_text)

        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

        findings = []

        for sentence in sentences:
            if len(sentence) < 10:
                continue

            result = self.classify_text(sentence)
            if not result:
                continue

            labels = result.get("labels", [])
            scores = result.get("scores", [])
            if not labels or not scores:
                continue

            # Confidence threshold
            threshold = 0.60
            for lbl, sc in zip(labels, scores):
                if sc < threshold:
                    continue

                # Search for label info
                info = self.label_info.get(lbl, {})
                display = info.get('display', lbl.replace('_', ' '))
                explain = info.get('explain', '')

                # Generate unbiased suggestion
                suggestion = self.generate_unbiased_phrase(sentence)

                findings.append({
                    "text": sentence,
                    "bias_type": display,
                    "confidence": f"{sc:.2%}",
                    "explanation": explain,
                    "suggestion": suggestion
                })
                break

        self._show_results(findings)

    def _show_results(self, findings):
        if not findings:
            print("✅ The model did not detect any obvious biases with high probability.")
        else:
            print(f"⚠️ Detected {len(findings)} potentially biased phrases:\n")
            for i, h in enumerate(findings, 1):
                print(f"Finding #{i}")
                print(f"  📝 Phrase: '{h['text']}'")
                print(f"  🤖 AI Detected Category: {h['bias_type'].upper()}")
                print(f"  📊 Confidence Level: {h['confidence']}")
                if h.get('suggestion'):
                    print(f"  💡 Unbiased Suggestion: '{h['suggestion']}'")
                else:
                    print(f"  ⚠️ Could not generate automatic suggestion")
                print("-" * 50)

# --- Bloque de Ejecución ---
if __name__ == "__main__":
    # Read test job posting from file
    with open("example.txt", "r", encoding="utf-8") as f:
        vacante_test = f.read()

    # Use the DiversIA_AI_Analyzer class
    # You can change the LLM model here. Options:
    # - "Qwen/Qwen2.5-3B-Instruct" (recommended, good balance)
    # - "meta-llama/Llama-3.2-3B-Instruct" (needs authentication)
    # - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (fast, smaller)
    ia_diversia = DiversIA_AI_Analyzer(
        model_path="model-bias-detection",
        vector_store_path="vector_store",
        llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    if ia_diversia.clasificador is None:
        print("\n⚠️  No model available. To train the model:")
        print("    python3 training.py")
        exit(1)

    # Load label metadata for enhanced presentation
    try:
        ia_diversia.load_label_info("dataset-bias-identification.csv")
    except Exception as e:
        print(f"Warning: label_info not loaded: {e}")

    # Evaluate test job posting
    ia_diversia.evaluate_job_posting(vacante_test)

    # Analyze sample from CSV (Random 10 rows)
    print("\n=== Sample analysis from dataset-bias-identification ===")
    try:
        df = pd.read_csv("dataset-bias-identification.csv", encoding="utf-8")
        for idx, row in df.sample(10).iterrows():
            texto = str(row.get("texto", "")).strip()
            if not texto:
                continue
            resultado = ia_diversia.classify_text(texto)
            if resultado:
                labels = resultado.get("labels", [])
                scores = resultado.get("scores", [])
                if labels and scores:
                    lbl = labels[0]
                    info = ia_diversia.label_info.get(lbl, {})
                    display = info.get('display', lbl.replace('_', ' '))
                    print(f"Row {idx}: {display} ({scores[0]:.2%})")
    except FileNotFoundError:
        print("Dataset not found.")