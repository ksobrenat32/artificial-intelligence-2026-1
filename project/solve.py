import os
import torch
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

class DiversIA_AI_Analyzer:
    """
    Advanced class for DiversIA using Transformers and PyTorch or embeddings+trainable classifier.
    """

    def __init__(self, model_path="model-bias-detection", debiasing_model_path="model-debiasing"):
        print("Loading fine-tuned AI models...")
        self.model_path = Path(model_path)
        self.debiasing_model_path = Path(debiasing_model_path)
        self.clasificador = None
        self.debiasing_pipeline = None
        self.label_info = {}
        
        # Load the fine-tuned bias-detection model
        if self.model_path.exists():
            try:
                self.clasificador = pipeline(
                    "text-classification",
                    model=str(self.model_path),
                    tokenizer=str(self.model_path),
                    device=0 if torch.cuda.is_available() else -1,
                    top_k=None  # retorna todas las etiquetas con scores
                )
                print(f"✓ Bias-detection model loaded from: {self.model_path.resolve()}")
            except Exception as e:
                print(f"Error loading bias-detection model: {e}")
                self.clasificador = None
        else:
            print(f"⚠️ Bias-detection model not found at {self.model_path.resolve()}")
            print("   Run first: python3 training.py")
            self.clasificador = None
        
        # Load debiasing correction model
        if self.debiasing_model_path.exists():
            try:
                self.debiasing_pipeline = pipeline(
                    "text2text-generation",
                    model=str(self.debiasing_model_path),
                    tokenizer=str(self.debiasing_model_path),
                    device=0 if torch.cuda.is_available() else -1
                )
                print(f"✓ Debiasing model loaded from: {self.debiasing_model_path.resolve()}")
            except Exception as e:
                print(f"Error loading debiasing model: {e}")
                self.debiasing_pipeline = None
        else:
            print(f"⚠️ Debiasing model not found at {self.debiasing_model_path.resolve()}")
            self.debiasing_pipeline = None


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
    
    def generate_unbiased_phrase(self, biased_phrase):
        """
        Generates an unbiased version of the phrase using the trained model.
        """
        if not self.debiasing_pipeline or not biased_phrase.strip():
            return None
        
        try:
            # Prepare input with the prefix expected by the model
            input_text = f"debias: {biased_phrase.strip()}"
            
            # Generate corrected text
            resultado = self.debiasing_pipeline(
                input_text,
                max_new_tokens=60,
                min_length=5,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                early_stopping=True,
                pad_token_id=self.debiasing_pipeline.tokenizer.eos_token_id
            )
            
            if resultado and len(resultado) > 0:
                texto_generado = resultado[0]['generated_text'].strip()
                
                # Clean the prefix if present
                if texto_generado.lower().startswith('debias:'):
                    texto_generado = texto_generado[7:].strip()
                
                # Check that it is different from the original
                if (len(texto_generado) > 5 and 
                    texto_generado.lower() != biased_phrase.lower()):
                    return texto_generado
                    
        except Exception as e:
            print(f"⚠️ Error al generar sugerencia: {str(e)}")
        
        return None

    def evaluate_job_posting(self, full_text):
        """
        Splits the job posting into sentences and analyzes each semantically.
        """
        print(f"\n--- Starting Analysis with AI (Deep Learning) ---")
        sentences = [s.strip() for s in full_text.split('.') if s.strip()]
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
    ia_diversia = DiversIA_AI_Analyzer(model_path="model-bias-detection", debiasing_model_path="model-debiasing")
    
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