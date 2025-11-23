import os
import torch
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

class DiversIA_AI_Analyzer:
    """
    Clase avanzada para DiversIA usando Transformers y PyTorch o embeddings+clasificador.
    model_type: 'zero-shot' (usa pipeline HF) o 'embed' (usa sentence-transformers + classifier entrenable).
    """

    def __init__(self, model_path="my_model"):
        print("Cargando modelo de Inteligencia Artificial fine-tuneado...")
        self.model_path = Path(model_path)
        self.clasificador = None
        self.label_info = {}
        
        # Intentar cargar el modelo fine-tuneado
        if self.model_path.exists():
            try:
                self.clasificador = pipeline(
                    "text-classification",
                    model=str(self.model_path),
                    tokenizer=str(self.model_path),
                    device=0 if torch.cuda.is_available() else -1,
                    top_k=None  # retorna todas las etiquetas con scores
                )
                print(f"✓ Modelo cargado desde: {self.model_path.resolve()}")
            except Exception as e:
                print(f"Error cargando modelo fine-tuneado: {e}")
                self.clasificador = None
        else:
            print(f"⚠️ Modelo no encontrado en {self.model_path.resolve()}")
            print("   Ejecuta primero: python3 training.py")
            self.clasificador = None





    def load_label_info(self, csv_path, label_col="etiqueta", id_col="label_id", explain_col="explicacion"):
        """
        Carga metadatos de etiquetas desde el CSV para enriquecer las predicciones.
        """
        df = pd.read_csv(csv_path, encoding="utf-8")
        info = {}
        
        if label_col not in df.columns:
            print(f"Advertencia: columna '{label_col}' no encontrada. Usando etiquetas del modelo.")
            return {}

        for _, row in df.drop_duplicates(subset=[label_col]).iterrows():
            raw = str(row[label_col]).strip()
            key = raw  # mantener el formato original del modelo
            
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
        print(f"Cargada información de {len(info)} etiquetas")
        return info



    def predict_text(self, texto):
        """
        Usa el modelo fine-tuneado para clasificar el texto.
        Retorna dict con 'labels' y 'scores'.
        """
        texto = str(texto).strip()
        if not texto or self.clasificador is None:
            return None

        # Pipeline con top_k=None retorna [[{'label': 'X', 'score': Y}, ...]]
        results = self.clasificador(texto)
        
        # Extraer la primera lista (resultados para el primer texto)
        if isinstance(results, list) and len(results) > 0:
            results = results[0] if isinstance(results[0], list) else results
        
        # Convertir a formato compatible
        labels = [r['label'] for r in results]
        scores = [r['score'] for r in results]
        
        return {"labels": labels, "scores": scores}

    def analizar_frase(self, frase):
        """
        Usa el método predict_text para obtener etiquetas y scores.
        """
        if not frase.strip():
            return None
        resultado = self.predict_text(frase)
        return resultado

    def evaluar_vacante(self, texto_completo):
        """
        Divide la vacante en oraciones y analiza cada una semánticamente.
        """
        print(f"\n--- Iniciando Análisis con IA (Deep Learning) ---")
        oraciones = [s.strip() for s in texto_completo.split('.') if s.strip()]
        hallazgos = []

        for oracion in oraciones:
            if len(oracion) < 10:
                continue

            resultado = self.analizar_frase(oracion)
            if not resultado:
                continue

            labels = resultado.get("labels", [])
            scores = resultado.get("scores", [])
            if not labels or not scores:
                continue

            # Umbral de confianza
            umbral = 0.40
            for lbl, sc in zip(labels, scores):
                if sc < umbral:
                    continue

                # Buscar info de la etiqueta
                info = self.label_info.get(lbl, {})
                display = info.get('display', lbl.replace('_', ' '))
                explain = info.get('explain', '')

                hallazgos.append({
                    "texto": oracion,
                    "tipo_sesgo": display,
                    "confianza": f"{sc:.2%}",
                    "explicacion": explain
                })
                break

        self._mostrar_resultados(hallazgos)

    def _mostrar_resultados(self, hallazgos):
        if not hallazgos:
            print("✅ El modelo no detectó sesgos evidentes con alta probabilidad.")
        else:
            print(f"⚠️ Se detectaron {len(hallazgos)} frases potencialmente sesgadas:\n")
            for i, h in enumerate(hallazgos, 1):
                print(f"Hallazgo #{i}")
                print(f"  📝 Frase: '{h['texto']}'")
                print(f"  🤖 Categoría detectada por IA: {h['tipo_sesgo'].upper()}")
                print(f"  📊 Nivel de Confianza: {h['confianza']}")
                print("-" * 50)

# --- Bloque de Ejecución ---
if __name__ == "__main__":
    vacante_test = """
¡Buscamos "Rockstars" de las Ventas para unirse a nuestra Tribu! 🚀

¿Tienes hambre de éxito y energía inagotable? En TechNova Solutions no buscamos empleados, buscamos guerreros que quieran comerse el mundo. Somos una empresa líder en el sector con un ambiente Work Hard, Play Hard.

¿Quién eres tú?

Eres un nativo digital; la tecnología corre por tus venas y no necesitas manuales para entender las nuevas tendencias.

Tienes una imagen impecable y profesional; sabes que la primera impresión es la que cuenta para cerrar tratos.

Posees un carácter fuerte, competitivo y dominante. No te asusta la presión ni los retos agresivos.

Tienes disponibilidad total (24/7); aquí la pasión no tiene horario de salida.

Requisitos Indispensables:

Español nativo (lengua materna): Buscamos a alguien que domine a la perfección los modismos locales y la cultura del país para conectar genuinamente con el cliente (abstenerse si no se cumple este requisito al 100%).

Egresado de universidades de prestigio nacionales.

Experiencia máxima de 5 a 8 años (queremos moldearte a nuestro estilo, sin vicios de la vieja escuela).

Fit cultural: Te encantan los after-office, las cervezas los viernes y el ambiente de camaradería intensa.

Ofrecemos:

Sueldo competitivo + comisiones agresivas.

Mesa de ping-pong, FIFA y cerveza ilimitada.

Un equipo joven y dinámico donde nunca te aburrirás.
    """

    # Usar el modelo fine-tuneado (debe existir el directorio my_model/)
    ia_diversia = DiversIA_AI_Analyzer(model_path="my_model")
    
    if ia_diversia.clasificador is None:
        print("\n⚠️  No hay modelo disponible. Para entrenar el modelo:")
        print("    python3 training.py")
        exit(1)

    # Cargar metadatos de etiquetas para presentación mejorada
    try:
        ia_diversia.load_label_info("dataset.csv")
    except Exception as e:
        print(f"Advertencia: no se cargó label_info: {e}")

    # Evaluar vacante de prueba
    ia_diversia.evaluar_vacante(vacante_test)

    # Analizar muestra del CSV (primeras 10 filas para no saturar)
    print("\n=== Análisis de muestra del dataset ===")
    try:
        df = pd.read_csv("dataset.csv", encoding="utf-8")
        for idx, row in df.head(10).iterrows():
            texto = str(row.get("texto", "")).strip()
            if not texto:
                continue
            resultado = ia_diversia.predict_text(texto)
            if resultado:
                labels = resultado.get("labels", [])
                scores = resultado.get("scores", [])
                if labels and scores:
                    lbl = labels[0]
                    info = ia_diversia.label_info.get(lbl, {})
                    display = info.get('display', lbl.replace('_', ' '))
                    print(f"Fila {idx}: {display} ({scores[0]:.2%})")
    except FileNotFoundError:
        print("Dataset no encontrado.")
