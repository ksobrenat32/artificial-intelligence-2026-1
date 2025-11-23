import torch
from transformers import pipeline

class DiversIA_AI_Analyzer:
    """
    Clase avanzada para DiversIA usando Transformers y PyTorch.
    Implementa la lógica descrita en la fase de diseño para clasificación de sesgos.
    """

    def __init__(self):
        print("Cargando modelos de Inteligencia Artificial... (esto puede tardar la primera vez)")
        
        # 1. Cargar Pipeline de Clasificación (Zero-Shot)
        # Usamos un modelo multilingüe capaz de entender español sin entrenamiento adicional.
        # Modelo: joeddav/xlm-roberta-large-xnli (Excelente para clasificación en español)
        try:
            self.clasificador = pipeline(
                "zero-shot-classification", 
                model="nahiar/zero-shot-classification",
                device=0 if torch.cuda.is_available() else -1 # Usa GPU si está disponible
            )
        except Exception as e:
            print(f"Error cargando el modelo: {e}")
            print("Asegúrate de tener conexión a internet para descargar los pesos del modelo.")

        # Definimos las etiquetas que queremos que la IA busque en el texto
        self.etiquetas_sesgo = [
            "discriminación de género", 
            "discriminación por edad", 
            "discriminación racial",
            "lenguaje inclusivo y neutral",
            "requisitos laborales técnicos"
        ]

    def analizar_frase(self, frase):
        """
        Usa el modelo Transformer para predecir si una frase contiene sesgo.
        """
        if not frase.strip():
            return None

        # El modelo calcula la probabilidad de que la frase pertenezca a cada etiqueta
        resultado = self.clasificador(
            frase, 
            candidate_labels=self.etiquetas_sesgo,
            multi_label=True # Una frase puede tener múltiples tipos de sesgo
        )
        
        return resultado

    def evaluar_vacante(self, texto_completo):
        """
        Divide la vacante en oraciones y analiza cada una semánticamente.
        """
        print(f"\n--- Iniciando Análisis con IA (Deep Learning) ---")
        
        # Separación simple por puntos (se puede mejorar con NLTK/Spacy)
        oraciones = texto_completo.split('.')
        
        hallazgos = []

        for oracion in oraciones:
            if len(oracion) < 10: continue # Ignorar oraciones muy cortas
            
            resultado = self.analizar_frase(oracion)
            
            # Filtramos: Si la IA está más de 60% segura de que es discriminación
            scores = resultado['scores']
            labels = resultado['labels']
            
            # Buscamos la etiqueta con mayor puntaje
            top_score = scores[0]
            top_label = labels[0]

            # Lógica de detección: Si detecta discriminación con > 50% de confianza
            if "discriminación" in top_label and top_score > 0.5:
                hallazgo = {
                    "texto": oracion.strip(),
                    "tipo_sesgo": top_label,
                    "confianza": f"{top_score:.2%}" # Formato porcentaje
                }
                hallazgos.append(hallazgo)

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
    # Texto de ejemplo basado en los problemas detectados en el documento [cite: 18]
    vacante_test = """
    Buscamos un joven dinámico para el puesto de ventas.
    Se requiere nacionalidad mexicana y buena presencia.
    El candidato ideal debe tener conocimientos en Python.
    """

    # Instancia y ejecución
    ia_diversia = DiversIA_AI_Analyzer()
    ia_diversia.evaluar_vacante(vacante_test)
