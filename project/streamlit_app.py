import streamlit as st
import re
from solve import DiversIA_AI_Analyzer
import pandas as pd

# Configure Streamlit page
st.set_page_config(
    page_title="DiversIA - Bias Detection & Correction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .bias-card {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .safe-card {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .suggestion-card {
        background-color: #e7f3ff;
        border: 1px solid #b8daff;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_analyzer():
    """Load the DiversIA analyzer with caching for better performance"""
    with st.spinner("Cargando modelos de IA... Esto puede tomar unos minutos la primera vez."):
        analyzer = DiversIA_AI_Analyzer(
            model_path="model-bias-detection",
            llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )

        try:
            analyzer.load_label_info("dataset-bias-identification.csv")
        except Exception as e:
            st.warning(f"No se pudo cargar información de etiquetas: {e}")

        return analyzer

def analyze_single_phrase(analyzer, phrase):
    """Analyze a single phrase for bias"""
    if not phrase.strip():
        return None

    result = analyzer.classify_text(phrase)
    if not result:
        return None

    labels = result.get("labels", [])
    scores = result.get("scores", [])

    if not labels or not scores:
        return None

    threshold = 0.60
    for lbl, sc in zip(labels, scores):
        if sc >= threshold:
            # Get label info
            info = analyzer.label_info.get(lbl, {})
            display = info.get('display', lbl.replace('_', ' '))
            explain = info.get('explain', '')

            suggestion = analyzer.generate_unbiased_phrase(phrase)

            return {
                "text": phrase,
                "bias_type": display,
                "confidence": sc,
                "explanation": explain,
                "suggestion": suggestion,
                "raw_label": lbl
            }

    return {"text": phrase, "bias_type": None, "confidence": 0}

def analyze_full_text(analyzer, full_text):
    """Analyze full text by splitting into sentences"""
    sentence_pattern = r'[.!?;\n:]+'
    sentences = re.split(sentence_pattern, full_text)

    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    findings = []
    for sentence in sentences:
        result = analyze_single_phrase(analyzer, sentence)
        if result and result.get("bias_type"):
            findings.append(result)

    return findings

def main():
    st.markdown('<h1 class="main-header">🤖 DiversIA</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Detección y Corrección de Sesgos en Texto</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("ℹ️ Información")
        st.write("""
        **DiversIA** utiliza modelos de inteligencia artificial para:

        - **Detectar sesgos** en texto
        - **Sugerir correcciones** inclusivas
        - **Mostrar nivel de confianza** de la predicción

        **Tipos de sesgo detectados:**
        - Discriminación de género
        - Discriminación por edad
        - Discriminación por nacionalidad
        - Lenguaje no inclusivo
        """)

        st.header("🎯 Modo de análisis")
        analysis_mode = st.radio(
            "Selecciona el modo:",
            ["Frase individual", "Texto completo"],
            help="Frase individual: analiza una sola oración. Texto completo: divide el texto en oraciones y analiza cada una."
        )

        st.header("⚙️ Configuración")
        confidence_threshold = st.slider(
            "Umbral de confianza",
            min_value=0.0,
            max_value=1.0,
            value=0.60,
            step=0.05,
            help="Mínimo nivel de confianza para mostrar una detección de sesgo"
        )

    try:
        analyzer = load_analyzer()

        if analyzer.clasificador is None:
            st.error("No se pudo cargar el modelo de detección de sesgos.")
            st.info("Ejecuta primero: `python3 training.py` para entrenar el modelo.")
            return

    except Exception as e:
        st.error(f"Error cargando el analizador: {str(e)}")
        return

    st.header("Ingresa tu texto")

    if analysis_mode == "Frase individual":
        user_input = st.text_input(
            "Escribe una frase para analizar:",
            placeholder="Ejemplo: Necesitamos un programador con experiencia...",
            help="Ingresa una sola oración para analizar sesgos"
        )

        if st.button("🔍 Analizar Frase", type="primary"):
            if user_input.strip():
                with st.spinner("Analizando..."):
                    result = analyze_single_phrase(analyzer, user_input)

                    if result and result.get("bias_type") and result["confidence"] >= confidence_threshold:
                        st.markdown("### ⚠️ Sesgo Detectado")

                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.markdown(f'<div class="bias-card">', unsafe_allow_html=True)
                            st.markdown(f"**Frase analizada:** {result['text']}")
                            st.markdown(f"**Tipo de sesgo:** {result['bias_type'].title()}")
                            st.markdown(f"**Confianza:** {result['confidence']:.1%}")
                            # if result.get('explanation'):
                            #     st.markdown(f"**💭 Explicación:** {result['explanation']}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            st.metric("Nivel de Sesgo", f"{result['confidence']:.1%}",
                                     delta=None, delta_color="inverse")

                        if result.get('suggestion'):
                            st.markdown("### 💡 Sugerencia de Corrección")
                            st.markdown(f'''<div class="suggestion-card">
                                <strong>Versión corregida:</strong> {result['suggestion']}
                            </div>''', unsafe_allow_html=True)

                            # Show comparison
                            st.markdown("### 📊 Comparación")
                            comparison_data = {
                                "Versión": ["Original", "Corregida"],
                                "Texto": [result['text'], result['suggestion']]
                            }
                            st.table(pd.DataFrame(comparison_data))
                        else:
                            st.warning("⚠️ No se pudo generar una sugerencia automática.")

                    else:
                        # No bias detected
                        st.markdown("### ✅ Texto Seguro")
                        st.markdown(f'<div class="safe-card">', unsafe_allow_html=True)
                        st.markdown(f"**Frase analizada:** {user_input}")
                        st.markdown("**Resultado:** No se detectaron sesgos significativos en esta frase.")
                        if result:
                            st.markdown(f"**Confianza máxima:** {result['confidence']:.1%}")
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Por favor ingresa una frase para analizar.")

    else:
        user_input = st.text_area(
            "Escribe el texto completo para analizar:",
            height=200,
            placeholder="Ejemplo: Buscamos un programador joven y dinámico para nuestro equipo. Debe ser una persona responsable y comprometida...",
            help="Ingresa un texto completo que será dividido en oraciones para análisis individual"
        )

        if st.button("🔍 Analizar Texto Completo", type="primary"):
            if user_input.strip():
                with st.spinner("Analizando texto completo..."):
                    findings = analyze_full_text(analyzer, user_input)

                    findings = [f for f in findings if f["confidence"] >= confidence_threshold]

                    if findings:
                        st.markdown(f"### ⚠️ Se detectaron {len(findings)} frases con posibles sesgos")

                        for i, finding in enumerate(findings, 1):
                            with st.expander(f"🔍 Hallazgo #{i}: {finding['bias_type'].title()}"):
                                col1, col2 = st.columns([3, 1])

                                with col1:
                                    st.markdown(f"**Frase:** {finding['text']}")
                                    st.markdown(f"**Tipo de sesgo:** {finding['bias_type'].title()}")
                                    # if finding.get('explanation'):
                                    #     st.write(f"**💭 Explicación:** {finding['explanation']}")
                                with col2:
                                    st.metric("Confianza", f"{finding['confidence']:.1%}")

                                if finding.get('suggestion'):
                                    st.markdown("**Sugerencia:**")
                                    st.info(finding['suggestion'])
                                else:
                                    st.warning("No se pudo generar sugerencia automática.")

                        st.markdown("### 📊 Resumen del Análisis")
                        bias_types = [f['bias_type'] for f in findings]
                        bias_counts = pd.Series(bias_types).value_counts()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total de Frases Analizadas", len(re.split(r'[.!?;\n:]+', user_input)))
                        with col2:
                            st.metric("Frases con Sesgo", len(findings))
                        with col3:
                            st.metric("Tipos de Sesgo", len(bias_counts))

                        if len(bias_counts) > 0:
                            st.markdown("**Distribución de tipos de sesgo:**")
                            st.bar_chart(bias_counts)

                    else:
                        st.markdown("### ✅ Texto Seguro")
                        st.markdown(f'<div class="safe-card">', unsafe_allow_html=True)
                        st.markdown("**Resultado:** No se detectaron sesgos significativos en este texto.")
                        sentences_count = len([s for s in re.split(r'[.!?;\n:]+', user_input) if s.strip() and len(s.strip()) > 10])
                        st.write(f"**📊 Frases analizadas:** {sentences_count}")
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Por favor ingresa un texto para analizar.")


    st.markdown("---")
    st.markdown("""
    <p style='text-align: center; color: #666; font-size: 0.8rem;'>
    🤖 DiversIA - Powered by Transformers & AI |
    Desarrollado para promover lenguaje inclusivo
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()