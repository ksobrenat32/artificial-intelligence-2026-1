#!/bin/bash

# DiversIA Streamlit App Launcher
echo "🤖 Iniciando DiversIA - Detección de Sesgos..."
echo "=========================================="

# Check if models exist
if [ ! -d "model-bias-detection" ]; then
    echo "⚠️  Modelo de detección de sesgos no encontrado."
    echo "💡 Ejecuta primero: python3 training.py"
    echo ""
fi

if [ ! -d "vector_store" ]; then
    echo "⚠️  Vector store no encontrado."
    echo "💡 Ejecuta primero: python3 training.py"
    echo ""
fi

# Install requirements if needed
if [ ! -f ".streamlit_deps_installed" ]; then
    echo "📦 Instalando dependencias..."
    pip install -r requirements.txt
    touch .streamlit_deps_installed
fi

# Create .streamlit directory and config if it doesn't exist
mkdir -p .streamlit

# Create streamlit config
cat > .streamlit/config.toml << EOL
[server]
headless = false
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
EOL

echo "🚀 Ejecutando Streamlit..."
echo "📱 La aplicación se abrirá en: http://localhost:8501"
echo ""

# Run streamlit
streamlit run streamlit_app.py