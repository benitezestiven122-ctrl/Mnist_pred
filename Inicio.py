import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os
import time

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y UX
# ==========================================
st.set_page_config(
    page_title="Lienzo Emocional",
    page_icon="✨",
    layout="wide"
)

# Diccionario de reinterpretación: Mapea los 10 resultados del modelo MNIST a emociones
diccionario_emociones = {
    0: {"estado": "Agotamiento / Sobrecarga 🌧️", "mensaje": "Recuerda que pausar también es avanzar. Tómate un respiro, recarga energías y sé amable contigo mismo hoy. El mundo puede esperar."},
    1: {"estado": "Alegría / Energía Vital ☀️", "mensaje": "¡Qué vibras tan increíbles! Esa luz que transmites es única. Aprovecha este impulso para crear algo hermoso o contagiar tu energía a otros."},
    2: {"estado": "Duda / Confusión 🌫️", "mensaje": "Es completamente normal no tener todas las respuestas ahora mismo. Ve un paso a la vez, el camino se irá aclarando a medida que camines."},
    3: {"estado": "Melancolía / Tristeza 💧", "mensaje": "Los días grises son necesarios para que crezcan las flores. Permítete sentir, abraza tu vulnerabilidad y recuerda que esto también pasará."},
    4: {"estado": "Estrés / Ansiedad 🌪️", "mensaje": "Respira profundo. Inhala paz, exhala tensión. No tienes que resolver todo hoy. Enfócate solo en el siguiente pequeño paso."},
    5: {"estado": "Creatividad / Inspiración 🎨", "mensaje": "¡Tienes una mente brillante en ebullición! Deja fluir tus ideas sin juzgarlas. Estás en un momento perfecto para expresarte."},
    6: {"estado": "Calma / Paz Interior 🍃", "mensaje": "Disfruta de esta tranquilidad. Estás en tu centro. Guarda esta sensación en tu memoria para cuando necesites volver a ella."},
    7: {"estado": "Frustración / Bloqueo 🧱", "mensaje": "Esa energía intensa y atrapada puedes canalizarla hacia algo positivo. Libérala en este lienzo o da un paseo corto para cambiar de perspectiva."},
    8: {"estado": "Nostalgia 🕰️", "mensaje": "Recordar es vivir. Abraza esos recuerdos que te visitan hoy, agradece lo vivido, pero mantén tus ojos y tu corazón abiertos al presente."},
    9: {"estado": "Esperanza / Motivación 🌱", "mensaje": "Estás en el camino correcto. Sigue confiando en tu proceso. Tienes todo lo necesario dentro de ti para lograr lo que te propones."}
}

# ==========================================
# 2. CARGA DEL MODELO
# ==========================================
@st.cache_resource
def load_mnist_model():
    """Carga el modelo MNIST base que usaremos como motor de análisis abstracto"""
    model_files = ['mnist_model.keras', 'mnist_model.h5']
    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                return keras.models.load_model(model_path)
            except Exception:
                continue
    
    st.error("❌ No se encontró el modelo base (mnist_model.keras).")
    st.stop()

model = load_mnist_model()

# ==========================================
# 3. INTERFAZ DE USUARIO (UI)
# ==========================================
st.title("✨ Lienzo Emocional: Dibuja tu Sentir")
st.markdown("""
Bienvenido a un espacio seguro. **No pienses en formas perfectas**, simplemente deja que tu mano se mueva 
según cómo te sientes en este momento. Nuestra Inteligencia Artificial analizará los patrones de tus trazos 
para conectarse con tu estado de ánimo.
""")

col1, col2 = st.columns([2, 1])

with col1:
    # Se mantiene fondo negro y trazo blanco para que el modelo MNIST no falle al procesar la matriz
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        width=500,
        height=400,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.write("### 🎮 Acciones")
    predict_btn = st.button("🔮 **Analizar mis emociones**", type="primary", use_container_width=True)
    clear_btn = st.button("🗑️ Borrar y empezar de nuevo", use_container_width=True)
    
    if clear_btn:
        st.rerun()
        
    st.info("💡 **Tip:** Puedes dibujar formas abstractas, garabatos rápidos si sientes estrés, o líneas suaves si buscas calma.")

# ==========================================
# 4. LÓGICA DE PREDICCIÓN Y RESPUESTA
# ==========================================
if predict_btn:
    if canvas_result.image_data is not None:
        if np.max(canvas_result.image_data) == 0:
            st.warning("⚠️ El lienzo está vacío. Por favor, expresa algo primero.")
        else:
            with st.spinner("Meditando sobre tus trazos..."):
                time.sleep(1.5) # Pausa dramática para mejorar la experiencia de usuario (UX)
                
                # Procesamiento de imagen exacto para MNIST
                image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
                image_resized = image.resize((28, 28), Image.Resampling.LANCZOS)
                img_array = np.array(image_resized) / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)
                
                # Predicción del modelo
                prediction = model.predict(img_array, verbose=0)
                emocion_id = np.argmax(prediction)
                
                # Extraer datos de nuestro diccionario
                resultado = diccionario_emociones[emocion_id]
                
                # Efecto visual de celebración si es una emoción positiva
                if emocion_id in [1, 5, 9]:
                    st.balloons()
            
            # ==========================================
            # 5. PRESENTACIÓN DE RESULTADOS (FEEDBACK)
            # ==========================================
            st.markdown("---")
            st.success(f"### El oráculo percibe: **{resultado['estado']}**")
            
            # Tarjeta destacada con el mensaje motivacional
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: #2e3b4e; color: white; border-left: 5px solid #4CAF50; margin-bottom: 20px;">
                <h4 style="margin-top:0;">Mensaje para ti:</h4>
                <p style="font-size: 1.1rem; font-style: italic;">"{resultado['mensaje']}"</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detalles técnicos convertidos en "Matices emocionales"
            st.write("#### 🌊 Matices emocionales detectados en tu trazo")
            
            # Obtener el top 3 de emociones
            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            
            col_m1, col_m2, col_m3 = st.columns(3)
            columnas = [col_m1, col_m2, col_m3]
            
            for i, idx in enumerate(top_3_indices):
                with columnas[i]:
                    estado_secundario = diccionario_emociones[idx]['estado'].split()[0] # Toma solo la primera palabra
                    porcentaje = prediction[0][idx] * 100
                    st.metric(label=f"Matiz {i+1}", value=f"{estado_secundario}", delta=f"{porcentaje:.1f}% de intensidad")

    else:
        st.warning("⚠️ Por favor, interactúa con el lienzo primero.")
