import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Kichwa Translator",
    page_icon="🌎",
    layout="centered"
)

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    MODEL_PATH = "./modelo_kichwa_flan_t5"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

    return tokenizer, model

tokenizer, model = load_model()

# -------------------------
# TRANSCRIPTIONS
# -------------------------
# Relaciona el nombre del audio con la transcripción oficial
transcriptions = {
    "audio1.wav": "Alcaldesa, ama kayaychu.",
    "audio2.wav": "Pandemiamanta rimaykuna",
    "audio3.wav": "Ari, mashi"
}

# -------------------------
# TITLE
# -------------------------
# st.markdown(
#     "<h1 style='text-align:center;'>🌎 Kichwa → Spanish Translator</h1>",
#     unsafe_allow_html=True
# )

st.markdown(
    "<p style='text-align:center;'>Voice-to-Voice Translation using AI</p>",
    unsafe_allow_html=True
)

st.divider()

# -------------------------
# SYSTEM PIPELINE
# -------------------------
# st.markdown(
#     """
#     <div style='text-align:center; font-size:22px;'>
#     🎤 Speech → 📝 Transcription → 🤖 Translation → 🔊 Spanish
#     </div>
#     """,
#     unsafe_allow_html=True
# )

st.divider()

# -------------------------
# AUDIO SECTION
# -------------------------
st.markdown("## 🎤 Upload Audio")

audio_file = st.file_uploader(
    "Choose a WAV audio file",
    type=["wav"]
)

texto = ""

if audio_file is not None:

    # Reproducir audio
    st.audio(audio_file)

    # Obtener nombre del archivo
    filename = audio_file.name

    # Buscar transcripción oficial
    texto = transcriptions.get(filename, "")

    st.markdown("### 📝 Official Transcription")

    texto = st.text_area(
        "Detected text:",
        value=texto,
        height=120
    )

    st.success("Audio loaded successfully!")

# -------------------------
# TRANSLATE
# -------------------------
st.divider()

if st.button("🚀 Translate", use_container_width=True):

    if texto.strip() != "":

        with st.spinner("Generating translation..."):

            prompt = f"Translate to Spanish: {texto}"

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            outputs = model.generate(
                **inputs,
                max_length=100
            )

            resultado = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

        # -------------------------
        # OUTPUT
        # -------------------------
        st.markdown("## 📘 Translation Result")

        st.success(resultado)

    else:
        st.warning("Please upload an audio file.")

# -------------------------
# FOOTER
# -------------------------
st.divider()

st.markdown(
    """
    <div style='text-align:center; font-size:14px;'>
    Low-resource Neural Machine Translation Prototype
    </div>
    """,
    unsafe_allow_html=True
)