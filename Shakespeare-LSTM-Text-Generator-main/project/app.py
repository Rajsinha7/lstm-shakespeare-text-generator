"""
Streamlit Web Application for local interactive Shakespeare text generation.
Loads preprocessed vocabularies and trained H5 model configurations to output live predictions.
"""

import os
import pickle
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Direct module imports relative to project workspace
from src.generate import generate_text_temperature, generate_text_beam_search, clean_seed_text

# Page Config
st.set_page_config(
    page_title="Shakespeare LSTM Text Generator",
    page_icon="🎭",
    layout="centered"
)

# Title Header
st.title("🎭 Shakespeare LSTM Text Generator")
st.markdown(
    "Generate custom, grammatically structured Shakespearean verses utilizing trained stacked Recurrent Neural Networks!"
)

# Paths Configuration
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

@st.cache_resource
def load_generator_resources():
    """
    Caches model and vocabulary mappings to maximize application response times.
    """
    if not os.path.exists(MODEL_PATH):
        return None, None, None, "Model file not found. Ensure training has run first!"
        
    try:
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer_data = pickle.load(f)
        word2idx = tokenizer_data["word2idx"]
        idx2word = tokenizer_data["idx2word"]
        return model, word2idx, idx2word, None
    except Exception as e:
        return None, None, None, f"Failed to load resources: {str(e)}"

# Sidebar Controls
st.sidebar.header("🛠️ Generation Parameters")

gen_method = st.sidebar.selectbox(
    "Decoding Strategy",
    options=["Temperature Sampling", "Beam Search"],
    help="Temperature Sampling offers variable randomness, whereas Beam Search explores continuous high-probability paths."
)

next_words_count = st.sidebar.slider(
    "Sequence Length to Generate",
    min_value=5,
    max_value=100,
    value=25,
    step=5
)

# Custom fields based on strategy selection
if gen_method == "Temperature Sampling":
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.5,
        value=0.5,
        step=0.1,
        help="Lower values make the text conservative and repetitive; higher values make it diverse but potentially chaotic."
    )
else:
    beam_width = st.sidebar.slider(
        "Beam Width",
        min_value=2,
        max_value=5,
        value=3,
        step=1,
        help="Number of continuous alternative branches analyzed simultaneously. Higher widths take slightly longer to process."
    )

# Main Form Container
st.subheader("💡 Enter Seed Text")
default_seed = "Shall I compare thee to a summer's day"
user_seed = st.text_input("Seed Sentence:", value=default_seed)

# Trigger Action
if st.button("🎭 Generate Shakespearean Verse"):
    model, word2idx, idx2word, err = load_generator_resources()
    
    if err:
        st.warning(err)
        st.info(
            "👉 Don't have a pre-trained model? Download this full package, run `src/train.py` on your machine, or test it out inside our interactive online simulation environment!"
        )
    else:
        with st.spinner("Writing the scroll..."):
            if gen_method == "Temperature Sampling":
                output = generate_text_temperature(
                    model=model,
                    word2idx=word2idx,
                    idx2word=idx2word,
                    seed_text=user_seed,
                    next_words=next_words_count,
                    temperature=temperature
                )
            else:
                output = generate_text_beam_search(
                    model=model,
                    word2idx=word2idx,
                    idx2word=idx2word,
                    seed_text=user_seed,
                    next_words=next_words_count,
                    beam_width=beam_width
                )
                
        st.success("📝 Generation Complete!")
        st.markdown(f"**Seed Context:** *\"{user_seed}\"*")
        st.markdown(f"### Generated Text:")
        st.code(output, language="text")

# Footer details
st.divider()
st.markdown(
    "Created as an interview-quality presentation project highlighting advanced sequence model training pipelines, custom temperature sampling decoders, and beam searching."
)
