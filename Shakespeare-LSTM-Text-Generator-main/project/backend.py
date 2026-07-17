"""
FastAPI Backend for local Shakespeare LSTM Text Generation.
Loads model and tokenizer, performing real TF/Keras inference.
"""

import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

# Global variables for model resources
model = None
word2idx = None
idx2word = None
load_error = None

# Attempt to load model and tokenizer
if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
    load_error = "Model not trained yet. Please run training first."
    print(f"[-] Load error: {load_error}")
else:
    try:
        from tensorflow.keras.models import load_model
        print("[*] Loading Keras model from:", MODEL_PATH)
        model = load_model(MODEL_PATH)
        print("[*] Loading Tokenizer from:", TOKENIZER_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer_data = pickle.load(f)
        word2idx = tokenizer_data["word2idx"]
        idx2word = tokenizer_data["idx2word"]
        print("[+] Model and Tokenizer loaded successfully!")
    except Exception as e:
        load_error = f"Failed to load resources: {str(e)}"
        print(f"[-] Load error: {load_error}")

app = FastAPI(title="Shakespeare LSTM Backend")

# Enable CORS for React dev server requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    seed_text: str
    temperature: float = 0.5
    next_words: int = 30
    strategy: str = "temperature"  # "temperature" or "beam"
    beam_width: int = 3

class AlternativeToken(BaseModel):
    token: str
    probability: float

class TokenDistribution(BaseModel):
    word: str
    alternatives: List[AlternativeToken]

class BeamSearchCandidate(BaseModel):
    sequence: str
    score: float

class BeamSearchStep(BaseModel):
    step: int
    candidates: List[BeamSearchCandidate]

class GenerateResponse(BaseModel):
    generated_text: str
    probabilities: List[TokenDistribution]
    beam_steps: Optional[List[BeamSearchStep]] = None

def sample_with_temperature(preds: np.ndarray, temperature: float = 1.0) -> int:
    preds = np.asarray(preds).astype("float64")
    preds = np.clip(preds, 1e-10, 1.0 - 1e-10)
    log_preds = np.log(preds) / temperature
    exp_preds = np.exp(log_preds)
    scaled_probs = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, scaled_probs, 1)
    return int(np.argmax(probas))

def clean_seed_text(text: str) -> str:
    import re
    text = text.lower().replace("\n", " ")
    text = re.sub(r"[^a-zA-Z0-9'\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_text(req: GenerateRequest):
    global model, word2idx, idx2word, load_error
    
    # Reload model if it was trained since startup and not loaded
    if model is None:
        if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
            try:
                from tensorflow.keras.models import load_model
                model = load_model(MODEL_PATH)
                with open(TOKENIZER_PATH, "rb") as f:
                    tokenizer_data = pickle.load(f)
                word2idx = tokenizer_data["word2idx"]
                idx2word = tokenizer_data["idx2word"]
                load_error = None
                print("[+] Model and Tokenizer reloaded successfully!")
            except Exception as e:
                load_error = f"Failed to reload model: {str(e)}"
    
    if model is None or word2idx is None or idx2word is None:
        raise HTTPException(
            status_code=400, 
            detail=load_error or "Model not trained yet. Please run training first."
        )
    
    seq_length = 20  # Model expects sequence of length 20
    
    try:
        if req.strategy == "beam":
            # Beam search generation
            cleaned_seed = clean_seed_text(req.seed_text)
            words = cleaned_seed.split()
            
            if len(words) < seq_length:
                padded_words = ["<PAD>"] * (seq_length - len(words)) + words
            else:
                padded_words = words[-seq_length:]
                
            candidates = [(list(padded_words), 0.0)]
            beam_steps = []
            
            for step in range(1, req.next_words + 1):
                all_expansions = []
                for seq, score in candidates:
                    input_words = seq[-seq_length:]
                    input_indices = [word2idx.get(w, 1) for w in input_words]
                    x_input = np.reshape(input_indices, (1, seq_length))
                    
                    preds = model.predict(x_input, verbose=0)[0]
                    preds = np.clip(preds, 1e-10, 1.0)
                    
                    # Sort indices
                    top_indices = np.argsort(preds)[-req.beam_width:]
                    for idx in top_indices:
                        next_word = idx2word.get(idx, "<UNK>")
                        prob = preds[idx]
                        new_score = score + np.log(prob)
                        new_seq = seq + [next_word]
                        all_expansions.append((new_seq, new_score))
                        
                all_expansions.sort(key=lambda x: x[1], reverse=True)
                candidates = all_expansions[:req.beam_width]
                
                # Log step candidates
                step_candidates = []
                for seq, score in candidates:
                    display_seq = [w for w in seq if w != "<PAD>"]
                    step_candidates.append({
                        "sequence": " ".join(display_seq),
                        "score": float(score)
                    })
                beam_steps.append({
                    "step": step,
                    "candidates": step_candidates
                })
                
            best_seq, best_score = candidates[0]
            display_words = [w for w in best_seq if w != "<PAD>"]
            generated_text = " ".join(display_words)
            
            # Reconstruct probabilities log along the chosen path
            probabilities = []
            seed_len = len(padded_words)
            for i in range(seed_len, len(best_seq)):
                input_words = best_seq[i - seq_length: i]
                next_word = best_seq[i]
                next_idx = word2idx.get(next_word, 1)
                
                input_indices = [word2idx.get(w, 1) for w in input_words]
                x_input = np.reshape(input_indices, (1, seq_length))
                
                preds = model.predict(x_input, verbose=0)[0]
                preds_clip = np.clip(preds, 1e-10, 1.0 - 1e-10)
                
                top_indices = np.argsort(preds_clip)[::-1]
                other_indices = [idx for idx in top_indices if idx != next_idx]
                alt_indices = [next_idx] + other_indices[:3]
                
                alternatives = [
                    {"token": idx2word.get(idx, "<UNK>"), "probability": float(preds_clip[idx])}
                    for idx in alt_indices
                ]
                probabilities.append({
                    "word": next_word,
                    "alternatives": alternatives
                })
                
            return {
                "generated_text": generated_text,
                "probabilities": probabilities,
                "beam_steps": beam_steps
            }
            
        else:
            # Temperature sampling generation
            cleaned_seed = clean_seed_text(req.seed_text)
            words = cleaned_seed.split()
            
            if len(words) < seq_length:
                input_words = ["<PAD>"] * (seq_length - len(words)) + words
            else:
                input_words = words[-seq_length:]
                
            generated_words = list(words)
            probabilities = []
            
            for _ in range(req.next_words):
                input_indices = [word2idx.get(w, 1) for w in input_words]
                x_input = np.reshape(input_indices, (1, seq_length))
                
                preds = model.predict(x_input, verbose=0)[0]
                
                # Sample
                next_idx = sample_with_temperature(preds, req.temperature)
                next_word = idx2word.get(next_idx, "<UNK>")
                
                # Calculate scaled probabilities
                preds_clip = np.clip(preds, 1e-10, 1.0 - 1e-10)
                log_preds = np.log(preds_clip) / req.temperature
                exp_preds = np.exp(log_preds)
                scaled_probs = exp_preds / np.sum(exp_preds)
                
                # Top candidates
                top_indices = np.argsort(scaled_probs)[::-1]
                other_indices = [idx for idx in top_indices if idx != next_idx]
                alt_indices = [next_idx] + other_indices[:3]
                
                alternatives = [
                    {"token": idx2word.get(idx, "<UNK>"), "probability": float(scaled_probs[idx])}
                    for idx in alt_indices
                ]
                probabilities.append({
                    "word": next_word,
                    "alternatives": alternatives
                })
                
                generated_words.append(next_word)
                input_words = input_words[1:] + [next_word]
                
            display_words = [w for w in generated_words if w != "<PAD>"]
            generated_text = " ".join(display_words)
            
            return {
                "generated_text": generated_text,
                "probabilities": probabilities
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
