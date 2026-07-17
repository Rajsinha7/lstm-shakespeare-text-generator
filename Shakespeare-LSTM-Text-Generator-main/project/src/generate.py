"""
Inference and Text Generation Module for Shakespeare LSTM.
Implements Temperature Sampling, Beam Search, and full seed-text cleaning and formatting.
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import List, Dict, Tuple

def sample_with_temperature(preds: np.ndarray, temperature: float = 1.0) -> int:
    """
    Applies temperature scaling to the prediction probabilities and draws a single sample.
    - Low temp (0.2) -> high confidence, deterministic, repetitive.
    - Mid temp (0.5) -> balanced.
    - High temp (1.0) -> diverse, creative, potentially incoherent.
    """
    preds = np.asarray(preds).astype("float64")
    
    # Avoid log(0) division by adding a tiny epsilon
    preds = np.clip(preds, 1e-10, 1.0 - 1e-10)
    
    # Scale predictions by temperature
    log_preds = np.log(preds) / temperature
    exp_preds = np.exp(log_preds)
    
    # Normalize back into probability distribution
    scaled_probs = exp_preds / np.sum(exp_preds)
    
    # Draw standard multinomial sample
    probas = np.random.multinomial(1, scaled_probs, 1)
    return int(np.argmax(probas))

def clean_seed_text(text: str) -> str:
    """
    Normalizes the input user seed to match the preprocessor specifications.
    """
    import re
    text = text.lower().replace("\n", " ")
    text = re.sub(r"[^a-zA-Z0-9'\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def generate_text_temperature(
    model, 
    word2idx: Dict[str, int], 
    idx2word: Dict[int, str], 
    seed_text: str, 
    next_words: int = 30, 
    seq_length: int = 20, 
    temperature: float = 0.5
) -> str:
    """
    Generates text using temperature sampling.
    """
    cleaned_seed = clean_seed_text(seed_text)
    words = cleaned_seed.split()
    
    # If the seed is shorter than sequence length, pad it with <PAD>
    if len(words) < seq_length:
        words = ["<PAD>"] * (seq_length - len(words)) + words
        
    generated_words = list(words)
    
    for _ in range(next_words):
        # Format the last N words as input sequence
        input_words = generated_words[-seq_length:]
        input_indices = [word2idx.get(w, 1) for w in input_words] # fallback to <UNK>
        
        # Reshape for prediction [1, seq_length]
        x_input = np.reshape(input_indices, (1, seq_length))
        
        # Get softmax prediction
        preds = model.predict(x_input, verbose=0)[0]
        
        # Sample index
        next_idx = sample_with_temperature(preds, temperature)
        next_word = idx2word.get(next_idx, "<UNK>")
        
        # Append to track
        generated_words.append(next_word)
        
    # Filter out special PAD tokens from final display
    display_words = [w for w in generated_words if w != "<PAD>"]
    return " ".join(display_words)

def generate_text_beam_search(
    model, 
    word2idx: Dict[str, int], 
    idx2word: Dict[int, str], 
    seed_text: str, 
    next_words: int = 15, 
    seq_length: int = 20, 
    beam_width: int = 3
) -> str:
    """
    Generates text using Beam Search to find sequences with the highest joint probability.
    Extremely effective at avoiding short-sighted, repetitive, or locally optimal predictions.
    """
    cleaned_seed = clean_seed_text(seed_text)
    words = cleaned_seed.split()
    
    if len(words) < seq_length:
        words = ["<PAD>"] * (seq_length - len(words)) + words
        
    # Candidate sequences represented as: (word_list, cumulative_log_probability)
    # Start with the cleaned seed sequence
    candidates = [(list(words), 0.0)]
    
    for _ in range(next_words):
        all_expansions = []
        
        for seq, score in candidates:
            # Prepare current input context
            input_words = seq[-seq_length:]
            input_indices = [word2idx.get(w, 1) for w in input_words]
            x_input = np.reshape(input_indices, (1, seq_length))
            
            # Predict probability distribution
            preds = model.predict(x_input, verbose=0)[0]
            preds = np.clip(preds, 1e-10, 1.0) # avoid log(0)
            
            # Get the top indices to expand
            top_indices = np.argsort(preds)[-beam_width:]
            
            for idx in top_indices:
                next_word = idx2word.get(idx, "<UNK>")
                prob = preds[idx]
                # Log score addition (log prob multiplying)
                new_score = score + np.log(prob)
                new_seq = seq + [next_word]
                all_expansions.append((new_seq, new_score))
                
        # Sort all expansions by their scores in descending order and select top candidates
        all_expansions.sort(key=lambda x: x[1], reverse=True)
        candidates = all_expansions[:beam_width]
        
    # Return candidate list's best-scoring sequence
    best_seq = candidates[0][0]
    display_words = [w for w in best_seq if w != "<PAD>"]
    return " ".join(display_words)

def run_sample_generation(
    model_path: str = None,
    mappings_dir: str = None,
    seed: str = None,
    length: int = 25,
    temperature: float = 0.5,
    strategy: str = "both",
    beam_width: int = 3
) -> None:
    """
    Demonstrates comparison of generation methods and parameters.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    if model_path is None:
        model_path = os.path.join(project_dir, "models", "best_model.keras")
    if mappings_dir is None:
        mappings_dir = os.path.join(project_dir, "models")
        
    print("[*] Loading trained LSTM model and vocabulary mappings...")
    try:
        model = load_model(model_path)
        with open(os.path.join(mappings_dir, "tokenizer.pkl"), "rb") as f:
            tokenizer_data = pickle.load(f)
        word2idx = tokenizer_data["word2idx"]
        idx2word = tokenizer_data["idx2word"]
    except Exception as e:
        print(f"[-] Loading failed: {e}. Please ensure the model is trained by running 'python src/train.py' or 'python src/train_quick.py' first.")
        return
        
    if seed:
        print(f"\n[+] Original Seed: '{seed}'\n" + "="*50)
        if strategy in ["temperature", "both"]:
            print(f"\n[Temperature: {temperature}]")
            generated = generate_text_temperature(model, word2idx, idx2word, seed, next_words=length, temperature=temperature)
            print(f"  > {generated}")
        if strategy in ["beam", "both"]:
            print(f"\n[Beam Search - Width {beam_width}]")
            generated_beam = generate_text_beam_search(model, word2idx, idx2word, seed, next_words=length, beam_width=beam_width)
            print(f"  > {generated_beam}")
    else:
        # Default behavior: generate sample outputs for required interview seeds
        test_seeds = ["to be or not", "the king", "love is"]
        print("[*] No custom seed provided. Generating sample outputs for default interview seeds:")
        for ts in test_seeds:
            print("\n" + "="*60)
            print(f"Seed Text: '{ts}'")
            print("="*60)
            
            print("[Temperature = 0.5]")
            out_temp = generate_text_temperature(model, word2idx, idx2word, ts, next_words=length, temperature=0.5)
            print(f"  > {out_temp}")
            
            print("[Beam Search - Width 3]")
            out_beam = generate_text_beam_search(model, word2idx, idx2word, ts, next_words=length, beam_width=3)
            print(f"  > {out_beam}")

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Generate text using trained Shakespeare LSTM model.")
    parser.add_argument("--seed", type=str, default=None, help="Seed text for generation.")
    parser.add_argument("--words", type=int, default=25, help="Number of words to generate.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature.")
    parser.add_argument("--strategy", type=str, default="both", choices=["temperature", "beam", "both"], help="Decoding strategy.")
    parser.add_argument("--beam_width", type=int, default=3, help="Beam width for beam search.")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    run_sample_generation(
        model_path=os.path.join(project_dir, "models", "best_model.keras"),
        mappings_dir=os.path.join(project_dir, "models"),
        seed=args.seed,
        length=args.words,
        temperature=args.temperature,
        strategy=args.strategy,
        beam_width=args.beam_width
    )
