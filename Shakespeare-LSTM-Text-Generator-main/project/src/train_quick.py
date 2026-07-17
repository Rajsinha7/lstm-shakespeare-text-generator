"""
Quick Training Script for Shakespeare LSTM.
Generates a tiny model and tokenizer in seconds for testing and development.
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Relative imports from src
from preprocess import download_dataset, clean_text, tokenize_and_build_vocab, create_sequences, save_preprocessor
from train import build_lstm_model

def run_quick_training():
    print("[*] Starting quick training for testing...")
    
    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    data_path = download_dataset(os.path.join(project_dir, "data", "t8.shakespeare.txt"))
    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    # Take a tiny slice of the text to train instantly
    raw_text = raw_text[:20000]
    
    print("[*] Cleaning text...")
    cleaned = clean_text(raw_text)
    
    print("[*] Tokenizing...")
    words, word2idx, idx2word = tokenize_and_build_vocab(cleaned)
    vocab_size = len(word2idx)
    print(f"[+] Vocabulary size: {vocab_size}")
    
    # Save the tokenizer mappings
    model_dir = os.path.join(project_dir, "models")
    save_preprocessor(word2idx, idx2word, dir_path=model_dir)
    
    print("[*] Creating sequences...")
    X, y = create_sequences(words, word2idx, seq_length=20)
    
    # Take only a small subset of sequences
    X = X[:500]
    y = y[:500]
    
    print(f"[+] Total sequence samples for quick training: {len(X)}")
    
    # Build a tiny LSTM model
    print("[*] Building a tiny LSTM model...")
    model = Sequential(name="Tiny_Shakespeare_LSTM")
    model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=20, name="Word_Embedding"))
    model.add(LSTM(units=32, return_sequences=True, name="LSTM_Layer_1"))
    model.add(BatchNormalization())
    model.add(LSTM(units=32, return_sequences=False, name="LSTM_Layer_2"))
    model.add(BatchNormalization())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(vocab_size, activation="softmax"))
    
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("[*] Training tiny model for 1 epoch...")
    model.fit(X, y, epochs=1, batch_size=32, verbose=1)
    
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "best_model.keras"))
    print(f"[+] Saved tiny model to {model_dir}/best_model.keras")
    print("[+] Quick training complete!")

if __name__ == "__main__":
    run_quick_training()
