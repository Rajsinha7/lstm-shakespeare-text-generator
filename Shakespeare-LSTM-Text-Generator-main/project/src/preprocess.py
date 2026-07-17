"""
Shakespeare Text Preprocessing Module for LSTM Text Generation.
Handles downloading the dataset, tokenization, sequence preparation, and mapping saves.
"""

import os
import re
import urllib.request
import pickle
import numpy as np
from typing import Tuple, List, Dict

# Resolve paths dynamically relative to script location
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)

# Project Gutenberg Complete Works of Shakespeare URL
SHAKESPEARE_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
# Fallback to TensorFlow Shakespeare Dataset URL
SHAKESPEARE_URL_FALLBACK = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"

def download_dataset(destination_path: str = None) -> str:
    """
    Downloads the Shakespeare dataset if it is not already present locally.
    """
    if destination_path is None:
        destination_path = os.path.join(PROJECT_DIR, "data", "t8.shakespeare.txt")
        
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    if not os.path.exists(destination_path):
        print(f"[*] Downloading Shakespeare dataset from {SHAKESPEARE_URL}...")
        try:
            # Gutenberg can sometimes block basic urllib requests, so we set a User-Agent header
            req = urllib.request.Request(
                SHAKESPEARE_URL,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(destination_path, 'wb') as out_file:
                out_file.write(response.read())
            print("[+] Download complete!")
        except Exception as e:
            print(f"[-] Failed to download from Gutenberg ({e}). Attempting fallback URL...")
            try:
                urllib.request.urlretrieve(SHAKESPEARE_URL_FALLBACK, destination_path)
                print("[+] Fallback download complete!")
            except Exception as fe:
                print(f"[-] Fallback download failed: {fe}")
                raise fe
    else:
        print(f"[+] Dataset already exists at {destination_path}")
    return destination_path

def strip_gutenberg_metadata(text: str) -> str:
    """
    Strips the Project Gutenberg header and footer text.
    Works by locating the start of the plays ("1609\n\nTHE SONNETS" or after copyright disclaimers)
    and the end of the plays ("THE END" before the final copyright block).
    """
    # Normalize line endings
    normalized = text.replace("\r\n", "\n")
    
    # 1. Strip Header
    # Look for the start of the plays
    header_marker = "1609\n\nTHE SONNETS"
    start_idx = normalized.find(header_marker)
    if start_idx != -1:
        print("[*] Detected Project Gutenberg header ending at '1609\\n\\nTHE SONNETS'. Stripping...")
        normalized = normalized[start_idx:]
    else:
        # Fallback: search for the second occurrence of the disclaimer end ">>"
        disclaimers = [m.end() for m in re.finditer(r">>", normalized)]
        if len(disclaimers) >= 2:
            print("[*] Detected Project Gutenberg header ending after second copyright notice. Stripping...")
            normalized = normalized[disclaimers[1]:]
            
    # 2. Strip Footer
    # Look for the last occurrence of the disclaimer start at the end of the file
    footer_marker = "<<THIS ELECTRONIC VERSION"
    end_idx = normalized.rfind(footer_marker)
    if end_idx != -1:
        # If "THE END" is just before the footer, we can chop there
        the_end_idx = normalized[:end_idx].rfind("THE END")
        if the_end_idx != -1 and (end_idx - the_end_idx) < 100:
            print("[*] Detected Project Gutenberg footer starting after 'THE END'. Stripping...")
            normalized = normalized[:the_end_idx + len("THE END")]
        else:
            print("[*] Detected Project Gutenberg footer. Stripping...")
            normalized = normalized[:end_idx]
    else:
        # Fallback: search for "End of this Etext" or "End of the Project Gutenberg"
        end_idx_fallback = normalized.rfind("End of this Etext")
        if end_idx_fallback != -1:
            print("[*] Detected Project Gutenberg footer ending statement. Stripping...")
            normalized = normalized[:end_idx_fallback]
            
    return normalized.strip()

def clean_text(text: str) -> str:
    """
    Converts text to lowercase and removes punctuation and unnecessary symbols.
    Also strips Gutenberg header and footer metadata.
    """
    # Detect and remove Project Gutenberg header and footer
    text = strip_gutenberg_metadata(text)
    
    # Lowercase
    text = text.lower()
    # Replace newlines with spaces to help with tokenization
    text = text.replace("\n", " ")
    # Keep only letters, numbers, spaces, and basic apostrophes
    text = re.sub(r"[^a-zA-Z0-9'\s]", "", text)
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_and_build_vocab(cleaned_text: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    Tokenizes text at the word level and creates bidirectional index mappings.
    """
    words = cleaned_text.split()
    unique_words = sorted(list(set(words)))
    
    # We add <PAD> and <UNK> tokens to our vocabulary for standard robust NLP practices
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for idx, word in enumerate(unique_words, start=2):
        word2idx[word] = idx
        
    idx2word = {idx: word for word, idx in word2idx.items()}
    return words, word2idx, idx2word

def create_sequences(
    words: List[str], 
    word2idx: Dict[str, int], 
    seq_length: int = 20, 
    step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates input-output sliding window sequences for next-word prediction.
    e.g. Input: "to be or not to", Output: "be"
    """
    X_indices = []
    y_indices = []
    
    for i in range(0, len(words) - seq_length, step):
        seq = words[i : i + seq_length]
        next_word = words[i + seq_length]
        
        X_indices.append([word2idx.get(w, 1) for w in seq])
        y_indices.append(word2idx.get(next_word, 1))
        
    return np.array(X_indices), np.array(y_indices)

def split_data(
    X: np.ndarray, 
    y: np.ndarray, 
    val_split: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the sequences into training and validation sets.
    """
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)
    
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    split_idx = int(num_samples * (1 - val_split))
    X_train, X_val = X_shuffled[:split_idx], X_shuffled[split_idx:]
    y_train, y_val = y_shuffled[:split_idx], y_shuffled[split_idx:]
    
    return X_train, y_train, X_val, y_val

def save_preprocessor(word2idx: Dict[str, int], idx2word: Dict[int, str], dir_path: str = None) -> None:
    """
    Saves vocabulary index mappings as pickled files for generation reference.
    """
    if dir_path is None:
        dir_path = os.path.join(PROJECT_DIR, "models")
        
    os.makedirs(dir_path, exist_ok=True)
    tokenizer_data = {
        "word2idx": word2idx,
        "idx2word": idx2word
    }
    with open(os.path.join(dir_path, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer_data, f)
    print(f"[+] Saved word2idx and idx2word mappings to single tokenizer.pkl in {dir_path}")

def run_preprocessing_pipeline(
    data_path: str = None,
    seq_length: int = 20,
    val_split: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Full preprocessing execution flow.
    """
    if data_path is None:
        data_path = os.path.join(PROJECT_DIR, "data", "t8.shakespeare.txt")
        
    filepath = download_dataset(data_path)
    
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    print(f"[*] Raw text length: {len(raw_text)} characters.")
    
    print("[*] Cleaning text...")
    cleaned = clean_text(raw_text)
    print(f"[*] Cleaned word count: {len(cleaned.split())} words.")
    
    print("[*] Tokenizing and building vocabulary...")
    words, word2idx, idx2word = tokenize_and_build_vocab(cleaned)
    vocab_size = len(word2idx)
    print(f"[+] Vocabulary size (including <PAD>/<UNK>): {vocab_size}")
    
    print(f"[*] Creating sequences of length {seq_length}...")
    X, y = create_sequences(words, word2idx, seq_length=seq_length)
    print(f"[+] Total sequence samples: {len(X)}")
    
    print("[*] Splitting data into Train and Validation sets...")
    X_train, y_train, X_val, y_val = split_data(X, y, val_split=val_split)
    print(f"[+] Train shape: {X_train.shape}, Validation shape: {X_val.shape}")
    
    # Save vocabulary for inference
    save_preprocessor(word2idx, idx2word, dir_path=os.path.join(PROJECT_DIR, "models"))
    
    return X_train, y_train, X_val, y_val, word2idx, idx2word

if __name__ == "__main__":
    # Test script execution
    run_preprocessing_pipeline()

