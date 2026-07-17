"""
Utility Module for the Shakespeare LSTM Text Generator.
Provides evaluation helpers, perplexity metrics, statistics visualizers, and project packaging routines.
"""

import os
import zipfile
import numpy as np
import tensorflow as tf
from typing import List, Dict

def calculate_perplexity(loss_values: List[float]) -> List[float]:
    """
    Calculates perplexity across training epochs given a list of cross-entropy losses.
    Perplexity = exp(Loss)
    """
    return [float(np.exp(loss)) for loss in loss_values]

def print_dataset_statistics(words: List[str], word2idx: Dict[str, int]) -> Dict[str, float]:
    """
    Analyzes dataset text lists and returns core stats.
    """
    stats = {
        "Total Words": len(words),
        "Unique Vocabulary Size": len(word2idx),
        "Lexical Richness": len(word2idx) / len(words) if words else 0
    }
    
    print("\n" + "="*40 + "\nDATASET STATISTICS\n" + "="*40)
    for k, v in stats.items():
        if isinstance(v, int):
            print(f"{k:<30}: {v:,}")
        else:
            print(f"{k:<30}: {v:.4f}")
    print("="*40 + "\n")
    return stats

def print_sequence_examples(X: np.ndarray, y: np.ndarray, idx2word: Dict[int, str], count: int = 3) -> None:
    """
    Prints input sequences matched with their respective labels for visual validation.
    """
    print("\n" + "="*50 + "\nSLIDING WINDOW SEQUENCE EXAMPLES\n" + "="*50)
    indices = np.random.choice(len(X), size=count, replace=False)
    for num, idx in enumerate(indices, start=1):
        seq_words = [idx2word.get(i, "<UNK>") for i in X[idx]]
        target_word = idx2word.get(y[idx], "<UNK>")
        
        print(f"\nExample #{num}:")
        print(f"  Input Sequence : \"{' '.join(seq_words)}\"")
        print(f"  Next Word Label: \"{target_word}\"")
    print("\n" + "="*50 + "\n")

def package_project_zip(project_root: str = None, output_filename: str = "shakespeare_lstm_project.zip") -> str:
    """
    Compresses the complete project structure into a clean zip archive for easy local or Colab export.
    Excludes large cached files, virtual environments, and weights.
    """
    if project_root is None:
        # Resolve to the 'project' directory relative to this script
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(src_dir)
        
    zip_path = os.path.join(os.path.dirname(project_root) or ".", output_filename)
    
    # Exclude system caches and large generated binaries from package
    exclude_dirs = {"__pycache__", ".ipynb_checkpoints", ".git", "venv", "node_modules", ".outputs"}
    exclude_extensions = {".pyc", ".h5", ".pkl", ".zip"}
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(project_root):
            # Prune directory listing in-place
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in exclude_extensions:
                    continue
                file_path = os.path.join(root, file)
                # Compute relative zip path
                arcname = os.path.relpath(file_path, os.path.dirname(project_root))
                zipf.write(file_path, arcname)
                
    print(f"[+] Compressed Python project successfully into: {zip_path}")
    return zip_path

if __name__ == "__main__":
    # Test packaging
    package_project_zip()
