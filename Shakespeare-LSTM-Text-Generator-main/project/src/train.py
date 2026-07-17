"""
Model Architecture and Training Module for the Shakespeare LSTM Text Generator.
Creates a robust stacked LSTM model with Embeddings, Dropout, and Batch Normalization.
Supports Sparse Categorical Crossentropy for computational and memory efficiency.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Dict

# Set up random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def build_lstm_model(
    vocab_size: int, 
    embedding_dim: int = 128, 
    seq_length: int = 20, 
    hidden_units: list = [256, 256],
    dropout_rate: float = 0.3
) -> Sequential:
    """
    Builds an advanced stacked LSTM model with Keras.
    Includes Embedding, Stacked LSTM, Batch Normalization, Dropout, and Dense Softmax.
    """
    model = Sequential(name="Advanced_Shakespeare_LSTM")
    
    # 1. Embedding Layer: maps word indices to continuous vector representations
    model.add(Embedding(
        input_dim=vocab_size, 
        output_dim=embedding_dim, 
        input_length=seq_length,
        name="Word_Embedding"
    ))
    
    # 2. Stacked LSTM Layers
    for i, units in enumerate(hidden_units):
        # All but the last LSTM layer must return sequences for stacking
        return_seq = (i < len(hidden_units) - 1)
        model.add(LSTM(
            units=units, 
            return_sequences=return_seq, 
            dropout=dropout_rate, 
            recurrent_dropout=0.0, # Kept 0.0 for cuDNN optimization
            name=f"LSTM_Layer_{i+1}"
        ))
        # Batch Normalization stabilizes training
        model.add(BatchNormalization(name=f"Batch_Norm_{i+1}"))
    
    # 3. Dense layer with Dropout for regularization
    model.add(Dense(128, activation="relu", name="Dense_Projection"))
    model.add(Dropout(dropout_rate, name="Dropout_Projection"))
    
    # 4. Output softmax layer predicting vocabulary probability distribution
    model.add(Dense(vocab_size, activation="softmax", name="Softmax_Prediction"))
    
    return model

def compile_and_train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    model_dir: str = "project/models",
    log_dir: str = "project/outputs/logs"
) -> Tuple[Sequential, tf.keras.callbacks.History]:
    """
    Compiles and trains the LSTM model, writing checkpoints and TensorBoard records.
    Uses Adam optimizer and Sparse Categorical Crossentropy to handle massive word vocabularies efficiently.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy", # highly efficient compared to categorical_crossentropy for raw word indices
        metrics=["accuracy"]
    )
    
    model.summary()
    
    # Define production-grade callbacks
    checkpoint_filepath = os.path.join(model_dir, "best_model.keras")
    
    callbacks = [
        # Stop training if validation loss halts improvement
        EarlyStopping(
            monitor="val_loss", 
            patience=3, 
            restore_best_weights=True,
            verbose=1
        ),
        # Save the best weights found during validation
        ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        # Decay learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
        # TensorBoard logging
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq="epoch"
        )
    ]
    
    print(f"[*] Starting LSTM Model Training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def plot_training_results(history: tf.keras.callbacks.History, save_dir: str = "project/outputs") -> None:
    """
    Generates training graphs showing training & validation loss and accuracy.
    Calculates and plots perplexity as well.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history.history["loss"]) + 1)
    
    # Create the figure
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history.history["loss"], "b-", label="Training Loss")
    plt.plot(epochs, history.history["val_loss"], "r-", label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history.history["accuracy"], "b-", label="Training Accuracy")
    plt.plot(epochs, history.history["val_accuracy"], "r-", label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Perplexity (exp of cross-entropy loss)
    train_perp = np.exp(history.history["loss"])
    val_perp = np.exp(history.history["val_loss"])
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_perp, "b-", label="Training Perplexity")
    plt.plot(epochs, val_perp, "r-", label="Validation Perplexity")
    plt.title("Model Perplexity")
    plt.xlabel("Epochs")
    plt.ylabel("Perplexity")
    plt.yscale("log") # Log scale helps with high initial perplexity values
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_filepath = os.path.join(save_dir, "training_metrics.png")
    plt.savefig(plot_filepath)
    plt.close()
    print(f"[+] Saved training accuracy, loss, and perplexity curves to {plot_filepath}")
if __name__ == "__main__":
    from preprocess import run_preprocessing_pipeline, download_dataset, clean_text, tokenize_and_build_vocab, create_sequences, save_preprocessor, split_data
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Train stacked LSTM model for Shakespeare text generation.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--quick", action="store_true", help="Train a tiny model for 1 epoch for fast validation/testing.")
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    data_path = os.path.join(project_dir, "data", "t8.shakespeare.txt")
    model_dir = os.path.join(project_dir, "models")
    log_dir = os.path.join(project_dir, "outputs", "logs")
    
    if args.quick:
        print("[*] Running QUICK training pipeline for testing...")
        # Resolve dataset path
        filepath = download_dataset(data_path)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
            
        # Take a tiny slice to train instantly
        raw_text = raw_text[:20000]
        cleaned = clean_text(raw_text)
        words, word2idx, idx2word = tokenize_and_build_vocab(cleaned)
        vocab_size = len(word2idx)
        print(f"[+] Quick vocabulary size: {vocab_size}")
        
        # Save preprocessor
        save_preprocessor(word2idx, idx2word, dir_path=model_dir)
        
        # Create sequences
        X, y = create_sequences(words, word2idx, seq_length=20)
        X = X[:500]
        y = y[:500]
        
        # Split
        X_train, y_train, X_val, y_val = split_data(X, y, val_split=0.2)
        print(f"[+] Quick train shape: {X_train.shape}, validation shape: {X_val.shape}")
        
        # Build a slightly smaller model for quick training
        print("[*] Building a quick LSTM model...")
        model = build_lstm_model(vocab_size=vocab_size, embedding_dim=32, seq_length=20, hidden_units=[32, 32], dropout_rate=0.1)
        
        # Train for 1 epoch
        model, history = compile_and_train_model(
            model,
            X_train, y_train,
            X_val, y_val,
            epochs=1,
            batch_size=32,
            model_dir=model_dir,
            log_dir=log_dir
        )
        print("[*] Plotting training results...")
        plot_training_results(history, save_dir=os.path.join(project_dir, "outputs"))
        print("[+] Quick training pipeline complete!")
        
    else:
        print("[*] Running standard training pipeline...")
        print("[*] Running preprocessing pipeline...")
        X_train, y_train, X_val, y_val, word2idx, idx2word = run_preprocessing_pipeline(data_path=data_path)
        
        vocab_size = len(word2idx)
        print(f"[+] Loaded dataset. Vocabulary size: {vocab_size}")
        
        print("[*] Building Stacked LSTM model...")
        model = build_lstm_model(vocab_size=vocab_size, seq_length=20)
        
        print("[*] Training model...")
        model, history = compile_and_train_model(
            model, 
            X_train, y_train, 
            X_val, y_val, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            model_dir=model_dir,
            log_dir=log_dir
        )
        
        print("[*] Plotting training results...")
        plot_training_results(history, save_dir=os.path.join(project_dir, "outputs"))
        print("[+] Standard training pipeline complete!")
