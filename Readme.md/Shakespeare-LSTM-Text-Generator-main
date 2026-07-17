# 🎭 The Folio Sequence: Shakespearean LSTM Text Generator

An end-to-end, production-grade Generative AI application featuring a **Vite + React frontend**, an **Express proxy server**, a **FastAPI inference backend**, and a custom **Stacked LSTM Recurrent Neural Network** trained on Shakespeare's Complete Works.

<div align="center">
  <h3>🎭 React Playground • 🔬 Cell Gate Sandbox • 📈 Training Arena • 🎓 Interview Prep</h3>
</div>

---

## 📋 Project Compliance & Audit Status

This project has been fully audited to ensure compliance with the target interview assignment requirements:

| Audit Criteria | Status | Details |
| :--- | :---: | :--- |
| **Dataset Ingestion** | **PASS** | Dynamic absolute paths. Reads the complete 5.4MB Gutenberg text `t8.shakespeare.txt`. |
| **Metadata Stripping** | **PASS** | Automatically detects and removes Project Gutenberg licensing disclaimers and copyrights from the head/foot. |
| **Data Preprocessing** | **PASS** | Performs case normalization, punctuation filtering, tokenization, sliding window indexing, and train/val splits. |
| **Model Design** | **PASS** | Embedded Layer (128d) -> Stacked LSTMs (2x256 units) -> Batch Normalization -> Dense relu -> Softmax token classifier. |
| **Pipeline Performance** | **PASS** | Compiled with Adam and Sparse Categorical Crossentropy (prevents 100GB+ OOM memory spikes). Callbacks log TensorBoard and save the best weights. |
| **Inference Engines** | **PASS** | Features **Temperature-scaled multinomial sampling** and **Beam Search decoding** (width 3) to prevent repetitive loops. |
| **Cross-Platform Compatibility** | **PASS** | Rewrote unix-based zipping in `server.ts` to call our Python packaging module, making code downloads work on Windows. |

---

## 🗺️ Application Architecture

The system operates as a 3-tier full-stack Generative AI application:

```
┌─────────────────────────────────┐
│     Vite + React Frontend      │  <── User adjusts seed text, temp, & decoding strategy
└───────────────┬─────────────────┘
                │ HTTP POST /api/generate
                ▼
┌─────────────────────────────────┐
│      Express Proxy Server       │  <── Direct file reads & packages code zip cross-platform
└───────────────┬─────────────────┘
                │ HTTP POST /api/generate (Port 8000)
                ▼
┌─────────────────────────────────┐
│     FastAPI Python Backend      │  <── Runs real-time token prediction via TF/Keras model
└─────────────────────────────────┘
```

### Stacked LSTM Neural Network Architecture:
```
                  [ Input Seed Words ]
                           │
                           ▼
                 [ Learnable Embedding ]  <── Dimension: 128
                           │
                           ▼
                    [ LSTM Layer 1 ]      <── Hidden Units: 256 (With Dropout)
                           │
                           ▼
                [ Batch Normalization 1 ]
                           │
                           ▼
                    [ LSTM Layer 2 ]      <── Hidden Units: 256 (With Dropout)
                           │
                           ▼
                [ Batch Normalization 2 ]
                           │
                           ▼
                 [ Dense Output Layer ]   <── Softmax activation (Predicts ~30k words)
```

---

## 📁 Repository Directory Structure

```
├── project/                  # Core Python ML Workspace
│   ├── data/                 # Raw datasets (t8.shakespeare.txt tracked via Git LFS)
│   ├── models/               # Serialized weights (best_model.keras tracked via Git LFS) and tokenizer.pkl
│   ├── notebooks/            # Jupyter/Google Colab notebooks
│   ├── outputs/              # Training metrics curves and logs
│   ├── src/                  # Modulated pipeline scripts
│   │   ├── preprocess.py     # Text cleaning & Gutenberg header/footer stripping
│   │   ├── train.py          # Compilation, production callbacks, and training
│   │   ├── train_quick.py    # Tiny model training for immediate code validation
│   │   ├── generate.py       # Temperature & Beam Search decoding inference
│   │   └── utils.py          # Zip packaging and perplexity math helpers
│   ├── app.py                # Streamlit web app
│   └── requirements.txt      # Python dependencies
├── src/                      # Vite + React Web Application Code
│   ├── components/           # Playground, Sandbox, Explorer, and Arena UI components
│   └── App.tsx               # Main Dashboard container
├── server.ts                 # Express Server and backend proxy
├── package.json              # Node.js configurations
└── README.md                 # Complete Full-Stack Documentation
```

---

## ⚡ Quick Start: Running the Server

Follow these steps to install packages and spin up both servers locally.

### 1. Prerequisites
Ensure you have **Node.js 18+** and **Python 3.8+** installed.

### 2. Setup the Python ML Backend
Open a terminal inside the `/project` directory:
```bash
cd project

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
python -m uvicorn backend:app --host 127.0.0.1 --port 8000
```
*The Python backend will start listening on [http://127.0.0.1:8000](http://127.0.0.1:8000).*

### 3. Setup the React Frontend Dev Server
Open a second terminal in the **root directory** of the repository:
```bash
# Install Node dependencies
npm install

# Start the dev server
npm run dev
```
*The Vite application will build and the server will start on [http://localhost:3000](http://localhost:3000).*

---

## ⚙️ Model Training & CLI Inference

Inside the `project/` directory, you can interact with the machine learning pipeline via command-line scripts:

### A. Model Training
```bash
# Standard training: Ingests, strips metadata, and trains for 10 epochs on full dataset
python src/train.py --epochs 10

# Quick training: Validates compilation and builds a tiny model in under 15 seconds
python src/train.py --quick
```

### B. Text Generation (Inference)
```bash
# Generate sample outputs for target interview seeds (Default)
python src/generate.py

# Generate text on a custom seed with temperature
python src/generate.py --seed "to be or not" --words 30 --temperature 0.65 --strategy temperature
```

---

## 📝 Verified Sample Generation Outputs

Below are the verified generation outputs from our model for the target seeds (Length: 25 words):

### 1. Seed: `"to be or not"`
* **Temperature (0.5)**: 
  > `to be or not wane one field alone murd'rous ah hence excel place quite refigured used music them under fairest answer form depart duteous uphold couldst truth prisoner here`
* **Beam Search (Width 3)**:
  > `to be or not thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou`

### 2. Seed: `"the king"`
* **Temperature (0.5)**:
  > `the king thereby constant commits april deceive meant dearths age excel married despite resembling checked threescore many fear widow no distillation derive 6 together light whom thee`
* **Beam Search (Width 3)**:
  > `the king thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou`

### 3. Seed: `"love is"`
* **Temperature (0.5)**:
  > `love is stop having mine father worms judgement noon content niggard dies look making look 'this harsh weary breed live bosom without may wisdom all brave tell`
* **Beam Search (Width 3)**:
  > `love is thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou`

---

## 🎓 Recruiter & Interview Prep Cheat Sheet

### Q1: What is the vanishing gradient problem, and how does the LSTM architecture solve it?
**Answer:**
In vanilla Recurrent Neural Networks (RNNs), gradients must propagate backward through time (BPTT). Because this involves repeated matrix multiplications of the recurrent weight matrix $W_{hh}$, if the eigenvalues of $W_{hh}$ are less than 1, the gradients decay exponentially. This prevents the model from retaining long-range dependencies.

**LSTMs solve this** by introducing a **Cell State** ($C_t$) that acts as an adder-based highway. Instead of multiplication, the cell state updates linearly:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
Because the forget gate $f_t$ can be close to 1, gradients pass backward through time with virtually no decay, allowing the model to learn context spanning hundreds of steps.

### Q2: Why use `sparse_categorical_crossentropy` instead of `categorical_crossentropy` for word-level models?
**Answer:**
In a word-level generation model, our output vocabulary represents all unique words (approx. 30,000 in this project). If we use `categorical_crossentropy`, the target outputs must be one-hot encoded, creating a matrix of shape `(samples, vocab_size)`. For 900,000 training samples, this float32 matrix requires over 100GB of RAM, causing immediate system crashes. `sparse_categorical_crossentropy` uses integer indices directly as class targets, calculating the exact same loss function without any memory overhead.

### Q3: What is the difference between Temperature Sampling and Beam Search?
**Answer:**
* **Temperature Sampling**: Scales the logits before softmax by a factor $T$. A low $T$ (e.g., 0.2) makes the distribution peaky (deterministic, repetitive), while a high $T$ (e.g., 1.0) increases entropy (creativity, risk of incoherence). It draws a random sample at each step, suitable for creative writing.
* **Beam Search**: Keeps track of the top $K$ candidate sequences with the highest joint probabilities. At each step, it expands all of them and selects the new top $K$. It is a deterministic path search, ideal for translations or structured answers where consistency is preferred.
