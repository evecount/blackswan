# BlackSwan: Embedding + CNN for Extreme Event Semantic Analysis

> **An Advanced Quantitative Case Study on Exogenous Information Latency and Semantic Microstructure**  
> *Coursework Extension & Advanced Research Workbook*

### 👤 Author & Academic Attribution
* **Student:** Gwendalynn Lim Wan Ting
* **Institution:** Singapore Institute of Technology (SIT)
* **Student Email:** [`2504137@sit.singaporetech.edu.sg`](mailto:2504137@sit.singaporetech.edu.sg)
* **Academic Program:** Information and Communications Technology / Applied Computing (AI & Machine Learning)
* **Course Context:** ICT3506C: Applied Natural Language Processing — Advanced Lab Extension (Lab 2.4: Embedding + CNN)

---

## 🧭 Executive Overview

This workbook represents an advanced, research-oriented extension of the foundational **"Embedding + Convolutional Neural Networks for Sentiment Analysis"** curriculum.

While standard natural language processing (NLP) labs introduce text classification on retail market sentiment (classifying financial news headlines into standard positive, neutral, or negative categories), quantitative research and market microstructure desks frequently apply this architecture to **exogenous shock forensics**.

In high-stakes, black-swan scenarios (e.g., unexpected macroeconomic announcements, geopolitical disruptions, or systemic market shocks), financial headlines do not simply reflect binary positive or negative sentiment. Instead, they reflect a rapid, high-entropy mutation from **ambiguous localized noise** (*"unconfirmed report"*, *"minor disruption"*) to **confirmed systemic structural threat** (*"catastrophic failure"*, *"emergency intervention"*).

This lab challenges students to build a production-grade **1D Convolutional Neural Network (CNN)** leveraging **pre-trained Transformer (BERT) embeddings** to model and measure the semantic phase-shift of information during extreme historical events.

---

## 🎯 Pedagogical Objectives & Alignment

This workbook preserves and deepens all core curricular requirements:

| Curricular Objective | Foundational Implementation | BlackSwan Advanced Extension |
| :--- | :--- | :--- |
| **1. Data Preparation** | Cleaning headline sentences and mapping tokens to index sequences with padding. | Constructing timestamped event sequences, tokenization, sequence alignment, and sliding-window temporal windows. |
| **2. Transfer Learning** | Importing pre-trained embeddings (e.g., BERT / FinBERT) to accelerate convergence. | Freezing pre-trained BERT layers for base representations with optional targeted unfreezing for domain-specific fine-tuning. |
| **3. Model Building (1D CNN)** | Single or multi-filter 1D convolutions to capture fixed $n$-grams. | **Multi-scale parallel 1D CNN kernels** ($k \in \{2, 3, 4, 5\}$) to simultaneously capture localized bigrams through pentagrams. |
| **4. Feature Pooling** | 1D Max Pooling across temporal dimension. | **Max-over-Time Pooling** to extract the dominant semantic activations invariant of sequence position. |
| **5. Training & Evaluation** | Classification report (Precision, Recall, F1, Loss curve). | Classification metrics supplemented by **temporal kernel activation analysis** (measuring latency from shock to market absorption). |

---

## 🏛️ Theoretical Architecture

```
                          [ Raw Timestamped News / Wire Stream ]
                                            │
                                            ▼
                           [ BERT WordPiece Tokenizer ]
                       (Input IDs, Attention Masks, Padding)
                                            │
                                            ▼
                       [ Pre-trained BERT Embedding Layer ]
                            (Frozen / Fine-Tuned Weights)
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               │                            │                            │
               ▼                            ▼                            ▼
      [ Conv1D Kernel: k=2 ]       [ Conv1D Kernel: k=3 ]       [ Conv1D Kernel: k=4 ]
         (Bigram Features)           (Trigram Features)          (4-gram Features)
               │                            │                            │
               ▼                            ▼                            ▼
        [ ReLU / GELU ]              [ ReLU / GELU ]              [ ReLU / GELU ]
               │                            │                            │
               ▼                            ▼                            ▼
      [ Max-over-Time Pool ]       [ Max-over-Time Pool ]       [ Max-over-Time Pool ]
               │                            │                            │
               └────────────────────────────┼────────────────────────────┘
                                            │
                                            ▼
                              [ Concatenated Feature Vector ]
                                            │
                                            ▼
                               [ Dropout Regularization ]
                                            │
                                            ▼
                                  [ Fully-Connected Dense ]
                                            │
                                            ▼
                             [ Softmax / Logits Output ]
                       (Uncertainty vs. Confirmed Shock)
```

---

## 📁 Repository Structure

```
D:\blackswan\
├── README.md                      # Academic context, architecture, & theoretical foundation
├── .gitignore                     # Standard Python/Jupyter ignore rules
├── WORKBOOK_SPECIFICATION.md      # Detailed problem set, math specifications, & rubrics
├── notebooks/
│   └── BlackSwan_Lab_Workbook.ipynb # Interactive student workbook with code scaffolds
└── data/
    └── sample_event_stream.json   # Seed schema for timestamped event forensics
```

---

## 🚀 Getting Started

### 1. Environment Setup
Ensure you have a Python 3.10+ environment with PyTorch and Hugging Face Transformers:

```bash
git clone https://github.com/evecount/blackswan.git
cd blackswan
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Install requirements:
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn jupyter
```

### 2. Launch the Workbook
```bash
jupyter notebook notebooks/BlackSwan_Lab_Workbook.ipynb
```

---

## ⚖️ Academic Integrity & Context
*This project is an open-source educational module created for advanced quantitative natural language processing research. It is designed to complement academic curricula by providing real-world market microstructure and historical forensic contexts for neural text classification.*
