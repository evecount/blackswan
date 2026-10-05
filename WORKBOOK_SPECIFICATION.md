# BlackSwan Lab: Workbook Specification & Mathematical Framework

**Student:** Gwendalynn Lim Wan Ting (`2504137@sit.singaporetech.edu.sg`)  
**Institution:** Singapore Institute of Technology (SIT)  
**Program:** Information and Communications Technology / Applied Computing (AI & ML)  
**Module:** ICT3506C Applied Natural Language Processing — Lab 2.4 Research Extension  
**AI Collaboration:** Pair-programmed with **Google Antigravity** (Google DeepMind)  

---

## 1. Problem Formulation: Exogenous Shock Latency

In conventional sentiment analysis, a sentence $S = (w_1, w_2, \dots, w_T)$ is assigned a static categorical label $y \in \{-1, 0, 1\}$.

In **forensic market microstructure**, an exogenous shock event generates a sequence of timestamped text observations:
$$\mathcal{D} = \{(S_1, t_1), (S_2, t_2), \dots, (S_N, t_N)\}$$

where each headline arrives at continuous time $t_i$. The objective of the neural feature extractor is twofold:
1. Extract localized semantic triggers (via parallel $n$-gram convolutions) indicating structural anomalies.
2. Measure the latency function $\tau$:
$$\tau = t_{\text{absorbed}} - t_{\text{onset}}$$
where $t_{\text{onset}}$ is the physical occurrence timestamp, and $t_{\text{absorbed}}$ is the timestamp where the model's confidence distribution over the **SYSTEMIC_SHOCK** class exceeds a threshold $\theta \ge 0.95$.

---

## 2. Neural Architecture Mathematical Specification

### A. Pre-trained Embedding Layer (Transfer Learning)
Given an input sequence of token indices $\mathbf{x} = [x_1, x_2, \dots, x_L] \in \mathbb{Z}^L$, we project each token into a dense semantic space using pre-trained BERT embeddings:
$$\mathbf{E} = \text{Embedding}(\mathbf{x}) \in \mathbb{R}^{L \times d}$$
where $L$ is sequence length (with padding) and $d = 768$ is the hidden embedding dimension.

### B. Multi-Scale 1D Convolutional Feature Maps
Instead of a single filter size, we deploy a bank of parallel convolutional kernels $W_k \in \mathbb{R}^{k \times d}$ for window sizes $k \in \{2, 3, 4, 5\}$.

For a filter of length $k$, the convolution operation on a sub-matrix $\mathbf{E}_{i:i+k-1}$ generates a feature $c_i$:
$$c_i = f(\mathbf{E}_{i:i+k-1} \ast W_k + b)$$
where:
- $\ast$ denotes the Frobenius inner product across the temporal slice,
- $b \in \mathbb{R}$ is a scalar bias,
- $f(\cdot)$ is a non-linear activation (ReLU or GELU).

Applying this filter across all possible windows yields the feature map:
$$\mathbf{c}^{(k)} = [c_1, c_2, \dots, c_{L-k+1}] \in \mathbb{R}^{L-k+1}$$

### C. Max-over-Time Pooling
To extract the most salient $n$-gram signal irrespective of where it appears in the headline, we apply global max-pooling over time:
$$\hat{c}^{(k)} = \max_{1 \le i \le L-k+1} c_i$$

For $M$ feature maps per filter size $k$, we obtain a pooled vector $\mathbf{h}^{(k)} \in \mathbb{R}^M$.

### D. Multi-Kernel Concatenation & Classification
The pooled representations from all filter sizes are concatenated into a unified semantic descriptor:
$$\mathbf{z} = [\mathbf{h}^{(2)} \,\|\, \mathbf{h}^{(3)} \,\|\, \mathbf{h}^{(4)} \,\|\, \mathbf{h}^{(5)}] \in \mathbb{R}^{4M}$$

Regularized by dropout with probability $p \in [0.2, 0.5]$:
$$\tilde{\mathbf{z}} = \text{Dropout}(\mathbf{z})$$

And projected through a linear classifier to produce class logits:
$$\hat{\mathbf{y}} = \mathbf{W}_c \tilde{\mathbf{z}} + \mathbf{b}_c$$
$$P(y = c \mid S) = \frac{\exp(\hat{y}_c)}{\sum_{j=1}^C \exp(\hat{y}_j)}$$

---

## 3. Student Exercises Breakdown (Self-Guided Implementation)

### Exercise 1: Tokenization and Sequence Alignment
- **Task:** Implement `preprocess_event_stream()` using Hugging Face's `BertTokenizerFast`.
- **Requirements:** 
  - Truncation to max sequence length $L = 64$.
  - Padding with special token `[PAD]`.
  - Generation of `input_ids` and `attention_mask` tensors.

### Exercise 2: Pre-trained BERT Backbone Configuration
- **Task:** Instantiate `BertModel.from_pretrained('bert-base-uncased')`.
- **Requirements:**
  - Freeze the base encoder weights (`param.requires_grad = False`).
  - Verify that gradient computation is restricted solely to the downstream CNN head.

### Exercise 3: Multi-Scale 1D CNN Head Implementation
- **Task:** Build the `MultiScaleCNNTextClassifier` PyTorch `nn.Module`.
- **Requirements:**
  - Use `nn.ModuleList` containing `nn.Conv1d(in_channels=768, out_channels=128, kernel_size=k)` for $k \in [2, 3, 4, 5]$.
  - Implement dynamic temporal max-pooling handling variable sequence lengths.
  - Concatenate pooled vectors and attach linear classification head.

### Exercise 4: Training Loop & Metric Convergence
- **Task:** Implement standard mini-batch training loop with `AdamW` optimizer and cross-entropy loss.
- **Evaluation:** Output standard `classification_report` (Precision, Recall, F1 score).

### Exercise 5: Forensic Timeline Activation Analysis
- **Task:** Feed the sequential event timeline $\mathcal{D}$ through the trained model.
- **Output:** Plot predicted class probabilities over the 8:46 AM to 9:15 AM window to visualize the transition threshold from localized uncertainty to systemic shock.

---

## 4. Discussion & Defense Questions
1. **Microstructure Latency & Adverse Selection:** Why do tier-1 quantitative market makers (e.g., **Jane Street**, **Citadel Securities**) treat retail headline scraping as an adverse selection risk rather than an execution alpha source? Why is a CNN inference latency of $\sim 15\text{ms}$ sufficient for macroeconomic risk adjustment, but completely non-viable for collocated HFT?
2. **Kernel Interpretability:** How do the activation maps of $k=2$ (bigram) filters differ from $k=4$ (4-gram) filters when processing compound phrases such as *"controlled airspace shutdown"*?
3. **Catastrophic Drift:** How does a language model pre-trained on modern web data perform on historical event syntax from 2001?
